"""M5 hardening: recoverable failures, cache resilience, input limits, run statistics, crashes."""

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from research_agent import cli
from research_agent.agent import InvalidQuestionError, ResearchAgent
from research_agent.agent.budget import Budget
from research_agent.config import ConfigError, ReportingConfig, load_config
from research_agent.llm import (
    LLMApiError,
    StubLLMClient,
    ToolCall,
    ToolResultMessage,
    Usage,
    text_response,
    tool_call_response,
)
from research_agent.llm.types import LLMResponse
from research_agent.logging_setup import REDACTOR, configure_logging, reset_logging
from research_agent.retrieval import OpenAlexClient
from research_agent.retrieval.cache import ResponseCache

REPORTING = ReportingConfig(3, 3)
SEARCH = "search_literature"
ROOT = Path(__file__).resolve().parents[1]
BASE_TOML = ROOT / "configs" / "base.toml"
OPENALEX_FIXTURE = Path(__file__).parent / "fixtures" / "openalex_search.json"


class FakeSearch:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def search(self, query, *, year_from=None, max_results=None):
        self.calls += 1
        return self.result


def agent(llm, search, limits):
    return ResearchAgent(llm, search, limits, REPORTING)


def final():
    return json.dumps({"claims": [], "limitations": []})


@pytest.fixture(autouse=True)
def clean_logging():
    reset_logging()
    yield
    reset_logging()


# --- the model's malformed tool call no longer kills the run ---------------------------------


def test_malformed_tool_arguments_go_back_to_the_model_and_the_run_continues(limits):
    bad = LLMResponse(
        None,
        (ToolCall("c1", SEARCH, {}, error="the tool arguments were not valid JSON"),),
        Usage(10, 5),
        "tool_calls",
    )
    llm = StubLLMClient([bad, text_response(final())])
    search = FakeSearch(None)
    result = agent(llm, search, limits).run("question")

    assert search.calls == 0  # no search was run or charged
    feedback = llm.calls[1].messages[-1]
    assert isinstance(feedback, ToolResultMessage) and feedback.call_id == "c1"
    assert "not valid JSON" in feedback.content and "valid JSON arguments" in feedback.content
    assert result.stats.search_calls == 0
    assert result.answer.limitations  # the run still produced a complete, honest answer


def test_malformed_call_alongside_good_call_only_fails_the_bad_one(limits, make_work, make_result):
    calls = (
        ToolCall("bad", SEARCH, {}, error="the tool arguments were not valid JSON"),
        ToolCall("good", SEARCH, {"query": "land cover"}),
    )
    first = LLMResponse(None, calls, Usage(10, 5), "tool_calls")
    llm = StubLLMClient([first, text_response(final())])
    search = FakeSearch(make_result([make_work()]))
    agent(llm, search, limits).run("question")
    results = {
        m.call_id: m.content for m in llm.calls[1].messages if isinstance(m, ToolResultMessage)
    }
    assert results["bad"].startswith("Error:")
    assert "[W1]" in results["good"]
    assert search.calls == 1


# --- cache failures never fail a run ---------------------------------------------------------


def test_cache_write_failure_is_logged_not_raised(tmp_path, caplog):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file where the cache folder should be", encoding="utf-8")
    cache = ResponseCache(blocker / "cache", ttl_days=1)
    with caplog.at_level(logging.WARNING):
        cache.put("k", {"a": 1})  # must not raise
    assert "could not write cache entry" in caplog.text
    assert cache.get("k") is None


def test_search_still_succeeds_when_the_cache_is_unwritable(base_config, tmp_path):
    fixture = json.loads(OPENALEX_FIXTURE.read_text(encoding="utf-8"))
    blocker = tmp_path / "blocked"
    blocker.write_text("x", encoding="utf-8")
    cfg = replace(base_config.openalex, cache_dir=blocker / "cache")
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=fixture)))
    client = OpenAlexClient(cfg, base_config.limits, http_client=http)
    result = client.search("land cover")
    assert len(result.works) == 3 and result.from_cache is False


# --- question limit --------------------------------------------------------------------------


def test_question_at_the_limit_is_accepted_and_one_over_is_rejected(limits):
    n = limits.max_question_chars
    llm = StubLLMClient([text_response(final())])
    agent(llm, FakeSearch(None), limits).run("q" * n)  # exactly at the limit
    over = StubLLMClient([])
    with pytest.raises(InvalidQuestionError, match=str(n)):
        agent(over, FakeSearch(None), limits).run("q" * (n + 1))
    assert over.calls == []  # rejected before any model call, so no tokens are spent


def test_empty_question_is_an_invalid_question_error(limits):
    llm = StubLLMClient([])
    with pytest.raises(InvalidQuestionError, match="non-empty"):
        agent(llm, FakeSearch(None), limits).run("   ")


def test_invalid_question_error_is_still_a_value_error():
    assert issubclass(InvalidQuestionError, ValueError)


@pytest.mark.parametrize("bad", ["0", "-5", "1.5", '"2000"'])
def test_question_limit_must_be_a_positive_integer(tmp_path, bad):
    text = BASE_TOML.read_text(encoding="utf-8")
    p = tmp_path / "c.toml"
    p.write_text(text.replace("max_question_chars = 2000", f"max_question_chars = {bad}"), "utf-8")
    with pytest.raises(ConfigError, match="max_question_chars"):
        load_config(p)


# --- per-run statistics ----------------------------------------------------------------------


def test_budget_counts_cached_searches_separately():
    b = Budget(load_config(BASE_TOML).limits)
    b.record_search()
    b.record_search(cached=True)
    assert (b.search_calls, b.cached_searches) == (2, 1)


def test_run_stats_report_cache_hits(limits, make_work, make_result):
    cached = replace(make_result([make_work()]), from_cache=True)
    first = tool_call_response(("c1", SEARCH, {"query": "a"}))
    llm = StubLLMClient([first, text_response(final())])
    stats = agent(llm, FakeSearch(cached), limits).run("question").stats
    assert (stats.search_calls, stats.cached_searches) == (1, 1)


def test_run_stats_count_network_searches_as_not_cached(limits, make_work, make_result):
    first = tool_call_response(("c1", SEARCH, {"query": "a"}))
    llm = StubLLMClient([first, text_response(final())])
    stats = agent(llm, FakeSearch(make_result([make_work()])), limits).run("question").stats
    assert (stats.search_calls, stats.cached_searches) == (1, 0)


# --- progress logging ------------------------------------------------------------------------


def test_progress_is_logged_at_info_level(limits, make_work, make_result, capsys):
    configure_logging(1)
    first = tool_call_response(("c1", SEARCH, {"query": "land cover"}))
    llm = StubLLMClient([first, text_response(final())])
    agent(llm, FakeSearch(make_result([make_work()], total=40)), limits).run("a private question")
    err = capsys.readouterr().err
    assert "turn 1:" in err and "turn 2:" in err
    assert 'search "land cover"' in err and "1 of 40 matches (network)" in err
    assert "done: 2 turn(s), 1 search(es) (0 cached)" in err
    assert "a private question" not in err  # the user's question text is not logged


def test_nothing_is_logged_by_default(limits, capsys):
    configure_logging(0)
    agent(StubLLMClient([text_response(final())]), FakeSearch(None), limits).run("q")
    assert capsys.readouterr().err == ""


# --- last-resort crash handling --------------------------------------------------------------


def run_with_failing_app(monkeypatch, capsys, exc, argv=("research-agent",)):
    def failing_app():
        raise exc

    monkeypatch.setattr(cli, "app", failing_app)
    monkeypatch.setattr(sys, "argv", list(argv))
    with pytest.raises(SystemExit) as info:
        cli.run()
    captured = capsys.readouterr()
    return info.value.code, captured


def test_unexpected_exception_exits_one_with_a_single_clear_line(monkeypatch, capsys):
    code, out = run_with_failing_app(monkeypatch, capsys, RuntimeError("something broke"))
    assert code == 1
    assert "unexpected failure (RuntimeError: something broke)" in out.err
    assert "Use -vv for a traceback" in out.err
    assert "Traceback" not in out.err  # no dump unless asked for
    assert out.out == ""


def test_unexpected_exception_message_is_redacted(monkeypatch, capsys):
    REDACTOR.add("sk-test-0123456789abcdef")
    _, out = run_with_failing_app(
        monkeypatch, capsys, RuntimeError("bad key sk-test-0123456789abcdef in request")
    )
    assert "sk-test-0123456789abcdef" not in out.err
    assert "<redacted>" in out.err


def test_traceback_available_with_vv_and_redacted(monkeypatch, capsys):
    configure_logging(2)
    REDACTOR.add("sk-test-0123456789abcdef")
    _, out = run_with_failing_app(
        monkeypatch, capsys, RuntimeError("leak sk-test-0123456789abcdef")
    )
    assert "Traceback" in out.err
    assert "sk-test-0123456789abcdef" not in out.err


def test_typer_is_configured_not_to_dump_local_variables():
    assert cli.app.pretty_exceptions_enable is False
    assert cli.app.pretty_exceptions_show_locals is False


def test_a_crash_inside_a_real_command_does_not_leak_a_local_secret(monkeypatch, capsys, tmp_path):
    """Regression: Typer's default crash output prints locals such as an API key."""
    secret = "sk-local-variable-secret-0123456789"

    def explode(*args):
        api_key = secret  # a local variable holding a secret, in the crashing frame  # noqa: F841
        raise RuntimeError("unexpected internal error")

    monkeypatch.setattr(cli, "_run_agent", explode)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef" + chr(10), encoding="utf-8")
    base = BASE_TOML
    monkeypatch.setattr(
        sys, "argv", ["research-agent", "ask", "q", "--config", str(base), "--env-file", str(env)]
    )
    with pytest.raises(SystemExit) as info:
        cli.run()
    out = capsys.readouterr()
    assert info.value.code == 1
    assert secret not in out.err and secret not in out.out
    assert "unexpected failure (RuntimeError: unexpected internal error)" in out.err


def test_expected_failures_still_use_their_own_clear_message(monkeypatch, capsys, tmp_path):
    def provider_down(*args):
        raise LLMApiError("OpenAI returned 500: down")

    monkeypatch.setattr(cli, "_run_agent", provider_down)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef" + chr(10), encoding="utf-8")
    base = BASE_TOML
    monkeypatch.setattr(
        sys, "argv", ["research-agent", "ask", "q", "--config", str(base), "--env-file", str(env)]
    )
    with pytest.raises(SystemExit) as info:
        cli.run()
    out = capsys.readouterr()
    assert info.value.code == 1
    assert "error: OpenAI returned 500: down" in out.err
    assert "unexpected" not in out.err


def test_verbose_flag_configures_logging_for_the_command(monkeypatch, capsys, tmp_path, limits):
    from contextlib import contextmanager

    llm = StubLLMClient([text_response(final())])

    @contextmanager
    def fake_open(cfg, creds):
        yield ResearchAgent(llm, FakeSearch(None), cfg.limits, cfg.reporting)

    monkeypatch.setattr(cli, "_open_agent", fake_open)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef" + chr(10), encoding="utf-8")
    base = BASE_TOML
    monkeypatch.setattr(
        sys,
        "argv",
        ["research-agent", "-v", "ask", "q", "--config", str(base), "--env-file", str(env)],
    )
    with pytest.raises(SystemExit) as info:
        cli.run()
    assert info.value.code == 0
    assert "turn 1:" in capsys.readouterr().err
