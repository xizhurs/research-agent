import io
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.agent import AgentResult, Answer, Claim, RunStats
from research_agent.cli import app, make_output_safe
from research_agent.llm import LLMRateLimitError
from research_agent.retrieval import RateLimitError, SearchResult, Work

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"
runner = CliRunner()
NL = chr(10)


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "show-config" in result.output


def test_show_config_prints_limits():
    result = runner.invoke(app, ["show-config", "--config", str(BASE)])
    assert result.exit_code == 0
    assert "max_iterations = 5" in result.output


def search_args(tmp_path):
    return ["search", "q", "--config", str(BASE), "--env-file", str(tmp_path / "none")]


class FakeClient:
    """Stands in for OpenAlexClient so CLI tests make no network calls."""

    outcome: object = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def search(self, query, *, year_from=None, max_results=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_search_prints_results(monkeypatch, tmp_path):
    work = Work(
        id="W1", url="https://openalex.org/W1", title="A title", year=2024, date=None,
        authors=("A", "B", "C", "D"), venue=None, doi=None, type=None, language=None,
        cited_by_count=None, abstract=None, abstract_truncated=False,
    )
    FakeClient.outcome = SearchResult("q", None, (work,), 10, 0, from_cache=True)
    monkeypatch.setattr("research_agent.cli.OpenAlexClient", FakeClient)
    result = runner.invoke(app, search_args(tmp_path))
    assert result.exit_code == 0
    assert "1 of 10 matches (cache)" in result.output
    assert "[W1] A title (2024) - A, B, C et al." in result.output
    assert "1 result(s) have no abstract" in result.output


def test_search_retrieval_error_exits_nonzero(monkeypatch, tmp_path):
    FakeClient.outcome = RateLimitError("budget exhausted")
    monkeypatch.setattr("research_agent.cli.OpenAlexClient", FakeClient)
    result = runner.invoke(app, search_args(tmp_path))
    assert result.exit_code == 1
    assert "budget exhausted" in result.output


def test_show_config_bad_path_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["show-config", "--config", str(tmp_path / "missing.toml")])
    assert result.exit_code == 1


def agent_result(make_work):
    w = make_work("W1", title="Crop maps", authors=("A", "B"))
    answer = Answer(
        claims=(
            Claim("Crops can be mapped.", "evidence", ("W1",)),
            Claim("It likely generalises.", "interpretation", ()),
        ),
        references=(w,),
        limitations=("Only abstracts were read.",),
    )
    return AgentResult(answer, RunStats(2, 1, 400, 100, 3))


def ask_args(tmp_path, question="How are crops mapped?"):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef\n", encoding="utf-8")
    return ["ask", question, "--config", str(BASE), "--env-file", str(env)]


def test_ask_prints_answer_references_limitations_and_stats(monkeypatch, tmp_path, make_work):
    monkeypatch.setattr("research_agent.cli._run_agent", lambda *a: agent_result(make_work))
    result = runner.invoke(app, ask_args(tmp_path))
    assert result.exit_code == 0
    assert "- (evidence) Crops can be mapped. [W1]" in result.output
    assert "- (interpretation) It likely generalises." in result.output
    assert "[W1] A, B (2023). Crop maps." in result.output
    assert "- Only abstracts were read." in result.output
    footer = "2 model turn(s), 1 search(es) (0 from cache), 500 tokens (400 in / 100 out)"
    assert footer in result.output


def test_ask_llm_error_exits_nonzero(monkeypatch, tmp_path):
    def boom(*args):
        raise LLMRateLimitError("quota exceeded")

    monkeypatch.setattr("research_agent.cli._run_agent", boom)
    result = runner.invoke(app, ask_args(tmp_path))
    assert result.exit_code == 1
    assert "quota exceeded" in result.output


def test_ask_without_openai_key_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = ["ask", "q", "--config", str(BASE), "--env-file", str(tmp_path / "absent")]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output


def test_ask_empty_question_exits_nonzero(tmp_path):
    result = runner.invoke(app, ask_args(tmp_path, question="  "))
    assert result.exit_code == 1
    assert "must not be empty" in result.output


def test_output_with_unencodable_characters_does_not_crash():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        stream.write("non‐breaking")  # a legacy console would crash here
        stream.flush()
    raw, stream = io.BytesIO(), None
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    make_output_safe(stream)
    stream.write("non‐breaking Αν")
    stream.flush()
    assert raw.getvalue() == b"non?breaking ??"


def test_make_output_safe_ignores_streams_that_cannot_be_reconfigured():
    make_output_safe(io.StringIO())  # no reconfigure(): must not raise


# --- eval command ----------------------------------------------------------------------------


class FakeEvalAgent:
    def __init__(self, result_for):
        self.result_for = result_for
        self.questions = []

    def run(self, question):
        self.questions.append(question)
        return self.result_for(question)


def patch_agent(monkeypatch, agent):
    from contextlib import contextmanager

    @contextmanager
    def fake_open(cfg, creds):
        yield agent

    monkeypatch.setattr("research_agent.cli._open_agent", fake_open)


def two_case_file(tmp_path):
    path = tmp_path / "cases.toml"
    path.write_text(
        "[[case]]" + NL + "id = 'one'" + NL + "question = 'Q1'" + NL + "min_references = 1" + NL
        + "[[case]]" + NL + "id = 'two'" + NL + "question = 'Q2'" + NL,
        encoding="utf-8",
    )
    return path


def eval_args(tmp_path, *extra):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef" + NL, encoding="utf-8")
    cases = two_case_file(tmp_path)
    return ["eval", "--cases", str(cases), "--config", str(BASE), "--env-file", str(env), *extra]


def test_eval_all_pass_exit_zero(monkeypatch, tmp_path, make_work):
    agent = FakeEvalAgent(lambda q: agent_result(make_work))
    patch_agent(monkeypatch, agent)
    result = runner.invoke(app, eval_args(tmp_path))
    assert result.exit_code == 0, result.output
    assert "[PASS] one" in result.output and "[PASS] two" in result.output
    assert "Summary: 2/2 passed" in result.output
    assert agent.questions == ["Q1", "Q2"]


def test_eval_failure_exits_nonzero_and_lists_reasons(monkeypatch, tmp_path, make_work):
    def result_for(question):
        res = agent_result(make_work)
        if question == "Q1":  # no references, but case 'one' demands one
            res = AgentResult(
                Answer(res.answer.claims, (), res.answer.limitations), res.stats
            )
        return res

    patch_agent(monkeypatch, FakeEvalAgent(result_for))
    result = runner.invoke(app, eval_args(tmp_path))
    assert result.exit_code == 1
    assert "[FAIL] one" in result.output
    assert "at least 1 reference" in result.output
    assert "Summary: 1/2 passed" in result.output


def test_eval_only_runs_selected_case(monkeypatch, tmp_path, make_work):
    agent = FakeEvalAgent(lambda q: agent_result(make_work))
    patch_agent(monkeypatch, agent)
    result = runner.invoke(app, eval_args(tmp_path, "--only", "two"))
    assert result.exit_code == 0
    assert agent.questions == ["Q2"]
    assert "Summary: 1/1 passed" in result.output


def test_eval_unknown_case_id_exits_nonzero(monkeypatch, tmp_path, make_work):
    patch_agent(monkeypatch, FakeEvalAgent(lambda q: agent_result(make_work)))
    result = runner.invoke(app, eval_args(tmp_path, "--only", "nope"))
    assert result.exit_code == 1
    assert "unknown case id" in result.output


def test_eval_bad_case_file_exits_nonzero(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef" + NL, encoding="utf-8")
    missing = tmp_path / "nope.toml"
    args = ["eval", "--cases", str(missing), "--config", str(BASE), "--env-file", str(env)]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "not found" in result.output


def test_eval_without_openai_key_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = [
        "eval", "--cases", str(two_case_file(tmp_path)), "--config", str(BASE),
        "--env-file", str(tmp_path / "absent"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output


def test_eval_auth_failure_aborts_with_exit_one(monkeypatch, tmp_path):
    from research_agent.llm import LLMAuthError

    def boom(question):
        raise LLMAuthError("OpenAI rejected the API key")

    patch_agent(monkeypatch, FakeEvalAgent(boom))
    result = runner.invoke(app, eval_args(tmp_path))
    assert result.exit_code == 1
    assert "rejected the API key" in result.output
