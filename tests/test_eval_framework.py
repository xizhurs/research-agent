"""The evaluation framework: run budget, recording, run records, offline rescoring and the CLI.

Everything here is offline: a scripted ``StubLLMClient`` stands in for the model.
"""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.agent import ResearchAgent
from research_agent.cli import app
from research_agent.config import ReportingConfig
from research_agent.evaluation import (
    EvalBudgetExceeded,
    EvalCase,
    EvalError,
    RecordingLLMClient,
    RunBudget,
    load_cases,
    read_records,
    request_hash,
    rescore,
    run_cases,
    select_cases,
    summarize_records,
)
from research_agent.llm import StubLLMClient, UserMessage, text_response, tool_call_response

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"
REPORTING = ReportingConfig(thin_evidence_works=3, stale_after_years=3)
runner = CliRunner()


def final(*sources):
    claims = [{"text": "Crops are mapped with CNNs.", "kind": "evidence", "sources": list(sources)}]
    return json.dumps({"claims": claims, "limitations": ["Only abstracts."]})


class FakeSearch:
    def __init__(self, result):
        self.result = result

    def search(self, query, *, year_from=None, max_results=None):
        return self.result


def search_then_answer():
    return [
        tool_call_response(("c1", "search_literature", {"query": "crop mapping"}),
                           input_tokens=100, output_tokens=20),
        text_response(final("W1"), input_tokens=300, output_tokens=50),
    ]


def agent_with(recorder, make_result, make_work, limits):
    search = FakeSearch(make_result([make_work("W1")], query="crop mapping"))
    return ResearchAgent(recorder, search, limits, REPORTING)


CASE = EvalCase("crop", "How are crops mapped?", min_references=1, category="literature")


# --- budget --------------------------------------------------------------------------------------


def test_budget_refuses_the_call_that_would_exceed_calls_tokens_or_cost():
    b = RunBudget(max_calls=2, max_tokens=1000)
    assert b.refusal(100, 100) is None
    b.record(400, 100)
    assert "token budget" in b.refusal(400, 200)
    b.record(10, 10)
    assert "model-call budget" in b.refusal(1, 1)
    priced = RunBudget(10, 10**9, max_cost_usd=0.01, price_input_per_mtok=1.0)
    assert priced.refusal(5_000, 0) is None and "cost budget" in priced.refusal(20_000, 0)
    assert priced.cost(1_000_000, 0) == pytest.approx(1.0)


def test_a_case_is_not_started_without_room_for_its_worst_case():
    b = RunBudget(max_calls=50, max_tokens=30_000)
    assert b.case_refusal(20_000) is None
    b.record(9_000, 2_000)
    assert "too low for another case" in b.case_refusal(20_000)


# --- recording -----------------------------------------------------------------------------------


def test_recording_keeps_hashes_usage_latency_and_tool_results(make_result, make_work, limits):
    ticks = iter([0.0, 0.5, 1.0, 1.25])
    recorder = RecordingLLMClient(
        StubLLMClient(search_then_answer()), RunBudget(10, 10_000), clock=lambda: next(ticks)
    )
    agent_with(recorder, make_result, make_work, limits).run("How are crops mapped?")
    first, second = recorder.log.calls
    assert first["latency_s"] == 0.5 and second["latency_s"] == 0.25
    assert first["usage"] == {"input": 100, "output": 20}
    assert first["tool_calls"][0]["name"] == "search_literature"
    assert first["request_hash"] != second["request_hash"]
    assert recorder.log.tool_results["c1"].startswith('Search "crop mapping"')
    assert recorder.budget.calls == 2 and recorder.budget.tokens == 470


def test_the_budget_stops_before_the_model_is_called():
    stub = StubLLMClient([])  # would raise AssertionError if it were called
    recorder = RecordingLLMClient(stub, RunBudget(max_calls=1, max_tokens=10))
    with pytest.raises(EvalBudgetExceeded, match="token budget"):
        recorder.complete(
            instructions="x" * 100, messages=[UserMessage("q")], tools=[], max_output_tokens=50
        )
    assert stub.calls == []


def test_request_hash_is_stable_and_sees_the_instructions():
    msgs = [UserMessage("q")]
    assert request_hash("a", msgs, [], 10, True) == request_hash("a", msgs, [], 10, True)
    assert request_hash("a", msgs, [], 10, True) != request_hash("b", msgs, [], 10, True)


# --- run records ---------------------------------------------------------------------------------


def run_two(tmp_path, make_result, make_work, limits, budget, script, cases=(CASE,)):
    recorder = RecordingLLMClient(StubLLMClient(script), budget)
    agent = agent_with(recorder, make_result, make_work, limits)
    return run_cases(
        list(cases), lambda case: agent.run(case.question), recorder, limits,
        out_dir=tmp_path / "run", run_id="r1", meta={"model": "m", "git_commit": "abc"},
    )  # fmt: skip


def test_a_record_holds_calls_tools_answer_and_stats(tmp_path, make_result, make_work, limits):
    (record,) = run_two(
        tmp_path, make_result, make_work, limits, RunBudget(10, 100_000), search_then_answer()
    )
    assert record["status"] == "ok" and record["failures"] == []
    assert (record["case_id"], record["category"], record["model"]) == ("crop", "literature", "m")
    (tool,) = record["tool_calls"]
    assert (tool["name"], tool["status"], tool["turn"]) == ("search_literature", "ok", 1)
    assert tool["result"].startswith('Search "crop mapping"')
    assert record["answer"]["works"][0]["id"] == "W1"
    assert record["answer"]["references"] == ["W1"] and record["stats"]["search_calls"] == 1
    assert record["prompt_hash"] and len(record["llm_calls"]) == 2
    stored = read_records(tmp_path / "run")
    assert stored == [record]


def test_budget_aborts_the_running_case_and_skips_the_rest(
    tmp_path, make_result, make_work, limits
):
    second = replace(CASE, id="crop-2")
    records = run_two(
        tmp_path, make_result, make_work, limits,
        RunBudget(max_calls=1, max_tokens=100_000), search_then_answer(), cases=(CASE, second),
    )  # fmt: skip
    assert [r["status"] for r in records] == ["aborted", "skipped"]
    assert "model-call budget" in records[0]["failures"][0]
    assert len(records[0]["llm_calls"]) == 1  # the first call ran, the second was refused
    summary = summarize_records(records, RunBudget(1, 1, price_input_per_mtok=2.0))
    assert (summary["aborted"], summary["skipped"], summary["passed"]) == (1, 1, 0)
    assert summary["input_tokens"] == 100 and summary["estimated_cost_usd"] == 0.0002


def test_rescore_uses_the_current_case_definitions_offline(
    tmp_path, make_result, make_work, limits
):
    records = run_two(
        tmp_path, make_result, make_work, limits, RunBudget(10, 100_000), search_then_answer()
    )
    stricter = replace(CASE, min_references=2, keywords_any=("transformer",))
    (rescored,) = rescore(records, [stricter], limits)
    assert any("at least 2 reference" in f for f in rescored["failures"])
    assert any("keywords" in f for f in rescored["failures"])
    (orphan,) = rescore(records, [], limits)
    assert orphan["failures"] == ["case no longer in the case file"]


def test_select_cases_by_split_category_and_id():
    cases = [
        EvalCase("a", "Q", split="dev", category="rainfall"),
        EvalCase("b", "Q", split="test", category="rainfall"),
        EvalCase("c", "Q", split="dev", category="enso"),
    ]
    assert [c.id for c in select_cases(cases, split="dev")] == ["a", "c"]
    assert [c.id for c in select_cases(cases, category="rainfall")] == ["a", "b"]
    assert [c.id for c in select_cases(cases, only=["b"], split="dev")] == ["b"]
    with pytest.raises(EvalError, match="unknown case id"):
        select_cases(cases, only=["zzz"])


def test_shipped_cases_have_categories_and_a_held_out_split():
    cases = load_cases(BASE.parents[1] / "evals" / "cases.toml")
    assert all(c.category for c in cases)
    test = {c.id for c in cases if c.split == "test"}
    assert len(test) == 8 and "flood-sar" in test and "temperature-cold-moscow" in test


def test_case_file_rejects_an_unknown_split_or_category(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[[case]]\nid = "a"\nquestion = "Q"\nsplit = "train"\n', encoding="utf-8")
    with pytest.raises(EvalError, match="split"):
        load_cases(path)


# --- CLI -----------------------------------------------------------------------------------------


def cases_file(tmp_path):
    path = tmp_path / "cases.toml"
    path.write_text(
        '[[case]]\nid = "crop"\nquestion = "How are crops mapped?"\nmin_references = 1\n'
        'category = "literature"\n[[case]]\nid = "held"\nquestion = "Q"\nsplit = "test"\n',
        encoding="utf-8",
    )
    return path


def test_run_without_live_prints_the_plan_and_builds_nothing(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("no agent may be built without --live")

    monkeypatch.setattr("research_agent.cli._open_agent", boom)
    args = ["eval", "run", "--cases", str(cases_file(tmp_path)), "--config", str(BASE)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Plan: 1 case(s): crop" in result.output  # the held-out case is not in the dev split
    assert "Worst case 20,000 tokens" in result.output and "nothing was run" in result.output


def test_a_cost_cap_needs_prices(tmp_path):
    args = ["eval", "run", "--cases", str(cases_file(tmp_path)), "--config", str(BASE),
            "--max-cost", "1"]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "needs prices" in result.output


def test_live_run_writes_records_and_score_rescores_them_offline(
    monkeypatch, tmp_path, make_result, make_work
):
    seen = {}

    @contextmanager
    def fake_open(cfg, creds, *, wrap_llm=None, http_client=None, today=None):
        seen["cache"] = cfg.openalex.cache_dir
        seen["http"] = http_client
        seen["today"] = today()
        llm = wrap_llm(StubLLMClient(search_then_answer()))
        search = FakeSearch(make_result([make_work("W1")]))
        yield ResearchAgent(llm, search, cfg.limits, cfg.reporting)

    monkeypatch.setattr("research_agent.cli._open_agent", fake_open)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef\n", encoding="utf-8")
    out = tmp_path / "run"
    cases = cases_file(tmp_path)
    args = ["eval", "run", "--live", "--yes", "--frozen-tools", "--out", str(out),
            "--cases", str(cases), "--config", str(BASE), "--env-file", str(env)]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "[PASS] crop" in result.output and "Summary: 1/1 passed" in result.output
    assert seen["cache"] == Path("evals/tool_cache/openalex") and seen["http"] is not None
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_calls"] == 2 and summary["input_tokens"] == 400
    (record,) = read_records(out)
    assert record["today"] == seen["today"].isoformat()  # pinned, so a replay can use it
    assert record["llm_calls"][0]["tools_hash"]

    scored = runner.invoke(app, ["eval", "score", str(out), "--cases", str(cases),
                                 "--config", str(BASE)])  # fmt: skip
    assert scored.exit_code == 0 and "[PASS] crop" in scored.output
    cases.write_text(cases.read_text(encoding="utf-8").replace("min_references = 1",
                                                               "min_references = 3"))
    rescored = runner.invoke(app, ["eval", "score", str(out), "--cases", str(cases),
                                   "--config", str(BASE)])  # fmt: skip
    assert rescored.exit_code == 1 and "at least 3 reference" in rescored.output


def test_list_shows_split_and_category(tmp_path):
    result = runner.invoke(app, ["eval", "list", "--cases", str(cases_file(tmp_path))])
    assert result.exit_code == 0
    assert "held" in result.output and "test" in result.output and "2 case(s)" in result.output
