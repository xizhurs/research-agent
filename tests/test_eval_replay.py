"""Phase 3: replaying stored runs through the current code (offline; no model is called)."""

import copy
import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from typer.testing import CliRunner

from research_agent.agent import ResearchAgent
from research_agent.cli import app
from research_agent.config import ReportingConfig
from research_agent.evaluation import (
    EvalCase,
    RecordingLLMClient,
    ReplayLLMClient,
    RunBudget,
    compare_answers,
    read_records,
    replay_date,
    replay_section,
    run_cases,
)
from research_agent.llm import StubLLMClient, text_response, tool_call_response

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"
REPORTING = ReportingConfig(thin_evidence_works=3, stale_after_years=3)
DAY = date(2026, 11, 13)
CASE = EvalCase("crop", "How are crops mapped?", min_references=1, category="literature")


class FakeSearch:
    def __init__(self, result):
        self.result = result

    def search(self, query, *, year_from=None, max_results=None):
        return self.result


def script():
    claims = [{"text": "Crops are mapped with CNNs.", "kind": "evidence", "sources": ["W1"]}]
    return [
        tool_call_response(("c1", "search_literature", {"query": "crop mapping"}),
                           input_tokens=100, output_tokens=20),
        text_response(json.dumps({"claims": claims, "limitations": ["Only abstracts."]}),
                      input_tokens=300, output_tokens=50),
    ]  # fmt: skip


def search(make_result, make_work, title="A title"):
    return FakeSearch(make_result([make_work("W1", title=title)], query="crop mapping"))


def record_run(tmp_path, make_result, make_work, limits):
    recorder = RecordingLLMClient(StubLLMClient(script()), RunBudget(10, 100_000))
    agent = ResearchAgent(recorder, search(make_result, make_work), limits, REPORTING,
                          today=lambda: DAY)  # fmt: skip
    return run_cases(
        [CASE], lambda case: agent.run(case.question), recorder, limits,
        out_dir=tmp_path / "run", run_id="r1", meta={"model": "m", "today": DAY.isoformat()},
    )  # fmt: skip


def replay(tmp_path, originals, search, limits, reporting=REPORTING, **agent_options):
    by_id = {r["case_id"]: r for r in originals}
    replayer = ReplayLLMClient()
    recorder = RecordingLLMClient(replayer, RunBudget(10**9, 10**15))
    agent = ResearchAgent(recorder, search, limits, reporting, today=lambda: DAY, **agent_options)

    def run(case):
        replayer.load(by_id[case.id])
        result = agent.run(case.question)
        replayer.finish()
        return result

    (record,) = run_cases(
        [CASE], run, recorder, limits, out_dir=tmp_path / "replay", run_id="r1-replay",
        meta={"mode": "replay"},
        annotate=lambda case, rec: {**rec, "replay": replay_section(by_id[case.id], rec)},
    )  # fmt: skip
    return record


def test_unchanged_code_replays_to_the_same_answer(tmp_path, make_result, make_work, limits):
    originals = record_run(tmp_path, make_result, make_work, limits)
    record = replay(tmp_path, originals, search(make_result, make_work), limits)
    assert record["status"] == "ok" and record["failures"] == []
    assert record["replay"] == {"of": "r1", "mode": "agentic", "original_failures": [],
                                "outcome": "same", "changes": []}  # fmt: skip
    assert [c["request_hash"] for c in record["llm_calls"]] == [
        c["request_hash"] for c in originals[0]["llm_calls"]
    ]
    assert record["answer"] == originals[0]["answer"] and record["stats"]["total_tokens"] == 470


def test_changed_tool_output_diverges_with_the_difference(tmp_path, make_result, make_work, limits):
    originals = record_run(tmp_path, make_result, make_work, limits)
    record = replay(tmp_path, originals, search(make_result, make_work, "B title"), limits)
    assert record["status"] == "diverged" and record["replay"]["outcome"] == "diverged"
    (failure,) = record["failures"]
    assert failure.startswith("diverged at call 2: tool result changed for search_literature c1")
    assert "was '" in failure and "B title" in failure
    assert len(record["llm_calls"]) == 1  # the diverging request was never answered


def test_a_changed_prompt_diverges_at_the_first_call(tmp_path, make_result, make_work, limits):
    originals = record_run(tmp_path, make_result, make_work, limits)
    record = replay(tmp_path, originals, search(make_result, make_work), limits,
                    rainfall=object())  # adds a tool and changes the instructions # fmt: skip
    assert record["failures"] == ["diverged at call 1: the instructions (prompt) changed"]


def test_a_change_after_the_model_is_reported_not_diverged(
    tmp_path, make_result, make_work, limits
):
    originals = record_run(tmp_path, make_result, make_work, limits)
    lenient = ReportingConfig(thin_evidence_works=3, stale_after_years=10)
    record = replay(tmp_path, originals, search(make_result, make_work), limits, lenient)
    assert record["status"] == "ok" and record["replay"]["outcome"] == "changed"
    (change,) = record["replay"]["changes"]
    assert change.startswith("limitations: - ")  # the 2023 work no longer counts as dated


def test_extra_or_missing_recorded_calls_diverge(tmp_path, make_result, make_work, limits):
    originals = record_run(tmp_path, make_result, make_work, limits)
    extra = copy.deepcopy(originals)
    extra[0]["llm_calls"].append(extra[0]["llm_calls"][-1])
    record = replay(tmp_path, extra, search(make_result, make_work), limits)
    assert record["failures"] == [
        "diverged at call 3: the run finished early (1 recorded model call(s) unused)"
    ]
    short = copy.deepcopy(originals)
    short[0]["llm_calls"].pop()
    record = replay(tmp_path, short, search(make_result, make_work), limits)
    assert record["failures"] == ["diverged at call 2: more model calls than recorded (1 recorded)"]


def test_replay_date_and_answer_comparison():
    assert replay_date({"today": "2026-01-02", "recorded_at": "2026-01-03T00:10:00+00:00"}) == (
        date(2026, 1, 2)
    )
    assert replay_date({"recorded_at": "2026-01-03T00:10:00+00:00"}) == date(2026, 1, 3)
    old = {"structured": True, "references": ["W1", "W2"], "data_references": [],
           "rejected_citations": [], "limitations": ["a"],
           "claims": [{"text": "t", "kind": "evidence", "sources": ["W1"]}]}  # fmt: skip
    new = {**old, "references": ["W1"], "limitations": ["a", "b"],
           "claims": [{"text": "t", "kind": "interpretation", "sources": []}]}  # fmt: skip
    assert compare_answers(old, old) == []
    assert compare_answers(old, new) == [
        "references: - W2",
        "limitations: + b",
        "claim 1: kind evidence -> interpretation; sources ['W1'] -> []",
    ]


# --- CLI -----------------------------------------------------------------------------------------


def test_cli_replay_needs_no_key_and_reports_outcomes(
    monkeypatch, tmp_path, make_result, make_work, limits
):
    originals = record_run(tmp_path, make_result, make_work, limits)
    aborted = {**originals[0], "case_id": "gone", "status": "aborted"}
    (tmp_path / "run" / "gone.json").write_text(json.dumps(aborted), encoding="utf-8")
    seen = {"title": "A title"}

    @contextmanager
    def fake_open(cfg, creds, *, llm=None, http_client=None, today=None, wrap_llm=None):
        seen["key"] = creds.openai_api_key
        seen["today"] = today()
        found = search(make_result, make_work, seen["title"])
        yield ResearchAgent(llm, found, cfg.limits, REPORTING, today=today)

    monkeypatch.setattr("research_agent.cli._open_agent", fake_open)
    cases = tmp_path / "cases.toml"
    cases.write_text('[[case]]\nid = "crop"\nquestion = "Q (edited)"\nmin_references = 1\n',
                     encoding="utf-8")  # fmt: skip
    out = tmp_path / "replayed"
    args = ["eval", "replay", str(tmp_path / "run"), "--out", str(out), "--cases", str(cases),
            "--config", str(BASE)]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "[NOT REPLAYED] gone: the original run ended 'aborted'" in result.output
    assert "[SAME] [PASS] crop" in result.output and "Replay: 1 same, 0 changed" in result.output
    assert seen["key"] is None and seen["today"] == DAY
    (record,) = read_records(out)
    assert record["question"] == CASE.question  # the recorded question, not the edited one
    assert record["mode"] == "replay" and record["today"] == DAY.isoformat()
    assert "replay of run r1: same" in (out / "report.md").read_text(encoding="utf-8")

    seen["title"] = "B title"
    result = CliRunner().invoke(app, [*args[:3], "--out", str(tmp_path / "r2"), *args[5:]])
    assert result.exit_code == 1
    assert "[DIVERGED] crop  call 2: tool result changed for search_literature c1" in result.output
    assert "Replay: 0 same, 0 changed, 1 diverged" in result.output


def test_replaying_skips_cases_dropped_from_the_case_file(tmp_path, make_result, make_work,
                                                          limits):  # fmt: skip
    originals = record_run(tmp_path, make_result, make_work, limits)
    cases = tmp_path / "cases.toml"
    cases.write_text('[[case]]\nid = "other"\nquestion = "Q"\n', encoding="utf-8")
    args = ["eval", "replay", str(tmp_path / "run"), "--cases", str(cases), "--config", str(BASE)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "[NOT REPLAYED] crop: no longer in the case file" in result.output
    assert "nothing to replay" in result.output and originals


def test_recorded_requests_carry_the_tool_definitions_hash(tmp_path, make_result, make_work,
                                                           limits):  # fmt: skip
    (record,) = record_run(tmp_path, make_result, make_work, limits)
    first, second = record["llm_calls"]
    assert first["tools_hash"] == second["tools_hash"] and record["today"] == DAY.isoformat()
