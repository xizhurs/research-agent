"""Phase 4a: the fixed-search baseline and paired run comparison (offline; stubbed model)."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.agent import ResearchAgent
from research_agent.cli import app
from research_agent.config import ReportingConfig
from research_agent.evaluation import (
    FIXED_CALL_ID,
    EvalCase,
    EvalError,
    FixedSearchLLMClient,
    RecordingLLMClient,
    RunBudget,
    ScoreSettings,
    compare_scores,
    fixed_query,
    load_cases,
    read_records,
    render_comparison,
    run_cases,
    score_record,
)
from research_agent.llm import StubLLMClient, UserMessage, text_response

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"
REPORTING = ReportingConfig(thin_evidence_works=3, stale_after_years=3)
SETTINGS = ScoreSettings(0.8, 0.3)
CASE = EvalCase("crop", "How are crops mapped?", min_references=1, category="literature")


class FakeSearch:
    def __init__(self, result):
        self.result = result
        self.queries = []

    def search(self, query, *, year_from=None, max_results=None):
        self.queries.append(query)
        return self.result


def final_answer(tokens=(300, 50)):
    claims = [{"text": "Crops are mapped with CNNs.", "kind": "evidence", "sources": ["W1"]}]
    return text_response(json.dumps({"claims": claims, "limitations": ["Only abstracts."]}),
                         input_tokens=tokens[0], output_tokens=tokens[1])  # fmt: skip


def test_fixed_query_defaults_to_the_question(tmp_path):
    assert fixed_query(CASE) == CASE.question
    path = tmp_path / "c.toml"
    path.write_text('[[case]]\nid = "a"\nquestion = "Q"\nfixed_query = " crop mapping "\n',
                    encoding="utf-8")  # fmt: skip
    (case,) = load_cases(path)
    assert fixed_query(case) == "crop mapping"
    path.write_text('[[case]]\nid = "a"\nquestion = "Q"\nfixed_query = ""\n', encoding="utf-8")
    with pytest.raises(EvalError, match="fixed_query"):
        load_cases(path)


def test_the_first_call_is_a_scripted_search_and_the_next_has_tools_off():
    stub = StubLLMClient([final_answer()])
    fixed = FixedSearchLLMClient(stub)
    with pytest.raises(RuntimeError, match="start"):
        fixed.complete(instructions="i", messages=[], tools=[], max_output_tokens=10)
    fixed.start("crop mapping")
    first = fixed.complete(instructions="i", messages=[UserMessage("q")], tools=[],
                           max_output_tokens=10)  # fmt: skip
    (call,) = first.tool_calls
    assert (call.id, call.name, call.arguments) == (FIXED_CALL_ID, "search_literature",
                                                    {"query": "crop mapping"})  # fmt: skip
    assert first.usage.input_tokens == 0 and stub.calls == []
    fixed.complete(instructions="i", messages=[], tools=[], max_output_tokens=10)
    assert stub.calls[0].allow_tools is False


def fixed_run(tmp_path, make_result, make_work, limits):
    search = FakeSearch(make_result([make_work("W1")], query=CASE.question))
    recorder = RecordingLLMClient(StubLLMClient([final_answer()]), RunBudget(10, 100_000))
    fixed = FixedSearchLLMClient(recorder)
    agent = ResearchAgent(fixed, search, limits, REPORTING)

    def run(case):
        fixed.start(fixed_query(case))
        return agent.run(case.question)

    records = run_cases(
        [CASE], run, recorder, limits, out_dir=tmp_path / "fixed", run_id="f1",
        meta={"mode": "fixed"},
        annotate=lambda case, record: {**record, "fixed_query": fixed_query(case)},
    )  # fmt: skip
    return records, search


def test_a_fixed_run_searches_once_with_the_question(tmp_path, make_result, make_work, limits):
    (record,), search = fixed_run(tmp_path, make_result, make_work, limits)
    assert search.queries == [CASE.question]
    assert record["status"] == "ok" and record["failures"] == [] and record["mode"] == "fixed"
    assert len(record["llm_calls"]) == 1 and record["llm_calls"][0]["allow_tools"] is False
    assert record["stats"]["search_calls"] == 1 and record["answer"]["references"] == ["W1"]
    (tool,) = record["tool_calls"]
    assert tool["id"] == FIXED_CALL_ID and tool["status"] == "ok"
    score = score_record(record, CASE, None, limits, SETTINGS)
    assert score["trajectory"]["stop"] == "fixed"  # tools off by design, not a forced stop
    assert score["efficiency"]["model_calls"] == 1


def scored(case_id, passed=True, tokens=1000, cited=2, unsupported=0, seconds=2.0):
    return {
        "case_id": case_id, "passed": passed, "tools": {},
        "key_points": {"coverage": None},
        "retrieval": {"precision": None, "recall": None, "works_cited": cited, "works_shown": 8},
        "numbers": {"unsupported": [{"numbers": ["1"]}] * unsupported},
        "support": {"low": []},
        "efficiency": {"input_tokens": tokens, "output_tokens": 0, "model_calls": 2,
                       "wall_seconds": seconds, "cost_usd": None},
    }  # fmt: skip


def test_compare_counts_improved_same_and_worse_by_direction():
    a = [scored("x", passed=False, tokens=4000), scored("y", tokens=6000, unsupported=1),
         scored("only-a"), {"case_id": "z", "passed": False, "efficiency": {}}]
    b = [scored("x", tokens=8000), scored("y", tokens=6000), scored("only-b"),
         scored("z")]  # fmt: skip
    result = compare_scores(a, b)
    assert result["paired_cases"] == ["x", "y"]
    assert result["only_in_a"] == ["only-a"] and result["only_in_b"] == ["only-b"]
    assert result["not_completed_in_both"] == ["z"]
    m = result["measures"]
    assert (m["passed"]["improved"], m["passed"]["same"], m["passed"]["worse"]) == (1, 1, 0)
    assert (m["tokens"]["improved"], m["tokens"]["worse"], m["tokens"]["mean_b"]) == (0, 1, 7000)
    assert m["unsupported_numbers"]["improved"] == 1
    assert m["works_cited"]["improved"] is None and m["works_cited"]["same"] == 2  # neutral
    assert m["cost_usd"]["paired"] == 0
    text = render_comparison(result, "fixed", "agentic")
    assert "| tokens | lower is better | 2 | 5000 | 7000 | 0 | 1 | 1 |" in text
    assert "| x | 0 -> 1 | 4000 -> 8000 |" in text and "Only in A: only-a" in text


def test_cli_fixed_run_then_compare_and_replay(monkeypatch, tmp_path, make_result, make_work):
    search_result = make_result([make_work("W1")])

    @contextmanager
    def fake_open(cfg, creds, *, wrap_llm=None, http_client=None, today=None, llm=None):
        model = llm if llm is not None else wrap_llm(StubLLMClient([final_answer((900, 90))]))
        yield ResearchAgent(model, FakeSearch(search_result), cfg.limits, REPORTING,
                            today=today)  # fmt: skip

    monkeypatch.setattr("research_agent.cli._open_agent", fake_open)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef\n", encoding="utf-8")
    cases = tmp_path / "cases.toml"
    cases.write_text(
        '[[case]]\nid = "crop"\nquestion = "How are crops mapped?"\nmin_references = 1\n'
        'category = "literature"\n[[case]]\nid = "rain"\nquestion = "Q"\ncategory = "rainfall"\n',
        encoding="utf-8",
    )
    common = ["--cases", str(cases), "--config", str(BASE)]
    fixed_dir = tmp_path / "fixed"
    result = CliRunner().invoke(app, ["eval", "run", "--live", "--yes", "--mode", "fixed",
                                      "--out", str(fixed_dir), "--env-file", str(env),
                                      *common])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "(1 other(s) left out)" in result.output
    assert "Plan (fixed): 1 case(s): crop" in result.output
    (record,) = read_records(fixed_dir)
    assert record["mode"] == "fixed" and record["fixed_query"] == "How are crops mapped?"

    replayed = CliRunner().invoke(app, ["eval", "replay", str(fixed_dir), "--out",
                                        str(tmp_path / "replayed"), *common])  # fmt: skip
    assert replayed.exit_code == 0, replayed.output
    assert "[SAME] [PASS] crop" in replayed.output

    agentic_dir = tmp_path / "agentic"
    agentic = dict(record, mode="agentic", run_id="a1")
    agentic["llm_calls"] = [*record["llm_calls"], *record["llm_calls"]]  # twice the tokens
    agentic_dir.mkdir()
    (agentic_dir / "crop.json").write_text(json.dumps(agentic), encoding="utf-8")
    out = tmp_path / "compare.md"
    compared = CliRunner().invoke(app, ["eval", "compare", str(fixed_dir), str(agentic_dir),
                                        "--out", str(out), *common])  # fmt: skip
    assert compared.exit_code == 0, compared.output
    assert "A = fixed (fixed), B = agentic (agentic)" in compared.output
    assert "| tokens | lower is better | 1 | 990 | 1980 | 0 | 0 | 1 |" in compared.output
    assert out.read_text(encoding="utf-8").startswith("# Comparison")

    empty = tmp_path / "empty"
    empty.mkdir()
    nothing = CliRunner().invoke(app, ["eval", "compare", str(fixed_dir), str(empty), *common])
    assert nothing.exit_code == 1 and "no case completed in both runs" in nothing.output
