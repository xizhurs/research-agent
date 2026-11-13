"""Phase 6: importing Langfuse traces as run records (offline; the fixture holds real rows).

``tests/fixtures/langfuse_observations.json`` is the raw ``GET /api/public/v2/observations``
output (all field groups, ``expandMetadata=attributes``) of two eval traces from the owner's
Langfuse project, fetched read-only on 2026-11-13 with the owner's OK: the
``rainfall-event-zhengzhou`` case and turn 3 of ``conv-amsterdam-rotterdam`` ("Compare the
two."). The project ID is replaced by a placeholder.
"""

import copy
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.evaluation import (
    EvalCase,
    LangfuseImportError,
    LangfuseReader,
    case_matcher,
    fetch_traces,
    load_cases,
    load_conversations,
    read_records,
    record_failures,
    rescore,
    trace_record,
)
from research_agent.evaluation.langfuse_import import _blocks, _work

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "base.toml"
ROWS = json.loads((ROOT / "tests/fixtures/langfuse_observations.json").read_text("utf-8"))["data"]
CASES = load_cases(ROOT / "evals/cases.toml") + [
    t for c in load_conversations(ROOT / "evals/conversations.toml") for t in c.turns
]


def trace(root_name):
    trace_id = next(r["traceId"] for r in ROWS if r["name"] == root_name)
    return [copy.deepcopy(r) for r in ROWS if r["traceId"] == trace_id]


def convert(rows, cases=CASES):
    return trace_record(rows, run_id="lf", case_for=case_matcher(cases))


# --- converting a trace --------------------------------------------------------------------------


def test_an_eval_case_trace_becomes_a_scoreable_record(limits):
    record, why = convert(trace("eval.case"))
    assert why is None
    assert (record["case_id"], record["category"], record["mode"], record["status"]) == (
        "rainfall-event-zhengzhou", "rainfall", "imported", "ok"
    )
    assert [c["usage"] for c in record["llm_calls"]] == [{"input": 2903, "output": 49},
                                                         {"input": 3445, "output": 204}]
    assert record["llm_calls"][0]["latency_s"] == 4.49 and record["llm_calls"][0]["allow_tools"]
    assert record["llm_calls"][0]["request_hash"] is None  # not in a trace: not replayable
    assert record["stats"]["tool_trace"] == [["analyze_rainfall_extremes[wet]"], []]
    assert (record["stats"]["iterations"], record["stats"]["rainfall_calls"],
            record["stats"]["total_tokens"]) == (2, 1, 6601)  # fmt: skip
    (tool,) = record["tool_calls"]
    assert (tool["name"], tool["status"], tool["result_id"], tool["turn"]) == (
        "analyze_rainfall_extremes", "ok", "R1", 1
    )
    assert tool["arguments"]["event_date"] == "2021-07-20"
    (data,) = record["answer"]["data"]
    assert data["id"] == "R1" and data["text"].startswith("[R1] Extreme rainfall (Rx1day")
    assert len(record["prompt_hash"]) == 16 and record["model"] == "gpt-6-luna"
    assert record["source"]["root"] == "eval.case" and "conversation" not in record
    case = next(c for c in CASES if c.id == "rainfall-event-zhengzhou")
    assert record_failures(case, record, limits) == []  # incl. expect_numbers 79.0 both sides


def test_a_conversation_turn_keeps_relation_context_and_recall(limits):
    record, _ = convert(trace("eval.turn"))
    assert record["case_id"] == "conv-amsterdam-rotterdam#3"
    conv = record["conversation"]
    assert (conv["id"], conv["turn"], conv["relation"], conv["relation_source"]) == (
        "conv-amsterdam-rotterdam", 3, "compare", "model"
    )
    assert (conv["context_tokens"], conv["context_turns"], conv["recalled"]) == (451, 2,
                                                                               ["R2", "R3"])
    assert conv["state"] is None  # not in the trace: state expectations are not judged
    assert [d["id"] for d in record["answer"]["data"]] == ["R2", "R3"]
    assert record["answer"]["data"][0]["text"].startswith("[R2] Annual precipitation totals")
    case = next(c for c in CASES if c.id == "conv-amsterdam-rotterdam#3")
    assert record_failures(case, record, limits) == []


def test_an_unmatched_trace_is_imported_with_invariant_checks(limits):
    record, _ = convert(trace("eval.case"), cases=[])
    assert record["case_id"].startswith("lf-") and record["category"] == "imported"
    (rescored,) = rescore([record], [], limits)
    assert rescored["failures"] == []  # invariants only, not "case no longer in the file"


def test_traces_that_cannot_be_imported_are_skipped():
    no_content = trace("eval.case")
    for row in no_content:
        row["input"] = row["output"] = None
    assert convert(no_content) == (None, "no content (traced with "
                                         "RESEARCH_AGENT_TRACE_CONTENT=false?)")
    ui_search = [{**trace("eval.case")[0], "name": "ui.search", "parentObservationId": None}]
    assert convert(ui_search)[1] == "not an agent trace (root 'ui.search')"
    orphans = [{**r, "parentObservationId": "x"} for r in trace("eval.case")]
    assert convert(orphans)[1].startswith("no root observation")


def test_a_failed_run_is_recorded_as_failed():
    rows = trace("eval.case")
    for row in rows:
        if row["name"] == "invoke_agent":
            row.update(output=None, level="ERROR", statusMessage="LLMRateLimitError: quota")
    record, _ = convert(rows)
    assert record["status"] == "failed" and record["error"] == "LLMRateLimitError: quota"
    assert record["answer"] is None and record["failures"] == ["run failed: LLMRateLimitError: "
                                                               "quota"]


def test_search_results_give_titles_and_abstracts():
    text = ('Search "q": showing results of 9 matches.\n'
            "[W1] Crop mapping (with CNNs) (Ann Author, J, 2023, cited by 5)\n"
            "Abstract: CNNs map crops.\n[W2] (already shown in an earlier search)")  # fmt: skip
    blocks = _blocks(text)
    assert _work(blocks["W1"]) == {"id": "W1", "title": "Crop mapping (with CNNs)",
                                   "abstract": "CNNs map crops.", "year": None}
    assert blocks["W2"] == "[W2] (already shown in an earlier search)"


# --- reading the API -----------------------------------------------------------------------------


def reader_with(handler, sleeps=None):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep = (sleeps.append if sleeps is not None else lambda s: None)
    return LangfuseReader("https://lf.example", "pk-lf-test-0001", "sk-lf-test-0001",
                          http_client=client, sleep=sleep)  # fmt: skip


def test_reader_follows_the_cursor_and_retries_rate_limits():
    seen, sleeps = [], []

    def handler(request):
        seen.append(dict(request.url.params))
        assert request.headers["authorization"].startswith("Basic ")
        if len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        if "cursor" not in request.url.params:
            return httpx.Response(200, json={"data": [{"id": "a"}], "meta": {"cursor": "c1"}})
        return httpx.Response(200, json={"data": [{"id": "b"}], "meta": {}})

    reader = reader_with(handler, sleeps)
    assert [r["id"] for r in reader.observations({"traceId": "t"})] == ["a", "b"]
    assert sleeps == [2.0] and seen[-1]["cursor"] == "c1" and seen[-1]["limit"] == "1000"
    assert request_count(reader) == 3


def request_count(reader):
    return reader.requests


def test_reader_reports_rejected_keys_without_them():
    reader = reader_with(lambda request: httpx.Response(401, text="bad key sk-lf-test-0001"))
    with pytest.raises(LangfuseImportError) as exc:
        list(reader.observations({}))
    assert "401" in str(exc.value) and "sk-lf" not in str(exc.value)


def fixture_handler(request):
    """Langfuse stand-in: roots for isRootObservation=true, else every row (newest first)."""
    params = request.url.params
    assert params["fromStartTime"] and params["toStartTime"]
    rows = sorted(ROWS, key=lambda r: r["startTime"], reverse=True)
    if params.get("isRootObservation") == "true":
        rows = [r for r in rows if not r["parentObservationId"]]
    if params.get("isRootObservation") == "true":
        if "filter" in params:  # tags are on root rows only (as on the real API)
            assert json.loads(params["filter"])[0]["value"] == ["eval"]
    else:
        assert "filter" not in params  # a tag filter would drop every child row
        rows = [r for r in rows if params["fromStartTime"] <= r["startTime"]
                < params["toStartTime"]]
    return httpx.Response(200, json={"data": rows, "meta": {}})


def test_fetch_traces_groups_rows_of_the_newest_traces():
    traces = fetch_traces(reader_with(fixture_handler), "2026-11-06T00:00:00Z",
                          "2026-11-14T00:00:00Z", "eval", 50)
    assert [len(rows) for rows in traces.values()] == [5, 5]
    newest = fetch_traces(reader_with(fixture_handler), "a", "b", "eval", 1)
    (rows,) = newest.values()
    assert any(r["name"] == "eval.turn" for r in rows)  # the 2026-11-13 turn is newer


# --- CLI -----------------------------------------------------------------------------------------


def test_cli_imports_scores_and_refuses_replay(monkeypatch, tmp_path):
    monkeypatch.setattr("research_agent.cli._langfuse_reader",
                        lambda env: reader_with(fixture_handler))  # fmt: skip
    out = tmp_path / "imported"
    common = ["--config", str(BASE)]
    result = CliRunner().invoke(app, ["eval", "import-langfuse", "--since", "2026-11-06",
                                      "--until", "2026-11-14", "--out", str(out), *common])
    assert result.exit_code == 0, result.output
    assert "Langfuse: 2 traces tagged 'eval' from 2026-11-06 to 2026-11-14" in result.output
    assert "[PASS] rainfall-event-zhengzhou" in result.output
    assert "[PASS] conv-amsterdam-rotterdam#3" in result.output
    assert {r["case_id"] for r in read_records(out)} == {"rainfall-event-zhengzhou",
                                                        "conv-amsterdam-rotterdam#3"}
    assert (out / "report.md").exists() and (out / "summary.json").exists()

    scored = CliRunner().invoke(app, ["eval", "score", str(out), *common])
    assert scored.exit_code == 0, scored.output

    replayed = CliRunner().invoke(app, ["eval", "replay", str(out), *common])
    assert "[NOT REPLAYED] rainfall-event-zhengzhou: imported from Langfuse" in replayed.output
    assert replayed.exit_code == 1 and "nothing to replay" in replayed.output


def test_cli_needs_the_langfuse_keys(tmp_path, monkeypatch):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    result = CliRunner().invoke(app, ["eval", "import-langfuse", "--env-file", str(env),
                                      "--config", str(BASE)])  # fmt: skip
    assert result.exit_code == 1 and "set LANGFUSE_PUBLIC_KEY" in result.output
    bad = CliRunner().invoke(app, ["eval", "import-langfuse", "--since", "13-11-2026",
                                   "--config", str(BASE)])  # fmt: skip
    assert bad.exit_code == 1 and "--since must be a date" in bad.output


def test_blank_case_gets_only_the_invariants(limits):
    record, _ = convert(trace("eval.case"), cases=[])
    assert record_failures(EvalCase(record["case_id"], record["question"]), record, limits) == []
