"""Phase 2 metrics on stored run records, case-declared checks, qrels and the report (offline)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.evaluation import (
    EvalCase,
    EvalError,
    ExpectedNumber,
    KeyPoint,
    Qrels,
    RunBudget,
    ScoreSettings,
    aggregate,
    check_record,
    flags,
    load_cases,
    load_conversations,
    load_qrels,
    score_record,
    write_scores,
)
from research_agent.evaluation.metrics import error_kind, redundant_calls, search_works

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"
ROOT_EVALS = BASE.parents[1] / "evals"
SETTINGS = ScoreSettings(redundant_query_jaccard=0.8, low_support_overlap=0.3)
RAIN = "analyze_rainfall_extremes"


def search_text(query, *works, again=()):
    lines = [f'Search "{query}": showing results of 100 matches.']
    meta = "(Ann Author, Journal, 2023, cited by 5)"
    lines += [f"[{w}] Title of {w} {meta}\nAbstract: x" for w in works]
    lines += [f"[{w}] (already shown in an earlier search)" for w in again]
    return "\n".join(lines)


def call(id, name, arguments, result, status="ok", result_id=None, turn=1):
    return {"turn": turn, "id": id, "name": name, "arguments": arguments, "status": status,
            "result_id": result_id, "result": result}  # fmt: skip


def llm_call(allow_tools=True, tokens=(1000, 100), latency=1.0):
    return {"allow_tools": allow_tools, "latency_s": latency,
            "usage": {"input": tokens[0], "output": tokens[1]}}  # fmt: skip


def make_record(*, tool_calls=(), claims=(), works=(), data=(), llm=None, iterations=None,
                status="ok", failures=(), searches=0, analyses=0):  # fmt: skip
    llm = list(llm or [llm_call(), llm_call()])
    answer = None
    stats = None
    if status == "ok":
        answer = {
            "structured": True,
            "claims": list(claims),
            "limitations": ["Only abstracts."],
            "references": [w["id"] for w in works],
            "data_references": [d["id"] for d in data],
            "rejected_citations": [],
            "works": list(works),
            "data": list(data),
        }
        stats = {
            "iterations": iterations or len(llm),
            "search_calls": searches,
            "rainfall_calls": analyses,
            "total_tokens": sum(c["usage"]["input"] + c["usage"]["output"] for c in llm),
            "tool_trace": [[c["name"] for c in tool_calls]],
        }
    return {"schema": 1, "case_id": "c", "category": "rainfall", "split": "dev",
            "status": status, "failures": list(failures), "wall_seconds": 3.0, "llm_calls": llm,
            "tool_calls": list(tool_calls), "answer": answer, "stats": stats}  # fmt: skip


def claim(text, *sources, kind="evidence"):
    return {"text": text, "kind": kind, "sources": list(sources)}


# --- tool metrics --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result, status, kind",
    [
        ("[R1] Rainfall ...", "ok", None),
        ("Error: 'city' is required.", "error", "invalid"),
        ("Error: no place named 'Atlantis' was found.", "error", "not_found"),
        ("Error: 'Springfield' is ambiguous ...", "ambiguous", "ambiguous"),
        ("Error: search limit reached; no more searches are allowed. Answer now.", "error",
         "budget_refused"),
        ("Error: result omitted because the retrieved-context budget is used up.", "error",
         "context_refused"),
        ("Error: the rainfall analysis failed (HTTP 503). You may retry once", "error", "failed"),
    ],
)  # fmt: skip
def test_tool_errors_are_classified_from_status_and_text(result, status, kind):
    assert error_kind(call("c1", RAIN, {}, result, status=status)) == kind


def test_redundant_calls_by_arguments_and_by_similar_query():
    calls = [
        call("a", "search_literature", {"query": "crop type mapping Sentinel-2"}, ""),
        call("b", "search_literature", {"query": "Sentinel-2 crop type  mapping"}, ""),
        call("c", RAIN, {"city": "Madrid", "scale": "yearly"}, ""),
        call("d", RAIN, {"city": " madrid ", "scale": "yearly"}, ""),
        call("e", RAIN, {"city": "Madrid", "scale": "daily"}, ""),
    ]
    found = redundant_calls(calls, 0.8)
    assert [(r["id"], r["repeats"], r["why"]) for r in found] == [
        ("b", "a", "similar query"),
        ("d", "c", "same arguments"),
    ]


def test_search_result_lines_give_new_and_repeated_works():
    new, again = search_works(search_text("q", "W1", "W2", again=("W9",)))
    assert new == [("W1", "Title of W1"), ("W2", "Title of W2")] and again == ["W9"]


# --- a whole record ------------------------------------------------------------------------------


def full_record():
    calls = [
        call("s1", "search_literature", {"query": "zhengzhou rain"}, search_text("q", "W1", "W2")),
        call("s2", "search_literature", {"query": "henan flood drivers"},
             search_text("q", "W3", again=("W1",)), turn=2),
        call("r1", RAIN, {"city": "Zhengzhou", "scale": "daily"},
             "[R1] Rx1day 79.0 mm on 2021-07-20, rank 1 of 76 years", result_id="R1", turn=2),
        call("r2", RAIN, {"city": "Atlantis"}, "Error: 'scale' is required.", status="error"),
    ]  # fmt: skip
    claims = [
        claim("Rx1day rank 1 of 76 years was 79.0 mm on 20 July 2021.", "R1"),
        claim("The rank 1 years value was 80.1 mm.", "R1"),
        claim("Typhoon moisture transport drove the storm.", "W3"),
        claim("Unrelated wording about glaciers melting quickly.", "W3"),
        claim("This is my interpretation.", kind="interpretation"),
    ]
    works = [{"id": "W3", "title": "Typhoon moisture transport and the Henan storm",
              "year": 2022, "abstract": "Moisture transport by a typhoon drove the storm."}]
    data = [{"id": "R1", "text": "[R1] Rx1day 79.0 mm on 2021-07-20, rank 1 of 76 years"}]
    llm = [llm_call(), llm_call(), llm_call(), llm_call(), llm_call(allow_tools=False)]
    return make_record(tool_calls=calls, claims=claims, works=works, data=data, llm=llm,
                       searches=2, analyses=2)  # fmt: skip


def test_score_of_a_complete_record(limits):
    case = EvalCase("c", "Q", key_points=(KeyPoint("rx1day value", ("79.0 mm",)),
                                          KeyPoint("drivers", ("monsoon",))))  # fmt: skip
    qrels = Qrels(relevant=frozenset({"W3"}), not_relevant=frozenset({"W2"}))
    s = score_record(full_record(), case, qrels, limits, SETTINGS)

    tools = s["tools"]
    assert tools["calls"] == 4 and tools["by_tool"] == {RAIN: 2, "search_literature": 2}
    assert tools["errors"]["invalid"] == 1 and tools["argument_validity"] == 0.75
    assert [u["id"] for u in tools["uncited"]] == ["s1"]  # W1, W2 shown, never cited
    assert tools["redundant"] == []

    t = s["trajectory"]
    assert (t["turns"], t["stop"], t["forced_by"]) == (5, "forced", "model turns")
    assert t["budget_left"]["model_turns"] == 0 and t["budget_left"]["searches"] == 1
    assert t["new_work_share_per_follow_up_search"] == [0.5]
    assert t["cited_from_follow_up_searches"] == 1  # W3 came from the second search

    r = s["retrieval"]
    assert (r["works_shown"], r["judged_shown"], r["precision"], r["recall"]) == (3, 2, 0.5, 1.0)
    assert r["unjudged"] == [{"id": "W1", "title": "Title of W1"}]

    (unsupported,) = s["numbers"]["unsupported"]
    assert unsupported["numbers"] == ["80.1"] and unsupported["sources"] == ["R1"]
    assert s["numbers"]["claims_checked"] == 2

    assert [c["text"] for c in s["support"]["low"]] == [
        "Unrelated wording about glaciers melting quickly."
    ]
    assert s["key_points"] == {"total": 2, "covered": ["rx1day value"], "coverage": 0.5}
    e = s["efficiency"]
    assert (e["model_calls"], e["tool_calls"], e["input_tokens"]) == (5, 4, 5000)
    assert e["tokens_per_cited_item"] == 2750 and e["cost_usd"] is None
    assert "1 claim(s) with unsupported numbers" in flags(s)
    assert "forced stop (model turns)" in flags(s)


def test_a_voluntary_stop_and_no_labels(limits):
    s = score_record(make_record(), None, None, limits, SETTINGS)
    assert s["trajectory"]["stop"] == "voluntary" and s["trajectory"]["forced_by"] is None
    assert s["retrieval"]["labelled"] is False and s["retrieval"]["precision"] is None
    assert s["key_points"]["coverage"] is None and flags(s) is None


def test_an_aborted_record_gets_only_efficiency(limits):
    record = make_record(status="aborted", failures=["aborted: budget"], llm=[llm_call()])
    s = score_record(record, None, None, limits, SETTINGS, RunBudget(1, 1, price_input_per_mtok=2))
    assert "tools" not in s and s["passed"] is False
    assert s["efficiency"]["cost_usd"] == 0.002 and flags(s) is None


def test_aggregate_and_report_are_deterministic(tmp_path, limits):
    qrels = Qrels(relevant=frozenset({"W3"}))
    scores = [score_record(full_record(), None, qrels, limits, SETTINGS),
              score_record(make_record(status="skipped", llm=[]), None, None, limits, SETTINGS)]
    agg = aggregate(scores)
    a = agg["all"]
    assert (a["cases"], a["passed"], a["forced_stops"], a["unsupported_numbers"]) == (2, 1, 1, 1)
    assert a["labelled_cases"] == 1 and a["mean_recall"] == 1.0
    assert set(agg["by_category"]) == {"rainfall"} and set(agg["by_split"]) == {"dev"}

    score_path, report_path = write_scores(tmp_path, scores, agg)
    first = score_path.read_bytes()
    write_scores(tmp_path, scores, aggregate(scores))
    assert score_path.read_bytes() == first
    report = report_path.read_text(encoding="utf-8")
    assert "| all | 2 | 1 |" in report
    assert "unsupported number(s) ['80.1']" in report
    assert "Unjudged works (label in evals/qrels.toml under [c])" in report
    assert "- W1 Title of W1" in report


# --- case-declared checks ------------------------------------------------------------------------


def test_expect_numbers_checks_the_tool_side_and_the_claims(limits):
    record = full_record()
    case = EvalCase("c", "Q", expect_numbers=(ExpectedNumber(79.0), ExpectedNumber(81.5)))
    tool_results = [c["result"] for c in record["tool_calls"]]
    failures = check_record(case, record["answer"], record["stats"], limits, tool_results)
    assert failures == [
        "expected value 81.5 (±0.05) not found in any tool result",
        "expected value 81.5 (±0.05) not found in any claim",
    ]
    # Without tool results (the live check_result path) only the claim side is checked.
    assert check_record(case, record["answer"], record["stats"], limits) == [
        "expected value 81.5 (±0.05) not found in any claim"
    ]


def test_max_tool_calls_and_min_key_points(limits):
    record = full_record()
    case = EvalCase("c", "Q", max_tool_calls=3, min_key_points=2,
                    key_points=(KeyPoint("a", ("79.0 mm",)), KeyPoint("b", ("monsoon",))))
    failures = check_record(case, record["answer"], record["stats"], limits)
    assert failures == [
        "requested 4 tool call(s) (case limit 3)",
        "covered 1 of 2 key point(s) (case minimum 2)",
    ]


def write(tmp_path, text, name="c.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_case_file_reads_the_new_keys(tmp_path):
    (case,) = load_cases(write(tmp_path, """[[case]]
id = "a"
question = "Q"
max_tool_calls = 4
expect_numbers = [{ value = -41.5 }, { value = 79, tolerance = 0.5 }]
key_points = [{ text = "the value", any_terms = ["79 MM", "79.0"] }]
min_key_points = 1
"""))
    assert case.expect_numbers == (ExpectedNumber(-41.5, 0.05), ExpectedNumber(79.0, 0.5))
    assert case.key_points == (KeyPoint("the value", ("79 mm", "79.0")),)
    assert (case.max_tool_calls, case.min_key_points) == (4, 1)


@pytest.mark.parametrize(
    "extra, message",
    [
        ('expect_numbers = [{ value = "x" }]', "numeric 'value'"),
        ("expect_numbers = [{ value = 1, tolerance = -1 }]", "tolerance"),
        ("expect_numbers = [{ value = 1, unit = 'mm' }]", "unknown keys"),
        ('key_points = [{ text = "t", any_terms = [] }]', "any_terms"),
        ('key_points = [{ text = "t", any_terms = ["a"] }]\nmin_key_points = 2', "min_key_points"),
        ("max_tool_calls = -1", "max_tool_calls"),
    ],
)
def test_case_file_rejects_bad_new_keys(tmp_path, extra, message):
    with pytest.raises(EvalError, match=message):
        load_cases(write(tmp_path, f'[[case]]\nid = "a"\nquestion = "Q"\n{extra}\n'))


def test_qrels_file(tmp_path):
    assert load_qrels(tmp_path / "missing.toml") == {}
    qrels = load_qrels(write(tmp_path, '[a]\nrelevant = ["W1"]\nnot_relevant = ["W2"]\n'))
    assert qrels == {"a": Qrels(frozenset({"W1"}), frozenset({"W2"}))}
    with pytest.raises(EvalError, match="both kinds"):
        load_qrels(write(tmp_path, '[a]\nrelevant = ["W1"]\nnot_relevant = ["W1"]\n'))
    with pytest.raises(EvalError, match="work IDs"):
        load_qrels(write(tmp_path, '[a]\nrelevant = ["R1"]\n'))


def test_shipped_files_load():
    root = BASE.parents[1] / "evals"
    cases = {c.id: c for c in load_cases(root / "cases.toml")}
    assert cases["temperature-cold-moscow"].expect_numbers == (ExpectedNumber(-41.5),)
    assert load_qrels(root / "qrels.toml") == {}


# --- CLI -----------------------------------------------------------------------------------------


def test_score_writes_the_report_and_flags_metrics(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "c.json").write_text(json.dumps(full_record()), encoding="utf-8")
    cases = write(
        tmp_path, '[[case]]\nid = "c"\nquestion = "Q"\nexpect_numbers = [{ value = 79.0 }]\n'
    )
    qrels = write(tmp_path, '[c]\nrelevant = ["W3"]\n[gone]\nrelevant = ["W5"]\n', "q.toml")
    args = ["eval", "score", str(run), "--cases", str(cases), "--qrels", str(qrels),
            "--config", str(BASE)]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "[PASS] c" in result.output
    assert "~ 1 claim(s) with unsupported numbers" in result.output
    assert "Metrics: argument validity 0.75" in result.output
    assert "labels unknown case(s): gone" in result.output
    assert (run / "report.md").exists() and (run / "score.json").exists()

    cases.write_text(cases.read_text(encoding="utf-8").replace("79.0", "81.5"), encoding="utf-8")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1 and "81.5 (±0.05) not found in any tool result" in result.output


# --- reviewed key points (evals/key_points.toml) -------------------------------------------------

KP_CASES = '[[case]]\nid = "a"\nquestion = "Q"\n[[case]]\nid = "b"\nquestion = "Q"\n'
KP_CONV = '[[conversation]]\nid = "c"\n[[conversation.turn]]\nquestion = "Q"\n'


def kp_entry(case_id, status, extra=""):
    return (f'["{case_id}"]\nstatus = "{status}"\n{extra}'
            f'[["{case_id}".point]]\ntext = "A fact."\nany_terms = ["Fact"]\n'
            f'source = "R1"\n')  # fmt: skip


def test_only_approved_key_points_are_attached(tmp_path):
    write(tmp_path, kp_entry("a", "approved", "min_key_points = 1\n") + kp_entry("b", "draft")
          + kp_entry("c#1", "approved"), "key_points.toml")  # fmt: skip
    a, b = load_cases(write(tmp_path, KP_CASES))
    assert a.key_points == (KeyPoint("A fact.", ("fact",)),) and a.min_key_points == 1
    assert b.key_points == ()  # a draft is never used
    (conv,) = load_conversations(write(tmp_path, KP_CONV, "conv.toml"))
    assert conv.turns[0].key_points == (KeyPoint("A fact.", ("fact",)),)


@pytest.mark.parametrize(
    "entry, message",
    [
        (kp_entry("a", "maybe"), "status must be one of"),
        (kp_entry("a", "approved", "min_key_points = 3\n"), "min_key_points exceeds"),
        ('["a"]\nstatus = "approved"\n', "needs at least one"),
        (kp_entry("a", "approved").replace('source = "R1"', 'note = "x"'), "unknown keys"),
    ],
)
def test_key_point_file_errors(tmp_path, entry, message):
    write(tmp_path, entry, "key_points.toml")
    with pytest.raises(EvalError, match=message):
        load_cases(write(tmp_path, KP_CASES))


def test_key_points_may_not_be_defined_twice(tmp_path):
    write(tmp_path, kp_entry("a", "approved"), "key_points.toml")
    inline = KP_CASES.replace('id = "a"\nquestion = "Q"\n',
                              'id = "a"\nquestion = "Q"\n'
                              'key_points = [{ text = "t", any_terms = ["t"] }]\n')  # fmt: skip
    with pytest.raises(EvalError, match="in both the case file and key_points.toml"):
        load_cases(write(tmp_path, inline))


def test_shipped_key_point_drafts_are_valid_and_unused():
    import tomllib

    raw = tomllib.loads((ROOT_EVALS / "key_points.toml").read_text(encoding="utf-8"))
    assert {v["status"] for v in raw.values()} == {"draft"}  # all await the owner's review
    cases = load_cases(ROOT_EVALS / "cases.toml")
    turns = [t for c in load_conversations(ROOT_EVALS / "conversations.toml") for t in c.turns]
    assert set(raw) <= {c.id for c in cases} | {t.id for t in turns}
    assert not any(c.key_points for c in [*cases, *turns])
