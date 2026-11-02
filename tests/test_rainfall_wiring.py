"""The rainfall tool's wiring into formatting, citations, limitations, budget, CLI and eval."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.agent import AgentResult, Answer, Budget, Claim, RunStats, estimate_tokens
from research_agent.agent.answer import parse_answer, validate_citations
from research_agent.agent.limitations import ALWAYS, SearchRecord, evidence_limitations
from research_agent.agent.tools import (
    RAINFALL_TOOL,
    ToolArgumentError,
    format_rainfall,
    parse_rainfall_args,
)
from research_agent.cli import app
from research_agent.evaluation import EvalCase, check_result, load_cases
from research_agent.rainfall import AmbiguousLocationError

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"
runner = CliRunner()
TODAY = date(2026, 11, 2)


@pytest.fixture
def r1(make_analysis):
    return replace(make_analysis(event_date=date(2004, 7, 1)), id="R1")


# --- tool schema, arguments and formatting ------------------------------------------------------


def test_tool_schema_requires_city_only():
    assert RAINFALL_TOOL.name == "analyze_extreme_rainfall"
    assert RAINFALL_TOOL.parameters["required"] == ["city"]
    assert set(RAINFALL_TOOL.parameters["properties"]) == {
        "city",
        "country_code",
        "region",
        "start_year",
        "end_year",
        "event_date",
    }


def test_parse_valid_and_minimal_arguments():
    full = parse_rainfall_args(
        {"city": "Paris", "country_code": "FR", "start_year": 1990, "event_date": "2003-08-12"}
    )
    assert (full.city, full.country_code, full.start_year) == ("Paris", "FR", 1990)
    assert full.event_date == date(2003, 8, 12) and full.region is None
    minimal = parse_rainfall_args({"city": "Paris", "region": None, "event_date": ""})
    assert minimal.event_date is None and minimal.region is None


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"city": " "},
        {"city": "Paris", "event_date": "2003-13-45"},
        {"city": "Paris", "event_date": 20030812},
        {"city": "Paris", "start_year": 1990.5},
        {"city": "Paris", "end_year": False},
        {"city": "Paris", "region": ["x"]},
    ],
)
def test_parse_rejects_unusable_arguments(arguments):
    with pytest.raises(ToolArgumentError):
        parse_rainfall_args(arguments)


def test_format_contains_every_number_the_model_may_quote(r1):
    text = format_rainfall(r1)
    assert text.startswith("[R1] Extreme rainfall")
    assert "Zhengzhou, Henan, China" in text
    assert "Valid years: 5 (2000-2004), 0 excluded." in text
    assert "Baseline 2000-2004 Rx1day (5 years): mean 30.0 mm, median 30.0 mm." in text
    assert "1) 2004-07-01 50.0 mm, 1.7x baseline mean" in text
    assert "Event 2004-07-01: 50.0 mm; rank 1 of 1,827 days" in text
    assert "would rank 1 of 5 annual maxima (percentile 80.0)" in text
    assert "Caveats: Values are ERA5 reanalysis" in text


def test_format_is_deterministic_and_compact(r1):
    assert format_rainfall(r1) == format_rainfall(r1)
    assert estimate_tokens(format_rainfall(r1)) < 700


def test_format_states_when_ranks_were_withheld(make_analysis):
    text = format_rainfall(
        replace(make_analysis(min_years=20, event_date=date(2004, 7, 1)), id="R2")
    )
    assert "ranks withheld: record too short" in text and "[R2]" in text


# --- citation validation across both kinds of evidence ------------------------------------------


def test_validate_accepts_work_and_analysis_ids(make_work, r1):
    claims = (Claim("Both.", "evidence", ("R1", "W1")),)
    checked, rejected, notes = validate_citations(claims, {"W1": make_work("W1")}, {"R1": r1})
    assert checked[0].sources == ("R1", "W1") and checked[0].kind == "evidence"
    assert rejected == () and notes == []


def test_validate_rejects_unknown_ids_and_downgrades_unsupported_evidence(r1):
    claims = (Claim("Made up.", "evidence", ("R2",)), Claim("Fine.", "evidence", ("R1",)))
    checked, rejected, notes = validate_citations(claims, {}, {"R1": r1})
    assert [c.kind for c in checked] == ["interpretation", "evidence"]
    assert rejected == ("R2",)
    assert any("R2" in n and "removed" in n for n in notes)
    assert any("shown as interpretation" in n for n in notes)


def test_analysis_ids_are_unknown_when_no_analyses_are_given():
    checked, rejected, _ = validate_citations((Claim("x", "evidence", ("R1",)),), {})
    assert rejected == ("R1",) and checked[0].kind == "interpretation"


def test_parse_answer_lists_data_references_in_order_of_first_citation(
    make_work, r1, make_analysis
):
    r2 = replace(make_analysis(), id="R2")
    reply = (
        '{"claims": [{"text": "a", "kind": "evidence", "sources": ["R2"]},'
        ' {"text": "b", "kind": "evidence", "sources": ["R1", "R2"]}], "limitations": []}'
    )
    answer = parse_answer(reply, {}, analyses={"R1": r1, "R2": r2})
    assert [a.id for a in answer.data_references] == ["R2", "R1"]
    assert answer.references == ()


def test_uncited_analyses_are_not_references(r1):
    reply = '{"claims": [{"text": "a", "kind": "interpretation", "sources": []}]}'
    assert parse_answer(reply, {}, analyses={"R1": r1}).data_references == ()


# --- code-derived limitations -------------------------------------------------------------------


def limits_for(searches, evidence, limits, reporting, **kwargs):
    return evidence_limitations(
        searches, evidence, limits=limits, reporting=reporting, today=TODAY, **kwargs
    )


def test_rainfall_only_run_has_rainfall_caveats_and_no_openalex_text(limits, reporting, r1):
    out = limits_for([], {}, limits, reporting, analyses=[r1], rainfall_calls=1)
    assert ALWAYS not in out and not any("literature" in x for x in out)
    assert out == list(r1.caveats)


def test_failed_rainfall_run_reports_missing_numerical_evidence(limits, reporting):
    out = limits_for([], {}, limits, reporting, rainfall_calls=1)
    assert out == [
        "A rainfall analysis was requested but produced no usable result, so no numerical "
        "evidence backs this answer."
    ]


def test_nothing_run_keeps_the_literature_wording(limits, reporting):
    out = limits_for([], {}, limits, reporting)
    assert out[0] == ALWAYS and any("No literature search was run" in x for x in out)


def test_both_tools_report_both_kinds_of_limitation(limits, reporting, make_work, r1):
    works = {"W1": make_work("W1", year=2025)}
    search = SearchRecord("flood", None, 10, 10, 1)
    out = limits_for([search], works, limits, reporting, analyses=[r1], rainfall_calls=1)
    assert ALWAYS in out and set(r1.caveats) <= set(out)


# --- budget -------------------------------------------------------------------------------------


def test_rainfall_budget_counts_and_caps(limits):
    b = Budget(limits, rainfall_enabled=True)
    assert b.rainfall_left == limits.max_rainfall_calls
    b.record_rainfall()
    b.record_rainfall(cached=True)
    assert (b.rainfall_calls, b.cached_rainfall) == (2, 1)
    assert b.rainfall_left == limits.max_rainfall_calls - 2


def test_rainfall_is_unavailable_when_not_enabled(limits):
    assert Budget(limits).rainfall_left == 0


def test_spent_searches_force_a_final_answer_only_when_rainfall_is_also_spent(limits):
    capped = replace(limits, max_search_calls=1, max_rainfall_calls=1)
    b = Budget(capped, rainfall_enabled=True)
    b.record_search()
    assert b.force_final_reason(100) is None  # a rainfall analysis is still allowed
    b.record_rainfall()
    assert "Search and rainfall-analysis limits reached" in b.force_final_reason(100)


def test_literature_only_budget_keeps_its_original_message(limits):
    capped = replace(limits, max_search_calls=1)
    b = Budget(capped)
    b.record_search()
    assert b.force_final_reason(100).startswith("Search limit reached")


# --- evaluation checker -------------------------------------------------------------------------


def eval_result(r1, *, claims, stats=None, references=()):
    answer = Answer(tuple(claims), tuple(references), ("note",), data_references=(r1,))
    return AgentResult(answer, stats or RunStats(2, 0, 1000, 100, 0, rainfall_calls=1))


def test_rainfall_citation_counts_as_a_resolved_reference(r1, limits):
    result = eval_result(r1, claims=[Claim("Peak 50 mm.", "evidence", ("R1",))])
    case = EvalCase("c", "Q", min_rainfall_calls=1, min_data_references=1, keywords_any=("mm",))
    assert check_result(case, result, limits) == []


def test_missing_rainfall_run_or_citation_fails_the_case(r1, limits):
    result = eval_result(
        r1,
        claims=[Claim("Nothing cited.", "interpretation", ())],
        stats=RunStats(1, 0, 10, 10, 0),
    )
    result = AgentResult(replace(result.answer, data_references=()), result.stats)
    case = EvalCase("c", "Q", min_rainfall_calls=1, min_data_references=1)
    failures = check_result(case, result, limits)
    assert any("rainfall analysis(es), ran 0" in f for f in failures)
    assert any("cited rainfall analysis" in f for f in failures)


def test_rainfall_call_cap_is_an_invariant(r1, limits):
    over = RunStats(2, 0, 10, 10, 0, rainfall_calls=limits.max_rainfall_calls + 1)
    result = eval_result(r1, claims=[Claim("x", "interpretation", ())], stats=over)
    assert any(
        "rainfall analyses (limit" in f for f in check_result(EvalCase("c", "Q"), result, limits)
    )


def test_shipped_cases_include_rainfall_cases():
    cases = {
        c.id: c for c in load_cases(Path(__file__).resolve().parents[1] / "evals" / "cases.toml")
    }
    assert cases["rainfall-event-zhengzhou"].min_data_references == 1
    assert cases["rainfall-and-literature"].min_searches == 1
    assert cases["rainfall-and-literature"].min_rainfall_calls == 1
    assert cases["rainfall-ambiguous-city"].max_evidence_claims == 0


# --- CLI ----------------------------------------------------------------------------------------


class FakeRainfallClient:
    outcome = None
    calls: list = []

    def __init__(self, config):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def analyze(self, city, **kwargs):
        FakeRainfallClient.calls.append({"city": city, **kwargs})
        if isinstance(FakeRainfallClient.outcome, Exception):
            raise FakeRainfallClient.outcome
        return FakeRainfallClient.outcome


@pytest.fixture
def fake_client(monkeypatch):
    FakeRainfallClient.calls = []
    monkeypatch.setattr("research_agent.cli.RainfallClient", FakeRainfallClient)
    return FakeRainfallClient


def test_rainfall_command_prints_analysis_provenance_and_source(fake_client, make_analysis):
    fake_client.outcome = make_analysis(event_date=date(2004, 7, 1))
    args = ["rainfall", "Zhengzhou", "--country-code", "CN", "--event-date", "2004-07-01"]
    result = runner.invoke(app, [*args, "--config", str(BASE)])
    assert result.exit_code == 0, result.output
    assert "[R1] Extreme rainfall" in result.output
    assert "Event 2004-07-01: 50.0 mm" in result.output
    assert "Request: https://archive.example/v1/archive?x=1" in result.output
    assert (
        "CC BY 4.0" in result.output and "(network; data 2000-01-01 to 2004-12-31)" in result.output
    )
    assert fake_client.calls[0]["country_code"] == "CN"
    assert fake_client.calls[0]["event_date"] == date(2004, 7, 1)


def test_rainfall_command_reports_ambiguity_with_exit_code_1(fake_client):
    fake_client.outcome = AmbiguousLocationError(
        "'Springfield' matches 4 places: 1) A. Do not guess."
    )
    result = runner.invoke(app, ["rainfall", "Springfield", "--config", str(BASE)])
    assert result.exit_code == 1
    assert "error: 'Springfield' matches 4 places" in result.output


def test_rainfall_command_rejects_a_bad_event_date_before_any_request(fake_client):
    result = runner.invoke(
        app, ["rainfall", "Paris", "--event-date", "last tuesday", "--config", str(BASE)]
    )
    assert result.exit_code == 1 and "YYYY-MM-DD" in result.output
    assert fake_client.calls == []


def test_ask_prints_a_data_section_and_the_rainfall_footer(monkeypatch, tmp_path, r1, make_work):
    answer = Answer(
        claims=(Claim("Peak was 50 mm.", "evidence", ("R1",)),),
        references=(),
        limitations=("ERA5 is not a gauge.",),
        data_references=(r1,),
    )
    stats = RunStats(2, 0, 400, 100, 0, rainfall_calls=1, cached_rainfall=1)
    monkeypatch.setattr("research_agent.cli._run_agent", lambda *a: AgentResult(answer, stats))
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef\n", encoding="utf-8")
    result = runner.invoke(app, ["ask", "q", "--config", str(BASE), "--env-file", str(env)])
    assert result.exit_code == 0, result.output
    assert "- (evidence) Peak was 50 mm. [R1]" in result.output
    assert "References" not in result.output  # no literature was cited
    assert "Data" in result.output
    assert "[R1] Zhengzhou, Henan, China: Rx1day 2000-2004. ERA5 test dataset" in result.output
    assert "1 rainfall analysis(es) (1 from cache), 500 tokens" in result.output


def test_format_trace_is_one_readable_line():
    from research_agent.cli import format_trace

    trace = (
        ("analyze_extreme_rainfall", "search_literature"),
        ("search_literature",),
        (),
    )
    assert format_trace(trace) == "1: rainfall, search | 2: search | 3: answer"


def test_ask_prints_tools_by_turn_on_stderr(monkeypatch, tmp_path):
    answer = Answer(claims=(Claim("x", "interpretation", ()),), references=(), limitations=("n",))
    stats = RunStats(
        2, 0, 10, 10, 0, rainfall_calls=1, tool_trace=(("analyze_extreme_rainfall",), ())
    )
    monkeypatch.setattr("research_agent.cli._run_agent", lambda *a: AgentResult(answer, stats))
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef\n", encoding="utf-8")
    result = runner.invoke(app, ["ask", "q", "--config", str(BASE), "--env-file", str(env)])
    assert "tools by turn: 1: rainfall | 2: answer" in result.output
