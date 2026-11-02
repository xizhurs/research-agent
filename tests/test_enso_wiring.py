"""The ENSO tool's wiring into formatting, arguments, citations, CLI and eval."""

from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.agent import AgentResult, Answer, Claim, RunStats
from research_agent.agent.tools import (
    ENSO_TOOL,
    ToolArgumentError,
    format_enso,
    parse_enso_args,
)
from research_agent.cli import app, format_trace
from research_agent.enso import EnsoApiError
from research_agent.evaluation import EvalCase, EvalError, check_result, load_cases

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"
runner = CliRunner()
ENSO = "analyze_enso_events"


@pytest.fixture
def e1(make_enso_analysis):
    return replace(make_enso_analysis(), id="R1")


# --- tool schema and arguments ----------------------------------------------------------------


def test_tool_schema_has_no_required_argument():
    assert ENSO_TOOL.name == ENSO
    assert ENSO_TOOL.parameters["required"] == []
    props = ENSO_TOOL.parameters["properties"]
    assert set(props) == {"phase", "start_year", "end_year", "year"}
    assert props["phase"]["enum"] == ["both", "el_nino", "la_nina"]
    assert "RONI" in ENSO_TOOL.description and "search_literature" in ENSO_TOOL.description


def test_parse_defaults_and_valid_arguments():
    default = parse_enso_args({})
    assert (default.phase, default.start_year, default.end_year, default.year) == (
        "both",
        None,
        None,
        None,
    )
    full = parse_enso_args({"phase": "la_nina", "start_year": 1990, "end_year": 2010, "year": 1999})
    assert (full.phase, full.start_year, full.end_year, full.year) == ("la_nina", 1990, 2010, 1999)
    assert parse_enso_args({"phase": None, "year": None}).phase == "both"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"phase": "warm"}, "'phase' must be"),
        ({"phase": 1}, "'phase' must be a string"),
        ({"year": "2010"}, "'year' must be an integer"),
        ({"year": True}, "'year' must be an integer"),
        ({"start_year": 1990.5}, "'start_year' must be an integer"),
        ({"end_year": "x"}, "'end_year' must be an integer"),
    ],
)
def test_parse_rejects_bad_arguments(arguments, message):
    with pytest.raises(ToolArgumentError, match=message):
        parse_enso_args(arguments)


# --- formatting (exact strings: these are the numbers the model quotes) ------------------------


def test_format_lists_events_with_exact_numbers(e1):
    lines = format_enso(e1).splitlines()
    assert lines[:7] == [
        "[R1] ENSO events from NOAA CPC RONI; record DJF 2000 to AMJ 2001; events starting "
        "2000-2001.",
        "Rule: index at or above +0.5 (El Nino) or at or below -0.5 (La Nina) for at least 5 "
        "consecutive overlapping 3-month seasons. Strength by peak value (convention): weak, "
        "moderate from 1, strong from 1.5, very strong from 2.",
        "Source: https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt",
        "Events: 1 El Nino, 1 La Nina.",
        "- El Nino FMA 2000 to JAS 2000 (6 seasons), peak +1.40 (AMJ 2000), moderate",
        "- La Nina OND 2000 to FMA 2001 (5 seasons), peak -1.80 (DJF 2001), strong",
        lines[6],
    ]
    assert lines[6].startswith("Caveats: ENSO events come from NOAA CPC's Relative Oceanic Nino")
    assert len(lines) == 7


def test_format_phase_filter_counts_only_the_selected_phase(make_enso_analysis):
    text = format_enso(replace(make_enso_analysis(phase="la_nina"), id="R1"))
    assert "Events: 1 La Nina." in text and "El Nino FMA 2000" not in text


def test_format_with_no_events_says_so(make_enso_analysis):
    text = format_enso(replace(make_enso_analysis([0.1] * 20), id="R3"))
    assert text.startswith("[R3] ENSO events")
    assert "Events: 0 El Nino, 0 La Nina." in text and "\n- " not in text


def test_format_marks_a_pending_run_and_both_record_edges(make_enso_analysis):
    pending = format_enso(replace(make_enso_analysis([0, 0.7, 0.7, 0.7]), id="R1"))
    assert (
        "Not yet an event: El Nino JFM 2000 to MAM 2000 (3 seasons), peak +0.70 (JFM 2000), "
        "ongoing: the record ends inside it; fewer than 5 seasons so far."
    ) in pending
    single = format_enso(replace(make_enso_analysis([0, 0.7]), id="R1"))
    assert "(1 season)," in single
    edges = format_enso(replace(make_enso_analysis([0.8] * 6), id="R1"))
    assert (
        "- El Nino DJF 2000 to MJJ 2000 (6 seasons), peak +0.80 (DJF 2000), weak, the record "
        "starts inside it so its start may be cut, ongoing: the record ends inside it"
    ) in edges


def test_format_focus_year_block(make_enso_analysis):
    text = format_enso(replace(make_enso_analysis(year=2000), id="R1"))
    assert "Year 2000 seasonal values: DJF +0.00, JFM +0.10, FMA +0.60," in text
    assert "AMJ +1.40" in text and text.count("Events overlapping 2000: ") == 1
    assert (
        "Events overlapping 2000: El Nino FMA 2000 to JAS 2000 (6 seasons), peak +1.40 "
        "(AMJ 2000), moderate; La Nina OND 2000 to FMA 2001 (5 seasons), peak -1.80 "
        "(DJF 2001), strong"
    ) in text


def test_format_focus_on_a_neutral_year_and_on_short_runs(make_enso_analysis):
    neutral = format_enso(replace(make_enso_analysis([0.1] * 30, year=2001), id="R1"))
    assert "No El Nino or La Nina event overlaps 2001." in neutral
    short = format_enso(replace(make_enso_analysis([0, 0.7, 0.7, 0.7, 0, 0], year=2000), id="R1"))
    assert "At-threshold runs in 2000 too short to be events: El Nino JFM 2000 to MAM 2000" in short
    assert "No El Nino or La Nina event overlaps 2000." in short


# --- CLI --------------------------------------------------------------------------------------


class FakeEnsoClient:
    calls: list[dict] = []
    outcome = None

    def __init__(self, config):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def analyze(self, **kwargs):
        FakeEnsoClient.calls.append(kwargs)
        if isinstance(FakeEnsoClient.outcome, Exception):
            raise FakeEnsoClient.outcome
        return FakeEnsoClient.outcome


@pytest.fixture
def fake_client(monkeypatch):
    FakeEnsoClient.calls = []
    monkeypatch.setattr("research_agent.cli.EnsoClient", FakeEnsoClient)
    return FakeEnsoClient


def test_enso_command_prints_events_and_the_data_origin(fake_client, make_enso_analysis):
    fake_client.outcome = make_enso_analysis()
    args = ["enso", "--phase", "el_nino", "--start-year", "1990", "--end-year", "2020"]
    result = runner.invoke(app, [*args, "--year", "2000", "--config", str(BASE)])
    assert result.exit_code == 0, result.output
    assert "[R1] ENSO events from NOAA CPC RONI" in result.output
    la_nina = "- La Nina OND 2000 to FMA 2001 (5 seasons), peak -1.80 (DJF 2001), strong"
    assert la_nina in result.output
    assert "(network; record DJF 2000 to AMJ 2001)" in result.output
    assert fake_client.calls == [
        {"phase": "el_nino", "start_year": 1990, "end_year": 2020, "year": 2000}
    ]


def test_enso_command_says_when_served_from_the_cache(fake_client, make_enso_analysis):
    fake_client.outcome = make_enso_analysis(from_cache=True)
    result = runner.invoke(app, ["enso", "--config", str(BASE)])
    assert result.exit_code == 0 and "(cache; record DJF 2000 to AMJ 2001)" in result.output
    defaults = {"phase": "both", "start_year": None, "end_year": None, "year": None}
    assert fake_client.calls == [defaults]


def test_enso_command_reports_failures_with_exit_code_1(fake_client):
    fake_client.outcome = EnsoApiError("NOAA CPC returned HTTP 404 for the RONI file")
    result = runner.invoke(app, ["enso", "--config", str(BASE)])
    assert result.exit_code == 1
    assert "error: NOAA CPC returned HTTP 404 for the RONI file" in result.output


def test_ask_data_section_names_the_enso_analysis(monkeypatch, tmp_path, e1):
    answer = Answer(
        claims=(Claim("A strong La Nina.", "evidence", ("R1",)),),
        references=(),
        limitations=("n",),
        data_references=(e1,),
    )
    stats = RunStats(2, 0, 10, 10, 0, rainfall_calls=1, tool_trace=((ENSO,), ()))
    monkeypatch.setattr("research_agent.cli._run_agent", lambda *a: AgentResult(answer, stats))
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef\n", encoding="utf-8")
    result = runner.invoke(app, ["ask", "q", "--config", str(BASE), "--env-file", str(env)])
    assert result.exit_code == 0, result.output
    assert (
        "[R1] ENSO events, NOAA CPC RONI, record DJF 2000 to AMJ 2001; events starting "
        "2000-2001."
    ) in result.output
    assert "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt" in result.output
    assert "tools by turn: 1: enso | 2: answer" in result.output


def test_format_trace_shortens_the_enso_tool():
    trace = ((ENSO, "search_literature"), ("analyze_rainfall_extremes[dry]",), ())
    assert format_trace(trace) == "1: enso, search | 2: rainfall[dry] | 3: answer"


# --- eval keys --------------------------------------------------------------------------------


def _result(e1, *trace):
    answer = Answer(
        claims=(Claim("x", "evidence", ("R1",)),),
        references=(),
        limitations=("n",),
        data_references=(e1,),
    )
    return AgentResult(answer, RunStats(2, 0, 10, 10, 0, rainfall_calls=1, tool_trace=trace))


def test_an_enso_citation_satisfies_the_universal_invariants(e1, limits):
    case = EvalCase("c", "Q", min_data_references=1)
    assert check_result(case, _result(e1, (ENSO,), ()), limits) == []


def test_expect_tool_is_judged_on_the_trace(e1, limits):
    case = EvalCase("c", "Q", expect_tool=ENSO)
    assert check_result(case, _result(e1, (ENSO,), ()), limits) == []
    failures = check_result(case, _result(e1, ("search_literature",), ()), limits)
    assert any(ENSO in f and "search_literature" in f for f in failures)


def test_expect_tool_ignores_the_metric_suffix(e1, limits):
    case = EvalCase("c", "Q", expect_tool="analyze_rainfall_extremes")
    assert check_result(case, _result(e1, ("analyze_rainfall_extremes[dry]",), ()), limits) == []


def test_expect_tool_counts_a_refused_call_as_the_choice_made(limits):
    case = EvalCase("c", "Q", expect_tool=ENSO)
    refused = AgentResult(
        Answer(claims=(Claim("x", "interpretation", ()),), references=(), limitations=("n",)),
        RunStats(2, 0, 10, 10, 0, tool_trace=((ENSO,), ())),
    )
    assert check_result(case, refused, limits) == []


def test_forbid_tool_fails_when_the_model_requests_it(e1, limits):
    case = EvalCase("c", "Q", forbid_tool=ENSO)
    assert check_result(case, _result(e1, ("search_literature",), ()), limits) == []
    failures = check_result(case, _result(e1, (ENSO,), ()), limits)
    assert any("forbids" in f for f in failures)


@pytest.mark.parametrize("key", ["expect_tool", "forbid_tool"])
def test_case_file_rejects_an_unknown_tool_name(tmp_path, key):
    path = tmp_path / "c.toml"
    path.write_text(f'[[case]]\nid = "a"\nquestion = "Q"\n{key} = "web_search"\n', encoding="utf-8")
    with pytest.raises(EvalError, match=key):
        load_cases(path)


def test_shipped_cases_cover_the_enso_tool_choices():
    root = Path(__file__).resolve().parents[1]
    cases = {c.id: c for c in load_cases(root / "evals" / "cases.toml")}
    assert len(cases) == 26
    for case_id in (
        "enso-events-since-1990",
        "enso-year-2010",
        "enso-strongest-la-nina-and-literature",
        "enso-current-conditions",
    ):
        assert cases[case_id].expect_tool == ENSO and cases[case_id].min_data_references >= 1
    assert cases["enso-literature-only"].forbid_tool == ENSO
    assert cases["enso-literature-only"].min_searches == 1
    assert cases["enso-strongest-la-nina-and-literature"].min_searches == 1


def test_case_file_accepts_known_tool_names(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        f'[[case]]\nid = "a"\nquestion = "Q"\nexpect_tool = "{ENSO}"\n'
        'forbid_tool = "search_literature"\n',
        encoding="utf-8",
    )
    (case,) = load_cases(path)
    assert (case.expect_tool, case.forbid_tool) == (ENSO, "search_literature")
