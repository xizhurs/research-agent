"""Agent loop with the temperature tool: scripted LLM, fake analyzers. No network.

Which tool a real model chooses is judged by the live eval cases; these tests cover what the code
controls: the tool is offered with rainfall, calls are dispatched and traced by name, the budget
is shared, citations and ambiguity work as for rainfall, and the map point reaches the analyzer.
"""

import json
from dataclasses import replace
from datetime import date

import pytest

from research_agent.agent import AgentResult, Answer, Claim, ResearchAgent, RunStats
from research_agent.agent.events import ToolEvent
from research_agent.agent.prompt import build_instructions
from research_agent.agent.tools import (
    RAINFALL_TOOL,
    SEARCH_TOOL,
    TEMPERATURE_TOOL,
    TEMPERATURE_TOOL_NAME,
    ToolArgumentError,
    format_recalled,
    format_temperature,
    parse_temperature_args,
)
from research_agent.cli import format_trace
from research_agent.config import ReportingConfig
from research_agent.conversation.state import _pending, evidence_label, turn_settings
from research_agent.evaluation import EvalCase, check_result
from research_agent.llm import StubLLMClient, ToolResultMessage, text_response, tool_call_response
from research_agent.rainfall import AmbiguousLocationError
from research_agent.temperature.analysis import compute_extremes, compute_yearly

RAIN, TEMP = "analyze_rainfall_extremes", TEMPERATURE_TOOL_NAME
TODAY = date(2026, 11, 12)
REPORTING = ReportingConfig(thin_evidence_works=3, stale_after_years=3)


@pytest.fixture
def make_temperature(rainfall_config, make_location, make_provenance, make_series):
    """A real temperature record (computed from a synthetic series), id still empty."""
    cfg = replace(rainfall_config, min_years=3, baseline_start_year=2000, baseline_end_year=2002)

    def _make(kind="hot", location=None):
        dates, values = make_series(2000, 2004, spikes={date(2003, 7, 10): 40.0}, base=20.0)
        args = {
            "start_year": 2000,
            "end_year": 2004,
            "last_complete_year": 2004,
            "config": cfg,
            "provenance": make_provenance(),
            "baseline_from_year": 2000,
        }
        if kind == "yearly":
            return compute_yearly(location or make_location(), dates, values, **args)
        return compute_extremes(location or make_location(), dates, values, metric=kind, **args)

    return _make


class FakeAnalyzer:
    """Scripted outcomes (the last repeats); records each call and each point call."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.points: list[tuple] = []

    def _next(self):
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, None

    def analyze(self, city, **kwargs):
        self.calls.append({"city": city, **kwargs})
        return self._next()

    def analyze_point(self, latitude, longitude, **kwargs):
        self.points.append((latitude, longitude, kwargs))
        return self._next()


class FakeSearch:
    def search(self, query, *, year_from=None, max_results=None):
        raise AssertionError("no search expected")


def final(*claims):
    return json.dumps({"claims": list(claims), "limitations": []})


def run(llm, limits, *, rainfall, temperature, map_point=None):
    agent = ResearchAgent(
        llm,
        FakeSearch(),
        limits,
        REPORTING,
        rainfall=rainfall,
        temperature=temperature,
        today=lambda: TODAY,
    )
    if map_point is not None:
        return agent.run("How hot was it?", map_point=map_point)
    return agent.run("How hot was it?")


def temp(call_id="t1", **args):
    return (call_id, TEMP, {"city": "Zhengzhou", "scale": "daily", **args})


def test_temperature_needs_the_rainfall_tool(limits, make_temperature):
    with pytest.raises(ValueError, match="together with the rainfall tool"):
        ResearchAgent(
            StubLLMClient([]), FakeSearch(), limits, REPORTING, temperature=FakeAnalyzer()
        )


def test_tool_and_prompt_are_offered_and_unchanged_without_temperature(limits, make_temperature):
    llm = StubLLMClient([text_response(final())])
    run(llm, limits, rainfall=FakeAnalyzer(), temperature=FakeAnalyzer(make_temperature()))
    rec = llm.calls[0]
    assert rec.tools == (SEARCH_TOOL, RAINFALL_TOOL, TEMPERATURE_TOOL)
    assert rec.instructions == build_instructions(limits, rainfall=True, temperature=True)
    assert "- analyze_temperature:" in rec.instructions
    assert f"at most {limits.max_rainfall_calls} rainfall or temperature analyses" in (
        rec.instructions
    )
    without = build_instructions(limits, rainfall=True)
    assert TEMP not in without and "temperature" not in without


def test_a_temperature_call_is_citable_and_traced_by_name(limits, make_temperature):
    analyzer = FakeAnalyzer(make_temperature())
    claim = {"text": "Hottest day.", "kind": "evidence", "sources": ["R1"]}
    llm = StubLLMClient(
        [
            tool_call_response(temp(metric="hot", event_date="2003-07-10")),
            text_response(final(claim)),
        ]
    )
    result = run(llm, limits, rainfall=FakeAnalyzer(), temperature=analyzer)
    assert analyzer.calls[0]["metric"] == "hot"
    assert analyzer.calls[0]["event_date"] == date(2003, 7, 10)
    shown = [m for m in llm.calls[1].messages if isinstance(m, ToolResultMessage)][0].content
    assert shown.startswith("[R1] Temperature extremes (TXx")
    assert result.answer.claims[0].kind == "evidence"
    assert [r.id for r in result.answer.data_references] == ["R1"]
    assert result.stats.tool_trace == ((f"{TEMP}[hot]",), ())
    assert result.stats.rainfall_calls == 1
    assert result.events[0].tool == TEMP and result.events[0].location.name == "Zhengzhou"
    assert format_trace(result.stats.tool_trace) == "1: temperature[hot] | 2: answer"


@pytest.mark.parametrize(
    "args, name",
    [
        ({"scale": "yearly"}, "yearly"),
        ({"scale": "monthly"}, "monthly"),
        ({"scale": "daily", "metric": "cold"}, "cold"),
        ({}, "hot"),  # a call without scale or metric keeps the daily default
    ],
)
def test_the_trace_names_the_scale_or_metric(limits, make_temperature, args, name):
    call = ("t1", TEMP, {"city": "Zhengzhou", **args})
    llm = StubLLMClient([tool_call_response(call), text_response(final())])
    result = run(llm, limits, rainfall=FakeAnalyzer(), temperature=FakeAnalyzer(make_temperature()))
    assert result.stats.tool_trace[0] == (f"{TEMP}[{name}]",)


def test_rainfall_temperature_and_enso_share_one_analysis_budget(
    limits, make_temperature, make_analysis
):
    rain = ("r1", RAIN, {"city": "Zhengzhou", "scale": "daily"})
    calls = [rain, temp("t1"), temp("t2", metric="cold")]
    llm = StubLLMClient(
        [
            tool_call_response(*calls[:2]),
            tool_call_response(calls[2], temp("t3", scale="yearly")),
            text_response(final()),
        ]
    )
    analyzer = FakeAnalyzer(make_temperature())
    result = run(
        llm, replace(limits, max_rainfall_calls=3), rainfall=FakeAnalyzer(make_analysis()),
        temperature=analyzer,
    )  # fmt: skip
    assert len(analyzer.calls) == 2 and result.stats.rainfall_calls == 3
    refused = [m for m in llm.calls[2].messages if isinstance(m, ToolResultMessage)][-1]
    assert refused.content.startswith("Error: analysis limit reached (rainfall, temperature")
    assert any("temperature analysis was not run" in x for x in result.answer.limitations)


def test_bad_arguments_cost_nothing(limits, make_temperature):
    analyzer = FakeAnalyzer(make_temperature())
    llm = StubLLMClient([tool_call_response(temp(metric="wet")), text_response(final())])
    result = run(llm, limits, rainfall=FakeAnalyzer(), temperature=analyzer)
    shown = [m for m in llm.calls[1].messages if isinstance(m, ToolResultMessage)][0].content
    assert shown == "Error: 'metric' must be 'hot' or 'cold'."
    assert analyzer.calls == [] and result.stats.rainfall_calls == 0


def test_an_ambiguous_city_is_reported_and_pending(limits, make_location):
    candidates = (make_location(id=1, name="Springfield"), make_location(id=2, name="Springfield"))
    analyzer = FakeAnalyzer(AmbiguousLocationError("several places", candidates))
    llm = StubLLMClient([tool_call_response(temp(city="Springfield")), text_response(final())])
    result = run(llm, limits, rainfall=FakeAnalyzer(), temperature=analyzer)
    assert result.events[0].status == "ambiguous"
    assert any("no temperature analysis was run" in x for x in result.answer.limitations)
    pending = _pending(result.events, None)
    assert pending is not None and len(pending.candidates) == 2


def test_the_map_point_goes_to_the_temperature_tool(limits, make_temperature):
    analyzer = FakeAnalyzer(make_temperature())
    call = ("t1", TEMP, {"use_map_point": True, "scale": "yearly"})
    llm = StubLLMClient([tool_call_response(call), text_response(final())])
    run(llm, limits, rainfall=FakeAnalyzer(), temperature=analyzer, map_point=(48.8534, 2.3488))
    assert analyzer.points[0][:2] == (48.8534, 2.3488) and analyzer.calls == []
    question = llm.calls[0].messages[-1].text
    assert f"call {RAIN} or {TEMP} with use_map_point=true" in question
    tools = {t.name: t for t in llm.calls[0].tools}
    assert "use_map_point" in tools[TEMP].parameters["properties"]


def test_parse_temperature_args():
    args = parse_temperature_args({"city": "Madrid", "scale": "yearly", "event_year": 2024})
    assert (args.city, args.scale, args.metric) == ("Madrid", "yearly", "hot")
    assert args.event_year == 2024
    with pytest.raises(ToolArgumentError, match="no map point"):
        parse_temperature_args({"use_map_point": True, "scale": "yearly"})


def test_recall_and_labels_render_temperature_records(make_temperature):
    record = replace(make_temperature("yearly"), id="R4")
    assert format_recalled(record) == format_temperature(record)
    assert evidence_label(record) == "Zhengzhou, Henan, China, annual mean temperature 2000-2004"


def test_conversation_state_labels_temperature_metrics(make_location):
    event = ToolEvent(
        tool=TEMP,
        arguments={"city": "Zhengzhou", "scale": "yearly", "start_year": 2015},
        status="ok",
        result_id="R1",
        location=make_location(),
    )
    settings = turn_settings([event])
    assert settings["metrics"] == ("temperature yearly",)
    assert settings["sources"] == ("ERA5 (Open-Meteo)",)
    assert settings["period"] == (2015, None)
    assert settings["places"][0].label == "Zhengzhou, Henan, China"


def _traced(record, *trace):
    answer = Answer(
        claims=(Claim("x", "evidence", ("R1",)),),
        references=(),
        limitations=("n",),
        data_references=(record,),
    )
    return AgentResult(answer, RunStats(2, 0, 10, 10, 0, rainfall_calls=1, tool_trace=trace))


def test_eval_metric_is_judged_on_the_expected_tool(limits, make_temperature):
    record = replace(make_temperature("yearly"), id="R1")
    case = EvalCase("c", "Q", expect_tool=TEMP, expect_metric="yearly")
    assert check_result(case, _traced(record, (f"{TEMP}[yearly]",), ()), limits) == []
    wrong_tool = check_result(case, _traced(record, (f"{RAIN}[yearly]",), ()), limits)
    assert any(f"expected {TEMP} to be requested with metric 'yearly'" in f for f in wrong_tool)
    rain_case = EvalCase("c", "Q", expect_metric="yearly")  # no tool named: rainfall, as before
    assert check_result(rain_case, _traced(record, (f"{TEMP}[yearly]",), ()), limits) != []
