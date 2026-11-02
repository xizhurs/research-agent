"""Agent loop with the ENSO tool: scripted LLM, fake analyzers and search. No network.

As with rainfall, which tool a real model *chooses* cannot be tested offline (the stub's script
makes the choice); the live ``eval`` cases judge that. These tests cover what the code controls.
"""

import json
from dataclasses import replace
from datetime import date

import pytest

from research_agent.agent import ResearchAgent
from research_agent.agent.prompt import build_instructions
from research_agent.agent.tools import ENSO_TOOL, RAINFALL_TOOL, SEARCH_TOOL, format_enso
from research_agent.config import ReportingConfig
from research_agent.enso import (
    EnsoApiError,
    EnsoNetworkError,
    InsufficientEnsoDataError,
    InvalidEnsoQueryError,
)
from research_agent.llm import (
    StubLLMClient,
    ToolResultMessage,
    text_response,
    tool_call_response,
)

SEARCH, RAIN, ENSO = "search_literature", "analyze_rainfall_extremes", "analyze_enso_events"
TODAY = date(2026, 11, 2)
REPORTING = ReportingConfig(thin_evidence_works=3, stale_after_years=3)


class FakeEnso:
    """Scripted outcomes (the last repeats); records every call's arguments."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def analyze(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeRainfall:
    def __init__(self, analysis):
        self.analysis = analysis

    def analyze(self, city, **kwargs):
        return self.analysis


class FakeSearch:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def search(self, query, *, year_from=None, max_results=None):
        self.calls.append(query)
        return self.result


def enso(call_id="e1", **args):
    return (call_id, ENSO, args)


def final(claims=(), limitations=()):
    return json.dumps({"claims": list(claims), "limitations": list(limitations)})


def evidence(text, *sources):
    return {"text": text, "kind": "evidence", "sources": list(sources)}


def run(llm, limits, *, enso_backend, rainfall=None, search=None, question="Which years?"):
    agent = ResearchAgent(
        llm,
        search or FakeSearch(),
        limits,
        REPORTING,
        rainfall=rainfall or FakeRainfall(None),
        enso=enso_backend,
        today=lambda: TODAY,
    )
    return agent.run(question)


def tool_results(llm, call_index):
    return [m for m in llm.calls[call_index].messages if isinstance(m, ToolResultMessage)]


# --- offering and dispatch --------------------------------------------------------------------


def test_three_tools_and_an_extended_prompt_are_offered_when_enso_is_enabled(
    limits, make_enso_analysis
):
    llm = StubLLMClient([text_response(final())])
    run(llm, limits, enso_backend=FakeEnso(make_enso_analysis()))
    rec = llm.calls[0]
    assert rec.tools == (SEARCH_TOOL, RAINFALL_TOOL, ENSO_TOOL)
    assert ENSO in rec.instructions and "three tools" in rec.instructions
    assert f"at most {limits.max_rainfall_calls} rainfall or ENSO analyses" in rec.instructions
    assert "never list ENSO years from memory" in rec.instructions


def test_without_enso_the_two_tool_agent_and_prompt_are_unchanged(limits, make_analysis):
    llm = StubLLMClient([text_response(final())])
    agent = ResearchAgent(
        llm, FakeSearch(), limits, REPORTING, rainfall=FakeRainfall(make_analysis())
    )
    agent.run("q")
    rec = llm.calls[0]
    assert rec.tools == (SEARCH_TOOL, RAINFALL_TOOL)
    assert ENSO not in rec.instructions and "El Nino" not in rec.instructions
    assert rec.instructions == build_instructions(limits, rainfall=True)
    assert f"at most {limits.max_rainfall_calls} rainfall analyses" in rec.instructions


def test_enso_without_rainfall_is_refused_at_construction(limits, make_enso_analysis):
    with pytest.raises(ValueError, match="together with the rainfall tool"):
        ResearchAgent(
            StubLLMClient([]), FakeSearch(), limits, REPORTING, enso=FakeEnso(make_enso_analysis())
        )
    with pytest.raises(ValueError, match="together with the rainfall tool"):
        build_instructions(limits, enso=True)


def test_enso_call_without_an_enso_backend_is_an_unknown_tool(limits, make_analysis):
    llm = StubLLMClient([tool_call_response(enso()), text_response(final())])
    agent = ResearchAgent(
        llm, FakeSearch(), limits, REPORTING, rainfall=FakeRainfall(make_analysis())
    )
    result = agent.run("q")
    assert "unknown tool" in tool_results(llm, 1)[0].content
    assert result.stats.rainfall_calls == 0


def test_enso_question_runs_only_the_enso_tool_and_cites_it(limits, make_enso_analysis):
    backend, search = FakeEnso(make_enso_analysis()), FakeSearch()
    reply = final([evidence("A strong La Nina peaked in DJF 2001.", "R1")], ["model-own note"])
    llm = StubLLMClient(
        [
            tool_call_response(enso(phase="la_nina", start_year=1990, end_year=2010, year=2001)),
            text_response(reply),
        ]
    )

    result = run(llm, limits, enso_backend=backend, search=search)

    assert backend.calls == [
        {"phase": "la_nina", "start_year": 1990, "end_year": 2010, "year": 2001}
    ]
    assert search.calls == []
    shown = tool_results(llm, 1)[0].content
    assert shown.startswith("[R1] ENSO events from NOAA CPC RONI") and "-1.80 (DJF 2001)" in shown
    ans = result.answer
    assert [a.id for a in ans.data_references] == ["R1"] and ans.references == ()
    assert ans.limitations[0] == "model-own note"
    text = " ".join(ans.limitations)
    assert "Relative Oceanic Nino Index" in text and "common convention" in text
    assert "OpenAlex" not in text and "No literature search" not in text
    stats = result.stats
    assert (stats.search_calls, stats.rainfall_calls) == (0, 1)
    assert stats.tool_trace == ((ENSO,), ())


def test_defaults_are_passed_when_the_model_gives_no_arguments(limits, make_enso_analysis):
    backend = FakeEnso(make_enso_analysis())
    llm = StubLLMClient([tool_call_response(enso()), text_response(final())])
    run(llm, limits, enso_backend=backend)
    assert backend.calls == [{"phase": "both", "start_year": None, "end_year": None, "year": None}]


def test_cached_result_is_counted_as_cached(limits, make_enso_analysis):
    llm = StubLLMClient([tool_call_response(enso()), text_response(final())])
    result = run(llm, limits, enso_backend=FakeEnso(make_enso_analysis(from_cache=True)))
    assert (result.stats.rainfall_calls, result.stats.cached_rainfall) == (1, 1)


# --- citations --------------------------------------------------------------------------------


def test_ids_continue_across_rainfall_and_enso_and_both_are_citable(
    limits, make_analysis, make_enso_analysis
):
    llm = StubLLMClient(
        [
            tool_call_response(("r1", RAIN, {"city": "Zhengzhou"}), enso("e1")),
            text_response(
                final(
                    [
                        evidence("Heavy rain.", "R1"),
                        evidence("A La Nina.", "R2"),
                        evidence("Made up.", "R3"),
                    ]
                )
            ),
        ]
    )
    result = run(
        llm,
        limits,
        enso_backend=FakeEnso(make_enso_analysis()),
        rainfall=FakeRainfall(make_analysis()),
    )
    ans = result.answer
    assert [a.id for a in ans.data_references] == ["R1", "R2"]
    assert type(ans.data_references[0]).__name__ == "RainfallAnalysis"
    assert type(ans.data_references[1]).__name__ == "EnsoAnalysis"
    assert ans.rejected_citations == ("R3",)
    assert ans.claims[2].kind == "interpretation"  # its only citation was fabricated
    assert [m.content[:4] for m in tool_results(llm, 1)] == ["[R1]", "[R2]"]


def test_literature_and_enso_results_are_both_citable(
    limits, make_enso_analysis, make_work, make_result
):
    search = FakeSearch(make_result([make_work("W1")]))
    llm = StubLLMClient(
        [
            tool_call_response(("c1", SEARCH, {"query": "la nina impacts"}), enso()),
            text_response(final([evidence("Impacts.", "W1"), evidence("Dates.", "R1")])),
        ]
    )
    result = run(llm, limits, enso_backend=FakeEnso(make_enso_analysis()), search=search)
    assert [w.id for w in result.answer.references] == ["W1"]
    assert [a.id for a in result.answer.data_references] == ["R1"]
    assert result.answer.rejected_citations == ()


def test_a_fabricated_enso_id_is_rejected(limits, make_enso_analysis):
    llm = StubLLMClient(
        [tool_call_response(enso()), text_response(final([evidence("Invented.", "R9")]))]
    )
    result = run(llm, limits, enso_backend=FakeEnso(make_enso_analysis()))
    assert result.answer.rejected_citations == ("R9",)
    assert result.answer.claims[0].kind == "interpretation"


# --- budgets and failures ---------------------------------------------------------------------


def test_enso_calls_share_the_rainfall_limit(limits, make_enso_analysis):
    one = replace(limits, max_rainfall_calls=1)
    backend = FakeEnso(make_enso_analysis())
    llm = StubLLMClient(
        [tool_call_response(enso("e1")), tool_call_response(enso("e2")), text_response(final())]
    )
    result = run(llm, one, enso_backend=backend)
    assert len(backend.calls) == 1
    second = tool_results(llm, 2)[-1].content  # the results so far: R1, then the refusal
    assert "analysis limit reached" in second and "share it" in second
    assert any("ENSO analysis was not run" in x for x in result.answer.limitations)
    assert result.stats.rainfall_calls == 1


@pytest.mark.parametrize(
    "arguments",
    [{"phase": "warm"}, {"year": "2010"}, {"start_year": True}, {"end_year": 2000.5}, {"phase": 3}],
)
def test_bad_arguments_go_back_to_the_model_and_cost_nothing(
    limits, make_enso_analysis, arguments
):
    backend = FakeEnso(make_enso_analysis())
    llm = StubLLMClient([tool_call_response(enso(**arguments)), text_response(final())])
    result = run(llm, limits, enso_backend=backend)
    assert tool_results(llm, 1)[0].content.startswith("Error: ")
    assert backend.calls == [] and result.stats.rainfall_calls == 0


def test_a_rejected_query_costs_nothing(limits):
    backend = FakeEnso(InvalidEnsoQueryError("start_year must not be after end_year"))
    llm = StubLLMClient(
        [tool_call_response(enso(start_year=2000, end_year=1990)), text_response(final())]
    )
    result = run(llm, limits, enso_backend=backend)
    assert "start_year must not be after end_year" in tool_results(llm, 1)[0].content
    assert result.stats.rainfall_calls == 0


@pytest.mark.parametrize(
    "error",
    [
        EnsoNetworkError("timeout"),
        EnsoApiError("unexpected RONI file"),
        InsufficientEnsoDataError("year 1940 lies outside it"),
    ],
)
def test_a_failed_analysis_costs_a_call_and_is_reported(limits, error):
    llm = StubLLMClient([tool_call_response(enso()), text_response(final())])
    result = run(llm, limits, enso_backend=FakeEnso(error))
    content = tool_results(llm, 1)[0].content
    assert content.startswith("Error: ") and str(error) in content
    text = " ".join(result.answer.limitations)
    assert type(error).__name__ in text and "no numerical evidence backs this answer" in text
    assert result.answer.data_references == () and result.stats.rainfall_calls == 1


def test_a_result_too_large_for_the_context_budget_is_refused_and_not_citable(
    limits, make_enso_analysis
):
    tiny = replace(limits, max_context_tokens=10)
    llm = StubLLMClient(
        [tool_call_response(enso()), text_response(final([evidence("Dates.", "R1")]))]
    )
    result = run(llm, tiny, enso_backend=FakeEnso(make_enso_analysis()))
    assert "budget is used up" in tool_results(llm, 1)[0].content
    assert result.answer.data_references == () and result.answer.rejected_citations == ("R1",)
    assert any("ENSO result was not shown" in x for x in result.answer.limitations)


def test_the_full_record_fits_the_default_context_budget(limits, make_enso_analysis):
    """About 45 events: a long list must not eat the whole retrieved-context budget."""
    values = ([0.0] * 2 + [1.0] * 6 + [0.0] * 2 + [-1.0] * 6) * 22
    text = format_enso(replace(make_enso_analysis(values), id="R1"))
    assert len(text) // 4 < limits.max_context_tokens * 0.4
