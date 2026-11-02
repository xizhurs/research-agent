"""Agent loop with the rainfall tool: scripted LLM, fake analyzer and search. No network.

Which tool a real model *chooses* cannot be tested offline: the stub's script makes the choice.
These tests cover what the code controls: both tools are offered, each call is dispatched to the
right backend, budgets and citation rules hold, and failures are reported honestly. Choice
quality is judged by the live ``eval`` cases.
"""

import json
from dataclasses import replace
from datetime import date

import pytest

from research_agent.agent import ResearchAgent, estimate_tokens
from research_agent.agent.tools import RAINFALL_TOOL, SEARCH_TOOL, format_rainfall
from research_agent.config import ReportingConfig
from research_agent.llm import (
    StubLLMClient,
    ToolResultMessage,
    text_response,
    tool_call_response,
)
from research_agent.rainfall import (
    AmbiguousLocationError,
    InsufficientDataError,
    InvalidRainfallQueryError,
    LocationNotFoundError,
    RainfallNetworkError,
)

SEARCH, RAIN = "search_literature", "analyze_extreme_rainfall"
TODAY = date(2026, 11, 2)
REPORTING = ReportingConfig(thin_evidence_works=3, stale_after_years=3)


class FakeRainfall:
    """Scripted analyzer outcomes (the last repeats); records every call's arguments."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def analyze(self, city, **kwargs):
        self.calls.append({"city": city, **kwargs})
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeSearch:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def search(self, query, *, year_from=None, max_results=None):
        self.calls.append(query)
        return self.result


def rain(call_id="r1", **args):
    return (call_id, RAIN, {"city": "Zhengzhou", **args})


def lit(call_id="c1", query="urban flood"):
    return (call_id, SEARCH, {"query": query})


def final(claims=(), limitations=()):
    return json.dumps({"claims": list(claims), "limitations": list(limitations)})


def evidence(text, *sources):
    return {"text": text, "kind": "evidence", "sources": list(sources)}


def run(llm, limits, *, search=None, rainfall=None, question="How extreme was it?"):
    agent = ResearchAgent(
        llm,
        search or FakeSearch(),
        limits,
        REPORTING,
        rainfall=rainfall,
        today=lambda: TODAY,
    )
    return agent.run(question)


def tool_results(llm, call_index):
    return [m for m in llm.calls[call_index].messages if isinstance(m, ToolResultMessage)]


# --- tool selection and dispatch ----------------------------------------------------------------


def test_both_tools_and_a_two_tool_prompt_are_offered_when_rainfall_is_enabled(
    limits, make_analysis
):
    llm = StubLLMClient([text_response(final())])
    run(llm, limits, rainfall=FakeRainfall(make_analysis()))
    rec = llm.calls[0]
    assert rec.tools == (SEARCH_TOOL, RAINFALL_TOOL)
    assert SEARCH in rec.instructions and RAIN in rec.instructions
    assert f"at most {limits.max_rainfall_calls} rainfall analyses" in rec.instructions
    assert "ambiguous" in rec.instructions and "R1" in rec.instructions


def test_without_a_rainfall_backend_the_agent_is_the_literature_agent_it_was(limits):
    llm = StubLLMClient([text_response(final())])
    run(llm, limits)
    rec = llm.calls[0]
    assert rec.tools == (SEARCH_TOOL,)
    assert RAIN not in rec.instructions
    assert f"at most {limits.max_search_calls} searches" in rec.instructions


def test_rainfall_call_without_a_backend_is_an_unknown_tool(limits):
    llm = StubLLMClient([tool_call_response(rain()), text_response(final())])
    result = run(llm, limits)
    assert "unknown tool" in tool_results(llm, 1)[0].content
    assert result.stats.rainfall_calls == 0


def test_rainfall_only_question_runs_only_the_rainfall_tool(limits, make_analysis):
    analyzer, search = FakeRainfall(make_analysis()), FakeSearch()
    reply = final([evidence("Zhengzhou's wettest year was 2004.", "R1")], ["model-own note"])
    llm = StubLLMClient([tool_call_response(rain()), text_response(reply)])

    result = run(llm, limits, search=search, rainfall=analyzer)

    assert len(analyzer.calls) == 1 and search.calls == []
    shown = tool_results(llm, 1)[0].content
    assert shown.startswith("[R1]") and "50.0 mm" in shown
    ans = result.answer
    assert [a.id for a in ans.data_references] == ["R1"] and ans.references == ()
    assert ans.limitations[0] == "model-own note"
    text = " ".join(ans.limitations)
    assert "ERA5 reanalysis" in text and "not gauge measurements" in text
    assert "OpenAlex" not in text and "No literature search" not in text
    assert (result.stats.search_calls, result.stats.rainfall_calls) == (0, 1)


def test_literature_only_question_with_rainfall_enabled_keeps_literature_caveats(
    limits, make_work, make_result, make_analysis
):
    analyzer = FakeRainfall(make_analysis())
    search = FakeSearch(make_result([make_work("W1")]))
    llm = StubLLMClient(
        [tool_call_response(lit()), text_response(final([evidence("Nets help.", "W1")]))]
    )
    result = run(llm, limits, search=search, rainfall=analyzer)
    assert analyzer.calls == [] and result.stats.rainfall_calls == 0
    assert any("OpenAlex" in x for x in result.answer.limitations)
    assert not any("ERA5" in x for x in result.answer.limitations)
    assert result.answer.data_references == ()


def test_numbers_and_explanations_use_both_tools_and_both_citation_kinds(
    limits, make_work, make_result, make_analysis
):
    analyzer = FakeRainfall(make_analysis())
    search = FakeSearch(make_result([make_work("W1", abstract="Monsoon drivers.")]))
    reply = final(
        [evidence("Peak was 50 mm.", "R1"), evidence("Driven by monsoon.", "W1")],
    )
    llm = StubLLMClient([tool_call_response(rain(), lit()), text_response(reply)])

    result = run(llm, limits, search=search, rainfall=analyzer)

    contents = [m.content for m in tool_results(llm, 1)]
    assert contents[0].startswith("[R1]") and "[W1]" in contents[1]
    ans = result.answer
    assert [a.id for a in ans.data_references] == ["R1"]
    assert [w.id for w in ans.references] == ["W1"]
    assert all(c.kind == "evidence" for c in ans.claims) and ans.rejected_citations == ()
    text = " ".join(ans.limitations)
    assert "OpenAlex" in text and "ERA5" in text
    assert (result.stats.search_calls, result.stats.rainfall_calls) == (1, 1)


def test_tool_arguments_reach_the_analyzer_with_parsed_types(limits, make_analysis):
    analyzer = FakeRainfall(make_analysis())
    args = {
        "country_code": "CN",
        "region": "Henan",
        "start_year": 1990,
        "end_year": 2024,
        "event_date": "2021-07-20",
    }
    llm = StubLLMClient([tool_call_response(rain(**args)), text_response(final())])
    run(llm, limits, rainfall=analyzer)
    assert analyzer.calls == [
        {
            "city": "Zhengzhou",
            "country_code": "CN",
            "region": "Henan",
            "start_year": 1990,
            "end_year": 2024,
            "event_date": date(2021, 7, 20),
        }
    ]


def test_second_analysis_gets_the_next_id_and_cache_hits_are_counted(
    limits, make_analysis, make_location
):
    first = make_analysis(from_cache=True)
    second = make_analysis(
        location=make_location(name="Mumbai", admin1="Maharashtra", country="India")
    )
    analyzer = FakeRainfall(first, second)
    reply = final([evidence("A", "R1"), evidence("B", "R2")])
    llm = StubLLMClient(
        [
            tool_call_response(rain("a")),
            tool_call_response(rain("b", city="Mumbai")),
            text_response(reply),
        ]
    )
    result = run(llm, limits, rainfall=analyzer)
    assert [a.id for a in result.answer.data_references] == ["R1", "R2"]
    assert [a.location.name for a in result.answer.data_references] == ["Zhengzhou", "Mumbai"]
    assert result.stats.cached_rainfall == 1 and result.stats.rainfall_calls == 2


# --- invalid input -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"city": ""}, "'city'"),
        ({"city": 5}, "'city'"),
        ({"start_year": "1990"}, "'start_year' must be an integer"),
        ({"end_year": True}, "'end_year' must be an integer"),
        ({"event_date": "20 July 2021"}, "'event_date' must be a date"),
        ({"event_date": 20210720}, "'event_date' must be a string"),
        ({"country_code": 7}, "'country_code' must be a string"),
    ],
)
def test_bad_arguments_go_back_to_the_model_and_cost_nothing(limits, args, message):
    analyzer = FakeRainfall(RuntimeError("must not be called"))
    llm = StubLLMClient(
        [tool_call_response(("r1", RAIN, {"city": "Zhengzhou", **args})), text_response(final())]
    )
    result = run(llm, limits, rainfall=analyzer)
    assert analyzer.calls == []
    assert message in tool_results(llm, 1)[0].content
    assert result.stats.rainfall_calls == 0


def test_missing_city_is_rejected(limits):
    analyzer = FakeRainfall(RuntimeError("must not be called"))
    llm = StubLLMClient([tool_call_response(("r1", RAIN, {})), text_response(final())])
    result = run(llm, limits, rainfall=analyzer)
    assert "'city' is required" in tool_results(llm, 1)[0].content
    assert analyzer.calls == [] and result.stats.rainfall_calls == 0


def test_semantically_invalid_request_is_returned_to_the_model_without_cost(limits):
    analyzer = FakeRainfall(InvalidRainfallQueryError("start_year must be between 1940 and 2026"))
    llm = StubLLMClient([tool_call_response(rain(start_year=1800)), text_response(final())])
    result = run(llm, limits, rainfall=analyzer)
    assert "between 1940 and 2026" in tool_results(llm, 1)[0].content
    assert result.stats.rainfall_calls == 0


# --- ambiguous locations and insufficient evidence ----------------------------------------------


def test_ambiguous_location_runs_no_analysis_and_is_reported(limits):
    message = "'Springfield' matches 4 places: 1) Springfield, Missouri. Do not guess."
    analyzer = FakeRainfall(AmbiguousLocationError(message, ()))
    reply = final([evidence("Springfield is ambiguous.", "R1")], ["location ambiguous"])
    llm = StubLLMClient([tool_call_response(rain(city="Springfield")), text_response(reply)])

    result = run(llm, limits, rainfall=analyzer)

    assert "Springfield, Missouri" in tool_results(llm, 1)[0].content
    ans = result.answer
    assert ans.data_references == ()
    assert ans.rejected_citations == ("R1",)  # nothing was computed, so R1 cannot be cited
    assert [c.kind for c in ans.claims] == ["interpretation"]
    text = " ".join(ans.limitations)
    assert "'Springfield' was ambiguous" in text and "no rainfall analysis was run" in text
    assert "no numerical evidence" in text
    assert result.stats.rainfall_calls == 1  # the geocoding request was made


def test_unknown_city_is_reported_as_not_found(limits):
    analyzer = FakeRainfall(LocationNotFoundError("no place named 'Atlantis' was found"))
    llm = StubLLMClient([tool_call_response(rain(city="Atlantis")), text_response(final())])
    result = run(llm, limits, rainfall=analyzer)
    assert "was not found" in " ".join(result.answer.limitations)
    assert result.answer.data_references == ()


@pytest.mark.parametrize(
    "error",
    [
        InsufficientDataError("no year between 2024 and 2024 has enough daily data"),
        RainfallNetworkError("Open-Meteo request failed after 4 attempts"),
    ],
)
def test_failed_or_insufficient_analysis_costs_a_call_and_is_reported(limits, error):
    analyzer = FakeRainfall(error)
    reply = final([evidence("It was extreme.", "R1")])
    llm = StubLLMClient([tool_call_response(rain()), text_response(reply)])

    result = run(llm, limits, rainfall=analyzer)

    assert "Error:" in tool_results(llm, 1)[0].content
    ans = result.answer
    assert result.stats.rainfall_calls == 1
    assert ans.claims[0].kind == "interpretation"  # unsupported evidence is downgraded
    text = " ".join(ans.limitations)
    assert type(error).__name__ in text and "no numerical evidence backs this answer" in text


def test_short_record_limitation_travels_into_the_answer(limits, make_analysis):
    analyzer = FakeRainfall(make_analysis(min_years=20, event_date=date(2004, 7, 1)))
    llm = StubLLMClient(
        [tool_call_response(rain()), text_response(final([evidence("Max 50 mm.", "R1")]))]
    )
    result = run(llm, limits, rainfall=analyzer)
    assert "ranks withheld" in tool_results(llm, 1)[0].content
    assert any("ranks and percentiles were not computed" in x for x in result.answer.limitations)


# --- budgets ------------------------------------------------------------------------------------


def test_rainfall_cap_refuses_extra_calls_but_answers_every_call_id(limits, make_analysis):
    capped = replace(limits, max_rainfall_calls=1)
    analyzer = FakeRainfall(make_analysis())
    llm = StubLLMClient(
        [tool_call_response(rain("a"), rain("b", city="Mumbai")), text_response(final())]
    )
    result = run(llm, capped, rainfall=analyzer)
    first, second = tool_results(llm, 1)
    assert (first.call_id, second.call_id) == ("a", "b")
    assert first.content.startswith("[R1]") and "rainfall-analysis limit reached" in second.content
    assert len(analyzer.calls) == 1 and result.stats.rainfall_calls == 1
    assert any("rainfall-analysis limit" in x for x in result.answer.limitations)


def test_spent_searches_do_not_block_the_rainfall_tool(
    limits, make_work, make_result, make_analysis
):
    capped = replace(limits, max_search_calls=1)
    analyzer = FakeRainfall(make_analysis())
    llm = StubLLMClient(
        [tool_call_response(lit()), tool_call_response(rain()), text_response(final())]
    )
    result = run(llm, capped, search=FakeSearch(make_result([make_work()])), rainfall=analyzer)
    assert [c.allow_tools for c in llm.calls] == [True, True, True]
    assert len(analyzer.calls) == 1 and result.stats.rainfall_calls == 1
    assert not any("Search limit" in x for x in result.answer.limitations)


def test_final_answer_is_forced_only_when_both_tool_budgets_are_spent(
    limits, make_work, make_result, make_analysis
):
    capped = replace(limits, max_search_calls=1, max_rainfall_calls=1)
    llm = StubLLMClient(
        [tool_call_response(lit()), tool_call_response(rain()), text_response(final())]
    )
    result = run(
        llm,
        capped,
        search=FakeSearch(make_result([make_work()])),
        rainfall=FakeRainfall(make_analysis()),
    )
    assert [c.allow_tools for c in llm.calls] == [True, True, False]
    assert any(
        "Search and rainfall-analysis limits reached" in x for x in result.answer.limitations
    )


def test_iteration_cap_still_applies_with_two_tools(limits, make_analysis):
    capped = replace(limits, max_iterations=2)
    llm = StubLLMClient([tool_call_response(rain()), text_response(final())])
    result = run(llm, capped, rainfall=FakeRainfall(make_analysis()))
    assert [c.allow_tools for c in llm.calls] == [True, False]
    assert result.stats.iterations == 2
    assert any("Iteration limit" in x for x in result.answer.limitations)


def test_result_that_does_not_fit_the_context_budget_is_not_shown_or_citable(limits, make_analysis):
    tiny = replace(limits, max_context_tokens=50)
    reply = final([evidence("Peak was 50 mm.", "R1")])
    llm = StubLLMClient([tool_call_response(rain()), text_response(reply)])

    result = run(llm, tiny, rainfall=FakeRainfall(make_analysis()))

    assert "retrieved-context budget" in tool_results(llm, 1)[0].content
    ans = result.answer
    assert ans.data_references == () and ans.rejected_citations == ("R1",)
    assert any("retrieved-context limit" in x for x in ans.limitations)
    assert result.stats.rainfall_calls == 1  # the analysis ran; only its display was refused


def test_rainfall_and_literature_share_one_context_budget(
    limits, make_work, make_result, make_analysis
):
    analysis = make_analysis()
    needed = estimate_tokens(format_rainfall(replace(analysis, id="R1")))
    shared = replace(limits, max_context_tokens=needed + 20)  # room for one rainfall result only
    big = make_work("W1", abstract="long " * 80)
    llm = StubLLMClient(
        [tool_call_response(rain("a")), tool_call_response(lit("b")), text_response(final())]
    )
    result = run(
        llm, shared, search=FakeSearch(make_result([big])), rainfall=FakeRainfall(analysis)
    )
    assert "(1 more result(s) omitted" in tool_results(llm, 2)[-1].content
    assert result.answer.limitations and any(
        "not shown to the model" in x for x in result.answer.limitations
    )


# --- citations ----------------------------------------------------------------------------------


def test_fabricated_rainfall_id_is_rejected_and_reported(limits, make_analysis):
    reply = final([evidence("Real.", "R1"), evidence("Invented.", "R9")])
    llm = StubLLMClient([tool_call_response(rain()), text_response(reply)])
    result = run(llm, limits, rainfall=FakeRainfall(make_analysis()))
    ans = result.answer
    assert ans.rejected_citations == ("R9",)
    assert [c.kind for c in ans.claims] == ["evidence", "interpretation"]
    assert ans.claims[1].sources == ()
    assert [a.id for a in ans.data_references] == ["R1"]
    assert any("R9" in x for x in ans.limitations)


def test_work_id_cited_for_numbers_that_were_never_retrieved_is_rejected(limits, make_analysis):
    reply = final([evidence("Peak was 50 mm.", "W123")])
    llm = StubLLMClient([tool_call_response(rain()), text_response(reply)])
    result = run(llm, limits, rainfall=FakeRainfall(make_analysis()))
    assert result.answer.rejected_citations == ("W123",)
    assert result.answer.claims[0].kind == "interpretation"
