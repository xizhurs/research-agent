"""Agent-loop tests: scripted LLM + fake search. No network, no cost."""

import json
from dataclasses import replace
from datetime import date

import pytest

from research_agent.agent import ResearchAgent
from research_agent.agent.tools import SEARCH_TOOL
from research_agent.config import ReportingConfig
from research_agent.llm import (
    AssistantMessage,
    LLMRateLimitError,
    LLMResponse,
    StubLLMClient,
    ToolResultMessage,
    Usage,
    UserMessage,
    text_response,
    tool_call_response,
)
from research_agent.retrieval import (
    AuthError,
    InvalidQueryError,
    RateLimitError,
)

SEARCH = "search_literature"


class FakeSearch:
    """Returns scripted results (or raises); the last outcome repeats. Records every call."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple] = []

    def search(self, query, *, year_from=None, max_results=None):
        self.calls.append((query, year_from, max_results))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def call(call_id="c1", query="land cover", **extra):
    return (call_id, SEARCH, {"query": query, **extra})


def final(claims=(), limitations=()):
    return json.dumps({"claims": list(claims), "limitations": list(limitations)})


def evidence_claim(text="Finding.", sources=("W1",)):
    return {"text": text, "kind": "evidence", "sources": list(sources)}


TODAY = date(2026, 11, 1)  # fixed clock so "outdated evidence" limitations are deterministic
REPORTING = ReportingConfig(thin_evidence_works=3, stale_after_years=3)


def run(llm, search, limits, question="What works for land cover?"):
    agent = ResearchAgent(llm, search, limits, REPORTING, today=lambda: TODAY)
    return agent.run(question)


# --- happy paths -----------------------------------------------------------------------------


def test_search_then_structured_answer(limits, make_work, make_result):
    w1, w2 = make_work("W1", abstract="Deep nets map land cover."), make_work("W2")
    search = FakeSearch(make_result([w1, w2]))
    reply = final([evidence_claim("Deep nets work.", ["W1"])], ["abstracts only"])
    llm = StubLLMClient([tool_call_response(call()), text_response(reply)])

    result = run(llm, search, limits)

    assert search.calls == [("land cover", None, None)]
    assert len(llm.calls) == 2
    second = llm.calls[1]
    assert second.allow_tools is True
    assert isinstance(second.messages[0], UserMessage)
    assert isinstance(second.messages[1], AssistantMessage)
    assert second.messages[1].tool_calls[0].id == "c1"
    tool_result = second.messages[2]
    assert isinstance(tool_result, ToolResultMessage) and tool_result.call_id == "c1"
    assert "[W1]" in tool_result.content and "Deep nets map land cover." in tool_result.content

    ans = result.answer
    assert [c.text for c in ans.claims] == ["Deep nets work."]
    assert [w.id for w in ans.references] == ["W1"]  # only cited works
    assert ans.limitations[0] == "abstracts only"  # the model's own limitation comes first
    assert any("OpenAlex" in x for x in ans.limitations)  # code-added evidence limitation
    assert (result.stats.iterations, result.stats.search_calls) == (2, 1)
    assert result.stats.total_tokens == 30


def test_can_answer_without_searching(limits):
    llm = StubLLMClient([text_response(final([evidence_claim()]))])
    search = FakeSearch(RateLimitError("should never be called"))
    result = run(llm, search, limits)
    assert search.calls == []
    assert result.stats.search_calls == 0 and result.stats.iterations == 1


def test_call_passes_instructions_tools_and_output_cap(limits):
    llm = StubLLMClient([text_response(final())])
    run(llm, FakeSearch(), limits)
    rec = llm.calls[0]
    assert rec.tools == (SEARCH_TOOL,)
    assert rec.max_output_tokens == limits.max_output_tokens
    assert f"at most {limits.max_search_calls} searches" in rec.instructions
    assert rec.messages == (UserMessage("What works for land cover?"),)


def test_provider_state_is_passed_back_untouched(limits, make_work, make_result):
    state = object()
    calls = tool_call_response(call()).tool_calls
    first = LLMResponse(None, calls, Usage(5, 5), "tool_calls", state)
    llm = StubLLMClient([first, text_response(final())])
    run(llm, FakeSearch(make_result([make_work()])), limits)
    assert llm.calls[1].messages[1].provider_state is state


def test_repeated_work_is_not_resent_in_full(limits, make_work, make_result):
    w1 = make_work("W1", abstract="unique abstract text")
    search = FakeSearch(make_result([w1]))
    llm = StubLLMClient(
        [
            tool_call_response(call("c1")),
            tool_call_response(call("c2", "again")),
            text_response(final()),
        ]
    )
    run(llm, search, limits)
    second_result = llm.calls[2].messages[-1]
    assert "already shown" in second_result.content
    assert "unique abstract text" not in second_result.content


# --- hard caps -------------------------------------------------------------------------------


def test_iteration_cap_forces_final_answer_and_never_exceeds_it(limits, make_work, make_result):
    capped = replace(limits, max_iterations=3, max_search_calls=10)
    llm = StubLLMClient(  # exactly 3 entries: a 4th call would raise AssertionError
        [
            tool_call_response(call("c1")),
            tool_call_response(call("c2", "b")),
            text_response(final()),
        ]
    )
    result = run(llm, FakeSearch(make_result([make_work()])), capped)
    assert [c.allow_tools for c in llm.calls] == [True, True, False]
    assert result.stats.iterations == 3
    assert any("Iteration limit" in x for x in result.answer.limitations)


def test_single_iteration_means_no_tools_at_all(limits):
    capped = replace(limits, max_iterations=1)
    llm = StubLLMClient([text_response(final(limitations=["no evidence gathered"]))])
    search = FakeSearch()
    result = run(llm, search, capped)
    assert [c.allow_tools for c in llm.calls] == [False]
    assert search.calls == []
    assert any("Iteration limit" in x for x in result.answer.limitations)


def test_model_ignoring_disabled_tools_is_handled_without_executing_them(
    limits, make_work, make_result
):
    capped = replace(limits, max_iterations=2)
    llm = StubLLMClient([tool_call_response(call("c1")), tool_call_response(call("c2", "b"))])
    search = FakeSearch(make_result([make_work()]))
    result = run(llm, search, capped)
    assert len(search.calls) == 1  # the forced turn's tool call was not executed
    assert llm.calls[1].allow_tools is False
    assert any("tried to call tools" in x for x in result.answer.limitations)
    assert result.answer.claims  # still returns an (unstructured) answer, never crashes


def test_search_cap_forces_final_answer(limits, make_work, make_result):
    capped = replace(limits, max_search_calls=2, max_iterations=10)
    llm = StubLLMClient(
        [
            tool_call_response(call("c1")),
            tool_call_response(call("c2", "b")),
            text_response(final()),
        ]
    )
    search = FakeSearch(make_result([make_work()]))
    result = run(llm, search, capped)
    assert len(search.calls) == 2
    assert [c.allow_tools for c in llm.calls] == [True, True, False]
    assert result.stats.search_calls == 2
    assert any("Search limit" in x for x in result.answer.limitations)


def test_parallel_calls_beyond_remaining_searches_are_refused_but_answered(
    limits, make_work, make_result
):
    capped = replace(limits, max_search_calls=2, max_iterations=10)
    three = tool_call_response(call("c1", "a"), call("c2", "b"), call("c3", "c"))
    llm = StubLLMClient([three, text_response(final())])
    search = FakeSearch(make_result([make_work()]))
    result = run(llm, search, capped)

    assert [q for q, *_ in search.calls] == ["a", "b"]
    results = {
        m.call_id: m.content for m in llm.calls[1].messages if isinstance(m, ToolResultMessage)
    }
    assert set(results) == {"c1", "c2", "c3"}  # every call ID gets a result
    assert "search limit reached" in results["c3"]
    assert llm.calls[1].allow_tools is False
    assert result.stats.search_calls == 2
    assert any("not run" in x for x in result.answer.limitations)


def test_context_budget_truncates_results_and_omitted_works_are_not_citable(
    limits, make_work, make_result
):
    tight = replace(limits, max_context_tokens=150)
    works = [make_work(f"W{i}", abstract="x" * 400) for i in (1, 2)]  # ~100+ tokens each
    reply = final([evidence_claim("a", ["W1"]), evidence_claim("b", ["W2"])])
    llm = StubLLMClient([tool_call_response(call()), text_response(reply)])
    result = run(llm, FakeSearch(make_result(works)), tight)

    shown = llm.calls[1].messages[-1].content
    assert "[W1]" in shown and "omitted" in shown and "[W2]" not in shown
    assert [w.id for w in result.answer.references] == ["W1"]
    assert any("retrieved-context limit" in x for x in result.answer.limitations)


def test_token_budget_forces_final_answer(limits, make_work, make_result):
    tight = replace(limits, max_total_tokens=1000, max_output_tokens=100, max_iterations=10)
    llm = StubLLMClient(
        [tool_call_response(call(), input_tokens=600, output_tokens=50), text_response(final())]
    )
    result = run(llm, FakeSearch(make_result([make_work()])), tight)
    assert [c.allow_tools for c in llm.calls] == [True, False]
    assert any("Token budget nearly exhausted" in x for x in result.answer.limitations)


def test_token_overshoot_is_reported(limits):
    tight = replace(limits, max_total_tokens=100)
    llm = StubLLMClient([text_response(final(), input_tokens=500, output_tokens=50)])
    result = run(llm, FakeSearch(), tight)
    assert any("Token budget exceeded" in x for x in result.answer.limitations)


# --- tool errors -----------------------------------------------------------------------------


def test_bad_tool_arguments_go_back_to_the_model_and_cost_no_search(limits):
    bad = tool_call_response(("c1", SEARCH, {"query": ""}))
    llm = StubLLMClient([bad, text_response(final())])
    search = FakeSearch()
    result = run(llm, search, limits)
    assert search.calls == []
    assert "Error" in llm.calls[1].messages[-1].content
    assert result.stats.search_calls == 0


def test_invalid_query_from_client_costs_no_search(limits):
    llm = StubLLMClient([tool_call_response(call(year_from=1)), text_response(final())])
    search = FakeSearch(InvalidQueryError("year_from must be between 1500 and 2100"))
    result = run(llm, search, limits)
    assert "year_from must be" in llm.calls[1].messages[-1].content
    assert result.stats.search_calls == 0


def test_retrieval_failure_is_reported_to_model_and_in_limitations(limits):
    llm = StubLLMClient([tool_call_response(call()), text_response(final())])
    search = FakeSearch(RateLimitError("daily budget exhausted"))
    result = run(llm, search, limits)
    assert "daily budget exhausted" in llm.calls[1].messages[-1].content
    assert result.stats.search_calls == 1
    assert any("search failed (RateLimitError)" in x for x in result.answer.limitations)


def test_auth_failure_aborts_the_run(limits):
    llm = StubLLMClient([tool_call_response(call())])
    with pytest.raises(AuthError):
        run(llm, FakeSearch(AuthError("key_not_activated")), limits)


def test_unknown_tool_gets_an_error_result(limits):
    unknown = tool_call_response(("c1", "delete_everything", {}))
    llm = StubLLMClient([unknown, text_response(final())])
    run(llm, FakeSearch(), limits)
    assert "unknown tool 'delete_everything'" in llm.calls[1].messages[-1].content


# --- answer handling and failures ------------------------------------------------------------


def test_unstructured_final_reply_becomes_honest_fallback(limits):
    llm = StubLLMClient([text_response("Land cover is mapped with CNNs.")])
    ans = run(llm, FakeSearch(), limits).answer
    assert ans.claims[0].text == "Land cover is mapped with CNNs."
    assert ans.claims[0].kind == "interpretation" and ans.claims[0].sources == ()
    assert "structured format" in ans.limitations[0]


def test_truncated_final_reply_is_flagged(limits):
    cut = LLMResponse('{"claims": [{"text": "unfinis', (), Usage(10, 5), "length")
    ans = run(StubLLMClient([cut]), FakeSearch(), limits).answer
    assert any("output limit" in x for x in ans.limitations)
    assert any("structured format" in x for x in ans.limitations)


def test_llm_errors_propagate(limits):
    llm = StubLLMClient([LLMRateLimitError("quota")])
    with pytest.raises(LLMRateLimitError):
        run(llm, FakeSearch(), limits)


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_rejected_before_any_call(limits, question):
    llm = StubLLMClient([])
    with pytest.raises(ValueError, match="non-empty"):
        run(llm, FakeSearch(), limits, question)
    assert llm.calls == []


# --- M4: citation validation and limitations through the whole loop ----------------------------


def test_hallucinated_citation_is_rejected_end_to_end(limits, make_work, make_result):
    reply = final([evidence_claim("Real.", ["W1"]), evidence_claim("Invented.", ["W404"])])
    llm = StubLLMClient([tool_call_response(call()), text_response(reply)])
    result = run(llm, FakeSearch(make_result([make_work("W1")])), limits)

    ans = result.answer
    assert ans.rejected_citations == ("W404",)
    assert [c.kind for c in ans.claims] == ["evidence", "interpretation"]
    assert [w.id for w in ans.references] == ["W1"]
    assert any("W404" in x for x in ans.limitations)


def test_citing_a_work_the_model_never_saw_is_rejected_even_if_it_was_returned(
    limits, make_work, make_result
):
    """W2 was returned by OpenAlex but hidden by the context budget, so it is not citable."""
    tight = replace(limits, max_context_tokens=150)
    works = [make_work(f"W{i}", abstract="x" * 400) for i in (1, 2)]
    reply = final([evidence_claim("Unseen.", ["W2"])])
    llm = StubLLMClient([tool_call_response(call()), text_response(reply)])
    ans = run(llm, FakeSearch(make_result(works)), tight).answer
    assert ans.rejected_citations == ("W2",)
    assert ans.claims[0].kind == "interpretation"


def test_every_answer_carries_limitations_even_if_the_model_gives_none(limits):
    llm = StubLLMClient([text_response(final())])  # model: no claims, no limitations
    ans = run(llm, FakeSearch(), limits).answer
    assert ans.limitations
    assert any("No literature search was run" in x for x in ans.limitations)


def test_fallback_answer_also_carries_evidence_limitations(limits, make_work, make_result):
    llm = StubLLMClient([tool_call_response(call()), text_response("plain prose")])
    ans = run(llm, FakeSearch(make_result([make_work()])), limits).answer
    assert ans.structured is False
    assert any("OpenAlex" in x for x in ans.limitations)
    assert any("thin" in x for x in ans.limitations)  # only 1 work retrieved


def test_search_records_drive_year_filter_and_coverage_limitations(
    limits, make_work, make_result
):
    works = [make_work(f"W{i}", year=2025) for i in range(1, 5)]
    result = make_result(works, total=900)
    llm = StubLLMClient(
        [tool_call_response(call(year_from=2022)), text_response(final([evidence_claim()]))]
    )
    ans = run(llm, FakeSearch(result), limits).answer
    assert any("excluded works published before 2022" in x for x in ans.limitations)
    assert any("4 of 900 matches" in x for x in ans.limitations)


def test_stats_report_works_shown(limits, make_work, make_result):
    works = [make_work(f"W{i}") for i in (1, 2, 3)]
    llm = StubLLMClient([tool_call_response(call()), text_response(final())])
    assert run(llm, FakeSearch(make_result(works)), limits).stats.works_shown == 3


def test_outdated_limitation_uses_the_injected_clock(limits, make_work, make_result):
    old = [make_work(f"W{i}", year=2019) for i in (1, 2, 3)]
    llm = StubLLMClient([tool_call_response(call()), text_response(final())])
    ans = run(llm, FakeSearch(make_result(old)), limits).answer
    assert any("most recent retrieved work is from 2019" in x for x in ans.limitations)
