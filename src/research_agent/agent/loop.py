"""The single-agent loop: model turns, tool execution, budgets and forced finalisation.

Guarantees, all enforced here in code (not in the prompt):
- at most ``max_iterations`` model calls per question;
- at most ``max_search_calls`` searches actually run;
- the retrieved content sent to the model stays within ``max_context_tokens``;
- when a budget is spent, the next call is made with tools disabled so the model must answer
  from what it has, and the answer's limitations say so.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from research_agent.agent.answer import Answer, build_answer
from research_agent.agent.budget import Budget, estimate_tokens
from research_agent.agent.limitations import SearchRecord, evidence_limitations
from research_agent.agent.prompt import build_instructions
from research_agent.agent.tools import (
    SEARCH_TOOL,
    SEARCH_TOOL_NAME,
    ToolArgumentError,
    format_results,
    parse_search_args,
)
from research_agent.config import Limits, ReportingConfig
from research_agent.llm import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from research_agent.retrieval import (
    AuthError,
    InvalidQueryError,
    RetrievalError,
    SearchResult,
    Work,
)

log = logging.getLogger(__name__)


class LiteratureSearch(Protocol):
    def search(
        self, query: str, *, year_from: int | None = None, max_results: int | None = None
    ) -> SearchResult: ...


class InvalidQuestionError(ValueError):
    """The question is empty or longer than ``limits.max_question_chars``."""


@dataclass(frozen=True)
class RunStats:
    iterations: int  # model calls
    search_calls: int  # searches that ran (including ones answered from the cache)
    input_tokens: int
    output_tokens: int
    works_shown: int  # distinct works the model saw in full
    cached_searches: int = 0  # of search_calls, answered from the local cache (no API cost)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class AgentResult:
    answer: Answer
    stats: RunStats


class ResearchAgent:
    def __init__(
        self,
        llm: LLMClient,
        search: LiteratureSearch,
        limits: Limits,
        reporting: ReportingConfig,
        *,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._llm = llm
        self._search = search
        self._limits = limits
        self._reporting = reporting
        self._today = today

    def run(self, question: str) -> AgentResult:
        """Answer ``question``. Raises LLMError (or AuthError from search) on hard failures."""
        question = question.strip()
        if not question:
            raise InvalidQuestionError("question must be a non-empty string")
        limits = self._limits
        if len(question) > limits.max_question_chars:
            raise InvalidQuestionError(
                f"question is {len(question)} characters; the limit is {limits.max_question_chars}"
            )

        budget = Budget(limits)
        instructions = build_instructions(limits)
        messages: list[Message] = [UserMessage(question)]
        evidence: dict[str, Work] = {}
        searches: list[SearchRecord] = []
        notes: list[str] = []
        next_input = estimate_tokens(instructions) + estimate_tokens(question)

        # Terminates: force_final_reason() is non-None once iterations + 1 >= max_iterations,
        # and a forced call always ends the loop.
        while True:
            reason = budget.force_final_reason(next_input)
            allow_tools = reason is None
            if reason:
                log.info("forcing final answer: %s", reason)
                notes.append(reason)
            response = self._llm.complete(
                instructions=instructions,
                messages=messages,
                tools=[SEARCH_TOOL],
                max_output_tokens=limits.max_output_tokens,
                allow_tools=allow_tools,
            )
            budget.record_call(response.usage)
            log.info(
                "turn %d: %d in / %d out tokens, stop=%s, tool calls=%d",
                budget.iterations,
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.stop_reason,
                len(response.tool_calls),
            )

            if not allow_tools or not response.tool_calls:
                return self._finish(response, budget, evidence, searches, notes)

            messages.append(response.to_message())
            results = self._run_tools(response.tool_calls, budget, evidence, searches, notes)
            messages.extend(results)
            next_input = (
                response.usage.input_tokens
                + response.usage.output_tokens
                + sum(estimate_tokens(r.content) for r in results)
            )

    def _finish(
        self,
        response: LLMResponse,
        budget: Budget,
        evidence: dict[str, Work],
        searches: list[SearchRecord],
        notes: list[str],
    ) -> AgentResult:
        if response.tool_calls:  # only possible when tools were disabled for this call
            notes.append(
                "The model tried to call tools after the budget was spent; those calls were "
                "ignored."
            )
        if response.stop_reason == "length":
            notes.append(
                f"The answer hit the output limit ({self._limits.max_output_tokens} tokens) and "
                "may be incomplete."
            )
        overshoot = budget.overshoot_reason()
        if overshoot:
            notes.append(overshoot)
        notes.extend(
            evidence_limitations(
                searches,
                evidence,
                limits=self._limits,
                reporting=self._reporting,
                today=self._today(),
            )
        )
        answer = build_answer(response.text, evidence, notes)
        stats = RunStats(
            iterations=budget.iterations,
            search_calls=budget.search_calls,
            input_tokens=budget.input_tokens,
            output_tokens=budget.output_tokens,
            works_shown=len(evidence),
            cached_searches=budget.cached_searches,
        )
        log.info(
            "done: %d turn(s), %d search(es) (%d cached), %d tokens",
            stats.iterations,
            stats.search_calls,
            stats.cached_searches,
            stats.total_tokens,
        )
        return AgentResult(answer=answer, stats=stats)

    def _run_tools(
        self,
        calls: Sequence[ToolCall],
        budget: Budget,
        evidence: dict[str, Work],
        searches: list[SearchRecord],
        notes: list[str],
    ) -> list[ToolResultMessage]:
        """Answer every tool call (the API requires a result for each call ID)."""
        results = []
        for call in calls:
            if call.error:
                content = f"Error: {call.error}. Call the tool again with valid JSON arguments."
            elif call.name == SEARCH_TOOL_NAME:
                content = self._do_search(call, budget, evidence, searches, notes)
            else:
                content = f"Error: unknown tool '{call.name}'."
            results.append(ToolResultMessage(call_id=call.id, content=content))
        return results

    def _do_search(
        self,
        call: ToolCall,
        budget: Budget,
        evidence: dict[str, Work],
        searches: list[SearchRecord],
        notes: list[str],
    ) -> str:
        try:
            args = parse_search_args(call.arguments)
        except ToolArgumentError as exc:
            return f"Error: {exc}."
        if budget.searches_left == 0:
            notes.append("A requested search was not run because the search limit was reached.")
            return "Error: search limit reached; no more searches are allowed. Answer now."

        try:
            result = self._search.search(
                args.query, year_from=args.year_from, max_results=args.max_results
            )
        except InvalidQueryError as exc:
            return f"Error: {exc}."  # rejected before any request, so it costs no search
        except AuthError:
            raise  # a configuration problem; answering without evidence would be misleading
        except RetrievalError as exc:
            budget.record_search()
            log.warning("search failed: %s", exc)
            notes.append(
                f"A literature search failed ({type(exc).__name__}); the evidence may be "
                "incomplete."
            )
            return (
                f"Error: the search failed ({exc}). You may retry once or answer with what "
                "you have."
            )

        budget.record_search(cached=result.from_cache)
        log.info(
            'search "%s" (year_from=%s): %d of %d matches (%s)',
            args.query,
            args.year_from,
            len(result.works),
            result.total_count,
            "cache" if result.from_cache else "network",
        )
        formatted = format_results(result, budget.context_left, evidence.keys())
        budget.record_context(formatted.context_tokens)
        evidence.update({w.id: w for w in formatted.shown})
        searches.append(
            SearchRecord(
                query=args.query,
                year_from=args.year_from,
                total_count=result.total_count,
                returned=len(result.works),
                shown=len(formatted.shown),
            )
        )
        if formatted.omitted:
            notes.append(
                f"{formatted.omitted} search result(s) were not shown to the model because the "
                f"retrieved-context limit ({self._limits.max_context_tokens} tokens) was reached."
            )
        return formatted.text
