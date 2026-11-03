"""The single-agent loop: model turns, tool execution, budgets and forced finalisation.

Guarantees, all enforced here in code (not in the prompt):
- at most ``max_iterations`` model calls per question;
- at most ``max_search_calls`` searches and ``max_rainfall_calls`` rainfall analyses actually run;
- the retrieved content (literature and rainfall results) sent to the model stays within
  ``max_context_tokens``;
- when every tool budget is spent, the next call is made with tools disabled so the model must
  answer from what it has, and the answer's limitations say so.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Protocol

from research_agent.agent.answer import Answer, DataRecord, build_answer
from research_agent.agent.budget import Budget, estimate_tokens
from research_agent.agent.limitations import SearchRecord, evidence_limitations
from research_agent.agent.prompt import build_instructions
from research_agent.agent.tools import (
    ENSO_TOOL,
    ENSO_TOOL_NAME,
    RAINFALL_TOOL,
    RAINFALL_TOOL_NAME,
    SEARCH_TOOL,
    SEARCH_TOOL_NAME,
    ToolArgumentError,
    format_enso,
    format_rainfall,
    format_results,
    parse_enso_args,
    parse_rainfall_args,
    parse_search_args,
)
from research_agent.config import Limits, ReportingConfig
from research_agent.enso import EnsoAnalysis, EnsoError, InvalidEnsoQueryError
from research_agent.llm import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from research_agent.rainfall import (
    AmbiguousLocationError,
    Archive,
    InvalidRainfallQueryError,
    LocationNotFoundError,
    RainfallError,
    RainfallRecord,
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


class RainfallAnalyzer(Protocol):
    def analyze(
        self,
        city: str,
        *,
        country_code: str | None = None,
        region: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        event_date: date | None = None,
        metric: str = "wet",
        event_year: int | None = None,
    ) -> tuple[RainfallRecord, Archive]: ...


class EnsoAnalyzer(Protocol):
    def analyze(
        self,
        *,
        phase: str = "both",
        start_year: int | None = None,
        end_year: int | None = None,
        year: int | None = None,
    ) -> EnsoAnalysis: ...


def _trace_name(call: ToolCall) -> str:
    """The tool as the model chose it; the rainfall tool also shows its metric, e.g. ``[dry]``."""
    if call.name != RAINFALL_TOOL_NAME:
        return call.name
    return f"{call.name}[{call.arguments.get('metric') or 'wet'}]"


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
    rainfall_calls: int = 0  # rainfall analyses that ran (including ones that failed)
    cached_rainfall: int = 0  # of rainfall_calls, answered from the local cache
    # What the model asked for on each turn, in order: the tool names it requested (even if the
    # call was refused or invalid). An empty tuple is a turn with no tool call, e.g. the answer.
    tool_trace: tuple[tuple[str, ...], ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class AgentResult:
    answer: Answer
    stats: RunStats


@dataclass
class _RunState:
    """Everything one question accumulates. Never shared between runs."""

    budget: Budget
    evidence: dict[str, Work] = field(default_factory=dict)  # works the model saw in full
    analyses: dict[str, DataRecord] = field(default_factory=dict)  # R1, R2... shown
    searches: list[SearchRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    trace: list[tuple[str, ...]] = field(default_factory=list)


class ResearchAgent:
    def __init__(
        self,
        llm: LLMClient,
        search: LiteratureSearch,
        limits: Limits,
        reporting: ReportingConfig,
        *,
        rainfall: RainfallAnalyzer | None = None,
        enso: EnsoAnalyzer | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        if enso is not None and rainfall is None:
            raise ValueError("the ENSO tool is offered together with the rainfall tool")
        self._llm = llm
        self._search = search
        self._rainfall = rainfall
        self._enso = enso
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

        state = _RunState(Budget(limits, rainfall_enabled=self._rainfall is not None))
        budget = state.budget
        instructions = build_instructions(
            limits, rainfall=self._rainfall is not None, enso=self._enso is not None
        )
        tools = [SEARCH_TOOL]
        if self._rainfall is not None:
            tools.append(RAINFALL_TOOL)
        if self._enso is not None:
            tools.append(ENSO_TOOL)
        messages: list[Message] = [UserMessage(question)]
        next_input = estimate_tokens(instructions) + estimate_tokens(question)

        # Terminates: force_final_reason() is non-None once iterations + 1 >= max_iterations,
        # and a forced call always ends the loop.
        while True:
            reason = budget.force_final_reason(next_input)
            allow_tools = reason is None
            if reason:
                log.info("forcing final answer: %s", reason)
                state.notes.append(reason)
            response = self._llm.complete(
                instructions=instructions,
                messages=messages,
                tools=tools,
                max_output_tokens=limits.max_output_tokens,
                allow_tools=allow_tools,
            )
            budget.record_call(response.usage)
            state.trace.append(tuple(_trace_name(c) for c in response.tool_calls))
            log.info(
                "turn %d: %d in / %d out tokens, stop=%s, tool calls=%d",
                budget.iterations,
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.stop_reason,
                len(response.tool_calls),
            )

            if not allow_tools or not response.tool_calls:
                return self._finish(response, state)

            messages.append(response.to_message())
            results = self._run_tools(response.tool_calls, state)
            messages.extend(results)
            next_input = (
                response.usage.input_tokens
                + response.usage.output_tokens
                + sum(estimate_tokens(r.content) for r in results)
            )

    def _finish(self, response: LLMResponse, state: _RunState) -> AgentResult:
        budget, notes = state.budget, state.notes
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
                state.searches,
                state.evidence,
                limits=self._limits,
                reporting=self._reporting,
                today=self._today(),
                analyses=list(state.analyses.values()),
                rainfall_calls=budget.rainfall_calls,
            )
        )
        answer = build_answer(response.text, state.evidence, notes, state.analyses)
        stats = RunStats(
            iterations=budget.iterations,
            search_calls=budget.search_calls,
            input_tokens=budget.input_tokens,
            output_tokens=budget.output_tokens,
            works_shown=len(state.evidence),
            cached_searches=budget.cached_searches,
            rainfall_calls=budget.rainfall_calls,
            cached_rainfall=budget.cached_rainfall,
            tool_trace=tuple(state.trace),
        )
        log.info(
            "done: %d turn(s), %d search(es) (%d cached), %d rainfall analysis(es) (%d cached), "
            "%d tokens",
            stats.iterations,
            stats.search_calls,
            stats.cached_searches,
            stats.rainfall_calls,
            stats.cached_rainfall,
            stats.total_tokens,
        )
        return AgentResult(answer=answer, stats=stats)

    def _run_tools(self, calls: Sequence[ToolCall], state: _RunState) -> list[ToolResultMessage]:
        """Answer every tool call (the API requires a result for each call ID)."""
        results = []
        for call in calls:
            if call.error:
                content = f"Error: {call.error}. Call the tool again with valid JSON arguments."
            elif call.name == SEARCH_TOOL_NAME:
                content = self._do_search(call, state)
            elif call.name == RAINFALL_TOOL_NAME and self._rainfall is not None:
                content = self._do_rainfall(call, self._rainfall, state)
            elif call.name == ENSO_TOOL_NAME and self._enso is not None:
                content = self._do_enso(call, self._enso, state)
            else:
                content = f"Error: unknown tool '{call.name}'."
            results.append(ToolResultMessage(call_id=call.id, content=content))
        return results

    def _do_search(self, call: ToolCall, state: _RunState) -> str:
        budget, notes = state.budget, state.notes
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
        formatted = format_results(result, budget.context_left, state.evidence.keys())
        budget.record_context(formatted.context_tokens)
        state.evidence.update({w.id: w for w in formatted.shown})
        state.searches.append(
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

    def _do_rainfall(self, call: ToolCall, analyzer: RainfallAnalyzer, state: _RunState) -> str:
        budget, notes = state.budget, state.notes
        try:
            args = parse_rainfall_args(call.arguments)
        except ToolArgumentError as exc:
            return f"Error: {exc}."
        if budget.rainfall_left == 0:
            notes.append(
                "A requested rainfall analysis was not run because the rainfall-analysis limit "
                "was reached."
            )
            return (
                "Error: rainfall-analysis limit reached; no more analyses are allowed. Answer now."
            )

        try:
            analysis, _daily = analyzer.analyze(  # the model sees statistics, not daily values
                args.city,
                country_code=args.country_code,
                region=args.region,
                start_year=args.start_year,
                end_year=args.end_year,
                event_date=args.event_date,
                metric=args.metric,
                event_year=args.event_year,
            )
        except InvalidRainfallQueryError as exc:
            return f"Error: {exc}."  # rejected before any request, so it costs no analysis
        except (AmbiguousLocationError, LocationNotFoundError) as exc:
            budget.record_rainfall()
            kind = "ambiguous" if isinstance(exc, AmbiguousLocationError) else "not found"
            notes.append(
                f"The location '{args.city}' was {kind}; no rainfall analysis was run for it."
            )
            return f"Error: {exc}."
        except RainfallError as exc:
            budget.record_rainfall()
            log.warning("rainfall analysis failed: %s", exc)
            notes.append(
                f"A rainfall analysis failed ({type(exc).__name__}); the numerical evidence may "
                "be incomplete."
            )
            return (
                f"Error: the rainfall analysis failed ({exc}). You may retry once or answer with "
                "what you have."
            )

        budget.record_rainfall(cached=analysis.provenance.from_cache)
        analysis = replace(analysis, id=f"R{len(state.analyses) + 1}")
        text = format_rainfall(analysis)
        tokens = estimate_tokens(text)
        log.info(
            'rainfall[%s] "%s": %d valid year(s), ~%d tokens (%s)',
            analysis.metric,
            analysis.location.label,
            len(analysis.valid_years),
            tokens,
            "cache" if analysis.provenance.from_cache else "network",
        )
        if tokens > budget.context_left:
            notes.append(
                "A rainfall result was not shown to the model because the retrieved-context "
                f"limit ({self._limits.max_context_tokens} tokens) was reached."
            )
            return "Error: result omitted because the retrieved-context budget is used up."
        budget.record_context(tokens)
        state.analyses[analysis.id] = analysis  # only results the model saw in full are citable
        return text

    def _do_enso(self, call: ToolCall, analyzer: EnsoAnalyzer, state: _RunState) -> str:
        """Same contract as ``_do_rainfall``; ENSO calls share the ``max_rainfall_calls`` limit."""
        budget, notes = state.budget, state.notes
        try:
            args = parse_enso_args(call.arguments)
        except ToolArgumentError as exc:
            return f"Error: {exc}."
        if budget.rainfall_left == 0:
            notes.append(
                "A requested ENSO analysis was not run because the rainfall/ENSO analysis limit "
                "was reached."
            )
            return (
                "Error: analysis limit reached (rainfall and ENSO analyses share it); no more "
                "analyses are allowed. Answer now."
            )

        try:
            analysis = analyzer.analyze(
                phase=args.phase,
                start_year=args.start_year,
                end_year=args.end_year,
                year=args.year,
            )
        except InvalidEnsoQueryError as exc:
            return f"Error: {exc}."  # rejected before any request, so it costs no analysis
        except EnsoError as exc:
            budget.record_rainfall()
            log.warning("ENSO analysis failed: %s", exc)
            notes.append(
                f"An ENSO analysis failed ({type(exc).__name__}); the numerical evidence may be "
                "incomplete."
            )
            return (
                f"Error: the ENSO analysis failed ({exc}). You may retry once or answer with "
                "what you have."
            )

        budget.record_rainfall(cached=analysis.from_cache)
        analysis = replace(analysis, id=f"R{len(state.analyses) + 1}")
        text = format_enso(analysis)
        tokens = estimate_tokens(text)
        log.info(
            "enso %s: %d event(s), ~%d tokens (%s)",
            analysis.index_name,
            len(analysis.events),
            tokens,
            "cache" if analysis.from_cache else "network",
        )
        if tokens > budget.context_left:
            notes.append(
                "An ENSO result was not shown to the model because the retrieved-context "
                f"limit ({self._limits.max_context_tokens} tokens) was reached."
            )
            return "Error: result omitted because the retrieved-context budget is used up."
        budget.record_context(tokens)
        state.analyses[analysis.id] = analysis  # only results the model saw in full are citable
        return text
