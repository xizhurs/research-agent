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
from typing import Any, Protocol

from research_agent import tracing
from research_agent.agent.answer import Answer, DataRecord, build_answer
from research_agent.agent.budget import Budget, estimate_tokens
from research_agent.agent.events import ToolEvent, TurnContext
from research_agent.agent.limitations import SearchRecord, evidence_limitations
from research_agent.agent.prompt import build_instructions
from research_agent.agent.tools import (
    ENSO_TOOL,
    ENSO_TOOL_NAME,
    RAINFALL_TOOL,
    RAINFALL_TOOL_NAME,
    RECALL_TOOL,
    RECALL_TOOL_NAME,
    SEARCH_TOOL,
    SEARCH_TOOL_NAME,
    ToolArgumentError,
    format_enso,
    format_rainfall,
    format_recalled,
    format_results,
    parse_enso_args,
    parse_rainfall_args,
    parse_recall_args,
    parse_search_args,
)
from research_agent.config import Limits, ReportingConfig
from research_agent.enso import EnsoAnalysis, EnsoError, InvalidEnsoQueryError
from research_agent.llm import (
    AssistantMessage,
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
    Location,
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


def _tool_calls_json(calls: Sequence[ToolCall]) -> list[dict[str, Any]]:
    return [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls]


def _messages_json(instructions: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
    """The conversation as role/content records for a trace. ``provider_state`` is left out."""
    out: list[dict[str, Any]] = [{"role": "system", "content": instructions}]
    for m in messages:
        if isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.text})
        elif isinstance(m, AssistantMessage):
            calls = _tool_calls_json(m.tool_calls)
            out.append({"role": "assistant", "content": m.text, "tool_calls": calls})
        else:
            out.append({"role": "tool", "tool_call_id": m.call_id, "content": m.content})
    return out


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
    events: tuple[ToolEvent, ...] = ()  # every tool call and its outcome, in order
    # Every work and analysis shown to the model in full this turn (new or recalled).
    retrieved: tuple[Work | DataRecord, ...] = ()


@dataclass
class _RunState:
    """Everything one question accumulates. Never shared between runs."""

    budget: Budget
    evidence: dict[str, Work] = field(default_factory=dict)  # works the model saw in full
    analyses: dict[str, DataRecord] = field(default_factory=dict)  # R1, R2... shown
    searches: list[SearchRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    trace: list[tuple[str, ...]] = field(default_factory=list)
    events: list[ToolEvent] = field(default_factory=list)
    context: TurnContext | None = None  # earlier turns of a conversation, if any
    next_data_number: int = 1  # the number of the next R ID
    reused: set[str] = field(default_factory=set)  # stored items shown again this turn
    ambiguous: tuple[Location, ...] = ()  # set by a rainfall call whose place was ambiguous


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

    def run(self, question: str, context: TurnContext | None = None) -> AgentResult:
        """Answer ``question``. Raises LLMError (or AuthError from search) on hard failures.

        ``context`` carries earlier turns of a conversation. Budgets, error handling and citation
        validation are the same for every turn; earlier items become citable only when shown
        again in this turn.
        """
        with tracing.observe("invoke_agent", "agent") as span:
            tracing.set_content(span, input=question)
            result = self._run(question, context)
            stats = result.stats
            span.set_attributes(
                {
                    "research_agent.iterations": stats.iterations,
                    "research_agent.search_calls": stats.search_calls,
                    "research_agent.cached_searches": stats.cached_searches,
                    "research_agent.rainfall_calls": stats.rainfall_calls,
                    "research_agent.cached_rainfall": stats.cached_rainfall,
                    "research_agent.works_shown": stats.works_shown,
                    "research_agent.input_tokens": stats.input_tokens,
                    "research_agent.output_tokens": stats.output_tokens,
                    "research_agent.structured": result.answer.structured,
                    "research_agent.limitations": len(result.answer.limitations),
                    "research_agent.rejected_citations": len(result.answer.rejected_citations),
                }
            )
            tracing.set_content(span, output=tracing.answer_json(result.answer))
            return result

    def _run(self, question: str, context: TurnContext | None) -> AgentResult:
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
            limits,
            rainfall=self._rainfall is not None,
            enso=self._enso is not None,
            conversation=context is not None,
        )
        tools = [SEARCH_TOOL]
        if self._rainfall is not None:
            tools.append(RAINFALL_TOOL)
        if self._enso is not None:
            tools.append(ENSO_TOOL)
        messages: list[Message] = [UserMessage(question)]
        next_input = estimate_tokens(instructions) + estimate_tokens(question)
        if context is not None:
            state.context = context
            state.next_data_number = context.next_data_number
            if context.recallable:
                tools.append(RECALL_TOOL)
            text = f"{context.block}\n\nCurrent question: {question}" if context.block else question
            messages = [*context.messages, UserMessage(text)]
            next_input += context.tokens
            if context.omitted:
                state.notes.append(
                    f"{context.omitted} earlier part(s) of the conversation were left out to stay "
                    "within the conversation context limit."
                )

        # Terminates: force_final_reason() is non-None once iterations + 1 >= max_iterations,
        # and a forced call always ends the loop.
        while True:
            reason = budget.force_final_reason(next_input)
            allow_tools = reason is None
            if reason:
                log.info("forcing final answer: %s", reason)
                state.notes.append(reason)
            turn = {
                "research_agent.turn": budget.iterations + 1,
                "research_agent.allow_tools": allow_tools,
            }
            with tracing.observe("chat", "generation", turn) as span:
                if span.is_recording():
                    tracing.set_content(span, input=_messages_json(instructions, messages))
                response = self._llm.complete(
                    instructions=instructions,
                    messages=messages,
                    tools=tools,
                    max_output_tokens=limits.max_output_tokens,
                    allow_tools=allow_tools,
                )
                usage = response.usage
                span.set_attributes(
                    {
                        "gen_ai.usage.input_tokens": usage.input_tokens,
                        "gen_ai.usage.output_tokens": usage.output_tokens,
                        "gen_ai.response.finish_reasons": [response.stop_reason],
                        "research_agent.tool_calls": len(response.tool_calls),
                    }
                )
                tracing.set_content(
                    span,
                    output={
                        "role": "assistant",
                        "content": response.text,
                        "tool_calls": _tool_calls_json(response.tool_calls),
                    },
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
                reused=len(state.reused),
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
        return AgentResult(
            answer=answer,
            stats=stats,
            events=tuple(state.events),
            retrieved=(*state.evidence.values(), *state.analyses.values()),
        )

    def _run_tools(self, calls: Sequence[ToolCall], state: _RunState) -> list[ToolResultMessage]:
        """Answer every tool call (the API requires a result for each call ID)."""
        results = []
        for call in calls:
            attrs = {"gen_ai.tool.name": call.name, "gen_ai.tool.call.id": call.id}
            with tracing.observe(f"execute_tool {call.name}", "tool", attrs) as span:
                tracing.set_content(span, input=call.arguments)
                content = self._run_tool(call, state)
                failed = content.startswith("Error:")  # every refused or failed call says so
                span.set_attribute("research_agent.tool.error", failed)
                if failed:
                    tracing.mark_failed(span, content)
                tracing.set_content(span, output=content)
            results.append(ToolResultMessage(call_id=call.id, content=content))
        return results

    def _run_tool(self, call: ToolCall, state: _RunState) -> str:
        """Run one call and record what it did, read from what it added to the run state."""
        works_before, analyses_before = set(state.evidence), set(state.analyses)
        state.ambiguous = ()
        content = self._dispatch(call, state)
        new_works = tuple(w for w in state.evidence if w not in works_before)
        new_analyses = tuple(a for a in state.analyses if a not in analyses_before)
        recalled = tuple(i for i in (*new_works, *new_analyses) if i in state.reused)
        fresh = [a for a in new_analyses if a not in recalled]
        record = state.analyses[fresh[0]] if fresh else None
        if state.ambiguous:
            status = "ambiguous"
        else:
            status = "error" if content.startswith("Error:") else "ok"
        state.events.append(
            ToolEvent(
                tool=call.name,
                arguments=dict(call.arguments),
                status=status,
                result_id=fresh[0] if fresh else None,
                work_ids=tuple(w for w in new_works if w not in recalled),
                location=getattr(record, "location", None),
                candidates=state.ambiguous,
                recalled_ids=recalled,
            )
        )
        return content

    def _dispatch(self, call: ToolCall, state: _RunState) -> str:
        if call.error:
            return f"Error: {call.error}. Call the tool again with valid JSON arguments."
        if call.name == SEARCH_TOOL_NAME:
            return self._do_search(call, state)
        if call.name == RAINFALL_TOOL_NAME and self._rainfall is not None:
            return self._do_rainfall(call, self._rainfall, state)
        if call.name == ENSO_TOOL_NAME and self._enso is not None:
            return self._do_enso(call, self._enso, state)
        if call.name == RECALL_TOOL_NAME and state.context and state.context.recallable:
            return self._do_recall(call, state)
        return f"Error: unknown tool '{call.name}'."

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
            if isinstance(exc, AmbiguousLocationError):
                state.ambiguous = tuple(c for c in exc.candidates if isinstance(c, Location))
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
        analysis = replace(analysis, id=f"R{state.next_data_number}")
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
        state.next_data_number += 1
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
        analysis = replace(analysis, id=f"R{state.next_data_number}")
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
        state.next_data_number += 1
        return text

    def _do_recall(self, call: ToolCall, state: _RunState) -> str:
        """Show stored items from earlier turns again. Costs context tokens, no search/analysis."""
        assert state.context is not None
        stored = state.context.recallable
        try:
            ids = parse_recall_args(call.arguments, state.context.max_recall_per_call)
        except ToolArgumentError as exc:
            return f"Error: {exc}."
        unknown = [i for i in ids if i not in stored]
        if len(unknown) == len(ids):
            return (
                f"Error: not in the stored evidence: {', '.join(unknown)}. Use IDs from the "
                "'Stored evidence' list, or run the tool again."
            )
        parts: list[str] = []
        for item_id in ids:
            if item_id in unknown:
                continue
            if item_id in state.evidence or item_id in state.analyses:
                parts.append(f"[{item_id}] (already shown in this turn)")
                continue
            item = stored[item_id]
            text = format_recalled(item)
            tokens = estimate_tokens(text)
            if tokens > state.budget.context_left:
                parts.append(f"[{item_id}] not shown: the retrieved-context budget is used up.")
                state.notes.append(
                    "An item from an earlier turn was not shown again because the "
                    f"retrieved-context limit ({self._limits.max_context_tokens} tokens) was "
                    "reached."
                )
                continue
            state.budget.record_context(tokens)
            if isinstance(item, Work):
                state.evidence[item_id] = item
            else:
                state.analyses[item_id] = item
            state.reused.add(item_id)
            parts.append(text)
        if unknown:
            parts.append(f"Not in the stored evidence: {', '.join(unknown)}.")
        log.info("recall: %d item(s) requested, %d reused so far", len(ids), len(state.reused))
        return "\n\n".join(parts)
