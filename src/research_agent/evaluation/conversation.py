"""Driving evaluation conversations through the real ``ConversationService``.

Each conversation gets its own session; its turns are asked in order, so context building,
task-state updates and evidence recall are the same code the ``chat`` command and the UI use.
After each turn the details the conversation checks need are taken from the service's outcome
and the stored session: the model's relation label, recalled IDs, the context sent, and the
task state the code built.
"""

from __future__ import annotations

from typing import Any

from research_agent.agent import AgentResult
from research_agent.config import ConversationConfig
from research_agent.conversation.models import Session, TaskState
from research_agent.conversation.service import ConversationService, RunAgent, TurnOutcome
from research_agent.conversation.store import InMemorySessionStore
from research_agent.evaluation.cases import Conversation, EvalCase


def state_dict(state: TaskState) -> dict[str, Any]:
    """The task state as plain data (place labels, not coordinates)."""
    return {
        "topic": state.topic,
        "places": [p.label for p in state.places],
        "period": list(state.period) if state.period else None,
        "event_date": state.event_date,
        "event_year": state.event_year,
        "event_month": state.event_month,
        "metrics": list(state.metrics),
        "sources": list(state.sources),
        "queries": list(state.queries),
        "pending": state.pending.city if state.pending else None,
    }


def turn_details(outcome: TurnOutcome, session: Session | None) -> dict[str, Any]:
    """What a turn record keeps about the conversation (its ``conversation`` section)."""
    context = outcome.context
    return {
        "relation": outcome.turn.relation,
        "relation_source": outcome.turn.relation_source,
        "context_tokens": context.tokens if context else 0,
        "context_turns": len(context.messages) // 2 if context else 0,
        "context_omitted": context.omitted if context else 0,
        "recalled": [i for event in outcome.result.events for i in event.recalled_ids],
        "state": state_dict(session.state) if session else None,
    }


class ConversationDriver:
    """Asks evaluation turns in order; a new conversation ID starts a new session."""

    def __init__(self, run_agent: RunAgent, cfg: ConversationConfig) -> None:
        self._store = InMemorySessionStore(cfg.max_sessions, cfg.session_ttl_minutes * 60)
        self._service = ConversationService(
            run_agent, self._store, cfg, span_name="eval.turn", tags=("eval",)
        )
        self._conversation: str | None = None
        self._session: str | None = None

    def ask(self, conversation: Conversation, turn: EvalCase) -> tuple[AgentResult, dict]:
        if conversation.id != self._conversation:
            self._conversation, self._session = conversation.id, None
        outcome = self._service.ask(self._session, turn.question)
        self._session = outcome.session_id
        return outcome.result, turn_details(outcome, self._store.get(outcome.session_id))
