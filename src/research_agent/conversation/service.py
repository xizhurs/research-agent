"""One conversation turn: load the session, build the context, run the agent, record the turn.

Platform-independent: the agent run is a callable and the store an interface, so the CLI, the
web UI and tests use the same code. A failed turn leaves the session unchanged.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from research_agent import tracing
from research_agent.agent import AgentResult
from research_agent.agent.events import TurnContext
from research_agent.config import ConversationConfig
from research_agent.conversation.context import build_context
from research_agent.conversation.models import Session, TurnRecord
from research_agent.conversation.state import apply_turn
from research_agent.conversation.store import SessionStore

RunAgent = Callable[[str, TurnContext | None], AgentResult]


@dataclass(frozen=True)
class TurnOutcome:
    session_id: str
    result: AgentResult
    turn: TurnRecord
    context: TurnContext | None  # what was sent from earlier turns (None on a first turn)


class ConversationService:
    def __init__(
        self,
        run_agent: RunAgent,
        store: SessionStore,
        cfg: ConversationConfig,
        *,
        span_name: str = "chat.turn",
        tags: Sequence[str] = ("chat",),
        clock: Callable[[], float] = time.time,
        new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._run_agent = run_agent
        self._store = store
        self._cfg = cfg
        self._span_name = span_name
        self._tags = tuple(tags)
        self._clock = clock
        self._new_id = new_id

    def turns(self, session_id: str | None) -> tuple[TurnRecord, ...]:
        """The recorded turns of a session (empty for an unknown or expired one)."""
        session = self._store.get(session_id) if session_id else None
        return tuple(session.turns) if session else ()

    def ask(self, session_id: str | None, question: str) -> TurnOutcome:
        """Answer ``question`` in ``session_id``'s conversation.

        An unknown, expired or missing ID starts a new session with a new ID (it never reads
        another session). Errors from the agent are re-raised and the session is not changed.
        """
        session = self._store.get(session_id) if session_id else None
        if session is None:
            now = self._clock()
            session = Session(id=self._new_id(), created_at=now, updated_at=now)
        context = build_context(session, self._cfg)
        attributes = {
            "langfuse.session.id": session.id,
            "research_agent.turn": len(session.turns) + 1,
            "research_agent.context.tokens": context.tokens if context else 0,
            "research_agent.context.turns_sent": len(context.messages) // 2 if context else 0,
            "research_agent.context.omitted": context.omitted if context else 0,
            "research_agent.context.recallable": len(context.recallable) if context else 0,
        }
        with tracing.observe(self._span_name, "span", attributes, tags=self._tags) as span:
            tracing.set_content(span, input=question)
            result = self._run_agent(question, context)
            updated = apply_turn(session, question, result, self._cfg, self._clock())
            turn = updated.turns[-1]
            recalled = [i for e in result.events for i in e.recalled_ids]
            span.set_attributes(
                {
                    "research_agent.relation": turn.relation,
                    "research_agent.relation_source": turn.relation_source,
                    "research_agent.recalled": len(recalled),
                    "research_agent.stored_evidence": len(updated.evidence),
                }
            )
            tracing.set_content(span, output=tracing.answer_json(result.answer))
        self._store.put(updated)
        return TurnOutcome(updated.id, result, turn, context)
