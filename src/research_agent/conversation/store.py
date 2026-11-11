"""Where sessions live. ``SessionStore`` is the interface; ``InMemorySessionStore`` the default.

A persistent store (SQLite, Redis, ...) can implement the same three methods. It would save turns,
task state and ``EvidenceItem.ref`` and rebuild ``EvidenceItem.record`` by re-running the tool
(mostly a cache hit), because the records themselves are large Python objects.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol

from research_agent.conversation.models import Session


class SessionStore(Protocol):
    def get(self, session_id: str) -> Session | None: ...

    def put(self, session: Session) -> None: ...

    def delete(self, session_id: str) -> None: ...


class InMemorySessionStore:
    """Process memory only: lost on restart and not shared between server instances.

    Bounded: at most ``max_sessions`` (least recently used dropped first), and a session unused
    for ``ttl_seconds`` is dropped. Safe to use from several threads.
    """

    def __init__(
        self,
        max_sessions: int,
        ttl_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_sessions
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, tuple[Session, float]] = OrderedDict()

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            session, used = entry
            now = self._clock()
            if now - used > self._ttl:
                del self._sessions[session_id]
                return None
            self._sessions[session_id] = (session, now)
            self._sessions.move_to_end(session_id)
            return session

    def put(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = (session, self._clock())
            self._sessions.move_to_end(session.id)
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
