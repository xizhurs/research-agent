"""What a turn did with its tools (``ToolEvent``) and what it is given from earlier turns
(``TurnContext``). Plain data: the agent never imports the conversation package; the
conversation package builds a ``TurnContext`` and reads the events back.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from research_agent.agent.answer import DataRecord
from research_agent.llm import Message
from research_agent.rainfall import Location
from research_agent.retrieval import Work

# "error" covers every result starting with "Error:" (refused, invalid, failed, not found).
EventStatus = Literal["ok", "ambiguous", "error"]


@dataclass(frozen=True)
class ToolEvent:
    """One tool call and its outcome, recorded by code (never from model text)."""

    tool: str
    arguments: Mapping[str, Any]
    status: EventStatus
    result_id: str | None = None  # R... for an analysis shown to the model
    work_ids: tuple[str, ...] = ()  # works shown in full by this call
    location: Location | None = None  # the place a rainfall analysis resolved to
    candidates: tuple[Location, ...] = ()  # places listed when the location was ambiguous
    recalled_ids: tuple[str, ...] = ()  # items re-shown by recall_evidence


@dataclass(frozen=True)
class TurnContext:
    """Earlier turns, prepared for one new question (see ``conversation.context``).

    ``messages`` come before the question; ``block`` (task state, evidence index, summaries) is
    put in front of the question text. ``recallable`` items can be re-shown in full with the
    ``recall_evidence`` tool; only items shown in this turn are citable.
    """

    messages: tuple[Message, ...] = ()
    block: str = ""
    recallable: Mapping[str, Work | DataRecord] = field(default_factory=dict)
    next_data_number: int = 1  # R IDs continue across the conversation
    tokens: int = 0  # estimated tokens of messages + block
    omitted: int = 0  # context items left out to stay within the conversation budget
    max_recall_per_call: int = 5
