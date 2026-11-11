"""Conversation data: sessions, turns, code-derived task state and stored evidence.

Everything a later turn relies on comes from code (tool events), never from model text: the task
state is built from tool calls, and stored evidence is the exact objects the model was shown.
Earlier answers are kept only as short digests, labelled as model-written.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from research_agent.agent.answer import DataRecord
from research_agent.retrieval import Work

RelationSource = Literal["model", "default"]
EvidenceKind = Literal["work", "rainfall", "enso"]


@dataclass(frozen=True)
class PlaceRef:
    """A place a rainfall tool call resolved to (or one of the candidates it listed)."""

    label: str
    country_code: str | None
    admin1: str | None
    latitude: float
    longitude: float
    geonames_id: int


@dataclass(frozen=True)
class Clarification:
    """A place name that was ambiguous, with the candidates the tool listed."""

    city: str
    candidates: tuple[PlaceRef, ...]


@dataclass(frozen=True)
class TaskState:
    """The settings of the current topic, taken from the tool calls that ran."""

    topic: int = 1
    places: tuple[PlaceRef, ...] = ()  # newest last
    period: tuple[int | None, int | None] | None = None  # (start_year, end_year) as requested
    event_date: str | None = None
    event_year: int | None = None
    event_month: int | None = None
    metrics: tuple[str, ...] = ()  # "wet", "dry" (daily), "yearly", "monthly", "enso"
    sources: tuple[str, ...] = ()  # "OpenAlex", "ERA5 (Open-Meteo)", "NOAA CPC RONI"
    queries: tuple[str, ...] = ()  # literature searches that ran
    pending: Clarification | None = None

    @property
    def empty(self) -> bool:
        return self == TaskState(topic=self.topic)


@dataclass(frozen=True)
class ToolRef:
    """How an item was produced, so a persistent store could recompute it."""

    tool: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    kind: EvidenceKind
    label: str  # one line for the evidence index (title and year, or place and metric)
    turn: int
    topic: int
    record: Work | DataRecord  # the object the model saw (in memory)
    ref: ToolRef


@dataclass(frozen=True)
class TurnRecord:
    number: int
    question: str
    relation: str  # one of answer.RELATIONS
    relation_source: RelationSource
    topic: int
    answer_digest: str  # claims as "(kind) text [IDs]", cut to a length limit; not evidence
    evidence_ids: tuple[str, ...]  # new items stored by this turn
    summary_line: str  # one line written by code, used once the turn is no longer recent
    tokens: int  # model tokens the turn used


@dataclass
class Session:
    id: str
    created_at: float
    updated_at: float
    turns: list[TurnRecord] = field(default_factory=list)
    state: TaskState = field(default_factory=TaskState)
    archived_topics: list[str] = field(default_factory=list)  # one line per finished topic
    evidence: dict[str, EvidenceItem] = field(default_factory=dict)  # insertion = age order
    next_data_number: int = 1
