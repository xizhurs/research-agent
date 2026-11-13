"""Applying a finished turn to a session. Pure: no I/O, no clock (``now`` is passed in).

The task state is built from the turn's tool events (code), and the model's ``relation`` label
only decides how it is merged with the earlier state:

- ``new_topic``: the old topic is archived as one line and a new state starts from this turn.
- ``compare``: places, metrics, sources and searches are added to the earlier ones, including
  those of stored analyses the turn re-showed with ``recall_evidence`` (comparing two recalled
  places keeps both in the state even if an earlier ``update`` had replaced one).
- ``update`` / ``continue``: each setting this turn used replaces the earlier one; the rest stay.

A missing or invalid label counts as ``update`` (it only changes what the turn touched).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import replace

from research_agent.agent import AgentResult
from research_agent.agent.events import ToolEvent
from research_agent.agent.tools import (
    ENSO_TOOL_NAME,
    RAINFALL_TOOL_NAME,
    SEARCH_TOOL_NAME,
    TEMPERATURE_TOOL_NAME,
)
from research_agent.config import ConversationConfig
from research_agent.conversation.models import (
    Clarification,
    EvidenceItem,
    PlaceRef,
    Session,
    TaskState,
    ToolRef,
    TurnRecord,
)
from research_agent.enso import EnsoAnalysis
from research_agent.rainfall import Location
from research_agent.retrieval import Work
from research_agent.temperature import TemperatureExtremes, TemperatureMonthly, TemperatureYearly

SOURCES = {
    SEARCH_TOOL_NAME: "OpenAlex",
    RAINFALL_TOOL_NAME: "ERA5 (Open-Meteo)",
    TEMPERATURE_TOOL_NAME: "ERA5 (Open-Meteo)",
    ENSO_TOOL_NAME: "NOAA CPC RONI",
}
PLACE_TOOLS = (RAINFALL_TOOL_NAME, TEMPERATURE_TOOL_NAME)
QUESTION_CHARS = 100  # in a one-line summary


def place_ref(location: Location) -> PlaceRef:
    return PlaceRef(
        label=location.label,
        country_code=location.country_code,
        admin1=location.admin1,
        latitude=location.latitude,
        longitude=location.longitude,
        geonames_id=location.id,
    )


def _union(old: Iterable, new: Iterable) -> tuple:
    return tuple(dict.fromkeys((*old, *new)))


def _period(arguments) -> tuple[int | None, int | None] | None:
    start, end = arguments.get("start_year"), arguments.get("end_year")
    return (start, end) if start is not None or end is not None else None


def _metric(tool: str, args) -> str:
    """The task-state metric of a rainfall or temperature call."""
    scale = str(args.get("scale") or "daily")
    if tool == RAINFALL_TOOL_NAME:
        return str(args.get("metric") or "wet") if scale == "daily" else scale
    kind = str(args.get("metric") or "hot") if scale == "daily" else scale
    return f"temperature {kind}"  # labelled, so it is not mistaken for yearly rainfall


def turn_settings(events: Iterable[ToolEvent]) -> dict:
    """The settings a turn's successful tool calls used, as ``TaskState`` field values."""
    places: list[PlaceRef] = []
    metrics: list[str] = []
    sources: list[str] = []
    queries: list[str] = []
    settings: dict = {}
    for event in events:
        if event.status != "ok" or event.recalled_ids:
            continue
        args = event.arguments
        sources.append(SOURCES.get(event.tool, event.tool))
        if event.tool == SEARCH_TOOL_NAME:
            queries.append(str(args.get("query", "")))
        elif event.tool in PLACE_TOOLS and event.location is not None:
            places.append(place_ref(event.location))
            metrics.append(_metric(event.tool, args))
            if _period(args):
                settings["period"] = _period(args)
            for key in ("event_date", "event_year", "event_month"):
                if args.get(key) is not None:
                    settings[key] = args[key]
        elif event.tool == ENSO_TOOL_NAME:
            metrics.append("enso")
            if _period(args):
                settings["period"] = _period(args)
    for name, values in (("places", places), ("metrics", metrics), ("sources", sources),
                         ("queries", queries)):
        if values:
            settings[name] = tuple(dict.fromkeys(values))
    return settings


def recalled_settings(events: Iterable[ToolEvent], evidence: dict[str, EvidenceItem]) -> dict:
    """Places, metrics and sources of stored analyses a turn re-showed with ``recall_evidence``,
    from the stored item and the arguments of the call that produced it (code, not model text).
    Works add nothing: their searches are already in the state."""
    places: list[PlaceRef] = []
    metrics: list[str] = []
    sources: list[str] = []
    for event in events:
        if event.status != "ok":
            continue
        for item_id in event.recalled_ids:
            item = evidence.get(item_id)
            if item is None or item.ref.tool not in (*PLACE_TOOLS, ENSO_TOOL_NAME):
                continue
            sources.append(SOURCES[item.ref.tool])
            if item.ref.tool == ENSO_TOOL_NAME:
                metrics.append("enso")
                continue
            location = getattr(item.record, "location", None)
            if location is not None:
                places.append(place_ref(location))
            metrics.append(_metric(item.ref.tool, item.ref.arguments))
    named = (("places", places), ("metrics", metrics), ("sources", sources))
    return {name: tuple(dict.fromkeys(values)) for name, values in named if values}


def _pending(events: Iterable[ToolEvent], previous: Clarification | None) -> Clarification | None:
    """An ambiguous place stays pending until a rainfall or temperature call resolves one."""
    pending = previous
    for event in events:
        if event.status == "ambiguous":
            city = str(event.arguments.get("city", ""))
            pending = Clarification(city, tuple(place_ref(c) for c in event.candidates))
        elif event.tool in PLACE_TOOLS and event.status == "ok" and event.location:
            pending = None
    return pending


def merge_state(state: TaskState, settings: dict, relation: str) -> TaskState:
    if relation == "new_topic":
        return TaskState(topic=state.topic + 1, **settings)
    if relation == "compare":
        merged = dict(settings)
        for name in ("places", "metrics", "sources", "queries"):
            if name in settings:
                merged[name] = _union(getattr(state, name), settings[name])
        return replace(state, **merged)
    return replace(state, **settings)  # update / continue


def describe_state(state: TaskState) -> str:
    """One line for an archived topic or a summary."""
    parts = []
    if state.places:
        parts.append("places " + "; ".join(p.label for p in state.places))
    if state.metrics:
        parts.append("metrics " + ", ".join(state.metrics))
    if state.period:
        parts.append(f"period {state.period[0] or 'default'}-{state.period[1] or 'default'}")
    if state.queries:
        parts.append("searches " + "; ".join(f'"{q}"' for q in state.queries))
    return ", ".join(parts) or "no tool settings"


def evidence_label(item: Work | object) -> str:
    if isinstance(item, Work):
        title = (item.title or "(no title)")[:100]
        return f"{title} ({item.year or 'n.d.'})"
    if isinstance(item, EnsoAnalysis):
        return f"ENSO events ({item.index_name}), starting {item.first_year}-{item.last_year}"
    years = item.valid_years  # a rainfall or temperature record
    span = f"{years[0]}-{years[-1]}" if years else "no valid years"
    return f"{item.location.label}, {item.summary} {span}"


def _kind(item) -> str:
    if isinstance(item, Work):
        return "work"
    if isinstance(item, TemperatureYearly | TemperatureMonthly | TemperatureExtremes):
        return "temperature"
    return "enso" if isinstance(item, EnsoAnalysis) else "rainfall"


def _digest(result: AgentResult, limit: int) -> str:
    lines = []
    for claim in result.answer.claims:
        cites = f" [{', '.join(claim.sources)}]" if claim.sources else ""
        lines.append(f"({claim.kind}) {claim.text}{cites}")
    text = "\n".join(lines)
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _producer(item_id: str, events: Iterable[ToolEvent]) -> ToolRef:
    for event in events:
        if item_id == event.result_id or item_id in event.work_ids:
            return ToolRef(event.tool, dict(event.arguments))
    return ToolRef("unknown", {})


def apply_turn(
    session: Session, question: str, result: AgentResult, cfg: ConversationConfig, now: float
) -> Session:
    """Return a new session with this turn recorded. The input session is not modified."""
    new = copy.copy(session)
    new.turns = list(session.turns)
    new.archived_topics = list(session.archived_topics)
    new.evidence = dict(session.evidence)

    number = len(session.turns) + 1
    first = not session.turns
    label = result.answer.relation
    source = "model" if label else "default"
    relation = label or ("new_topic" if first else "update")

    settings = turn_settings(result.events)
    if relation == "compare" and not first:  # what is compared includes the recalled analyses
        recalled = recalled_settings(result.events, session.evidence)
        for name, values in recalled.items():
            settings[name] = _union(values, settings.get(name, ()))
    if first:  # the first turn fills topic 1; it does not open topic 2
        state, pending = merge_state(session.state, settings, "update"), None
    elif relation == "new_topic":
        new.archived_topics.append(f"Topic {session.state.topic}: {describe_state(session.state)}")
        state, pending = merge_state(session.state, settings, "new_topic"), None
    else:
        state, pending = merge_state(session.state, settings, relation), session.state.pending
    new.state = replace(state, pending=_pending(result.events, pending))

    stored: list[str] = []
    for item in result.retrieved:
        item_id = item.id
        if item_id in new.evidence:
            continue
        new.evidence[item_id] = EvidenceItem(
            id=item_id,
            kind=_kind(item),
            label=evidence_label(item),
            turn=number,
            topic=new.state.topic,
            record=item,
            ref=_producer(item_id, result.events),
        )
        stored.append(item_id)
        if item_id.startswith("R") and item_id[1:].isdigit():
            new.next_data_number = max(new.next_data_number, int(item_id[1:]) + 1)
    while len(new.evidence) > cfg.max_stored_evidence:
        del new.evidence[next(iter(new.evidence))]  # oldest first

    analyses = [i for i in stored if i.startswith("R")]
    works = len(stored) - len(analyses)
    found = [f"{i} ({new.evidence[i].label})" for i in analyses if i in new.evidence]
    if works:
        found.append(f"{works} work(s)")
    if new.state.pending and any(e.status == "ambiguous" for e in result.events):
        found.append(f"'{new.state.pending.city}' was ambiguous")
    asked = question if len(question) <= QUESTION_CHARS else question[: QUESTION_CHARS - 3] + "..."
    summary = f"Turn {number} ({relation}): {asked}" + (f" -> {'; '.join(found)}" if found else "")

    new.turns.append(
        TurnRecord(
            number=number,
            question=question,
            relation=relation,
            relation_source=source,
            topic=new.state.topic,
            answer_digest=_digest(result, cfg.answer_digest_chars),
            evidence_ids=tuple(stored),
            summary_line=summary,
            tokens=result.stats.total_tokens,
        )
    )
    new.updated_at = now
    return new
