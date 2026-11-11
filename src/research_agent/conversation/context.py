"""Choosing what of a conversation goes with the next question. Pure.

Never the whole conversation and never tool output: only the code-derived task state, recent
questions with short digests of their answers (labelled as model-written, not evidence), an
ID-and-label index of stored evidence, and one-line summaries of older turns. Items are added in
priority order until ``max_context_tokens`` is used:

1. task state and any pending clarification (always included),
2. the last turn,
3. the evidence index of the current topic, newest first,
4. the other recent turns (``recent_turns``), newest first,
5. summaries of older turns and archived topics, newest first.

Whatever does not fit is counted in ``TurnContext.omitted``; the agent reports it as a limitation.
"""

from __future__ import annotations

from research_agent.agent.budget import estimate_tokens
from research_agent.agent.events import TurnContext
from research_agent.config import ConversationConfig
from research_agent.conversation.models import Session, TaskState, TurnRecord
from research_agent.llm import AssistantMessage, Message, UserMessage

EARLIER_ANSWER = "[Earlier answer, model-written, not evidence]"
INDEX_HEADER = (
    "Stored evidence from earlier turns (show again with recall_evidence before citing; IDs from "
    "older topics appear in the summaries):"
)
SUMMARY_HEADER = "Earlier turns (summaries written by code):"
MAX_LISTED = 8  # places, searches or candidates shown in the state section


def render_state(state: TaskState) -> str:
    lines = ["Conversation state (from earlier tool calls, written by code):"]
    if state.places:
        places = state.places[-MAX_LISTED:]
        listed = "; ".join(f"{p.label} ({p.latitude:.2f}, {p.longitude:.2f})" for p in places)
        lines.append(f"- Places: {listed}")
    if state.period:
        start, end = state.period
        lines.append(f"- Period: {start or 'default start'} to {end or 'default end'}")
    if state.event_date:
        lines.append(f"- Event date: {state.event_date}")
    if state.event_year:
        lines.append(f"- Event year: {state.event_year}")
    if state.event_month:
        lines.append(f"- Event month: {state.event_month}")
    if state.metrics:
        lines.append("- Metrics: " + ", ".join(state.metrics))
    if state.sources:
        lines.append("- Sources: " + ", ".join(state.sources))
    if state.queries:
        lines.append(
            "- Literature searches: " + "; ".join(f'"{q}"' for q in state.queries[-MAX_LISTED:])
        )
    if len(lines) == 1:
        lines.append("- No tool settings yet.")
    if state.pending:
        listed = "; ".join(
            f"{i}) {c.label}" for i, c in enumerate(state.pending.candidates[:MAX_LISTED], start=1)
        )
        lines.append(
            f"Pending clarification: '{state.pending.city}' matched several places: {listed}."
        )
    return "\n".join(lines)


def _turn_messages(turn: TurnRecord) -> tuple[Message, Message]:
    return (
        UserMessage(turn.question),
        AssistantMessage(text=f"{EARLIER_ANSWER}\n{turn.answer_digest}"),
    )


def _messages_tokens(turn: TurnRecord) -> int:
    return sum(estimate_tokens(m.text or "") for m in _turn_messages(turn))


def build_context(session: Session, cfg: ConversationConfig) -> TurnContext | None:
    """The context for the next question, or None for the first question of a session."""
    if not session.turns:
        return None
    budget = cfg.max_context_tokens
    state_text = render_state(session.state)
    # Every piece is costed with its separator, and a header with its section's first line, so
    # the sum of the costs is never below the size of the assembled block.
    used = estimate_tokens(state_text + "\n\n")  # always included, even beyond the budget
    omitted = 0

    def fits(cost: int) -> bool:
        nonlocal used, omitted
        if used + cost <= budget:
            used += cost
            return True
        omitted += 1
        return False

    turns = session.turns
    recent = turns[-cfg.recent_turns :]
    older = turns[: -cfg.recent_turns] if len(turns) > cfg.recent_turns else []

    included: set[int] = set()
    if fits(_messages_tokens(recent[-1])):
        included.add(recent[-1].number)

    index_lines: list[str] = []
    topic_items = [e for e in session.evidence.values() if e.topic == session.state.topic]
    for item in reversed(topic_items):  # newest first
        line = f"- {item.id}: {item.label} (turn {item.turn})"
        header = "" if index_lines else INDEX_HEADER + "\n"
        if fits(estimate_tokens(line + "\n") + estimate_tokens(header + "\n")):
            index_lines.append(line)

    for turn in reversed(recent[:-1]):
        if fits(_messages_tokens(turn)):
            included.add(turn.number)

    summaries: list[str] = []
    lines = [t.summary_line for t in reversed(older)]
    lines += [f"Earlier {line}" for line in reversed(session.archived_topics)]
    for line in lines:
        header = "" if summaries else SUMMARY_HEADER + "\n"
        if fits(estimate_tokens(line + "\n") + estimate_tokens(header + "\n")):
            summaries.append(line)

    messages: list[Message] = []
    for turn in turns:  # chronological order
        if turn.number in included:
            messages.extend(_turn_messages(turn))

    sections = [state_text]
    if index_lines:
        sections.append(INDEX_HEADER + "\n" + "\n".join(index_lines))
    if summaries:
        sections.append(SUMMARY_HEADER + "\n" + "\n".join(reversed(summaries)))
    block = "\n\n".join(sections)
    tokens = estimate_tokens(block) + sum(estimate_tokens(m.text or "") for m in messages)
    return TurnContext(
        messages=tuple(messages),
        block=block,
        recallable={item.id: item.record for item in session.evidence.values()},
        next_data_number=session.next_data_number,
        tokens=tokens,
        omitted=omitted,
        max_recall_per_call=cfg.max_recall_per_call,
    )
