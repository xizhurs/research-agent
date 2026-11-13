"""Replay a stored run: the recorded model replies through the current code (offline, free).

``ReplayLLMClient`` answers call k with the reply recorded for call k, but only if the request
is byte-for-byte the recorded one (``request_hash``). Then the code after the model's decisions
(tool execution from the frozen cache, answer parsing, citation validation, limitations, case
checks) runs for real and ``compare_answers`` says what changed. If anything before the model's
decision changed (prompt, tool definitions, a tool's output, when the loop forces the final
answer), the case stops as "diverged" with the first reason found. Nothing is sent anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from research_agent.evaluation.recording import fingerprint, request_hash, tools_hash
from research_agent.llm import (
    LLMResponse,
    Message,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    Usage,
    UserMessage,
)

EXCERPT = 60


class ReplayDiverged(Exception):
    """A request differed from the recorded one (or the call count did): replay stops."""

    def __init__(self, call: int, reason: str, detail: str = "") -> None:
        super().__init__(f"call {call}: {reason}" + (f" ({detail})" if detail else ""))
        self.call = call
        self.reason = reason
        self.detail = detail


def first_difference(old: str, new: str) -> str:
    """Both texts around the first character where they differ."""
    i = next((k for k, (a, b) in enumerate(zip(old, new, strict=False)) if a != b), None)
    if i is None:
        i = min(len(old), len(new))
    start = max(0, i - EXCERPT // 3)
    return (
        f"at character {i}: was {old[start : i + EXCERPT]!r}, now {new[start : i + EXCERPT]!r}"
    )


def replay_date(record: dict[str, Any]) -> date:
    """The 'today' the original run used (older records: the date it was recorded)."""
    return date.fromisoformat(record.get("today") or record["recorded_at"][:10])


class ReplayLLMClient:
    """An ``LLMClient`` that returns one record's recorded replies (``load`` per case)."""

    def __init__(self) -> None:
        self._record: dict[str, Any] = {"llm_calls": [], "tool_calls": []}
        self._next = 0

    def load(self, record: dict[str, Any]) -> None:
        self._record = record
        self._next = 0

    def finish(self) -> None:
        """Raise if the run ended before using every recorded reply."""
        left = len(self._record["llm_calls"]) - self._next
        if left:
            raise ReplayDiverged(
                self._next + 1, "the run finished early", f"{left} recorded model call(s) unused"
            )

    def complete(
        self,
        *,
        instructions: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
        allow_tools: bool = True,
    ) -> LLMResponse:
        calls = self._record["llm_calls"]
        k = self._next
        if k >= len(calls):
            raise ReplayDiverged(k + 1, "more model calls than recorded", f"{len(calls)} recorded")
        recorded = calls[k]
        if request_hash(instructions, messages, tools, max_output_tokens, allow_tools) != (
            recorded["request_hash"]
        ):
            reason, detail = self._diagnose(
                recorded, instructions, messages, tools, max_output_tokens, allow_tools
            )
            raise ReplayDiverged(k + 1, reason, detail)
        self._next += 1
        return LLMResponse(
            text=recorded["text"],
            tool_calls=tuple(
                ToolCall(c["id"], c["name"], c["arguments"], c.get("error"))
                for c in recorded["tool_calls"]
            ),
            usage=Usage(recorded["usage"]["input"], recorded["usage"]["output"]),
            stop_reason=recorded["stop_reason"],
        )

    def _diagnose(
        self,
        recorded: dict[str, Any],
        instructions: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
        allow_tools: bool,
    ) -> tuple[str, str]:
        """The first explanation for a request that differs from the recorded one."""
        results = {c["id"]: c for c in self._record["tool_calls"]}
        for m in messages:
            if not isinstance(m, ToolResultMessage):
                continue
            old = results.get(m.call_id)
            if old is None:
                return "a tool call ran that did not run before", m.call_id
            if old["result"] != m.content:
                return (
                    f"tool result changed for {old['name']} {m.call_id}",
                    first_difference(old["result"], m.content),
                )
        # The question is the last user message; in a conversation turn the context block and
        # earlier turns come before it ("...Current question: <question>").
        asked = [m.text for m in messages if isinstance(m, UserMessage)]
        if asked and not asked[-1].endswith(self._record["question"]):
            return "the question message changed", ""
        prompt = self._record.get("prompt_hash")
        if prompt and fingerprint(instructions) != prompt:
            return "the instructions (prompt) changed", ""
        if "tools_hash" in recorded and tools_hash(tools) != recorded["tools_hash"]:
            return "the tool definitions changed", ""
        if allow_tools != recorded["allow_tools"]:
            now = "off" if not allow_tools else "on"
            return "the final answer is forced at a different call", f"tools now {now}"
        if max_output_tokens != recorded["max_output_tokens"]:
            before = recorded["max_output_tokens"]
            return "max_output_tokens changed", f"{before} -> {max_output_tokens}"
        if "tools_hash" not in recorded:
            return "the request changed (tool definitions were not recorded)", ""
        if (self._record.get("conversation") or {}).get("turn", 1) > 1:
            return "the conversation context (earlier turns) or the request structure changed", ""
        return "the request structure changed", ""


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 80 else text[:79] + "…"


def compare_answers(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """What differs between the recorded answer and the replayed one (empty: the same)."""
    changes = []
    if old["structured"] != new["structured"]:
        changes.append(f"structured: {old['structured']} -> {new['structured']}")
    for key in ("references", "data_references", "rejected_citations", "limitations"):
        before, after = old[key], new[key]
        if before == after:
            continue
        diff = [f"+ {_short(x)}" for x in after if x not in before]
        diff += [f"- {_short(x)}" for x in before if x not in after]
        changes.append(f"{key}: " + ("; ".join(diff) if diff else "same items, new order"))
    before, after = old["claims"], new["claims"]
    if before != after:
        if len(before) != len(after):
            changes.append(f"claims: {len(before)} -> {len(after)}")
        for i, (a, b) in enumerate(zip(before, after, strict=False), start=1):
            if a == b:
                continue
            parts = [
                f"{field} {_short(a[field])} -> {_short(b[field])}"
                for field in ("kind", "sources", "text")
                if a[field] != b[field]
            ]
            changes.append(f"claim {i}: " + "; ".join(parts))
    return changes


def replay_section(original: dict[str, Any], replayed: dict[str, Any]) -> dict[str, Any]:
    """The ``replay`` part of a replayed record: outcome same / changed / diverged."""
    section: dict[str, Any] = {
        "of": original["run_id"],
        "mode": original.get("mode", "agentic"),
        "original_failures": original["failures"],
    }
    if replayed["status"] == "diverged":
        return {**section, "outcome": "diverged", "reason": replayed["error"], "changes": []}
    if replayed["answer"] is None:
        return {**section, "outcome": replayed["status"], "changes": []}
    changes = compare_answers(original["answer"], replayed["answer"])
    return {**section, "outcome": "changed" if changes else "same", "changes": changes}
