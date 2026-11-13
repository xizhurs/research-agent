"""Model-call recording and the run budget for paid evaluation runs.

``RecordingLLMClient`` wraps the real client. Before each call it asks ``RunBudget`` whether one
more call fits (calls, tokens and, with prices configured, cost, across the whole run); if not it
raises ``EvalBudgetExceeded`` and nothing is sent. Each call's request hash, reply, usage and
latency is kept for the run record, and the tool results the model saw are read from the next
request, so the agent loop needs no changes.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from research_agent.agent.budget import estimate_tokens
from research_agent.llm import (
    AssistantMessage,
    LLMClient,
    LLMResponse,
    Message,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)


class EvalBudgetExceeded(Exception):
    """The evaluation run's budget would be exceeded by the next model call."""


@dataclass
class RunBudget:
    """Caps for one evaluation run. A price of 0 means unknown: cost is then not estimated."""

    max_calls: int
    max_tokens: int
    max_cost_usd: float = 0.0  # 0 = no cost cap
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def priced(self) -> bool:
        return self.price_input_per_mtok > 0 or self.price_output_per_mtok > 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost(self, input_tokens: int | None = None, output_tokens: int | None = None) -> float:
        """Estimated USD for the given tokens (default: what was used so far)."""
        tin = self.input_tokens if input_tokens is None else input_tokens
        tout = self.output_tokens if output_tokens is None else output_tokens
        return (tin * self.price_input_per_mtok + tout * self.price_output_per_mtok) / 1e6

    def refusal(self, projected_input: int, max_output: int) -> str | None:
        """Why one more call (of up to ``projected_input`` + ``max_output`` tokens) must not run."""
        if self.calls + 1 > self.max_calls:
            return f"model-call budget reached ({self.calls} of {self.max_calls} calls)"
        if self.tokens + projected_input + max_output > self.max_tokens:
            return f"token budget reached ({self.tokens:,} of {self.max_tokens:,} tokens used)"
        if self.max_cost_usd > 0:
            after = self.cost(self.input_tokens + projected_input, self.output_tokens + max_output)
            if after > self.max_cost_usd:
                return f"cost budget reached (${self.cost():.4f} of ${self.max_cost_usd:.2f})"
        return None

    def case_refusal(self, worst_case_tokens: int) -> str | None:
        """Why a whole case (up to ``worst_case_tokens``) should not start."""
        if self.calls >= self.max_calls:
            return f"model-call budget reached ({self.calls} of {self.max_calls} calls)"
        if self.tokens + worst_case_tokens > self.max_tokens:
            return (
                f"token budget too low for another case ({self.tokens:,} of "
                f"{self.max_tokens:,} used; a case may use up to {worst_case_tokens:,})"
            )
        return None

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


def messages_json(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Messages as plain data (``provider_state`` left out: it is the provider's, not ours)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserMessage):
            out.append({"role": "user", "text": m.text})
        elif isinstance(m, AssistantMessage):
            calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in m.tool_calls]
            out.append({"role": "assistant", "text": m.text, "tool_calls": calls})
        else:
            out.append({"role": "tool", "call_id": m.call_id, "content": m.content})
    return out


def request_hash(
    instructions: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    max_output_tokens: int,
    allow_tools: bool,
) -> str:
    """A stable fingerprint of one model request (to detect when a replay diverges)."""
    payload = {
        "instructions": instructions,
        "messages": messages_json(messages),
        "tools": [{"name": t.name, "description": t.description, "parameters": t.parameters}
                  for t in tools],
        "max_output_tokens": max_output_tokens,
        "allow_tools": allow_tools,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class _CaseLog:
    calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, str] = field(default_factory=dict)  # call ID -> text the model saw
    instructions: str | None = None


class RecordingLLMClient:
    """An ``LLMClient`` that enforces ``budget`` and records every call of the current case.

    ``inner`` may be set after construction (the CLI builds the recorder before the real client).
    """

    def __init__(
        self,
        inner: LLMClient | None,
        budget: RunBudget,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.inner = inner
        self.budget = budget
        self._clock = clock
        self.log = _CaseLog()

    def start_case(self) -> None:
        self.log = _CaseLog()

    def complete(
        self,
        *,
        instructions: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
        allow_tools: bool = True,
    ) -> LLMResponse:
        for m in messages:
            if isinstance(m, ToolResultMessage):
                self.log.tool_results[m.call_id] = m.content
        if self.log.instructions is None:
            self.log.instructions = instructions
        projected = estimate_tokens(instructions) + estimate_tokens(
            json.dumps(messages_json(messages), ensure_ascii=False)
        )
        reason = self.budget.refusal(projected, max_output_tokens)
        if reason:
            raise EvalBudgetExceeded(reason)
        if self.inner is None:
            raise RuntimeError("RecordingLLMClient has no inner client")
        started = self._clock()
        response = self.inner.complete(
            instructions=instructions,
            messages=messages,
            tools=tools,
            max_output_tokens=max_output_tokens,
            allow_tools=allow_tools,
        )
        latency = self._clock() - started
        usage = response.usage
        self.budget.record(usage.input_tokens, usage.output_tokens)
        self.log.calls.append(
            {
                "request_hash": request_hash(
                    instructions, messages, tools, max_output_tokens, allow_tools
                ),
                "allow_tools": allow_tools,
                "max_output_tokens": max_output_tokens,
                "latency_s": round(latency, 3),
                "usage": {"input": usage.input_tokens, "output": usage.output_tokens},
                "stop_reason": response.stop_reason,
                "text": response.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments, "error": c.error}
                    for c in response.tool_calls
                ],
            }
        )
        return response
