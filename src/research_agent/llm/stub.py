"""Scripted LLM client for tests and offline development. No network, no cost."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from research_agent.llm.types import (
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)


@dataclass(frozen=True)
class RecordedCall:
    instructions: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    max_output_tokens: int
    allow_tools: bool


def text_response(text: str, *, input_tokens: int = 10, output_tokens: int = 5) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=(),
        usage=Usage(input_tokens, output_tokens),
        stop_reason="stop",
    )


def tool_call_response(
    *calls: tuple[str, str, dict[str, Any]], input_tokens: int = 10, output_tokens: int = 5
) -> LLMResponse:
    """Build a response requesting tool calls, each given as ``(call_id, name, arguments)``."""
    return LLMResponse(
        text=None,
        tool_calls=tuple(ToolCall(id=i, name=n, arguments=a) for i, n, a in calls),
        usage=Usage(input_tokens, output_tokens),
        stop_reason="tool_calls",
    )


class StubLLMClient:
    """Returns scripted responses in order; an ``Exception`` entry is raised instead.

    Every call is recorded in ``calls``. Calling more often than scripted is a test bug and
    raises AssertionError.
    """

    def __init__(self, script: Sequence[LLMResponse | Exception]) -> None:
        self._script = list(script)
        self.calls: list[RecordedCall] = []

    def complete(
        self,
        *,
        instructions: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
        allow_tools: bool = True,
    ) -> LLMResponse:
        record = RecordedCall(
            instructions, tuple(messages), tuple(tools), max_output_tokens, allow_tools
        )
        self.calls.append(record)
        if not self._script:
            raise AssertionError("StubLLMClient script exhausted: more calls than scripted")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
