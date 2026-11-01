"""Provider-agnostic LLM types and the ``LLMClient`` protocol.

The agent loop talks only to these types. Provider details (message formats, tool-call
encodings, reasoning items) live inside each adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

StopReason = Literal["stop", "tool_calls", "length", "other"]


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class AssistantMessage:
    """A model turn to replay in the conversation.

    ``provider_state`` is opaque data an adapter may attach (e.g. items it must send back
    verbatim on the next request). Callers must pass it through untouched.
    """

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    provider_state: Any = None


@dataclass(frozen=True)
class ToolResultMessage:
    call_id: str
    content: str


Message = UserMessage | AssistantMessage | ToolResultMessage


@dataclass(frozen=True)
class LLMResponse:
    text: str | None
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    stop_reason: StopReason
    provider_state: Any = None

    def to_message(self) -> AssistantMessage:
        """The message to append to the conversation before sending tool results."""
        return AssistantMessage(
            text=self.text, tool_calls=self.tool_calls, provider_state=self.provider_state
        )


class LLMClient(Protocol):
    def complete(
        self,
        *,
        instructions: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
        allow_tools: bool = True,
    ) -> LLMResponse:
        """Run one model turn.

        ``allow_tools=False`` forbids tool calls this turn (used to force a final answer when
        a budget is exhausted). Raises ``LLMError`` subclasses; never returns partial data.
        """
        ...
