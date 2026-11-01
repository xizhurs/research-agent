"""OpenAI adapter (Responses API) implementing ``LLMClient``.

Everything OpenAI-specific lives here: the request/response item formats, reasoning-item
replay and error mapping. Retries on 429/5xx/connection errors are done by the SDK itself
(``max_retries`` from config), so this module does not add a second retry loop.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

import openai

from research_agent.config import LLMConfig
from research_agent.credentials import Credentials
from research_agent.llm.errors import (
    LLMApiError,
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
)
from research_agent.llm.types import (
    AssistantMessage,
    LLMResponse,
    Message,
    StopReason,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    Usage,
    UserMessage,
)

MAX_ERROR_CHARS = 300

log = logging.getLogger(__name__)


def tools_to_openai(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """Responses-API function tools are flat (no nested ``function`` object).

    ``strict`` is set explicitly to False so schemas are used as given; our own code
    validates tool arguments.
    """
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": dict(t.parameters),
            "strict": False,
        }
        for t in tools
    ]


def messages_to_input(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert provider-agnostic messages to Responses ``input`` items."""
    items: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserMessage):
            items.append({"role": "user", "content": m.text})
        elif isinstance(m, AssistantMessage):
            if m.provider_state is not None:
                # Replay the model's own output items verbatim (includes any reasoning items).
                items.extend(m.provider_state)
                continue
            if m.text:
                items.append({"role": "assistant", "content": m.text})
            for call in m.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    }
                )
        elif isinstance(m, ToolResultMessage):
            items.append(
                {"type": "function_call_output", "call_id": m.call_id, "output": m.content}
            )
        else:  # pragma: no cover - guarded by the Message union
            raise TypeError(f"unsupported message type: {type(m).__name__}")
    return items


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Return ``(arguments, error)``. Bad arguments are the model's mistake, not a crash."""
    try:
        args = json.loads(raw)
    except (TypeError, ValueError):
        return {}, "the tool arguments were not valid JSON"
    if not isinstance(args, dict):
        return {}, "the tool arguments were not a JSON object"
    return args, None


def parse_response(response: Any) -> LLMResponse:
    """Convert an OpenAI Responses object into an ``LLMResponse``."""
    status = getattr(response, "status", None)
    if status in ("failed", "cancelled"):
        err = getattr(response, "error", None)
        detail = getattr(err, "message", None) or status
        raise LLMApiError(f"OpenAI response {status}: {str(detail)[:MAX_ERROR_CHARS]}")

    usage = getattr(response, "usage", None)
    if usage is None:
        raise LLMResponseError("OpenAI response has no usage data; cannot enforce token budget")

    texts: list[str] = []
    calls: list[ToolCall] = []
    refused = False
    for item in response.output:
        if item.type == "function_call":
            arguments, error = _parse_arguments(item.arguments)
            calls.append(
                ToolCall(id=item.call_id, name=item.name, arguments=arguments, error=error)
            )
        elif item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    texts.append(part.text)
                elif part.type == "refusal":
                    texts.append(part.refusal)
                    refused = True

    stop_reason: StopReason
    if status == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        stop_reason = "length" if reason == "max_output_tokens" else "other"
    elif refused:
        stop_reason = "other"
    elif calls:
        stop_reason = "tool_calls"
    else:
        stop_reason = "stop"

    return LLMResponse(
        text="".join(texts) or None,
        tool_calls=tuple(calls),
        usage=Usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens),
        stop_reason=stop_reason,
        provider_state=tuple(item.model_dump(exclude_none=True) for item in response.output),
    )


class OpenAIClient:
    def __init__(self, config: LLMConfig, client: Any) -> None:
        """``client`` is an ``openai.OpenAI`` (or any object with ``responses.create``)."""
        self._config = config
        self._client = client

    @classmethod
    def from_credentials(cls, config: LLMConfig, credentials: Credentials) -> OpenAIClient:
        key = credentials.require_openai_key()
        sdk = openai.OpenAI(
            api_key=key.reveal(),
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        return cls(config, sdk)

    def complete(
        self,
        *,
        instructions: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
        allow_tools: bool = True,
    ) -> LLMResponse:
        # No temperature: reasoning-capable models reject it unless reasoning is off.
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "instructions": instructions,
            "input": messages_to_input(messages),
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": self._config.reasoning_effort},
        }
        if tools:
            kwargs["tools"] = tools_to_openai(tools)
            kwargs["tool_choice"] = "auto" if allow_tools else "none"
        log.debug(
            "calling %s: %d input item(s), %d tool(s), max_output_tokens=%d, tools=%s",
            self._config.model,
            len(kwargs["input"]),
            len(tools),
            max_output_tokens,
            "on" if allow_tools else "off",
        )
        try:
            response = self._client.responses.create(**kwargs)
        except (openai.AuthenticationError, openai.PermissionDeniedError):
            # Server text for auth failures can contain a (masked) fragment of the key.
            raise LLMAuthError(
                "OpenAI rejected the API key (401/403). Check OPENAI_API_KEY."
            ) from None
        except openai.RateLimitError as exc:
            raise LLMRateLimitError(
                f"OpenAI rate limit or quota exceeded after retries: {_short(exc)}"
            ) from exc
        except openai.APIConnectionError as exc:  # includes APITimeoutError
            raise LLMConnectionError(
                f"could not reach OpenAI after retries: {type(exc).__name__}"
            ) from exc
        except openai.APIStatusError as exc:
            raise LLMApiError(f"OpenAI returned {exc.status_code}: {_short(exc)}") from exc
        except openai.APIError as exc:
            raise LLMApiError(f"OpenAI API error: {_short(exc)}") from exc
        return parse_response(response)


def _short(exc: Exception) -> str:
    return str(getattr(exc, "message", exc))[:MAX_ERROR_CHARS]
