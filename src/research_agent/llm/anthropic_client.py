"""Anthropic Messages API adapter for structured JSON replies (the evaluation judge).

Separate from the agent's ``LLMClient``: the judge makes one request with no tools and asks for
a reply that matches a JSON schema (``output_config.format``), so the text is schema-valid JSON
unless the model refused or ran out of tokens. The SDK (``anthropic``, the ``judge`` extra) is
imported only when a client is built from credentials, so tests and CI run without it.

Model notes for Claude Haiku 5.5 (from Anthropic's API reference): adaptive thinking is on by
default and counts toward ``max_tokens``; depth is set with ``output_config.effort``; sampling
parameters (``temperature`` ...) and assistant prefill are rejected, so none are sent; a reply
can start with ``thinking`` blocks, so text is read by block type; a safety refusal returns
``stop_reason == "refusal"`` with no server-side fallback for this model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from research_agent.config import JudgeConfig
from research_agent.credentials import Credentials
from research_agent.llm.errors import (
    LLMApiError,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
)
from research_agent.llm.types import Usage


@dataclass(frozen=True)
class JsonReply:
    text: str  # the concatenated text blocks (JSON when stop_reason is "end_turn")
    usage: Usage
    stop_reason: str  # "end_turn", "max_tokens", "refusal", ...


class JsonClient(Protocol):
    def complete_json(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_output_tokens: int
    ) -> JsonReply: ...


def _translate(exc: Exception) -> LLMError:
    """SDK exceptions to the project's LLM errors. Server text is not copied into auth errors
    (it can quote part of the key)."""
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return LLMAuthError("Anthropic rejected the API key (CLAUDE_API_KEY)")
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimitError("Anthropic rate limit reached after retries")
    if isinstance(exc, anthropic.APITimeoutError | anthropic.APIConnectionError):
        return LLMConnectionError(f"Anthropic request failed: {type(exc).__name__}")
    if isinstance(exc, anthropic.APIStatusError):
        return LLMApiError(f"Anthropic API error {exc.status_code}: {exc.message}")
    return LLMApiError(f"Anthropic request failed: {type(exc).__name__}")


class AnthropicJsonClient:
    """``complete_json`` over ``messages.create`` with ``output_config.format`` (json_schema)."""

    def __init__(self, sdk: Any, model: str, effort: str) -> None:
        self._sdk = sdk
        self.model = model
        self._effort = effort

    @classmethod
    def from_credentials(
        cls, config: JudgeConfig, model: str, credentials: Credentials
    ) -> AnthropicJsonClient:
        key = credentials.require_claude_key()
        try:
            import anthropic
        except ModuleNotFoundError as exc:
            raise LLMError("the judge needs the extra: uv sync --extra judge") from exc
        sdk = anthropic.Anthropic(
            api_key=key.reveal(), timeout=config.timeout_seconds, max_retries=config.max_retries
        )
        return cls(sdk, model, config.effort)

    def complete_json(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_output_tokens: int
    ) -> JsonReply:
        try:
            response = self._sdk.messages.create(
                model=self.model,
                max_tokens=max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": self._effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except Exception as exc:
            if type(exc).__module__.split(".")[0] != "anthropic":
                raise
            raise _translate(exc) from exc
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
        return JsonReply(text=text, usage=usage, stop_reason=str(response.stop_reason))
