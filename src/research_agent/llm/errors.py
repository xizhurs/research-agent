"""LLM error hierarchy. The agent loop catches LLMError; messages never contain secrets."""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM failures."""


class LLMAuthError(LLMError):
    """The provider rejected the credentials (401/403)."""


class LLMRateLimitError(LLMError):
    """The provider kept answering 429 after the SDK's retries."""


class LLMConnectionError(LLMError):
    """The request could not complete (timeout or connection failure) after retries."""


class LLMApiError(LLMError):
    """Any other provider-side failure (4xx/5xx, or a response that reports it failed)."""


class LLMResponseError(LLMError):
    """The provider answered, but the response is unusable (bad tool arguments, no usage...)."""
