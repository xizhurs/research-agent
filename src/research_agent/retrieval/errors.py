"""Retrieval error hierarchy. Callers (e.g. a later agent tool layer) catch RetrievalError."""

from __future__ import annotations


class RetrievalError(Exception):
    """Base class for all literature-retrieval failures."""


class InvalidQueryError(RetrievalError):
    """The caller supplied an unusable query or argument."""


class NetworkError(RetrievalError):
    """The request could not complete (timeout, connection failure) after retries."""


class RateLimitError(RetrievalError):
    """OpenAlex kept answering 429 (per-second limit or daily budget) after retries."""


class AuthError(RetrievalError):
    """OpenAlex rejected the credentials (401/403, e.g. an API key that is not activated)."""


class ApiError(RetrievalError):
    """Non-retryable API failure, or a response that is not the expected shape."""
