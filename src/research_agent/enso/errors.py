"""ENSO error hierarchy. The agent catches EnsoError."""

from __future__ import annotations


class EnsoError(Exception):
    """Base class for all ENSO-analysis failures."""


class InvalidEnsoQueryError(EnsoError):
    """The caller supplied an unusable argument. Rejected before any request."""


class EnsoNetworkError(EnsoError):
    """The request could not complete (timeout, connection failure, 5xx) after retries."""


class EnsoApiError(EnsoError):
    """Non-retryable HTTP failure, or a file that is not the expected RONI table."""


class InsufficientEnsoDataError(EnsoError):
    """The requested period or year lies outside the record, so nothing can be computed."""
