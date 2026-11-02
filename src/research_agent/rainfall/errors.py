"""Rainfall error hierarchy. The agent catches RainfallError."""

from __future__ import annotations

from collections.abc import Sequence


class RainfallError(Exception):
    """Base class for all rainfall-analysis failures."""


class InvalidRainfallQueryError(RainfallError):
    """The caller supplied an unusable argument. Rejected before any request."""


class LocationNotFoundError(RainfallError):
    """The geocoder found no place matching the name and qualifiers."""


class AmbiguousLocationError(RainfallError):
    """Several places match and none clearly dominates. No analysis is run.

    ``candidates`` are listed in the message so the model can report them.
    """

    def __init__(self, message: str, candidates: Sequence[object] = ()) -> None:
        super().__init__(message)
        self.candidates = tuple(candidates)


class InsufficientDataError(RainfallError):
    """The archive returned too little usable data to compute anything meaningful."""


class RainfallNetworkError(RainfallError):
    """The request could not complete (timeout, connection failure) after retries."""


class RainfallRateLimitError(RainfallError):
    """Open-Meteo kept answering 429 after retries."""


class RainfallApiError(RainfallError):
    """Non-retryable API failure, or a response that is not the expected shape."""
