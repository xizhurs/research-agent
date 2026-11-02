"""Deterministic extreme-rainfall analysis (no LLM)."""

from research_agent.rainfall.client import RainfallClient
from research_agent.rainfall.errors import (
    AmbiguousLocationError,
    InsufficientDataError,
    InvalidRainfallQueryError,
    LocationNotFoundError,
    RainfallApiError,
    RainfallError,
    RainfallNetworkError,
    RainfallRateLimitError,
)
from research_agent.rainfall.models import (
    AnnualMax,
    Baseline,
    EventContext,
    Location,
    Provenance,
    RainfallAnalysis,
)

__all__ = [
    "AmbiguousLocationError",
    "AnnualMax",
    "Baseline",
    "EventContext",
    "InsufficientDataError",
    "InvalidRainfallQueryError",
    "Location",
    "LocationNotFoundError",
    "Provenance",
    "RainfallAnalysis",
    "RainfallApiError",
    "RainfallClient",
    "RainfallError",
    "RainfallNetworkError",
    "RainfallRateLimitError",
]
