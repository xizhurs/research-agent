"""Deterministic extreme-rainfall analysis (no LLM)."""

from research_agent.rainfall.client import Archive, RainfallClient
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
    AnnualDry,
    AnnualMax,
    Baseline,
    DryAnalysis,
    DryBaseline,
    DryEventContext,
    DrySpell,
    DryYearContext,
    EventContext,
    Location,
    Provenance,
    RainfallAnalysis,
    RainfallRecord,
)

__all__ = [
    "AmbiguousLocationError",
    "AnnualDry",
    "Archive",
    "AnnualMax",
    "Baseline",
    "DryAnalysis",
    "DryBaseline",
    "DryEventContext",
    "DrySpell",
    "DryYearContext",
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
    "RainfallRecord",
]
