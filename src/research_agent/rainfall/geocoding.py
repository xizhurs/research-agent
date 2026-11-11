"""City geocoding with Open-Meteo: request building, parsing and disambiguation. Pure.

Disambiguation never guesses silently. Candidates within ``same_place_km`` of a larger one (for
example a city's districts, which the geocoder lists separately) are merged into it first. Then
one candidate, or one that clearly dominates by population (``dominance_ratio``), is accepted and
the alternatives are reported; otherwise ``AmbiguousLocationError`` lists the candidates and no
analysis runs.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

from research_agent.rainfall.errors import (
    AmbiguousLocationError,
    LocationNotFoundError,
    RainfallApiError,
)
from research_agent.rainfall.models import Location

log = logging.getLogger(__name__)

MAX_LISTED = 10
EARTH_RADIUS_KM = 6371.0


def build_geocode_params(city: str, country_code: str | None, count: int) -> dict[str, str]:
    """Query parameters for the geocoding search. ``region`` is filtered locally (exact match)."""
    params = {"name": city, "count": str(count), "language": "en", "format": "json"}
    if country_code:
        params["countryCode"] = country_code
    return params


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def parse_candidates(payload: Any) -> tuple[Location, ...]:
    """Parse a geocoding response. A response without ``results`` means no match (not an error).

    Raises RainfallApiError if the envelope is not an object. Unparseable records are counted
    and logged, not dropped silently.
    """
    if not isinstance(payload, dict):
        raise RainfallApiError("unexpected geocoding response: top-level JSON is not an object")
    raw = payload.get("results", [])
    if not isinstance(raw, list):
        raise RainfallApiError("unexpected geocoding response: 'results' is not a list")
    found: list[Location] = []
    skipped = 0
    for record in raw:
        lat = _number(record.get("latitude")) if isinstance(record, dict) else None
        lon = _number(record.get("longitude")) if isinstance(record, dict) else None
        ident = record.get("id") if isinstance(record, dict) else None
        name = _text(record.get("name")) if isinstance(record, dict) else None
        if (
            lat is None
            or lon is None
            or name is None
            or isinstance(ident, bool)
            or not isinstance(ident, int)
        ):
            skipped += 1
            log.warning("skipped unparseable geocoding record: %.120r", record)
            continue
        population = record.get("population")
        found.append(
            Location(
                id=ident,
                name=name,
                admin1=_text(record.get("admin1")),
                country=_text(record.get("country")),
                country_code=_text(record.get("country_code")),
                latitude=lat,
                longitude=lon,
                population=population
                if isinstance(population, int) and not isinstance(population, bool)
                else None,
                timezone=_text(record.get("timezone")),
            )
        )
    return tuple(found)


def _describe(loc: Location) -> str:
    pop = f"population {loc.population:,}" if loc.population else "population unknown"
    return f"{loc.label} ({pop})"


def distance_km(a: Location, b: Location) -> float:
    """Great-circle (haversine) distance between two places."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.latitude, a.longitude, b.latitude, b.longitude))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(
        (lon2 - lon1) / 2
    ) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def resolve_location(
    candidates: Sequence[Location],
    *,
    city: str,
    region: str | None,
    dominance_ratio: float,
    same_place_km: float,
) -> tuple[Location, tuple[Location, ...]]:
    """Pick the place to analyse. Returns ``(chosen, other_matches)``.

    ``other_matches`` holds every other candidate, including the ones merged as nearby.
    Raises LocationNotFoundError (no match) or AmbiguousLocationError (no clear winner).
    """
    pool = list(candidates)
    if region is not None:
        pool = [c for c in pool if (c.admin1 or "").casefold() == region.strip().casefold()]
    if not pool:
        where = f" in region '{region}'" if region else ""
        raise LocationNotFoundError(f"no place named '{city}'{where} was found")
    pool.sort(key=lambda c: c.population or 0, reverse=True)  # stable: ties keep API order
    kept: list[Location] = []
    merged: list[Location] = []
    for candidate in pool:  # larger places first, so a district merges into its city
        near = any(distance_km(candidate, k) <= same_place_km for k in kept)
        (merged if near else kept).append(candidate)
    others = tuple(c for c in pool if c is not kept[0])
    if len(kept) == 1:
        return kept[0], others
    top, second = kept[0], kept[1]
    if top.population and top.population >= dominance_ratio * (second.population or 0):
        return top, others
    pool = kept
    listed = "; ".join(f"{i}) {_describe(c)}" for i, c in enumerate(pool[:MAX_LISTED], start=1))
    raise AmbiguousLocationError(
        f"'{city}' matches {len(pool)} places and none clearly dominates: {listed}. Do not guess: "
        "call the tool again with country_code and/or region only if the user's question says "
        "which place is meant; otherwise tell the user the location is ambiguous and list "
        "these options.",
        pool,
    )
