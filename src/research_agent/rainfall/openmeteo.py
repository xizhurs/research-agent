"""Open-Meteo plumbing shared by the rainfall and temperature clients: input checks, cached GETs
with bounded retries, and city resolution. No analysis here.

Errors are the ``RainfallError`` family for both clients: they describe Open-Meteo and geocoding
outcomes, not rainfall itself.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable
from datetime import date
from typing import Any

import httpx

from research_agent.config import RainfallConfig
from research_agent.rainfall.errors import (
    InvalidRainfallQueryError,
    RainfallApiError,
    RainfallNetworkError,
    RainfallRateLimitError,
)
from research_agent.rainfall.geocoding import (
    build_geocode_params,
    parse_candidates,
    resolve_location,
)
from research_agent.rainfall.models import Location
from research_agent.retrieval.cache import ResponseCache, cache_key

log = logging.getLogger(__name__)

MAX_CITY_CHARS = 200
SCALES = ("daily", "monthly", "yearly")
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_COUNTRY_CODE = re.compile(r"^[A-Za-z]{2}$")


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def validate_point(latitude: object, longitude: object) -> tuple[float, float]:
    """Check coordinates before any request: finite numbers within the valid ranges."""
    checked = []
    for name, value, limit in (("latitude", latitude, 90), ("longitude", longitude, 180)):
        value_f = number(value)
        if value_f is None or not math.isfinite(value_f):
            raise InvalidRainfallQueryError(f"{name} must be a number")
        if not -limit <= value_f <= limit:
            raise InvalidRainfallQueryError(f"{name} must be between -{limit} and {limit}")
        checked.append(value_f)
    return checked[0], checked[1]


def point_location(latitude: float, longitude: float) -> Location:
    """A ``Location`` for a point given by coordinates; its name is the coordinates."""
    return Location(
        id=0,
        name=f"{latitude:.4f}, {longitude:.4f}",
        admin1=None,
        country=None,
        country_code=None,
        latitude=latitude,
        longitude=longitude,
        population=None,
        timezone=None,
    )


def point_caveat(location: Location, grid_latitude: float, grid_longitude: float) -> str:
    return (
        f"The location is a point given by coordinates ({location.name}); the values are for "
        f"the ERA5 grid cell at {grid_latitude:.4f}, {grid_longitude:.4f} (about "
        "0.25 degrees), not for the exact point."
    )


def validate_city(
    city: object, country_code: object, region: object
) -> tuple[str, str | None, str | None]:
    if not isinstance(city, str) or not city.strip():
        raise InvalidRainfallQueryError("city must be a non-empty string")
    city = city.strip()
    if len(city) > MAX_CITY_CHARS:
        raise InvalidRainfallQueryError(f"city is longer than {MAX_CITY_CHARS} characters")
    if country_code is not None and (
        not isinstance(country_code, str) or not _COUNTRY_CODE.match(country_code.strip())
    ):
        raise InvalidRainfallQueryError("country_code must be a 2-letter ISO code, e.g. 'FR'")
    if region is not None and (not isinstance(region, str) or not region.strip()):
        raise InvalidRainfallQueryError("region must be a non-empty string")
    return (
        city,
        country_code.strip().upper() if isinstance(country_code, str) else None,
        region.strip() if isinstance(region, str) else None,
    )


def validate_period(
    cfg: RainfallConfig,
    today: date,
    *,
    start_year: object,
    end_year: object,
    event_date: object,
    event_year: object,
    scale: object,
    event_month: object,
    daily_event_year: bool,
    daily_event_year_hint: str,
) -> tuple[int, int]:
    """Check the scale, period and event arguments; return ``(first, last)`` year.

    ``daily_event_year`` says whether ``event_year`` is accepted at the daily scale (the rainfall
    dry metric accepts it); ``daily_event_year_hint`` is added to the error when it is not.
    """
    if scale not in SCALES:
        raise InvalidRainfallQueryError("scale must be 'daily', 'monthly' or 'yearly'")

    first = cfg.default_start_year if start_year is None else start_year
    # Monthly questions may be about this year's finished months; the others need whole years.
    default_last = today.year if scale == "monthly" else today.year - 1
    last = default_last if end_year is None else end_year
    for name, year in (("start_year", first), ("end_year", last)):
        if isinstance(year, bool) or not isinstance(year, int):
            raise InvalidRainfallQueryError(f"{name} must be an integer year")
        if not cfg.earliest_year <= year <= today.year:
            raise InvalidRainfallQueryError(
                f"{name} must be between {cfg.earliest_year} and {today.year}"
            )
    if first > last:
        raise InvalidRainfallQueryError("start_year must not be after end_year")
    if event_year is not None:
        if scale == "daily" and not daily_event_year:
            raise InvalidRainfallQueryError(
                f"event_year needs scale 'yearly' or 'monthly'{daily_event_year_hint}; for one "
                "day use event_date"
            )
        if (
            isinstance(event_year, bool)
            or not isinstance(event_year, int)
            or not first <= event_year <= last
        ):
            raise InvalidRainfallQueryError(
                f"event_year must be an integer between {first} and {last}"
            )
    if event_month is not None:
        if scale != "monthly" or event_year is None:
            raise InvalidRainfallQueryError("event_month needs scale 'monthly' and event_year")
        if isinstance(event_month, bool) or event_month not in range(1, 13):
            raise InvalidRainfallQueryError("event_month must be an integer from 1 to 12")
    if event_date is not None:
        if scale != "daily":
            raise InvalidRainfallQueryError(
                "event_date is for scale 'daily'; for a month or year use event_year "
                "(and event_month)"
            )
        latest = min(date(last, 12, 31), today)
        if not isinstance(event_date, date) or not date(first, 1, 1) <= event_date <= latest:
            raise InvalidRainfallQueryError(
                f"event_date must lie between {first}-01-01 and {latest.isoformat()} "
                "(widen start_year/end_year to include it)"
            )
    return first, last


def _retry_after_seconds(response: httpx.Response) -> float | None:
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and isinstance(body.get("reason"), str):
        return body["reason"][:200]
    return str(body)[:200]


class OpenMeteoFetcher:
    """Cached GETs against Open-Meteo with bounded retries, and city resolution."""

    def __init__(
        self,
        config: RainfallConfig,
        *,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._http = http_client or httpx.Client(timeout=config.timeout_seconds)
        self._owns_http = http_client is None
        self._sleep = sleep
        self._cache = ResponseCache(config.cache_dir, config.cache_ttl_days, now=now)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def resolve_city(
        self, city: str, country_code: str | None, region: str | None
    ) -> tuple[Location, tuple[Location, ...], bool]:
        """Geocode a validated city: ``(location, other matches, served from cache)``."""
        cfg = self._config
        geo_params = build_geocode_params(city, country_code, cfg.geocode_candidates)
        candidates, geo_cached = self.cached(cfg.geocoding_url, geo_params, parse_candidates)
        location, others = resolve_location(
            candidates,
            city=city,
            region=region,
            dominance_ratio=cfg.dominance_ratio,
            same_place_km=cfg.same_place_km,
        )
        return location, others, geo_cached

    def cached[T](
        self, url: str, params: dict[str, str], parse: Callable[[Any], T]
    ) -> tuple[T, bool]:
        """Fetch ``url`` through the cache. Only validated responses are cached."""
        key = cache_key(url, params)
        cached = self._cache.get(key)
        log.debug("open-meteo %s: cache %s", url, "hit" if cached is not None else "miss")
        if cached is not None:
            try:
                return parse(cached), True
            except RainfallApiError as exc:
                log.warning("ignoring cached response with unexpected shape: %s", exc)
        payload = self._get_json(url, params)
        parsed = parse(payload)
        self._cache.put(key, payload)
        return parsed, False

    def _get_json(self, url: str, params: dict[str, str]) -> Any:
        cfg = self._config
        last_failure = "no attempt made"
        for attempt in range(cfg.max_retries + 1):
            try:
                response = self._http.get(url, params=params, timeout=cfg.timeout_seconds)
            except httpx.TransportError as exc:  # includes timeouts
                last_failure = f"{type(exc).__name__}: {exc}"
                wait = self._backoff(attempt)
                if attempt == cfg.max_retries:
                    raise RainfallNetworkError(
                        f"Open-Meteo request failed after {attempt + 1} attempts: {last_failure}"
                    ) from exc
            else:
                status = response.status_code
                if status == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise RainfallApiError(
                            "Open-Meteo returned a non-JSON 200 response"
                        ) from exc
                if status not in RETRYABLE_STATUS:
                    raise RainfallApiError(
                        f"Open-Meteo returned {status}: {_error_detail(response)}"
                    )
                last_failure = f"HTTP {status}: {_error_detail(response)}"
                retry_after = _retry_after_seconds(response)
                if retry_after is not None and retry_after > cfg.max_backoff_seconds:
                    raise RainfallRateLimitError(
                        f"Open-Meteo asked to wait {retry_after:.0f}s (> max "
                        f"{cfg.max_backoff_seconds:.0f}s); giving up. {last_failure}"
                    )
                wait = retry_after if retry_after is not None else self._backoff(attempt)
                if attempt == cfg.max_retries:
                    error_cls = RainfallRateLimitError if status == 429 else RainfallApiError
                    raise error_cls(
                        f"Open-Meteo request failed after {attempt + 1} attempts: {last_failure}"
                    )
            log.warning(
                "Open-Meteo request failed (%s); retry %d/%d in %.1fs",
                last_failure,
                attempt + 1,
                cfg.max_retries,
                wait,
            )
            self._sleep(wait)
        raise RainfallApiError(f"Open-Meteo request failed: {last_failure}")  # pragma: no cover

    def _backoff(self, attempt: int) -> float:
        cfg = self._config
        return min(cfg.backoff_base_seconds * 2**attempt, cfg.max_backoff_seconds)
