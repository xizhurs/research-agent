"""Deterministic extreme-rainfall analysis: geocode a city, fetch ERA5 daily precipitation from
Open-Meteo, compute Rx1day statistics. No LLM involved.

``RainfallClient.analyze`` is the single entry point the agent tool calls. It validates input
before any request, caches raw responses on disk, retries only transient failures, and raises
``RainfallError`` subclasses; it never returns partial or guessed data. No credentials are used.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlencode

import httpx

from research_agent.config import RainfallConfig
from research_agent.rainfall.analysis import compute_analysis
from research_agent.rainfall.dryness import compute_dry_analysis
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
from research_agent.rainfall.models import Location, Provenance, RainfallRecord
from research_agent.retrieval.cache import ResponseCache, cache_key

log = logging.getLogger(__name__)

MAX_CITY_CHARS = 200
METRICS = ("wet", "dry")
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_COUNTRY_CODE = re.compile(r"^[A-Za-z]{2}$")
VARIABLE = "precipitation_sum: daily total precipitation in mm, local calendar day"
LICENSE = "CC BY 4.0 (attribution to Open-Meteo.com required)"


@dataclass(frozen=True)
class Archive:
    """A parsed archive response: a daily series plus the grid cell that was actually used."""

    dates: tuple[date, ...]
    values: tuple[float | None, ...]
    latitude: float
    longitude: float
    elevation_m: float | None
    timezone: str | None


def build_archive_params(
    latitude: float, longitude: float, start: date, end: date, model: str
) -> dict[str, str]:
    """Archive query parameters. The model is always pinned (the API default mixes products)."""
    return {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "precipitation_sum",
        "timezone": "auto",
        "models": model,
    }


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def parse_archive(payload: Any) -> Archive:
    """Validate and parse an archive response. Raises RainfallApiError on any unexpected shape."""
    if not isinstance(payload, dict):
        raise RainfallApiError("unexpected archive response: top-level JSON is not an object")
    daily = payload.get("daily")
    units = payload.get("daily_units")
    if not isinstance(daily, dict) or not isinstance(units, dict):
        raise RainfallApiError("unexpected archive response: missing 'daily' or 'daily_units'")
    if units.get("precipitation_sum") != "mm":
        raise RainfallApiError(
            f"unexpected archive response: precipitation unit is {units.get('precipitation_sum')!r}"
        )
    times, raw = daily.get("time"), daily.get("precipitation_sum")
    if not isinstance(times, list) or not isinstance(raw, list) or len(times) != len(raw):
        raise RainfallApiError("unexpected archive response: 'time' and values do not line up")
    if not times:
        raise RainfallApiError("unexpected archive response: the daily series is empty")
    lat, lon = _number(payload.get("latitude")), _number(payload.get("longitude"))
    if lat is None or lon is None:
        raise RainfallApiError("unexpected archive response: grid latitude/longitude missing")
    try:
        days = tuple(date.fromisoformat(t) for t in times)
    except (TypeError, ValueError) as exc:
        raise RainfallApiError("unexpected archive response: invalid date in 'time'") from exc
    values: list[float | None] = []
    for v in raw:
        if v is None:
            values.append(None)
            continue
        number = _number(v)
        if number is None or number < 0:
            raise RainfallApiError("unexpected archive response: invalid precipitation value")
        values.append(number)
    timezone = payload.get("timezone")
    return Archive(
        dates=days,
        values=tuple(values),
        latitude=lat,
        longitude=lon,
        elevation_m=_number(payload.get("elevation")),
        timezone=timezone if isinstance(timezone, str) else None,
    )


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


class RainfallClient:
    def __init__(
        self,
        config: RainfallConfig,
        *,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._config = config
        self._http = http_client or httpx.Client(timeout=config.timeout_seconds)
        self._owns_http = http_client is None
        self._sleep = sleep
        self._today = today
        self._cache = ResponseCache(config.cache_dir, config.cache_ttl_days, now=now)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> RainfallClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _archive_params(self, location: Location, start_year: int, end_year: int) -> dict[str, str]:
        return build_archive_params(
            location.latitude,
            location.longitude,
            date(start_year, 1, 1),
            min(date(end_year, 12, 31), self._today()),
            self._config.model,
        )

    def fetch_archive(self, location: Location, start_year: int, end_year: int) -> Archive:
        """The daily series ``analyze`` used for ``location`` and these years.

        Builds the same request as ``analyze``, so it is a cache hit right after an analysis (it
        does not geocode again, which could resolve differently). Used to plot an analysis that
        only the statistics were kept for.
        """
        archive, _ = self._cached(
            self._config.archive_url,
            self._archive_params(location, start_year, end_year),
            parse_archive,
        )
        return archive

    def analyze(
        self,
        city: str,
        *,
        country_code: str | None = None,
        region: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        event_date: date | None = None,
        metric: str = "wet",
        event_year: int | None = None,
    ) -> tuple[RainfallRecord, Archive]:
        """Analyse ``city``: returns the statistics and the daily series they were computed from.

        ``metric`` is ``"wet"`` (Rx1day) or ``"dry"`` (CDD, annual totals). ``event_year`` (dry
        only) asks how one year compares; ``event_date`` asks about one day. The archive covers
        the requested period, so a plot of the daily values needs no second request.
        """
        cfg = self._config
        today = self._today()
        last_complete_year = today.year - 1
        city, country_code, region, start_year, end_year = self._validate(
            city, country_code, region, start_year, end_year, event_date, today, metric, event_year
        )

        geo_params = build_geocode_params(city, country_code, cfg.geocode_candidates)
        candidates, geo_cached = self._cached(cfg.geocoding_url, geo_params, parse_candidates)
        location, others = resolve_location(
            candidates, city=city, region=region, dominance_ratio=cfg.dominance_ratio
        )

        params = self._archive_params(location, start_year, end_year)
        archive, archive_cached = self._cached(cfg.archive_url, params, parse_archive)
        provenance = Provenance(
            dataset=(
                f"ERA5 reanalysis (ECMWF) via the Open-Meteo Historical Weather API "
                f"(model={cfg.model})"
            ),
            variable=VARIABLE,
            license=LICENSE,
            request_url=f"{cfg.archive_url}?{urlencode(params)}",
            grid_latitude=archive.latitude,
            grid_longitude=archive.longitude,
            elevation_m=archive.elevation_m,
            timezone=archive.timezone,
            data_start=archive.dates[0].isoformat(),
            data_end=archive.dates[-1].isoformat(),
            from_cache=geo_cached and archive_cached,
        )
        if metric == "dry":
            record: RainfallRecord = compute_dry_analysis(
                location,
                archive.dates,
                archive.values,
                start_year=start_year,
                end_year=end_year,
                last_complete_year=last_complete_year,
                config=cfg,
                provenance=provenance,
                event_date=event_date,
                event_year=event_year,
                other_matches=others,
            )
        else:
            record = compute_analysis(
                location,
                archive.dates,
                archive.values,
                start_year=start_year,
                end_year=end_year,
                last_complete_year=last_complete_year,
                config=cfg,
                provenance=provenance,
                event_date=event_date,
                other_matches=others,
            )
        return record, archive

    def _validate(
        self,
        city: object,
        country_code: object,
        region: object,
        start_year: object,
        end_year: object,
        event_date: object,
        today: date,
        metric: object,
        event_year: object,
    ) -> tuple[str, str | None, str | None, int, int]:
        cfg = self._config
        if metric not in METRICS:
            raise InvalidRainfallQueryError("metric must be 'wet' or 'dry'")
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

        first = cfg.default_start_year if start_year is None else start_year
        last = today.year - 1 if end_year is None else end_year
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
            if metric != "dry":
                raise InvalidRainfallQueryError("event_year can only be used with metric 'dry'")
            if (
                isinstance(event_year, bool)
                or not isinstance(event_year, int)
                or not first <= event_year <= last
            ):
                raise InvalidRainfallQueryError(
                    f"event_year must be an integer between {first} and {last}"
                )
        if event_date is not None:
            latest = min(date(last, 12, 31), today)
            if not isinstance(event_date, date) or not date(first, 1, 1) <= event_date <= latest:
                raise InvalidRainfallQueryError(
                    f"event_date must lie between {first}-01-01 and {latest.isoformat()} "
                    "(widen start_year/end_year to include it)"
                )
        return (
            city,
            country_code.strip().upper() if isinstance(country_code, str) else None,
            region.strip() if isinstance(region, str) else None,
            first,
            last,
        )

    def _cached[T](
        self, url: str, params: dict[str, str], parse: Callable[[Any], T]
    ) -> tuple[T, bool]:
        """Fetch ``url`` through the cache. Only validated responses are cached."""
        key = cache_key(url, params)
        cached = self._cache.get(key)
        log.debug("rainfall %s: cache %s", url, "hit" if cached is not None else "miss")
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
