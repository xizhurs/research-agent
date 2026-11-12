"""Deterministic extreme-rainfall analysis: geocode a city, fetch ERA5 daily precipitation from
Open-Meteo, compute Rx1day statistics. No LLM involved.

``RainfallClient.analyze`` is the single entry point the agent tool calls. It validates input
before any request, caches raw responses on disk, retries only transient failures, and raises
``RainfallError`` subclasses; it never returns partial or guessed data. No credentials are used.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from typing import Any
from urllib.parse import urlencode

import httpx

from research_agent.config import RainfallConfig
from research_agent.rainfall.analysis import compute_analysis
from research_agent.rainfall.dryness import compute_dry_analysis
from research_agent.rainfall.errors import InvalidRainfallQueryError, RainfallApiError
from research_agent.rainfall.models import Location, Provenance, RainfallRecord
from research_agent.rainfall.openmeteo import (
    OpenMeteoFetcher,
    number,
    point_caveat,
    point_location,
    validate_city,
    validate_period,
    validate_point,
)
from research_agent.rainfall.totals import compute_monthly_analysis, compute_yearly_analysis

log = logging.getLogger(__name__)

METRICS = ("wet", "dry")
VARIABLE = "precipitation_sum: daily total precipitation in mm, local calendar day"
LICENSE = "CC BY 4.0 (attribution to Open-Meteo.com required)"
DATASET = "ERA5 reanalysis (ECMWF) via the Open-Meteo Historical Weather API (model={model})"

__all__ = ["Archive", "RainfallClient", "build_archive_params", "parse_archive", "validate_point"]


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
    lat, lon = number(payload.get("latitude")), number(payload.get("longitude"))
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
        mm = number(v)
        if mm is None or mm < 0:
            raise RainfallApiError("unexpected archive response: invalid precipitation value")
        values.append(mm)
    timezone = payload.get("timezone")
    return Archive(
        dates=days,
        values=tuple(values),
        latitude=lat,
        longitude=lon,
        elevation_m=number(payload.get("elevation")),
        timezone=timezone if isinstance(timezone, str) else None,
    )


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
        self._today = today
        self._fetcher = OpenMeteoFetcher(config, http_client=http_client, sleep=sleep, now=now)

    def close(self) -> None:
        self._fetcher.close()

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
        archive, _ = self._fetcher.cached(
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
        scale: str = "daily",
        event_month: int | None = None,
    ) -> tuple[RainfallRecord, Archive]:
        """Analyse ``city``: returns the statistics and the daily series they were computed from.

        ``scale`` ``"daily"``: ``metric`` ``"wet"`` (Rx1day) or ``"dry"`` (CDD, annual totals);
        ``event_date`` asks about one day, ``event_year`` (dry only) about one year. ``"yearly"``:
        annual totals; ``event_year`` asks about one year. ``"monthly"``: monthly totals;
        ``event_year`` (with ``event_month``) asks about one month, or each month of a year. For
        yearly and monthly the series starts at the baseline period at the latest, so the
        baseline is always available. A plot of the daily values needs no second request.
        """
        today = self._today()
        city, country_code, region = validate_city(city, country_code, region)
        start_year, end_year = self._validate(
            start_year, end_year, event_date, today, metric, event_year, scale, event_month
        )

        location, others, geo_cached = self._fetcher.resolve_city(city, country_code, region)
        return self._analyze_location(
            location,
            others,
            geo_cached,
            today=today,
            start_year=start_year,
            end_year=end_year,
            event_date=event_date,
            metric=metric,
            event_year=event_year,
            scale=scale,
            event_month=event_month,
        )

    def analyze_point(
        self,
        latitude: float,
        longitude: float,
        *,
        start_year: int | None = None,
        end_year: int | None = None,
        event_date: date | None = None,
        metric: str = "wet",
        event_year: int | None = None,
        scale: str = "daily",
        event_month: int | None = None,
    ) -> tuple[RainfallRecord, Archive]:
        """Like ``analyze``, for a point given by coordinates (for example a map click).

        No geocoding: the archive request goes straight to the point, and the record adds a
        caveat naming the ERA5 grid cell actually used.
        """
        today = self._today()
        latitude, longitude = validate_point(latitude, longitude)
        start_year, end_year = self._validate(
            start_year, end_year, event_date, today, metric, event_year, scale, event_month
        )
        location = point_location(latitude, longitude)
        record, archive = self._analyze_location(
            location,
            (),
            True,
            today=today,
            start_year=start_year,
            end_year=end_year,
            event_date=event_date,
            metric=metric,
            event_year=event_year,
            scale=scale,
            event_month=event_month,
        )
        caveat = point_caveat(location, archive.latitude, archive.longitude)
        return replace(record, caveats=(*record.caveats, caveat)), archive

    def _analyze_location(
        self,
        location: Location,
        others: tuple[Location, ...],
        geo_cached: bool,
        *,
        today: date,
        start_year: int,
        end_year: int,
        event_date: date | None,
        metric: str,
        event_year: int | None,
        scale: str,
        event_month: int | None,
    ) -> tuple[RainfallRecord, Archive]:
        """Fetch the archive for an already resolved ``location`` and compute the statistics."""
        cfg = self._config
        last_complete_year = today.year - 1
        series_start = start_year if scale == "daily" else min(start_year, cfg.baseline_start_year)
        params = self._archive_params(location, series_start, end_year)
        archive, archive_cached = self._fetcher.cached(cfg.archive_url, params, parse_archive)
        provenance = Provenance(
            dataset=DATASET.format(model=cfg.model),
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
        if scale == "yearly":
            record: RainfallRecord = compute_yearly_analysis(
                location,
                archive.dates,
                archive.values,
                start_year=start_year,
                end_year=end_year,
                last_complete_year=last_complete_year,
                config=cfg,
                provenance=provenance,
                baseline_from_year=series_start,
                event_year=event_year,
                other_matches=others,
            )
        elif scale == "monthly":
            last_month = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
            record = compute_monthly_analysis(
                location,
                archive.dates,
                archive.values,
                start_year=start_year,
                end_year=end_year,
                last_complete_month=last_month,
                config=cfg,
                provenance=provenance,
                baseline_from_year=series_start,
                event_year=event_year,
                event_month=event_month,
                other_matches=others,
            )
        elif metric == "dry":
            record = compute_dry_analysis(
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
        start_year: object,
        end_year: object,
        event_date: object,
        today: date,
        metric: object,
        event_year: object,
        scale: object = "daily",
        event_month: object = None,
    ) -> tuple[int, int]:
        if metric not in METRICS:
            raise InvalidRainfallQueryError("metric must be 'wet' or 'dry'")
        return validate_period(
            self._config,
            today,
            start_year=start_year,
            end_year=end_year,
            event_date=event_date,
            event_year=event_year,
            scale=scale,
            event_month=event_month,
            daily_event_year=metric == "dry",
            daily_event_year_hint=" (or metric 'dry')",
        )
