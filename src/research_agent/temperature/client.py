"""Deterministic temperature analysis: geocode a city (or take a map point), fetch ERA5 daily
maximum, minimum and mean temperature from Open-Meteo, compute statistics. No LLM involved.

Shares the Open-Meteo plumbing, the ``[rainfall]`` settings (URLs, pinned ERA5 model, cache,
baseline years, coverage rules) and the ``RainfallError`` family with the rainfall client. One
archive request carries all three variables, so every scale and metric of a place shares one
cache entry.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from typing import Any
from urllib.parse import urlencode

import httpx

from research_agent.config import RainfallConfig
from research_agent.rainfall.errors import InvalidRainfallQueryError, RainfallApiError
from research_agent.rainfall.models import Location, Provenance
from research_agent.rainfall.openmeteo import (
    OpenMeteoFetcher,
    number,
    point_caveat,
    point_location,
    validate_city,
    validate_period,
    validate_point,
)
from research_agent.temperature.analysis import (
    METRICS,
    compute_extremes,
    compute_monthly,
    compute_yearly,
)
from research_agent.temperature.models import TemperatureRecord

VARIABLES = ("temperature_2m_max", "temperature_2m_min", "temperature_2m_mean")
VARIABLE = (
    "temperature_2m_max, temperature_2m_min, temperature_2m_mean: daily maximum, minimum and "
    "mean air temperature at 2 m in degrees C, local calendar day"
)
LICENSE = "CC BY 4.0 (attribution to Open-Meteo.com required)"
DATASET = "ERA5 reanalysis (ECMWF) via the Open-Meteo Historical Weather API (model={model})"
MAX_ABS_C = 100.0  # anything beyond this is not a temperature


@dataclass(frozen=True)
class TemperatureArchive:
    """A parsed archive response: three daily series plus the grid cell actually used."""

    dates: tuple[date, ...]
    tmax: tuple[float | None, ...]
    tmin: tuple[float | None, ...]
    tmean: tuple[float | None, ...]
    latitude: float
    longitude: float
    elevation_m: float | None
    timezone: str | None

    def series(self, metric: str) -> tuple[float | None, ...]:
        """The series a metric is computed from: maxima (hot), minima (cold), else means."""
        return {"hot": self.tmax, "cold": self.tmin}.get(metric, self.tmean)


def build_temperature_params(
    latitude: float, longitude: float, start: date, end: date, model: str
) -> dict[str, str]:
    """Archive query parameters. The model is always pinned (the API default mixes products)."""
    return {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(VARIABLES),
        "timezone": "auto",
        "models": model,
    }


def parse_temperature_archive(payload: Any) -> TemperatureArchive:
    """Validate and parse an archive response. Raises RainfallApiError on any unexpected shape."""
    if not isinstance(payload, dict):
        raise RainfallApiError("unexpected archive response: top-level JSON is not an object")
    daily = payload.get("daily")
    units = payload.get("daily_units")
    if not isinstance(daily, dict) or not isinstance(units, dict):
        raise RainfallApiError("unexpected archive response: missing 'daily' or 'daily_units'")
    times = daily.get("time")
    if not isinstance(times, list) or not times:
        raise RainfallApiError("unexpected archive response: the daily series is empty")
    series: list[tuple[float | None, ...]] = []
    for name in VARIABLES:
        if units.get(name) != "°C":
            raise RainfallApiError(
                f"unexpected archive response: {name} unit is {units.get(name)!r}"
            )
        raw = daily.get(name)
        if not isinstance(raw, list) or len(raw) != len(times):
            raise RainfallApiError("unexpected archive response: 'time' and values do not line up")
        values: list[float | None] = []
        for v in raw:
            if v is None:
                values.append(None)
                continue
            celsius = number(v)
            if celsius is None or not math.isfinite(celsius) or abs(celsius) > MAX_ABS_C:
                raise RainfallApiError(f"unexpected archive response: invalid {name} value")
            values.append(celsius)
        series.append(tuple(values))
    lat, lon = number(payload.get("latitude")), number(payload.get("longitude"))
    if lat is None or lon is None:
        raise RainfallApiError("unexpected archive response: grid latitude/longitude missing")
    try:
        days = tuple(date.fromisoformat(t) for t in times)
    except (TypeError, ValueError) as exc:
        raise RainfallApiError("unexpected archive response: invalid date in 'time'") from exc
    timezone = payload.get("timezone")
    return TemperatureArchive(
        dates=days,
        tmax=series[0],
        tmin=series[1],
        tmean=series[2],
        latitude=lat,
        longitude=lon,
        elevation_m=number(payload.get("elevation")),
        timezone=timezone if isinstance(timezone, str) else None,
    )


class TemperatureClient:
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

    def __enter__(self) -> TemperatureClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _params(self, location: Location, start_year: int, end_year: int) -> dict[str, str]:
        return build_temperature_params(
            location.latitude,
            location.longitude,
            date(start_year, 1, 1),
            min(date(end_year, 12, 31), self._today()),
            self._config.model,
        )

    def fetch_archive(
        self, location: Location, start_year: int, end_year: int
    ) -> TemperatureArchive:
        """The daily series ``analyze`` used (a cache hit right after an analysis; no geocoding)."""
        archive, _ = self._fetcher.cached(
            self._config.archive_url,
            self._params(location, start_year, end_year),
            parse_temperature_archive,
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
        metric: str = "hot",
        event_year: int | None = None,
        scale: str = "daily",
        event_month: int | None = None,
    ) -> tuple[TemperatureRecord, TemperatureArchive]:
        """Analyse ``city``: returns the statistics and the daily series they came from.

        ``scale`` ``"daily"``: ``metric`` ``"hot"`` (TXx) or ``"cold"`` (TNn); ``event_date`` asks
        about one day. ``"yearly"``: annual means; ``event_year`` asks about one year.
        ``"monthly"``: monthly means; ``event_year`` (with ``event_month``) asks about one month,
        or each month of a year.
        """
        today = self._today()
        city, country_code, region = validate_city(city, country_code, region)
        first, last = self._validate(
            start_year, end_year, event_date, today, metric, event_year, scale, event_month
        )
        location, others, geo_cached = self._fetcher.resolve_city(city, country_code, region)
        return self._analyze_location(
            location,
            others,
            geo_cached,
            today=today,
            start_year=first,
            end_year=last,
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
        metric: str = "hot",
        event_year: int | None = None,
        scale: str = "daily",
        event_month: int | None = None,
    ) -> tuple[TemperatureRecord, TemperatureArchive]:
        """Like ``analyze``, for a point given by coordinates (no geocoding)."""
        today = self._today()
        latitude, longitude = validate_point(latitude, longitude)
        first, last = self._validate(
            start_year, end_year, event_date, today, metric, event_year, scale, event_month
        )
        location = point_location(latitude, longitude)
        record, archive = self._analyze_location(
            location,
            (),
            True,
            today=today,
            start_year=first,
            end_year=last,
            event_date=event_date,
            metric=metric,
            event_year=event_year,
            scale=scale,
            event_month=event_month,
        )
        caveat = point_caveat(location, archive.latitude, archive.longitude)
        return replace(record, caveats=(*record.caveats, caveat)), archive

    def _validate(
        self,
        start_year: object,
        end_year: object,
        event_date: object,
        today: date,
        metric: object,
        event_year: object,
        scale: object,
        event_month: object,
    ) -> tuple[int, int]:
        if metric not in METRICS:
            raise InvalidRainfallQueryError("metric must be 'hot' or 'cold'")
        return validate_period(
            self._config,
            today,
            start_year=start_year,
            end_year=end_year,
            event_date=event_date,
            event_year=event_year,
            scale=scale,
            event_month=event_month,
            daily_event_year=False,
            daily_event_year_hint="",
        )

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
    ) -> tuple[TemperatureRecord, TemperatureArchive]:
        cfg = self._config
        series_start = min(start_year, cfg.baseline_start_year)
        params = self._params(location, series_start, end_year)
        archive, archive_cached = self._fetcher.cached(
            cfg.archive_url, params, parse_temperature_archive
        )
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
        common: dict[str, Any] = {
            "start_year": start_year,
            "end_year": end_year,
            "config": cfg,
            "provenance": provenance,
            "baseline_from_year": series_start,
            "other_matches": others,
        }
        record: TemperatureRecord
        if scale == "yearly":
            record = compute_yearly(
                location,
                archive.dates,
                archive.tmean,
                last_complete_year=today.year - 1,
                event_year=event_year,
                **common,
            )
        elif scale == "monthly":
            last_month = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
            record = compute_monthly(
                location,
                archive.dates,
                archive.tmean,
                last_complete_month=last_month,
                event_year=event_year,
                event_month=event_month,
                **common,
            )
        else:
            record = compute_extremes(
                location,
                archive.dates,
                archive.series(metric),
                metric=metric,
                last_complete_year=today.year - 1,
                event_date=event_date,
                **common,
            )
        return record, archive
