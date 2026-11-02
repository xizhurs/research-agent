"""Rx1day (annual maximum daily precipitation) statistics. Pure: no I/O, no LLM.

All numbers shown to the model come from here. Missing data is never filled in: a year needs at
least ``min_year_coverage`` of its days, otherwise it is excluded and reported. Ranks and
percentiles are withheld when the record is shorter than ``min_years``.
"""

from __future__ import annotations

import calendar
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from research_agent.config import RainfallConfig
from research_agent.rainfall.errors import InsufficientDataError
from research_agent.rainfall.models import (
    AnnualMax,
    Baseline,
    EventContext,
    Location,
    Provenance,
    RainfallAnalysis,
)

REANALYSIS_CAVEAT = (
    "Values are ERA5 reanalysis (model output on a grid of about 0.25 degrees, roughly 25 km), "
    "not gauge measurements; a grid-cell daily total usually understates the extremes recorded "
    "at a single station."
)
RX1DAY_CAVEAT = (
    "Rx1day is the largest single-day total of a year; it does not capture rain that accumulated "
    "over several days."
)
MAX_YEARS_LISTED = 5


@dataclass
class _YearStats:
    days_with_data: int = 0
    best_mm: float = -1.0
    best_date: date | None = None


def _group_by_year(
    dates: Sequence[date], values: Sequence[float | None], start_year: int, end_year: int
) -> dict[int, _YearStats]:
    stats: dict[int, _YearStats] = {}
    for day, mm in zip(dates, values, strict=True):
        if mm is None or not start_year <= day.year <= end_year:
            continue
        year = stats.setdefault(day.year, _YearStats())
        year.days_with_data += 1
        # Strictly greater: on a tie the earliest date wins (input is in date order).
        if mm > year.best_mm:
            year.best_mm, year.best_date = mm, day
    return stats


def compute_analysis(
    location: Location,
    dates: Sequence[date],
    values: Sequence[float | None],
    *,
    start_year: int,
    end_year: int,
    last_complete_year: int,
    config: RainfallConfig,
    provenance: Provenance,
    event_date: date | None = None,
    other_matches: Sequence[Location] = (),
) -> RainfallAnalysis:
    """Compute Rx1day per year, a baseline, the largest events and an optional day's context.

    ``dates`` and ``values`` are the daily series (``None`` = no data). Raises
    InsufficientDataError if no year has enough data or the requested day has no data.
    """
    per_year = _group_by_year(dates, values, start_year, end_year)

    annual: list[AnnualMax] = []
    excluded: list[tuple[int, str]] = []
    for year in range(start_year, end_year + 1):
        if year > last_complete_year:
            excluded.append((year, "year not complete"))
            continue
        expected = 366 if calendar.isleap(year) else 365
        stats = per_year.get(year)
        have = stats.days_with_data if stats else 0
        if have < config.min_year_coverage * expected or stats is None or stats.best_date is None:
            excluded.append((year, f"only {have} of {expected} days have data"))
            continue
        annual.append(AnnualMax(year, stats.best_date.isoformat(), stats.best_mm))
    if not annual:
        raise InsufficientDataError(
            f"no year between {start_year} and {end_year} has enough daily data "
            f"(at least {config.min_year_coverage:.0%} of days)"
        )

    valid_years = {a.year for a in annual}
    label = location.label
    ranking_available = len(annual) >= config.min_years

    in_baseline = [
        a.mm for a in annual if config.baseline_start_year <= a.year <= config.baseline_end_year
    ]
    baseline = (
        Baseline(
            config.baseline_start_year,
            config.baseline_end_year,
            len(in_baseline),
            statistics.fmean(in_baseline),
            statistics.median(in_baseline),
        )
        if len(in_baseline) >= config.min_years
        else None
    )

    top = tuple(sorted(annual, key=lambda a: (-a.mm, a.date))[: config.top_events])

    event = None
    event_year_valid = True
    if event_date is not None:
        mm = next((v for d, v in zip(dates, values, strict=True) if d == event_date), None)
        if mm is None:
            raise InsufficientDataError(
                f"no data for {event_date.isoformat()} (the archive lags real time by several days)"
            )
        event_year_valid = event_date.year in valid_years
        record = [
            v
            for d, v in zip(dates, values, strict=True)
            if v is not None and d.year in valid_years
        ]
        maxima = [a.mm for a in annual]
        event = EventContext(
            date=event_date.isoformat(),
            mm=mm,
            rank_among_days=1 + sum(v > mm for v in record) if ranking_available else None,
            days=len(record),
            rank_among_annual_maxima=1 + sum(m > mm for m in maxima) if ranking_available else None,
            annual_percentile=100 * sum(m < mm for m in maxima) / len(maxima)
            if ranking_available
            else None,
            ratio_to_baseline_mean=mm / baseline.mean_mm
            if baseline and baseline.mean_mm > 0
            else None,
        )

    caveats = [REANALYSIS_CAVEAT, RX1DAY_CAVEAT]
    if excluded:
        listed = ", ".join(f"{y} ({r})" for y, r in excluded[:MAX_YEARS_LISTED])
        more = "..." if len(excluded) > MAX_YEARS_LISTED else ""
        caveats.append(
            f"{label}: {len(excluded)} year(s) were excluded from the annual maxima: "
            f"{listed}{more}."
        )
    if not ranking_available:
        caveats.append(
            f"{label}: only {len(annual)} valid year(s) (minimum {config.min_years}); ranks and "
            "percentiles were not computed."
        )
    if baseline is None:
        caveats.append(
            f"{label}: fewer than {config.min_years} valid years in the "
            f"{config.baseline_start_year}-{config.baseline_end_year} baseline; no baseline "
            "comparison."
        )
    if event is not None and not event_year_valid:
        caveats.append(
            f"{label}: {event.date} lies in an excluded year, so it is compared with the other "
            "years' record only."
        )
    if other_matches:
        names = "; ".join(m.label for m in other_matches[:3])
        caveats.append(
            f"'{location.name}' also matched {len(other_matches)} other place(s) (for example "
            f"{names}); {label} was used because its population is at least "
            f"{config.dominance_ratio:g} times larger."
        )

    return RainfallAnalysis(
        id="",
        location=location,
        provenance=provenance,
        start_year=start_year,
        end_year=end_year,
        annual_maxima=tuple(annual),
        excluded_years=tuple(excluded),
        baseline=baseline,
        ranking_available=ranking_available,
        top_events=top,
        event=event,
        caveats=tuple(caveats),
    )
