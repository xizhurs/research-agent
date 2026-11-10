"""Baselines for a daily precipitation series, for plotting. Pure: no I/O, no plotting.

Daily: for each calendar day (366 slots, 29 February included) the mean over the baseline years,
smoothed with a centred moving window that wraps around the year end.

Monthly: for each month, the 10th and 90th percentile of the monthly totals over the baseline
years.

Only valid years take part (complete, with enough days of data: the same rule the annual
statistics use), and a missing day is skipped, never counted as zero. These are different
statistics from ``Baseline`` in ``models.py`` (the mean of yearly maxima or yearly totals): they
describe typical rain, and are not thresholds for extremes.
"""

from __future__ import annotations

import calendar
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from research_agent.rainfall.analysis import classify_years

SLOTS = 366
MAX_WINDOW_DAYS = 365


def _slot(day: date) -> int:
    """Index 0..365 of a month and day in a leap year, so 1 March is the same slot every year."""
    return date(2000, day.month, day.day).timetuple().tm_yday - 1


def _valid_baseline_years(
    dates: Sequence[date],
    values: Sequence[float | None],
    start_year: int,
    end_year: int,
    last_complete_year: int,
    min_years: int,
    min_year_coverage: float,
) -> list[int] | None:
    """Valid years of the baseline period in the series, or None if fewer than ``min_years``."""
    days_with_data = Counter(d.year for d, v in zip(dates, values, strict=True) if v is not None)
    valid, _excluded = classify_years(
        days_with_data, start_year, end_year, last_complete_year, min_year_coverage
    )
    return valid if len(valid) >= min_years else None


@dataclass(frozen=True)
class DailyBaseline:
    start_year: int  # the baseline period asked for
    end_year: int
    years: tuple[int, ...]  # valid years that were averaged
    window_days: int
    mm_by_slot: tuple[float | None, ...]  # smoothed mean mm/day per slot; None if no data at all

    def at(self, day: date) -> float | None:
        """The smoothed baseline for the calendar day of ``day`` (any year)."""
        return self.mm_by_slot[_slot(day)]


def daily_baseline(
    dates: Sequence[date],
    values: Sequence[float | None],
    *,
    start_year: int,
    end_year: int,
    last_complete_year: int,
    min_years: int,
    min_year_coverage: float,
    window_days: int,
) -> DailyBaseline | None:
    """Smoothed day-of-year mean over the valid years in ``start_year``-``end_year``.

    ``dates`` and ``values`` are the daily series (``None`` = no data). Returns None when fewer
    than ``min_years`` valid years are available, as the annual baseline does. ``window_days``
    must be odd, 1 meaning no smoothing.
    """
    if window_days < 1 or window_days % 2 == 0 or window_days > MAX_WINDOW_DAYS:
        raise ValueError(f"window_days must be an odd number from 1 to {MAX_WINDOW_DAYS}")

    valid = _valid_baseline_years(
        dates, values, start_year, end_year, last_complete_year, min_years, min_year_coverage
    )
    if valid is None:
        return None
    in_baseline = set(valid)

    sums = [0.0] * SLOTS
    counts = [0] * SLOTS
    for day, mm in zip(dates, values, strict=True):
        if mm is not None and day.year in in_baseline:
            slot = _slot(day)
            sums[slot] += mm
            counts[slot] += 1
    raw = [sums[i] / counts[i] if counts[i] else None for i in range(SLOTS)]

    half = window_days // 2
    smoothed: list[float | None] = []
    for i in range(SLOTS):
        window = [raw[(i + k) % SLOTS] for k in range(-half, half + 1)]
        present = [v for v in window if v is not None]
        smoothed.append(statistics.fmean(present) if present else None)
    return DailyBaseline(start_year, end_year, tuple(valid), window_days, tuple(smoothed))


# --- monthly totals against a percentile band ---------------------------------------------------

BAND_LOW_PERCENTILE = 10
BAND_HIGH_PERCENTILE = 90


def _month_totals(
    dates: Sequence[date], values: Sequence[float | None], min_year_coverage: float
) -> dict[tuple[int, int], float]:
    """Total precipitation of every ``(year, month)`` that has enough days with data.

    A month needs at least ``min_year_coverage`` of its days; a month with fewer (the current
    month, for instance) is left out, and a missing day is never counted as zero. A month with a
    few missing days therefore understates its total by up to the missing share.
    """
    sums: dict[tuple[int, int], float] = {}
    counts: Counter[tuple[int, int]] = Counter()
    for day, mm in zip(dates, values, strict=True):
        if mm is not None:
            key = (day.year, day.month)
            sums[key] = sums.get(key, 0.0) + mm
            counts[key] += 1
    return {
        (year, month): total
        for (year, month), total in sums.items()
        if counts[year, month] >= min_year_coverage * calendar.monthrange(year, month)[1]
    }


def monthly_totals(
    dates: Sequence[date],
    values: Sequence[float | None],
    *,
    year: int,
    min_year_coverage: float,
) -> tuple[float | None, ...]:
    """The 12 monthly totals (mm) of ``year``; None for a month without enough data."""
    totals = _month_totals(dates, values, min_year_coverage)
    return tuple(totals.get((year, month)) for month in range(1, 13))


@dataclass(frozen=True)
class MonthlyBaseline:
    start_year: int  # the baseline period asked for
    end_year: int
    years: tuple[int, ...]  # valid years that were used
    low_percentile: int
    high_percentile: int
    low_mm: tuple[float | None, ...]  # per month (January first); None if too few totals
    high_mm: tuple[float | None, ...]


def monthly_baseline(
    dates: Sequence[date],
    values: Sequence[float | None],
    *,
    start_year: int,
    end_year: int,
    last_complete_year: int,
    min_years: int,
    min_year_coverage: float,
    low_percentile: int = BAND_LOW_PERCENTILE,
    high_percentile: int = BAND_HIGH_PERCENTILE,
) -> MonthlyBaseline | None:
    """Percentile band of each month's total over the valid years in the baseline period.

    Percentiles use linear interpolation between the sorted totals. Returns None when fewer than
    ``min_years`` valid years are available; an individual month with fewer than ``min_years``
    usable totals has no band (None). By construction about ``low + (100 - high)`` percent of
    years fall outside the band.
    """
    if not 0 < low_percentile < high_percentile < 100:
        raise ValueError("percentiles must satisfy 0 < low < high < 100")
    valid = _valid_baseline_years(
        dates, values, start_year, end_year, last_complete_year, min_years, min_year_coverage
    )
    if valid is None:
        return None
    totals = _month_totals(dates, values, min_year_coverage)

    low: list[float | None] = []
    high: list[float | None] = []
    for month in range(1, 13):
        sample = [totals[y, month] for y in valid if (y, month) in totals]
        if len(sample) < max(min_years, 2):
            low.append(None)
            high.append(None)
            continue
        cuts = statistics.quantiles(sample, n=100, method="inclusive")
        low.append(cuts[low_percentile - 1])
        high.append(cuts[high_percentile - 1])
    return MonthlyBaseline(
        start_year, end_year, tuple(valid), low_percentile, high_percentile, tuple(low), tuple(high)
    )
