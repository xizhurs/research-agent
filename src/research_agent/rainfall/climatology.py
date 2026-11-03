"""A smoothed day-of-year baseline for a daily precipitation series. Pure: no I/O, no plotting.

For each calendar day (366 slots, 29 February included) it averages the daily values over the
baseline years, then smooths that annual cycle with a centred moving window that wraps around the
year end. Only valid years take part (complete, with enough days of data: the same rule the
annual statistics use), and a missing day is skipped, never counted as zero.

This is a different statistic from ``Baseline`` in ``models.py`` (the mean of yearly maxima or
yearly totals): it describes the typical rain on a given day, and is not a threshold for extremes.
"""

from __future__ import annotations

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

    days_with_data = Counter(
        d.year for d, v in zip(dates, values, strict=True) if v is not None
    )
    valid, _excluded = classify_years(
        days_with_data, start_year, end_year, last_complete_year, min_year_coverage
    )
    if len(valid) < min_years:
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
