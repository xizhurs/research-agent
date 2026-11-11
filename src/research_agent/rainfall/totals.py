"""Precipitation totals per year and per month. Pure: no I/O, no LLM.

Yearly: each valid year's total, the wettest and driest years, a baseline (mean, median and the
10th-90th percentile band of the baseline years' totals) and, optionally, how one year compares.

Monthly: totals of the months that have enough daily data, the wettest months, a baseline per
calendar month and, optionally, how one month (or each month of one year) compares with the same
calendar month in other years.

Validity follows the other analyses: a year needs ``min_year_coverage`` of its days and must be
complete; a month needs ``min_year_coverage`` of its days and must have ended. A missing day is
skipped, never counted as zero. Ranks are withheld when fewer than ``min_years`` totals are
compared. The daily series starts at the baseline period when the question asks about a shorter
recent period, so the baseline comparison is always possible; ranks still cover the requested
period only.
"""

from __future__ import annotations

import calendar
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import date

from research_agent.config import RainfallConfig
from research_agent.rainfall.analysis import (
    REANALYSIS_CAVEAT,
    classify_years,
    excluded_years_caveat,
    other_matches_caveat,
)
from research_agent.rainfall.climatology import (
    BAND_HIGH_PERCENTILE,
    BAND_LOW_PERCENTILE,
    all_month_totals,
)
from research_agent.rainfall.errors import InsufficientDataError
from research_agent.rainfall.models import (
    Location,
    MonthlyAnalysis,
    PeriodTotal,
    Provenance,
    TotalContext,
    TotalsBaseline,
    YearlyAnalysis,
)

TOTAL_CAVEAT = (
    "A total sums the daily ERA5 values; a day without data is skipped, never counted as zero, so "
    "a period with missing days reads slightly low."
)
MONTH_NAMES = tuple(calendar.month_name[1:])


def _band_caveat(what: str) -> str:
    outside = BAND_LOW_PERCENTILE + 100 - BAND_HIGH_PERCENTILE
    return (
        f"The baseline band spans the {BAND_LOW_PERCENTILE}th to {BAND_HIGH_PERCENTILE}th "
        f"percentile of {what} in the baseline years; about {outside}% of years fall outside it "
        "by construction, so 'above' or 'below' is unusual, not necessarily extreme."
    )


def baseline_of(sample: Sequence[float], config: RainfallConfig) -> TotalsBaseline | None:
    """Mean, median and percentile band of the baseline totals; None if too few."""
    if len(sample) < max(config.min_years, 2):
        return None
    cuts = statistics.quantiles(sample, n=100, method="inclusive")
    return TotalsBaseline(
        config.baseline_start_year,
        config.baseline_end_year,
        len(sample),
        statistics.fmean(sample),
        statistics.median(sample),
        BAND_LOW_PERCENTILE,
        BAND_HIGH_PERCENTILE,
        cuts[BAND_LOW_PERCENTILE - 1],
        cuts[BAND_HIGH_PERCENTILE - 1],
    )


def _context(
    year: int,
    month: int | None,
    total: float | None,
    compared: Sequence[float],
    baseline: TotalsBaseline | None,
    min_years: int,
) -> TotalContext:
    ranked = total is not None and len(compared) >= min_years
    position = None
    if total is not None and baseline is not None:
        if total < baseline.low_mm:
            position = "below"
        elif total > baseline.high_mm:
            position = "above"
        else:
            position = "within"
    return TotalContext(
        year=year,
        month=month,
        total_mm=total,
        rank_wettest=1 + sum(t > total for t in compared) if ranked else None,
        compared=len(compared),
        percent_drier=100 * sum(t < total for t in compared) / len(compared) if ranked else None,
        ratio_to_baseline_mean=total / baseline.mean_mm
        if total is not None and baseline and baseline.mean_mm > 0
        else None,
        band_position=position,
    )


def _common_caveats(
    label: str,
    baseline_missing: bool,
    config: RainfallConfig,
    location: Location,
    other_matches: Sequence[Location],
) -> list[str]:
    caveats = []
    if baseline_missing:
        caveats.append(
            f"{label}: fewer than {config.min_years} usable totals in the "
            f"{config.baseline_start_year}-{config.baseline_end_year} baseline; no baseline "
            "comparison."
        )
    if other_matches:
        caveats.append(
            other_matches_caveat(
                location, other_matches, config.dominance_ratio, config.same_place_km
            )
        )
    return caveats


def compute_yearly_analysis(
    location: Location,
    dates: Sequence[date],
    values: Sequence[float | None],
    *,
    start_year: int,
    end_year: int,
    last_complete_year: int,
    config: RainfallConfig,
    provenance: Provenance,
    baseline_from_year: int,
    event_year: int | None = None,
    other_matches: Sequence[Location] = (),
) -> YearlyAnalysis:
    """Annual totals of the requested period, compared with the baseline years' totals.

    Raises InsufficientDataError if no year of the period has enough data.
    """
    counts: Counter[int] = Counter()
    sums: defaultdict[int, float] = defaultdict(float)
    for day, mm in zip(dates, values, strict=True):
        if mm is not None:
            counts[day.year] += 1
            sums[day.year] += mm
    cov = config.min_year_coverage
    valid, excluded = classify_years(counts, start_year, end_year, last_complete_year, cov)
    if not valid:
        raise InsufficientDataError(
            f"no year between {start_year} and {end_year} has enough daily data "
            f"(at least {cov:.0%} of days)"
        )
    base_years, _ = classify_years(
        counts, config.baseline_start_year, config.baseline_end_year, last_complete_year, cov
    )
    baseline = baseline_of([sums[y] for y in base_years], config)
    totals = [PeriodTotal(y, None, sums[y]) for y in valid]
    ranking_available = len(totals) >= config.min_years
    label = location.label

    caveats = [REANALYSIS_CAVEAT, TOTAL_CAVEAT]
    if baseline:
        caveats.append(_band_caveat("the annual totals"))
    year_focus = None
    if event_year is not None:
        if event_year in valid:
            year_focus = _context(
                event_year,
                None,
                sums[event_year],
                [t.total_mm for t in totals],
                baseline,
                config.min_years,
            )
        else:
            reason = dict(excluded).get(event_year, "outside the analysed period")
            caveats.append(f"{label}: {event_year} was excluded ({reason}), so it was not ranked.")
    if excluded:
        caveats.append(excluded_years_caveat(label, excluded, "the annual totals"))
    if not ranking_available:
        caveats.append(
            f"{label}: only {len(totals)} valid year(s) in the period (minimum "
            f"{config.min_years}); ranks were not computed, the baseline comparison still was."
        )
    missing = sum((366 if calendar.isleap(y) else 365) - counts[y] for y in valid)
    if missing:
        caveats.append(f"{label}: {missing} day(s) without data fall in valid years.")
    caveats += _common_caveats(label, baseline is None, config, location, other_matches)

    by_wet = sorted(totals, key=lambda t: (-t.total_mm, t.year))
    by_dry = sorted(totals, key=lambda t: (t.total_mm, t.year))
    return YearlyAnalysis(
        id="",
        location=location,
        provenance=provenance,
        start_year=start_year,
        end_year=end_year,
        baseline_from_year=baseline_from_year,
        totals=tuple(totals),
        excluded_years=tuple(excluded),
        baseline=baseline,
        ranking_available=ranking_available,
        wettest=tuple(by_wet[: config.top_events]),
        driest=tuple(by_dry[: config.top_events]),
        year_focus=year_focus,
        caveats=tuple(caveats),
    )


def compute_monthly_analysis(
    location: Location,
    dates: Sequence[date],
    values: Sequence[float | None],
    *,
    start_year: int,
    end_year: int,
    last_complete_month: tuple[int, int],
    config: RainfallConfig,
    provenance: Provenance,
    baseline_from_year: int,
    event_year: int | None = None,
    event_month: int | None = None,
    other_matches: Sequence[Location] = (),
) -> MonthlyAnalysis:
    """Monthly totals of the requested period against each calendar month's baseline.

    ``last_complete_month`` is ``(year, month)`` of the last month that has ended. Raises
    InsufficientDataError if no month of the period has enough data.
    """
    all_totals = {
        key: mm
        for key, mm in all_month_totals(dates, values, config.min_year_coverage).items()
        if key <= last_complete_month
    }
    period = {k: v for k, v in all_totals.items() if start_year <= k[0] <= end_year}
    if not period:
        raise InsufficientDataError(
            f"no month between {start_year} and {end_year} has enough daily data "
            f"(at least {config.min_year_coverage:.0%} of days)"
        )
    expected = sum(
        1
        for y in range(start_year, end_year + 1)
        for m in range(1, 13)
        if (y, m) <= last_complete_month
    )
    climate = tuple(
        baseline_of(
            [
                mm
                for (y, m), mm in all_totals.items()
                if m == month and config.baseline_start_year <= y <= config.baseline_end_year
            ],
            config,
        )
        for month in range(1, 13)
    )
    same_month = {
        month: [mm for (_, m), mm in period.items() if m == month] for month in range(1, 13)
    }
    label = location.label

    def context(year: int, month: int) -> TotalContext:
        return _context(
            year,
            month,
            period.get((year, month)),
            same_month[month],
            climate[month - 1],
            config.min_years,
        )

    caveats = [REANALYSIS_CAVEAT, TOTAL_CAVEAT]
    if any(climate):
        caveats.append(_band_caveat("that calendar month's totals"))
    month_focus = None
    year_months: tuple[TotalContext, ...] = ()
    if event_year is not None and event_month is not None:
        month_focus = context(event_year, event_month)
        if month_focus.total_mm is None:
            caveats.append(
                f"{label}: {MONTH_NAMES[event_month - 1]} {event_year} has not ended or has too "
                "little daily data, so it was not assessed."
            )
    elif event_year is not None:
        year_months = tuple(context(event_year, m) for m in range(1, 13))
        blank = sum(c.total_mm is None for c in year_months)
        if blank:
            caveats.append(
                f"{label}: {blank} month(s) of {event_year} have not ended or have too little "
                "daily data and were not assessed."
            )
    if expected > len(period):
        caveats.append(
            f"{label}: {expected - len(period)} month(s) of the period had too little daily data "
            "and were left out."
        )
    if min(len(v) for v in same_month.values()) < config.min_years:
        caveats.append(
            f"{label}: a calendar month with fewer than {config.min_years} totals in the period "
            "is not ranked; the baseline comparison still is."
        )
    caveats += _common_caveats(label, not any(climate), config, location, other_matches)

    wettest = sorted(period.items(), key=lambda kv: (-kv[1], kv[0]))[: config.top_events]
    return MonthlyAnalysis(
        id="",
        location=location,
        provenance=provenance,
        start_year=start_year,
        end_year=end_year,
        baseline_from_year=baseline_from_year,
        years=tuple(sorted({y for y, _ in period})),
        months_used=len(period),
        months_missing=expected - len(period),
        climate=climate,
        wettest=tuple(PeriodTotal(y, m, mm) for (y, m), mm in wettest),
        month_focus=month_focus,
        year_months=year_months,
        caveats=tuple(caveats),
    )
