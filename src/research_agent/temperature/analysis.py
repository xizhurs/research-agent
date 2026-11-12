"""Temperature statistics from ERA5 daily series. Pure: no I/O, no LLM.

Yearly: each valid year's mean temperature, the warmest and coolest years, a baseline (mean,
median and the 10th-90th percentile band of the baseline years' means) and, optionally, how one
year compares. Monthly: the same per calendar month, plus the months furthest above and below
their own calendar month's baseline mean. Daily extremes: TXx (highest daily maximum of a year,
metric "hot") or TNn (lowest daily minimum, metric "cold"), the most extreme years, and how one
day compares with the record and with the baseline mean for its calendar day.

Validity follows the rainfall analyses: a year needs ``min_year_coverage`` of its days and must be
complete; a month needs ``min_year_coverage`` of its days and must have ended. A missing day is
skipped, never filled in. Ranks are withheld when fewer than ``min_years`` values are compared.
The daily series starts at the baseline period at the latest, so the baseline comparison is
always possible; ranks cover the requested period only. No trend is computed.
"""

from __future__ import annotations

import calendar
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import date

from research_agent.config import RainfallConfig
from research_agent.rainfall.analysis import (
    classify_years,
    excluded_years_caveat,
    other_matches_caveat,
)
from research_agent.rainfall.climatology import (
    BAND_HIGH_PERCENTILE,
    BAND_LOW_PERCENTILE,
    daily_baseline,
)
from research_agent.rainfall.errors import InsufficientDataError, InvalidRainfallQueryError
from research_agent.rainfall.models import Location, Provenance
from research_agent.temperature.models import (
    AnnualExtreme,
    DayContext,
    ExtremeBaseline,
    MeanContext,
    MonthAnomaly,
    PeriodMean,
    TempBaseline,
    TemperatureExtremes,
    TemperatureMonthly,
    TemperatureYearly,
)

METRICS = ("hot", "cold")
DAY_OF_YEAR_WINDOW = 15  # days in the centred window that smooths the calendar-day baseline
MONTH_NAMES = tuple(calendar.month_name[1:])

REANALYSIS_CAVEAT = (
    "Values are ERA5 reanalysis 2 m air temperature for a grid cell of about 0.25 degrees "
    "(roughly 25 km), not station measurements; a grid cell smooths local extremes and does not "
    "represent urban heat."
)
MEAN_CAVEAT = (
    "A mean averages the daily means (Open-Meteo temperature_2m_mean) of the days with data; a "
    "day without data is skipped, never filled in."
)
NO_TREND_CAVEAT = (
    "No trend was computed: differences between single years are not a trend estimate."
)
HOT_CAVEAT = (
    "TXx is the highest daily maximum temperature of a year; one hot day is not a heatwave, "
    "which needs several days above a threshold (definitions vary)."
)
COLD_CAVEAT = "TNn is the lowest daily minimum temperature of a year (usually a night)."


def _elevation_caveat(provenance: Provenance) -> list[str]:
    if provenance.elevation_m is None:
        return []
    return [
        f"The ERA5 grid cell's elevation is {provenance.elevation_m:.0f} m; where the place lies "
        "higher or lower, its temperature differs (roughly 6.5 degrees C per 1,000 m). No "
        "correction was applied."
    ]


def _band_caveat(what: str) -> str:
    outside = BAND_LOW_PERCENTILE + 100 - BAND_HIGH_PERCENTILE
    return (
        f"The baseline band spans the {BAND_LOW_PERCENTILE}th to {BAND_HIGH_PERCENTILE}th "
        f"percentile of {what} in the baseline years; about {outside}% of years fall outside it "
        "by construction, so 'above' or 'below' is unusual, not necessarily extreme."
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
            f"{label}: fewer than {config.min_years} usable values in the "
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


# --- means -------------------------------------------------------------------------------------


def year_counts_and_means(
    dates: Sequence[date], values: Sequence[float | None]
) -> tuple[Counter[int], dict[int, float]]:
    """Days with data and the mean over those days, per calendar year (no validity check)."""
    counts: Counter[int] = Counter()
    sums: defaultdict[int, float] = defaultdict(float)
    for day, value in zip(dates, values, strict=True):
        if value is not None:
            counts[day.year] += 1
            sums[day.year] += value
    return counts, {y: sums[y] / counts[y] for y in counts}


def valid_annual_means(
    dates: Sequence[date],
    values: Sequence[float | None],
    *,
    start_year: int,
    end_year: int,
    last_complete_year: int,
    min_year_coverage: float,
) -> tuple[dict[int, float], list[tuple[int, str]]]:
    """Mean of each valid year in the range, and the excluded years with reasons."""
    counts, means = year_counts_and_means(dates, values)
    valid, excluded = classify_years(
        counts, start_year, end_year, last_complete_year, min_year_coverage
    )
    return {y: means[y] for y in valid}, excluded


def monthly_means(
    dates: Sequence[date], values: Sequence[float | None], min_year_coverage: float
) -> dict[tuple[int, int], float]:
    """Mean of every month that has at least ``min_year_coverage`` of its days with data.

    Whether the month has ended is the caller's check.
    """
    counts: Counter[tuple[int, int]] = Counter()
    sums: defaultdict[tuple[int, int], float] = defaultdict(float)
    for day, value in zip(dates, values, strict=True):
        if value is not None:
            key = (day.year, day.month)
            counts[key] += 1
            sums[key] += value
    return {
        key: sums[key] / n
        for key, n in counts.items()
        if n >= min_year_coverage * calendar.monthrange(*key)[1]
    }


def baseline_of(sample: Sequence[float], config: RainfallConfig) -> TempBaseline | None:
    """Mean, median and percentile band of the baseline values; None if too few."""
    if len(sample) < max(config.min_years, 2):
        return None
    cuts = statistics.quantiles(sample, n=100, method="inclusive")
    return TempBaseline(
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
    mean: float | None,
    compared: Sequence[float],
    baseline: TempBaseline | None,
    min_years: int,
) -> MeanContext:
    ranked = mean is not None and len(compared) >= min_years
    position = None
    if mean is not None and baseline is not None:
        if mean < baseline.low_c:
            position = "below"
        elif mean > baseline.high_c:
            position = "above"
        else:
            position = "within"
    return MeanContext(
        year=year,
        month=month,
        mean_c=mean,
        rank_warmest=1 + sum(t > mean for t in compared) if ranked else None,
        compared=len(compared),
        percent_cooler=100 * sum(t < mean for t in compared) / len(compared) if ranked else None,
        anomaly_c=mean - baseline.mean_c if mean is not None and baseline else None,
        band_position=position,
    )


def compute_yearly(
    location: Location,
    dates: Sequence[date],
    tmean: Sequence[float | None],
    *,
    start_year: int,
    end_year: int,
    last_complete_year: int,
    config: RainfallConfig,
    provenance: Provenance,
    baseline_from_year: int,
    event_year: int | None = None,
    other_matches: Sequence[Location] = (),
) -> TemperatureYearly:
    """Annual means of the requested period, compared with the baseline years' means.

    Raises InsufficientDataError if no year of the period has enough data.
    """
    counts, means = year_counts_and_means(dates, tmean)
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
    baseline = baseline_of([means[y] for y in base_years], config)
    period = [PeriodMean(y, None, means[y]) for y in valid]
    ranking_available = len(period) >= config.min_years
    label = location.label

    caveats = [REANALYSIS_CAVEAT, MEAN_CAVEAT, NO_TREND_CAVEAT, *_elevation_caveat(provenance)]
    if baseline:
        caveats.append(_band_caveat("the annual means"))
    year_focus = None
    if event_year is not None:
        if event_year in valid:
            year_focus = _context(
                event_year,
                None,
                means[event_year],
                [m.mean_c for m in period],
                baseline,
                config.min_years,
            )
        else:
            reason = dict(excluded).get(event_year, "outside the analysed period")
            caveats.append(f"{label}: {event_year} was excluded ({reason}), so it was not ranked.")
    if excluded:
        caveats.append(excluded_years_caveat(label, excluded, "the annual means"))
    if not ranking_available:
        caveats.append(
            f"{label}: only {len(period)} valid year(s) in the period (minimum "
            f"{config.min_years}); ranks were not computed, the baseline comparison still was."
        )
    missing = sum((366 if calendar.isleap(y) else 365) - counts[y] for y in valid)
    if missing:
        caveats.append(f"{label}: {missing} day(s) without data fall in valid years.")
    caveats += _common_caveats(label, baseline is None, config, location, other_matches)

    by_warm = sorted(period, key=lambda m: (-m.mean_c, m.year))
    by_cool = sorted(period, key=lambda m: (m.mean_c, m.year))
    return TemperatureYearly(
        id="",
        location=location,
        provenance=provenance,
        start_year=start_year,
        end_year=end_year,
        baseline_from_year=baseline_from_year,
        means=tuple(period),
        excluded_years=tuple(excluded),
        baseline=baseline,
        ranking_available=ranking_available,
        warmest=tuple(by_warm[: config.top_events]),
        coolest=tuple(by_cool[: config.top_events]),
        year_focus=year_focus,
        caveats=tuple(caveats),
    )


def compute_monthly(
    location: Location,
    dates: Sequence[date],
    tmean: Sequence[float | None],
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
) -> TemperatureMonthly:
    """Monthly means of the requested period against each calendar month's baseline.

    ``last_complete_month`` is ``(year, month)`` of the last month that has ended. Raises
    InsufficientDataError if no month of the period has enough data.
    """
    all_means = {
        key: value
        for key, value in monthly_means(dates, tmean, config.min_year_coverage).items()
        if key <= last_complete_month
    }
    period = {k: v for k, v in all_means.items() if start_year <= k[0] <= end_year}
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
                value
                for (y, m), value in all_means.items()
                if m == month and config.baseline_start_year <= y <= config.baseline_end_year
            ],
            config,
        )
        for month in range(1, 13)
    )
    same_month = {
        month: [v for (_, m), v in period.items() if m == month] for month in range(1, 13)
    }
    label = location.label

    def context(year: int, month: int) -> MeanContext:
        return _context(
            year,
            month,
            period.get((year, month)),
            same_month[month],
            climate[month - 1],
            config.min_years,
        )

    caveats = [REANALYSIS_CAVEAT, MEAN_CAVEAT, *_elevation_caveat(provenance)]
    if any(climate):
        caveats.append(_band_caveat("that calendar month's means"))
    month_focus = None
    year_months: tuple[MeanContext, ...] = ()
    if event_year is not None and event_month is not None:
        month_focus = context(event_year, event_month)
        if month_focus.mean_c is None:
            caveats.append(
                f"{label}: {MONTH_NAMES[event_month - 1]} {event_year} has not ended or has too "
                "little daily data, so it was not assessed."
            )
    elif event_year is not None:
        year_months = tuple(context(event_year, m) for m in range(1, 13))
        blank = sum(c.mean_c is None for c in year_months)
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
            f"{label}: a calendar month with fewer than {config.min_years} means in the period "
            "is not ranked; the baseline comparison still is."
        )
    caveats += _common_caveats(label, not any(climate), config, location, other_matches)

    anomalies = [
        MonthAnomaly(y, m, v, v - base.mean_c)
        for (y, m), v in sorted(period.items())
        if (base := climate[m - 1]) is not None
    ]
    if any(climate):
        caveats.append(
            "Warm and cool months are ranked by their difference from the baseline mean of the "
            "same calendar month, not by absolute temperature."
        )
    warm = sorted(anomalies, key=lambda a: (-a.anomaly_c, a.year, a.month))
    cool = sorted(anomalies, key=lambda a: (a.anomaly_c, a.year, a.month))
    return TemperatureMonthly(
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
        warm_anomalies=tuple(warm[: config.top_events]),
        cool_anomalies=tuple(cool[: config.top_events]),
        month_focus=month_focus,
        year_months=year_months,
        caveats=tuple(caveats),
    )


# --- daily extremes ----------------------------------------------------------------------------


def _annual_extremes(
    dates: Sequence[date], values: Sequence[float | None], hot: bool
) -> tuple[Counter[int], dict[int, AnnualExtreme]]:
    """Days with data and the most extreme value per calendar year (earliest date on a tie)."""
    counts: Counter[int] = Counter()
    best: dict[int, AnnualExtreme] = {}
    for day, value in zip(dates, values, strict=True):
        if value is None:
            continue
        counts[day.year] += 1
        current = best.get(day.year)
        # Strictly more extreme: on a tie the earliest date wins (input is in date order).
        if current is None or (value > current.value_c if hot else value < current.value_c):
            best[day.year] = AnnualExtreme(day.year, day.isoformat(), value)
    return counts, best


def compute_extremes(
    location: Location,
    dates: Sequence[date],
    values: Sequence[float | None],
    *,
    metric: str,
    start_year: int,
    end_year: int,
    last_complete_year: int,
    config: RainfallConfig,
    provenance: Provenance,
    baseline_from_year: int,
    event_date: date | None = None,
    other_matches: Sequence[Location] = (),
) -> TemperatureExtremes:
    """TXx or TNn per year of the period, a baseline and an optional day's context.

    ``values`` is the daily maximum series for ``metric`` "hot" and the daily minimum series for
    "cold". Raises InsufficientDataError if no year has enough data or the day has no data.
    """
    if metric not in METRICS:
        raise InvalidRainfallQueryError("metric must be 'hot' or 'cold'")
    hot = metric == "hot"
    cov = config.min_year_coverage
    counts, best = _annual_extremes(dates, values, hot)
    valid, excluded = classify_years(counts, start_year, end_year, last_complete_year, cov)
    if not valid:
        raise InsufficientDataError(
            f"no year between {start_year} and {end_year} has enough daily data "
            f"(at least {cov:.0%} of days)"
        )
    annual = [best[y] for y in valid]
    base_years, _ = classify_years(
        counts, config.baseline_start_year, config.baseline_end_year, last_complete_year, cov
    )
    in_baseline = [best[y].value_c for y in base_years]
    baseline = (
        ExtremeBaseline(
            config.baseline_start_year,
            config.baseline_end_year,
            len(in_baseline),
            statistics.fmean(in_baseline),
            statistics.median(in_baseline),
        )
        if len(in_baseline) >= config.min_years
        else None
    )
    ranking_available = len(annual) >= config.min_years
    label = location.label
    sign = 1 if hot else -1  # rank on sign * value, so "more extreme" is always "larger"

    event = None
    event_year_valid = True
    if event_date is not None:
        value = next((v for d, v in zip(dates, values, strict=True) if d == event_date), None)
        if value is None:
            raise InsufficientDataError(
                f"no data for {event_date.isoformat()} (the archive lags real time by several days)"
            )
        event_year_valid = event_date.year in valid
        valid_set = set(valid)
        record = [
            v for d, v in zip(dates, values, strict=True) if v is not None and d.year in valid_set
        ]
        extremes = [a.value_c for a in annual]
        doy = daily_baseline(
            dates,
            values,
            start_year=config.baseline_start_year,
            end_year=config.baseline_end_year,
            last_complete_year=last_complete_year,
            min_years=config.min_years,
            min_year_coverage=cov,
            window_days=DAY_OF_YEAR_WINDOW,
        )
        doy_mean = doy.at(event_date) if doy else None
        event = DayContext(
            date=event_date.isoformat(),
            value_c=value,
            rank_among_days=1 + sum(sign * v > sign * value for v in record)
            if ranking_available
            else None,
            days=len(record),
            rank_among_annual_extremes=1 + sum(sign * e > sign * value for e in extremes)
            if ranking_available
            else None,
            annual_percentile=100 * sum(sign * e < sign * value for e in extremes) / len(extremes)
            if ranking_available
            else None,
            difference_from_baseline_extreme_c=value - baseline.mean_c if baseline else None,
            day_of_year_mean_c=doy_mean,
            difference_from_day_of_year_c=value - doy_mean if doy_mean is not None else None,
        )

    caveats = [REANALYSIS_CAVEAT, HOT_CAVEAT if hot else COLD_CAVEAT]
    caveats += _elevation_caveat(provenance)
    name = "TXx" if hot else "TNn"
    if event is not None and event.day_of_year_mean_c is not None:
        caveats.append(
            f"The calendar-day mean is the {config.baseline_start_year}-"
            f"{config.baseline_end_year} mean daily {'maximum' if hot else 'minimum'} for that "
            f"day, smoothed over {DAY_OF_YEAR_WINDOW} days; it is typical weather, not an "
            "extreme threshold."
        )
    if excluded:
        caveats.append(excluded_years_caveat(label, excluded, f"the annual {name} values"))
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
            f"{label}: {event.date} lies in an excluded year or outside the period, so it is "
            "compared with the other years' record only."
        )
    if other_matches:
        caveats.append(
            other_matches_caveat(
                location, other_matches, config.dominance_ratio, config.same_place_km
            )
        )

    top = sorted(annual, key=lambda a: (-sign * a.value_c, a.date))[: config.top_events]
    return TemperatureExtremes(
        id="",
        location=location,
        provenance=provenance,
        metric=metric,
        start_year=start_year,
        end_year=end_year,
        baseline_from_year=baseline_from_year,
        annual=tuple(annual),
        excluded_years=tuple(excluded),
        baseline=baseline,
        ranking_available=ranking_available,
        top=tuple(top),
        event=event,
        caveats=tuple(caveats),
    )
