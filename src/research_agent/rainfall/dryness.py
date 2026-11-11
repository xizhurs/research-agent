"""Dry-condition statistics: consecutive dry days (CDD) and annual totals. Pure: no I/O, no LLM.

A day is dry if it has less than ``dry_day_mm`` of precipitation (the ETCCDI CDD convention is
1 mm). A missing day breaks a dry run and is never counted as dry. Two views of the same days:

- yearly CDD: the longest run inside each calendar year (runs restart on 1 January);
- unbroken spells: maximal runs over the whole valid record, not split at New Year.

Year validity, withheld ranks and caveat wording for excluded years and other places with the same
name are shared with the wet analysis (``analysis.py``). This is not a drought index.
"""

from __future__ import annotations

import calendar
import statistics
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, timedelta

from research_agent.config import RainfallConfig
from research_agent.rainfall.analysis import (
    classify_years,
    excluded_years_caveat,
    other_matches_caveat,
)
from research_agent.rainfall.errors import InsufficientDataError
from research_agent.rainfall.models import (
    AnnualDry,
    DryAnalysis,
    DryBaseline,
    DryEventContext,
    DrySpell,
    DryYearContext,
    Location,
    Provenance,
)

REANALYSIS_CAVEAT = (
    "Values are ERA5 reanalysis (model output on a grid of about 0.25 degrees, roughly 25 km), "
    "not gauge measurements."
)
LIGHT_RAIN_CAVEAT = (
    "Regional studies find that ERA5 produces too many light-rain days, which can make dry "
    "spells look shorter than observed; dry-spell lengths here may be underestimates."
)
SEASON_CAVEAT = (
    "Yearly CDD restarts on 1 January, so a dry spell that crosses New Year is split in that "
    "figure but not in the list of unbroken spells; where there is a distinct dry season, yearly "
    "CDD mostly reflects the length of that season."
)


def _runs(records: Sequence[tuple[date, float]], dry_day_mm: float) -> list[DrySpell]:
    """Maximal runs of dry days on consecutive calendar days. ``records`` are in date order."""
    spells: list[DrySpell] = []
    start: date | None = None
    prev: date | None = None
    days, total = 0, 0.0
    for day, mm in records:
        dry = mm < dry_day_mm
        contiguous = prev is not None and day - prev == timedelta(days=1)
        if start is not None and prev is not None and not (dry and contiguous):
            spells.append(DrySpell(start.isoformat(), prev.isoformat(), days, total))
            start = None
        if dry:
            if start is None:
                start, days, total = day, 0, 0.0
            days += 1
            total += mm
        prev = day
    if start is not None and prev is not None:
        spells.append(DrySpell(start.isoformat(), prev.isoformat(), days, total))
    return spells


def _longest(spells: Sequence[DrySpell]) -> DrySpell | None:
    """The longest spell; the earliest wins a tie."""
    return min(spells, key=lambda s: (-s.days, s.start), default=None)


def compute_dry_analysis(
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
    event_year: int | None = None,
    other_matches: Sequence[Location] = (),
) -> DryAnalysis:
    """Yearly CDD, unbroken dry spells, annual totals, a baseline and optional focus blocks.

    ``dates`` and ``values`` are the daily series in date order (``None`` = no data). Raises
    InsufficientDataError if no year has enough data or the requested day has no data.
    """
    thr = config.dry_day_mm
    counts: dict[int, int] = defaultdict(int)
    for day, mm in zip(dates, values, strict=True):
        if mm is not None and start_year <= day.year <= end_year:
            counts[day.year] += 1
    valid_list, excluded = classify_years(
        counts, start_year, end_year, last_complete_year, config.min_year_coverage
    )
    if not valid_list:
        raise InsufficientDataError(
            f"no year between {start_year} and {end_year} has enough daily data "
            f"(at least {config.min_year_coverage:.0%} of days)"
        )
    valid = set(valid_list)
    label = location.label

    records = [
        (day, mm)
        for day, mm in zip(dates, values, strict=True)
        if mm is not None and day.year in valid
    ]
    by_year: dict[int, list[tuple[date, float]]] = defaultdict(list)
    for record in records:
        by_year[record[0].year].append(record)

    annual: list[AnnualDry] = []
    for year in valid_list:
        best = _longest(_runs(by_year[year], thr))
        annual.append(
            AnnualDry(
                year,
                best.days if best else 0,
                best.start if best else None,
                best.end if best else None,
                sum(mm for _, mm in by_year[year]),
            )
        )
    unbroken = _runs(records, thr)
    ranking_available = len(annual) >= config.min_years

    in_base = [
        a for a in annual if config.baseline_start_year <= a.year <= config.baseline_end_year
    ]
    baseline = (
        DryBaseline(
            config.baseline_start_year,
            config.baseline_end_year,
            len(in_base),
            statistics.fmean(a.cdd_days for a in in_base),
            statistics.median(a.cdd_days for a in in_base),
            statistics.fmean(a.total_mm for a in in_base),
            statistics.median(a.total_mm for a in in_base),
        )
        if len(in_base) >= config.min_years
        else None
    )

    longest = tuple(sorted(unbroken, key=lambda s: (-s.days, s.start))[: config.top_events])
    driest = tuple(sorted(annual, key=lambda a: (a.total_mm, a.year))[: config.top_events])

    caveats = [
        REANALYSIS_CAVEAT,
        LIGHT_RAIN_CAVEAT,
        f"A dry spell is a run of days with less than {thr:g} mm of rain; it is not a drought "
        "index (no evaporation, soil moisture or accumulated deficit).",
        SEASON_CAVEAT,
    ]

    target = event_year if event_year is not None else (event_date.year if event_date else None)
    year_focus = None
    if target is not None:
        row = next((a for a in annual if a.year == target), None)
        if row is None:
            reason = dict(excluded).get(target, "outside the analysed period")
            caveats.append(
                f"{label}: {target} was excluded ({reason}), so its total and CDD were not ranked."
            )
        else:
            totals = [a.total_mm for a in annual]
            year_focus = DryYearContext(
                year=target,
                total_mm=row.total_mm,
                total_rank=1 + sum(t < row.total_mm for t in totals) if ranking_available else None,
                percent_wetter=100 * sum(t > row.total_mm for t in totals) / len(totals)
                if ranking_available
                else None,
                ratio_to_baseline_mean=row.total_mm / baseline.total_mean_mm
                if baseline and baseline.total_mean_mm > 0
                else None,
                cdd_days=row.cdd_days,
            )

    event = None
    if event_date is not None:
        value = next((mm for d, mm in zip(dates, values, strict=True) if d == event_date), None)
        if value is None:
            raise InsufficientDataError(
                f"no data for {event_date.isoformat()} (the archive lags real time by several days)"
            )
        dry_day = value < thr
        iso = event_date.isoformat()
        spell = next((s for s in unbroken if s.start <= iso <= s.end), None) if dry_day else None
        if dry_day and spell is None:
            caveats.append(
                f"{label}: {iso} lies in an excluded year, so its dry spell was not measured."
            )
        event = DryEventContext(
            date=iso,
            value_mm=value,
            dry_day=dry_day,
            spell=spell,
            spell_rank=1 + sum(s.days > spell.days for s in unbroken)
            if spell and ranking_available
            else None,
            spells=len(unbroken),
        )

    if excluded:
        caveats.append(excluded_years_caveat(label, excluded, "the dry-condition statistics"))
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
    missing = sum((366 if calendar.isleap(y) else 365) - counts[y] for y in valid_list)
    if missing:
        caveats.append(
            f"{label}: {missing} day(s) without data fall in valid years; they break dry runs and "
            "make the totals of those years slightly low."
        )
    if other_matches:
        caveats.append(
            other_matches_caveat(
                location, other_matches, config.dominance_ratio, config.same_place_km
            )
        )

    return DryAnalysis(
        id="",
        location=location,
        provenance=provenance,
        start_year=start_year,
        end_year=end_year,
        dry_day_mm=thr,
        annual=tuple(annual),
        excluded_years=tuple(excluded),
        baseline=baseline,
        ranking_available=ranking_available,
        longest_spells=longest,
        driest_years=driest,
        year_focus=year_focus,
        event=event,
        caveats=tuple(caveats),
    )
