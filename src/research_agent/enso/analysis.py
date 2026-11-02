"""Pure ENSO event detection on the RONI series. No I/O, no LLM.

The file is a table of overlapping 3-month seasons (DJF, JFM, ... NDJ), 12 per year, one index
value each. An event is a maximal run of consecutive seasons at or beyond +/-threshold, kept when
it spans at least ``min_seasons`` seasons. All numbers shown to the model come from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from research_agent.config import EnsoConfig
from research_agent.enso.errors import EnsoApiError, InsufficientEnsoDataError
from research_agent.enso.models import (
    EL_NINO,
    LA_NINA,
    PHASE_LABELS,
    EnsoAnalysis,
    EnsoFocus,
    EnsoRun,
)

INDEX_NAME = "RONI"
SEASONS = ("DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ")
# The index is a few degrees at most. A larger value is a missing-value sentinel (such as -99.9)
# or a corrupt file, and would silently create or break events.
MAX_ABS_INDEX = 10.0
HEADER = ["SEAS", "YR", "ANOM"]


@dataclass(frozen=True)
class Season:
    year: int
    slot: int  # 0 = DJF ... 11 = NDJ; seasons within a year run in calendar order
    value: float

    @property
    def number(self) -> int:
        """Consecutive across years: the next season always has number + 1."""
        return self.year * 12 + self.slot

    @property
    def label(self) -> str:
        return f"{SEASONS[self.slot]} {self.year}"


def parse_roni(text: str) -> tuple[Season, ...]:
    """Parse the RONI table strictly. Raises EnsoApiError on anything unexpected.

    Gaps, duplicates and out-of-order rows are rejected because a missing season would split
    an event in two (or hide one) without any sign of it in the result.
    """
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows or rows[0] != HEADER:
        raise EnsoApiError("unexpected RONI file: the header line is not 'SEAS YR ANOM'")
    seasons: list[Season] = []
    for tokens in rows[1:]:
        shown = " ".join(tokens)[:60]
        if len(tokens) != 3 or tokens[0] not in SEASONS:
            raise EnsoApiError(f"unexpected RONI file: bad row {shown!r}")
        try:
            year, value = int(tokens[1]), float(tokens[2])
        except ValueError:
            raise EnsoApiError(f"unexpected RONI file: bad number in row {shown!r}") from None
        if not math.isfinite(value) or abs(value) > MAX_ABS_INDEX:
            raise EnsoApiError(f"unexpected RONI file: implausible value in row {shown!r}")
        season = Season(year, SEASONS.index(tokens[0]), value)
        if seasons and season.number != seasons[-1].number + 1:
            raise EnsoApiError(
                f"unexpected RONI file: row {shown!r} does not follow {seasons[-1].label}"
            )
        seasons.append(season)
    if not seasons:
        raise EnsoApiError("unexpected RONI file: it has no data rows")
    return tuple(seasons)


def _phase(value: float, threshold: float) -> str | None:
    """The threshold is inclusive: exactly +/-threshold counts."""
    if value >= threshold:
        return EL_NINO
    if value <= -threshold:
        return LA_NINA
    return None


def _runs(seasons: tuple[Season, ...], threshold: float) -> list[tuple[str, int, int]]:
    """Maximal runs of one phase as (phase, first index, last index). Seasons are consecutive."""
    runs: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(seasons) + 1):
        current = _phase(seasons[i].value, threshold) if i < len(seasons) else None
        previous = _phase(seasons[i - 1].value, threshold)
        if previous is not None and current != previous:
            runs.append((previous, start, i - 1))
        if current != previous:
            start = i
    return runs


def strength_label(peak: float, config: EnsoConfig) -> str:
    size = abs(peak)
    if size >= config.strength_very_strong:
        return "very strong"
    if size >= config.strength_strong:
        return "strong"
    if size >= config.strength_moderate:
        return "moderate"
    return "weak"


def _make_run(
    seasons: tuple[Season, ...], run: tuple[str, int, int], config: EnsoConfig
) -> EnsoRun:
    phase, first, last = run
    members = seasons[first : last + 1]
    peak = max(members, key=lambda s: abs(s.value))  # the earliest season wins a tie
    is_event = len(members) >= config.min_seasons
    return EnsoRun(
        phase=phase,
        first_season=members[0].label,
        last_season=members[-1].label,
        seasons=len(members),
        peak=peak.value,
        peak_season=peak.label,
        strength=strength_label(peak.value, config) if is_event else None,
        ongoing=last == len(seasons) - 1,
        starts_at_record_start=first == 0,
    )


def _overlaps(seasons: tuple[Season, ...], run: tuple[str, int, int], year: int) -> bool:
    return seasons[run[1]].year <= year <= seasons[run[2]].year


def compute_analysis(
    seasons: tuple[Season, ...],
    *,
    config: EnsoConfig,
    source_url: str,
    from_cache: bool,
    start_year: int | None = None,
    end_year: int | None = None,
    phase: str = "both",
    year: int | None = None,
) -> EnsoAnalysis:
    """Events that *start* in ``start_year..end_year`` (default: the whole record).

    Events are always detected on the full record and then selected, so a window never cuts an
    event short. ``year`` adds what the record says about that calendar year. Raises
    InsufficientEnsoDataError when the period or the year lies outside the record.
    """
    first_year, last_year = seasons[0].year, seasons[-1].year
    if (start_year is not None and start_year > last_year) or (
        end_year is not None and end_year < first_year
    ):
        raise InsufficientEnsoDataError(
            f"the record covers {seasons[0].label} to {seasons[-1].label}; the requested period "
            f"{start_year or first_year}-{end_year or last_year} lies outside it"
        )
    if year is not None and not first_year <= year <= last_year:
        raise InsufficientEnsoDataError(
            f"the record covers {seasons[0].label} to {seasons[-1].label}; year {year} lies "
            "outside it"
        )
    lo = max(start_year, first_year) if start_year is not None else first_year
    hi = min(end_year, last_year) if end_year is not None else last_year

    runs = _runs(seasons, config.threshold)
    event_runs = [r for r in runs if r[2] - r[1] + 1 >= config.min_seasons]
    short_runs = [r for r in runs if r[2] - r[1] + 1 < config.min_seasons]

    def wanted(run: tuple[str, int, int]) -> bool:
        return phase in ("both", run[0])

    selected = [
        _make_run(seasons, r, config)
        for r in event_runs
        if wanted(r) and lo <= seasons[r[1]].year <= hi
    ]
    last_run = short_runs[-1] if short_runs else None
    pending = (
        _make_run(seasons, last_run, config)
        if last_run is not None
        and last_run[2] == len(seasons) - 1
        and wanted(last_run)
        and hi == last_year
        else None
    )

    focus = None
    if year is not None:
        focus = EnsoFocus(
            year=year,
            values=tuple((SEASONS[s.slot], s.value) for s in seasons if s.year == year),
            events=tuple(
                _make_run(seasons, r, config) for r in event_runs if _overlaps(seasons, r, year)
            ),
            short_runs=tuple(
                _make_run(seasons, r, config) for r in short_runs if _overlaps(seasons, r, year)
            ),
        )

    return EnsoAnalysis(
        id="",
        index_name=INDEX_NAME,
        source_url=source_url,
        from_cache=from_cache,
        record_first=seasons[0].label,
        record_last=seasons[-1].label,
        first_year=lo,
        last_year=hi,
        phase=phase,
        threshold=config.threshold,
        min_seasons=config.min_seasons,
        strength_moderate=config.strength_moderate,
        strength_strong=config.strength_strong,
        strength_very_strong=config.strength_very_strong,
        events=tuple(selected),
        pending=pending,
        focus=focus,
        caveats=_caveats(seasons, config, selected, pending, start_year, end_year, year, lo, hi),
    )


def _caveats(
    seasons: tuple[Season, ...],
    config: EnsoConfig,
    events: list[EnsoRun],
    pending: EnsoRun | None,
    start_year: int | None,
    end_year: int | None,
    year: int | None,
    lo: int,
    hi: int,
) -> tuple[str, ...]:
    t, n = config.threshold, config.min_seasons
    notes = [
        "ENSO events come from NOAA CPC's Relative Oceanic Nino Index (RONI, 1991-2020 base "
        "period), a three-month running mean index of equatorial Pacific sea-surface "
        "temperature. It is not station or rainfall data and says nothing about impacts at a "
        "place.",
        f"An event follows NOAA's rule: the index at or beyond +/-{t:g} for at least {n} "
        f"consecutive overlapping seasons (a value of exactly {t:g} counts). Papers that use "
        "other indices (such as ONI, MEI or Nino 3), thresholds or periods list different "
        "events and dates.",
        f"Strength labels (weak from {t:g}, moderate from {config.strength_moderate:g}, strong "
        f"from {config.strength_strong:g}, very strong from {config.strength_very_strong:g}, by "
        "peak value) are a common convention, not part of NOAA's definition; a stricter version "
        "requires the class to be reached for 3 consecutive seasons and gives a lower label for "
        "some events.",
        "Recent seasons may be revised by NOAA, and the file does not state which "
        "sea-surface-temperature dataset version it uses.",
    ]
    for event in events:
        if event.starts_at_record_start:
            notes.append(
                f"The record begins inside the {PHASE_LABELS[event.phase]} event listed from "
                f"{event.first_season}: it may have started earlier, so its start, length and "
                "peak are incomplete."
            )
        if event.ongoing:
            notes.append(
                f"The record ends inside the {PHASE_LABELS[event.phase]} event that began in "
                f"{event.first_season}: its end, length and peak may still change."
            )
    if pending is not None:
        notes.append(
            f"{PHASE_LABELS[pending.phase]} conditions reach the threshold for "
            f"{pending.seasons} consecutive season(s) ({pending.first_season} to "
            f"{pending.last_season}), fewer than the {n} needed: this is not yet an event by "
            "NOAA's rule."
        )
    if start_year is not None and start_year < lo:
        notes.append(
            f"The requested start year {start_year} is before the record begins "
            f"({seasons[0].label}); events are listed from {lo}."
        )
    if end_year is not None and end_year > hi:
        notes.append(
            f"The requested end year {end_year} is after the record ends ({seasons[-1].label}); "
            f"events are listed up to {hi}."
        )
    if year is not None and year == seasons[-1].year and seasons[-1].slot < len(SEASONS) - 1:
        notes.append(f"{year} is incomplete in the record (last season {seasons[-1].label}).")
    return tuple(notes)
