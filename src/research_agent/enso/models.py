"""Frozen result types of the ENSO analysis."""

from __future__ import annotations

from dataclasses import dataclass

EL_NINO = "el_nino"
LA_NINA = "la_nina"
PHASES = (EL_NINO, LA_NINA)
PHASE_LABELS = {EL_NINO: "El Nino", LA_NINA: "La Nina"}


@dataclass(frozen=True)
class EnsoRun:
    """A maximal run of consecutive seasons at or beyond the threshold, all in one phase.

    It is an *event* when ``seasons`` reaches the minimum; ``strength`` is set only then.
    ``ongoing`` means the run reaches the last season of the record, so its end, length and peak
    may still change.
    """

    phase: str
    first_season: str  # e.g. "ASO 1997"
    last_season: str
    seasons: int
    peak: float  # signed index value of the largest |value|; the earliest season on a tie
    peak_season: str
    strength: str | None  # "weak" | "moderate" | "strong" | "very strong" for events
    ongoing: bool
    starts_at_record_start: bool  # the run may have begun before the record, so its start is cut


@dataclass(frozen=True)
class EnsoFocus:
    """What the record says about one calendar year (the seasons labelled with that year)."""

    year: int
    values: tuple[tuple[str, float], ...]  # (season code, index value) present in the record
    events: tuple[EnsoRun, ...]  # events with at least one season in this year
    short_runs: tuple[EnsoRun, ...]  # at-threshold runs too short to be events, same condition


@dataclass(frozen=True)
class EnsoAnalysis:
    id: str  # assigned by the agent loop (R1, R2...), empty until then
    index_name: str
    source_url: str
    from_cache: bool
    record_first: str  # first season in the file, e.g. "DJF 1950"
    record_last: str
    first_year: int  # years the events were selected for (clipped to the record)
    last_year: int
    phase: str  # "el_nino", "la_nina" or "both"
    threshold: float
    min_seasons: int
    strength_moderate: float
    strength_strong: float
    strength_very_strong: float
    events: tuple[EnsoRun, ...]  # chronological, starting in the selected years and phase
    pending: EnsoRun | None  # latest at-threshold run, reaching the end of the record, too short
    focus: EnsoFocus | None
    caveats: tuple[str, ...]
