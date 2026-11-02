"""Plain data for the rainfall analysis. No I/O, no LLM concepts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Location:
    """A geocoded place (one Open-Meteo geocoding result)."""

    id: int
    name: str
    admin1: str | None
    country: str | None
    country_code: str | None
    latitude: float
    longitude: float
    population: int | None
    timezone: str | None

    @property
    def label(self) -> str:
        parts = (self.name, self.admin1, self.country or self.country_code)
        return ", ".join(p for p in parts if p)


@dataclass(frozen=True)
class AnnualMax:
    """The largest daily precipitation total of one year (Rx1day)."""

    year: int
    date: str  # ISO date of the wettest day (the earliest one if tied)
    mm: float


@dataclass(frozen=True)
class Baseline:
    start_year: int
    end_year: int
    years: int  # valid years inside the baseline period
    mean_mm: float
    median_mm: float


@dataclass(frozen=True)
class EventContext:
    """How one requested day compares with the record. Rank 1 is the wettest."""

    date: str
    mm: float
    rank_among_days: int | None  # 1 + days in the valid record that were wetter
    days: int  # days with data in the valid record
    rank_among_annual_maxima: int | None  # where this value would sit among the yearly Rx1day
    annual_percentile: float | None  # % of valid annual maxima strictly below this value
    ratio_to_baseline_mean: float | None


@dataclass(frozen=True)
class Provenance:
    """What exactly was fetched, so the result can be reproduced and attributed."""

    dataset: str
    variable: str
    license: str
    request_url: str  # the archive request, with its parameters (contains no credentials)
    grid_latitude: float
    grid_longitude: float
    elevation_m: float | None
    timezone: str | None
    data_start: str  # first and last date in the response
    data_end: str
    from_cache: bool  # True only if the geocoding and the archive were both served from cache


@dataclass(frozen=True)
class RainfallAnalysis:
    """Result of one deterministic rainfall analysis; the unit of numerical evidence.

    ``id`` (``R1``, ``R2``...) is assigned by the agent loop when the result is shown to the
    model; it is the citation key, like a work ID for literature.
    """

    id: str
    location: Location
    provenance: Provenance
    start_year: int  # requested period
    end_year: int
    annual_maxima: tuple[AnnualMax, ...]  # valid years only, in year order
    excluded_years: tuple[tuple[int, str], ...]  # (year, reason)
    baseline: Baseline | None  # None when too few valid baseline years
    ranking_available: bool  # False when fewer than ``min_years`` valid years
    top_events: tuple[AnnualMax, ...]  # the largest annual maxima, wettest first
    event: EventContext | None
    caveats: tuple[str, ...]
    metric: ClassVar[str] = "wet"

    @property
    def valid_years(self) -> tuple[int, ...]:
        return tuple(a.year for a in self.annual_maxima)


@dataclass(frozen=True)
class AnnualDry:
    """Dry-condition numbers for one valid year."""

    year: int
    cdd_days: int  # longest run of dry days inside the calendar year (0 if no dry day)
    cdd_start: str | None  # first and last day of that run (the earliest run if tied)
    cdd_end: str | None
    total_mm: float  # precipitation summed over every day of the year


@dataclass(frozen=True)
class DrySpell:
    """A maximal run of consecutive dry days, not split at New Year."""

    start: str
    end: str
    days: int
    total_mm: float


@dataclass(frozen=True)
class DryBaseline:
    start_year: int
    end_year: int
    years: int  # valid years inside the baseline period
    cdd_mean: float
    cdd_median: float
    total_mean_mm: float
    total_median_mm: float


@dataclass(frozen=True)
class DryYearContext:
    """How one requested year compares with the record. Rank 1 is the driest."""

    year: int
    total_mm: float
    total_rank: int | None  # 1 + valid years that were drier; None when ranks are withheld
    percent_wetter: float | None  # % of valid years with a strictly higher total
    ratio_to_baseline_mean: float | None
    cdd_days: int


@dataclass(frozen=True)
class DryEventContext:
    """How one requested day relates to dry spells. Rank 1 is the longest spell."""

    date: str
    value_mm: float
    dry_day: bool
    spell: DrySpell | None  # the unbroken spell containing the day (None if not dry or unmeasured)
    spell_rank: int | None  # 1 + spells that were longer; None when ranks are withheld
    spells: int  # unbroken dry spells in the valid record


@dataclass(frozen=True)
class DryAnalysis:
    """Dry-extremes counterpart of ``RainfallAnalysis``; cited as ``R…`` in the same way."""

    id: str
    location: Location
    provenance: Provenance
    start_year: int  # requested period
    end_year: int
    dry_day_mm: float  # threshold: a day below this is dry
    annual: tuple[AnnualDry, ...]  # valid years only, in year order
    excluded_years: tuple[tuple[int, str], ...]  # (year, reason)
    baseline: DryBaseline | None
    ranking_available: bool
    longest_spells: tuple[DrySpell, ...]  # longest unbroken spells, longest first
    driest_years: tuple[AnnualDry, ...]  # lowest annual totals, driest first
    year_focus: DryYearContext | None
    event: DryEventContext | None
    caveats: tuple[str, ...]
    metric: ClassVar[str] = "dry"

    @property
    def valid_years(self) -> tuple[int, ...]:
        return tuple(a.year for a in self.annual)


RainfallRecord = RainfallAnalysis | DryAnalysis
