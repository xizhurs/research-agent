"""Plain data for the temperature analysis. No I/O, no LLM concepts.

Places and provenance are the rainfall package's ``Location`` and ``Provenance``: both tools use
the same geocoder and the same ERA5 archive. All temperatures are in degrees Celsius; anomalies
are differences in degrees Celsius, never ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from research_agent.rainfall.models import Location, Provenance


@dataclass(frozen=True)
class TempBaseline:
    """Typical values over the baseline years: mean, median and a percentile band."""

    start_year: int
    end_year: int
    years: int  # values in the baseline period that were used
    mean_c: float
    median_c: float
    low_percentile: int
    high_percentile: int
    low_c: float
    high_c: float


@dataclass(frozen=True)
class PeriodMean:
    """Mean temperature of one year (``month`` None) or one month."""

    year: int
    month: int | None
    mean_c: float


@dataclass(frozen=True)
class MeanContext:
    """How one year's or one month's mean compares. Rank 1 is the warmest."""

    year: int
    month: int | None
    mean_c: float | None  # None: not enough daily data for this period
    rank_warmest: int | None  # None when ranks are withheld
    compared: int  # means it was ranked against (same calendar month for a month)
    percent_cooler: float | None  # % of compared means strictly lower
    anomaly_c: float | None  # difference from the baseline mean
    band_position: str | None  # "below", "within" or "above" the baseline band; None without one


@dataclass(frozen=True)
class MonthAnomaly:
    """A month's mean and its difference from the baseline mean of the same calendar month."""

    year: int
    month: int
    mean_c: float
    anomaly_c: float


@dataclass(frozen=True)
class TemperatureYearly:
    """Annual mean temperature: warmest and coolest years, an optional year in focus."""

    id: str
    location: Location
    provenance: Provenance
    start_year: int  # requested period
    end_year: int
    baseline_from_year: int  # the series starts here so the baseline is always available
    means: tuple[PeriodMean, ...]  # valid years of the requested period, in year order
    excluded_years: tuple[tuple[int, str], ...]
    baseline: TempBaseline | None
    ranking_available: bool
    warmest: tuple[PeriodMean, ...]
    coolest: tuple[PeriodMean, ...]
    year_focus: MeanContext | None
    caveats: tuple[str, ...]
    metric: ClassVar[str] = "mean"
    scale: ClassVar[str] = "yearly"
    summary: ClassVar[str] = "annual mean temperature"

    @property
    def series_start_year(self) -> int:
        return self.baseline_from_year

    @property
    def valid_years(self) -> tuple[int, ...]:
        return tuple(m.year for m in self.means)


@dataclass(frozen=True)
class TemperatureMonthly:
    """Monthly mean temperature against the same calendar month in other years."""

    id: str
    location: Location
    provenance: Provenance
    start_year: int
    end_year: int
    baseline_from_year: int
    years: tuple[int, ...]  # years of the period with at least one month of enough data
    months_used: int
    months_missing: int  # ended months of the period left out for lack of data
    climate: tuple[TempBaseline | None, ...]  # per calendar month, January first
    warm_anomalies: tuple[MonthAnomaly, ...]  # largest positive anomalies first
    cool_anomalies: tuple[MonthAnomaly, ...]  # largest negative anomalies first
    month_focus: MeanContext | None
    year_months: tuple[MeanContext, ...]  # the 12 months of a year in focus (no month given)
    caveats: tuple[str, ...]
    metric: ClassVar[str] = "mean"
    scale: ClassVar[str] = "monthly"
    summary: ClassVar[str] = "monthly mean temperature"

    @property
    def series_start_year(self) -> int:
        return self.baseline_from_year

    @property
    def valid_years(self) -> tuple[int, ...]:
        return self.years


@dataclass(frozen=True)
class AnnualExtreme:
    """The highest daily maximum (hot) or lowest daily minimum (cold) of one year."""

    year: int
    date: str  # ISO date (the earliest one if tied)
    value_c: float


@dataclass(frozen=True)
class ExtremeBaseline:
    start_year: int
    end_year: int
    years: int  # valid years inside the baseline period
    mean_c: float
    median_c: float


@dataclass(frozen=True)
class DayContext:
    """How one requested day compares. Rank 1 is the hottest (hot) or coldest (cold)."""

    date: str
    value_c: float  # the day's maximum (hot) or minimum (cold)
    rank_among_days: int | None  # 1 + days in the valid record that were more extreme
    days: int  # days with data in the valid record
    rank_among_annual_extremes: int | None
    annual_percentile: float | None  # % of annual extremes that were less extreme
    difference_from_baseline_extreme_c: float | None  # value minus the baseline mean TXx/TNn
    day_of_year_mean_c: float | None  # baseline mean for this calendar day (smoothed)
    difference_from_day_of_year_c: float | None


@dataclass(frozen=True)
class TemperatureExtremes:
    """TXx (``metric`` "hot") or TNn ("cold") per year, the most extreme days, a day in focus."""

    id: str
    location: Location
    provenance: Provenance
    metric: str  # "hot" or "cold"
    start_year: int
    end_year: int
    baseline_from_year: int
    annual: tuple[AnnualExtreme, ...]  # valid years of the requested period, in year order
    excluded_years: tuple[tuple[int, str], ...]
    baseline: ExtremeBaseline | None
    ranking_available: bool
    top: tuple[AnnualExtreme, ...]  # most extreme annual values first
    event: DayContext | None
    caveats: tuple[str, ...]
    scale: ClassVar[str] = "daily"

    @property
    def summary(self) -> str:
        if self.metric == "hot":
            return "TXx (hottest day per year)"
        return "TNn (coldest night per year)"

    @property
    def series_start_year(self) -> int:
        return self.baseline_from_year

    @property
    def valid_years(self) -> tuple[int, ...]:
        return tuple(a.year for a in self.annual)


TemperatureRecord = TemperatureYearly | TemperatureMonthly | TemperatureExtremes
