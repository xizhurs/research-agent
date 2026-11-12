"""The ``search_literature`` tool: schema, argument validation and result formatting.

Formatting is deterministic and enforces the retrieved-context budget, so the model never sees
more than ``max_context_tokens`` of retrieved text across a whole question.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from research_agent.agent.budget import estimate_tokens
from research_agent.enso import PHASE_LABELS, EnsoAnalysis, EnsoRun
from research_agent.llm import ToolSpec
from research_agent.rainfall import (
    DryAnalysis,
    MonthlyAnalysis,
    RainfallAnalysis,
    RainfallRecord,
    TotalContext,
    TotalsBaseline,
    YearlyAnalysis,
)
from research_agent.retrieval import SearchResult, Work
from research_agent.temperature import (
    MeanContext,
    TempBaseline,
    TemperatureExtremes,
    TemperatureMonthly,
    TemperatureRecord,
    TemperatureYearly,
)

SEARCH_TOOL_NAME = "search_literature"

SEARCH_TOOL = ToolSpec(
    name=SEARCH_TOOL_NAME,
    description=(
        "Search scholarly literature (OpenAlex) by title and abstract. Returns titles, authors, "
        "year, venue and abstracts, each with a work ID such as W4388333579. Use short keyword "
        "queries. Results are matched on words in the title or abstract."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Short keyword query."},
            "year_from": {
                "type": "integer",
                "description": "Optional. Only works published in or after this year.",
            },
            "max_results": {
                "type": "integer",
                "description": "Optional. Number of results wanted; capped by the system.",
            },
        },
        "required": ["query"],
    },
)


RAINFALL_TOOL_NAME = "analyze_rainfall_extremes"

RAINFALL_TOOL = ToolSpec(
    name=RAINFALL_TOOL_NAME,
    description=(
        "Quantify past rainfall in one city from ERA5 reanalysis (daily precipitation), at one "
        "time scale. scale='yearly': annual totals, the wettest and driest years, how a year "
        "(event_year) compares with the 1991-2020 baseline; use it for general questions such as "
        "'how was the rainfall in the last 10 years' or 'how wet was 2023'. scale='monthly': "
        "monthly totals, the wettest months, how a month (event_year + event_month) or each "
        "month of a year (event_year) compares with the same calendar month. scale='daily': "
        "single days and spells; metric='wet' gives Rx1day (largest one-day rainfall of each "
        "year), the largest events and how extreme a day (event_date) was; metric='dry' gives "
        "consecutive dry days (CDD), the longest dry spells, the driest years and how dry a year "
        "or spell was. A dry spell is not a drought index. Returns a numeric record with an ID "
        "such as R1. Use search_literature for explanations and mechanisms. Put only the place "
        "name in 'city' and use 'country_code' (ISO alpha-2, e.g. FR) and 'region' (state or "
        "province) to disambiguate. Not for forecasts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name only, e.g. 'Zhengzhou'."},
            "scale": {
                "type": "string",
                "enum": ["yearly", "monthly", "daily"],
                "description": (
                    "'yearly' for annual amounts and general questions, 'monthly' for months or "
                    "seasons, 'daily' for heavy-rain days (metric 'wet') or dry spells (metric "
                    "'dry')."
                ),
            },
            "country_code": {
                "type": "string",
                "description": "Optional. ISO 3166-1 alpha-2 country code, e.g. 'CN'.",
            },
            "region": {
                "type": "string",
                "description": "Optional. State or province name, e.g. 'Henan'.",
            },
            "start_year": {"type": "integer", "description": "Optional. First year to analyse."},
            "end_year": {
                "type": "integer",
                "description": "Optional. Last year to analyse (default: last complete year).",
            },
            "event_date": {
                "type": "string",
                "description": (
                    "Optional, scale 'daily' only. A day to assess, as YYYY-MM-DD (metric 'dry': "
                    "a day inside the dry spell of interest)."
                ),
            },
            "metric": {
                "type": "string",
                "enum": ["wet", "dry"],
                "description": (
                    "Optional, scale 'daily' only. 'wet' (default) for heavy-rain days, 'dry' "
                    "for dry spells."
                ),
            },
            "event_year": {
                "type": "integer",
                "description": (
                    "Optional. A year to assess, e.g. 2022 (scale 'yearly' or 'monthly', or "
                    "scale 'daily' with metric 'dry')."
                ),
            },
            "event_month": {
                "type": "integer",
                "description": "Optional, scale 'monthly' with event_year. A month, 1-12.",
            },
        },
        "required": ["city", "scale"],
    },
)


def rainfall_tool(map_point: bool) -> ToolSpec:
    """The rainfall tool. With ``map_point`` it also takes ``use_map_point`` (``city`` optional).

    Without a map point it is ``RAINFALL_TOOL`` itself, unchanged.
    """
    return _with_map_point(RAINFALL_TOOL) if map_point else RAINFALL_TOOL


def _with_map_point(spec: ToolSpec) -> ToolSpec:
    """``spec`` with an extra ``use_map_point`` flag; ``city`` is then optional."""
    params = spec.parameters
    properties = {
        **params["properties"],
        "use_map_point": {
            "type": "boolean",
            "description": (
                "Optional. true to analyse the point the user selected on the map (the code "
                "supplies its coordinates); then leave city, country_code and region unset."
            ),
        },
    }
    return replace(spec, parameters={**params, "properties": properties, "required": ["scale"]})


TEMPERATURE_TOOL_NAME = "analyze_temperature"

TEMPERATURE_TOOL = ToolSpec(
    name=TEMPERATURE_TOOL_NAME,
    description=(
        "Quantify past air temperature in one city from ERA5 reanalysis (daily maximum, minimum "
        "and mean at 2 m), at one time scale. scale='yearly': annual mean temperature, the "
        "warmest and coolest years, how a year (event_year) compares with the 1991-2020 "
        "baseline; use it for general questions such as 'how warm were the last 10 years' or "
        "'was 2024 the warmest year'. scale='monthly': monthly means, how a month (event_year + "
        "event_month) or each month of a year (event_year) compares with the same calendar "
        "month, and the months furthest above or below normal. scale='daily': single days; "
        "metric='hot' gives TXx (hottest day of each year), the hottest days and how hot a day "
        "(event_date) was; metric='cold' gives TNn (coldest night of each year), the coldest "
        "nights and how cold a day was. A hot day is not a heatwave index. Returns a numeric "
        "record with an ID such as R1. Use analyze_rainfall_extremes for rain and "
        "search_literature for explanations. Put only the place name in 'city' and use "
        "'country_code' and 'region' to disambiguate. Not for forecasts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name only, e.g. 'Madrid'."},
            "scale": {
                "type": "string",
                "enum": ["yearly", "monthly", "daily"],
                "description": (
                    "'yearly' for annual means and general questions, 'monthly' for months or "
                    "seasons, 'daily' for hot days (metric 'hot') or cold nights (metric "
                    "'cold')."
                ),
            },
            "country_code": {
                "type": "string",
                "description": "Optional. ISO 3166-1 alpha-2 country code, e.g. 'ES'.",
            },
            "region": {
                "type": "string",
                "description": "Optional. State or province name, e.g. 'Madrid'.",
            },
            "start_year": {"type": "integer", "description": "Optional. First year to analyse."},
            "end_year": {
                "type": "integer",
                "description": "Optional. Last year to analyse (default: last complete year).",
            },
            "event_date": {
                "type": "string",
                "description": "Optional, scale 'daily' only. A day to assess, as YYYY-MM-DD.",
            },
            "metric": {
                "type": "string",
                "enum": ["hot", "cold"],
                "description": (
                    "Optional, scale 'daily' only. 'hot' (default) for the hottest days, 'cold' "
                    "for the coldest nights."
                ),
            },
            "event_year": {
                "type": "integer",
                "description": "Optional, scale 'yearly' or 'monthly'. A year to assess.",
            },
            "event_month": {
                "type": "integer",
                "description": "Optional, scale 'monthly' with event_year. A month, 1-12.",
            },
        },
        "required": ["city", "scale"],
    },
)


def temperature_tool(map_point: bool) -> ToolSpec:
    """The temperature tool; with ``map_point`` it also takes ``use_map_point``."""
    return _with_map_point(TEMPERATURE_TOOL) if map_point else TEMPERATURE_TOOL


ENSO_TOOL_NAME = "analyze_enso_events"

ENSO_TOOL = ToolSpec(
    name=ENSO_TOOL_NAME,
    description=(
        "List El Nino and La Nina events from NOAA CPC's Relative Oceanic Nino Index (RONI), the "
        "official ENSO index, over the whole record since 1950: first and last season, number "
        "of seasons, peak value and strength. An event follows NOAA's rule (index at or beyond "
        "+/-0.5 for at least 5 consecutive overlapping 3-month seasons). With 'year', also "
        "report what the record says about that calendar year. Returns a numeric record with an "
        "ID such as R1. Use it to find or date ENSO events, count them or compare their "
        "strength; use search_literature for causes, impacts, forecasting methods and for what "
        "a paper counted as an event. It is a global index: it takes no place and says nothing "
        "about rainfall at a location. Not for forecasts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "enum": ["both", "el_nino", "la_nina"],
                "description": "Optional. Which events to list (default 'both').",
            },
            "start_year": {
                "type": "integer",
                "description": "Optional. Only events that begin in or after this year.",
            },
            "end_year": {
                "type": "integer",
                "description": "Optional. Only events that begin in or before this year.",
            },
            "year": {
                "type": "integer",
                "description": (
                    "Optional. A calendar year to describe: its seasonal values and any event "
                    "overlapping it."
                ),
            },
        },
        "required": [],
    },
)


RECALL_TOOL_NAME = "recall_evidence"

RECALL_TOOL = ToolSpec(
    name=RECALL_TOOL_NAME,
    description=(
        "Show again, in full, works (W...) or analyses (R...) retrieved in earlier turns of this "
        "conversation, as listed under 'Stored evidence'. Only items shown in this turn, by a "
        "new tool call or by this tool, may be cited. It runs no new search or analysis."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs from the stored-evidence list, for example [\"R1\", \"W123\"].",
            }
        },
        "required": ["ids"],
    },
)


class ToolArgumentError(ValueError):
    """The model supplied unusable tool arguments. The message is safe to show the model."""


def parse_recall_args(arguments: dict[str, Any], max_ids: int) -> tuple[str, ...]:
    ids = arguments.get("ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise ToolArgumentError("'ids' is required and must be a non-empty list of strings")
    unique = tuple(dict.fromkeys(i.strip() for i in ids if i.strip()))
    if not unique:
        raise ToolArgumentError("'ids' must contain at least one ID")
    if len(unique) > max_ids:
        raise ToolArgumentError(f"at most {max_ids} IDs per call")
    return unique


def format_recalled(item: Work | RainfallRecord | EnsoAnalysis | TemperatureRecord) -> str:
    """The full text the model saw when the item was first retrieved."""
    if isinstance(item, Work):
        return _format_work(item)
    if isinstance(item, EnsoAnalysis):
        return format_enso(item)
    if isinstance(item, TemperatureYearly | TemperatureMonthly | TemperatureExtremes):
        return format_temperature(item)
    return format_rainfall(item)


@dataclass(frozen=True)
class SearchArgs:
    query: str
    year_from: int | None
    max_results: int | None


def _optional_int(arguments: dict[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolArgumentError(f"'{key}' must be an integer")
    return value


def parse_search_args(arguments: dict[str, Any]) -> SearchArgs:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolArgumentError("'query' is required and must be a non-empty string")
    return SearchArgs(
        query=query,
        year_from=_optional_int(arguments, "year_from"),
        max_results=_optional_int(arguments, "max_results"),
    )


def _format_work(work: Work) -> str:
    authors = ", ".join(work.authors[:3]) + (" et al." if len(work.authors) > 3 else "")
    meta = ", ".join(
        part
        for part in (
            authors,
            work.venue,
            str(work.year) if work.year else None,
            f"cited by {work.cited_by_count}" if work.cited_by_count is not None else None,
        )
        if part
    )
    head = f"[{work.id}] {work.title or '(no title)'}" + (f" ({meta})" if meta else "")
    return f"{head}\nAbstract: {work.abstract or '(not available)'}"


@dataclass(frozen=True)
class FormattedResults:
    text: str
    shown: tuple[Work, ...]  # works whose full record the model saw (citable evidence)
    omitted: int  # results left out because the context budget ran out
    context_tokens: int  # estimated tokens of retrieved content in ``text``


def format_results(
    result: SearchResult, context_left: int, already_seen: Collection[str]
) -> FormattedResults:
    """Render results for the model within ``context_left`` estimated tokens.

    Results keep OpenAlex's relevance order; once one does not fit, it and all later ones are
    omitted. Works already shown in an earlier search are listed by ID only.
    """
    lines = [f'Search "{result.query}": showing results of {result.total_count} matches.']
    used = estimate_tokens(lines[0])
    shown: list[Work] = []
    omitted = 0
    for index, work in enumerate(result.works):
        block = (
            f"[{work.id}] (already shown in an earlier search)"
            if work.id in already_seen
            else _format_work(work)
        )
        cost = estimate_tokens(block)
        if used + cost > context_left:
            omitted = len(result.works) - index
            break
        lines.append(block)
        used += cost
        if work.id not in already_seen:
            shown.append(work)
    if not result.works:
        lines.append("No results.")
    if omitted:
        lines.append(f"({omitted} more result(s) omitted: retrieved-context budget reached.)")
    return FormattedResults(
        text="\n\n".join(lines), shown=tuple(shown), omitted=omitted, context_tokens=used
    )


@dataclass(frozen=True)
class PlaceArgs:
    """Arguments of a rainfall or temperature call."""

    city: str | None  # None when use_map_point is set
    country_code: str | None
    region: str | None
    start_year: int | None
    end_year: int | None
    event_date: date | None
    metric: str
    event_year: int | None
    scale: str = "daily"
    event_month: int | None = None
    use_map_point: bool = False


RainfallArgs = PlaceArgs


def _optional_str(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolArgumentError(f"'{key}' must be a string")
    return value


def parse_rainfall_args(arguments: dict[str, Any], *, map_point: bool = False) -> PlaceArgs:
    """Validate a rainfall call. ``map_point``: a map point is selected, so ``use_map_point``
    may replace ``city``."""
    return _parse_place_args(arguments, map_point, ("wet", "dry"))


def parse_temperature_args(arguments: dict[str, Any], *, map_point: bool = False) -> PlaceArgs:
    """Validate a temperature call (metric 'hot' or 'cold'); otherwise as for rainfall."""
    return _parse_place_args(arguments, map_point, ("hot", "cold"))


def _parse_place_args(
    arguments: dict[str, Any], map_point: bool, metrics: tuple[str, str]
) -> PlaceArgs:
    """``metrics``: the two allowed metrics, the default first."""
    use_point = arguments.get("use_map_point", False)
    if not isinstance(use_point, bool):
        raise ToolArgumentError("'use_map_point' must be true or false")
    if use_point and not map_point:
        raise ToolArgumentError("no map point is selected; give a 'city' instead")
    city = arguments.get("city")
    if use_point:
        if any(arguments.get(k) is not None for k in ("city", "country_code", "region")):
            raise ToolArgumentError(
                "with 'use_map_point' leave 'city', 'country_code' and 'region' unset"
            )
    elif not isinstance(city, str) or not city.strip():
        raise ToolArgumentError("'city' is required and must be a non-empty string")
    raw_date = _optional_str(arguments, "event_date")
    try:
        event_date = date.fromisoformat(raw_date) if raw_date else None
    except ValueError:
        raise ToolArgumentError("'event_date' must be a date formatted YYYY-MM-DD") from None
    metric = _optional_str(arguments, "metric") or metrics[0]
    if metric not in metrics:
        raise ToolArgumentError(f"'metric' must be '{metrics[0]}' or '{metrics[1]}'")
    # Required in the schema; a call without it keeps the original (daily) behaviour.
    scale = _optional_str(arguments, "scale") or "daily"
    if scale not in ("daily", "monthly", "yearly"):
        raise ToolArgumentError("'scale' must be 'yearly', 'monthly' or 'daily'")
    return PlaceArgs(
        city=city,
        country_code=_optional_str(arguments, "country_code"),
        region=_optional_str(arguments, "region"),
        start_year=_optional_int(arguments, "start_year"),
        end_year=_optional_int(arguments, "end_year"),
        event_date=event_date,
        metric=metric,
        event_year=_optional_int(arguments, "event_year"),
        scale=scale,
        event_month=_optional_int(arguments, "event_month"),
        use_map_point=use_point,
    )


@dataclass(frozen=True)
class EnsoArgs:
    phase: str
    start_year: int | None
    end_year: int | None
    year: int | None


def parse_enso_args(arguments: dict[str, Any]) -> EnsoArgs:
    phase = _optional_str(arguments, "phase") or "both"
    if phase not in ("both", "el_nino", "la_nina"):
        raise ToolArgumentError("'phase' must be 'both', 'el_nino' or 'la_nina'")
    return EnsoArgs(
        phase=phase,
        start_year=_optional_int(arguments, "start_year"),
        end_year=_optional_int(arguments, "end_year"),
        year=_optional_int(arguments, "year"),
    )


def _mm(value: float) -> str:
    return f"{value:.1f} mm"


def format_rainfall(record: RainfallRecord) -> str:
    """Render one analysis for the model: compact, deterministic, numbers only from code."""
    if isinstance(record, YearlyAnalysis):
        return _format_yearly(record)
    if isinstance(record, MonthlyAnalysis):
        return _format_monthly(record)
    if isinstance(record, DryAnalysis):
        return _format_dry(record)
    return _format_wet(record)


MAX_YEARS_LISTED = 30  # a yearly result lists every year of a period up to this long
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _band(b: TotalsBaseline) -> str:
    return (
        f"mean {_mm(b.mean_mm)}, median {_mm(b.median_mm)}, {b.low_percentile}th-"
        f"{b.high_percentile}th percentile band {_mm(b.low_mm)} to {_mm(b.high_mm)}"
    )


def _total_context(c: TotalContext, what: str) -> str:
    if c.total_mm is None:
        return f"{what}: not assessed (not ended or too little data)"
    parts = [f"{what}: {_mm(c.total_mm)}"]
    if c.rank_wettest is not None and c.percent_drier is not None:
        parts.append(
            f"rank {c.rank_wettest} of {c.compared} (1 = wettest; {c.percent_drier:.0f}% were "
            "drier)"
        )
    else:
        parts.append("rank withheld: too few comparable totals")
    if c.ratio_to_baseline_mean is not None:
        parts.append(f"{c.ratio_to_baseline_mean:.2f}x the baseline mean")
    if c.band_position is not None:
        parts.append(f"{c.band_position} the baseline band")
    return ", ".join(parts)


def _header(record: RainfallRecord | TemperatureRecord, title: str) -> list[str]:
    loc, prov = record.location, record.provenance
    return [
        f"[{record.id}] {title} for {loc.label} ({loc.latitude:.3f}, {loc.longitude:.3f}).",
        f"Data: {prov.dataset}; grid cell centre ({prov.grid_latitude:.2f}, "
        f"{prov.grid_longitude:.2f}); {prov.variable}.",
    ]


def _format_yearly(a: YearlyAnalysis) -> str:
    lines = _header(a, "Annual precipitation totals")
    lines.append(
        f"Valid years: {len(a.totals)} ({a.valid_years[0]}-{a.valid_years[-1]}), "
        f"{len(a.excluded_years)} excluded."
    )
    base = a.baseline
    if base:
        lines.append(
            f"Baseline {base.start_year}-{base.end_year} annual total ({base.years} years): "
            f"{_band(base)}."
        )
    if len(a.totals) <= MAX_YEARS_LISTED:
        listed = []
        for t in a.totals:
            ratio = f" ({t.total_mm / base.mean_mm:.2f}x)" if base and base.mean_mm > 0 else ""
            listed.append(f"{t.year} {_mm(t.total_mm)}{ratio}")
        suffix = " (x = times the baseline mean)" if base else ""
        lines.append(f"Every year{suffix}: " + "; ".join(listed) + ".")
    lines.append(
        "Wettest years: "
        + "; ".join(f"{i}) {t.year} {_mm(t.total_mm)}" for i, t in enumerate(a.wettest, 1))
        + "."
    )
    lines.append(
        "Driest years: "
        + "; ".join(f"{i}) {t.year} {_mm(t.total_mm)}" for i, t in enumerate(a.driest, 1))
        + "."
    )
    if a.year_focus:
        lines.append(_total_context(a.year_focus, f"Year {a.year_focus.year}") + ".")
    lines.append("Caveats: " + " ".join(a.caveats))
    return "\n".join(lines)


def _format_monthly(a: MonthlyAnalysis) -> str:
    lines = _header(a, "Monthly precipitation totals")
    lines.append(
        f"Months with enough data: {a.months_used} ({a.years[0]}-{a.years[-1]}), "
        f"{a.months_missing} left out."
    )
    climate = [
        f"{MONTHS[i]} {_band(b)}" for i, b in enumerate(a.climate) if b is not None
    ]
    if climate:
        first = next(b for b in a.climate if b is not None)
        lines.append(
            f"Baseline {first.start_year}-{first.end_year} per calendar month: "
            + "; ".join(climate)
            + "."
        )
    lines.append(
        "Wettest months: "
        + "; ".join(
            f"{i}) {MONTHS[t.month - 1]} {t.year} {_mm(t.total_mm)}"
            for i, t in enumerate(a.wettest, 1)
            if t.month is not None
        )
        + "."
    )
    if a.month_focus:
        c = a.month_focus
        lines.append(_total_context(c, f"{MONTHS[c.month - 1]} {c.year}") + ".")
    for c in a.year_months:
        lines.append(_total_context(c, f"{MONTHS[c.month - 1]} {c.year}") + ".")
    lines.append("Caveats: " + " ".join(a.caveats))
    return "\n".join(lines)


def _format_wet(analysis: RainfallAnalysis) -> str:
    loc, prov = analysis.location, analysis.provenance
    valid = analysis.annual_maxima
    lines = [
        f"[{analysis.id}] Extreme rainfall (Rx1day = largest daily precipitation of the year) "
        f"for {loc.label} ({loc.latitude:.3f}, {loc.longitude:.3f}).",
        f"Data: {prov.dataset}; grid cell centre ({prov.grid_latitude:.2f}, "
        f"{prov.grid_longitude:.2f}); {prov.variable}.",
        f"Valid years: {len(valid)} ({valid[0].year}-{valid[-1].year}), "
        f"{len(analysis.excluded_years)} excluded.",
    ]
    base = analysis.baseline
    if base:
        lines.append(
            f"Baseline {base.start_year}-{base.end_year} Rx1day ({base.years} years): mean "
            f"{_mm(base.mean_mm)}, median {_mm(base.median_mm)}."
        )
    ratio_base = base.mean_mm if base else None
    top = []
    for rank, a in enumerate(analysis.top_events, start=1):
        times = f", {a.mm / ratio_base:.1f}x baseline mean" if ratio_base else ""
        top.append(f"{rank}) {a.date} {_mm(a.mm)}{times}")
    lines.append("Largest annual maxima: " + "; ".join(top) + ".")
    ev = analysis.event
    if ev:
        parts = [f"Event {ev.date}: {_mm(ev.mm)}"]
        if ev.rank_among_days is not None:
            parts.append(f"rank {ev.rank_among_days:,} of {ev.days:,} days in the record")
        if ev.rank_among_annual_maxima is not None and ev.annual_percentile is not None:
            parts.append(
                f"would rank {ev.rank_among_annual_maxima} of {len(valid)} annual maxima "
                f"(percentile {ev.annual_percentile:.1f})"
            )
        if ev.ratio_to_baseline_mean is not None:
            parts.append(f"{ev.ratio_to_baseline_mean:.1f}x the baseline mean Rx1day")
        if ev.rank_among_days is None:
            parts.append("ranks withheld: record too short")
        lines.append("; ".join(parts) + ".")
    lines.append("Caveats: " + " ".join(analysis.caveats))
    return "\n".join(lines)


def _format_dry(a: DryAnalysis) -> str:
    loc, prov = a.location, a.provenance
    lines = [
        f"[{a.id}] Extreme dry conditions (CDD = longest run of days under {a.dry_day_mm:g} mm; "
        f"annual total precipitation) for {loc.label} ({loc.latitude:.3f}, {loc.longitude:.3f}).",
        f"Data: {prov.dataset}; grid cell centre ({prov.grid_latitude:.2f}, "
        f"{prov.grid_longitude:.2f}); {prov.variable}.",
        f"Valid years: {len(a.annual)} ({a.valid_years[0]}-{a.valid_years[-1]}), "
        f"{len(a.excluded_years)} excluded.",
    ]
    base = a.baseline
    if base:
        lines.append(
            f"Baseline {base.start_year}-{base.end_year} ({base.years} years): yearly CDD mean "
            f"{base.cdd_mean:.1f} days, median {base.cdd_median:.1f} days; annual total mean "
            f"{_mm(base.total_mean_mm)}, median {_mm(base.total_median_mm)}."
        )
    spells = [
        f"{i}) {s.start} to {s.end}, {s.days} days, {_mm(s.total_mm)}"
        for i, s in enumerate(a.longest_spells, start=1)
    ]
    lines.append("Longest unbroken dry spells: " + "; ".join(spells) + ".")
    years = []
    for i, y in enumerate(a.driest_years, start=1):
        times = f" ({y.total_mm / base.total_mean_mm:.2f}x baseline mean)" if base else ""
        years.append(f"{i}) {y.year} {_mm(y.total_mm)}{times}")
    lines.append("Driest years (annual total): " + "; ".join(years) + ".")
    yf = a.year_focus
    if yf:
        parts = [f"Year {yf.year}: total {_mm(yf.total_mm)}"]
        if yf.total_rank is not None and yf.percent_wetter is not None:
            parts.append(
                f"rank {yf.total_rank} of {len(a.annual)} (1 = driest; {yf.percent_wetter:.1f}% "
                "of years were wetter)"
            )
        else:
            parts.append("rank withheld: record too short")
        if yf.ratio_to_baseline_mean is not None:
            parts.append(f"{yf.ratio_to_baseline_mean:.2f}x the baseline mean total")
        parts.append(f"longest dry run that year {yf.cdd_days} days")
        lines.append("; ".join(parts) + ".")
    ev = a.event
    if ev:
        if not ev.dry_day:
            lines.append(f"Event {ev.date}: {_mm(ev.value_mm)}, not a dry day.")
        elif ev.spell is None:
            lines.append(
                f"Event {ev.date}: {_mm(ev.value_mm)}, a dry day; its dry spell was not measured "
                "(excluded year)."
            )
        else:
            parts = [
                f"Event {ev.date}: {_mm(ev.value_mm)}, a dry day inside the unbroken dry spell "
                f"{ev.spell.start} to {ev.spell.end} ({ev.spell.days} days)"
            ]
            if ev.spell_rank is not None:
                parts.append(f"rank {ev.spell_rank} of {ev.spells} unbroken spells (1 = longest)")
            else:
                parts.append("rank withheld: record too short")
            lines.append("; ".join(parts) + ".")
    lines.append("Caveats: " + " ".join(a.caveats))
    return "\n".join(lines)


def _c(value: float) -> str:
    return f"{value:.1f} °C"


def _signed(value: float) -> str:
    return f"{value:+.1f} °C"


def format_temperature(record: TemperatureRecord) -> str:
    """Render one temperature analysis for the model: compact, numbers only from code."""
    if isinstance(record, TemperatureYearly):
        return _format_temp_yearly(record)
    if isinstance(record, TemperatureMonthly):
        return _format_temp_monthly(record)
    return _format_temp_extremes(record)


def _temp_band(b: TempBaseline) -> str:
    return (
        f"mean {_c(b.mean_c)}, median {_c(b.median_c)}, {b.low_percentile}th-"
        f"{b.high_percentile}th percentile band {_c(b.low_c)} to {_c(b.high_c)}"
    )


def _mean_context(c: MeanContext, what: str) -> str:
    if c.mean_c is None:
        return f"{what}: not assessed (not ended or too little data)"
    parts = [f"{what}: mean {_c(c.mean_c)}"]
    if c.rank_warmest is not None and c.percent_cooler is not None:
        parts.append(
            f"rank {c.rank_warmest} of {c.compared} (1 = warmest; {c.percent_cooler:.0f}% were "
            "cooler)"
        )
    else:
        parts.append("rank withheld: too few comparable means")
    if c.anomaly_c is not None:
        parts.append(f"{_signed(c.anomaly_c)} against the baseline mean")
    if c.band_position is not None:
        parts.append(f"{c.band_position} the baseline band")
    return ", ".join(parts)


def _format_temp_yearly(a: TemperatureYearly) -> str:
    lines = _header(a, "Annual mean temperature")
    lines.append(
        f"Valid years: {len(a.means)} ({a.valid_years[0]}-{a.valid_years[-1]}), "
        f"{len(a.excluded_years)} excluded."
    )
    base = a.baseline
    if base:
        lines.append(
            f"Baseline {base.start_year}-{base.end_year} annual mean ({base.years} years): "
            f"{_temp_band(base)}."
        )
    if len(a.means) <= MAX_YEARS_LISTED:
        listed = []
        for m in a.means:
            anomaly = f" ({_signed(m.mean_c - base.mean_c)})" if base else ""
            listed.append(f"{m.year} {_c(m.mean_c)}{anomaly}")
        suffix = " (in brackets: difference from the baseline mean)" if base else ""
        lines.append(f"Every year{suffix}: " + "; ".join(listed) + ".")
    lines.append(
        "Warmest years: "
        + "; ".join(f"{i}) {m.year} {_c(m.mean_c)}" for i, m in enumerate(a.warmest, 1))
        + "."
    )
    lines.append(
        "Coolest years: "
        + "; ".join(f"{i}) {m.year} {_c(m.mean_c)}" for i, m in enumerate(a.coolest, 1))
        + "."
    )
    if a.year_focus:
        lines.append(_mean_context(a.year_focus, f"Year {a.year_focus.year}") + ".")
    lines.append("Caveats: " + " ".join(a.caveats))
    return "\n".join(lines)


def _format_temp_monthly(a: TemperatureMonthly) -> str:
    lines = _header(a, "Monthly mean temperature")
    lines.append(
        f"Months with enough data: {a.months_used} ({a.years[0]}-{a.years[-1]}), "
        f"{a.months_missing} left out."
    )
    climate = [f"{MONTHS[i]} {_temp_band(b)}" for i, b in enumerate(a.climate) if b is not None]
    if climate:
        first = next(b for b in a.climate if b is not None)
        lines.append(
            f"Baseline {first.start_year}-{first.end_year} per calendar month: "
            + "; ".join(climate)
            + "."
        )
    for title, months in (
        ("Months furthest above their calendar month's baseline mean", a.warm_anomalies),
        ("Months furthest below their calendar month's baseline mean", a.cool_anomalies),
    ):
        if months:
            lines.append(
                f"{title}: "
                + "; ".join(
                    f"{i}) {MONTHS[m.month - 1]} {m.year} {_c(m.mean_c)} ({_signed(m.anomaly_c)})"
                    for i, m in enumerate(months, 1)
                )
                + "."
            )
    contexts = (a.month_focus,) if a.month_focus else a.year_months
    for c in contexts:
        lines.append(_mean_context(c, f"{MONTHS[(c.month or 1) - 1]} {c.year}") + ".")
    lines.append("Caveats: " + " ".join(a.caveats))
    return "\n".join(lines)


def _format_temp_extremes(a: TemperatureExtremes) -> str:
    hot = a.metric == "hot"
    name, daily, superlative = (
        ("TXx", "daily maximum", "hottest") if hot else ("TNn", "daily minimum", "coldest")
    )
    lines = _header(
        a, f"Temperature extremes ({name} = {'highest' if hot else 'lowest'} {daily} of the year)"
    )
    lines.append(
        f"Valid years: {len(a.annual)} ({a.valid_years[0]}-{a.valid_years[-1]}), "
        f"{len(a.excluded_years)} excluded."
    )
    base = a.baseline
    if base:
        lines.append(
            f"Baseline {base.start_year}-{base.end_year} {name} ({base.years} years): mean "
            f"{_c(base.mean_c)}, median {_c(base.median_c)}."
        )
    lines.append(
        f"Most extreme years ({superlative} first): "
        + "; ".join(f"{i}) {e.date} {_c(e.value_c)}" for i, e in enumerate(a.top, 1))
        + "."
    )
    ev = a.event
    if ev:
        parts = [f"Event {ev.date}: {daily} {_c(ev.value_c)}"]
        if ev.rank_among_days is not None:
            parts.append(
                f"rank {ev.rank_among_days:,} of {ev.days:,} days in the record "
                f"(1 = {superlative})"
            )
        if ev.rank_among_annual_extremes is not None and ev.annual_percentile is not None:
            parts.append(
                f"would rank {ev.rank_among_annual_extremes} of {len(a.annual)} annual {name} "
                f"values ({ev.annual_percentile:.1f}% of them were less extreme)"
            )
        if ev.difference_from_baseline_extreme_c is not None:
            parts.append(
                f"{_signed(ev.difference_from_baseline_extreme_c)} against the baseline mean "
                f"{name}"
            )
        if ev.day_of_year_mean_c is not None and ev.difference_from_day_of_year_c is not None:
            parts.append(
                f"{_signed(ev.difference_from_day_of_year_c)} against the baseline mean "
                f"{daily} for that calendar day ({_c(ev.day_of_year_mean_c)})"
            )
        if ev.rank_among_days is None:
            parts.append("ranks withheld: record too short")
        lines.append("; ".join(parts) + ".")
    lines.append("Caveats: " + " ".join(a.caveats))
    return "\n".join(lines)


def _enso_run(run: EnsoRun) -> str:
    text = (
        f"{PHASE_LABELS[run.phase]} {run.first_season} to {run.last_season} "
        f"({run.seasons} season{'s' if run.seasons != 1 else ''}), peak {run.peak:+.2f} "
        f"({run.peak_season})"
    )
    if run.strength:
        text += f", {run.strength}"
    if run.starts_at_record_start:
        text += ", the record starts inside it so its start may be cut"
    if run.ongoing:
        text += ", ongoing: the record ends inside it"
    return text


def format_enso(a: EnsoAnalysis) -> str:
    """Render one ENSO analysis for the model: compact, deterministic, numbers only from code."""
    t = a.threshold
    phases = [p for p in ("el_nino", "la_nina") if a.phase in ("both", p)]
    counts = ", ".join(
        f"{sum(e.phase == p for e in a.events)} {PHASE_LABELS[p]}" for p in phases
    )
    lines = [
        f"[{a.id}] ENSO events from NOAA CPC {a.index_name}; record {a.record_first} to "
        f"{a.record_last}; events starting {a.first_year}-{a.last_year}.",
        f"Rule: index at or above +{t:g} (El Nino) or at or below -{t:g} (La Nina) for at least "
        f"{a.min_seasons} consecutive overlapping 3-month seasons. Strength by peak value "
        f"(convention): weak, moderate from {a.strength_moderate:g}, strong from "
        f"{a.strength_strong:g}, very strong from {a.strength_very_strong:g}.",
        f"Source: {a.source_url}",
        f"Events: {counts}.",
    ]
    lines.extend(f"- {_enso_run(e)}" for e in a.events)
    if a.pending:
        lines.append(
            f"Not yet an event: {_enso_run(a.pending)}; fewer than {a.min_seasons} seasons so far."
        )
    f = a.focus
    if f:
        values = ", ".join(f"{season} {value:+.2f}" for season, value in f.values)
        lines.append(f"Year {f.year} seasonal values: {values}.")
        if f.events:
            overlapping = "; ".join(_enso_run(e) for e in f.events)
            lines.append(f"Events overlapping {f.year}: {overlapping}")
        else:
            lines.append(f"No El Nino or La Nina event overlaps {f.year}.")
        if f.short_runs:
            lines.append(
                f"At-threshold runs in {f.year} too short to be events: "
                + "; ".join(_enso_run(r) for r in f.short_runs)
            )
    lines.append("Caveats: " + " ".join(a.caveats))
    return "\n".join(lines)
