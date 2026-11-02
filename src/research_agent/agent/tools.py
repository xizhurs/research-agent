"""The ``search_literature`` tool: schema, argument validation and result formatting.

Formatting is deterministic and enforces the retrieved-context budget, so the model never sees
more than ``max_context_tokens`` of retrieved text across a whole question.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import date
from typing import Any

from research_agent.agent.budget import estimate_tokens
from research_agent.llm import ToolSpec
from research_agent.rainfall import RainfallAnalysis
from research_agent.retrieval import SearchResult, Work

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


RAINFALL_TOOL_NAME = "analyze_extreme_rainfall"

RAINFALL_TOOL = ToolSpec(
    name=RAINFALL_TOOL_NAME,
    description=(
        "Quantify historical extreme rainfall in one city from ERA5 reanalysis (daily "
        "precipitation). Computes Rx1day (the largest one-day rainfall of each year), a "
        "climatological baseline, the largest events on record and, if event_date is given, how "
        "extreme that day was. Returns a numeric record with an ID such as R1. Use it for "
        "quantitative questions about past rainfall at a place; use search_literature for "
        "explanations and mechanisms. Put only the place name in 'city' and use 'country_code' "
        "(ISO alpha-2, e.g. FR) and 'region' (state or province) to disambiguate. Not for "
        "forecasts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name only, e.g. 'Zhengzhou'."},
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
                "description": "Optional. A day to assess, as YYYY-MM-DD.",
            },
        },
        "required": ["city"],
    },
)


class ToolArgumentError(ValueError):
    """The model supplied unusable tool arguments. The message is safe to show the model."""


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
class RainfallArgs:
    city: str
    country_code: str | None
    region: str | None
    start_year: int | None
    end_year: int | None
    event_date: date | None


def _optional_str(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolArgumentError(f"'{key}' must be a string")
    return value


def parse_rainfall_args(arguments: dict[str, Any]) -> RainfallArgs:
    city = arguments.get("city")
    if not isinstance(city, str) or not city.strip():
        raise ToolArgumentError("'city' is required and must be a non-empty string")
    raw_date = _optional_str(arguments, "event_date")
    try:
        event_date = date.fromisoformat(raw_date) if raw_date else None
    except ValueError:
        raise ToolArgumentError("'event_date' must be a date formatted YYYY-MM-DD") from None
    return RainfallArgs(
        city=city,
        country_code=_optional_str(arguments, "country_code"),
        region=_optional_str(arguments, "region"),
        start_year=_optional_int(arguments, "start_year"),
        end_year=_optional_int(arguments, "end_year"),
        event_date=event_date,
    )


def _mm(value: float) -> str:
    return f"{value:.1f} mm"


def format_rainfall(analysis: RainfallAnalysis) -> str:
    """Render one analysis for the model: compact, deterministic, numbers only from code."""
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
