"""What each tab does when its button is pressed. No Gradio imports: plain values in and out.

Every handler returns Markdown or text for display and never raises for an expected failure
(bad input, provider or network error): the message goes into the status line instead. Each status
line starts with whether the language model was used, because that is what costs tokens.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

from research_agent.agent import InvalidQuestionError
from research_agent.agent.tools import format_rainfall
from research_agent.cli import _run_agent, format_trace
from research_agent.config import Config
from research_agent.credentials import CredentialError, Credentials
from research_agent.enso import EnsoAnalysis
from research_agent.llm import LLMError
from research_agent.rainfall import Archive, RainfallClient, RainfallError, RainfallRecord
from research_agent.retrieval import OpenAlexClient, RetrievalError, Work
from research_agent.ui.plots import (
    DEFAULT_WINDOW_DAYS,
    PlotError,
    daily_figure,
    default_plot_year,
    monthly_figure,
)

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

log = logging.getLogger(__name__)

NO_LLM = "**LLM used: no** (deterministic code only)"
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>~$])")
_SAFE_URL = re.compile(r"^https?://[^\s<>()]+$")


def md(text: object) -> str:
    """Escape text from outside (paper titles, model output) so it cannot become Markdown."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(text).split()))


def _error(exc: Exception) -> str:
    return f"**Error:** {md(exc)}"


def _whole(value: float | None, name: str) -> int | None:
    """A Gradio number box gives a float or None; years and counts must be whole numbers."""
    if value is None:
        return None
    if value != int(value):
        raise ValueError(f"{name} must be a whole number")
    return int(value)


def _text(value: str | None) -> str | None:
    return value.strip() or None if value else None


def _authors(work: Work) -> str:
    return ", ".join(work.authors[:3]) + (" et al." if len(work.authors) > 3 else "") or "n/a"


def _link(url: str | None) -> str:
    return f"<{url}>" if url and _SAFE_URL.match(url) else md(url or "")


# --- Search -------------------------------------------------------------------------------------


def search(
    cfg: Config,
    creds: Credentials,
    query: str,
    year_from: float | None,
    max_results: float | None,
    show_abstracts: bool,
) -> tuple[str, str]:
    """OpenAlex search. Returns ``(status, results)`` as Markdown."""
    key = creds.openalex_api_key
    try:
        with OpenAlexClient(
            cfg.openalex, cfg.limits, api_key=key.reveal() if key else None
        ) as client:
            result = client.search(
                query,
                year_from=_whole(year_from, "year from"),
                max_results=_whole(max_results, "results"),
            )
    except (RetrievalError, ValueError) as exc:
        return _error(exc), ""

    where = "cache" if result.from_cache else ("network, API key" if key else "network, no API key")
    status = [NO_LLM, f"{len(result.works)} of {result.total_count:,} matches ({where})"]
    if result.missing_abstract_count:
        status.append(f"{result.missing_abstract_count} without an abstract")
    if result.skipped_records:
        status.append(f"{result.skipped_records} record(s) could not be parsed")

    lines = []
    for w in result.works:
        venue = f" · *{md(w.venue)}*" if w.venue else ""
        cited = f" · cited by {w.cited_by_count}" if w.cited_by_count is not None else ""
        lines.append(
            f"- **[{w.id}]** {md(w.title or '(no title)')} ({w.year or 'n.d.'}) — "
            f"{md(_authors(w))}{venue}{cited} · {_link(w.doi or w.url)}"
        )
        if show_abstracts:
            lines.append(f"  > {md(w.abstract) if w.abstract else '(no abstract)'}")
    return " · ".join(status), "\n".join(lines) or "No works matched."


# --- Rainfall -----------------------------------------------------------------------------------


def _plot(
    cfg: Config,
    record: RainfallRecord,
    client: RainfallClient | None,
    year: int | None,
    window_days: int,
    archive: Archive | None = None,
    view: str = "daily",
) -> tuple[Figure | None, str]:
    """The plot for ``record``, or ``(None, why not)``.

    ``view`` is ``"daily"`` or ``"monthly"``. ``archive`` is the daily series; without it the
    client fetches it (from the cache, normally).
    """
    try:
        if archive is None:
            if client is None:
                raise ValueError("no daily series to plot")
            archive = client.fetch_archive(record.location, record.start_year, record.end_year)
        year = year if year is not None else default_plot_year(record)
        if view == "daily":
            return daily_figure(
                record, archive, year=year, window_days=window_days, config=cfg.rainfall
            )
        if view == "monthly":
            return monthly_figure(record, archive, year=year, config=cfg.rainfall)
        raise ValueError(f"unknown plot view {view!r}")
    except (PlotError, RainfallError, ValueError) as exc:
        return None, f"No plot: {md(exc)}"


def replot(
    cfg: Config,
    series: tuple[RainfallRecord, Archive] | None,
    plot_year: float | None,
    window_days: float,
    view: str,
) -> tuple[Figure | None, str]:
    """Redraw the Rainfall plot from the last analysis (no request) when the view changes."""
    if series is None:
        return None, ""
    try:
        year = _whole(plot_year, "plot year")
        window = _whole(window_days, "smoothing window") or DEFAULT_WINDOW_DAYS
    except ValueError as exc:
        return None, f"No plot: {md(exc)}"
    record, archive = series
    return _plot(cfg, record, None, year, window, archive, view)


def rainfall(
    cfg: Config,
    city: str,
    country_code: str,
    region: str,
    start_year: float | None,
    end_year: float | None,
    event_date: str,
    metric: str,
    event_year: float | None,
    plot_year: float | None,
    window_days: float,
    view: str = "daily",
) -> tuple[str, str, Figure | None, str, tuple[RainfallRecord, Archive] | None]:
    """ERA5 rainfall analysis.

    Returns ``(status, statistics, figure, plot note, series)``; ``series`` is the record and
    daily data for ``replot`` (None when the analysis failed).
    """
    try:
        day = date.fromisoformat(event_date.strip()) if event_date and event_date.strip() else None
    except ValueError:
        return "**Error:** event date must be formatted YYYY-MM-DD", "", None, "", None
    try:
        years = (
            _whole(start_year, "start year"),
            _whole(end_year, "end year"),
            _whole(event_year, "event year"),
            _whole(plot_year, "plot year"),
        )
        window = _whole(window_days, "smoothing window") or DEFAULT_WINDOW_DAYS
        with RainfallClient(cfg.rainfall) as client:
            record, archive = client.analyze(
                city,
                country_code=_text(country_code),
                region=_text(region),
                start_year=years[0],
                end_year=years[1],
                event_date=day,
                metric=metric,
                event_year=years[2],
            )
            figure, note = _plot(cfg, record, client, years[3], window, archive, view)
    except (RainfallError, ValueError) as exc:
        return _error(exc), "", None, "", None

    prov = record.provenance
    origin = "cache" if prov.from_cache else "network"
    status = (
        f"{NO_LLM} · {md(record.location.label)} · {origin} · "
        f"data {prov.data_start} to {prov.data_end}"
    )
    text = (
        f"{format_rainfall(replace(record, id='R1'))}\n\n"
        f"Source: {prov.dataset}. {prov.license}.\nRequest: {prov.request_url}"
    )
    return status, text, figure, note, (record, archive)


# --- Ask ----------------------------------------------------------------------------------------


def _format_answer(result) -> str:
    answer = result.answer
    out = ["### Answer"]
    for claim in answer.claims:
        cites = f" {md('[' + ', '.join(claim.sources) + ']')}" if claim.sources else ""
        label = "**evidence**" if claim.kind == "evidence" else "*interpretation*"
        out.append(f"- {label}: {md(claim.text)}{cites}")
    if answer.references:
        out.append("\n### References")
        for w in answer.references:
            venue = f" {md(w.venue)}." if w.venue else ""
            title = md(w.title or "(no title)")
            out.append(
                f"- **[{w.id}]** {md(_authors(w))} ({w.year or 'n.d.'}). {title}."
                f"{venue} {_link(w.doi or w.url)}"
            )
    if answer.data_references:
        out.append("\n### Data")
        for a in answer.data_references:
            if isinstance(a, EnsoAnalysis):
                out.append(
                    f"- **[{a.id}]** ENSO events, NOAA CPC {md(a.index_name)}, record "
                    f"{a.record_first} to {a.record_last}; events starting "
                    f"{a.first_year}-{a.last_year}. {_link(a.source_url)}"
                )
                continue
            p = a.provenance
            what = "dry spells and annual totals" if a.metric == "dry" else "Rx1day"
            out.append(
                f"- **[{a.id}]** {md(a.location.label)}: {what} {a.valid_years[0]}-"
                f"{a.valid_years[-1]}. {md(p.dataset)}; grid cell "
                f"({p.grid_latitude:.2f}, {p.grid_longitude:.2f}). {md(p.license)}."
            )
    out.append("\n### Limitations")
    out.extend(f"- {md(x)}" for x in answer.limitations or ("(none reported)",))
    return "\n".join(out)


def _ask_status(cfg: Config, stats) -> str:
    parts = [
        f"**LLM used: yes** ({md(cfg.llm.model)}) · {stats.iterations} model call(s), "
        f"{stats.total_tokens:,} tokens ({stats.input_tokens:,} in / {stats.output_tokens:,} out)",
        f"{stats.search_calls} search(es) ({stats.cached_searches} from cache)",
    ]
    if stats.rainfall_calls:
        parts.append(
            f"{stats.rainfall_calls} rainfall/ENSO analysis(es) "
            f"({stats.cached_rainfall} from cache)"
        )
    if stats.tool_trace:
        parts.append(f"tools by turn: {md(format_trace(stats.tool_trace))}")
    return " · ".join(parts)


def ask(
    cfg: Config, creds: Credentials, question: str, view: str = "daily"
) -> tuple[str, str, Figure | None, str, tuple[RainfallRecord, Archive] | None]:
    """Run the agent. Returns ``(status, answer, figure, plot note, series)``.

    The figure plots the first rainfall analysis the answer cites, in ``view`` (``"daily"`` or
    ``"monthly"``), from the same cached daily series; the language model never sees or draws it.
    ``series`` is that analysis and its daily data, for ``replot`` (None if nothing was plotted).
    """
    if not question.strip():
        return "**Error:** the question must not be empty", "", None, "", None
    try:
        creds.require_openai_key()
        result = _run_agent(cfg, creds, question)
    except (CredentialError, LLMError, RetrievalError, InvalidQuestionError) as exc:
        return f"**LLM used: no** (stopped before an answer) · {_error(exc)}", "", None, "", None

    body = _format_answer(result)
    records = [a for a in result.answer.data_references if not isinstance(a, EnsoAnalysis)]
    figure, note, series = None, "", None
    if records:
        record = records[0]
        try:
            with RainfallClient(cfg.rainfall) as client:
                archive = client.fetch_archive(record.location, record.start_year, record.end_year)
        except RainfallError as exc:
            note = f"No plot: {md(exc)}"
        else:
            series = (record, archive)
            figure, note = _plot(cfg, record, None, None, DEFAULT_WINDOW_DAYS, archive, view)
            if len(records) > 1:  # said here, not in the note, which a redraw replaces
                body += (
                    f"\n\n*Only {record.id} is plotted; "
                    f"{len(records) - 1} other analysis(es) not plotted.*"
                )
    return _ask_status(cfg, result.stats), body, figure, note, series
