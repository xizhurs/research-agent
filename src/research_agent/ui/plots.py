"""The precipitation plots: one year of ERA5 values against a baseline, daily or monthly.

Daily: bars against a smoothed day-of-year mean. Monthly: bars of monthly totals against the
10th-90th percentile band of the baseline years (see ``rainfall/climatology.py``). Plotly is
imported inside the figure functions so the rest of the interface (and its tests) work without
the optional ``ui`` extra.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING

from research_agent.config import RainfallConfig
from research_agent.rainfall import Archive, RainfallRecord
from research_agent.rainfall.climatology import daily_baseline, monthly_baseline, monthly_totals

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

DEFAULT_WINDOW_DAYS = 15
BAR_COLOR = "#4c78a8"
BASELINE_COLOR = "#d95f02"
BAND_COLOR = "rgba(217, 95, 2, 0.25)"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


class PlotError(ValueError):
    """The requested plot cannot be drawn from the data at hand."""


def default_plot_year(record: RainfallRecord) -> int:
    """The year a question is about: the event's year, else the compared year, else the latest."""
    if record.event is not None:
        return int(record.event.date[:4])
    year_focus = getattr(record, "year_focus", None)  # dry analyses only
    if year_focus is not None:
        return year_focus.year
    return record.valid_years[-1]


def daily_figure(
    record: RainfallRecord,
    archive: Archive,
    *,
    year: int,
    window_days: int,
    config: RainfallConfig,
    today: Callable[[], date] = date.today,
) -> tuple[Figure, str]:
    """Return ``(figure, note)``: daily mm for ``year`` as bars, the baseline as a line.

    The baseline is the mean for each calendar day over the valid years of the configured
    baseline period that the archive covers, smoothed over ``window_days``. When too few such
    years are available the bars are drawn alone and the note says why. Raises ``PlotError`` if
    the archive has no data for ``year``.
    """
    import plotly.graph_objects as go

    days = [(d, v) for d, v in zip(archive.dates, archive.values, strict=True) if d.year == year]
    if not days:
        raise PlotError(
            f"the daily series has no data for {year}; it covers "
            f"{archive.dates[0].isoformat()} to {archive.dates[-1].isoformat()}"
        )
    baseline = daily_baseline(
        archive.dates,
        archive.values,
        start_year=config.baseline_start_year,
        end_year=config.baseline_end_year,
        last_complete_year=today().year - 1,
        min_years=config.min_years,
        min_year_coverage=config.min_year_coverage,
        window_days=window_days,
    )

    fig = go.Figure()
    fig.add_bar(
        x=[d for d, _ in days],
        y=[v for _, v in days],
        name=f"{year} daily total",
        marker_color=BAR_COLOR,
        hovertemplate="%{x|%d %b %Y}: %{y:.1f} mm<extra></extra>",
    )
    if baseline is None:
        note = (
            f"No baseline line: fewer than {config.min_years} valid years of "
            f"{config.baseline_start_year}-{config.baseline_end_year} are in the daily series "
            f"({archive.dates[0].year}-{archive.dates[-1].year}). Analyse a period that covers the "
            "baseline years to see it."
        )
    else:
        fig.add_scatter(
            x=[d for d, _ in days],
            y=[baseline.at(d) for d, _ in days],
            mode="lines",
            name=f"baseline ({baseline.window_days}-day smoothed mean)",
            line={"color": BASELINE_COLOR, "width": 2},
            hovertemplate="baseline %{y:.2f} mm/day<extra></extra>",
        )
        note = (
            f"Baseline: mean precipitation of each calendar day over {len(baseline.years)} valid "
            f"years of {baseline.start_year}-{baseline.end_year}, smoothed with a "
            f"{baseline.window_days}-day centred window. It is a mean, which a few heavy days "
            "dominate: neither a typical day nor an extreme threshold. Source: ERA5 reanalysis "
            "grid cell, not a rain gauge."
        )
    if record.event is not None and record.event.date[:4] == str(year):
        fig.add_shape(
            type="line",
            x0=record.event.date,
            x1=record.event.date,
            y0=0,
            y1=1,
            yref="paper",
            line={"color": "gray", "dash": "dot"},
        )
        fig.add_annotation(
            x=record.event.date, y=1, yref="paper", text="event", showarrow=False, yanchor="bottom"
        )
    fig.update_layout(
        title=f"{record.location.label}: daily precipitation in {year}",
        yaxis_title="mm per day",
        template="plotly_white",
        height=420,
        margin={"l": 50, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.15},
        bargap=0.1,
    )
    return fig, note


def monthly_figure(
    record: RainfallRecord,
    archive: Archive,
    *,
    year: int,
    config: RainfallConfig,
    today: Callable[[], date] = date.today,
) -> tuple[Figure, str]:
    """Return ``(figure, note)``: monthly totals of ``year`` against the baseline percentile band.

    The band covers the 10th to 90th percentile of each month's total over the valid years of the
    configured baseline period that the archive covers. Months without enough daily data are left
    blank, and months without enough baseline totals have no band. Raises ``PlotError`` if no
    month of ``year`` has enough data.
    """
    import plotly.graph_objects as go

    totals = monthly_totals(
        archive.dates, archive.values, year=year, min_year_coverage=config.min_year_coverage
    )
    if all(t is None for t in totals):
        raise PlotError(
            f"the daily series has no month with enough data in {year}; it covers "
            f"{archive.dates[0].isoformat()} to {archive.dates[-1].isoformat()}"
        )
    band = monthly_baseline(
        archive.dates,
        archive.values,
        start_year=config.baseline_start_year,
        end_year=config.baseline_end_year,
        last_complete_year=today().year - 1,
        min_years=config.min_years,
        min_year_coverage=config.min_year_coverage,
    )

    fig = go.Figure()
    note = ""
    if band is None:
        note = (
            f"No baseline band: fewer than {config.min_years} valid years of "
            f"{config.baseline_start_year}-{config.baseline_end_year} are in the daily series "
            f"({archive.dates[0].year}-{archive.dates[-1].year}). Analyse a period that covers the "
            "baseline years to see it."
        )
    else:
        months = [i for i in range(12) if band.low_mm[i] is not None]
        fig.add_bar(
            x=[MONTHS[i] for i in months],
            y=[band.high_mm[i] - band.low_mm[i] for i in months],
            base=[band.low_mm[i] for i in months],
            customdata=[(band.low_mm[i], band.high_mm[i]) for i in months],
            width=0.8,
            name=f"baseline, percentile {band.low_percentile} to {band.high_percentile}",
            marker_color=BAND_COLOR,
            hovertemplate="baseline range: %{customdata[0]:.0f} to %{customdata[1]:.0f} mm"
            "<extra></extra>",
        )
        outside = band.low_percentile + 100 - band.high_percentile
        note = (
            f"Band: percentile {band.low_percentile} to {band.high_percentile} of each month's "
            f"total over {len(band.years)} valid years of {band.start_year}-{band.end_year}; "
            f"about {outside}% of years fall outside it by construction."
        )
        if len(months) < 12:
            note += f" {12 - len(months)} month(s) have too few baseline totals for a band."
    fig.add_bar(
        x=list(MONTHS),
        y=list(totals),
        width=0.4,
        name=f"{year} monthly total",
        marker_color=BAR_COLOR,
        hovertemplate="%{x}: %{y:.0f} mm<extra></extra>",
    )
    if any(t is None for t in totals):
        note += " Months without enough daily data are left blank."
    note += " Source: ERA5 reanalysis grid cell, not a rain gauge."
    fig.update_layout(
        title=f"{record.location.label}: monthly precipitation in {year}",
        yaxis_title="mm per month",
        xaxis={"categoryorder": "array", "categoryarray": list(MONTHS)},
        barmode="overlay",
        template="plotly_white",
        height=420,
        margin={"l": 50, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.15},
    )
    return fig, note.strip()
