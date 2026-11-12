"""The temperature plots: ERA5 values against a baseline, daily, monthly or yearly.

Daily: one year's daily maximum (hot), minimum (cold) or mean (other analyses) as a line against
the smoothed calendar-day mean of the baseline years. Monthly: one year's monthly means against
the 10th-90th percentile band of the baseline years. Yearly: the annual means of the analysed
period against the band and mean of the baseline years' means. Lines and markers, not bars: a
bar from 0 degrees C would suggest a meaning it does not have. The means come from the same
functions as the analysis (``temperature/analysis.py``), so the plot and the text agree. Plotly
is imported inside the functions, as in ``plots.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING

from research_agent.config import RainfallConfig
from research_agent.rainfall.analysis import classify_years
from research_agent.rainfall.climatology import daily_baseline
from research_agent.temperature import TemperatureArchive, TemperatureRecord
from research_agent.temperature.analysis import (
    baseline_of,
    monthly_means,
    year_counts_and_means,
)
from research_agent.ui.plots import (
    BAND_COLOR,
    BASELINE_COLOR,
    MONTHS,
    PlotError,
)

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

LINE_COLOR = "#c0392b"
SOURCE_NOTE = " Source: ERA5 reanalysis 2 m air temperature of a grid cell, not a station."
_SERIES_NAME = {"hot": "daily maximum", "cold": "daily minimum"}


def _layout(fig: Figure, title: str, y_title: str) -> None:
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        template="plotly_white",
        height=420,
        margin={"l": 50, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.15},
    )


def temperature_daily_figure(
    record: TemperatureRecord,
    archive: TemperatureArchive,
    *,
    year: int,
    window_days: int,
    config: RainfallConfig,
    today: Callable[[], date] = date.today,
) -> tuple[Figure, str]:
    """Return ``(figure, note)``: one year of daily values and the calendar-day baseline.

    The series is the daily maximum for a hot analysis, the minimum for a cold one and the mean
    otherwise. Raises ``PlotError`` if the archive has no data for ``year``.
    """
    import plotly.graph_objects as go

    metric = getattr(record, "metric", "mean")
    values = archive.series(metric)
    what = _SERIES_NAME.get(metric, "daily mean")
    days = [(d, v) for d, v in zip(archive.dates, values, strict=True) if d.year == year]
    if not days:
        raise PlotError(
            f"the daily series has no data for {year}; it covers "
            f"{archive.dates[0].isoformat()} to {archive.dates[-1].isoformat()}"
        )
    baseline = daily_baseline(
        archive.dates,
        values,
        start_year=config.baseline_start_year,
        end_year=config.baseline_end_year,
        last_complete_year=today().year - 1,
        min_years=config.min_years,
        min_year_coverage=config.min_year_coverage,
        window_days=window_days,
    )
    fig = go.Figure()
    if baseline is None:
        note = (
            f"No baseline line: fewer than {config.min_years} valid years of "
            f"{config.baseline_start_year}-{config.baseline_end_year} are in the daily series."
        )
    else:
        fig.add_scatter(
            x=[d for d, _ in days],
            y=[baseline.at(d) for d, _ in days],
            mode="lines",
            name=f"baseline ({baseline.window_days}-day smoothed mean)",
            line={"color": BASELINE_COLOR, "width": 2, "dash": "dash"},
            hovertemplate="baseline %{y:.1f} °C<extra></extra>",
        )
        note = (
            f"Baseline: mean {what} of each calendar day over {len(baseline.years)} valid years "
            f"of {baseline.start_year}-{baseline.end_year}, smoothed with a "
            f"{baseline.window_days}-day centred window: typical weather, not an extreme "
            "threshold."
        )
    fig.add_scatter(
        x=[d for d, _ in days],
        y=[v for _, v in days],
        mode="lines",
        name=f"{year} {what}",
        line={"color": LINE_COLOR, "width": 1.5},
        hovertemplate="%{x|%d %b %Y}: %{y:.1f} °C<extra></extra>",
    )
    event = getattr(record, "event", None)
    if event is not None and event.date[:4] == str(year):
        fig.add_shape(
            type="line",
            x0=event.date,
            x1=event.date,
            y0=0,
            y1=1,
            yref="paper",
            line={"color": "gray", "dash": "dot"},
        )
        fig.add_annotation(
            x=event.date, y=1, yref="paper", text="event", showarrow=False, yanchor="bottom"
        )
    _layout(fig, f"{record.location.label}: {what} temperature in {year}", "°C")
    return fig, note + SOURCE_NOTE


def temperature_monthly_figure(
    record: TemperatureRecord,
    archive: TemperatureArchive,
    *,
    year: int,
    config: RainfallConfig,
    today: Callable[[], date] = date.today,
) -> tuple[Figure, str]:
    """Return ``(figure, note)``: monthly means of ``year`` against the baseline percentile band.

    Raises ``PlotError`` if no month of ``year`` has enough data.
    """
    import plotly.graph_objects as go

    means = monthly_means(archive.dates, archive.tmean, config.min_year_coverage)
    last_complete = today().year - 1
    counts, _ = year_counts_and_means(archive.dates, archive.tmean)
    base_years, _ = classify_years(
        counts,
        config.baseline_start_year,
        config.baseline_end_year,
        last_complete,
        config.min_year_coverage,
    )
    in_base = set(base_years)
    climate = [
        baseline_of([v for (y, m), v in means.items() if m == month and y in in_base], config)
        for month in range(1, 13)
    ]
    shown = [means.get((year, m)) for m in range(1, 13)]
    if all(v is None for v in shown):
        raise PlotError(
            f"the daily series has no month with enough data in {year}; it covers "
            f"{archive.dates[0].isoformat()} to {archive.dates[-1].isoformat()}"
        )
    fig = go.Figure()
    bands = [(MONTHS[i], b) for i, b in enumerate(climate) if b is not None]
    if bands:
        first = bands[0][1]
        fig.add_bar(
            x=[m for m, _ in bands],
            y=[b.high_c - b.low_c for _, b in bands],
            base=[b.low_c for _, b in bands],
            customdata=[(b.low_c, b.high_c) for _, b in bands],
            width=0.8,
            name=f"baseline, percentile {first.low_percentile} to {first.high_percentile}",
            marker_color=BAND_COLOR,
            hovertemplate="baseline range: %{customdata[0]:.1f} to %{customdata[1]:.1f} °C"
            "<extra></extra>",
        )
        outside = first.low_percentile + 100 - first.high_percentile
        note = (
            f"Band: percentile {first.low_percentile} to {first.high_percentile} of each month's "
            f"mean over the valid years of {first.start_year}-{first.end_year}; about {outside}% "
            "of years fall outside it by construction."
        )
    else:
        note = (
            f"No baseline band: fewer than {config.min_years} valid years of "
            f"{config.baseline_start_year}-{config.baseline_end_year} are in the daily series."
        )
    fig.add_scatter(
        x=list(MONTHS),
        y=shown,
        mode="lines+markers",
        name=f"{year} monthly mean",
        line={"color": LINE_COLOR},
        hovertemplate="%{x}: %{y:.1f} °C<extra></extra>",
    )
    if any(v is None for v in shown):
        note += " Months without enough daily data, or not yet ended, are left blank."
    _layout(fig, f"{record.location.label}: monthly mean temperature in {year}", "°C")
    fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": list(MONTHS)})
    return fig, note + SOURCE_NOTE


def temperature_yearly_figure(
    record: TemperatureRecord,
    archive: TemperatureArchive,
    *,
    year: int | None,
    config: RainfallConfig,
    today: Callable[[], date] = date.today,
) -> tuple[Figure, str]:
    """Return ``(figure, note)``: annual means of the analysed years against the baseline.

    Raises ``PlotError`` if no year of the period has enough data.
    """
    import plotly.graph_objects as go

    last_complete = today().year - 1
    cov = config.min_year_coverage
    counts, means = year_counts_and_means(archive.dates, archive.tmean)
    years, _ = classify_years(counts, record.start_year, record.end_year, last_complete, cov)
    if not years:
        raise PlotError(
            f"no complete year with enough data between {record.start_year} and "
            f"{record.end_year}"
        )
    base_years, _ = classify_years(
        counts, config.baseline_start_year, config.baseline_end_year, last_complete, cov
    )
    base = baseline_of([means[y] for y in base_years], config)
    fig = go.Figure()
    if base is None:
        note = (
            f"No baseline: fewer than {config.min_years} valid years of "
            f"{config.baseline_start_year}-{config.baseline_end_year} are in the daily series."
        )
    else:
        fig.add_hrect(
            y0=base.low_c, y1=base.high_c, fillcolor=BAND_COLOR, line_width=0, layer="below"
        )
        fig.add_hline(y=base.mean_c, line_color=BASELINE_COLOR, line_dash="dash")
        outside = base.low_percentile + 100 - base.high_percentile
        note = (
            f"Shaded: percentile {base.low_percentile} to {base.high_percentile} of the annual "
            f"means of {base.years} valid years of {base.start_year}-{base.end_year} (about "
            f"{outside}% of years fall outside it by construction); dashed line: their mean, "
            f"{base.mean_c:.1f} °C."
        )
    fig.add_scatter(
        x=years,
        y=[means[y] for y in years],
        mode="lines+markers",
        name="annual mean",
        line={"color": LINE_COLOR},
        marker={
            "size": [11 if y == year else 6 for y in years],
            "color": [BASELINE_COLOR if y == year else LINE_COLOR for y in years],
        },
        hovertemplate="%{x}: %{y:.2f} °C<extra></extra>",
    )
    if year in years:
        note += f" {year} is highlighted."
    _layout(
        fig, f"{record.location.label}: annual mean temperature {years[0]}-{years[-1]}", "°C"
    )
    fig.update_layout(showlegend=False)
    return fig, note.strip() + SOURCE_NOTE
