"""The daily plot: traces, baseline values, the no-baseline case. Skipped without plotly."""

from dataclasses import replace
from datetime import date

import pytest

pytest.importorskip("plotly")

from research_agent.rainfall import Archive  # noqa: E402
from research_agent.ui.plots import (  # noqa: E402
    PlotError,
    daily_figure,
    default_plot_year,
    monthly_figure,
)

TODAY = lambda: date(2026, 11, 2)  # noqa: E731


@pytest.fixture
def cfg(rainfall_config):
    return replace(rainfall_config, min_years=3, baseline_start_year=2000, baseline_end_year=2004)


@pytest.fixture
def archive(make_analysis, make_series):
    # The series make_analysis uses: 1 mm every day, July 1 = 10, 20, 30, 40, 50 mm in 2000-2004.
    july_1 = {2000: 10.0, 2001: 20.0, 2002: 30.0, 2003: 40.0, 2004: 50.0}
    spikes = {date(y, 7, 1): mm for y, mm in july_1.items()}
    dates, values = make_series(2000, 2004, spikes)
    return Archive(tuple(dates), tuple(values), 34.75, 113.75, 110.0, "Asia/Shanghai")


def figure(record, archive, cfg, year=2002, window_days=1):
    return daily_figure(
        record, archive, year=year, window_days=window_days, config=cfg, today=TODAY
    )


def test_bars_are_the_chosen_years_daily_values_and_the_line_is_the_baseline(
    make_analysis, archive, cfg
):
    fig, note = figure(make_analysis(), archive, cfg, year=2002)
    bars, line = fig.data
    assert len(bars.x) == 365 and bars.x[0] == date(2002, 1, 1)
    assert dict(zip(bars.x, bars.y, strict=True))[date(2002, 7, 1)] == 30.0  # that year's spike
    base = dict(zip(line.x, line.y, strict=True))
    assert base[date(2002, 7, 1)] == pytest.approx(30.0)  # mean of 10..50
    assert base[date(2002, 1, 1)] == pytest.approx(1.0)
    assert "5 valid years of 2000-2004" in note and "1-day" in note


def test_smoothing_spreads_the_baseline_spike_over_the_window(make_analysis, archive, cfg):
    fig, _ = figure(make_analysis(), archive, cfg, year=2002, window_days=15)
    base = dict(zip(fig.data[1].x, fig.data[1].y, strict=True))
    assert base[date(2002, 7, 1)] == pytest.approx((14 * 1.0 + 30.0) / 15)
    assert base[date(2002, 7, 8)] == base[date(2002, 6, 24)]  # symmetric around the spike


def test_no_baseline_line_when_the_series_has_too_few_baseline_years(make_analysis, archive, cfg):
    cfg = replace(cfg, baseline_start_year=2010, baseline_end_year=2020)  # no overlap with 2000-04
    fig, note = figure(make_analysis(), archive, cfg)
    assert len(fig.data) == 1  # bars only
    assert "No baseline line" in note and "2010-2020" in note


def test_a_partly_covered_baseline_period_uses_the_years_available_and_says_how_many(
    make_analysis, archive, cfg
):
    cfg = replace(cfg, baseline_start_year=1991, baseline_end_year=2020)
    fig, note = figure(make_analysis(), archive, cfg)
    assert len(fig.data) == 2
    assert "5 valid years of 1991-2020" in note


def test_year_without_data_is_a_plot_error_naming_the_coverage(make_analysis, archive, cfg):
    with pytest.raises(PlotError, match="2010.*2000-01-01 to 2004-12-31"):
        figure(make_analysis(), archive, cfg, year=2010)


def test_event_day_is_marked_only_when_it_lies_in_the_plotted_year(make_analysis, archive, cfg):
    record = make_analysis(event_date=date(2003, 7, 1))
    marked, _ = figure(record, archive, cfg, year=2003)
    assert [s.x0 for s in marked.layout.shapes] == ["2003-07-01"]
    other, _ = figure(record, archive, cfg, year=2002)
    assert other.layout.shapes == ()


def test_default_year_is_the_event_year_then_the_compared_year_then_the_latest(
    make_analysis, make_dry_analysis
):
    assert default_plot_year(make_analysis(event_date=date(2002, 7, 1))) == 2002
    assert default_plot_year(make_analysis()) == 2004
    assert default_plot_year(make_dry_analysis(event_year=2001)) == 2001
    assert default_plot_year(make_dry_analysis(event_date=date(2003, 3, 3))) == 2003
    assert default_plot_year(make_dry_analysis()) == 2004


# --- monthly view -------------------------------------------------------------------------------

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly(record, archive, cfg, year=2002):
    return monthly_figure(record, archive, year=year, config=cfg, today=TODAY)


def test_monthly_bars_are_the_years_totals_and_the_band_is_the_baseline_percentiles(
    make_analysis, archive, cfg
):
    # July is 30 mm of 1 mm days plus the 1 July spike: totals 40, 50, 60, 70, 80 in 2000-04.
    fig, note = monthly(make_analysis(), archive, cfg, year=2002)
    band, bars = fig.data
    assert list(bars.x) == MONTHS
    assert bars.y[0] == 31.0 and bars.y[1] == 28.0 and bars.y[6] == 60.0
    july = list(band.x).index("Jul")
    assert band.base[july] == pytest.approx(44.0)  # 10th percentile of 40..80
    assert band.base[july] + band.y[july] == pytest.approx(76.0)  # 90th percentile
    assert fig.layout.barmode == "overlay"
    assert fig.layout.title.text == "Zhengzhou, Henan, China: monthly precipitation in 2002"
    assert "percentile 10 to 90" in note and "5 valid years of 2000-2004" in note
    assert "20% of years fall outside" in note


def test_no_monthly_band_when_the_series_has_too_few_baseline_years(make_analysis, archive, cfg):
    cfg = replace(cfg, baseline_start_year=2010, baseline_end_year=2020)
    fig, note = monthly(make_analysis(), archive, cfg)
    assert len(fig.data) == 1  # the year's bars only
    assert "No baseline band" in note and "2010-2020" in note


def test_months_without_enough_data_are_left_blank_and_the_note_says_so(
    make_analysis, make_series, cfg
):
    last_quarter = [d for d in make_series(2004, 2004)[0] if d.month >= 10]
    dates, values = make_series(2000, 2004, missing=last_quarter)
    archive = Archive(tuple(dates), tuple(values), 34.75, 113.75, 110.0, "Asia/Shanghai")
    fig, note = monthly(make_analysis(), archive, cfg, year=2004)
    totals = fig.data[-1].y
    assert totals[8] == 30.0 and tuple(totals[9:]) == (None, None, None)
    assert "left blank" in note
    assert "4 valid years of 2000-2004" in note  # 2004 is too incomplete to be a baseline year


def test_a_year_with_no_usable_month_is_a_plot_error(make_analysis, archive, cfg):
    with pytest.raises(PlotError, match="no month with enough data in 2010"):
        monthly(make_analysis(), archive, cfg, year=2010)
