"""Yearly and monthly totals: statistics, validation, tool wiring and plots. No network.

Hand-built series (``make_series``): ``base`` mm on every day, so a year's total is its number
of days times ``base`` and a month's total its number of days, unless ``spikes`` change a day.
"""

from dataclasses import replace
from datetime import date
from urllib.parse import parse_qs

import httpx
import pytest
from test_rainfall_client import FakeServer, analyze_with_archive

from research_agent.agent.loop import _trace_name
from research_agent.agent.tools import format_rainfall, parse_rainfall_args
from research_agent.llm import ToolCall
from research_agent.rainfall import InsufficientDataError, InvalidRainfallQueryError
from research_agent.rainfall.totals import compute_monthly_analysis, compute_yearly_analysis


@pytest.fixture
def cfg(rainfall_config):
    return replace(
        rainfall_config, min_years=3, baseline_start_year=2000, baseline_end_year=2004, top_events=3
    )


def yearly(cfg, make_location, make_provenance, make_series, *, start=2000, end=2004,
           spikes=None, missing=(), event_year=None):
    dates, values = make_series(2000, 2004, spikes, missing=missing)
    return compute_yearly_analysis(
        make_location(), dates, values, start_year=start, end_year=end, last_complete_year=2025,
        config=cfg, provenance=make_provenance(), baseline_from_year=2000, event_year=event_year,
    )


def monthly(cfg, make_location, make_provenance, make_series, *, start=2000, end=2004,
            spikes=None, missing=(), last=(2004, 12), event_year=None, event_month=None):
    dates, values = make_series(2000, 2004, spikes, missing=missing)
    return compute_monthly_analysis(
        make_location(), dates, values, start_year=start, end_year=end, last_complete_month=last,
        config=cfg, provenance=make_provenance(), baseline_from_year=2000,
        event_year=event_year, event_month=event_month,
    )


# --- yearly ---------------------------------------------------------------------------------------


def test_yearly_totals_rank_wettest_and_driest(cfg, make_location, make_provenance, make_series):
    # 2003 gets 100 mm more on one day; 2000 and 2004 are leap years (366 mm), others 365 mm.
    a = yearly(cfg, make_location, make_provenance, make_series,
               spikes={date(2003, 5, 1): 101.0}, event_year=2003)
    assert [t.total_mm for t in a.totals] == [366, 365, 365, 465, 366]
    assert [t.year for t in a.wettest] == [2003, 2000, 2004]
    assert [t.year for t in a.driest] == [2001, 2002, 2000]  # ties: earliest year first
    assert a.baseline.years == 5 and a.baseline.mean_mm == pytest.approx(385.4)
    focus = a.year_focus
    assert (focus.total_mm, focus.rank_wettest, focus.compared) == (465, 1, 5)
    assert focus.percent_drier == 80.0 and focus.band_position == "above"
    assert focus.ratio_to_baseline_mean == pytest.approx(465 / 385.4)


def test_a_short_recent_period_still_gets_the_baseline(
    cfg, make_location, make_provenance, make_series
):
    a = yearly(cfg, make_location, make_provenance, make_series, start=2003, end=2004)
    assert a.valid_years == (2003, 2004) and not a.ranking_available
    assert a.baseline is not None and a.baseline.years == 5  # from the 2000-2004 baseline years
    assert any("ranks were not computed, the baseline comparison still was" in c
               for c in a.caveats)


def test_an_incomplete_year_is_excluded_and_reported(
    cfg, make_location, make_provenance, make_series
):
    gap = [date(2001, 1, 1 + i) for i in range(30)] + [date(2001, 2, i) for i in range(1, 28)]
    a = yearly(cfg, make_location, make_provenance, make_series, missing=gap, event_year=2001)
    assert 2001 not in a.valid_years and a.year_focus is None
    assert any("2001 was excluded" in c for c in a.caveats)


def test_yearly_without_any_valid_year_raises(cfg, make_location, make_provenance, make_series):
    with pytest.raises(InsufficientDataError):
        yearly(cfg, make_location, make_provenance, make_series, start=2010, end=2012)


def test_format_yearly_lists_every_year_and_the_focus(
    cfg, make_location, make_provenance, make_series
):
    a = replace(yearly(cfg, make_location, make_provenance, make_series, event_year=2001), id="R1")
    text = format_rainfall(a)
    assert text.startswith("[R1] Annual precipitation totals for Zhengzhou")
    assert "Every year (x = times the baseline mean): 2000 366.0 mm (1.00x); 2001 365.0 mm" in text
    assert "Wettest years: 1) 2000 366.0 mm" in text and "Driest years: 1) 2001 365.0 mm" in text
    assert "Year 2001: 365.0 mm, rank 3 of 5 (1 = wettest" in text
    assert "baseline band" in text


# --- monthly --------------------------------------------------------------------------------------


def test_a_month_is_compared_with_the_same_calendar_month(
    cfg, make_location, make_provenance, make_series
):
    a = monthly(cfg, make_location, make_provenance, make_series,
                spikes={date(2002, 7, 20): 81.0}, event_year=2002, event_month=7)
    c = a.month_focus
    assert (c.year, c.month, c.total_mm) == (2002, 7, 111.0)  # 31 days, one of them 81 mm
    assert (c.rank_wettest, c.compared, c.percent_drier) == (1, 5, 80.0)
    assert c.band_position == "above"
    assert a.climate[1].years == 5 and a.climate[1].mean_mm == pytest.approx(28.4)  # February
    assert (a.wettest[0].year, a.wettest[0].month) == (2002, 7)
    assert a.months_used == 60 and a.months_missing == 0


def test_a_year_in_focus_gives_all_twelve_months(cfg, make_location, make_provenance, make_series):
    a = monthly(cfg, make_location, make_provenance, make_series, event_year=2001)
    assert [c.month for c in a.year_months] == list(range(1, 13))
    assert a.year_months[1].total_mm == 28.0 and a.month_focus is None


def test_a_month_that_has_not_ended_is_not_assessed(
    cfg, make_location, make_provenance, make_series
):
    a = monthly(cfg, make_location, make_provenance, make_series, last=(2004, 6),
                event_year=2004, event_month=8)
    assert a.month_focus.total_mm is None and a.month_focus.rank_wettest is None
    assert any("August 2004 has not ended" in c for c in a.caveats)
    assert (2004, 7) not in {(t.year, t.month) for t in a.wettest}


def test_a_month_with_too_little_data_is_left_out(
    cfg, make_location, make_provenance, make_series
):
    gap = [date(2003, 3, d) for d in range(1, 11)]
    a = monthly(cfg, make_location, make_provenance, make_series, missing=gap)
    assert a.months_missing == 1 and any("1 month(s) of the period" in c for c in a.caveats)


def test_format_monthly(cfg, make_location, make_provenance, make_series):
    a = monthly(cfg, make_location, make_provenance, make_series, event_year=2002, event_month=2)
    text = format_rainfall(replace(a, id="R2"))
    assert text.startswith("[R2] Monthly precipitation totals for Zhengzhou")
    assert "Feb 2002: 28.0 mm, rank 3 of 5 (1 = wettest; 0% were drier)" in text
    assert "Baseline 2000-2004 per calendar month: Jan mean 31.0 mm" in text


# --- client: validation and the series period -----------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scale": "weekly"}, "scale must be"),
        ({"scale": "yearly", "event_date": date(2020, 1, 1)}, "event_date is for scale 'daily'"),
        ({"scale": "yearly", "event_month": 7, "event_year": 2020}, "event_month needs"),
        ({"scale": "monthly", "event_month": 7}, "event_month needs"),
        ({"scale": "monthly", "event_month": 13, "event_year": 2020}, "1 to 12"),
        ({"event_year": 2020}, "event_year needs scale 'yearly' or 'monthly'"),
    ],
)
def test_invalid_scale_combinations_are_rejected_before_any_request(cfg, kwargs, message):
    server = FakeServer()
    with pytest.raises(InvalidRainfallQueryError, match=message):
        analyze_with_archive(cfg, server, **kwargs)
    assert server.requests == []


def test_yearly_fetches_the_series_from_the_baseline_start(rainfall_config):
    server = FakeServer(archive=[httpx.Response(400, json={"error": True, "reason": "stop"})])
    cfg = replace(rainfall_config, baseline_start_year=1991)
    with pytest.raises(Exception):  # noqa: B017 - only the request matters here
        analyze_with_archive(cfg, server, scale="yearly", start_year=2016, end_year=2025)
    archive = next(r for r in server.requests if "archive" in r.url.host)
    assert parse_qs(archive.url.query.decode())["start_date"] == ["1991-01-01"]


# --- tool arguments and trace ---------------------------------------------------------------------


def test_tool_arguments_carry_the_scale():
    args = parse_rainfall_args(
        {"city": "Amsterdam", "scale": "monthly", "event_year": 2021, "event_month": 7}
    )
    assert (args.scale, args.event_year, args.event_month) == ("monthly", 2021, 7)
    assert parse_rainfall_args({"city": "Amsterdam"}).scale == "daily"  # schema requires it


def test_the_tool_trace_names_the_scale():
    def name(**args):
        return _trace_name(ToolCall("c", "analyze_rainfall_extremes", {"city": "X", **args}))

    assert name(scale="yearly") == "analyze_rainfall_extremes[yearly]"
    assert name(scale="monthly", metric="dry") == "analyze_rainfall_extremes[monthly]"
    assert name(scale="daily", metric="dry") == "analyze_rainfall_extremes[dry]"
    assert name() == "analyze_rainfall_extremes[wet]"


# --- plots ----------------------------------------------------------------------------------------


def test_yearly_figure_draws_the_period_against_the_baseline(
    cfg, make_location, make_provenance, make_series
):
    pytest.importorskip("plotly")
    from research_agent.rainfall import Archive
    from research_agent.ui.plots import default_plot_year, yearly_figure

    record = yearly(cfg, make_location, make_provenance, make_series,
                    spikes={date(2003, 5, 1): 101.0}, event_year=2003)
    dates, values = make_series(2000, 2004, {date(2003, 5, 1): 101.0})
    archive = Archive(tuple(dates), tuple(values), 34.75, 113.75, 110.0, "Asia/Shanghai")
    year = default_plot_year(record)
    figure, note = yearly_figure(
        record, archive, year=year, config=cfg, today=lambda: date(2026, 1, 1)
    )
    bars = figure.data[0]
    assert list(bars.x) == [2000, 2001, 2002, 2003, 2004]
    assert list(bars.y) == [366, 365, 365, 465, 366]
    assert year == 2003 and "2003 is highlighted" in note and "about 20% of years" in note
