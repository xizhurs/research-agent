"""Temperature statistics: pure functions on synthetic daily series. No network."""

from dataclasses import replace
from datetime import date, timedelta

import pytest

from research_agent.rainfall import InsufficientDataError, InvalidRainfallQueryError
from research_agent.temperature.analysis import (
    compute_extremes,
    compute_monthly,
    compute_yearly,
    monthly_means,
    valid_annual_means,
)


@pytest.fixture
def cfg(rainfall_config):
    return replace(
        rainfall_config, min_years=3, baseline_start_year=2000, baseline_end_year=2002, top_events=3
    )


def common(cfg, make_location, make_provenance, **extra):
    return {
        "start_year": 2000,
        "end_year": 2004,
        "config": cfg,
        "provenance": make_provenance(),
        "baseline_from_year": 2000,
        **extra,
    } | {"location": make_location()}


def series(make_series, by_year, spikes=None, missing=()):
    """Every day of a year at ``by_year[year]`` degrees, with optional spikes and gaps."""
    dates, values = make_series(2000, 2004, spikes=spikes, missing=missing)
    values = [
        None if v is None else spikes.get(d, by_year[d.year]) if spikes else by_year[d.year]
        for d, v in zip(dates, values, strict=True)
    ]
    return dates, values


YEARS = {2000: 10.0, 2001: 11.0, 2002: 12.0, 2003: 14.0, 2004: 9.0}


def yearly(cfg, make_location, make_provenance, dates, values, **kw):
    args = common(cfg, make_location, make_provenance, **kw)
    location = args.pop("location")
    return compute_yearly(location, dates, values, last_complete_year=2004, **args)


# --- means ---------------------------------------------------------------------------------------


def test_annual_means_skip_missing_days_and_need_coverage(make_series, cfg):
    dates, values = series(make_series, YEARS, missing=[date(2001, 1, d) for d in range(1, 11)])
    means, excluded = valid_annual_means(
        dates, values, start_year=2000, end_year=2004, last_complete_year=2004,
        min_year_coverage=0.9,
    )  # fmt: skip
    assert means[2001] == pytest.approx(11.0)  # ten missing days skipped, not counted as 0
    assert excluded == []
    gap = [date(2002, 1, 1) + timedelta(days=i) for i in range(60)]
    dates, values = series(make_series, YEARS, missing=gap)
    means, excluded = valid_annual_means(
        dates, values, start_year=2000, end_year=2004, last_complete_year=2004,
        min_year_coverage=0.9,
    )  # fmt: skip
    assert 2002 not in means and excluded[0][0] == 2002


def test_monthly_means_need_coverage_per_month(make_series):
    dates, values = series(make_series, YEARS, missing=[date(2003, 2, d) for d in range(1, 10)])
    means = monthly_means(dates, values, 0.9)
    assert means[(2003, 1)] == pytest.approx(14.0)
    assert (2003, 2) not in means  # 19 of 28 days


# --- yearly --------------------------------------------------------------------------------------


def test_yearly_ranks_warmest_and_coolest_with_anomalies_in_degrees(
    cfg, make_location, make_provenance, make_series
):
    dates, values = series(make_series, YEARS)
    a = yearly(cfg, make_location, make_provenance, dates, values, event_year=2003)
    assert [m.year for m in a.warmest] == [2003, 2002, 2001]
    assert [m.year for m in a.coolest] == [2004, 2000, 2001]
    assert a.baseline.mean_c == pytest.approx(11.0) and a.baseline.years == 3
    focus = a.year_focus
    assert focus.rank_warmest == 1 and focus.compared == 5
    assert focus.percent_cooler == pytest.approx(80.0)
    assert focus.anomaly_c == pytest.approx(3.0)  # a difference, never a ratio
    assert focus.band_position == "above"
    assert any("No trend was computed" in c for c in a.caveats)
    assert any("elevation is 110 m" in c for c in a.caveats)


def test_yearly_withholds_ranks_below_min_years(cfg, make_location, make_provenance, make_series):
    dates, values = series(make_series, YEARS)
    a = yearly(
        replace(cfg, min_years=10), make_location, make_provenance, dates, values, event_year=2003
    )
    assert a.ranking_available is False and a.year_focus.rank_warmest is None
    assert a.baseline is None


def test_yearly_without_any_valid_year_raises(cfg, make_location, make_provenance, make_series):
    dates, values = make_series(2000, 2004, base=None)
    with pytest.raises(InsufficientDataError):
        yearly(cfg, make_location, make_provenance, dates, [None] * len(values))


# --- monthly -------------------------------------------------------------------------------------


def test_monthly_compares_a_month_with_the_same_calendar_month(
    cfg, make_location, make_provenance, make_series
):
    spikes = {date(2003, 7, d): 30.0 for d in range(1, 32)}
    dates, values = series(make_series, YEARS, spikes=spikes)
    args = common(cfg, make_location, make_provenance)
    location = args.pop("location")
    m = compute_monthly(
        location, dates, values, last_complete_month=(2004, 12), event_year=2003, event_month=7,
        **args,
    )  # fmt: skip
    assert m.month_focus.mean_c == pytest.approx(30.0)
    assert m.month_focus.rank_warmest == 1 and m.month_focus.anomaly_c == pytest.approx(19.0)
    assert (m.warm_anomalies[0].year, m.warm_anomalies[0].month) == (2003, 7)
    assert m.cool_anomalies[0].year == 2004  # 9 degrees, the coolest against the baseline
    assert m.months_used == 60 and m.months_missing == 0


# --- extremes ------------------------------------------------------------------------------------


def extremes(cfg, make_location, make_provenance, dates, values, metric, **kw):
    args = common(cfg, make_location, make_provenance, **kw)
    location = args.pop("location")
    return compute_extremes(
        location, dates, values, metric=metric, last_complete_year=2004, **args
    )


def test_hot_extremes_txx_per_year_ties_go_to_the_earliest_day(
    cfg, make_location, make_provenance, make_series
):
    spikes = {date(2001, 6, 1): 35.0, date(2001, 8, 1): 35.0, date(2003, 7, 10): 40.0}
    dates, values = series(make_series, YEARS, spikes=spikes)
    a = extremes(
        cfg, make_location, make_provenance, dates, values, "hot", event_date=date(2003, 7, 10)
    )
    assert a.summary.startswith("TXx")
    assert [(e.year, e.date, e.value_c) for e in a.annual][1] == (2001, "2001-06-01", 35.0)
    assert a.top[0].date == "2003-07-10"
    ev = a.event
    assert ev.rank_among_days == 1 and ev.rank_among_annual_extremes == 1
    assert ev.annual_percentile == pytest.approx(80.0)
    # baseline TXx 2000-2002: 10, 35, 12 -> mean 19
    assert ev.difference_from_baseline_extreme_c == pytest.approx(40.0 - 19.0)
    assert ev.day_of_year_mean_c is not None and ev.difference_from_day_of_year_c > 0


def test_cold_extremes_handle_negative_values_and_rank_the_coldest_first(
    cfg, make_location, make_provenance, make_series
):
    spikes = {date(2002, 1, 5): -20.0, date(2004, 12, 31): -25.5}
    dates, values = series(make_series, YEARS, spikes=spikes)
    a = extremes(
        cfg, make_location, make_provenance, dates, values, "cold", event_date=date(2002, 1, 5)
    )
    assert a.summary.startswith("TNn")
    assert [e.value_c for e in a.top[:2]] == [-25.5, -20.0]
    assert a.event.rank_among_days == 2 and a.event.rank_among_annual_extremes == 2
    assert any("TNn is the lowest" in c for c in a.caveats)


def test_extremes_reject_an_unknown_metric(cfg, make_location, make_provenance, make_series):
    dates, values = series(make_series, YEARS)
    with pytest.raises(InvalidRainfallQueryError):
        extremes(cfg, make_location, make_provenance, dates, values, "wet")


def test_extremes_need_data_on_the_event_day(cfg, make_location, make_provenance, make_series):
    dates, values = series(make_series, YEARS, missing=[date(2003, 7, 10)])
    with pytest.raises(InsufficientDataError, match="no data for 2003-07-10"):
        extremes(
            cfg, make_location, make_provenance, dates, values, "hot",
            event_date=date(2003, 7, 10),
        )  # fmt: skip
