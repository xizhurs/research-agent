"""Pure Rx1day statistics, checked against hand-computed values. No network."""

from dataclasses import replace
from datetime import date

import pytest

from research_agent.rainfall import InsufficientDataError
from research_agent.rainfall.analysis import compute_analysis


@pytest.fixture
def cfg(rainfall_config):
    # Baseline = the five synthetic years, and three valid years are enough to rank.
    return replace(
        rainfall_config, min_years=3, baseline_start_year=2000, baseline_end_year=2004, top_events=3
    )


@pytest.fixture
def compute(cfg, make_location, make_provenance, make_series):
    def _compute(
        spikes=None, *, start=2000, end=2004, last_complete=2025, missing=(), config=None, **kw
    ):
        dates, values = make_series(start, end, spikes, missing=missing)
        return compute_analysis(
            make_location(),
            dates,
            values,
            start_year=start,
            end_year=end,
            last_complete_year=last_complete,
            config=config or cfg,
            provenance=make_provenance(),
            **kw,
        )

    return _compute


JULY = {date(2000 + i, 7, 1): mm for i, mm in enumerate([10.0, 20.0, 30.0, 40.0, 50.0])}


def test_rx1day_per_year_and_date_of_the_maximum(compute):
    result = compute(JULY)
    assert [(a.year, a.date, a.mm) for a in result.annual_maxima] == [
        (2000, "2000-07-01", 10.0),
        (2001, "2001-07-01", 20.0),
        (2002, "2002-07-01", 30.0),
        (2003, "2003-07-01", 40.0),
        (2004, "2004-07-01", 50.0),
    ]
    assert result.excluded_years == ()


def test_tied_maxima_report_the_earliest_date(compute):
    result = compute({date(2001, 3, 5): 20.0, date(2001, 9, 9): 20.0, **JULY})
    assert next(a for a in result.annual_maxima if a.year == 2001).date == "2001-03-05"


def test_baseline_mean_and_median(compute):
    base = compute(JULY).baseline
    assert (base.start_year, base.end_year, base.years) == (2000, 2004, 5)
    assert base.mean_mm == pytest.approx(30.0)
    assert base.median_mm == pytest.approx(30.0)
    skewed = compute({**JULY, date(2004, 7, 1): 120.0}).baseline  # maxima 10 20 30 40 120
    assert skewed.mean_mm == pytest.approx(44.0)
    assert skewed.median_mm == pytest.approx(30.0)


def test_top_events_are_wettest_first_and_capped(compute):
    top = compute(JULY).top_events
    assert [(a.year, a.mm) for a in top] == [(2004, 50.0), (2003, 40.0), (2002, 30.0)]


def test_event_rank_percentile_and_ratio(compute):
    # 35 mm on 1 Aug 2004: only the 2003 (40) and 2004 (50) maxima are wetter.
    spikes = {**JULY, date(2004, 8, 1): 35.0}
    ev = compute(spikes, event_date=date(2004, 8, 1)).event
    assert ev.mm == 35.0
    assert ev.rank_among_days == 3  # two wetter days in the whole record
    assert ev.days == 1827  # 5 years incl. leap 2000 and 2004
    assert ev.rank_among_annual_maxima == 3
    assert ev.annual_percentile == pytest.approx(60.0)  # 3 of 5 maxima are below 35
    assert ev.ratio_to_baseline_mean == pytest.approx(35.0 / 30.0)


def test_event_equal_to_the_record_maximum_ranks_first(compute):
    ev = compute(JULY, event_date=date(2004, 7, 1)).event
    assert ev.rank_among_days == 1 and ev.rank_among_annual_maxima == 1
    assert ev.annual_percentile == pytest.approx(80.0)


def test_year_with_too_little_data_is_excluded_and_reported(compute):
    gaps = [date(2002, 1, 1) + (date(2002, 1, 2) - date(2002, 1, 1)) * i for i in range(60)]
    result = compute(JULY, missing=gaps)  # 2002 keeps 305 of 365 days: under 90%
    assert [a.year for a in result.annual_maxima] == [2000, 2001, 2003, 2004]
    assert result.excluded_years == ((2002, "only 305 of 365 days have data"),)
    assert any("excluded" in c and "2002" in c for c in result.caveats)


def test_a_few_missing_days_do_not_exclude_a_year(compute):
    result = compute(JULY, missing=[date(2002, 1, 1), date(2002, 1, 2)])
    assert 2002 in {a.year for a in result.annual_maxima}


def test_incomplete_current_year_is_never_used_for_annual_maxima(compute):
    result = compute(JULY, last_complete=2003)
    assert [a.year for a in result.annual_maxima] == [2000, 2001, 2002, 2003]
    assert (2004, "year not complete") in result.excluded_years


def test_no_valid_year_raises(compute):
    with pytest.raises(InsufficientDataError, match="no year"):
        compute(JULY, last_complete=1990)


def test_short_record_withholds_ranks_and_baseline(compute, rainfall_config):
    strict = replace(
        rainfall_config, min_years=20, baseline_start_year=2000, baseline_end_year=2004
    )
    result = compute({**JULY, date(2004, 8, 1): 35.0}, config=strict, event_date=date(2004, 8, 1))
    assert result.ranking_available is False
    assert result.baseline is None
    ev = result.event
    assert (ev.rank_among_days, ev.rank_among_annual_maxima, ev.annual_percentile) == (
        None,
        None,
        None,
    )
    assert ev.ratio_to_baseline_mean is None and ev.mm == 35.0
    assert any("ranks and percentiles were not computed" in c for c in result.caveats)
    assert any("no baseline comparison" in c for c in result.caveats)


def test_event_day_without_data_raises(compute):
    with pytest.raises(InsufficientDataError, match="2004-08-01"):
        compute(JULY, missing=[date(2004, 8, 1)], event_date=date(2004, 8, 1))


def test_event_in_an_excluded_year_is_compared_with_the_other_years(compute):
    result = compute(JULY, last_complete=2003, event_date=date(2004, 7, 1))
    assert result.event.mm == 50.0
    assert result.event.rank_among_annual_maxima == 1  # wetter than all four valid maxima
    assert any("excluded year" in c for c in result.caveats)


def test_reanalysis_and_rx1day_caveats_are_always_present(compute):
    caveats = " ".join(compute(JULY).caveats)
    assert "ERA5 reanalysis" in caveats and "not gauge measurements" in caveats
    assert "Rx1day" in caveats


def test_other_matches_are_disclosed(compute, make_location):
    other = make_location(id=2, name="Zhengzhoucun", admin1="Guangxi", population=None)
    caveat = " ".join(compute(JULY, other_matches=[other]).caveats)
    assert "also matched 1 other place" in caveat and "Zhengzhoucun" in caveat


def test_no_other_matches_means_no_disclosure(compute):
    assert not any("also matched" in c for c in compute(JULY).caveats)
