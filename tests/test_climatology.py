"""Smoothed day-of-year baseline: synthetic series with hand-checkable results."""

from datetime import date, timedelta

import pytest

from research_agent.rainfall.climatology import SLOTS, daily_baseline

LAST_COMPLETE = 2025


def series(years, value):
    """Every day of ``years``; ``value(day)`` gives the precipitation (or None)."""
    dates = []
    day, end = date(years[0], 1, 1), date(years[-1], 12, 31)
    while day <= end:
        dates.append(day)
        day += timedelta(days=1)
    return dates, [value(d) for d in dates]


def baseline(dates, values, **kwargs):
    defaults = {
        "start_year": 2001,
        "end_year": 2003,
        "last_complete_year": LAST_COMPLETE,
        "min_years": 3,
        "min_year_coverage": 0.9,
        "window_days": 1,
    }
    return daily_baseline(dates, values, **{**defaults, **kwargs})


def test_unsmoothed_baseline_is_the_mean_of_each_calendar_day_over_the_years():
    # 2001, 2002, 2003 get 1, 2, 3 mm on 15 June and nothing on every other day.
    dates, values = series(
        [2001, 2002, 2003],
        lambda d: float(d.year - 2000) if (d.month, d.day) == (6, 15) else 0.0,
    )
    b = baseline(dates, values)
    assert b.years == (2001, 2002, 2003)
    assert b.at(date(2010, 6, 15)) == pytest.approx(2.0)  # any year maps to the same slot
    assert b.at(date(2010, 6, 16)) == 0.0
    assert len(b.mm_by_slot) == SLOTS


def test_29_february_is_averaged_over_leap_years_only_and_marches_do_not_shift():
    # Baseline 2000-2002 (2000 is the only leap year): 29 Feb is 9 mm there, 1 March is 5 mm.
    mm = {(2, 29): 9.0, (3, 1): 5.0}
    dates, values = series([2000, 2001, 2002], lambda d: mm.get((d.month, d.day), 0.0))
    b = baseline(dates, values, start_year=2000, end_year=2002)
    assert b.at(date(2004, 2, 29)) == 9.0
    assert b.at(date(2001, 3, 1)) == 5.0  # not diluted by the leap day shifting later years


def test_a_missing_day_is_skipped_not_counted_as_zero():
    dates, values = series(
        [2001, 2002, 2003],
        lambda d: None if (d.year == 2002 and (d.month, d.day) == (7, 1)) else 6.0,
    )
    assert baseline(dates, values).at(date(2001, 7, 1)) == 6.0


def test_smoothing_is_a_centred_window_that_wraps_around_the_year_end():
    # 30 mm on 1 January only, in every baseline year; a 3-day window spreads it to 31 December,
    # 1 January and 2 January.
    dates, values = series(
        [2001, 2002, 2003], lambda d: 30.0 if (d.month, d.day) == (1, 1) else 0.0
    )
    b = baseline(dates, values, window_days=3)
    assert b.at(date(2002, 12, 31)) == pytest.approx(10.0)
    assert b.at(date(2002, 1, 1)) == pytest.approx(10.0)
    assert b.at(date(2002, 1, 2)) == pytest.approx(10.0)
    assert b.at(date(2002, 1, 3)) == 0.0
    assert b.at(date(2002, 12, 30)) == 0.0


def test_smoothing_keeps_the_annual_mean():
    # 2000 is a leap year, so all 366 slots have data and the circular average is exact.
    dates, values = series([2000, 2001, 2002], lambda d: float(d.timetuple().tm_yday % 7))
    period = {"start_year": 2000, "end_year": 2002}
    raw = baseline(dates, values, **period)
    smooth = baseline(dates, values, window_days=15, **period)
    assert sum(smooth.mm_by_slot) == pytest.approx(sum(raw.mm_by_slot))
    assert max(smooth.mm_by_slot) < max(raw.mm_by_slot)  # and it is smoother


def test_only_valid_years_inside_the_baseline_period_are_used():
    # 2002 has too little data and 2004 lies outside the period: neither may take part.
    def value(d):
        if d.year == 2002 and d.month > 2:
            return None
        return 100.0 if d.year == 2004 else 1.0

    dates, values = series([2001, 2002, 2003, 2004], value)
    b = baseline(dates, values, min_years=2)
    assert b.years == (2001, 2003)
    assert b.at(date(2001, 8, 8)) == 1.0


def test_too_few_valid_years_gives_no_baseline():
    dates, values = series([2001, 2002], lambda d: 1.0)  # the period asks for 2001-2003
    assert baseline(dates, values) is None
    assert baseline(dates, values, min_years=2) is not None


def test_an_incomplete_current_year_is_not_a_baseline_year():
    dates, values = series([2024, 2025, 2026], lambda d: 1.0)
    assert baseline(dates, values, start_year=2024, end_year=2026, min_years=3) is None


@pytest.mark.parametrize("window", [0, -3, 2, 14, 367])
def test_window_must_be_odd_and_at_most_a_year(window):
    dates, values = series([2001, 2002, 2003], lambda d: 1.0)
    with pytest.raises(ValueError, match="odd"):
        baseline(dates, values, window_days=window)


def test_a_slot_with_no_data_is_none_unless_smoothing_reaches_a_neighbour():
    # No leap year in 2001-2003, so 29 February has no data of its own.
    dates, values = series([2001, 2002, 2003], lambda d: 4.0)
    assert baseline(dates, values).at(date(2004, 2, 29)) is None
    assert baseline(dates, values, window_days=3).at(date(2004, 2, 29)) == 4.0
