"""Pure dry-condition statistics, checked against the hand-computed series in conftest.py."""

from dataclasses import replace
from datetime import date

import pytest

from research_agent.rainfall import InsufficientDataError
from research_agent.rainfall.dryness import compute_dry_analysis


@pytest.fixture
def cfg(rainfall_config):
    return replace(
        rainfall_config,
        min_years=3,
        baseline_start_year=2000,
        baseline_end_year=2004,
        top_events=3,
    )


@pytest.fixture
def compute(cfg, make_location, make_provenance, make_series, dry_spikes):
    def _compute(spikes=None, *, base=5.0, start=2000, end=2004, last_complete=2025, **kw):
        spikes = dry_spikes if spikes is None else spikes
        dates, values = make_series(start, end, spikes, base=base, missing=kw.pop("missing", ()))
        return compute_dry_analysis(
            make_location(),
            dates,
            values,
            start_year=start,
            end_year=end,
            last_complete_year=last_complete,
            config=kw.pop("config", cfg),
            provenance=make_provenance(),
            **kw,
        )

    return _compute


def test_yearly_cdd_restarts_on_new_year_and_totals_are_summed(compute):
    result = compute()
    assert [(a.year, a.cdd_days, a.cdd_start, a.cdd_end) for a in result.annual] == [
        (2000, 20, "2000-07-01", "2000-07-20"),
        (2001, 12, "2001-12-20", "2001-12-31"),  # the spell is cut at New Year here...
        (2002, 9, "2002-01-01", "2002-01-09"),
        (2003, 0, None, None),  # a year with no dry day
        (2004, 29, "2004-08-01", "2004-08-29"),
    ]
    assert [a.total_mm for a in result.annual] == pytest.approx([1680, 1765, 1755, 1825, 1685])


def test_unbroken_spells_are_not_split_at_new_year_and_sorted_longest_first(compute):
    spells = compute().longest_spells
    assert [(s.days, s.start, s.end) for s in spells] == [
        (29, "2004-08-01", "2004-08-29"),
        (21, "2001-12-20", "2002-01-09"),  # ...but not here
        (20, "2000-07-01", "2000-07-20"),
    ]
    assert spells[0].total_mm == 0.0


def test_driest_years_are_ranked_by_annual_total(compute):
    driest = compute().driest_years
    assert [a.year for a in driest] == [2000, 2004, 2002]
    assert [a.total_mm for a in driest] == pytest.approx([1680, 1685, 1755])


def test_baseline_means_and_medians(compute):
    base = compute().baseline
    assert (base.start_year, base.end_year, base.years) == (2000, 2004, 5)
    assert base.cdd_mean == pytest.approx(14.0) and base.cdd_median == pytest.approx(12.0)
    assert base.total_mean_mm == pytest.approx(1742.0)
    assert base.total_median_mm == pytest.approx(1755.0)


def test_exactly_the_threshold_is_a_wet_day_and_just_below_is_dry(compute):
    spikes = {date(2003, 3, 3): 0.99, date(2003, 3, 4): 1.0}  # adjacent; only 2003 has dry days
    result = compute(spikes)
    row = next(a for a in result.annual if a.year == 2003)
    assert (row.cdd_days, row.cdd_start) == (1, "2003-03-03")


def test_a_missing_day_breaks_a_dry_run_and_is_never_counted_as_dry(compute):
    july = {date(2000, 7, d): 0.0 for d in range(1, 21)}
    result = compute(july, missing=[date(2000, 7, 10)])
    assert [(s.days, s.start, s.end) for s in result.longest_spells][:2] == [
        (10, "2000-07-11", "2000-07-20"),
        (9, "2000-07-01", "2000-07-09"),
    ]
    assert any("without data fall in valid years" in c for c in result.caveats)


def test_tied_yearly_runs_report_the_earliest(compute):
    spikes = {
        date(2000, m, d): 0.0 for m, days in ((2, range(1, 6)), (9, range(1, 6))) for d in days
    }
    row = next(a for a in compute(spikes).annual if a.year == 2000)
    assert (row.cdd_days, row.cdd_start) == (5, "2000-02-01")


def test_year_focus_rank_percentile_and_ratio(compute):
    focus = compute(event_year=2004).year_focus
    assert focus.year == 2004 and focus.total_mm == pytest.approx(1685)
    assert focus.total_rank == 2  # only 2000 was drier
    assert focus.percent_wetter == pytest.approx(60.0)  # 2002, 2001 and 2003 were wetter
    assert focus.ratio_to_baseline_mean == pytest.approx(1685 / 1742)
    assert focus.cdd_days == 29


def test_driest_year_has_rank_one(compute):
    assert compute(event_year=2000).year_focus.total_rank == 1


def test_event_date_inside_a_spell_that_crosses_new_year(compute):
    result = compute(event_date=date(2001, 12, 25))
    ev = result.event
    assert (ev.dry_day, ev.value_mm) == (True, 0.0)
    assert (ev.spell.days, ev.spell.start, ev.spell.end) == (21, "2001-12-20", "2002-01-09")
    assert (ev.spell_rank, ev.spells) == (2, 5)  # the 29-day spell of 2004 is longer
    assert result.year_focus.year == 2001  # defaults to the event day's year
    assert result.year_focus.total_rank == 4


def test_event_year_overrides_the_event_dates_year(compute):
    result = compute(event_date=date(2001, 12, 25), event_year=2004)
    assert result.year_focus.year == 2004 and result.event.spell.days == 21


def test_event_date_on_a_wet_day_is_not_a_dry_day(compute):
    ev = compute(event_date=date(2003, 6, 15)).event
    assert (ev.dry_day, ev.spell, ev.spell_rank) == (False, None, None)
    assert ev.value_mm == 5.0


def test_event_day_without_data_raises(compute):
    with pytest.raises(InsufficientDataError, match="2003-06-15"):
        compute(event_date=date(2003, 6, 15), missing=[date(2003, 6, 15)])


def test_year_with_too_little_data_is_excluded_and_cuts_spells(compute, dry_spikes):
    gaps = [date(2002, 3, 1) + (date(2002, 3, 2) - date(2002, 3, 1)) * i for i in range(60)]
    result = compute(missing=gaps, event_date=date(2002, 1, 5), event_year=2002)
    assert [y for y, _ in result.excluded_years] == [2002]
    assert [a.year for a in result.annual] == [2000, 2001, 2003, 2004]
    assert result.year_focus is None  # an excluded year is reported, not ranked
    assert any("2002 was excluded" in c for c in result.caveats)
    # The January 2002 days are outside the valid record, so the spell stops at New Year.
    assert result.event.dry_day is True and result.event.spell is None
    assert any("lies in an excluded year" in c for c in result.caveats)
    assert (21, "2001-12-20", "2002-01-09") not in [
        (s.days, s.start, s.end) for s in result.longest_spells
    ]


def test_incomplete_year_is_excluded(compute):
    result = compute(last_complete=2003)
    assert [a.year for a in result.annual] == [2000, 2001, 2002, 2003]
    assert (2004, "year not complete") in result.excluded_years


def test_short_record_withholds_ranks_and_baseline(compute, rainfall_config):
    strict = replace(
        rainfall_config,
        min_years=20,
        baseline_start_year=2000,
        baseline_end_year=2004,
        top_events=3,
    )
    result = compute(config=strict, event_date=date(2001, 12, 25), event_year=2004)
    assert result.ranking_available is False and result.baseline is None
    assert result.year_focus.total_rank is None and result.year_focus.percent_wetter is None
    assert result.year_focus.ratio_to_baseline_mean is None
    assert result.event.spell.days == 21 and result.event.spell_rank is None
    assert any("ranks and percentiles were not computed" in c for c in result.caveats)
    assert any("no baseline comparison" in c for c in result.caveats)


def test_no_valid_year_raises(compute):
    with pytest.raises(InsufficientDataError, match="no year"):
        compute(last_complete=1990)


def test_dry_caveats_are_always_present_and_do_not_claim_a_drought(compute):
    text = " ".join(compute().caveats)
    assert "not gauge measurements" in text
    assert "too many light-rain days" in text and "underestimates" in text
    assert "not a drought index" in text and "less than 1 mm" in text
    assert "restarts on 1 January" in text


def test_threshold_comes_from_config(compute, cfg):
    strict = replace(cfg, dry_day_mm=6.0)  # now the 5 mm "wet" days count as dry too
    result = compute(config=strict)
    assert result.dry_day_mm == 6.0
    assert result.longest_spells[0].days > 365  # everything is dry, so one very long spell
    assert "less than 6 mm" in " ".join(result.caveats)


def test_other_matches_are_disclosed(compute, make_location):
    other = make_location(id=2, name="Zhengzhoucun", admin1="Guangxi", population=None)
    assert any("also matched 1 other place" in c for c in compute(other_matches=[other]).caveats)
