"""Pure ENSO logic: parsing the RONI table and finding events. Hand-built series, no I/O.

Seasons are numbered consecutively from DJF of the first year, so index ``i`` of a series that
starts at DJF 2000 is season ``i`` of the record (0 = DJF 2000, 12 = DJF 2001).
"""

from dataclasses import replace
from pathlib import Path

import pytest

from research_agent.enso import InsufficientEnsoDataError
from research_agent.enso.analysis import (
    compute_analysis,
    parse_roni,
    strength_label,
)
from research_agent.enso.errors import EnsoApiError

FIXTURE = Path(__file__).parent / "fixtures" / "roni_1950_2001.txt"
GOOD = "SEAS   YR  ANOM\nDJF  2000 -0.10\nJFM  2000  0.55\nFMA  2000  1.00\n"


def analyze(make_seasons, enso_config, values, *, year=2000, slot=0, **kwargs):
    return compute_analysis(
        make_seasons(values, year=year, slot=slot),
        config=enso_config,
        source_url="https://example.test/RONI.ascii.txt",
        from_cache=False,
        **kwargs,
    )


# --- event rule -------------------------------------------------------------------------------


def test_hand_built_series_gives_the_expected_events(make_enso_analysis):
    a = make_enso_analysis()
    assert a.record_first == "DJF 2000" and a.record_last == "AMJ 2001"
    nino, nina = a.events
    assert (nino.phase, nino.first_season, nino.last_season, nino.seasons) == (
        "el_nino",
        "FMA 2000",
        "JAS 2000",
        6,
    )
    assert (nino.peak, nino.peak_season, nino.strength) == (1.4, "AMJ 2000", "moderate")
    assert (nina.phase, nina.first_season, nina.last_season, nina.seasons) == (
        "la_nina",
        "OND 2000",
        "FMA 2001",
        5,
    )
    assert (nina.peak, nina.peak_season, nina.strength) == (-1.8, "DJF 2001", "strong")
    assert not nino.ongoing and not nina.ongoing
    assert not nino.starts_at_record_start and a.pending is None


def test_five_consecutive_seasons_make_an_event_and_four_do_not(make_enso_analysis):
    a = make_enso_analysis([0] + [0.7] * 5 + [0] + [0.7] * 4 + [0])
    assert [(e.first_season, e.seasons) for e in a.events] == [("JFM 2000", 5)]


def test_a_value_of_exactly_the_threshold_counts(make_enso_analysis):
    nino = make_enso_analysis([0] + [0.5] * 5 + [0]).events
    nina = make_enso_analysis([0] + [-0.5] * 5 + [0]).events
    assert [e.phase for e in nino] == ["el_nino"] and [e.phase for e in nina] == ["la_nina"]


def test_a_value_just_below_the_threshold_breaks_the_run(make_enso_analysis):
    a = make_enso_analysis([0] + [0.5, 0.5, 0.49, 0.5, 0.5, 0.5] + [0])
    assert a.events == ()  # runs of 2 and 3 seasons


def test_the_other_sign_breaks_a_run_and_opposite_events_can_be_adjacent(make_enso_analysis):
    a = make_enso_analysis([0] + [0.9] * 5 + [-0.9] * 5 + [0])
    assert [(e.phase, e.seasons) for e in a.events] == [("el_nino", 5), ("la_nina", 5)]
    mixed = make_enso_analysis([0] + [0.9, 0.9, -0.9, 0.9, 0.9, 0.9] + [0])
    assert mixed.events == ()


def test_peak_is_the_largest_absolute_value_and_the_earliest_on_a_tie(make_enso_analysis):
    a = make_enso_analysis([0] + [0.6, 1.2, 1.2, 0.8, 0.7] + [0])
    (event,) = a.events
    assert (event.peak, event.peak_season) == (1.2, "FMA 2000")


def test_a_strength_is_set_for_events_only(make_enso_analysis):
    a = make_enso_analysis([0] + [0.7] * 3 + [0] * 3 + [0.7] * 6 + [0] * 3 + [0.7] * 2)
    assert [e.strength for e in a.events] == ["weak"]


@pytest.mark.parametrize(
    ("peak", "label"),
    [
        (0.5, "weak"),
        (0.99, "weak"),
        (1.0, "moderate"),
        (1.49, "moderate"),
        (1.5, "strong"),
        (1.99, "strong"),
        (2.0, "very strong"),
        (2.4, "very strong"),
        (-0.5, "weak"),
        (-1.0, "moderate"),
        (-1.5, "strong"),
        (-2.0, "very strong"),
    ],
)
def test_strength_classes_use_the_absolute_peak_and_inclusive_boundaries(
    enso_config, peak, label
):
    assert strength_label(peak, enso_config) == label


def test_threshold_and_minimum_length_come_from_the_config(
    make_seasons, enso_config
):
    strict = replace(enso_config, threshold=0.8, min_seasons=3, strength_moderate=1.0)
    a = compute_analysis(
        make_seasons([0, 0.7, 0.7, 0.7, 0, 0.9, 0.9, 0.9, 0]),
        config=strict,
        source_url="u",
        from_cache=False,
    )
    # Index 5 of a series starting at DJF is MJJ; 0.7 is below the 0.8 threshold.
    assert [(e.first_season, e.seasons) for e in a.events] == [("MJJ 2000", 3)]


def test_an_event_can_cross_a_year_boundary(make_seasons, enso_config):
    a = analyze(make_seasons, enso_config, [0] + [0.8] * 8 + [0], slot=8)
    (event,) = a.events
    assert (event.first_season, event.last_season, event.seasons) == ("SON 2000", "AMJ 2001", 8)


# --- the edges of the record ------------------------------------------------------------------


def test_an_event_at_the_end_of_the_record_is_ongoing(make_enso_analysis):
    a = make_enso_analysis([0] + [0.7] * 6)
    (event,) = a.events
    assert event.ongoing and a.pending is None
    assert any("record ends inside the El Nino event" in c for c in a.caveats)


def test_a_short_run_at_the_end_is_pending_and_not_an_event(make_enso_analysis):
    a = make_enso_analysis([0] + [0.7] * 3)
    assert a.events == ()
    assert a.pending is not None
    assert (a.pending.phase, a.pending.seasons, a.pending.ongoing) == ("el_nino", 3, True)
    assert (a.pending.first_season, a.pending.last_season) == ("JFM 2000", "MAM 2000")
    assert a.pending.strength is None
    assert any("not yet an event" in c and "3 consecutive" in c for c in a.caveats)


def test_pending_is_hidden_when_the_selection_excludes_the_end_of_the_record(
    make_enso_analysis,
):
    values = [0] * 20 + [0.7] * 3  # the record ends in 2001
    assert make_enso_analysis(values).pending is not None
    assert make_enso_analysis(values, end_year=2000).pending is None
    assert make_enso_analysis(values, phase="la_nina").pending is None


def test_a_run_that_begins_with_the_record_may_be_cut_at_the_start(make_enso_analysis):
    a = make_enso_analysis([0.8] * 6 + [0])
    (event,) = a.events
    assert event.starts_at_record_start and not event.ongoing
    assert any("record begins inside the El Nino event" in c for c in a.caveats)


def test_a_finished_short_run_is_neither_event_nor_pending(make_enso_analysis):
    a = make_enso_analysis([0] + [0.7] * 3 + [0] * 3)
    assert a.events == () and a.pending is None


# --- selecting by period and phase ------------------------------------------------------------

# One El Nino in JFM-MJJ 2000 and one in ASO 2004-DJF 2005, then a quiet tail (to 2005).
TWO_EVENTS = [0] + [0.8] * 5 + [0] * 50 + [0.8] * 5 + [0] * 3


def test_period_selects_events_by_their_start_year(make_enso_analysis):
    full = make_enso_analysis(TWO_EVENTS)
    assert [e.first_season for e in full.events] == ["JFM 2000", "ASO 2004"]
    assert [e.first_season for e in make_enso_analysis(TWO_EVENTS, start_year=2004).events] == [
        "ASO 2004"
    ]
    assert make_enso_analysis(TWO_EVENTS, start_year=2005).events == ()
    assert [e.first_season for e in make_enso_analysis(TWO_EVENTS, end_year=2000).events] == [
        "JFM 2000"
    ]


def test_a_window_never_cuts_an_event_short(make_enso_analysis):
    a = make_enso_analysis(TWO_EVENTS, end_year=2004)
    late = a.events[-1]
    assert (late.first_season, late.last_season, late.seasons) == ("ASO 2004", "DJF 2005", 5)


def test_phase_filter_keeps_only_that_phase(make_enso_analysis):
    assert [e.phase for e in make_enso_analysis(phase="la_nina").events] == ["la_nina"]
    assert [e.phase for e in make_enso_analysis(phase="el_nino").events] == ["el_nino"]
    assert len(make_enso_analysis(phase="both").events) == 2


def test_a_period_outside_the_record_is_an_error(make_enso_analysis):
    with pytest.raises(InsufficientEnsoDataError, match="outside"):
        make_enso_analysis(start_year=2010)
    with pytest.raises(InsufficientEnsoDataError, match="outside"):
        make_enso_analysis(end_year=1990)
    with pytest.raises(InsufficientEnsoDataError, match="year 1990"):
        make_enso_analysis(year=1990)


def test_a_period_partly_outside_the_record_is_clipped_and_reported(make_enso_analysis):
    a = make_enso_analysis(start_year=1990, end_year=2030)
    assert (a.first_year, a.last_year) == (2000, 2001)
    assert any("start year 1990 is before the record begins" in c for c in a.caveats)
    assert any("end year 2030 is after the record ends" in c for c in a.caveats)


# --- focus year -------------------------------------------------------------------------------


def test_focus_year_reports_values_and_overlapping_events(make_enso_analysis):
    f = make_enso_analysis(year=2000).focus
    assert [s for s, _ in f.values] == list(
        ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    )
    assert f.values[4] == ("AMJ", 1.4)
    assert [e.phase for e in f.events] == ["el_nino", "la_nina"]  # La Nina starts in OND 2000
    assert f.short_runs == ()


def test_focus_year_includes_an_event_that_began_the_year_before(make_enso_analysis):
    a = make_enso_analysis(year=2001)
    assert [e.phase for e in a.focus.events] == ["la_nina"]
    assert a.focus.values == (
        ("DJF", -1.8),
        ("JFM", -1.2),
        ("FMA", -0.7),
        ("MAM", 0.0),
        ("AMJ", 0.1),
    )
    assert any("2001 is incomplete in the record (last season AMJ 2001)" in c for c in a.caveats)


def test_focus_year_lists_short_runs_that_are_not_events(make_enso_analysis):
    f = make_enso_analysis([0.0, 0.7, 0.7, 0.7, 0.0], year=2000).focus
    assert f.events == ()
    assert [(r.phase, r.seasons, r.strength) for r in f.short_runs] == [("el_nino", 3, None)]


def test_a_neutral_year_has_no_events_or_short_runs(make_enso_analysis):
    f = make_enso_analysis([0.1] * 30, year=2001).focus
    assert f.events == () and f.short_runs == () and len(f.values) == 12


def test_focus_year_ignores_the_phase_filter(make_enso_analysis):
    a = make_enso_analysis(year=2000, phase="la_nina")
    assert [e.phase for e in a.events] == ["la_nina"]
    assert [e.phase for e in a.focus.events] == ["el_nino", "la_nina"]


# --- caveats ----------------------------------------------------------------------------------


def test_caveats_state_the_index_rule_convention_and_revision_risk(make_enso_analysis):
    text = " ".join(make_enso_analysis().caveats)
    assert "Relative Oceanic Nino Index" in text and "not station or rainfall data" in text
    assert "+/-0.5" in text and "at least 5 consecutive overlapping seasons" in text
    assert "exactly 0.5 counts" in text and "other indices" in text
    assert "common convention, not part of NOAA's definition" in text
    assert "may be revised" in text and "does not state" in text


# --- parsing ----------------------------------------------------------------------------------


def test_parse_reads_the_real_layout():
    seasons = parse_roni(GOOD)
    assert [(s.label, s.value) for s in seasons] == [
        ("DJF 2000", -0.1),
        ("JFM 2000", 0.55),
        ("FMA 2000", 1.0),
    ]


def test_parse_ignores_blank_lines_and_extra_spaces():
    text = "\n SEAS YR   ANOM \n\nDJF   2000   -0.10\n\n\nJFM 2000 0.55\n\n"
    assert len(parse_roni(text)) == 2


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "header"),
        ("SEAS YR TOTAL ANOM\nDJF 2000 25.0 -0.1\n", "header"),
        ("SEAS YR ANOM\n", "no data rows"),
        ("SEAS YR ANOM\nDJF 2000\n", "bad row"),
        ("SEAS YR ANOM\nDJF 2000 0.1 extra\n", "bad row"),
        ("SEAS YR ANOM\nXYZ 2000 0.1\n", "bad row"),
        ("SEAS YR ANOM\nDJF year 0.1\n", "bad number"),
        ("SEAS YR ANOM\nDJF 2000 high\n", "bad number"),
        ("SEAS YR ANOM\nDJF 2000 nan\n", "implausible"),
        ("SEAS YR ANOM\nDJF 2000 inf\n", "implausible"),
        ("SEAS YR ANOM\nDJF 2000 -99.9\n", "implausible"),
        ("SEAS YR ANOM\nDJF 2000 0.1\nFMA 2000 0.2\n", "does not follow"),  # a season is missing
        ("SEAS YR ANOM\nDJF 2000 0.1\nDJF 2000 0.1\n", "does not follow"),  # duplicate
        ("SEAS YR ANOM\nJFM 2000 0.1\nDJF 2000 0.1\n", "does not follow"),  # out of order
        ("SEAS YR ANOM\nNDJ 2000 0.1\nJFM 2001 0.2\n", "does not follow"),  # gap over New Year
    ],
)
def test_parse_rejects_anything_unexpected(text, message):
    with pytest.raises(EnsoApiError, match=message):
        parse_roni(text)


def test_parse_accepts_the_new_year_rollover():
    seasons = parse_roni("SEAS YR ANOM\nNDJ 2000 0.1\nDJF 2001 0.2\n")
    assert [s.label for s in seasons] == ["NDJ 2000", "DJF 2001"]


# --- a trimmed copy of the real file (1950-2001) ----------------------------------------------


@pytest.fixture(scope="module")
def real_events():
    from research_agent.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.toml").enso
    return compute_analysis(
        parse_roni(FIXTURE.read_text(encoding="utf-8")),
        config=cfg,
        source_url="fixture",
        from_cache=False,
    )


def summary(event):
    return (event.phase, event.first_season, event.last_season, event.seasons, event.strength)


def test_real_file_snapshot_of_well_known_events(real_events):
    """Snapshot of what this code computes from the recorded real file (see CLAUDE.md for the
    cross-check against NOAA's published table)."""
    found = {summary(e): (e.peak, e.peak_season) for e in real_events.events}
    assert found[("el_nino", "AMJ 1982", "AMJ 1983", 13, "very strong")] == (2.40, "DJF 1983")
    assert found[("el_nino", "AMJ 1997", "MAM 1998", 12, "very strong")] == (2.28, "NDJ 1997")
    assert found[("la_nina", "MAM 1988", "AMJ 1989", 14, "strong")] == (-1.95, "OND 1988")
    assert found[("la_nina", "JJA 1998", "MJJ 2000", 24, "strong")] == (-1.61, "NDJ 1998")


def test_real_file_counts_and_edges(real_events):
    phases = [e.phase for e in real_events.events]
    assert (phases.count("el_nino"), phases.count("la_nina")) == (15, 12)
    first = real_events.events[0]
    assert (first.phase, first.first_season, first.starts_at_record_start) == (
        "la_nina",
        "DJF 1950",
        True,
    )
    assert not any(e.ongoing for e in real_events.events) and real_events.pending is None
