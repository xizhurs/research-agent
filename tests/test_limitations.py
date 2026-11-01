from dataclasses import replace
from datetime import date

import pytest

from research_agent.agent.limitations import ALWAYS, SearchRecord, evidence_limitations

TODAY = date(2026, 11, 1)


def rec(query="land cover", year_from=None, total=10, returned=10, shown=10):
    return SearchRecord(query, year_from, total, returned, shown)


def notes(searches, works, limits, reporting, today=TODAY):
    evidence = {w.id: w for w in works}
    return evidence_limitations(
        searches, evidence, limits=limits, reporting=reporting, today=today
    )


@pytest.fixture
def good_works(make_work):
    return [make_work(f"W{i}", year=2025) for i in range(1, 6)]


def test_always_present_even_for_a_clean_run(good_works, limits, reporting):
    out = notes([rec()], good_works, limits, reporting)
    assert out == [ALWAYS]  # nothing else applies: enough recent works, no filters, no gaps


def test_no_search_run(limits, reporting):
    out = notes([], [], limits, reporting)
    assert out[0] == ALWAYS
    assert any("No literature search was run" in x for x in out)
    assert len(out) == 2


def test_searches_returned_nothing_usable(limits, reporting):
    out = notes([rec(total=0, returned=0, shown=0)], [], limits, reporting)
    assert any("returned no usable works" in x for x in out)
    assert len(out) == 2


def test_thin_evidence_boundary(make_work, limits, reporting):
    threshold = reporting.thin_evidence_works  # 3
    below = [make_work(f"W{i}", year=2025) for i in range(1, threshold)]
    at = [make_work(f"W{i}", year=2025) for i in range(1, threshold + 1)]
    assert any("evidence base is thin" in x for x in notes([rec()], below, limits, reporting))
    assert not any("thin" in x for x in notes([rec()], at, limits, reporting))


def test_thin_message_counts_distinct_works(make_work, limits, reporting):
    out = notes([rec()], [make_work("W1", year=2025)], limits, reporting)
    assert any("Only 1 distinct work(s)" in x for x in out)


def test_top_ranked_only_message_uses_the_widest_gap(good_works, limits, reporting):
    searches = [rec("narrow", total=9, returned=8), rec("broad", total=2714, returned=8)]
    out = notes(searches, good_works, limits, reporting)
    (msg,) = [x for x in out if "top-ranked" in x]
    assert '8 of 2,714 matches for "broad"' in msg


def test_no_top_ranked_message_when_everything_was_read(good_works, limits, reporting):
    out = notes([rec(total=5, returned=5)], good_works, limits, reporting)
    assert not any("top-ranked" in x for x in out)


def test_year_filter_reported_only_if_all_searches_used_one(good_works, limits, reporting):
    both = [rec(year_from=2022), rec(year_from=2020)]
    out = notes(both, good_works, limits, reporting)
    assert any("excluded works published before 2020" in x for x in out)  # least restrictive
    mixed = [rec(year_from=2022), rec(year_from=None)]
    assert not any("excluded works" in x for x in notes(mixed, good_works, limits, reporting))


def test_missing_abstracts_counted(make_work, limits, reporting):
    works = [make_work("W1", year=2025, abstract=None)] + [
        make_work(f"W{i}", year=2025) for i in (2, 3, 4)
    ]
    out = notes([rec()], works, limits, reporting)
    assert any("1 of 4 retrieved work(s) have no abstract" in x for x in out)


def test_truncated_abstracts_counted_with_the_configured_limit(make_work, limits, reporting):
    works = [replace(make_work(f"W{i}", year=2025), abstract_truncated=(i <= 2)) for i in (1, 2, 3)]
    out = notes([rec()], works, limits, reporting)
    assert any(
        f"2 abstract(s) were truncated to {limits.abstract_max_chars} characters" in x for x in out
    )


def test_outdated_evidence_boundary(make_work, limits, reporting):
    gap = reporting.stale_after_years  # 3
    fresh = [make_work(f"W{i}", year=TODAY.year - gap + 1) for i in (1, 2, 3)]
    stale = [make_work(f"W{i}", year=TODAY.year - gap) for i in (1, 2, 3)]
    assert not any("newer research" in x for x in notes([rec()], fresh, limits, reporting))
    out = notes([rec()], stale, limits, reporting)
    assert any(f"most recent retrieved work is from {TODAY.year - gap}" in x for x in out)


def test_outdated_uses_the_newest_work_not_the_oldest(make_work, limits, reporting):
    works = [make_work("W1", year=2010), make_work("W2", year=2025), make_work("W3", year=2020)]
    assert not any("newer research" in x for x in notes([rec()], works, limits, reporting))


def test_works_without_a_year_do_not_break_the_staleness_check(make_work, limits, reporting):
    works = [make_work(f"W{i}", year=None) for i in (1, 2, 3)]
    assert not any("newer research" in x for x in notes([rec()], works, limits, reporting))
