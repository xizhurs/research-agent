import json
from pathlib import Path

import pytest

from research_agent.retrieval.errors import ApiError
from research_agent.retrieval.openalex import (
    SELECT_FIELDS,
    build_params,
    parse_response,
    parse_work,
)

FIXTURE = Path(__file__).parent / "fixtures" / "openalex_search.json"


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_build_params_basic():
    p = build_params("land cover", None, 8)
    assert p == {
        "search.title_and_abstract": "land cover",
        "per_page": "8",
        "select": SELECT_FIELDS,
    }


def test_build_params_year_filter_and_special_chars_stay_out_of_filter():
    p = build_params("a,b|c:d", 2020, 5)
    assert p["filter"] == "from_publication_date:2020-01-01"
    assert p["search.title_and_abstract"] == "a,b|c:d"


def test_parse_recorded_response(payload):
    works, total, skipped = parse_response(payload, 1200)
    assert total == payload["meta"]["count"]
    assert skipped == 0
    assert len(works) == 3
    w = works[0]
    assert w.id == "W4388333579"
    assert w.url == "https://openalex.org/W4388333579"
    assert w.title.startswith("Land Use and Land Cover Classification")
    assert w.year == 2023
    assert w.venue == "Sensors"
    assert "Shengyu Zhao" in w.authors
    assert w.abstract and w.abstract.startswith("As one of")
    assert len(w.abstract) <= 1200


def test_abstract_limit_applied_at_parse_time(payload):
    works, _, _ = parse_response(payload, 100)
    for w in works:
        assert w.abstract is not None
        assert len(w.abstract) <= 100
        assert w.abstract_truncated is True


def test_parse_work_tolerates_missing_optional_fields():
    w = parse_work({"id": "https://openalex.org/W1"}, 100)
    assert w is not None
    assert w.id == "W1"
    assert w.title is None and w.year is None and w.abstract is None
    assert w.authors == () and w.venue is None and w.doi is None
    assert w.abstract_truncated is False


def test_parse_work_null_nested_objects():
    record = {
        "id": "https://openalex.org/W2",
        "primary_location": {"source": None},
        "authorships": [{"author": None}, None, {"author": {"display_name": "A B"}}],
        "abstract_inverted_index": None,
        "title": "   ",
    }
    w = parse_work(record, 100)
    assert w is not None
    assert w.venue is None
    assert w.authors == ("A B",)
    assert w.abstract is None
    assert w.title is None


@pytest.mark.parametrize(
    "record",
    [None, "x", {}, {"id": 5}, {"id": "https://openalex.org/A123"}, {"id": "W"}],
)
def test_parse_work_rejects_records_without_work_id(record):
    assert parse_work(record, 100) is None


def test_unparseable_records_are_counted_not_dropped_silently(payload):
    payload["results"].append({"no": "id"})
    works, _, skipped = parse_response(payload, 1200)
    assert len(works) == 3
    assert skipped == 1


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"results": []},
        {"meta": {"count": 1}},
        {"results": {}, "meta": {"count": 1}},
        {"results": [], "meta": {}},
        {"results": [], "meta": {"count": "3"}},
    ],
)
def test_malformed_envelope_raises(bad):
    with pytest.raises(ApiError):
        parse_response(bad, 100)
