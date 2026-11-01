import pytest

from research_agent.agent.budget import estimate_tokens
from research_agent.agent.tools import (
    SEARCH_TOOL,
    ToolArgumentError,
    format_results,
    parse_search_args,
)


def test_tool_schema_requires_query_only():
    assert SEARCH_TOOL.name == "search_literature"
    assert SEARCH_TOOL.parameters["required"] == ["query"]
    assert set(SEARCH_TOOL.parameters["properties"]) == {"query", "year_from", "max_results"}


def test_parse_valid_args():
    a = parse_search_args({"query": "land cover", "year_from": 2020, "max_results": 3})
    assert (a.query, a.year_from, a.max_results) == ("land cover", 2020, 3)


def test_optional_args_may_be_missing_or_null():
    a = parse_search_args({"query": "q", "year_from": None})
    assert a.year_from is None and a.max_results is None


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": 5},
        {"query": "q", "year_from": "2020"},
        {"query": "q", "year_from": True},
        {"query": "q", "max_results": 2.5},
    ],
)
def test_bad_args_raise_with_model_safe_message(arguments):
    with pytest.raises(ToolArgumentError):
        parse_search_args(arguments)


def test_format_includes_id_metadata_and_abstract(make_work, make_result):
    w = make_work("W7", title="Crop maps", abstract="We map crops.", authors=("A", "B", "C", "D"))
    out = format_results(make_result([w], query="crops", total=42), 5000, set())
    assert 'Search "crops"' in out.text and "42 matches" in out.text
    assert "[W7] Crop maps (A, B, C et al., Journal, 2023, cited by 5)" in out.text
    assert "Abstract: We map crops." in out.text
    assert out.shown == (w,)
    assert out.omitted == 0
    assert 0 < out.context_tokens <= estimate_tokens(out.text) + 3


def test_missing_abstract_is_stated_not_invented(make_work, make_result):
    w = make_work("W1", abstract=None)
    out = format_results(make_result([w]), 5000, set())
    assert "Abstract: (not available)" in out.text


def test_no_results_message(make_result):
    out = format_results(make_result([]), 5000, set())
    assert "No results." in out.text
    assert out.shown == ()


def test_context_budget_omits_in_rank_order(make_work, make_result):
    works = [make_work(f"W{i}", abstract="x" * 400) for i in range(1, 5)]  # ~100+ tokens each
    out = format_results(make_result(works), 260, set())
    assert [w.id for w in out.shown] == ["W1", "W2"]
    assert out.omitted == 2
    assert "2 more result(s) omitted" in out.text
    assert "W3" not in out.text.split("omitted")[0]
    assert out.context_tokens <= 260


def test_zero_context_left_shows_nothing(make_work, make_result):
    out = format_results(make_result([make_work("W1")]), 0, set())
    assert out.shown == ()
    assert out.omitted == 1


def test_already_seen_works_are_listed_by_id_only(make_work, make_result):
    w1, w2 = make_work("W1", abstract="long abstract " * 20), make_work("W2")
    out = format_results(make_result([w1, w2]), 5000, {"W1"})
    assert "[W1] (already shown in an earlier search)" in out.text
    assert "long abstract" not in out.text
    assert out.shown == (w2,)  # only newly shown works become new evidence
