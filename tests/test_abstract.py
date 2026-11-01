import pytest

from research_agent.retrieval.abstract import reconstruct_abstract, truncate_abstract


def test_reconstruct_orders_by_position_and_repeats_words():
    index = {"the": [0, 3], "cat": [1], "sat": [2], "mat": [4]}
    assert reconstruct_abstract(index) == "the cat sat the mat"


def test_reconstruct_handles_gaps_in_positions():
    assert reconstruct_abstract({"a": [0], "b": [5]}) == "a b"


@pytest.mark.parametrize("bad", [None, {}, [], "text", {"a": "not a list"}, {"a": [-1, True, "x"]}])
def test_reconstruct_returns_none_when_unusable(bad):
    assert reconstruct_abstract(bad) is None


def test_reconstruct_ignores_only_the_invalid_parts():
    assert reconstruct_abstract({"ok": [0], "bad": "x", 5: [1]}) == "ok"


def test_truncate_short_text_unchanged():
    assert truncate_abstract("short text", 50) == ("short text", False)


def test_truncate_exact_length_unchanged():
    assert truncate_abstract("12345", 5) == ("12345", False)


def test_truncate_cuts_on_word_boundary_within_limit():
    text, truncated = truncate_abstract("alpha beta gamma delta", 14)
    assert truncated is True
    assert text == "alpha beta…"
    assert len(text) <= 14


def test_truncate_never_exceeds_limit():
    text = "word " * 100
    for limit in (1, 2, 3, 10, 57):
        out, truncated = truncate_abstract(text, limit)
        assert truncated is True
        assert len(out) <= limit


def test_truncate_single_long_word_cut_hard():
    out, truncated = truncate_abstract("a" * 50, 10)
    assert truncated is True
    assert out == "a" * 9 + "…"


def test_truncate_rejects_invalid_limit():
    with pytest.raises(ValueError):
        truncate_abstract("x", 0)
