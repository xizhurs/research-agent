"""Pure abstract handling: rebuild plain text from OpenAlex's inverted index, then truncate."""

from __future__ import annotations

from typing import Any

ELLIPSIS = "…"


def reconstruct_abstract(inverted_index: Any) -> str | None:
    """Rebuild abstract text from ``{word: [positions]}``.

    Returns None when there is no usable abstract (null, empty, or not the expected shape).
    Words are ordered by position; invalid positions are ignored.
    """
    if not isinstance(inverted_index, dict) or not inverted_index:
        return None
    placed: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and not isinstance(pos, bool) and pos >= 0:
                placed.append((pos, word))
    if not placed:
        return None
    placed.sort()
    text = " ".join(word for _, word in placed).strip()
    return text or None


def truncate_abstract(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate to at most ``max_chars`` characters (ellipsis included), on a word boundary.

    Returns ``(text, was_truncated)``.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if len(text) <= max_chars:
        return text, False
    budget = max_chars - len(ELLIPSIS)
    if budget <= 0:
        return ELLIPSIS[:max_chars], True
    cut = text[:budget]
    # Cut at the last space when the next character would have continued a word.
    if text[budget] != " " and " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut.rstrip() + ELLIPSIS, True
