"""Plain data returned by retrieval. No I/O, no LLM concepts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Work:
    """One scholarly work, reduced to the fields an answer can cite.

    ``id`` is the short OpenAlex ID (e.g. ``W4388333579``) used as the citation key.
    """

    id: str
    url: str
    title: str | None
    year: int | None
    date: str | None
    authors: tuple[str, ...]
    venue: str | None
    doi: str | None
    type: str | None
    language: str | None
    cited_by_count: int | None
    abstract: str | None
    abstract_truncated: bool


@dataclass(frozen=True)
class SearchResult:
    """Outcome of one search.

    ``total_count`` is how many works matched in OpenAlex overall, not how many were returned.
    ``skipped_records`` counts records in the response that could not be parsed; they are never
    dropped silently.
    """

    query: str
    year_from: int | None
    works: tuple[Work, ...]
    total_count: int
    skipped_records: int
    from_cache: bool

    @property
    def missing_abstract_count(self) -> int:
        return sum(1 for w in self.works if w.abstract is None)
