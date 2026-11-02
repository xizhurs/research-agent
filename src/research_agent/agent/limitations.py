"""Limitations the code derives from what was actually retrieved. Pure and deterministic.

These are added to every answer next to the model's own limitations, so a reader always sees
what the evidence can and cannot support, even when the model says nothing about it.
Budget hits, search failures and citation problems are reported where they happen (loop and
answer modules); this module covers the evidence itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from research_agent.config import Limits, ReportingConfig
from research_agent.rainfall import RainfallAnalysis
from research_agent.retrieval import Work

ALWAYS = (
    "Evidence is limited to titles, abstracts and metadata from OpenAlex; full texts and other "
    "databases were not searched."
)


@dataclass(frozen=True)
class SearchRecord:
    """One search that ran successfully."""

    query: str
    year_from: int | None
    total_count: int  # matches in OpenAlex overall
    returned: int  # works returned (after the per-search cap)
    shown: int  # works the model actually saw in full (context budget may hide some)


def evidence_limitations(
    searches: Sequence[SearchRecord],
    evidence: Mapping[str, Work],
    *,
    limits: Limits,
    reporting: ReportingConfig,
    today: date,
    analyses: Sequence[RainfallAnalysis] = (),
    rainfall_calls: int = 0,
) -> list[str]:
    """Limitations implied by the searches, the works seen and the rainfall analyses shown.

    The literature caveats apply when a search ran, or when nothing ran at all; a rainfall-only
    answer is not described as OpenAlex evidence.
    """
    notes: list[str] = []
    if searches or not rainfall_calls:
        notes.extend(_literature_limitations(searches, evidence, limits, reporting, today))
    if rainfall_calls and not analyses:
        notes.append(
            "A rainfall analysis was requested but produced no usable result, so no numerical "
            "evidence backs this answer."
        )
    for analysis in analyses:
        notes.extend(analysis.caveats)  # identical sentences are removed when the answer is built
    return notes


def _literature_limitations(
    searches: Sequence[SearchRecord],
    evidence: Mapping[str, Work],
    limits: Limits,
    reporting: ReportingConfig,
    today: date,
) -> list[str]:
    notes = [ALWAYS]
    works = list(evidence.values())

    if not searches:
        notes.append(
            "No literature search was run, so this answer is not backed by retrieved evidence."
        )
        return notes
    if not works:
        notes.append(
            "The searches returned no usable works, so this answer is not backed by retrieved "
            "evidence."
        )
        return notes

    if len(works) < reporting.thin_evidence_works:
        notes.append(
            f"Only {len(works)} distinct work(s) were retrieved; the evidence base is thin."
        )

    widest = max(searches, key=lambda s: s.total_count - s.returned)
    if widest.total_count > widest.returned:
        notes.append(
            "Only the top-ranked results of each search were read (for example "
            f'{widest.returned} of {widest.total_count:,} matches for "{widest.query}"); '
            "relevant work may have been missed."
        )

    cutoffs = [s.year_from for s in searches]
    if all(year is not None for year in cutoffs):
        notes.append(f"All searches excluded works published before {min(cutoffs)}.")

    missing = sum(1 for w in works if w.abstract is None)
    if missing:
        notes.append(
            f"{missing} of {len(works)} retrieved work(s) have no abstract and could be judged "
            "by title only."
        )
    truncated = sum(1 for w in works if w.abstract_truncated)
    if truncated:
        notes.append(
            f"{truncated} abstract(s) were truncated to {limits.abstract_max_chars} characters."
        )

    years = [w.year for w in works if w.year is not None]
    if years and today.year - max(years) >= reporting.stale_after_years:
        notes.append(
            f"The most recent retrieved work is from {max(years)}; newer research may exist."
        )
    return notes
