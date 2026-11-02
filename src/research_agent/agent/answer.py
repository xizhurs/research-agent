"""The structured ``Answer`` and parsing of the model's final JSON reply. Pure.

The model returns JSON text; this module turns it into typed data. References are built from
works the agent actually retrieved, never from text the model wrote, and citations to works
that were never retrieved are rejected (see ``validate_citations``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from research_agent.rainfall import RainfallRecord
from research_agent.retrieval import Work

ClaimKind = Literal["evidence", "interpretation"]
_KINDS: tuple[ClaimKind, ...] = ("evidence", "interpretation")
_FENCE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)


class AnswerFormatError(ValueError):
    """The model's reply is not the required JSON structure."""


@dataclass(frozen=True)
class Claim:
    text: str
    kind: ClaimKind
    sources: tuple[str, ...]  # short IDs: works ("W4388333579") or rainfall analyses ("R1")


@dataclass(frozen=True)
class Answer:
    claims: tuple[Claim, ...]
    references: tuple[Work, ...]  # cited works that were retrieved, in order of first citation
    limitations: tuple[str, ...]
    # IDs the model cited that were never retrieved; they are removed from the claims.
    rejected_citations: tuple[str, ...] = ()
    # False when the model ignored the required format and the answer is a plain-text fallback.
    structured: bool = True
    # Numerical evidence: cited rainfall analyses, in order of first citation.
    data_references: tuple[RainfallRecord, ...] = ()


def _load_json(text: str) -> Any:
    candidate = text.strip()
    fenced = _FENCE.match(candidate)
    if fenced:
        candidate = fenced.group(1)
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    # Tolerate stray prose around the object.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except ValueError:
            pass
    raise AnswerFormatError("reply is not valid JSON")


def _short_id(raw: str) -> str:
    return raw.strip().rstrip("/").rsplit("/", 1)[-1]


def _parse_claim(raw: Any) -> Claim:
    if not isinstance(raw, dict):
        raise AnswerFormatError("a claim is not an object")
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        raise AnswerFormatError("a claim has no text")
    kind = raw.get("kind")
    if kind not in _KINDS:
        raise AnswerFormatError("a claim has an invalid 'kind'")
    sources = raw.get("sources", [])
    if not isinstance(sources, list) or not all(isinstance(s, str) and s.strip() for s in sources):
        raise AnswerFormatError("a claim has invalid 'sources'")
    return Claim(text=text.strip(), kind=kind, sources=tuple(_short_id(s) for s in sources))


def _dedupe(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(i for i in items if i))


def _references(claims: Sequence[Claim], evidence: Mapping[str, Work]) -> tuple[Work, ...]:
    ordered = dict.fromkeys(s for c in claims for s in c.sources)
    return tuple(evidence[i] for i in ordered if i in evidence)


def _data_references(
    claims: Sequence[Claim], analyses: Mapping[str, RainfallRecord]
) -> tuple[RainfallRecord, ...]:
    ordered = dict.fromkeys(s for c in claims for s in c.sources)
    return tuple(analyses[i] for i in ordered if i in analyses)


def validate_citations(
    claims: Sequence[Claim],
    evidence: Mapping[str, Work],
    analyses: Mapping[str, RainfallRecord] | None = None,
) -> tuple[tuple[Claim, ...], tuple[str, ...], list[str]]:
    """Enforce that every citation points to a work or rainfall analysis that was retrieved.

    - Citations to anything not retrieved or computed in this run are removed and returned as
      ``rejected``.
    - An "evidence" claim left with no valid citation is downgraded to "interpretation":
      nothing supports it as evidence, so it must not be presented as such.
    Returns ``(claims, rejected_ids, limitation_notes)``; every change is reported.
    """
    known = {**evidence, **(analyses or {})}
    rejected: dict[str, None] = {}
    downgraded = 0
    checked: list[Claim] = []
    for claim in claims:
        valid = tuple(dict.fromkeys(s for s in claim.sources if s in known))
        rejected.update(dict.fromkeys(s for s in claim.sources if s not in known))
        kind: ClaimKind = claim.kind
        if kind == "evidence" and not valid:
            kind = "interpretation"
            downgraded += 1
        checked.append(Claim(text=claim.text, kind=kind, sources=valid))

    notes: list[str] = []
    if rejected:
        notes.append(
            f"{len(rejected)} cited ID(s) were not among the retrieved works or computed "
            f"analyses and were removed: {', '.join(rejected)}."
        )
    if downgraded:
        notes.append(
            f"{downgraded} claim(s) labelled as evidence cited no retrieved work or analysis and "
            "are shown as interpretation."
        )
    return tuple(checked), tuple(rejected), notes


def parse_answer(
    text: str,
    evidence: Mapping[str, Work],
    notes: Sequence[str] = (),
    analyses: Mapping[str, RainfallRecord] | None = None,
) -> Answer:
    """Parse the model's JSON reply. Raises AnswerFormatError if the structure is wrong.

    ``notes`` are limitations added by the code (e.g. a budget was hit); they are appended
    after the model's own limitations.
    """
    data = _load_json(text)
    if not isinstance(data, dict):
        raise AnswerFormatError("top-level JSON is not an object")
    raw_claims = data.get("claims")
    if not isinstance(raw_claims, list):
        raise AnswerFormatError("'claims' is missing or not a list")
    parsed = tuple(_parse_claim(c) for c in raw_claims)
    raw_limits = data.get("limitations", [])
    if not isinstance(raw_limits, list) or not all(isinstance(x, str) for x in raw_limits):
        raise AnswerFormatError("'limitations' is not a list of strings")
    claims, rejected, citation_notes = validate_citations(parsed, evidence, analyses)
    return Answer(
        claims=claims,
        references=_references(claims, evidence),
        data_references=_data_references(claims, analyses or {}),
        limitations=_dedupe([*(x.strip() for x in raw_limits), *notes, *citation_notes]),
        rejected_citations=rejected,
    )


def fallback_answer(text: str | None, reason: str, notes: Sequence[str] = ()) -> Answer:
    """An honest unstructured answer for when the model ignored the required format.

    The raw text is kept as a single, unsourced interpretation so nothing is silently lost.
    """
    body = (text or "").strip() or "(the model returned no answer text)"
    return Answer(
        claims=(Claim(text=body, kind="interpretation", sources=()),),
        references=(),
        limitations=_dedupe(
            [
                f"The model did not return the required structured format ({reason}); this "
                "answer is unstructured and its statements are not linked to sources.",
                *notes,
            ]
        ),
        structured=False,
    )


def build_answer(
    text: str | None,
    evidence: Mapping[str, Work],
    notes: Sequence[str],
    analyses: Mapping[str, RainfallRecord] | None = None,
) -> Answer:
    """Parse ``text`` if possible, otherwise fall back to an unstructured answer."""
    if text is None:
        return fallback_answer(None, "empty reply", notes)
    try:
        return parse_answer(text, evidence, notes, analyses)
    except AnswerFormatError as exc:
        return fallback_answer(text, str(exc), notes)
