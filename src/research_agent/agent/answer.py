"""The structured ``Answer`` and parsing of the model's final JSON reply. Pure.

The model returns JSON text; this module turns it into typed data. References are built from
works the agent actually retrieved, never from text the model wrote. Checking that cited IDs
were really retrieved is a later milestone (M4); here unknown IDs are kept as given.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

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
    sources: tuple[str, ...]  # short work IDs, e.g. "W4388333579"


@dataclass(frozen=True)
class Answer:
    claims: tuple[Claim, ...]
    references: tuple[Work, ...]  # cited works that were retrieved, in order of first citation
    limitations: tuple[str, ...]


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


def parse_answer(text: str, evidence: Mapping[str, Work], notes: Sequence[str] = ()) -> Answer:
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
    claims = tuple(_parse_claim(c) for c in raw_claims)
    raw_limits = data.get("limitations", [])
    if not isinstance(raw_limits, list) or not all(isinstance(x, str) for x in raw_limits):
        raise AnswerFormatError("'limitations' is not a list of strings")
    return Answer(
        claims=claims,
        references=_references(claims, evidence),
        limitations=_dedupe([*(x.strip() for x in raw_limits), *notes]),
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
    )


def build_answer(text: str | None, evidence: Mapping[str, Work], notes: Sequence[str]) -> Answer:
    """Parse ``text`` if possible, otherwise fall back to an unstructured answer."""
    if text is None:
        return fallback_answer(None, "empty reply", notes)
    try:
        return parse_answer(text, evidence, notes)
    except AnswerFormatError as exc:
        return fallback_answer(text, str(exc), notes)
