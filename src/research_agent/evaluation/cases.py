"""Evaluation cases: a TOML file of questions and the *properties* each answer must have.

Answers are model output and vary between runs, so cases specify expectations (searched,
cited enough works, mentions a topic keyword...) rather than exact text. ``split`` separates the
development cases from the held-out test cases; ``category`` groups them in reports.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_agent.agent.tools import (
    ENSO_TOOL_NAME,
    RAINFALL_TOOL_NAME,
    SEARCH_TOOL_NAME,
    TEMPERATURE_TOOL_NAME,
)

KNOWN_TOOLS = frozenset(
    {SEARCH_TOOL_NAME, RAINFALL_TOOL_NAME, TEMPERATURE_TOOL_NAME, ENSO_TOOL_NAME}
)
METRICS = ("wet", "dry", "yearly", "monthly", "hot", "cold")
SPLITS = ("dev", "test")
CATEGORIES = ("literature", "safety", "rainfall", "dry", "enso", "scale", "temperature")

_ALLOWED_KEYS = {
    "id",
    "question",
    "note",
    "split",
    "category",
    "min_searches",
    "min_references",
    "min_rainfall_calls",
    "min_data_references",
    "expect_metric",
    "expect_tool",
    "forbid_tool",
    "max_evidence_claims",
    "keywords_any",
    "forbidden_phrases",
    "expect_numbers",
    "max_tool_calls",
    "key_points",
    "min_key_points",
}
DEFAULT_NUMBER_TOLERANCE = 0.05
_WORK_ID = re.compile(r"^W\d+$")


class EvalError(ValueError):
    """The case file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ExpectedNumber:
    """A verified value that must appear in a tool result and in a claim (``checks``)."""

    value: float
    tolerance: float = DEFAULT_NUMBER_TOLERANCE


@dataclass(frozen=True)
class KeyPoint:
    """A verified fact an answer should cover: covered when any term appears (lower-case) in a
    claim that cites something. Curated once, never generated per run."""

    text: str
    any_terms: tuple[str, ...]


@dataclass(frozen=True)
class Qrels:
    """Relevance labels for the works a case's searches may show (``evals/qrels.toml``)."""

    relevant: frozenset[str] = frozenset()
    not_relevant: frozenset[str] = frozenset()


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    min_searches: int = 0
    min_references: int = 0
    max_evidence_claims: int | None = None
    keywords_any: tuple[str, ...] = ()  # at least one must appear in the claims (case-insensitive)
    forbidden_phrases: tuple[str, ...] = ()  # none may appear in the claims (case-insensitive)
    min_rainfall_calls: int = 0
    min_data_references: int = 0  # cited rainfall analyses (R1, R2...)
    # The rainfall tool must have been requested with this metric ("wet" or "dry") at least once.
    # Checked on the trace, so it judges the model's choice even if the tool refused the call.
    expect_metric: str | None = None
    # A tool name that must (expect_tool) or must not (forbid_tool) appear in the trace, e.g.
    # "analyze_enso_events". Like expect_metric, this judges the model's choice of tool.
    expect_tool: str | None = None
    forbid_tool: str | None = None
    split: str = "dev"  # "test" cases are held out: run them only to measure, never to tune
    category: str | None = None
    expect_numbers: tuple[ExpectedNumber, ...] = ()
    max_tool_calls: int | None = None  # tool calls requested (all turns of the trace)
    key_points: tuple[KeyPoint, ...] = ()
    min_key_points: int = 0  # key points that must be covered (0: coverage is only reported)


def _int_field(raw: dict[str, Any], key: str, default: int | None, case_id: str) -> int | None:
    value = raw.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvalError(f"case {case_id!r}: {key} must be an integer >= 0, got {value!r}")
    return value


def _choice(raw: dict[str, Any], key: str, allowed: tuple[str, ...], case_id: str) -> str | None:
    value = raw.get(key)
    if value is not None and value not in allowed:
        raise EvalError(f"case {case_id!r}: {key} must be one of {list(allowed)}")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _tables(raw: dict[str, Any], key: str, allowed: set[str], case_id: str) -> list[dict]:
    items = raw.get(key, [])
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise EvalError(f"case {case_id!r}: {key} must be a list of tables")
    for item in items:
        unknown = item.keys() - allowed
        if unknown:
            raise EvalError(f"case {case_id!r}: {key} has unknown keys {sorted(unknown)}")
    return items


def _expected_numbers(raw: dict[str, Any], case_id: str) -> tuple[ExpectedNumber, ...]:
    out = []
    for item in _tables(raw, "expect_numbers", {"value", "tolerance"}, case_id):
        value = item.get("value")
        tolerance = item.get("tolerance", DEFAULT_NUMBER_TOLERANCE)
        if not _is_number(value):
            raise EvalError(f"case {case_id!r}: expect_numbers needs a numeric 'value'")
        if not _is_number(tolerance) or tolerance < 0:
            raise EvalError(f"case {case_id!r}: expect_numbers 'tolerance' must be >= 0")
        out.append(ExpectedNumber(float(value), float(tolerance)))
    return tuple(out)


def _key_points(raw: dict[str, Any], case_id: str) -> tuple[KeyPoint, ...]:
    out = []
    for item in _tables(raw, "key_points", {"text", "any_terms"}, case_id):
        text, terms = item.get("text"), item.get("any_terms")
        if not isinstance(text, str) or not text.strip():
            raise EvalError(f"case {case_id!r}: every key point needs a 'text'")
        if not isinstance(terms, list) or not terms or not all(
            isinstance(t, str) and t.strip() for t in terms
        ):
            raise EvalError(f"case {case_id!r}: key point 'any_terms' must be non-empty strings")
        out.append(KeyPoint(text.strip(), tuple(t.strip().lower() for t in terms)))
    return tuple(out)


def _parse_case(raw: Any, index: int) -> EvalCase:
    if not isinstance(raw, dict):
        raise EvalError(f"case #{index}: not a table")
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise EvalError(f"case #{index}: 'id' is required")
    unknown = raw.keys() - _ALLOWED_KEYS
    if unknown:
        raise EvalError(f"case {case_id!r}: unknown keys {sorted(unknown)}")
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise EvalError(f"case {case_id!r}: 'question' is required")
    expect_metric = _choice(raw, "expect_metric", METRICS, case_id)
    tools: dict[str, str | None] = {}
    for key in ("expect_tool", "forbid_tool"):
        name = raw.get(key)
        if name is not None and not (isinstance(name, str) and name in KNOWN_TOOLS):
            raise EvalError(f"case {case_id!r}: {key} must be one of {sorted(KNOWN_TOOLS)}")
        tools[key] = name
    phrase_lists: dict[str, tuple[str, ...]] = {}
    for key in ("keywords_any", "forbidden_phrases"):
        values = raw.get(key, [])
        if not isinstance(values, list) or not all(
            isinstance(v, str) and v.strip() for v in values
        ):
            raise EvalError(f"case {case_id!r}: {key} must be a list of non-empty strings")
        phrase_lists[key] = tuple(v.strip().lower() for v in values)
    key_points = _key_points(raw, case_id)
    min_key_points = _int_field(raw, "min_key_points", 0, case_id) or 0
    if min_key_points > len(key_points):
        raise EvalError(f"case {case_id!r}: min_key_points exceeds the number of key points")
    return EvalCase(
        id=case_id.strip(),
        question=question.strip(),
        min_searches=_int_field(raw, "min_searches", 0, case_id) or 0,
        min_references=_int_field(raw, "min_references", 0, case_id) or 0,
        min_rainfall_calls=_int_field(raw, "min_rainfall_calls", 0, case_id) or 0,
        min_data_references=_int_field(raw, "min_data_references", 0, case_id) or 0,
        max_evidence_claims=_int_field(raw, "max_evidence_claims", None, case_id),
        keywords_any=phrase_lists["keywords_any"],
        forbidden_phrases=phrase_lists["forbidden_phrases"],
        expect_metric=expect_metric,
        expect_tool=tools["expect_tool"],
        forbid_tool=tools["forbid_tool"],
        split=_choice(raw, "split", SPLITS, case_id) or "dev",
        category=_choice(raw, "category", CATEGORIES, case_id),
        expect_numbers=_expected_numbers(raw, case_id),
        max_tool_calls=_int_field(raw, "max_tool_calls", None, case_id),
        key_points=key_points,
        min_key_points=min_key_points,
    )


def load_cases(path: Path) -> list[EvalCase]:
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise EvalError(f"case file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise EvalError(f"invalid TOML in {path}: {exc}") from exc
    items = raw.get("case")
    if not isinstance(items, list) or not items:
        raise EvalError(f"{path}: expected at least one [[case]] table")
    cases = [_parse_case(item, i) for i, item in enumerate(items, start=1)]
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise EvalError(f"{path}: duplicate case ids: {duplicates}")
    return cases


def select_cases(
    cases: list[EvalCase],
    *,
    only: list[str] | None = None,
    split: str | None = None,
    category: str | None = None,
) -> list[EvalCase]:
    """Filter by ID (``only``), split and category. Unknown IDs raise ``EvalError``.

    ``split`` None keeps both splits; named IDs are kept whatever their split.
    """
    if only:
        unknown = sorted(set(only) - {c.id for c in cases})
        if unknown:
            raise EvalError(f"unknown case id(s): {', '.join(unknown)}")
        return [c for c in cases if c.id in only and (category is None or c.category == category)]
    return [
        c
        for c in cases
        if (split is None or c.split == split) and (category is None or c.category == category)
    ]


def load_qrels(path: Path) -> dict[str, Qrels]:
    """Relevance labels per case ID. A missing file means no labels yet.

    Format: one table per case, ``[case-id]`` with ``relevant`` and ``not_relevant`` lists of
    OpenAlex work IDs (``W…``). A work in both lists is an error."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise EvalError(f"invalid TOML in {path}: {exc}") from exc
    out: dict[str, Qrels] = {}
    for case_id, table in raw.items():
        if not isinstance(table, dict):
            raise EvalError(f"{path}: [{case_id}] must be a table")
        unknown = table.keys() - {"relevant", "not_relevant"}
        if unknown:
            raise EvalError(f"{path}: [{case_id}] has unknown keys {sorted(unknown)}")
        lists = {}
        for key in ("relevant", "not_relevant"):
            ids = table.get(key, [])
            if not isinstance(ids, list) or not all(
                isinstance(i, str) and _WORK_ID.match(i) for i in ids
            ):
                raise EvalError(f"{path}: [{case_id}] {key} must be a list of work IDs (W…)")
            lists[key] = frozenset(ids)
        both = sorted(lists["relevant"] & lists["not_relevant"])
        if both:
            raise EvalError(f"{path}: [{case_id}] lists {', '.join(both)} as both kinds")
        out[case_id] = Qrels(lists["relevant"], lists["not_relevant"])
    return out
