"""Evaluation harness: run a fixed question set and check *properties* of each answer.

Answers are model output and vary between runs, so cases specify expectations (searched,
cited enough works, mentions a topic keyword...) rather than exact text. Every case is also
checked against invariants that must hold for any answer; those encode the project's contract.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from research_agent.agent import AgentResult, InvalidQuestionError
from research_agent.agent.budget import OVERSHOOT_PREFIX
from research_agent.config import Limits
from research_agent.llm import LLMAuthError, LLMError
from research_agent.retrieval import AuthError, RetrievalError

_ALLOWED_KEYS = {
    "id",
    "question",
    "note",
    "min_searches",
    "min_references",
    "min_rainfall_calls",
    "min_data_references",
    "max_evidence_claims",
    "keywords_any",
    "forbidden_phrases",
}


class EvalError(ValueError):
    """The case file is missing, malformed or inconsistent."""


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


class EvalAgent(Protocol):
    def run(self, question: str) -> AgentResult: ...


def _int_field(raw: dict[str, Any], key: str, default: int | None, case_id: str) -> int | None:
    value = raw.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvalError(f"case {case_id!r}: {key} must be an integer >= 0, got {value!r}")
    return value


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
    phrase_lists: dict[str, tuple[str, ...]] = {}
    for key in ("keywords_any", "forbidden_phrases"):
        values = raw.get(key, [])
        if not isinstance(values, list) or not all(
            isinstance(v, str) and v.strip() for v in values
        ):
            raise EvalError(f"case {case_id!r}: {key} must be a list of non-empty strings")
        phrase_lists[key] = tuple(v.strip().lower() for v in values)
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


def check_result(case: EvalCase, result: AgentResult, limits: Limits) -> list[str]:
    """Return a list of failed expectations (empty means the case passed)."""
    answer, stats = result.answer, result.stats
    failures: list[str] = []

    # Invariants: must hold for every answer.
    if not answer.structured:
        failures.append("answer was not in the required structured format")
    if not answer.limitations:
        failures.append("answer carries no limitations")
    reference_ids = {w.id for w in answer.references} | {a.id for a in answer.data_references}
    for claim in answer.claims:
        if claim.kind == "evidence" and not claim.sources:
            failures.append(f"evidence claim has no citation: {claim.text[:60]!r}")
        for source in claim.sources:
            if source not in reference_ids:
                failures.append(f"claim cites {source}, which is not in the references")
    if stats.iterations > limits.max_iterations:
        failures.append(f"used {stats.iterations} model turns (limit {limits.max_iterations})")
    if stats.search_calls > limits.max_search_calls:
        failures.append(f"ran {stats.search_calls} searches (limit {limits.max_search_calls})")
    if stats.rainfall_calls > limits.max_rainfall_calls:
        failures.append(
            f"ran {stats.rainfall_calls} rainfall analyses (limit {limits.max_rainfall_calls})"
        )
    if stats.total_tokens > limits.max_total_tokens and not any(
        x.startswith(OVERSHOOT_PREFIX) for x in answer.limitations
    ):
        failures.append("exceeded the token limit without reporting it in the limitations")

    # Case-specific expectations.
    if stats.search_calls < case.min_searches:
        failures.append(
            f"expected at least {case.min_searches} search(es), ran {stats.search_calls}"
        )
    if stats.rainfall_calls < case.min_rainfall_calls:
        failures.append(
            f"expected at least {case.min_rainfall_calls} rainfall analysis(es), "
            f"ran {stats.rainfall_calls}"
        )
    if len(answer.data_references) < case.min_data_references:
        failures.append(
            f"expected at least {case.min_data_references} cited rainfall analysis(es), "
            f"got {len(answer.data_references)}"
        )
    if len(answer.references) < case.min_references:
        failures.append(
            f"expected at least {case.min_references} reference(s), got {len(answer.references)}"
        )
    evidence_claims = sum(1 for c in answer.claims if c.kind == "evidence")
    if case.max_evidence_claims is not None and evidence_claims > case.max_evidence_claims:
        failures.append(
            f"expected at most {case.max_evidence_claims} evidence claim(s), got {evidence_claims}"
        )
    text = " ".join(c.text for c in answer.claims).lower()
    if case.keywords_any and not any(k in text for k in case.keywords_any):
        failures.append(f"none of the expected keywords appear: {list(case.keywords_any)}")
    leaked = [p for p in case.forbidden_phrases if p in text]
    if leaked:
        failures.append(f"answer contains forbidden phrase(s): {leaked}")
    return failures


@dataclass(frozen=True)
class CaseReport:
    case_id: str
    failures: tuple[str, ...]
    result: AgentResult | None  # None when the run itself failed
    error: str | None = None

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def rejected_citations(self) -> int:
        return len(self.result.answer.rejected_citations) if self.result else 0


def run_case(agent: EvalAgent, case: EvalCase, limits: Limits) -> CaseReport:
    """Run one case. A provider/search failure fails that case; auth failures abort the run."""
    try:
        result = agent.run(case.question)
    except (LLMAuthError, AuthError):
        raise  # a configuration problem: every remaining case would fail the same way
    except (LLMError, RetrievalError, InvalidQuestionError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        return CaseReport(case.id, (f"run failed: {message}",), None, error=message)
    return CaseReport(case.id, tuple(check_result(case, result, limits)), result)


@dataclass(frozen=True)
class EvalSummary:
    total: int
    passed: int
    total_tokens: int
    searches: int
    rejected_citations: int
    unstructured: int
    rainfall_calls: int = 0


def summarize(reports: list[CaseReport]) -> EvalSummary:
    results = [r.result for r in reports if r.result is not None]
    return EvalSummary(
        total=len(reports),
        passed=sum(1 for r in reports if r.passed),
        total_tokens=sum(r.stats.total_tokens for r in results),
        searches=sum(r.stats.search_calls for r in results),
        rejected_citations=sum(r.rejected_citations for r in reports),
        unstructured=sum(1 for r in results if not r.answer.structured),
        rainfall_calls=sum(r.stats.rainfall_calls for r in results),
    )
