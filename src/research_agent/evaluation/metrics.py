"""Deterministic metrics of one stored run record (pure, offline, free).

Everything here is reported, never a case failure (the owner's choice, 2026-11-13); failures
come only from ``checks``. Sections: tools, trajectory, retrieval (against ``evals/qrels.toml``),
key points, lexical citation support, numbers, efficiency. A record that did not complete gets
only its status and efficiency.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from typing import Any

from research_agent.agent.tools import SEARCH_TOOL_NAME
from research_agent.config import Limits
from research_agent.evaluation.cases import EvalCase, Qrels
from research_agent.evaluation.checks import covered_key_points
from research_agent.evaluation.numbers import extract_numbers, unsupported_numbers
from research_agent.evaluation.recording import RunBudget

ERROR_KINDS = ("invalid", "not_found", "ambiguous", "budget_refused", "context_refused", "failed")
EXCERPT_CHARS = 100

# A work line of a search result: "[W123] Title (authors, venue, year, cited by N)".
_WORK_LINE = re.compile(r"^\[(W\d+)\] (.*)$", re.MULTILINE)
_ALREADY_SHOWN = "(already shown in an earlier search)"
_STOP_WORDS = frozenset(
    (
        "about", "after", "also", "among", "based", "been", "before", "being", "between",
        "both", "could", "does", "during", "each", "from", "have", "into", "more", "most",
        "other", "over", "same", "some", "such", "than", "that", "their", "them", "then",
        "there", "these", "they", "this", "those", "through", "used", "using", "very",
        "were", "what", "when", "where", "which", "while", "will", "with", "within",
        "would", "shows", "show", "study", "studies", "paper", "papers",
    )
)


@dataclass(frozen=True)
class ScoreSettings:
    """From ``[eval]``: ``redundant_query_jaccard`` and ``low_support_overlap``."""

    redundant_query_jaccard: float
    low_support_overlap: float


def _excerpt(text: str) -> str:
    return text if len(text) <= EXCERPT_CHARS else text[: EXCERPT_CHARS - 1] + "…"


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


# --- tools ---------------------------------------------------------------------------------------


def error_kind(call: dict[str, Any]) -> str | None:
    """How a tool call went wrong, from its recorded status and the text the model saw (None:
    it worked). The texts are the agent loop's own ``Error: ...`` results."""
    if call["status"] == "ambiguous":
        return "ambiguous"
    text = call["result"]
    if not text.startswith("Error:"):
        return None
    if "limit reached" in text:
        return "budget_refused"
    if "retrieved-context budget is used up" in text:
        return "context_refused"
    if "search failed (" in text or "analysis failed (" in text:
        return "failed"
    if text.startswith("Error: no place named"):
        return "not_found"
    return "invalid"  # bad arguments, unknown tool, malformed JSON, out-of-range request


def _normalised(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, dict):
        return {k: _normalised(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalised(v) for v in value]
    return value


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def redundant_calls(calls: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Calls repeating an earlier one: same tool and normalised arguments, or a search whose
    query words overlap an earlier query's by at least ``threshold`` (Jaccard)."""
    out = []
    for j, later in enumerate(calls):
        later_key = json.dumps(_normalised(later["arguments"]), sort_keys=True)
        for earlier in calls[:j]:
            if earlier["name"] != later["name"]:
                continue
            if json.dumps(_normalised(earlier["arguments"]), sort_keys=True) == later_key:
                why = "same arguments"
            elif later["name"] == SEARCH_TOOL_NAME and jaccard(
                _words(str(earlier["arguments"].get("query", ""))),
                _words(str(later["arguments"].get("query", ""))),
            ) >= threshold:
                why = "similar query"
            else:
                continue
            out.append({"id": later["id"], "tool": later["name"], "repeats": earlier["id"],
                        "why": why})  # fmt: skip
            break
    return out


def search_works(result_text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """(newly shown works as (ID, title line), IDs listed as already shown) of a search result."""
    new, again = [], []
    for match in _WORK_LINE.finditer(result_text):
        work_id, rest = match.groups()
        if rest.startswith(_ALREADY_SHOWN):
            again.append(work_id)
        else:
            title = rest.rsplit(" (", 1)[0] if rest.endswith(")") and " (" in rest else rest
            new.append((work_id, title))
    return new, again


def _tools(calls: list[dict], cited: set[str], settings: ScoreSettings) -> dict[str, Any]:
    by_tool: dict[str, int] = {}
    errors = dict.fromkeys(ERROR_KINDS, 0)
    uncited = []
    for call in calls:
        by_tool[call["name"]] = by_tool.get(call["name"], 0) + 1
        kind = error_kind(call)
        if kind:
            errors[kind] += 1
            continue
        if call["name"] == SEARCH_TOOL_NAME:
            new_ids = {work_id for work_id, _ in search_works(call["result"])[0]}
            if new_ids and not new_ids & cited:
                uncited.append({"id": call["id"], "tool": call["name"],
                                "why": f"none of {len(new_ids)} new work(s) cited"})  # fmt: skip
        elif call["result_id"] and call["result_id"] not in cited:
            uncited.append({"id": call["id"], "tool": call["name"],
                            "why": f"{call['result_id']} not cited"})  # fmt: skip
    return {
        "calls": len(calls),
        "by_tool": dict(sorted(by_tool.items())),
        "errors": errors,
        "argument_validity": round(1 - errors["invalid"] / len(calls), 3) if calls else None,
        "redundant": redundant_calls(calls, settings.redundant_query_jaccard),
        "uncited": uncited,
    }


# --- trajectory ----------------------------------------------------------------------------------


def _forced_reason(stats: dict[str, Any], limits: Limits) -> str:
    """Which budget forced the final answer (from the counts, in the loop's order)."""
    if stats["iterations"] >= limits.max_iterations:
        return "model turns"
    if stats["search_calls"] >= limits.max_search_calls:
        if stats["rainfall_calls"] >= limits.max_rainfall_calls:
            return "searches and analyses"
        return "searches"  # only possible for the literature-only agent
    return "tokens"


def _trajectory(record: dict, calls: list[dict], cited: set[str], limits: Limits) -> dict:
    stats = record["stats"]
    llm = record["llm_calls"]
    fixed = bool(record.get("fixed_query"))  # tools are off by design in fixed-search mode
    forced = bool(llm) and not llm[-1]["allow_tools"] and not fixed
    first_shown: dict[str, int] = {}  # work ID -> index of the search that first showed it
    follow_ups = []
    searches = [c for c in calls if c["name"] == SEARCH_TOOL_NAME and not error_kind(c)]
    for index, call in enumerate(searches):
        new, again = search_works(call["result"])
        for work_id, _ in new:
            first_shown.setdefault(work_id, index)
        if index > 0:
            listed = len(new) + len(again)
            follow_ups.append(round(len(new) / listed, 3) if listed else None)
    return {
        "turns": len(llm),
        "tools_per_turn": [len(turn) for turn in stats["tool_trace"]],
        "stop": "fixed" if fixed else "forced" if forced else "voluntary",
        "forced_by": _forced_reason(stats, limits) if forced else None,
        "budget_left": {
            "model_turns": limits.max_iterations - stats["iterations"],
            "searches": limits.max_search_calls - stats["search_calls"],
            "analyses": limits.max_rainfall_calls - stats["rainfall_calls"],
            "tokens": limits.max_total_tokens - stats["total_tokens"],
        },
        "new_work_share_per_follow_up_search": follow_ups,
        "cited_from_follow_up_searches": sum(
            1 for work_id, index in first_shown.items() if index > 0 and work_id in cited
        ),
    }


# --- retrieval, key points, support, numbers -----------------------------------------------------


def _retrieval(calls: list[dict], answer: dict, qrels: Qrels | None) -> dict[str, Any]:
    shown: dict[str, str] = {}
    for call in calls:
        if call["name"] == SEARCH_TOOL_NAME and not error_kind(call):
            for work_id, title in search_works(call["result"])[0]:
                shown.setdefault(work_id, title)
    labels = qrels or Qrels()
    judged = [w for w in shown if w in labels.relevant or w in labels.not_relevant]
    relevant_shown = [w for w in shown if w in labels.relevant]
    return {
        "works_shown": len(shown),
        "works_cited": len(answer["works"]),
        "labelled": qrels is not None,
        "judged_shown": len(judged),
        "precision": round(len(relevant_shown) / len(judged), 3) if judged else None,
        "recall": round(len(relevant_shown) / len(labels.relevant), 3)
        if labels.relevant
        else None,
        "unjudged": [{"id": w, "title": _excerpt(t)} for w, t in shown.items() if w not in judged],
    }


def _source_texts(answer: dict) -> dict[str, str]:
    texts = {w["id"]: f"{w['title'] or ''}\n{w['abstract'] or ''}" for w in answer["works"]}
    texts.update({d["id"]: d["text"] for d in answer["data"]})
    return texts


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[^\W\d_]{4,}", text.lower()) if w not in _STOP_WORDS}


def _support(claims: list[dict], sources: dict[str, str], settings: ScoreSettings) -> dict:
    overlaps, low = [], []
    for claim in claims:
        if claim["kind"] != "evidence" or not claim["sources"]:
            continue
        words = _content_words(claim["text"])
        if not words:
            continue
        cited = _content_words(" ".join(sources.get(s, "") for s in claim["sources"]))
        overlap = len(words & cited) / len(words)
        overlaps.append(overlap)
        if overlap < settings.low_support_overlap:
            low.append({"text": _excerpt(claim["text"]), "sources": claim["sources"],
                        "overlap": round(overlap, 3)})  # fmt: skip
    return {"claims_scored": len(overlaps), "median_overlap": _median(overlaps), "low": low}


def _numbers(claims: list[dict], sources: dict[str, str]) -> dict[str, Any]:
    checked_claims = checked_numbers = 0
    unsupported = []
    for claim in claims:
        if claim["kind"] != "evidence" or not claim["sources"]:
            continue
        found = extract_numbers(claim["text"])
        if not found:
            continue
        checked_claims += 1
        checked_numbers += len(found)
        missing = unsupported_numbers(claim["text"], [sources.get(s, "") for s in claim["sources"]])
        if missing:
            unsupported.append({"text": _excerpt(claim["text"]), "sources": claim["sources"],
                                "numbers": missing})  # fmt: skip
    return {"claims_checked": checked_claims, "numbers_checked": checked_numbers,
            "unsupported": unsupported}  # fmt: skip


def _key_points(case: EvalCase | None, claims: list[dict]) -> dict[str, Any]:
    points = case.key_points if case else ()
    covered = covered_key_points(points, claims)
    return {
        "total": len(points),
        "covered": [k.text for k in covered],
        "coverage": round(len(covered) / len(points), 3) if points else None,
    }


# --- efficiency and the whole record -------------------------------------------------------------


def _efficiency(record: dict, budget: RunBudget | None) -> dict[str, Any]:
    llm = record["llm_calls"]
    tin = sum(c["usage"]["input"] for c in llm)
    tout = sum(c["usage"]["output"] for c in llm)
    answer = record["answer"]
    cited = len(answer["references"]) + len(answer["data_references"]) if answer else 0
    return {
        "wall_seconds": record["wall_seconds"],
        "model_calls": len(llm),
        "tool_calls": len(record["tool_calls"]),
        "input_tokens": tin,
        "output_tokens": tout,
        "median_call_seconds": _median([c["latency_s"] for c in llm]),
        "cost_usd": round(budget.cost(tin, tout), 6) if budget and budget.priced else None,
        "tokens_per_cited_item": round((tin + tout) / cited) if cited else None,
    }


def score_record(
    record: dict[str, Any],
    case: EvalCase | None,
    qrels: Qrels | None,
    limits: Limits,
    settings: ScoreSettings,
    budget: RunBudget | None = None,
) -> dict[str, Any]:
    """All metrics of one record. ``case`` (for key points) may be None for a case no longer in
    the case file; ``qrels`` None means the case has no relevance labels."""
    out: dict[str, Any] = {
        "case_id": record["case_id"],
        "category": record.get("category"),
        "split": record.get("split"),
        "status": record["status"],
        "passed": record["status"] == "ok" and not record["failures"],
        "failures": record["failures"],
        "efficiency": _efficiency(record, budget),
    }
    if "replay" in record:
        out["replay"] = record["replay"]
    answer = record["answer"]
    if record["status"] != "ok" or answer is None:
        return out
    claims = answer["claims"]
    cited = {s for c in claims for s in c["sources"]}
    calls = record["tool_calls"]
    sources = _source_texts(answer)
    out.update(
        tools=_tools(calls, cited, settings),
        trajectory=_trajectory(record, calls, cited, limits),
        retrieval=_retrieval(calls, answer, qrels),
        key_points=_key_points(case, claims),
        support=_support(claims, sources, settings),
        numbers=_numbers(claims, sources),
    )
    return out
