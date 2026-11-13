"""Deterministic checks of one answer against its case. One implementation for both a live
``AgentResult`` and a stored run record, so ``eval score`` judges stored runs exactly as
``eval run`` judged them (with the current case definitions and checks)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from research_agent.agent import AgentResult
from research_agent.agent.budget import OVERSHOOT_PREFIX
from research_agent.agent.tools import RAINFALL_TOOL_NAME, TEMPERATURE_TOOL_NAME, format_recalled
from research_agent.config import Limits
from research_agent.evaluation.cases import EvalCase, KeyPoint, TurnExpect
from research_agent.evaluation.numbers import value_found
from research_agent.tracing import answer_json


def answer_dict(result: AgentResult) -> dict[str, Any]:
    """The answer as stored in a run record: claims, IDs, limitations, plus the cited works'
    text and the cited analyses as the model saw them (for offline faithfulness checks)."""
    answer = result.answer
    return {
        **answer_json(answer),
        "works": [
            {"id": w.id, "title": w.title, "year": w.year, "abstract": w.abstract}
            for w in answer.references
        ],
        "data": [{"id": d.id, "text": format_recalled(d)} for d in answer.data_references],
    }


def stats_dict(result: AgentResult) -> dict[str, Any]:
    stats = result.stats
    data = asdict(stats)
    data["tool_trace"] = [list(turn) for turn in stats.tool_trace]
    data["total_tokens"] = stats.total_tokens
    return data


def check_result(case: EvalCase, result: AgentResult, limits: Limits) -> list[str]:
    """Return a list of failed expectations (empty means the case passed)."""
    return check_record(case, answer_dict(result), stats_dict(result), limits)


def covered_key_points(key_points: tuple[KeyPoint, ...], claims: list[dict]) -> list[KeyPoint]:
    """Key points with a term in a claim that cites something (citations are already valid)."""
    cited_text = " ".join(c["text"] for c in claims if c["sources"]).lower()
    return [k for k in key_points if any(term in cited_text for term in k.any_terms)]


def check_record(
    case: EvalCase,
    answer: dict[str, Any],
    stats: dict[str, Any],
    limits: Limits,
    tool_results: list[str] | None = None,
) -> list[str]:
    """The checks on stored data: ``answer`` and ``stats`` as written by ``answer_dict`` and
    ``stats_dict``; ``tool_results`` are the texts the model saw (from a run record; without
    them the tool side of ``expect_numbers`` is not checked). Returns the failed expectations
    (empty means the case passed)."""
    failures: list[str] = []
    claims = answer["claims"]
    trace = [list(turn) for turn in stats["tool_trace"]]

    # Invariants: must hold for every answer.
    if not answer["structured"]:
        failures.append("answer was not in the required structured format")
    if not answer["limitations"]:
        failures.append("answer carries no limitations")
    reference_ids = set(answer["references"]) | set(answer["data_references"])
    for claim in claims:
        if claim["kind"] == "evidence" and not claim["sources"]:
            failures.append(f"evidence claim has no citation: {claim['text'][:60]!r}")
        for source in claim["sources"]:
            if source not in reference_ids:
                failures.append(f"claim cites {source}, which is not in the references")
    if stats["iterations"] > limits.max_iterations:
        failures.append(f"used {stats['iterations']} model turns (limit {limits.max_iterations})")
    if stats["search_calls"] > limits.max_search_calls:
        failures.append(
            f"ran {stats['search_calls']} searches (limit {limits.max_search_calls})"
        )
    if stats["rainfall_calls"] > limits.max_rainfall_calls:
        failures.append(
            f"ran {stats['rainfall_calls']} rainfall analyses (limit {limits.max_rainfall_calls})"
        )
    if stats["total_tokens"] > limits.max_total_tokens and not any(
        x.startswith(OVERSHOOT_PREFIX) for x in answer["limitations"]
    ):
        failures.append("exceeded the token limit without reporting it in the limitations")

    # Case-specific expectations.
    if stats["search_calls"] < case.min_searches:
        failures.append(
            f"expected at least {case.min_searches} search(es), ran {stats['search_calls']}"
        )
    # The metric belongs to the temperature tool when the case expects that tool, else rainfall.
    metric_tool = (
        TEMPERATURE_TOOL_NAME if case.expect_tool == TEMPERATURE_TOOL_NAME else RAINFALL_TOOL_NAME
    )
    if case.expect_metric and not any(
        name == f"{metric_tool}[{case.expect_metric}]" for turn in trace for name in turn
    ):
        failures.append(
            f"expected {metric_tool} to be requested with metric {case.expect_metric!r}; "
            f"tools requested: {trace}"
        )
    requested = {name.partition("[")[0] for turn in trace for name in turn}
    if case.expect_tool and case.expect_tool not in requested:
        failures.append(
            f"expected the model to request {case.expect_tool}; tools requested: {trace}"
        )
    if case.forbid_tool and case.forbid_tool in requested:
        failures.append(f"the model requested {case.forbid_tool}, which this case forbids")
    if stats["rainfall_calls"] < case.min_rainfall_calls:
        failures.append(
            f"expected at least {case.min_rainfall_calls} rainfall analysis(es), "
            f"ran {stats['rainfall_calls']}"
        )
    if len(answer["data_references"]) < case.min_data_references:
        failures.append(
            f"expected at least {case.min_data_references} cited rainfall analysis(es), "
            f"got {len(answer['data_references'])}"
        )
    if len(answer["references"]) < case.min_references:
        failures.append(
            f"expected at least {case.min_references} reference(s), "
            f"got {len(answer['references'])}"
        )
    evidence_claims = sum(1 for c in claims if c["kind"] == "evidence")
    if case.max_evidence_claims is not None and evidence_claims > case.max_evidence_claims:
        failures.append(
            f"expected at most {case.max_evidence_claims} evidence claim(s), got {evidence_claims}"
        )
    text = " ".join(c["text"] for c in claims).lower()
    if case.keywords_any and not any(k in text for k in case.keywords_any):
        failures.append(f"none of the expected keywords appear: {list(case.keywords_any)}")
    leaked = [p for p in case.forbidden_phrases if p in text]
    if leaked:
        failures.append(f"answer contains forbidden phrase(s): {leaked}")

    requested_calls = sum(len(turn) for turn in trace)
    if case.max_tool_calls is not None and requested_calls > case.max_tool_calls:
        failures.append(
            f"requested {requested_calls} tool call(s) (case limit {case.max_tool_calls})"
        )
    # Tool side: the data is still right (signed). Claim side: it was reported (by absolute
    # value, so "41.5 °C below zero" counts for -41.5).
    claim_texts = [c["text"] for c in claims]
    for expected in case.expect_numbers:
        label = f"{expected.value:g} (±{expected.tolerance:g})"
        if tool_results is not None and not value_found(
            expected.value, expected.tolerance, tool_results
        ):
            failures.append(f"expected value {label} not found in any tool result")
        if not value_found(expected.value, expected.tolerance, claim_texts, signed=False):
            failures.append(f"expected value {label} not found in any claim")
    if case.min_key_points:
        covered = len(covered_key_points(case.key_points, claims))
        if covered < case.min_key_points:
            failures.append(
                f"covered {covered} of {len(case.key_points)} key point(s) "
                f"(case minimum {case.min_key_points})"
            )
    return failures


def check_turn(expect: TurnExpect, record: dict[str, Any]) -> list[str]:
    """The conversation checks of one completed turn record: the model's relation label and
    recalled IDs, the analyses it ran, and the task state the code built after the turn."""
    info = record.get("conversation") or {}
    failures = []
    relation = info.get("relation")
    if expect.expect_relation and relation not in expect.expect_relation:
        failures.append(
            f"relation {relation!r} ({info.get('relation_source')}) is not one of "
            f"{list(expect.expect_relation)}"
        )
    recalled = info.get("recalled", [])
    if expect.expect_recall is True and not recalled:
        failures.append("expected stored evidence to be recalled; nothing was")
    if expect.expect_recall is False and recalled:
        failures.append(f"expected no recall; recalled {recalled}")
    analyses = record["stats"]["rainfall_calls"]
    if expect.max_new_analyses is not None and analyses > expect.max_new_analyses:
        failures.append(
            f"ran {analyses} new analysis(es) (turn limit {expect.max_new_analyses})"
        )
    state = info.get("state")
    if state is None:  # not recorded (an imported trace): state expectations are not judged
        return failures
    places = state.get("places", [])
    for want in expect.expect_places:
        if not any(want in place.lower() for place in places):
            failures.append(f"task state has no place matching {want!r}; places: {places}")
    metrics = state.get("metrics", [])
    for metric in expect.expect_state_metrics:
        if metric not in metrics:
            failures.append(f"task state lacks metric {metric!r}; metrics: {metrics}")
    pending = state.get("pending")
    if expect.expect_pending is True and pending is None:
        failures.append("expected a pending clarification (ambiguous place); there is none")
    if expect.expect_pending is False and pending is not None:
        failures.append(f"expected no pending clarification; still pending for {pending!r}")
    return failures
