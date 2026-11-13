"""Paired comparison of two scored runs, case by case (pure, offline).

Used for fixed vs agentic search, a model or prompt change, or a run vs its replay. Only cases
present and completed in both runs are paired. For each measure the per-case values of run B
are compared with run A's: improved, same or worse (by the measure's direction), and the means.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any

# (name, value of one score or None, direction: +1 higher is better, -1 lower, 0 neutral)
Measure = tuple[str, Callable[[dict[str, Any]], float | None], int]


def _judged(s: dict[str, Any], key: str) -> float | None:
    """A judge figure, only when every claim of the record has a cached verdict."""
    judge = s.get("judge")
    if not judge or not judge["judged"] or judge["unjudged"]:
        return None
    return float(judge[key])


def _tokens(s: dict[str, Any]) -> float:
    return s["efficiency"]["input_tokens"] + s["efficiency"]["output_tokens"]


MEASURES: tuple[Measure, ...] = (
    ("passed", lambda s: float(s["passed"]), 1),
    ("key_point_coverage", lambda s: s["key_points"]["coverage"], 1),
    ("precision", lambda s: s["retrieval"]["precision"], 1),
    ("recall", lambda s: s["retrieval"]["recall"], 1),
    ("unsupported_numbers",
     lambda s: float(sum(len(u["numbers"]) for u in s["numbers"]["unsupported"])), -1),
    ("low_support_claims", lambda s: float(len(s["support"]["low"])), -1),
    ("works_cited", lambda s: float(s["retrieval"]["works_cited"]), 0),
    ("works_shown", lambda s: float(s["retrieval"]["works_shown"]), 0),
    ("tokens", _tokens, -1),
    ("model_calls", lambda s: float(s["efficiency"]["model_calls"]), -1),
    ("wall_seconds", lambda s: s["efficiency"]["wall_seconds"], -1),
    ("cost_usd", lambda s: s["efficiency"]["cost_usd"], -1),
    ("judge_unsupported", lambda s: _judged(s, "unsupported"), -1),
    ("judge_supported_share", lambda s: _judged(s, "supported_share"), 1),
)  # fmt: skip


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def compare_scores(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-measure paired results of run B against run A, plus the unpaired cases."""
    by_a = {s["case_id"]: s for s in a}
    by_b = {s["case_id"]: s for s in b}
    both = [c for c in by_a if c in by_b]
    paired = [c for c in both if "tools" in by_a[c] and "tools" in by_b[c]]
    measures = {}
    for name, value, direction in MEASURES:
        pairs = [(value(by_a[c]), value(by_b[c]), c) for c in paired]
        pairs = [(x, y, c) for x, y, c in pairs if x is not None and y is not None]
        better = worse = 0
        for x, y, _ in pairs:
            if direction and x != y:
                if (y - x) * direction > 0:
                    better += 1
                else:
                    worse += 1
        measures[name] = {
            "direction": {1: "higher is better", -1: "lower is better", 0: "neutral"}[direction],
            "paired": len(pairs),
            "mean_a": _mean([x for x, _, _ in pairs]),
            "mean_b": _mean([y for _, y, _ in pairs]),
            "improved": better if direction else None,
            "same": sum(1 for x, y, _ in pairs if x == y),
            "worse": worse if direction else None,
            "per_case": {c: [x, y] for x, y, c in pairs},
        }
    return {
        "paired_cases": paired,
        "not_completed_in_both": sorted(set(both) - set(paired)),
        "only_in_a": sorted(set(by_a) - set(by_b)),
        "only_in_b": sorted(set(by_b) - set(by_a)),
        "measures": measures,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:g}" if isinstance(value, float) else str(value)


def render_comparison(result: dict[str, Any], label_a: str, label_b: str) -> str:
    """A Markdown table: one row per measure, then the per-case values of the main measures."""
    lines = [f"# Comparison: A = {label_a}, B = {label_b}", "",
             f"{len(result['paired_cases'])} paired case(s). Improved / worse are B against A.",
             "",
             "| measure | direction | paired | mean A | mean B | improved | same | worse |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, m in result["measures"].items():
        lines.append(
            f"| {name} | {m['direction']} | {m['paired']} | {_fmt(m['mean_a'])} | "
            f"{_fmt(m['mean_b'])} | {_fmt(m['improved'])} | {m['same']} | {_fmt(m['worse'])} |"
        )
    for key, title in (("only_in_a", "Only in A"), ("only_in_b", "Only in B"),
                       ("not_completed_in_both", "Not completed in both runs")):  # fmt: skip
        if result[key]:
            lines += ["", f"{title}: {', '.join(result[key])}"]
    shown = ("passed", "tokens", "works_cited", "unsupported_numbers", "low_support_claims")
    lines += ["", "## Per case (A -> B)", "",
              "| case | " + " | ".join(shown) + " |", "|---|" + "---|" * len(shown)]  # fmt: skip
    for case in result["paired_cases"]:
        cells = []
        for name in shown:
            pair = result["measures"][name]["per_case"].get(case)
            cells.append("-" if pair is None else f"{pair[0]:g} -> {pair[1]:g}")
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
