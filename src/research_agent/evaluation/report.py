"""Scores of a run: per-case metrics, totals per run/category/split, ``score.json`` and
``report.md`` in the run folder (pure apart from writing the two files)."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from research_agent.config import Limits
from research_agent.evaluation.cases import EvalCase, Qrels
from research_agent.evaluation.judge import INTERPRETATION_SCORES, verdict_score
from research_agent.evaluation.metrics import ERROR_KINDS, ScoreSettings, score_record
from research_agent.evaluation.recording import RunBudget

SCORE_FILE = "score.json"
REPORT_FILE = "report.md"
REPLAY_OUTCOMES = ("same", "changed", "diverged")


def score_records(
    records: list[dict[str, Any]],
    cases: list[EvalCase],
    qrels: dict[str, Qrels],
    limits: Limits,
    settings: ScoreSettings,
    budget: RunBudget | None = None,
    judge_lookup: Callable[[dict[str, Any], EvalCase | None], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Metrics per record; ``judge_lookup`` (cached verdicts, no call) adds a judge section."""
    by_id = {c.id: c for c in cases}
    out = []
    for r in records:
        case = by_id.get(r["case_id"])
        judge = judge_lookup(r, case) if judge_lookup else None
        out.append(score_record(r, case, qrels.get(r["case_id"]), limits, settings, budget, judge))
    return out


def _judge_totals(scores: list[dict[str, Any]]) -> dict[str, Any] | None:
    sections = [s["judge"] for s in scores if s.get("judge")]
    if not sections:
        return None
    totals = {k: sum(j[k] for j in sections)
              for k in ("claims", "judged", "unjudged", "supported", "partial", "unsupported")}
    totals["supported_share"] = (
        round(totals["supported"] / totals["judged"], 3) if totals["judged"] else None
    )
    totals["score"] = verdict_score(totals)
    totals["model"] = sections[0]["model"]
    interp = [j["interpretation"] for j in sections if j.get("interpretation")]
    i_totals = {k: sum(i[k] for i in interp)
                for k in ("claims", "judged", "unjudged", "within", "stretch", "overreach")}
    i_totals["score"] = verdict_score(i_totals, INTERPRETATION_SCORES)
    totals["interpretation"] = i_totals
    return totals  # fmt: skip


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def _group(scores: list[dict[str, Any]]) -> dict[str, Any]:
    done = [s for s in scores if "tools" in s]
    tool_calls = sum(s["tools"]["calls"] for s in done)
    errors = {k: sum(s["tools"]["errors"][k] for s in done) for k in ERROR_KINDS}
    tokens = sum(s["efficiency"]["input_tokens"] + s["efficiency"]["output_tokens"] for s in scores)
    costs = [s["efficiency"]["cost_usd"] for s in scores]
    passed = sum(1 for s in scores if s["passed"])
    return {
        "cases": len(scores),
        "passed": passed,
        "pass_rate": round(passed / len(scores), 3) if scores else None,
        "tool_calls": tool_calls,
        "argument_validity": round(1 - errors["invalid"] / tool_calls, 3) if tool_calls else None,
        "tool_errors": errors,
        "redundant_calls": sum(len(s["tools"]["redundant"]) for s in done),
        "uncited_calls": sum(len(s["tools"]["uncited"]) for s in done),
        "forced_stops": sum(1 for s in done if s["trajectory"]["stop"] == "forced"),
        "unsupported_numbers": sum(
            len(u["numbers"]) for s in done for u in s["numbers"]["unsupported"]
        ),
        "low_support_claims": sum(len(s["support"]["low"]) for s in done),
        "median_key_point_coverage": _median(
            [s["key_points"]["coverage"] for s in done if s["key_points"]["coverage"] is not None]
        ),
        "labelled_cases": sum(1 for s in done if s["retrieval"]["labelled"]),
        "mean_precision": _mean(
            [s["retrieval"]["precision"] for s in done if s["retrieval"]["precision"] is not None]
        ),
        "mean_recall": _mean(
            [s["retrieval"]["recall"] for s in done if s["retrieval"]["recall"] is not None]
        ),
        "unjudged_works": sum(len(s["retrieval"]["unjudged"]) for s in done),
        "tokens": tokens,
        "median_tokens": _median(
            [s["efficiency"]["input_tokens"] + s["efficiency"]["output_tokens"] for s in done]
        ),
        "median_wall_seconds": _median(
            [s["efficiency"]["wall_seconds"] for s in done if s["efficiency"]["wall_seconds"]]
        ),
        "cost_usd": round(sum(costs), 6) if costs and None not in costs else None,
        "passed_per_100k_tokens": round(passed / tokens * 100_000, 2) if tokens else None,
        "replay": {
            outcome: sum(1 for s in scores if s.get("replay", {}).get("outcome") == outcome)
            for outcome in REPLAY_OUTCOMES
        },
        "judge": _judge_totals(scores),
    }


def conversations(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Conversations whose every recorded turn passed (turns of a conversation are records
    with a ``conversation`` section)."""
    turns: dict[str, list[bool]] = {}
    for s in scores:
        if "conversation" in s:
            turns.setdefault(s["conversation"]["id"], []).append(bool(s["passed"]))
    return {
        "total": len(turns),
        "passed": sum(1 for passed in turns.values() if all(passed)),
        "failed": sorted(c for c, passed in turns.items() if not all(passed)),
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    def by(key: str) -> dict[str, Any]:
        names = sorted({s[key] or "none" for s in scores})
        return {n: _group([s for s in scores if (s[key] or "none") == n]) for n in names}

    return {
        "all": _group(scores),
        "by_category": by("category"),
        "by_split": by("split"),
        "conversations": conversations(scores),
    }


def flags(score: dict[str, Any]) -> str | None:
    """A one-line list of what the metrics flagged for a case (None: nothing)."""
    if "tools" not in score:
        return None
    parts = []
    errors = sum(score["tools"]["errors"].values())
    for count, label in (
        (len(score["numbers"]["unsupported"]), "claim(s) with unsupported numbers"),
        (len(score["support"]["low"]), "low-support claim(s)"),
        (len(score["tools"]["redundant"]), "redundant call(s)"),
        (len(score["tools"]["uncited"]), "uncited call(s)"),
        (errors, "tool error(s)"),
    ):
        if count:
            parts.append(f"{count} {label}")
    if score["trajectory"]["stop"] == "forced":
        parts.append(f"forced stop ({score['trajectory']['forced_by']})")
    judge = score.get("judge")
    if judge and (judge["unsupported"] or judge["partial"]):
        parts.append(f"judge: {judge['unsupported']} unsupported, {judge['partial']} partial")
    interp = (judge or {}).get("interpretation") or {}
    if interp.get("overreach") or interp.get("stretch"):
        parts.append(f"judge: {interp['overreach']} overreaching, {interp['stretch']} "
                     "stretching interpretation(s)")  # fmt: skip
    return ", ".join(parts) or None


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def metrics_line(agg: dict[str, Any]) -> str:
    a = agg["all"]
    return (
        f"Metrics: argument validity {_fmt(a['argument_validity'])} | "
        f"{sum(a['tool_errors'].values())} tool error(s) | {a['redundant_calls']} redundant, "
        f"{a['uncited_calls']} uncited call(s) | {a['forced_stops']} forced stop(s) | "
        f"{a['unsupported_numbers']} unsupported number(s) | "
        f"{a['low_support_claims']} low-support claim(s) | "
        f"key points {_fmt(a['median_key_point_coverage'])} | "
        f"{a['labelled_cases']} labelled case(s), {a['unjudged_works']} unjudged work(s)"
        + (f" | {judge_line(a['judge'])}" if a["judge"] else "")
    )


def judge_line(totals: dict[str, Any]) -> str:
    return (
        f"judge ({totals['model']}): {totals['supported']} supported, {totals['partial']} "
        f"partial, {totals['unsupported']} unsupported of {totals['judged']} judged claim(s)"
        + (f", score {_fmt(totals['score'])}" if totals.get("score") is not None else "")
        + (f", {totals['unjudged']} not judged yet" if totals["unjudged"] else "")
        + _interpretation_line(totals.get("interpretation"))
    )


def _interpretation_line(i: dict[str, Any] | None) -> str:
    if not i or not i["claims"]:
        return ""
    return (
        f"; interpretations: {i['within']} within, {i['stretch']} stretch, {i['overreach']} "
        f"overreach of {i['judged']} judged"
        + (f", score {_fmt(i['score'])}" if i["score"] is not None else "")
        + (f", {i['unjudged']} not judged yet" if i["unjudged"] else "")
    )


_JUDGE_COLUMNS = (
    ("judged", "judged"), ("supported", "supported"), ("partial", "partial"),
    ("unsupported", "unsupported"), ("unjudged", "not judged"),
    ("supported_share", "supported share"), ("score", "score"),
)  # fmt: skip
_INTERP_COLUMNS = (
    ("judged", "interp. judged"), ("within", "within"), ("stretch", "stretch"),
    ("overreach", "overreach"), ("unjudged", "interp. not judged"), ("score", "interp. score"),
)  # fmt: skip


def _judge_cells(j: dict[str, Any]) -> str:
    cells = [_fmt(j.get(k)) for k, _ in _JUDGE_COLUMNS]
    i = j.get("interpretation") or {}
    cells += [_fmt(i.get(k)) for k, _ in _INTERP_COLUMNS]
    return " | ".join(cells)


def _key_point_cell(points: list[dict[str, Any]] | None) -> str:
    if not points:
        return "-"
    covered = sum(p["coverage"] == "covered" for p in points)
    partial = sum(p["coverage"] == "partial" for p in points)
    contradicted = sum(p["consistency"] == "contradicted" for p in points)
    return f"{covered}/{len(points)} covered, {partial} partial, {contradicted} contradicted"


def judge_tables(scores: list[dict[str, Any]], agg: dict[str, Any]) -> list[str]:
    """The Judge section: totals by run, category and split, then one row per case.

    Score = mean over judged claims of supported 1, partial 0.5, unsupported 0."""
    totals = agg["all"]["judge"]
    if not totals:
        return []
    columns = (*_JUDGE_COLUMNS, *_INTERP_COLUMNS)
    head = " | ".join(label for _, label in columns)
    lines = [
        "## Judge", "",
        f"Model {totals['model']}. Each evidence claim is rated against the text it cites; "
        "score = mean over judged claims of supported 1, partial 0.5, unsupported 0. Each "
        "interpretation claim is rated against the evidence it rests on (its cited sources, or "
        "all the answer's evidence when it cites none); interp. score = mean of within 1, "
        "stretch 0.5, overreach 0. Reported only, never a case failure.", "",
        f"| group | claims | {head} |", "|---|---|" + "---|" * len(columns),
    ]  # fmt: skip
    groups = [("all", agg["all"])]
    groups += [(f"category {k}", v) for k, v in agg["by_category"].items()]
    groups += [(f"split {k}", v) for k, v in agg["by_split"].items()]
    for name, group in groups:
        j = group.get("judge")
        if j:
            lines.append(f"| {name} | {j['claims']} | {_judge_cells(j)} |")
    lines += ["", f"| case | claims | {head} | key points |",
              "|---|---|" + "---|" * (len(columns) + 1)]  # fmt: skip
    for s in scores:
        j = s.get("judge")
        if j:
            cells = _judge_cells(j)
            lines.append(f"| {s['case_id']} | {j['claims']} | {cells} | "
                         f"{_key_point_cell(j.get('key_points'))} |")  # fmt: skip
    return [*lines, ""]


_COLUMNS = (
    ("cases", "cases"), ("passed", "passed"), ("median_tokens", "median tokens"),
    ("forced_stops", "forced"), ("unsupported_numbers", "unsupported numbers"),
    ("low_support_claims", "low support"), ("redundant_calls", "redundant"),
    ("uncited_calls", "uncited"), ("argument_validity", "arg validity"),
    ("median_key_point_coverage", "key points"), ("mean_precision", "precision"),
    ("mean_recall", "recall"), ("unjudged_works", "unjudged"),
)  # fmt: skip


def _table(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    head = "| group | " + " | ".join(label for _, label in _COLUMNS) + " |"
    lines = [head, "|" + "---|" * (len(_COLUMNS) + 1)]
    for name, group in rows:
        lines.append(f"| {name} | " + " | ".join(_fmt(group[k]) for k, _ in _COLUMNS) + " |")
    return lines


def _case_section(s: dict[str, Any]) -> list[str]:
    status = ("PASS" if s["passed"] else "FAIL") if s["status"] == "ok" else s["status"].upper()
    e = s["efficiency"]
    lines = [
        f"### {s['case_id']} ({s['category'] or '-'}, {s['split'] or '-'}): {status}",
        "",
        f"{e['input_tokens'] + e['output_tokens']} tokens, {e['model_calls']} model call(s), "
        f"{e['tool_calls']} tool call(s), {_fmt(e['wall_seconds'])} s",
    ]
    items = [f"- failure: {f}" for f in s["failures"]]
    conv = s.get("conversation")
    if conv and "relation" in conv:
        state = conv.get("state") or {}
        items.append(
            f"- turn {conv['turn']} of {conv['id']}: relation {conv['relation']} "
            f"({conv['relation_source']}), context {conv['context_tokens']} tokens from "
            f"{conv['context_turns']} turn(s) ({conv['context_omitted']} part(s) left out), "
            f"recalled {conv['recalled'] or 'nothing'}"
        )
        items.append(
            f"- task state: places {state.get('places')}, metrics {state.get('metrics')}, "
            f"period {state.get('period')}, pending {state.get('pending')}"
        )
    replay = s.get("replay")
    if replay:
        items.append(f"- replay of run {replay['of']}: {replay['outcome']}")
        items += [f"  - {change}" for change in replay["changes"]]
    if "tools" in s:
        t, n = s["tools"], s["numbers"]
        items += [f"- unsupported number(s) {u['numbers']} in {u['sources']}: {u['text']}"
                  for u in n["unsupported"]]  # fmt: skip
        items += [f"- low support ({u['overlap']}) for {u['sources']}: {u['text']}"
                  for u in s["support"]["low"]]  # fmt: skip
        items += [f"- redundant {r['tool']} {r['id']} ({r['why']} as {r['repeats']})"
                  for r in t["redundant"]]  # fmt: skip
        items += [f"- uncited {u['tool']} {u['id']}: {u['why']}" for u in t["uncited"]]
        errors = {k: v for k, v in t["errors"].items() if v}
        if errors:
            items.append(f"- tool errors: {errors}")
        if s["trajectory"]["stop"] == "forced":
            items.append(f"- forced stop by the {s['trajectory']['forced_by']} budget")
        kp = s["key_points"]
        if kp["total"]:
            items.append(f"- key points covered: {len(kp['covered'])} of {kp['total']}")
    judge = s.get("judge")
    if judge:
        items.append(
            f"- judge ({judge['model']}): {judge['supported']} supported, {judge['partial']} "
            f"partial, {judge['unsupported']} unsupported, {judge['unjudged']} not judged, "
            f"score {_fmt(judge.get('score'))}"
        )
        items += [f"  - {f['verdict']} {f['sources']}: {f['text']} -- {f['reason']}"
                  for f in judge["flagged"]]  # fmt: skip
        interp = judge.get("interpretation") or {}
        if interp.get("claims"):
            items.append(
                f"- interpretations: {interp['within']} within, {interp['stretch']} stretch, "
                f"{interp['overreach']} overreach, {interp['unjudged']} not judged, "
                f"score {_fmt(interp['score'])}"
            )
            items += [f"  - {f['verdict']}: {f['text']} -- {f['reason']}"
                      for f in interp["flagged"]]  # fmt: skip
        for point in judge.get("key_points") or []:
            items.append(f"  - key point ({point['coverage']}, {point['consistency']}): "
                         f"{point['text']} -- {point['reason']}")  # fmt: skip
    lines += ["", *items] if items else []
    unjudged = s.get("retrieval", {}).get("unjudged", [])
    if unjudged:
        lines += ["", f"Unjudged works (label in evals/qrels.toml under [{s['case_id']}]):", ""]
        lines += [f"- {w['id']} {w['title']}" for w in unjudged]
    return [*lines, ""]


def render_report(scores: list[dict[str, Any]], agg: dict[str, Any]) -> str:
    lines = ["# Evaluation report", "",
             "Metrics are reported only; failures come from the case checks.", "",
             "## Totals", ""]  # fmt: skip
    rows = [("all", agg["all"])]
    rows += [(f"category {k}", v) for k, v in agg["by_category"].items()]
    rows += [(f"split {k}", v) for k, v in agg["by_split"].items()]
    lines += _table(rows)
    conv = agg.get("conversations") or {}
    if conv.get("total"):
        lines += ["", f"Conversations: {conv['passed']}/{conv['total']} with every turn passed"]
    replay = agg["all"]["replay"]
    if any(replay.values()):
        lines += ["", "Replay: " + ", ".join(f"{v} {k}" for k, v in replay.items())]
    lines.append("")
    lines += judge_tables(scores, agg)  # empty when nothing has a judge section
    lines += ["## Cases", ""]
    for s in scores:
        lines += _case_section(s)
    return "\n".join(lines)


def write_scores(
    directory: Path, scores: list[dict[str, Any]], agg: dict[str, Any]
) -> tuple[Path, Path]:
    """``score.json`` (byte-identical for the same records) and ``report.md``."""
    directory.mkdir(parents=True, exist_ok=True)
    score_path = directory / SCORE_FILE
    score_path.write_text(
        json.dumps({"aggregate": agg, "cases": scores}, indent=1, sort_keys=True,
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    report_path = directory / REPORT_FILE
    report_path.write_text(render_report(scores, agg), encoding="utf-8")
    return score_path, report_path
