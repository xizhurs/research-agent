"""Running cases and keeping one JSON run record per case, then summarising records.

A run record holds everything later scoring needs (the case, every model call with its request
hash, usage and latency, every tool call with the text the model saw, the answer with the cited
works' text and analyses, the stats and the failures), so ``eval score`` never runs the agent.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from research_agent.agent import AgentResult, InvalidQuestionError
from research_agent.config import Limits
from research_agent.evaluation.cases import EvalCase
from research_agent.evaluation.checks import answer_dict, check_record, check_result, stats_dict
from research_agent.evaluation.recording import (
    EvalBudgetExceeded,
    RecordingLLMClient,
    RunBudget,
    fingerprint,
)
from research_agent.evaluation.replay import ReplayDiverged
from research_agent.llm import LLMAuthError, LLMError
from research_agent.retrieval import AuthError, RetrievalError

SCHEMA_VERSION = 1
STATUSES = ("ok", "failed", "aborted", "skipped", "diverged")
# failed: the run failed; aborted/skipped: the run budget; diverged: a replay request differed


class EvalAgent(Protocol):
    def run(self, question: str) -> AgentResult: ...


# --- the original one-case API (kept for callers and tests) ------------------------------------


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


# --- run records -------------------------------------------------------------------------------


def git_commit(cwd: Path | None = None) -> str | None:
    """The current commit, or None outside a git checkout (best effort, never raises)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _tool_calls(log_calls: list[dict], tool_results: dict[str, str], events) -> list[dict]:
    """Every tool call that ran, with the result text the model saw and its recorded outcome.

    A call ran if its result was sent back to the model; results are in execution order, like
    the agent's events. Calls in a turn made with tools disabled were ignored by the loop and
    have no result. A call no recorded model reply asked for (the fixed-search mode's scripted
    search) takes its name and arguments from the agent's event and has turn None."""
    requested = {}
    for turn, call in enumerate(log_calls, start=1):
        for tool_call in call["tool_calls"]:
            requested[tool_call["id"]] = (turn, tool_call)
    out = []
    for index, (call_id, result) in enumerate(tool_results.items()):
        event = events[index] if index < len(events) else None
        turn, tool_call = requested.get(call_id, (None, None))
        if tool_call is None:
            tool_call = {"name": event.tool if event else None,
                         "arguments": dict(event.arguments) if event else {}}  # fmt: skip
        out.append(
            {
                "turn": turn,
                "id": call_id,
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
                "status": event.status if event else None,
                "result_id": event.result_id if event else None,
                "result": result,
            }
        )
    return out


def build_record(
    case: EvalCase,
    *,
    run_id: str,
    meta: dict[str, Any],
    status: str,
    failures: Iterable[str] = (),
    result: AgentResult | None = None,
    log=None,
    wall_seconds: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    calls = list(log.calls) if log else []
    instructions = log.instructions if log else None
    return {
        "schema": SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case.id,
        "split": case.split,
        "category": case.category,
        "mode": meta.get("mode", "agentic"),
        "question": case.question,
        "git_commit": meta.get("git_commit"),
        "model": meta.get("model"),
        "config_hash": meta.get("config_hash"),
        "prompt_hash": fingerprint(instructions) if instructions else None,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "today": meta.get("today"),  # the date the tools and the agent used (replay pins it)
        "status": status,
        "error": error,
        "wall_seconds": None if wall_seconds is None else round(wall_seconds, 3),
        "llm_calls": calls,
        "tool_calls": (
            _tool_calls(calls, log.tool_results, result.events) if log and result else []
        ),
        "answer": answer_dict(result) if result else None,
        "stats": stats_dict(result) if result else None,
        "failures": list(failures),
    }


def write_record(directory: Path, record: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['case_id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def read_records(directory: Path) -> list[dict[str, Any]]:
    """Every run record in ``directory`` (``summary.json`` and other files are ignored)."""
    if not directory.is_dir():
        raise FileNotFoundError(f"no run directory: {directory}")
    records = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema") == SCHEMA_VERSION and "case_id" in data:
            records.append(data)
    return records


def run_cases(
    cases: list[EvalCase],
    run: Callable[[EvalCase], AgentResult],
    recorder: RecordingLLMClient,
    limits: Limits,
    *,
    out_dir: Path,
    run_id: str,
    meta: dict[str, Any],
    on_record: Callable[[dict[str, Any]], None] = lambda record: None,
    clock: Callable[[], float] = time.perf_counter,
    annotate: Callable[[EvalCase, dict[str, Any]], dict[str, Any]] = lambda case, record: record,
) -> list[dict[str, Any]]:
    """Run each case within the run budget, writing one record per case to ``out_dir``.

    A case is skipped when the budget cannot cover one more worst-case case; a case cut off by
    the budget mid-way is recorded as aborted, and a replayed case whose request differed from
    the recording as diverged. Auth failures abort the whole run (re-raised). ``annotate`` may
    add to each record before it is written (replay adds its comparison).
    """
    records = []
    for case in cases:
        reason = recorder.budget.case_refusal(limits.max_total_tokens)
        recorder.start_case()
        if reason:
            record = build_record(case, run_id=run_id, meta=meta, status="skipped",
                                  failures=[f"skipped: {reason}"], error=reason)
        else:
            started = clock()
            try:
                result = run(case)
            except (LLMAuthError, AuthError):
                raise
            except EvalBudgetExceeded as exc:
                record = build_record(
                    case, run_id=run_id, meta=meta, status="aborted", log=recorder.log,
                    failures=[f"aborted: {exc}"], wall_seconds=clock() - started, error=str(exc),
                )
            except ReplayDiverged as exc:
                record = build_record(
                    case, run_id=run_id, meta=meta, status="diverged", log=recorder.log,
                    failures=[f"diverged at {exc}"], wall_seconds=clock() - started,
                    error=str(exc),
                )
            except (LLMError, RetrievalError, InvalidQuestionError) as exc:
                message = f"{type(exc).__name__}: {exc}"
                record = build_record(
                    case, run_id=run_id, meta=meta, status="failed", log=recorder.log,
                    failures=[f"run failed: {message}"], wall_seconds=clock() - started,
                    error=message,
                )
            else:
                record = build_record(
                    case, run_id=run_id, meta=meta, status="ok", log=recorder.log,
                    result=result, wall_seconds=clock() - started,
                )
                record["failures"] = record_failures(case, record, limits)
        record = annotate(case, record)
        write_record(out_dir, record)
        records.append(record)
        on_record(record)
    return records


def rescore(
    records: list[dict[str, Any]], cases: list[EvalCase], limits: Limits
) -> list[dict[str, Any]]:
    """The records with ``failures`` recomputed from the current cases and checks (offline).

    Records of runs that did not finish keep their failures; a case no longer in the case file
    is reported as such."""
    by_id = {c.id: c for c in cases}
    out = []
    for record in records:
        case = by_id.get(record["case_id"])
        if record["status"] != "ok":
            out.append(record)
        elif case is None:
            out.append({**record, "failures": ["case no longer in the case file"]})
        else:
            out.append({**record, "failures": record_failures(case, record, limits)})
    return out


def record_failures(case: EvalCase, record: dict[str, Any], limits: Limits) -> list[str]:
    """The checks of a completed record, including the tool side of ``expect_numbers``."""
    tool_results = [call["result"] for call in record["tool_calls"]]
    return check_record(case, record["answer"], record["stats"], limits, tool_results)


# --- summary -----------------------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def summarize_records(
    records: list[dict[str, Any]], budget: RunBudget | None = None
) -> dict[str, Any]:
    """Totals over run records: outcomes, tokens, estimated cost, calls and latency, per
    category too. Cost is None without configured prices."""
    ok = [r for r in records if r["status"] == "ok"]
    stats = [r["stats"] for r in ok]
    tin = sum(c["usage"]["input"] for r in records for c in r["llm_calls"])
    tout = sum(c["usage"]["output"] for r in records for c in r["llm_calls"])
    categories: dict[str, dict[str, int]] = {}
    for r in records:
        entry = categories.setdefault(r.get("category") or "none", {"cases": 0, "passed": 0})
        entry["cases"] += 1
        entry["passed"] += int(r["status"] == "ok" and not r["failures"])
    return {
        "cases": len(records),
        "passed": sum(1 for r in ok if not r["failures"]),
        "failed_checks": sum(1 for r in ok if r["failures"]),
        "run_failed": sum(1 for r in records if r["status"] == "failed"),
        "aborted": sum(1 for r in records if r["status"] == "aborted"),
        "skipped": sum(1 for r in records if r["status"] == "skipped"),
        "diverged": sum(1 for r in records if r["status"] == "diverged"),
        "model_calls": sum(len(r["llm_calls"]) for r in records),
        "input_tokens": tin,
        "output_tokens": tout,
        "estimated_cost_usd": round(budget.cost(tin, tout), 4)
        if budget is not None and budget.priced
        else None,
        "searches": sum(s["search_calls"] for s in stats),
        "data_analyses": sum(s["rainfall_calls"] for s in stats),
        "rejected_citations": sum(len(r["answer"]["rejected_citations"]) for r in ok),
        "unstructured": sum(1 for r in ok if not r["answer"]["structured"]),
        "median_wall_seconds": _median([r["wall_seconds"] for r in ok if r["wall_seconds"]]),
        "median_tokens_per_case": _median([s["total_tokens"] for s in stats]),
        "median_model_call_seconds": _median(
            [c["latency_s"] for r in records for c in r["llm_calls"]]
        ),
        "categories": categories,
    }


def write_summary(directory: Path, summary: dict[str, Any]) -> Path:
    path = directory / "summary.json"
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return path
