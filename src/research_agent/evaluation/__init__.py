"""Evaluation: cases, deterministic checks, budgeted paid runs and offline scoring of records.

Tiers: ``pytest`` (stubbed model) and ``eval score`` (stored run records) are offline and free;
``eval run --live`` runs the real agent within a run budget and writes one record per case.
"""

from research_agent.evaluation.cases import (
    CATEGORIES,
    KNOWN_TOOLS,
    METRICS,
    SPLITS,
    EvalCase,
    EvalError,
    load_cases,
    select_cases,
)
from research_agent.evaluation.checks import check_record, check_result
from research_agent.evaluation.recording import (
    EvalBudgetExceeded,
    RecordingLLMClient,
    RunBudget,
    request_hash,
)
from research_agent.evaluation.runner import (
    CaseReport,
    EvalAgent,
    EvalSummary,
    read_records,
    rescore,
    run_case,
    run_cases,
    summarize,
    summarize_records,
    write_summary,
)

__all__ = [
    "CATEGORIES",
    "KNOWN_TOOLS",
    "METRICS",
    "SPLITS",
    "CaseReport",
    "EvalAgent",
    "EvalBudgetExceeded",
    "EvalCase",
    "EvalError",
    "EvalSummary",
    "RecordingLLMClient",
    "RunBudget",
    "check_record",
    "check_result",
    "load_cases",
    "read_records",
    "request_hash",
    "rescore",
    "run_case",
    "run_cases",
    "select_cases",
    "summarize",
    "summarize_records",
    "write_summary",
]
