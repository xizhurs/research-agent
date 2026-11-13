"""Evaluation: cases, checks, budgeted paid runs, offline scoring and replay of records.

Tiers: ``pytest`` (stubbed model) and ``eval score`` (stored run records) are offline and free;
``eval run --live`` runs the real agent within a run budget and writes one record per case;
``eval replay`` feeds a run's recorded model replies through the current code (offline, free).
"""

from research_agent.evaluation.cases import (
    CATEGORIES,
    KNOWN_TOOLS,
    METRICS,
    SPLITS,
    EvalCase,
    EvalError,
    ExpectedNumber,
    KeyPoint,
    Qrels,
    load_cases,
    load_qrels,
    select_cases,
)
from research_agent.evaluation.checks import check_record, check_result
from research_agent.evaluation.metrics import ScoreSettings, score_record
from research_agent.evaluation.numbers import extract_numbers, unsupported_numbers, value_found
from research_agent.evaluation.recording import (
    EvalBudgetExceeded,
    RecordingLLMClient,
    RunBudget,
    fingerprint,
    request_hash,
    tools_hash,
)
from research_agent.evaluation.replay import (
    ReplayDiverged,
    ReplayLLMClient,
    compare_answers,
    replay_date,
    replay_section,
)
from research_agent.evaluation.report import (
    aggregate,
    flags,
    metrics_line,
    score_records,
    write_scores,
)
from research_agent.evaluation.runner import (
    CaseReport,
    EvalAgent,
    EvalSummary,
    read_records,
    record_failures,
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
    "ExpectedNumber",
    "KeyPoint",
    "Qrels",
    "RecordingLLMClient",
    "ReplayDiverged",
    "ReplayLLMClient",
    "RunBudget",
    "ScoreSettings",
    "aggregate",
    "check_record",
    "check_result",
    "compare_answers",
    "extract_numbers",
    "fingerprint",
    "flags",
    "load_cases",
    "load_qrels",
    "metrics_line",
    "read_records",
    "record_failures",
    "replay_date",
    "replay_section",
    "request_hash",
    "rescore",
    "run_case",
    "run_cases",
    "score_record",
    "score_records",
    "select_cases",
    "summarize",
    "summarize_records",
    "tools_hash",
    "unsupported_numbers",
    "value_found",
    "write_scores",
    "write_summary",
]
