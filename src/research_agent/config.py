"""Typed configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or invalid."""


def _is_number(value: object) -> bool:
    # bool is an int subclass; reject it explicitly.
    return isinstance(value, int | float) and not isinstance(value, bool)


@dataclass(frozen=True)
class Limits:
    """Per-question resource limits. All values must be positive integers."""

    max_iterations: int
    max_search_calls: int
    results_per_search: int
    abstract_max_chars: int
    max_context_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    max_question_chars: int
    max_rainfall_calls: int

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"limits.{f.name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class OpenAlexConfig:
    """OpenAlex client settings. Credentials are never stored here (env var only)."""

    base_url: str
    timeout_seconds: float
    max_retries: int
    backoff_base_seconds: float
    max_backoff_seconds: float
    cache_dir: Path
    cache_ttl_days: float

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.startswith(
            ("https://", "http://")
        ):
            raise ConfigError(f"openalex.base_url must be an http(s) URL, got {self.base_url!r}")
        if not isinstance(self.cache_dir, Path):
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        for name in (
            "timeout_seconds",
            "backoff_base_seconds",
            "max_backoff_seconds",
            "cache_ttl_days",
        ):
            value = getattr(self, name)
            if not _is_number(value) or value <= 0:
                raise ConfigError(f"openalex.{name} must be a positive number, got {value!r}")
        retries = self.max_retries
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ConfigError(f"openalex.max_retries must be an integer >= 0, got {retries!r}")


@dataclass(frozen=True)
class RainfallConfig:
    """Open-Meteo geocoding + ERA5 archive settings and the thresholds of the rainfall analysis."""

    geocoding_url: str
    archive_url: str
    model: str  # pinned reanalysis product; the API's default mixes products (verified live)
    timeout_seconds: float
    max_retries: int
    backoff_base_seconds: float
    max_backoff_seconds: float
    cache_dir: Path
    cache_ttl_days: float
    earliest_year: int  # first year the archive serves
    default_start_year: int
    baseline_start_year: int
    baseline_end_year: int
    min_years: int  # fewer valid years than this: no ranking / no baseline comparison
    min_year_coverage: float  # a year needs at least this fraction of days with data
    top_events: int
    geocode_candidates: int
    dominance_ratio: float  # top candidate wins only if its population is this many times larger
    same_place_km: float  # a candidate this close to a larger one is merged into it (districts)
    dry_day_mm: float  # a day with less precipitation than this is dry (ETCCDI CDD uses 1 mm)

    def __post_init__(self) -> None:
        for name in ("geocoding_url", "archive_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.startswith(("https://", "http://")):
                raise ConfigError(f"rainfall.{name} must be an http(s) URL, got {value!r}")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigError(f"rainfall.model must be a non-empty string, got {self.model!r}")
        if not isinstance(self.cache_dir, Path):
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        for name in (
            "timeout_seconds",
            "backoff_base_seconds",
            "max_backoff_seconds",
            "cache_ttl_days",
            "dominance_ratio",
            "same_place_km",
            "dry_day_mm",
        ):
            value = getattr(self, name)
            if not _is_number(value) or value <= 0:
                raise ConfigError(f"rainfall.{name} must be a positive number, got {value!r}")
        for name in (
            "earliest_year",
            "default_start_year",
            "baseline_start_year",
            "baseline_end_year",
            "min_years",
            "top_events",
            "geocode_candidates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"rainfall.{name} must be a positive integer, got {value!r}")
        retries = self.max_retries
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ConfigError(f"rainfall.max_retries must be an integer >= 0, got {retries!r}")
        coverage = self.min_year_coverage
        if not _is_number(coverage) or not 0 < coverage <= 1:
            raise ConfigError(f"rainfall.min_year_coverage must be in (0, 1], got {coverage!r}")
        if self.default_start_year < self.earliest_year:
            raise ConfigError("rainfall.default_start_year must not be before earliest_year")
        if self.baseline_start_year > self.baseline_end_year:
            raise ConfigError("rainfall.baseline_start_year must not be after baseline_end_year")


@dataclass(frozen=True)
class EnsoConfig:
    """NOAA CPC RONI file settings and the event rule (threshold, minimum length, strength)."""

    url: str
    timeout_seconds: float
    max_retries: int
    backoff_base_seconds: float
    max_backoff_seconds: float
    cache_dir: Path
    cache_ttl_days: float
    threshold: float  # |index| at or above this counts as El Nino / La Nina (inclusive)
    min_seasons: int  # an event needs this many consecutive overlapping seasons
    strength_moderate: float  # peak |index| at or above: moderate (below: weak)
    strength_strong: float
    strength_very_strong: float

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.startswith(("https://", "http://")):
            raise ConfigError(f"enso.url must be an http(s) URL, got {self.url!r}")
        if not isinstance(self.cache_dir, Path):
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        for name in (
            "timeout_seconds",
            "backoff_base_seconds",
            "max_backoff_seconds",
            "cache_ttl_days",
            "threshold",
            "strength_moderate",
            "strength_strong",
            "strength_very_strong",
        ):
            value = getattr(self, name)
            if not _is_number(value) or value <= 0:
                raise ConfigError(f"enso.{name} must be a positive number, got {value!r}")
        for name in ("max_retries",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(f"enso.{name} must be an integer >= 0, got {value!r}")
        seasons = self.min_seasons
        if isinstance(seasons, bool) or not isinstance(seasons, int) or seasons <= 0:
            raise ConfigError(f"enso.min_seasons must be a positive integer, got {seasons!r}")
        ladder = (
            self.threshold,
            self.strength_moderate,
            self.strength_strong,
            self.strength_very_strong,
        )
        if list(ladder) != sorted(set(ladder)):
            raise ConfigError(
                "enso: threshold < strength_moderate < strength_strong < strength_very_strong "
                "is required"
            )


@dataclass(frozen=True)
class LLMConfig:
    """LLM settings for the OpenAI adapter. The API key is never stored here (env var only)."""

    model: str
    reasoning_effort: str
    timeout_seconds: float
    max_retries: int

    def __post_init__(self) -> None:
        for name in ("model", "reasoning_effort"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"llm.{name} must be a non-empty string, got {value!r}")
        if not _is_number(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ConfigError(
                f"llm.timeout_seconds must be a positive number, got {self.timeout_seconds!r}"
            )
        retries = self.max_retries
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ConfigError(f"llm.max_retries must be an integer >= 0, got {retries!r}")


@dataclass(frozen=True)
class ReportingConfig:
    """Thresholds for the limitations the code adds to every answer."""

    thin_evidence_works: int  # fewer distinct retrieved works than this counts as thin evidence
    stale_after_years: int  # newest retrieved work at least this many years old counts as dated

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"reporting.{f.name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class ConversationConfig:
    """Follow-up questions: how much of earlier turns is sent, and how sessions are kept."""

    max_context_tokens: int  # earlier turns, task state and evidence index sent with a follow-up
    recent_turns: int  # turns sent as messages; older ones become one-line summaries
    answer_digest_chars: int  # an earlier answer is cut to this many characters
    max_stored_evidence: int  # works and analyses kept per session for recall_evidence
    max_recall_per_call: int
    max_sessions: int  # in-memory sessions kept (least recently used dropped first)
    session_ttl_minutes: int  # a session idle this long is dropped

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(
                    f"conversation.{f.name} must be a positive integer, got {value!r}"
                )


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation runs: where records go, the frozen tool cache, and the paid-run budget.

    A price of 0 means unknown (cost is then not estimated); ``max_cost_usd`` 0 means no cost cap.
    """

    runs_dir: Path  # one folder of run records per evaluation run
    tool_cache_dir: Path  # OpenAlex / Open-Meteo / RONI responses kept for reproducible runs
    max_model_calls: int  # per evaluation run, across all cases
    max_total_tokens: int  # per evaluation run, across all cases
    max_cost_usd: float
    price_input_per_mtok: float  # USD per million input tokens of the agent's model
    price_output_per_mtok: float
    redundant_query_jaccard: float  # two searches this similar (word sets) count as redundant
    low_support_overlap: float  # claims sharing fewer of their words with the cited text: flagged

    def __post_init__(self) -> None:
        for name in ("redundant_query_jaccard", "low_support_overlap"):
            value = getattr(self, name)
            if not _is_number(value) or not 0 < value <= 1:
                raise ConfigError(f"eval.{name} must be a number in (0, 1], got {value!r}")
        for name in ("runs_dir", "tool_cache_dir"):
            if not isinstance(getattr(self, name), Path):
                object.__setattr__(self, name, Path(getattr(self, name)))
        for name in ("max_model_calls", "max_total_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"eval.{name} must be a positive integer, got {value!r}")
        for name in ("max_cost_usd", "price_input_per_mtok", "price_output_per_mtok"):
            value = getattr(self, name)
            if not _is_number(value) or value < 0:
                raise ConfigError(f"eval.{name} must be a number >= 0, got {value!r}")


@dataclass(frozen=True)
class Config:
    limits: Limits
    openalex: OpenAlexConfig
    llm: LLMConfig
    reporting: ReportingConfig
    rainfall: RainfallConfig
    enso: EnsoConfig
    conversation: ConversationConfig
    eval: EvalConfig


def _build_section[T](cls: type[T], raw: dict[str, Any], section: str, path: Path) -> T:
    table = raw.get(section)
    if not isinstance(table, dict):
        raise ConfigError(f"{path}: missing [{section}] table")
    expected = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    missing = expected - table.keys()
    unknown = table.keys() - expected
    if missing:
        raise ConfigError(f"{path}: missing {section} keys: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{path}: unknown {section} keys: {sorted(unknown)}")
    return cls(**table)


def load_config(path: Path) -> Config:
    """Load and validate the configuration at ``path``."""
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    return Config(
        limits=_build_section(Limits, raw, "limits", path),
        openalex=_build_section(OpenAlexConfig, raw, "openalex", path),
        llm=_build_section(LLMConfig, raw, "llm", path),
        reporting=_build_section(ReportingConfig, raw, "reporting", path),
        rainfall=_build_section(RainfallConfig, raw, "rainfall", path),
        enso=_build_section(EnsoConfig, raw, "enso", path),
        conversation=_build_section(ConversationConfig, raw, "conversation", path),
        eval=_build_section(EvalConfig, raw, "eval", path),
    )
