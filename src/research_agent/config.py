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
class Config:
    limits: Limits
    openalex: OpenAlexConfig


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
    )
