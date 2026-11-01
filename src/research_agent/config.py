"""Typed configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or invalid."""


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
            # bool is an int subclass; reject it explicitly.
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"limits.{f.name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class Config:
    limits: Limits


def load_config(path: Path) -> Config:
    """Load and validate the configuration at ``path``."""
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    limits_raw = raw.get("limits")
    if not isinstance(limits_raw, dict):
        raise ConfigError(f"{path}: missing [limits] table")

    expected = {f.name for f in fields(Limits)}
    missing = expected - limits_raw.keys()
    unknown = limits_raw.keys() - expected
    if missing:
        raise ConfigError(f"{path}: missing limits keys: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{path}: unknown limits keys: {sorted(unknown)}")

    return Config(limits=Limits(**limits_raw))
