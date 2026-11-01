"""Credential loading and validation.

Sources, highest priority first: the process environment, then a local ``.env`` file.
Loading never mutates ``os.environ``. Secrets are wrapped in ``Secret`` so they cannot leak
through ``repr``, ``str``, f-strings, logging or exception messages by accident; the raw value
is only available via an explicit ``reveal()``. Error messages name the variable, never the value.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_ENV_FILE = Path(".env")
ENV_FILE_ENV_VAR = "RESEARCH_AGENT_ENV_FILE"

OPENALEX_VAR = "OPENALEX_API_KEY"
OPENAI_VAR = "OPENAI_API_KEY"

MIN_KEY_LENGTH = 8
# Values copied from .env.example (or similar) that were never replaced.
_PLACEHOLDER = re.compile(r"^(your[-_ ]|<|changeme|xxx)", re.IGNORECASE)


class CredentialError(ValueError):
    """A credential is missing or malformed. The message never contains the secret value."""


class Secret:
    """Holds a secret string; every implicit string conversion is redacted."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(<redacted>)"

    __str__ = __repr__

    def __format__(self, format_spec: str) -> str:
        return "<redacted>"


@dataclass(frozen=True)
class Credentials:
    openalex_api_key: Secret | None
    openai_api_key: Secret | None

    def require_openai_key(self) -> Secret:
        if self.openai_api_key is None:
            raise CredentialError(
                f"{OPENAI_VAR} is not set. Add it to your .env file or the environment."
            )
        return self.openai_api_key


def _validate(name: str, raw: str | None) -> Secret | None:
    """Return a Secret, or None when unset/blank. Raise CredentialError when malformed."""
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if _PLACEHOLDER.match(value):
        raise CredentialError(
            f"{name} still has a placeholder value. Put your real key in, or leave it blank."
        )
    if re.search(r"\s", value):
        raise CredentialError(f"{name} contains whitespace; check for a stray space or newline.")
    if len(value) < MIN_KEY_LENGTH:
        raise CredentialError(f"{name} is too short to be an API key.")
    return Secret(value)


def load_credentials(
    env_file: Path | None = DEFAULT_ENV_FILE, environ: Mapping[str, str] | None = None
) -> Credentials:
    """Load and validate credentials. The process environment overrides the .env file."""
    merged: dict[str, str | None] = {}
    if env_file is not None and env_file.is_file():
        merged.update(dotenv_values(env_file))
    merged.update(os.environ if environ is None else environ)
    return Credentials(
        openalex_api_key=_validate(OPENALEX_VAR, merged.get(OPENALEX_VAR)),
        openai_api_key=_validate(OPENAI_VAR, merged.get(OPENAI_VAR)),
    )
