"""Logging configuration for the CLI: verbosity levels, format, and secret redaction.

- Logs go to stderr so stdout stays clean for answers.
- Only the ``research_agent`` loggers follow the verbosity flag. Third-party loggers (httpx,
  openai, ...) stay at WARNING because their DEBUG output can contain full prompts.
- Any registered secret (API key) is replaced by ``<redacted>`` in every log line and in
  tracebacks attached to log records, as a second line of defence behind ``Secret``.
"""

from __future__ import annotations

import logging
import sys

from research_agent.credentials import Credentials

REDACTED = "<redacted>"
MIN_SECRET_CHARS = 8  # never "redact" tiny strings: that would mangle ordinary text
_HANDLER_TAG = "_research_agent_handler"
_THIRD_PARTY = ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "opentelemetry")


class SecretRedactor(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._secrets: set[str] = set()

    def add(self, secret: str | None) -> None:
        if secret and len(secret) >= MIN_SECRET_CHARS:
            self._secrets.add(secret)

    def add_credentials(self, credentials: Credentials) -> None:
        keys = (credentials.openalex_api_key, credentials.openai_api_key,
                credentials.claude_api_key)  # fmt: skip
        for key in keys:
            if key is not None:
                self.add(key.reveal())

    def clear(self) -> None:
        self._secrets.clear()

    def redact(self, text: str) -> str:
        for secret in sorted(self._secrets, key=len, reverse=True):
            text = text.replace(secret, REDACTED)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        record.msg = self.redact(record.getMessage())
        record.args = None
        if record.exc_info:
            # Always rebuild the traceback text: another handler may already have cached an
            # unredacted ``exc_text`` on this record, and Formatter reuses whatever is there.
            record.exc_text = self.redact(logging.Formatter().formatException(record.exc_info))
        return True


REDACTOR = SecretRedactor()


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is at emit time (safe when streams are swapped)."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr


def level_for(verbosity: int) -> int:
    """0 -> WARNING, 1 -> INFO, 2 or more -> DEBUG."""
    if verbosity <= 0:
        return logging.WARNING
    return logging.INFO if verbosity == 1 else logging.DEBUG


def _remove_our_handlers(root: logging.Logger) -> None:
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)


def configure_logging(verbosity: int = 0) -> None:
    """Install the stderr handler (idempotent) and set the project's log level."""
    root = logging.getLogger()
    _remove_our_handlers(root)
    handler = _StderrHandler()
    setattr(handler, _HANDLER_TAG, True)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    )
    handler.addFilter(REDACTOR)
    root.addHandler(handler)
    logging.getLogger("research_agent").setLevel(level_for(verbosity))
    for name in _THIRD_PARTY:
        logging.getLogger(name).setLevel(logging.WARNING)


def reset_logging() -> None:
    """Undo ``configure_logging`` (used by tests)."""
    _remove_our_handlers(logging.getLogger())
    logging.getLogger("research_agent").setLevel(logging.NOTSET)
    for name in _THIRD_PARTY:
        logging.getLogger(name).setLevel(logging.NOTSET)
    REDACTOR.clear()
