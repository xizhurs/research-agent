import logging
import sys

import pytest

from research_agent.credentials import Credentials, Secret
from research_agent.logging_setup import (
    REDACTED,
    REDACTOR,
    SecretRedactor,
    configure_logging,
    level_for,
    reset_logging,
)

KEY = "sk-test-0123456789abcdef"
OTHER = "oa-test-fedcba9876543210"


@pytest.fixture(autouse=True)
def clean_logging():
    reset_logging()
    yield
    reset_logging()


def emit(name, level, message, *args, **kwargs):
    logging.getLogger(name).log(level, message, *args, **kwargs)


# --- levels and routing ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verbosity", "level"),
    [
        (0, logging.WARNING),
        (1, logging.INFO),
        (2, logging.DEBUG),
        (5, logging.DEBUG),
        (-1, logging.WARNING),
    ],
)
def test_level_for(verbosity, level):
    assert level_for(verbosity) == level


def test_default_shows_warnings_but_not_progress(capsys):
    configure_logging(0)
    emit("research_agent.agent.loop", logging.INFO, "progress")
    emit("research_agent.agent.loop", logging.WARNING, "something odd")
    err = capsys.readouterr().err
    assert "something odd" in err
    assert "progress" not in err


def test_verbose_shows_progress_on_stderr_not_stdout(capsys):
    configure_logging(1)
    emit("research_agent.agent.loop", logging.INFO, "turn 1 done")
    emit("research_agent.agent.loop", logging.DEBUG, "fine detail")
    captured = capsys.readouterr()
    assert "turn 1 done" in captured.err and "INFO" in captured.err
    assert "fine detail" not in captured.err
    assert captured.out == ""


def test_very_verbose_shows_debug(capsys):
    configure_logging(2)
    emit("research_agent.retrieval.openalex", logging.DEBUG, "cache miss")
    assert "cache miss" in capsys.readouterr().err


def test_third_party_loggers_stay_quiet_even_at_max_verbosity(capsys):
    configure_logging(2)
    for name in ("httpx", "openai", "httpcore"):
        emit(name, logging.DEBUG, "request body with the whole prompt")
        emit(name, logging.INFO, "HTTP Request: POST ...")
    assert capsys.readouterr().err == ""


def test_configure_is_idempotent_no_duplicate_lines(capsys):
    configure_logging(1)
    configure_logging(1)
    emit("research_agent.x", logging.INFO, "once")
    assert capsys.readouterr().err.count("once") == 1


def test_handler_follows_a_swapped_stderr(capsys, monkeypatch):
    import io

    configure_logging(1)
    replacement = io.StringIO()
    monkeypatch.setattr(sys, "stderr", replacement)
    emit("research_agent.x", logging.INFO, "to the new stream")
    assert "to the new stream" in replacement.getvalue()


# --- secret redaction ------------------------------------------------------------------------


def test_registered_secrets_are_redacted_from_messages_and_args(capsys):
    configure_logging(1)
    REDACTOR.add(KEY)
    emit("research_agent.x", logging.INFO, "key is %s and %s", KEY, "fine")
    emit("research_agent.x", logging.WARNING, f"fstring {KEY}")
    err = capsys.readouterr().err
    assert KEY not in err
    assert f"key is {REDACTED} and fine" in err
    assert f"fstring {REDACTED}" in err


def test_secrets_are_redacted_from_tracebacks(capsys):
    configure_logging(1)
    REDACTOR.add(KEY)
    try:
        raise RuntimeError(f"boom with {KEY}")
    except RuntimeError:
        logging.getLogger("research_agent.x").error("failed", exc_info=True)
    err = capsys.readouterr().err
    assert KEY not in err
    assert "RuntimeError: boom with " + REDACTED in err
    assert "Traceback" in err  # the traceback itself is still there


def test_add_credentials_registers_both_keys(capsys):
    configure_logging(1)
    REDACTOR.add_credentials(Credentials(Secret(OTHER), Secret(KEY)))
    emit("research_agent.x", logging.INFO, "%s %s", KEY, OTHER)
    assert capsys.readouterr().err.count(REDACTED) == 2


def test_missing_credentials_are_fine():
    redactor = SecretRedactor()
    redactor.add_credentials(Credentials(None, None))
    assert redactor.redact("nothing to hide") == "nothing to hide"


def test_longest_secret_is_replaced_first():
    redactor = SecretRedactor()
    redactor.add("abcdefgh")
    redactor.add("abcdefghijkl")
    assert redactor.redact("x abcdefghijkl y") == f"x {REDACTED} y"


def test_tiny_strings_are_never_treated_as_secrets():
    redactor = SecretRedactor()
    redactor.add("abc")
    redactor.add("")
    redactor.add(None)
    assert redactor.redact("abc def") == "abc def"


def test_without_secrets_messages_pass_through_untouched(capsys):
    configure_logging(1)
    emit("research_agent.x", logging.INFO, "value=%d", 5)
    assert "value=5" in capsys.readouterr().err
