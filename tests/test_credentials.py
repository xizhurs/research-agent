import logging
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.credentials import (
    CredentialError,
    Credentials,
    Secret,
    load_credentials,
)

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()

REAL_LOOKING = "sk-test-0123456789abcdef"
OTHER = "oa-test-fedcba9876543210"


def env_file(tmp_path, text):
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return p


def test_secret_never_leaks_implicitly(caplog):
    s = Secret(REAL_LOOKING)
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("t").warning("value=%s repr=%r", s, s)
    percent = "%s" % s  # noqa: UP031 (deliberately exercising %-formatting)
    dotformat = "{}".format(s)  # noqa: UP032 (deliberately exercising str.format)
    rendered = [repr(s), str(s), f"{s}", f"{s!s}", percent, dotformat, caplog.text]
    assert all(REAL_LOOKING not in r for r in rendered)
    assert s.reveal() == REAL_LOOKING


def test_credentials_repr_does_not_leak():
    creds = Credentials(Secret(REAL_LOOKING), Secret(OTHER))
    assert REAL_LOOKING not in repr(creds)
    assert OTHER not in repr(creds)
    assert REAL_LOOKING not in str(creds)


def test_loads_from_env_file_only(tmp_path):
    f = env_file(tmp_path, f"OPENALEX_API_KEY={OTHER}\nOPENAI_API_KEY={REAL_LOOKING}\n")
    creds = load_credentials(f, environ={})
    assert creds.openalex_api_key.reveal() == OTHER
    assert creds.openai_api_key.reveal() == REAL_LOOKING


def test_process_environment_overrides_env_file(tmp_path):
    f = env_file(tmp_path, f"OPENAI_API_KEY={REAL_LOOKING}\n")
    creds = load_credentials(f, environ={"OPENAI_API_KEY": "from-the-environment-1"})
    assert creds.openai_api_key.reveal() == "from-the-environment-1"


def test_missing_file_is_fine_and_keys_are_none(tmp_path):
    creds = load_credentials(tmp_path / "absent", environ={})
    assert creds.openalex_api_key is None and creds.openai_api_key is None


def test_blank_values_count_as_unset(tmp_path):
    f = env_file(tmp_path, "OPENALEX_API_KEY=\nOPENAI_API_KEY=   \n")
    creds = load_credentials(f, environ={})
    assert creds.openalex_api_key is None and creds.openai_api_key is None


def test_loading_does_not_mutate_process_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    f = env_file(tmp_path, f"OPENAI_API_KEY={REAL_LOOKING}\n")
    load_credentials(f)
    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.parametrize(
    "value", ["your-openai-api-key", "Your_Key_Here", "<paste key>", "changeme", "xxx-xxx-xxx"]
)
def test_placeholder_values_rejected(value):
    with pytest.raises(CredentialError, match="OPENAI_API_KEY.*placeholder"):
        load_credentials(None, environ={"OPENAI_API_KEY": value})


def test_env_example_unedited_is_rejected(tmp_path):
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    with pytest.raises(CredentialError, match="placeholder"):
        load_credentials(env_file(tmp_path, example), environ={})


def test_whitespace_and_short_values_rejected_without_echoing_them():
    bad_space = "sk-live-secret part-two"
    with pytest.raises(CredentialError, match="whitespace") as e1:
        load_credentials(None, environ={"OPENAI_API_KEY": bad_space})
    assert "sk-live-secret" not in str(e1.value) and "part-two" not in str(e1.value)

    with pytest.raises(CredentialError, match="too short") as e2:
        load_credentials(None, environ={"OPENALEX_API_KEY": "abc123"})
    assert "abc123" not in str(e2.value)


def test_surrounding_whitespace_is_stripped():
    creds = load_credentials(None, environ={"OPENAI_API_KEY": f"  {REAL_LOOKING}\n"})
    assert creds.openai_api_key.reveal() == REAL_LOOKING


def test_require_openai_key():
    creds = load_credentials(None, environ={})
    with pytest.raises(CredentialError, match="OPENAI_API_KEY is not set"):
        creds.require_openai_key()
    creds = load_credentials(None, environ={"OPENAI_API_KEY": REAL_LOOKING})
    assert creds.require_openai_key().reveal() == REAL_LOOKING


# --- git hygiene -----------------------------------------------------------------------------


def git_ignored(name):
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", name], cwd=ROOT, capture_output=True
        ).returncode
        == 0
    )


def test_real_env_files_are_git_ignored_but_example_is_not():
    assert git_ignored(".env")
    assert git_ignored(".env.local")
    assert git_ignored(".env.production")
    assert not git_ignored(".env.example")


def test_env_example_contains_only_placeholders():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    values = [
        line.split("=", 1)[1].strip()
        for line in text.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    ]
    assert values, "expected at least one variable in .env.example"
    for v in values:
        with pytest.raises(CredentialError, match="placeholder"):
            load_credentials(None, environ={"OPENAI_API_KEY": v})


# --- CLI -------------------------------------------------------------------------------------


def test_check_credentials_reports_status_without_printing_values(tmp_path):
    f = env_file(tmp_path, f"OPENALEX_API_KEY={OTHER}\nOPENAI_API_KEY={REAL_LOOKING}\n")
    result = runner.invoke(app, ["check-credentials", "--env-file", str(f)])
    assert result.exit_code == 0
    assert "OPENALEX_API_KEY: set" in result.output
    assert "OPENAI_API_KEY: set" in result.output
    assert REAL_LOOKING not in result.output and OTHER not in result.output


def test_check_credentials_unset_is_ok_unless_required(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    args = ["check-credentials", "--env-file", str(tmp_path / "absent")]
    assert runner.invoke(app, args).exit_code == 0
    required = runner.invoke(app, [*args, "--require-openai"])
    assert required.exit_code == 1
    assert "OPENAI_API_KEY is required" in required.output


def test_check_credentials_malformed_exits_nonzero_without_leaking(tmp_path):
    f = env_file(tmp_path, "OPENAI_API_KEY=sk-has a-space-inside\n")
    result = runner.invoke(app, ["check-credentials", "--env-file", str(f)])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output and "whitespace" in result.output
    assert "sk-has" not in result.output


def test_env_file_can_come_from_environment_variable(tmp_path, monkeypatch):
    f = env_file(tmp_path, f"OPENAI_API_KEY={REAL_LOOKING}\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("RESEARCH_AGENT_ENV_FILE", str(f))
    result = runner.invoke(app, ["check-credentials"])
    assert result.exit_code == 0
    assert "OPENAI_API_KEY: set" in result.output
