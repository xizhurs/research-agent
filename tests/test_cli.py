from pathlib import Path

from typer.testing import CliRunner

from research_agent.cli import app

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"
runner = CliRunner()


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "show-config" in result.output


def test_show_config_prints_limits():
    result = runner.invoke(app, ["show-config", "--config", str(BASE)])
    assert result.exit_code == 0
    assert "max_iterations = 5" in result.output


def test_show_config_bad_path_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["show-config", "--config", str(tmp_path / "missing.toml")])
    assert result.exit_code == 1
