"""The Gradio app builds, and the ``ui`` command launches it (or explains the missing extra)."""

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.credentials import Credentials

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"
runner = CliRunner()


def ui_args(tmp_path, *extra):
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    return ["ui", "--config", str(BASE), "--env-file", str(env), *extra]


def test_ui_command_explains_the_missing_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "gradio", None)  # makes `import gradio` fail
    monkeypatch.delitem(sys.modules, "research_agent.ui.app", raising=False)
    result = runner.invoke(app, ui_args(tmp_path))
    assert result.exit_code == 1
    assert "uv sync --extra ui" in result.output


def test_ui_command_launches_on_localhost_by_default(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    launched = {}

    class FakeDemo:
        def launch(self, **kwargs):
            launched.update(kwargs)

    monkeypatch.setattr("research_agent.ui.app.build_app", lambda cfg, creds: FakeDemo())
    assert runner.invoke(app, ui_args(tmp_path)).exit_code == 0
    assert launched == {"server_name": "127.0.0.1", "server_port": 7860}

    launched.clear()
    runner.invoke(app, ui_args(tmp_path, "--host", "0.0.0.0", "--port", "9000"))
    assert launched == {"server_name": "0.0.0.0", "server_port": 9000}  # noqa: S104


def _capture_launch(monkeypatch):
    launched = {}

    class FakeDemo:
        def launch(self, **kwargs):
            launched.update(kwargs)

    monkeypatch.setattr("research_agent.ui.app.build_app", lambda cfg, creds: FakeDemo())
    return launched


def test_ui_login_comes_from_the_environment(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    launched = _capture_launch(monkeypatch)
    monkeypatch.setenv("UI_USERNAME", "alice")
    monkeypatch.setenv("UI_PASSWORD", "s3cret-pw")
    assert runner.invoke(app, ui_args(tmp_path, "--host", "0.0.0.0")).exit_code == 0  # noqa: S104
    assert launched["auth"] == ("alice", "s3cret-pw")


def test_ui_refuses_half_a_login(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    launched = _capture_launch(monkeypatch)
    monkeypatch.setenv("UI_USERNAME", "alice")
    monkeypatch.delenv("UI_PASSWORD", raising=False)
    result = runner.invoke(app, ui_args(tmp_path))
    assert result.exit_code == 1
    assert "UI_PASSWORD" in result.output
    assert launched == {}


def test_ui_warns_when_public_without_login(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    _capture_launch(monkeypatch)
    monkeypatch.delenv("UI_USERNAME", raising=False)
    monkeypatch.delenv("UI_PASSWORD", raising=False)
    result = runner.invoke(app, ui_args(tmp_path, "--host", "0.0.0.0"))  # noqa: S104
    assert result.exit_code == 0
    assert "no login" in result.output


def test_ui_login_can_come_from_the_env_file(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    launched = _capture_launch(monkeypatch)
    monkeypatch.delenv("UI_USERNAME", raising=False)
    monkeypatch.delenv("UI_PASSWORD", raising=False)
    args = ui_args(tmp_path)
    login = "UI_USERNAME=bob\nUI_PASSWORD=long-pass-word\n"
    (tmp_path / ".env").write_text(login, encoding="utf-8")
    assert runner.invoke(app, args).exit_code == 0
    assert launched["auth"] == ("bob", "long-pass-word")


def test_share_needs_a_login(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    launched = _capture_launch(monkeypatch)
    monkeypatch.delenv("UI_USERNAME", raising=False)
    monkeypatch.delenv("UI_PASSWORD", raising=False)
    result = runner.invoke(app, ui_args(tmp_path, "--share"))
    assert result.exit_code == 1 and "--share makes the app public" in result.output
    assert launched == {}


def test_share_with_a_login_asks_gradio_for_a_public_link(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    launched = _capture_launch(monkeypatch)
    monkeypatch.setenv("UI_USERNAME", "alice")
    monkeypatch.setenv("UI_PASSWORD", "s3cret-pw")
    assert runner.invoke(app, ui_args(tmp_path, "--share")).exit_code == 0
    assert launched["share"] is True and launched["auth"] == ("alice", "s3cret-pw")
    assert launched["server_name"] == "127.0.0.1"  # the local server itself stays local


def test_app_builds_with_the_three_tabs_and_no_analytics(base_config):
    gr = pytest.importorskip("gradio")
    from research_agent.ui.app import build_app

    demo = build_app(base_config, Credentials(None, None))
    assert isinstance(demo, gr.Blocks)
    tabs = [b.label for b in demo.blocks.values() if isinstance(b, gr.Tab)]
    assert tabs == ["Ask", "Search", "Rainfall"]
    assert demo.analytics_enabled is False
    badges = [b.value for b in demo.blocks.values() if isinstance(b, gr.HTML)]
    assert [("Uses the LLM" in b) for b in badges] == [True, False, False]
    radios = [b for b in demo.blocks.values() if isinstance(b, gr.Radio)]
    views = [r for r in radios if [v for _, v in r.choices] == ["daily", "monthly", "yearly"]]
    assert len(views) == 3  # plot view on the Ask and Rainfall tabs, time scale on Rainfall
    assert sum(r.label == "Time scale" for r in views) == 1
    years = [b.label for b in demo.blocks.values() if isinstance(b, gr.Number)]
    assert any(label.startswith("Year to plot (press Enter") for label in years)  # Ask tab
