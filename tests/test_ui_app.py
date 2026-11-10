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
    views = [b for b in demo.blocks.values() if isinstance(b, gr.Radio) and b.value == "daily"]
    assert len(views) == 2  # a Daily/Monthly toggle on the Ask tab and on the Rainfall tab
