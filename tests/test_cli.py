from pathlib import Path

from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.retrieval import RateLimitError, SearchResult, Work

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


def search_args(tmp_path):
    return ["search", "q", "--config", str(BASE), "--env-file", str(tmp_path / "none")]


class FakeClient:
    """Stands in for OpenAlexClient so CLI tests make no network calls."""

    outcome: object = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def search(self, query, *, year_from=None, max_results=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_search_prints_results(monkeypatch, tmp_path):
    work = Work(
        id="W1", url="https://openalex.org/W1", title="A title", year=2024, date=None,
        authors=("A", "B", "C", "D"), venue=None, doi=None, type=None, language=None,
        cited_by_count=None, abstract=None, abstract_truncated=False,
    )
    FakeClient.outcome = SearchResult("q", None, (work,), 10, 0, from_cache=True)
    monkeypatch.setattr("research_agent.cli.OpenAlexClient", FakeClient)
    result = runner.invoke(app, search_args(tmp_path))
    assert result.exit_code == 0
    assert "1 of 10 matches (cache)" in result.output
    assert "[W1] A title (2024) - A, B, C et al." in result.output
    assert "1 result(s) have no abstract" in result.output


def test_search_retrieval_error_exits_nonzero(monkeypatch, tmp_path):
    FakeClient.outcome = RateLimitError("budget exhausted")
    monkeypatch.setattr("research_agent.cli.OpenAlexClient", FakeClient)
    result = runner.invoke(app, search_args(tmp_path))
    assert result.exit_code == 1
    assert "budget exhausted" in result.output


def test_show_config_bad_path_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["show-config", "--config", str(tmp_path / "missing.toml")])
    assert result.exit_code == 1
