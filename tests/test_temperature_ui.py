"""Temperature in the CLI and the UI: command, tab handler, Ask-tab plot and the figures.

Fake clients only; the figure tests are skipped without plotly.
"""

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.agent import AgentResult, Answer, Claim, RunStats
from research_agent.cli import app
from research_agent.credentials import Credentials, Secret
from research_agent.rainfall import AmbiguousLocationError
from research_agent.temperature import TemperatureArchive
from research_agent.temperature.analysis import compute_extremes, compute_yearly
from research_agent.ui import handlers

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"
CREDS = Credentials(openalex_api_key=None, openai_api_key=Secret("sk-test-0123456789abcdef"))
runner = CliRunner()


@pytest.fixture
def cfg(rainfall_config):
    return replace(rainfall_config, min_years=3, baseline_start_year=2000, baseline_end_year=2002)


@pytest.fixture
def archive(make_series):
    """2000-2004: max 25, min 5, mean 15 degrees every day; 10 July 2003 reaches 40."""
    dates, _ = make_series(2000, 2004)
    hot = date(2003, 7, 10)
    tmax = tuple(40.0 if d == hot else 25.0 for d in dates)
    return TemperatureArchive(
        tuple(dates), tmax, (5.0,) * len(dates), (15.0,) * len(dates), 34.75, 113.75, 110.0, None
    )


@pytest.fixture
def records(cfg, make_location, make_provenance, archive):
    args = {
        "start_year": 2000,
        "end_year": 2004,
        "last_complete_year": 2004,
        "config": cfg,
        "provenance": make_provenance(),
        "baseline_from_year": 2000,
    }
    loc = make_location()
    hot = compute_extremes(
        loc, archive.dates, archive.tmax, metric="hot", event_date=date(2003, 7, 10), **args
    )
    yearly = compute_yearly(loc, archive.dates, archive.tmean, event_year=2003, **args)
    return {"hot": hot, "yearly": yearly}


class FakeTemperature:
    calls: list = []
    archive_calls: list = []
    outcome = None

    def __init__(self, config):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def analyze(self, city, **kwargs):
        FakeTemperature.calls.append({"city": city, **kwargs})
        if isinstance(FakeTemperature.outcome, Exception):
            raise FakeTemperature.outcome
        return FakeTemperature.outcome, "ARCHIVE"

    def analyze_point(self, latitude, longitude, **kwargs):
        FakeTemperature.calls.append({"point": (latitude, longitude), **kwargs})
        return FakeTemperature.outcome, "ARCHIVE"

    def fetch_archive(self, location, start_year, end_year):
        FakeTemperature.archive_calls.append((location.name, start_year, end_year))
        return "REFETCHED"


@pytest.fixture
def fake(monkeypatch):
    FakeTemperature.calls, FakeTemperature.archive_calls, FakeTemperature.outcome = [], [], None
    monkeypatch.setattr("research_agent.cli.TemperatureClient", FakeTemperature)
    monkeypatch.setattr(handlers, "TemperatureClient", FakeTemperature)
    drawn = []

    def figure(kind):
        def draw(record, archive, **kwargs):
            drawn.append({"kind": kind, "record": record, "archive": archive, **kwargs})
            return f"{kind} figure", f"{kind} note"

        return draw

    for kind in ("daily", "monthly", "yearly"):
        monkeypatch.setattr(handlers, f"temperature_{kind}_figure", figure(kind))
    FakeTemperature.drawn = drawn
    return FakeTemperature


# --- CLI -----------------------------------------------------------------------------------------


def test_temperature_command_prints_the_analysis_and_its_source(fake, records):
    fake.outcome = records["hot"]
    args = ["temperature", "Zhengzhou", "--event-date", "2003-07-10", "--config", str(BASE)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "[R1] Temperature extremes (TXx" in result.output
    assert "Event 2003-07-10: daily maximum 40.0 °C" in result.output
    assert "Request: https://archive.example/v1/archive?x=1" in result.output
    call = fake.calls[0]
    assert (call["metric"], call["scale"]) == ("hot", "daily")
    assert call["event_date"] == date(2003, 7, 10)


def test_temperature_command_reports_ambiguity_with_exit_code_1(fake):
    fake.outcome = AmbiguousLocationError("'Springfield' matches several places")
    result = runner.invoke(app, ["temperature", "Springfield", "--config", str(BASE)])
    assert result.exit_code == 1 and "matches several places" in result.output


# --- Temperature tab -----------------------------------------------------------------------------


def analyse(cfg, **overrides):
    args = {
        "city": "Zhengzhou",
        "country_code": "",
        "region": "",
        "start_year": None,
        "end_year": None,
        "event_date": "",
        "metric": "hot",
        "event_year": None,
        "plot_year": None,
        "window_days": 15.0,
        "view": "daily",
    }
    return handlers.temperature(cfg, **{**args, **overrides})


def test_tab_passes_options_and_returns_statistics_figure_and_series(
    fake, base_config, records
):
    fake.outcome = records["yearly"]
    status, text, figure, note, series = analyse(
        base_config, scale="yearly", event_year=2003.0, view="yearly", country_code=" cn "
    )
    assert status.startswith(handlers.NO_LLM) and "Zhengzhou" in status
    assert text.startswith("[R1] Annual mean temperature") and "Request: https://" in text
    assert (figure, note) == ("yearly figure", "yearly note")
    assert series == (records["yearly"], "ARCHIVE")
    call = fake.calls[0]
    assert (call["scale"], call["event_year"], call["country_code"]) == ("yearly", 2003, "cn")
    assert fake.drawn[0]["year"] == 2003


def test_tab_point_mode_and_errors(fake, base_config, records):
    fake.outcome = records["hot"]
    analyse(base_config, location_mode="point", latitude=48.85, longitude=2.35)
    assert fake.calls[-1]["point"] == (48.85, 2.35)
    status, *_ = analyse(base_config, location_mode="point", latitude=None, longitude=2.35)
    assert status.startswith("**Error:** click the map")
    fake.outcome = AmbiguousLocationError("several places")
    status, text, figure, _, series = analyse(base_config)
    assert status.startswith("**Error:**") and (text, figure, series) == ("", None, None)


def test_replot_redraws_a_temperature_series_with_the_temperature_figures(
    fake, base_config, records
):
    series = (records["hot"], "ARCHIVE")
    figure, note = handlers.replot(base_config, series, 2003.0, 15.0, "monthly")
    assert (figure, note) == ("monthly figure", "monthly note")
    assert fake.drawn[-1]["record"] is records["hot"] and fake.drawn[-1]["year"] == 2003


# --- Ask tab -------------------------------------------------------------------------------------


def test_ask_plots_a_cited_temperature_analysis_from_its_cached_series(
    monkeypatch, fake, base_config, records
):
    r1 = replace(records["hot"], id="R1")
    answer = Answer(
        claims=(Claim("Hottest day.", "evidence", ("R1",)),),
        references=(),
        limitations=("ERA5 is not a station.",),
        data_references=(r1,),
    )
    stats = RunStats(
        2, 0, 100, 50, 0, rainfall_calls=1, tool_trace=(("analyze_temperature[hot]",), ())
    )
    monkeypatch.setattr(handlers, "_run_agent", lambda *a, **k: AgentResult(answer, stats))
    status, body, figure, note, series = handlers.ask(base_config, CREDS, "How hot?")
    assert r"tools by turn: 1: temperature\[hot\] \| 2: answer" in status
    assert "**[R1]** Zhengzhou, Henan, China: TXx (hottest day per year) 2000-2004" in body
    assert fake.archive_calls == [("Zhengzhou", 2000, 2004)]
    assert (figure, series) == ("daily figure", (r1, "REFETCHED"))
    assert fake.drawn[0]["year"] == 2003  # the event's year


# --- figures -------------------------------------------------------------------------------------


def test_daily_figure_draws_the_metric_series_and_the_calendar_day_baseline(
    archive, records, cfg
):
    pytest.importorskip("plotly")
    from research_agent.ui.temperature_plots import temperature_daily_figure

    today = lambda: date(2026, 1, 1)  # noqa: E731
    fig, note = temperature_daily_figure(
        records["hot"], archive, year=2003, window_days=1, config=cfg, today=today
    )
    baseline, values = fig.data
    assert max(values.y) == 40.0 and min(values.y) == 25.0  # the daily maxima of 2003
    assert set(baseline.y) == {25.0}  # mean daily maximum of 2000-2002 per calendar day
    assert "daily maximum" in note and "not a station" in note
    assert fig.layout.shapes  # the event marker


def test_monthly_and_yearly_figures_use_the_mean_series_with_a_band(archive, records, cfg):
    pytest.importorskip("plotly")
    from research_agent.ui.temperature_plots import (
        temperature_monthly_figure,
        temperature_yearly_figure,
    )

    today = lambda: date(2026, 1, 1)  # noqa: E731
    fig, note = temperature_monthly_figure(
        records["yearly"], archive, year=2003, config=cfg, today=today
    )
    band, means = fig.data
    assert list(means.y) == [15.0] * 12 and list(band.base) == [15.0] * 12
    assert "percentile 10 to 90" in note
    fig, note = temperature_yearly_figure(
        records["yearly"], archive, year=2003, config=cfg, today=today
    )
    (line,) = fig.data
    assert list(line.x) == [2000, 2001, 2002, 2003, 2004] and set(line.y) == {15.0}
    assert "2003 is highlighted" in note and "15.0 °C" in note
