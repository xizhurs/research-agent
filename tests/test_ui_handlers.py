"""UI handlers with fake clients and a fake figure builder: no network, no Gradio, no plotly."""

import re
from dataclasses import replace
from datetime import date

import pytest

from research_agent.agent import AgentResult, Answer, Claim, RunStats
from research_agent.credentials import Credentials, Secret
from research_agent.llm import LLMError
from research_agent.rainfall import AmbiguousLocationError
from research_agent.retrieval import RetrievalError
from research_agent.ui import handlers
from research_agent.ui.plots import DEFAULT_WINDOW_DAYS, PlotError

CREDS = Credentials(openalex_api_key=None, openai_api_key=Secret("sk-test-0123456789abcdef"))
NO_OPENAI = Credentials(openalex_api_key=None, openai_api_key=None)


# --- escaping -------------------------------------------------------------------------------------


def test_md_escapes_markdown_and_html_and_collapses_whitespace():
    out = handlers.md("**bold** [x](http://e) <img src=x>\n$$m$$  `c`")
    assert not re.search(r"(?<!\\)[*\[\]()<>$`]", out)  # every markup character is escaped
    assert "\n" not in out and "  " not in out


# --- search ---------------------------------------------------------------------------------------


class FakeOpenAlex:
    calls: list = []
    keys: list = []
    outcome = None

    def __init__(self, cfg, limits, api_key=None):
        FakeOpenAlex.keys.append(api_key)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def search(self, query, **kwargs):
        FakeOpenAlex.calls.append({"query": query, **kwargs})
        if isinstance(FakeOpenAlex.outcome, Exception):
            raise FakeOpenAlex.outcome
        return FakeOpenAlex.outcome


@pytest.fixture
def openalex(monkeypatch):
    FakeOpenAlex.calls, FakeOpenAlex.keys = [], []
    monkeypatch.setattr(handlers, "OpenAlexClient", FakeOpenAlex)
    return FakeOpenAlex


def test_search_passes_the_options_and_says_no_llm(openalex, base_config, make_work, make_result):
    openalex.outcome = make_result([make_work(title="Crop mapping")], total=1234)
    status, body = handlers.search(base_config, CREDS, "crops", 2020.0, 5.0, False)
    assert openalex.calls == [{"query": "crops", "year_from": 2020, "max_results": 5}]
    assert status.startswith(handlers.NO_LLM)
    assert "1 of 1,234 matches (network, no API key)" in status
    assert "**[W1]** Crop mapping (2023) — Ann Author · *Journal* · cited by 5" in body
    assert "> " not in body  # abstracts only when asked


def test_search_shows_abstracts_when_asked_and_flags_missing_ones(
    openalex, base_config, make_work, make_result
):
    works = [make_work("W1", abstract="Text."), make_work("W2", abstract=None)]
    openalex.outcome = make_result(works)
    status, body = handlers.search(base_config, CREDS, "q", None, None, True)
    assert "> Text." in body and "> (no abstract)" in body
    assert "1 without an abstract" in status
    assert openalex.calls[0]["year_from"] is None and openalex.calls[0]["max_results"] is None


def test_search_never_renders_titles_or_links_from_the_source_as_markup(
    openalex, base_config, make_work, make_result
):
    title = "**x** [click](http://evil) <script>"
    work = replace(make_work(title=title), url="javascript:alert(1)")
    openalex.outcome = make_result([work])
    _, body = handlers.search(base_config, CREDS, "q", None, None, False)
    assert "**x**" not in body and "[click](" not in body and "<script>" not in body
    assert "<javascript" not in body  # a non-http link is shown as text, not made clickable


def test_search_links_http_urls(openalex, base_config, make_work, make_result):
    openalex.outcome = make_result([make_work()])
    _, body = handlers.search(base_config, CREDS, "q", None, None, False)
    assert "<https://openalex.org/W1>" in body


def test_search_uses_the_openalex_key_when_there_is_one(
    openalex, base_config, make_work, make_result
):
    openalex.outcome = make_result([make_work()])
    creds = Credentials(openalex_api_key=Secret("oa-key-0123456789"), openai_api_key=None)
    status, _ = handlers.search(base_config, creds, "q", None, None, False)
    assert openalex.keys == ["oa-key-0123456789"] and "network, API key" in status


def test_search_failure_is_shown_not_raised(openalex, base_config):
    openalex.outcome = RetrievalError("OpenAlex is down")
    status, body = handlers.search(base_config, CREDS, "q", None, None, False)
    assert status == "**Error:** OpenAlex is down" and body == ""


def test_search_rejects_a_fractional_year_before_any_request(openalex, base_config):
    status, _ = handlers.search(base_config, CREDS, "q", 2020.5, None, False)
    assert "year from must be a whole number" in status and openalex.calls == []


# --- rainfall -------------------------------------------------------------------------------------


class FakeRainfall:
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
        FakeRainfall.calls.append({"city": city, **kwargs})
        if isinstance(FakeRainfall.outcome, Exception):
            raise FakeRainfall.outcome
        return FakeRainfall.outcome, "ARCHIVE"

    def fetch_archive(self, location, start_year, end_year):
        FakeRainfall.archive_calls.append((location.name, start_year, end_year))
        return "REFETCHED"


class FakeFigure:
    """Stands in for ``daily_figure`` and records each call's arguments."""

    calls: list = []
    error = None

    def __call__(self, record, archive, **kwargs):
        FakeFigure.calls.append({"record": record, "archive": archive, **kwargs})
        if FakeFigure.error:
            raise FakeFigure.error
        return "FIGURE", "baseline note"


@pytest.fixture
def rain(monkeypatch):
    FakeRainfall.calls, FakeRainfall.archive_calls = [], []
    FakeRainfall.outcome, FakeFigure.calls, FakeFigure.error = None, [], None
    monkeypatch.setattr(handlers, "RainfallClient", FakeRainfall)
    monkeypatch.setattr(handlers, "daily_figure", FakeFigure())
    return FakeRainfall


def analyse(cfg, **overrides):
    args = {
        "city": "Zhengzhou",
        "country_code": "",
        "region": "",
        "start_year": None,
        "end_year": None,
        "event_date": "",
        "metric": "wet",
        "event_year": None,
        "plot_year": None,
        "window_days": 15.0,
    }
    return handlers.rainfall(cfg, **{**args, **overrides})


def test_rainfall_passes_options_and_returns_statistics_figure_and_note(
    rain, base_config, make_analysis
):
    rain.outcome = make_analysis(event_date=date(2002, 7, 1))
    status, text, figure, note = analyse(
        base_config,
        country_code=" CN ",
        region="Henan",
        start_year=2000.0,
        end_year=2004.0,
        event_date="2002-07-01",
        window_days=31.0,
    )
    assert rain.calls == [
        {
            "city": "Zhengzhou",
            "country_code": "CN",
            "region": "Henan",
            "start_year": 2000,
            "end_year": 2004,
            "event_date": date(2002, 7, 1),
            "metric": "wet",
            "event_year": None,
        }
    ]
    assert status.startswith(handlers.NO_LLM) and "Zhengzhou, Henan, China" in status
    assert "[R1] Extreme rainfall" in text and "Request: https://archive.example" in text
    assert (figure, note) == ("FIGURE", "baseline note")
    call = FakeFigure.calls[0]
    assert call["archive"] == "ARCHIVE"  # the series analyze returned, not a second lookup
    assert (call["year"], call["window_days"]) == (2002, 31)  # the event's year by default


def test_rainfall_blank_options_become_none_and_an_explicit_plot_year_wins(
    rain, base_config, make_analysis
):
    rain.outcome = make_analysis(event_date=date(2002, 7, 1))
    analyse(base_config, plot_year=2004.0, event_date="  ")
    assert rain.calls[0]["country_code"] is None and rain.calls[0]["event_date"] is None
    assert FakeFigure.calls[0]["year"] == 2004


def test_rainfall_dry_metric_and_event_year(rain, base_config, make_dry_analysis):
    rain.outcome = make_dry_analysis(event_year=2001)
    analyse(base_config, metric="dry", event_year=2001.0)
    assert rain.calls[0]["metric"] == "dry" and rain.calls[0]["event_year"] == 2001
    assert FakeFigure.calls[0]["year"] == 2001


@pytest.mark.parametrize(
    "overrides",
    [{"event_date": "last tuesday"}, {"start_year": 1990.5}, {"window_days": 7.5}],
)
def test_rainfall_bad_input_is_an_error_before_any_request(rain, base_config, overrides):
    status, text, figure, _ = analyse(base_config, **overrides)
    assert status.startswith("**Error:**") and (text, figure) == ("", None)
    assert rain.calls == []


def test_rainfall_ambiguous_city_is_an_error_with_no_plot(rain, base_config):
    rain.outcome = AmbiguousLocationError("'Springfield' matches 4 places. Do not guess.")
    status, text, figure, _ = analyse(base_config, city="Springfield")
    assert "matches 4 places" in status and (text, figure) == ("", None)
    assert FakeFigure.calls == []


def test_rainfall_plot_failure_keeps_the_statistics(rain, base_config, make_analysis):
    rain.outcome = make_analysis()
    FakeFigure.error = PlotError("the daily series has no data for 2030")
    status, text, figure, note = analyse(base_config, plot_year=2030.0)
    assert status.startswith(handlers.NO_LLM) and "[R1]" in text
    assert figure is None and note == "No plot: the daily series has no data for 2030"


# --- ask ------------------------------------------------------------------------------------------


def agent_result(answer, **stats):
    defaults = {"iterations": 2, "search_calls": 1, "input_tokens": 1200, "output_tokens": 300}
    return AgentResult(answer, RunStats(works_shown=1, **{**defaults, **stats}))


@pytest.fixture
def run_agent(monkeypatch):
    calls, outcome = [], {}

    def fake(cfg, creds, question):
        calls.append(question)
        if isinstance(outcome["value"], Exception):
            raise outcome["value"]
        return outcome["value"]

    monkeypatch.setattr(handlers, "_run_agent", fake)
    fake.calls, fake.outcome = calls, outcome
    return fake


def test_ask_says_the_llm_was_used_with_tokens_and_tools(run_agent, rain, base_config, make_work):
    answer = Answer(
        claims=(
            Claim("CNNs are common <b>.", "evidence", ("W1",)),
            Claim("This suggests more.", "interpretation", ()),
        ),
        references=(make_work("W1", title="Crop *mapping*"),),
        limitations=("OpenAlex only.",),
    )
    run_agent.outcome["value"] = agent_result(
        answer, cached_searches=1, tool_trace=(("search_literature",), ())
    )
    status, body, figure, note = handlers.ask(base_config, CREDS, "How are crops mapped?")
    assert run_agent.calls == ["How are crops mapped?"]
    assert status.startswith(f"**LLM used: yes** ({base_config.llm.model})")
    assert "2 model call(s), 1,500 tokens (1,200 in / 300 out)" in status
    assert "1 search(es) (1 from cache)" in status
    assert "tools by turn: 1: search \\| 2: answer" in status
    assert "- **evidence**: CNNs are common \\<b\\>. \\[W1\\]" in body  # escaped, then cited
    assert "- *interpretation*: This suggests more." in body
    assert "**[W1]** Ann Author (2023). Crop \\*mapping\\*." in body
    assert "- OpenAlex only." in body
    assert (figure, note) == (None, "")  # no rainfall analysis, nothing to plot


def test_ask_without_an_openai_key_does_not_call_the_agent(run_agent, base_config):
    status, body, figure, _ = handlers.ask(base_config, NO_OPENAI, "q")
    assert "LLM used: no" in status and "OPENAI\\_API\\_KEY is not set" in status
    assert run_agent.calls == [] and (body, figure) == ("", None)


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_rejects_an_empty_question(run_agent, base_config, question):
    status, *_ = handlers.ask(base_config, CREDS, question)
    assert "must not be empty" in status and run_agent.calls == []


def test_ask_reports_provider_failures_instead_of_raising(run_agent, base_config):
    run_agent.outcome["value"] = LLMError("provider is down")
    status, body, _, _ = handlers.ask(base_config, CREDS, "q")
    assert "provider is down" in status and body == ""


def test_ask_plots_the_first_rainfall_analysis_from_the_cached_series(
    run_agent, rain, base_config, make_analysis, make_dry_analysis
):
    r1 = replace(make_analysis(), id="R1")
    r2 = replace(make_dry_analysis(), id="R2")
    answer = Answer(
        claims=(Claim("Peak.", "evidence", ("R1",)),),
        references=(),
        limitations=("ERA5 is not a gauge.",),
        data_references=(r1, r2),
    )
    run_agent.outcome["value"] = agent_result(
        answer, rainfall_calls=2, cached_rainfall=1, search_calls=0
    )
    status, body, figure, note = handlers.ask(base_config, CREDS, "q")
    assert "2 rainfall/ENSO analysis(es) (1 from cache)" in status
    assert "**[R1]** Zhengzhou, Henan, China: Rx1day 2000-2004" in body
    assert rain.archive_calls == [("Zhengzhou", 2000, 2004)]  # no new analysis, no geocoding
    call = FakeFigure.calls[0]
    assert call["record"] is r1 and call["archive"] == "REFETCHED"
    assert (call["year"], call["window_days"]) == (2004, DEFAULT_WINDOW_DAYS)
    assert figure == "FIGURE"
    assert note.startswith("Plotted [R1]; 1 other analysis(es) not plotted.")


def test_ask_with_only_an_enso_analysis_has_nothing_to_plot(
    run_agent, rain, base_config, make_enso_analysis
):
    enso = replace(make_enso_analysis(), id="R1")
    answer = Answer(
        claims=(Claim("El Nino in 1997.", "evidence", ("R1",)),),
        references=(),
        limitations=("RONI.",),
        data_references=(enso,),
    )
    run_agent.outcome["value"] = agent_result(answer, rainfall_calls=1, search_calls=0)
    _, body, figure, note = handlers.ask(base_config, CREDS, "q")
    assert "**[R1]** ENSO events, NOAA CPC" in body
    assert (figure, note) == (None, "") and rain.archive_calls == []
