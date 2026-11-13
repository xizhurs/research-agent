"""Command-line entry point."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path
from typing import TextIO

import httpx
import typer

from research_agent.agent import AgentResult, InvalidQuestionError, ResearchAgent
from research_agent.agent.events import TurnContext
from research_agent.agent.tools import format_enso, format_rainfall, format_temperature
from research_agent.config import Config, ConfigError, load_config
from research_agent.conversation import ConversationService, InMemorySessionStore
from research_agent.credentials import (
    DEFAULT_ENV_FILE,
    ENV_FILE_ENV_VAR,
    OPENAI_VAR,
    OPENALEX_VAR,
    CredentialError,
    Credentials,
    load_credentials,
    read_env,
)
from research_agent.enso import EnsoAnalysis, EnsoClient, EnsoError
from research_agent.evaluation import (
    EvalCase,
    EvalError,
    Qrels,
    RecordingLLMClient,
    RunBudget,
    ScoreSettings,
    aggregate,
    flags,
    load_cases,
    load_qrels,
    metrics_line,
    read_records,
    rescore,
    run_cases,
    score_records,
    select_cases,
    summarize_records,
    write_scores,
    write_summary,
)
from research_agent.evaluation.runner import fingerprint, git_commit
from research_agent.llm import LLMAuthError, LLMClient, LLMError, OpenAIClient
from research_agent.logging_setup import REDACTOR, configure_logging
from research_agent.rainfall import RainfallClient, RainfallError
from research_agent.retrieval import AuthError, OpenAlexClient, RetrievalError
from research_agent.temperature import TemperatureClient
from research_agent.tracing import (
    TracingError,
    answer_json,
    configure_tracing,
    observe,
    set_content,
)

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/base.toml")
ENV_FILE_HELP = f"Local env file with API keys (or set {ENV_FILE_ENV_VAR})."


def _load_or_exit(path: Path) -> Config:
    try:
        return load_config(path)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _credentials_or_exit(env_file: Path) -> Credentials:
    try:
        creds = load_credentials(env_file)
    except CredentialError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    REDACTOR.add_credentials(creds)  # from here on, keys never appear in logs
    return creds


def _tracing_or_exit(env_file: Path) -> None:
    """Turn on tracing if LANGFUSE_* or OTEL_* settings are present (see tracing.py)."""
    try:
        configure_tracing(read_env(env_file), os.environ)
    except (TracingError, CredentialError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


# Typer's default crash output prints local variables, which can include API keys. Keep that
# off, and let unexpected exceptions reach run() so they are reported without a data dump.
app = typer.Typer(
    help="GeoAI / remote-sensing literature research agent.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)


def make_output_safe(*streams: TextIO) -> None:
    """Never crash on characters the console cannot encode (paper titles contain many).

    Windows consoles often use a legacy code page; unencodable characters become '?'.
    """
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


@app.callback()
def main(
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Log progress to stderr: -v for steps, -vv for debug. Put it before the command.",
    ),
) -> None:
    """GeoAI / remote-sensing literature research agent."""
    make_output_safe(sys.stdout, sys.stderr)
    configure_logging(verbose)


def run() -> None:
    """Console-script entry point: run the app and report unexpected failures concisely.

    Expected failures (bad config, provider errors...) are handled inside the commands and exit
    with code 1. Anything else is a bug: show one redacted line, keep the traceback for -vv.
    """
    try:
        app()
    except Exception as exc:
        log.debug("unexpected failure", exc_info=True)
        message = REDACTOR.redact(f"{type(exc).__name__}: {exc}")
        typer.echo(f"error: unexpected failure ({message}). Use -vv for a traceback.", err=True)
        sys.exit(1)


@app.command("show-config")
def show_config(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
) -> None:
    """Validate the config file and print the resolved resource limits."""
    cfg = _load_or_exit(config)
    for name, value in asdict(cfg.limits).items():
        typer.echo(f"{name} = {value}")


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search text (matched against title and abstract)."),
    year_from: int | None = typer.Option(None, "--year-from", help="Earliest publication year."),
    max_results: int | None = typer.Option(
        None, "--max-results", "-n", help="Number of results (capped by limits.results_per_search)."
    ),
    show_abstracts: bool = typer.Option(False, "--abstracts", help="Print each abstract."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", envvar=ENV_FILE_ENV_VAR, help=ENV_FILE_HELP
    ),
) -> None:
    """Search OpenAlex deterministically (no LLM) and print the results."""
    cfg = _load_or_exit(config)
    key = _credentials_or_exit(env_file).openalex_api_key
    try:
        with OpenAlexClient(
            cfg.openalex, cfg.limits, api_key=key.reveal() if key else None
        ) as client:
            result = client.search(query, year_from=year_from, max_results=max_results)
    except RetrievalError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    network = "network, API key" if key else "network, no API key"
    source = "cache" if result.from_cache else network
    typer.echo(f"{len(result.works)} of {result.total_count} matches ({source})")
    for w in result.works:
        authors = ", ".join(w.authors[:3]) + (" et al." if len(w.authors) > 3 else "")
        typer.echo(f"[{w.id}] {w.title or '(no title)'} ({w.year or 'n.d.'}) - {authors or 'n/a'}")
        if show_abstracts:
            typer.echo(f"    {w.abstract or '(no abstract)'}")
    if result.missing_abstract_count:
        typer.echo(f"note: {result.missing_abstract_count} result(s) have no abstract")
    if result.skipped_records:
        typer.echo(f"note: {result.skipped_records} record(s) could not be parsed", err=True)


@app.command("rainfall")
def rainfall(
    city: str = typer.Argument(..., help="City name only, e.g. Zhengzhou."),
    country_code: str | None = typer.Option(None, "--country-code", help="ISO alpha-2, e.g. CN."),
    region: str | None = typer.Option(None, "--region", help="State or province, e.g. Henan."),
    start_year: int | None = typer.Option(None, "--start-year", help="First year to analyse."),
    end_year: int | None = typer.Option(None, "--end-year", help="Last year to analyse."),
    event_date: str | None = typer.Option(
        None, "--event-date", help="A day to assess, as YYYY-MM-DD."
    ),
    scale: str = typer.Option(
        "daily", "--scale", help="'daily' (see --metric), 'monthly' or 'yearly' totals."
    ),
    metric: str = typer.Option(
        "wet", "--metric", help="Daily scale: 'wet' (Rx1day) or 'dry' (CDD, totals)."
    ),
    event_year: int | None = typer.Option(
        None, "--event-year", help="A year to assess (--scale yearly/monthly, or --metric dry)."
    ),
    event_month: int | None = typer.Option(
        None, "--event-month", help="With --scale monthly and --event-year: a month, 1-12."
    ),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
) -> None:
    """Analyse rainfall (ERA5) for a city at a daily, monthly or yearly scale (no LLM)."""
    cfg = _load_or_exit(config)
    try:
        day = date.fromisoformat(event_date) if event_date else None
    except ValueError:
        typer.echo("error: --event-date must be formatted YYYY-MM-DD", err=True)
        raise typer.Exit(code=1) from None
    try:
        with RainfallClient(cfg.rainfall) as client:
            analysis, _daily = client.analyze(
                city,
                country_code=country_code,
                region=region,
                start_year=start_year,
                end_year=end_year,
                event_date=day,
                metric=metric,
                event_year=event_year,
                scale=scale,
                event_month=event_month,
            )
    except RainfallError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    prov = analysis.provenance
    typer.echo(format_rainfall(replace(analysis, id="R1")))
    typer.echo(f"\nSource: {prov.dataset}. {prov.license}.")
    typer.echo(f"Request: {prov.request_url}")
    origin = "cache" if prov.from_cache else "network"
    typer.echo(f"({origin}; data {prov.data_start} to {prov.data_end})")


@app.command("temperature")
def temperature(
    city: str = typer.Argument(..., help="City name only, e.g. Madrid."),
    country_code: str | None = typer.Option(None, "--country-code", help="ISO alpha-2, e.g. ES."),
    region: str | None = typer.Option(None, "--region", help="State or province."),
    start_year: int | None = typer.Option(None, "--start-year", help="First year to analyse."),
    end_year: int | None = typer.Option(None, "--end-year", help="Last year to analyse."),
    event_date: str | None = typer.Option(
        None, "--event-date", help="A day to assess, as YYYY-MM-DD (--scale daily)."
    ),
    scale: str = typer.Option(
        "daily", "--scale", help="'daily' (see --metric), 'monthly' or 'yearly' means."
    ),
    metric: str = typer.Option(
        "hot", "--metric", help="Daily scale: 'hot' (TXx) or 'cold' (TNn)."
    ),
    event_year: int | None = typer.Option(
        None, "--event-year", help="A year to assess (--scale yearly or monthly)."
    ),
    event_month: int | None = typer.Option(
        None, "--event-month", help="With --scale monthly and --event-year: a month, 1-12."
    ),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
) -> None:
    """Analyse air temperature (ERA5) for a city at a daily, monthly or yearly scale (no LLM)."""
    cfg = _load_or_exit(config)
    try:
        day = date.fromisoformat(event_date) if event_date else None
    except ValueError:
        typer.echo("error: --event-date must be formatted YYYY-MM-DD", err=True)
        raise typer.Exit(code=1) from None
    try:
        with TemperatureClient(cfg.rainfall) as client:
            analysis, _daily = client.analyze(
                city,
                country_code=country_code,
                region=region,
                start_year=start_year,
                end_year=end_year,
                event_date=day,
                metric=metric,
                event_year=event_year,
                scale=scale,
                event_month=event_month,
            )
    except RainfallError as exc:  # the Open-Meteo error family, shared with rainfall
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    prov = analysis.provenance
    typer.echo(format_temperature(replace(analysis, id="R1")))
    typer.echo(f"\nSource: {prov.dataset}. {prov.license}.")
    typer.echo(f"Request: {prov.request_url}")
    origin = "cache" if prov.from_cache else "network"
    typer.echo(f"({origin}; data {prov.data_start} to {prov.data_end})")


@app.command("enso")
def enso(
    phase: str = typer.Option("both", "--phase", help="'both', 'el_nino' or 'la_nina'."),
    start_year: int | None = typer.Option(None, "--start-year", help="Events that begin in/after."),
    end_year: int | None = typer.Option(None, "--end-year", help="Events that begin in/before."),
    year: int | None = typer.Option(None, "--year", help="Describe this calendar year."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
) -> None:
    """List El Nino / La Nina events from NOAA CPC's RONI deterministically (no LLM)."""
    cfg = _load_or_exit(config)
    try:
        with EnsoClient(cfg.enso) as client:
            analysis = client.analyze(
                phase=phase, start_year=start_year, end_year=end_year, year=year
            )
    except EnsoError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_enso(replace(analysis, id="R1")))
    origin = "cache" if analysis.from_cache else "network"
    typer.echo(f"\n({origin}; record {analysis.record_first} to {analysis.record_last})")


@app.command("ui")
def ui(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Interface to listen on. Whoever can reach it can spend tokens."
    ),
    port: int = typer.Option(7860, "--port", help="Port to listen on."),
    share: bool = typer.Option(
        False,
        "--share",
        help="Also open a temporary public *.gradio.live link to this machine. Needs "
        "UI_USERNAME and UI_PASSWORD.",
    ),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", envvar=ENV_FILE_ENV_VAR, help=ENV_FILE_HELP
    ),
) -> None:
    """Serve the web interface (Ask, Search, Rainfall, Temperature). Needs: uv sync --extra ui."""
    cfg = _load_or_exit(config)
    creds = _credentials_or_exit(env_file)
    _tracing_or_exit(env_file)
    try:
        from research_agent.ui.app import build_app
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in {"gradio", "plotly"}:
            raise
        typer.echo("error: the web interface needs the extra: uv sync --extra ui", err=True)
        raise typer.Exit(code=1) from exc
    # Optional login, from the environment or the .env file (never a flag, so it stays out of
    # shell history).
    env = read_env(env_file)
    username, password = env.get("UI_USERNAME"), env.get("UI_PASSWORD")
    if bool(username) != bool(password):
        typer.echo("error: set both UI_USERNAME and UI_PASSWORD, or neither.", err=True)
        raise typer.Exit(code=1)
    if share and not username:
        typer.echo(
            "error: --share makes the app public; set UI_USERNAME and UI_PASSWORD first.", err=True
        )
        raise typer.Exit(code=1)
    launch_args: dict = {"server_name": host, "server_port": port}
    if share:
        launch_args["share"] = True
    if username and password:
        launch_args["auth"] = (username, password)
    elif host not in {"127.0.0.1", "localhost", "::1"}:
        typer.echo(
            "warning: no login set (UI_USERNAME/UI_PASSWORD) and the server is not local: "
            "anyone who can reach it can spend the OpenAI budget.",
            err=True,
        )
    build_app(cfg, creds).launch(**launch_args)


@app.command("check-credentials")
def check_credentials(
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", envvar=ENV_FILE_ENV_VAR, help=ENV_FILE_HELP
    ),
    require_openai: bool = typer.Option(
        False, "--require-openai", help="Exit non-zero if OPENAI_API_KEY is not set."
    ),
) -> None:
    """Check that API keys are present and well-formed. Never prints key values.

    This is an offline check: it cannot tell whether a provider accepts the key.
    """
    creds = _credentials_or_exit(env_file)
    where = "found" if env_file.is_file() else "not found, using process environment only"
    typer.echo(f"env file: {env_file} ({where})")
    typer.echo(
        f"{OPENALEX_VAR}: "
        + ("set" if creds.openalex_api_key else "not set (optional; small anonymous budget)")
    )
    typer.echo(f"{OPENAI_VAR}: " + ("set" if creds.openai_api_key else "not set (needed for ask)"))
    if require_openai and creds.openai_api_key is None:
        typer.echo(f"error: {OPENAI_VAR} is required but not set", err=True)
        raise typer.Exit(code=1)


@contextmanager
def _open_agent(
    cfg: Config,
    creds: Credentials,
    *,
    wrap_llm: Callable[[LLMClient], LLMClient] | None = None,
    http_client: httpx.Client | None = None,
) -> Iterator[ResearchAgent]:
    """Build the real clients and yield an agent. Patched out in CLI tests.

    ``wrap_llm`` wraps the model client (evaluation recording and budget); ``http_client`` is
    used by every tool client (evaluation with a frozen tool cache: no network).
    """
    key = creds.openalex_api_key
    llm: LLMClient = OpenAIClient.from_credentials(cfg.llm, creds)
    if wrap_llm is not None:
        llm = wrap_llm(llm)
    api_key = key.reveal() if key else None
    with (
        OpenAlexClient(
            cfg.openalex, cfg.limits, api_key=api_key, http_client=http_client
        ) as search,
        RainfallClient(cfg.rainfall, http_client=http_client) as rain,
        TemperatureClient(cfg.rainfall, http_client=http_client) as temp,
        EnsoClient(cfg.enso, http_client=http_client) as enso_client,
    ):
        yield ResearchAgent(
            llm,
            search,
            cfg.limits,
            cfg.reporting,
            rainfall=rain,
            enso=enso_client,
            temperature=temp,
        )


def _run_agent(
    cfg: Config,
    creds: Credentials,
    question: str,
    context: TurnContext | None = None,
    map_point: tuple[float, float] | None = None,
) -> AgentResult:
    """Answer one question with the real clients. Patched out in CLI tests."""
    with _open_agent(cfg, creds) as agent:
        if map_point is not None:
            return agent.run(question, context, map_point=map_point)
        return agent.run(question, context) if context is not None else agent.run(question)


@app.command("ask")
def ask(
    question: str = typer.Argument(
        ..., help="A GeoAI / remote-sensing literature or extreme-rainfall question."
    ),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", envvar=ENV_FILE_ENV_VAR, help=ENV_FILE_HELP
    ),
) -> None:
    """Answer from OpenAlex literature and/or ERA5 rainfall analysis, within the limits."""
    if not question.strip():
        typer.echo("error: question must not be empty", err=True)
        raise typer.Exit(code=1)
    cfg = _load_or_exit(config)
    creds = _credentials_or_exit(env_file)
    _tracing_or_exit(env_file)
    try:
        creds.require_openai_key()
        with observe("ask", "span", tags=("cli",)) as span:
            set_content(span, input=question)
            result = _run_agent(cfg, creds, question)
            set_content(span, output=answer_json(result.answer))
    except (CredentialError, LLMError, RetrievalError, InvalidQuestionError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_result(result)


def _print_result(result: AgentResult) -> None:
    """The answer on stdout; the tool trace and usage footer on stderr."""
    answer = result.answer
    typer.echo("Answer")
    for claim in answer.claims:
        cites = f" [{', '.join(claim.sources)}]" if claim.sources else ""
        typer.echo(f"- ({claim.kind}) {claim.text}{cites}")
    if answer.references:
        typer.echo("\nReferences")
        for w in answer.references:
            authors = ", ".join(w.authors[:3]) + (" et al." if len(w.authors) > 3 else "")
            title = w.title or "(no title)"
            typer.echo(f"[{w.id}] {authors or 'n/a'} ({w.year or 'n.d.'}). {title}.")
            typer.echo(f"    {w.venue + '. ' if w.venue else ''}{w.doi or w.url}")
    if answer.data_references:
        typer.echo("\nData")
        for a in answer.data_references:
            if isinstance(a, EnsoAnalysis):
                typer.echo(
                    f"[{a.id}] ENSO events, NOAA CPC {a.index_name}, record {a.record_first} to "
                    f"{a.record_last}; events starting {a.first_year}-{a.last_year}."
                )
                typer.echo(f"    {a.source_url}")
                continue
            p = a.provenance
            typer.echo(
                f"[{a.id}] {a.location.label}: {a.summary} {a.valid_years[0]}-"
                f"{a.valid_years[-1]}. {p.dataset}; grid cell "
                f"({p.grid_latitude:.2f}, {p.grid_longitude:.2f})."
            )
            typer.echo(f"    {p.license}. {p.request_url}")
    typer.echo("\nLimitations")
    for limitation in answer.limitations or ("(none reported)",):
        typer.echo(f"- {limitation}")
    stats = result.stats
    rain = (
        f"{stats.rainfall_calls} rainfall/temperature/ENSO analysis(es) "
        f"({stats.cached_rainfall} from cache), "
        if stats.rainfall_calls
        else ""
    )
    if stats.tool_trace:
        typer.echo(f"\ntools by turn: {format_trace(stats.tool_trace)}", err=True)
    typer.echo(
        f"\nrun: {stats.iterations} model turn(s), {stats.search_calls} search(es) "
        f"({stats.cached_searches} from cache), {rain}{stats.total_tokens} tokens "
        f"({stats.input_tokens} in / {stats.output_tokens} out)",
        err=True,
    )


def format_trace(trace: tuple[tuple[str, ...], ...]) -> str:
    """One line: the tools the model requested on each turn, e.g. ``1: rainfall | 2: answer``."""
    short = {
        "search_literature": "search",
        "analyze_rainfall_extremes": "rainfall",
        "analyze_temperature": "temperature",
        "analyze_enso_events": "enso",
    }

    def shorten(name: str) -> str:
        base, bracket, rest = name.partition("[")
        return short.get(base, base) + bracket + rest

    turns = [", ".join(shorten(n) for n in names) or "answer" for names in trace]
    return " | ".join(f"{i}: {turn}" for i, turn in enumerate(turns, start=1))


CHAT_EXIT = {"", "exit", "quit"}


@app.command("chat")
def chat(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", envvar=ENV_FILE_ENV_VAR, help=ENV_FILE_HELP
    ),
) -> None:
    """Ask questions with follow-ups ("what about Rotterdam?"). Empty line or 'exit' ends it.

    Each turn has the same budgets and citation checks as `ask`; earlier turns are sent as a
    capped summary (see [conversation] in the config) and kept in memory only.
    """
    cfg = _load_or_exit(config)
    creds = _credentials_or_exit(env_file)
    _tracing_or_exit(env_file)
    try:
        creds.require_openai_key()
    except CredentialError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    conv = cfg.conversation
    store = InMemorySessionStore(conv.max_sessions, conv.session_ttl_minutes * 60)
    session_id: str | None = None
    typer.echo("Ask a question; follow-ups keep the context. Empty line or 'exit' ends.", err=True)
    with _open_agent(cfg, creds) as agent:
        service = ConversationService(lambda q, ctx: agent.run(q, ctx), store, conv)
        while True:
            try:
                question = input("you> ")
            except EOFError:
                break
            if question.strip().lower() in CHAT_EXIT:
                break
            try:
                outcome = service.ask(session_id, question)
            except (LLMAuthError, AuthError) as exc:
                typer.echo(f"error: {exc}", err=True)
                raise typer.Exit(code=1) from exc
            except (LLMError, RetrievalError, InvalidQuestionError) as exc:
                typer.echo(f"error: {exc} (the conversation continues)", err=True)
                continue
            session_id = outcome.session_id
            _print_result(outcome.result)
            turn = outcome.turn
            sent = outcome.context.tokens if outcome.context else 0
            typer.echo(
                f"turn {turn.number} ({turn.relation}): ~{sent} context tokens from earlier turns",
                err=True,
            )


DEFAULT_CASES = Path("evals/cases.toml")
DEFAULT_QRELS = Path("evals/qrels.toml")
FROZEN_TTL_DAYS = 36500.0  # the evaluation tool cache never expires; delete it to refresh

eval_app = typer.Typer(
    help="Evaluation: list cases, run them (paid, only with --live), score stored runs (free).",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)
app.add_typer(eval_app, name="eval")

CASES_OPTION = typer.Option(DEFAULT_CASES, "--cases", help="TOML file with evaluation cases.")
ONLY_OPTION = typer.Option([], "--only", help="Only this case id (repeatable; any split).")
SPLIT_OPTION = typer.Option("dev", "--split", help="'dev', 'test' (held out) or 'all'.")
CATEGORY_OPTION = typer.Option(None, "--category", help="e.g. literature, rainfall, enso.")
QRELS_OPTION = typer.Option(
    DEFAULT_QRELS, "--qrels", help="Relevance labels per case (a missing file means none)."
)


def _qrels_or_exit(path: Path, cases: list[EvalCase]) -> dict[str, Qrels]:
    try:
        qrels = load_qrels(path)
    except EvalError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    unknown = sorted(qrels.keys() - {c.id for c in cases})
    if unknown:
        typer.echo(f"warning: {path} labels unknown case(s): {', '.join(unknown)}", err=True)
    return qrels


def _score_and_report(
    cfg: Config,
    run_dir: Path,
    records: list[dict],
    cases: list[EvalCase],
    qrels: dict[str, Qrels],
    budget: RunBudget,
) -> tuple[list[dict], dict]:
    """Metrics of the records (reported, never failures), written to score.json and report.md."""
    settings = ScoreSettings(cfg.eval.redundant_query_jaccard, cfg.eval.low_support_overlap)
    scores = score_records(records, cases, qrels, cfg.limits, settings, budget)
    agg = aggregate(scores)
    write_scores(run_dir, scores, agg)
    return scores, agg


def _cases_or_exit(
    path: Path, only: list[str], split: str, category: str | None
) -> list[EvalCase]:
    if split not in ("dev", "test", "all"):
        typer.echo("error: --split must be 'dev', 'test' or 'all'", err=True)
        raise typer.Exit(code=1)
    try:
        return select_cases(
            load_cases(path),
            only=only,
            split=None if split == "all" else split,
            category=category,
        )
    except EvalError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _budget(cfg: Config, calls: int | None, tokens: int | None, cost: float | None) -> RunBudget:
    e = cfg.eval
    return RunBudget(
        max_calls=calls or e.max_model_calls,
        max_tokens=tokens or e.max_total_tokens,
        max_cost_usd=e.max_cost_usd if cost is None else cost,
        price_input_per_mtok=e.price_input_per_mtok,
        price_output_per_mtok=e.price_output_per_mtok,
    )


def _eval_tool_config(cfg: Config, frozen: bool) -> Config:
    """Tool caches in the evaluation folder that never expire; ``frozen``: no retries either
    (no network is used, so a miss fails at once)."""
    root = cfg.eval.tool_cache_dir
    retries = {"max_retries": 0} if frozen else {}
    return replace(
        cfg,
        openalex=replace(
            cfg.openalex, cache_dir=root / "openalex", cache_ttl_days=FROZEN_TTL_DAYS, **retries
        ),
        rainfall=replace(
            cfg.rainfall, cache_dir=root / "open-meteo", cache_ttl_days=FROZEN_TTL_DAYS, **retries
        ),
        enso=replace(cfg.enso, cache_dir=root / "enso", cache_ttl_days=FROZEN_TTL_DAYS, **retries),
    )


def _refuse_network(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError(
        "frozen tool cache: this response is not cached and the network is off", request=request
    )


def _record_line(record: dict) -> str:
    status = record["status"]
    label = {"ok": "PASS" if not record["failures"] else "FAIL"}.get(status, status.upper())
    if record["stats"]:
        st = record["stats"]
        trace = format_trace(tuple(tuple(t) for t in st["tool_trace"]))
        detail = (
            f"turns={st['iterations']} searches={st['search_calls']} "
            f"analyses={st['rainfall_calls']} tokens={st['total_tokens']} "
            f"time={record['wall_seconds']}s tools=[{trace}] "
            f"rejected_citations={len(record['answer']['rejected_citations'])}"
        )
    else:
        detail = record["error"] or ""
    lines = [f"[{label}] {record['case_id']}  {detail}"]
    lines += [f"    - {f}" for f in record["failures"]]
    return "\n".join(lines)


def _summary_line(summary: dict) -> str:
    cost = summary["estimated_cost_usd"]
    money = f"~${cost:.4f}" if cost is not None else "cost unknown (no prices in [eval])"
    not_run = summary["run_failed"] + summary["aborted"] + summary["skipped"]
    return (
        f"Summary: {summary['passed']}/{summary['cases']} passed | "
        f"{summary['failed_checks']} failed checks, {not_run} not completed "
        f"({summary['aborted']} aborted, {summary['skipped']} skipped by the budget) | "
        f"{summary['model_calls']} model calls, "
        f"{summary['input_tokens'] + summary['output_tokens']} tokens, {money} | "
        f"{summary['searches']} searches | {summary['data_analyses']} data analyses | "
        f"{summary['rejected_citations']} rejected citation(s) | "
        f"{summary['unstructured']} unstructured answer(s)"
    )


@eval_app.command("list")
def eval_list(
    cases: Path = CASES_OPTION,
    only: list[str] = ONLY_OPTION,
    split: str = typer.Option("all", "--split", help="'dev', 'test' (held out) or 'all'."),
    category: str | None = CATEGORY_OPTION,
) -> None:
    """List evaluation cases (free: nothing is run)."""
    selected = _cases_or_exit(cases, only, split, category)
    for c in selected:
        typer.echo(f"{c.id:40} {c.split:4} {c.category or '-':11} {c.question[:70]}")
    typer.echo(f"{len(selected)} case(s)")


@eval_app.command("run")
def eval_run(
    cases: Path = CASES_OPTION,
    only: list[str] = ONLY_OPTION,
    split: str = SPLIT_OPTION,
    category: str | None = CATEGORY_OPTION,
    live: bool = typer.Option(
        False, "--live", help="Really run: calls the model (costs tokens). Without it: plan only."
    ),
    yes: bool = typer.Option(False, "--yes", help="Do not ask for confirmation (scripts)."),
    frozen_tools: bool = typer.Option(
        False, "--frozen-tools", help="Tools answer only from the evaluation cache (no network)."
    ),
    max_calls: int | None = typer.Option(None, "--max-calls", min=1, help="Model-call cap."),
    max_tokens: int | None = typer.Option(None, "--max-tokens", min=1, help="Token cap."),
    max_cost: float | None = typer.Option(None, "--max-cost", min=0.0, help="USD cap."),
    out: Path | None = typer.Option(None, "--out", help="Run folder (default: [eval] runs_dir)."),
    qrels_file: Path = QRELS_OPTION,
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", envvar=ENV_FILE_ENV_VAR, help=ENV_FILE_HELP
    ),
) -> None:
    """Run cases with the real model within a run budget; one record per case is written,
    then score.json and report.md (metrics, reported only).

    Without --live only the plan is printed (cases, worst-case tokens, budget): nothing is
    called. Exits non-zero if any case fails, is aborted or is skipped.
    """
    cfg = _load_or_exit(config)
    selected = _cases_or_exit(cases, only, split, category)
    if not selected:
        typer.echo("error: no case matches the selection", err=True)
        raise typer.Exit(code=1)
    qrels = _qrels_or_exit(qrels_file, _cases_or_exit(cases, [], "all", None))
    budget = _budget(cfg, max_calls, max_tokens, max_cost)
    if budget.max_cost_usd > 0 and not budget.priced:
        typer.echo("error: a cost cap needs prices in [eval] (price_*_per_mtok)", err=True)
        raise typer.Exit(code=1)

    worst = len(selected) * cfg.limits.max_total_tokens
    typer.echo(f"Plan: {len(selected)} case(s): {', '.join(c.id for c in selected)}")
    typer.echo(
        f"Worst case {worst:,} tokens ({len(selected)} x limits.max_total_tokens "
        f"{cfg.limits.max_total_tokens:,}); run budget {budget.max_calls} model calls, "
        f"{budget.max_tokens:,} tokens"
        + (f", ${budget.max_cost_usd:.2f}" if budget.max_cost_usd else "")
        + "."
    )
    if budget.priced:
        typer.echo(f"Worst-case cost at the configured prices: ~${budget.cost(worst, 0):.2f}.")
    typer.echo(f"Tools: {'frozen cache, no network' if frozen_tools else 'evaluation cache'}.")
    if not live:
        typer.echo("Plan only: nothing was run. Add --live to run it (calls the model).")
        return

    creds = _credentials_or_exit(env_file)
    _tracing_or_exit(env_file)
    try:
        creds.require_openai_key()
    except CredentialError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not yes and not typer.confirm("Run these cases with the real model?"):
        typer.echo("cancelled: nothing was run", err=True)
        raise typer.Exit(code=1)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = out or cfg.eval.runs_dir / run_id
    meta = {
        "mode": "agentic",
        "git_commit": git_commit(),
        "model": cfg.llm.model,
        "config_hash": fingerprint(asdict(cfg)),
    }
    recorder = RecordingLLMClient(None, budget)

    def wrap(llm: LLMClient) -> LLMClient:
        recorder.inner = llm
        return recorder

    http = httpx.Client(transport=httpx.MockTransport(_refuse_network)) if frozen_tools else None
    try:
        with _open_agent(
            _eval_tool_config(cfg, frozen_tools), creds, wrap_llm=wrap, http_client=http
        ) as agent:

            def run_one(case: EvalCase) -> AgentResult:
                with observe("eval.case", "span", tags=("eval", case.id)) as span:
                    set_content(span, input=case.question)
                    result = agent.run(case.question)
                    set_content(span, output=answer_json(result.answer))
                    return result

            records = run_cases(
                selected,
                run_one,
                recorder,
                cfg.limits,
                out_dir=out_dir,
                run_id=run_id,
                meta=meta,
                on_record=lambda record: typer.echo(_record_line(record)),
            )
    except (CredentialError, LLMError, RetrievalError) as exc:
        typer.echo(f"error: {exc} (records so far are in {out_dir})", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if http is not None:
            http.close()

    summary = summarize_records(records, budget)
    write_summary(out_dir, {"run_id": run_id, **meta, **summary})
    typer.echo(_summary_line(summary))
    _, agg = _score_and_report(cfg, out_dir, records, selected, qrels, budget)
    typer.echo(metrics_line(agg))
    typer.echo(f"Records and report.md: {out_dir}")
    if summary["passed"] != summary["cases"]:
        raise typer.Exit(code=1)


@eval_app.command("score")
def eval_score(
    run_dir: Path = typer.Argument(..., help="A run folder written by eval run."),
    cases: Path = CASES_OPTION,
    qrels_file: Path = QRELS_OPTION,
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
) -> None:
    """Score stored run records again with the current cases and checks (free, offline).

    Also writes score.json and report.md into RUN_DIR (metrics: tools, trajectory, numbers,
    citation support, key points, retrieval against --qrels, efficiency). Metrics are reported
    only; the exit code depends on the case checks.
    """
    cfg = _load_or_exit(config)
    try:
        records = read_records(run_dir)
        case_list = load_cases(cases)
    except (FileNotFoundError, EvalError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not records:
        typer.echo(f"error: no run records in {run_dir}", err=True)
        raise typer.Exit(code=1)
    qrels = _qrels_or_exit(qrels_file, case_list)
    budget = _budget(cfg, None, None, None)
    rescored = rescore(records, case_list, cfg.limits)
    scores, agg = _score_and_report(cfg, run_dir, rescored, case_list, qrels, budget)
    for record, score in zip(rescored, scores, strict=True):
        typer.echo(_record_line(record))
        flagged = flags(score)
        if flagged:
            typer.echo(f"    ~ {flagged}")
    summary = summarize_records(rescored, budget)
    typer.echo(_summary_line(summary))
    typer.echo(metrics_line(agg))
    typer.echo(f"Report: {run_dir / 'report.md'}")
    if summary["passed"] != summary["cases"]:
        raise typer.Exit(code=1)
