"""Command-line entry point."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import TextIO

import typer

from research_agent.agent import AgentResult, InvalidQuestionError, ResearchAgent
from research_agent.agent.events import TurnContext
from research_agent.agent.tools import format_enso, format_rainfall
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
from research_agent.evaluation import EvalError, load_cases, run_case, summarize
from research_agent.llm import LLMAuthError, LLMError, OpenAIClient
from research_agent.logging_setup import REDACTOR, configure_logging
from research_agent.rainfall import RainfallClient, RainfallError
from research_agent.retrieval import AuthError, OpenAlexClient, RetrievalError
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
    """Serve the web interface (Ask, Search, Rainfall). Needs the extra: uv sync --extra ui."""
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
def _open_agent(cfg: Config, creds: Credentials) -> Iterator[ResearchAgent]:
    """Build the real clients and yield an agent. Patched out in CLI tests."""
    key = creds.openalex_api_key
    llm = OpenAIClient.from_credentials(cfg.llm, creds)
    with (
        OpenAlexClient(cfg.openalex, cfg.limits, api_key=key.reveal() if key else None) as search,
        RainfallClient(cfg.rainfall) as rain,
        EnsoClient(cfg.enso) as enso_client,
    ):
        yield ResearchAgent(
            llm, search, cfg.limits, cfg.reporting, rainfall=rain, enso=enso_client
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
        f"{stats.rainfall_calls} rainfall/ENSO analysis(es) ({stats.cached_rainfall} from cache), "
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


@app.command("eval")
def evaluate(
    cases: Path = typer.Option(DEFAULT_CASES, "--cases", help="TOML file with evaluation cases."),
    only: list[str] = typer.Option(
        [], "--only", help="Run only this case id (repeatable). Default: all cases."
    ),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", envvar=ENV_FILE_ENV_VAR, help=ENV_FILE_HELP
    ),
) -> None:
    """Run the evaluation cases with the real model and OpenAlex (costs tokens and searches).

    Exits non-zero if any case fails.
    """
    cfg = _load_or_exit(config)
    creds = _credentials_or_exit(env_file)
    _tracing_or_exit(env_file)
    try:
        selected = load_cases(cases)
    except EvalError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if only:
        unknown = sorted(set(only) - {c.id for c in selected})
        if unknown:
            typer.echo(f"error: unknown case id(s): {', '.join(unknown)}", err=True)
            raise typer.Exit(code=1)
        selected = [c for c in selected if c.id in only]

    reports = []
    try:
        creds.require_openai_key()
        with _open_agent(cfg, creds) as agent:
            for case in selected:
                with observe("eval.case", "span", tags=("eval", case.id)) as span:
                    set_content(span, input=case.question)
                    report = run_case(agent, case, cfg.limits)
                    span.set_attribute("research_agent.eval.passed", report.passed)
                    if report.result:
                        set_content(span, output=answer_json(report.result.answer))
                reports.append(report)
                if report.result:
                    st = report.result.stats
                    detail = (
                        f"turns={st.iterations} searches={st.search_calls} "
                        f"rainfall={st.rainfall_calls} tokens={st.total_tokens}"
                        f" tools=[{format_trace(st.tool_trace)}]"
                        f" rejected_citations={report.rejected_citations}"
                    )
                else:
                    detail = "run failed"
                typer.echo(f"[{'PASS' if report.passed else 'FAIL'}] {case.id}  {detail}")
                for failure in report.failures:
                    typer.echo(f"    - {failure}")
    except (CredentialError, LLMError, RetrievalError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    total = summarize(reports)
    typer.echo(
        f"Summary: {total.passed}/{total.total} passed | {total.total_tokens} tokens | "
        f"{total.searches} searches | {total.rainfall_calls} rainfall analyses | "
        f"{total.rejected_citations} rejected citation(s) | "
        f"{total.unstructured} unstructured answer(s)"
    )
    if total.passed != total.total:
        raise typer.Exit(code=1)
