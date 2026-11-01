"""Command-line entry point."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import typer

from research_agent.agent import AgentResult, ResearchAgent
from research_agent.config import Config, ConfigError, load_config
from research_agent.credentials import (
    DEFAULT_ENV_FILE,
    ENV_FILE_ENV_VAR,
    OPENAI_VAR,
    OPENALEX_VAR,
    CredentialError,
    Credentials,
    load_credentials,
)
from research_agent.llm import LLMError, OpenAIClient
from research_agent.retrieval import OpenAlexClient, RetrievalError

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
        return load_credentials(env_file)
    except CredentialError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


app = typer.Typer(help="GeoAI / remote-sensing literature research agent.", no_args_is_help=True)


def make_output_safe(*streams: TextIO) -> None:
    """Never crash on characters the console cannot encode (paper titles contain many).

    Windows consoles often use a legacy code page; unencodable characters become '?'.
    """
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


@app.callback()
def main() -> None:
    """GeoAI / remote-sensing literature research agent."""
    make_output_safe(sys.stdout, sys.stderr)


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
    typer.echo(
        f"{OPENAI_VAR}: " + ("set" if creds.openai_api_key else "not set (needed for ask)")
    )
    if require_openai and creds.openai_api_key is None:
        typer.echo(f"error: {OPENAI_VAR} is required but not set", err=True)
        raise typer.Exit(code=1)


def _run_agent(cfg: Config, creds: Credentials, question: str) -> AgentResult:
    """Build the real clients and answer one question. Patched out in CLI tests."""
    key = creds.openalex_api_key
    llm = OpenAIClient.from_credentials(cfg.llm, creds)
    with OpenAlexClient(
        cfg.openalex, cfg.limits, api_key=key.reveal() if key else None
    ) as search:
        return ResearchAgent(llm, search, cfg.limits).run(question)


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="A GeoAI / remote-sensing literature question."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
    env_file: Path = typer.Option(
        DEFAULT_ENV_FILE, "--env-file", envvar=ENV_FILE_ENV_VAR, help=ENV_FILE_HELP
    ),
) -> None:
    """Answer a question from OpenAlex literature, within the configured resource limits."""
    if not question.strip():
        typer.echo("error: question must not be empty", err=True)
        raise typer.Exit(code=1)
    cfg = _load_or_exit(config)
    creds = _credentials_or_exit(env_file)
    try:
        creds.require_openai_key()
        result = _run_agent(cfg, creds, question)
    except (CredentialError, LLMError, RetrievalError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

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
    typer.echo("\nLimitations")
    for limitation in answer.limitations or ("(none reported)",):
        typer.echo(f"- {limitation}")
    stats = result.stats
    typer.echo(
        f"\nrun: {stats.iterations} model turn(s), {stats.search_calls} search(es), "
        f"{stats.total_tokens} tokens ({stats.input_tokens} in / {stats.output_tokens} out)",
        err=True,
    )
