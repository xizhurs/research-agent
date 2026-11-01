"""Command-line entry point."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer

from research_agent.config import Config, ConfigError, load_config
from research_agent.retrieval import OpenAlexClient, RetrievalError, api_key_from_env

DEFAULT_CONFIG = Path("configs/base.toml")


def _load_or_exit(path: Path) -> Config:
    try:
        return load_config(path)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

app = typer.Typer(help="GeoAI / remote-sensing literature research agent.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """GeoAI / remote-sensing literature research agent."""


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
) -> None:
    """Search OpenAlex deterministically (no LLM) and print the results."""
    cfg = _load_or_exit(config)
    try:
        with OpenAlexClient(cfg.openalex, cfg.limits, api_key=api_key_from_env()) as client:
            result = client.search(query, year_from=year_from, max_results=max_results)
    except RetrievalError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    source = "cache" if result.from_cache else "network"
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
