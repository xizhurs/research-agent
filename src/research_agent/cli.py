"""Command-line entry point."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer

from research_agent.config import ConfigError, load_config

DEFAULT_CONFIG = Path("configs/base.toml")

app = typer.Typer(help="GeoAI / remote-sensing literature research agent.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """GeoAI / remote-sensing literature research agent."""


@app.command("show-config")
def show_config(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to the TOML config."),
) -> None:
    """Validate the config file and print the resolved resource limits."""
    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for name, value in asdict(cfg.limits).items():
        typer.echo(f"{name} = {value}")
