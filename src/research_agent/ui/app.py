"""The Gradio app: three tabs (Ask, Search, Rainfall) over the handlers in ``handlers.py``."""

from __future__ import annotations

from functools import partial

import gradio as gr

from research_agent.config import Config
from research_agent.credentials import Credentials
from research_agent.ui import handlers
from research_agent.ui.plots import DEFAULT_WINDOW_DAYS

_BADGE = (
    '<div style="display:inline-block;padding:4px 10px;margin:4px 0 12px;border-radius:6px;'
    'border:1px solid {color};color:{color};font-weight:600">{text}</div>'
)


def _badge(uses_llm: bool, text: str) -> str:
    return _BADGE.format(color="#b45309" if uses_llm else "#15803d", text=text)


def build_app(cfg: Config, creds: Credentials) -> gr.Blocks:
    """Build the interface. Handlers get the config and credentials; nothing is global."""
    with gr.Blocks(title="GeoAI research agent", analytics_enabled=False) as demo:
        gr.Markdown("# GeoAI research agent")

        with gr.Tabs():
            with gr.Tab("Ask"):
                gr.HTML(_badge(True, f"Uses the LLM ({cfg.llm.model}): costs tokens"))
                gr.Markdown(
                    "A GeoAI / remote-sensing literature or extreme-rainfall question. The model "
                    "decides which tools to call; the numbers come from code, not from the model."
                )
                question = gr.Textbox(
                    label="Question", lines=3, max_length=cfg.limits.max_question_chars
                )
                ask_button = gr.Button("Ask", variant="primary")
                ask_status = gr.Markdown()
                ask_answer = gr.Markdown()
                ask_plot = gr.Plot(label="Daily rainfall vs baseline")
                ask_note = gr.Markdown()
                ask_button.click(
                    partial(handlers.ask, cfg, creds),
                    question,
                    [ask_status, ask_answer, ask_plot, ask_note],
                    api_name="ask",
                )

            with gr.Tab("Search"):
                gr.HTML(_badge(False, "No LLM: deterministic OpenAlex search"))
                query = gr.Textbox(label="Search text (matched against title and abstract)")
                with gr.Row():
                    year_from = gr.Number(label="Earliest year (optional)", precision=0)
                    max_results = gr.Slider(
                        1,
                        cfg.limits.results_per_search,
                        value=cfg.limits.results_per_search,
                        step=1,
                        label="Results",
                    )
                    show_abstracts = gr.Checkbox(label="Show abstracts")
                search_button = gr.Button("Search", variant="primary")
                search_status = gr.Markdown()
                search_results = gr.Markdown()
                search_button.click(
                    partial(handlers.search, cfg, creds),
                    [query, year_from, max_results, show_abstracts],
                    [search_status, search_results],
                    api_name="search",
                )

            with gr.Tab("Rainfall"):
                gr.HTML(_badge(False, "No LLM: deterministic ERA5 analysis (Open-Meteo)"))
                with gr.Row():
                    city = gr.Textbox(label="City (name only)", placeholder="Zhengzhou")
                    country_code = gr.Textbox(
                        label="Country code (optional)", placeholder="CN", max_length=2
                    )
                    region = gr.Textbox(label="Region (optional)", placeholder="Henan")
                metric = gr.Radio(
                    [
                        ("Wet: heaviest day of each year (Rx1day)", "wet"),
                        ("Dry: spells and totals", "dry"),
                    ],
                    value="wet",
                    label="Metric",
                )
                with gr.Row():
                    start_year = gr.Number(
                        label=f"Start year (default {cfg.rainfall.default_start_year})", precision=0
                    )
                    end_year = gr.Number(
                        label="End year (default: last complete year)", precision=0
                    )
                with gr.Row():
                    event_date = gr.Textbox(label="Event date (optional)", placeholder="YYYY-MM-DD")
                    event_year = gr.Number(label="Event year (dry metric only)", precision=0)
                with gr.Accordion("Plot options", open=False):
                    plot_year = gr.Number(
                        label="Year to plot (default: event year, else latest valid year)",
                        precision=0,
                    )
                    window = gr.Slider(
                        1,
                        61,
                        value=DEFAULT_WINDOW_DAYS,
                        step=2,
                        label="Baseline smoothing window (days)",
                    )
                rain_button = gr.Button("Analyse", variant="primary")
                rain_status = gr.Markdown()
                rain_text = gr.Textbox(label="Analysis", lines=18, max_lines=40, interactive=False)
                rain_plot = gr.Plot(label="Daily rainfall vs baseline")
                rain_note = gr.Markdown()
                rain_button.click(
                    partial(handlers.rainfall, cfg),
                    [
                        city,
                        country_code,
                        region,
                        start_year,
                        end_year,
                        event_date,
                        metric,
                        event_year,
                        plot_year,
                        window,
                    ],
                    [rain_status, rain_text, rain_plot, rain_note],
                    api_name="rainfall",
                )
    return demo
