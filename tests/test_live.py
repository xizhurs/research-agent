"""Opt-in smoke test against the real API. Run with: uv run pytest -m live"""

from dataclasses import replace
from pathlib import Path

import pytest

from research_agent.config import load_config
from research_agent.credentials import load_credentials
from research_agent.llm import OpenAIClient, ToolResultMessage, ToolSpec, UserMessage
from research_agent.rainfall import RainfallClient
from research_agent.retrieval import OpenAlexClient

pytestmark = pytest.mark.live


def test_real_search_returns_works_with_abstracts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs" / "base.toml")
    key = load_credentials(root / ".env").openalex_api_key
    cfg = replace(base, openalex=replace(base.openalex, cache_dir=tmp_path))
    api_key = key.reveal() if key else None
    with OpenAlexClient(cfg.openalex, cfg.limits, api_key=api_key) as client:
        result = client.search(
            "deep learning land cover classification", year_from=2022, max_results=3
        )
    assert result.from_cache is False
    assert 1 <= len(result.works) <= 3
    assert result.skipped_records == 0
    assert all(w.id.startswith("W") for w in result.works)
    assert any(w.abstract for w in result.works)


def test_real_rainfall_analysis_for_one_city(tmp_path):
    """Geocode + ERA5 archive for Zhengzhou: one pinned-model request, then served from cache."""
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs" / "base.toml")
    cfg = replace(base.rainfall, cache_dir=tmp_path)
    with RainfallClient(cfg) as client:
        first = client.analyze("Zhengzhou", country_code="CN", start_year=1991, end_year=2020)
        second = client.analyze("Zhengzhou", country_code="CN", start_year=1991, end_year=2020)
    assert first.provenance.from_cache is False and second.provenance.from_cache is True
    assert first.location.country_code == "CN" and first.location.admin1 == "Henan"
    assert len(first.annual_maxima) == 30 and first.baseline.years == 30
    assert all(a.mm > 0 for a in first.annual_maxima)
    assert "models=era5" in first.provenance.request_url


def test_openai_tool_calling_round_trip():
    """One real tool-calling round trip: model requests a tool, we answer, model uses the result."""
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "base.toml")
    creds = load_credentials(root / ".env")
    if creds.openai_api_key is None:
        pytest.skip("OPENAI_API_KEY is not set")

    tool = ToolSpec(
        "lookup_secret_number",
        "Returns the secret number for a given name. Always use this tool to answer.",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
    client = OpenAIClient.from_credentials(cfg.llm, creds)
    instructions = "Answer in one short sentence. You must use the tool to find the number."
    messages = [UserMessage("What is the secret number for 'alice'?")]

    first = client.complete(
        instructions=instructions, messages=messages, tools=[tool], max_output_tokens=300
    )
    assert first.stop_reason == "tool_calls"
    (call,) = first.tool_calls
    assert call.name == "lookup_secret_number"
    assert isinstance(call.arguments.get("name"), str)
    assert first.usage.total_tokens > 0

    messages += [first.to_message(), ToolResultMessage(call.id, "4217")]
    second = client.complete(
        instructions=instructions,
        messages=messages,
        tools=[tool],
        max_output_tokens=300,
        allow_tools=False,
    )
    assert second.stop_reason == "stop"
    assert second.tool_calls == ()
    assert "4217" in (second.text or "")


def test_agent_end_to_end_stays_within_limits(tmp_path):
    """Real model + real OpenAlex: one question answered within every configured limit."""
    from research_agent.agent import ResearchAgent

    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs" / "base.toml")
    cfg = replace(base, openalex=replace(base.openalex, cache_dir=tmp_path))
    creds = load_credentials(root / ".env")
    if creds.openai_api_key is None:
        pytest.skip("OPENAI_API_KEY is not set")
    key = creds.openalex_api_key
    llm = OpenAIClient.from_credentials(cfg.llm, creds)
    with OpenAlexClient(cfg.openalex, cfg.limits, api_key=key.reveal() if key else None) as search:
        result = ResearchAgent(llm, search, cfg.limits).run(
            "Which deep learning methods are used for crop type mapping from Sentinel-2?"
        )
    stats, limits = result.stats, cfg.limits
    assert 1 <= stats.iterations <= limits.max_iterations
    assert stats.search_calls <= limits.max_search_calls
    assert result.answer.claims
    assert result.answer.limitations
    print(result.answer, stats)
