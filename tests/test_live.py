"""Opt-in smoke test against the real API. Run with: uv run pytest -m live"""

from dataclasses import replace
from pathlib import Path

import pytest

from research_agent.config import load_config
from research_agent.retrieval import OpenAlexClient, api_key_from_env

pytestmark = pytest.mark.live


def test_real_search_returns_works_with_abstracts(tmp_path):
    base = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.toml")
    cfg = replace(base, openalex=replace(base.openalex, cache_dir=tmp_path))
    with OpenAlexClient(cfg.openalex, cfg.limits, api_key=api_key_from_env()) as client:
        result = client.search(
            "deep learning land cover classification", year_from=2022, max_results=3
        )
    assert result.from_cache is False
    assert 1 <= len(result.works) <= 3
    assert result.skipped_records == 0
    assert all(w.id.startswith("W") for w in result.works)
    assert any(w.abstract for w in result.works)
