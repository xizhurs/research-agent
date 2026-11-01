from pathlib import Path

import pytest

from research_agent.config import load_config
from research_agent.retrieval import SearchResult, Work

BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.toml"


@pytest.fixture
def base_config():
    return load_config(BASE_CONFIG)


@pytest.fixture
def limits(base_config):
    return base_config.limits


@pytest.fixture
def make_work():
    def _make(
        id="W1",
        title="A title",
        abstract="An abstract.",
        year=2023,
        authors=("Ann Author",),
        venue="Journal",
        cited_by_count=5,
    ):
        return Work(
            id=id,
            url=f"https://openalex.org/{id}",
            title=title,
            year=year,
            date=None,
            authors=tuple(authors),
            venue=venue,
            doi=None,
            type="article",
            language="en",
            cited_by_count=cited_by_count,
            abstract=abstract,
            abstract_truncated=False,
        )

    return _make


@pytest.fixture
def make_result():
    def _make(works, query="q", total=None):
        return SearchResult(
            query=query,
            year_from=None,
            works=tuple(works),
            total_count=len(works) if total is None else total,
            skipped_records=0,
            from_cache=False,
        )

    return _make
