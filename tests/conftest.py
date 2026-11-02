from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from research_agent.config import load_config
from research_agent.rainfall import Location, Provenance
from research_agent.rainfall.analysis import compute_analysis
from research_agent.retrieval import SearchResult, Work

BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.toml"


@pytest.fixture
def base_config():
    return load_config(BASE_CONFIG)


@pytest.fixture
def limits(base_config):
    return base_config.limits


@pytest.fixture
def reporting(base_config):
    return base_config.reporting


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


# --- rainfall ---------------------------------------------------------------------------------

@pytest.fixture
def rainfall_config(base_config, tmp_path):
    """Rainfall settings with a temporary cache folder."""
    return replace(base_config.rainfall, cache_dir=tmp_path / "rainfall-cache")


@pytest.fixture
def make_location():
    def _make(
        id=1784658,
        name="Zhengzhou",
        admin1="Henan",
        country="China",
        country_code="CN",
        latitude=34.75778,
        longitude=113.64861,
        population=6650532,
        timezone="Asia/Shanghai",
    ):
        return Location(
            id, name, admin1, country, country_code, latitude, longitude, population, timezone
        )

    return _make


@pytest.fixture
def make_provenance():
    def _make(from_cache=False):
        return Provenance(
            dataset="ERA5 test dataset",
            variable="precipitation_sum",
            license="CC BY 4.0",
            request_url="https://archive.example/v1/archive?x=1",
            grid_latitude=34.75,
            grid_longitude=113.75,
            elevation_m=110.0,
            timezone="Asia/Shanghai",
            data_start="2000-01-01",
            data_end="2004-12-31",
            from_cache=from_cache,
        )

    return _make


@pytest.fixture
def make_series():
    """A complete daily series: ``base`` mm every day, ``spikes`` overriding given dates."""

    def _make(start_year, end_year, spikes=None, base=1.0, missing=()):
        spikes, missing = spikes or {}, set(missing)
        day = date(start_year, 1, 1)
        end = date(end_year, 12, 31)
        dates, values = [], []
        while day <= end:
            dates.append(day)
            values.append(None if day in missing else spikes.get(day, base))
            day += timedelta(days=1)
        return dates, values

    return _make


YEAR_MAXIMA = {2000: 10.0, 2001: 20.0, 2002: 30.0, 2003: 40.0, 2004: 50.0}


@pytest.fixture
def make_analysis(rainfall_config, make_location, make_provenance, make_series):
    """A real RainfallAnalysis (computed, not hand-built) for 2000-2004, id still empty."""

    def _make(location=None, event_date=None, from_cache=False, extra_spikes=None, **config):
        settings = {"min_years": 3, "baseline_start_year": 2000, "baseline_end_year": 2004}
        cfg = replace(rainfall_config, **{**settings, **config})
        spikes = {date(y, 7, 1): mm for y, mm in YEAR_MAXIMA.items()}
        spikes.update(extra_spikes or {})
        dates, values = make_series(2000, 2004, spikes)
        return compute_analysis(
            location or make_location(),
            dates,
            values,
            start_year=2000,
            end_year=2004,
            last_complete_year=2025,
            config=cfg,
            provenance=make_provenance(from_cache),
            event_date=event_date,
        )

    return _make


