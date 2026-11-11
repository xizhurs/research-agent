from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from research_agent.config import load_config
from research_agent.enso.analysis import Season
from research_agent.enso.analysis import compute_analysis as compute_enso_analysis
from research_agent.rainfall import Location, Provenance
from research_agent.rainfall.analysis import compute_analysis
from research_agent.rainfall.dryness import compute_dry_analysis
from research_agent.retrieval import SearchResult, Work

BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.toml"

TRACING_VARS = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "RESEARCH_AGENT_TRACE_CONTENT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
)


@pytest.fixture(autouse=True)
def _no_tracing_export(monkeypatch):
    """Tests never send traces, even if the developer's shell has tracing variables set."""
    for name in TRACING_VARS:
        monkeypatch.delenv(name, raising=False)


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




def day_range(start, end):
    """Every date from ``start`` to ``end`` inclusive."""
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


# A wet year is 5 mm every day; these ranges are dry (0 mm). Hand-computed expectations:
# unbroken spells 10, 20, 21 (2001-12-20 to 2002-01-09, crosses New Year), 5 and 29 days;
# yearly CDD 2000:20 2001:12 2002:9 2003:0 2004:29; annual totals (mm) 2000:1680 2001:1765
# 2002:1755 2003:1825 2004:1685.
DRY_RANGES = [
    (date(2000, 1, 1), date(2000, 1, 10)),
    (date(2000, 7, 1), date(2000, 7, 20)),
    (date(2001, 12, 20), date(2002, 1, 9)),
    (date(2002, 6, 1), date(2002, 6, 5)),
    (date(2004, 8, 1), date(2004, 8, 29)),
]


@pytest.fixture
def dry_spikes():
    return {d: 0.0 for start, end in DRY_RANGES for d in day_range(start, end)}


@pytest.fixture
def make_dry_analysis(rainfall_config, make_location, make_provenance, make_series, dry_spikes):
    """A real DryAnalysis (computed, not hand-built) for 2000-2004, id still empty."""

    def _make(location=None, event_date=None, event_year=None, from_cache=False, **config):
        settings = {"min_years": 3, "baseline_start_year": 2000, "baseline_end_year": 2004}
        cfg = replace(rainfall_config, **{**settings, "top_events": 3, **config})
        dates, values = make_series(2000, 2004, dry_spikes, base=5.0)
        return compute_dry_analysis(
            location or make_location(),
            dates,
            values,
            start_year=2000,
            end_year=2004,
            last_complete_year=2025,
            config=cfg,
            provenance=make_provenance(from_cache),
            event_date=event_date,
            event_year=event_year,
        )

    return _make


# --- ENSO -------------------------------------------------------------------------------------


@pytest.fixture
def enso_config(base_config, tmp_path):
    """ENSO settings with a temporary cache folder."""
    return replace(base_config.enso, cache_dir=tmp_path / "enso-cache")


@pytest.fixture
def make_seasons():
    """Consecutive seasons from index values, starting at DJF of ``year`` (or at ``slot``)."""

    def _make(values, year=2000, slot=0):
        first = year * 12 + slot
        return tuple(Season((first + i) // 12, (first + i) % 12, v) for i, v in enumerate(values))

    return _make


# Hand-built series, starting DJF 2000: two quiet seasons, an El Nino of 6 seasons (FMA-JAS 2000,
# peak +1.40 in AMJ 2000), two quiet seasons, a La Nina of 5 seasons (OND 2000-FMA 2001, peak
# -1.80 in DJF 2001), two quiet seasons.
ENSO_SERIES = (
    [0.0, 0.1]
    + [0.6, 1.0, 1.4, 1.2, 0.8, 0.6]
    + [0.2, -0.2]
    + [-0.6, -1.0, -1.8, -1.2, -0.7]
    + [0.0, 0.1]
)


@pytest.fixture
def make_enso_analysis(enso_config, make_seasons):
    """A real EnsoAnalysis (computed, not hand-built) over ENSO_SERIES, id still empty."""

    def _make(values=None, from_cache=False, **kwargs):
        return compute_enso_analysis(
            make_seasons(ENSO_SERIES if values is None else values),
            config=enso_config,
            source_url=enso_config.url,
            from_cache=from_cache,
            **kwargs,
        )

    return _make
