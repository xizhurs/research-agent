"""TemperatureClient: request, strict parsing, cache, point mode, validation. Mock HTTP only."""

import copy
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import httpx
import pytest

from research_agent.rainfall import (
    AmbiguousLocationError,
    InvalidRainfallQueryError,
    RainfallApiError,
    RainfallClient,
)
from research_agent.temperature import (
    TemperatureClient,
    TemperatureExtremes,
    TemperatureMonthly,
    TemperatureYearly,
)
from research_agent.temperature.client import parse_temperature_archive

FIXTURES = Path(__file__).parent / "fixtures"
# Paris, 2019-2021, recorded from Open-Meteo (models=era5) on 2026-11-12.
ARCHIVE = json.loads((FIXTURES / "openmeteo_temperature.json").read_text(encoding="utf-8"))
RAIN_ARCHIVE = json.loads((FIXTURES / "openmeteo_archive.json").read_text(encoding="utf-8"))
SPRINGFIELD = json.loads((FIXTURES / "openmeteo_geocode_springfield.json").read_text("utf-8"))
PARIS = {
    "results": [
        {
            "id": 2988507,
            "name": "Paris",
            "latitude": 48.85341,
            "longitude": 2.3488,
            "country_code": "FR",
            "country": "France",
            "admin1": "Île-de-France",
            "timezone": "Europe/Paris",
            "population": 2138551,
        }
    ]
}
TODAY = date(2026, 11, 12)


class FakeServer:
    """Routes by host and by the requested variable; records every request."""

    def __init__(self, geocode=PARIS, archive=None):
        self.geocode = geocode
        self.archive = archive if archive is not None else ARCHIVE
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        if "geocoding" in request.url.host:
            return httpx.Response(200, json=self.geocode)
        if "precipitation_sum" in request.url.params.get("daily", ""):
            return httpx.Response(200, json=RAIN_ARCHIVE)
        return httpx.Response(200, json=self.archive)

    def hosts(self):
        return [r.url.host.split(".")[0] for r in self.requests]


@pytest.fixture
def cfg(rainfall_config):
    return replace(rainfall_config, min_years=3, baseline_start_year=2019, baseline_end_year=2021)


def client_for(cfg, server, cls=TemperatureClient):
    return cls(
        cfg,
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
        sleep=lambda _s: None,
        today=lambda: TODAY,
    )


def analyze(cfg, server, **kwargs):
    kwargs.setdefault("start_year", 2019)
    kwargs.setdefault("end_year", 2021)
    return client_for(cfg, server).analyze("Paris", **kwargs)


def test_hot_day_from_recorded_data_with_the_three_variables_requested(cfg):
    server = FakeServer()
    record, archive = analyze(cfg, server, event_date=date(2019, 7, 25))
    assert isinstance(record, TemperatureExtremes) and record.metric == "hot"
    assert record.event.value_c == 41.4 and record.event.rank_among_days == 1
    assert [a.year for a in record.annual] == [2019, 2020, 2021]
    params = server.requests[-1].url.params
    assert params["daily"] == "temperature_2m_max,temperature_2m_min,temperature_2m_mean"
    assert params["models"] == "era5" and params["timezone"] == "auto"
    assert (record.provenance.grid_latitude, record.provenance.grid_longitude) == (48.75, 2.25)
    assert len(archive.tmax) == len(archive.dates) == 1096
    assert server.hosts() == ["geocoding-api", "archive-api"]


def test_every_scale_and_metric_shares_one_archive_request(cfg, tmp_path):
    server = FakeServer()
    client = client_for(cfg, server)
    years = {"start_year": 2019, "end_year": 2021}
    cold, _ = client.analyze("Paris", metric="cold", **years)
    yearly, _ = client.analyze("Paris", scale="yearly", event_year=2020, **years)
    monthly, _ = client.analyze(
        "Paris", scale="monthly", event_year=2019, event_month=7, **years
    )
    assert cold.summary.startswith("TNn") and cold.top[0].value_c < 0
    assert isinstance(yearly, TemperatureYearly) and yearly.year_focus.year == 2020
    assert isinstance(monthly, TemperatureMonthly) and monthly.month_focus.mean_c is not None
    assert server.hosts() == ["geocoding-api", "archive-api"]  # then all from the cache
    assert yearly.provenance.from_cache and monthly.provenance.from_cache


def test_rainfall_and_temperature_share_the_geocoder_cache_but_not_the_archive(cfg):
    server = FakeServer()
    client_for(cfg, server, RainfallClient).analyze("Paris", start_year=2019, end_year=2021)
    analyze(cfg, server)
    assert server.hosts() == ["geocoding-api", "archive-api", "archive-api"]


def test_point_mode_skips_geocoding_and_names_the_grid_cell(cfg):
    server = FakeServer()
    record, _ = client_for(cfg, server).analyze_point(
        48.8534, 2.3488, start_year=2019, end_year=2021, scale="yearly"
    )
    assert server.hosts() == ["archive-api"]
    assert record.location.name == "48.8534, 2.3488"
    assert "ERA5 grid cell at 48.7500, 2.2500" in record.caveats[-1]


def test_an_ambiguous_city_runs_no_analysis(cfg):
    server = FakeServer(geocode=SPRINGFIELD)
    with pytest.raises(AmbiguousLocationError):
        analyze(cfg, server)
    assert server.hosts() == ["geocoding-api"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"metric": "wet"}, "metric must be 'hot' or 'cold'"),
        ({"scale": "weekly"}, "scale must be"),
        ({"event_year": 2020}, "event_year needs scale 'yearly' or 'monthly'; for one day"),
        ({"scale": "yearly", "event_date": date(2020, 1, 1)}, "event_date is for scale 'daily'"),
        ({"scale": "yearly", "event_month": 7, "event_year": 2020}, "event_month needs"),
        ({"start_year": 2022, "end_year": 2021}, "start_year must not be after end_year"),
    ],
)
def test_bad_arguments_are_rejected_before_any_request(cfg, kwargs, message):
    server = FakeServer()
    with pytest.raises(InvalidRainfallQueryError, match=message):
        analyze(cfg, server, **kwargs)
    assert server.requests == []


def test_a_bad_point_is_rejected_before_any_request(cfg):
    server = FakeServer()
    with pytest.raises(InvalidRainfallQueryError, match="latitude"):
        client_for(cfg, server).analyze_point(95.0, 0.0)
    assert server.requests == []


def _broken(**changes):
    payload = copy.deepcopy(ARCHIVE)
    for path, value in changes.items():
        section, key = path.split("__")
        payload[section][key] = value
    return payload


@pytest.mark.parametrize(
    "payload, message",
    [
        (_broken(daily_units__temperature_2m_max="°F"), "unit"),
        (_broken(daily__temperature_2m_min=[1.0]), "do not line up"),
        (_broken(daily__temperature_2m_mean=[999.0] * 1096), "invalid temperature_2m_mean"),
        ({"daily": {}}, "missing"),
        ([], "not an object"),
    ],
)
def test_parsing_is_strict(payload, message):
    with pytest.raises(RainfallApiError, match=message):
        parse_temperature_archive(payload)


def test_missing_values_are_kept_as_missing():
    payload = _broken(daily__temperature_2m_max=[None] + ARCHIVE["daily"]["temperature_2m_max"][1:])
    archive = parse_temperature_archive(payload)
    assert archive.tmax[0] is None and archive.tmin[0] is not None
    assert archive.series("hot") is archive.tmax and archive.series("mean") is archive.tmean
