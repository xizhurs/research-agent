"""RainfallClient: validation, caching, retries and error mapping. Mock HTTP, no network."""

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import httpx
import pytest

from research_agent.rainfall import (
    AmbiguousLocationError,
    InsufficientDataError,
    InvalidRainfallQueryError,
    LocationNotFoundError,
    RainfallApiError,
    RainfallClient,
    RainfallNetworkError,
    RainfallRateLimitError,
)

FIXTURES = Path(__file__).parent / "fixtures"
ARCHIVE = json.loads((FIXTURES / "openmeteo_archive.json").read_text(encoding="utf-8"))
SPRINGFIELD = json.loads((FIXTURES / "openmeteo_geocode_springfield.json").read_text("utf-8"))
ZHENGZHOU = {
    "results": [
        {
            "id": 1784658,
            "name": "Zhengzhou",
            "latitude": 34.75778,
            "longitude": 113.64861,
            "country_code": "CN",
            "country": "China",
            "admin1": "Henan",
            "timezone": "Asia/Shanghai",
            "population": 6650532,
        }
    ]
}
TODAY = date(2026, 11, 2)


class FakeServer:
    """Routes by host: ``geocode`` and ``archive`` are lists of scripted steps (last repeats)."""

    def __init__(self, geocode=None, archive=None):
        ok_geocode = [httpx.Response(200, json=ZHENGZHOU)]
        ok_archive = [httpx.Response(200, json=ARCHIVE)]
        self.geocode = list(geocode if geocode is not None else ok_geocode)
        self.archive = list(archive if archive is not None else ok_archive)
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        steps = self.geocode if "geocoding" in request.url.host else self.archive
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, Exception):
            raise step
        return step

    def hosts(self):
        return [r.url.host.split(".")[0] for r in self.requests]


@pytest.fixture
def cfg(rainfall_config):
    # The fixture holds 2019-2021; a three-year baseline over it needs three valid years.
    return replace(rainfall_config, min_years=3, baseline_start_year=2019, baseline_end_year=2021)


def make_client(cfg, server, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return RainfallClient(
        cfg,
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
        sleep=sleeps.append,
        today=lambda: TODAY,
    )


def analyze(cfg, server, **kwargs):
    kwargs.setdefault("start_year", 2019)
    kwargs.setdefault("end_year", 2021)
    return make_client(cfg, server).analyze("Zhengzhou", **kwargs)


# --- happy path and provenance ----------------------------------------------------------------


def test_analyze_computes_from_recorded_data_with_provenance(cfg):
    server = FakeServer()
    result = analyze(cfg, server, country_code="cn", event_date=date(2021, 7, 20))

    assert [a.year for a in result.annual_maxima] == [2019, 2020, 2021]
    assert result.baseline.years == 3
    assert result.event.date == "2021-07-20"
    assert result.location.label == "Zhengzhou, Henan, China"
    prov = result.provenance
    assert (prov.grid_latitude, prov.grid_longitude) == (ARCHIVE["latitude"], ARCHIVE["longitude"])
    assert prov.from_cache is False
    assert (prov.data_start, prov.data_end) == ("2019-01-01", "2021-12-31")
    assert "models=era5" in prov.request_url and "timezone=auto" in prov.request_url
    assert "CC BY 4.0" in prov.license
    assert server.hosts() == ["geocoding-api", "archive-api"]


def test_requests_pin_the_model_and_carry_no_credentials(cfg):
    server = FakeServer()
    analyze(cfg, server, country_code="CN")
    geocode, archive = server.requests
    assert geocode.url.params["countryCode"] == "CN" and geocode.url.params["name"] == "Zhengzhou"
    q = archive.url.params
    assert (q["models"], q["daily"], q["timezone"]) == ("era5", "precipitation_sum", "auto")
    assert (q["start_date"], q["end_date"]) == ("2019-01-01", "2021-12-31")
    assert q["latitude"] == "34.7578" and q["longitude"] == "113.6486"
    for request in server.requests:
        assert "authorization" not in request.headers and "apikey" not in request.url.params


def test_default_period_is_configured_start_to_last_complete_year(cfg):
    server = FakeServer()
    result = make_client(cfg, server).analyze("Zhengzhou")
    q = server.requests[1].url.params
    assert (q["start_date"], q["end_date"]) == ("1950-01-01", "2025-12-31")
    assert (result.start_year, result.end_year) == (1950, 2025)
    assert len(result.excluded_years) == 73  # the fixture only covers 2019-2021


def test_current_year_is_requested_up_to_today_and_excluded_as_incomplete(cfg):
    server = FakeServer()
    result = make_client(cfg, server).analyze("Zhengzhou", start_year=2019, end_year=2026)
    assert server.requests[1].url.params["end_date"] == "2026-11-02"
    assert (2026, "year not complete") in result.excluded_years


# --- cache ------------------------------------------------------------------------------------


def test_second_identical_analysis_is_served_from_cache(cfg):
    server = FakeServer()
    first = analyze(cfg, server)
    assert len(server.requests) == 2 and first.provenance.from_cache is False
    second = analyze(cfg, server)
    assert len(server.requests) == 2  # no new network call
    assert second.provenance.from_cache is True
    assert second.annual_maxima == first.annual_maxima


def test_changing_the_period_misses_the_archive_cache_but_reuses_geocoding(cfg):
    server = FakeServer()
    analyze(cfg, server)
    analyze(cfg, server, end_year=2020)
    assert server.hosts() == ["geocoding-api", "archive-api", "archive-api"]


def test_malformed_response_is_not_cached(cfg):
    broken = {**ARCHIVE, "daily_units": {"time": "iso8601", "precipitation_sum": "inch"}}
    server = FakeServer(
        archive=[httpx.Response(200, json=broken), httpx.Response(200, json=ARCHIVE)]
    )
    with pytest.raises(RainfallApiError, match="unit"):
        analyze(cfg, server)
    assert analyze(cfg, server).annual_maxima  # the retry reached the network, not a bad cache


def test_cache_write_failure_is_not_fatal(cfg, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    cfg = replace(cfg, cache_dir=blocker / "sub")  # a folder cannot be created under a file
    assert analyze(cfg, FakeServer()).annual_maxima


# --- locations --------------------------------------------------------------------------------


def test_ambiguous_city_stops_before_any_archive_request(cfg):
    server = FakeServer(geocode=[httpx.Response(200, json=SPRINGFIELD)])
    with pytest.raises(AmbiguousLocationError):
        make_client(cfg, server).analyze("Springfield", start_year=2019, end_year=2021)
    assert server.hosts() == ["geocoding-api"]


def test_region_resolves_an_ambiguous_city(cfg):
    server = FakeServer(geocode=[httpx.Response(200, json=SPRINGFIELD)])
    result = make_client(cfg, server).analyze(
        "Springfield", region="Illinois", start_year=2019, end_year=2021
    )
    assert result.location.admin1 == "Illinois"


def test_unknown_city_is_not_found(cfg):
    server = FakeServer(geocode=[httpx.Response(200, json={"generationtime_ms": 0.1})])
    with pytest.raises(LocationNotFoundError, match="Zhengzhou"):
        analyze(cfg, server)
    assert server.hosts() == ["geocoding-api"]


def test_event_day_without_data_is_insufficient(cfg):
    nulls = {**ARCHIVE, "daily": {**ARCHIVE["daily"]}}
    values = list(nulls["daily"]["precipitation_sum"])
    values[nulls["daily"]["time"].index("2021-07-20")] = None
    nulls["daily"]["precipitation_sum"] = values
    with pytest.raises(InsufficientDataError, match="2021-07-20"):
        server = FakeServer(archive=[httpx.Response(200, json=nulls)])
        analyze(cfg, server, event_date=date(2021, 7, 20))


# --- invalid input never reaches the network --------------------------------------------------


@pytest.mark.parametrize(
    ("city", "kwargs"),
    [
        ("", {}),
        ("   ", {}),
        ("x" * 201, {}),
        ("Paris", {"country_code": "FRA"}),
        ("Paris", {"country_code": "1A"}),
        ("Paris", {"country_code": 5}),
        ("Paris", {"region": " "}),
        ("Paris", {"start_year": 1939}),
        ("Paris", {"end_year": 2027}),
        ("Paris", {"start_year": 2021, "end_year": 2019}),
        ("Paris", {"start_year": "2019"}),
        ("Paris", {"start_year": True}),
        ("Paris", {"start_year": 2019, "end_year": 2021, "event_date": date(2022, 1, 1)}),
        ("Paris", {"start_year": 2019, "end_year": 2021, "event_date": date(2018, 12, 31)}),
        ("Paris", {"event_date": date(2026, 11, 3)}),  # in the future
        (None, {}),
    ],
)
def test_invalid_input_is_rejected_before_any_request(cfg, city, kwargs):
    server = FakeServer()
    with pytest.raises(InvalidRainfallQueryError):
        make_client(cfg, server).analyze(city, **kwargs)
    assert server.requests == []


def test_event_in_the_current_year_is_allowed_when_end_year_covers_it(cfg):
    client = make_client(cfg, FakeServer())
    # The fixture has no 2026 data, so the day has no value: reaching that error (and not
    # InvalidRainfallQueryError) proves the date passed validation.
    with pytest.raises(InsufficientDataError):
        client.analyze("Zhengzhou", start_year=2019, end_year=2026, event_date=date(2026, 10, 1))


# --- transport behaviour ----------------------------------------------------------------------


def test_transient_failure_is_retried_with_backoff(cfg):
    busy = httpx.Response(503, json={"reason": "busy"})
    server = FakeServer(archive=[busy, httpx.Response(200, json=ARCHIVE)])
    sleeps: list[float] = []
    result = make_client(cfg, server, sleeps).analyze("Zhengzhou", start_year=2019, end_year=2021)
    assert result.annual_maxima and sleeps == [cfg.backoff_base_seconds]


def test_retry_after_is_honoured_when_short(cfg):
    limited = httpx.Response(429, headers={"Retry-After": "2"}, json={"reason": "slow down"})
    server = FakeServer(archive=[limited, httpx.Response(200, json=ARCHIVE)])
    sleeps: list[float] = []
    make_client(cfg, server, sleeps).analyze("Zhengzhou", start_year=2019, end_year=2021)
    assert sleeps == [2.0]


def test_long_retry_after_fails_fast(cfg):
    limited = httpx.Response(429, headers={"Retry-After": "86400"}, json={"reason": "daily limit"})
    sleeps: list[float] = []
    with pytest.raises(RainfallRateLimitError, match="daily limit"):
        make_client(cfg, FakeServer(archive=[limited]), sleeps).analyze(
            "Zhengzhou", start_year=2019, end_year=2021
        )
    assert sleeps == []


def test_persistent_429_becomes_a_rate_limit_error_after_bounded_retries(cfg):
    server = FakeServer(archive=[httpx.Response(429, json={"reason": "too many"})])
    with pytest.raises(RainfallRateLimitError, match="4 attempts"):
        analyze(cfg, server)
    assert server.hosts().count("archive-api") == cfg.max_retries + 1


def test_network_failure_after_retries(cfg):
    server = FakeServer(archive=[httpx.ConnectError("boom")])
    sleeps: list[float] = []
    with pytest.raises(RainfallNetworkError, match="ConnectError"):
        make_client(cfg, server, sleeps).analyze("Zhengzhou", start_year=2019, end_year=2021)
    assert len(sleeps) == cfg.max_retries


def test_client_error_is_not_retried_and_shows_the_reason(cfg):
    bad = httpx.Response(400, json={"error": True, "reason": "start_date is out of allowed range"})
    server = FakeServer(archive=[bad])
    with pytest.raises(RainfallApiError, match="out of allowed range"):
        analyze(cfg, server)
    assert server.hosts().count("archive-api") == 1


def test_non_json_success_is_an_api_error(cfg):
    with pytest.raises(RainfallApiError, match="non-JSON"):
        analyze(cfg, FakeServer(archive=[httpx.Response(200, text="<html>")]))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: {**d, "daily": {"time": d["daily"]["time"], "precipitation_sum": [1.0]}},
        lambda d: {k: v for k, v in d.items() if k != "latitude"},
        lambda d: {**d, "daily": {"time": ["not-a-date"], "precipitation_sum": [1.0]}},
        lambda d: {**d, "daily": {"time": ["2020-01-01"], "precipitation_sum": [-5.0]}},
        lambda d: {**d, "daily": {"time": [], "precipitation_sum": []}},
        lambda d: [],
    ],
)
def test_unexpected_archive_shapes_are_api_errors(cfg, mutate):
    with pytest.raises(RainfallApiError):
        analyze(cfg, FakeServer(archive=[httpx.Response(200, json=mutate(ARCHIVE))]))


# --- dry metric ---------------------------------------------------------------------------------


def test_dry_metric_computes_dry_statistics_from_the_recorded_data(cfg):
    result = analyze(cfg, FakeServer(), metric="dry", event_year=2020)
    assert result.metric == "dry" and result.valid_years == (2019, 2020, 2021)
    assert result.year_focus.year == 2020 and result.baseline.years == 3
    assert result.longest_spells and result.dry_day_mm == 1.0
    assert "models=era5" in result.provenance.request_url


def test_dry_after_wet_for_the_same_city_makes_no_new_request(cfg):
    server = FakeServer()
    wet = analyze(cfg, server)
    assert wet.metric == "wet" and len(server.requests) == 2
    dry = analyze(cfg, server, metric="dry")
    assert len(server.requests) == 2  # geocoding and archive both came from the cache
    assert dry.metric == "dry" and dry.provenance.from_cache is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "drought"},
        {"metric": None},
        {"event_year": 2020},  # event_year needs metric 'dry'
        {"metric": "wet", "event_year": 2020},
        {"metric": "dry", "event_year": 2018},  # before start_year
        {"metric": "dry", "event_year": 2022},  # after end_year
        {"metric": "dry", "event_year": True},
        {"metric": "dry", "event_year": "2020"},
    ],
)
def test_invalid_metric_or_event_year_is_rejected_before_any_request(cfg, kwargs):
    server = FakeServer()
    with pytest.raises(InvalidRainfallQueryError):
        make_client(cfg, server).analyze("Paris", start_year=2019, end_year=2021, **kwargs)
    assert server.requests == []


def test_event_year_in_the_current_year_is_reported_not_ranked(cfg):
    result = make_client(cfg, FakeServer()).analyze(
        "Zhengzhou", start_year=2019, end_year=2026, metric="dry", event_year=2026
    )
    assert result.year_focus is None
    assert any("2026 was excluded (year not complete)" in c for c in result.caveats)
