"""EnsoClient: validation, caching, retries and error mapping. Mock HTTP, no network."""

from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from research_agent.enso import (
    EnsoApiError,
    EnsoClient,
    EnsoNetworkError,
    InsufficientEnsoDataError,
    InvalidEnsoQueryError,
)

REAL = (Path(__file__).parent / "fixtures" / "roni_1950_2001.txt").read_text(encoding="utf-8")


class FakeServer:
    """Scripted responses or exceptions (the last one repeats); records every request."""

    def __init__(self, *steps):
        self.steps = list(steps or [httpx.Response(200, text=REAL)])
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return step


def cached_files(cfg):
    return list(cfg.cache_dir.glob("*.json")) if cfg.cache_dir.exists() else []


def make_client(cfg, server, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return EnsoClient(
        cfg,
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
        sleep=sleeps.append,
    )


# --- happy path and caching -------------------------------------------------------------------


def test_analyze_downloads_parses_and_reports_the_source(enso_config):
    server = FakeServer()
    a = make_client(enso_config, server).analyze()
    assert a.index_name == "RONI" and a.source_url == enso_config.url
    assert (a.record_first, a.record_last) == ("DJF 1950", "NDJ 2001")
    assert a.from_cache is False and len(a.events) == 27
    assert [str(r.url) for r in server.requests] == [enso_config.url]


def test_second_call_is_served_from_the_cache_without_a_request(enso_config):
    server = FakeServer()
    client = make_client(enso_config, server)
    first = client.analyze()
    second = client.analyze(phase="la_nina", start_year=1990)
    assert first.from_cache is False and second.from_cache is True
    assert len(server.requests) == 1
    assert second.events and all(e.phase == "la_nina" for e in second.events)


def test_a_new_client_shares_the_cache_on_disk(enso_config):
    make_client(enso_config, FakeServer()).analyze()
    again = FakeServer()
    assert make_client(enso_config, again).analyze().from_cache is True
    assert again.requests == []


def test_an_expired_cache_entry_is_downloaded_again(enso_config):
    make_client(enso_config, FakeServer()).analyze()
    server = FakeServer()
    stale = replace(enso_config, cache_ttl_days=1e-9)
    assert make_client(stale, server).analyze().from_cache is False
    assert len(server.requests) == 1


def test_a_cached_file_with_unexpected_content_is_ignored_and_replaced(enso_config):
    make_client(enso_config, FakeServer()).analyze()
    (cached,) = list(enso_config.cache_dir.glob("*.json"))
    cached.write_text('"not a RONI table"', encoding="utf-8")
    server = FakeServer()
    assert make_client(enso_config, server).analyze().from_cache is False
    assert len(server.requests) == 1


# --- validation (before any request) ----------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase": "warm"},
        {"phase": None},
        {"start_year": 0},
        {"start_year": -1990},
        {"start_year": True},
        {"end_year": "2000"},
        {"end_year": 1990.5},
        {"year": 0},
        {"year": False},
        {"start_year": 2000, "end_year": 1999},
    ],
)
def test_invalid_arguments_are_rejected_before_any_request(enso_config, kwargs):
    server = FakeServer()
    with pytest.raises(InvalidEnsoQueryError):
        make_client(enso_config, server).analyze(**kwargs)
    assert server.requests == []


def test_a_year_outside_the_record_is_reported_after_the_download(enso_config):
    server = FakeServer()
    with pytest.raises(InsufficientEnsoDataError, match="year 1940"):
        make_client(enso_config, server).analyze(year=1940)
    assert len(server.requests) == 1


# --- retries and errors -----------------------------------------------------------------------


def test_server_error_then_success_retries_with_backoff(enso_config):
    sleeps = []
    server = FakeServer(httpx.Response(503), httpx.Response(200, text=REAL))
    a = make_client(enso_config, server, sleeps).analyze()
    assert len(server.requests) == 2 and a.events
    assert sleeps == [enso_config.backoff_base_seconds]


def test_rate_limit_is_retried_too(enso_config):
    server = FakeServer(httpx.Response(429), httpx.Response(429), httpx.Response(200, text=REAL))
    make_client(enso_config, server).analyze()
    assert len(server.requests) == 3


def test_persistent_server_error_gives_a_clear_api_error_after_the_retry_budget(enso_config):
    sleeps = []
    server = FakeServer(httpx.Response(500))
    with pytest.raises(EnsoApiError, match="failed after 4 attempts: HTTP 500"):
        make_client(enso_config, server, sleeps).analyze()
    assert len(server.requests) == enso_config.max_retries + 1
    assert len(sleeps) == enso_config.max_retries
    assert cached_files(enso_config) == []


def test_backoff_doubles_and_is_capped(enso_config):
    sleeps = []
    cfg = replace(enso_config, backoff_base_seconds=10.0, max_backoff_seconds=25.0)
    with pytest.raises(EnsoApiError):
        make_client(cfg, FakeServer(httpx.Response(502)), sleeps).analyze()
    assert sleeps == [10.0, 20.0, 25.0]


def test_connection_failures_raise_a_network_error_after_retries(enso_config):
    server = FakeServer(httpx.ConnectError("boom"))
    with pytest.raises(EnsoNetworkError, match="ConnectError"):
        make_client(enso_config, server).analyze()
    assert len(server.requests) == enso_config.max_retries + 1


def test_a_missing_file_is_not_retried(enso_config):
    server = FakeServer(httpx.Response(404))
    with pytest.raises(EnsoApiError, match="HTTP 404"):
        make_client(enso_config, server).analyze()
    assert len(server.requests) == 1


def test_a_download_that_is_not_a_roni_table_is_an_error_and_is_not_cached(enso_config):
    server = FakeServer(httpx.Response(200, text="<html>maintenance</html>"))
    with pytest.raises(EnsoApiError, match="header"):
        make_client(enso_config, server).analyze()
    assert cached_files(enso_config) == []


def test_a_file_with_a_gap_is_rejected_rather_than_splitting_events(enso_config):
    lines = REAL.splitlines()
    broken = "\n".join(lines[:100] + lines[101:]) + "\n"  # one season removed
    with pytest.raises(EnsoApiError, match="does not follow"):
        make_client(enso_config, FakeServer(httpx.Response(200, text=broken))).analyze()


def test_no_credentials_are_sent(enso_config):
    server = FakeServer()
    make_client(enso_config, server).analyze()
    sent = {k.lower() for k in server.requests[0].headers}
    assert "authorization" not in sent and "x-api-key" not in sent
