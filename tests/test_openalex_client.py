import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from research_agent.config import load_config
from research_agent.retrieval import (
    ApiError,
    AuthError,
    InvalidQueryError,
    NetworkError,
    OpenAlexClient,
    RateLimitError,
)
from research_agent.retrieval.openalex import api_key_from_env

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "openalex_search.json"
PAYLOAD = json.loads(FIXTURE.read_text(encoding="utf-8"))


class FakeServer:
    """Scripted HTTP responses; records every request."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return step


def ok():
    return httpx.Response(200, json=PAYLOAD)


@pytest.fixture
def cfg(tmp_path):
    base = load_config(ROOT / "configs" / "base.toml")
    return replace(base, openalex=replace(base.openalex, cache_dir=tmp_path / "cache"))


def make_client(cfg, server, sleeps=None, api_key=None):
    sleeps = sleeps if sleeps is not None else []
    return OpenAlexClient(
        cfg.openalex,
        cfg.limits,
        api_key=api_key,
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
        sleep=sleeps.append,
    )


def test_search_returns_parsed_works(cfg):
    server = FakeServer(ok())
    result = make_client(cfg, server).search("land cover", year_from=2022)
    assert [w.id for w in result.works] == ["W4388333579", "W4379984088", "W4323022345"]
    assert result.total_count == PAYLOAD["meta"]["count"]
    assert result.from_cache is False
    assert result.missing_abstract_count == 0
    req = server.requests[0]
    assert req.url.path == "/works"
    assert req.url.params["search.title_and_abstract"] == "land cover"
    assert req.url.params["filter"] == "from_publication_date:2022-01-01"
    assert req.url.params["per_page"] == str(cfg.limits.results_per_search)


def test_second_identical_search_hits_cache_without_network(cfg):
    server = FakeServer(ok())
    client = make_client(cfg, server)
    first = client.search("land cover")
    second = client.search("land cover")
    assert len(server.requests) == 1
    assert second.from_cache is True
    assert second.works == first.works


def test_different_query_is_not_served_from_cache(cfg):
    server = FakeServer(ok())
    client = make_client(cfg, server)
    client.search("land cover")
    client.search("crop mapping")
    assert len(server.requests) == 2


def test_cache_persists_across_clients(cfg):
    make_client(cfg, FakeServer(ok())).search("q")
    server = FakeServer(ok())
    result = make_client(cfg, server).search("q")
    assert result.from_cache is True
    assert server.requests == []


def test_cached_abstracts_follow_current_limit(cfg):
    make_client(cfg, FakeServer(ok())).search("q")
    tight = replace(cfg, limits=replace(cfg.limits, abstract_max_chars=50))
    result = make_client(tight, FakeServer(ok())).search("q")
    assert result.from_cache is True
    assert all(len(w.abstract) <= 50 for w in result.works)


def test_cached_response_with_bad_shape_is_refetched(cfg):
    client = make_client(cfg, FakeServer(ok()))
    client.search("q")
    (entry,) = (cfg.openalex.cache_dir).glob("*.json")
    entry.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    server = FakeServer(ok())
    result = make_client(cfg, server).search("q")
    assert result.from_cache is False
    assert len(server.requests) == 1


def test_malformed_response_is_not_cached(cfg):
    server = FakeServer(httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ApiError):
        make_client(cfg, server).search("q")
    assert list(cfg.openalex.cache_dir.glob("*.json")) == []


def test_max_results_can_lower_but_never_exceed_cap(cfg):
    server = FakeServer(ok())
    client = make_client(cfg, server)
    client.search("a", max_results=2)
    client.search("b", max_results=10_000)
    assert server.requests[0].url.params["per_page"] == "2"
    assert server.requests[1].url.params["per_page"] == str(cfg.limits.results_per_search)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 501},
        {"query": "ok", "year_from": 1},
        {"query": "ok", "year_from": True},
        {"query": "ok", "max_results": 0},
    ],
)
def test_invalid_arguments_rejected_before_any_request(cfg, kwargs):
    server = FakeServer(ok())
    q = kwargs.pop("query")
    with pytest.raises(InvalidQueryError):
        make_client(cfg, server).search(q, **kwargs)
    assert server.requests == []


def test_api_key_sent_as_header_never_in_url(cfg):
    server = FakeServer(ok())
    make_client(cfg, server, api_key="secret-key").search("q")
    req = server.requests[0]
    assert req.headers["Authorization"] == "Bearer secret-key"
    assert "secret-key" not in str(req.url)


def test_no_api_key_sends_no_auth_header(cfg):
    server = FakeServer(ok())
    make_client(cfg, server).search("q")
    assert "Authorization" not in server.requests[0].headers


def test_cache_files_do_not_contain_the_api_key(cfg):
    make_client(cfg, FakeServer(ok()), api_key="secret-key").search("q")
    for f in cfg.openalex.cache_dir.glob("*"):
        assert "secret-key" not in f.name
        assert "secret-key" not in f.read_text(encoding="utf-8")


def test_retries_429_with_exponential_backoff_then_succeeds(cfg):
    server = FakeServer(httpx.Response(429), httpx.Response(429), ok())
    sleeps: list[float] = []
    result = make_client(cfg, server, sleeps).search("q")
    assert len(result.works) == 3
    assert len(server.requests) == 3
    assert sleeps == [1.0, 2.0]


def test_retry_after_header_is_honoured(cfg):
    server = FakeServer(httpx.Response(429, headers={"Retry-After": "7"}), ok())
    sleeps: list[float] = []
    make_client(cfg, server, sleeps).search("q")
    assert sleeps == [7.0]


def test_retry_after_beyond_cap_fails_fast_without_sleeping(cfg):
    server = FakeServer(httpx.Response(429, headers={"Retry-After": "40000"}))
    sleeps: list[float] = []
    with pytest.raises(RateLimitError, match="giving up"):
        make_client(cfg, server, sleeps).search("q")
    assert len(server.requests) == 1
    assert sleeps == []


def test_rate_limit_exhausts_retries(cfg):
    server = FakeServer(httpx.Response(429, json={"error": "not_enough_credits"}))
    sleeps: list[float] = []
    with pytest.raises(RateLimitError, match="not_enough_credits"):
        make_client(cfg, server, sleeps).search("q")
    assert len(server.requests) == cfg.openalex.max_retries + 1
    assert len(sleeps) == cfg.openalex.max_retries


def test_server_error_retried_then_raises_api_error(cfg):
    server = FakeServer(httpx.Response(503, text="down"))
    with pytest.raises(ApiError, match="503"):
        make_client(cfg, server).search("q")
    assert len(server.requests) == cfg.openalex.max_retries + 1


def test_backoff_is_capped(cfg):
    capped = replace(cfg, openalex=replace(cfg.openalex, max_retries=6, max_backoff_seconds=3.0))
    server = FakeServer(httpx.Response(500))
    sleeps: list[float] = []
    with pytest.raises(ApiError):
        make_client(capped, server, sleeps).search("q")
    assert sleeps == [1.0, 2.0, 3.0, 3.0, 3.0, 3.0]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_not_retried(cfg, status):
    server = FakeServer(httpx.Response(status, json={"error": "key_not_activated"}))
    with pytest.raises(AuthError, match="key_not_activated"):
        make_client(cfg, server).search("q")
    assert len(server.requests) == 1


@pytest.mark.parametrize("status", [400, 404])
def test_other_client_errors_not_retried(cfg, status):
    server = FakeServer(httpx.Response(status, text="nope"))
    with pytest.raises(ApiError, match=str(status)):
        make_client(cfg, server).search("q")
    assert len(server.requests) == 1


def test_timeout_retried_then_network_error(cfg):
    server = FakeServer(httpx.ReadTimeout("slow"))
    sleeps: list[float] = []
    with pytest.raises(NetworkError, match="ReadTimeout"):
        make_client(cfg, server, sleeps).search("q")
    assert len(server.requests) == cfg.openalex.max_retries + 1


def test_transient_network_failure_recovers(cfg):
    server = FakeServer(httpx.ConnectError("refused"), ok())
    result = make_client(cfg, server).search("q")
    assert len(result.works) == 3


def test_zero_retries_means_single_attempt(cfg):
    once = replace(cfg, openalex=replace(cfg.openalex, max_retries=0))
    server = FakeServer(httpx.Response(429))
    with pytest.raises(RateLimitError):
        make_client(once, server).search("q")
    assert len(server.requests) == 1


def test_non_json_200_is_api_error(cfg):
    server = FakeServer(httpx.Response(200, text="<html>"))
    with pytest.raises(ApiError, match="non-JSON"):
        make_client(cfg, server).search("q")


def test_api_key_from_env(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    assert api_key_from_env() is None
    monkeypatch.setenv("OPENALEX_API_KEY", "  ")
    assert api_key_from_env() is None
    monkeypatch.setenv("OPENALEX_API_KEY", " abc ")
    assert api_key_from_env() == "abc"
