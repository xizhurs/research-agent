import os
from pathlib import Path

import pytest

from research_agent.retrieval.cache import ResponseCache, cache_key


def test_key_is_deterministic_and_order_independent():
    a = cache_key("https://x/works", {"a": "1", "b": "2"})
    b = cache_key("https://x/works", {"b": "2", "a": "1"})
    assert a == b


def test_key_changes_with_params_and_url():
    base = cache_key("https://x/works", {"a": "1"})
    assert base != cache_key("https://x/works", {"a": "2"})
    assert base != cache_key("https://y/works", {"a": "1"})


def test_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path / "c", ttl_days=1)
    assert cache.get("k") is None
    cache.put("k", {"a": [1, 2], "t": "é"})
    assert cache.get("k") == {"a": [1, 2], "t": "é"}
    assert not list((tmp_path / "c").glob("*.tmp"))


def test_expired_entry_is_a_miss(tmp_path):
    clock = [1_000_000.0]
    cache = ResponseCache(tmp_path, ttl_days=1, now=lambda: clock[0])
    cache.put("k", {"v": 1})
    os.utime(tmp_path / "k.json", (clock[0], clock[0]))
    clock[0] += 86400 - 1
    assert cache.get("k") == {"v": 1}
    clock[0] += 2
    assert cache.get("k") is None


def test_corrupt_entry_is_a_miss_and_logged(tmp_path, caplog):
    cache = ResponseCache(tmp_path, ttl_days=1)
    (tmp_path / "k.json").write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert cache.get("k") is None
    assert "unreadable cache entry" in caplog.text


@pytest.mark.parametrize("error", [NotADirectoryError, PermissionError, OSError])
def test_unexpected_os_errors_on_lookup_are_a_logged_miss(tmp_path, monkeypatch, caplog, error):
    """Linux raises NotADirectoryError (not FileNotFoundError) when a parent path is a file.

    Simulated here so the behaviour is checked on every OS, not only on Linux CI.
    """
    cache = ResponseCache(tmp_path, ttl_days=1)

    def failing_stat(self, *args, **kwargs):
        raise error(20, "simulated")

    monkeypatch.setattr(Path, "stat", failing_stat)
    with caplog.at_level("WARNING"):
        assert cache.get("k") is None
    assert "cannot read cache entry" in caplog.text


def test_plain_missing_entry_is_a_silent_miss(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        assert ResponseCache(tmp_path, ttl_days=1).get("absent") is None
    assert caplog.text == ""
