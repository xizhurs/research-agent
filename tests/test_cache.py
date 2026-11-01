import os

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
