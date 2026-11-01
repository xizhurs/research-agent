from pathlib import Path

import pytest

from research_agent.config import ConfigError, load_config

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"

VALID = """
[limits]
max_iterations = 5
max_search_calls = 3
results_per_search = 8
abstract_max_chars = 1200
max_context_tokens = 6000
max_output_tokens = 800
max_total_tokens = 20000

[openalex]
base_url = "https://api.openalex.org"
timeout_seconds = 20
max_retries = 3
backoff_base_seconds = 1.0
max_backoff_seconds = 30.0
cache_dir = ".cache/openalex"
cache_ttl_days = 30
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_base_config_loads_documented_defaults():
    limits = load_config(BASE).limits
    assert limits.max_iterations == 5
    assert limits.max_search_calls == 3
    assert limits.results_per_search == 8
    assert limits.max_total_tokens == 20000


def test_base_config_openalex_section():
    oa = load_config(BASE).openalex
    assert oa.base_url == "https://api.openalex.org"
    assert oa.max_retries == 3
    assert oa.cache_dir == Path(".cache/openalex")


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ('base_url = "https://api.openalex.org"', 'base_url = "ftp://x"', "base_url"),
        ("timeout_seconds = 20", "timeout_seconds = 0", "timeout_seconds"),
        ("max_retries = 3", "max_retries = -1", "max_retries"),
        ("max_retries = 3", "max_retries = true", "max_retries"),
        ("cache_ttl_days = 30", 'cache_ttl_days = "30"', "cache_ttl_days"),
    ],
)
def test_invalid_openalex_values_rejected(tmp_path, old, new, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, VALID.replace(old, new)))


def test_zero_retries_allowed(tmp_path):
    cfg = load_config(write(tmp_path, VALID.replace("max_retries = 3", "max_retries = 0")))
    assert cfg.openalex.max_retries == 0


def test_missing_openalex_table(tmp_path):
    text = VALID.split("[openalex]")[0]
    with pytest.raises(ConfigError, match=r"missing \[openalex\]"):
        load_config(write(tmp_path, text))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_malformed_toml(tmp_path):
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(write(tmp_path, "[limits\n"))


def test_missing_limits_table(tmp_path):
    with pytest.raises(ConfigError, match=r"missing \[limits\]"):
        load_config(write(tmp_path, "x = 1\n"))


def test_missing_key(tmp_path):
    text = VALID.replace("max_iterations = 5\n", "")
    with pytest.raises(ConfigError, match="missing limits keys"):
        load_config(write(tmp_path, text))


def test_unknown_key(tmp_path):
    with pytest.raises(ConfigError, match="unknown limits keys"):
        load_config(write(tmp_path, VALID.replace("[limits]\n", "[limits]\ntypo = 1\n")))


def test_unknown_openalex_key(tmp_path):
    with pytest.raises(ConfigError, match="unknown openalex keys"):
        load_config(write(tmp_path, VALID + "typo = 1\n"))


@pytest.mark.parametrize("bad", ["0", "-1", "2.5", '"5"', "true"])
def test_non_positive_or_non_int_rejected(tmp_path, bad):
    text = VALID.replace("max_iterations = 5", f"max_iterations = {bad}")
    with pytest.raises(ConfigError, match="max_iterations"):
        load_config(write(tmp_path, text))
