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
max_question_chars = 2000
max_rainfall_calls = 3

[openalex]
base_url = "https://api.openalex.org"
timeout_seconds = 20
max_retries = 3
backoff_base_seconds = 1.0
max_backoff_seconds = 30.0
cache_dir = ".cache/openalex"
cache_ttl_days = 30

[llm]
model = "gpt-6-luna"
reasoning_effort = "none"
timeout_seconds = 60
max_retries = 2

[reporting]
thin_evidence_works = 3
stale_after_years = 3

[rainfall]
geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
archive_url = "https://archive-api.open-meteo.com/v1/archive"
model = "era5"
timeout_seconds = 30
max_retries = 3
backoff_base_seconds = 1.0
max_backoff_seconds = 30.0
cache_dir = ".cache/rainfall"
cache_ttl_days = 90
earliest_year = 1940
default_start_year = 1950
baseline_start_year = 1991
baseline_end_year = 2020
min_years = 20
min_year_coverage = 0.9
top_events = 5
geocode_candidates = 10
dominance_ratio = 10.0
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


def test_base_config_llm_section():
    llm = load_config(BASE).llm
    assert llm.model == "gpt-6-luna"
    assert llm.reasoning_effort == "none"
    assert llm.max_retries == 2


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ('model = "gpt-6-luna"', 'model = ""', "llm.model"),
        ('reasoning_effort = "none"', "reasoning_effort = 3", "llm.reasoning_effort"),
        ("timeout_seconds = 60", "timeout_seconds = -1", "llm.timeout_seconds"),
        ("max_retries = 2", "max_retries = 1.5", "llm.max_retries"),
    ],
)
def test_invalid_llm_values_rejected(tmp_path, old, new, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, VALID.replace(old, new)))


def test_base_config_reporting_section():
    rep = load_config(BASE).reporting
    assert (rep.thin_evidence_works, rep.stale_after_years) == (3, 3)


@pytest.mark.parametrize("bad", ["0", "-1", "2.5", '"3"', "true"])
def test_invalid_reporting_values_rejected(tmp_path, bad):
    text = VALID.replace("thin_evidence_works = 3", f"thin_evidence_works = {bad}")
    with pytest.raises(ConfigError, match="reporting.thin_evidence_works"):
        load_config(write(tmp_path, text))


def test_missing_reporting_table(tmp_path):
    text = VALID.split("[reporting]")[0]
    with pytest.raises(ConfigError, match=r"missing \[reporting\]"):
        load_config(write(tmp_path, text))


def test_missing_llm_table(tmp_path):
    text = VALID.split("[llm]")[0]
    with pytest.raises(ConfigError, match=r"missing \[llm\]"):
        load_config(write(tmp_path, text))


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
    text = VALID.replace("[openalex]\n", "[openalex]\ntypo = 1\n")
    with pytest.raises(ConfigError, match="unknown openalex keys"):
        load_config(write(tmp_path, text))


def test_unknown_llm_key(tmp_path):
    text = VALID.replace("[llm]\n", "[llm]\ntypo = 1\n")
    with pytest.raises(ConfigError, match="unknown llm keys"):
        load_config(write(tmp_path, text))


def test_unknown_reporting_key(tmp_path):
    text = VALID.replace("[reporting]\n", "[reporting]\ntypo = 1\n")
    with pytest.raises(ConfigError, match="unknown reporting keys"):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("bad", ["0", "-1", "2.5", '"5"', "true"])
def test_non_positive_or_non_int_rejected(tmp_path, bad):
    text = VALID.replace("max_iterations = 5", f"max_iterations = {bad}")
    with pytest.raises(ConfigError, match="max_iterations"):
        load_config(write(tmp_path, text))




def test_base_config_rainfall_section():
    cfg = load_config(BASE)
    assert cfg.rainfall.model == "era5"
    assert cfg.rainfall.cache_dir == Path(".cache/rainfall")
    assert (cfg.rainfall.baseline_start_year, cfg.rainfall.baseline_end_year) == (1991, 2020)
    assert cfg.limits.max_rainfall_calls == 3


@pytest.mark.parametrize(
    ("old", "new", "name"),
    [
        ('model = "era5"', 'model = ""', "rainfall.model"),
        ("min_year_coverage = 0.9", "min_year_coverage = 0", "min_year_coverage"),
        ("min_year_coverage = 0.9", "min_year_coverage = 1.5", "min_year_coverage"),
        ("min_years = 20", "min_years = 0", "min_years"),
        ("top_events = 5", "top_events = true", "top_events"),
        ("dominance_ratio = 10.0", "dominance_ratio = -1", "dominance_ratio"),
        ("default_start_year = 1950", "default_start_year = 1900", "default_start_year"),
        ("baseline_start_year = 1991", "baseline_start_year = 2021", "baseline_start_year"),
        ('archive_url = "https', 'archive_url = "ftp', "archive_url"),
        ("cache_ttl_days = 90", "cache_ttl_days = 0", "rainfall.cache_ttl_days"),
    ],
)
def test_invalid_rainfall_values_rejected(tmp_path, old, new, name):
    text = VALID.replace(old, new)
    assert text != VALID
    with pytest.raises(ConfigError, match=name):
        load_config(write(tmp_path, text))


def test_missing_rainfall_table(tmp_path):
    text = VALID[: VALID.index("[rainfall]")]
    with pytest.raises(ConfigError, match=r"missing \[rainfall\]"):
        load_config(write(tmp_path, text))


def test_unknown_rainfall_key(tmp_path):
    with pytest.raises(ConfigError, match="unknown rainfall keys"):
        load_config(write(tmp_path, VALID + "bogus = 1\n"))


def test_missing_max_rainfall_calls(tmp_path):
    with pytest.raises(ConfigError, match="max_rainfall_calls"):
        load_config(write(tmp_path, VALID.replace("max_rainfall_calls = 3\n", "")))
