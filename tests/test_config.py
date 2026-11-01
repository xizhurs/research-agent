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
        load_config(write(tmp_path, VALID + "typo = 1\n"))


@pytest.mark.parametrize("bad", ["0", "-1", "2.5", '"5"', "true"])
def test_non_positive_or_non_int_rejected(tmp_path, bad):
    text = VALID.replace("max_iterations = 5", f"max_iterations = {bad}")
    with pytest.raises(ConfigError, match="max_iterations"):
        load_config(write(tmp_path, text))
