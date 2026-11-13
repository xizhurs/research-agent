"""Numbers in text, for checking answers against the text they cite (pure, offline).

A number in a claim is supported when the cited text holds the same number to the precision the
claim wrote it with ("79 mm" matches 79.0, "80.1 mm" does not). Values are compared by absolute
value, so a sign error is not caught, but "41.5 °C below zero" matches -41.5. Numbers the model
derived itself ("about 3 times", "top 5%") are reported as unsupported: the metric is reported,
never a failure.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# A sign counts only at the start of a token, so "2021-07-20" is 2021, 07, 20 and "1991-2020" is
# 1991, 2020. Digits glued to letters ("W123", "R1", "Rx1day", "Sentinel2") are not numbers.
_NUMBER = re.compile(r"(?<![\w.,])([-−+]?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")


@dataclass(frozen=True)
class Number:
    value: float
    decimals: int  # digits after the decimal point, as written
    text: str  # as it appears in the text


def extract_numbers(text: str) -> list[Number]:
    out = []
    for match in _NUMBER.finditer(text):
        sign, whole, fraction = match.groups()
        value = float(whole.replace(",", "") + (f".{fraction}" if fraction else ""))
        if sign in ("-", "−"):
            value = -value
        out.append(Number(value, len(fraction) if fraction else 0, match.group(0)))
    return out


def _half_unit(decimals: int) -> float:
    """Half a unit of the last digit written (a rounded value can be this far off)."""
    return 0.5 * 10.0**-decimals + 1e-9


def unsupported_numbers(claim_text: str, source_texts: Iterable[str]) -> list[str]:
    """The numbers in ``claim_text`` that no source text holds (by absolute value, within the
    claim's own rounding)."""
    available = [abs(n.value) for text in source_texts for n in extract_numbers(text)]
    return [
        n.text
        for n in extract_numbers(claim_text)
        if not any(abs(abs(n.value) - v) <= _half_unit(n.decimals) for v in available)
    ]


def value_found(
    value: float, tolerance: float, texts: Iterable[str], *, signed: bool = True
) -> bool:
    """Whether any number in ``texts`` equals ``value`` within ``tolerance`` or within that
    number's own rounding ("286 mm" matches 286.2). ``signed=False`` compares absolute values."""
    target = value if signed else abs(value)
    for text in texts:
        for n in extract_numbers(text):
            seen = n.value if signed else abs(n.value)
            if abs(seen - target) <= max(tolerance, _half_unit(n.decimals)):
                return True
    return False
