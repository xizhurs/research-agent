"""Numbers in claims against the cited text (evaluation/numbers.py; pure)."""

import pytest

from research_agent.evaluation import extract_numbers, unsupported_numbers, value_found


def values(text):
    return [n.value for n in extract_numbers(text)]


def test_dates_ranges_thousands_signs_percent_and_ordinals():
    text = "On 2021-07-20: 79.0 mm, rank 1 of 27,759 days, 1991-2020, −41.5 °C, -0.7, 95%, 4th"
    assert values(text) == [2021, 7, 20, 79.0, 1, 27759, 1991, 2020, -41.5, -0.7, 95, 4]
    assert [n.decimals for n in extract_numbers("79.0 and 79 and 0.25")] == [1, 0, 2]


def test_ids_and_glued_digits_are_not_numbers():
    assert values("W2741809807 and R1 cite Rx1day from Sentinel-2") == [2]


def test_a_claim_number_matches_within_its_own_rounding():
    source = ["[R1] Rx1day 79.0 mm on 2021-07-20; TXx 41.4 °C; TNn -41.5 °C"]
    assert unsupported_numbers("It rained 79 mm (79.0 mm) on 20 July 2021.", source) == []
    assert unsupported_numbers("About 41 °C, and 41.5 °C below zero.", source) == []
    assert unsupported_numbers("It rained 80.1 mm, then 43 °C.", source) == ["80.1", "43"]
    assert unsupported_numbers("No numbers here.", source) == []


@pytest.mark.parametrize(
    "value, texts, signed, found",
    [
        (79.0, ["79.0 mm"], True, True),
        (79.0, ["79 mm"], True, True),  # the text's own rounding
        (286.2, ["286 mm"], True, True),
        (79.0, ["80 mm"], True, False),
        (-41.5, ["TNn 41.5 °C"], True, False),  # the sign matters on the tool side
        (-41.5, ["41.5 °C below zero"], False, True),
        (-41.5, ["TNn −41.5 °C"], True, True),
        (15.7, ["15.66"], True, True),  # within the 0.05 tolerance
    ],
)
def test_value_found(value, texts, signed, found):
    assert value_found(value, 0.05, texts, signed=signed) is found
