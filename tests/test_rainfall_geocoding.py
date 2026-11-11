"""Geocoding: parsing recorded responses and the disambiguation rules. No network."""

import json
import logging
from pathlib import Path

import pytest

from research_agent.rainfall import (
    AmbiguousLocationError,
    LocationNotFoundError,
    RainfallApiError,
)
from research_agent.rainfall.geocoding import (
    build_geocode_params,
    distance_km,
    parse_candidates,
    resolve_location,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def resolve(candidates, city="X", region=None, ratio=10.0, same_place_km=25.0):
    return resolve_location(
        candidates, city=city, region=region, dominance_ratio=ratio, same_place_km=same_place_km
    )


def test_params_include_country_code_only_when_given():
    base = build_geocode_params("Paris", None, 10)
    assert base == {"name": "Paris", "count": "10", "language": "en", "format": "json"}
    assert build_geocode_params("Paris", "FR", 10)["countryCode"] == "FR"


def test_parse_recorded_response():
    found = parse_candidates(load("openmeteo_geocode_springfield.json"))
    assert len(found) == 4
    first = found[0]
    assert (first.name, first.admin1, first.country_code) == ("Springfield", "Missouri", "US")
    assert first.population == 170188 and first.latitude == pytest.approx(37.21533)
    assert first.label.startswith("Springfield, Missouri")


def test_response_without_results_means_no_match():
    assert parse_candidates({"generationtime_ms": 0.1}) == ()


@pytest.mark.parametrize("payload", [[], "text", {"results": "oops"}])
def test_unexpected_envelope_is_an_error(payload):
    with pytest.raises(RainfallApiError):
        parse_candidates(payload)


def test_unparseable_records_are_logged_not_silently_dropped(caplog):
    good = load("openmeteo_geocode_paris.json")["results"][0]
    with caplog.at_level(logging.WARNING):
        found = parse_candidates({"results": [good, {"name": "No coordinates", "id": 5}, "junk"]})
    assert len(found) == 1
    assert caplog.text.count("skipped unparseable geocoding record") == 2


def test_dominant_candidate_is_chosen_and_alternatives_reported():
    found = parse_candidates(load("openmeteo_geocode_paris.json"))
    chosen, others = resolve(found, city="Paris")
    assert (chosen.name, chosen.country_code) == ("Paris", "FR")
    assert len(others) == len(found) - 1


def test_ambiguous_city_returns_candidates_and_never_guesses():
    found = parse_candidates(load("openmeteo_geocode_springfield.json"))
    with pytest.raises(AmbiguousLocationError) as err:
        resolve(found, city="Springfield")
    message = str(err.value)
    assert "Springfield, Missouri" in message and "Springfield, Illinois" in message
    assert "Do not guess" in message
    assert len(err.value.candidates) == 4
    assert err.value.candidates[0].population >= err.value.candidates[-1].population


def test_region_filter_is_exact_and_case_insensitive(make_location):
    a = make_location(id=1, name="Springfield", admin1="Illinois", population=114394)
    b = make_location(id=2, name="Springfield", admin1="Missouri", population=170188)
    chosen, others = resolve([a, b], city="Springfield", region="illinois")
    assert chosen.id == 1 and others == ()


def test_region_with_no_match_is_not_found(make_location):
    a = make_location(admin1="Illinois")
    with pytest.raises(LocationNotFoundError, match="Ohio"):
        resolve([a], city="Springfield", region="Ohio")


def test_no_candidates_is_not_found():
    with pytest.raises(LocationNotFoundError, match="Nowhere"):
        resolve([], city="Nowhere")


def test_single_candidate_is_accepted_even_without_population(make_location):
    only = make_location(population=None)
    assert resolve([only]) == (only, ())


def test_unknown_population_cannot_win_a_dominance_contest(make_location):
    a = make_location(id=1, population=None)
    b = make_location(id=2, population=None, latitude=40.0)  # far apart: not merged
    with pytest.raises(AmbiguousLocationError):
        resolve([a, b])


def test_dominance_needs_the_configured_ratio(make_location):
    big = make_location(id=1, population=900_000)
    small = make_location(id=2, population=100_000, latitude=40.0)  # far apart: not merged
    with pytest.raises(AmbiguousLocationError):
        resolve([big, small], ratio=10.0)  # only 9x
    assert resolve([big, small], ratio=9.0)[0].id == 1


# --- nearby candidates (a city's districts) are the same place ------------------------------------


def test_distance_km_matches_a_known_pair(make_location):
    amsterdam = make_location(latitude=52.374, longitude=4.890)
    rotterdam = make_location(latitude=51.923, longitude=4.479)
    assert distance_km(amsterdam, rotterdam) == pytest.approx(57.8, abs=1.0)
    assert distance_km(amsterdam, amsterdam) == 0


def test_amsterdam_districts_merge_into_the_city():
    # Recorded response: Amsterdam (741,636) plus 6 districts within ~10 km (New West 151,677,
    # only 4.9x smaller), a heliport and Amsterdamscheveld (Drenthe, 190 people, ~150 km away).
    found = parse_candidates(load("openmeteo_geocode_amsterdam.json"))
    chosen, others = resolve(found, city="Amsterdam")
    assert chosen.id == 2759794 and chosen.admin1 == "North Holland"
    assert len(others) == len(found) - 1 == 9
    with pytest.raises(AmbiguousLocationError):  # the old rule (no merging) could not decide
        resolve(found, city="Amsterdam", same_place_km=0.0)


def test_a_nearby_place_without_population_is_merged(make_location):
    city = make_location(id=1, population=500_000)
    heliport = make_location(id=2, population=None, latitude=city.latitude + 0.05)
    assert resolve([heliport, city]) == (city, (heliport,))


def test_far_apart_places_of_similar_size_stay_ambiguous(make_location):
    a = make_location(id=1, population=170_000)
    b = make_location(id=2, population=114_000, latitude=a.latitude + 1.0)  # ~111 km
    with pytest.raises(AmbiguousLocationError) as err:
        resolve([a, b])
    assert len(err.value.candidates) == 2


def test_merged_districts_are_not_listed_as_ambiguous_options(make_location):
    a = make_location(id=1, name="A", population=300_000)
    a_district = make_location(id=2, name="A-North", population=200_000, latitude=a.latitude + 0.05)
    b = make_location(id=3, name="B", population=250_000, latitude=a.latitude + 2.0)
    with pytest.raises(AmbiguousLocationError) as err:
        resolve([a, a_district, b])
    assert [c.id for c in err.value.candidates] == [1, 3]
