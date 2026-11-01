import json

import pytest

from research_agent.agent import AnswerFormatError, parse_answer
from research_agent.agent.answer import build_answer, fallback_answer


def reply(claims, limitations=()):
    return json.dumps({"claims": claims, "limitations": list(limitations)})


def claim(text="X is true.", kind="evidence", sources=("W1",)):
    return {"text": text, "kind": kind, "sources": list(sources)}


def test_parses_claims_references_and_limitations(make_work):
    ev = {"W1": make_work("W1"), "W2": make_work("W2")}
    text = reply(
        [claim("a", "evidence", ["W2"]), claim("b", "interpretation", ["W1", "W2"])],
        ["only abstracts"],
    )
    ans = parse_answer(text, ev)
    assert [c.kind for c in ans.claims] == ["evidence", "interpretation"]
    assert ans.claims[1].sources == ("W1", "W2")
    assert [w.id for w in ans.references] == ["W2", "W1"]  # order of first citation
    assert ans.limitations == ("only abstracts",)


def test_references_only_include_retrieved_works(make_work):
    ans = parse_answer(reply([claim(sources=["W1", "W999"])]), {"W1": make_work("W1")})
    assert [w.id for w in ans.references] == ["W1"]
    assert ans.claims[0].sources == ("W1", "W999")  # unknown IDs are kept as given (M4 rejects)


def test_unreferenced_retrieved_works_are_not_references(make_work):
    ev = {"W1": make_work("W1"), "W2": make_work("W2")}
    assert [w.id for w in parse_answer(reply([claim(sources=["W1"])]), ev).references] == ["W1"]


def test_openalex_urls_are_normalised_to_short_ids(make_work):
    ans = parse_answer(reply([claim(sources=["https://openalex.org/W1"])]), {"W1": make_work("W1")})
    assert ans.claims[0].sources == ("W1",)
    assert len(ans.references) == 1


def test_accepts_fenced_json_and_surrounding_prose():
    body = reply([claim()])
    assert parse_answer(f"```json\n{body}\n```", {}).claims
    assert parse_answer(f"Here you go:\n{body}\nThanks!", {}).claims


def test_sources_and_limitations_are_optional():
    text = json.dumps({"claims": [{"text": "t", "kind": "interpretation"}]})
    ans = parse_answer(text, {})
    assert ans.claims[0].sources == ()
    assert ans.limitations == ()


def test_empty_claims_allowed_when_evidence_is_missing():
    ans = parse_answer(reply([], ["no relevant works found"]), {})
    assert ans.claims == ()
    assert ans.limitations == ("no relevant works found",)


def test_code_notes_are_appended_and_duplicates_removed():
    ans = parse_answer(reply([claim()], ["a", "b"]), {}, notes=["b", "budget hit"])
    assert ans.limitations == ("a", "b", "budget hit")


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        "[]",
        '{"claims": "nope"}',
        '{"limitations": []}',
        reply([{"kind": "evidence"}]),
        reply([claim(kind="fact")]),
        reply([claim(text="  ")]),
        reply(["a string"]),
        json.dumps({"claims": [{"text": "t", "kind": "evidence", "sources": "W1"}]}),
        json.dumps({"claims": [{"text": "t", "kind": "evidence", "sources": [1]}]}),
        json.dumps({"claims": [], "limitations": "text"}),
    ],
)
def test_malformed_replies_raise(bad):
    with pytest.raises(AnswerFormatError):
        parse_answer(bad, {})


def test_fallback_keeps_raw_text_as_unsourced_interpretation():
    ans = fallback_answer("  plain prose answer ", "reply is not valid JSON", notes=["note"])
    assert len(ans.claims) == 1
    assert ans.claims[0].text == "plain prose answer"
    assert ans.claims[0].kind == "interpretation"
    assert ans.claims[0].sources == ()
    assert ans.references == ()
    assert "structured format" in ans.limitations[0]
    assert ans.limitations[-1] == "note"


def test_build_answer_falls_back_on_bad_format_and_on_none():
    bad = build_answer("just words", {}, ["n"])
    assert bad.claims[0].text == "just words"
    assert "n" in bad.limitations
    empty = build_answer(None, {}, [])
    assert "no answer text" in empty.claims[0].text
