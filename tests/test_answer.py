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
    assert ans.claims[0].sources == ("W1",)  # the unretrieved ID is removed from the claim


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


def test_code_notes_are_appended_and_duplicates_removed(make_work):
    ans = parse_answer(
        reply([claim()], ["a", "b"]), {"W1": make_work("W1")}, notes=["b", "budget hit"]
    )
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


# --- citation validation (M4) ---------------------------------------------------------------


def test_citation_to_unretrieved_work_is_removed_and_reported(make_work):
    ev = {"W1": make_work("W1")}
    ans = parse_answer(reply([claim("a", "evidence", ["W1", "W404"])]), ev)
    assert ans.claims[0].sources == ("W1",)
    assert ans.claims[0].kind == "evidence"  # still supported by the valid citation
    assert ans.rejected_citations == ("W404",)
    assert any("W404" in x and "not among the retrieved" in x for x in ans.limitations)


def test_evidence_claim_with_only_unretrieved_citations_is_downgraded(make_work):
    ans = parse_answer(reply([claim("made up", "evidence", ["W404"])]), {"W1": make_work("W1")})
    assert ans.claims[0].kind == "interpretation"
    assert ans.claims[0].sources == ()
    assert ans.references == ()
    assert any("labelled as evidence cited no retrieved work" in x for x in ans.limitations)


def test_evidence_claim_with_no_citation_at_all_is_downgraded():
    ans = parse_answer(reply([claim("uncited", "evidence", [])]), {})
    assert ans.claims[0].kind == "interpretation"
    assert ans.rejected_citations == ()  # nothing was cited, so nothing was rejected
    assert any("cited no retrieved work" in x for x in ans.limitations)


def test_interpretation_with_unretrieved_citation_keeps_claim_but_drops_citation():
    ans = parse_answer(reply([claim("my view", "interpretation", ["W404"])]), {})
    assert ans.claims[0].kind == "interpretation"
    assert ans.claims[0].sources == ()
    assert ans.rejected_citations == ("W404",)
    assert not any("labelled as evidence" in x for x in ans.limitations)


def test_valid_answer_produces_no_citation_notes(make_work):
    ans = parse_answer(reply([claim()]), {"W1": make_work("W1")})
    assert ans.rejected_citations == ()
    assert ans.limitations == ()


def test_rejected_ids_are_listed_once_and_duplicate_sources_collapse(make_work):
    ev = {"W1": make_work("W1")}
    text = reply(
        [claim("a", "evidence", ["W1", "W1", "W9"]), claim("b", "interpretation", ["W9", "W8"])]
    )
    ans = parse_answer(text, ev)
    assert ans.rejected_citations == ("W9", "W8")
    assert ans.claims[0].sources == ("W1",)


def test_every_surviving_citation_resolves_to_a_reference(make_work):
    ev = {f"W{i}": make_work(f"W{i}") for i in (1, 2, 3)}
    text = reply(
        [
            claim("a", "evidence", ["W1", "W7"]),
            claim("b", "evidence", ["W9"]),
            claim("c", "interpretation", ["W2", "W8", "W3"]),
        ]
    )
    ans = parse_answer(text, ev)
    reference_ids = {w.id for w in ans.references}
    for c in ans.claims:
        assert set(c.sources) <= reference_ids
        if c.kind == "evidence":
            assert c.sources  # evidence claims always keep at least one valid citation
    assert reference_ids == {"W1", "W2", "W3"}


def test_structured_flag():
    assert parse_answer(reply([claim("t", "interpretation", [])]), {}).structured is True
    assert fallback_answer("prose", "reason").structured is False
    assert build_answer("not json", {}, []).structured is False
