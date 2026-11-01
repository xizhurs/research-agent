from dataclasses import replace
from pathlib import Path

import pytest

from research_agent.agent import AgentResult, Answer, Claim, RunStats
from research_agent.evaluation import (
    EvalCase,
    EvalError,
    check_result,
    load_cases,
    run_case,
    summarize,
)
from research_agent.llm import LLMApiError, LLMAuthError
from research_agent.retrieval import AuthError, RateLimitError

ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = ROOT / "evals" / "cases.toml"


def write(tmp_path, text):
    p = tmp_path / "cases.toml"
    p.write_text(text, encoding="utf-8")
    return p


# --- loading ---------------------------------------------------------------------------------


def test_shipped_cases_file_is_valid_and_well_formed():
    cases = load_cases(CASES_FILE)
    assert len(cases) >= 10
    assert len({c.id for c in cases}) == len(cases)
    on_topic = [c for c in cases if c.min_searches > 0]
    assert len(on_topic) >= 8
    assert any(c.max_evidence_claims == 0 for c in cases)  # a guard case exists
    assert any(c.forbidden_phrases for c in cases)  # and a leak-check case


def test_load_full_case(tmp_path):
    path = write(
        tmp_path,
        """
[[case]]
id = "a"
question = "  Q?  "
min_searches = 1
min_references = 2
max_evidence_claims = 0
keywords_any = ["LSTM", "cnn"]
forbidden_phrases = ["Secret"]
""",
    )
    (case,) = load_cases(path)
    assert case == EvalCase("a", "Q?", 1, 2, 0, ("lstm", "cnn"), ("secret",))


def test_defaults_for_a_minimal_case(tmp_path):
    (case,) = load_cases(write(tmp_path, '[[case]]\nid = "a"\nquestion = "Q"\n'))
    assert (case.min_searches, case.min_references) == (0, 0)
    assert case.max_evidence_claims is None
    assert case.keywords_any == () and case.forbidden_phrases == ()


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "at least one"),
        ("[[case]]\nquestion = 'Q'\n", "'id' is required"),
        ("[[case]]\nid = 'a'\n", "'question' is required"),
        ("[[case]]\nid = 'a'\nquestion = 'Q'\ntypo = 1\n", "unknown keys"),
        ("[[case]]\nid = 'a'\nquestion = 'Q'\nmin_searches = -1\n", "min_searches"),
        ("[[case]]\nid = 'a'\nquestion = 'Q'\nmin_references = 'x'\n", "min_references"),
        ("[[case]]\nid = 'a'\nquestion = 'Q'\nkeywords_any = 'x'\n", "keywords_any"),
        ("[[case]]\nid = 'a'\nquestion = 'Q'\nforbidden_phrases = ['']\n", "forbidden_phrases"),
        ("[[case]]\nid = 'a'\nquestion = 'Q'\n[[case]]\nid = 'a'\nquestion = 'R'\n", "duplicate"),
        ("[[case\n", "invalid TOML"),
    ],
)
def test_invalid_case_files_rejected(tmp_path, text, match):
    with pytest.raises(EvalError, match=match):
        load_cases(write(tmp_path, text))


def test_missing_case_file(tmp_path):
    with pytest.raises(EvalError, match="not found"):
        load_cases(tmp_path / "nope.toml")


# --- checking --------------------------------------------------------------------------------


def make_agent_result(
    make_work,
    *,
    claims=None,
    references=None,
    limitations=("Evidence is limited to abstracts.",),
    structured=True,
    stats=None,
):
    refs = references if references is not None else (make_work("W1"), make_work("W2"))
    claims = claims if claims is not None else (
        Claim("LSTM models are used for crops.", "evidence", ("W1", "W2")),
        Claim("This likely generalises.", "interpretation", ()),
    )
    answer = Answer(tuple(claims), tuple(refs), tuple(limitations), structured=structured)
    return AgentResult(answer, stats or RunStats(3, 2, 4000, 400, 6))


CASE = EvalCase("c", "Q", min_searches=1, min_references=2, keywords_any=("lstm",))


def test_good_answer_passes(make_work, limits):
    assert check_result(CASE, make_agent_result(make_work), limits) == []


def test_unstructured_answer_fails(make_work, limits):
    r = make_agent_result(make_work, structured=False)
    assert any("structured format" in f for f in check_result(CASE, r, limits))


def test_missing_limitations_fail(make_work, limits):
    r = make_agent_result(make_work, limitations=())
    assert any("no limitations" in f for f in check_result(CASE, r, limits))


def test_evidence_claim_without_citation_fails(make_work, limits):
    r = make_agent_result(make_work, claims=(Claim("Uncited fact.", "evidence", ()),))
    assert any("no citation" in f for f in check_result(CASE, r, limits))


def test_claim_citing_a_work_not_in_references_fails(make_work, limits):
    r = make_agent_result(make_work, claims=(Claim("x lstm", "evidence", ("W9",)),))
    assert any("W9" in f and "not in the references" in f for f in check_result(CASE, r, limits))


def test_budget_violations_fail(make_work, limits):
    r = make_agent_result(make_work, stats=RunStats(limits.max_iterations + 1, 2, 10, 10, 6))
    assert any("model turns" in f for f in check_result(CASE, r, limits))
    r = make_agent_result(make_work, stats=RunStats(2, limits.max_search_calls + 1, 10, 10, 6))
    assert any("searches" in f for f in check_result(CASE, r, limits))


def test_token_overshoot_must_be_reported(make_work, limits):
    over = RunStats(2, 1, limits.max_total_tokens, 500, 6)
    r = make_agent_result(make_work, stats=over)
    assert any("without reporting it" in f for f in check_result(CASE, r, limits))
    ok = make_agent_result(
        make_work, stats=over, limitations=("Token budget exceeded: 20500 tokens used.",)
    )
    assert check_result(CASE, ok, limits) == []


def test_case_specific_expectations(make_work, limits):
    few = make_agent_result(make_work, stats=RunStats(2, 0, 10, 10, 0), references=())
    failures = check_result(CASE, few, limits)
    assert any("at least 1 search" in f for f in failures)
    assert any("at least 2 reference" in f for f in failures)


def test_keyword_expectation_is_case_insensitive_and_any_of(make_work, limits):
    hit = replace(CASE, keywords_any=("transformer", "lstm"))
    assert check_result(hit, make_agent_result(make_work), limits) == []
    miss = replace(CASE, keywords_any=("transformer",))
    failures = check_result(miss, make_agent_result(make_work), limits)
    assert any("expected keywords" in f for f in failures)


def test_max_evidence_claims_and_forbidden_phrases(make_work, limits):
    guard = EvalCase(
        "g", "Q", max_evidence_claims=0, forbidden_phrases=("lstm models",),
    )
    failures = check_result(guard, make_agent_result(make_work), limits)
    assert any("at most 0 evidence" in f for f in failures)
    assert any("forbidden phrase" in f for f in failures)


# --- running ---------------------------------------------------------------------------------


class FakeAgent:
    def __init__(self, outcome):
        self.outcome = outcome
        self.questions = []

    def run(self, question):
        self.questions.append(question)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_run_case_passes_and_reports(make_work, limits):
    agent = FakeAgent(make_agent_result(make_work))
    report = run_case(agent, CASE, limits)
    assert report.passed and report.failures == ()
    assert agent.questions == ["Q"]
    assert report.rejected_citations == 0


def test_run_case_records_rejected_citations(make_work, limits):
    base = make_agent_result(make_work)
    answer = replace(base.answer, rejected_citations=("W9", "W8"))
    report = run_case(FakeAgent(AgentResult(answer, base.stats)), CASE, limits)
    assert report.rejected_citations == 2


def test_provider_failure_fails_only_that_case(limits):
    report = run_case(FakeAgent(LLMApiError("500 from provider")), CASE, limits)
    assert not report.passed
    assert report.result is None
    assert "500 from provider" in report.failures[0]


def test_search_failure_fails_only_that_case(limits):
    report = run_case(FakeAgent(RateLimitError("daily budget")), CASE, limits)
    assert not report.passed and "daily budget" in report.error


@pytest.mark.parametrize("exc", [LLMAuthError("bad key"), AuthError("bad key")])
def test_auth_failures_abort_instead_of_failing_every_case(limits, exc):
    with pytest.raises(type(exc)):
        run_case(FakeAgent(exc), CASE, limits)


def test_summary_totals(make_work, limits):
    good = run_case(FakeAgent(make_agent_result(make_work)), CASE, limits)
    bad = run_case(FakeAgent(make_agent_result(make_work, structured=False)), CASE, limits)
    errored = run_case(FakeAgent(LLMApiError("x")), CASE, limits)
    s = summarize([good, bad, errored])
    assert (s.total, s.passed) == (3, 1)
    assert s.total_tokens == 2 * 4400  # the errored case has no usage to count
    assert s.searches == 4
    assert s.unstructured == 1
