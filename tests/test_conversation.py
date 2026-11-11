"""Follow-up questions: task state, context building, evidence reuse, sessions. No network.

A scripted LLM plays the model across turns; a fake analyzer and search stand in for the tools.
Which relation a real model picks is judged live, not here.
"""

import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from datetime import date

import pytest
from test_agent_rainfall import FakeRainfall, FakeSearch
from typer.testing import CliRunner

from research_agent.agent import AgentResult, ResearchAgent, RunStats
from research_agent.agent.answer import Answer, Claim
from research_agent.agent.prompt import CONVERSATION_RULES, build_instructions
from research_agent.cli import app
from research_agent.config import ReportingConfig
from research_agent.conversation import (
    ConversationService,
    EvidenceItem,
    InMemorySessionStore,
    Session,
    TaskState,
    ToolRef,
    TurnRecord,
    build_context,
)
from research_agent.llm import (
    LLMAuthError,
    LLMRateLimitError,
    StubLLMClient,
    UserMessage,
    text_response,
    tool_call_response,
)
from research_agent.rainfall import AmbiguousLocationError

RAIN, SEARCH, RECALL = "analyze_rainfall_extremes", "search_literature", "recall_evidence"
REPORTING = ReportingConfig(thin_evidence_works=3, stale_after_years=3)
TODAY = date(2026, 11, 2)


def reply(claims=(), relation=None):
    data = {"claims": list(claims), "limitations": []}
    if relation:
        data["relation"] = relation
    return text_response(json.dumps(data))


def evidence(text, *sources):
    return {"text": text, "kind": "evidence", "sources": list(sources)}


@pytest.fixture
def places(make_location):
    return {
        "amsterdam": make_location(
            id=2759794, name="Amsterdam", admin1="North Holland", country="Netherlands",
            country_code="NL", latitude=52.37, longitude=4.89,
        ),
        "rotterdam": make_location(
            id=2747891, name="Rotterdam", admin1="South Holland", country="Netherlands",
            country_code="NL", latitude=51.92, longitude=4.48,
        ),
        "springfield_il": make_location(
            id=4250542, name="Springfield", admin1="Illinois", country="United States",
            country_code="US", latitude=39.80, longitude=-89.64,
        ),
        "springfield_mo": make_location(
            id=4409896, name="Springfield", admin1="Missouri", country="United States",
            country_code="US", latitude=37.22, longitude=-93.30,
        ),
    }


@pytest.fixture
def conv(base_config):
    return base_config.conversation


def make_service(llm, limits, conv, *, rainfall=None, search=None, store=None):
    agent = ResearchAgent(
        llm, search or FakeSearch(), limits, REPORTING, rainfall=rainfall, today=lambda: TODAY
    )
    store = store or InMemorySessionStore(conv.max_sessions, conv.session_ttl_minutes * 60)
    service = ConversationService(lambda q, ctx: agent.run(q, ctx), store, conv)
    return service, store


def last_user_text(call):
    return [m for m in call.messages if isinstance(m, UserMessage)][-1].text


# --- follow-up resolution ---------------------------------------------------------------------


def test_a_follow_up_inherits_place_period_and_metric(limits, conv, make_analysis, places):
    rainfall = FakeRainfall(
        make_analysis(location=places["amsterdam"]), make_analysis(location=places["rotterdam"])
    )
    llm = StubLLMClient(
        [
            tool_call_response(("a", RAIN, {"city": "Amsterdam", "start_year": 2015,
                                            "end_year": 2024})),
            reply([evidence("Amsterdam figures.", "R1")]),
            tool_call_response(("b", RAIN, {"city": "Rotterdam", "start_year": 2015,
                                            "end_year": 2024})),
            reply([evidence("Rotterdam figures.", "R2")], relation="update"),
        ]
    )
    service, store = make_service(llm, limits, conv, rainfall=rainfall)

    first = service.ask(None, "How was the rainfall in Amsterdam in the last 10 years?")
    second = service.ask(first.session_id, "What about Rotterdam?")

    assert first.context is None  # a first question is sent exactly as in `ask`
    follow_up = llm.calls[2]
    assert CONVERSATION_RULES in follow_up.instructions
    block = last_user_text(follow_up)
    assert "Amsterdam, North Holland" in block and "Period: 2015 to 2024" in block
    assert "Metrics: wet" in block and block.endswith("Current question: What about Rotterdam?")
    assert follow_up.messages[0] == UserMessage(
        "How was the rainfall in Amsterdam in the last 10 years?"
    )
    assert "[Earlier answer, model-written, not evidence]" in follow_up.messages[1].text

    assert [r.id for r in second.result.answer.data_references] == ["R2"]  # IDs continue
    state = store.get(first.session_id).state
    assert [p.label for p in state.places] == ["Rotterdam, South Holland, Netherlands"]
    assert state.period == (2015, 2024) and state.metrics == ("wet",)
    assert second.turn.relation == "update" and second.turn.relation_source == "model"


def test_a_pending_clarification_is_offered_and_cleared(limits, conv, make_analysis, places):
    ambiguous = AmbiguousLocationError(
        "'Springfield' matches 2 places", [places["springfield_mo"], places["springfield_il"]]
    )
    rainfall = FakeRainfall(ambiguous, make_analysis(location=places["springfield_il"]))
    llm = StubLLMClient(
        [
            tool_call_response(("a", RAIN, {"city": "Springfield"})),
            reply([{"text": "Ambiguous.", "kind": "interpretation", "sources": []}]),
            tool_call_response(("b", RAIN, {"city": "Springfield", "region": "Illinois"})),
            reply([evidence("Illinois figures.", "R1")], relation="continue"),
        ]
    )
    service, store = make_service(llm, limits, conv, rainfall=rainfall)

    first = service.ask(None, "Rainfall extremes in Springfield?")
    pending = store.get(first.session_id).state.pending
    assert pending.city == "Springfield" and len(pending.candidates) == 2
    service.ask(first.session_id, "The one in Illinois.")

    block = last_user_text(llm.calls[2])
    assert "Pending clarification: 'Springfield' matched several places" in block
    assert "Springfield, Illinois" in block
    state = store.get(first.session_id).state
    assert state.pending is None and state.places[0].admin1 == "Illinois"
    assert "was ambiguous" in store.get(first.session_id).turns[0].summary_line


# --- comparing, switching topic -----------------------------------------------------------------


def test_compare_recalls_r1_without_spending_analysis_budget(limits, conv, make_analysis, places):
    rainfall = FakeRainfall(
        make_analysis(location=places["amsterdam"]), make_analysis(location=places["rotterdam"])
    )
    llm = StubLLMClient(
        [
            tool_call_response(("a", RAIN, {"city": "Amsterdam"})),
            reply([evidence("Amsterdam.", "R1")]),
            tool_call_response(
                ("r", RECALL, {"ids": ["R1"]}), ("b", RAIN, {"city": "Rotterdam"})
            ),
            reply([evidence("Rotterdam vs Amsterdam.", "R1", "R2")], relation="compare"),
        ]
    )
    service, store = make_service(llm, limits, conv, rainfall=rainfall)
    first = service.ask(None, "Amsterdam rainfall?")
    second = service.ask(first.session_id, "Compare with Rotterdam.")

    answer = second.result.answer
    assert answer.rejected_citations == ()
    assert [r.id for r in answer.data_references] == ["R1", "R2"]
    assert second.result.stats.rainfall_calls == 1  # the recall is not an analysis
    assert len(rainfall.calls) == 2
    assert any("reused from earlier turns" in x for x in answer.limitations)
    assert [p.geonames_id for p in store.get(first.session_id).state.places] == [2759794, 2747891]
    recall_event = second.result.events[0]
    assert recall_event.tool == RECALL and recall_event.recalled_ids == ("R1",)


def test_citing_an_earlier_id_without_recall_is_rejected(limits, conv, make_analysis, places):
    rainfall = FakeRainfall(make_analysis(location=places["amsterdam"]))
    llm = StubLLMClient(
        [
            tool_call_response(("a", RAIN, {"city": "Amsterdam"})),
            reply([evidence("Amsterdam.", "R1")]),
            reply([evidence("As said before, Amsterdam had 50 mm.", "R1")], relation="continue"),
        ]
    )
    service, _ = make_service(llm, limits, conv, rainfall=rainfall)
    first = service.ask(None, "Amsterdam rainfall?")
    second = service.ask(first.session_id, "Repeat the maximum.")

    answer = second.result.answer
    assert answer.rejected_citations == ("R1",)
    assert answer.claims[0].kind == "interpretation"  # earlier answers are never evidence


def test_a_new_topic_archives_the_old_one(limits, conv, make_analysis, places, make_work,
                                          make_result):
    rainfall = FakeRainfall(make_analysis(location=places["amsterdam"]))
    search = FakeSearch(make_result([make_work("W1", title="Crop mapping with CNNs")]))
    llm = StubLLMClient(
        [
            tool_call_response(("a", RAIN, {"city": "Amsterdam"})),
            reply([evidence("Amsterdam.", "R1")]),
            tool_call_response(("s", SEARCH, {"query": "crop mapping"})),
            reply([evidence("CNNs map crops.", "W1")], relation="new_topic"),
        ]
    )
    service, store = make_service(llm, limits, conv, rainfall=rainfall, search=search)
    first = service.ask(None, "Amsterdam rainfall?")
    service.ask(first.session_id, "Which deep learning methods map crops?")

    session = store.get(first.session_id)
    assert session.state.topic == 2 and session.state.places == ()
    assert session.state.queries == ("crop mapping",)
    assert session.archived_topics == ["Topic 1: places Amsterdam, North Holland, Netherlands, "
                                       "metrics wet"]
    context = build_context(session, conv)
    assert "- Places:" not in context.block  # the old topic's settings are not carried over
    assert "- W1: Crop mapping with CNNs" in context.block and "- R1:" not in context.block
    assert set(context.recallable) == {"R1", "W1"}  # old evidence can still be recalled by ID


def test_a_missing_relation_counts_as_update(limits, conv, make_analysis, places):
    rainfall = FakeRainfall(
        make_analysis(location=places["amsterdam"]), make_analysis(location=places["rotterdam"])
    )
    llm = StubLLMClient(
        [
            tool_call_response(("a", RAIN, {"city": "Amsterdam", "metric": "dry"})),
            reply([evidence("A.", "R1")]),
            tool_call_response(("b", RAIN, {"city": "Rotterdam", "metric": "dry"})),
            reply([evidence("R.", "R2")]),  # no relation given
        ]
    )
    service, store = make_service(llm, limits, conv, rainfall=rainfall)
    first = service.ask(None, "How dry was Amsterdam?")
    second = service.ask(first.session_id, "And Rotterdam?")
    assert (second.turn.relation, second.turn.relation_source) == ("update", "default")
    assert store.get(first.session_id).state.metrics == ("dry",)


# --- per-turn guarantees ------------------------------------------------------------------------


def test_budgets_are_per_turn(limits, conv, make_analysis, places):
    one = replace(limits, max_rainfall_calls=1, max_search_calls=1)
    rainfall = FakeRainfall(
        make_analysis(location=places["amsterdam"]), make_analysis(location=places["rotterdam"])
    )
    llm = StubLLMClient(
        [
            tool_call_response(("a", RAIN, {"city": "Amsterdam"})),
            reply([evidence("A.", "R1")]),
            tool_call_response(("b", RAIN, {"city": "Rotterdam"})),
            reply([evidence("R.", "R2")], relation="update"),
        ]
    )
    service, _ = make_service(llm, one, conv, rainfall=rainfall)
    first = service.ask(None, "Amsterdam?")
    second = service.ask(first.session_id, "Rotterdam?")
    assert len(rainfall.calls) == 2  # turn 1's spent budget does not block turn 2
    assert second.result.stats.rainfall_calls == 1


def test_a_failed_turn_leaves_the_session_unchanged(limits, conv, make_analysis, places):
    rainfall = FakeRainfall(make_analysis(location=places["amsterdam"]))
    llm = StubLLMClient(
        [
            tool_call_response(("a", RAIN, {"city": "Amsterdam"})),
            reply([evidence("A.", "R1")]),
            LLMRateLimitError("quota"),
        ]
    )
    service, store = make_service(llm, limits, conv, rainfall=rainfall)
    first = service.ask(None, "Amsterdam?")
    before = store.get(first.session_id)
    with pytest.raises(LLMRateLimitError):
        service.ask(first.session_id, "Rotterdam?")
    after = store.get(first.session_id)
    assert after is before and len(after.turns) == 1


def test_recall_is_offered_only_with_stored_evidence_and_rejects_unknown_ids(
    limits, conv, make_analysis, places
):
    rainfall = FakeRainfall(make_analysis(location=places["amsterdam"]))
    llm = StubLLMClient(
        [
            tool_call_response(("a", RAIN, {"city": "Amsterdam"})),
            reply([evidence("A.", "R1")]),
            tool_call_response(("r", RECALL, {"ids": ["R9"]})),
            reply([], relation="continue"),
        ]
    )
    service, _ = make_service(llm, limits, conv, rainfall=rainfall)
    first = service.ask(None, "Amsterdam?")
    second = service.ask(first.session_id, "More detail?")
    assert RECALL not in {t.name for t in llm.calls[0].tools}
    assert RECALL in {t.name for t in llm.calls[2].tools}
    tool_result = llm.calls[3].messages[-1]
    assert tool_result.content.startswith("Error: not in the stored evidence: R9")
    assert second.result.events[0].status == "error"


def test_without_context_the_prompt_is_unchanged(limits):
    plain = build_instructions(limits, rainfall=True, enso=True)
    assert CONVERSATION_RULES not in plain
    assert build_instructions(limits, rainfall=True, enso=True, conversation=True) == (
        plain + CONVERSATION_RULES
    )


def test_left_out_context_is_reported_as_a_limitation(limits, base_config, make_analysis, places):
    small = replace(base_config.conversation, max_context_tokens=1)
    rainfall = FakeRainfall(make_analysis(location=places["amsterdam"]))
    llm = StubLLMClient(
        [
            tool_call_response(("a", RAIN, {"city": "Amsterdam"})),
            reply([evidence("A.", "R1")]),
            reply([], relation="continue"),
        ]
    )
    service, _ = make_service(llm, limits, small, rainfall=rainfall)
    first = service.ask(None, "Amsterdam?")
    second = service.ask(first.session_id, "More?")
    assert second.context.omitted > 0
    assert any("left out" in x for x in second.result.answer.limitations)


# --- context budget (pure) ----------------------------------------------------------------------


def _turn(n, topic=1):
    return TurnRecord(
        number=n, question=f"question {n}", relation="update", relation_source="model",
        topic=topic, answer_digest=f"(interpretation) answer {n}", evidence_ids=(),
        summary_line=f"Turn {n} (update): question {n}", tokens=100,
    )


def _session(make_work, turns=5):
    work = make_work("W7", abstract="A very long abstract that must never be sent again. " * 50)
    item = EvidenceItem("W7", "work", "A title (2023)", 1, 1, work, ToolRef(SEARCH, {}))
    return Session(
        id="s", created_at=0, updated_at=0, turns=[_turn(n) for n in range(1, turns + 1)],
        state=TaskState(queries=("crop mapping",)), evidence={"W7": item},
    )


def test_context_keeps_recent_turns_and_summarises_older_ones(conv, make_work):
    context = build_context(_session(make_work), replace(conv, recent_turns=2))
    sent = [m.text for m in context.messages]
    assert sent[0] == "question 4" and sent[2] == "question 5" and len(sent) == 4
    assert "Turn 1 (update): question 1" in context.block
    assert "Turn 3 (update): question 3" in context.block
    assert "- W7: A title (2023) (turn 1)" in context.block
    assert "must never be sent again" not in context.block  # tool output is never resent
    assert context.omitted == 0 and context.tokens > 0


def test_a_tiny_budget_keeps_only_the_task_state(conv, make_work):
    context = build_context(_session(make_work), replace(conv, max_context_tokens=1))
    assert context.messages == ()
    assert context.block.startswith("Conversation state") and "crop mapping" in context.block
    assert context.omitted == 1 + 1 + 2 + 2  # last turn, index line, 2 recent turns, 2 summaries


def test_a_middle_budget_drops_the_lowest_priority_first(conv, make_work):
    session = _session(make_work)
    full = build_context(session, conv)
    cut = build_context(session, replace(conv, max_context_tokens=full.tokens - 5))
    assert cut.omitted >= 1
    assert cut.messages[-2].text == "question 5"  # the last turn survives
    assert "Turn 1 (update)" not in cut.block  # the oldest summary goes first


def test_the_context_never_exceeds_its_budget(conv, make_work):
    session = _session(make_work, turns=8)
    full = build_context(session, conv).tokens
    floor = build_context(session, replace(conv, max_context_tokens=1)).tokens  # state only
    for budget in range(floor, full + 10, 3):
        context = build_context(session, replace(conv, max_context_tokens=budget))
        assert context.tokens <= budget, budget


def test_the_first_question_gets_no_context(conv):
    assert build_context(Session(id="s", created_at=0, updated_at=0), conv) is None


# --- sessions -----------------------------------------------------------------------------------


def _fake_result(text="ok"):
    answer = Answer(claims=(Claim(text, "interpretation", ()),), references=(), limitations=())
    return AgentResult(answer, RunStats(1, 0, 10, 5, 0))


def test_sessions_do_not_share_turns(conv):
    seen = []

    def run(question, context):
        seen.append(context)
        return _fake_result(question)

    store = InMemorySessionStore(10, 3600)
    service = ConversationService(run, store, conv)
    a = service.ask(None, "first in A")
    b = service.ask(None, "first in B")
    service.ask(a.session_id, "second in A")
    assert a.session_id != b.session_id
    assert seen[1] is None  # B starts empty
    assert "first in A" in [m.text for m in seen[2].messages]
    assert "first in B" not in [m.text for m in seen[2].messages]
    assert [t.question for t in service.turns(b.session_id)] == ["first in B"]


def test_an_unknown_session_id_starts_a_new_session(conv):
    service = ConversationService(lambda q, c: _fake_result(), InMemorySessionStore(10, 3600), conv)
    outcome = service.ask("not-a-real-id", "hello")
    assert outcome.session_id != "not-a-real-id" and outcome.turn.number == 1


def test_store_drops_least_recently_used_and_expired_sessions():
    now = [0.0]
    store = InMemorySessionStore(2, ttl_seconds=100, clock=lambda: now[0])
    for sid in ("a", "b"):
        store.put(Session(id=sid, created_at=0, updated_at=0))
    assert store.get("a") is not None  # "a" is now the most recently used
    store.put(Session(id="c", created_at=0, updated_at=0))
    assert store.get("b") is None and store.get("a") is not None
    now[0] = 1000.0
    assert store.get("a") is None and len(store) == 1


def test_store_is_safe_under_concurrent_use():
    store = InMemorySessionStore(50, 3600)

    def work(prefix):
        for i in range(200):
            store.put(Session(id=f"{prefix}{i % 60}", created_at=0, updated_at=0))
            store.get(f"{prefix}{(i * 7) % 60}")

    threads = [threading.Thread(target=work, args=(p,)) for p in "abcd"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store) == 50


# --- CLI ----------------------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.contexts = []

    def run(self, question, context=None):
        self.contexts.append(context)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _chat(monkeypatch, tmp_path, agent, lines):
    @contextmanager
    def fake_open(cfg, creds):
        yield agent

    monkeypatch.setattr("research_agent.cli._open_agent", fake_open)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test-0123456789abcdef\n", encoding="utf-8")
    args = ["chat", "--config", "configs/base.toml", "--env-file", str(env)]
    return CliRunner().invoke(app, args, input="\n".join(lines) + "\n")


def test_chat_sends_earlier_turns_with_a_follow_up(monkeypatch, tmp_path):
    agent = _FakeAgent(_fake_result("Amsterdam answer"), _fake_result("Rotterdam answer"))
    lines = ["Amsterdam rainfall?", "What about Rotterdam?", ""]
    result = _chat(monkeypatch, tmp_path, agent, lines)
    assert result.exit_code == 0
    assert agent.contexts[0] is None
    assert [m.text for m in agent.contexts[1].messages][0] == "Amsterdam rainfall?"
    assert "Amsterdam answer" in result.output and "Rotterdam answer" in result.output
    assert "turn 2 (update)" in result.output


def test_chat_continues_after_a_provider_error(monkeypatch, tmp_path):
    agent = _FakeAgent(LLMRateLimitError("quota"), _fake_result("second works"))
    result = _chat(monkeypatch, tmp_path, agent, ["one", "two", "exit"])
    assert result.exit_code == 0
    assert "error: quota (the conversation continues)" in result.output
    assert "second works" in result.output
    assert agent.contexts[1] is None  # the failed turn was not recorded


def test_chat_stops_on_an_auth_error(monkeypatch, tmp_path):
    agent = _FakeAgent(LLMAuthError("OpenAI rejected the API key"))
    result = _chat(monkeypatch, tmp_path, agent, ["one", "two"])
    assert result.exit_code == 1 and "rejected the API key" in result.output
