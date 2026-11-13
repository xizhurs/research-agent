"""Phase 4b: multi-turn evaluation conversations (offline; stubbed model, fake tools)."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from test_agent_rainfall import FakeRainfall, FakeSearch
from typer.testing import CliRunner

from research_agent.agent import ResearchAgent
from research_agent.cli import app
from research_agent.config import ReportingConfig
from research_agent.evaluation import (
    ConversationDriver,
    EvalError,
    RecordingLLMClient,
    ReplayLLMClient,
    RunBudget,
    TurnExpect,
    check_turn,
    load_conversations,
    read_records,
    replay_section,
    rescore,
    run_conversations,
)
from research_agent.llm import LLMRateLimitError, StubLLMClient, text_response, tool_call_response

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.toml"
REPORTING = ReportingConfig(thin_evidence_works=3, stale_after_years=3)
RAIN, RECALL = "analyze_rainfall_extremes", "recall_evidence"

CONVERSATION = """[[conversation]]
id = "c"

[[conversation.turn]]
question = "Amsterdam rainfall?"
expect_tool = "analyze_rainfall_extremes"
min_data_references = 1
expect_places = ["amsterdam"]

[[conversation.turn]]
question = "Compare with Rotterdam."
expect_relation = ["compare"]
expect_recall = true
max_new_analyses = 1
min_data_references = 2
expect_places = ["amsterdam", "rotterdam"]
"""


def reply(claims, relation=None):
    data = {"claims": claims, "limitations": ["Only ERA5."]}
    if relation:
        data["relation"] = relation
    return text_response(json.dumps(data))


def script():
    return [
        tool_call_response(("a", RAIN, {"city": "Amsterdam"})),
        reply([{"text": "Amsterdam.", "kind": "evidence", "sources": ["R1"]}]),
        tool_call_response(("r", RECALL, {"ids": ["R1"]}), ("b", RAIN, {"city": "Rotterdam"})),
        reply([{"text": "Rotterdam vs Amsterdam.", "kind": "evidence", "sources": ["R1", "R2"]}],
              relation="compare"),
    ]  # fmt: skip


@pytest.fixture
def cities(make_location, make_analysis):
    amsterdam = make_location(id=2759794, name="Amsterdam", admin1="North Holland",
                              country="Netherlands", country_code="NL")  # fmt: skip
    rotterdam = make_location(id=2747891, name="Rotterdam", admin1="South Holland",
                              country="Netherlands", country_code="NL")  # fmt: skip
    return lambda: FakeRainfall(make_analysis(location=amsterdam),
                                make_analysis(location=rotterdam))  # fmt: skip


def write(tmp_path, text, name="conv.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def record_conversation(tmp_path, llm, rainfall, limits, conv_cfg, out="run", recorder=None):
    talks = load_conversations(write(tmp_path, CONVERSATION))
    recorder = recorder or RecordingLLMClient(llm, RunBudget(100, 10**6))
    agent = ResearchAgent(recorder, FakeSearch(), limits, REPORTING, rainfall=rainfall)
    driver = ConversationDriver(lambda q, ctx, **kw: agent.run(q, ctx, **kw), conv_cfg)
    return talks, recorder, driver


# --- the conversation file -----------------------------------------------------------------------


def test_conversation_file_gives_one_case_per_turn(tmp_path):
    (conv,) = load_conversations(write(tmp_path, CONVERSATION))
    first, second = conv.turns
    assert (conv.id, conv.split) == ("c", "dev")
    assert (first.id, first.category, first.split) == ("c#1", "conversation", "dev")
    assert first.expect_tool == RAIN and first.turn == TurnExpect("c", 1,
                                                                  expect_places=("amsterdam",))
    assert second.turn.expect_relation == ("compare",) and second.turn.expect_recall is True
    assert second.turn.max_new_analyses == 1 and second.min_data_references == 2
    assert load_conversations(tmp_path / "missing.toml") == []


@pytest.mark.parametrize(
    "text, message",
    [
        ('[[conversation]]\nid = "a#b"\n', "without '#'"),
        ('[[conversation]]\nid = "a"\n', "at least one"),
        ('[[conversation]]\nid = "a"\nbogus = 1\n[[conversation.turn]]\nquestion = "Q"\n',
         "unknown keys"),
        ('[[conversation]]\nid = "a"\n[[conversation.turn]]\nquestion = "Q"\nsplit = "test"\n',
         "belong to the conversation"),
        ('[[conversation]]\nid = "a"\n[[conversation.turn]]\nquestion = "Q"\n'
         'expect_relation = ["agree"]\n', "expect_relation"),
        ('[[conversation]]\nid = "a"\n[[conversation.turn]]\nquestion = "Q"\n'
         'expect_recall = "yes"\n', "true or false"),
        ('[[conversation]]\nid = "a"\n[[conversation.turn]]\nquestion = "Q"\nmin_searches = -1\n',
         "min_searches"),
    ],
)  # fmt: skip
def test_conversation_file_errors(tmp_path, text, message):
    with pytest.raises(EvalError, match=message):
        load_conversations(write(tmp_path, text))


def test_shipped_conversations_load():
    talks = load_conversations(BASE.parents[1] / "evals" / "conversations.toml")
    assert [len(c.turns) for c in talks] == [3, 2, 2, 2]
    assert [c.id for c in talks if c.split == "test"] == ["conv-new-topic"]


# --- turn checks ---------------------------------------------------------------------------------


def test_check_turn_judges_relation_recall_analyses_and_state():
    expect = TurnExpect("c", 2, expect_relation=("compare",), expect_recall=True,
                        max_new_analyses=0, expect_places=("rotterdam",),
                        expect_state_metrics=("yearly",), expect_pending=False)  # fmt: skip
    record = {"stats": {"rainfall_calls": 1}, "conversation": {
        "relation": "update", "relation_source": "default", "recalled": [],
        "state": {"places": ["Amsterdam, North Holland"], "metrics": ["wet"],
                  "pending": "Springfield"}}}  # fmt: skip
    assert check_turn(expect, record) == [
        "relation 'update' (default) is not one of ['compare']",
        "expected stored evidence to be recalled; nothing was",
        "ran 1 new analysis(es) (turn limit 0)",
        "task state has no place matching 'rotterdam'; places: ['Amsterdam, North Holland']",
        "task state lacks metric 'yearly'; metrics: ['wet']",
        "expected no pending clarification; still pending for 'Springfield'",
    ]
    assert check_turn(TurnExpect("c", 1, expect_recall=False, expect_pending=True),
                      {"stats": {"rainfall_calls": 0}, "conversation": {
                          "recalled": ["R1"], "state": {"pending": None}}}) == [
        "expected no recall; recalled ['R1']",
        "expected a pending clarification (ambiguous place); there is none",
    ]  # fmt: skip


# --- running, rescoring and replaying ------------------------------------------------------------


def test_a_conversation_runs_turn_by_turn_in_one_session(tmp_path, cities, limits, base_config):
    talks, recorder, driver = record_conversation(
        tmp_path, StubLLMClient(script()), cities(), limits, base_config.conversation
    )
    first, second = run_conversations(talks, driver.ask, recorder, limits,
                                      out_dir=tmp_path / "run", run_id="r1", meta={})  # fmt: skip
    assert (first["case_id"], second["case_id"]) == ("c#1", "c#2")
    assert first["failures"] == [] and second["failures"] == []
    conv = second["conversation"]
    assert (conv["id"], conv["turn"], conv["relation"], conv["relation_source"]) == (
        "c", 2, "compare", "model"
    )
    assert conv["recalled"] == ["R1"] and conv["context_turns"] == 1 and conv["context_tokens"] > 0
    assert [p.split(",")[0] for p in conv["state"]["places"]] == ["Amsterdam", "Rotterdam"]
    assert first["conversation"]["context_tokens"] == 0  # a first turn has no context
    assert len(second["llm_calls"]) == 2  # each turn's record holds only its own calls

    stricter = [replace(t, turn=replace(t.turn, expect_relation=("new_topic",)))
                for t in talks[0].turns]  # fmt: skip
    rescored = rescore([first, second], stricter, limits)
    assert rescored[1]["failures"] == ["relation 'compare' (model) is not one of ['new_topic']"]


def test_later_turns_are_skipped_after_a_failed_turn(tmp_path, cities, limits, base_config):
    llm = StubLLMClient([LLMRateLimitError("quota")])
    talks, recorder, driver = record_conversation(tmp_path, llm, cities(), limits,
                                                  base_config.conversation)  # fmt: skip
    first, second = run_conversations(talks, driver.ask, recorder, limits,
                                      out_dir=tmp_path / "run", run_id="r1", meta={})  # fmt: skip
    assert first["status"] == "failed" and second["status"] == "skipped"
    assert second["failures"] == ["skipped: turn 1 of c ended 'failed'"]
    assert second["conversation"] == {"id": "c", "turn": 2}


def replay_conversation(tmp_path, records, rainfall, limits, conv_cfg):
    originals = {r["case_id"]: r for r in records}
    replayer = ReplayLLMClient()
    talks, recorder, driver = record_conversation(
        tmp_path, None, rainfall, limits, conv_cfg,
        recorder=RecordingLLMClient(replayer, RunBudget(10**9, 10**15)),
    )  # fmt: skip

    def run_turn(conv, case):
        replayer.load(originals[case.id])
        outcome = driver.ask(conv, case)
        replayer.finish()
        return outcome

    return run_conversations(
        talks, run_turn, recorder, limits, out_dir=tmp_path / "replay", run_id="r1-replay",
        meta={}, annotate=lambda c, rec: {**rec, "replay": replay_section(originals[c.id], rec)},
    )  # fmt: skip


def test_a_conversation_replays_turn_by_turn(tmp_path, cities, limits, base_config):
    conv_cfg = base_config.conversation
    talks, recorder, driver = record_conversation(tmp_path, StubLLMClient(script()), cities(),
                                                  limits, conv_cfg)  # fmt: skip
    records = run_conversations(talks, driver.ask, recorder, limits, out_dir=tmp_path / "run",
                                run_id="r1", meta={})  # fmt: skip
    same = replay_conversation(tmp_path, records, cities(), limits, conv_cfg)
    assert [r["replay"]["outcome"] for r in same] == ["same", "same"]

    tighter = replace(conv_cfg, max_context_tokens=60)  # the second turn's context changes
    diverged = replay_conversation(tmp_path, records, cities(), limits, tighter)
    assert diverged[0]["replay"]["outcome"] == "same"
    assert diverged[1]["failures"] == [
        "diverged at call 1: the conversation context (earlier turns) or the request structure "
        "changed"
    ]


def test_cli_runs_scores_and_replays_conversations(
    monkeypatch, tmp_path, cities, base_config
):
    @contextmanager
    def fake_open(cfg, creds, *, wrap_llm=None, http_client=None, today=None, llm=None):
        model = llm if llm is not None else wrap_llm(StubLLMClient(script()))
        yield ResearchAgent(model, FakeSearch(), cfg.limits, REPORTING, rainfall=cities(),
                            today=today)  # fmt: skip

    monkeypatch.setattr("research_agent.cli._open_agent", fake_open)
    env = write(tmp_path, "OPENAI_API_KEY=sk-test-0123456789abcdef\n", ".env")
    conversations = write(tmp_path, CONVERSATION)
    common = ["--conversations", str(conversations), "--config", str(BASE)]
    out = tmp_path / "run"
    result = CliRunner().invoke(app, ["eval", "run", "--suite", "conversations", "--live",
                                      "--yes", "--out", str(out), "--env-file", str(env),
                                      *common])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Plan (conversations): 1 conversation(s), 2 turn(s): c" in result.output
    assert "[PASS] c#1" in result.output and "[PASS] c#2" in result.output
    assert "Conversations: 1/1 with every turn passed" in result.output
    assert "turn 2 of c: relation compare (model)" in (out / "report.md").read_text("utf-8")

    scored = CliRunner().invoke(app, ["eval", "score", str(out), *common])
    assert scored.exit_code == 0 and "Conversations: 1/1" in scored.output

    replayed = CliRunner().invoke(app, ["eval", "replay", str(out), "--out",
                                        str(tmp_path / "replayed"), *common])  # fmt: skip
    assert replayed.exit_code == 0, replayed.output
    assert "[SAME] [PASS] c#2" in replayed.output and "Replay: 2 same" in replayed.output
    assert [r["case_id"] for r in read_records(tmp_path / "replayed")] == ["c#1", "c#2"]

    bad = CliRunner().invoke(app, ["eval", "run", "--suite", "conversations", "--mode", "fixed",
                                   *common])  # fmt: skip
    assert bad.exit_code == 1 and "neither --mode fixed" in bad.output
