"""Tracing tests: spans are recorded in memory, never exported. No network, no cost."""

import base64
import sys

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from test_agent_loop import FakeSearch, call, evidence_claim, final
from test_openai_client import CFG, FakeSDK, make_response, message_item
from typer.testing import CliRunner

from research_agent import tracing
from research_agent.agent import ResearchAgent
from research_agent.cli import app
from research_agent.config import ReportingConfig
from research_agent.credentials import CredentialError
from research_agent.llm import (
    LLMRateLimitError,
    OpenAIClient,
    StubLLMClient,
    UserMessage,
    text_response,
    tool_call_response,
)
from research_agent.logging_setup import REDACTOR
from research_agent.retrieval import RateLimitError

REPORTING = ReportingConfig(thin_evidence_works=3, stale_after_years=3)
PK, SK = "pk-lf-0123456789", "sk-lf-abcdefghijkl"
LANGFUSE = {
    "LANGFUSE_PUBLIC_KEY": PK,
    "LANGFUSE_SECRET_KEY": SK,
    "LANGFUSE_HOST": "https://cloud.langfuse.com/",
}

_EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="session", autouse=True)
def _provider():
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
    trace.set_tracer_provider(provider)


@pytest.fixture
def spans(monkeypatch):
    _EXPORTER.clear()
    monkeypatch.setattr(tracing, "_capture_content", True)
    yield _EXPORTER
    _EXPORTER.clear()
    REDACTOR.clear()


def by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def run_agent(llm, search, limits, question="What works for land cover?"):
    return ResearchAgent(llm, search, limits, REPORTING).run(question)


# --- spans from the agent loop ------------------------------------------------------------------


def test_a_search_run_gives_agent_generation_and_tool_spans(spans, limits, make_work, make_result):
    search = FakeSearch(make_result([make_work("W1")]))
    reply = final([evidence_claim("Deep nets work.", ["W1"])])
    llm = StubLLMClient(
        [tool_call_response(call(), input_tokens=30, output_tokens=4), text_response(reply)]
    )

    run_agent(llm, search, limits)

    names = [s.name for s in spans.get_finished_spans()]
    assert names == ["chat", "execute_tool search_literature", "chat", "invoke_agent"]
    s = by_name(spans)
    agent, tool = s["invoke_agent"], s["execute_tool search_literature"]
    chats = [x for x in spans.get_finished_spans() if x.name == "chat"]
    for child in (*chats, tool):
        assert child.parent.span_id == agent.context.span_id
    assert agent.attributes["langfuse.observation.type"] == "agent"
    assert tool.attributes["langfuse.observation.type"] == "tool"
    assert chats[0].attributes["langfuse.observation.type"] == "generation"
    assert chats[0].attributes["gen_ai.usage.input_tokens"] == 30
    assert chats[0].attributes["gen_ai.usage.output_tokens"] == 4
    assert [c.attributes["research_agent.turn"] for c in chats] == [1, 2]
    assert agent.attributes["research_agent.search_calls"] == 1
    assert agent.attributes["research_agent.input_tokens"] == 40
    assert tool.attributes["research_agent.tool.error"] is False
    # content
    assert agent.attributes["langfuse.observation.input"] == "What works for land cover?"
    assert '"Deep nets work."' in agent.attributes["langfuse.observation.output"]
    assert '"role": "system"' in chats[0].attributes["langfuse.observation.input"]
    assert '"land cover"' in tool.attributes["langfuse.observation.input"]
    assert "[W1]" in tool.attributes["langfuse.observation.output"]


def test_content_is_left_out_when_switched_off(spans, monkeypatch, limits, make_work, make_result):
    monkeypatch.setattr(tracing, "_capture_content", False)
    search = FakeSearch(make_result([make_work("W1")]))
    llm = StubLLMClient([tool_call_response(call()), text_response(final())])

    run_agent(llm, search, limits, question="a private question")

    for span in spans.get_finished_spans():
        assert "langfuse.observation.input" not in span.attributes
        assert "langfuse.observation.output" not in span.attributes
    assert by_name(spans)["chat"].attributes["gen_ai.usage.input_tokens"] == 10  # metadata kept


def test_a_failed_tool_call_is_flagged(spans, limits):
    search = FakeSearch(RateLimitError("slow down"))
    llm = StubLLMClient([tool_call_response(call()), text_response(final())])

    run_agent(llm, search, limits)

    tool = by_name(spans)["execute_tool search_literature"]
    assert tool.attributes["research_agent.tool.error"] is True
    assert tool.status.status_code == StatusCode.ERROR
    assert by_name(spans)["invoke_agent"].status.status_code != StatusCode.ERROR


def test_a_model_error_marks_spans_with_redacted_text(spans, limits):
    REDACTOR.add("sk-secret-in-message")
    llm = StubLLMClient([LLMRateLimitError("quota for sk-secret-in-message")])

    with pytest.raises(LLMRateLimitError):
        run_agent(llm, FakeSearch(), limits)

    for name in ("chat", "invoke_agent"):
        span = by_name(spans)[name]
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description == "quota for <redacted>"
        assert span.attributes["error.type"] == "LLMRateLimitError"
        assert not span.events  # no exception event with an unredacted message or traceback


def test_the_openai_adapter_names_the_model(spans):
    sdk = FakeSDK(make_response([message_item("hi")]))
    with tracing.observe("chat", "generation"):
        OpenAIClient(CFG, sdk).complete(
            instructions="x", messages=[UserMessage("q")], tools=[], max_output_tokens=10
        )
    assert by_name(spans)["chat"].attributes["gen_ai.request.model"] == CFG.model


# --- settings -----------------------------------------------------------------------------------


def test_nothing_set_means_no_tracing():
    assert tracing.export_target({}, {}) is None
    assert tracing.configure_tracing({}, {}) is False


def test_langfuse_target():
    target = tracing.export_target(LANGFUSE, {})
    assert target.endpoint == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    auth = base64.b64encode(f"{PK}:{SK}".encode()).decode()
    assert target.headers == {
        "Authorization": f"Basic {auth}",
        "x-langfuse-ingestion-version": "4",
    }
    assert set(target.secrets) == {SK, auth}


@pytest.mark.parametrize("missing", ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"])
def test_langfuse_needs_all_three(missing):
    env = {k: v for k, v in LANGFUSE.items() if k != missing}
    with pytest.raises(tracing.TracingError, match=missing):
        tracing.export_target(env, {})


def test_langfuse_and_otel_together_is_an_error():
    with pytest.raises(tracing.TracingError, match="choose one"):
        tracing.export_target(LANGFUSE, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://x:4318"})


def test_bad_host_and_placeholder_key_are_rejected_without_echoing_keys():
    with pytest.raises(tracing.TracingError) as exc:
        tracing.export_target({**LANGFUSE, "LANGFUSE_HOST": "cloud.langfuse.com"}, {})
    assert SK not in str(exc.value)
    with pytest.raises(CredentialError, match="placeholder"):
        tracing.export_target({**LANGFUSE, "LANGFUSE_SECRET_KEY": "your-secret-key"}, {})


def test_otel_target_leaves_settings_to_the_sdk_and_redacts_header_values():
    process = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example",
        "OTEL_EXPORTER_OTLP_HEADERS": "api-key=abcdef123456,x-team=platform-team",
    }
    target = tracing.export_target({}, process)
    assert (target.endpoint, target.headers) == (None, None)
    assert target.secrets == ("abcdef123456", "platform-team")


def test_otel_variables_in_the_env_file_alone_do_not_switch_tracing_on():
    assert tracing.export_target({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://x"}, {}) is None


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, True), ("", True), ("true", True), ("0", False), ("False", False)]
)
def test_content_setting(raw, expected):
    env = {} if raw is None else {"RESEARCH_AGENT_TRACE_CONTENT": raw}
    assert tracing.capture_content_setting(env) is expected


def test_content_setting_rejects_other_values():
    with pytest.raises(tracing.TracingError):
        tracing.capture_content_setting({"RESEARCH_AGENT_TRACE_CONTENT": "maybe"})


def test_configure_installs_a_provider_and_registers_secrets(spans, monkeypatch):
    installed = []
    monkeypatch.setattr(tracing.trace, "set_tracer_provider", installed.append)
    env = {**LANGFUSE, "RESEARCH_AGENT_TRACE_CONTENT": "false"}

    assert tracing.configure_tracing(env, {}) is True

    assert len(installed) == 1
    assert tracing._capture_content is False
    assert REDACTOR.redact(f"key {SK}") == "key <redacted>"
    installed[0].shutdown()  # nothing was recorded, so nothing is sent


def test_configure_without_the_extra_says_how_to_install(spans, monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    with pytest.raises(tracing.TracingError, match="uv sync --extra tracing"):
        tracing.configure_tracing(LANGFUSE, {})


def test_cli_exits_on_incomplete_langfuse_settings(spans, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        f"OPENAI_API_KEY=sk-test-0123456789abcdef\nLANGFUSE_SECRET_KEY={SK}\n", encoding="utf-8"
    )
    args = ["ask", "q", "--config", "configs/base.toml", "--env-file", str(env)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "LANGFUSE_PUBLIC_KEY" in result.output
    assert SK not in result.output


def test_conversation_turns_share_a_langfuse_session(spans, base_config):
    from research_agent.agent import AgentResult, RunStats
    from research_agent.agent.answer import Answer, Claim
    from research_agent.conversation import ConversationService, InMemorySessionStore

    def run(question, context):
        answer = Answer((Claim("ok", "interpretation", ()),), (), ())
        return AgentResult(answer, RunStats(1, 0, 10, 5, 0))

    service = ConversationService(run, InMemorySessionStore(10, 3600), base_config.conversation)
    first = service.ask(None, "first question")
    service.ask(first.session_id, "follow-up")

    turns = [s for s in spans.get_finished_spans() if s.name == "chat.turn"]
    assert [t.attributes["langfuse.session.id"] for t in turns] == [first.session_id] * 2
    assert [t.attributes["research_agent.turn"] for t in turns] == [1, 2]
    assert turns[0].attributes["research_agent.context.tokens"] == 0
    assert turns[1].attributes["research_agent.context.tokens"] > 0
    assert turns[1].attributes["research_agent.relation"] == "update"
    assert turns[1].attributes["research_agent.relation_source"] == "default"
    assert turns[1].attributes["langfuse.observation.input"] == "follow-up"
