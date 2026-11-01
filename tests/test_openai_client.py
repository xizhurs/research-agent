"""OpenAI adapter tests. A fake SDK client returns real SDK ``Response`` objects; no network."""

import json
from pathlib import Path

import httpx2
import openai
import pytest
from openai.types.responses import Response

from research_agent.config import load_config
from research_agent.credentials import CredentialError, Credentials, Secret
from research_agent.llm import (
    AssistantMessage,
    LLMApiError,
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    OpenAIClient,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from research_agent.llm.openai_client import messages_to_input, tools_to_openai

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "base.toml").llm
SPEC = ToolSpec(
    "search_literature",
    "Search papers",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
REQ = httpx2.Request("POST", "https://api.openai.com/v1/responses")


def make_response(output, *, status="completed", incomplete=None, usage=True):
    data = {
        "id": "resp_1",
        "created_at": 0.0,
        "model": "m",
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
        "output": output,
    }
    if usage:
        data["usage"] = {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
    if incomplete:
        data["incomplete_details"] = {"reason": incomplete}
    return Response.model_validate(data)


def message_item(text):
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def call_item(call_id="call_1", name="search_literature", arguments='{"query": "x"}'):
    return {
        "type": "function_call",
        "id": "fc_" + call_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


class FakeSDK:
    """Mimics ``openai.OpenAI``: ``client.responses.create(**kwargs)``."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def run(sdk, *, tools=(SPEC,), allow_tools=True, messages=None):
    client = OpenAIClient(CFG, sdk)
    return client.complete(
        instructions="be brief",
        messages=messages or [UserMessage("question")],
        tools=list(tools),
        max_output_tokens=123,
        allow_tools=allow_tools,
    )


# --- request shape ---------------------------------------------------------------------------


def test_request_shape():
    sdk = FakeSDK(make_response([message_item("hi")]))
    run(sdk)
    (req,) = sdk.requests
    assert req["model"] == CFG.model
    assert req["instructions"] == "be brief"
    assert req["max_output_tokens"] == 123
    assert req["reasoning"] == {"effort": CFG.reasoning_effort}
    assert req["input"] == [{"role": "user", "content": "question"}]
    assert req["tool_choice"] == "auto"
    assert "temperature" not in req


def test_tools_are_flat_function_tools_with_strict_false():
    assert tools_to_openai([SPEC]) == [
        {
            "type": "function",
            "name": "search_literature",
            "description": "Search papers",
            "parameters": dict(SPEC.parameters),
            "strict": False,
        }
    ]


def test_allow_tools_false_sets_tool_choice_none():
    sdk = FakeSDK(make_response([message_item("final")]))
    run(sdk, allow_tools=False)
    assert sdk.requests[0]["tool_choice"] == "none"
    assert sdk.requests[0]["tools"]  # tools still declared so earlier calls stay valid


def test_no_tools_means_no_tool_params():
    sdk = FakeSDK(make_response([message_item("hi")]))
    run(sdk, tools=())
    assert "tools" not in sdk.requests[0]
    assert "tool_choice" not in sdk.requests[0]


# --- message conversion ----------------------------------------------------------------------


def test_messages_without_provider_state_are_translated():
    items = messages_to_input(
        [
            UserMessage("q"),
            AssistantMessage(text="thinking", tool_calls=(ToolCall("c1", "search", {"a": 1}),)),
            ToolResultMessage("c1", "result text"),
        ]
    )
    assert items == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "thinking"},
        {"type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"a": 1}'},
        {"type": "function_call_output", "call_id": "c1", "output": "result text"},
    ]


def test_provider_state_is_replayed_verbatim_instead_of_translated():
    first = run(FakeSDK(make_response([call_item("c9")])))
    assert first.provider_state  # captured for replay
    items = messages_to_input(
        [UserMessage("q"), first.to_message(), ToolResultMessage("c9", "out")]
    )
    assert items[1:-1] == list(first.provider_state)
    assert items[-1] == {"type": "function_call_output", "call_id": "c9", "output": "out"}
    json.dumps(items)  # replayed state must be plain JSON-able data


# --- response parsing ------------------------------------------------------------------------


def test_text_response():
    r = run(FakeSDK(make_response([message_item("the answer")])))
    assert r.text == "the answer"
    assert r.tool_calls == ()
    assert r.stop_reason == "stop"
    assert (r.usage.input_tokens, r.usage.output_tokens) == (11, 7)


def test_tool_call_response_parses_json_arguments():
    r = run(FakeSDK(make_response([call_item("c1", arguments='{"query": "land cover"}')])))
    assert r.stop_reason == "tool_calls"
    assert r.text is None
    assert r.tool_calls == (ToolCall("c1", "search_literature", {"query": "land cover"}),)


def test_multiple_tool_calls_keep_order_and_ids():
    r = run(FakeSDK(make_response([call_item("c1"), call_item("c2", arguments="{}")])))
    assert [c.id for c in r.tool_calls] == ["c1", "c2"]


def test_text_and_tool_call_together():
    r = run(FakeSDK(make_response([message_item("let me search"), call_item()])))
    assert r.text == "let me search"
    assert len(r.tool_calls) == 1
    assert r.stop_reason == "tool_calls"


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]", '"str"', "null"])
def test_invalid_tool_arguments_are_reported_not_raised_and_not_echoed(bad):
    r = run(FakeSDK(make_response([call_item("call_1", arguments=bad)])))
    (call,) = r.tool_calls
    assert call.id == "call_1" and call.name == "search_literature"
    assert call.arguments == {}
    assert call.error and "JSON" in call.error
    assert bad not in call.error
    assert r.stop_reason == "tool_calls"


def test_valid_tool_arguments_have_no_error():
    r = run(FakeSDK(make_response([call_item()])))
    assert r.tool_calls[0].error is None


def test_incomplete_for_max_tokens_is_length():
    resp = make_response(
        [message_item("cut off")], status="incomplete", incomplete="max_output_tokens"
    )
    r = run(FakeSDK(resp))
    assert r.stop_reason == "length"
    assert r.text == "cut off"


def test_incomplete_for_other_reason_is_other():
    r = run(FakeSDK(make_response([], status="incomplete", incomplete="content_filter")))
    assert r.stop_reason == "other"
    assert r.text is None


def test_refusal_is_returned_as_text_with_stop_reason_other():
    item = message_item("")
    item["content"] = [{"type": "refusal", "refusal": "I can't help with that."}]
    r = run(FakeSDK(make_response([item])))
    assert r.text == "I can't help with that."
    assert r.stop_reason == "other"


def test_missing_usage_raises_because_budget_cannot_be_enforced():
    with pytest.raises(LLMResponseError, match="usage"):
        run(FakeSDK(make_response([message_item("x")], usage=False)))


def test_failed_response_raises_api_error():
    resp = make_response([], status="failed")
    with pytest.raises(LLMApiError, match="failed"):
        run(FakeSDK(resp))


# --- error mapping ---------------------------------------------------------------------------


def status_error(cls, status, message="boom"):
    return cls(message, response=httpx2.Response(status, request=REQ), body=None)


@pytest.mark.parametrize(
    ("exc_cls", "status"), [(openai.AuthenticationError, 401), (openai.PermissionDeniedError, 403)]
)
def test_auth_errors_hide_server_text_that_may_contain_key_fragments(exc_cls, status):
    exc = status_error(exc_cls, status, "Incorrect API key provided: sk-proj-****abcd")
    with pytest.raises(LLMAuthError) as e:
        run(FakeSDK(exc))
    assert "sk-proj" not in str(e.value)
    assert "OPENAI_API_KEY" in str(e.value)
    assert e.value.__cause__ is None
    assert e.value.__suppress_context__ is True


def test_rate_limit_error_mapped():
    with pytest.raises(LLMRateLimitError, match="quota"):
        run(FakeSDK(status_error(openai.RateLimitError, 429, "You exceeded your quota")))


@pytest.mark.parametrize(
    "exc", [openai.APITimeoutError(request=REQ), openai.APIConnectionError(request=REQ)]
)
def test_connection_and_timeout_errors_mapped(exc):
    with pytest.raises(LLMConnectionError):
        run(FakeSDK(exc))


def test_other_status_errors_mapped_with_status_code():
    with pytest.raises(LLMApiError, match="500"):
        run(FakeSDK(status_error(openai.InternalServerError, 500, "server exploded")))
    with pytest.raises(LLMApiError, match="400"):
        run(FakeSDK(status_error(openai.BadRequestError, 400, "bad param")))


def test_long_server_messages_are_truncated():
    with pytest.raises(LLMApiError) as e:
        run(FakeSDK(status_error(openai.BadRequestError, 400, "x" * 5000)))
    assert len(str(e.value)) < 400


# --- construction ----------------------------------------------------------------------------


def test_from_credentials_requires_the_openai_key():
    with pytest.raises(CredentialError, match="OPENAI_API_KEY is not set"):
        OpenAIClient.from_credentials(CFG, Credentials(None, None))


def test_from_credentials_builds_sdk_client_with_config_and_does_not_leak_key():
    key = "sk-test-0123456789abcdef"
    client = OpenAIClient.from_credentials(CFG, Credentials(None, Secret(key)))
    sdk = client._client
    assert isinstance(sdk, openai.OpenAI)
    assert sdk.max_retries == CFG.max_retries
    assert sdk.timeout == CFG.timeout_seconds
    assert key not in repr(client)
