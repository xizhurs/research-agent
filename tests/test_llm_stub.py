import pytest

from research_agent.llm import (
    AssistantMessage,
    LLMClient,
    LLMRateLimitError,
    StubLLMClient,
    ToolCall,
    ToolSpec,
    Usage,
    UserMessage,
    text_response,
    tool_call_response,
)

SPEC = ToolSpec("search", "Search papers", {"type": "object", "properties": {}})


def call(client, **overrides):
    kwargs = {
        "instructions": "be brief",
        "messages": [UserMessage("hi")],
        "tools": [SPEC],
        "max_output_tokens": 100,
    }
    kwargs.update(overrides)
    return client.complete(**kwargs)


def test_usage_total():
    assert Usage(3, 4).total_tokens == 7


def test_stub_satisfies_protocol():
    client: LLMClient = StubLLMClient([text_response("x")])
    assert call(client).text == "x"


def test_responses_returned_in_order():
    client = StubLLMClient([text_response("one"), text_response("two")])
    assert call(client).text == "one"
    assert call(client).text == "two"


def test_text_response_fields():
    r = text_response("answer", input_tokens=7, output_tokens=3)
    assert r.stop_reason == "stop"
    assert r.tool_calls == ()
    assert r.usage == Usage(7, 3)


def test_tool_call_response_fields():
    r = tool_call_response(("c1", "search", {"q": "a"}), ("c2", "search", {"q": "b"}))
    assert r.stop_reason == "tool_calls"
    assert r.text is None
    assert r.tool_calls == (
        ToolCall("c1", "search", {"q": "a"}),
        ToolCall("c2", "search", {"q": "b"}),
    )


def test_calls_are_recorded_with_all_arguments():
    client = StubLLMClient([text_response("x")])
    call(client, allow_tools=False, max_output_tokens=55)
    (rec,) = client.calls
    assert rec.instructions == "be brief"
    assert rec.messages == (UserMessage("hi"),)
    assert rec.tools == (SPEC,)
    assert rec.max_output_tokens == 55
    assert rec.allow_tools is False


def test_recorded_messages_are_a_snapshot():
    client = StubLLMClient([text_response("a"), text_response("b")])
    history = [UserMessage("hi")]
    call(client, messages=history)
    history.append(UserMessage("later"))
    call(client, messages=history)
    assert len(client.calls[0].messages) == 1
    assert len(client.calls[1].messages) == 2


def test_scripted_exception_is_raised():
    client = StubLLMClient([LLMRateLimitError("slow down"), text_response("ok")])
    with pytest.raises(LLMRateLimitError):
        call(client)
    assert call(client).text == "ok"


def test_exhausted_script_is_a_loud_test_failure():
    client = StubLLMClient([text_response("only")])
    call(client)
    with pytest.raises(AssertionError, match="exhausted"):
        call(client)


def test_response_to_message_carries_calls_and_provider_state():
    r = tool_call_response(("c1", "search", {"q": "a"}))
    msg = r.to_message()
    assert isinstance(msg, AssistantMessage)
    assert msg.tool_calls == r.tool_calls
    assert msg.text is None
    assert msg.provider_state is None
