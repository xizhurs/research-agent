"""Provider-agnostic LLM interface plus a scripted stub and the first real adapter."""

from research_agent.llm.errors import (
    LLMApiError,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
)
from research_agent.llm.openai_client import OpenAIClient
from research_agent.llm.stub import StubLLMClient, text_response, tool_call_response
from research_agent.llm.types import (
    AssistantMessage,
    LLMClient,
    LLMResponse,
    Message,
    StopReason,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    Usage,
    UserMessage,
)

__all__ = [
    "AssistantMessage",
    "LLMApiError",
    "LLMAuthError",
    "LLMClient",
    "LLMConnectionError",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMResponseError",
    "Message",
    "OpenAIClient",
    "StopReason",
    "StubLLMClient",
    "ToolCall",
    "ToolResultMessage",
    "ToolSpec",
    "Usage",
    "UserMessage",
    "text_response",
    "tool_call_response",
]
