"""The fixed-search baseline: one search with a fixed query, then one model call with tools off.

``FixedSearchLLMClient`` wraps the model client. The agent's first model call is answered by
code with a scripted ``search_literature`` call (no model, no tokens); the next call goes to the
model with tools disabled, so it must answer from that one search. Everything else is the
agent's own code (prompt, result formatting, citation validation, limitations), so a fixed run
and an agentic run differ only in who chooses the searches.
"""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.agent.tools import SEARCH_TOOL_NAME
from research_agent.evaluation.cases import EvalCase
from research_agent.llm import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, Usage

FIXED_CALL_ID = "fixed_search_1"


def fixed_query(case: EvalCase) -> str:
    """The case's ``fixed_query``, else the question as asked (the owner's choice)."""
    return case.fixed_query or case.question


class FixedSearchLLMClient:
    """Scripts the first call of each case as one search; later calls go to ``inner`` with
    tools disabled. Call ``start(query)`` before each case."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self._query: str | None = None
        self._searched = False

    def start(self, query: str) -> None:
        self._query = query
        self._searched = False

    def complete(
        self,
        *,
        instructions: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
        allow_tools: bool = True,
    ) -> LLMResponse:
        if self._query is None:
            raise RuntimeError("FixedSearchLLMClient.start() was not called")
        if not self._searched:
            self._searched = True
            call = ToolCall(FIXED_CALL_ID, SEARCH_TOOL_NAME, {"query": self._query})
            return LLMResponse(text=None, tool_calls=(call,), usage=Usage(0, 0),
                               stop_reason="tool_calls")  # fmt: skip
        return self.inner.complete(
            instructions=instructions,
            messages=messages,
            tools=tools,
            max_output_tokens=max_output_tokens,
            allow_tools=False,
        )
