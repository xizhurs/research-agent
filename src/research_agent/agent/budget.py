"""Per-question resource accounting. Pure: no I/O, no LLM.

The loop asks ``force_final_reason`` before every model call. A non-None result means tools
are withheld for that call so the model must answer with what it has.
"""

from __future__ import annotations

from research_agent.config import Limits
from research_agent.llm import Usage

CHARS_PER_TOKEN = 4
OVERSHOOT_PREFIX = "Token budget exceeded"


def estimate_tokens(text: str) -> int:
    """Rough token count (ceil(chars / 4)) for planning. Real usage comes from the API."""
    return -(-len(text) // CHARS_PER_TOKEN)


class Budget:
    def __init__(self, limits: Limits, *, rainfall_enabled: bool = False) -> None:
        self._limits = limits
        self._rainfall_enabled = rainfall_enabled
        self.iterations = 0  # completed model calls
        self.search_calls = 0
        self.cached_searches = 0  # of search_calls, answered from the local cache
        self.rainfall_calls = 0
        self.cached_rainfall = 0  # of rainfall_calls, answered from the local cache
        self.input_tokens = 0
        self.output_tokens = 0
        self.context_tokens = 0  # estimated tokens of retrieved content sent to the model

    @property
    def tokens_used(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def searches_left(self) -> int:
        return max(0, self._limits.max_search_calls - self.search_calls)

    @property
    def rainfall_left(self) -> int:
        if not self._rainfall_enabled:
            return 0
        return max(0, self._limits.max_rainfall_calls - self.rainfall_calls)

    @property
    def context_left(self) -> int:
        return max(0, self._limits.max_context_tokens - self.context_tokens)

    def record_call(self, usage: Usage) -> None:
        self.iterations += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def record_search(self, *, cached: bool = False) -> None:
        self.search_calls += 1
        if cached:
            self.cached_searches += 1

    def record_rainfall(self, *, cached: bool = False) -> None:
        self.rainfall_calls += 1
        if cached:
            self.cached_rainfall += 1

    def record_context(self, tokens: int) -> None:
        self.context_tokens += tokens

    def force_final_reason(self, next_input_estimate: int) -> str | None:
        """Why the next call must be the final answer, or None if tools are still allowed.

        The token rule is predictive: a tool turn is allowed only if there is room for it
        *and* for a following final call of similar size.
        """
        lim = self._limits
        if self.iterations + 1 >= lim.max_iterations:
            return (
                f"Iteration limit reached ({lim.max_iterations} model turns): the final answer "
                "was written without further searching."
            )
        if self.searches_left == 0 and not self._rainfall_enabled:
            return (
                f"Search limit reached ({self.search_calls} of {lim.max_search_calls} searches "
                "used): the answer is based only on the evidence retrieved so far."
            )
        if self.searches_left == 0 and self.rainfall_left == 0:
            return (
                f"Search and rainfall-analysis limits reached ({self.search_calls} of "
                f"{lim.max_search_calls} searches, {self.rainfall_calls} of "
                f"{lim.max_rainfall_calls} rainfall, temperature or ENSO analyses used): the "
                "answer is based only on the evidence retrieved so far."
            )
        next_call_cost = next_input_estimate + lim.max_output_tokens
        if self.tokens_used + 2 * next_call_cost > lim.max_total_tokens:
            return (
                f"Token budget nearly exhausted ({self.tokens_used} of {lim.max_total_tokens} "
                "tokens used): the final answer was written without further searching."
            )
        return None

    def overshoot_reason(self) -> str | None:
        """Set when actual usage ended above the total-token limit."""
        limit = self._limits.max_total_tokens
        if self.tokens_used > limit:
            return (
                f"{OVERSHOOT_PREFIX}: {self.tokens_used} tokens used against a limit of "
                f"{limit}."
            )
        return None
