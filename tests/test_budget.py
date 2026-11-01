from dataclasses import replace

from research_agent.agent import Budget, estimate_tokens
from research_agent.llm import Usage


def test_estimate_tokens_rounds_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_counters(limits):
    b = Budget(limits)
    b.record_call(Usage(100, 20))
    b.record_call(Usage(50, 5))
    b.record_search()
    b.record_context(300)
    assert (b.iterations, b.search_calls) == (2, 1)
    assert (b.input_tokens, b.output_tokens, b.tokens_used) == (150, 25, 175)
    assert b.searches_left == limits.max_search_calls - 1
    assert b.context_left == limits.max_context_tokens - 300


def test_context_left_never_negative(limits):
    b = Budget(limits)
    b.record_context(limits.max_context_tokens + 500)
    assert b.context_left == 0


def test_fresh_budget_allows_tools(limits):
    assert Budget(limits).force_final_reason(next_input_estimate=500) is None


def test_final_iteration_is_forced(limits):
    b = Budget(limits)
    for _ in range(limits.max_iterations - 2):
        b.record_call(Usage(1, 1))
        assert b.force_final_reason(10) is None
    b.record_call(Usage(1, 1))  # the next call is the last allowed one
    assert "Iteration limit" in b.force_final_reason(10)


def test_single_iteration_budget_forces_immediately(limits):
    b = Budget(replace(limits, max_iterations=1))
    assert "Iteration limit" in b.force_final_reason(10)


def test_search_exhaustion_forces_final(limits):
    b = Budget(replace(limits, max_search_calls=2, max_iterations=10))
    b.record_search()
    assert b.force_final_reason(10) is None
    b.record_search()
    assert "Search limit" in b.force_final_reason(10)
    assert b.searches_left == 0


def test_token_rule_needs_room_for_this_call_and_a_final_call(limits):
    tight = replace(limits, max_total_tokens=2000, max_output_tokens=100, max_iterations=10)
    b = Budget(tight)
    # cost of one call = 400 + 100 = 500; need used + 2 * 500 <= 2000
    b.record_call(Usage(500, 500))  # used = 1000
    assert b.force_final_reason(400) is None  # 1000 + 1000 == 2000: allowed
    b.record_call(Usage(1, 0))  # used = 1001
    assert "Token budget" in b.force_final_reason(400)


def test_first_matching_reason_wins(limits):
    b = Budget(replace(limits, max_iterations=1, max_search_calls=1))
    b.record_search()
    assert "Iteration limit" in b.force_final_reason(10)


def test_overshoot_reported_only_above_limit(limits):
    tight = replace(limits, max_total_tokens=100)
    b = Budget(tight)
    b.record_call(Usage(60, 40))
    assert b.overshoot_reason() is None
    b.record_call(Usage(1, 0))
    assert "exceeded" in b.overshoot_reason()
