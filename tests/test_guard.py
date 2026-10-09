"""Unit tests for TokenGuard CircuitBreaker and Guard Engine."""

import pytest
import time
from tokenguard.guard import (
    BudgetExceededError,
    CircuitBreaker,
    KillSwitchActiveError,
    LoopDetectedError,
)


def test_pricing_calculation():
    guard = CircuitBreaker()
    # gpt-4o: $2.50 / 1M input, $10.00 / 1M output
    # 1000 input tokens = $0.0025, 500 output tokens = $0.0050 => $0.0075
    cost = guard.compute_cost("gpt-4o", 1000, 500)
    assert pytest.approx(cost, 1e-6) == 0.0075

    # gpt-4o-mini: $0.15 / 1M input, $0.60 / 1M output
    # 10,000 input = $0.0015, 2,000 output = $0.0012 => $0.0027
    cost_mini = guard.compute_cost("gpt-4o-mini", 10000, 2000)
    assert pytest.approx(cost_mini, 1e-6) == 0.0027

    # Prefix match: gpt-4o-2024-08-06 should match gpt-4o rates
    cost_prefix = guard.compute_cost("gpt-4o-2024-08-06", 1000, 500)
    assert pytest.approx(cost_prefix, 1e-6) == 0.0075

    # deepseek-chat: $0.14 / 1M input, $0.28 / 1M output
    # 10,000 input = $0.0014, 2,000 output = $0.00056 => $0.00196
    cost_ds_chat = guard.compute_cost("deepseek-chat", 10000, 2000)
    assert pytest.approx(cost_ds_chat, 1e-6) == 0.00196

    # deepseek-reasoner: $0.55 / 1M input, $2.19 / 1M output
    # 10,000 input = $0.0055, 2,000 output = $0.00438 => $0.00988
    cost_ds_reasoner = guard.compute_cost("deepseek-reasoner", 10000, 2000)
    assert pytest.approx(cost_ds_reasoner, 1e-6) == 0.00988

    # Prefix match: deepseek-chat-v3 and provider prefix deepseek/deepseek-reasoner
    cost_ds_prefix = guard.compute_cost("deepseek-chat-v3", 10000, 2000)
    assert pytest.approx(cost_ds_prefix, 1e-6) == 0.00196
    cost_ds_provider = guard.compute_cost("deepseek/deepseek-reasoner", 10000, 2000)
    assert pytest.approx(cost_ds_provider, 1e-6) == 0.00988

    # Unknown model fallback to default ($1.00 input, $2.00 output)
    cost_default = guard.compute_cost("unknown-custom-model", 1000, 1000)
    assert pytest.approx(cost_default, 1e-6) == 0.0030


def test_prompt_hash_deterministic():
    messages_1 = [{"role": "user", "content": "Hello, how are you?"}]
    messages_2 = [{"role": "user", "content": "Hello, how are you?"}]
    messages_3 = [{"role": "user", "content": "Different content"}]

    hash_1 = CircuitBreaker.compute_prompt_hash(messages_1)
    hash_2 = CircuitBreaker.compute_prompt_hash(messages_2)
    hash_3 = CircuitBreaker.compute_prompt_hash(messages_3)

    assert hash_1 == hash_2
    assert hash_1 != hash_3


def test_loop_detector_tripping():
    guard = CircuitBreaker(loop_threshold=3, loop_window_size=20)
    messages_loop = [{"role": "user", "content": "Run infinite task"}]
    messages_other = [{"role": "user", "content": "Another normal prompt"}]

    # 1st call: Allowed
    h1 = guard.check_request(messages_loop, "gpt-4o")
    assert len(guard.recent_hashes) == 1

    # 2nd consecutive call: Allowed
    h2 = guard.check_request(messages_loop, "gpt-4o")
    assert len(guard.recent_hashes) == 2
    assert h1 == h2

    # 3rd consecutive call: TRIPPED!
    with pytest.raises(LoopDetectedError) as exc_info:
        guard.check_request(messages_loop, "gpt-4o")
    assert "infinite agent loop detected" in str(exc_info.value.message)
    assert exc_info.value.error_type == "loop_detected"
    # Ensure rejected attempt did NOT append to recent_hashes
    assert len(guard.recent_hashes) == 2

    # Multiple client retries on 429 should still be blocked but not poison future distinct prompts
    for _ in range(5):
        with pytest.raises(LoopDetectedError):
            guard.check_request(messages_loop, "gpt-4o")
    assert len(guard.recent_hashes) == 2

    # Sending a NEW, distinct prompt IMMEDIATELY succeeds!
    h_other = guard.check_request(messages_other, "gpt-4o")
    assert h_other != h1
    assert len(guard.recent_hashes) == 3

    # Now the previous message can be called again without immediate trip (history tail is now [h1, h_other])
    h_retry = guard.check_request(messages_loop, "gpt-4o")
    assert h_retry == h1


def test_guard_reset_and_check_loop():
    guard = CircuitBreaker(loop_threshold=3, loop_window_size=20)
    hash_a = "aaa"
    hash_b = "bbb"

    assert guard.check_loop(hash_a) is False
    guard.record_request_hash(hash_a)
    assert guard.check_loop(hash_a) is False
    guard.record_request_hash(hash_a)
    # 2 previous entries equal hash_a => next hash_a trips
    assert guard.check_loop(hash_a) is True
    # Different hash_b does NOT trip
    assert guard.check_loop(hash_b) is False

    # Reset clears hashes
    guard.reset()
    assert len(guard.recent_hashes) == 0
    assert guard.check_loop(hash_a) is False


def test_remove_last_hash_on_upstream_failure():
    guard = CircuitBreaker(loop_threshold=3, loop_window_size=20)
    hash_a = "aaa"

    # Pre-flight records hash
    guard.record_request_hash(hash_a)
    assert len(guard.recent_hashes) == 1

    # Upstream fails -> remove hash
    guard.remove_last_hash(hash_a)
    assert len(guard.recent_hashes) == 0


def test_loop_time_window_expiration():
    # 1 second loop window
    guard = CircuitBreaker(loop_threshold=3, loop_window_size=20, loop_window_seconds=0.1)
    hash_a = "aaa"

    guard.record_request_hash(hash_a)
    guard.record_request_hash(hash_a)
    assert guard.check_loop(hash_a) is True

    # Sleep beyond window
    time.sleep(0.15)
    # Hashes expired, should no longer trip!
    assert guard.check_loop(hash_a) is False


def test_loop_threshold_disabled():
    guard = CircuitBreaker(loop_threshold=0)
    hash_a = "aaa"
    guard.record_request_hash(hash_a)
    guard.record_request_hash(hash_a)
    guard.record_request_hash(hash_a)
    assert guard.check_loop(hash_a) is False


def test_budget_guard_tripping():
    # Hourly limit set to $0.0100
    guard = CircuitBreaker(hourly_limit=0.0100)
    messages = [{"role": "user", "content": "Process massive text" * 2000}]

    # Request 1: $0.0060 spend
    h1 = guard.check_request([{"role": "user", "content": "Small prompt 1"}], "gpt-4o")
    guard.record_success("gpt-4o", 2000, 100, 0.0060, h1)
    assert pytest.approx(guard.get_hourly_spend(), 1e-6) == 0.0060

    # Request 2: $0.0035 spend (Total: $0.0095 <= $0.0100)
    h2 = guard.check_request([{"role": "user", "content": "Small prompt 2"}], "gpt-4o")
    guard.record_success("gpt-4o", 1000, 50, 0.0035, h2)
    assert pytest.approx(guard.get_hourly_spend(), 1e-6) == 0.0095

    # Request 3: Large prompt with estimated cost > $0.0005 remaining budget -> TRIPPED!
    with pytest.raises(BudgetExceededError) as exc_info:
        guard.check_request(messages, "gpt-4o")
    assert "budget limit exceeded" in str(exc_info.value.message)
    assert exc_info.value.error_type == "budget_exceeded"


def test_kill_switch():
    guard = CircuitBreaker()
    messages = [{"role": "user", "content": "Hello"}]

    # Normal execution
    assert guard.check_request(messages, "gpt-4o")

    # Engage Kill Switch
    guard.toggle_kill_switch(True)
    assert guard.kill_switch_active is True

    # Check request should be rejected
    with pytest.raises(KillSwitchActiveError) as exc_info:
        guard.check_request(messages, "gpt-4o")
    assert "kill switch active" in str(exc_info.value.message)
    assert exc_info.value.error_type == "kill_switch_active"

    # Disengage Kill Switch
    guard.toggle_kill_switch(False)
    assert guard.check_request(messages, "gpt-4o")


def test_avoided_cost_calculation():
    guard = CircuitBreaker()
    # gpt-4o: prompt 100 tokens, 500 completion tokens
    # (100 * 2.50 + 500 * 10.00) / 1,000,000 = (250 + 5000) / 1M = 0.00525
    saved_gpt4o = guard.calculate_avoided_cost("gpt-4o", prompt_tokens=100, avoided_completion_tokens=500)
    assert pytest.approx(saved_gpt4o, 1e-6) == 0.00525

    # deepseek-chat: prompt 200 tokens, 500 completion tokens
    # (200 * 0.14 + 500 * 0.28) / 1,000,000 = (28 + 140) / 1M = 0.000168
    saved_ds = guard.calculate_avoided_cost("deepseek-chat", prompt_tokens=200, avoided_completion_tokens=500)
    assert pytest.approx(saved_ds, 1e-6) == 0.000168


def test_notification_on_loop_block(monkeypatch):
    """Verify that a desktop notification is triggered non-blockingly when a loop is tripped."""
    notified = []

    def mock_send(title, message, sound=True):
        notified.append({"title": title, "message": message})

    monkeypatch.setattr("tokenguard.guard.send_block_notification", mock_send)

    guard = CircuitBreaker(loop_threshold=2)
    messages = [{"role": "user", "content": "Loop test prompt"}]

    # 1st request -> allowed
    guard.check_request(messages, "gpt-4o")
    assert len(notified) == 0

    # 2nd request (duplicate) -> tripped!
    with pytest.raises(LoopDetectedError):
        guard.check_request(messages, "gpt-4o")

    assert len(notified) == 1
    assert "Loop" in notified[0]["title"] or "TokenGuard" in notified[0]["title"]
    assert "gpt-4o" in notified[0]["message"]


