"""Unit tests for TokenGuard CircuitBreaker and Guard Engine."""

import pytest
import time
from tokenguard.guard import (
    BudgetExceededError,
    CircuitBreaker,
    KillSwitchActiveError,
    LoopDetectedError,
    BurnRateExceededError,
    ContextBallooningError,
    parse_burn_rate,
)


def test_pricing_calculation():
    guard = CircuitBreaker()
    cost = guard.compute_cost("gpt-4o", 1000, 500)
    assert pytest.approx(cost, 1e-6) == 0.0075

    cost_mini = guard.compute_cost("gpt-4o-mini", 10000, 2000)
    assert pytest.approx(cost_mini, 1e-6) == 0.0027

    cost_prefix = guard.compute_cost("gpt-4o-2024-08-06", 1000, 500)
    assert pytest.approx(cost_prefix, 1e-6) == 0.0075

    cost_ds_chat = guard.compute_cost("deepseek-chat", 10000, 2000)
    assert pytest.approx(cost_ds_chat, 1e-6) == 0.00196

    cost_ds_reasoner = guard.compute_cost("deepseek-reasoner", 10000, 2000)
    assert pytest.approx(cost_ds_reasoner, 1e-6) == 0.00988

    cost_ds_prefix = guard.compute_cost("deepseek-chat-v3", 10000, 2000)
    assert pytest.approx(cost_ds_prefix, 1e-6) == 0.00196
    cost_ds_provider = guard.compute_cost("deepseek/deepseek-reasoner", 10000, 2000)
    assert pytest.approx(cost_ds_provider, 1e-6) == 0.00988

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

    h1 = guard.check_request(messages_loop, "gpt-4o")
    assert len(guard.recent_hashes) == 1

    h2 = guard.check_request(messages_loop, "gpt-4o")
    assert len(guard.recent_hashes) == 2
    assert h1 == h2

    with pytest.raises(LoopDetectedError) as exc_info:
        guard.check_request(messages_loop, "gpt-4o")
    assert "infinite agent loop detected" in str(exc_info.value.message)
    assert exc_info.value.error_type == "loop_detected"
    assert len(guard.recent_hashes) == 2

    for _ in range(5):
        with pytest.raises(LoopDetectedError):
            guard.check_request(messages_loop, "gpt-4o")
    assert len(guard.recent_hashes) == 2

    h_other = guard.check_request(messages_other, "gpt-4o")
    assert h_other != h1
    assert len(guard.recent_hashes) == 3

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
    assert guard.check_loop(hash_a) is True
    assert guard.check_loop(hash_b) is False

    guard.reset()
    assert len(guard.recent_hashes) == 0
    assert guard.check_loop(hash_a) is False


def test_remove_last_hash_on_upstream_failure():
    guard = CircuitBreaker(loop_threshold=3, loop_window_size=20)
    hash_a = "aaa"

    guard.record_request_hash(hash_a)
    assert len(guard.recent_hashes) == 1

    guard.remove_last_hash(hash_a)
    assert len(guard.recent_hashes) == 0


def test_loop_time_window_expiration():
    guard = CircuitBreaker(loop_threshold=3, loop_window_size=20, loop_window_seconds=0.1)
    hash_a = "aaa"

    guard.record_request_hash(hash_a)
    guard.record_request_hash(hash_a)
    assert guard.check_loop(hash_a) is True

    time.sleep(0.15)
    assert guard.check_loop(hash_a) is False


def test_loop_threshold_disabled():
    guard = CircuitBreaker(loop_threshold=0)
    hash_a = "aaa"
    guard.record_request_hash(hash_a)
    guard.record_request_hash(hash_a)
    guard.record_request_hash(hash_a)
    assert guard.check_loop(hash_a) is False


def test_budget_guard_tripping():
    guard = CircuitBreaker(hourly_limit=0.0100)
    messages = [{"role": "user", "content": "Process massive text" * 2000}]

    h1 = guard.check_request([{"role": "user", "content": "Small prompt 1"}], "gpt-4o")
    guard.record_success("gpt-4o", 2000, 100, 0.0060, h1)
    assert pytest.approx(guard.get_hourly_spend(), 1e-6) == 0.0060

    h2 = guard.check_request([{"role": "user", "content": "Small prompt 2"}], "gpt-4o")
    guard.record_success("gpt-4o", 1000, 50, 0.0035, h2)
    assert pytest.approx(guard.get_hourly_spend(), 1e-6) == 0.0095

    with pytest.raises(BudgetExceededError) as exc_info:
        guard.check_request(messages, "gpt-4o")
    assert "budget limit exceeded" in str(exc_info.value.message)
    assert exc_info.value.error_type == "budget_exceeded"


def test_kill_switch():
    guard = CircuitBreaker()
    messages = [{"role": "user", "content": "Hello"}]

    assert guard.check_request(messages, "gpt-4o")

    guard.toggle_kill_switch(True)
    assert guard.kill_switch_active is True

    with pytest.raises(KillSwitchActiveError) as exc_info:
        guard.check_request(messages, "gpt-4o")
    assert "kill switch active" in str(exc_info.value.message)
    assert exc_info.value.error_type == "kill_switch_active"

    guard.toggle_kill_switch(False)
    assert guard.check_request(messages, "gpt-4o")


def test_avoided_cost_calculation():
    guard = CircuitBreaker()
    saved_gpt4o = guard.calculate_avoided_cost("gpt-4o", prompt_tokens=100, avoided_completion_tokens=500)
    assert pytest.approx(saved_gpt4o, 1e-6) == 0.00525

    saved_ds = guard.calculate_avoided_cost("deepseek-chat", prompt_tokens=200, avoided_completion_tokens=500)
    assert pytest.approx(saved_ds, 1e-6) == 0.000168


def test_notification_on_loop_block(monkeypatch):
    """Verify that a desktop notification is triggered non-blockingly when a loop is tripped."""
    import sys
    import tokenguard.guard
    guard_mod = sys.modules["tokenguard.guard"]
    notified = []

    def mock_send(title, message, sound=True):
        notified.append({"title": title, "message": message})

    monkeypatch.setattr(guard_mod, "send_block_notification", mock_send)

    guard = CircuitBreaker(loop_threshold=2)
    messages = [{"role": "user", "content": "Loop test prompt"}]

    guard.check_request(messages, "gpt-4o")
    assert len(notified) == 0

    with pytest.raises(LoopDetectedError):
        guard.check_request(messages, "gpt-4o")

    assert len(notified) == 1
    assert "Loop" in notified[0]["title"] or "TokenGuard" in notified[0]["title"]
    assert "gpt-4o" in notified[0]["message"]


def test_passive_mode_non_blocking():
    """Verify that passive mode allows infinite repeated requests and exceeds budget without throwing exceptions."""
    guard = CircuitBreaker(
        hourly_limit=0.0001,
        loop_threshold=2,
        active_profile="passive",
    )
    messages = [{"role": "user", "content": "Repeated prompt in passive mode"}]

    h1 = guard.check_request(messages, "gpt-4o")
    guard.record_success("gpt-4o", 1000, 500, 0.05, h1)

    h2 = guard.check_request(messages, "gpt-4o")
    assert h2 == h1

    h3 = guard.check_request(messages, "gpt-4o")
    assert h3 == h1


def test_parse_burn_rate():
    assert parse_burn_rate("$0.50/min") == (0.50, 60.0)
    assert parse_burn_rate("$2.00/5m") == (2.00, 300.0)
    assert parse_burn_rate("1.50/30s") == (1.50, 30.0)
    assert parse_burn_rate("$10.00/1h") == (10.00, 3600.0)
    assert parse_burn_rate("$0.25/10sec") == (0.25, 10.0)
    assert parse_burn_rate("$5/minute") == (5.0, 60.0)

    with pytest.raises(ValueError):
        parse_burn_rate("invalid")
    with pytest.raises(ValueError):
        parse_burn_rate("$0/min")


def test_burn_rate_circuit_breaker_tripping():
    guard = CircuitBreaker(max_burn_rate="$0.50/min")
    now = time.time()
    guard.burn_rate_records.append((now - 10, 0.35))

    messages = [{"role": "user", "content": "Hello"}]
    guard.check_request(messages, "gpt-4o")

    guard.burn_rate_records.append((now - 5, 0.20))

    with pytest.raises(BurnRateExceededError) as exc_info:
        guard.check_request(messages, "gpt-4o")

    err = exc_info.value
    assert err.error_type == "tokenguard_budget_breach"
    assert err.code == "burn_rate_exceeded"
    assert err.budget_limit == 0.50
    assert err.current_spend == pytest.approx(0.55, 1e-4)

    payload = err.to_dict()
    assert payload["error"]["type"] == "tokenguard_budget_breach"
    assert payload["error"]["code"] == "burn_rate_exceeded"
    assert payload["error"]["budget_limit"] == 0.50
    assert payload["error"]["current_spend"] == pytest.approx(0.55, 1e-4)
    assert "Burn rate exceeded" in payload["error"]["message"]
    assert "Hard limit: $0.50/min" in payload["error"]["message"]


def test_burn_rate_sliding_window_expiration():
    guard = CircuitBreaker(max_burn_rate="$0.50/min")
    now = time.time()
    guard.burn_rate_records.append((now - 70, 0.60))

    messages = [{"role": "user", "content": "Hello after window"}]
    guard.check_request(messages, "gpt-4o")


def test_context_ballooning_tripping():
    guard = CircuitBreaker(enable_context_ballooning=True)
    session_id = "agent-session-42"

    msg1 = [{"role": "user", "content": "Initial prompt specification"}]
    guard.check_request(msg1, "gpt-4o", session_id=session_id, incoming_tokens=100)
    guard.record_success("gpt-4o", prompt_tokens=100, completion_tokens=20, cost=0.001, session_id=session_id)

    msg2 = [{"role": "user", "content": "Initial prompt plus compiler error 1 and stack trace"}]
    guard.check_request(msg2, "gpt-4o", session_id=session_id, incoming_tokens=250)
    guard.record_success("gpt-4o", prompt_tokens=250, completion_tokens=20, cost=0.002, session_id=session_id)

    msg3 = [{"role": "user", "content": "Initial prompt plus compiler errors and intermediate thoughts"}]
    with pytest.raises(ContextBallooningError) as exc_info:
        guard.check_request(msg3, "gpt-4o", session_id=session_id, incoming_tokens=450)

    err = exc_info.value
    assert err.error_type == "tokenguard_budget_breach"
    assert err.code == "context_ballooning_detected"
    payload = err.to_dict()
    assert payload["error"]["code"] == "context_ballooning_detected"
    assert "Context ballooning detected" in payload["error"]["message"]


def test_context_ballooning_bypass_with_task_or_cache():
    guard = CircuitBreaker(enable_context_ballooning=True)
    session_id = "agent-session-bypass"

    msg1 = [{"role": "user", "content": "Step 1 instruction"}]
    guard.check_request(msg1, "gpt-4o", session_id=session_id, incoming_tokens=100)
    guard.record_success("gpt-4o", prompt_tokens=100, completion_tokens=20, cost=0.001, session_id=session_id)

    msg2 = [{"role": "user", "content": "Step 2 instruction with stack trace"}]
    guard.check_request(msg2, "gpt-4o", session_id=session_id, incoming_tokens=250)
    guard.record_success("gpt-4o", prompt_tokens=250, completion_tokens=20, cost=0.002, session_id=session_id)

    msg3 = [{"role": "user", "content": "Step 3 instruction with large context"}]
    guard.check_request(msg3, "gpt-4o", session_id=session_id, incoming_tokens=500, has_task_update=True)

    guard.check_request(msg3, "gpt-4o", session_id="different-session", incoming_tokens=500)


def test_tool_ping_pong_loop_detection():
    from tokenguard.guard import ToolPingPongError
    guard = CircuitBreaker(tool_loop_threshold=3)
    session = "agent-session-1"

    guard.check_tool_loop(session, "read_file", {"path": "/var/log/app.log"})
    guard.record_tool_call(session, "read_file", {"path": "/var/log/app.log"})

    guard.check_tool_loop(session, "read_file", {"path": "/var/log/app.log"})
    guard.record_tool_call(session, "read_file", {"path": "/var/log/app.log"})

    with pytest.raises(ToolPingPongError) as exc_info:
        guard.check_tool_loop(session, "read_file", {"path": "/var/log/app.log"})

    err = exc_info.value
    assert err.code == "TOOL_PING_PONG_LOOP"
    assert err.error_type == "circuit_breaker"
    assert err.to_dict() == {
        "error": {
            "type": "circuit_breaker",
            "code": "TOOL_PING_PONG_LOOP",
            "message": "TokenGuard intercepted repeated tool call loop: read_file",
            "tool_name": "read_file",
            "repeats": 3,
        }
    }


def test_tool_ping_pong_fuzzy_similarity_detection():
    from tokenguard.guard import ToolPingPongError
    guard = CircuitBreaker(tool_loop_threshold=3)
    session = "agent-session-fuzzy"

    guard.check_tool_loop(session, "bash", {"command": "curl http://127.0.0.1/health", "timeout": 10})
    guard.record_tool_call(session, "bash", {"command": "curl http://127.0.0.1/health", "timeout": 10})

    guard.check_tool_loop(session, "bash", {"command": "curl http://127.0.0.1/health ", "timeout": 10})
    guard.record_tool_call(session, "bash", {"command": "curl http://127.0.0.1/health ", "timeout": 10})

    with pytest.raises(ToolPingPongError) as exc_info:
        guard.check_tool_loop(session, "bash", {"command": "curl http://127.0.0.1/health", "timeout": 10})

    assert exc_info.value.code == "TOOL_PING_PONG_LOOP"


def test_tool_ping_pong_progression_allowed():
    guard = CircuitBreaker(tool_loop_threshold=3)
    session = "agent-session-progress"

    guard.check_tool_loop(session, "read_file", {"path": "module_a.py"})
    guard.record_tool_call(session, "read_file", {"path": "module_a.py"})

    guard.check_tool_loop(session, "read_file", {"path": "completely_different_component_b.py"})
    guard.record_tool_call(session, "read_file", {"path": "completely_different_component_b.py"})

    guard.check_tool_loop(session, "read_file", {"path": "third_unrelated_service_c.py"})
    guard.record_tool_call(session, "read_file", {"path": "third_unrelated_service_c.py"})


def test_tool_ping_pong_in_messages_history():
    from tokenguard.guard import ToolPingPongError
    guard = CircuitBreaker(tool_loop_threshold=3)

    messages = [
        {"role": "user", "content": "Fix bug in app"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "bug.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "FileNotFoundError"},
        {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "bug.py"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "FileNotFoundError"},
        {"role": "assistant", "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "bug.py"}'}}]},
    ]

    with pytest.raises(ToolPingPongError):
        guard.check_request(messages, "gpt-4o", session_id="msg-history-session")


def test_tool_cycle_normal_varied_calls_pass():
    """Verify that varied tool calls with different tools and arguments pass without tripping."""
    guard = CircuitBreaker(tool_loop_threshold=3)
    session = "agent-normal-calls"

    calls = [
        ("read_file", {"path": "src/main.py"}),
        ("bash", {"command": "pytest tests/"}),
        ("search", {"query": "AttributeError in runner"}),
        ("git_status", {}),
        ("edit_file", {"path": "src/main.py", "diff": "+ fix"}),
        ("bash", {"command": "git commit -m 'fix'"}),
    ]

    for tool_name, args in calls:
        guard.check_tool_loop(session, tool_name, args)
        guard.record_tool_call(session, tool_name, args)

    assert len(guard._tool_history[session]) == 6


def test_tool_cycle_standard_consecutive_loop_trips():
    """Verify standard consecutive loop (A-A-A) trips ToolPingPongError immediately on 3rd call."""
    from tokenguard.guard import ToolPingPongError
    guard = CircuitBreaker(tool_loop_threshold=3)
    session = "agent-consecutive-loop"

    tool = "read_file"
    args = {"path": "missing_config.yaml"}

    guard.check_tool_loop(session, tool, args)
    guard.record_tool_call(session, tool, args)

    guard.check_tool_loop(session, tool, args)
    guard.record_tool_call(session, tool, args)

    with pytest.raises(ToolPingPongError) as exc_info:
        guard.check_tool_loop(session, tool, args)

    err = exc_info.value
    assert err.code == "TOOL_PING_PONG_LOOP"
    assert err.status_code == 429
    assert err.tool_name == "read_file"
    assert err.repeats == 3
    assert err.cycle_length == 1
    assert err.to_dict()["error"]["code"] == "TOOL_PING_PONG_LOOP"


def test_tool_cycle_alternating_loop_trips():
    """Verify alternating oscillating tool calls (A-B-A-B) trip ToolPingPongError on 4th call."""
    from tokenguard.guard import ToolPingPongError
    guard = CircuitBreaker(tool_loop_threshold=3)
    session = "agent-alternating-loop"

    tool_a = ("read_file", {"path": "server.py"})
    tool_b = ("bash", {"command": "python server.py"})

    guard.check_tool_loop(session, tool_a[0], tool_a[1])
    guard.record_tool_call(session, tool_a[0], tool_a[1])

    guard.check_tool_loop(session, tool_b[0], tool_b[1])
    guard.record_tool_call(session, tool_b[0], tool_b[1])

    guard.check_tool_loop(session, tool_a[0], tool_a[1])
    guard.record_tool_call(session, tool_a[0], tool_a[1])

    with pytest.raises(ToolPingPongError) as exc_info:
        guard.check_tool_loop(session, tool_b[0], tool_b[1])

    err = exc_info.value
    assert err.code == "TOOL_PING_PONG_LOOP"
    assert err.status_code == 429
    assert err.cycle_length == 2
    assert err.pattern == ["read_file", "bash"]
    assert "alternating" in err.message.lower()
    err_dict = err.to_dict()["error"]
    assert err_dict["code"] == "TOOL_PING_PONG_LOOP"
    assert err_dict["cycle_length"] == 2
    assert err_dict["pattern"] == ["read_file", "bash"]


def test_tool_cycle_length_3_trips():
    """Verify 3-step cycle (A-B-C-A-B-C) trips ToolPingPongError on 6th call."""
    from tokenguard.guard import ToolPingPongError
    guard = CircuitBreaker(tool_loop_threshold=3)
    session = "agent-length-3-loop"

    tool_a = ("read_file", {"path": "module.py"})
    tool_b = ("bash", {"command": "cat module.py"})
    tool_c = ("search", {"query": "module syntax error"})

    sequence = [tool_a, tool_b, tool_c, tool_a, tool_b]
    for t_name, t_args in sequence:
        guard.check_tool_loop(session, t_name, t_args)
        guard.record_tool_call(session, t_name, t_args)

    with pytest.raises(ToolPingPongError) as exc_info:
        guard.check_tool_loop(session, tool_c[0], tool_c[1])

    err = exc_info.value
    assert err.code == "TOOL_PING_PONG_LOOP"
    assert err.status_code == 429
    assert err.cycle_length == 3
    assert err.pattern == ["read_file", "bash", "search"]


def test_tool_cycle_in_messages_alternating():
    """Verify alternating tool calls inside conversation messages history trip circuit breaker."""
    from tokenguard.guard import ToolPingPongError
    guard = CircuitBreaker(tool_loop_threshold=3)

    messages = [
        {"role": "user", "content": "Debug server failure"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "server.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "SyntaxError: line 42"},
        {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "python server.py"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "Process failed with code 1"},
        {"role": "assistant", "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "server.py"}'}}]},
        {"role": "tool", "tool_call_id": "c3", "content": "SyntaxError: line 42"},
        {"role": "assistant", "tool_calls": [{"id": "c4", "type": "function", "function": {"name": "bash", "arguments": '{"command": "python server.py"}'}}]},
    ]

    with pytest.raises(ToolPingPongError) as exc_info:
        guard.check_request(messages, "gpt-4o", session_id="agent-msg-alt")

    assert exc_info.value.code == "TOOL_PING_PONG_LOOP"
    assert exc_info.value.cycle_length == 2


def test_prompt_cycle_alternating_loop_trips():
    """Verify prompt-level alternating loops (A-B-A-B) trip LoopDetectedError."""
    from tokenguard.guard import LoopDetectedError
    guard = CircuitBreaker(loop_threshold=3)

    msg_a = [{"role": "user", "content": "Prompt Alpha"}]
    msg_b = [{"role": "user", "content": "Prompt Beta"}]

    guard.check_request(msg_a, "gpt-4o", session_id="sess-alt-prompt")
    guard.check_request(msg_b, "gpt-4o", session_id="sess-alt-prompt")
    guard.check_request(msg_a, "gpt-4o", session_id="sess-alt-prompt")

    with pytest.raises(LoopDetectedError) as exc_info:
        guard.check_request(msg_b, "gpt-4o", session_id="sess-alt-prompt")

    assert exc_info.value.status_code == 429
    assert "infinite agent loop detected" in exc_info.value.message


def test_tool_history_sliding_window_maxlen():
    """Verify sliding window of recent call signatures is bounded to maxlen=10 per session."""
    guard = CircuitBreaker()
    session = "agent-sliding-bound"

    for i in range(25):
        guard.record_tool_call(session, f"tool_{i}", {"index": i})

    assert len(guard._tool_history[session]) == 10
    last_entry = guard._tool_history[session][-1]
    assert last_entry[1] == "tool_24"


def test_normalize_tool_arguments_strips_dynamic_nonces():
    """Verify normalize_tool_arguments strips call_id, timestamp, ts, nonce, _id, request_id."""
    from tokenguard.guard import normalize_tool_arguments

    args_dict = {
        "command": "curl http://127.0.0.1/health",
        "call_id": "call_abc123",
        "timestamp": "2026-10-28T12:00:00Z",
        "ts": 1714000000,
        "nonce": "xyz987",
        "_id": "doc_456",
        "request_id": "req-999",
        "timeout": 30,
    }

    norm_str, clean_obj = normalize_tool_arguments(args_dict)
    assert "call_id" not in clean_obj
    assert "timestamp" not in clean_obj
    assert "ts" not in clean_obj
    assert "nonce" not in clean_obj
    assert "_id" not in clean_obj
    assert "request_id" not in clean_obj
    assert clean_obj == {"command": "curl http://127.0.0.1/health", "timeout": 30}
    assert norm_str == '{"command":"curl http://127.0.0.1/health","timeout":30}'


def test_normalize_tool_arguments_json_string_with_nested_nonces():
    """Verify JSON string inputs and nested structures have nonces stripped recursively."""
    import json
    from tokenguard.guard import normalize_tool_arguments

    raw_json = json.dumps({
        "file": "main.py ",
        "meta": {
            "request_id": "req-1",
            "timestamp": 12345,
            "line": 42,
        },
        "items": [
            {"_id": "1", "name": "foo "},
            {"nonce": "n1", "name": "bar"},
        ],
    })

    norm_str, clean_obj = normalize_tool_arguments(raw_json)
    assert clean_obj["file"] == "main.py"
    assert "request_id" not in clean_obj["meta"]
    assert "timestamp" not in clean_obj["meta"]
    assert clean_obj["meta"] == {"line": 42}
    assert clean_obj["items"] == [{"name": "foo"}, {"name": "bar"}]


def test_tool_loop_detection_trips_across_varying_dynamic_nonces():
    """Verify tool loop circuit breaker catches repetitive calls even when nonces/timestamps vary."""
    from tokenguard.guard import CircuitBreaker, ToolPingPongError

    guard = CircuitBreaker(tool_loop_threshold=3)
    session = "sess-nonce-test"

    call1 = {"path": "/workspace/config.json", "call_id": "call-1", "ts": 1000, "nonce": "n1"}
    call2 = {"path": "/workspace/config.json", "call_id": "call-2", "ts": 1001, "nonce": "n2"}
    call3 = {"path": "/workspace/config.json", "call_id": "call-3", "ts": 1002, "nonce": "n3"}

    guard.check_tool_loop(session, "read_file", call1)
    guard.record_tool_call(session, "read_file", call1)

    guard.check_tool_loop(session, "read_file", call2)
    guard.record_tool_call(session, "read_file", call2)

    with pytest.raises(ToolPingPongError) as exc_info:
        guard.check_tool_loop(session, "read_file", call3)

    assert exc_info.value.tool_name == "read_file"
    assert exc_info.value.repeats == 3


