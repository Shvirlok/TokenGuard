"""Tests for Loop Nudge Synthetic Recovery Mode (v0.3.0-nudge)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import tokenguard.proxy as proxy_module
from tokenguard.cli import cmd_simulate, create_parser
from tokenguard.config import Settings, TokenGuardConfig, set_settings
from tokenguard.guard import CircuitBreaker
from tokenguard.proxy import create_app


def _mock_openai_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_resp = {
        "id": "chatcmpl-upstream-ok",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Upstream response OK"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
    }

    async def mock_post(url, json=None, headers=None):
        return httpx.Response(
            status_code=200,
            json=mock_resp,
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr(proxy_module.http_client, "post", mock_post)


def _mock_anthropic_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_resp = {
        "id": "msg_upstream_ok",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Anthropic upstream OK"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 40, "output_tokens": 12},
    }

    async def mock_post(url, json=None, headers=None):
        return httpx.Response(
            status_code=200,
            json=mock_resp,
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr(proxy_module.http_client, "post", mock_post)


def _append_openai_tool_step(history: list, idx: int, tool_name: str, path: str) -> list:
    return list(history) + [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({"path": path}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{idx}",
            "content": f"FileNotFoundError: {path}",
        },
    ]


def _append_anthropic_tool_step(history: list, idx: int, tool_name: str, path: str) -> list:
    return list(history) + [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"toolu_{idx}",
                    "name": tool_name,
                    "input": {"path": path},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"toolu_{idx}",
                    "content": f"FileNotFoundError: {path}",
                }
            ],
        },
    ]


def test_first_trip_returns_synthetic_200(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify valid synthetic 200 OK JSON structure, headers, and DB status for OpenAI and Anthropic."""
    settings = TokenGuardConfig(
        db_path=tmp_path / "nudge_first.db",
        max_tool_repeats=3,
        on_trip="nudge",
    )
    set_settings(settings)
    app = create_app(settings)

    with TestClient(app) as client:
        # 1. OpenAI /v1/chat/completions
        _mock_openai_upstream(monkeypatch)
        openai_headers = {"x-session-id": "sess-nudge-openai"}
        history = [{"role": "user", "content": "Read deployment configuration"}]

        for i in range(1, 3):
            history = _append_openai_tool_step(history, i, "read_file", f"cfg_{i}.yaml")
            r = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": history},
                headers=openai_headers,
            )
            assert r.status_code == 200
            assert r.headers.get("X-TokenGuard-Action") is None

        # 3rd consecutive read_file -> trips ToolRetryCeilingError -> synthetic 200 nudge
        history = _append_openai_tool_step(history, 3, "read_file", "cfg_3.yaml")
        nudge_resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": history},
            headers=openai_headers,
        )
        assert nudge_resp.status_code == 200
        assert nudge_resp.headers.get("X-TokenGuard-Action") == "nudged"
        assert "ToolRetryCeilingTripped" in nudge_resp.headers.get("X-TokenGuard-Reason", "")

        data = nudge_resp.json()
        assert data["id"].startswith("chatcmpl-tokenguard-nudge-")
        assert data["object"] == "chat.completion"
        assert isinstance(data["created"], int)
        assert data["model"] == "gpt-4o"
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["role"] == "assistant"
        expected_snippet = (
            "[TokenGuard Alert] Execution loop detected: "
            "You have repeated tool call 'read_file' 3 times without state progression."
        )
        assert expected_snippet in choice["message"]["content"]
        assert data["usage"] == {"prompt_tokens": 0, "completion_tokens": 30, "total_tokens": 30}

        # 2. Anthropic /v1/messages
        _mock_anthropic_upstream(monkeypatch)
        anth_headers = {"x-session-id": "sess-nudge-anthropic"}
        anth_history = [{"role": "user", "content": "Read deployment configuration"}]

        for i in range(1, 3):
            anth_history = _append_anthropic_tool_step(anth_history, i, "read_file", f"cfg_{i}.yaml")
            r = client.post(
                "/v1/messages",
                json={"model": "claude-3-5-sonnet-20241022", "max_tokens": 512, "messages": anth_history},
                headers=anth_headers,
            )
            assert r.status_code == 200
            assert r.headers.get("X-TokenGuard-Action") is None

        anth_history = _append_anthropic_tool_step(anth_history, 3, "read_file", "cfg_3.yaml")
        anth_nudge = client.post(
            "/v1/messages",
            json={"model": "claude-3-5-sonnet-20241022", "max_tokens": 512, "messages": anth_history},
            headers=anth_headers,
        )
        assert anth_nudge.status_code == 200
        assert anth_nudge.headers.get("X-TokenGuard-Action") == "nudged"
        assert "ToolRetryCeilingTripped" in anth_nudge.headers.get("X-TokenGuard-Reason", "")

        anth_data = anth_nudge.json()
        assert anth_data["type"] == "message"
        assert anth_data["role"] == "assistant"
        assert anth_data["model"] == "claude-3-5-sonnet-20241022"
        assert anth_data["stop_reason"] == "end_turn"
        assert isinstance(anth_data["content"], list) and len(anth_data["content"]) == 1
        assert anth_data["content"][0]["type"] == "text"
        assert expected_snippet in anth_data["content"][0]["text"]
        assert anth_data["usage"] == {"input_tokens": 0, "output_tokens": 30}

        # Verify SQLite telemetry records status="nudged_loop"
        reqs = client.get("/api/requests?status=nudged").json()["requests"]
        assert len(reqs) == 2
        assert all(row["status"] == "nudged_loop" for row in reqs)

        metrics = client.get("/api/stats").json()
        assert metrics["nudged_loop_count"] == 2
        assert metrics["all_time_nudged_loops"] == 2
        assert metrics["on_trip"] == "nudge"


def test_consecutive_trip_escalates_to_429(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify second consecutive violation escalates to HTTP 429, while valid recovery clears nudge state."""
    settings = TokenGuardConfig(
        db_path=tmp_path / "nudge_escalate.db",
        max_tool_repeats=3,
        on_trip="nudge",
    )
    set_settings(settings)
    app = create_app(settings)

    with TestClient(app) as client:
        _mock_openai_upstream(monkeypatch)

        # Scenario A: Agent ignores the nudge on the next turn -> hard HTTP 429 escalation
        headers_a = {"x-session-id": "sess-stubborn-agent"}
        history_a = [{"role": "user", "content": "Find missing file"}]
        for i in range(1, 3):
            history_a = _append_openai_tool_step(history_a, i, "read_file", f"missing_{i}.txt")
            assert client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": history_a}, headers=headers_a).status_code == 200

        # 1st violation -> 200 OK nudge
        history_a = _append_openai_tool_step(history_a, 3, "read_file", "missing_3.txt")
        first_trip = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": history_a}, headers=headers_a)
        assert first_trip.status_code == 200
        assert first_trip.headers.get("X-TokenGuard-Action") == "nudged"
        assert proxy_module.circuit_breaker._session_nudged.get("sess-stubborn-agent") is True

        # Agent framework appends synthetic assistant message, then still calls read_file again!
        nudge_msg = first_trip.json()["choices"][0]["message"]
        history_a = list(history_a) + [nudge_msg, {"role": "user", "content": "Keep trying"}]
        history_a = _append_openai_tool_step(history_a, 4, "read_file", "missing_4.txt")

        second_trip = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": history_a}, headers=headers_a)
        assert second_trip.status_code == 429
        assert "ToolRetryCeilingTripped" in second_trip.headers.get("X-TokenGuard-Reason", "")
        assert "sess-stubborn-agent" not in proxy_module.circuit_breaker._session_nudged

        # Scenario B: Agent heeds the nudge and switches to a different valid tool -> clears _session_nudged
        headers_b = {"x-session-id": "sess-recovering-agent"}
        history_b = [{"role": "user", "content": "Find missing file"}]
        for i in range(1, 3):
            history_b = _append_openai_tool_step(history_b, i, "read_file", f"a_{i}.txt")
            assert client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": history_b}, headers=headers_b).status_code == 200

        history_b = _append_openai_tool_step(history_b, 3, "read_file", "a_3.txt")
        nudge_b = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": history_b}, headers=headers_b)
        assert nudge_b.status_code == 200
        assert nudge_b.headers.get("X-TokenGuard-Action") == "nudged"
        assert proxy_module.circuit_breaker._session_nudged.get("sess-recovering-agent") is True

        # Next turn: agent alters strategy and calls write_file instead -> 200 OK upstream, nudge flag cleared
        history_b = list(history_b) + [nudge_b.json()["choices"][0]["message"]]
        history_b = _append_openai_tool_step(history_b, 4, "write_file", "summary.md")
        recovery_resp = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": history_b}, headers=headers_b)
        assert recovery_resp.status_code == 200
        assert recovery_resp.headers.get("X-TokenGuard-Action") is None
        assert recovery_resp.json()["id"] == "chatcmpl-upstream-ok"
        assert "sess-recovering-agent" not in proxy_module.circuit_breaker._session_nudged


def test_nudge_streaming_sse_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify streaming requests ('stream': true) receive valid chunked SSE payloads ending with [DONE]."""
    settings = TokenGuardConfig(
        db_path=tmp_path / "nudge_stream.db",
        max_tool_repeats=2,
        on_trip="nudge",
    )
    set_settings(settings)
    app = create_app(settings)

    with TestClient(app) as client:
        # 1. OpenAI SSE stream
        _mock_openai_upstream(monkeypatch)
        headers_oai = {"x-session-id": "sess-stream-oai"}
        hist = _append_openai_tool_step([{"role": "user", "content": "Start"}], 1, "read_file", "x.py")
        assert client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": hist}, headers=headers_oai).status_code == 200

        hist = _append_openai_tool_step(hist, 2, "read_file", "y.py")
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": hist, "stream": True},
            headers=headers_oai,
        ) as stream_resp:
            assert stream_resp.status_code == 200
            assert "text/event-stream" in stream_resp.headers.get("content-type", "")
            assert stream_resp.headers.get("X-TokenGuard-Action") == "nudged"
            assert "ToolRetryCeilingTripped" in stream_resp.headers.get("X-TokenGuard-Reason", "")
            raw_sse = stream_resp.read().decode("utf-8")

        assert raw_sse.endswith("data: [DONE]\n\n")
        data_lines = [line[len("data: "):] for line in raw_sse.splitlines() if line.startswith("data: ")]
        assert data_lines[-1] == "[DONE]"
        parsed_chunks = [json.loads(item) for item in data_lines[:-1]]
        assert len(parsed_chunks) >= 1
        combined_text = "".join(
            ch["choices"][0].get("delta", {}).get("content", "")
            for ch in parsed_chunks
            if ch.get("choices")
        )
        assert "[TokenGuard Alert] Execution loop detected" in combined_text
        assert any(ch["choices"][0].get("finish_reason") == "stop" for ch in parsed_chunks if ch.get("choices"))

        # 2. Anthropic SSE stream
        _mock_anthropic_upstream(monkeypatch)
        headers_anth = {"x-session-id": "sess-stream-anth"}
        anth_hist = _append_anthropic_tool_step([{"role": "user", "content": "Start"}], 1, "read_file", "x.py")
        assert client.post("/v1/messages", json={"model": "claude-3-5-sonnet-20241022", "messages": anth_hist}, headers=headers_anth).status_code == 200

        anth_hist = _append_anthropic_tool_step(anth_hist, 2, "read_file", "y.py")
        with client.stream(
            "POST",
            "/v1/messages",
            json={"model": "claude-3-5-sonnet-20241022", "messages": anth_hist, "stream": True},
            headers=headers_anth,
        ) as anth_stream:
            assert anth_stream.status_code == 200
            assert "text/event-stream" in anth_stream.headers.get("content-type", "")
            assert anth_stream.headers.get("X-TokenGuard-Action") == "nudged"
            anth_raw_sse = anth_stream.read().decode("utf-8")

        assert anth_raw_sse.endswith("data: [DONE]\n\n")
        anth_data_lines = [line[len("data: "):] for line in anth_raw_sse.splitlines() if line.startswith("data: ")]
        assert anth_data_lines[-1] == "[DONE]"
        anth_events = [json.loads(item) for item in anth_data_lines[:-1]]
        anth_text = "".join(
            ev.get("delta", {}).get("text", "")
            for ev in anth_events
            if ev.get("type") == "content_block_delta"
        )
        assert "[TokenGuard Alert] Execution loop detected" in anth_text


def test_cli_and_simulate_nudge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Verify --on-trip flag, TOKENGUARD_ON_TRIP env var, LRU session bound, and simulate --type nudge."""
    monkeypatch.setenv("TOKENGUARD_ON_TRIP", "nudge")
    env_settings = Settings(db_path=tmp_path / "env.db")
    assert env_settings.on_trip == "nudge"

    parser = create_parser()
    parsed_start = parser.parse_args(["start", "--on-trip", "nudge"])
    assert parsed_start.on_trip == "nudge"

    parsed_sim = parser.parse_args(["simulate", "--type", "nudge", "--port", "59999"])
    assert parsed_sim.type == "nudge"
    cmd_simulate(parsed_sim)
    captured = capsys.readouterr().out
    assert "Loop Nudge" in captured
    assert "429" in captured

    # Verify /api/simulate --type nudge endpoint
    app = create_app(Settings(db_path=tmp_path / "sim_api.db", admin_token_required=False))
    with TestClient(app) as client:
        sim_res = client.post("/api/simulate", json={"type": "nudge", "model": "gpt-4o"})
        assert sim_res.status_code == 200
        sim_body = sim_res.json()
        assert sim_body["simulation"] == "nudge"
        assert sim_body["nudged"] is True
        assert sim_body["escalated"] is True
        statuses = [item["status"] for item in sim_body["results"]]
        assert "nudged_loop" in statuses
        assert "blocked_tool_retry_ceiling" in statuses

    # Verify LRU eviction cleans up _session_nudged
    cb = CircuitBreaker(max_active_sessions=2, on_trip="nudge", max_tool_repeats=1, enable_notifications=False)
    for sid in ("s1", "s2", "s3"):
        with pytest.raises(Exception) as exc_info:
            cb.check_tool_retry_ceiling(sid, "read_file")
        assert cb.evaluate_trip_action(sid, exc_info.value) == "nudge"
    assert len(cb._session_nudged) <= 2
    assert "s1" not in cb._session_nudged
