from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncGenerator
import httpx
import pytest
from starlette.testclient import TestClient

from tokenguard.config import Settings, get_or_create_admin_token, set_settings
from tokenguard.db import get_metrics, get_recent_requests, init_db, log_request
from tokenguard.guard import (
    CircuitBreaker,
    ExpiringLRUDict,
    SessionBudgetLimitError,
)
from tokenguard.proxy import StreamLoopDetector, app, create_app
import tokenguard.proxy as proxy_module


@pytest.mark.asyncio
async def test_metrics_breakdown_counts(tmp_path: Path):
    db_file = tmp_path / "metrics_test.db"
    await init_db(db_file)

    await log_request(
        db_path=db_file,
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.001,
        latency_ms=120,
        prompt_hash="hash_success",
        status="success",
    )
    await log_request(
        db_path=db_file,
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=0,
        cost_usd=0.0,
        latency_ms=0,
        prompt_hash="hash_retry",
        status="blocked_tool_retry_ceiling",
        blocked_reason="ToolRetryCeilingTripped: read_file called 3 times consecutively",
    )
    await log_request(
        db_path=db_file,
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=0,
        cost_usd=0.0,
        latency_ms=0,
        prompt_hash="hash_stag",
        status="blocked_stagnation",
        blocked_reason="SessionStagnationTripped: 6 steps without state change",
    )
    await log_request(
        db_path=db_file,
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=0,
        cost_usd=0.0,
        latency_ms=0,
        prompt_hash="hash_secret",
        status="blocked_secret_leak",
        blocked_reason="SecretLeakDetected: Found matching pattern OPENAI_KEY",
    )
    await log_request(
        db_path=db_file,
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=0,
        cost_usd=0.0,
        latency_ms=0,
        prompt_hash="hash_loop",
        status="blocked_loop",
        blocked_reason="Loop detected",
    )

    metrics = await get_metrics(db_file, time_window_hours=1.0)

    assert metrics["total_requests"] == 5
    assert metrics["success_requests"] == 1
    assert metrics["blocked_requests"] == 4
    assert metrics["blocked_retry_ceiling_count"] == 1
    assert metrics["blocked_stagnation_count"] == 1
    assert metrics["blocked_secret_count"] == 1
    assert metrics["blocked_loop_count"] == 4


def test_lru_session_eviction():
    d = ExpiringLRUDict(max_size=3, ttl_seconds=0.1)
    d["s1"] = "val1"
    d["s2"] = "val2"
    d["s3"] = "val3"

    assert d["s1"] == "val1"
    d["s4"] = "val4"

    assert "s2" not in d
    assert "s1" in d
    assert "s3" in d
    assert "s4" in d

    time.sleep(0.12)
    assert "s1" not in d
    assert "s3" not in d
    assert "s4" not in d

    cb = CircuitBreaker(max_active_sessions=2, session_ttl_seconds=0.1)
    cb.check_request([{"role": "user", "content": "1"}], model="gpt-4o", session_id="s1")
    cb.check_request([{"role": "user", "content": "2"}], model="gpt-4o", session_id="s2")
    cb.check_request([{"role": "user", "content": "3"}], model="gpt-4o", session_id="s3")

    assert len(cb._session_last_seen) <= 2
    assert "s1" not in cb._session_last_seen

    time.sleep(0.12)
    cb._evict_idle_sessions()
    assert len(cb._session_last_seen) == 0


def test_per_session_budget_limit(tmp_path: Path, monkeypatch):
    cb = CircuitBreaker(session_limit=0.01, hourly_limit=10.0)

    cb.check_request([{"role": "user", "content": "start"}], model="gpt-4o", session_id="session-a")
    cb.record_success(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=1000,
        cost_usd=0.015,
        session_id="session-a",
    )

    with pytest.raises(SessionBudgetLimitError) as exc_info:
        cb.check_request(
            [{"role": "user", "content": "next task"}],
            model="gpt-4o",
            session_id="session-a",
        )
    assert "SessionBudgetLimitTripped" in exc_info.value.reason
    assert exc_info.value.session_id == "session-a"

    prompt_hash_b = cb.check_request(
        [{"role": "user", "content": "different session"}],
        model="gpt-4o",
        session_id="session-b",
    )
    assert prompt_hash_b is not None
    assert cb.get_session_spend("session-b") == 0.0

    db_file = tmp_path / "session_budget.db"
    settings = Settings(
        session_limit=0.01,
        hourly_limit=10.0,
        db_path=db_file,
    )
    set_settings(settings)
    test_app = create_app()

    async def mock_post(url, *args, **kwargs):
        payload = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        return httpx.Response(200, json=payload)

    with TestClient(test_app) as client:
        guard_inst = proxy_module.get_guard()
        guard_inst.session_limit = 0.01
        guard_inst.record_success(
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=1000,
            cost_usd=0.015,
            session_id="capped-session",
        )

        r_blocked = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Blocked?"}]},
            headers={"X-Session-ID": "capped-session"},
        )
        assert r_blocked.status_code == 429
        assert "SessionBudgetLimitTripped" in r_blocked.headers.get("X-TokenGuard-Reason", "")

        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)
        r_allowed = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Allowed?"}]},
            headers={"X-Session-ID": "fresh-session"},
        )
        assert r_allowed.status_code == 200


@pytest.mark.asyncio
async def test_mid_stream_repetition_abort(tmp_path: Path, monkeypatch):
    detector = StreamLoopDetector(max_repeats=5)
    for _ in range(4):
        tripped = detector.feed(b'data: {"choices":[{"delta":{"content":"loop "}}]}\n\n')
        assert tripped is False

    tripped = detector.feed(b'data: {"choices":[{"delta":{"content":"loop "}}]}\n\n')
    assert tripped is True
    assert detector.repeated_phrase == "loop "

    db_file = tmp_path / "stream_abort.db"
    settings = Settings(db_path=db_file)
    set_settings(settings)
    test_app = create_app()

    chunks_yielded = 0

    async def mock_stream_bytes():
        nonlocal chunks_yielded
        for i in range(12):
            chunks_yielded += 1
            yield f'data: {{"id":"cmpl","choices":[{{"delta":{{"content":"repeat "}}}}]}}\n\n'.encode("utf-8")
        yield b'data: {"id":"cmpl","choices":[{"delta":{"content":"final"}}]}\\n\n'
        yield b"data: [DONE]\n\n"

    mock_resp = httpx.Response(
        200,
        content=mock_stream_bytes(),
        headers={"Content-Type": "text/event-stream"},
    )

    async def mock_send(request, stream=False):
        return mock_resp

    with TestClient(test_app) as client:
        monkeypatch.setattr(proxy_module.http_client, "send", mock_send)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Trigger mid-stream loop"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        text = resp.text
        assert text.count("repeat ") == 4
        assert "final" not in text
        assert chunks_yielded == 5

        records = await get_recent_requests(db_file)
        assert len(records) > 0
        assert records[0]["status"] == "blocked_stream_loop"
        assert "mid_stream_loop_aborted" in records[0]["blocked_reason"]
