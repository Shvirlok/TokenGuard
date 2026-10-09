"""Unit tests for SQLite database schema and logging helpers."""

import pytest
import aiosqlite
from pathlib import Path
from tokenguard.db import (
    init_db,
    log_request,
    get_metrics,
    get_recent_requests,
    get_hourly_history,
)


@pytest.mark.asyncio
async def test_db_init_and_logging(tmp_path: Path):
    db_file = tmp_path / "test_tokenguard.db"
    await init_db(db_file)

    # Verify WAL mode is set
    async with aiosqlite.connect(db_file) as db:
        async with db.execute("PRAGMA journal_mode;") as cursor:
            row = await cursor.fetchone()
            assert row[0].lower() == "wal"

    # Log requests
    req_id1 = await log_request(
        db_path=db_file,
        model="gpt-4o",
        prompt_tokens=150,
        completion_tokens=50,
        cost_usd=0.000875,
        latency_ms=230,
        prompt_hash="abc123hash",
        status="success",
    )
    assert req_id1 > 0

    req_id2 = await log_request(
        db_path=db_file,
        model="gpt-4o",
        prompt_tokens=150,
        completion_tokens=0,
        cost_usd=0.0,
        latency_ms=0,
        prompt_hash="abc123hash",
        status="blocked_loop",
        blocked_reason="TokenGuard: infinite agent loop detected",
    )
    assert req_id2 > req_id1

    # Check metrics
    metrics = await get_metrics(db_path=db_file, time_window_hours=1.0)
    assert metrics["total_requests"] == 2
    assert metrics["success_requests"] == 1
    assert metrics["blocked_requests"] == 1
    assert metrics["blocked_loop_count"] == 1
    assert pytest.approx(metrics["total_spent"], 1e-6) == 0.000875
    assert metrics["total_tokens"] == 200
    assert metrics["avg_latency_ms"] == 230.0
    # Avoided: (150 * 2.50 + 500 * 10.00) / 1,000,000 = 0.005375 -> round 0.0054
    assert metrics["saved_cost_estimate"] == 0.0054

    # Check recent requests
    recent = await get_recent_requests(db_path=db_file, limit=10)
    assert len(recent) == 2
    assert recent[0]["id"] == req_id2
    assert recent[0]["status"] == "blocked_loop"
    assert recent[1]["id"] == req_id1
    assert recent[1]["status"] == "success"

    # Check filtered requests
    blocked_only = await get_recent_requests(db_path=db_file, status_filter="blocked")
    assert len(blocked_only) == 1
    assert blocked_only[0]["status"] == "blocked_loop"

    # Check hourly history
    history = await get_hourly_history(db_path=db_file, hours=24)
    assert len(history) >= 1
    assert history[0]["request_count"] == 2
    assert history[0]["success_count"] == 1
    assert history[0]["blocked_count"] == 1


@pytest.mark.asyncio
async def test_saved_cost_calculation_blocked_loops(tmp_path: Path):
    """Verify saved_cost_estimate calculates model pricing accurately even when no successful requests exist."""
    db_file = tmp_path / "test_saved_cost.db"
    await init_db(db_file)

    # Log 3 blocked loop requests on gpt-4o (100 prompt tokens each)
    for _ in range(3):
        await log_request(
            db_path=db_file,
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=0,
            cost_usd=0.0,
            status="blocked_loop",
            blocked_reason="TokenGuard: infinite agent loop detected",
        )

    # 1 blocked loop on deepseek-chat (200 prompt tokens)
    await log_request(
        db_path=db_file,
        model="deepseek-chat",
        prompt_tokens=200,
        completion_tokens=0,
        cost_usd=0.0,
        status="blocked_loop",
        blocked_reason="TokenGuard: infinite agent loop detected",
    )

    # gpt-4o: 3 * ((100 * 2.50 + 500 * 10.00) / 1,000,000) = 3 * 0.00525 = 0.01575
    # deepseek-chat: 1 * ((200 * 0.14 + 500 * 0.28) / 1,000,000) = 0.000168
    # Total: 0.01575 + 0.000168 = 0.015918 -> round(..., 4) = 0.0159
    metrics = await get_metrics(db_path=db_file)
    assert metrics["blocked_requests"] == 4
    assert metrics["blocked_loop_count"] == 4
    assert metrics["total_spent"] == 0.0
    assert metrics["saved_cost_estimate"] == 0.0159

