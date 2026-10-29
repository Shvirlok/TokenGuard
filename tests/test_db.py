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

    async with aiosqlite.connect(db_file) as db:
        async with db.execute("PRAGMA journal_mode;") as cursor:
            row = await cursor.fetchone()
            assert row[0].lower() == "wal"

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

    metrics = await get_metrics(db_path=db_file, time_window_hours=1.0)
    assert metrics["total_requests"] == 2
    assert metrics["success_requests"] == 1
    assert metrics["blocked_requests"] == 1
    assert metrics["blocked_loop_count"] == 1
    assert pytest.approx(metrics["total_spent"], 1e-6) == 0.000875
    assert metrics["total_tokens"] == 200
    assert metrics["avg_latency_ms"] == 230.0
    assert metrics["saved_cost_estimate"] == 0.0054

    recent = await get_recent_requests(db_path=db_file, limit=10)
    assert len(recent) == 2
    assert recent[0]["id"] == req_id2
    assert recent[0]["status"] == "blocked_loop"
    assert recent[1]["id"] == req_id1
    assert recent[1]["status"] == "success"

    blocked_only = await get_recent_requests(db_path=db_file, status_filter="blocked")
    assert len(blocked_only) == 1
    assert blocked_only[0]["status"] == "blocked_loop"

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

    await log_request(
        db_path=db_file,
        model="deepseek-chat",
        prompt_tokens=200,
        completion_tokens=0,
        cost_usd=0.0,
        status="blocked_loop",
        blocked_reason="TokenGuard: infinite agent loop detected",
    )

    metrics = await get_metrics(db_path=db_file)
    assert metrics["blocked_requests"] == 4
    assert metrics["blocked_loop_count"] == 4
    assert metrics["total_spent"] == 0.0
    assert metrics["saved_cost_estimate"] == 0.0159


@pytest.mark.asyncio
async def test_db_session_id_column_and_index(tmp_path: Path):
    """Verify session_id TEXT column and idx_requests_session_id index in requests table."""
    from tokenguard.db import get_all_requests_for_export

    db_file = tmp_path / "test_session_id.db"
    await init_db(db_file)

    async with aiosqlite.connect(db_file) as db:
        async with db.execute("PRAGMA table_info(requests);") as cursor:
            cols = await cursor.fetchall()
            col_names = [c[1] for c in cols]
            assert "session_id" in col_names

        async with db.execute("PRAGMA index_list(requests);") as cursor:
            indices = await cursor.fetchall()
            index_names = [idx[1] for idx in indices]
            assert "idx_requests_session_id" in index_names

        async with db.execute("PRAGMA index_list(requests);") as cursor:
            indices = await cursor.fetchall()
            index_names = [idx[1] for idx in indices]
            assert "idx_requests_session_id" in index_names

    id1 = await log_request(
        db_path=db_file,
        model="gpt-4o",
        session_id="agent-run-123",
        status="success",
    )
    id2 = await log_request(
        db_path=db_file,
        model="gpt-4o",
        status="success",
    )

    recent = await get_recent_requests(db_path=db_file)
    assert recent[0]["id"] == id2
    assert recent[0]["session_id"] is None
    assert recent[1]["id"] == id1
    assert recent[1]["session_id"] == "agent-run-123"

    all_exported = await get_all_requests_for_export(db_path=db_file)
    assert all_exported[0]["session_id"] == "agent-run-123"
    assert all_exported[1]["session_id"] == ""


@pytest.mark.asyncio
async def test_db_migration_adds_session_id(tmp_path: Path):
    """Verify init_db gracefully alters an existing legacy requests table lacking session_id."""
    db_file = tmp_path / "test_legacy.db"

    async with aiosqlite.connect(db_file) as db:
        await db.execute("""
            CREATE TABLE requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                latency_ms INTEGER DEFAULT 0,
                prompt_hash TEXT,
                status TEXT NOT NULL,
                blocked_reason TEXT
            );
        """)
        await db.commit()

    await init_db(db_file)

    async with aiosqlite.connect(db_file) as db:
        async with db.execute("PRAGMA table_info(requests);") as cursor:
            cols = await cursor.fetchall()
            col_names = [c[1] for c in cols]
            assert "session_id" in col_names

        async with db.execute("PRAGMA index_list(requests);") as cursor:
            indices = await cursor.fetchall()
            index_names = [idx[1] for idx in indices]
            assert "idx_requests_session_id" in index_names

