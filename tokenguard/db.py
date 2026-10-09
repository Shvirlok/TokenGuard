"""Database module for TokenGuard using aiosqlite in WAL mode."""

from __future__ import annotations

from contextlib import asynccontextmanager
import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import aiosqlite

from tokenguard.guard import calculate_avoided_cost


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS requests (
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

CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);
CREATE INDEX IF NOT EXISTS idx_requests_prompt_hash ON requests(prompt_hash);
"""


@asynccontextmanager
async def get_db(db_path: Union[str, Path]) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Create a configured SQLite connection context with WAL mode enabled."""
    db_path = Path(db_path)
    if db_path.parent:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode = WAL;")
        await db.execute("PRAGMA synchronous = NORMAL;")
        await db.execute("PRAGMA busy_timeout = 5000;")
        yield db


@asynccontextmanager
async def get_readonly_db(db_path: Union[str, Path]) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Create a read-only SQLite connection context in WAL mode without blocking writers."""
    db_path = Path(db_path)
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode = WAL;")
        await db.execute("PRAGMA query_only = ON;")
        await db.execute("PRAGMA busy_timeout = 5000;")
        yield db


async def init_db(db_path: Union[str, Path]) -> None:
    """Initialize SQLite database tables and indexes."""
    async with get_db(db_path) as db:
        await db.executescript(SCHEMA_SQL)
        await db.commit()


async def log_request(
    db_path: Union[str, Path],
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
    prompt_hash: Optional[str] = None,
    status: str = "success",
    blocked_reason: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> int:
    """Log an LLM request or blocked intercept to SQLite.

    Returns the inserted row ID.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    async with get_db(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO requests (
                timestamp, model, prompt_tokens, completion_tokens,
                cost_usd, latency_ms, prompt_hash, status, blocked_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                model,
                prompt_tokens,
                completion_tokens,
                cost_usd,
                latency_ms,
                prompt_hash,
                status,
                blocked_reason,
            ),
        )
        await db.commit()
        return cursor.lastrowid or 0


async def get_metrics(
    db_path: Union[str, Path],
    time_window_hours: float = 1.0,
    prices: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Calculate aggregated metrics over a sliding time window (default 1 hour)."""
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(hours=time_window_hours)
    ).isoformat()

    async with get_db(db_path) as db:
        # Metrics for the sliding window
        window_query = """
            SELECT
                COUNT(*) AS total_requests,
                COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) AS success_requests,
                COALESCE(SUM(CASE WHEN status LIKE 'blocked%' THEN 1 ELSE 0 END), 0) AS blocked_requests,
                COALESCE(SUM(CASE WHEN status = 'blocked_budget' THEN 1 ELSE 0 END), 0) AS blocked_budget_count,
                COALESCE(SUM(CASE WHEN status = 'blocked_loop' THEN 1 ELSE 0 END), 0) AS blocked_loop_count,
                COALESCE(SUM(CASE WHEN status = 'blocked_killswitch' THEN 1 ELSE 0 END), 0) AS blocked_killswitch_count,
                COALESCE(SUM(CASE WHEN status = 'success' THEN cost_usd ELSE 0.0 END), 0.0) AS total_spent,
                COALESCE(SUM(CASE WHEN status = 'success' THEN prompt_tokens ELSE 0 END), 0) AS total_prompt_tokens,
                COALESCE(SUM(CASE WHEN status = 'success' THEN completion_tokens ELSE 0 END), 0) AS total_completion_tokens,
                COALESCE(AVG(CASE WHEN status = 'success' AND latency_ms > 0 THEN latency_ms ELSE NULL END), 0.0) AS avg_latency_ms
            FROM requests
            WHERE timestamp >= ?
        """
        async with db.execute(window_query, (cutoff,)) as cursor:
            row = await cursor.fetchone()
            window_data = dict(row) if row else {}

        # All-time aggregated counters
        all_time_query = """
            SELECT
                COUNT(*) AS all_time_total_requests,
                COALESCE(SUM(CASE WHEN status = 'success' THEN cost_usd ELSE 0.0 END), 0.0) AS all_time_spent,
                COALESCE(SUM(CASE WHEN status = 'success' THEN prompt_tokens + completion_tokens ELSE 0 END), 0) AS all_time_tokens,
                COALESCE(SUM(CASE WHEN status = 'blocked_loop' THEN 1 ELSE 0 END), 0) AS all_time_blocked_loops,
                COALESCE(SUM(CASE WHEN status = 'blocked_budget' THEN 1 ELSE 0 END), 0) AS all_time_blocked_budget,
                COALESCE(SUM(CASE WHEN status LIKE 'blocked%' THEN 1 ELSE 0 END), 0) AS all_time_blocked_total
            FROM requests
        """
        async with db.execute(all_time_query) as cursor:
            all_time_row = await cursor.fetchone()
            all_time_data = dict(all_time_row) if all_time_row else {}

        # Calculate estimated savings for blocked requests (e.g. blocked loops, budget overruns, kill switch)
        # Based on the model's pricing: prompt tokens + average avoided generation of ~500 completion tokens
        blocked_rows_query = """
            SELECT model, prompt_tokens, status
            FROM requests
            WHERE status LIKE 'blocked%'
        """
        async with db.execute(blocked_rows_query) as cursor:
            blocked_rows = await cursor.fetchall()

        total_saved_cost = 0.0
        for b_row in blocked_rows:
            p_tokens = b_row["prompt_tokens"] or 0
            model_name = b_row["model"] or "default"
            total_saved_cost += calculate_avoided_cost(
                model=model_name,
                prompt_tokens=p_tokens,
                avoided_completion_tokens=500,
                prices=prices,
            )

        saved_cost_estimate = round(total_saved_cost, 4)

        return {
            "time_window_hours": time_window_hours,
            "total_requests": window_data.get("total_requests", 0),
            "success_requests": window_data.get("success_requests", 0),
            "blocked_requests": window_data.get("blocked_requests", 0),
            "blocked_budget_count": window_data.get("blocked_budget_count", 0),
            "blocked_loop_count": window_data.get("blocked_loop_count", 0),
            "blocked_killswitch_count": window_data.get("blocked_killswitch_count", 0),
            "total_spent": round(window_data.get("total_spent", 0.0), 6),
            "total_prompt_tokens": window_data.get("total_prompt_tokens", 0),
            "total_completion_tokens": window_data.get("total_completion_tokens", 0),
            "total_tokens": window_data.get("total_prompt_tokens", 0) + window_data.get("total_completion_tokens", 0),
            "avg_latency_ms": round(window_data.get("avg_latency_ms", 0.0), 1),
            "all_time_spent": round(all_time_data.get("all_time_spent", 0.0), 6),
            "all_time_tokens": all_time_data.get("all_time_tokens", 0),
            "all_time_blocked_loops": all_time_data.get("all_time_blocked_loops", 0),
            "all_time_blocked_budget": all_time_data.get("all_time_blocked_budget", 0),
            "all_time_blocked_total": all_time_data.get("all_time_blocked_total", 0),
            "saved_cost_estimate": saved_cost_estimate,
        }


async def get_recent_requests(
    db_path: Union[str, Path],
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve the most recent requests for UI and inspection."""
    async with get_db(db_path) as db:
        if status_filter and status_filter != "all":
            if status_filter == "blocked":
                query = """
                    SELECT id, timestamp, model, prompt_tokens, completion_tokens,
                           cost_usd, latency_ms, prompt_hash, status, blocked_reason
                    FROM requests
                    WHERE status LIKE 'blocked%'
                    ORDER BY id DESC
                    LIMIT ? OFFSET ?
                """
                params = (limit, offset)
            else:
                query = """
                    SELECT id, timestamp, model, prompt_tokens, completion_tokens,
                           cost_usd, latency_ms, prompt_hash, status, blocked_reason
                    FROM requests
                    WHERE status = ?
                    ORDER BY id DESC
                    LIMIT ? OFFSET ?
                """
                params = (status_filter, limit, offset)
        else:
            query = """
                SELECT id, timestamp, model, prompt_tokens, completion_tokens,
                       cost_usd, latency_ms, prompt_hash, status, blocked_reason
                FROM requests
                ORDER BY id DESC
                LIMIT ? OFFSET ?
            """
            params = (limit, offset)

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "model": row["model"],
                    "prompt_tokens": row["prompt_tokens"],
                    "completion_tokens": row["completion_tokens"],
                    "total_tokens": row["prompt_tokens"] + row["completion_tokens"],
                    "cost_usd": round(row["cost_usd"], 6),
                    "latency_ms": row["latency_ms"],
                    "prompt_hash": row["prompt_hash"],
                    "status": row["status"],
                    "blocked_reason": row["blocked_reason"],
                }
                for row in rows
            ]


async def get_spend_in_window(
    db_path: Union[str, Path],
    hours: float = 1.0,
) -> float:
    """Calculate total USD spend in the sliding window of N hours."""
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(hours=hours)
    ).isoformat()

    async with get_db(db_path) as db:
        query = """
            SELECT COALESCE(SUM(cost_usd), 0.0) AS spend
            FROM requests
            WHERE status = 'success' AND timestamp >= ?
        """
        async with db.execute(query, (cutoff,)) as cursor:
            row = await cursor.fetchone()
            return float(row["spend"]) if row and row["spend"] is not None else 0.0


async def get_hourly_history(
    db_path: Union[str, Path],
    hours: int = 24,
) -> List[Dict[str, Any]]:
    """Get time-bucketed spending and request counts for Chart.js dashboard charts."""
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(hours=hours)
    ).isoformat()

    async with get_db(db_path) as db:
        query = """
            SELECT
                strftime('%Y-%m-%d %H:00', timestamp) AS hour_bucket,
                COUNT(*) AS request_count,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count,
                SUM(CASE WHEN status LIKE 'blocked%' THEN 1 ELSE 0 END) AS blocked_count,
                COALESCE(SUM(CASE WHEN status = 'success' THEN cost_usd ELSE 0.0 END), 0.0) AS spend_usd,
                COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS total_tokens
            FROM requests
            WHERE timestamp >= ?
            GROUP BY hour_bucket
            ORDER BY hour_bucket ASC
        """
        async with db.execute(query, (cutoff,)) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "hour": row["hour_bucket"],
                    "request_count": row["request_count"],
                    "success_count": row["success_count"],
                    "blocked_count": row["blocked_count"],
                    "spend_usd": round(row["spend_usd"], 6),
                    "total_tokens": row["total_tokens"],
                }
                for row in rows
            ]


async def get_all_requests_for_export(
    db_path: Union[str, Path],
    status_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all request records for export without limit using a non-blocking read-only connection."""
    async with get_readonly_db(db_path) as db:
        if status_filter and status_filter != "all":
            if status_filter == "blocked":
                query = """
                    SELECT id, timestamp, model, prompt_tokens, completion_tokens,
                           cost_usd, latency_ms, prompt_hash, status, blocked_reason
                    FROM requests
                    WHERE status LIKE 'blocked%'
                    ORDER BY id ASC
                """
                params: tuple = ()
            else:
                query = """
                    SELECT id, timestamp, model, prompt_tokens, completion_tokens,
                           cost_usd, latency_ms, prompt_hash, status, blocked_reason
                    FROM requests
                    WHERE status = ?
                    ORDER BY id ASC
                """
                params = (status_filter,)
        else:
            query = """
                SELECT id, timestamp, model, prompt_tokens, completion_tokens,
                       cost_usd, latency_ms, prompt_hash, status, blocked_reason
                FROM requests
                ORDER BY id ASC
            """
            params = ()

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "model": row["model"],
                    "prompt_tokens": row["prompt_tokens"],
                    "completion_tokens": row["completion_tokens"],
                    "total_tokens": (row["prompt_tokens"] or 0) + (row["completion_tokens"] or 0),
                    "cost_usd": round(row["cost_usd"], 6),
                    "latency_ms": row["latency_ms"],
                    "prompt_hash": row["prompt_hash"],
                    "status": row["status"],
                    "blocked_reason": row["blocked_reason"] or "",
                }
                for row in rows
            ]


async def clear_all_requests(db_path: Union[str, Path]) -> None:
    """Clear all request log entries from the database."""
    async with get_db(db_path) as db:
        await db.execute("DELETE FROM requests;")
        await db.commit()
