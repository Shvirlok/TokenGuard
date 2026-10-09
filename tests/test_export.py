"""Tests for telemetry export functionality (JSONL, HAR 1.2, JSON, CSV)."""

import json
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from tokenguard.db import init_db, log_request
from tokenguard.export import (
    export_telemetry_async,
    format_csv_export,
    format_har_export,
    format_json_export,
    format_jsonl_export,
)
from tokenguard.cli import create_parser, cmd_export
from tokenguard.config import Settings, set_settings
from tokenguard.proxy import app


@pytest.fixture
async def sample_db(tmp_path: Path):
    """Create and seed a temporary SQLite database with test requests."""
    db_file = tmp_path / "test_export.db"
    await init_db(db_file)

    # 1. Successful request
    await log_request(
        db_path=db_file,
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.00075,
        latency_ms=120,
        prompt_hash="hash_success_1",
        status="success",
        blocked_reason=None,
    )

    # 2. Blocked loop request (429)
    await log_request(
        db_path=db_file,
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=0,
        cost_usd=0.0,
        latency_ms=5,
        prompt_hash="hash_loop_1",
        status="blocked_loop",
        blocked_reason="infinite_loop_detected (3 identical consecutive requests)",
    )

    # 3. Blocked budget request (429)
    await log_request(
        db_path=db_file,
        model="deepseek-chat",
        prompt_tokens=500,
        completion_tokens=0,
        cost_usd=0.0,
        latency_ms=2,
        prompt_hash="hash_budget_1",
        status="blocked_budget",
        blocked_reason="hourly_budget_exceeded",
    )

    return db_file


@pytest.mark.asyncio
async def test_format_har_export():
    """Verify HAR 1.2 schema structure and custom telemetry fields."""
    records = [
        {
            "id": 1,
            "timestamp": "2026-10-09T12:00:00Z",
            "model": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost_usd": 0.00075,
            "latency_ms": 120,
            "prompt_hash": "hash123",
            "status": "success",
            "blocked_reason": None,
        },
        {
            "id": 2,
            "timestamp": "2026-10-09T12:01:00Z",
            "model": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 0,
            "total_tokens": 100,
            "cost_usd": 0.0,
            "latency_ms": 10,
            "prompt_hash": "hash123",
            "status": "blocked_loop",
            "blocked_reason": "infinite_loop_detected",
        },
    ]

    har = format_har_export(records, host="127.0.0.1", port=8080)

    assert "log" in har
    assert har["log"]["version"] == "1.2"
    assert har["log"]["creator"]["name"] == "TokenGuard"
    assert len(har["log"]["entries"]) == 2

    # Verify entry 1 (success)
    e1 = har["log"]["entries"][0]
    assert e1["startedDateTime"] == "2026-10-09T12:00:00Z"
    assert e1["time"] == 120
    assert e1["request"]["method"] == "POST"
    assert e1["request"]["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert e1["response"]["status"] == 200
    assert e1["response"]["statusText"] == "OK"
    assert e1["_tokenCost"] == 0.00075
    assert e1["_blockedReason"] is None
    assert e1["_model"] == "gpt-4o"
    assert e1["_promptTokens"] == 100
    assert e1["_completionTokens"] == 50
    assert e1["_totalTokens"] == 150

    # Verify entry 2 (blocked)
    e2 = har["log"]["entries"][1]
    assert e2["response"]["status"] == 429
    assert e2["response"]["statusText"] == "Too Many Requests"
    assert e2["_tokenCost"] == 0.0
    assert e2["_blockedReason"] == "infinite_loop_detected"
    assert e2["_tokenGuardStatus"] == "blocked_loop"


@pytest.mark.asyncio
async def test_format_jsonl_export():
    """Verify JSONL format creates valid newline-delimited JSON with custom fields."""
    records = [
        {
            "id": 1,
            "timestamp": "2026-10-09T12:00:00Z",
            "model": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost_usd": 0.00075,
            "latency_ms": 120,
            "prompt_hash": "hash123",
            "status": "success",
            "blocked_reason": None,
        }
    ]

    jsonl_output = format_jsonl_export(records)
    lines = [line for line in jsonl_output.strip().split("\n") if line]
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["model"] == "gpt-4o"
    assert parsed["_tokenCost"] == 0.00075
    assert parsed["_blockedReason"] is None


@pytest.mark.asyncio
async def test_export_telemetry_async_formats(sample_db: Path, tmp_path: Path):
    """Test exporting to JSONL and HAR files on disk."""
    # 1. Export JSONL
    out_jsonl = tmp_path / "custom_export.jsonl"
    target, count, content = await export_telemetry_async(
        db_path=sample_db,
        format_type="jsonl",
        output_path=out_jsonl,
    )
    assert target == out_jsonl
    assert count == 3
    assert out_jsonl.exists()
    lines = [json.loads(l) for l in out_jsonl.read_text().strip().split("\n")]
    assert len(lines) == 3

    # 2. Export HAR
    out_har = tmp_path / "custom_export.har"
    target_har, count_har, content_har = await export_telemetry_async(
        db_path=sample_db,
        format_type="har",
        output_path=out_har,
    )
    assert target_har == out_har
    assert count_har == 3
    assert out_har.exists()
    har_data = json.loads(out_har.read_text())
    assert har_data["log"]["version"] == "1.2"
    assert len(har_data["log"]["entries"]) == 3

    # 3. Export with status filter
    target_filtered, count_filtered, _ = await export_telemetry_async(
        db_path=sample_db,
        format_type="jsonl",
        output_path=tmp_path / "blocked_only.jsonl",
        status_filter="blocked",
    )
    assert count_filtered == 2


@pytest.mark.asyncio
async def test_cli_export_subcommand(sample_db: Path, tmp_path: Path):
    """Test CLI parser and cmd_export execution."""
    parser = create_parser()

    # Test parser
    args = parser.parse_args(["export", "--format", "har", "--output", str(tmp_path / "cli_test.har"), "--db-path", str(sample_db)])
    assert args.command == "export"
    assert args.format == "har"
    assert args.output == tmp_path / "cli_test.har"

    # Execute cmd_export
    cmd_export(args)
    assert (tmp_path / "cli_test.har").exists()
    har_obj = json.loads((tmp_path / "cli_test.har").read_text())
    assert len(har_obj["log"]["entries"]) == 3


@pytest.mark.asyncio
async def test_proxy_api_export_endpoints(sample_db: Path):
    """Test FastAPI /api/export endpoint supporting JSONL and HAR formats."""
    settings = Settings(db_path=sample_db)
    set_settings(settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. JSONL Export
        res_jsonl = await client.get("/api/export?format=jsonl")
        assert res_jsonl.status_code == 200
        assert "attachment" in res_jsonl.headers.get("content-disposition", "")
        lines = [l for l in res_jsonl.text.strip().split("\n") if l]
        assert len(lines) == 3

        # 2. HAR Export
        res_har = await client.get("/api/export?format=har")
        assert res_har.status_code == 200
        har_data = res_har.json()
        assert har_data["log"]["version"] == "1.2"
        assert len(har_data["log"]["entries"]) == 3

        # 3. Filtered Export
        res_filtered = await client.get("/api/export?format=jsonl&status=blocked")
        assert res_filtered.status_code == 200
        lines_filtered = [l for l in res_filtered.text.strip().split("\n") if l]
        assert len(lines_filtered) == 2
