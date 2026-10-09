"""Telemetry export module for TokenGuard supporting JSONL, HAR 1.2, JSON, and CSV."""

from __future__ import annotations

import csv
import datetime
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tokenguard.db import get_all_requests_for_export


def format_har_export(
    records: List[Dict[str, Any]],
    host: str = "127.0.0.1",
    port: int = 8080,
) -> Dict[str, Any]:
    """Format request records into a valid HTTP Archive (HAR) 1.2 schema.
    
    Compatible with Postman, Chrome DevTools, and Charles Proxy.
    Embeds custom fields `_tokenCost`, `_blockedReason`, and token breakdown.
    """
    entries = []
    for rec in records:
        ts = rec.get("timestamp")
        if not ts:
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        elif not (ts.endswith("Z") or "+" in ts or "-" in ts[10:]):
            ts = f"{ts}Z"

        latency_ms = int(rec.get("latency_ms") or 0)
        status_str = str(rec.get("status", "success"))
        is_blocked = status_str.startswith("blocked")
        http_status = 429 if is_blocked else 200
        http_status_text = "Too Many Requests" if is_blocked else "OK"
        model = rec.get("model", "default")
        cost = float(rec.get("cost_usd") or 0.0)
        blocked_reason = rec.get("blocked_reason") or None
        prompt_tokens = int(rec.get("prompt_tokens") or 0)
        completion_tokens = int(rec.get("completion_tokens") or 0)
        prompt_hash = rec.get("prompt_hash") or ""

        post_body = json.dumps({
            "model": model,
            "prompt_hash": prompt_hash,
            "estimated_prompt_tokens": prompt_tokens,
        })
        resp_body = json.dumps({
            "status": status_str,
            "blocked_reason": blocked_reason,
            "model": model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "cost_usd": cost,
        })

        entry = {
            "startedDateTime": ts,
            "time": latency_ms,
            "request": {
                "method": "POST",
                "url": f"http://{host}:{port}/v1/chat/completions",
                "httpVersion": "HTTP/1.1",
                "cookies": [],
                "headers": [
                    {"name": "Host", "value": f"{host}:{port}"},
                    {"name": "Content-Type", "value": "application/json"},
                    {"name": "Accept", "value": "application/json"},
                    {"name": "User-Agent", "value": "TokenGuard-Proxy/0.1.0"},
                ],
                "queryString": [],
                "postData": {
                    "mimeType": "application/json",
                    "text": post_body,
                },
                "headersSize": -1,
                "bodySize": len(post_body.encode("utf-8")),
            },
            "response": {
                "status": http_status,
                "statusText": http_status_text,
                "httpVersion": "HTTP/1.1",
                "cookies": [],
                "headers": [
                    {"name": "Content-Type", "value": "application/json"},
                    {"name": "X-TokenGuard-Status", "value": status_str},
                    {"name": "X-TokenGuard-Cost", "value": f"{cost:.6f}"},
                ],
                "content": {
                    "size": len(resp_body.encode("utf-8")),
                    "mimeType": "application/json",
                    "text": resp_body,
                },
                "redirectURL": "",
                "headersSize": -1,
                "bodySize": len(resp_body.encode("utf-8")),
            },
            "cache": {},
            "timings": {
                "blocked": -1,
                "dns": -1,
                "connect": -1,
                "send": 0,
                "wait": latency_ms,
                "receive": 0,
                "ssl": -1,
            },
            "_tokenCost": cost,
            "_blockedReason": blocked_reason,
            "_model": model,
            "_promptTokens": prompt_tokens,
            "_completionTokens": completion_tokens,
            "_totalTokens": prompt_tokens + completion_tokens,
            "_promptHash": prompt_hash,
            "_tokenGuardStatus": status_str,
        }
        entries.append(entry)

    return {
        "log": {
            "version": "1.2",
            "creator": {
                "name": "TokenGuard",
                "version": "0.1.0",
                "comment": "TokenGuard LLM Circuit Breaker & Telemetry Export",
            },
            "pages": [],
            "entries": entries,
        }
    }


def format_jsonl_export(records: List[Dict[str, Any]]) -> str:
    """Format request records into newline-delimited JSON (JSONL)."""
    lines = []
    for rec in records:
        entry = dict(rec)
        entry["_tokenCost"] = float(rec.get("cost_usd") or 0.0)
        entry["_blockedReason"] = rec.get("blocked_reason") or None
        lines.append(json.dumps(entry))
    return "\n".join(lines) + ("\n" if lines else "")


def format_json_export(records: List[Dict[str, Any]]) -> str:
    """Format request records into structured JSON."""
    enriched = []
    for rec in records:
        entry = dict(rec)
        entry["_tokenCost"] = float(rec.get("cost_usd") or 0.0)
        entry["_blockedReason"] = rec.get("blocked_reason") or None
        enriched.append(entry)
    return json.dumps({"requests": enriched, "count": len(enriched)}, indent=2)


def format_csv_export(records: List[Dict[str, Any]]) -> str:
    """Format request records into standard CSV."""
    output = io.StringIO()
    fieldnames = [
        "id",
        "timestamp",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "latency_ms",
        "prompt_hash",
        "status",
        "blocked_reason",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for rec in records:
        filtered = {k: rec.get(k, "") for k in fieldnames}
        writer.writerow(filtered)
    return output.getvalue()


async def export_telemetry_async(
    db_path: Union[str, Path],
    format_type: str = "jsonl",
    output_path: Optional[Union[str, Path]] = None,
    status_filter: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> Tuple[Path, int, str]:
    """Export request telemetry directly from SQLite WAL database.
    
    Returns:
        (resolved_output_path, record_count, formatted_content)
    """
    records = await get_all_requests_for_export(db_path, status_filter=status_filter)
    fmt = format_type.strip().lower()

    if fmt == "har":
        har_obj = format_har_export(records, host=host, port=port)
        content = json.dumps(har_obj, indent=2)
        ext = "har"
    elif fmt == "json":
        content = format_json_export(records)
        ext = "json"
    elif fmt == "csv":
        content = format_csv_export(records)
        ext = "csv"
    else:
        # Default to jsonl
        content = format_jsonl_export(records)
        ext = "jsonl"
        fmt = "jsonl"

    if output_path is not None:
        target = Path(output_path)
        if target.is_dir():
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            target = target / f"tokenguard_export_{ts_str}.{ext}"
    else:
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target = Path(f"tokenguard_export_{ts_str}.{ext}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    return target, len(records), content


def export_telemetry_sync(
    db_path: Union[str, Path],
    format_type: str = "jsonl",
    output_path: Optional[Union[str, Path]] = None,
    status_filter: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> Tuple[Path, int, str]:
    """Synchronous wrapper for export_telemetry_async, safe to call from any context."""
    import asyncio
    import concurrent.futures

    coro = export_telemetry_async(
        db_path=db_path,
        format_type=format_type,
        output_path=output_path,
        status_filter=status_filter,
        host=host,
        port=port,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)

