"""Security tests for TokenGuard (v0.2.1-sec).

Tests:
1. test_self_proxy_loop_detected (fallback when base URL points to localhost and hop header detection)
2. test_admin_token_protection (/api/kill-switch, /api/config, /api/reset, /api/clear-logs require X-TokenGuard-Admin-Token)
3. test_secret_blocking (block requests containing secrets with HTTP 400 when block_secrets=True)
4. test_secret_redaction (redact secrets in-flight before upstream forwarding when redact_secrets=True)
5. test_payload_too_large (reject payloads larger than max_payload_size_mb with HTTP 413)
6. test_db_file_permissions (enforce 0600 on SQLite database file)
7. test_cors_restrictions (CORS allowed strictly for localhost/127.0.0.1, not malicious origins)
"""

import os
import stat
import pytest
import httpx
from pathlib import Path
from fastapi.testclient import TestClient

from tokenguard.config import Settings, get_or_create_admin_token, set_settings
from tokenguard.db import init_db
from tokenguard.proxy import create_app, resolve_upstream_url
import tokenguard.proxy as proxy_module


def test_self_proxy_loop_detected(monkeypatch):
    """Verify fallback when base URL points to localhost:8080 and hop loop detection."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    url = resolve_upstream_url("openai", endpoint="chat/completions")
    assert "localhost" not in url
    assert url == "https://api.openai.com/v1/chat/completions"

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080")
    url2 = resolve_upstream_url("openai", endpoint="chat/completions")
    assert "127.0.0.1" not in url2
    assert url2 == "https://api.openai.com/v1/chat/completions"

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:8080")
    url3 = resolve_upstream_url("anthropic", endpoint="messages")
    assert "localhost" not in url3
    assert url3 == "https://api.anthropic.com/v1/messages"

    settings = Settings(admin_token_required=False)
    set_settings(settings)
    app = create_app(settings)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            headers={"X-TokenGuard-Hop": "1"},
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 508
        data = resp.json()
        assert data["error"]["type"] == "loop_detected"


def test_admin_token_protection(tmp_path: Path):
    """Verify POST /api/kill-switch and control routes return 401 without token, and 200 with valid token."""
    db_file = tmp_path / "test_admin.db"
    admin_token = "test-secret-token-abcdef1234567890"
    settings = Settings(
        admin_token=admin_token,
        admin_token_required=True,
        db_path=db_file,
    )
    set_settings(settings)
    app = create_app(settings)

    with TestClient(app) as client:
        resp = client.post("/api/kill-switch", json={"active": True})
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "unauthorized"

        resp = client.post(
            "/api/kill-switch",
            json={"active": True},
            headers={"X-TokenGuard-Admin-Token": "invalid-token"},
        )
        assert resp.status_code == 401

        resp = client.post(
            "/api/kill-switch",
            json={"active": True},
            headers={"X-TokenGuard-Admin-Token": admin_token},
        )
        assert resp.status_code == 200
        assert resp.json()["kill_switch_active"] is True

        resp = client.post("/api/config", json={"hourly_limit": 10.0})
        assert resp.status_code == 401
        resp = client.post(
            "/api/config",
            json={"hourly_limit": 10.0},
            headers={"X-TokenGuard-Admin-Token": admin_token},
        )
        assert resp.status_code == 200

        resp = client.post("/api/reset")
        assert resp.status_code == 401
        resp = client.post(
            "/api/reset",
            headers={"X-TokenGuard-Admin-Token": admin_token},
        )
        assert resp.status_code == 200

        resp = client.post("/api/clear-logs")
        assert resp.status_code == 401
        resp = client.post(
            "/api/clear-logs",
            headers={"X-TokenGuard-Admin-Token": admin_token},
        )
        assert resp.status_code == 200


def test_secret_blocking(tmp_path: Path, monkeypatch):
    """Verify request containing secrets is blocked with HTTP 400 when block_secrets is active."""
    db_file = tmp_path / "test_secrets.db"
    settings = Settings(
        block_secrets=True,
        admin_token_required=False,
        db_path=db_file,
    )
    set_settings(settings)
    app = create_app(settings)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Here is key: sk-proj-1234567890123456789012345"}],
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "SecretLeakDetected" in data["error"]["message"]
        assert data["error"]["type"] == "secret_leak_detected"

        resp2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Check sk-ant-api03-1234567890abcdefghijklmnop"}],
            },
        )
        assert resp2.status_code == 400
        assert "SecretLeakDetected" in resp2.json()["error"]["message"]

        resp3 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "AWS config AKIAIOSFODNN7EXAMPLE"}],
            },
        )
        assert resp3.status_code == 400
        assert "SecretLeakDetected" in resp3.json()["error"]["message"]

        resp4 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."}],
            },
        )
        assert resp4.status_code == 400
        assert "SecretLeakDetected" in resp4.json()["error"]["message"]

        async def mock_post(url, json=None, headers=None):
            return httpx.Response(
                status_code=200,
                json={"id": "test", "choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )

        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)
        resp_safe = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Clean prompt without credentials"}],
            },
        )
        assert resp_safe.status_code == 200


def test_secret_redaction(tmp_path: Path, monkeypatch):
    """Verify secrets are redacted in forwarded payload when redact_secrets is active."""
    db_file = tmp_path / "test_redaction.db"
    settings = Settings(
        block_secrets=False,
        redact_secrets=True,
        admin_token_required=False,
        db_path=db_file,
    )
    set_settings(settings)
    app = create_app(settings)

    captured_body = {}

    async def mock_post(url, json=None, headers=None):
        nonlocal captured_body
        captured_body = json
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-redact",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
            },
        )

    with TestClient(app) as client:
        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)
        raw_key = "sk-ant-api03-1234567890abcdefghijklmnop"
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": f"Connect using {raw_key} today"}],
            },
        )
        assert resp.status_code == 200
        assert captured_body is not None
        forwarded_content = captured_body["messages"][0]["content"]
        assert raw_key not in forwarded_content
        assert "***REDACTED_SECRET***" in forwarded_content


def test_payload_too_large(tmp_path: Path):
    """Verify request exceeding max_payload_size_mb returns HTTP 413."""
    db_file = tmp_path / "test_large.db"
    settings = Settings(
        max_payload_size_mb=1.0,
        admin_token_required=False,
        db_path=db_file,
    )
    set_settings(settings)
    app = create_app(settings)

    with TestClient(app) as client:
        large_len = 2 * 1024 * 1024
        resp = client.post(
            "/v1/chat/completions",
            headers={"Content-Length": str(large_len)},
            content=b'{"messages": []}',
        )
        assert resp.status_code == 413
        assert resp.json()["error"]["type"] == "payload_too_large"
        assert "Payload Too Large" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_db_file_permissions(tmp_path: Path):
    """Verify SQLite database file is created with 0600 permissions."""
    db_path = tmp_path / "secure_tokenguard.db"
    await init_db(db_path)
    assert db_path.exists()
    mode = stat.S_IMODE(os.stat(db_path).st_mode)
    assert mode == 0o600


def test_cors_restrictions(tmp_path: Path):
    """Verify CORS allow_origin rejects untrusted external origins and accepts localhost."""
    db_file = tmp_path / "test_cors.db"
    settings = Settings(admin_token_required=False, db_path=db_file)
    set_settings(settings)
    app = create_app(settings)

    with TestClient(app) as client:
        resp_evil = client.get("/health", headers={"Origin": "https://evil.com"})
        assert resp_evil.status_code == 200
        assert resp_evil.headers.get("access-control-allow-origin") != "https://evil.com"
        assert resp_evil.headers.get("access-control-allow-origin") != "*"

        resp_local = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert resp_local.status_code == 200
        assert resp_local.headers.get("access-control-allow-origin") == "http://localhost:3000"
