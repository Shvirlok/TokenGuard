import httpx
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from tokenguard.config import Settings, set_settings
from tokenguard.proxy import create_app
import tokenguard.proxy as proxy_module


@pytest.fixture
def sdk_test_app(tmp_path: Path):
    db_file = tmp_path / "test_sdk.db"
    settings = Settings(
        host="127.0.0.1",
        port=8080,
        hourly_limit=5.0,
        daily_limit=50.0,
        loop_threshold=3,
        loop_window_seconds=60.0,
        db_path=db_file,
    )
    set_settings(settings)
    app = create_app(settings)
    return app


def test_sdk_behavior(sdk_test_app, monkeypatch):
    # Mock upstream OpenAI response
    mock_resp = {
        "id": "chatcmpl-sdk-test",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from mock!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }

    async def mock_post(url, json=None, headers=None):
        return httpx.Response(status_code=200, json=mock_resp, headers={"Content-Type": "application/json"})

    with TestClient(sdk_test_app) as client:
        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        # 1. Reset guard
        reset_res = client.post("/api/reset")
        assert reset_res.status_code == 200

        # 2. Call prompt A twice
        payload_a = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Prompt A"}]}
        r1 = client.post("/v1/chat/completions", json=payload_a)
        assert r1.status_code == 200

        r2 = client.post("/v1/chat/completions", json=payload_a)
        assert r2.status_code == 200

        # 3. 3rd time prompt A trips
        r3 = client.post("/v1/chat/completions", json=payload_a)
        assert r3.status_code == 429
        assert r3.json()["error"]["type"] == "loop_detected"

        # 4. Prompt B immediately succeeds without 429!
        payload_b = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Prompt B"}]}
        r_b = client.post("/v1/chat/completions", json=payload_b)
        assert r_b.status_code == 200
        assert r_b.json()["choices"][0]["message"]["content"] == "Hello from mock!"


def test_sdk_deepseek_behavior(sdk_test_app, monkeypatch):
    captured = {}
    mock_resp = {
        "id": "chatcmpl-ds-sdk",
        "object": "chat.completion",
        "model": "deepseek-chat",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from DeepSeek via OpenAI SDK!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
    }

    async def mock_post(url, json=None, headers=None):
        captured["url"] = str(url)
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(status_code=200, json=mock_resp, headers={"Content-Type": "application/json"})

    with TestClient(sdk_test_app) as client:
        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello TokenGuard with DeepSeek"}],
        }
        res = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer sk-deepseek-test-key"}
        )
        assert res.status_code == 200
        assert res.json()["choices"][0]["message"]["content"] == "Hello from DeepSeek via OpenAI SDK!"
        assert "api.deepseek.com" in captured["url"]
        assert captured["headers"]["Authorization"] == "Bearer sk-deepseek-test-key"

