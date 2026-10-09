"""Comprehensive tests for multi-provider proxying (Gemini, Claude, Qwen, DeepSeek, Groq, Mistral, OpenRouter)."""

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import httpx

from tokenguard.config import Settings, set_settings
from tokenguard.proxy import create_app, detect_provider, resolve_upstream_url, resolve_auth_headers
import tokenguard.proxy as proxy_module


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_multi_provider.db"


@pytest.fixture
def custom_settings(temp_db: Path) -> Settings:
    settings = Settings(
        db_path=temp_db,
        hourly_limit=10.0,
        daily_limit=50.0,
        loop_threshold=3,
        loop_window_seconds=60.0,
        upstream_url="https://api.openai.com",
        gemini_url="https://generativelanguage.googleapis.com/v1beta/openai",
        anthropic_url="https://api.anthropic.com",
        dashscope_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        deepseek_url="https://api.deepseek.com",
        groq_url="https://api.groq.com/openai/v1",
        mistral_url="https://api.mistral.ai/v1",
        openrouter_url="https://openrouter.ai/api/v1",
    )
    set_settings(settings)
    return settings


def test_provider_detection_and_urls(custom_settings: Settings):
    """Verify provider detection logic and upstream URL resolution for all major LLMs."""
    assert detect_provider("gpt-4o") == "openai"
    assert detect_provider("gpt-4o-mini") == "openai"
    assert detect_provider("o1") == "openai"

    assert detect_provider("gemini-2.0-flash") == "gemini"
    assert detect_provider("gemini-1.5-pro") == "gemini"
    assert detect_provider("gemini-1.5-flash-8b") == "gemini"

    assert detect_provider("claude-3-5-sonnet-20241022") == "anthropic"
    assert detect_provider("claude-3-7-sonnet") == "anthropic"
    assert detect_provider("claude-3-opus") == "anthropic"

    assert detect_provider("qwen-max") == "dashscope"
    assert detect_provider("qwen-turbo") == "dashscope"
    assert detect_provider("qwen-2.5-coder") == "dashscope"
    assert detect_provider("qwq-32b") == "dashscope"

    assert detect_provider("deepseek-chat") == "deepseek"
    assert detect_provider("deepseek-reasoner") == "deepseek"

    assert detect_provider("llama-3.3-70b-versatile") == "groq"
    assert detect_provider("mixtral-8x7b-32768") == "groq"
    assert detect_provider("gemma2-9b-it") == "groq"

    assert detect_provider("mistral-large-latest") == "mistral"
    assert detect_provider("codestral-latest") == "mistral"

    assert detect_provider("openrouter/meta-llama/llama-3") == "openrouter"

    # URL resolution checks
    assert resolve_upstream_url("gemini") == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert resolve_upstream_url("dashscope") == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert resolve_upstream_url("deepseek") == "https://api.deepseek.com/chat/completions"
    assert resolve_upstream_url("groq") == "https://api.groq.com/openai/v1/chat/completions"
    assert resolve_upstream_url("mistral") == "https://api.mistral.ai/v1/chat/completions"
    assert resolve_upstream_url("openrouter") == "https://openrouter.ai/api/v1/chat/completions"
    assert resolve_upstream_url("anthropic", endpoint="messages") == "https://api.anthropic.com/v1/messages"


@pytest.mark.asyncio
async def test_gemini_openai_gateway_proxying(custom_settings: Settings, monkeypatch):
    """Test Google Gemini calls routed through OpenAI-compatible gateway with pricing and DB recording."""
    app = create_app(custom_settings)

    with TestClient(app) as client:
        captured_url = None
        captured_headers = None

        async def mock_post(url, json=None, headers=None, **kwargs):
            nonlocal captured_url, captured_headers
            captured_url = str(url)
            captured_headers = headers

            resp_payload = {
                "id": "chatcmpl-gemini-12345",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gemini-2.0-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Gemini 2.0 Flash via TokenGuard!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }
            return httpx.Response(200, json=resp_payload, headers={"Content-Type": "application/json"})

        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyFakeGeminiKey123456789")
        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.0-flash",
                "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello from Gemini 2.0 Flash via TokenGuard!"
        assert "generativelanguage.googleapis.com" in captured_url
        assert captured_headers.get("Authorization") == "Bearer AIzaSyFakeGeminiKey123456789"

        # Check stats
        stats_resp = client.get("/api/stats")
        stats = stats_resp.json()
        assert stats["total_requests"] == 1
        assert stats["success_requests"] == 1
        # gemini-2.0-flash input: $0.10/1M, output: $0.40/1M -> (100*0.10 + 50*0.40)/1M = 0.000030 USD
        assert stats["total_spent"] > 0.000025


@pytest.mark.asyncio
async def test_qwen_dashscope_proxying(custom_settings: Settings, monkeypatch):
    """Test Qwen / DashScope calls with token pricing calculation."""
    app = create_app(custom_settings)

    with TestClient(app) as client:
        captured_url = None
        captured_headers = None

        async def mock_post(url, json=None, headers=None, **kwargs):
            nonlocal captured_url, captured_headers
            captured_url = str(url)
            captured_headers = headers

            resp_payload = {
                "id": "chatcmpl-qwen-9988",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "qwen-max",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response from Qwen-Max through TokenGuard."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                },
            }
            return httpx.Response(200, json=resp_payload, headers={"Content-Type": "application/json"})

        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-secret-token")
        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen-max",
                "messages": [{"role": "user", "content": "Write Python code for quicksort."}],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "dashscope" in captured_url
        assert captured_headers.get("Authorization") == "Bearer sk-dashscope-secret-token"

        # Check stats
        stats = client.get("/api/stats").json()
        assert stats["total_requests"] == 1
        # qwen-max: input 2.80, output 8.40 -> (200*2.80 + 100*8.40)/1M = 0.00140 USD
        assert 0.0013 < stats["total_spent"] < 0.0015


@pytest.mark.asyncio
async def test_anthropic_native_messages_endpoint(custom_settings: Settings, monkeypatch):
    """Test Anthropic Native /v1/messages API with circuit breaker and usage metrics."""
    app = create_app(custom_settings)

    with TestClient(app) as client:
        captured_url = None
        captured_headers = None

        async def mock_post(url, json=None, headers=None, **kwargs):
            nonlocal captured_url, captured_headers
            captured_url = str(url)
            captured_headers = headers

            resp_payload = {
                "id": "msg_01XyZ999",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Claude 3.5 Sonnet!"}],
                "model": "claude-3-5-sonnet",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 25,
                },
            }
            return httpx.Response(200, json=resp_payload, headers={"Content-Type": "application/json"})

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test-key")
        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "Hello Claude"}],
                "max_tokens": 1024,
            },
            headers={"x-api-key": "sk-ant-custom-client-key"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["content"][0]["text"] == "Hello from Claude 3.5 Sonnet!"
        assert "api.anthropic.com/v1/messages" in captured_url
        assert captured_headers.get("x-api-key") == "sk-ant-custom-client-key"

        # Check stats
        stats = client.get("/api/stats").json()
        assert stats["total_requests"] == 1
        # claude-3-5-sonnet: input $3.00/1M, output $15.00/1M -> (50*3.0 + 25*15.0)/1M = 0.000525 USD
        assert 0.00050 < stats["total_spent"] < 0.00055


@pytest.mark.asyncio
async def test_anthropic_messages_loop_breaker(custom_settings: Settings, monkeypatch):
    """Test that native /v1/messages calls trigger the infinite loop breaker and return Anthropic 429 schema."""
    app = create_app(custom_settings)

    with TestClient(app) as client:
        async def mock_post(*args, **kwargs):
            return httpx.Response(200, json={
                "id": "msg_loop",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Loop answer"}],
                "usage": {"input_tokens": 10, "output_tokens": 10},
            }, headers={"Content-Type": "application/json"})

        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        payload = {
            "model": "claude-3-5-haiku",
            "messages": [{"role": "user", "content": "Identical Anthropic Agent Loop Prompt"}],
            "max_tokens": 500,
        }

        # Attempt 1 -> 200 Allowed
        r1 = client.post("/v1/messages", json=payload)
        assert r1.status_code == 200

        # Attempt 2 -> 200 Allowed
        r2 = client.post("/v1/messages", json=payload)
        assert r2.status_code == 200

        # Attempt 3 (threshold=3) -> 429 Blocked
        r3 = client.post("/v1/messages", json=payload)
        assert r3.status_code == 429
        data = r3.json()
        assert data.get("type") == "error"
        assert data.get("error", {}).get("type") == "rate_limit_error"
        assert "infinite agent loop detected" in data.get("error", {}).get("message", "")

        # Check stats
        stats = client.get("/api/stats").json()
        assert stats["total_requests"] == 3
        assert stats["blocked_requests"] == 1
        assert stats["saved_cost_estimate"] > 0.0


@pytest.mark.asyncio
async def test_claude_openai_adapter(custom_settings: Settings, monkeypatch):
    """Test OpenAI SDK clients calling Claude models via /v1/chat/completions (automatic translation)."""
    app = create_app(custom_settings)

    with TestClient(app) as client:
        async def mock_post(url, json=None, headers=None, **kwargs):
            # Assert that the adapter converted OpenAI schema to Anthropic Messages schema
            assert "messages" in json
            assert json["messages"][0]["role"] == "user"
            assert json["system"] == "You are a helpful coding assistant"
            assert headers.get("x-api-key") == "sk-ant-test-key"

            resp_payload = {
                "id": "msg_translated_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Translated Claude answer"}],
                "model": "claude-3-5-sonnet",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 60, "output_tokens": 40},
            }
            return httpx.Response(200, json=resp_payload, headers={"Content-Type": "application/json"})

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        # Send standard OpenAI chat completions format
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet",
                "messages": [
                    {"role": "system", "content": "You are a helpful coding assistant"},
                    {"role": "user", "content": "Write Hello World in Rust"},
                ],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        # Verify adapted response is standard OpenAI chat.completion
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Translated Claude answer"
        assert data["usage"]["total_tokens"] == 100


@pytest.mark.asyncio
async def test_groq_and_mistral_proxying(custom_settings: Settings, monkeypatch):
    """Test Groq (Llama) and Mistral routing."""
    app = create_app(custom_settings)

    with TestClient(app) as client:
        groq_called = False
        mistral_called = False

        async def mock_post(url, json=None, headers=None, **kwargs):
            nonlocal groq_called, mistral_called
            url_str = str(url)
            if "api.groq.com" in url_str:
                groq_called = True
            if "api.mistral.ai" in url_str:
                mistral_called = True

            return httpx.Response(200, json={
                "id": "chatcmpl-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            }, headers={"Content-Type": "application/json"})

        monkeypatch.setenv("GROQ_API_KEY", "gsk_groq_key_123")
        monkeypatch.setenv("MISTRAL_API_KEY", "mistral_api_key_456")
        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        # Groq Llama call
        r_groq = client.post(
            "/v1/chat/completions",
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "test groq"}]},
        )
        assert r_groq.status_code == 200
        assert groq_called is True

        # Mistral call
        r_mistral = client.post(
            "/v1/chat/completions",
            json={"model": "mistral-large-latest", "messages": [{"role": "user", "content": "test mistral"}]},
        )
        assert r_mistral.status_code == 200
        assert mistral_called is True
