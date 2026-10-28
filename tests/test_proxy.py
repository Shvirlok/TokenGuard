"""Integration tests for TokenGuard FastAPI proxy, circuit breaker intercepts, and streaming."""

import json
import time
import pytest
import httpx
from pathlib import Path
from fastapi.testclient import TestClient

from tokenguard.config import Settings, set_settings
from tokenguard.proxy import create_app
import tokenguard.proxy as proxy_module


@pytest.fixture
def test_app(tmp_path: Path):
    db_file = tmp_path / "test_proxy.db"
    settings = Settings(
        host="127.0.0.1",
        port=8080,
        hourly_limit=0.010,  # $0.01 limit for testing budget tripping
        daily_limit=0.100,
        loop_threshold=3,
        db_path=db_file,
    )
    set_settings(settings)
    app = create_app(settings)
    return app


@pytest.mark.asyncio
async def test_dashboard_and_health_endpoints(test_app):
    with TestClient(test_app) as client:
        # Health check
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

        # Stats API
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_spent" in data
        assert "hourly_limit" in data
        assert data["hourly_limit"] == 0.010

        # Dashboard static file
        resp = client.get("/")
        assert resp.status_code == 200
        assert "TokenGuard" in resp.text


@pytest.mark.asyncio
async def test_kill_switch_blocking(test_app):
    with TestClient(test_app) as client:
        # Toggle kill switch ON
        resp = client.post("/api/kill-switch", json={"active": True})
        assert resp.status_code == 200
        assert resp.json()["kill_switch_active"] is True

        # Send chat completion request
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello TokenGuard"}],
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 429
        err = resp.json()
        assert err["error"]["type"] == "kill_switch_active"
        assert "kill switch active" in err["error"]["message"]

        # Check that it was logged to DB
        reqs_resp = client.get("/api/requests")
        assert reqs_resp.status_code == 200
        records = reqs_resp.json()["requests"]
        assert len(records) >= 1
        assert records[0]["status"] == "blocked_killswitch"

        # Toggle kill switch OFF
        resp = client.post("/api/kill-switch", json={"active": False})
        assert resp.status_code == 200
        assert resp.json()["kill_switch_active"] is False


@pytest.mark.asyncio
async def test_infinite_loop_detector(test_app, monkeypatch):
    """Verify that 3 consecutive identical prompt hashes trigger HTTP 429."""
    with TestClient(test_app) as client:
        # Mock upstream httpx.AsyncClient.post
        mock_openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Loop response"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }

        async def mock_post(url, json=None, headers=None):
            return httpx.Response(
                status_code=200,
                json=mock_openai_response,
                headers={"Content-Type": "application/json"}
            )

        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Repeat this identical prompt"}],
        }

        # 1st request -> Allowed
        r1 = client.post("/v1/chat/completions", json=payload)
        assert r1.status_code == 200

        # 2nd request -> Allowed
        r2 = client.post("/v1/chat/completions", json=payload)
        assert r2.status_code == 200

        # 3rd request -> Tripped Loop Detector!
        r3 = client.post("/v1/chat/completions", json=payload)
        assert r3.status_code == 429
        err = r3.json()
        assert err["error"]["type"] == "loop_detected"
        assert "infinite agent loop detected" in err["error"]["message"]

        # 4th request (retry of same prompt) -> Still blocked with 429
        r4 = client.post("/v1/chat/completions", json=payload)
        assert r4.status_code == 429

        # Sending a NEW, DISTINCT prompt must IMMEDIATELY SUCCEED!
        payload_distinct = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Completely new and different prompt"}],
        }
        r_new = client.post("/v1/chat/completions", json=payload_distinct)
        assert r_new.status_code == 200

        # Check DB log for blocked request
        reqs_resp = client.get("/api/requests?status=blocked")
        assert reqs_resp.status_code == 200
        blocked_records = reqs_resp.json()["requests"]
        assert len(blocked_records) >= 2
        assert blocked_records[0]["status"] == "blocked_loop"

        # Check /api/stats reflecting estimated money saved
        stats_resp = client.get("/api/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.json()
        assert stats_data["blocked_loop_count"] >= 2
        assert stats_data["saved_cost_estimate"] > 0.0

        # Test POST /api/reset clears circuit breaker
        reset_resp = client.post("/api/reset")
        assert reset_resp.status_code == 200
        assert reset_resp.json()["status"] == "reset"


@pytest.mark.asyncio
async def test_budget_limit_guard(test_app, monkeypatch):
    """Verify that exceeding hourly limit trips the circuit breaker with 429."""
    with TestClient(test_app) as client:
        # Set limit to $0.0100
        client.post("/api/config", json={"hourly_limit": 0.0100})

        mock_openai_response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Costly answer"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500
            }
        }

        async def mock_post(url, json=None, headers=None):
            return httpx.Response(
                status_code=200,
                json=mock_openai_response,
                headers={"Content-Type": "application/json"}
            )

        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        # 1st request costs $0.0075 (within $0.0100 limit) -> Allowed
        r1 = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Large task 1"}],
        })
        assert r1.status_code == 200

        # Now tighten limit to $0.0050 or let next request exceed $0.0100
        client.post("/api/config", json={"hourly_limit": 0.0050})

        # 2nd request should be blocked immediately with 429
        r2 = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Large task 2"}],
        })
        assert r2.status_code == 429
        err = r2.json()
        assert err["error"]["type"] == "budget_exceeded"
        assert "budget limit exceeded" in err["error"]["message"]


@pytest.mark.asyncio
async def test_streaming_chat_completions(test_app, monkeypatch):
    """Verify SSE streaming with on-the-fly usage extraction from final chunk."""
    with TestClient(test_app) as client:
        # Prepare mock SSE chunks
        chunk1 = b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}\n\n'
        chunk2 = b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world!"}}]}\n\n'
        chunk_usage = b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":15,"completion_tokens":5,"total_tokens":20}}\n\n'
        chunk_done = b'data: [DONE]\n\n'

        async def mock_aiter_bytes():
            yield chunk1
            yield chunk2
            yield chunk_usage
            yield chunk_done

        class MockStreamResponse:
            status_code = 200
            headers = {"Content-Type": "text/event-stream"}

            def aiter_bytes(self):
                return mock_aiter_bytes()

            async def aclose(self):
                pass

        async def mock_send(request, stream=False):
            return MockStreamResponse()

        monkeypatch.setattr(proxy_module.http_client, "send", mock_send)

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Tell me a joke in stream"}],
            "stream": True,
        }

        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["Content-Type"]

        # Ensure all chunks and [DONE] were delivered
        body_text = resp.text
        assert "Hello" in body_text
        assert " world!" in body_text
        assert "[DONE]" in body_text
        assert '"prompt_tokens":15' in body_text

        # Verify DB logged the streaming usage
        reqs_resp = client.get("/api/requests")
        records = reqs_resp.json()["requests"]
        assert len(records) >= 1
        stream_record = records[0]
        assert stream_record["status"] == "success"
        assert stream_record["prompt_tokens"] == 15
        assert stream_record["completion_tokens"] == 5
        assert stream_record["cost_usd"] > 0.0


@pytest.mark.asyncio
async def test_deepseek_routing_and_cost(test_app, monkeypatch):
    """Verify that requests with model='deepseek-chat' route to DeepSeek URL and calculate accurate costs."""
    with TestClient(test_app) as client:
        captured_requests = []

        mock_deepseek_resp = {
            "id": "deepseek-cmpl-789",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from DeepSeek V3!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10000,
                "completion_tokens": 2000,
                "total_tokens": 12000,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 10000,
            }
        }

        async def mock_post(url, json=None, headers=None):
            captured_requests.append({"url": str(url), "json": json, "headers": headers})
            return httpx.Response(
                status_code=200,
                json=mock_deepseek_resp,
                headers={"Content-Type": "application/json"}
            )

        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Explain quantum computing."}],
        }
        resp = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer sk-client-ds-key"}
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "Hello from DeepSeek V3!"

        # 1. Verify Upstream Routing URL & Client Auth Header
        assert len(captured_requests) == 1
        assert "api.deepseek.com" in captured_requests[0]["url"]
        assert captured_requests[0]["headers"]["Authorization"] == "Bearer sk-client-ds-key"

        # 2. Verify Database logging and cost calculation
        # deepseek-chat: 10,000 * $0.14/1M + 2,000 * $0.28/1M = $0.0014 + $0.00056 = $0.00196
        reqs_resp = client.get("/api/requests")
        assert reqs_resp.status_code == 200
        records = reqs_resp.json()["requests"]
        assert len(records) >= 1
        ds_record = records[0]
        assert ds_record["model"] == "deepseek-chat"
        assert ds_record["prompt_tokens"] == 10000
        assert ds_record["completion_tokens"] == 2000
        assert pytest.approx(ds_record["cost_usd"], 1e-6) == 0.00196


@pytest.mark.asyncio
async def test_deepseek_env_api_key_fallback_and_reasoner_cost(test_app, monkeypatch):
    """Verify fallback to DEEPSEEK_API_KEY when no header is supplied, and deepseek-reasoner cost calculation."""
    with TestClient(test_app) as client:
        captured_requests = []
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-deepseek-secret")

        mock_reasoner_resp = {
            "id": "deepseek-reasoner-123",
            "object": "chat.completion",
            "model": "deepseek-reasoner",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "42"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10000,
                "completion_tokens": 2000,
                "total_tokens": 12000,
            }
        }

        async def mock_post(url, json=None, headers=None):
            captured_requests.append({"url": str(url), "json": json, "headers": headers})
            return httpx.Response(
                status_code=200,
                json=mock_reasoner_resp,
                headers={"Content-Type": "application/json"}
            )

        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        payload = {
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": "Solve math problem"}],
        }
        # Send request without Authorization header
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

        # Verify fallback to DEEPSEEK_API_KEY from environment
        assert len(captured_requests) == 1
        assert "api.deepseek.com" in captured_requests[0]["url"]
        assert captured_requests[0]["headers"]["Authorization"] == "Bearer sk-env-deepseek-secret"

        # Verify deepseek-reasoner cost calculation:
        # deepseek-reasoner: 10,000 * $0.55/1M + 2,000 * $2.19/1M = $0.0055 + $0.00438 = $0.00988
        reqs_resp = client.get("/api/requests")
        records = reqs_resp.json()["requests"]
        assert records[0]["model"] == "deepseek-reasoner"
        assert pytest.approx(records[0]["cost_usd"], 1e-6) == 0.00988


@pytest.mark.asyncio
async def test_deepseek_streaming_chat(test_app, monkeypatch):
    """Verify DeepSeek streaming chat completion forwards chunks and calculates usage/cost accurately."""
    with TestClient(test_app) as client:
        captured_requests = []

        chunk1 = b'data: {"id":"ds-stream","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Thinking..."}}]}\n\n'
        chunk2 = b'data: {"id":"ds-stream","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" Done!"}}]}\n\n'
        chunk_usage = b'data: {"id":"ds-stream","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":10000,"completion_tokens":2000,"total_tokens":12000}}\n\n'
        chunk_done = b'data: [DONE]\n\n'

        async def mock_aiter_bytes():
            yield chunk1
            yield chunk2
            yield chunk_usage
            yield chunk_done

        class MockStreamResponse:
            status_code = 200
            headers = {"Content-Type": "text/event-stream"}

            def aiter_bytes(self):
                return mock_aiter_bytes()

            async def aclose(self):
                pass

        async def mock_send(request, stream=False):
            captured_requests.append(request)
            return MockStreamResponse()

        monkeypatch.setattr(proxy_module.http_client, "send", mock_send)

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Stream poem"}],
            "stream": True,
        }

        resp = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer sk-deepseek-stream-key"}
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["Content-Type"]

        # Verify routed to DeepSeek
        assert len(captured_requests) == 1
        assert "api.deepseek.com" in str(captured_requests[0].url)

        # Verify DB logged the streaming usage with DeepSeek pricing
        reqs_resp = client.get("/api/requests")
        records = reqs_resp.json()["requests"]
        assert len(records) >= 1
        stream_record = records[0]
        assert stream_record["model"] == "deepseek-chat"
        assert stream_record["prompt_tokens"] == 10000
        assert stream_record["completion_tokens"] == 2000
        assert pytest.approx(stream_record["cost_usd"], 1e-6) == 0.00196


@pytest.mark.asyncio
async def test_export_endpoints_csv_and_json(test_app, monkeypatch):
    """Verify that /api/export returns valid CSV and JSON file responses."""
    with TestClient(test_app) as client:
        # Mock upstream response
        mock_resp = {
            "id": "chatcmpl-exp",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Export test"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }

        async def mock_post(url, json=None, headers=None):
            return httpx.Response(status_code=200, json=mock_resp, headers={"Content-Type": "application/json"})

        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        # Generate a request
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Export test prompt"}]},
        )

        # 1. Test CSV Export
        csv_resp = client.get("/api/export?format=csv")
        assert csv_resp.status_code == 200
        assert "text/csv" in csv_resp.headers["Content-Type"]
        assert 'filename="tokenguard_requests.csv"' in csv_resp.headers["Content-Disposition"]
        csv_text = csv_resp.text
        assert "id,timestamp,model,prompt_tokens" in csv_text
        assert "gpt-4o" in csv_text

        # 2. Test JSON Export
        json_resp = client.get("/api/export?format=json")
        assert json_resp.status_code == 200
        assert "application/json" in json_resp.headers["Content-Type"]
        assert 'filename="tokenguard_requests.json"' in json_resp.headers["Content-Disposition"]
        json_data = json_resp.json()
        assert "requests" in json_data
        assert "count" in json_data
        assert json_data["count"] >= 1
        assert json_data["requests"][0]["model"] == "gpt-4o"

        # 3. Test Invalid Format Error
        bad_resp = client.get("/api/export?format=xml")
        assert bad_resp.status_code == 400
        assert "Unsupported export format" in bad_resp.json()["detail"]


@pytest.mark.asyncio
async def test_simulate_and_clear_logs_endpoints(test_app):
    """Verify that /api/simulate triggers events and /api/clear-logs flushes DB."""
    with TestClient(test_app) as client:
        # 1. Simulate Loop
        sim_loop = client.post("/api/simulate", json={"type": "loop", "model": "gpt-4o"})
        assert sim_loop.status_code == 200
        assert sim_loop.json()["simulation"] == "loop"

        # Check that blocked loop was logged
        stats = client.get("/api/stats").json()
        assert stats["blocked_loop_count"] >= 1
        assert stats["saved_cost_estimate"] > 0.0

        # 2. Simulate Success
        sim_succ = client.post("/api/simulate", json={"type": "success", "model": "gpt-4o-mini"})
        assert sim_succ.status_code == 200
        assert sim_succ.json()["simulation"] == "success"

        # 3. Simulate Alternating Loop (A-B-A-B)
        sim_alt = client.post("/api/simulate", json={"type": "alternating_loop", "model": "gpt-4o"})
        assert sim_alt.status_code == 200
        alt_data = sim_alt.json()
        assert alt_data["simulation"] == "alternating_loop"
        assert alt_data["tripped"] is True
        assert alt_data["cycle_length"] == 2
        assert alt_data["pattern"] == ["read_file", "bash"]
        assert len(alt_data["results"]) == 4
        assert alt_data["results"][0]["status"] == "allowed"
        assert alt_data["results"][1]["status"] == "allowed"
        assert alt_data["results"][2]["status"] == "allowed"
        assert alt_data["results"][3]["status"] == "blocked_tool_loop"

        # 4. Clear Logs
        clear_resp = client.post("/api/clear-logs")
        assert clear_resp.status_code == 200
        assert clear_resp.json()["status"] == "cleared"

        # Verify DB is emptied
        reqs_resp = client.get("/api/requests")
        assert len(reqs_resp.json()["requests"]) == 0


@pytest.mark.asyncio
async def test_sse_stream_realtime_events(test_app):
    """Verify that SSE stream broadcasts state changes, request events, and log flushes."""
    import asyncio

    # Test queue subscription directly through broadcaster
    q = proxy_module.broadcaster.subscribe()
    try:
        with TestClient(test_app) as client:
            # 1. Trigger kill switch
            client.post("/api/kill-switch", json={"active": True, "source": "cli"})
            msg = q.get_nowait()
            assert msg["event"] == "state_change"
            assert msg["data"]["kill_switch_active"] is True
            assert msg["data"]["source"] == "cli"

            # 2. Trigger config / profile change
            client.post("/api/config", json={"hourly_limit": 15.0, "daily_limit": 100.0, "loop_threshold": 4, "profile": "standard", "source": "cli"})
            msg2 = q.get_nowait()
            assert msg2["event"] == "state_change"
            assert msg2["data"]["active_profile"] == "standard"
            assert msg2["data"]["hourly_limit"] == 15.0

            # 3. Simulate request logging
            client.post("/api/simulate", json={"type": "success", "model": "gpt-4o"})
            msg3 = q.get_nowait()
            assert msg3["event"] == "request_logged"
            assert msg3["data"]["model"] == "gpt-4o"
            assert msg3["data"]["status"] == "success"

            # 4. Clear logs
            client.post("/api/clear-logs", json={"source": "cli"})
            msg4 = q.get_nowait()
            assert msg4["event"] == "logs_cleared"
            assert msg4["data"]["source"] == "cli"
    finally:
        proxy_module.broadcaster.unsubscribe(q)


@pytest.mark.asyncio
async def test_burn_rate_proxy_429_json_response(test_app):
    """Verify that burn rate exceeded triggers HTTP 429 with explicit JSON schema."""
    with TestClient(test_app) as client:
        guard = proxy_module.get_guard()
        guard.max_burn_rate = "$0.50/min"
        guard.burn_rate_limit = 0.50
        guard.burn_rate_window_seconds = 60.0
        now = time.time()
        # Seed spend exceeding $0.50 in the last minute
        guard.burn_rate_records = [(now - 10, 0.35), (now - 5, 0.25)]

        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Burn rate test prompt"}],
        }

        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 429
        data = resp.json()
        assert "error" in data
        err = data["error"]
        assert err["type"] == "tokenguard_budget_breach"
        assert err["code"] == "burn_rate_exceeded"
        assert err["budget_limit"] == 0.50
        assert err["current_spend"] >= 0.60
        assert "Burn rate exceeded" in err["message"]
        assert "Hard limit: $0.50/min" in err["message"]

        # Verify DB logged the blocked burn rate request
        reqs_resp = client.get("/api/requests?status=blocked")
        assert reqs_resp.status_code == 200
        records = reqs_resp.json()["requests"]
        assert any(r["status"] == "blocked_burn_rate" for r in records)


@pytest.mark.asyncio
async def test_context_ballooning_proxy_429_json_response(test_app, monkeypatch):
    """Verify that monotonic context expansion > 300% across 3 requests trips 429."""
    with TestClient(test_app) as client:
        guard = proxy_module.get_guard()
        guard.enable_context_ballooning = True

        async def mock_send(request, **kwargs):
            req_body = json.loads(request.content)
            req_msg = req_body["messages"][0]["content"]
            tok_count = max(10, len(req_msg) // 4)
            resp_payload = {
                "id": "chatcmpl-cb",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Short reply"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": tok_count, "completion_tokens": 10, "total_tokens": tok_count + 10},
            }
            return httpx.Response(200, json=resp_payload)

        monkeypatch.setattr(proxy_module.http_client, "send", mock_send)

        headers = {"x-session-id": "agent-loop-sess"}

        # Request 1: ~100 tokens
        words_100 = "word " * 100
        r1 = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": words_100}],
        }, headers=headers)
        assert r1.status_code == 200

        # Request 2: ~250 tokens
        words_250 = "word " * 250
        r2 = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": words_250}],
        }, headers=headers)
        assert r2.status_code == 200

        # Request 3: ~450 tokens (> 300% expansion from baseline)
        words_450 = "word " * 450
        r3 = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": words_450}],
        }, headers=headers)
        assert r3.status_code == 429
        err = r3.json()["error"]
        assert err["type"] == "tokenguard_budget_breach"
        assert err["code"] == "context_ballooning_detected"
        assert "Context ballooning detected" in err["message"]

        # Request with task update status header should bypass
        headers_with_task = {"x-session-id": "agent-loop-sess", "x-task-status": "refactored_subsystem"}
        r_bypass = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": words_450}],
        }, headers=headers_with_task)
        assert r_bypass.status_code == 200


@pytest.mark.asyncio
async def test_zero_latency_streaming_omitted_usage(test_app, monkeypatch):
    """Verify that when upstream omits usage, StreamAccumulator computes tokens and DB logs them."""
    with TestClient(test_app) as client:
        chunk1 = b'data: {"id":"chatcmpl-omitted","choices":[{"delta":{"content":"Zero "}}]}\n\n'
        chunk2 = b'data: {"id":"chatcmpl-omitted","choices":[{"delta":{"content":"latency "}}]}\n\n'
        chunk3 = b'data: {"id":"chatcmpl-omitted","choices":[{"delta":{"content":"streaming accumulator."}}]}\n\n'
        chunk_done = b'data: [DONE]\n\n'

        async def mock_aiter_bytes():
            yield chunk1
            yield chunk2
            yield chunk3
            yield chunk_done

        class MockStreamResponse:
            status_code = 200
            headers = {"Content-Type": "text/event-stream"}
            def aiter_bytes(self):
                return mock_aiter_bytes()
            async def aclose(self):
                pass

        async def mock_send(request, stream=False):
            return MockStreamResponse()

        monkeypatch.setattr(proxy_module.http_client, "send", mock_send)

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Stream without usage"}],
            "stream": True,
        }

        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        text = resp.text
        assert "Zero " in text
        assert "latency " in text
        assert "streaming accumulator." in text
        assert "[DONE]" in text

        # Verify DB logged completion tokens > 0 (calculated from text chunks)
        reqs_resp = client.get("/api/requests")
        records = reqs_resp.json()["requests"]
        assert len(records) >= 1
        record = records[0]
        assert record["status"] == "success"
        assert record["completion_tokens"] > 0
        assert record["cost_usd"] > 0.0


@pytest.mark.asyncio
async def test_streaming_client_disconnect_closes_upstream(test_app, monkeypatch):
    """Verify that client disconnection / cancellation ensures upstream response is closed."""
    import asyncio
    closed_called = False

    class MockStreamResponse:
        status_code = 200
        headers = {"Content-Type": "text/event-stream"}
        async def aiter_bytes(self):
            for i in range(10):
                yield b'data: {"choices":[{"delta":{"content":"Chunk"}}}\n\n'
                await asyncio.sleep(0.005)
        async def aclose(self):
            nonlocal closed_called
            closed_called = True

    async def mock_send(request, stream=False):
        return MockStreamResponse()

    with TestClient(test_app) as client:
        monkeypatch.setattr(proxy_module.http_client, "send", mock_send)
        try:
            with client.stream("POST", "/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Cancel test"}],
                "stream": True,
            }) as resp:
                for line in resp.iter_lines():
                    if line:
                        break  # Disconnect after first chunk
        except Exception:
            pass

    assert closed_called is True


def test_tool_ping_pong_proxy_intercept(test_app):
    """Verify that repeated tool call loops trip the circuit breaker and return 429 with TOOL_PING_PONG_LOOP."""
    with TestClient(test_app) as client:
        guard = proxy_module.get_guard()
        guard.reset()
        guard.tool_loop_threshold = 3

        headers = {"x-session-id": "test-tool-loop-session"}
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Execute tool"},
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "config.json"}'}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "Error: FileNotFoundError: config.json"},
                {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "config.json"}'}}]},
                {"role": "tool", "tool_call_id": "c2", "content": "Error: FileNotFoundError: config.json"},
                {"role": "assistant", "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "config.json"}'}}]},
            ],
        }

        resp = client.post("/v1/chat/completions", json=body, headers=headers)
        assert resp.status_code == 429
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "circuit_breaker"
        assert data["error"]["code"] == "TOOL_PING_PONG_LOOP"
        assert "TokenGuard intercepted repeated tool call loop: read_file" in data["error"]["message"]


def test_alternating_tool_loop_proxy_intercept(test_app):
    """Verify that oscillating alternating tool calls (A-B-A-B) trip the circuit breaker and return 429."""
    with TestClient(test_app) as client:
        guard = proxy_module.get_guard()
        guard.reset()
        guard.tool_loop_threshold = 3

        headers = {"x-agent-id": "agent-oscillating-worker"}
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Execute alternating tools"},
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "data.csv"}'}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "Error: Permission denied"},
                {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "chmod 777 data.csv"}'}}]},
                {"role": "tool", "tool_call_id": "c2", "content": "Operation not permitted"},
                {"role": "assistant", "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "data.csv"}'}}]},
                {"role": "tool", "tool_call_id": "c3", "content": "Error: Permission denied"},
                {"role": "assistant", "tool_calls": [{"id": "c4", "type": "function", "function": {"name": "bash", "arguments": '{"command": "chmod 777 data.csv"}'}}]},
            ],
        }

        resp = client.post("/v1/chat/completions", json=body, headers=headers)
        assert resp.status_code == 429
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "circuit_breaker"
        assert data["error"]["code"] == "TOOL_PING_PONG_LOOP"
        assert data["error"]["cycle_length"] == 2
        assert data["error"]["pattern"] == ["read_file", "bash"]


@pytest.mark.asyncio
async def test_api_events_sse_endpoint(test_app):
    """Verify that GET /api/events establishes SSE text/event-stream StreamingResponse."""
    from starlette.requests import Request
    scope = {"type": "http", "method": "GET", "path": "/api/events", "headers": []}
    req = Request(scope)
    endpoint = None
    for route in test_app.routes:
        if getattr(route, "path", None) == "/api/events":
            endpoint = route.endpoint
            break
    assert endpoint is not None
    resp = await endpoint(req)
    assert resp.media_type == "text/event-stream"
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["connection"] == "keep-alive"



def test_simulate_tool_loop_endpoint(test_app):
    """Verify that POST /api/test/simulate-tool-loop trips circuit breaker and returns intercepted result."""
    with TestClient(test_app) as client:
        resp = client.post("/api/test/simulate-tool-loop", json={
            "tool_name": "read_file",
            "arguments": {"path": "/workspace/credentials.env"}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["simulation"] == "tool_loop"
        assert data["tool_name"] == "read_file"
        assert data["intercepted"] is True
        assert len(data["results"]) == 3
        assert data["results"][-1]["status"] == "blocked_tool_loop"
        assert "[TOOL RETRY LOOP: HARD STOP]" in data["results"][-1]["message"]

        reqs = client.get("/api/requests").json()["requests"]
        tool_loop_reqs = [r for r in reqs if r["status"] == "blocked_tool_loop"]
        assert len(tool_loop_reqs) >= 1
        assert "[TOOL RETRY LOOP: HARD STOP]" in tool_loop_reqs[0]["blocked_reason"]


@pytest.mark.asyncio
async def test_circuit_breaker_saved_telemetry_headers_and_body(test_app, monkeypatch):
    """Verify X-TokenGuard-Status, Reason, Tokens-Saved, USD-Saved headers and JSON body telemetry on 429."""
    with TestClient(test_app) as client:
        mock_openai_response = {
            "id": "chatcmpl-telemetry",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200},
        }

        async def mock_post(url, json=None, headers=None):
            return httpx.Response(status_code=200, json=mock_openai_response, headers={"Content-Type": "application/json"})

        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Telemetry test prompt runaway loop"}],
        }

        # 1st request -> allowed
        r1 = client.post("/v1/chat/completions", json=payload)
        assert r1.status_code == 200

        # 2nd request -> allowed
        r2 = client.post("/v1/chat/completions", json=payload)
        assert r2.status_code == 200

        # 3rd request -> tripped (loop_threshold = 3)
        r3 = client.post("/v1/chat/completions", json=payload)
        assert r3.status_code == 429

        # Headers check
        assert r3.headers.get("x-tokenguard-status") == "tripped"
        assert "infinite agent loop detected" in r3.headers.get("x-tokenguard-reason", "")
        tokens_saved_header = r3.headers.get("x-tokenguard-tokens-saved")
        assert tokens_saved_header is not None
        assert int(tokens_saved_header) > 0
        usd_saved_header = r3.headers.get("x-tokenguard-usd-saved")
        assert usd_saved_header is not None
        assert float(usd_saved_header) > 0.0

        # JSON body check
        body = r3.json()
        assert body["tokens_saved"] == int(tokens_saved_header)
        assert pytest.approx(body["usd_saved"], 1e-4) == float(usd_saved_header)
        assert body["error"]["tokens_saved"] == int(tokens_saved_header)
        assert pytest.approx(body["error"]["usd_saved"], 1e-4) == float(usd_saved_header)


@pytest.mark.asyncio
async def test_dry_run_observation_mode_chat_completions(tmp_path: Path, monkeypatch):
    """Verify that in dry_run mode, requests that trip circuit breaker pass through upstream with 200 OK and headers."""
    db_file = tmp_path / "test_dryrun.db"
    dry_settings = Settings(
        host="127.0.0.1",
        port=8080,
        loop_threshold=2,
        db_path=db_file,
        dry_run=True,
    )
    set_settings(dry_settings)
    app = create_app(dry_settings)

    with TestClient(app) as client:
        mock_response = {
            "id": "chatcmpl-dryrun",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Upstream response in shadow mode"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }

        async def mock_post(url, json=None, headers=None):
            return httpx.Response(status_code=200, json=mock_response, headers={"Content-Type": "application/json"})

        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Same prompt to trigger trip in dry-run"}],
        }

        # Request 1 -> 200 OK (Allowed)
        r1 = client.post("/v1/chat/completions", json=payload)
        assert r1.status_code == 200
        assert r1.headers.get("x-tokenguard-dry-run") == "true"
        assert r1.headers.get("x-tokenguard-status") is None

        # Request 2 -> Would normally trip with 429, but dry_run passes through with 200 OK!
        r2 = client.post("/v1/chat/completions", json=payload)
        assert r2.status_code == 200
        assert r2.headers.get("x-tokenguard-dry-run") == "true"
        assert r2.headers.get("x-tokenguard-status") == "tripped"
        assert "infinite agent loop detected" in r2.headers.get("x-tokenguard-reason", "")
        assert r2.headers.get("x-tokenguard-tokens-saved") is not None
        assert int(r2.headers.get("x-tokenguard-tokens-saved")) > 0
        assert r2.headers.get("x-tokenguard-usd-saved") is not None
        assert float(r2.headers.get("x-tokenguard-usd-saved")) > 0.0
        assert r2.json()["choices"][0]["message"]["content"] == "Upstream response in shadow mode"

        # Verify DB logged trip as blocked_loop with blocked_reason
        reqs = client.get("/api/requests").json()["requests"]
        assert len(reqs) >= 2
        trip_records = [r for r in reqs if r["status"] == "blocked_loop"]
        assert len(trip_records) == 1
        assert "infinite agent loop detected" in trip_records[0]["blocked_reason"]


@pytest.mark.asyncio
async def test_dry_run_observation_mode_anthropic_messages(tmp_path: Path, monkeypatch):
    """Verify that in dry_run mode, Anthropic /v1/messages passes tripped requests through upstream."""
    db_file = tmp_path / "test_dryrun_anthropic.db"
    dry_settings = Settings(
        host="127.0.0.1",
        port=8080,
        loop_threshold=2,
        db_path=db_file,
        dry_run=True,
    )
    set_settings(dry_settings)
    app = create_app(dry_settings)

    with TestClient(app) as client:
        mock_response = {
            "id": "msg_dryrun_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Anthropic upstream shadow answer"}],
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 60, "output_tokens": 25},
        }

        async def mock_post(url, json=None, headers=None):
            return httpx.Response(status_code=200, json=mock_response, headers={"Content-Type": "application/json"})

        monkeypatch.setattr(proxy_module.http_client, "post", mock_post)

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Anthropic loop trip prompt"}],
        }

        # Request 1 -> 200 OK
        r1 = client.post("/v1/messages", json=payload, headers={"x-api-key": "test-key"})
        assert r1.status_code == 200
        assert r1.headers.get("x-tokenguard-dry-run") == "true"

        # Request 2 -> Tripped, but passes through with 200 OK
        r2 = client.post("/v1/messages", json=payload, headers={"x-api-key": "test-key"})
        assert r2.status_code == 200
        assert r2.headers.get("x-tokenguard-dry-run") == "true"
        assert r2.headers.get("x-tokenguard-status") == "tripped"
        assert r2.headers.get("x-tokenguard-tokens-saved") is not None
        assert r2.headers.get("x-tokenguard-usd-saved") is not None
        assert r2.json()["content"][0]["text"] == "Anthropic upstream shadow answer"

        # Verify DB recorded the trip
        reqs = client.get("/api/requests").json()["requests"]
        assert len(reqs) >= 2
        trip_records = [r for r in reqs if r["status"] == "blocked_loop"]
        assert len(trip_records) == 1
        assert "infinite agent loop detected" in trip_records[0]["blocked_reason"]










