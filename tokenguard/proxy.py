"""FastAPI proxy application for TokenGuard."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tokenguard.config import Settings, get_api_key, get_settings
from tokenguard.db import (
    clear_all_requests,
    get_all_requests_for_export,
    get_hourly_history,
    get_metrics,
    get_recent_requests,
    init_db,
    log_request,
)
from tokenguard.export import (
    format_csv_export,
    format_har_export,
    format_json_export,
    format_jsonl_export,
)
from tokenguard.guard import (
    BudgetExceededError,
    CircuitBreaker,
    CircuitBreakerError,
    KillSwitchActiveError,
    LoopDetectedError,
)

logger = logging.getLogger("tokenguard.proxy")


class EventBroadcaster:
    """Manages Server-Sent Events (SSE) subscriptions and real-time broadcasting."""

    def __init__(self):
        self._subscribers: Set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def broadcast(self, event_type: str, data: Dict[str, Any]) -> None:
        if not self._subscribers:
            return
        payload = {"event": event_type, "data": data}
        for q in list(self._subscribers):
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except Exception:
                        pass
                q.put_nowait(payload)
            except Exception:
                pass


# Global state instances
broadcaster = EventBroadcaster()
circuit_breaker: Optional[CircuitBreaker] = None
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for setup and teardown."""
    global circuit_breaker, http_client
    settings = get_settings()

    # Initialize SQLite database
    await init_db(settings.db_path)

    # Initialize CircuitBreaker engine
    circuit_breaker = CircuitBreaker(
        hourly_limit=settings.hourly_limit,
        daily_limit=settings.daily_limit,
        loop_threshold=settings.loop_threshold,
        loop_window_size=settings.loop_window_size,
        loop_window_seconds=settings.loop_window_seconds,
        prices_path=settings.prices_path,
        kill_switch=settings.kill_switch,
        active_profile=settings.profile,
    )

    # Persistent HTTP client for upstream proxying
    http_client = httpx.AsyncClient(
        base_url=settings.upstream_url,
        timeout=httpx.Timeout(settings.timeout, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
    )

    yield

    # Teardown
    if http_client is not None:
        await http_client.aclose()


async def _record_and_broadcast_request(
    db_path: Path,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_ms: int,
    prompt_hash: Optional[str],
    status: str,
    blocked_reason: Optional[str] = None,
) -> int:
    """Log an LLM request to SQLite and broadcast a real-time event to all SSE clients."""
    rec_id = await log_request(
        db_path=db_path,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        prompt_hash=prompt_hash,
        status=status,
        blocked_reason=blocked_reason,
    )
    req_payload = {
        "id": rec_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "prompt_hash": prompt_hash,
        "status": status,
        "blocked_reason": blocked_reason,
    }
    await broadcaster.broadcast("request_logged", req_payload)
    return rec_id


def detect_provider(model: str) -> str:
    """Detect upstream provider from model identifier."""
    m = (model or "").lower().strip()
    if m.startswith("openrouter/") or "openrouter" in m:
        return "openrouter"
    if "claude" in m or m.startswith("anthropic/"):
        return "anthropic"
    if "gemini" in m or m.startswith("google/"):
        return "gemini"
    if "qwen" in m or "qwq" in m or m.startswith("dashscope/"):
        return "dashscope"
    if "deepseek" in m:
        return "deepseek"
    if any(k in m for k in ("llama", "gemma2", "gemma-2", "mixtral", "groq")):
        return "groq"
    if any(k in m for k in ("mistral", "codestral", "pixtral", "ministral")):
        return "mistral"
    return "openai"


def resolve_upstream_url(provider: str, endpoint: str = "chat/completions") -> str:
    """Resolve destination URL for a given provider and endpoint."""
    settings = get_settings()
    p = provider.lower().strip()

    if p == "anthropic":
        base = os.environ.get("ANTHROPIC_BASE_URL", settings.anthropic_url).rstrip("/")
        if endpoint == "messages":
            if base.endswith("/v1/messages") or base.endswith("/messages"):
                return base
            elif base.endswith("/v1"):
                return f"{base}/messages"
            return f"{base}/v1/messages"
        else:
            if base.endswith("/chat/completions") or base.endswith("/v1/chat/completions"):
                return base
            elif base.endswith("/v1"):
                return f"{base}/chat/completions"
            return f"{base}/v1/chat/completions"

    elif p == "gemini":
        base = os.environ.get("GEMINI_BASE_URL", os.environ.get("GOOGLE_BASE_URL", settings.gemini_url)).rstrip("/")
        if base.endswith("/chat/completions") or base.endswith("/v1/chat/completions"):
            return base
        return f"{base}/chat/completions"

    elif p in ("dashscope", "qwen", "qwq"):
        base = os.environ.get("DASHSCOPE_BASE_URL", os.environ.get("QWEN_BASE_URL", settings.dashscope_url)).rstrip("/")
        if base.endswith("/chat/completions") or base.endswith("/v1/chat/completions"):
            return base
        return f"{base}/chat/completions"

    elif p == "deepseek":
        base = os.environ.get("DEEPSEEK_BASE_URL", settings.deepseek_url).rstrip("/")
        if base.endswith("/chat/completions") or base.endswith("/v1/chat/completions"):
            return base
        elif base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"

    elif p == "groq":
        base = os.environ.get("GROQ_BASE_URL", settings.groq_url).rstrip("/")
        if base.endswith("/chat/completions") or base.endswith("/v1/chat/completions"):
            return base
        return f"{base}/chat/completions"

    elif p == "mistral":
        base = os.environ.get("MISTRAL_BASE_URL", settings.mistral_url).rstrip("/")
        if base.endswith("/chat/completions") or base.endswith("/v1/chat/completions"):
            return base
        return f"{base}/chat/completions"

    elif p == "openrouter":
        base = os.environ.get("OPENROUTER_BASE_URL", settings.openrouter_url).rstrip("/")
        if base.endswith("/chat/completions") or base.endswith("/v1/chat/completions"):
            return base
        return f"{base}/chat/completions"

    else:
        base = os.environ.get("OPENAI_BASE_URL", settings.upstream_url).rstrip("/")
        if base.endswith("/chat/completions") or base.endswith("/v1/chat/completions"):
            return base
        elif base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"


def resolve_auth_headers(
    provider: str,
    request_headers: Dict[str, str],
    auth_header: Optional[str] = None,
    x_api_key: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve authorization headers for upstream provider."""
    headers: Dict[str, str] = {}
    p = provider.lower().strip()

    client_auth = auth_header or request_headers.get("authorization")
    client_key = x_api_key or request_headers.get("x-api-key")

    if p == "anthropic":
        key = (
            client_key
            or (client_auth.replace("Bearer ", "").strip() if client_auth else None)
            or get_api_key("anthropic")
        )
        if key:
            headers["x-api-key"] = key
        headers["anthropic-version"] = request_headers.get("anthropic-version", "2023-06-01")
        if "anthropic-beta" in request_headers:
            headers["anthropic-beta"] = request_headers["anthropic-beta"]
    else:
        bearer_val = client_auth
        if not bearer_val:
            key = get_api_key(p)
            if key:
                bearer_val = key if key.startswith("Bearer ") else f"Bearer {key}"
        if bearer_val:
            headers["Authorization"] = bearer_val

    # Forward vendor-specific headers
    for h, v in request_headers.items():
        hl = h.lower()
        if hl in ("openai-organization", "openai-project", "openai-beta", "x-goog-api-client", "x-goog-api-key"):
            headers[h] = v

    return headers


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="TokenGuard",
        description="Lightweight universal local proxy and circuit breaker for LLM calls (OpenAI, Gemini, Claude, Qwen, DeepSeek, Groq, Mistral)",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Enable CORS for browser dashboard and web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure static directory exists
    static_dir = Path(settings.static_path)
    static_dir.mkdir(parents=True, exist_ok=True)
    index_file = static_dir / "index.html"

    # Mount static assets
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    next_dir = static_dir / "_next"
    if next_dir.exists():
        app.mount("/_next", StaticFiles(directory=str(next_dir)), name="next_static")

    # --------------------------------------------------------------------------
    # Frontend Dashboard Routes
    # --------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard():
        """Serve the TokenGuard single-page dashboard."""
        if index_file.exists():
            return FileResponse(
                str(index_file),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return HTMLResponse("<h1>TokenGuard Dashboard is initializing...</h1>")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "TokenGuard",
            "version": "0.2.0",
            "kill_switch": circuit_breaker.kill_switch_active if circuit_breaker else False,
            "providers": ["openai", "anthropic", "gemini", "dashscope", "deepseek", "groq", "mistral", "openrouter"],
        }

    # --------------------------------------------------------------------------
    # Dashboard API Endpoints
    # --------------------------------------------------------------------------
    @app.get("/api/stats")
    async def api_stats():
        """Get aggregate metrics, spending status, and circuit breaker health."""
        if circuit_breaker is None:
            raise HTTPException(status_code=500, detail="Circuit breaker not initialized")

        current_settings = get_settings()
        metrics = await get_metrics(
            current_settings.db_path,
            time_window_hours=1.0,
            prices=circuit_breaker.prices,
        )
        cb_state = circuit_breaker.get_state()

        return {
            **metrics,
            **cb_state,
            "prices_count": len(circuit_breaker.prices),
        }

    @app.get("/api/requests")
    async def api_requests(
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ):
        """Retrieve recent LLM requests logged in SQLite."""
        current_settings = get_settings()
        records = await get_recent_requests(
            current_settings.db_path,
            limit=min(limit, 200),
            offset=offset,
            status_filter=status,
        )
        return {"requests": records, "count": len(records)}

    @app.get("/api/export")
    async def api_export(
        format: str = "jsonl",
        status: Optional[str] = None,
    ):
        """Export all logged requests in JSONL, HAR 1.2, CSV, or JSON format."""
        current_settings = get_settings()
        records = await get_all_requests_for_export(
            current_settings.db_path,
            status_filter=status,
        )

        format_clean = format.strip().lower()
        if format_clean == "har":
            har_obj = format_har_export(records, host=current_settings.host, port=current_settings.port)
            har_str = json.dumps(har_obj, indent=2)
            return Response(
                content=har_str,
                media_type="application/json",
                headers={
                    "Content-Disposition": 'attachment; filename="tokenguard_requests.har"',
                },
            )
        elif format_clean == "jsonl":
            jsonl_str = format_jsonl_export(records)
            return Response(
                content=jsonl_str,
                media_type="application/x-ndjson",
                headers={
                    "Content-Disposition": 'attachment; filename="tokenguard_requests.jsonl"',
                },
            )
        elif format_clean == "csv":
            csv_str = format_csv_export(records)
            return Response(
                content=csv_str,
                media_type="text/csv",
                headers={
                    "Content-Disposition": 'attachment; filename="tokenguard_requests.csv"',
                },
            )
        elif format_clean == "json":
            json_str = format_json_export(records)
            return Response(
                content=json_str,
                media_type="application/json",
                headers={
                    "Content-Disposition": 'attachment; filename="tokenguard_requests.json"',
                },
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported export format '{format}'. Supported formats: 'jsonl', 'har', 'json', 'csv'",
            )

    @app.get("/api/history")
    async def api_history(hours: int = 24):
        """Retrieve hourly spend and request breakdown for chart rendering."""
        current_settings = get_settings()
        history = await get_hourly_history(current_settings.db_path, hours=min(hours, 72))
        return {"history": history}

    @app.get("/api/stream")
    async def api_stream(request: Request):
        """Server-Sent Events endpoint broadcasting live state changes and logged requests in real-time."""
        async def event_generator():
            q = broadcaster.subscribe()
            try:
                if circuit_breaker is not None:
                    initial_state = circuit_breaker.get_state()
                    yield f"event: initial_state\ndata: {json.dumps(initial_state)}\n\n"

                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        payload = await asyncio.wait_for(q.get(), timeout=15.0)
                        event_name = payload.get("event", "message")
                        event_data = json.dumps(payload.get("data", {}))
                        yield f"event: {event_name}\ndata: {event_data}\n\n"
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                broadcaster.unsubscribe(q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/kill-switch")
    async def api_toggle_kill_switch(payload: Optional[Dict[str, Any]] = None):
        """Toggle or set the manual kill switch and broadcast state change."""
        if circuit_breaker is None:
            raise HTTPException(status_code=500, detail="Circuit breaker not initialized")

        active_target = payload.get("active") if payload and "active" in payload else None
        source = payload.get("source", "web") if payload else "web"
        new_state = circuit_breaker.toggle_kill_switch(active_target)
        action_text = "Kill Switch Engaged" if new_state else "Traffic Resumed"

        state_data = {
            **circuit_breaker.get_state(),
            "source": source,
            "action": action_text,
        }
        await broadcaster.broadcast("state_change", state_data)

        return {
            "kill_switch_active": new_state,
            "message": action_text,
            "state": state_data,
        }

    @app.post("/api/config")
    async def api_update_config(payload: Dict[str, Any]):
        """Dynamically update limits (hourly budget, daily budget, loop threshold, profile)."""
        if circuit_breaker is None:
            raise HTTPException(status_code=500, detail="Circuit breaker not initialized")

        hourly_limit = payload.get("hourly_limit")
        daily_limit = payload.get("daily_limit")
        loop_threshold = payload.get("loop_threshold")
        profile = payload.get("profile")
        source = payload.get("source", "web")

        circuit_breaker.update_limits(
            hourly_limit=hourly_limit,
            daily_limit=daily_limit,
            loop_threshold=loop_threshold,
            profile=profile,
        )

        action_text = f"Profile set to {str(profile).capitalize()}" if profile else "Limits updated"
        state_data = {
            **circuit_breaker.get_state(),
            "source": source,
            "action": action_text,
        }
        await broadcaster.broadcast("state_change", state_data)

        return {
            "status": "updated",
            "state": state_data,
        }

    @app.post("/api/reset")
    async def api_reset(payload: Optional[Dict[str, Any]] = None):
        """Reset in-memory circuit breaker state, spending tracking, and hash history."""
        if circuit_breaker is None:
            raise HTTPException(status_code=500, detail="Circuit breaker not initialized")

        source = payload.get("source", "web") if payload else "web"
        circuit_breaker.reset()
        state_data = {
            **circuit_breaker.get_state(),
            "source": source,
            "action": "Circuit breaker in-memory state cleared",
        }
        await broadcaster.broadcast("state_change", state_data)

        return {
            "status": "reset",
            "message": "Circuit breaker in-memory state and hash history cleared",
            "state": state_data,
        }

    @app.post("/api/clear-logs")
    async def api_clear_logs(payload: Optional[Dict[str, Any]] = None):
        """Clear database request logs and reset in-memory circuit breaker state."""
        if circuit_breaker is None:
            raise HTTPException(status_code=500, detail="Circuit breaker not initialized")

        source = payload.get("source", "web") if payload else "web"
        current_settings = get_settings()
        await clear_all_requests(current_settings.db_path)
        circuit_breaker.reset()

        state_data = {
            **circuit_breaker.get_state(),
            "source": source,
            "action": "All logs and metrics cleared",
        }
        await broadcaster.broadcast("logs_cleared", state_data)

        return {
            "status": "cleared",
            "message": "All database logs and in-memory circuit breaker state cleared",
            "state": state_data,
        }

    @app.post("/api/simulate")
    async def api_simulate(payload: Optional[Dict[str, Any]] = None):
        """Simulate LLM events (loop test, standard request, budget spike) for instant onboarding testing."""
        if circuit_breaker is None:
            raise HTTPException(status_code=500, detail="Circuit breaker not initialized")

        payload = payload or {}
        sim_type = payload.get("type", "loop")
        model = str(payload.get("model", "gpt-4o"))
        current_settings = get_settings()

        if sim_type == "loop":
            sample_messages = [{"role": "user", "content": "Rogue agent repeating identical prompt sequence for simulation"}]
            prompt_hash = circuit_breaker.compute_prompt_hash(sample_messages)
            est_tokens = circuit_breaker.estimate_prompt_tokens(sample_messages)

            results = []
            for i in range(1, circuit_breaker.loop_threshold + 1):
                try:
                    circuit_breaker.check_request(sample_messages, model)
                    cost = circuit_breaker.compute_cost(model, est_tokens, 150)
                    circuit_breaker.record_success(model, est_tokens, 150, cost, prompt_hash)
                    await _record_and_broadcast_request(
                        db_path=current_settings.db_path,
                        model=model,
                        prompt_tokens=est_tokens,
                        completion_tokens=150,
                        cost_usd=cost,
                        latency_ms=160 + i * 20,
                        prompt_hash=prompt_hash,
                        status="success",
                    )
                    results.append({"attempt": i, "status": "allowed", "cost": cost})
                except LoopDetectedError as err:
                    await _record_and_broadcast_request(
                        db_path=current_settings.db_path,
                        model=model,
                        prompt_tokens=est_tokens,
                        completion_tokens=0,
                        cost_usd=0.0,
                        latency_ms=0,
                        prompt_hash=prompt_hash,
                        status="blocked_loop",
                        blocked_reason=err.message,
                    )
                    results.append({"attempt": i, "status": "blocked_loop", "message": err.message})

            return {"simulation": "loop", "results": results, "threshold": circuit_breaker.loop_threshold}

        elif sim_type == "success":
            sample_messages = [{"role": "user", "content": f"Simulated user prompt #{int(time.time())}"}]
            prompt_hash = circuit_breaker.compute_prompt_hash(sample_messages)
            est_tokens = 85
            completion_tokens = 140
            cost = circuit_breaker.compute_cost(model, est_tokens, completion_tokens)
            circuit_breaker.record_success(model, est_tokens, completion_tokens, cost, prompt_hash)
            await _record_and_broadcast_request(
                db_path=current_settings.db_path,
                model=model,
                prompt_tokens=est_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                latency_ms=210,
                prompt_hash=prompt_hash,
                status="success",
            )
            return {"simulation": "success", "model": model, "cost": cost}

        elif sim_type == "budget":
            sample_messages = [{"role": "user", "content": "Simulated heavy payload budget overrun"}]
            prompt_hash = circuit_breaker.compute_prompt_hash(sample_messages)
            await _record_and_broadcast_request(
                db_path=current_settings.db_path,
                model=model,
                prompt_tokens=30000,
                completion_tokens=0,
                cost_usd=0.0,
                latency_ms=0,
                prompt_hash=prompt_hash,
                status="blocked_budget",
                blocked_reason="TokenGuard: budget limit exceeded",
            )
            return {"simulation": "budget", "status": "blocked_budget"}

        return {"status": "unknown simulation type"}

    @app.get("/api/prices")
    async def api_prices():
        """Get the current pricing table."""
        if circuit_breaker is None:
            raise HTTPException(status_code=500, detail="Circuit breaker not initialized")
        return {"prices": circuit_breaker.prices}

    # --------------------------------------------------------------------------
    # Universal Models Endpoint
    # --------------------------------------------------------------------------
    @app.get("/v1/models")
    async def proxy_models(request: Request, authorization: Optional[str] = Header(None)):
        """Transparently proxy the /v1/models endpoint."""
        if http_client is None:
            raise HTTPException(status_code=500, detail="HTTP client not initialized")

        headers = {}
        auth_header = (
            authorization
            or request.headers.get("authorization")
            or get_api_key("openai")
            or get_api_key("deepseek")
            or get_api_key("gemini")
            or get_api_key("dashscope")
            or get_api_key("groq")
            or get_api_key("mistral")
        )
        if auth_header:
            headers["Authorization"] = auth_header if auth_header.startswith("Bearer ") else f"Bearer {auth_header}"

        try:
            current_settings = get_settings()
            resp = await http_client.get(f"{current_settings.upstream_url.rstrip('/')}/v1/models", headers=headers)
            if resp.status_code == 200:
                return Response(content=resp.content, status_code=200, media_type="application/json")
        except Exception:
            pass

        # Fallback to registered models
        if circuit_breaker:
            model_items = [
                {"id": m_id, "object": "model", "created": 1700000000, "owned_by": detect_provider(m_id)}
                for m_id in circuit_breaker.prices.keys()
                if m_id != "default"
            ]
            return JSONResponse(status_code=200, content={"object": "list", "data": model_items})

        return JSONResponse(status_code=200, content={"object": "list", "data": [{"id": "gpt-4o", "object": "model"}]})

    # --------------------------------------------------------------------------
    # Anthropic Native Protocol (/v1/messages)
    # --------------------------------------------------------------------------
    @app.post("/v1/messages")
    async def anthropic_messages(
        request: Request,
        authorization: Optional[str] = Header(None),
        x_api_key: Optional[str] = Header(None),
        anthropic_version: Optional[str] = Header(None),
    ):
        """Native Anthropic Messages API proxy with full TokenGuard circuit breaker protection."""
        if circuit_breaker is None or http_client is None:
            raise HTTPException(status_code=500, detail="TokenGuard server initializing")

        current_settings = get_settings()

        # 1. Parse JSON payload
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": f"Invalid JSON payload: {str(e)}"},
                },
            )

        messages = body.get("messages", [])
        system_prompt = body.get("system")
        if system_prompt:
            # Combine system prompt with messages for prompt hash calculation
            check_payload = [{"role": "system", "content": system_prompt}] + list(messages)
        else:
            check_payload = messages

        model = str(body.get("model", "claude-3-5-sonnet"))
        is_streaming = bool(body.get("stream", False))
        prompt_hash = circuit_breaker.compute_prompt_hash(check_payload)

        # 2. Pre-flight Circuit Breaker Guards
        try:
            circuit_breaker.check_request(check_payload, model)
        except CircuitBreakerError as cb_err:
            status_tag = f"blocked_{cb_err.error_type}" if not cb_err.error_type.startswith("blocked_") else cb_err.error_type
            if cb_err.error_type == "budget_exceeded":
                status_tag = "blocked_budget"
            elif cb_err.error_type == "loop_detected":
                status_tag = "blocked_loop"
            elif cb_err.error_type == "kill_switch_active":
                status_tag = "blocked_killswitch"

            await _record_and_broadcast_request(
                db_path=current_settings.db_path,
                model=model,
                prompt_tokens=circuit_breaker.estimate_prompt_tokens(check_payload),
                completion_tokens=0,
                cost_usd=0.0,
                latency_ms=0,
                prompt_hash=prompt_hash,
                status=status_tag,
                blocked_reason=cb_err.message,
            )

            # Return Anthropic-native error schema
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": cb_err.message,
                    },
                },
            )

        # 3. Resolve Destination and Headers
        target_url = resolve_upstream_url("anthropic", endpoint="messages")
        forward_headers = resolve_auth_headers(
            provider="anthropic",
            request_headers=dict(request.headers),
            auth_header=authorization,
            x_api_key=x_api_key,
        )
        forward_headers["Content-Type"] = "application/json"
        if is_streaming:
            forward_headers["Accept"] = "text/event-stream"

        # 4. Handle Streaming vs Non-Streaming
        if is_streaming:
            return await _handle_anthropic_streaming(
                http_client=http_client,
                target_url=target_url,
                body=body,
                headers=forward_headers,
                model=model,
                prompt_hash=prompt_hash,
                circuit_breaker=circuit_breaker,
                db_path=current_settings.db_path,
                messages=check_payload,
            )

        return await _handle_anthropic_non_streaming(
            http_client=http_client,
            target_url=target_url,
            body=body,
            headers=forward_headers,
            model=model,
            prompt_hash=prompt_hash,
            circuit_breaker=circuit_breaker,
            db_path=current_settings.db_path,
            messages=check_payload,
        )

    # --------------------------------------------------------------------------
    # Universal /v1/chat/completions (OpenAI, Gemini, DashScope/Qwen, Claude, etc.)
    # --------------------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        """Universal OpenAI-compatible chat completions proxy with auto-routing across all LLM providers."""
        if circuit_breaker is None or http_client is None:
            raise HTTPException(status_code=500, detail="TokenGuard server initializing")

        current_settings = get_settings()

        # 1. Parse JSON payload
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Invalid JSON payload: {str(e)}", "type": "invalid_request_error"}},
            )

        messages = body.get("messages", [])
        model = str(body.get("model", "default"))
        is_streaming = bool(body.get("stream", False))
        prompt_hash = circuit_breaker.compute_prompt_hash(messages)

        # 2. Pre-flight Circuit Breaker Guards
        try:
            circuit_breaker.check_request(messages, model)
        except CircuitBreakerError as cb_err:
            status_tag = f"blocked_{cb_err.error_type}" if not cb_err.error_type.startswith("blocked_") else cb_err.error_type
            if cb_err.error_type == "budget_exceeded":
                status_tag = "blocked_budget"
            elif cb_err.error_type == "loop_detected":
                status_tag = "blocked_loop"
            elif cb_err.error_type == "kill_switch_active":
                status_tag = "blocked_killswitch"

            await _record_and_broadcast_request(
                db_path=current_settings.db_path,
                model=model,
                prompt_tokens=circuit_breaker.estimate_prompt_tokens(messages),
                completion_tokens=0,
                cost_usd=0.0,
                latency_ms=0,
                prompt_hash=prompt_hash,
                status=status_tag,
                blocked_reason=cb_err.message,
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=cb_err.to_dict(),
            )

        # 3. Detect Provider and Resolve Routing
        provider = detect_provider(model)

        # Special Adapter: OpenAI client calling Claude model -> translate to Anthropic messages protocol
        if provider == "anthropic":
            anthropic_base = os.environ.get("ANTHROPIC_BASE_URL", current_settings.anthropic_url)
            # If standard Anthropic URL is used, use Anthropic Messages adapter
            if "api.anthropic.com" in anthropic_base:
                return await _handle_openai_to_anthropic_adapter(
                    http_client=http_client,
                    body=body,
                    headers=dict(request.headers),
                    model=model,
                    prompt_hash=prompt_hash,
                    circuit_breaker=circuit_breaker,
                    db_path=current_settings.db_path,
                    messages=messages,
                    authorization=authorization,
                )

        # Standard OpenAI-compatible upstream routing (OpenAI, Gemini, Qwen/DashScope, DeepSeek, Groq, Mistral, OpenRouter)
        upstream_url = resolve_upstream_url(provider, endpoint="chat/completions")
        forward_headers = resolve_auth_headers(
            provider=provider,
            request_headers=dict(request.headers),
            auth_header=authorization,
        )
        forward_headers["Content-Type"] = "application/json"
        forward_headers["Accept"] = "text/event-stream" if is_streaming else "application/json"

        # 4. Handle Streaming vs Non-Streaming
        if is_streaming:
            if "stream_options" not in body or not isinstance(body["stream_options"], dict):
                body["stream_options"] = {"include_usage": True}
            else:
                body["stream_options"]["include_usage"] = True

            return await _handle_streaming_chat(
                http_client=http_client,
                target_url=upstream_url,
                body=body,
                headers=forward_headers,
                model=model,
                prompt_hash=prompt_hash,
                circuit_breaker=circuit_breaker,
                db_path=current_settings.db_path,
                messages=messages,
            )

        return await _handle_non_streaming_chat(
            http_client=http_client,
            target_url=upstream_url,
            body=body,
            headers=forward_headers,
            model=model,
            prompt_hash=prompt_hash,
            circuit_breaker=circuit_breaker,
            db_path=current_settings.db_path,
            messages=messages,
        )

    # --------------------------------------------------------------------------
    # Universal /v1/embeddings Endpoint
    # --------------------------------------------------------------------------
    @app.post("/v1/embeddings")
    async def embeddings(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        """Proxy embeddings endpoint with cost calculation and circuit breaker checks."""
        if circuit_breaker is None or http_client is None:
            raise HTTPException(status_code=500, detail="TokenGuard server initializing")

        current_settings = get_settings()
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": {"message": str(e), "type": "invalid_request_error"}})

        model = str(body.get("model", "text-embedding-3-small"))
        input_data = body.get("input", "")
        prompt_hash = circuit_breaker.compute_prompt_hash(input_data)

        try:
            circuit_breaker.check_request(input_data, model)
        except CircuitBreakerError as cb_err:
            return JSONResponse(status_code=429, content=cb_err.to_dict())

        provider = detect_provider(model)
        target_url = f"{current_settings.upstream_url.rstrip('/')}/v1/embeddings"
        forward_headers = resolve_auth_headers(provider, dict(request.headers), auth_header=authorization)
        forward_headers["Content-Type"] = "application/json"

        start_time = time.perf_counter()
        try:
            resp = await http_client.post(target_url, json=body, headers=forward_headers)
        except Exception as exc:
            circuit_breaker.remove_last_hash(prompt_hash)
            return JSONResponse(status_code=502, content={"error": {"message": str(exc), "type": "bad_gateway"}})

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        if resp.status_code != 200:
            circuit_breaker.remove_last_hash(prompt_hash)
            return Response(content=resp.content, status_code=resp.status_code, headers={"Content-Type": "application/json"})

        resp_data = resp.json()
        usage = resp_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens") or circuit_breaker.estimate_prompt_tokens(input_data)
        cost_usd = circuit_breaker.compute_cost(model, prompt_tokens, 0)
        circuit_breaker.record_success(model, prompt_tokens, 0, cost_usd, prompt_hash)

        await _record_and_broadcast_request(
            db_path=current_settings.db_path,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            prompt_hash=prompt_hash,
            status="success",
        )

        return Response(content=resp.content, status_code=200, headers={"Content-Type": "application/json"})

    return app


# ------------------------------------------------------------------------------
# Handlers: Anthropic Native Protocol
# ------------------------------------------------------------------------------
async def _handle_anthropic_non_streaming(
    http_client: httpx.AsyncClient,
    target_url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    model: str,
    prompt_hash: str,
    circuit_breaker: CircuitBreaker,
    db_path: Path,
    messages: Any,
) -> Response:
    """Execute non-streaming request against Anthropic Messages API."""
    start_time = time.perf_counter()
    try:
        upstream_resp = await http_client.post(target_url, json=body, headers=headers)
    except Exception as exc:
        logger.error(f"Anthropic upstream request failed: {exc}")
        circuit_breaker.remove_last_hash(prompt_hash)
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": f"TokenGuard upstream error: {str(exc)}"}},
        )

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    if upstream_resp.status_code != 200:
        circuit_breaker.remove_last_hash(prompt_hash)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers={"Content-Type": "application/json"},
        )

    try:
        resp_data = upstream_resp.json()
    except Exception:
        circuit_breaker.remove_last_hash(prompt_hash)
        return Response(content=upstream_resp.content, status_code=upstream_resp.status_code, headers={"Content-Type": "application/json"})

    usage = resp_data.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or circuit_breaker.estimate_prompt_tokens(messages))
    completion_tokens = int(usage.get("output_tokens") or 0)

    cost_usd = circuit_breaker.compute_cost(model, prompt_tokens, completion_tokens)
    circuit_breaker.record_success(model, prompt_tokens, completion_tokens, cost_usd, prompt_hash)

    await _record_and_broadcast_request(
        db_path=db_path,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        prompt_hash=prompt_hash,
        status="success",
    )

    return Response(content=upstream_resp.content, status_code=200, headers={"Content-Type": "application/json"})


async def _handle_anthropic_streaming(
    http_client: httpx.AsyncClient,
    target_url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    model: str,
    prompt_hash: str,
    circuit_breaker: CircuitBreaker,
    db_path: Path,
    messages: Any,
) -> StreamingResponse:
    """Handle streaming SSE completions from Anthropic Messages API."""
    start_time = time.perf_counter()
    try:
        upstream_req = http_client.build_request("POST", target_url, json=body, headers=headers)
        upstream_resp = await http_client.send(upstream_req, stream=True)
    except Exception as exc:
        circuit_breaker.remove_last_hash(prompt_hash)
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": f"TokenGuard upstream error: {str(exc)}"}},
        )

    if upstream_resp.status_code != 200:
        circuit_breaker.remove_last_hash(prompt_hash)
        error_content = await upstream_resp.aread()
        await upstream_resp.aclose()
        return Response(content=error_content, status_code=upstream_resp.status_code, headers={"Content-Type": "application/json"})

    async def anthropic_sse_generator() -> AsyncGenerator[bytes, None]:
        buffer = ""
        prompt_tokens = 0
        completion_tokens = 0
        usage_found = False

        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
                try:
                    text_chunk = chunk.decode("utf-8", errors="ignore")
                    buffer += text_chunk

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue

                        data_str = line[5:].strip()
                        if not data_str:
                            continue

                        try:
                            payload = json.loads(data_str)
                            ptype = payload.get("type")
                            if ptype == "message_start":
                                msg = payload.get("message", {})
                                u = msg.get("usage", {})
                                if "input_tokens" in u:
                                    prompt_tokens = int(u["input_tokens"])
                                    usage_found = True
                            elif ptype == "message_delta":
                                u = payload.get("usage", {})
                                if "output_tokens" in u:
                                    completion_tokens = int(u["output_tokens"])
                                    usage_found = True
                        except Exception:
                            pass
                except Exception:
                    pass
        finally:
            await upstream_resp.aclose()
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            if not usage_found or (prompt_tokens == 0 and completion_tokens == 0):
                prompt_tokens = circuit_breaker.estimate_prompt_tokens(messages)
                completion_tokens = max(1, completion_tokens)

            cost_usd = circuit_breaker.compute_cost(model, prompt_tokens, completion_tokens)
            circuit_breaker.record_success(model, prompt_tokens, completion_tokens, cost_usd, prompt_hash)

            try:
                await _record_and_broadcast_request(
                    db_path=db_path,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    prompt_hash=prompt_hash,
                    status="success",
                )
            except Exception as e:
                logger.error(f"Failed to log Anthropic stream: {e}")

    return StreamingResponse(
        anthropic_sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------------------
# Handlers: OpenAI to Anthropic Schema Adapter
# ------------------------------------------------------------------------------
async def _handle_openai_to_anthropic_adapter(
    http_client: httpx.AsyncClient,
    body: Dict[str, Any],
    headers: Dict[str, str],
    model: str,
    prompt_hash: str,
    circuit_breaker: CircuitBreaker,
    db_path: Path,
    messages: List[Dict[str, Any]],
    authorization: Optional[str] = None,
) -> Response:
    """Translate OpenAI chat completion request to Anthropic Messages API and adapt response back."""
    target_url = resolve_upstream_url("anthropic", endpoint="messages")
    forward_headers = resolve_auth_headers("anthropic", headers, auth_header=authorization)
    forward_headers["Content-Type"] = "application/json"

    # 1. Transform OpenAI messages to Anthropic format
    system_prompt = None
    anthropic_msgs = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content if system_prompt is None else f"{system_prompt}\n{content}"
        else:
            anthropic_msgs.append({"role": role, "content": content})

    anthropic_payload: Dict[str, Any] = {
        "model": model,
        "messages": anthropic_msgs,
        "max_tokens": body.get("max_tokens", 4096),
    }
    if system_prompt:
        anthropic_payload["system"] = system_prompt
    if "temperature" in body:
        anthropic_payload["temperature"] = body["temperature"]
    if "top_p" in body:
        anthropic_payload["top_p"] = body["top_p"]

    is_streaming = bool(body.get("stream", False))
    anthropic_payload["stream"] = is_streaming

    if is_streaming:
        # Stream Anthropic deltas translated into OpenAI chunks
        start_time = time.perf_counter()
        try:
            req = http_client.build_request("POST", target_url, json=anthropic_payload, headers=forward_headers)
            upstream_resp = await http_client.send(req, stream=True)
        except Exception as exc:
            circuit_breaker.remove_last_hash(prompt_hash)
            return JSONResponse(status_code=502, content={"error": {"message": str(exc), "type": "bad_gateway"}})

        if upstream_resp.status_code != 200:
            circuit_breaker.remove_last_hash(prompt_hash)
            err_bytes = await upstream_resp.aread()
            await upstream_resp.aclose()
            return Response(content=err_bytes, status_code=upstream_resp.status_code, headers={"Content-Type": "application/json"})

        async def adapted_stream_generator() -> AsyncGenerator[bytes, None]:
            buffer = ""
            prompt_tokens = 0
            completion_tokens = 0
            chunk_id = f"chatcmpl-claude-{int(time.time())}"

            try:
                async for chunk in upstream_resp.aiter_bytes():
                    buffer += chunk.decode("utf-8", errors="ignore")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue

                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue

                        try:
                            payload = json.loads(data_str)
                            ptype = payload.get("type")
                            if ptype == "message_start":
                                u = payload.get("message", {}).get("usage", {})
                                if "input_tokens" in u:
                                    prompt_tokens = int(u["input_tokens"])
                            elif ptype == "content_block_delta":
                                text = payload.get("delta", {}).get("text", "")
                                openai_chunk = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(openai_chunk)}\n\n".encode("utf-8")
                            elif ptype == "message_delta":
                                u = payload.get("usage", {})
                                if "output_tokens" in u:
                                    completion_tokens = int(u["output_tokens"])
                        except Exception:
                            pass

                # Final stop chunk
                stop_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                yield f"data: {json.dumps(stop_chunk)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            finally:
                await upstream_resp.aclose()
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                if prompt_tokens == 0:
                    prompt_tokens = circuit_breaker.estimate_prompt_tokens(messages)
                if completion_tokens == 0:
                    completion_tokens = 50

                cost_usd = circuit_breaker.compute_cost(model, prompt_tokens, completion_tokens)
                circuit_breaker.record_success(model, prompt_tokens, completion_tokens, cost_usd, prompt_hash)

                try:
                    await _record_and_broadcast_request(
                        db_path=db_path,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=cost_usd,
                        latency_ms=latency_ms,
                        prompt_hash=prompt_hash,
                        status="success",
                    )
                except Exception as e:
                    logger.error(f"Failed to record adapted Claude stream: {e}")

        return StreamingResponse(
            adapted_stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # Non-streaming OpenAI to Anthropic
    start_time = time.perf_counter()
    try:
        resp = await http_client.post(target_url, json=anthropic_payload, headers=forward_headers)
    except Exception as exc:
        circuit_breaker.remove_last_hash(prompt_hash)
        return JSONResponse(status_code=502, content={"error": {"message": str(exc), "type": "bad_gateway"}})

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    if resp.status_code != 200:
        circuit_breaker.remove_last_hash(prompt_hash)
        return Response(content=resp.content, status_code=resp.status_code, headers={"Content-Type": "application/json"})

    try:
        data = resp.json()
    except Exception:
        circuit_breaker.remove_last_hash(prompt_hash)
        return Response(content=resp.content, status_code=resp.status_code, headers={"Content-Type": "application/json"})

    # Extract text from Anthropic content blocks
    text_content = ""
    for block in data.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            text_content += block.get("text", "")

    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("input_tokens") or circuit_breaker.estimate_prompt_tokens(messages))
    completion_tokens = int(usage.get("output_tokens") or 0)

    cost_usd = circuit_breaker.compute_cost(model, prompt_tokens, completion_tokens)
    circuit_breaker.record_success(model, prompt_tokens, completion_tokens, cost_usd, prompt_hash)

    await _record_and_broadcast_request(
        db_path=db_path,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        prompt_hash=prompt_hash,
        status="success",
    )

    openai_response = {
        "id": f"chatcmpl-{data.get('id', 'claude')}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text_content},
                "finish_reason": "stop" if data.get("stop_reason") == "end_turn" else data.get("stop_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    return JSONResponse(status_code=200, content=openai_response)


# ------------------------------------------------------------------------------
# Handlers: Standard OpenAI-Compatible Upstream
# ------------------------------------------------------------------------------
async def _handle_non_streaming_chat(
    http_client: httpx.AsyncClient,
    target_url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    model: str,
    prompt_hash: str,
    circuit_breaker: CircuitBreaker,
    db_path: Path,
    messages: Any,
) -> Response:
    """Execute standard non-streaming chat completions upstream request."""
    start_time = time.perf_counter()

    try:
        upstream_resp = await http_client.post(
            target_url,
            json=body,
            headers=headers,
        )
    except Exception as exc:
        logger.error(f"Upstream request failed: {exc}")
        circuit_breaker.remove_last_hash(prompt_hash)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"TokenGuard upstream error: {str(exc)}", "type": "bad_gateway"}},
        )

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    if upstream_resp.status_code != 200:
        circuit_breaker.remove_last_hash(prompt_hash)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers={"Content-Type": upstream_resp.headers.get("Content-Type", "application/json")},
        )

    try:
        resp_data = upstream_resp.json()
    except Exception:
        circuit_breaker.remove_last_hash(prompt_hash)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers={"Content-Type": "application/json"},
        )

    usage = resp_data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = circuit_breaker.estimate_prompt_tokens(messages)
    else:
        prompt_tokens = int(prompt_tokens)
    completion_tokens = int(usage.get("completion_tokens") or 0)

    cost_usd = circuit_breaker.compute_cost(model, prompt_tokens, completion_tokens)
    circuit_breaker.record_success(model, prompt_tokens, completion_tokens, cost_usd, prompt_hash)

    await _record_and_broadcast_request(
        db_path=db_path,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        prompt_hash=prompt_hash,
        status="success",
        blocked_reason=None,
    )

    return Response(
        content=upstream_resp.content,
        status_code=200,
        headers={"Content-Type": "application/json"},
    )


async def _handle_streaming_chat(
    http_client: httpx.AsyncClient,
    target_url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    model: str,
    prompt_hash: str,
    circuit_breaker: CircuitBreaker,
    db_path: Path,
    messages: Any,
) -> StreamingResponse:
    """Handle streaming SSE completions with on-the-fly token usage extraction."""
    start_time = time.perf_counter()

    try:
        upstream_req = http_client.build_request(
            "POST",
            target_url,
            json=body,
            headers=headers,
        )
        upstream_resp = await http_client.send(upstream_req, stream=True)
    except Exception as exc:
        logger.error(f"Failed to initiate upstream stream: {exc}")
        circuit_breaker.remove_last_hash(prompt_hash)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"TokenGuard upstream error: {str(exc)}", "type": "bad_gateway"}},
        )

    if upstream_resp.status_code != 200:
        circuit_breaker.remove_last_hash(prompt_hash)
        error_content = await upstream_resp.aread()
        await upstream_resp.aclose()
        return Response(
            content=error_content,
            status_code=upstream_resp.status_code,
            headers={"Content-Type": upstream_resp.headers.get("Content-Type", "application/json")},
        )

    async def sse_generator() -> AsyncGenerator[bytes, None]:
        buffer = ""
        prompt_tokens = 0
        completion_tokens = 0
        usage_found = False

        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk

                try:
                    text_chunk = chunk.decode("utf-8", errors="ignore")
                    buffer += text_chunk

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue

                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            continue

                        if '"usage"' in data_str:
                            try:
                                payload = json.loads(data_str)
                                usage = payload.get("usage")
                                if usage and isinstance(usage, dict):
                                    p_tok = usage.get("prompt_tokens")
                                    c_tok = usage.get("completion_tokens")
                                    if p_tok is not None:
                                        prompt_tokens = int(p_tok)
                                    if c_tok is not None:
                                        completion_tokens = int(c_tok)
                                    usage_found = True
                            except Exception:
                                pass
                except Exception:
                    pass

        finally:
            await upstream_resp.aclose()
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            if not usage_found or (prompt_tokens == 0 and completion_tokens == 0):
                prompt_tokens = circuit_breaker.estimate_prompt_tokens(messages)
                completion_tokens = max(1, completion_tokens)

            cost_usd = circuit_breaker.compute_cost(model, prompt_tokens, completion_tokens)
            circuit_breaker.record_success(model, prompt_tokens, completion_tokens, cost_usd, prompt_hash)

            try:
                await _record_and_broadcast_request(
                    db_path=db_path,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    prompt_hash=prompt_hash,
                    status="success",
                    blocked_reason=None,
                )
            except Exception as e:
                logger.error(f"Failed to log streaming request to DB: {e}")

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Default application instance for uvicorn
app = create_app()

