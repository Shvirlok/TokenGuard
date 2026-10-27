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
    get_recent_spends,
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
    BurnRateExceededError,
    CircuitBreaker,
    CircuitBreakerError,
    ContextBallooningError,
    KillSwitchActiveError,
    LoopDetectedError,
    ToolPingPongError,
    parse_burn_rate,
)

logger = logging.getLogger("tokenguard.proxy")


class StreamAccumulator:
    """Lightweight background character/token accumulator for SSE streams.

    Processes SSE chunks asynchronously in a background task without blocking
    the pass-through streaming loop. Extracts native provider usage blocks
    if available (OpenAI stream_options include_usage, Anthropic message_start/message_delta),
    or accumulates characters and estimates tokens if native usage is omitted.
    """

    def __init__(self, is_anthropic: bool = False, prompt_tokens_estimate: int = 0):
        self.is_anthropic = is_anthropic
        self.prompt_tokens_estimate = prompt_tokens_estimate
        self.prompt_tokens: Optional[int] = None
        self.completion_tokens: Optional[int] = None
        self.accumulated_chars: int = 0
        self.accumulated_tokens: int = 0
        self.usage_found: bool = False
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=2000)
        self._task: Optional[asyncio.Task] = None
        self._buffer: str = ""

    def start(self) -> None:
        self._task = asyncio.create_task(self._process_queue())

    def feed(self, chunk: bytes) -> None:
        """Non-blocking chunk ingestion into background queue."""
        try:
            self._queue.put_nowait(chunk)
        except (asyncio.QueueFull, Exception):
            pass

    async def _process_queue(self) -> None:
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break

                try:
                    text = chunk.decode("utf-8", errors="ignore")
                    self._buffer += text

                    while "\n" in self._buffer:
                        line, self._buffer = self._buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue

                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue

                        self._process_line(data_str)
                except Exception:
                    pass
        except asyncio.CancelledError:
            pass

    def _process_line(self, data_str: str) -> None:
        try:
            if not (data_str.startswith("{") and data_str.endswith("}")):
                return

            payload = json.loads(data_str)
            if self.is_anthropic:
                self._process_anthropic_payload(payload)
            else:
                self._process_openai_payload(payload)
        except Exception:
            pass

    def _process_openai_payload(self, payload: Dict[str, Any]) -> None:
        usage = payload.get("usage")
        if usage and isinstance(usage, dict):
            p_tok = usage.get("prompt_tokens")
            c_tok = usage.get("completion_tokens")
            if p_tok is not None:
                self.prompt_tokens = int(p_tok)
            if c_tok is not None:
                self.completion_tokens = int(c_tok)
            self.usage_found = True

        choices = payload.get("choices")
        if choices and isinstance(choices, list):
            for choice in choices:
                delta = choice.get("delta")
                if delta and isinstance(delta, dict):
                    content = delta.get("content")
                    if content and isinstance(content, str):
                        self.accumulated_chars += len(content)
                        self.accumulated_tokens += max(1, len(content) // 4)

    def _process_anthropic_payload(self, payload: Dict[str, Any]) -> None:
        ptype = payload.get("type")
        if ptype == "message_start":
            msg = payload.get("message", {})
            u = msg.get("usage", {})
            if "input_tokens" in u:
                self.prompt_tokens = int(u["input_tokens"])
                self.usage_found = True
        elif ptype == "message_delta":
            u = payload.get("usage", {})
            if "output_tokens" in u:
                self.completion_tokens = int(u["output_tokens"])
                self.usage_found = True
        elif ptype == "content_block_delta":
            delta = payload.get("delta", {})
            text = delta.get("text", "")
            if text:
                self.accumulated_chars += len(text)
                self.accumulated_tokens += max(1, len(text) // 4)

    async def finish(self) -> Tuple[int, int]:
        """Signal end of stream and return (prompt_tokens, completion_tokens)."""
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=0.5)
            except Exception:
                if not self._task.done():
                    self._task.cancel()

        if self.prompt_tokens is not None and self.prompt_tokens > 0:
            final_prompt = self.prompt_tokens
        else:
            final_prompt = self.prompt_tokens_estimate

        if self.completion_tokens is not None and self.completion_tokens > 0:
            final_completion = self.completion_tokens
        elif self.accumulated_tokens > 0:
            final_completion = self.accumulated_tokens
        elif self.accumulated_chars > 0:
            final_completion = max(1, self.accumulated_chars // 4)
        else:
            final_completion = 1

        return final_prompt, final_completion


def _detect_task_update(
    headers: Dict[str, str],
    body: Dict[str, Any],
    messages: Any,
) -> bool:
    """Detect if request contains an explicit task status update."""
    for h in ("x-task-status", "x-status-update", "x-task-state", "x-task-complete"):
        if h in headers and headers[h]:
            return True

    for f in ("task_status", "status_update", "task_state", "status"):
        if body.get(f):
            return True

    if isinstance(messages, list):
        for msg in reversed(messages[-3:]):
            if isinstance(msg, dict):
                content = str(msg.get("content", "")).lower()
                if any(k in content for k in (
                    "task_status",
                    "status: complete",
                    "status: done",
                    "status: in_progress",
                    "status: update",
                    "[task_update]",
                    "[status_update]",
                    "task status:",
                )):
                    return True
                tool_calls = msg.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        fn = tc.get("function", {}).get("name", "").lower()
                        if any(k in fn for k in ("task_update", "update_task", "set_status", "complete_task")):
                            return True
    return False


def _detect_cache_hit(
    headers: Dict[str, str],
    body: Dict[str, Any],
) -> bool:
    """Detect if request is marked as a prompt cache hit."""
    for h in ("x-cache-hit", "x-prompt-cache-hit", "x-cache"):
        val = str(headers.get(h, "")).lower()
        if val in ("true", "1", "hit"):
            return True

    if body.get("cache_hit") is True or body.get("prompt_cache_hit", False):
        return True
    if int(body.get("prompt_cache_hit_tokens", 0) or 0) > 0:
        return True

    return False


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


def get_guard() -> CircuitBreaker:
    """Return active CircuitBreaker instance, initializing a default if not yet created."""
    global circuit_breaker
    if circuit_breaker is None:
        settings = get_settings()
        circuit_breaker = CircuitBreaker(
            hourly_limit=settings.hourly_limit,
            daily_limit=settings.daily_limit,
            loop_threshold=settings.loop_threshold,
            loop_window_size=settings.loop_window_size,
            loop_window_seconds=settings.loop_window_seconds,
            prices_path=settings.prices_path,
            kill_switch=settings.kill_switch,
            active_profile=settings.profile,
            max_burn_rate=settings.max_burn_rate,
            enable_context_ballooning=settings.enable_context_ballooning,
        )
    return circuit_breaker


def get_client() -> httpx.AsyncClient:
    """Return active httpx.AsyncClient instance."""
    global http_client
    if http_client is None:
        settings = get_settings()
        http_client = httpx.AsyncClient(
            base_url=settings.upstream_url,
            timeout=httpx.Timeout(settings.timeout, connect=10.0),
        )
    return http_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for setup and teardown."""
    global circuit_breaker, http_client
    settings = get_settings()

    # Initialize SQLite database
    await init_db(settings.db_path)

    # Initialize CircuitBreaker engine with burn rate and context ballooning guard
    circuit_breaker = CircuitBreaker(
        hourly_limit=settings.hourly_limit,
        daily_limit=settings.daily_limit,
        loop_threshold=settings.loop_threshold,
        loop_window_size=settings.loop_window_size,
        loop_window_seconds=settings.loop_window_seconds,
        prices_path=settings.prices_path,
        kill_switch=settings.kill_switch,
        active_profile=settings.profile,
        max_burn_rate=settings.max_burn_rate,
        enable_context_ballooning=settings.enable_context_ballooning,
    )

    # Seed recent spends from SQLite to back sliding window burn-rate engine
    recent_spends = await get_recent_spends(settings.db_path, hours=24.0)
    circuit_breaker.seed_initial_spends(recent_spends)

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

    @app.get("/api/events")
    @app.get("/api/stream")
    async def api_events(request: Request):
        """Server-Sent Events endpoint broadcasting live state changes, logged requests, and reactive metrics in real-time."""
        async def event_generator():
            q = broadcaster.subscribe()
            try:
                yield f"event: connected\ndata: {json.dumps({'status': 'connected', 'timestamp': time.time()})}\n\n"

                if circuit_breaker is not None:
                    current_settings = get_settings()
                    try:
                        init_metrics = await get_metrics(current_settings.db_path)
                        init_metrics = {**circuit_breaker.get_state(), **init_metrics}
                        yield f"event: stats\ndata: {json.dumps(init_metrics)}\n\n"
                    except Exception:
                        initial_state = circuit_breaker.get_state()
                        yield f"event: stats\ndata: {json.dumps(initial_state)}\n\n"

                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        payload = await asyncio.wait_for(q.get(), timeout=1.0)
                        event_name = payload.get("event", "message")
                        event_data = json.dumps(payload.get("data", {}))
                        yield f"event: {event_name}\ndata: {event_data}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            except (asyncio.CancelledError, GeneratorExit):
                pass
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
        max_burn_rate = payload.get("max_burn_rate")
        enable_context_ballooning = payload.get("enable_context_ballooning")
        source = payload.get("source", "web")

        circuit_breaker.update_limits(
            hourly_limit=hourly_limit,
            daily_limit=daily_limit,
            loop_threshold=loop_threshold,
            profile=profile,
            max_burn_rate=max_burn_rate,
            enable_context_ballooning=enable_context_ballooning,
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

        elif sim_type in ("tool_loop", "tool"):
            tool_name = "read_file"
            tool_args = {"path": "/workspace/config.json"}
            sample_messages = [
                {"role": "user", "content": "Autonomous agent executing tool calls"},
                {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Error: FileNotFoundError: /workspace/config.json does not exist"}
            ]
            prompt_hash = circuit_breaker.compute_prompt_hash(sample_messages)
            est_tokens = circuit_breaker.estimate_prompt_tokens(sample_messages)
            results = []
            sim_session = f"sim_tool_{int(time.time())}"

            for i in range(1, circuit_breaker.tool_loop_threshold + 1):
                try:
                    circuit_breaker.check_tool_loop(sim_session, tool_name, tool_args)
                    circuit_breaker.record_tool_call(sim_session, tool_name, tool_args)
                    cost = circuit_breaker.compute_cost(model, est_tokens, 80)
                    circuit_breaker.record_success(model, est_tokens, 80, cost, prompt_hash)
                    await _record_and_broadcast_request(
                        db_path=current_settings.db_path,
                        model=model,
                        prompt_tokens=est_tokens,
                        completion_tokens=80,
                        cost_usd=cost,
                        latency_ms=120 + i * 20,
                        prompt_hash=prompt_hash,
                        status="success",
                    )
                    results.append({"attempt": i, "status": "allowed", "tool": tool_name})
                except ToolPingPongError as err:
                    await _record_and_broadcast_request(
                        db_path=current_settings.db_path,
                        model=model,
                        prompt_tokens=est_tokens,
                        completion_tokens=0,
                        cost_usd=0.0,
                        latency_ms=0,
                        prompt_hash=prompt_hash,
                        status="blocked_tool_loop",
                        blocked_reason=err.message,
                    )
                    results.append({"attempt": i, "status": "blocked_tool_loop", "message": err.message})

            return {"simulation": "tool_loop", "results": results, "threshold": circuit_breaker.tool_loop_threshold}

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

    @app.post("/api/test/simulate-tool-loop")
    async def api_simulate_tool_loop(payload: Optional[Dict[str, Any]] = None):
        """Programmatically feed consecutive identical tool-call payloads through guard.py, trip the circuit breaker, and broadcast over SSE."""
        if circuit_breaker is None:
            raise HTTPException(status_code=500, detail="Circuit breaker not initialized")

        payload = payload or {}
        model = str(payload.get("model", "gpt-4o"))
        tool_name = str(payload.get("tool_name", "read_file"))
        tool_args = payload.get("arguments", {"path": "/workspace/credentials.env"})
        current_settings = get_settings()

        sim_session = f"sim_tool_loop_{int(time.time() * 1000)}"
        results = []
        sample_messages = [
            {"role": "user", "content": "Autonomous agent executing tool calls"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_sim_1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_sim_1", "content": f"Error: FileNotFoundError: {tool_args.get('path', '')}"},
        ]
        prompt_hash = circuit_breaker.compute_prompt_hash(sample_messages)
        est_tokens = circuit_breaker.estimate_prompt_tokens(sample_messages)

        start_time = time.time()
        for i in range(1, circuit_breaker.tool_loop_threshold + 1):
            try:
                circuit_breaker.check_tool_loop(sim_session, tool_name, tool_args)
                circuit_breaker.record_tool_call(sim_session, tool_name, tool_args)
                cost = circuit_breaker.compute_cost(model, est_tokens, 80)
                circuit_breaker.record_success(model, est_tokens, 80, cost, prompt_hash)
                await _record_and_broadcast_request(
                    db_path=current_settings.db_path,
                    model=model,
                    prompt_tokens=est_tokens,
                    completion_tokens=80,
                    cost_usd=cost,
                    latency_ms=120 + i * 20,
                    prompt_hash=prompt_hash,
                    status="success",
                )
                results.append({"attempt": i, "status": "allowed", "tool": tool_name})
            except ToolPingPongError:
                elapsed_s = max(0.1, round(time.time() - start_time + 4.2, 1))
                trip_reason = f"[TOOL RETRY LOOP: HARD STOP] Identical tool arguments repeated {i}x within {elapsed_s}s: tool='{tool_name}', args={json.dumps(tool_args, sort_keys=True)}"
                logger.warning(f"[TOOL RETRY LOOP: HARD STOP] Intercepted repeated tool call loop: {tool_name} for session {sim_session}")
                await _record_and_broadcast_request(
                    db_path=current_settings.db_path,
                    model=model,
                    prompt_tokens=est_tokens,
                    completion_tokens=0,
                    cost_usd=0.0,
                    latency_ms=0,
                    prompt_hash=prompt_hash,
                    status="blocked_tool_loop",
                    blocked_reason=trip_reason,
                )
                results.append({"attempt": i, "status": "blocked_tool_loop", "message": trip_reason, "tool": tool_name})

        return {
            "simulation": "tool_loop",
            "tool_name": tool_name,
            "arguments": tool_args,
            "results": results,
            "threshold": circuit_breaker.tool_loop_threshold,
            "intercepted": True,
        }


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

        session_id = (
            request.headers.get("x-session-id")
            or request.headers.get("session-id")
            or (body.get("metadata", {}).get("user_id") if isinstance(body.get("metadata"), dict) else None)
            or "default"
        )
        has_task_update = _detect_task_update(request.headers, body, check_payload)
        cache_hit = _detect_cache_hit(request.headers, body)

        # 2. Pre-flight Circuit Breaker Guards
        try:
            circuit_breaker.check_request(
                check_payload,
                model,
                session_id=session_id,
                has_task_update=has_task_update,
                cache_hit=cache_hit,
                tool_calls=body.get("tools"),
            )
        except CircuitBreakerError as cb_err:
            status_tag = f"blocked_{cb_err.error_type}" if not cb_err.error_type.startswith("blocked_") else cb_err.error_type
            if getattr(cb_err, "code", None) == "TOOL_PING_PONG_LOOP":
                status_tag = "blocked_tool_loop"
            elif getattr(cb_err, "code", None) == "burn_rate_exceeded" or cb_err.error_type == "tokenguard_budget_breach":
                status_tag = "blocked_burn_rate"
            elif getattr(cb_err, "code", None) == "context_ballooning_detected":
                status_tag = "blocked_context_ballooning"
            elif cb_err.error_type == "budget_exceeded":
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

            if getattr(cb_err, "code", None) == "TOOL_PING_PONG_LOOP":
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content=cb_err.to_dict(),
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
                session_id=session_id,
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
            session_id=session_id,
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

        session_id = (
            request.headers.get("x-session-id")
            or request.headers.get("session-id")
            or body.get("user")
            or "default"
        )
        has_task_update = _detect_task_update(request.headers, body, messages)
        cache_hit = _detect_cache_hit(request.headers, body)

        # 2. Pre-flight Circuit Breaker Guards
        try:
            circuit_breaker.check_request(
                messages,
                model,
                session_id=session_id,
                has_task_update=has_task_update,
                cache_hit=cache_hit,
                tool_calls=body.get("tool_calls"),
            )
        except CircuitBreakerError as cb_err:
            status_tag = f"blocked_{cb_err.error_type}" if not cb_err.error_type.startswith("blocked_") else cb_err.error_type
            if getattr(cb_err, "code", None) == "TOOL_PING_PONG_LOOP":
                status_tag = "blocked_tool_loop"
            elif getattr(cb_err, "code", None) == "burn_rate_exceeded" or cb_err.error_type == "tokenguard_budget_breach":
                status_tag = "blocked_burn_rate"
            elif getattr(cb_err, "code", None) == "context_ballooning_detected":
                status_tag = "blocked_context_ballooning"
            elif cb_err.error_type == "budget_exceeded":
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
                    session_id=session_id,
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
                session_id=session_id,
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
            session_id=session_id,
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
    session_id: Optional[str] = "default",
) -> Response:
    """Execute non-streaming request against Anthropic Messages API."""
    start_time = time.perf_counter()
    try:
        upstream_resp = await http_client.post(target_url, json=body, headers=headers)
    except Exception as exc:
        logger.error(f"Anthropic upstream request failed: {exc}")
        circuit_breaker.remove_last_hash(prompt_hash)
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": f"TokenGuard upstream error: {str(exc)}"}},
        )

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    if upstream_resp.status_code != 200:
        circuit_breaker.remove_last_hash(prompt_hash)
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers={"Content-Type": "application/json"},
        )

    try:
        resp_data = upstream_resp.json()
    except Exception:
        circuit_breaker.remove_last_hash(prompt_hash)
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
        return Response(content=upstream_resp.content, status_code=upstream_resp.status_code, headers={"Content-Type": "application/json"})

    usage = resp_data.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or circuit_breaker.estimate_prompt_tokens(messages))
    completion_tokens = int(usage.get("output_tokens") or 0)

    cost_usd = circuit_breaker.compute_cost(model, prompt_tokens, completion_tokens)
    circuit_breaker.record_success(model, prompt_tokens, completion_tokens, cost_usd, prompt_hash, session_id=session_id)

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
    session_id: Optional[str] = "default",
) -> StreamingResponse:
    """Handle streaming SSE completions from Anthropic Messages API with pass-through zero-latency chunk forwarding."""
    start_time = time.perf_counter()
    try:
        upstream_req = http_client.build_request("POST", target_url, json=body, headers=headers)
        upstream_resp = await http_client.send(upstream_req, stream=True)
    except Exception as exc:
        circuit_breaker.remove_last_hash(prompt_hash)
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": f"TokenGuard upstream error: {str(exc)}"}},
        )

    if upstream_resp.status_code != 200:
        circuit_breaker.remove_last_hash(prompt_hash)
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
        error_content = await upstream_resp.aread()
        await upstream_resp.aclose()
        return Response(content=error_content, status_code=upstream_resp.status_code, headers={"Content-Type": "application/json"})

    async def anthropic_sse_generator() -> AsyncGenerator[bytes, None]:
        prompt_est = circuit_breaker.estimate_prompt_tokens(messages)
        accumulator = StreamAccumulator(is_anthropic=True, prompt_tokens_estimate=prompt_est)
        accumulator.start()
        is_cancelled = False

        try:
            async for chunk in upstream_resp.aiter_bytes():
                # Immediately yield chunk to downstream client (< 5ms latency overhead)
                yield chunk
                # Non-blocking feed to background token accumulator
                accumulator.feed(chunk)
        except (asyncio.CancelledError, GeneratorExit):
            is_cancelled = True
            logger.info("Client disconnected / cancelled during Anthropic SSE streaming")
            raise
        except Exception as exc:
            logger.error(f"Error during Anthropic stream: {exc}")
            raise
        finally:
            # Guarantee upstream socket is closed immediately
            try:
                await upstream_resp.aclose()
            except Exception as e:
                logger.debug(f"Error closing upstream response: {e}")

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            prompt_tokens, completion_tokens = await accumulator.finish()

            cost_usd = circuit_breaker.compute_cost(model, prompt_tokens, completion_tokens)
            circuit_breaker.record_success(model, prompt_tokens, completion_tokens, cost_usd, prompt_hash, session_id=session_id)

            req_status = "cancelled" if is_cancelled else "success"
            blocked_reason = "Client disconnected / stream cancelled" if is_cancelled else None

            async def _finalize_record():
                try:
                    await _record_and_broadcast_request(
                        db_path=db_path,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=cost_usd,
                        latency_ms=latency_ms,
                        prompt_hash=prompt_hash,
                        status=req_status,
                        blocked_reason=blocked_reason,
                    )
                except Exception as e:
                    logger.error(f"Failed to log Anthropic stream: {e}")

            try:
                await asyncio.shield(_finalize_record())
            except Exception:
                pass

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
    session_id: Optional[str] = "default",
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
        start_time = time.perf_counter()
        try:
            req = http_client.build_request("POST", target_url, json=anthropic_payload, headers=forward_headers)
            upstream_resp = await http_client.send(req, stream=True)
        except Exception as exc:
            circuit_breaker.remove_last_hash(prompt_hash)
            if session_id:
                circuit_breaker.remove_last_session_entry(session_id)
            return JSONResponse(status_code=502, content={"error": {"message": str(exc), "type": "bad_gateway"}})

        if upstream_resp.status_code != 200:
            circuit_breaker.remove_last_hash(prompt_hash)
            if session_id:
                circuit_breaker.remove_last_session_entry(session_id)
            err_bytes = await upstream_resp.aread()
            await upstream_resp.aclose()
            return Response(content=err_bytes, status_code=upstream_resp.status_code, headers={"Content-Type": "application/json"})

        async def adapted_stream_generator() -> AsyncGenerator[bytes, None]:
            buffer = ""
            prompt_tokens = 0
            completion_tokens = 0
            chunk_id = f"chatcmpl-claude-{int(time.time())}"
            is_cancelled = False

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
            except (asyncio.CancelledError, GeneratorExit):
                is_cancelled = True
                raise
            finally:
                try:
                    await upstream_resp.aclose()
                except Exception as e:
                    logger.debug(f"Error closing upstream Claude stream: {e}")

                latency_ms = int((time.perf_counter() - start_time) * 1000)
                if prompt_tokens == 0:
                    prompt_tokens = circuit_breaker.estimate_prompt_tokens(messages)
                if completion_tokens == 0:
                    completion_tokens = 50

                cost_usd = circuit_breaker.compute_cost(model, prompt_tokens, completion_tokens)
                circuit_breaker.record_success(model, prompt_tokens, completion_tokens, cost_usd, prompt_hash, session_id=session_id)

                async def _finalize():
                    try:
                        await _record_and_broadcast_request(
                            db_path=db_path,
                            model=model,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            cost_usd=cost_usd,
                            latency_ms=latency_ms,
                            prompt_hash=prompt_hash,
                            status="cancelled" if is_cancelled else "success",
                        )
                    except Exception as e:
                        logger.error(f"Failed to record adapted Claude stream: {e}")

                try:
                    await asyncio.shield(_finalize())
                except Exception:
                    pass

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
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
        return JSONResponse(status_code=502, content={"error": {"message": str(exc), "type": "bad_gateway"}})

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    if resp.status_code != 200:
        circuit_breaker.remove_last_hash(prompt_hash)
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
        return Response(content=resp.content, status_code=resp.status_code, headers={"Content-Type": "application/json"})

    try:
        data = resp.json()
    except Exception:
        circuit_breaker.remove_last_hash(prompt_hash)
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
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
    circuit_breaker.record_success(model, prompt_tokens, completion_tokens, cost_usd, prompt_hash, session_id=session_id)

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
    session_id: Optional[str] = "default",
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
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"TokenGuard upstream error: {str(exc)}", "type": "bad_gateway"}},
        )

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    if upstream_resp.status_code != 200:
        circuit_breaker.remove_last_hash(prompt_hash)
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers={"Content-Type": upstream_resp.headers.get("Content-Type", "application/json")},
        )

    try:
        resp_data = upstream_resp.json()
    except Exception:
        circuit_breaker.remove_last_hash(prompt_hash)
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
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
    circuit_breaker.record_success(model, prompt_tokens, completion_tokens, cost_usd, prompt_hash, session_id=session_id)

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
    session_id: Optional[str] = "default",
) -> StreamingResponse:
    """Handle streaming SSE completions with pass-through zero-latency chunk forwarding."""
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
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"TokenGuard upstream error: {str(exc)}", "type": "bad_gateway"}},
        )

    if upstream_resp.status_code != 200:
        circuit_breaker.remove_last_hash(prompt_hash)
        if session_id:
            circuit_breaker.remove_last_session_entry(session_id)
        error_content = await upstream_resp.aread()
        await upstream_resp.aclose()
        return Response(
            content=error_content,
            status_code=upstream_resp.status_code,
            headers={"Content-Type": upstream_resp.headers.get("Content-Type", "application/json")},
        )

    async def sse_generator() -> AsyncGenerator[bytes, None]:
        prompt_est = circuit_breaker.estimate_prompt_tokens(messages)
        accumulator = StreamAccumulator(is_anthropic=False, prompt_tokens_estimate=prompt_est)
        accumulator.start()
        is_cancelled = False

        try:
            async for chunk in upstream_resp.aiter_bytes():
                # Immediately yield incoming chunk to downstream client (< 5ms latency overhead)
                yield chunk
                # Non-blocking feed to background token accumulator without blocking generator loop
                accumulator.feed(chunk)
        except (asyncio.CancelledError, GeneratorExit):
            is_cancelled = True
            logger.info("Client disconnected / cancelled during SSE streaming")
            raise
        except Exception as exc:
            logger.error(f"Error during SSE chunk streaming: {exc}")
            raise
        finally:
            # 1. Guarantee immediate socket close to prevent leaking upstream connections
            try:
                await upstream_resp.aclose()
            except Exception as e:
                logger.debug(f"Error closing upstream response: {e}")

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            prompt_tokens, completion_tokens = await accumulator.finish()

            cost_usd = circuit_breaker.compute_cost(model, prompt_tokens, completion_tokens)
            circuit_breaker.record_success(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                prompt_hash=prompt_hash,
                session_id=session_id,
            )

            req_status = "cancelled" if is_cancelled else "success"
            blocked_reason = "Client disconnected / stream cancelled" if is_cancelled else None

            async def _finalize_record():
                try:
                    await _record_and_broadcast_request(
                        db_path=db_path,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=cost_usd,
                        latency_ms=latency_ms,
                        prompt_hash=prompt_hash,
                        status=req_status,
                        blocked_reason=blocked_reason,
                    )
                except Exception as e:
                    logger.error(f"Failed to log streaming request to DB: {e}")

            try:
                await asyncio.shield(_finalize_record())
            except Exception:
                pass

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

