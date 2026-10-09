"""Zero-friction Python SDK for TokenGuard LLM call protection.

Allows developers to protect LLM calls directly in Python via decorators and context managers.

Usage examples:
    # 1. Function Decorator (Sync or Async)
    @tokenguard.protect(max_repeats=3, budget_limit=10.0)
    def ask_agent(prompt: str):
        client = openai.OpenAI()
        return client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])

    # 2. Context Manager
    with tokenguard.guard(max_repeats=2):
        response = client.chat.completions.create(...)
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload

logger = logging.getLogger("tokenguard.sdk")

F = TypeVar("F", bound=Callable[..., Any])

INJECTED_ENV_VARS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "ANTHROPIC_BASE_URL",
    "GEMINI_API_BASE",
    "DASHSCOPE_BASE_URL",
    "QWEN_BASE_URL",
    "DEEPSEEK_BASE_URL",
    "GROQ_BASE_URL",
    "MISTRAL_API_BASE",
    "OPENROUTER_BASE_URL",
)


class guard:
    """Context manager that temporarily routes LLM client calls through TokenGuard proxy.
    
    Supports both synchronous `with guard():` and asynchronous `async with guard():` contexts.
    Restores original environment variables upon exiting the block, even on exceptions.
    """

    def __init__(
        self,
        max_repeats: Optional[int] = 3,
        budget_limit: Optional[float] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        auto_start: bool = True,
        passive: bool = False,
    ):
        self.max_repeats = max_repeats
        self.budget_limit = budget_limit
        self.host = host
        self.port = port
        self.auto_start = auto_start
        self.passive = passive
        self._saved_env: Dict[str, Optional[str]] = {}

    def _ensure_daemon_running(self) -> None:
        """Check if TokenGuard proxy is running; if not and auto_start is True, spawn it in background."""
        try:
            from tokenguard.cli import is_server_running, start_daemon
            if not is_server_running(self.host, self.port):
                if self.auto_start:
                    start_daemon(
                        host=self.host,
                        port=self.port,
                        limit=self.budget_limit or 5.0,
                        loop_threshold=self.max_repeats or 3,
                        passive=self.passive,
                    )
        except Exception as e:
            logger.debug(f"TokenGuard daemon auto-start check skipped: {e}")

    def _enter(self) -> guard:
        """Inject proxy environment variables and backup originals."""
        self._ensure_daemon_running()

        proxy_url = f"http://{self.host}:{self.port}/v1"
        anthropic_url = f"http://{self.host}:{self.port}"

        target_mapping = {
            "OPENAI_BASE_URL": proxy_url,
            "OPENAI_API_BASE": proxy_url,
            "ANTHROPIC_BASE_URL": anthropic_url,
            "GEMINI_API_BASE": proxy_url,
            "DASHSCOPE_BASE_URL": proxy_url,
            "QWEN_BASE_URL": proxy_url,
            "DEEPSEEK_BASE_URL": proxy_url,
            "GROQ_BASE_URL": proxy_url,
            "MISTRAL_API_BASE": proxy_url,
            "OPENROUTER_BASE_URL": proxy_url,
        }

        # Backup existing values
        for key in target_mapping:
            self._saved_env[key] = os.environ.get(key)
            os.environ[key] = target_mapping[key]

        return self

    def _exit(self) -> None:
        """Restore previous environment variables."""
        for key, original_val in self._saved_env.items():
            if original_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_val
        self._saved_env.clear()

    def __enter__(self) -> guard:
        return self._enter()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._exit()

    async def __aenter__(self) -> guard:
        return self._enter()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._exit()


@overload
def protect(
    func: F,
    *,
    max_repeats: Optional[int] = 3,
    budget_limit: Optional[float] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    auto_start: bool = True,
    passive: bool = False,
) -> F:
    ...


@overload
def protect(
    func: None = None,
    *,
    max_repeats: Optional[int] = 3,
    budget_limit: Optional[float] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    auto_start: bool = True,
    passive: bool = False,
) -> Callable[[F], F]:
    ...


def protect(
    func: Optional[F] = None,
    *,
    max_repeats: Optional[int] = 3,
    budget_limit: Optional[float] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    auto_start: bool = True,
    passive: bool = False,
) -> Union[F, Callable[[F], F]]:
    """Decorator to protect any sync or async function with TokenGuard circuit breaker proxy.
    
    Can be used with or without arguments:
        @protect
        def my_function(): ...

        @protect(max_repeats=2, budget_limit=5.0)
        async def my_async_function(): ...
    """
    def decorator(target_fn: F) -> F:
        if inspect.iscoroutinefunction(target_fn):
            @functools.wraps(target_fn)
            async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                with guard(
                    max_repeats=max_repeats,
                    budget_limit=budget_limit,
                    host=host,
                    port=port,
                    auto_start=auto_start,
                    passive=passive,
                ):
                    return await target_fn(*args, **kwargs)

            return async_wrapped  # type: ignore[return-value]
        else:
            @functools.wraps(target_fn)
            def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
                with guard(
                    max_repeats=max_repeats,
                    budget_limit=budget_limit,
                    host=host,
                    port=port,
                    auto_start=auto_start,
                    passive=passive,
                ):
                    return target_fn(*args, **kwargs)

            return sync_wrapped  # type: ignore[return-value]

    if func is not None and callable(func):
        return decorator(func)
    return decorator
