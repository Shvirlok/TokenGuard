"""Unit tests for TokenGuard Python SDK (@protect decorator and with guard() context manager)."""

import os
import pytest

import tokenguard
from tokenguard.sdk import guard, protect


def test_guard_context_manager_sync():
    """Verify with guard(): sets proxy environment variables and restores them cleanly."""
    # Ensure starting from known state
    os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
    os.environ.pop("ANTHROPIC_BASE_URL", None)

    with guard(host="127.0.0.1", port=8080, auto_start=False):
        assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1"
        assert os.environ["OPENAI_API_BASE"] == "http://127.0.0.1:8080/v1"
        assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
        assert os.environ["GEMINI_API_BASE"] == "http://127.0.0.1:8080/v1"
        assert os.environ["DEEPSEEK_BASE_URL"] == "http://127.0.0.1:8080/v1"
        assert os.environ["MISTRAL_API_BASE"] == "http://127.0.0.1:8080/v1"
        assert os.environ["OPENROUTER_BASE_URL"] == "http://127.0.0.1:8080/v1"

    # Verify restoration
    assert os.environ["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert "ANTHROPIC_BASE_URL" not in os.environ


def test_guard_context_manager_exception_restoration():
    """Verify environment variables are restored even if an exception occurs inside with guard()."""
    os.environ.pop("OPENAI_BASE_URL", None)

    with pytest.raises(ValueError, match="Test error"):
        with guard(auto_start=False):
            assert "OPENAI_BASE_URL" in os.environ
            raise ValueError("Test error")

    assert "OPENAI_BASE_URL" not in os.environ


@pytest.mark.asyncio
async def test_guard_context_manager_async():
    """Verify async with guard(): works in coroutines."""
    os.environ.pop("OPENAI_BASE_URL", None)

    async with guard(host="127.0.0.1", port=9090, auto_start=False):
        assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:9090/v1"

    assert "OPENAI_BASE_URL" not in os.environ


def test_protect_decorator_sync():
    """Verify @protect on sync functions with and without arguments."""
    os.environ.pop("OPENAI_BASE_URL", None)

    # 1. Bare decorator @protect
    @protect(auto_start=False)
    def my_sync_task(a: int, b: int) -> int:
        assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1"
        return a + b

    result = my_sync_task(10, 20)
    assert result == 30
    assert "OPENAI_BASE_URL" not in os.environ

    # 2. Parameterized decorator @protect(...)
    @protect(max_repeats=2, budget_limit=15.0, port=8888, auto_start=False)
    def my_custom_task(msg: str) -> str:
        assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:8888/v1"
        return f"Echo: {msg}"

    res = my_custom_task("hello")
    assert res == "Echo: hello"
    assert "OPENAI_BASE_URL" not in os.environ


@pytest.mark.asyncio
async def test_protect_decorator_async():
    """Verify @protect on async coroutines."""
    os.environ.pop("OPENAI_BASE_URL", None)

    @protect(auto_start=False)
    async def my_async_task(val: str) -> str:
        assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1"
        return f"async-{val}"

    res = await my_async_task("ok")
    assert res == "async-ok"
    assert "OPENAI_BASE_URL" not in os.environ


def test_top_level_package_exports():
    """Verify tokenguard package exposes protect and guard at top level."""
    assert hasattr(tokenguard, "protect")
    assert hasattr(tokenguard, "guard")
    assert callable(tokenguard.protect)
    assert isinstance(tokenguard.guard(), tokenguard.sdk.guard)


def test_guard_auto_start_invokes_daemon(monkeypatch):
    """Verify that guard auto_start triggers daemon when server is stopped."""
    spawned = []

    def mock_is_running(host, port):
        return False

    def mock_start_daemon(host, port, limit, loop_threshold, passive=False):
        spawned.append({"host": host, "port": port, "limit": limit, "loop_threshold": loop_threshold})
        return 12345

    monkeypatch.setattr("tokenguard.cli.is_server_running", mock_is_running)
    monkeypatch.setattr("tokenguard.cli.start_daemon", mock_start_daemon)

    with guard(max_repeats=4, budget_limit=25.0, port=8099, auto_start=True):
        pass

    assert len(spawned) == 1
    assert spawned[0]["port"] == 8099
    assert spawned[0]["limit"] == 25.0
    assert spawned[0]["loop_threshold"] == 4
