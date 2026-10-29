"""Unit and proxy integration tests for Per-Tool Retry Ceiling and Read-Only Stagnation Guard."""

from __future__ import annotations

import concurrent.futures
import json
import logging
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import tokenguard.proxy as proxy_module
from tokenguard.cli import create_parser
from tokenguard.config import Settings, TokenGuardConfig, set_settings
from tokenguard.guard import (
    CircuitBreaker,
    SessionStagnationError,
    ToolRetryCeilingError,
)
from tokenguard.proxy import create_app


def _mock_openai_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_resp = {
        "id": "chatcmpl-cb-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
    }

    async def mock_post(url, json=None, headers=None):
        return httpx.Response(
            status_code=200,
            json=mock_resp,
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr(proxy_module.http_client, "post", mock_post)


def _mock_anthropic_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_resp = {
        "id": "msg_cb_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Anthropic OK"}],
        "model": "claude-3-5-sonnet-20241022",
        "usage": {"input_tokens": 40, "output_tokens": 10},
    }

    async def mock_post(url, json=None, headers=None):
        return httpx.Response(
            status_code=200,
            json=mock_resp,
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr(proxy_module.http_client, "post", mock_post)


def test_tool_argument_drift_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Calling read_file 3 times with drifting arguments trips ToolRetryCeilingTripped on the 3rd attempt."""
    settings = TokenGuardConfig(
        db_path=tmp_path / "drift.db",
        max_tool_repeats=3,
        max_stagnant_steps=6,
    )
    set_settings(settings)
    app = create_app(settings)

    with TestClient(app) as client:
        _mock_openai_upstream(monkeypatch)
        headers = {"x-session-id": "sess-drift-openai"}
        drifted_paths = ["config.json", "./config.json", "/app/config.json"]

        history = [{"role": "user", "content": "Locate and read config.json"}]
        responses = []
        for idx, p in enumerate(drifted_paths, start=1):
            history = list(history) + [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": p}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_{idx}",
                    "content": f"FileNotFoundError: {p}",
                },
            ]
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": history},
                headers=headers,
            )
            responses.append(resp)

        assert responses[0].status_code == 200
        assert responses[1].status_code == 200
        assert responses[2].status_code == 429

        body = responses[2].json()
        assert body["reason"] == "ToolRetryCeilingTripped"
        assert body["error"]["code"] == "ToolRetryCeilingTripped"
        assert responses[2].headers.get("X-TokenGuard-Status") == "tripped"
        assert (
            responses[2].headers.get("X-TokenGuard-Reason")
            == "ToolRetryCeilingTripped: read_file called 3 times consecutively"
        )

    settings_anth = TokenGuardConfig(
        db_path=tmp_path / "drift_anthropic.db",
        max_tool_repeats=3,
    )
    set_settings(settings_anth)
    app_anth = create_app(settings_anth)

    with TestClient(app_anth) as client:
        _mock_anthropic_upstream(monkeypatch)
        headers = {"x-session-id": "sess-drift-anthropic", "x-api-key": "sk-ant-test"}

        for idx, p in enumerate(["config.json", "./config.json", "/app/config.json"], start=1):
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": f"Attempt {idx}"}],
                    "tools": [{"name": "read_file", "input": {"path": p}}],
                },
                headers=headers,
            )
            if idx < 3:
                assert resp.status_code == 200
            else:
                assert resp.status_code == 429
                assert (
                    resp.headers.get("X-TokenGuard-Reason")
                    == "ToolRetryCeilingTripped: read_file called 3 times consecutively"
                )

        recovery_resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "Switch strategy"}],
                "tools": [{"name": "bash", "input": {"command": "find / -name config.json"}}],
            },
            headers=headers,
        )
        assert recovery_resp.status_code == 200


def test_tool_switch_resets_ceiling():
    """Calling read_file -> read_file -> bash resets the consecutive counter for read_file."""
    guard = CircuitBreaker(max_tool_repeats=3, max_stagnant_steps=6, enable_notifications=False)
    session_id = "sess-switch-reset"

    def _msg(step: int, tool_name: str, arg_val: str):
        return [
            {"role": "user", "content": f"Step {step}"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"tc_{step}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps({"target": arg_val}),
                        },
                    }
                ],
            },
        ]

    guard.check_request(_msg(1, "read_file", "config.json"), "gpt-4o", session_id=session_id)
    guard.check_request(_msg(2, "read_file", "./config.json"), "gpt-4o", session_id=session_id)
    guard.check_request(_msg(3, "bash", "ls -la"), "gpt-4o", session_id=session_id)

    guard.check_request(_msg(4, "read_file", "/etc/config.json"), "gpt-4o", session_id=session_id)
    guard.check_request(_msg(5, "read_file", "/var/config.json"), "gpt-4o", session_id=session_id)

    with pytest.raises(ToolRetryCeilingError) as exc_info:
        guard.check_request(_msg(6, "read_file", "/opt/config.json"), "gpt-4o", session_id=session_id)

    assert exc_info.value.code == "ToolRetryCeilingTripped"
    assert exc_info.value.message == "ToolRetryCeilingTripped: read_file called 3 times consecutively"

    guard.check_request(_msg(7, "bash", "pwd"), "gpt-4o", session_id=session_id)
    guard.check_request(_msg(8, "read_file", "final.json"), "gpt-4o", session_id=session_id)


def test_session_stagnation_guard_trips_and_resets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """6 consecutive non-mutating steps trip SessionStagnationTripped, while mutating actions reset it."""
    settings = TokenGuardConfig(
        db_path=tmp_path / "stagnation.db",
        max_tool_repeats=3,
        max_stagnant_steps=6,
    )
    set_settings(settings)
    app = create_app(settings)

    with TestClient(app) as client:
        _mock_openai_upstream(monkeypatch)
        headers = {"x-session-id": "sess-stagnant"}

        readonly_sequence = [
            ("read_file", {"path": "a.py"}),
            ("grep", {"pattern": "TODO"}),
            ("list_dir", {"path": "src"}),
            ("search", {"query": "handler"}),
            ("git_status", {}),
            ("view_file", {"path": "b.py"}),
        ]

        for idx, (tool_name, args) in enumerate(readonly_sequence, start=1):
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": f"Inspect codebase step {idx}"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"ro_{idx}",
                                "type": "function",
                                "function": {"name": tool_name, "arguments": json.dumps(args)},
                            }
                        ],
                    },
                ],
            }
            resp = client.post("/v1/chat/completions", json=payload, headers=headers)
            if idx < 6:
                assert resp.status_code == 200
            else:
                assert resp.status_code == 429
                data = resp.json()
                assert data["reason"] == "SessionStagnationTripped"
                assert data["error"]["code"] == "SessionStagnationTripped"
                assert resp.headers.get("X-TokenGuard-Status") == "tripped"
                assert (
                    resp.headers.get("X-TokenGuard-Reason")
                    == "SessionStagnationTripped: 6 steps without state change"
                )

        mutate_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Apply fix"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "mut_1",
                                "type": "function",
                                "function": {
                                    "name": "edit_file",
                                    "arguments": json.dumps({"path": "a.py", "diff": "+ fixed"}),
                                },
                            }
                        ],
                    },
                ],
            },
            headers=headers,
        )
        assert mutate_resp.status_code == 200

        post_reset_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Verify fix"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "ro_after",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "a.py"}),
                                },
                            }
                        ],
                    },
                ],
            },
            headers=headers,
        )
        assert post_reset_resp.status_code == 200


def test_dry_run_observation_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Both ToolRetryCeilingTripped and SessionStagnationTripped log shadow alerts and return 200 in --dry-run."""
    dry_settings = TokenGuardConfig(
        db_path=tmp_path / "dry_run_breakers.db",
        max_tool_repeats=3,
        max_stagnant_steps=6,
        dry_run=True,
    )
    set_settings(dry_settings)
    app = create_app(dry_settings)

    caplog.set_level(logging.WARNING, logger="tokenguard.proxy")

    with TestClient(app) as client:
        _mock_openai_upstream(monkeypatch)

        drift_headers = {"x-session-id": "dryrun-ceiling-sess"}
        for idx, path_arg in enumerate(["config.json", "./config.json", "/app/config.json"], start=1):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": f"Read attempt {idx}"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": f"dr_c_{idx}",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": path_arg}),
                                    },
                                }
                            ],
                        },
                    ],
                },
                headers=drift_headers,
            )
            assert resp.status_code == 200
            assert resp.headers.get("X-TokenGuard-Dry-Run") == "true"
            if idx < 3:
                assert resp.headers.get("X-TokenGuard-Status") is None
            else:
                assert resp.headers.get("X-TokenGuard-Status") == "tripped"
                assert (
                    resp.headers.get("X-TokenGuard-Reason")
                    == "ToolRetryCeilingTripped: read_file called 3 times consecutively"
                )

        stag_headers = {"x-session-id": "dryrun-stagnation-sess"}
        ro_tools = ["read_file", "grep", "list_dir", "search", "git_status", "view_file"]
        for idx, t_name in enumerate(ro_tools, start=1):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": f"Readonly step {idx}"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": f"dr_s_{idx}",
                                    "type": "function",
                                    "function": {
                                        "name": t_name,
                                        "arguments": json.dumps({"step": idx}),
                                    },
                                }
                            ],
                        },
                    ],
                },
                headers=stag_headers,
            )
            assert resp.status_code == 200
            assert resp.headers.get("X-TokenGuard-Dry-Run") == "true"
            if idx < 6:
                assert resp.headers.get("X-TokenGuard-Status") is None
            else:
                assert resp.headers.get("X-TokenGuard-Status") == "tripped"
                assert (
                    resp.headers.get("X-TokenGuard-Reason")
                    == "SessionStagnationTripped: 6 steps without state change"
                )

        log_messages = "\n".join(r.message for r in caplog.records)
        assert "[DRY-RUN SHADOW MODE]" in log_messages
        assert "ToolRetryCeilingTripped: read_file called 3 times consecutively" in log_messages
        assert "SessionStagnationTripped: 6 steps without state change" in log_messages

        logged_reqs = client.get("/api/requests?status=blocked").json()["requests"]
        statuses = {r["status"] for r in logged_reqs}
        assert "blocked_tool_retry_ceiling" in statuses
        assert "blocked_stagnation" in statuses


def test_config_and_cli_options(monkeypatch: pytest.MonkeyPatch):
    """Verify CLI flags and TOKENGUARD_MAX_TOOL_REPEATS / TOKENGUARD_MAX_STAGNANT_STEPS env vars."""
    assert TokenGuardConfig is Settings

    default_cfg = TokenGuardConfig()
    assert default_cfg.max_tool_repeats == 3
    assert default_cfg.max_stagnant_steps == 6

    monkeypatch.setenv("TOKENGUARD_MAX_TOOL_REPEATS", "5")
    monkeypatch.setenv("TOKENGUARD_MAX_STAGNANT_STEPS", "9")

    env_cfg = TokenGuardConfig()
    assert env_cfg.max_tool_repeats == 5
    assert env_cfg.max_stagnant_steps == 9

    parser = create_parser()
    env_args = parser.parse_args(["start"])
    assert env_args.max_tool_repeats == 5
    assert env_args.max_stagnant_steps == 9

    cli_args = parser.parse_args(["start", "--max-tool-repeats", "4", "--max-stagnant-steps", "8"])
    assert cli_args.max_tool_repeats == 4
    assert cli_args.max_stagnant_steps == 8


def test_thread_safety_concurrent_sessions():
    """Verify thread-safe non-interfering session state under concurrent threads."""
    guard = CircuitBreaker(max_tool_repeats=3, max_stagnant_steps=6, enable_notifications=False)

    def _worker(worker_id: int) -> str:
        sess = f"concurrent-sess-{worker_id}"
        for step, p in enumerate(["config.json", "./config.json", "/app/config.json"], start=1):
            msgs = [
                {"role": "user", "content": f"Worker {worker_id} step {step}"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"w_{worker_id}_{step}",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": json.dumps({"path": p})},
                        }
                    ],
                },
            ]
            if step < 3:
                guard.check_request(msgs, "gpt-4o", session_id=sess)
            else:
                with pytest.raises(ToolRetryCeilingError) as exc:
                    guard.check_request(msgs, "gpt-4o", session_id=sess)
                return exc.value.code
        return "unexpected"

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(16)))

    assert results == ["ToolRetryCeilingTripped"] * 16

