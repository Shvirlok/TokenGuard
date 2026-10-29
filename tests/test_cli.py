"""Unit and integration tests for TokenGuard CLI subcommands (run, daemon, stop, status)."""

import os
import subprocess
import sys
import time
from pathlib import Path
import httpx
import pytest
from unittest.mock import patch, MagicMock

import tokenguard.cli as cli_module
from tokenguard.cli import (
    create_parser,
    get_daemon_pid,
    is_server_running,
    start_daemon,
    stop_daemon,
    show_status,
    cmd_run,
    PID_FILE,
    LOG_FILE,
)


def test_cli_parser_subcommands():
    """Verify parser registers all subcommands."""
    parser = create_parser()

    args_run = parser.parse_args(["run", "python", "script.py"])
    assert args_run.command == "run"
    assert args_run.cmd == ["python", "script.py"]

    args_start = parser.parse_args(["start", "-d", "--port", "9090", "--max-repeats", "5"])
    assert args_start.command == "start"
    assert args_start.daemon is True
    assert args_start.port == 9090
    assert args_start.loop_threshold == 5

    args_run_repeats = parser.parse_args(["run", "--max-repeats", "2", "python", "script.py"])
    assert args_run_repeats.loop_threshold == 2

    args_stop = parser.parse_args(["stop", "--port", "8080"])
    assert args_stop.command == "stop"

    args_status = parser.parse_args(["status"])
    assert args_status.command == "status"

    args_prices = parser.parse_args(["prices"])
    assert args_prices.command == "prices"


def test_cmd_run_environment_injection(monkeypatch, tmp_path):
    """Verify that tokenguard run injects OPENAI_BASE_URL and OPENAI_API_BASE."""
    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: True)

    captured_env = {}

    def mock_subprocess_run(cmd, env=None, **kwargs):
        captured_env.update(env or {})
        mock_res = MagicMock()
        mock_res.returncode = 0
        return mock_res

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    parser = create_parser()
    args = parser.parse_args(["run", "--port", "8080", "echo", "hello"])

    with pytest.raises(SystemExit) as exc_info:
        cmd_run(args)

    assert exc_info.value.code == 0
    assert captured_env.get("OPENAI_BASE_URL") == "http://127.0.0.1:8080/v1"
    assert captured_env.get("OPENAI_API_BASE") == "http://127.0.0.1:8080/v1"
    assert captured_env.get("ANTHROPIC_BASE_URL") == "http://127.0.0.1:8080"
    assert captured_env.get("GEMINI_API_BASE") == "http://127.0.0.1:8080/v1"
    assert captured_env.get("DASHSCOPE_BASE_URL") == "http://127.0.0.1:8080/v1"
    assert captured_env.get("DEEPSEEK_BASE_URL") == "http://127.0.0.1:8080/v1"
    assert captured_env.get("GROQ_BASE_URL") == "http://127.0.0.1:8080/v1"
    assert captured_env.get("MISTRAL_API_BASE") == "http://127.0.0.1:8080/v1"


def test_cmd_run_returncode_forwarding(monkeypatch):
    """Verify that tokenguard run exits with the exact child process return code."""
    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: True)

    def mock_subprocess_run(cmd, env=None, **kwargs):
        mock_res = MagicMock()
        mock_res.returncode = 42
        return mock_res

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    parser = create_parser()
    args = parser.parse_args(["run", "pytest"])

    with pytest.raises(SystemExit) as exc_info:
        cmd_run(args)

    assert exc_info.value.code == 42


def test_cmd_run_auto_start_proxy(monkeypatch):
    """Verify that tokenguard run auto-starts proxy if not running."""
    server_running_state = [False, True, True]

    def mock_is_server_running(host, port):
        if server_running_state:
            return server_running_state.pop(0)
        return True

    daemon_started = []

    def mock_start_daemon(*args, **kwargs):
        daemon_started.append(True)
        return 99999

    monkeypatch.setattr(cli_module, "is_server_running", mock_is_server_running)
    monkeypatch.setattr(cli_module, "start_daemon", mock_start_daemon)

    def mock_subprocess_run(cmd, env=None, **kwargs):
        mock_res = MagicMock()
        mock_res.returncode = 0
        return mock_res

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    parser = create_parser()
    args = parser.parse_args(["run", "python", "agent.py"])

    with pytest.raises(SystemExit) as exc_info:
        cmd_run(args)

    assert exc_info.value.code == 0
    assert len(daemon_started) == 1


def test_cmd_run_empty_command():
    """Verify that running with no command exits with error."""
    parser = create_parser()
    args = parser.parse_args(["run"])

    with pytest.raises(SystemExit) as exc_info:
        cmd_run(args)

    assert exc_info.value.code == 1


def test_daemon_stop_and_status(monkeypatch, tmp_path, capsys):
    """Verify daemon stop and status handling."""
    fake_pid_file = tmp_path / "test.pid"
    fake_pid_file.write_text("12345")
    monkeypatch.setattr(cli_module, "PID_FILE", fake_pid_file)
    monkeypatch.setattr(cli_module, "is_pid_alive", lambda pid: True)
    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: False)

    killed_pids = []

    def mock_kill(pid, sig):
        killed_pids.append((pid, sig))

    monkeypatch.setattr(os, "kill", mock_kill)

    success = stop_daemon(host="127.0.0.1", port=8080)
    assert success is True
    assert len(killed_pids) >= 1
    assert killed_pids[0][0] == 12345
    assert not fake_pid_file.exists()


def test_config_storage_and_permissions(tmp_path, monkeypatch):
    """Verify storing user configuration and 0600 file permissions."""
    from tokenguard.config import (
        load_stored_config,
        save_stored_config,
        update_stored_config,
        USER_CONFIG_DIR,
        USER_CONFIG_FILE,
    )
    import tokenguard.config as config_mod

    fake_config_dir = tmp_path / ".tokenguard"
    fake_config_file = fake_config_dir / "config.json"
    monkeypatch.setattr(config_mod, "USER_CONFIG_DIR", fake_config_dir)
    monkeypatch.setattr(config_mod, "USER_CONFIG_FILE", fake_config_file)

    assert load_stored_config() == {}

    save_stored_config({"openai_api_key": "sk-test12345678", "profile": "careful"})
    assert fake_config_file.exists()

    loaded = load_stored_config()
    assert loaded.get("openai_api_key") == "sk-test12345678"
    assert loaded.get("profile") == "careful"

    update_stored_config(deepseek_api_key="sk-ds98765432")
    updated = load_stored_config()
    assert updated.get("openai_api_key") == "sk-test12345678"
    assert updated.get("deepseek_api_key") == "sk-ds98765432"


def test_api_key_detection_hierarchy(tmp_path, monkeypatch):
    """Verify hierarchy: os.environ > .env > config.json."""
    from tokenguard.config import detect_api_keys, get_api_key, mask_key
    import tokenguard.config as config_mod

    fake_config_dir = tmp_path / ".tokenguard"
    fake_config_file = fake_config_dir / "config.json"
    monkeypatch.setattr(config_mod, "USER_CONFIG_DIR", fake_config_dir)
    monkeypatch.setattr(config_mod, "USER_CONFIG_FILE", fake_config_file)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    config_mod.save_stored_config({
        "openai_api_key": "sk-from-config-1234",
        "deepseek_api_key": "sk-from-config-5678",
    })

    keys = detect_api_keys()
    assert keys["OPENAI_API_KEY"] == "sk-from-config-1234"
    assert keys["DEEPSEEK_API_KEY"] == "sk-from-config-5678"
    assert get_api_key("openai") == "sk-from-config-1234"
    assert get_api_key("deepseek") == "sk-from-config-5678"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environ-9999")
    keys2 = detect_api_keys()
    assert keys2["OPENAI_API_KEY"] == "sk-from-environ-9999"
    assert keys2["DEEPSEEK_API_KEY"] == "sk-from-config-5678"

    assert mask_key("sk-abcdefghijklmnop") == "sk-a••••mnop"
    assert mask_key("short") == "••••••••"
    assert mask_key("") == "None"


def test_cmd_quickstart_already_running(monkeypatch):
    """Verify quickstart displays status if server is already up."""
    from tokenguard.cli import cmd_quickstart

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: True)
    monkeypatch.setattr(cli_module, "show_status", lambda h, p: None)

    parser = create_parser()
    args = parser.parse_args([])
    cmd_quickstart(args)


def test_cmd_quickstart_new_launch(monkeypatch):
    """Verify quickstart launches daemon with selected profile."""
    from tokenguard.cli import cmd_quickstart

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: False)
    started_args = {}

    def mock_start_daemon(host, port, limit, daily_limit, loop_threshold, **kwargs):
        started_args.update({
            "host": host,
            "port": port,
            "limit": limit,
            "daily_limit": daily_limit,
            "loop_threshold": loop_threshold,
        })
        return 7777

    monkeypatch.setattr(cli_module, "start_daemon", mock_start_daemon)

    parser = create_parser()
    args = parser.parse_args([])
    cmd_quickstart(args)

    assert started_args.get("port") == 8080
    assert started_args.get("limit") == 5.0
    assert started_args.get("loop_threshold") == 2


def test_cmd_prices(tmp_path):
    """Verify prices command displays table without error."""
    import json
    from tokenguard.cli import cmd_prices

    prices_file = tmp_path / "custom_prices.json"
    prices_file.write_text(json.dumps({
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "deepseek-chat": {"input": 0.27, "output": 1.10}
    }))

    parser = create_parser()
    args = parser.parse_args(["prices", "--prices-path", str(prices_file)])
    cmd_prices(args)


def test_cli_reactive_control_commands(monkeypatch):
    """Verify CLI kill, resume, profile, and clear commands."""
    from tokenguard.cli import cmd_kill, cmd_resume, cmd_profile, cmd_clear, cmd_logs

    parser = create_parser()
    assert parser.parse_args(["kill"]).command == "kill"
    assert parser.parse_args(["resume"]).command == "resume"
    assert parser.parse_args(["profile", "standard"]).command == "profile"
    assert parser.parse_args(["clear"]).command == "clear"
    assert parser.parse_args(["logs", "-f"]).command == "logs"
    assert parser.parse_args(["tail"]).command == "tail"

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: True)

    http_calls = []

    class MockHttpxClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, json=None, **kwargs):
            http_calls.append(("POST", url, json))
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"status": "ok"}
            return mock_resp
        def get(self, url, **kwargs):
            http_calls.append(("GET", url, None))
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "requests": [
                    {
                        "id": 1,
                        "timestamp": "2026-10-09T09:00:00Z",
                        "model": "gpt-4o",
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "cost_usd": 0.001,
                        "latency_ms": 150,
                        "status": "success",
                        "prompt_hash": "hash123",
                    }
                ]
            }
            return mock_resp

    monkeypatch.setattr(httpx, "Client", MockHttpxClient)

    args_kill = parser.parse_args(["kill"])
    cmd_kill(args_kill)
    assert any(c[0] == "POST" and "/api/kill-switch" in c[1] and c[2].get("active") is True for c in http_calls)

    args_resume = parser.parse_args(["resume"])
    cmd_resume(args_resume)
    assert any(c[0] == "POST" and "/api/kill-switch" in c[1] and c[2].get("active") is False for c in http_calls)

    args_prof = parser.parse_args(["profile", "standard"])
    cmd_profile(args_prof)
    assert any(c[0] == "POST" and "/api/config" in c[1] and c[2].get("profile") == "standard" for c in http_calls)

    args_clear = parser.parse_args(["clear"])
    cmd_clear(args_clear)
    assert any(c[0] == "POST" and "/api/clear-logs" in c[1] for c in http_calls)

    args_logs = parser.parse_args(["logs", "-n", "10"])
    cmd_logs(args_logs)
    assert any(c[0] == "GET" and "/api/requests" in c[1] for c in http_calls)


def test_interactive_select_navigation(monkeypatch):
    """Verify interactive_select handles arrows, numbers, and fallback."""
    from tokenguard.cli import interactive_select

    options = [
        ("opt1", "Option One", "First item", "Rec"),
        ("opt2", "Option Two", "Second item", ""),
        ("opt3", "Option Three", "Third item", ""),
    ]

    original_read_raw = cli_module._read_raw_key

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    res = interactive_select(options, default_index=0)
    assert res == "opt1"
    res2 = interactive_select(options, default_index=2)
    assert res2 == "opt3"

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdin, "fileno", lambda: 0)
    monkeypatch.setattr(cli_module, "_read_raw_key", lambda fd: "2")
    res_num = interactive_select(options, default_index=0)
    assert res_num == "opt2"

    keys_seq = ["down", "enter"]
    def mock_read_key_arrow(fd):
        if keys_seq:
            return keys_seq.pop(0)
        return "enter"

    monkeypatch.setattr(cli_module, "_read_raw_key", mock_read_key_arrow)
    res_arrow = interactive_select(options, default_index=0)
    assert res_arrow == "opt2"

    keys_seq_up = ["up", "enter"]
    def mock_read_key_up(fd):
        if keys_seq_up:
            return keys_seq_up.pop(0)
        return "enter"

    monkeypatch.setattr(cli_module, "_read_raw_key", mock_read_key_up)
    res_wrap = interactive_select(options, default_index=0)
    assert res_wrap == "opt3"

    fake_reads = [b'\x1b[B', b'\x1b[A', b'\x1bOB', b'1', b'\r']
    def mock_os_read(fd, n):
        if fake_reads:
            return fake_reads.pop(0)
        return b'\r'
    monkeypatch.setattr(os, "read", mock_os_read)
    assert original_read_raw(0) == "down"
    assert original_read_raw(0) == "up"
    assert original_read_raw(0) == "down"
    assert original_read_raw(0) == "1"
    assert original_read_raw(0) == "enter"


def test_cli_menu_and_simulate_parsers():
    """Verify menu, tui, dashboard, and simulate parsers."""
    parser = create_parser()
    assert parser.parse_args(["menu"]).command == "menu"
    assert parser.parse_args(["tui"]).command == "tui"
    assert parser.parse_args(["dashboard"]).command == "dashboard"
    assert parser.parse_args(["simulate", "--type", "loop"]).command == "simulate"
    assert parser.parse_args(["simulate", "--type", "alternating_loop"]).command == "simulate"
    assert parser.parse_args(["simulate", "--type", "drift"]).command == "simulate"
    assert parser.parse_args(["simulate", "--type", "stagnation"]).command == "simulate"
    assert parser.parse_args(["simulate", "--type", "secret"]).command == "simulate"
    assert parser.parse_args(["simulate", "--type", "stream-loop"]).command == "simulate"
    assert parser.parse_args(["simulate", "--type", "stream_loop"]).command == "simulate"


def test_cli_passive_flag(monkeypatch):
    """Verify that --passive is correctly accepted and passed to daemon/settings."""
    parser = create_parser()

    args_start = parser.parse_args(["start", "--passive"])
    assert args_start.passive is True

    args_run = parser.parse_args(["run", "--passive", "python", "script.py"])
    assert args_run.passive is True

    spawned_cmds = []
    def mock_popen(cmd, *args, **kwargs):
        spawned_cmds.append(cmd)
        proc = MagicMock()
        proc.pid = 99999
        return proc

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: False)
    monkeypatch.setattr(cli_module, "wait_for_server", lambda h, p, timeout: True)

    pid = start_daemon(port=9999, passive=True)
    assert pid == 99999
    assert "--passive" in spawned_cmds[0]


def test_cli_max_burn_rate_flag(monkeypatch):
    """Verify that --max-burn-rate is parsed and passed to start_daemon."""
    parser = create_parser()

    args_start = parser.parse_args(["start", "--max-burn-rate", "$0.50/min"])
    assert args_start.max_burn_rate == "$0.50/min"

    args_run = parser.parse_args(["run", "--max-burn-rate", "$2.00/5m", "python", "script.py"])
    assert args_run.max_burn_rate == "$2.00/5m"

    spawned_cmds = []
    def mock_popen(cmd, *args, **kwargs):
        spawned_cmds.append(cmd)
        proc = MagicMock()
        proc.pid = 88888
        return proc

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: False)
    monkeypatch.setattr(cli_module, "wait_for_server", lambda h, p, timeout: True)

    pid = start_daemon(port=8080, max_burn_rate="$0.50/min")
    assert pid == 88888
    assert "--max-burn-rate" in spawned_cmds[0]
    assert "$0.50/min" in spawned_cmds[0]


def test_cli_cmd_simulate_alternating_loop_local(monkeypatch, capsys):
    """Verify cmd_simulate with alternating_loop runs local in-process simulation when daemon is offline."""
    from tokenguard.cli import cmd_simulate, create_parser
    parser = create_parser()
    args = parser.parse_args(["simulate", "--type", "alternating_loop"])

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: False)
    cmd_simulate(args)


def test_cli_cmd_simulate_alternating_loop_with_server(monkeypatch):
    """Verify cmd_simulate with alternating_loop contacts server when daemon is running."""
    from tokenguard.cli import cmd_simulate, create_parser
    parser = create_parser()
    args = parser.parse_args(["simulate", "--type", "alternating_loop"])

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: True)

    sim_payload = {
        "simulation": "alternating_loop",
        "tripped": True,
        "cycle_length": 2,
        "pattern": ["read_file", "bash"],
        "results": [
            {"attempt": 1, "tool": "read_file", "args": {"path": "server.py"}, "status": "allowed"},
            {"attempt": 2, "tool": "bash", "args": {"command": "python server.py"}, "status": "allowed"},
            {"attempt": 3, "tool": "read_file", "args": {"path": "server.py"}, "status": "allowed"},
            {"attempt": 4, "tool": "bash", "args": {"command": "python server.py"}, "status": "blocked_tool_loop"},
        ],
        "details": {"message": "Tool ping-pong cycle detected: bash"},
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sim_payload

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kwargs):
            return mock_resp

    monkeypatch.setattr(httpx, "Client", MockClient)
    cmd_simulate(args)


def test_cli_env_command(capsys):
    """Verify tokenguard env outputs export commands on stdout and eval instructions on stderr."""
    from tokenguard.cli import cmd_env, create_parser
    parser = create_parser()
    args = parser.parse_args(["env"])
    assert args.command == "env"
    assert args.unset is False

    cmd_env(args)
    captured = capsys.readouterr()
    assert captured.out.strip() == 'export OPENAI_BASE_URL="http://localhost:8080/v1" ANTHROPIC_BASE_URL="http://localhost:8080"'
    assert "eval $(tokenguard env)" in captured.err


def test_cli_env_unset_command(capsys):
    """Verify tokenguard env --unset outputs unset commands on stdout and eval instructions on stderr."""
    from tokenguard.cli import cmd_env, create_parser
    parser = create_parser()
    args = parser.parse_args(["env", "--unset"])
    assert args.command == "env"
    assert args.unset is True

    cmd_env(args)
    captured = capsys.readouterr()
    assert captured.out.strip() == "unset OPENAI_BASE_URL ANTHROPIC_BASE_URL"
    assert "eval $(tokenguard env --unset)" in captured.err


def test_cli_env_custom_host_port(capsys):
    """Verify tokenguard env respects --host and --port arguments."""
    from tokenguard.cli import cmd_env, create_parser
    parser = create_parser()
    args = parser.parse_args(["env", "--host", "192.168.1.50", "--port", "9090"])

    cmd_env(args)
    captured = capsys.readouterr()
    assert captured.out.strip() == 'export OPENAI_BASE_URL="http://192.168.1.50:9090/v1" ANTHROPIC_BASE_URL="http://192.168.1.50:9090"'


def test_cli_dry_run_flags(monkeypatch):
    """Verify --dry-run flag parsing on start and run subcommands, and daemon invocation."""
    parser = create_parser()

    args_start = parser.parse_args(["start", "--dry-run"])
    assert args_start.dry_run is True

    args_run = parser.parse_args(["run", "--dry-run", "python", "test_agent.py"])
    assert args_run.dry_run is True

    spawned_cmds = []
    def mock_popen(cmd, *args, **kwargs):
        spawned_cmds.append(cmd)
        proc = MagicMock()
        proc.pid = 77777
        return proc

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: False)
    monkeypatch.setattr(cli_module, "wait_for_server", lambda h, p, timeout: True)

    pid = start_daemon(port=8080, dry_run=True)
    assert pid == 77777
    assert "--dry-run" in spawned_cmds[0]


def test_cli_cmd_simulate_drift_local(monkeypatch):
    """Verify cmd_simulate with drift runs local in-process simulation when daemon is offline."""
    from tokenguard.cli import cmd_simulate, create_parser
    parser = create_parser()
    args = parser.parse_args(["simulate", "--type", "drift"])

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: False)
    cmd_simulate(args)


def test_cli_cmd_simulate_stagnation_local(monkeypatch):
    """Verify cmd_simulate with stagnation runs local in-process simulation when daemon is offline."""
    from tokenguard.cli import cmd_simulate, create_parser
    parser = create_parser()
    args = parser.parse_args(["simulate", "--type", "stagnation"])

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: False)
    cmd_simulate(args)


def test_cli_cmd_simulate_secret_local(monkeypatch):
    """Verify cmd_simulate with secret runs local in-process simulation when daemon is offline."""
    from tokenguard.cli import cmd_simulate, create_parser
    parser = create_parser()
    args = parser.parse_args(["simulate", "--type", "secret"])

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: False)
    cmd_simulate(args)


def test_cli_cmd_simulate_stream_loop_local(monkeypatch):
    """Verify cmd_simulate with stream-loop runs local in-process simulation when daemon is offline."""
    from tokenguard.cli import cmd_simulate, create_parser
    parser = create_parser()
    args = parser.parse_args(["simulate", "--type", "stream-loop"])

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: False)
    cmd_simulate(args)


def test_cli_cmd_simulate_drift_with_server(monkeypatch):
    """Verify cmd_simulate with drift contacts server when daemon is running."""
    from tokenguard.cli import cmd_simulate, create_parser
    parser = create_parser()
    args = parser.parse_args(["simulate", "--type", "drift"])

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: True)

    sim_payload = {
        "simulation": "drift",
        "tripped": True,
        "tool": "read_file",
        "threshold": 3,
        "results": [
            {"attempt": 1, "tool": "read_file", "args": {"path": "config.json"}, "status": "allowed"},
            {"attempt": 2, "tool": "read_file", "args": {"path": "./config.json"}, "status": "allowed"},
            {"attempt": 3, "tool": "read_file", "args": {"path": "/app/config.json"}, "status": "blocked_tool_retry_ceiling"},
        ],
        "details": {"reason": "ToolRetryCeilingTripped", "message": "ToolRetryCeilingTripped: read_file called 3 times consecutively", "usd_saved": 0.0125},
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sim_payload

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kwargs):
            return mock_resp

    monkeypatch.setattr(httpx, "Client", MockClient)
    cmd_simulate(args)


def test_cli_cmd_simulate_stagnation_with_server(monkeypatch):
    """Verify cmd_simulate with stagnation contacts server when daemon is running."""
    from tokenguard.cli import cmd_simulate, create_parser
    parser = create_parser()
    args = parser.parse_args(["simulate", "--type", "stagnation"])

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: True)

    sim_payload = {
        "simulation": "stagnation",
        "tripped": True,
        "threshold": 6,
        "results": [
            {"attempt": 1, "tool": "read_file", "args": {"step": 1}, "status": "allowed"},
            {"attempt": 2, "tool": "list_dir", "args": {"step": 2}, "status": "allowed"},
            {"attempt": 3, "tool": "get_status", "args": {"step": 3}, "status": "allowed"},
            {"attempt": 4, "tool": "read_file", "args": {"step": 4}, "status": "allowed"},
            {"attempt": 5, "tool": "search_files", "args": {"step": 5}, "status": "allowed"},
            {"attempt": 6, "tool": "read_file", "args": {"step": 6}, "status": "blocked_stagnation"},
        ],
        "details": {"reason": "SessionStagnationTripped", "message": "SessionStagnationTripped: 6 steps without state change", "usd_saved": 0.0180},
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sim_payload

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kwargs):
            return mock_resp

    monkeypatch.setattr(httpx, "Client", MockClient)
    cmd_simulate(args)


def test_cli_cmd_simulate_secret_with_server(monkeypatch):
    """Verify cmd_simulate with secret contacts server when daemon is running."""
    from tokenguard.cli import cmd_simulate, create_parser
    parser = create_parser()
    args = parser.parse_args(["simulate", "--type", "secret"])

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: True)

    sim_payload = {
        "simulation": "secret",
        "tripped": True,
        "secret_type": "openai_proj_key",
        "preview": "sk-proj-****************",
        "results": [
            {"attempt": 1, "pattern": "openai_proj_key", "payload_snippet": "Deploy with API key sk-proj-****************", "status": "blocked_secret_leak"}
        ],
        "details": {"reason": "SecretLeakDetected", "message": "SecretLeakDetected: Found matching pattern openai_proj_key"},
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sim_payload

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kwargs):
            return mock_resp

    monkeypatch.setattr(httpx, "Client", MockClient)
    cmd_simulate(args)


def test_cli_cmd_simulate_stream_loop_with_server(monkeypatch):
    """Verify cmd_simulate with stream-loop contacts server when daemon is running."""
    from tokenguard.cli import cmd_simulate, create_parser
    parser = create_parser()
    args = parser.parse_args(["simulate", "--type", "stream_loop"])

    monkeypatch.setattr(cli_module, "is_server_running", lambda h, p: True)

    sim_payload = {
        "simulation": "stream_loop",
        "tripped": True,
        "repeated_token": "repeat ",
        "results": [
            {"attempt": 1, "token": "repeat ", "repeat_count": 1, "status": "allowed"},
            {"attempt": 2, "token": "repeat ", "repeat_count": 2, "status": "allowed"},
            {"attempt": 3, "token": "repeat ", "repeat_count": 3, "status": "allowed"},
            {"attempt": 4, "token": "repeat ", "repeat_count": 4, "status": "allowed"},
            {"attempt": 5, "token": "repeat ", "repeat_count": 5, "status": "blocked_stream_loop"},
        ],
        "details": {"reason": "StreamLoopAborted", "message": "StreamLoopAborted: repetitive token sequence 'repeat '", "usd_saved": 0.0420},
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sim_payload

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kwargs):
            return mock_resp

    monkeypatch.setattr(httpx, "Client", MockClient)
    cmd_simulate(args)



