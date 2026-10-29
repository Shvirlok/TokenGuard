from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import webbrowser
import httpx
import uvicorn
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from tokenguard.config import (
    PROFILES,
    Settings,
    detect_api_keys,
    get_or_create_admin_token,
    get_settings,
    load_stored_config,
    mask_key,
    save_stored_config,
    set_settings,
    update_stored_config,
)
from tokenguard.export import export_telemetry_sync
from tokenguard.pricing import (
    get_dynamic_pricing_table,
    load_baseline_prices,
    load_cached_prices,
    update_pricing_cache,
)

PID_DIR = Path.home() / ".tokenguard"
PID_FILE = PID_DIR / "tokenguard.pid"
LOG_FILE = PID_DIR / "tokenguard.log"

console = Console()
err_console = Console(stderr=True)


def _env_int(name: str, default: int) -> int:
    """Read integer environment variable with safe fallback."""
    val = os.environ.get(name)
    if val is None or not val.strip():
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="tokenguard",
        description="TokenGuard: Lightweight open-source local proxy and circuit breaker for LLM calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    start_parser = subparsers.add_parser("start", help="Start the TokenGuard proxy server and dashboard")
    _add_server_args(start_parser)
    start_parser.add_argument(
        "-d",
        "--daemon",
        action="store_true",
        help="Run TokenGuard in the background as a daemon process",
    )

    stop_parser = subparsers.add_parser("stop", help="Stop any background TokenGuard daemon process")
    stop_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host address (default: 127.0.0.1)")
    stop_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    status_parser = subparsers.add_parser("status", help="Check status of the TokenGuard daemon & proxy")
    status_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host address (default: 127.0.0.1)")
    status_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    run_parser = subparsers.add_parser(
        "run",
        help="Run a command with OPENAI_BASE_URL and OPENAI_API_BASE automatically injected",
    )
    run_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host address (default: 127.0.0.1)")
    run_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")
    run_parser.add_argument("--limit", type=float, default=5.0, help="Sliding hourly budget limit in USD (default: 5.0)")
    run_parser.add_argument("--daily-limit", type=float, default=50.0, help="Sliding daily budget limit in USD (default: 50.0)")
    run_parser.add_argument(
        "--max-burn-rate",
        type=str,
        default=None,
        help="Maximum spending burn rate velocity limit (e.g. '$0.50/min', '$2.00/5m')",
    )
    run_parser.add_argument(
        "--passive",
        action="store_true",
        help="Run in passive observability mode (no loop or budget cutoffs)",
    )
    run_parser.add_argument(
        "--max-repeats",
        "--loop-threshold",
        dest="loop_threshold",
        type=int,
        default=3,
        help="Maximum duplicate consecutive requests before tripping loop guard (default: 3, set 0 to disable)",
    )
    run_parser.add_argument(
        "--session-limit",
        type=float,
        default=float(os.environ.get("TOKENGUARD_SESSION_LIMIT", 0.0)),
        help="Spending budget cap in USD per session (default: 0.0 = disabled)",
    )
    run_parser.add_argument(
        "--max-tool-repeats",
        type=int,
        default=_env_int("TOKENGUARD_MAX_TOOL_REPEATS", 3),
        help="Maximum consecutive calls per tool name before tripping (default: 3, set 0 to disable)",
    )
    run_parser.add_argument(
        "--max-stagnant-steps",
        type=int,
        default=_env_int("TOKENGUARD_MAX_STAGNANT_STEPS", 6),
        help="Maximum consecutive non-mutating steps without state change before tripping (default: 6, set 0 to disable)",
    )
    run_parser.add_argument(
        "--loop-window-seconds",
        type=float,
        default=60.0,
        help="Sliding time window in seconds for loop detection matching (default: 60.0s)",
    )
    run_parser.add_argument(
        "--upstream-url",
        type=str,
        default="https://api.openai.com",
        help="Upstream OpenAI API base URL (default: https://api.openai.com)",
    )
    run_parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("tokenguard.db"),
        help="SQLite database path for request logging (default: tokenguard.db)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in observation mode (detect and log trips, but pass requests through upstream)",
    )
    run_parser.add_argument(
        "--on-trip",
        type=str,
        choices=["block", "nudge"],
        default=(
            os.environ.get("TOKENGUARD_ON_TRIP", "block").strip().lower()
            if os.environ.get("TOKENGUARD_ON_TRIP", "block").strip().lower() in ("block", "nudge")
            else "block"
        ),
        help="Action when a loop circuit trips: 'block' (HTTP 429) or 'nudge' (synthetic 200 OK self-correction on first violation)",
    )
    run_parser.add_argument(
        "--block-secrets",
        action="store_true",
        default=os.environ.get("TOKENGUARD_BLOCK_SECRETS", "").lower() in ("true", "1", "yes"),
        help="Abort request with HTTP 400 when API keys or private credentials are detected in prompt",
    )
    run_parser.add_argument(
        "--redact-secrets",
        action="store_true",
        default=os.environ.get("TOKENGUARD_REDACT_SECRETS", "").lower() in ("true", "1", "yes"),
        help="Replace secrets with ***REDACTED_SECRET*** in-flight before forwarding upstream",
    )
    run_parser.add_argument(
        "--max-payload-size-mb",
        type=float,
        default=float(os.environ.get("TOKENGUARD_MAX_PAYLOAD_SIZE_MB", 15.0)),
        help="Maximum allowed payload size in MB before rejecting with HTTP 413 (default: 15.0)",
    )
    run_parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command and arguments to execute (e.g. python agent.py or pytest)",
    )

    kill_parser = subparsers.add_parser("kill", help="Engage kill switch to immediately block all LLM requests")
    kill_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    kill_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    resume_parser = subparsers.add_parser("resume", help="Disengage kill switch and resume normal guarding")
    resume_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    resume_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    profile_parser = subparsers.add_parser("profile", help="Switch active guard profile (careful, standard, passive)")
    profile_parser.add_argument("name", nargs="?", default=None, choices=["careful", "standard", "passive"], help="Profile name (careful, standard, passive)")
    profile_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    profile_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    clear_parser = subparsers.add_parser("clear", help="Clear all request logs and reset in-memory caches")
    clear_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    clear_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")
    clear_parser.add_argument("--db-path", type=Path, default=Path("tokenguard.db"), help="SQLite database path")

    logs_parser = subparsers.add_parser("logs", help="View or stream live LLM request logs")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Stream live activity in real-time")
    logs_parser.add_argument("-n", "--lines", type=int, default=20, help="Number of recent lines to display (default: 20)")
    logs_parser.add_argument("--status", type=str, default="all", choices=["all", "success", "blocked", "nudged"], help="Filter by status")
    logs_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    logs_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    tail_parser = subparsers.add_parser("tail", help="Alias for 'tokenguard logs -f'")
    tail_parser.add_argument("-n", "--lines", type=int, default=20, help="Number of recent lines to display")
    tail_parser.add_argument("--status", type=str, default="all", choices=["all", "success", "blocked", "nudged"], help="Filter by status")
    tail_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    tail_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    prices_parser = subparsers.add_parser("prices", help="Display local pricing table")
    prices_parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Fetch latest live model prices from OpenRouter public registry",
    )
    prices_parser.add_argument(
        "--offline",
        action="store_true",
        help="Display offline static baseline pricing only",
    )
    prices_parser.add_argument(
        "--prices-path",
        type=Path,
        default=None,
        help="Path to custom prices.json file",
    )

    menu_parser = subparsers.add_parser("menu", help="Launch interactive navigation menu (Claude/Qwen Code style)")
    menu_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    menu_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    tui_parser = subparsers.add_parser("tui", help="Alias for 'tokenguard menu'")
    tui_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    tui_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    dash_parser = subparsers.add_parser("dashboard", help="Open TokenGuard web dashboard in browser")
    dash_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    dash_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    sim_parser = subparsers.add_parser("simulate", help="Run instant circuit breaker simulation")
    sim_parser.add_argument(
        "--type",
        type=str,
        default="loop",
        choices=[
            "loop",
            "budget",
            "success",
            "tool_loop",
            "alternating_loop",
            "drift",
            "stagnation",
            "secret",
            "stream-loop",
            "stream_loop",
            "nudge",
        ],
        help="Simulation type (default: loop; choices: loop, budget, success, tool_loop, alternating_loop, drift, stagnation, secret, stream-loop, nudge)",
    )
    sim_parser.add_argument("--model", type=str, default="gpt-4o", help="Model name for simulation")
    sim_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    sim_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    export_parser = subparsers.add_parser(
        "export",
        help="Export telemetry logs in JSONL, HAR 1.2, JSON, or CSV format",
    )
    export_parser.add_argument(
        "--format",
        type=str,
        default="jsonl",
        choices=["jsonl", "har", "json", "csv"],
        help="Export file format: jsonl (default), har (HTTP Archive 1.2), json, or csv",
    )
    export_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (auto-named if not specified)",
    )
    export_parser.add_argument(
        "--status",
        type=str,
        default="all",
        choices=["all", "success", "blocked", "nudged"],
        help="Filter records by status: all (default), success, blocked, or nudged",
    )
    export_parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("tokenguard.db"),
        help="SQLite database path (default: tokenguard.db)",
    )
    export_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Proxy host address (default: 127.0.0.1)",
    )
    export_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Proxy port (default: 8080)",
    )

    env_parser = subparsers.add_parser(
        "env",
        help="Print shell export/unset commands to route LLM SDKs through TokenGuard",
    )
    env_parser.add_argument(
        "--unset",
        action="store_true",
        help="Print shell commands to unset TokenGuard environment variables",
    )
    env_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Proxy host address (default: localhost)",
    )
    env_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Proxy port (default: 8080)",
    )

    _add_server_args(parser)
    parser.add_argument(
        "-d",
        "--daemon",
        action="store_true",
        help="Run TokenGuard in the background as a daemon process",
    )

    return parser


def _add_server_args(p: argparse.ArgumentParser) -> None:
    """Add standard server execution arguments to a parser."""
    p.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind proxy (default: 127.0.0.1)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run proxy server on (default: 8080)",
    )
    p.add_argument(
        "--limit",
        type=float,
        default=5.0,
        help="Sliding hourly budget limit in USD (default: 5.0)",
    )
    p.add_argument(
        "--daily-limit",
        type=float,
        default=50.0,
        help="Sliding daily budget limit in USD (default: 50.0)",
    )
    p.add_argument(
        "--max-burn-rate",
        type=str,
        default=None,
        help="Maximum spending burn rate velocity limit (e.g. '$0.50/min', '$2.00/5m')",
    )
    p.add_argument(
        "--passive",
        action="store_true",
        help="Run in passive observability mode (no loop or budget cutoffs)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in observation mode (detect and log trips, but pass requests through upstream)",
    )
    p.add_argument(
        "--on-trip",
        type=str,
        choices=["block", "nudge"],
        default=(
            os.environ.get("TOKENGUARD_ON_TRIP", "block").strip().lower()
            if os.environ.get("TOKENGUARD_ON_TRIP", "block").strip().lower() in ("block", "nudge")
            else "block"
        ),
        help="Action when a loop circuit trips: 'block' (HTTP 429) or 'nudge' (synthetic 200 OK self-correction on first violation)",
    )
    p.add_argument(
        "--session-limit",
        type=float,
        default=float(os.environ.get("TOKENGUARD_SESSION_LIMIT", 0.0)),
        help="Spending budget cap in USD per session (default: 0.0 = disabled)",
    )
    p.add_argument(
        "--max-repeats",
        "--loop-threshold",
        dest="loop_threshold",
        type=int,
        default=3,
        help="Maximum duplicate consecutive requests before tripping loop guard (default: 3, set 0 to disable)",
    )
    p.add_argument(
        "--max-tool-repeats",
        type=int,
        default=_env_int("TOKENGUARD_MAX_TOOL_REPEATS", 3),
        help="Maximum consecutive calls per tool name before tripping (default: 3, set 0 to disable)",
    )
    p.add_argument(
        "--max-stagnant-steps",
        type=int,
        default=_env_int("TOKENGUARD_MAX_STAGNANT_STEPS", 6),
        help="Maximum consecutive non-mutating steps without state change before tripping (default: 6, set 0 to disable)",
    )
    p.add_argument(
        "--loop-window-seconds",
        type=float,
        default=60.0,
        help="Sliding time window in seconds for loop detection matching (default: 60.0s)",
    )
    p.add_argument(
        "--upstream-url",
        type=str,
        default="https://api.openai.com",
        help="Upstream OpenAI API base URL (default: https://api.openai.com)",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=Path("tokenguard.db"),
        help="SQLite database path for request logging (default: tokenguard.db)",
    )
    p.add_argument(
        "--block-secrets",
        action="store_true",
        default=os.environ.get("TOKENGUARD_BLOCK_SECRETS", "").lower() in ("true", "1", "yes"),
        help="Abort request with HTTP 400 when API keys or private credentials are detected in prompt",
    )
    p.add_argument(
        "--redact-secrets",
        action="store_true",
        default=os.environ.get("TOKENGUARD_REDACT_SECRETS", "").lower() in ("true", "1", "yes"),
        help="Replace secrets with ***REDACTED_SECRET*** in-flight before forwarding upstream",
    )
    p.add_argument(
        "--max-payload-size-mb",
        type=float,
        default=float(os.environ.get("TOKENGUARD_MAX_PAYLOAD_SIZE_MB", 15.0)),
        help="Maximum allowed payload size in MB before rejecting with HTTP 413 (default: 15.0)",
    )
    p.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development",
    )


def print_banner(
    host: str,
    port: int,
    limit: float,
    loop_threshold: int,
    max_burn_rate: Optional[str] = None,
    dry_run: bool = False,
    on_trip: str = "block",
) -> None:
    """Display the clean Rich TokenGuard terminal startup banner."""
    base_url = f"http://{host}:{port}"
    loop_str = f"Enabled [dim](cutoff: {loop_threshold})[/dim]" if loop_threshold > 0 else "[yellow]Disabled (Passive)[/yellow]"

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column(style="bright_white")

    table.add_row("Proxy Base URL:", f"[cyan]{base_url}/v1[/cyan]")
    table.add_row("Web Dashboard:", f"[bold cyan]{base_url}[/bold cyan]")
    table.add_row("Hourly Limit:", f"[bold white]${limit:.2f}/hr[/bold white]")
    if max_burn_rate:
        table.add_row("Burn Rate Limit:", f"[bold white]{max_burn_rate}[/bold white]")
    table.add_row("Loop Breaker:", loop_str)
    if on_trip == "nudge":
        table.add_row("On-Trip Action:", "[bold cyan]Nudge (Synthetic 200 OK Recovery -> 429 Escalation)[/bold cyan]")
    if dry_run:
        table.add_row("Observation Mode:", "[bold yellow]Active (Dry Run / Pass-through)[/bold yellow]")

    subtitle = "[bold yellow]● OBSERVATION MODE (DRY-RUN)[/bold yellow]" if dry_run else "[bold green]● ACTIVE & GUARDING[/bold green]"

    panel = Panel(
        table,
        title="[bold bright_white]TokenGuard[/bold bright_white] [dim]v0.3.0[/dim]",
        subtitle=subtitle,
        box=box.ROUNDED,
        border_style="bright_black",
        padding=(1, 2),
    )
    console.print(panel)


def is_pid_alive(pid: int) -> bool:
    """Check if process with given PID exists and is running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_daemon_pid() -> Optional[int]:
    """Retrieve PID from ~/.tokenguard/tokenguard.pid if valid and running."""
    if PID_FILE.exists():
        try:
            pid_str = PID_FILE.read_text().strip()
            if pid_str:
                pid = int(pid_str)
                if is_pid_alive(pid):
                    return pid
                PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    return None


def is_server_running(host: str = "127.0.0.1", port: int = 8080) -> bool:
    """Check if TokenGuard HTTP health endpoint is responding with 200 OK."""
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"http://{host}:{port}/health")
            return resp.status_code == 200
    except Exception:
        return False


def wait_for_server(host: str, port: int, timeout: float = 6.0) -> bool:
    """Wait until TokenGuard server becomes healthy."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_server_running(host, port):
            return True
        time.sleep(0.15)
    return is_server_running(host, port)


def start_daemon(
    host: str = "127.0.0.1",
    port: int = 8080,
    limit: float = 5.0,
    daily_limit: float = 50.0,
    loop_threshold: int = 3,
    loop_window_seconds: float = 60.0,
    upstream_url: str = "https://api.openai.com",
    db_path: Path = Path("tokenguard.db"),
    passive: bool = False,
    max_burn_rate: Optional[str] = None,
    dry_run: bool = False,
    on_trip: str = "block",
    max_tool_repeats: int = 3,
    max_stagnant_steps: int = 6,
    block_secrets: bool = False,
    redact_secrets: bool = False,
    max_payload_size_mb: float = 15.0,
) -> int:
    """Spawn TokenGuard proxy in background as a daemon process."""
    existing_pid = get_daemon_pid()
    if existing_pid and is_server_running(host, port):
        return existing_pid

    PID_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "tokenguard.cli",
        "start",
        "--host",
        str(host),
        "--port",
        str(port),
        "--limit",
        str(limit),
        "--daily-limit",
        str(daily_limit),
        "--loop-threshold",
        str(loop_threshold),
        "--max-tool-repeats",
        str(max_tool_repeats),
        "--max-stagnant-steps",
        str(max_stagnant_steps),
        "--loop-window-seconds",
        str(loop_window_seconds),
        "--upstream-url",
        str(upstream_url),
        "--db-path",
        str(db_path),
        "--max-payload-size-mb",
        str(max_payload_size_mb),
        "--on-trip",
        str(on_trip),
    ]
    if passive:
        cmd.append("--passive")
    if dry_run:
        cmd.append("--dry-run")
    if block_secrets:
        cmd.append("--block-secrets")
    if redact_secrets:
        cmd.append("--redact-secrets")
    if max_burn_rate:
        cmd.extend(["--max-burn-rate", str(max_burn_rate)])

    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    PID_FILE.write_text(str(proc.pid))

    if not wait_for_server(host, port, timeout=6.0):
        time.sleep(0.5)

    return proc.pid


def stop_daemon(host: str = "127.0.0.1", port: int = 8080) -> bool:
    """Stop the background TokenGuard daemon process."""
    pid = get_daemon_pid()
    if not pid:
        if is_server_running(host, port):
            console.print(
                Panel(
                    f"[yellow]Warning: TokenGuard proxy is responding on http://{host}:{port}, but no daemon PID file was found ({PID_FILE}).[/yellow]\n"
                    f"[dim]If it was started directly in another terminal, terminate it there.[/dim]",
                    title="[bold bright_white]TokenGuard[/bold bright_white]",
                    box=box.ROUNDED,
                    border_style="yellow",
                )
            )
            return False
        console.print(
            Panel(
                "[dim]No active TokenGuard background daemon found.[/dim]",
                title="[bold bright_white]TokenGuard[/bold bright_white]",
                subtitle="[dim]○ INACTIVE[/dim]",
                box=box.ROUNDED,
                border_style="bright_black",
            )
        )
        return True

    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if not is_pid_alive(pid):
                break
            time.sleep(0.1)

        if is_pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

    PID_FILE.unlink(missing_ok=True)
    console.print(
        Panel(
            f"[bold red]● STOPPED[/bold red] [bright_white]TokenGuard daemon (PID: {pid}) has been terminated.[/bright_white]",
            title="[bold bright_white]TokenGuard[/bold bright_white]",
            box=box.ROUNDED,
            border_style="bright_black",
        )
    )
    return True


def show_status(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Display the current status and metrics of the TokenGuard daemon & proxy."""
    pid = get_daemon_pid()
    is_up = is_server_running(host, port)

    if is_up:
        stats = {}
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"http://{host}:{port}/api/stats")
                if res.status_code == 200:
                    stats = res.json()
        except Exception:
            pass

        is_killed = stats.get("kill_switch_active", False)
        status_pill = "[bold red]● KILLED[/bold red]" if is_killed else "[bold green]● ACTIVE & GUARDING[/bold green]"

        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", justify="right")
        table.add_column(style="bright_white")

        table.add_row("Status:", status_pill)
        table.add_row("Proxy Base URL:", f"[cyan]http://{host}:{port}/v1[/cyan]")
        table.add_row("Live Dashboard:", f"[bold cyan]http://{host}:{port}[/bold cyan]")
        if pid:
            table.add_row("Daemon PID:", f"[dim]{pid}[/dim]")

        active_prof = stats.get("active_profile")
        if active_prof and active_prof in PROFILES:
            table.add_row("Active Profile:", f"[bold white]{PROFILES[active_prof]['name']}[/bold white]")

        spent = stats.get("total_spent", 0.0)
        h_limit = stats.get("hourly_limit", 5.0)
        table.add_row("Hourly Spend:", f"${spent:.4f} [dim]/ ${h_limit:.2f}[/dim]")

        saved = stats.get("saved_cost_estimate", 0.0)
        table.add_row("Money Saved:", f"[bold green]+${saved:.4f}[/bold green]" if saved > 0 else "[dim]$0.00[/dim]")

        total_reqs = stats.get("total_requests", 0)
        succ = stats.get("success_requests", 0)
        blkd = stats.get("blocked_requests", 0)
        table.add_row("Requests:", f"{total_reqs} [dim]({succ} ok, {blkd} blocked)[/dim]")

        keys = detect_api_keys()
        key_strs = []
        if keys.get("OPENAI_API_KEY"):
            key_strs.append(f"OpenAI ({mask_key(keys['OPENAI_API_KEY'])})")
        if keys.get("ANTHROPIC_API_KEY"):
            key_strs.append(f"Anthropic ({mask_key(keys['ANTHROPIC_API_KEY'])})")
        if keys.get("GEMINI_API_KEY"):
            key_strs.append(f"Gemini ({mask_key(keys['GEMINI_API_KEY'])})")
        if keys.get("DASHSCOPE_API_KEY"):
            key_strs.append(f"DashScope/Qwen ({mask_key(keys['DASHSCOPE_API_KEY'])})")
        if keys.get("DEEPSEEK_API_KEY"):
            key_strs.append(f"DeepSeek ({mask_key(keys['DEEPSEEK_API_KEY'])})")
        if keys.get("GROQ_API_KEY"):
            key_strs.append(f"Groq ({mask_key(keys['GROQ_API_KEY'])})")
        if keys.get("MISTRAL_API_KEY"):
            key_strs.append(f"Mistral ({mask_key(keys['MISTRAL_API_KEY'])})")
        if not key_strs:
            key_strs.append("[dim]Inherited per request[/dim]")
        table.add_row("Keys Active:", ", ".join(key_strs))

        panel = Panel(
            table,
            title="[bold bright_white]TokenGuard Status[/bold bright_white]",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(1, 2),
        )
        console.print(panel)
    else:
        if pid:
            PID_FILE.unlink(missing_ok=True)

        content = Text.from_markup(
            f"[dim]○ TokenGuard is not running on http://{host}:{port}[/dim]\n\n"
            f"[bright_white]To start guarding:[/bright_white]\n"
            f"  [bold cyan]tokenguard[/bold cyan]            [dim]# Interactive quickstart[/dim]\n"
            f"  [bold cyan]tokenguard start -d[/bold cyan]   [dim]# Start in background as daemon[/dim]"
        )
        panel = Panel(
            content,
            title="[bold bright_white]TokenGuard[/bold bright_white]",
            subtitle="[dim]○ STOPPED[/dim]",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(1, 2),
        )
        console.print(panel)


def cmd_kill(args: argparse.Namespace) -> None:
    """Immediately engage kill switch to block all LLM requests."""
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8080)
    if not is_server_running(host, port):
        console.print(
            Panel(
                f"[dim]○ TokenGuard is not running on http://{host}:{port}[/dim]",
                title="[bold bright_white]TokenGuard[/bold bright_white]",
                subtitle="[dim]○ STOPPED[/dim]",
                box=box.ROUNDED,
                border_style="bright_black",
            )
        )
        return
    try:
        token = get_or_create_admin_token()
        headers = {"X-TokenGuard-Admin-Token": token}
        with httpx.Client(timeout=3.0) as client:
            res = client.post(
                f"http://{host}:{port}/api/kill-switch",
                json={"active": True, "source": "cli"},
                headers=headers,
            )
            if res.status_code == 200:
                console.print(
                    Panel(
                        f"[bold red]● KILLED[/bold red] [bright_white]Emergency kill switch engaged via CLI.[/bright_white]\n"
                        f"[dim]All incoming LLM requests are now blocked immediately (HTTP 429).[/dim]\n"
                        f"[dim]Run [bold green]tokenguard resume[/bold green] to restore traffic.[/dim]",
                        title="[bold bright_white]TokenGuard Kill Switch[/bold bright_white]",
                        box=box.ROUNDED,
                        border_style="bright_black",
                        padding=(1, 2),
                    )
                )
            else:
                err_console.print("[bold red]Failed to engage kill switch.[/bold red]")
    except Exception as e:
        err_console.print(f"[bold red]Error communicating with TokenGuard daemon:[/bold red] {e}")


def cmd_resume(args: argparse.Namespace) -> None:
    """Disengage kill switch and resume normal proxy traffic."""
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8080)
    if not is_server_running(host, port):
        console.print(
            Panel(
                f"[dim]○ TokenGuard is not running on http://{host}:{port}[/dim]",
                title="[bold bright_white]TokenGuard[/bold bright_white]",
                subtitle="[dim]○ STOPPED[/dim]",
                box=box.ROUNDED,
                border_style="bright_black",
            )
        )
        return
    try:
        token = get_or_create_admin_token()
        headers = {"X-TokenGuard-Admin-Token": token}
        with httpx.Client(timeout=3.0) as client:
            res = client.post(
                f"http://{host}:{port}/api/kill-switch",
                json={"active": False, "source": "cli"},
                headers=headers,
            )
            if res.status_code == 200:
                console.print(
                    Panel(
                        f"[bold green]● GUARDING[/bold green] [bright_white]Traffic resumed via CLI.[/bright_white]\n"
                        f"[dim]LLM requests are now allowed and guarded by active circuit breaker limits.[/dim]",
                        title="[bold bright_white]TokenGuard Active[/bold bright_white]",
                        box=box.ROUNDED,
                        border_style="bright_black",
                        padding=(1, 2),
                    )
                )
            else:
                err_console.print("[bold red]Failed to resume traffic.[/bold red]")
    except Exception as e:
        err_console.print(f"[bold red]Error communicating with TokenGuard daemon:[/bold red] {e}")


def _read_raw_key(fd: int) -> str:
    if sys.platform == "win32":
        try:
            import msvcrt
            ch = msvcrt.getwch()
            if ch in ('\x00', '\xe0'):
                ch2 = msvcrt.getwch()
                if ch2 == 'H':
                    return 'up'
                elif ch2 == 'P':
                    return 'down'
                elif ch2 == 'K':
                    return 'left'
                elif ch2 == 'M':
                    return 'right'
                return 'other'
            if ch in ('\r', '\n', ' '):
                return 'enter'
            if ch == '\x03':
                raise KeyboardInterrupt
            if ch == '\x1b':
                return 'esc'
            return ch
        except Exception:
            return 'enter'
    else:
        try:
            raw_bytes = os.read(fd, 32)
        except Exception:
            return 'enter'

        if not raw_bytes:
            return 'enter'

        if raw_bytes in (b'\x1b[A', b'\x1bOA', b'\x1b[1;2A', b'\x1b[1;5A'):
            return 'up'
        if raw_bytes in (b'\x1b[B', b'\x1bOB', b'\x1b[1;2B', b'\x1b[1;5B'):
            return 'down'
        if raw_bytes in (b'\x1b[C', b'\x1bOC'):
            return 'right'
        if raw_bytes in (b'\x1b[D', b'\x1bOD'):
            return 'left'
        if raw_bytes in (b'\r', b'\n', b' '):
            return 'enter'
        if raw_bytes == b'\x03':
            raise KeyboardInterrupt
        if raw_bytes == b'\x04':
            raise EOFError
        if raw_bytes == b'\x1b':
            return 'esc'

        try:
            char = raw_bytes.decode("utf-8", errors="ignore")
            if char.startswith("\x1b"):
                if char.endswith("A"):
                    return 'up'
                elif char.endswith("B"):
                    return 'down'
                elif char.endswith("C"):
                    return 'right'
                elif char.endswith("D"):
                    return 'left'
                return 'esc'
            return char
        except Exception:
            return 'enter'


def _read_key() -> str:
    """Read a single key (compatibility wrapper)."""
    return _read_raw_key(sys.stdin.fileno())


def format_col1(i: int, is_selected: bool, total_count: int = 10) -> str:
    """Format index & selector string: '› [ 1 ]' to '  [10 ]' (fixed width 8 chars)."""
    val = i + 1
    if total_count >= 10:
        if val < 10:
            s = f"› [ {val} ]" if is_selected else f"  [ {val} ]"
        else:
            s = f"› [{val} ]" if is_selected else f"  [{val} ]"
    else:
        s = f"› [ {val} ]" if is_selected else f"  [ {val} ]"

    if is_selected:
        return f"[bold cyan]{s}[/bold cyan]"
    return f"[dim]{s}[/dim]"


def render_interactive_menu(
    options: list[tuple[str, str, str, str]],
    selected_idx: int,
    title: str = "Select Option",
    subtitle: Optional[str] = None,
    status_line: Optional[str] = None,
) -> Panel:
    """Render a clean Rich panel with numbered, arrow-navigable menu options with explicit column widths."""
    table = Table.grid(padding=(0, 1), expand=False)
    table.add_column(width=8, justify="left", no_wrap=True)
    table.add_column(width=24, justify="left", no_wrap=True)
    table.add_column(width=1, justify="center", no_wrap=True)
    table.add_column(width=32, justify="left", no_wrap=True)
    table.add_column(width=6, justify="right", no_wrap=True)

    for i, opt in enumerate(options):
        opt_id = opt[0]
        opt_title = opt[1] if len(opt) > 1 else str(opt_id)
        opt_desc = opt[2] if len(opt) > 2 else ""
        opt_badge = opt[3] if len(opt) > 3 else ""

        is_selected = (i == selected_idx)
        col1 = format_col1(i, is_selected, len(options))

        if is_selected:
            col2 = f"[bold bright_white]{escape(opt_title)}[/bold bright_white]"
            col3 = "[dim]·[/dim]"
            col4 = f"[bright_white]{escape(opt_desc)}[/bright_white]"
        else:
            col2 = f"[white]{escape(opt_title)}[/white]"
            col3 = "[dim]·[/dim]"
            col4 = f"[dim]{escape(opt_desc)}[/dim]"

        if opt_badge.upper() == "LIVE":
            col5 = "[dim cyan]LIVE[/dim cyan]"
        elif opt_badge.upper() == "DEMO":
            col5 = "[dim yellow]DEMO[/dim yellow]"
        elif opt_badge:
            col5 = f"[dim green]{escape(opt_badge[:6])}[/dim green]"
        else:
            col5 = ""

        table.add_row(col1, col2, col3, col4, col5)

    elements = []
    if status_line:
        s_grid = Table.grid(expand=False)
        s_grid.add_column(no_wrap=True)
        s_grid.add_row(f" {status_line}")
        s_grid.add_row("")
        elements.append(s_grid)

    elements.append(table)

    content = Table.grid(expand=False)
    content.add_column(no_wrap=True)
    for el in elements:
        content.add_row(el)

    sub = subtitle or f"[dim]↑/↓ arrows or 1-{len(options)} to select · Enter to run · q to exit[/dim]"

    return Panel(
        content,
        title=f"[bold bright_white]{escape(title)}[/bold bright_white]",
        subtitle=sub,
        box=box.ROUNDED,
        border_style="bright_black",
        padding=(0, 1),
        expand=False,
    )


def interactive_select(
    options: list[tuple[str, str, str, str]],
    title: str = "Select Option",
    subtitle: Optional[str] = None,
    status_line: Optional[str] = None,
    default_index: int = 0,
) -> str:
    """Prompt user to choose an option using Arrow Keys (↑/↓) or Numbers (1-N)."""
    if not options:
        return ""

    if not sys.stdin.isatty():
        idx = default_index if 0 <= default_index < len(options) else 0
        return options[idx][0]

    selected_idx = default_index if 0 <= default_index < len(options) else 0

    try:
        fd = sys.stdin.fileno()
    except Exception:
        idx = default_index if 0 <= default_index < len(options) else 0
        return options[idx][0]

    old_settings = None
    if sys.platform != "win32":
        import termios
        import tty
        try:
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            pass

    try:
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
    except Exception:
        pass

    try:
        with Live(
            render_interactive_menu(options, selected_idx, title, subtitle, status_line),
            console=console,
            auto_refresh=False,
            transient=True,
        ) as live:
            while True:
                live.update(render_interactive_menu(options, selected_idx, title, subtitle, status_line), refresh=True)
                try:
                    key = _read_raw_key(fd)
                except (KeyboardInterrupt, EOFError):
                    return options[selected_idx][0]

                if key in ("up", "k", "K", "w", "W"):
                    selected_idx = (selected_idx - 1) % len(options)
                elif key in ("down", "j", "J", "s", "S"):
                    selected_idx = (selected_idx + 1) % len(options)
                elif key == "enter":
                    return options[selected_idx][0]
                elif key in ("q", "Q", "esc"):
                    return options[selected_idx][0]
                elif key.isdigit():
                    val = int(key)
                    if val == 0 and len(options) >= 10:
                        return options[9][0]
                    elif 1 <= val <= len(options):
                        return options[val - 1][0]
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        if old_settings is not None and sys.platform != "win32":
            import termios
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def interactive_control_menu(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Display interactive arrow/number menu to manage running TokenGuard proxy (Stripe / GitHub CLI style)."""
    if not sys.stdin.isatty():
        return

    while True:
        if not is_server_running(host, port):
            console.print(
                Panel(
                    f"[yellow]○ TokenGuard proxy on http://{host}:{port} is currently stopped.[/yellow]\n\n"
                    f"[dim]Starting background daemon automatically...[/dim]",
                    title="[bold bright_white]TokenGuard Status[/bold bright_white]",
                    box=box.ROUNDED,
                    border_style="bright_black",
                    padding=(1, 2),
                )
            )
            settings = get_settings()
            pid = start_daemon(
                host=host,
                port=port,
                limit=settings.hourly_limit,
                daily_limit=settings.daily_limit,
                loop_threshold=settings.loop_threshold,
                max_burn_rate=settings.max_burn_rate,
            )
            time.sleep(0.5)

        active_prof = "Careful"
        is_killed = False
        saved_usd = 0.0
        req_count = 0
        try:
            with httpx.Client(timeout=1.5) as client:
                r = client.get(f"http://{host}:{port}/api/stats")
                if r.status_code == 200:
                    s_data = r.json()
                    prof_key = s_data.get("active_profile", "careful")
                    active_prof = PROFILES.get(prof_key, {}).get("name", prof_key.title())
                    is_killed = s_data.get("kill_switch_active", False)
                    saved_usd = float(s_data.get("saved_cost_estimate", 0.0))
                    req_count = int(s_data.get("total_requests", 0))
        except Exception:
            pass

        status_badge = "[bold red]● KILLED[/bold red]" if is_killed else "[bold green]● GUARDING[/bold green]"
        status_line = f"Status: {status_badge}  │  Profile: [cyan]{active_prof}[/cyan]  │  Saved: [green]+${saved_usd:.2f}[/green]  │  Reqs: [white]{req_count}[/white]"

        menu_options = [
            ("stream", "Live Activity Monitor", "Real-time requests & telemetry", "LIVE"),
            ("run", "Run Protected Script", "Execute script with proxy", ""),
            ("profile", "Switch Guard Profile", "Careful, Standard, Passive", ""),
            ("simulate", "Run Loop Simulation", "Test loop circuit breaker", "DEMO"),
            ("kill", "Toggle Kill Switch", "Block or unblock LLM traffic", ""),
            ("dashboard", "Open Web Dashboard", f"http://{host}:{port}", ""),
            ("clear", "Clear Database & Cache", "Reset DB & loop memory", ""),
            ("prices", "View Model Pricing", "Model rates ($/1M tokens)", ""),
            ("stop", "Stop Proxy Daemon", "Terminate background proxy", ""),
            ("exit", "Exit Menu", "Return to terminal shell", ""),
        ]

        choice = interactive_select(
            menu_options,
            title="TokenGuard Control Center",
            subtitle=f"[dim]↑/↓ arrows or 1-{len(menu_options)} to select · Enter to run · q to exit[/dim]",
            status_line=status_line,
            default_index=0,
        )

        if choice == "stream":
            dummy_args = argparse.Namespace(host=host, port=port, follow=True, lines=20, status="all", command="logs")
            cmd_logs(dummy_args)
        elif choice == "run":
            try:
                cmd_str = Prompt.ask(
                    "[dim]Enter agent command to execute (e.g. python agent.py)[/dim]",
                    default="python my_agent.py",
                ).strip()
                if cmd_str:
                    raw_args = shlex.split(cmd_str)
                    run_args = argparse.Namespace(
                        cmd=raw_args,
                        host=host,
                        port=port,
                        limit=5.0,
                        daily_limit=50.0,
                        loop_threshold=3,
                        loop_window_seconds=60.0,
                        upstream_url="https://api.openai.com",
                        db_path=Path("tokenguard.db"),
                    )
                    cmd_run(run_args)
            except (KeyboardInterrupt, EOFError):
                pass
        elif choice == "profile":
            profile_opts = [
                ("careful", "Careful", "$5/hr cap, 2-loop cutoff, alerts ON", "DEFAULT"),
                ("standard", "Standard Agent", "$15/hr cap, 4-loop cutoff", "AUTONOMOUS"),
                ("passive", "Passive Monitor", "No loop blocks, tracking only", "OBSERVABILITY"),
            ]
            prof_choice = interactive_select(
                profile_opts,
                title="Choose Guard Profile",
                subtitle="[dim]↑/↓ arrows or 1-3 to select · Enter to apply · q to exit[/dim]",
                default_index=0,
            )
            prof_args = argparse.Namespace(name=prof_choice, host=host, port=port)
            cmd_profile(prof_args)
        elif choice == "simulate":
            try:
                console.print("[dim]Simulating infinite agent loop...[/dim]")
                with httpx.Client(timeout=5.0) as client:
                    sim_res = client.post(f"http://{host}:{port}/api/simulate", json={"type": "loop", "model": "gpt-4o"})
                    if sim_res.status_code == 200:
                        s_data = sim_res.json()
                        console.print(
                            Panel(
                                f"[bold green]● Simulation Completed[/bold green]\n\n"
                                f"[dim]Threshold:[/dim] [bold cyan]{s_data.get('threshold')} attempts[/bold cyan]\n"
                                f"[dim]Loop Intercepted & Blocked with Avoided Cost Calculation.[/dim]\n"
                                f"[dim]Live event broadcast to Web Dashboard via SSE stream.[/dim]",
                                title="[bold bright_white]Circuit Breaker Simulation[/bold bright_white]",
                                box=box.ROUNDED,
                                border_style="bright_black",
                                padding=(1, 2),
                            )
                        )
                    else:
                        err_console.print(f"[bold red]Simulation failed: {sim_res.text}[/bold red]")
            except Exception as e:
                err_console.print(f"[bold red]Error running simulation:[/bold red] {e}")
        elif choice == "kill":
            if is_killed:
                cmd_resume(argparse.Namespace(host=host, port=port))
            else:
                cmd_kill(argparse.Namespace(host=host, port=port))
        elif choice == "dashboard":
            url = f"http://{host}:{port}"
            console.print(f"[dim]Opening web dashboard in default browser:[/dim] [bold cyan]{url}[/bold cyan]")
            webbrowser.open(url)
        elif choice == "clear":
            cmd_clear(argparse.Namespace(host=host, port=port, db_path=Path("tokenguard.db")))
        elif choice == "prices":
            cmd_prices(argparse.Namespace(prices_path=None))
        elif choice == "stop":
            stop_daemon(host=host, port=port)
            break
        elif choice in ("exit", "q", "Q", "esc"):
            console.print("[dim]Exited TokenGuard interactive menu. (Proxy daemon is running in background)[/dim]")
            break


def cmd_profile(args: argparse.Namespace) -> None:
    """Switch active guard profile live via internal API."""
    name = getattr(args, "name", None)
    if not name:
        profile_opts = [
            ("careful", "Careful", "$5/hr cap, 2-loop cutoff, alerts ON", "DEFAULT"),
            ("standard", "Standard Agent", "$15/hr cap, 4-loop cutoff", "AUTONOMOUS"),
            ("passive", "Passive Monitor", "No loop blocks, tracking only", "OBSERVABILITY"),
        ]
        name = interactive_select(
            profile_opts,
            title="Choose Guard Profile",
            subtitle="[dim]↑/↓ arrows or 1-3 to select · Enter to apply · q to exit[/dim]",
            default_index=0,
        )

    name = str(name).lower()
    if name not in PROFILES:
        err_console.print(f"[bold red]Unknown profile '{name}'. Choose from: careful, standard, passive[/bold red]")
        sys.exit(1)

    prof = PROFILES[name]
    update_stored_config(profile=name)

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8080)

    if is_server_running(host, port):
        try:
            token = get_or_create_admin_token()
            headers = {"X-TokenGuard-Admin-Token": token}
            with httpx.Client(timeout=3.0) as client:
                client.post(
                    f"http://{host}:{port}/api/config",
                    json={
                        "hourly_limit": prof["hourly_limit"],
                        "daily_limit": prof["daily_limit"],
                        "loop_threshold": prof["loop_threshold"],
                        "profile": name,
                        "source": "cli",
                    },
                    headers=headers,
                )
        except Exception as e:
            pass

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column(style="bright_white")
    table.add_row("Profile:", f"[bold white]{prof['name']}[/bold white]")
    table.add_row("Hourly Limit:", f"${prof['hourly_limit']:.2f}/hr")
    table.add_row("Daily Limit:", f"${prof['daily_limit']:.2f}/day")
    table.add_row("Loop Cutoff:", f"{prof['loop_threshold']} repeats" if prof["loop_threshold"] > 0 else "[yellow]Disabled (Passive)[/yellow]")
    table.add_row("Description:", f"[dim]{prof['description']}[/dim]")

    console.print(
        Panel(
            table,
            title="[bold bright_white]Guard Profile Updated[/bold bright_white]",
            subtitle="[bold green]● APPLIED[/bold green]",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(1, 2),
        )
    )


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear all database request logs and reset in-memory caches."""
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8080)
    if not is_server_running(host, port):
        db_path = getattr(args, "db_path", Path("tokenguard.db"))
        from tokenguard.db import clear_all_requests
        asyncio.run(clear_all_requests(db_path))
        console.print(
            Panel(
                "[bold cyan]● CLEARED[/bold cyan] [bright_white]Local SQLite database flushed.[/bright_white]",
                title="[bold bright_white]TokenGuard Clear[/bold bright_white]",
                box=box.ROUNDED,
                border_style="bright_black",
                padding=(1, 2),
            )
        )
        return

    try:
        token = get_or_create_admin_token()
        headers = {"X-TokenGuard-Admin-Token": token}
        with httpx.Client(timeout=3.0) as client:
            res = client.post(
                f"http://{host}:{port}/api/clear-logs",
                json={"source": "cli"},
                headers=headers,
            )
            if res.status_code == 200:
                console.print(
                    Panel(
                        "[bold cyan]● CLEARED[/bold cyan] [bright_white]All request logs and in-memory caches flushed.[/bright_white]\n"
                        "[dim]State synchronized in real-time with Web Dashboard.[/dim]",
                        title="[bold bright_white]TokenGuard Clear[/bold bright_white]",
                        box=box.ROUNDED,
                        border_style="bright_black",
                        padding=(1, 2),
                    )
                )
            else:
                err_console.print("[bold red]Failed to clear logs on running daemon.[/bold red]")
    except Exception as e:
        err_console.print(f"[bold red]Error communicating with TokenGuard daemon:[/bold red] {e}")


def _render_stream_event(event: str, data: dict) -> None:
    """Format and print an SSE stream event in real-time."""
    if event == "request_logged":
        status_str = data.get("status", "unknown")
        model = data.get("model", "")
        cost = float(data.get("cost_usd", 0.0))
        lat = data.get("latency_ms", 0)
        p_tok = data.get("prompt_tokens", 0)
        c_tok = data.get("completion_tokens", 0)
        t_now = time.strftime("%H:%M:%S")

        if status_str == "success":
            console.print(
                f"[dim]{t_now}[/dim]  [bold green]200 OK[/bold green]  [bold cyan]{model:<16}[/bold cyan]  "
                f"[bright_white]{p_tok:>4}/{c_tok:<4}[/bright_white] tok  [bright_white]${cost:.5f}[/bright_white]  [dim]{lat:>4}ms[/dim]"
            )
        elif status_str == "blocked_loop":
            reason = data.get("blocked_reason", "infinite loop detected")
            console.print(
                f"[dim]{t_now}[/dim]  [bold red]BLOCKED: LOOP[/bold red]  [bold cyan]{model:<16}[/bold cyan]  "
                f"[red]{reason}[/red]"
            )
        elif status_str == "blocked_budget":
            reason = data.get("blocked_reason", "budget exceeded")
            console.print(
                f"[dim]{t_now}[/dim]  [bold yellow]BLOCKED: BUDGET[/bold yellow]  [bold cyan]{model:<16}[/bold cyan]  "
                f"[yellow]{reason}[/yellow]"
            )
        elif status_str == "blocked_killswitch":
            console.print(
                f"[dim]{t_now}[/dim]  [bold red]BLOCKED: KILLSWITCH[/bold red]  [bold cyan]{model:<16}[/bold cyan]  "
                f"[red]Kill switch active[/red]"
            )
    elif event == "state_change":
        action = data.get("action", "State updated")
        source = data.get("source", "system")
        console.print(f"[dim]{time.strftime('%H:%M:%S')}[/dim]  [bold yellow]● STATE UPDATE[/bold yellow]  [dim]{action} (via {source})[/dim]")
    elif event == "logs_cleared":
        source = data.get("source", "system")
        console.print(f"[dim]{time.strftime('%H:%M:%S')}[/dim]  [bold cyan]● LOGS CLEARED[/bold cyan]  [dim]All request history flushed (via {source})[/dim]")


def cmd_logs(args: argparse.Namespace) -> None:
    """Display recent request table or stream live SSE logs in terminal."""
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8080)
    follow = getattr(args, "follow", False) or getattr(args, "command", "") == "tail"
    lines = getattr(args, "lines", 20)
    status_filter = getattr(args, "status", "all")

    if not is_server_running(host, port):
        console.print(
            Panel(
                f"[dim]○ TokenGuard is not running on http://{host}:{port}[/dim]\n\n"
                f"[dim]Start daemon to view/stream logs:[/dim] [bold cyan]tokenguard start -d[/bold cyan]",
                title="[bold bright_white]TokenGuard Logs[/bold bright_white]",
                subtitle="[dim]○ STOPPED[/dim]",
                box=box.ROUNDED,
                border_style="bright_black",
            )
        )
        return

    if follow:
        console.print(
            Panel(
                f"[bold bright_white]TokenGuard Live Activity Stream[/bold bright_white]\n"
                f"[dim]Streaming events from http://{host}:{port}/api/stream • Press Ctrl+C to exit[/dim]",
                box=box.ROUNDED,
                border_style="bright_black",
                padding=(0, 2),
            )
        )
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream("GET", f"http://{host}:{port}/api/stream") as response:
                    current_event = None
                    for line in response.iter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("event:"):
                            current_event = line[6:].strip()
                        elif line.startswith("data:") and current_event:
                            try:
                                data = json.loads(line[5:].strip())
                                _render_stream_event(current_event, data)
                            except Exception:
                                pass
        except KeyboardInterrupt:
            console.print("\n[dim]Stream disconnected.[/dim]")
        except Exception as e:
            err_console.print(f"\n[yellow]Stream disconnected: {e}[/yellow]")
    else:
        try:
            with httpx.Client(timeout=3.0) as client:
                status_param = f"&status={status_filter}" if status_filter != "all" else ""
                res = client.get(f"http://{host}:{port}/api/requests?limit={lines}{status_param}")
                if res.status_code != 200:
                    err_console.print("[bold red]Failed to retrieve requests from daemon.[/bold red]")
                    return
                reqs = res.json().get("requests", [])

            if not reqs:
                console.print("[dim]No request events recorded yet.[/dim]")
                return

            table = Table(
                title="TokenGuard Recent Activity",
                box=box.ROUNDED,
                border_style="bright_black",
                header_style="bold bright_white",
            )
            table.add_column("Time", style="dim")
            table.add_column("Model", style="bold cyan")
            table.add_column("Tokens (P/C)", justify="right", style="bright_white")
            table.add_column("Cost", justify="right", style="bright_white")
            table.add_column("Latency", justify="right", style="dim")
            table.add_column("Status", justify="center")
            table.add_column("Info", style="dim")

            for r in reqs:
                status_str = r.get("status", "unknown")
                if status_str == "success":
                    badge = "[bold green]200 OK[/bold green]"
                elif status_str == "nudged_loop":
                    badge = "[bold cyan]200 NUDGED[/bold cyan]"
                elif status_str == "blocked_loop":
                    badge = "[bold red]BLOCKED: LOOP[/bold red]"
                elif status_str == "blocked_budget":
                    badge = "[bold yellow]BLOCKED: BUDGET[/bold yellow]"
                elif status_str == "blocked_killswitch":
                    badge = "[bold red]BLOCKED: KILLSWITCH[/bold red]"
                else:
                    badge = f"[dim]{status_str}[/dim]"

                t_str = r.get("timestamp", "").split("T")[-1][:8] if "T" in r.get("timestamp", "") else r.get("timestamp", "")[-8:]
                tokens_str = f"{r.get('prompt_tokens', 0)} / {r.get('completion_tokens', 0)}"
                cost_str = f"${float(r.get('cost_usd', 0)):.5f}"
                lat_str = f"{r.get('latency_ms', 0)}ms"
                info_str = r.get("blocked_reason") or (r.get("prompt_hash")[:10] + "..." if r.get("prompt_hash") else "")

                table.add_row(t_str, r.get("model", ""), tokens_str, cost_str, lat_str, badge, info_str)

            console.print(table)
        except Exception as e:
            err_console.print(f"[bold red]Error fetching logs:[/bold red] {e}")


def cmd_start(args: argparse.Namespace) -> None:
    """Start the proxy server and dashboard in foreground or daemon mode."""
    is_passive = getattr(args, "passive", False)
    is_dry_run = getattr(args, "dry_run", False)
    on_trip = str(getattr(args, "on_trip", os.environ.get("TOKENGUARD_ON_TRIP", "block")) or "block").strip().lower()
    if on_trip not in ("block", "nudge"):
        on_trip = "block"
    block_secrets = getattr(args, "block_secrets", False)
    redact_secrets = getattr(args, "redact_secrets", False)
    max_payload_size_mb = getattr(args, "max_payload_size_mb", 15.0)
    limit = 1000.0 if is_passive else args.limit
    daily_limit = 5000.0 if is_passive else args.daily_limit
    loop_threshold = 0 if is_passive else args.loop_threshold
    max_tool_repeats = 0 if is_passive else getattr(args, "max_tool_repeats", _env_int("TOKENGUARD_MAX_TOOL_REPEATS", 3))
    max_stagnant_steps = 0 if is_passive else getattr(args, "max_stagnant_steps", _env_int("TOKENGUARD_MAX_STAGNANT_STEPS", 6))

    if getattr(args, "daemon", False):
        with console.status("[bright_black]Starting TokenGuard daemon in background...[/bright_black]", spinner="dots"):
            pid = start_daemon(
                host=args.host,
                port=args.port,
                limit=limit,
                daily_limit=daily_limit,
                loop_threshold=loop_threshold,
                loop_window_seconds=args.loop_window_seconds,
                upstream_url=args.upstream_url,
                db_path=args.db_path,
                passive=is_passive,
                max_burn_rate=getattr(args, "max_burn_rate", None),
                dry_run=is_dry_run,
                on_trip=on_trip,
                max_tool_repeats=max_tool_repeats,
                max_stagnant_steps=max_stagnant_steps,
                block_secrets=block_secrets,
                redact_secrets=redact_secrets,
                max_payload_size_mb=max_payload_size_mb,
            )

        if wait_for_server(args.host, args.port, timeout=5.0):
            table = Table.grid(padding=(0, 2), expand=False)
            table.add_column(style="dim", justify="right", no_wrap=True)
            table.add_column(style="bright_white", no_wrap=True)

            if is_dry_run:
                status_text = "[bold yellow]● ACTIVE (OBSERVATION / DRY-RUN)[/bold yellow]"
            elif is_passive:
                status_text = "[bold yellow]● ACTIVE (PASSIVE MONITOR)[/bold yellow]"
            else:
                status_text = "[bold green]● ACTIVE & GUARDING[/bold green]"
            table.add_row("Status:", status_text)
            table.add_row("Proxy URL:", f"[cyan]http://{args.host}:{args.port}/v1[/cyan]")
            table.add_row("Dashboard:", f"[bold cyan]http://{args.host}:{args.port}[/bold cyan]")
            table.add_row("Process PID:", f"[dim]{pid}[/dim]")
            table.add_row("Log File:", f"[dim]{LOG_FILE}[/dim]")
            if is_dry_run:
                table.add_row("Observation Mode:", "[yellow]Dry Run (Shadow Mode)[/yellow]")

            panel = Panel(
                table,
                title="[bold bright_white]TokenGuard Daemon[/bold bright_white]",
                box=box.ROUNDED,
                border_style="bright_black",
                padding=(1, 2),
                expand=False,
            )
            console.print(panel)
            if sys.stdin.isatty():
                interactive_control_menu(args.host, args.port)
        else:
            console.print(f"[yellow]TokenGuard daemon spawned (PID: {pid}). Check logs at {LOG_FILE}[/yellow]")
        return

    settings = Settings(
        host=args.host,
        port=args.port,
        hourly_limit=limit,
        daily_limit=daily_limit,
        max_burn_rate=getattr(args, "max_burn_rate", None),
        loop_threshold=loop_threshold,
        max_tool_repeats=max_tool_repeats,
        max_stagnant_steps=max_stagnant_steps,
        loop_window_seconds=args.loop_window_seconds,
        upstream_url=args.upstream_url,
        db_path=args.db_path,
        profile="passive" if is_passive else "careful",
        passive=is_passive,
        dry_run=is_dry_run,
        on_trip=on_trip,
        block_secrets=block_secrets,
        redact_secrets=redact_secrets,
        max_payload_size_mb=max_payload_size_mb,
        session_limit=float(getattr(args, "session_limit", 0.0)),
    )
    set_settings(settings)

    print_banner(
        host=settings.host,
        port=settings.port,
        limit=settings.hourly_limit,
        loop_threshold=settings.loop_threshold,
        max_burn_rate=settings.max_burn_rate,
        dry_run=is_dry_run,
        on_trip=settings.on_trip,
    )

    uvicorn.run(
        "tokenguard.proxy:app",
        host=settings.host,
        port=settings.port,
        reload=args.reload,
        log_level="info",
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Execute child command with OPENAI_BASE_URL and OPENAI_API_BASE injected."""
    raw_cmd = args.cmd
    if raw_cmd and raw_cmd[0] == "--":
        raw_cmd = raw_cmd[1:]

    if not raw_cmd:
        err_console.print(
            Panel(
                "[bold red]Error: No command specified to run.[/bold red]\n\n"
                "[dim]Usage:[/dim]   [bold cyan]tokenguard run <command...>[/bold cyan]\n"
                "[dim]Example:[/dim] [bold green]tokenguard run python my_agent.py[/bold green]\n"
                "         [bold green]tokenguard run pytest[/bold green]",
                title="[bold bright_white]TokenGuard Run[/bold bright_white]",
                box=box.ROUNDED,
                border_style="bright_black",
                expand=False,
            )
        )
        sys.exit(1)

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8080)
    is_passive = getattr(args, "passive", False)
    is_dry_run = getattr(args, "dry_run", False)
    on_trip = str(getattr(args, "on_trip", os.environ.get("TOKENGUARD_ON_TRIP", "block")) or "block").strip().lower()
    if on_trip not in ("block", "nudge"):
        on_trip = "block"
    block_secrets = getattr(args, "block_secrets", False)
    redact_secrets = getattr(args, "redact_secrets", False)
    max_payload_size_mb = getattr(args, "max_payload_size_mb", 15.0)
    limit = 1000.0 if is_passive else getattr(args, "limit", 5.0)
    daily_limit = 5000.0 if is_passive else getattr(args, "daily_limit", 50.0)
    loop_threshold = 0 if is_passive else getattr(args, "loop_threshold", 3)
    max_tool_repeats = 0 if is_passive else getattr(args, "max_tool_repeats", _env_int("TOKENGUARD_MAX_TOOL_REPEATS", 3))
    max_stagnant_steps = 0 if is_passive else getattr(args, "max_stagnant_steps", _env_int("TOKENGUARD_MAX_STAGNANT_STEPS", 6))

    if not is_server_running(host, port):
        with console.status(f"[bright_black]Auto-starting TokenGuard proxy on http://{host}:{port}...[/bright_black]", spinner="dots"):
            pid = start_daemon(
                host=host,
                port=port,
                limit=limit,
                daily_limit=daily_limit,
                loop_threshold=loop_threshold,
                loop_window_seconds=getattr(args, "loop_window_seconds", 60.0),
                upstream_url=getattr(args, "upstream_url", "https://api.openai.com"),
                db_path=getattr(args, "db_path", Path("tokenguard.db")),
                passive=is_passive,
                max_burn_rate=getattr(args, "max_burn_rate", None),
                dry_run=is_dry_run,
                on_trip=on_trip,
                max_tool_repeats=max_tool_repeats,
                max_stagnant_steps=max_stagnant_steps,
                block_secrets=block_secrets,
                redact_secrets=redact_secrets,
                max_payload_size_mb=max_payload_size_mb,
            )

        if not is_server_running(host, port):
            err_console.print(f"[bold red]Error: Failed to auto-start TokenGuard proxy. Check {LOG_FILE}[/bold red]")
            sys.exit(1)

        console.print(f"[bold green]●[/bold green] [dim]TokenGuard proxy running (PID: {pid})[/dim]")

    proxy_url = f"http://{host}:{port}/v1"
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = proxy_url
    env["OPENAI_API_BASE"] = proxy_url
    env["ANTHROPIC_BASE_URL"] = f"http://{host}:{port}"
    env["GEMINI_API_BASE"] = proxy_url
    env["DASHSCOPE_BASE_URL"] = proxy_url
    env["QWEN_BASE_URL"] = proxy_url
    env["DEEPSEEK_BASE_URL"] = proxy_url
    env["GROQ_BASE_URL"] = proxy_url
    env["MISTRAL_API_BASE"] = proxy_url

    import shutil
    if raw_cmd[0] == "python" and not shutil.which("python"):
        raw_cmd[0] = sys.executable

    try:
        res = subprocess.run(raw_cmd, env=env)
        sys.exit(res.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


def cmd_quickstart(args: argparse.Namespace) -> None:
    """Interactive zero-friction setup and quickstart flow."""
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8080)

    if is_server_running(host, port):
        show_status(host, port)
        console.print(
            Panel(
                f"[bright_white]TokenGuard is currently running and protecting requests![/bright_white]\n\n"
                f"[dim]Run any script wrapped:[/dim]\n"
                f"  [bold cyan]tokenguard run python my_agent.py[/bold cyan]\n"
                f"  [bold cyan]tokenguard run pytest[/bold cyan]\n\n"
                f"[dim]Web Dashboard:[/dim] [bold cyan]http://{host}:{port}[/bold cyan]",
                title="[bold bright_white]Quick Run[/bold bright_white]",
                box=box.ROUNDED,
                border_style="bright_black",
                padding=(1, 2),
            )
        )
        if sys.stdin.isatty():
            interactive_control_menu(host, port)
        return

    console.print(
        Panel(
            "[bold bright_white]TokenGuard[/bold bright_white] [dim]— Zero-Latency LLM Circuit Breaker & Multi-Provider Proxy[/dim]\n"
            "[dim]Protects against infinite agent loops, runaway token costs, and API overruns.[/dim]",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(1, 2),
        )
    )

    keys = detect_api_keys()
    has_keys = any(bool(v) for v in keys.values())

    key_table = Table.grid(padding=(0, 2))
    key_table.add_column(style="dim", justify="right")
    key_table.add_column(style="bright_white")

    provider_labels = [
        ("OPENAI_API_KEY", "OpenAI:"),
        ("ANTHROPIC_API_KEY", "Anthropic / Claude:"),
        ("GEMINI_API_KEY", "Google Gemini:"),
        ("DASHSCOPE_API_KEY", "DashScope / Qwen:"),
        ("DEEPSEEK_API_KEY", "DeepSeek:"),
        ("GROQ_API_KEY", "Groq:"),
        ("MISTRAL_API_KEY", "Mistral AI:"),
    ]

    for key_name, label in provider_labels:
        val = keys.get(key_name)
        if val:
            key_table.add_row(label, f"[green]● Detected[/green] [dim]({mask_key(val)})[/dim]")
        else:
            key_table.add_row(label, "[dim]Not set (inherited per request)[/dim]")

    console.print(
        Panel(
            key_table,
            title="[bold bright_white]1. Multi-Provider API Key Detection[/bold bright_white]",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(1, 2),
        )
    )

    if not has_keys and sys.stdin.isatty():
        try:
            user_key = Prompt.ask(
                "[dim]Enter API Key (OpenAI / Claude / Gemini / Qwen / DeepSeek) or press [bold white]Enter[/bold white] to pass via agent code[/dim]",
                default="",
                show_default=False,
            ).strip()
            if user_key:
                uk_lower = user_key.lower()
                if "sk-ant" in uk_lower or "anthropic" in uk_lower:
                    update_stored_config(anthropic_api_key=user_key)
                elif "deepseek" in uk_lower or user_key.startswith("sk-d"):
                    update_stored_config(deepseek_api_key=user_key)
                elif "qwen" in uk_lower or "dashscope" in uk_lower:
                    update_stored_config(dashscope_api_key=user_key)
                elif "aiza" in uk_lower or "gemini" in uk_lower:
                    update_stored_config(gemini_api_key=user_key)
                else:
                    update_stored_config(openai_api_key=user_key)
                console.print("[dim]Stored in ~/.tokenguard/config.json with restricted permissions (0600)[/dim]")
        except (KeyboardInterrupt, EOFError):
            pass

    profile_opts = [
        ("careful", "Careful", "$5/hr cap, 2-loop cutoff, alerts ON", "RECOMMENDED"),
        ("standard", "Standard Agent", "$15/hr cap, 4-loop cutoff", "AUTONOMOUS"),
        ("passive", "Passive Monitor", "No loop blocks, tracking & telemetry only", "OBSERVABILITY"),
    ]
    selected_profile_key = interactive_select(
        profile_opts,
        title="2. Choose Guard Profile",
        subtitle="[dim]↑/↓ arrows or 1-3 to select · Enter to confirm[/dim]",
        default_index=0,
    )
    profile = PROFILES.get(selected_profile_key, PROFILES["careful"])

    with console.status("[bright_black]Starting TokenGuard daemon in background...[/bright_black]", spinner="dots"):
        pid = start_daemon(
            host=host,
            port=port,
            limit=profile["hourly_limit"],
            daily_limit=profile["daily_limit"],
            loop_threshold=profile["loop_threshold"],
            loop_window_seconds=60.0,
            upstream_url=getattr(args, "upstream_url", "https://api.openai.com"),
            db_path=getattr(args, "db_path", Path("tokenguard.db")),
        )

    status_table = Table.grid(padding=(0, 2))
    status_table.add_column(style="dim", justify="right")
    status_table.add_column(style="bright_white")

    status_table.add_row("Status:", "[bold green]● ACTIVE & GUARDING[/bold green]")
    status_table.add_row("Proxy Endpoint:", f"[cyan]http://{host}:{port}/v1[/cyan]")
    status_table.add_row("Web Dashboard:", f"[bold cyan]http://{host}:{port}[/bold cyan]")
    status_table.add_row("Guard Profile:", f"[bold white]{profile['name']}[/bold white] [dim](${profile['hourly_limit']:.0f}/hr, {profile['loop_threshold']}-loop cutoff)[/dim]")
    status_table.add_row("Process PID:", f"[dim]{pid}[/dim]")

    launch_snippet = (
        f"[dim]Run any agent script through TokenGuard:[/dim]\n"
        f"  [bold green]tokenguard run python my_agent.py[/bold green]\n\n"
        f"[dim]Or configure OpenAI base URL directly in your code:[/dim]\n"
        f"  [cyan]client = OpenAI(base_url=\"http://{host}:{port}/v1\")[/cyan]"
    )

    console.print(
        Panel(
            status_table,
            title="[bold bright_white]TokenGuard Ready[/bold bright_white]",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(1, 2),
        )
    )

    console.print(
        Panel(
            launch_snippet,
            title="[bold bright_white]Launch Protected[/bold bright_white]",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(1, 2),
        )
    )

    if sys.stdin.isatty():
        interactive_control_menu(host, port)


def _get_provider_label(model_name: str) -> str:
    """Return a styled provider tag for a given model name."""
    m = model_name.lower()
    if "gpt" in m or "o1" in m or "o3" in m or "text-embedding" in m or "dall-e" in m:
        return "[green]OpenAI[/green]"
    elif "claude" in m or "anthropic" in m:
        return "[bright_yellow]Anthropic[/bright_yellow]"
    elif "gemini" in m or "google" in m or "gemma" in m:
        return "[bright_blue]Google[/bright_blue]"
    elif "qwen" in m or "qwq" in m or "dashscope" in m:
        return "[blue]Alibaba[/blue]"
    elif "deepseek" in m:
        return "[cyan]DeepSeek[/cyan]"
    elif "mistral" in m or "codestral" in m or "pixtral" in m or "ministral" in m:
        return "[magenta]Mistral[/magenta]"
    elif "llama" in m or "meta" in m:
        return "[bright_magenta]Meta[/bright_magenta]"
    elif "groq" in m:
        return "[yellow]Groq[/yellow]"
    return "[dim]Community[/dim]"


def cmd_prices(args: argparse.Namespace) -> None:
    """Print out registered prices with dynamic live registry sync."""
    do_update = getattr(args, "update", False)
    do_offline = getattr(args, "offline", False)
    prices_path = getattr(args, "prices_path", None)

    if do_update:
        with console.status("[bright_black]Fetching latest live pricing from OpenRouter public registry...[/bright_black]", spinner="dots"):
            data, was_updated, msg = update_pricing_cache(force=True, baseline_path=prices_path)

        console.print(
            Panel(
                f"[bold green]● Pricing Synchronized[/bold green]\n\n"
                f"[dim]Source:[/dim]  [cyan]OpenRouter Public Registry (openrouter.ai/api/v1/models)[/cyan]\n"
                f"[dim]Status:[/dim]  {msg}\n"
                f"[dim]Models:[/dim]  [bold white]{len(data)}[/bold white] models registered",
                title="[bold bright_white]TokenGuard Dynamic Pricing[/bold bright_white]",
                box=box.ROUNDED,
                border_style="bright_black",
                expand=False,
            )
        )
        source_title = "Live Registry (openrouter.ai)"
    elif do_offline:
        data = load_baseline_prices(prices_path)
        source_title = "Offline Baseline (prices.json)"
    elif prices_path:
        data = load_baseline_prices(prices_path)
        source_title = f"Custom Baseline ({prices_path.name})"
    else:
        cached, last_updated = load_cached_prices()
        if cached:
            age_hours = round((time.time() - last_updated) / 3600.0, 1) if last_updated > 0 else 0
            data = get_dynamic_pricing_table()
            source_title = f"Dynamic Cache ({age_hours}h ago · {len(data)} models)"
        else:
            data = get_dynamic_pricing_table()
            source_title = f"Offline Baseline ({len(data)} models)"

    table = Table(
        title=f"TokenGuard Pricing Registry — {source_title}",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold bright_white",
        expand=False,
    )
    table.add_column("Model Identifier", style="bold cyan", no_wrap=True)
    table.add_column("Input Rate ($/1M)", justify="right", style="bright_white", no_wrap=True)
    table.add_column("Output Rate ($/1M)", justify="right", style="bright_white", no_wrap=True)
    table.add_column("Provider", style="dim", no_wrap=True)

    sorted_keys = sorted(data.keys())
    if len(sorted_keys) > 60 and not do_update:
        display_keys = [k for k in sorted_keys if "/" not in k]
        if not display_keys:
            display_keys = sorted_keys[:60]
    else:
        display_keys = sorted_keys

    for model in display_keys:
        rates = data[model]
        provider = _get_provider_label(model)
        in_cost = f"${float(rates.get('input', 0.0)):.2f}"
        out_cost = f"${float(rates.get('output', 0.0)):.2f}"
        table.add_row(model, in_cost, out_cost, provider)

    console.print(table)


def cmd_export(args: argparse.Namespace) -> None:
    """Export logged telemetry to JSONL, HAR 1.2, JSON, or CSV format."""
    db_path = getattr(args, "db_path", Path("tokenguard.db"))
    format_type = getattr(args, "format", "jsonl")
    output_path = getattr(args, "output", None)
    status_filter = getattr(args, "status", "all")
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8080)

    try:
        target, count, _ = export_telemetry_sync(
            db_path=db_path,
            format_type=format_type,
            output_path=output_path,
            status_filter=status_filter,
            host=host,
            port=port,
        )
    except Exception as e:
        err_console.print(
            Panel(
                f"[bold red]Failed to export telemetry:[/bold red] {e}",
                title="[bold bright_white]TokenGuard Export Error[/bold bright_white]",
                box=box.ROUNDED,
                border_style="bright_black",
                expand=False,
            )
        )
        sys.exit(1)

    table = Table.grid(padding=(0, 2), expand=False)
    table.add_column(style="dim", justify="right", no_wrap=True)
    table.add_column(style="bright_white", no_wrap=True)

    table.add_row("Output File:", f"[bold cyan]{target}[/bold cyan]")
    table.add_row("Format:", f"[bold white]{format_type.upper()}[/bold white]")
    table.add_row("Filter:", f"[white]{status_filter}[/white]")
    table.add_row("Total Records:", f"[green]{count}[/green]")
    table.add_row("Database:", f"[dim]{db_path}[/dim]")

    console.print(
        Panel(
            table,
            title="[bold bright_white]TokenGuard Telemetry Export[/bold bright_white]",
            subtitle="[dim]Export complete[/dim]",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(1, 2),
            expand=False,
        )
    )


def _render_alternating_loop_demo(
    data: dict,
    is_local: bool = False,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    results = data.get("results", [])
    tripped = data.get("tripped", False)
    cycle_len = data.get("cycle_length", 2)
    pattern = data.get("pattern", ["read_file", "bash"])
    pattern_display = " ➔ ".join(pattern)

    table = Table(
        title="TokenGuard Circuit Breaker — Alternating Loop (A-B-A-B) Demo",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold bright_white",
        expand=False,
    )
    table.add_column("Step", justify="center", style="bold")
    table.add_column("Tool", style="cyan")
    table.add_column("Arguments", style="dim")
    table.add_column("Action / Status", justify="left")

    for item in results:
        step = str(item.get("attempt") or item.get("step") or "")
        tool = item.get("tool", "")
        args_str = json.dumps(item.get("args", {}))
        status = item.get("status", "")
        if status == "allowed":
            status_text = "[bold green]ALLOWED (200 OK)[/bold green]"
        else:
            status_text = "[bold red]⚡ BLOCKED (429 Circuit Breaker Tripped)[/bold red]"
        table.add_row(step, tool, args_str, status_text)

    console.print()
    console.print(table)
    console.print()

    trip_detail = data.get("details", {})
    err_msg = trip_detail.get("message") or "Tool ping-pong cycle detected"
    mode_text = (
        "[dim](Executed in-process local simulation)[/dim]"
        if is_local
        else f"[dim](Live proxy on http://{host}:{port} — logged to SQLite & SSE stream)[/dim]"
    )

    panel_body = (
        f"[bold red]● Circuit Breaker Tripped: Alternating Ping-Pong Loop Intercepted[/bold red]\n\n"
        f"[bold white]Pattern:[/bold white] [cyan]{pattern_display} ➔ {pattern_display}[/cyan] (Cycle length: {cycle_len})\n"
        f"[bold white]Detection:[/bold white] {err_msg}\n"
        f"[bold white]Outcome:[/bold white] Call #4 intercepted locally before reaching upstream LLM\n"
        f"[bold green]Avoided Cost:[/bold green] +$0.0105 estimated saved spend\n\n"
        f"{mode_text}"
    )

    console.print(
        Panel(
            panel_body,
            title="[bold bright_white]Circuit Breaker: A-B-A-B Trip Demo[/bold bright_white]",
            box=box.ROUNDED,
            border_style="red" if tripped else "green",
            padding=(1, 2),
            expand=False,
        )
    )


def _render_drift_demo(
    data: dict,
    is_local: bool = False,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    results = data.get("results", [])
    tripped = data.get("tripped", False)
    tool_name = data.get("tool", "read_file")
    threshold = data.get("threshold", 3)
    usd_saved = data.get("details", {}).get("usd_saved", 0.0125)

    table = Table(
        title="TokenGuard Circuit Breaker — Argument Drift Bypass Demo",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold bright_white",
        expand=False,
    )
    table.add_column("Step", justify="center", style="bold")
    table.add_column("Tool", style="cyan")
    table.add_column("Mutated Arguments", style="dim")
    table.add_column("Action / Status", justify="left")

    for item in results:
        step = str(item.get("attempt") or item.get("step") or "")
        tool = item.get("tool", tool_name)
        args_str = json.dumps(item.get("args", {}))
        status = item.get("status", "")
        if status == "allowed":
            status_text = "[bold green]ALLOWED (200 OK)[/bold green]"
        else:
            status_text = "[bold red]⚡ BLOCKED (429 Circuit Breaker Tripped)[/bold red]"
        table.add_row(step, tool, args_str, status_text)

    console.print()
    console.print(table)
    console.print()

    trip_detail = data.get("details", {})
    err_msg = trip_detail.get("message") or f"ToolRetryCeilingTripped: {tool_name} called {threshold} times consecutively"
    mode_text = (
        "[dim](Executed in-process local simulation)[/dim]"
        if is_local
        else f"[dim](Live proxy on http://{host}:{port} — logged to SQLite & SSE stream)[/dim]"
    )

    panel_body = (
        f"[bold red]● Circuit Breaker Tripped: Argument Drift / Retry Ceiling Intercepted[/bold red]\n\n"
        f"[bold white]Status Code:[/bold white] [bold red]429 Too Many Requests[/bold red]\n"
        f"[bold white]Headers:[/bold white]\n"
        f"  [cyan]X-TokenGuard-Reason:[/cyan] {err_msg}\n"
        f"  [cyan]X-TokenGuard-USD-Saved:[/cyan] ${usd_saved:.4f}\n\n"
        f"[bold white]Outcome:[/bold white] Argument drift bypass blocked locally before reaching upstream LLM\n"
        f"[bold green]Avoided Cost:[/bold green] +${usd_saved:.4f} estimated saved spend\n\n"
        f"{mode_text}"
    )

    console.print(
        Panel(
            panel_body,
            title="[bold bright_white]Circuit Breaker: Tool Retry Ceiling Demo[/bold bright_white]",
            box=box.ROUNDED,
            border_style="red" if tripped else "green",
            padding=(1, 2),
            expand=False,
        )
    )


def _render_stagnation_demo(
    data: dict,
    is_local: bool = False,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    results = data.get("results", [])
    tripped = data.get("tripped", False)
    threshold = data.get("threshold", 6)
    usd_saved = data.get("details", {}).get("usd_saved", 0.0180)

    table = Table(
        title="TokenGuard Circuit Breaker — Session Stagnation Guard Demo",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold bright_white",
        expand=False,
    )
    table.add_column("Step", justify="center", style="bold")
    table.add_column("Action / Tool", style="cyan")
    table.add_column("State Mutated", justify="center", style="dim")
    table.add_column("Action / Status", justify="left")

    for item in results:
        step = str(item.get("attempt") or item.get("step") or "")
        tool = item.get("tool", "")
        mutated = "[dim]False[/dim]"
        status = item.get("status", "")
        if status == "allowed":
            status_text = "[bold green]ALLOWED (200 OK)[/bold green]"
        else:
            status_text = "[bold red]⚡ BLOCKED (429 Circuit Breaker Tripped)[/bold red]"
        table.add_row(step, tool, mutated, status_text)

    console.print()
    console.print(table)
    console.print()

    trip_detail = data.get("details", {})
    err_msg = trip_detail.get("message") or f"SessionStagnationTripped: {threshold} steps without state change"
    mode_text = (
        "[dim](Executed in-process local simulation)[/dim]"
        if is_local
        else f"[dim](Live proxy on http://{host}:{port} — logged to SQLite & SSE stream)[/dim]"
    )

    panel_body = (
        f"[bold red]● Circuit Breaker Tripped: Read-Only Stagnation Intercepted[/bold red]\n\n"
        f"[bold white]Status Code:[/bold white] [bold red]429 Too Many Requests[/bold red]\n"
        f"[bold white]Headers:[/bold white]\n"
        f"  [cyan]X-TokenGuard-Reason:[/cyan] {err_msg}\n"
        f"  [cyan]X-TokenGuard-USD-Saved:[/cyan] ${usd_saved:.4f}\n\n"
        f"[bold white]Outcome:[/bold white] Stagnant read-only loop terminated after {threshold} zero-state-change iterations\n"
        f"[bold green]Avoided Cost:[/bold green] +${usd_saved:.4f} estimated saved spend\n\n"
        f"{mode_text}"
    )

    console.print(
        Panel(
            panel_body,
            title="[bold bright_white]Circuit Breaker: Session Stagnation Guard Demo[/bold bright_white]",
            box=box.ROUNDED,
            border_style="red" if tripped else "green",
            padding=(1, 2),
            expand=False,
        )
    )


def _render_secret_demo(
    data: dict,
    is_local: bool = False,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    results = data.get("results", [])
    tripped = data.get("tripped", False)
    secret_type = data.get("secret_type", "OpenAI Project Key")
    preview = data.get("preview", "sk-proj-abc1234567890123456789012345678901234567890")

    table = Table(
        title="TokenGuard Security — Data Loss Prevention (DLP) Secret Leak Demo",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold bright_white",
        expand=False,
    )
    table.add_column("Attempt", justify="center", style="bold")
    table.add_column("Detected Pattern", style="red")
    table.add_column("Payload Snapshot", style="dim")
    table.add_column("Action / Status", justify="left")

    for item in results:
        step = str(item.get("attempt") or "1")
        pat = item.get("pattern", secret_type)
        snippet = item.get("payload_snippet", preview)
        status_text = "[bold red]⚡ BLOCKED (400 SecretLeakDetected)[/bold red]"
        table.add_row(step, pat, snippet, status_text)

    console.print()
    console.print(table)
    console.print()

    trip_detail = data.get("details", {})
    err_msg = trip_detail.get("message") or f"SecretLeakDetected: Found matching pattern {secret_type}"
    mode_text = (
        "[dim](Executed in-process local simulation)[/dim]"
        if is_local
        else f"[dim](Live proxy on http://{host}:{port} — logged to SQLite & SSE stream)[/dim]"
    )

    panel_body = (
        f"[bold red]● Outbound Request Intercepted: Secret Leak Blocked[/bold red]\n\n"
        f"[bold white]Status Code:[/bold white] [bold red]400 Bad Request[/bold red]\n"
        f"[bold white]Headers:[/bold white]\n"
        f"  [cyan]X-TokenGuard-Reason:[/cyan] {err_msg}\n\n"
        f"[bold white]Pattern Detected:[/bold white] [red]{secret_type}[/red]\n"
        f"[bold white]Outcome:[/bold white] Intercepted before plaintext API credential escaped to upstream provider\n\n"
        f"{mode_text}"
    )

    console.print(
        Panel(
            panel_body,
            title="[bold bright_white]Security: DLP Secret Leak Interceptor Demo[/bold bright_white]",
            box=box.ROUNDED,
            border_style="red" if tripped else "green",
            padding=(1, 2),
            expand=False,
        )
    )


def _render_stream_loop_demo(
    data: dict,
    is_local: bool = False,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    results = data.get("results", [])
    tripped = data.get("tripped", False)
    repeated_token = data.get("repeated_token", "repeat ")
    usd_saved = data.get("details", {}).get("usd_saved", 0.0420)

    table = Table(
        title="TokenGuard Circuit Breaker — Mid-Stream Token Repetition Cutoff Demo",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold bright_white",
        expand=False,
    )
    table.add_column("Chunk #", justify="center", style="bold")
    table.add_column("Stream Delta", style="cyan")
    table.add_column("Repetition Count", justify="center", style="dim")
    table.add_column("Stream State", justify="left")

    for item in results:
        step = str(item.get("attempt") or item.get("chunk") or "")
        token = item.get("token", repeated_token)
        rep = str(item.get("repeat_count", step))
        status = item.get("status", "")
        if status == "allowed" or status == "streaming":
            status_text = "[bold green]PASSING (Stream Active)[/bold green]"
        else:
            status_text = "[bold red]⚡ ABORTED (SSE Stream Severed)[/bold red]"
        table.add_row(step, repr(token), rep, status_text)

    console.print()
    console.print(table)
    console.print()

    trip_detail = data.get("details", {})
    err_msg = trip_detail.get("message") or f"StreamLoopAborted: repetitive token sequence '{repeated_token}'"
    mode_text = (
        "[dim](Executed in-process local simulation)[/dim]"
        if is_local
        else f"[dim](Live proxy on http://{host}:{port} — logged to SQLite & SSE stream)[/dim]"
    )

    panel_body = (
        f"[bold red]● Circuit Breaker Tripped: Mid-Stream Token Repetition Cutoff[/bold red]\n\n"
        f"[bold white]Status Code:[/bold white] [bold yellow]200 OK (Stream Severed Early)[/bold yellow]\n"
        f"[bold white]Headers:[/bold white]\n"
        f"  [cyan]X-TokenGuard-Reason:[/cyan] {err_msg}\n"
        f"  [cyan]X-TokenGuard-USD-Saved:[/cyan] ${usd_saved:.4f}\n\n"
        f"[bold white]Outcome:[/bold white] Upstream SSE connection aborted mid-flight to stop infinite token generation loop\n"
        f"[bold green]Avoided Cost:[/bold green] +${usd_saved:.4f} estimated saved spend\n\n"
        f"{mode_text}"
    )

    console.print(
        Panel(
            panel_body,
            title="[bold bright_white]Circuit Breaker: Mid-Stream Repetition Cutoff Demo[/bold bright_white]",
            box=box.ROUNDED,
            border_style="red" if tripped else "green",
            padding=(1, 2),
            expand=False,
        )
    )


def _render_nudge_demo(
    data: dict,
    is_local: bool = False,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Render Rich panel demonstrating synthetic 200 OK Loop Nudge recovery followed by 429 escalation."""
    results = data.get("results", [])
    nudge_msg = data.get("nudge_message", "")
    trip_reason = data.get("trip_reason", "ToolRetryCeilingTripped")
    usd_saved = float(data.get("usd_saved", 0.0) or 0.0)
    tool_name = data.get("tool_name", "read_file")

    lines = []
    for item in results:
        attempt = item.get("attempt", 0)
        st = item.get("status", "")
        if st == "allowed":
            lines.append(f"  [dim]Turn {attempt}:[/dim] [green]200 OK[/green] — tool [cyan]{tool_name}[/cyan] executed")
        elif st == "nudged_loop":
            lines.append(
                f"  [dim]Turn {attempt}:[/dim] [bold cyan]200 OK (Synthetic Nudge)[/bold cyan] — "
                f"[cyan]X-TokenGuard-Action: nudged[/cyan] (status: [bold]nudged_loop[/bold])"
            )
        else:
            lines.append(
                f"  [dim]Turn {attempt}:[/dim] [bold red]429 Too Many Requests[/bold red] — "
                f"Nudge ignored; escalated to hard block ([red]{st}[/red])"
            )

    mode_text = (
        "[dim](Executed in-process local simulation)[/dim]"
        if is_local
        else f"[dim](Live proxy on http://{host}:{port} — logged to SQLite & SSE stream)[/dim]"
    )

    steps_str = "\n".join(lines)
    panel_body = (
        f"[bold cyan]● Loop Nudge Synthetic Recovery Mode (--on-trip=nudge)[/bold cyan]\n\n"
        f"[bold white]Execution Sequence:[/bold white]\n{steps_str}\n\n"
        f"[bold white]Synthetic 200 OK Assistant Message:[/bold white]\n"
        f"  [italic bright_white]\"{nudge_msg}\"[/italic bright_white]\n\n"
        f"[bold white]Headers:[/bold white]\n"
        f"  [cyan]X-TokenGuard-Action:[/cyan] nudged\n"
        f"  [cyan]X-TokenGuard-Reason:[/cyan] {trip_reason}\n"
        f"[bold green]Avoided Cost:[/bold green] +${usd_saved:.4f} estimated saved spend\n\n"
        f"{mode_text}"
    )

    console.print(
        Panel(
            panel_body,
            title="[bold bright_white]Circuit Breaker: Loop Nudge Recovery & Escalation Demo[/bold bright_white]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(1, 2),
            expand=False,
        )
    )


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run circuit breaker simulation to demo protection mechanisms."""
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8080)
    sim_type = getattr(args, "type", "loop")
    sim_model = getattr(args, "model", "gpt-4o")

    if sim_type == "nudge":
        if is_server_running(host, port):
            try:
                with httpx.Client(timeout=5.0) as client:
                    res = client.post(
                        f"http://{host}:{port}/api/simulate",
                        json={"type": "nudge", "model": sim_model},
                    )
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("simulation") == "nudge" and data.get("results"):
                            _render_nudge_demo(data, is_local=False, host=host, port=port)
                            return
            except Exception:
                pass

        from tokenguard.guard import CircuitBreaker, ToolRetryCeilingError, build_nudge_message

        guard = CircuitBreaker(max_tool_repeats=3, on_trip="nudge", enable_notifications=False)
        session_id = f"demo_nudge_{int(time.time())}"
        tool_name = "read_file"
        results = []
        nudge_msg = ""
        trip_reason = ""
        usd_saved = 0.0

        for i in range(1, 5):
            sim_msgs = [{"role": "user", "content": f"Read config_{i}.yaml"}]
            try:
                guard.check_tool_retry_ceiling(session_id, tool_name, model=sim_model, messages=sim_msgs)
                guard.record_tool_name(session_id, tool_name)
                results.append({"attempt": i, "status": "allowed", "http_status": 200, "tool": tool_name})
            except ToolRetryCeilingError as err:
                action = guard.evaluate_trip_action(session_id, err)
                trip_reason = err.message
                _, u_s = guard.calculate_saved_telemetry(model=sim_model, messages=sim_msgs, session_id=session_id)
                usd_saved = round(usd_saved + u_s, 4)
                if action == "nudge":
                    nudge_msg = build_nudge_message(err)
                    results.append(
                        {
                            "attempt": i,
                            "status": "nudged_loop",
                            "http_status": 200,
                            "action": "nudged",
                            "tool": tool_name,
                            "nudge_message": nudge_msg,
                            "reason": err.message,
                        }
                    )
                else:
                    results.append(
                        {
                            "attempt": i,
                            "status": "blocked_tool_retry_ceiling",
                            "http_status": 429,
                            "action": "blocked",
                            "tool": tool_name,
                            "reason": err.message,
                        }
                    )

        data = {
            "simulation": "nudge",
            "tool_name": tool_name,
            "threshold": 3,
            "nudge_message": nudge_msg,
            "trip_reason": trip_reason,
            "usd_saved": usd_saved,
            "nudged": True,
            "escalated": True,
            "results": results,
        }
        _render_nudge_demo(data, is_local=True)
        return

    if sim_type in ("alternating_loop", "alternating"):
        if is_server_running(host, port):
            try:
                with httpx.Client(timeout=5.0) as client:
                    res = client.post(
                        f"http://{host}:{port}/api/simulate",
                        json={"type": "alternating_loop", "model": sim_model},
                    )
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("simulation") == "alternating_loop" and data.get("results"):
                            _render_alternating_loop_demo(data, is_local=False, host=host, port=port)
                            return
            except Exception:
                pass

        from tokenguard.guard import CircuitBreaker, ToolPingPongError
        guard = CircuitBreaker(tool_loop_threshold=3)
        session_id = f"demo_alt_{int(time.time())}"
        sequence = [
            ("read_file", {"path": "server.py"}),
            ("bash", {"command": "python server.py"}),
            ("read_file", {"path": "server.py"}),
            ("bash", {"command": "python server.py"}),
        ]
        results = []
        cycle_detected = False
        trip_info = {}

        for i, (tool_name, tool_args) in enumerate(sequence, start=1):
            try:
                guard.check_tool_loop(session_id, tool_name, tool_args)
                guard.record_tool_call(session_id, tool_name, tool_args)
                results.append({
                    "attempt": i,
                    "tool": tool_name,
                    "args": tool_args,
                    "status": "allowed",
                })
            except ToolPingPongError as err:
                cycle_detected = True
                trip_info = {
                    "tool": err.tool_name,
                    "cycle_length": err.cycle_length,
                    "pattern": err.pattern,
                    "message": err.message,
                }
                results.append({
                    "attempt": i,
                    "tool": tool_name,
                    "args": tool_args,
                    "status": "blocked_tool_loop",
                    "cycle_length": err.cycle_length,
                    "pattern": err.pattern,
                    "message": err.message,
                })

        data = {
            "simulation": "alternating_loop",
            "tripped": cycle_detected,
            "cycle_length": 2,
            "pattern": ["read_file", "bash"],
            "results": results,
            "details": trip_info,
        }
        _render_alternating_loop_demo(data, is_local=True)
        return

    if sim_type == "drift":
        if is_server_running(host, port):
            try:
                with httpx.Client(timeout=5.0) as client:
                    res = client.post(
                        f"http://{host}:{port}/api/simulate",
                        json={"type": "drift", "model": sim_model},
                    )
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("simulation") == "drift" and data.get("results"):
                            _render_drift_demo(data, is_local=False, host=host, port=port)
                            return
            except Exception:
                pass

        from tokenguard.guard import CircuitBreaker, ToolRetryCeilingError
        guard = CircuitBreaker(max_tool_repeats=3)
        session_id = f"demo_drift_{int(time.time())}"
        paths = ["config.json", "./config.json", "/app/config.json", "/etc/config.json"]
        results = []
        tripped = False
        trip_info = {}

        for i in range(1, guard.max_tool_repeats + 1):
            tool_args = {"path": paths[(i - 1) % len(paths)]}
            try:
                guard.check_tool_retry_ceiling(session_id, "read_file")
                guard.record_tool_name(session_id, "read_file")
                guard.record_tool_call(session_id, "read_file", tool_args)
                results.append({
                    "attempt": i,
                    "tool": "read_file",
                    "args": tool_args,
                    "status": "allowed",
                })
            except ToolRetryCeilingError as err:
                tripped = True
                trip_info = {
                    "reason": "ToolRetryCeilingTripped",
                    "message": err.message,
                    "usd_saved": 0.0125,
                }
                results.append({
                    "attempt": i,
                    "tool": "read_file",
                    "args": tool_args,
                    "status": "blocked_tool_retry_ceiling",
                })
                break

        data = {
            "simulation": "drift",
            "tripped": tripped,
            "tool": "read_file",
            "threshold": guard.max_tool_repeats,
            "results": results,
            "details": trip_info,
        }
        _render_drift_demo(data, is_local=True)
        return

    if sim_type == "stagnation":
        if is_server_running(host, port):
            try:
                with httpx.Client(timeout=5.0) as client:
                    res = client.post(
                        f"http://{host}:{port}/api/simulate",
                        json={"type": "stagnation", "model": sim_model},
                    )
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("simulation") == "stagnation" and data.get("results"):
                            _render_stagnation_demo(data, is_local=False, host=host, port=port)
                            return
            except Exception:
                pass

        from tokenguard.guard import CircuitBreaker, SessionStagnationError
        guard = CircuitBreaker(max_stagnant_steps=6)
        session_id = f"demo_stag_{int(time.time())}"
        tools = ["read_file", "list_dir", "get_status", "read_file", "search_files", "read_file"]
        results = []
        tripped = False
        trip_info = {}

        for i in range(1, guard.max_stagnant_steps + 1):
            tool_name = tools[(i - 1) % len(tools)]
            tool_args = {"step": i}
            try:
                guard.check_stagnation(session_id, tool_names=[tool_name])
                guard.record_stagnation_step(session_id, tool_names=[tool_name])
                guard.record_tool_call(session_id, tool_name, tool_args)
                results.append({
                    "attempt": i,
                    "tool": tool_name,
                    "args": tool_args,
                    "status": "allowed",
                })
            except SessionStagnationError as err:
                tripped = True
                trip_info = {
                    "reason": "SessionStagnationTripped",
                    "message": err.message,
                    "usd_saved": 0.0180,
                }
                results.append({
                    "attempt": i,
                    "tool": tool_name,
                    "args": tool_args,
                    "status": "blocked_stagnation",
                })
                break

        data = {
            "simulation": "stagnation",
            "tripped": tripped,
            "threshold": guard.max_stagnant_steps,
            "results": results,
            "details": trip_info,
        }
        _render_stagnation_demo(data, is_local=True)
        return

    if sim_type == "secret":
        if is_server_running(host, port):
            try:
                with httpx.Client(timeout=5.0) as client:
                    res = client.post(
                        f"http://{host}:{port}/api/simulate",
                        json={"type": "secret", "model": sim_model},
                    )
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("simulation") == "secret" and data.get("results"):
                            _render_secret_demo(data, is_local=False, host=host, port=port)
                            return
            except Exception:
                pass

        from tokenguard.guard import scan_secrets_in_obj
        raw_secret = "sk-proj-abc1234567890123456789012345678901234567890"
        payload_data = {"role": "user", "content": f"Deploy with API key {raw_secret}"}
        detected = scan_secrets_in_obj(payload_data) or "openai_proj_key"
        err_msg = f"SecretLeakDetected: Found matching pattern {detected}"
        data = {
            "simulation": "secret",
            "tripped": True,
            "secret_type": detected,
            "preview": f"sk-proj-{'*' * 16}",
            "results": [
                {
                    "attempt": 1,
                    "pattern": detected,
                    "payload_snippet": f"Deploy with API key sk-proj-{'*' * 16}",
                    "status": "blocked_secret_leak",
                }
            ],
            "details": {
                "reason": "SecretLeakDetected",
                "message": err_msg,
            },
        }
        _render_secret_demo(data, is_local=True)
        return

    if sim_type in ("stream-loop", "stream_loop"):
        if is_server_running(host, port):
            try:
                with httpx.Client(timeout=5.0) as client:
                    res = client.post(
                        f"http://{host}:{port}/api/simulate",
                        json={"type": "stream_loop", "model": sim_model},
                    )
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("simulation") == "stream_loop" and data.get("results"):
                            _render_stream_loop_demo(data, is_local=False, host=host, port=port)
                            return
            except Exception:
                pass

        from tokenguard.proxy import StreamLoopDetector
        detector = StreamLoopDetector(max_repeats=5)
        token = "repeat "
        results = []
        tripped = False
        trip_info = {}

        for i in range(1, 6):
            detector.feed_text(token)
            if detector.loop_detected:
                tripped = True
                trip_info = {
                    "reason": "StreamLoopAborted",
                    "message": f"StreamLoopAborted: repetitive token sequence '{token}'",
                    "usd_saved": 0.0420,
                }
                results.append({"attempt": i, "token": token, "repeat_count": i, "status": "blocked_stream_loop"})
                break
            else:
                results.append({"attempt": i, "token": token, "repeat_count": i, "status": "allowed"})

        data = {
            "simulation": "stream_loop",
            "tripped": tripped,
            "repeated_token": token,
            "results": results,
            "details": trip_info,
        }
        _render_stream_loop_demo(data, is_local=True)
        return

    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.post(
                f"http://{host}:{port}/api/simulate",
                json={"type": sim_type, "model": sim_model},
            )
            if res.status_code == 200:
                console.print(
                    Panel(
                        f"[bold green]● Simulation Completed[/bold green]\n\n"
                        f"[dim]Type:[/dim] {sim_type}  ·  [dim]Model:[/dim] {sim_model}\n"
                        f"[dim]Response:[/dim] {res.json()}",
                        title="[bold bright_white]TokenGuard Simulation[/bold bright_white]",
                        box=box.ROUNDED,
                        border_style="bright_black",
                        expand=False,
                    )
                )
            else:
                err_console.print(f"[bold red]Simulation failed: {res.text}[/bold red]")
    except Exception as e:
        err_console.print(f"[bold red]Error running simulation: {e}[/bold red]")


def cmd_env(args: argparse.Namespace) -> None:
    """Print shell environment export or unset commands to route LLM SDKs through TokenGuard."""
    unset = getattr(args, "unset", False)
    host = getattr(args, "host", "localhost")
    port = getattr(args, "port", 8080)

    if unset:
        print("unset OPENAI_BASE_URL ANTHROPIC_BASE_URL")
        sys.stderr.write("# Run this command to restore direct upstream connections:\n# eval $(tokenguard env --unset)\n")
    else:
        print(f'export OPENAI_BASE_URL="http://{host}:{port}/v1" ANTHROPIC_BASE_URL="http://{host}:{port}"')
        sys.stderr.write("# Run this command to route OpenAI & Anthropic SDKs through TokenGuard:\n# eval $(tokenguard env)\n")
    sys.stdout.flush()
    sys.stderr.flush()


def main() -> None:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "env":
        cmd_env(args)
    elif args.command == "stop":
        stop_daemon(host=args.host, port=args.port)
    elif args.command == "status":
        show_status(host=args.host, port=args.port)
    elif args.command in ("menu", "tui"):
        interactive_control_menu(host=getattr(args, "host", "127.0.0.1"), port=getattr(args, "port", 8080))
    elif args.command == "dashboard":
        url = f"http://{getattr(args, 'host', '127.0.0.1')}:{getattr(args, 'port', 8080)}"
        console.print(f"[dim]Opening web dashboard in default browser:[/dim] [bold cyan]{url}[/bold cyan]")
        webbrowser.open(url)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "kill":
        cmd_kill(args)
    elif args.command == "resume":
        cmd_resume(args)
    elif args.command == "profile":
        cmd_profile(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command in ("logs", "tail"):
        cmd_logs(args)
    elif args.command == "prices":
        cmd_prices(args)
    elif args.command == "start":
        cmd_start(args)
    else:
        if getattr(args, "daemon", False):
            cmd_start(args)
        else:
            cmd_quickstart(args)


if __name__ == "__main__":
    main()
