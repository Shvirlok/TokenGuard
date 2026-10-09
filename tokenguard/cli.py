"""CLI entrypoint and background daemon manager for TokenGuard."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from tokenguard.config import (
    PROFILES,
    Settings,
    detect_api_keys,
    get_settings,
    load_stored_config,
    mask_key,
    save_stored_config,
    set_settings,
    update_stored_config,
)

PID_DIR = Path.home() / ".tokenguard"
PID_FILE = PID_DIR / "tokenguard.pid"
LOG_FILE = PID_DIR / "tokenguard.log"

console = Console()
err_console = Console(stderr=True)


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="tokenguard",
        description="🛡️ TokenGuard: Lightweight open-source local proxy and circuit breaker for LLM calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # 1. Start command
    start_parser = subparsers.add_parser("start", help="Start the TokenGuard proxy server and dashboard")
    _add_server_args(start_parser)
    start_parser.add_argument(
        "-d",
        "--daemon",
        action="store_true",
        help="Run TokenGuard in the background as a daemon process",
    )

    # 2. Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop any background TokenGuard daemon process")
    stop_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host address (default: 127.0.0.1)")
    stop_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    # 3. Status command
    status_parser = subparsers.add_parser("status", help="Check status of the TokenGuard daemon & proxy")
    status_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host address (default: 127.0.0.1)")
    status_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    # 4. Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a command with OPENAI_BASE_URL and OPENAI_API_BASE automatically injected",
    )
    run_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host address (default: 127.0.0.1)")
    run_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")
    run_parser.add_argument("--limit", type=float, default=5.0, help="Sliding hourly budget limit in USD (default: 5.0)")
    run_parser.add_argument("--daily-limit", type=float, default=50.0, help="Sliding daily budget limit in USD (default: 50.0)")
    run_parser.add_argument(
        "--max-repeats",
        "--loop-threshold",
        dest="loop_threshold",
        type=int,
        default=3,
        help="Maximum duplicate consecutive requests before tripping loop guard (default: 3, set 0 to disable)",
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
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command and arguments to execute (e.g. python agent.py or pytest)",
    )

    # 5. Kill command (Reactive CLI Control)
    kill_parser = subparsers.add_parser("kill", help="Engage kill switch to immediately block all LLM requests")
    kill_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    kill_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    # 6. Resume command (Reactive CLI Control)
    resume_parser = subparsers.add_parser("resume", help="Disengage kill switch and resume normal guarding")
    resume_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    resume_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    # 7. Profile command (Reactive CLI Control)
    profile_parser = subparsers.add_parser("profile", help="Switch active guard profile (careful, standard, passive)")
    profile_parser.add_argument("name", nargs="?", default=None, choices=["careful", "standard", "passive"], help="Profile name (careful, standard, passive)")
    profile_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    profile_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    # 8. Clear command (Reactive CLI Control)
    clear_parser = subparsers.add_parser("clear", help="Clear all request logs and reset in-memory caches")
    clear_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    clear_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")
    clear_parser.add_argument("--db-path", type=Path, default=Path("tokenguard.db"), help="SQLite database path")

    # 9. Logs & Tail command (Live Stream)
    logs_parser = subparsers.add_parser("logs", help="View or stream live LLM request logs")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Stream live activity in real-time")
    logs_parser.add_argument("-n", "--lines", type=int, default=20, help="Number of recent lines to display (default: 20)")
    logs_parser.add_argument("--status", type=str, default="all", choices=["all", "success", "blocked"], help="Filter by status")
    logs_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    logs_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    tail_parser = subparsers.add_parser("tail", help="Alias for 'tokenguard logs -f'")
    tail_parser.add_argument("-n", "--lines", type=int, default=20, help="Number of recent lines to display")
    tail_parser.add_argument("--status", type=str, default="all", choices=["all", "success", "blocked"], help="Filter by status")
    tail_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    tail_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    # 10. Prices command
    prices_parser = subparsers.add_parser("prices", help="Display local pricing table")
    prices_parser.add_argument(
        "--prices-path",
        type=Path,
        default=None,
        help="Path to custom prices.json file",
    )

    # 11. Interactive Menu / TUI command
    menu_parser = subparsers.add_parser("menu", help="Launch interactive navigation menu (Claude/Qwen Code style)")
    menu_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    menu_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    tui_parser = subparsers.add_parser("tui", help="Alias for 'tokenguard menu'")
    tui_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    tui_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    # 12. Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Open TokenGuard web dashboard in browser")
    dash_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    dash_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    # 13. Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run instant circuit breaker simulation")
    sim_parser.add_argument("--type", type=str, default="loop", choices=["loop", "budget", "success"], help="Simulation type")
    sim_parser.add_argument("--model", type=str, default="gpt-4o", help="Model name for simulation")
    sim_parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    sim_parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")

    # Also allow root flags for quick default start
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
        "--max-repeats",
        "--loop-threshold",
        dest="loop_threshold",
        type=int,
        default=3,
        help="Maximum duplicate consecutive requests before tripping loop guard (default: 3, set 0 to disable)",
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
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development",
    )


def print_banner(host: str, port: int, limit: float, loop_threshold: int) -> None:
    """Display the clean Rich TokenGuard terminal startup banner."""
    base_url = f"http://{host}:{port}"
    loop_str = f"Enabled [dim](cutoff: {loop_threshold})[/dim]" if loop_threshold > 0 else "[yellow]Disabled (Passive)[/yellow]"

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column(style="bright_white")

    table.add_row("Proxy Base URL:", f"[cyan]{base_url}/v1[/cyan]")
    table.add_row("Web Dashboard:", f"[bold cyan]{base_url}[/bold cyan]")
    table.add_row("Hourly Limit:", f"[bold white]${limit:.2f}/hr[/bold white]")
    table.add_row("Loop Breaker:", loop_str)

    panel = Panel(
        table,
        title="[bold bright_white]TokenGuard[/bold bright_white] [dim]v0.1.0[/dim]",
        subtitle="[bold green]● ACTIVE & GUARDING[/bold green]",
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
                # Remove stale PID file
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
        "--loop-window-seconds",
        str(loop_window_seconds),
        "--upstream-url",
        str(upstream_url),
        "--db-path",
        str(db_path),
    ]

    log_file = open(LOG_FILE, "a", encoding="utf-8")
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
                    f"[yellow]⚠️ TokenGuard proxy is responding on http://{host}:{port}, but no daemon PID file was found ({PID_FILE}).[/yellow]\n"
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

        # Detected keys
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
        with httpx.Client(timeout=3.0) as client:
            res = client.post(f"http://{host}:{port}/api/kill-switch", json={"active": True, "source": "cli"})
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
        with httpx.Client(timeout=3.0) as client:
            res = client.post(f"http://{host}:{port}/api/kill-switch", json={"active": False, "source": "cli"})
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
    """Read a raw keystroke or ANSI escape sequence from a cbreak file descriptor."""
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

        # ANSI / VT100 / xterm arrow key sequences
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


def render_interactive_menu(
    options: list[tuple[str, str, str, str]],
    selected_idx: int,
    title: str = "Select Option",
    subtitle: Optional[str] = None,
) -> Panel:
    """Render a clean Rich panel with numbered, arrow-navigable menu options."""
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left", style="bold")
    table.add_column(justify="left")
    table.add_column(justify="left", style="dim")
    table.add_column(justify="right")

    for i, (opt_id, opt_title, opt_desc, opt_badge) in enumerate(options):
        is_selected = (i == selected_idx)
        num_str = f"[{i + 1}]"

        if is_selected:
            prefix = f"[bold cyan]❯ {num_str}[/bold cyan]"
            title_styled = f"[bold bright_white]{opt_title}[/bold bright_white]"
            desc_styled = f"[bright_white]{opt_desc}[/bright_white]" if opt_desc else ""
            badge_styled = f"[bold green]{opt_badge}[/bold green]" if opt_badge else ""
        else:
            prefix = f"[dim]  {num_str}[/dim]"
            title_styled = f"[white]{opt_title}[/white]"
            desc_styled = f"[dim]{opt_desc}[/dim]" if opt_desc else ""
            badge_styled = f"[dim]{opt_badge}[/dim]" if opt_badge else ""

        table.add_row(prefix, title_styled, desc_styled, badge_styled)

    sub = subtitle or f"[dim]Use [bold white]↑/↓[/bold white] arrows or press [bold white]1-{len(options)}[/bold white], [bold white]Enter[/bold white] to confirm, [bold white]q[/bold white] to exit[/dim]"

    return Panel(
        table,
        title=f"[bold bright_white]{title}[/bold bright_white]",
        subtitle=sub,
        box=box.ROUNDED,
        border_style="bright_black",
        padding=(1, 2),
    )


def interactive_select(
    options: list[tuple[str, str, str, str]],
    title: str = "Select Option",
    subtitle: Optional[str] = None,
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

    # Hide terminal cursor during interactive selection
    try:
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
    except Exception:
        pass

    try:
        with Live(
            render_interactive_menu(options, selected_idx, title, subtitle),
            console=console,
            auto_refresh=False,
            transient=True,
        ) as live:
            while True:
                live.update(render_interactive_menu(options, selected_idx, title, subtitle), refresh=True)
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
                    if 1 <= val <= len(options):
                        return options[val - 1][0]
    finally:
        # Restore terminal cursor and settings
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        if old_settings is not None and sys.platform != "win32":
            import termios
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def interactive_control_menu(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Display interactive arrow/number menu to manage running TokenGuard proxy (Claude/Qwen Code style)."""
    if not sys.stdin.isatty():
        return

    while True:
        # Check if server is running; if not, ask if user wants to start it
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
            pid = start_daemon(host=host, port=port, limit=settings.hourly_limit, loop_threshold=settings.loop_threshold)
            time.sleep(0.5)

        # Query live stats from daemon
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

        status_badge = "[bold red]🛑 KILLED[/bold red]" if is_killed else "[bold green]● GUARDING[/bold green]"
        status_line = f"Status: {status_badge}  •  Profile: [bold cyan]{active_prof}[/bold cyan]  •  Saved: [bold green]+${saved_usd:.2f}[/bold green]  •  Requests: [bold white]{req_count}[/bold white]"

        menu_options = [
            ("stream", "📡 Stream Live Activity Monitor", "Real-time requests, tokens & loop breaker logs", "Live"),
            ("run", "🚀 Run / Wrap Agent Script", "Execute script with TokenGuard proxy injected", ""),
            ("profile", "🛡️ Switch Guard Profile", "Careful ($5/hr) / Standard ($15/hr) / Passive", ""),
            ("simulate", "⚡ Run Instant Loop Simulation", "Test circuit breaker tripping (no API key needed)", "Demo"),
            ("kill", "🛑 Toggle Kill Switch", "Block / Unblock all LLM traffic instantly", ""),
            ("dashboard", "🌐 Open Web Dashboard", f"http://{host}:{port}", ""),
            ("clear", "🧹 Clear Logs & Reset Breakers", "Flush request database & loop breaker caches", ""),
            ("prices", "📊 View Pricing Registry", "Inspect registered model rates ($/1M tokens)", ""),
            ("stop", "⏹️ Stop TokenGuard Daemon", "Terminate background proxy server", ""),
            ("exit", "🚪 Exit Menu", "Return to terminal (daemon remains active)", ""),
        ]

        choice = interactive_select(
            menu_options,
            title="🛡️ TokenGuard Control Center",
            subtitle=f"{status_line}\n[dim]Navigate with [bold white]↑/↓[/bold white] arrows or press [bold white]1-9[/bold white], [bold white]Enter[/bold white] to run, [bold white]q[/bold white] to exit[/dim]",
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
                    raw_args = cmd_str.split()
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
                ("careful", "Careful", "$5/hr cap, 2-loop cutoff, alerts ON", "Default"),
                ("standard", "Standard Agent", "$15/hr cap, 4-loop cutoff", "Autonomous"),
                ("passive", "Passive Monitor", "No loop blocks, tracking only", "Observability"),
            ]
            prof_choice = interactive_select(
                profile_opts,
                title="Choose Guard Profile",
                subtitle="[dim]Use [bold white]↑/↓[/bold white] arrows or press [bold white]1-3[/bold white], [bold white]Enter[/bold white] to apply[/dim]",
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
                                f"[bold green]✓ Simulation Completed[/bold green]\n\n"
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
            ("careful", "Careful", "$5/hr cap, 2-loop cutoff, alerts ON", "Default"),
            ("standard", "Standard Agent", "$15/hr cap, 4-loop cutoff", "Autonomous"),
            ("passive", "Passive Monitor", "No loop blocks, tracking only", "Observability"),
        ]
        name = interactive_select(
            profile_opts,
            title="Choose Guard Profile",
            subtitle="[dim]Use [bold white]↑/↓[/bold white] arrows or press [bold white]1-3[/bold white], [bold white]Enter[/bold white] to apply[/dim]",
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
        with httpx.Client(timeout=3.0) as client:
            res = client.post(f"http://{host}:{port}/api/clear-logs", json={"source": "cli"})
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
    if getattr(args, "daemon", False):
        with console.status("[bright_black]Starting TokenGuard daemon in background...[/bright_black]", spinner="dots"):
            pid = start_daemon(
                host=args.host,
                port=args.port,
                limit=args.limit,
                daily_limit=args.daily_limit,
                loop_threshold=args.loop_threshold,
                loop_window_seconds=args.loop_window_seconds,
                upstream_url=args.upstream_url,
                db_path=args.db_path,
            )

        if wait_for_server(args.host, args.port, timeout=5.0):
            table = Table.grid(padding=(0, 2))
            table.add_column(style="dim", justify="right")
            table.add_column(style="bright_white")

            table.add_row("Status:", "[bold green]● ACTIVE & GUARDING[/bold green]")
            table.add_row("Proxy URL:", f"[cyan]http://{args.host}:{args.port}/v1[/cyan]")
            table.add_row("Dashboard:", f"[bold cyan]http://{args.host}:{args.port}[/bold cyan]")
            table.add_row("Process PID:", f"[dim]{pid}[/dim]")
            table.add_row("Log File:", f"[dim]{LOG_FILE}[/dim]")

            panel = Panel(
                table,
                title="[bold bright_white]TokenGuard Daemon[/bold bright_white]",
                box=box.ROUNDED,
                border_style="bright_black",
                padding=(1, 2),
            )
            console.print(panel)
            if sys.stdin.isatty():
                interactive_control_menu(args.host, args.port)
        else:
            console.print(f"[yellow]⚠️ TokenGuard daemon spawned (PID: {pid}). Check logs at {LOG_FILE}[/yellow]")
        return

    settings = Settings(
        host=args.host,
        port=args.port,
        hourly_limit=args.limit,
        daily_limit=args.daily_limit,
        loop_threshold=args.loop_threshold,
        loop_window_seconds=args.loop_window_seconds,
        upstream_url=args.upstream_url,
        db_path=args.db_path,
    )
    set_settings(settings)

    print_banner(
        host=settings.host,
        port=settings.port,
        limit=settings.hourly_limit,
        loop_threshold=settings.loop_threshold,
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
            )
        )
        sys.exit(1)

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8080)

    # 1. Check if TokenGuard proxy is already running; if not, auto-start daemon
    if not is_server_running(host, port):
        with console.status(f"[bright_black]Auto-starting TokenGuard proxy on http://{host}:{port}...[/bright_black]", spinner="dots"):
            pid = start_daemon(
                host=host,
                port=port,
                limit=getattr(args, "limit", 5.0),
                daily_limit=getattr(args, "daily_limit", 50.0),
                loop_threshold=getattr(args, "loop_threshold", 3),
                loop_window_seconds=getattr(args, "loop_window_seconds", 60.0),
                upstream_url=getattr(args, "upstream_url", "https://api.openai.com"),
                db_path=getattr(args, "db_path", Path("tokenguard.db")),
            )

        if not is_server_running(host, port):
            err_console.print(f"[bold red]❌ Error: Failed to auto-start TokenGuard proxy. Check {LOG_FILE}[/bold red]")
            sys.exit(1)

        console.print(f"[bold green]●[/bold green] [dim]TokenGuard proxy running (PID: {pid})[/dim]")

    # 2. Inject environment variables pointing to TokenGuard proxy
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

    # Resolve executable if 'python' was invoked but only python3/sys.executable exists in environment
    import shutil
    if raw_cmd[0] == "python" and not shutil.which("python"):
        raw_cmd[0] = sys.executable

    # 3. Transparently execute user command
    try:
        res = subprocess.run(raw_cmd, env=env)
        sys.exit(res.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


def cmd_quickstart(args: argparse.Namespace) -> None:
    """Interactive zero-friction setup and quickstart flow."""
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8080)

    # If proxy is already running, display status and open interactive command menu
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

    # Header Panel
    console.print(
        Panel(
            "[bold bright_white]TokenGuard[/bold bright_white] [dim]— Zero-Latency LLM Circuit Breaker & Multi-Provider Proxy[/dim]\n"
            "[dim]Protects against infinite agent loops, runaway token costs, and API overruns.[/dim]",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(1, 2),
        )
    )

    # 1. Detect existing API keys
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

    # If no keys found and running interactively, offer quick input
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
                console.print("[dim]✓ Stored in ~/.tokenguard/config.json with restricted permissions (0600)[/dim]")
        except (KeyboardInterrupt, EOFError):
            pass

    # 2. Guard Profile Selection via Arrow Keys & Numbers
    profile_opts = [
        ("careful", "Careful", "$5/hr cap, 2-loop cutoff, alerts ON", "Recommended"),
        ("standard", "Standard Agent", "$15/hr cap, 4-loop cutoff", "Autonomous"),
        ("passive", "Passive Monitor", "No loop blocks, tracking & telemetry only", "Observability"),
    ]
    selected_profile_key = interactive_select(
        profile_opts,
        title="2. Choose Guard Profile",
        subtitle="[dim]Use [bold white]↑/↓[/bold white] arrows or press [bold white]1-3[/bold white], [bold white]Enter[/bold white] to confirm[/dim]",
        default_index=0,
    )
    profile = PROFILES.get(selected_profile_key, PROFILES["careful"])

    # 3. Start background daemon
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

    # 4. Display Claude Code / Qwen Code Status & Launch Panel
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

    # If running interactively, offer the control menu immediately
    if sys.stdin.isatty():
        interactive_control_menu(host, port)


def cmd_prices(args: argparse.Namespace) -> None:
    """Print out registered prices in a clean Rich table."""
    settings = get_settings()
    prices_path = args.prices_path or settings.prices_path
    if not (prices_path and prices_path.exists()):
        err_console.print(f"[bold red]Error: Prices file not found at {prices_path}[/bold red]")
        sys.exit(1)

    with open(prices_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    table = Table(
        title=f"TokenGuard Pricing Registry ({prices_path.name})",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold bright_white",
    )
    table.add_column("Model Name", style="bold cyan")
    table.add_column("Input Rate ($/1M)", justify="right", style="bright_white")
    table.add_column("Output Rate ($/1M)", justify="right", style="bright_white")
    table.add_column("Provider", style="dim")

    for model, rates in sorted(data.items()):
        is_ds = "deepseek" in model.lower()
        provider = "[cyan]DeepSeek[/cyan]" if is_ds else "[green]OpenAI[/green]"
        in_cost = f"${float(rates.get('input', 0.0)):.2f}"
        out_cost = f"${float(rates.get('output', 0.0)):.2f}"
        table.add_row(model, in_cost, out_cost, provider)

    console.print(table)


def main() -> None:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
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
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8080)
        sim_type = getattr(args, "type", "loop")
        sim_model = getattr(args, "model", "gpt-4o")
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.post(f"http://{host}:{port}/api/simulate", json={"type": sim_type, "model": sim_model})
                if res.status_code == 200:
                    console.print(
                        Panel(
                            f"[bold green]✓ Simulation Completed[/bold green]\n\n"
                            f"[dim]Type:[/dim] {sim_type}  •  [dim]Model:[/dim] {sim_model}\n"
                            f"[dim]Response:[/dim] {res.json()}",
                            title="[bold bright_white]TokenGuard Simulation[/bold bright_white]",
                            box=box.ROUNDED,
                            border_style="bright_black",
                        )
                    )
                else:
                    err_console.print(f"[bold red]Simulation failed: {res.text}[/bold red]")
        except Exception as e:
            err_console.print(f"[bold red]Error running simulation: {e}[/bold red]")
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
