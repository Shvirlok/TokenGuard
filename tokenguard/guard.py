from __future__ import annotations

import collections
from collections import OrderedDict
import difflib
import hashlib
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
import threading
from threading import RLock
import time
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("tokenguard.guard")


def parse_burn_rate(burn_rate_str: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse burn rate string into (budget_limit, window_seconds).

    Examples:
        "$0.50/min" -> (0.50, 60.0)
        "$2.00/5m"  -> (2.00, 300.0)
        "1.50/30s"  -> (1.50, 30.0)
        "$10.00/1h" -> (10.0, 3600.0)
    """
    if not burn_rate_str or not str(burn_rate_str).strip():
        return None

    s = str(burn_rate_str).strip().lower()
    if "/" not in s:
        raise ValueError(
            f"Invalid burn rate format: '{burn_rate_str}'. Expected format like '$0.50/min' or '$2.00/5m'."
        )

    amt_part, window_part = s.split("/", 1)
    amt_part = amt_part.replace("$", "").strip()
    try:
        limit = float(amt_part)
    except ValueError:
        raise ValueError(f"Invalid cost amount in burn rate: '{amt_part}'")

    window_part = window_part.strip()
    m = re.match(r"^(\d*\.?\d*)\s*([a-z]+)$", window_part)
    if not m:
        raise ValueError(f"Invalid time window in burn rate: '{window_part}'")

    num_str, unit = m.groups()
    multiplier = float(num_str) if num_str else 1.0

    if unit in ("s", "sec", "second", "seconds"):
        seconds = multiplier * 1.0
    elif unit in ("m", "min", "minute", "minutes"):
        seconds = multiplier * 60.0
    elif unit in ("h", "hr", "hour", "hours"):
        seconds = multiplier * 3600.0
    elif unit in ("d", "day", "days"):
        seconds = multiplier * 86400.0
    else:
        raise ValueError(f"Unknown time unit '{unit}' in burn rate window. Supported: s, m, h, d.")

    if seconds <= 0:
        raise ValueError("Burn rate window duration must be greater than 0")
    if limit <= 0:
        raise ValueError("Burn rate limit amount must be greater than 0")

    return (limit, seconds)


def send_block_notification(
    title: str = "TokenGuard Alert",
    message: str = "Infinite LLM agent loop detected and blocked.",
    sound: bool = True,
) -> None:
    """Trigger a lightweight non-blocking system notification or terminal bell.

    Runs in a detached background daemon thread so it never blocks the request flow.
    Fails completely gracefully if tools or desktop environments are unavailable.
    """
    def _notify() -> None:
        try:
            if sound:
                try:
                    sys.stderr.write("\a")
                    sys.stderr.flush()
                except Exception:
                    pass

            system = platform.system().lower()
            if system == "darwin":
                safe_title = title.replace('"', '\\"')
                safe_msg = message.replace('"', '\\"')
                script = f'display notification "{safe_msg}" with title "{safe_title}" sound name "Ping"'
                subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    timeout=2.0,
                    check=False,
                )
            elif system == "linux":
                if shutil.which("notify-send"):
                    subprocess.run(
                        ["notify-send", "-u", "critical", "-a", "TokenGuard", title, message],
                        capture_output=True,
                        timeout=2.0,
                        check=False,
                    )
        except Exception:
            pass

    try:
        t = threading.Thread(target=_notify, daemon=True)
        t.start()
    except Exception:
        pass


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker tripping."""

    def __init__(
        self,
        message: str,
        error_type: str,
        status_code: int = 429,
        code: Optional[str] = None,
        current_spend: Optional[float] = None,
        budget_limit: Optional[float] = None,
        tokens_saved: Optional[int] = None,
        usd_saved: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.code = code
        self.current_spend = current_spend
        self.budget_limit = budget_limit
        self.tokens_saved = tokens_saved
        self.usd_saved = usd_saved
        self.extra = kwargs

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "message": self.message,
            "type": self.error_type,
        }
        if self.code is not None:
            data["code"] = self.code
        if self.current_spend is not None:
            data["current_spend"] = self.current_spend
        if self.budget_limit is not None:
            data["budget_limit"] = self.budget_limit
        if self.tokens_saved is not None:
            data["tokens_saved"] = self.tokens_saved
        if self.usd_saved is not None:
            data["usd_saved"] = self.usd_saved
        for k, v in self.extra.items():
            if k not in data:
                data[k] = v
        return {"error": data}


class BudgetExceededError(CircuitBreakerError):
    def __init__(
        self,
        message: str = "TokenGuard: budget limit exceeded",
        current_spend: Optional[float] = None,
        budget_limit: Optional[float] = None,
        tokens_saved: Optional[int] = None,
        usd_saved: Optional[float] = None,
    ):
        super().__init__(
            message=message,
            error_type="budget_exceeded",
            status_code=429,
            current_spend=current_spend,
            budget_limit=budget_limit,
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
        )


class BurnRateExceededError(CircuitBreakerError):
    """Tripped when cumulative spend in a sliding window exceeds max burn rate."""

    def __init__(
        self,
        message: str,
        current_spend: float,
        budget_limit: float,
        code: str = "burn_rate_exceeded",
        error_type: str = "tokenguard_budget_breach",
        tokens_saved: Optional[int] = None,
        usd_saved: Optional[float] = None,
    ):
        super().__init__(
            message=message,
            error_type=error_type,
            status_code=429,
            code=code,
            current_spend=round(current_spend, 2),
            budget_limit=round(budget_limit, 2),
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
        )


class ContextBallooningError(CircuitBreakerError):
    """Tripped when prompt tokens expand abnormal runaway sequence without cache hit or task update."""

    def __init__(
        self,
        message: str = "TokenGuard Circuit Breaker Tripped: Runaway context expansion detected.",
        prompt_tokens: Optional[int] = None,
        baseline_tokens: Optional[int] = None,
        growth_rate: Optional[float] = None,
        code: str = "context_ballooning_detected",
        error_type: str = "tokenguard_budget_breach",
        tokens_saved: Optional[int] = None,
        usd_saved: Optional[float] = None,
        **kwargs: Any,
    ):
        extra: Dict[str, Any] = dict(kwargs)
        if prompt_tokens is not None:
            extra["current_tokens"] = prompt_tokens
        if baseline_tokens is not None:
            extra["baseline_tokens"] = baseline_tokens
        if growth_rate is not None:
            extra["growth_rate"] = growth_rate
        super().__init__(
            message=message,
            error_type=error_type,
            status_code=429,
            code=code,
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
            **extra,
        )


class LoopDetectedError(CircuitBreakerError):
    def __init__(
        self,
        message: str = "TokenGuard: infinite agent loop detected",
        cycle_length: int = 1,
        pattern: Optional[List[str]] = None,
        tokens_saved: Optional[int] = None,
        usd_saved: Optional[float] = None,
    ):
        super().__init__(
            message=message,
            error_type="loop_detected",
            status_code=429,
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
        )
        self.cycle_length = cycle_length
        self.pattern = pattern


class ToolPingPongError(CircuitBreakerError):
    """Tripped when an autonomous agent gets stuck in a repetitive tool-call loop."""

    def __init__(
        self,
        tool_name: str,
        message: Optional[str] = None,
        repeats: int = 3,
        repeat_count: Optional[int] = None,
        tool_args: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
        cycle_length: int = 1,
        pattern: Optional[List[str]] = None,
        tokens_saved: Optional[int] = None,
        usd_saved: Optional[float] = None,
    ):
        final_repeats = repeat_count if repeat_count is not None else repeats
        if cycle_length > 1 and pattern:
            cycle_desc = " -> ".join(pattern)
            msg = message or f"TokenGuard intercepted alternating tool call loop (Length {cycle_length}: {cycle_desc})"
        else:
            msg = message or f"TokenGuard intercepted repeated tool call loop: {tool_name}"
        super().__init__(
            message=msg,
            error_type="circuit_breaker",
            status_code=429,
            code="TOOL_PING_PONG_LOOP",
            tool_name=tool_name,
            repeats=final_repeats,
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
        )
        self.tool_name = tool_name
        self.repeats = final_repeats
        self.repeat_count = final_repeats
        self.tool_args = tool_args
        self.elapsed_seconds = elapsed_seconds
        self.cycle_length = cycle_length
        self.pattern = pattern or [tool_name]

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "circuit_breaker",
            "code": "TOOL_PING_PONG_LOOP",
            "message": self.message,
            "tool_name": self.tool_name,
            "repeats": self.repeats,
        }
        if self.cycle_length > 1:
            data["cycle_length"] = self.cycle_length
            if self.pattern:
                data["pattern"] = self.pattern
        if self.tokens_saved is not None:
            data["tokens_saved"] = self.tokens_saved
        if self.usd_saved is not None:
            data["usd_saved"] = self.usd_saved
        return {
            "error": data
        }


class ToolRetryCeilingError(CircuitBreakerError):
    """Tripped when the same tool_name is invoked >= max_tool_repeats consecutively, ignoring arguments."""

    def __init__(
        self,
        tool_name: str,
        repeats: int,
        message: Optional[str] = None,
        tokens_saved: Optional[int] = None,
        usd_saved: Optional[float] = None,
    ):
        msg = message or f"ToolRetryCeilingTripped: {tool_name} called {repeats} times consecutively"
        super().__init__(
            message=msg,
            error_type="circuit_breaker",
            status_code=429,
            code="ToolRetryCeilingTripped",
            reason="ToolRetryCeilingTripped",
            tool_name=tool_name,
            repeats=repeats,
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
        )
        self.reason = "ToolRetryCeilingTripped"
        self.tool_name = tool_name
        self.repeats = repeats

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "circuit_breaker",
            "code": "ToolRetryCeilingTripped",
            "reason": "ToolRetryCeilingTripped",
            "message": self.message,
            "tool_name": self.tool_name,
            "repeats": self.repeats,
        }
        if self.tokens_saved is not None:
            data["tokens_saved"] = self.tokens_saved
        if self.usd_saved is not None:
            data["usd_saved"] = self.usd_saved
        return {
            "reason": "ToolRetryCeilingTripped",
            "error": data,
        }


class SessionStagnationError(CircuitBreakerError):
    """Tripped when an agent executes >= max_stagnant_steps without state change or output milestones."""

    def __init__(
        self,
        steps: int,
        message: Optional[str] = None,
        tokens_saved: Optional[int] = None,
        usd_saved: Optional[float] = None,
    ):
        msg = message or f"SessionStagnationTripped: {steps} steps without state change"
        super().__init__(
            message=msg,
            error_type="circuit_breaker",
            status_code=429,
            code="SessionStagnationTripped",
            reason="SessionStagnationTripped",
            steps=steps,
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
        )
        self.reason = "SessionStagnationTripped"
        self.steps = steps

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "circuit_breaker",
            "code": "SessionStagnationTripped",
            "reason": "SessionStagnationTripped",
            "message": self.message,
            "steps": self.steps,
        }
        if self.tokens_saved is not None:
            data["tokens_saved"] = self.tokens_saved
        if self.usd_saved is not None:
            data["usd_saved"] = self.usd_saved
        return {
            "reason": "SessionStagnationTripped",
            "error": data,
        }


class KillSwitchActiveError(CircuitBreakerError):
    def __init__(
        self,
        message: str = "TokenGuard kill switch active: all LLM calls manually blocked",
        tokens_saved: Optional[int] = None,
        usd_saved: Optional[float] = None,
    ):
        super().__init__(
            message=message,
            error_type="kill_switch_active",
            status_code=429,
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
        )


class SessionBudgetLimitError(CircuitBreakerError):
    def __init__(
        self,
        current_spend: float,
        session_limit: float,
        session_id: str = "",
        message: Optional[str] = None,
        tokens_saved: Optional[int] = None,
        usd_saved: Optional[float] = None,
    ):
        msg = message or f"SessionBudgetLimitTripped: session spend ${current_spend:.4f} exceeded limit ${session_limit:.4f}"
        super().__init__(
            message=msg,
            error_type="circuit_breaker",
            status_code=429,
            code="SessionBudgetLimitTripped",
            reason="SessionBudgetLimitTripped",
            current_spend=round(current_spend, 4),
            session_limit=round(session_limit, 4),
            session_id=session_id,
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
        )
        self.reason = "SessionBudgetLimitTripped"
        self.current_spend = current_spend
        self.session_limit = session_limit
        self.session_id = session_id

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "circuit_breaker",
            "code": "SessionBudgetLimitTripped",
            "reason": "SessionBudgetLimitTripped",
            "message": self.message,
            "session_id": self.session_id,
            "current_spend": round(self.current_spend, 4),
            "session_limit": round(self.session_limit, 4),
        }
        if self.tokens_saved is not None:
            data["tokens_saved"] = self.tokens_saved
        if self.usd_saved is not None:
            data["usd_saved"] = self.usd_saved
        return {
            "reason": "SessionBudgetLimitTripped",
            "error": data,
        }


class SecretLeakError(CircuitBreakerError):
    """Tripped when an outgoing request contains raw secrets (API keys, private keys)."""

    def __init__(
        self,
        pattern_type: str,
        message: Optional[str] = None,
        tokens_saved: Optional[int] = None,
        usd_saved: Optional[float] = None,
    ):
        msg = message or f"SecretLeakDetected: Found matching pattern {pattern_type}"
        super().__init__(
            message=msg,
            error_type="secret_leak_detected",
            status_code=400,
            code="SecretLeakDetected",
            pattern_type=pattern_type,
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
        )
        self.pattern_type = pattern_type

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "secret_leak_detected",
            "code": "SecretLeakDetected",
            "message": self.message,
            "pattern_type": self.pattern_type,
        }
        if self.tokens_saved is not None:
            data["tokens_saved"] = self.tokens_saved
        if self.usd_saved is not None:
            data["usd_saved"] = self.usd_saved
        return {"error": data}


SECRET_PATTERNS: List[Tuple[str, str]] = [
    ("pem_private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("openai_proj_key", r"sk-proj-[a-zA-Z0-9_\-]{20,}"),
    ("openai_key", r"sk-[a-zA-Z0-9]{20,}"),
    ("anthropic_key", r"sk-ant-[a-zA-Z0-9_\-]{20,}"),
    ("aws_access_key", r"AKIA[0-9A-Z]{16}"),
    ("github_pat", r"github_pat_[a-zA-Z0-9_]{82}"),
    ("github_token", r"ghp_[a-zA-Z0-9]{36}"),
]

COMBINED_SECRET_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in SECRET_PATTERNS))


def scan_secrets_in_obj(obj: Any) -> Optional[str]:
    """Recursively search for secrets in objects (strings, dicts, lists).
    
    Returns pattern group name (e.g. 'openai_proj_key', 'anthropic_key') or None.
    """
    if isinstance(obj, str):
        m = COMBINED_SECRET_RE.search(obj)
        if m:
            return m.lastgroup
        return None
    elif isinstance(obj, dict):
        for v in obj.values():
            found = scan_secrets_in_obj(v)
            if found:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = scan_secrets_in_obj(item)
            if found:
                return found
    return None


def redact_secrets_in_obj(obj: Any) -> Tuple[Any, bool]:
    """Recursively replace matching secrets with '***REDACTED_SECRET***' in objects."""
    if isinstance(obj, str):
        if COMBINED_SECRET_RE.search(obj):
            return COMBINED_SECRET_RE.sub("***REDACTED_SECRET***", obj), True
        return obj, False
    elif isinstance(obj, dict):
        new_dict = {}
        any_redacted = False
        for k, v in obj.items():
            new_v, redacted = redact_secrets_in_obj(v)
            new_dict[k] = new_v
            if redacted:
                any_redacted = True
        return new_dict, any_redacted
    elif isinstance(obj, list):
        new_list = []
        any_redacted = False
        for item in obj:
            new_item, redacted = redact_secrets_in_obj(item)
            new_list.append(new_item)
            if redacted:
                any_redacted = True
        return new_list, any_redacted
    return obj, False


DYNAMIC_NONCE_KEYS: frozenset[str] = frozenset({
    "call_id",
    "timestamp",
    "ts",
    "nonce",
    "_id",
    "request_id",
})


def _strip_dynamic_keys(obj: Any) -> Any:
    """Recursively strip dynamic nonces/IDs and trim string values."""
    if isinstance(obj, dict):
        return {
            str(k): _strip_dynamic_keys(v)
            for k, v in obj.items()
            if str(k).lower() not in DYNAMIC_NONCE_KEYS
        }
    elif isinstance(obj, (list, tuple)):
        return [_strip_dynamic_keys(item) for item in obj]
    elif isinstance(obj, str):
        return obj.strip()
    return obj


def normalize_tool_arguments(args: Any) -> Tuple[str, Any]:
    """Normalize tool arguments into stable canonical JSON string and parsed object.

    Strips whitespace, dynamic nonces/IDs (call_id, timestamp, ts, nonce, _id, request_id),
    sorts keys recursively, and formats without extra spacing.
    """
    if isinstance(args, str):
        s = args.strip()
        try:
            parsed = json.loads(s)
            cleaned = _strip_dynamic_keys(parsed)
            return json.dumps(cleaned, sort_keys=True, separators=(",", ":")), cleaned
        except Exception:
            return s, s
    elif isinstance(args, dict):
        cleaned = _strip_dynamic_keys(args)
        return json.dumps(cleaned, sort_keys=True, separators=(",", ":")), cleaned
    elif isinstance(args, (list, tuple)):
        cleaned = _strip_dynamic_keys(list(args))
        return json.dumps(cleaned, sort_keys=True, separators=(",", ":")), cleaned
    else:
        s = str(args).strip()
        return s, s


def compute_argument_similarity(args_str1: str, args_str2: str) -> float:
    """Compute normalized argument overlap between 0.0 and 1.0 using standard library difflib."""
    if args_str1 == args_str2:
        return 1.0
    if not args_str1 or not args_str2:
        return 0.0
    return difflib.SequenceMatcher(None, args_str1, args_str2).ratio()


def detect_cycle_in_sequence(
    items: Sequence[Any],
    match_fn: Optional[Callable[[Any, Any], bool]] = None,
    length_1_repeats: int = 3,
) -> Optional[Dict[str, Any]]:
    """Detect repeating multi-length cycles at the tail of a sequence.

    Supported patterns:
    - Length 1 cycle: A-A-A (three identical consecutive calls)
    - Length 2 cycle: A-B-A-B (alternating calls repeating twice, where A != B)
    - Length 3 cycle: A-B-C-A-B-C (three-step call sequence repeating twice)

    Returns:
        Dict with keys:
            'cycle_length': 1, 2, or 3
            'repeats': number of times cycle pattern repeated
            'pattern': list of items forming the base cycle pattern
            'matched_items': the full matched tail slice
        or None if no repeating cycle is detected.
    """
    if not items:
        return None

    def _eq(a: Any, b: Any) -> bool:
        if match_fn is not None:
            return match_fn(a, b)
        return a == b

    n = len(items)

    if length_1_repeats >= 1 and n >= length_1_repeats:
        tail_1 = items[-length_1_repeats:]
        first = tail_1[0]
        if all(_eq(item, first) for item in tail_1[1:]):
            return {
                "cycle_length": 1,
                "repeats": length_1_repeats,
                "pattern": [first],
                "matched_items": list(tail_1),
            }

    if n >= 4:
        s0, s1, s2, s3 = items[-4], items[-3], items[-2], items[-1]
        if _eq(s0, s2) and _eq(s1, s3) and not _eq(s0, s1):
            return {
                "cycle_length": 2,
                "repeats": 2,
                "pattern": [s0, s1],
                "matched_items": [s0, s1, s2, s3],
            }

    if n >= 6:
        s0, s1, s2, s3, s4, s5 = items[-6:]
        if _eq(s0, s3) and _eq(s1, s4) and _eq(s2, s5):
            if not (_eq(s0, s1) and _eq(s1, s2)):
                return {
                    "cycle_length": 3,
                    "repeats": 2,
                    "pattern": [s0, s1, s2],
                    "matched_items": [s0, s1, s2, s3, s4, s5],
                }

    return None


DEFAULT_MUTATING_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "edit_file",
    "create_file",
    "delete_file",
    "remove_file",
    "move_file",
    "rename_file",
    "copy_file",
    "patch_file",
    "apply_patch",
    "replace_in_file",
    "str_replace_editor",
    "multi_edit",
    "notebook_edit",
    "bash",
    "sh",
    "zsh",
    "shell",
    "run_command",
    "run_terminal_cmd",
    "execute",
    "execute_command",
    "exec_command",
    "terminal",
    "git_commit",
    "git_push",
    "git_checkout",
    "git_merge",
    "git_rebase",
    "git_reset",
    "sql_execute",
    "db_write",
    "insert",
    "update",
    "delete",
    "upsert",
    "create_issue",
    "create_pr",
    "post_comment",
    "send_message",
    "send_email",
})

_MUTATING_PREFIXES: Tuple[str, ...] = (
    "write_",
    "edit_",
    "create_",
    "delete_",
    "remove_",
    "update_",
    "insert_",
    "upsert_",
    "patch_",
    "apply_",
    "exec_",
    "execute_",
    "run_",
    "save_",
    "put_",
    "post_",
    "send_",
    "commit_",
    "push_",
    "install_",
    "mkdir",
    "rm_",
    "mv_",
)


def is_mutating_tool(tool_name: str, custom_mutating: Optional[frozenset[str]] = None) -> bool:
    """Return True if tool_name represents a state-changing / mutating action."""
    if not tool_name:
        return False
    clean = tool_name.strip().lower()
    if custom_mutating is not None:
        return clean in custom_mutating
    return clean in DEFAULT_MUTATING_TOOLS or clean.startswith(_MUTATING_PREFIXES)


def _is_schema_tool_definition(item: Dict[str, Any]) -> bool:
    """Return True if dict represents a JSON Schema tool definition rather than an invocation."""
    if "input_schema" in item or "parameters" in item:
        return True
    fn = item.get("function")
    if isinstance(fn, dict) and ("parameters" in fn or "input_schema" in fn):
        return True
    return False


def _is_tool_result_message(msg: Dict[str, Any]) -> bool:
    """Return True if chat message is a tool/function result frame."""
    role = msg.get("role")
    if role in ("tool", "function"):
        return True
    if role == "user":
        content = msg.get("content")
        if isinstance(content, list) and content:
            return all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return False


def _extract_raw_names_from_dict(item: Dict[str, Any]) -> List[str]:
    """Extract raw tool names from a single message or tool-call dict without normalizing args."""
    names: List[str] = []
    if _is_schema_tool_definition(item):
        return names

    tool_calls = item.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict) and not _is_schema_tool_definition(tc):
                fn = tc.get("function")
                raw_name = fn.get("name") if isinstance(fn, dict) else tc.get("name")
                if isinstance(raw_name, str) and raw_name.strip():
                    names.append(raw_name.strip())

    fn_call = item.get("function_call")
    if isinstance(fn_call, dict):
        raw_name = fn_call.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            names.append(raw_name.strip())

    content = item.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                raw_name = block.get("name")
                if isinstance(raw_name, str) and raw_name.strip():
                    names.append(raw_name.strip())

    tools = item.get("tools")
    if isinstance(tools, list):
        for t in tools:
            if isinstance(t, dict) and not _is_schema_tool_definition(t):
                fn = t.get("function")
                raw_name = fn.get("name") if isinstance(fn, dict) else t.get("name")
                if isinstance(raw_name, str) and raw_name.strip():
                    names.append(raw_name.strip())

    if not names and "role" not in item:
        fn = item.get("function")
        raw_name = fn.get("name") if isinstance(fn, dict) else item.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            names.append(raw_name.strip())

    return names


def extract_tool_calls(payload_or_messages: Any) -> List[Tuple[str, str, Any]]:
    """Extract tool calls from request payload, tool list, or messages list.

    Returns a list of (tool_name, normalized_args_string, raw_args).
    """
    results: List[Tuple[str, str, Any]] = []

    def _add(name: Any, args: Any):
        if name and isinstance(name, str) and name.strip():
            norm_str, raw_obj = normalize_tool_arguments(args)
            results.append((name.strip(), norm_str, raw_obj))

    def _process_item(item: Any):
        if not isinstance(item, dict) or _is_schema_tool_definition(item):
            return

        before_len = len(results)
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict) and not _is_schema_tool_definition(tc):
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        _add(fn.get("name"), fn.get("arguments", {}))
                    else:
                        _add(tc.get("name"), tc.get("arguments") if "arguments" in tc else tc.get("input", {}))

        fn_call = item.get("function_call")
        if isinstance(fn_call, dict):
            _add(fn_call.get("name"), fn_call.get("arguments", {}))

        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    _add(block.get("name"), block.get("input", {}))

        tools = item.get("tools")
        if isinstance(tools, list) and len(results) == before_len:
            for t in tools:
                if isinstance(t, dict) and not _is_schema_tool_definition(t):
                    fn = t.get("function")
                    if isinstance(fn, dict):
                        _add(fn.get("name"), fn.get("arguments", {}))
                    else:
                        _add(t.get("name"), t.get("arguments") if "arguments" in t else t.get("input", {}))

        if len(results) == before_len and "role" not in item:
            fn = item.get("function")
            if isinstance(fn, dict):
                _add(fn.get("name"), fn.get("arguments", {}))
            elif "name" in item:
                _add(item.get("name"), item.get("arguments") if "arguments" in item else item.get("input", {}))

    if isinstance(payload_or_messages, list):
        for msg in payload_or_messages:
            _process_item(msg)
    elif isinstance(payload_or_messages, dict):
        messages = payload_or_messages.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                _process_item(msg)
            if not results:
                _process_item(payload_or_messages)
        else:
            _process_item(payload_or_messages)

    return results


def _extract_active_step_tools(
    messages: Any,
    tool_calls: Optional[List[Any]] = None,
) -> Tuple[List[Tuple[str, str, Any]], bool]:
    """Extract only the active/current iteration's tool calls and whether args were explicitly supplied.

    Returns:
        (active_tools, has_explicit_args)
    """
    if tool_calls:
        explicit = extract_tool_calls(tool_calls)
        if explicit:
            has_args = any(
                isinstance(tc, dict)
                and (
                    "arguments" in tc
                    or "input" in tc
                    or (isinstance(tc.get("function"), dict) and "arguments" in tc["function"])
                )
                for tc in (tool_calls if isinstance(tool_calls, list) else [tool_calls])
            )
            return explicit, has_args

    target = messages
    if isinstance(target, dict):
        top_tc = target.get("tool_calls") or target.get("tools")
        if top_tc:
            explicit = extract_tool_calls(top_tc)
            if explicit:
                has_args = any(
                    isinstance(tc, dict)
                    and (
                        "arguments" in tc
                        or "input" in tc
                        or (isinstance(tc.get("function"), dict) and "arguments" in tc["function"])
                    )
                    for tc in (top_tc if isinstance(top_tc, list) else [top_tc])
                )
                return explicit, has_args
        target = target.get("messages")

    if isinstance(target, list) and target:
        last = target[-1]
        if isinstance(last, dict) and "role" not in last:
            extracted = extract_tool_calls(target)
            has_args = any(
                isinstance(tc, dict)
                and (
                    "arguments" in tc
                    or "input" in tc
                    or (isinstance(tc.get("function"), dict) and "arguments" in tc["function"])
                )
                for tc in target
            )
            return extracted, has_args

        idx = len(target) - 1
        while idx >= 0:
            msg = target[idx]
            if not isinstance(msg, dict):
                idx -= 1
                continue
            if _is_tool_result_message(msg):
                idx -= 1
                continue
            extracted = extract_tool_calls([msg])
            return extracted, bool(extracted)

    return [], False


def _compute_message_tail_streaks(
    messages: Any,
    mutating_set: Optional[frozenset[str]] = None,
) -> Tuple[Optional[str], int, int]:
    """Scan backwards from the tail of messages to compute consecutive tool name & stagnant step streaks.

    Returns:
        (tail_tool_name, consecutive_same_tool_count, consecutive_stagnant_steps)
    """
    target = messages.get("messages") if isinstance(messages, dict) else messages
    if not isinstance(target, list) or not target:
        return None, 0, 0

    if isinstance(target[-1], dict) and "role" not in target[-1]:
        names: List[str] = []
        for item in target:
            if isinstance(item, dict):
                names.extend(_extract_raw_names_from_dict(item))
        if not names:
            return None, 0, 0
        latest = names[-1]
        same_cnt = 0
        for nm in reversed(names):
            if nm == latest:
                same_cnt += 1
            else:
                break
        stag_cnt = 0
        for nm in reversed(names):
            if is_mutating_tool(nm, mutating_set):
                break
            stag_cnt += 1
        return latest, same_cnt, stag_cnt

    tail_tool_name: Optional[str] = None
    same_tool_count = 0
    same_tool_active = True
    stagnant_steps = 0
    stagnant_active = True
    saw_any_step = False

    idx = len(target) - 1
    while idx >= 0:
        msg = target[idx]
        idx -= 1
        if not isinstance(msg, dict):
            continue
        if _is_tool_result_message(msg):
            continue

        turn_names = _extract_raw_names_from_dict(msg)
        if not turn_names:
            if not saw_any_step:
                return None, 0, 0
            if msg.get("role") == "assistant":
                break
            continue

        saw_any_step = True
        if same_tool_active:
            for nm in reversed(turn_names):
                if tail_tool_name is None:
                    tail_tool_name = nm
                    same_tool_count = 1
                elif nm == tail_tool_name:
                    same_tool_count += 1
                else:
                    same_tool_active = False
                    break

        if stagnant_active:
            if any(is_mutating_tool(nm, mutating_set) for nm in turn_names):
                stagnant_active = False
            else:
                stagnant_steps += 1

        if not same_tool_active and not stagnant_active:
            break

    return tail_tool_name, same_tool_count, stagnant_steps


DEFAULT_PRICES: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o1": {"input": 15.00, "output": 60.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "default": {"input": 1.00, "output": 2.00},
}


def lookup_model_pricing(
    model: str,
    prices: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Lookup token pricing for a given model name with prefix and fallback handling."""
    table = prices if prices is not None else DEFAULT_PRICES
    if not model:
        return table.get("default", {"input": 1.00, "output": 2.00})

    model_clean = model.strip().lower()

    if model_clean in table:
        return table[model_clean]

    for key, rates in table.items():
        if key != "default" and model_clean.startswith(key):
            return rates

    short_name = model_clean.split("/")[-1]
    if short_name in table:
        return table[short_name]
    for key, rates in table.items():
        if key != "default" and short_name.startswith(key):
            return rates

    return table.get("default", {"input": 1.00, "output": 2.00})


def calculate_avoided_cost(
    model: str,
    prompt_tokens: int = 0,
    avoided_completion_tokens: int = 500,
    prices: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Calculate estimated money saved for a blocked LLM request.

    Assumes avoided generation of completion tokens (default 500) plus prompt tokens,
    multiplied by model input and output rates per 1,000,000 tokens.
    """
    pricing = lookup_model_pricing(model, prices=prices)
    input_rate = pricing.get("input", 1.00)
    output_rate = pricing.get("output", 2.00)

    prompt_cost = (max(0, prompt_tokens) * input_rate) / 1_000_000.0
    completion_cost = (max(0, avoided_completion_tokens) * output_rate) / 1_000_000.0
    return prompt_cost + completion_cost


_MISSING = object()


class ExpiringLRUDict(OrderedDict):
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600.0,
        default_factory: Optional[Callable[[], Any]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.default_factory = default_factory
        self._timestamps: Dict[Any, float] = {}
        self._lock = RLock()
        super().__init__()
        if args or kwargs:
            self.update(*args, **kwargs)

    def _delete_raw(self, key: Any) -> None:
        """Delete a key directly via OrderedDict.__delitem__ without calling OrderedDict.pop.

        In Python 3.10, OrderedDict.pop(self, key, default) internally invokes
        self.__contains__(key) and self.__getitem__(key) in C, causing infinite
        recursion if called from __contains__ or __getitem__.
        """
        self._timestamps.pop(key, None)
        if super().__contains__(key):
            super().__delitem__(key)

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            if not super().__contains__(key):
                if self.default_factory is not None:
                    val = self.default_factory()
                    self[key] = val
                    return val
                raise KeyError(key)

            now = time.time()
            if now - self._timestamps.get(key, 0.0) > self.ttl_seconds:
                self._delete_raw(key)
                if self.default_factory is not None:
                    val = self.default_factory()
                    self[key] = val
                    return val
                raise KeyError(key)

            self._timestamps[key] = now
            try:
                super().move_to_end(key)
            except KeyError:
                pass
            return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._lock:
            now = time.time()
            self._prune(now)
            super().__setitem__(key, value)
            self._timestamps[key] = now
            try:
                super().move_to_end(key)
            except KeyError:
                pass

            while super().__len__() > self.max_size:
                oldest = next(super().__iter__())
                self._delete_raw(oldest)

    def __delitem__(self, key: Any) -> None:
        with self._lock:
            if not super().__contains__(key):
                self._timestamps.pop(key, None)
                raise KeyError(key)
            self._delete_raw(key)

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            if not super().__contains__(key):
                return False
            now = time.time()
            if now - self._timestamps.get(key, 0.0) > self.ttl_seconds:
                self._delete_raw(key)
                return False
            return True

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            if not super().__contains__(key):
                return default
            now = time.time()
            if now - self._timestamps.get(key, 0.0) > self.ttl_seconds:
                self._delete_raw(key)
                return default
            self._timestamps[key] = now
            try:
                super().move_to_end(key)
            except KeyError:
                pass
            return super().__getitem__(key)

    def pop(self, key: Any, default: Any = _MISSING) -> Any:
        with self._lock:
            if not super().__contains__(key):
                self._timestamps.pop(key, None)
                if default is not _MISSING:
                    return default
                raise KeyError(key)
            ts = self._timestamps.pop(key, 0.0)
            val = super().__getitem__(key)
            super().__delitem__(key)
            if time.time() - ts > self.ttl_seconds:
                if default is not _MISSING:
                    return default
                raise KeyError(key)
            return val

    def _prune(self, now: float) -> None:
        expired = [
            k for k, ts in list(self._timestamps.items())
            if now - ts > self.ttl_seconds
        ]
        for k in expired:
            self._delete_raw(k)

    def clear(self) -> None:
        with self._lock:
            self._timestamps.clear()
            super().clear()

    def touch(self, key: Any) -> None:
        with self._lock:
            now = time.time()
            if super().__contains__(key):
                try:
                    super().move_to_end(key)
                except KeyError:
                    pass
                self._timestamps[key] = now

    def prune_expired(self) -> None:
        with self._lock:
            self._prune(time.time())


class CircuitBreaker:
    """In-memory circuit breaker tracking spending windows, loop detection, burn rate, and manual kill switch."""

    def __init__(
        self,
        hourly_limit: float = 5.0,
        daily_limit: float = 50.0,
        loop_threshold: int = 3,
        tool_loop_threshold: int = 3,
        max_tool_repeats: int = 3,
        max_stagnant_steps: int = 6,
        mutating_tools: Optional[Iterable[str]] = None,
        loop_window_size: int = 20,
        loop_window_seconds: float = 60.0,
        prices_path: Optional[Path] = None,
        kill_switch: bool = False,
        enable_notifications: bool = True,
        active_profile: str = "careful",
        max_burn_rate: Optional[str] = None,
        enable_context_ballooning: bool = True,
        dry_run: bool = False,
        block_secrets: bool = False,
        redact_secrets: bool = False,
        session_limit: float = 0.0,
        max_active_sessions: int = 1000,
        session_ttl_seconds: float = 3600.0,
    ):
        self._lock = threading.RLock()
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.loop_threshold = loop_threshold
        self.tool_loop_threshold = tool_loop_threshold
        self.max_tool_repeats = max_tool_repeats
        self.max_stagnant_steps = max_stagnant_steps
        self._custom_mutating_tools: Optional[frozenset[str]] = (
            frozenset(t.strip().lower() for t in mutating_tools if isinstance(t, str) and t.strip())
            if mutating_tools is not None
            else None
        )
        self.loop_window_size = loop_window_size
        self.loop_window_seconds = loop_window_seconds
        self.kill_switch_active = kill_switch
        self.prices_path = prices_path
        self.enable_notifications = enable_notifications
        self.active_profile = active_profile
        self.max_burn_rate = max_burn_rate
        self.enable_context_ballooning = enable_context_ballooning
        self.dry_run = dry_run
        self.block_secrets = block_secrets
        self.redact_secrets = redact_secrets
        self.session_limit = session_limit
        self.max_active_sessions = max_active_sessions
        self.session_ttl_seconds = session_ttl_seconds

        self.burn_rate_limit: Optional[float] = None
        self.burn_rate_window_seconds: Optional[float] = None
        if max_burn_rate:
            parsed = parse_burn_rate(max_burn_rate)
            if parsed:
                self.burn_rate_limit, self.burn_rate_window_seconds = parsed

        self._spend_records: Deque[Tuple[float, float]] = collections.deque()
        self.recent_hashes: Deque[Tuple[float, str]] = collections.deque(maxlen=loop_window_size)
        self._session_hashes: ExpiringLRUDict = ExpiringLRUDict(
            max_size=max_active_sessions,
            ttl_seconds=session_ttl_seconds,
            default_factory=lambda: collections.deque(maxlen=10),
        )
        self._session_history: ExpiringLRUDict = ExpiringLRUDict(
            max_size=max_active_sessions,
            ttl_seconds=session_ttl_seconds,
            default_factory=lambda: collections.deque(maxlen=20),
        )
        self._tool_history: ExpiringLRUDict = ExpiringLRUDict(
            max_size=max_active_sessions,
            ttl_seconds=session_ttl_seconds,
            default_factory=lambda: collections.deque(maxlen=10),
        )
        self._tool_repeat_state: ExpiringLRUDict = ExpiringLRUDict(
            max_size=max_active_sessions,
            ttl_seconds=session_ttl_seconds,
        )
        self._stagnant_steps: ExpiringLRUDict = ExpiringLRUDict(
            max_size=max_active_sessions,
            ttl_seconds=session_ttl_seconds,
        )
        self._session_spend: ExpiringLRUDict = ExpiringLRUDict(
            max_size=max_active_sessions,
            ttl_seconds=session_ttl_seconds,
        )
        self._session_last_seen: ExpiringLRUDict = ExpiringLRUDict(
            max_size=max_active_sessions,
            ttl_seconds=session_ttl_seconds,
        )
        self.prices: Dict[str, Dict[str, float]] = self._load_prices()

    @property
    def _hash_history(self) -> List[str]:
        """Backward compatibility alias returning list of hash strings."""
        return [h for _, h in self.recent_hashes]

    @property
    def burn_rate_records(self) -> Deque[Tuple[float, float]]:
        """Access in-memory sliding spend records."""
        return self._spend_records

    @burn_rate_records.setter
    def burn_rate_records(self, val: Any) -> None:
        self._spend_records = collections.deque(val)

    def _load_prices(self) -> Dict[str, Dict[str, float]]:
        """Load pricing table from dynamic registry, custom prices.json, or fallback defaults."""
        try:
            from tokenguard.pricing import get_dynamic_pricing_table
            return get_dynamic_pricing_table(self.prices_path)
        except Exception:
            if self.prices_path and self.prices_path.exists():
                try:
                    with open(self.prices_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
            return dict(DEFAULT_PRICES)

    def get_model_pricing(self, model: str) -> Dict[str, float]:
        """Lookup token pricing for a given model name with prefix and fallback handling."""
        return lookup_model_pricing(model, prices=self.prices)

    def compute_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate total USD cost for prompt and completion tokens."""
        pricing = self.get_model_pricing(model)
        input_rate = pricing.get("input", 1.00)
        output_rate = pricing.get("output", 2.00)

        input_cost = (prompt_tokens * input_rate) / 1_000_000.0
        output_cost = (completion_tokens * output_rate) / 1_000_000.0
        return round(input_cost + output_cost, 8)

    def set_dry_run(self, enabled: bool) -> None:
        """Toggle dry-run / observation mode on the circuit breaker."""
        self.dry_run = bool(enabled)

    def _touch_session(self, session_id: str) -> None:
        if not session_id:
            return
        now = time.time()
        self._session_last_seen[session_id] = now
        self._evict_idle_sessions(now)

    def _evict_idle_sessions(self, now: Optional[float] = None) -> None:
        t = now if now is not None else time.time()
        expired = [
            sid for sid, ts in list(self._session_last_seen._timestamps.items())
            if t - ts > self.session_ttl_seconds
        ]
        for sid in expired:
            self._evict_session(sid)
        while len(self._session_last_seen) > self.max_active_sessions:
            oldest_sid = next(iter(self._session_last_seen))
            self._evict_session(oldest_sid)

    def _evict_session(self, session_id: str) -> None:
        self._session_last_seen.pop(session_id, None)
        self._session_hashes.pop(session_id, None)
        self._session_history.pop(session_id, None)
        self._tool_history.pop(session_id, None)
        self._tool_repeat_state.pop(session_id, None)
        self._stagnant_steps.pop(session_id, None)
        self._session_spend.pop(session_id, None)

    def get_session_spend(self, session_id: str) -> float:
        with self._lock:
            self._touch_session(session_id)
            return float(self._session_spend.get(session_id, 0.0))

    def record_session_spend(self, session_id: str, cost: float) -> None:
        with self._lock:
            self._touch_session(session_id)
            self._session_spend[session_id] = self._session_spend.get(session_id, 0.0) + cost

    def calculate_avoided_cost(
        self,
        model: str,
        prompt_tokens: int = 0,
        avoided_completion_tokens: int = 500,
    ) -> float:
        """Calculate avoided cost for a blocked request using instance prices."""
        return calculate_avoided_cost(
            model=model,
            prompt_tokens=prompt_tokens,
            avoided_completion_tokens=avoided_completion_tokens,
            prices=self.prices,
        )

    def calculate_saved_telemetry(
        self,
        model: str,
        messages: Any = None,
        session_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        avoided_completion_tokens: int = 500,
    ) -> Tuple[int, float]:
        """Calculate estimated tokens and USD saved by intercepting runaway execution.

        Returns:
            (tokens_saved, usd_saved)
        """
        if prompt_tokens is not None and prompt_tokens > 0:
            p_tokens = prompt_tokens
        elif messages is not None:
            p_tokens = self.estimate_prompt_tokens(messages)
        else:
            p_tokens = 150

        sess = session_id or "default"
        history = self._session_history.get(sess)
        if history:
            recent_p = [h["prompt_tokens"] for h in history if "prompt_tokens" in h]
            if recent_p:
                avg_p = int(sum(recent_p) / len(recent_p))
                if avg_p > p_tokens:
                    p_tokens = avg_p

        total_tokens_saved = p_tokens + avoided_completion_tokens
        usd_saved = round(
            self.calculate_avoided_cost(
                model=model,
                prompt_tokens=p_tokens,
                avoided_completion_tokens=avoided_completion_tokens,
            ),
            4,
        )
        return total_tokens_saved, usd_saved

    @staticmethod
    def compute_prompt_hash(messages: Any) -> str:
        """Generate SHA-256 hash of the input messages for deterministic loop detection."""
        try:
            if isinstance(messages, (list, dict)):
                serialized = json.dumps(messages, sort_keys=True)
            else:
                serialized = str(messages)
        except Exception:
            serialized = str(messages)

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def estimate_prompt_tokens(self, messages: Any) -> int:
        """Fast heuristic estimation of prompt token count before making upstream API call."""
        try:
            if isinstance(messages, list):
                total_chars = 0
                for msg in messages:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            total_chars += len(content)
                        elif isinstance(content, list):
                            total_chars += sum(len(str(item)) for item in content)
                    else:
                        total_chars += len(str(msg))
                return max(1, total_chars // 4)
            elif isinstance(messages, str):
                return max(1, len(messages) // 4)
        except Exception:
            pass
        return 100

    def estimate_cost(self, model: str, messages: Any) -> float:
        """Estimate request cost for pre-flight budget checks."""
        est_tokens = self.estimate_prompt_tokens(messages)
        pricing = self.get_model_pricing(model)
        input_rate = pricing.get("input", 1.00)
        output_rate = pricing.get("output", 2.00)
        return ((est_tokens * input_rate) + (50 * output_rate)) / 1_000_000.0

    def _prune_old_spends(self, now: float) -> None:
        """Remove spend records older than 24 hours."""
        cutoff_24h = now - 86400.0
        while self._spend_records and self._spend_records[0][0] < cutoff_24h:
            self._spend_records.popleft()

    def _prune_old_hashes(self, now: float) -> None:
        """Remove prompt hashes older than loop_window_seconds (default 60s)."""
        cutoff = now - self.loop_window_seconds
        while self.recent_hashes and self.recent_hashes[0][0] < cutoff:
            self.recent_hashes.popleft()
        for sess, queue in list(self._session_hashes.items()):
            while queue and queue[0][0] < cutoff:
                queue.popleft()

    def get_sliding_spend(self, window_seconds: float = 3600.0) -> float:
        """Get total spend in the last `window_seconds`."""
        now = time.time()
        self._prune_old_spends(now)
        cutoff = now - window_seconds
        return sum(cost for ts, cost in self._spend_records if ts >= cutoff)

    def get_hourly_spend(self) -> float:
        """Get sliding 1-hour total spend."""
        return self.get_sliding_spend(3600.0)

    def get_daily_spend(self) -> float:
        """Get sliding 24-hour total spend."""
        return self.get_sliding_spend(86400.0)

    def get_burn_rate_spend(self, window_seconds: float) -> Tuple[float, int]:
        """Calculate cumulative spend and duration in seconds of earliest spend in sliding window.

        Returns:
            (cumulative_spend, elapsed_seconds)
        """
        now = time.time()
        self._prune_old_spends(now)
        cutoff = now - window_seconds
        records = [(ts, cost) for ts, cost in self._spend_records if ts >= cutoff]
        if not records:
            return 0.0, 0
        total_spend = sum(cost for _, cost in records)
        earliest_ts = min(ts for ts, _ in records)
        elapsed_seconds = max(1, int(now - earliest_ts))
        return total_spend, elapsed_seconds

    def check_loop(self, incoming_hash: str, session_id: Optional[str] = None) -> bool:
        """Check if incoming prompt hash triggers multi-length cycle loop detector.

        Checks:
        - Length 1 cycle: A-A-A (three identical consecutive calls)
        - Length 2 cycle: A-B-A-B (alternating calls repeating twice)
        - Length 3 cycle: A-B-C-A-B-C (three-call sequence repeating twice)
        """
        return self.check_loop_details(incoming_hash, session_id=session_id) is not None

    def check_loop_details(
        self,
        incoming_hash: str,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return cycle details if incoming prompt hash matches a repeating cycle, else None."""
        if self.active_profile == "passive" or self.loop_threshold <= 1:
            return None

        now = time.time()
        self._prune_old_hashes(now)

        hashes = [h for _, h in self.recent_hashes]
        cycle = detect_cycle_in_sequence(hashes + [incoming_hash], length_1_repeats=self.loop_threshold)
        if cycle is not None:
            return cycle

        sess = session_id or "default"
        if sess in self._session_hashes:
            sess_hashes = [h for _, h in self._session_hashes[sess]]
            cycle = detect_cycle_in_sequence(sess_hashes + [incoming_hash], length_1_repeats=self.loop_threshold)
            if cycle is not None:
                return cycle

        return None

    def check_tool_loop(
        self,
        session_id: str,
        tool_name: str,
        arguments: Any,
        model: Optional[str] = None,
    ) -> None:
        """Check if an incoming tool invocation triggers multi-length cycle detection.

        Checks:
        - Length 1 cycle: A-A-A (three identical consecutive calls)
        - Length 2 cycle: A-B-A-B (alternating calls repeating twice)
        - Length 3 cycle: A-B-C-A-B-C (three-call sequence repeating twice)
        """
        if self.active_profile == "passive" or self.tool_loop_threshold <= 1:
            return

        if self.block_secrets:
            secret_match = scan_secrets_in_obj(arguments)
            if secret_match:
                t_saved, u_saved = self.calculate_saved_telemetry(model, session_id=session_id) if model else (None, None)
                raise SecretLeakError(
                    pattern_type=secret_match,
                    tokens_saved=t_saved,
                    usd_saved=u_saved,
                )

        canonical_args, _ = normalize_tool_arguments(arguments)
        now = time.time()
        incoming_entry = (now, tool_name, canonical_args)

        history = list(self._tool_history[session_id])
        prospective = history + [incoming_entry]

        def _tool_match(entry_a: Tuple[float, str, str], entry_b: Tuple[float, str, str]) -> bool:
            name_a, args_a = entry_a[1], entry_a[2]
            name_b, args_b = entry_b[1], entry_b[2]
            if name_a != name_b:
                return False
            if args_a == args_b:
                return True
            return compute_argument_similarity(args_a, args_b) >= 0.90

        cycle = detect_cycle_in_sequence(
            prospective,
            match_fn=_tool_match,
            length_1_repeats=self.tool_loop_threshold,
        )

        if cycle is not None:
            cycle_len = cycle["cycle_length"]
            pattern_entries = cycle["pattern"]
            pattern_names = [e[1] for e in pattern_entries]
            matched_entries = cycle["matched_items"]
            earliest_ts = matched_entries[0][0]
            elapsed = max(0.1, round(now - earliest_ts, 1))

            if cycle_len == 1:
                msg = f"TokenGuard intercepted repeated tool call loop: {tool_name}"
                notify_msg = f"[TOOL RETRY LOOP: HARD STOP] Tool '{tool_name}' repeated {self.tool_loop_threshold}x within {elapsed}s"
            elif cycle_len == 2:
                cycle_str = " -> ".join(pattern_names)
                msg = f"TokenGuard intercepted alternating tool call loop (Length 2: {cycle_str})"
                notify_msg = f"[TOOL CYCLE LOOP: HARD STOP] Alternating tool loop ({cycle_str}) repeated twice within {elapsed}s"
            else:
                cycle_str = " -> ".join(pattern_names)
                msg = f"TokenGuard intercepted cyclical tool call loop (Length 3: {cycle_str})"
                notify_msg = f"[TOOL CYCLE LOOP: HARD STOP] Three-step tool loop ({cycle_str}) repeated twice within {elapsed}s"

            if self.enable_notifications:
                send_block_notification(
                    title="TokenGuard Tool Loop Alert",
                    message=notify_msg,
                )

            if model is not None:
                t_saved, u_saved = self.calculate_saved_telemetry(
                    model=model,
                    session_id=session_id,
                )
            else:
                t_saved, u_saved = None, None

            raise ToolPingPongError(
                tool_name=tool_name,
                message=msg,
                repeats=self.tool_loop_threshold if cycle_len == 1 else 2,
                repeat_count=self.tool_loop_threshold if cycle_len == 1 else 2,
                tool_args=canonical_args,
                elapsed_seconds=elapsed,
                cycle_length=cycle_len,
                pattern=pattern_names,
                tokens_saved=t_saved,
                usd_saved=u_saved,
            )

    def record_tool_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: Any,
    ) -> None:
        """Record an executed tool invocation into the session's rolling window."""
        canonical_args, _ = normalize_tool_arguments(arguments)
        with self._lock:
            self._tool_history[session_id].append((time.time(), tool_name, canonical_args))

    def is_mutating_action(self, tool_name: str) -> bool:
        """Return True if tool_name is classified as state-changing / mutating."""
        return is_mutating_tool(tool_name, self._custom_mutating_tools)

    def check_tool_retry_ceiling(
        self,
        session_id: str,
        tool_name: str,
        model: Optional[str] = None,
        messages: Any = None,
        min_streak: int = 1,
    ) -> None:
        """Check if consecutive invocations of tool_name (ignoring arguments) reach max_tool_repeats.

        Non-poisoning: does not mutate session state when tripping.
        """
        if self.active_profile == "passive" or self.max_tool_repeats <= 0:
            return
        if not tool_name:
            return

        sess = session_id or "default"
        now = time.time()
        with self._lock:
            prev = self._tool_repeat_state.get(sess)
            if prev is not None:
                last_name, count, last_ts = prev
                if self.loop_window_seconds > 0 and (now - last_ts) > self.loop_window_seconds:
                    sess_prospective = 1
                elif last_name == tool_name:
                    sess_prospective = count + 1
                else:
                    sess_prospective = 1
            else:
                sess_prospective = 1

        effective_count = sess_prospective if sess_prospective >= min_streak else min_streak
        if effective_count >= self.max_tool_repeats:
            msg = f"ToolRetryCeilingTripped: {tool_name} called {effective_count} times consecutively"
            if self.enable_notifications:
                send_block_notification(
                    title="TokenGuard Tool Retry Ceiling Alert",
                    message=msg,
                )
            if model is not None or messages is not None:
                t_saved, u_saved = self.calculate_saved_telemetry(
                    model=model or "default",
                    messages=messages,
                    session_id=sess,
                )
            else:
                t_saved, u_saved = None, None
            raise ToolRetryCeilingError(
                tool_name=tool_name,
                repeats=effective_count,
                message=msg,
                tokens_saved=t_saved,
                usd_saved=u_saved,
            )

    def record_tool_name(self, session_id: str, tool_name: str) -> None:
        """Update per-session consecutive tool counter after pre-flight checks pass."""
        if not tool_name:
            return
        sess = session_id or "default"
        now = time.time()
        with self._lock:
            prev = self._tool_repeat_state.get(sess)
            if prev is not None:
                last_name, count, last_ts = prev
                if self.loop_window_seconds > 0 and (now - last_ts) > self.loop_window_seconds:
                    self._tool_repeat_state[sess] = (tool_name, 1, now)
                elif last_name == tool_name:
                    self._tool_repeat_state[sess] = (tool_name, count + 1, now)
                else:
                    self._tool_repeat_state[sess] = (tool_name, 1, now)
            else:
                self._tool_repeat_state[sess] = (tool_name, 1, now)

    def check_stagnation(
        self,
        session_id: str,
        tool_names: Optional[Sequence[str]] = None,
        has_task_update: bool = False,
        has_user_output: bool = False,
        model: Optional[str] = None,
        messages: Any = None,
        min_steps: int = 0,
    ) -> None:
        """Check if session has performed >= max_stagnant_steps without state change or output milestones.

        Non-poisoning: does not increment stored counter on a tripped request.
        """
        if self.active_profile == "passive" or self.max_stagnant_steps <= 0:
            return
        if has_task_update or has_user_output:
            return
        if not tool_names:
            return
        if any(self.is_mutating_action(name) for name in tool_names):
            return

        sess = session_id or "default"
        with self._lock:
            current_stagnant = self._stagnant_steps.get(sess, 0)
            sess_prospective = current_stagnant + 1

        effective_steps = sess_prospective if sess_prospective >= min_steps else min_steps
        if effective_steps >= self.max_stagnant_steps:
            msg = f"SessionStagnationTripped: {effective_steps} steps without state change"
            if self.enable_notifications:
                send_block_notification(
                    title="TokenGuard Session Stagnation Alert",
                    message=msg,
                )
            if model is not None or messages is not None:
                t_saved, u_saved = self.calculate_saved_telemetry(
                    model=model or "default",
                    messages=messages,
                    session_id=sess,
                )
            else:
                t_saved, u_saved = None, None
            raise SessionStagnationError(
                steps=effective_steps,
                message=msg,
                tokens_saved=t_saved,
                usd_saved=u_saved,
            )

    def record_stagnation_step(
        self,
        session_id: str,
        tool_names: Optional[Sequence[str]] = None,
        has_task_update: bool = False,
        has_user_output: bool = False,
    ) -> None:
        """Update session stagnation counter after pre-flight checks succeed."""
        sess = session_id or "default"
        with self._lock:
            if has_task_update or has_user_output or not tool_names:
                self._stagnant_steps[sess] = 0
                return
            if any(self.is_mutating_action(name) for name in tool_names):
                self._stagnant_steps[sess] = 0
                return
            self._stagnant_steps[sess] = self._stagnant_steps.get(sess, 0) + 1

    def _check_session_exact_or_cycle_tool_loop(
        self,
        session_id: str,
        tool_name: str,
        arguments: Any,
        model: Optional[str] = None,
    ) -> None:
        """Check session tool history for exact duplicate args (length 1) or alternating cycles (length 2/3)."""
        if self.active_profile == "passive" or self.tool_loop_threshold <= 1:
            return

        if self.block_secrets:
            secret_match = scan_secrets_in_obj(arguments)
            if secret_match:
                t_saved, u_saved = self.calculate_saved_telemetry(model, session_id=session_id) if model else (None, None)
                raise SecretLeakError(
                    pattern_type=secret_match,
                    tokens_saved=t_saved,
                    usd_saved=u_saved,
                )

        canonical_args, _ = normalize_tool_arguments(arguments)
        now = time.time()
        incoming_entry = (now, tool_name, canonical_args)
        with self._lock:
            history = list(self._tool_history[session_id])
        prospective = history + [incoming_entry]

        if len(prospective) >= self.tool_loop_threshold:
            tail_1 = prospective[-self.tool_loop_threshold:]
            first = tail_1[0]
            if all(e[1] == first[1] and e[2] == first[2] for e in tail_1[1:]):
                elapsed = max(0.1, round(now - first[0], 1))
                msg = f"TokenGuard intercepted repeated tool call loop: {tool_name}"
                if self.enable_notifications:
                    send_block_notification(
                        title="TokenGuard Tool Loop Alert",
                        message=f"[TOOL RETRY LOOP: HARD STOP] Tool '{tool_name}' repeated {self.tool_loop_threshold}x within {elapsed}s",
                    )
                t_saved, u_saved = (
                    self.calculate_saved_telemetry(model=model, session_id=session_id)
                    if model is not None
                    else (None, None)
                )
                raise ToolPingPongError(
                    tool_name=tool_name,
                    message=msg,
                    repeats=self.tool_loop_threshold,
                    repeat_count=self.tool_loop_threshold,
                    tool_args=canonical_args,
                    elapsed_seconds=elapsed,
                    cycle_length=1,
                    pattern=[tool_name],
                    tokens_saved=t_saved,
                    usd_saved=u_saved,
                )

        def _cycle_match(entry_a: Tuple[float, str, str], entry_b: Tuple[float, str, str]) -> bool:
            if entry_a[1] != entry_b[1]:
                return False
            if entry_a[2] == entry_b[2]:
                return True
            return compute_argument_similarity(entry_a[2], entry_b[2]) >= 0.90

        cycle = detect_cycle_in_sequence(prospective, match_fn=_cycle_match, length_1_repeats=0)
        if cycle is not None:
            cycle_len = cycle["cycle_length"]
            pattern_names = [e[1] for e in cycle["pattern"]]
            earliest_ts = cycle["matched_items"][0][0]
            elapsed = max(0.1, round(now - earliest_ts, 1))
            cycle_str = " -> ".join(pattern_names)
            if cycle_len == 2:
                msg = f"TokenGuard intercepted alternating tool call loop (Length 2: {cycle_str})"
            else:
                msg = f"TokenGuard intercepted cyclical tool call loop (Length 3: {cycle_str})"
            if self.enable_notifications:
                send_block_notification(title="TokenGuard Tool Loop Alert", message=msg)
            t_saved, u_saved = (
                self.calculate_saved_telemetry(model=model, session_id=session_id)
                if model is not None
                else (None, None)
            )
            raise ToolPingPongError(
                tool_name=tool_name,
                message=msg,
                repeats=2,
                repeat_count=2,
                tool_args=canonical_args,
                elapsed_seconds=elapsed,
                cycle_length=cycle_len,
                pattern=pattern_names,
                tokens_saved=t_saved,
                usd_saved=u_saved,
            )

    def check_message_tool_loop(
        self,
        messages: Any,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Check if the messages history itself contains repeated exact tool calls or multi-tool cycles."""
        if self.active_profile == "passive" or self.tool_loop_threshold <= 1:
            return

        extracted = extract_tool_calls(messages)
        if len(extracted) < min(self.tool_loop_threshold, 4):
            return

        cycle: Optional[Dict[str, Any]] = None
        if len(extracted) >= self.tool_loop_threshold:
            tail_1 = extracted[-self.tool_loop_threshold:]
            first = tail_1[0]
            if all(item[0] == first[0] and item[1] == first[1] for item in tail_1[1:]):
                cycle = {
                    "cycle_length": 1,
                    "repeats": self.tool_loop_threshold,
                    "pattern": [first],
                    "matched_items": list(tail_1),
                }

        if cycle is None:
            def _msg_tool_match(a: Tuple[str, str, Any], b: Tuple[str, str, Any]) -> bool:
                name_a, args_a = a[0], a[1]
                name_b, args_b = b[0], b[1]
                if name_a != name_b:
                    return False
                if args_a == args_b:
                    return True
                return compute_argument_similarity(args_a, args_b) >= 0.90

            cycle = detect_cycle_in_sequence(
                extracted,
                match_fn=_msg_tool_match,
                length_1_repeats=0,
            )

        if cycle is not None:
            cycle_len = cycle["cycle_length"]
            pattern = [e[0] for e in cycle["pattern"]]
            tool_name = extracted[-1][0]

            if cycle_len == 1:
                msg = f"TokenGuard intercepted repeated tool call loop: {tool_name}"
            elif cycle_len == 2:
                cycle_str = " -> ".join(pattern)
                msg = f"TokenGuard intercepted alternating tool call loop (Length 2: {cycle_str})"
            else:
                cycle_str = " -> ".join(pattern)
                msg = f"TokenGuard intercepted cyclical tool call loop (Length 3: {cycle_str})"

            if self.enable_notifications:
                send_block_notification(
                    title="TokenGuard Tool Loop Alert",
                    message=msg,
                )

            t_saved, u_saved = self.calculate_saved_telemetry(
                model=model or "default",
                messages=messages,
                session_id=session_id,
            )
            raise ToolPingPongError(
                tool_name=tool_name,
                message=msg,
                repeats=self.tool_loop_threshold if cycle_len == 1 else 2,
                repeat_count=self.tool_loop_threshold if cycle_len == 1 else 2,
                cycle_length=cycle_len,
                pattern=pattern,
                tokens_saved=t_saved,
                usd_saved=u_saved,
            )

    def check_context_ballooning(
        self,
        session_id: str,
        incoming_tokens: int,
        has_task_update: bool = False,
        cache_hit: bool = False,
        model: Optional[str] = None,
    ) -> None:
        """Check if incoming request represents abnormal runaway context expansion.

        Trips if:
        1. Context ballooning guard is enabled and active profile is not 'passive'
        2. At least 2 prior requests exist in this session sequence (making 3 consecutive requests)
        3. None of the 3 requests had an explicit task status update or cache hit
        4. prompt_tokens expands by > 300% across the 3 consecutive requests (monotonically growing)
        """
        if self.active_profile == "passive" or not self.enable_context_ballooning:
            return

        if has_task_update or cache_hit:
            return

        history = self._session_history.get(session_id)
        if not history or len(history) < 2:
            return

        r1 = history[-2]
        r2 = history[-1]

        if r1.get("has_task_update") or r1.get("cache_hit") or r2.get("has_task_update") or r2.get("cache_hit"):
            return

        tokens_1 = r1.get("prompt_tokens", 0)
        tokens_2 = r2.get("prompt_tokens", 0)
        tokens_3 = incoming_tokens

        if tokens_1 >= 50 and tokens_3 > tokens_2 and tokens_2 > tokens_1:
            growth_ratio = (tokens_3 - tokens_1) / float(tokens_1)
            if growth_ratio > 3.0:
                growth_pct = int(growth_ratio * 100)
                msg = (
                    f"TokenGuard Circuit Breaker Tripped: Context ballooning detected - Runaway context expansion "
                    f"(prompt tokens expanded by {growth_pct}% across 3 consecutive requests from {tokens_1} to {tokens_3}). "
                    f"Hard limit: 300%."
                )
                if self.enable_notifications:
                    send_block_notification(
                        title="TokenGuard Context Ballooning Alert",
                        message=msg,
                    )
                t_saved, u_saved = self.calculate_saved_telemetry(
                    model=model or "default",
                    prompt_tokens=tokens_3,
                    session_id=session_id,
                )
                raise ContextBallooningError(
                    message=msg,
                    prompt_tokens=tokens_3,
                    baseline_tokens=tokens_1,
                    growth_rate=round(growth_ratio, 2),
                    tokens_saved=t_saved,
                    usd_saved=u_saved,
                )

    def record_request_hash(self, prompt_hash: str, session_id: Optional[str] = None) -> None:
        """Record an allowed request's prompt hash with current timestamp into sliding history."""
        now = time.time()
        self._prune_old_hashes(now)
        self.recent_hashes.append((now, prompt_hash))
        sess = session_id or "default"
        self._session_hashes[sess].append((now, prompt_hash))

    def remove_last_hash(self, prompt_hash: Optional[str] = None, session_id: Optional[str] = None) -> None:
        """Remove the last recorded prompt hash if upstream call failed."""
        if self.recent_hashes:
            if prompt_hash is None or self.recent_hashes[-1][1] == prompt_hash:
                self.recent_hashes.pop()
        sess = session_id or "default"
        if sess in self._session_hashes and self._session_hashes[sess]:
            if prompt_hash is None or self._session_hashes[sess][-1][1] == prompt_hash:
                self._session_hashes[sess].pop()

    def remove_last_session_entry(self, session_id: Optional[str] = None) -> None:
        """Remove the last session prompt entry if upstream call failed or was rejected."""
        sess = session_id or "default"
        if sess in self._session_history and self._session_history[sess]:
            self._session_history[sess].pop()

    def check_request(
        self,
        messages: Any,
        model: str,
        session_id: Optional[str] = "default",
        has_task_update: bool = False,
        cache_hit: bool = False,
        incoming_tokens: Optional[int] = None,
        tool_calls: Optional[List[Any]] = None,
    ) -> str:
        """Run pre-flight checks against circuit breaker guards.

        Raises:
            KillSwitchActiveError: If manual kill switch is enabled.
            LoopDetectedError: If the same prompt hash repeated >= threshold times consecutively.
            ToolPingPongError: If exact repeated tool calls or multi-tool cycles detected.
            ToolRetryCeilingError: If the same tool_name is invoked >= max_tool_repeats consecutively.
            SessionStagnationError: If >= max_stagnant_steps occur without state change.
            ContextBallooningError: If prompt tokens expand > 300% across 3 consecutive requests.
            BurnRateExceededError: If sliding window burn rate limit is exceeded.
            BudgetExceededError: If sliding hourly or daily budget limit is exceeded.

        Returns:
            The calculated prompt_hash if allowed.
        """
        with self._lock:
            active_tools, has_explicit_args = _extract_active_step_tools(messages, tool_calls)
            if tool_calls and active_tools:
                prompt_hash = self.compute_prompt_hash({"messages": messages, "tool_calls": tool_calls})
            else:
                prompt_hash = self.compute_prompt_hash(messages)
            sess = session_id or "default"

            if self.block_secrets:
                secret_match = scan_secrets_in_obj(messages)
                if not secret_match and tool_calls:
                    secret_match = scan_secrets_in_obj(tool_calls)
                if secret_match:
                    t_saved, u_saved = self.calculate_saved_telemetry(model, messages, session_id=sess)
                    raise SecretLeakError(
                        pattern_type=secret_match,
                        tokens_saved=t_saved,
                        usd_saved=u_saved,
                    )

            if self.kill_switch_active:
                t_saved, u_saved = self.calculate_saved_telemetry(model, messages, session_id=sess)
                raise KillSwitchActiveError(tokens_saved=t_saved, usd_saved=u_saved)

            if self.active_profile == "passive":
                self.record_request_hash(prompt_hash, session_id=sess)
                return prompt_hash

            if active_tools:
                if has_explicit_args:
                    self.check_message_tool_loop(messages, model=model, session_id=sess)
                    latest_tool_name, _, raw_args = active_tools[-1]
                    self._check_session_exact_or_cycle_tool_loop(sess, latest_tool_name, raw_args, model=model)
                else:
                    latest_tool_name, _, raw_args = active_tools[-1]

                tail_tool, tail_same_cnt, tail_stag_cnt = _compute_message_tail_streaks(
                    messages,
                    self._custom_mutating_tools,
                )
                min_same_streak = tail_same_cnt if tail_tool == latest_tool_name else 1
                self.check_tool_retry_ceiling(
                    sess,
                    latest_tool_name,
                    model=model,
                    messages=messages,
                    min_streak=min_same_streak,
                )
                step_tool_names = [t[0] for t in active_tools]
                self.check_stagnation(
                    sess,
                    tool_names=step_tool_names,
                    has_task_update=has_task_update,
                    model=model,
                    messages=messages,
                    min_steps=tail_stag_cnt,
                )

            loop_cycle = self.check_loop_details(prompt_hash, session_id=sess)
            if loop_cycle is not None:
                cycle_len = loop_cycle.get("cycle_length", 1)
                if cycle_len > 1:
                    msg = f"TokenGuard: infinite agent loop detected (cycle length {cycle_len})"
                else:
                    msg = f"TokenGuard: infinite agent loop detected ({self.loop_threshold} identical consecutive requests)"
                if self.enable_notifications:
                    send_block_notification(
                        title="TokenGuard Alert",
                        message=f"Infinite loop detected & blocked for model '{model}'",
                    )
                t_saved, u_saved = self.calculate_saved_telemetry(model, messages, session_id=sess)
                raise LoopDetectedError(
                    message=msg,
                    cycle_length=cycle_len,
                    tokens_saved=t_saved,
                    usd_saved=u_saved,
                )

            est_tokens = incoming_tokens if incoming_tokens is not None else self.estimate_prompt_tokens(messages)

            self.check_context_ballooning(
                session_id=sess,
                incoming_tokens=est_tokens,
                has_task_update=has_task_update,
                cache_hit=cache_hit,
                model=model,
            )

            if self.burn_rate_limit is not None and self.burn_rate_window_seconds is not None and self.burn_rate_limit > 0:
                window_spend, elapsed_s = self.get_burn_rate_spend(self.burn_rate_window_seconds)
                estimated_cost = self.estimate_cost(model, messages)
                total_projected = window_spend + estimated_cost
                if window_spend >= self.burn_rate_limit or total_projected > self.burn_rate_limit:
                    spend_to_report = window_spend if window_spend >= self.burn_rate_limit else total_projected
                    elapsed_display = max(1, elapsed_s)
                    limit_str = self.max_burn_rate or f"${self.burn_rate_limit:.2f}/{int(self.burn_rate_window_seconds)}s"
                    msg = f"TokenGuard Circuit Breaker Tripped: Burn rate exceeded (${spend_to_report:.2f} in {elapsed_display}s). Hard limit: {limit_str}."
                    if self.enable_notifications:
                        send_block_notification(
                            title="TokenGuard Burn Rate Alert",
                            message=msg,
                        )
                    t_saved, u_saved = self.calculate_saved_telemetry(model, messages, session_id=sess, prompt_tokens=est_tokens)
                    raise BurnRateExceededError(
                        message=msg,
                        current_spend=round(spend_to_report, 2),
                        budget_limit=round(self.burn_rate_limit, 2),
                        tokens_saved=t_saved,
                        usd_saved=u_saved,
                    )

            if self.session_limit > 0 and sess:
                current_session_spend = self.get_session_spend(sess)
                estimated_cost = self.estimate_cost(model, messages)
                projected = current_session_spend + estimated_cost
                if current_session_spend >= self.session_limit or projected > self.session_limit:
                    spend_to_report = current_session_spend if current_session_spend >= self.session_limit else projected
                    t_saved, u_saved = self.calculate_saved_telemetry(model, messages, session_id=sess, prompt_tokens=est_tokens)
                    raise SessionBudgetLimitError(
                        current_spend=spend_to_report,
                        session_limit=self.session_limit,
                        session_id=sess,
                        tokens_saved=t_saved,
                        usd_saved=u_saved,
                    )

            if self.hourly_limit > 0:
                current_hourly_spend = self.get_hourly_spend()
                estimated_cost = self.estimate_cost(model, messages)
                if (current_hourly_spend + estimated_cost) > self.hourly_limit:
                    t_saved, u_saved = self.calculate_saved_telemetry(model, messages, session_id=sess, prompt_tokens=est_tokens)
                    raise BudgetExceededError(
                        "TokenGuard: budget limit exceeded",
                        tokens_saved=t_saved,
                        usd_saved=u_saved,
                    )

            if self.daily_limit > 0:
                current_daily_spend = self.get_daily_spend()
                estimated_cost = self.estimate_cost(model, messages)
                if (current_daily_spend + estimated_cost) > self.daily_limit:
                    t_saved, u_saved = self.calculate_saved_telemetry(model, messages, session_id=sess, prompt_tokens=est_tokens)
                    raise BudgetExceededError(
                        "TokenGuard: daily budget limit exceeded",
                        tokens_saved=t_saved,
                        usd_saved=u_saved,
                    )

            if active_tools:
                latest_tool_name, _, raw_args = active_tools[-1]
                self.record_tool_call(sess, latest_tool_name, raw_args)
                for t_name, _, _ in active_tools:
                    self.record_tool_name(sess, t_name)
                self.record_stagnation_step(
                    sess,
                    tool_names=[t[0] for t in active_tools],
                    has_task_update=has_task_update,
                )
            else:
                self.record_stagnation_step(sess, tool_names=None, has_task_update=True)

            self.record_request_hash(prompt_hash, session_id=sess)
            self._session_history[sess].append({
                "prompt_tokens": est_tokens,
                "has_task_update": has_task_update,
                "cache_hit": cache_hit,
                "timestamp": time.time(),
            })
            return prompt_hash

    def record_success(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: Optional[float] = None,
        prompt_hash: Optional[str] = None,
        session_id: Optional[str] = None,
        cost: Optional[float] = None,
    ) -> None:
        """Record completed request in in-memory spending history."""
        now = time.time()
        final_cost = cost_usd if cost_usd is not None else (cost if cost is not None else 0.0)
        self._spend_records.append((now, final_cost))
        self._prune_old_spends(now)
        sess = session_id or "default"
        if sess:
            self.record_session_spend(sess, final_cost)
        if sess in self._session_history and self._session_history[sess]:
            self._session_history[sess][-1]["prompt_tokens"] = prompt_tokens

    def record_blocked(self, prompt_hash: str) -> None:
        """Explicitly log a blocked intercept without altering loop detection queue."""
        pass

    def clear_hashes(self) -> None:
        """Clear prompt hash history."""
        self.recent_hashes.clear()
        self._session_hashes.clear()

    def reset(self) -> None:
        """Reset all in-memory rolling state (hash history, spend records, session tracking, tool tracking, kill switch)."""
        with self._lock:
            self.recent_hashes.clear()
            self._session_hashes.clear()
            self._spend_records.clear()
            self._session_history.clear()
            self._tool_history.clear()
            self._tool_repeat_state.clear()
            self._stagnant_steps.clear()
            self._session_spend.clear()
            self._session_last_seen.clear()
            self.kill_switch_active = False

    def toggle_kill_switch(self, active: Optional[bool] = None) -> bool:
        """Toggle or explicitly set the kill switch state."""
        with self._lock:
            if active is None:
                self.kill_switch_active = not self.kill_switch_active
            else:
                self.kill_switch_active = bool(active)
            return self.kill_switch_active

    def update_limits(
        self,
        hourly_limit: Optional[float] = None,
        daily_limit: Optional[float] = None,
        loop_threshold: Optional[int] = None,
        tool_loop_threshold: Optional[int] = None,
        max_tool_repeats: Optional[int] = None,
        max_stagnant_steps: Optional[int] = None,
        profile: Optional[str] = None,
        max_burn_rate: Optional[str] = None,
        enable_context_ballooning: Optional[bool] = None,
        session_limit: Optional[float] = None,
    ) -> None:
        """Dynamically update runtime limits and profile."""
        with self._lock:
            if hourly_limit is not None and hourly_limit > 0:
                self.hourly_limit = float(hourly_limit)
            if daily_limit is not None and daily_limit > 0:
                self.daily_limit = float(daily_limit)
            if loop_threshold is not None and loop_threshold >= 0:
                self.loop_threshold = int(loop_threshold)
            if tool_loop_threshold is not None and tool_loop_threshold >= 0:
                self.tool_loop_threshold = int(tool_loop_threshold)
            if max_tool_repeats is not None and max_tool_repeats >= 0:
                self.max_tool_repeats = int(max_tool_repeats)
            if max_stagnant_steps is not None and max_stagnant_steps >= 0:
                self.max_stagnant_steps = int(max_stagnant_steps)
            if session_limit is not None and session_limit >= 0:
                self.session_limit = float(session_limit)
            if profile:
                self.active_profile = str(profile).lower()
            if max_burn_rate is not None:
                self.max_burn_rate = max_burn_rate
                parsed = parse_burn_rate(max_burn_rate)
                if parsed:
                    self.burn_rate_limit, self.burn_rate_window_seconds = parsed
                else:
                    self.burn_rate_limit, self.burn_rate_window_seconds = None, None
            if enable_context_ballooning is not None:
                self.enable_context_ballooning = bool(enable_context_ballooning)

    def seed_initial_spends(self, spend_entries: List[Tuple[float, float]]) -> None:
        """Seed initial spend entries (e.g. from DB on startup) into in-memory tracker."""
        now = time.time()
        cutoff_24h = now - 86400.0
        for ts, cost in spend_entries:
            if ts >= cutoff_24h:
                self._spend_records.append((ts, cost))
        self._prune_old_spends(now)

    def get_state(self) -> Dict[str, Any]:
        """Return snapshot of the circuit breaker state."""
        hourly_spend = self.get_hourly_spend()
        daily_spend = self.get_daily_spend()
        burn_spend, _ = (
            self.get_burn_rate_spend(self.burn_rate_window_seconds)
            if self.burn_rate_window_seconds
            else (0.0, 0)
        )
        return {
            "kill_switch_active": self.kill_switch_active,
            "active_profile": self.active_profile,
            "dry_run": self.dry_run,
            "hourly_limit": self.hourly_limit,
            "daily_limit": self.daily_limit,
            "session_limit": self.session_limit,
            "max_active_sessions": self.max_active_sessions,
            "session_ttl_seconds": self.session_ttl_seconds,
            "active_sessions_count": len(self._session_last_seen),
            "max_burn_rate": self.max_burn_rate,
            "burn_rate_limit": self.burn_rate_limit,
            "burn_rate_window_seconds": self.burn_rate_window_seconds,
            "current_burn_rate_spend": round(burn_spend, 6),
            "current_hourly_spend": round(hourly_spend, 6),
            "current_daily_spend": round(daily_spend, 6),
            "hourly_budget_used_percent": round(
                (hourly_spend / self.hourly_limit * 100.0) if self.hourly_limit > 0 else 0.0, 2
            ),
            "loop_threshold": self.loop_threshold,
            "tool_loop_threshold": self.tool_loop_threshold,
            "max_tool_repeats": self.max_tool_repeats,
            "max_stagnant_steps": self.max_stagnant_steps,
            "loop_window_size": self.loop_window_size,
            "recent_hashes_count": len(self.recent_hashes),
        }
