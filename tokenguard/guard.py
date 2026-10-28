"""Circuit breaker and guard engine for TokenGuard."""

from __future__ import annotations

import collections
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
import time
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

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

    # 1. Length 1 cycle: A-A-A (three identical consecutive calls)
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

    # 2. Length 2 cycle: A-B-A-B (alternating calls repeating twice)
    if n >= 4:
        s0, s1, s2, s3 = items[-4], items[-3], items[-2], items[-1]
        if _eq(s0, s2) and _eq(s1, s3) and not _eq(s0, s1):
            return {
                "cycle_length": 2,
                "repeats": 2,
                "pattern": [s0, s1],
                "matched_items": [s0, s1, s2, s3],
            }

    # 3. Length 3 cycle: A-B-C-A-B-C (three-step sequence repeating twice)
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


def extract_tool_calls(payload_or_messages: Any) -> List[Tuple[str, str, Any]]:
    """Extract tool calls from request payload or messages list.

    Returns a list of (tool_name, normalized_args_string, raw_args).
    """
    results: List[Tuple[str, str, Any]] = []

    def _add(name: Any, args: Any):
        if name and isinstance(name, str) and name.strip():
            norm_str, raw_obj = normalize_tool_arguments(args)
            results.append((name.strip(), norm_str, raw_obj))

    def _process_item(item: Any):
        if not isinstance(item, dict):
            return

        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        _add(fn.get("name"), fn.get("arguments", {}))
                    else:
                        _add(tc.get("name"), tc.get("arguments") or tc.get("input", {}))

        fn_call = item.get("function_call")
        if isinstance(fn_call, dict):
            _add(fn_call.get("name"), fn_call.get("arguments", {}))

        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    _add(block.get("name"), block.get("input", {}))

    if isinstance(payload_or_messages, list):
        for msg in payload_or_messages:
            _process_item(msg)
    elif isinstance(payload_or_messages, dict):
        messages = payload_or_messages.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                _process_item(msg)
        else:
            _process_item(payload_or_messages)

    return results


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


class CircuitBreaker:
    """In-memory circuit breaker tracking spending windows, loop detection, burn rate, and manual kill switch."""

    def __init__(
        self,
        hourly_limit: float = 5.0,
        daily_limit: float = 50.0,
        loop_threshold: int = 3,
        tool_loop_threshold: int = 3,
        loop_window_size: int = 20,
        loop_window_seconds: float = 60.0,
        prices_path: Optional[Path] = None,
        kill_switch: bool = False,
        enable_notifications: bool = True,
        active_profile: str = "careful",
        max_burn_rate: Optional[str] = None,
        enable_context_ballooning: bool = True,
        dry_run: bool = False,
    ):
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.loop_threshold = loop_threshold
        self.tool_loop_threshold = tool_loop_threshold
        self.loop_window_size = loop_window_size
        self.loop_window_seconds = loop_window_seconds
        self.kill_switch_active = kill_switch
        self.prices_path = prices_path
        self.enable_notifications = enable_notifications
        self.active_profile = active_profile
        self.max_burn_rate = max_burn_rate
        self.enable_context_ballooning = enable_context_ballooning
        self.dry_run = dry_run

        self.burn_rate_limit: Optional[float] = None
        self.burn_rate_window_seconds: Optional[float] = None
        if max_burn_rate:
            parsed = parse_burn_rate(max_burn_rate)
            if parsed:
                self.burn_rate_limit, self.burn_rate_window_seconds = parsed

        self._spend_records: Deque[Tuple[float, float]] = collections.deque()
        self.recent_hashes: Deque[Tuple[float, str]] = collections.deque(maxlen=loop_window_size)
        self._session_hashes: Dict[str, Deque[Tuple[float, str]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=10)
        )
        self._session_history: Dict[str, Deque[Dict[str, Any]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=20)
        )
        self._tool_history: Dict[str, Deque[Tuple[float, str, str]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=10)
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
        # Assume input tokens cost plus minimum estimated completion tokens (50)
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

        # 1. Check global sliding window
        hashes = [h for _, h in self.recent_hashes]
        cycle = detect_cycle_in_sequence(hashes + [incoming_hash], length_1_repeats=self.loop_threshold)
        if cycle is not None:
            return cycle

        # 2. Check per-session sliding window
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
        self._tool_history[session_id].append((time.time(), tool_name, canonical_args))

    def check_message_tool_loop(
        self,
        messages: Any,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Check if the messages history itself contains repeated consecutive tool calls."""
        if self.active_profile == "passive" or self.tool_loop_threshold <= 1:
            return

        extracted = extract_tool_calls(messages)
        if len(extracted) < min(self.tool_loop_threshold, 4):
            return

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
            length_1_repeats=self.tool_loop_threshold,
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
            ToolPingPongError: If repeated tool calls detected.
            ContextBallooningError: If prompt tokens expand > 300% across 3 consecutive requests.
            BurnRateExceededError: If sliding window burn rate limit is exceeded.
            BudgetExceededError: If sliding hourly or daily budget limit is exceeded.

        Returns:
            The calculated prompt_hash if allowed.
        """
        prompt_hash = self.compute_prompt_hash(messages)
        sess = session_id or "default"

        if self.kill_switch_active:
            t_saved, u_saved = self.calculate_saved_telemetry(model, messages, session_id=sess)
            raise KillSwitchActiveError(tokens_saved=t_saved, usd_saved=u_saved)

        if self.active_profile == "passive":
            self.record_request_hash(prompt_hash)
            return prompt_hash

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

        self.check_message_tool_loop(messages, model=model, session_id=sess)
        extracted_tools = extract_tool_calls(tool_calls or messages)
        if extracted_tools:
            latest_tool_name, _, raw_args = extracted_tools[-1]
            self.check_tool_loop(sess, latest_tool_name, raw_args, model=model)
            self.record_tool_call(sess, latest_tool_name, raw_args)

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
        self.recent_hashes.clear()
        self._session_hashes.clear()
        self._spend_records.clear()
        self._session_history.clear()
        self._tool_history.clear()
        self.kill_switch_active = False

    def toggle_kill_switch(self, active: Optional[bool] = None) -> bool:
        """Toggle or explicitly set the kill switch state."""
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
        profile: Optional[str] = None,
        max_burn_rate: Optional[str] = None,
        enable_context_ballooning: Optional[bool] = None,
    ) -> None:
        """Dynamically update runtime limits and profile."""
        if hourly_limit is not None and hourly_limit > 0:
            self.hourly_limit = float(hourly_limit)
        if daily_limit is not None and daily_limit > 0:
            self.daily_limit = float(daily_limit)
        if loop_threshold is not None and loop_threshold >= 0:
            self.loop_threshold = int(loop_threshold)
        if tool_loop_threshold is not None and tool_loop_threshold >= 0:
            self.tool_loop_threshold = int(tool_loop_threshold)
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
            "loop_window_size": self.loop_window_size,
            "recent_hashes_count": len(self.recent_hashes),
        }
