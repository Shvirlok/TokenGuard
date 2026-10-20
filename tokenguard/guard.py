"""Circuit breaker and guard engine for TokenGuard."""

from __future__ import annotations

import collections
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
from typing import Any, Deque, Dict, List, Optional, Tuple

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
            # 1. Audible terminal bell (if sound enabled)
            if sound:
                try:
                    sys.stderr.write("\a")
                    sys.stderr.flush()
                except Exception:
                    pass

            system = platform.system().lower()
            if system == "darwin":
                # macOS AppleScript notification
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
                # Linux notify-send
                if shutil.which("notify-send"):
                    subprocess.run(
                        ["notify-send", "-u", "critical", "-a", "TokenGuard", title, message],
                        capture_output=True,
                        timeout=2.0,
                        check=False,
                    )
        except Exception:
            # Fail silently and gracefully without disrupting proxy operation
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
        **kwargs: Any,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.code = code
        self.current_spend = current_spend
        self.budget_limit = budget_limit
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
        for k, v in self.extra.items():
            data[k] = v
        return {"error": data}


class BudgetExceededError(CircuitBreakerError):
    def __init__(self, message: str = "TokenGuard: budget limit exceeded"):
        super().__init__(message=message, error_type="budget_exceeded", status_code=429)


class BurnRateExceededError(CircuitBreakerError):
    """Tripped when cumulative spend in a sliding window exceeds max burn rate."""

    def __init__(
        self,
        message: str,
        current_spend: float,
        budget_limit: float,
        code: str = "burn_rate_exceeded",
        error_type: str = "tokenguard_budget_breach",
    ):
        super().__init__(
            message=message,
            error_type=error_type,
            status_code=429,
            code=code,
            current_spend=round(current_spend, 2),
            budget_limit=round(budget_limit, 2),
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
    ):
        extra: Dict[str, Any] = {}
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
            **extra,
        )


class LoopDetectedError(CircuitBreakerError):
    def __init__(self, message: str = "TokenGuard: infinite agent loop detected"):
        super().__init__(message=message, error_type="loop_detected", status_code=429)


class KillSwitchActiveError(CircuitBreakerError):
    def __init__(self, message: str = "TokenGuard: kill switch active"):
        super().__init__(message=message, error_type="kill_switch_active", status_code=429)



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

    # Exact match
    if model_clean in table:
        return table[model_clean]

    # Prefix matching (e.g. gpt-4o-2024-08-06 -> gpt-4o, deepseek-chat-v3 -> deepseek-chat)
    for key, rates in table.items():
        if key != "default" and model_clean.startswith(key):
            return rates

    # Strip optional provider prefix like "deepseek/deepseek-chat" or "openai/gpt-4o"
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
        loop_window_size: int = 20,
        loop_window_seconds: float = 60.0,
        prices_path: Optional[Path] = None,
        kill_switch: bool = False,
        enable_notifications: bool = True,
        active_profile: str = "careful",
        max_burn_rate: Optional[str] = None,
        enable_context_ballooning: bool = True,
    ):
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.loop_threshold = loop_threshold
        self.loop_window_size = loop_window_size
        self.loop_window_seconds = loop_window_seconds
        self.kill_switch_active = kill_switch
        self.prices_path = prices_path
        self.enable_notifications = enable_notifications
        self.active_profile = active_profile
        self.max_burn_rate = max_burn_rate
        self.enable_context_ballooning = enable_context_ballooning

        self.burn_rate_limit: Optional[float] = None
        self.burn_rate_window_seconds: Optional[float] = None
        if max_burn_rate:
            parsed = parse_burn_rate(max_burn_rate)
            if parsed:
                self.burn_rate_limit, self.burn_rate_window_seconds = parsed

        # Rolling window of recent spends: deque of (epoch_timestamp, cost_usd)
        self._spend_records: Deque[Tuple[float, float]] = collections.deque()

        # Bounded in-memory sliding window for prompt hashes: deque of (epoch_timestamp, hash_str)
        self.recent_hashes: Deque[Tuple[float, str]] = collections.deque(maxlen=loop_window_size)

        # In-memory session tracking for context ballooning guard: session_id -> deque of request records
        self._session_history: Dict[str, Deque[Dict[str, Any]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=20)
        )

        # In-memory pricing table
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

    def check_loop(self, incoming_hash: str) -> bool:
        """Check if incoming prompt hash triggers infinite loop detector.

        Only triggers if:
        1. loop_threshold >= 2
        2. At least (threshold - 1) previous requests exist in the active time window
        3. ALL of those previous (threshold - 1) requests match incoming_hash
        """
        if self.active_profile == "passive" or self.loop_threshold <= 1:
            return False

        now = time.time()
        self._prune_old_hashes(now)

        needed_prior = self.loop_threshold - 1
        if len(self.recent_hashes) < needed_prior:
            return False

        tail = [h for _, h in list(self.recent_hashes)[-needed_prior:]]
        return all(h == incoming_hash for h in tail)

    def check_context_ballooning(
        self,
        session_id: str,
        incoming_tokens: int,
        has_task_update: bool = False,
        cache_hit: bool = False,
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

        # If any prior request in window had a task update or cache hit, do not trip
        if r1.get("has_task_update") or r1.get("cache_hit") or r2.get("has_task_update") or r2.get("cache_hit"):
            return

        tokens_1 = r1.get("prompt_tokens", 0)
        tokens_2 = r2.get("prompt_tokens", 0)
        tokens_3 = incoming_tokens

        # Check monotonic context growth across 3 consecutive requests
        # and minimum token baseline (>= 50 tokens) to prevent false positives on tiny prompts
        if tokens_1 >= 50 and tokens_3 > tokens_2 and tokens_2 > tokens_1:
            growth_ratio = (tokens_3 - tokens_1) / float(tokens_1)
            # Expands by > 300%
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
                raise ContextBallooningError(
                    message=msg,
                    prompt_tokens=tokens_3,
                    baseline_tokens=tokens_1,
                    growth_rate=round(growth_ratio, 2),
                )

    def record_request_hash(self, prompt_hash: str) -> None:
        """Record an allowed request's prompt hash with current timestamp into sliding history."""
        now = time.time()
        self._prune_old_hashes(now)
        self.recent_hashes.append((now, prompt_hash))

    def remove_last_hash(self, prompt_hash: Optional[str] = None) -> None:
        """Remove the last recorded prompt hash if upstream call failed."""
        if self.recent_hashes:
            if prompt_hash is None or self.recent_hashes[-1][1] == prompt_hash:
                self.recent_hashes.pop()

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
    ) -> str:
        """Run pre-flight checks against circuit breaker guards.

        Raises:
            KillSwitchActiveError: If manual kill switch is enabled.
            LoopDetectedError: If the same prompt hash repeated >= threshold times consecutively.
            ContextBallooningError: If prompt tokens expand > 300% across 3 consecutive requests.
            BurnRateExceededError: If sliding window burn rate limit is exceeded.
            BudgetExceededError: If sliding hourly or daily budget limit is exceeded.

        Returns:
            The calculated prompt_hash if allowed.
        """
        prompt_hash = self.compute_prompt_hash(messages)
        sess = session_id or "default"

        # 1. Kill Switch Check
        if self.kill_switch_active:
            raise KillSwitchActiveError()

        # Passive mode bypasses loop & budget cutoffs for pure telemetry
        if self.active_profile == "passive":
            self.record_request_hash(prompt_hash)
            return prompt_hash

        # 2. Infinite Loop Detection Check
        if self.check_loop(prompt_hash):
            if self.enable_notifications:
                send_block_notification(
                    title="TokenGuard Alert",
                    message=f"Infinite loop detected & blocked for model '{model}' ({self.loop_threshold} identical consecutive requests)",
                )
            # Rejection must NOT poison the history window
            raise LoopDetectedError(
                f"TokenGuard: infinite agent loop detected ({self.loop_threshold} identical consecutive requests)"
            )

        est_tokens = incoming_tokens if incoming_tokens is not None else self.estimate_prompt_tokens(messages)

        # 3. Context Ballooning Guard Check
        self.check_context_ballooning(
            session_id=sess,
            incoming_tokens=est_tokens,
            has_task_update=has_task_update,
            cache_hit=cache_hit,
        )

        # 4. Burn-Rate Engine Check
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
                raise BurnRateExceededError(
                    message=msg,
                    current_spend=round(spend_to_report, 2),
                    budget_limit=round(self.burn_rate_limit, 2),
                )

        # 5. Budget Guard Check
        if self.hourly_limit > 0:
            current_hourly_spend = self.get_hourly_spend()
            estimated_cost = self.estimate_cost(model, messages)
            if (current_hourly_spend + estimated_cost) > self.hourly_limit:
                # Rejection must NOT poison the history window
                raise BudgetExceededError("TokenGuard: budget limit exceeded")

        if self.daily_limit > 0:
            current_daily_spend = self.get_daily_spend()
            estimated_cost = self.estimate_cost(model, messages)
            if (current_daily_spend + estimated_cost) > self.daily_limit:
                # Rejection must NOT poison the history window
                raise BudgetExceededError("TokenGuard: daily budget limit exceeded")

        # Request passed all pre-flight checks: append to recent_hashes and session sequence
        self.record_request_hash(prompt_hash)
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

    def reset(self) -> None:
        """Reset all in-memory rolling state (hash history, spend records, session tracking, kill switch)."""
        self.recent_hashes.clear()
        self._spend_records.clear()
        self._session_history.clear()
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
            "loop_window_size": self.loop_window_size,
            "recent_hashes_count": len(self.recent_hashes),
        }
