"""TokenGuard: Lightweight open-source local proxy and circuit breaker for LLM calls."""

from tokenguard.config import Settings, get_settings
from tokenguard.guard import (
    BudgetExceededError,
    CircuitBreaker,
    CircuitBreakerError,
    KillSwitchActiveError,
    LoopDetectedError,
)
from tokenguard.proxy import create_app

__version__ = "0.1.0"
__all__ = [
    "Settings",
    "get_settings",
    "CircuitBreaker",
    "CircuitBreakerError",
    "BudgetExceededError",
    "LoopDetectedError",
    "KillSwitchActiveError",
    "create_app",
]
