"""TokenGuard: Lightweight open-source local proxy and circuit breaker for LLM calls."""

from tokenguard.config import Settings, get_settings
from tokenguard.guard import (
    BudgetExceededError,
    BurnRateExceededError,
    CircuitBreaker,
    CircuitBreakerError,
    ContextBallooningError,
    KillSwitchActiveError,
    LoopDetectedError,
    ToolPingPongError,
)
from tokenguard.proxy import create_app
from tokenguard.sdk import guard, protect

__version__ = "0.2.0"
__all__ = [
    "guard",
    "protect",
    "Settings",
    "get_settings",
    "CircuitBreaker",
    "CircuitBreakerError",
    "BudgetExceededError",
    "BurnRateExceededError",
    "ContextBallooningError",
    "LoopDetectedError",
    "ToolPingPongError",
    "KillSwitchActiveError",
    "create_app",
]
