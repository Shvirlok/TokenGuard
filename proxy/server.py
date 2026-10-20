"""Proxy server module shim pointing to tokenguard.proxy."""
from tokenguard.proxy import (
    app,
    StreamAccumulator,
    get_client,
    get_guard,
    _handle_streaming_chat,
    _handle_anthropic_streaming,
    _handle_openai_to_anthropic_adapter,
)

__all__ = [
    "app",
    "StreamAccumulator",
    "get_client",
    "get_guard",
    "_handle_streaming_chat",
    "_handle_anthropic_streaming",
    "_handle_openai_to_anthropic_adapter",
]
