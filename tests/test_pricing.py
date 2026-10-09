"""Tests for dynamic pricing engine, OpenRouter sync, and offline fallbacks."""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from tokenguard.pricing import (
    fetch_openrouter_prices,
    get_dynamic_pricing_table,
    load_baseline_prices,
    load_cached_prices,
    save_cached_prices,
    update_pricing_cache,
)
from tokenguard.guard import CircuitBreaker
from tokenguard.cli import create_parser, cmd_prices


def test_load_baseline_prices():
    """Verify baseline offline prices load correctly."""
    prices = load_baseline_prices()
    assert "gpt-4o" in prices
    assert prices["gpt-4o"]["input"] == 2.50
    assert prices["gpt-4o"]["output"] == 10.00
    assert "deepseek-chat" in prices
    assert "claude-3-5-sonnet" in prices
    assert "default" in prices


def test_fetch_openrouter_prices_parsing(monkeypatch):
    """Verify OpenRouter models JSON parsing and rate calculation (USD per 1M tokens)."""
    mock_response_data = {
        "data": [
            {
                "id": "openai/gpt-4o-2024-11-20",
                "name": "OpenAI: GPT-4o",
                "pricing": {
                    "prompt": "0.0000025",
                    "completion": "0.00001",
                },
            },
            {
                "id": "anthropic/claude-3.5-sonnet",
                "name": "Claude 3.5 Sonnet",
                "pricing": {
                    "prompt": "0.000003",
                    "completion": "0.000015",
                },
            },
            {
                "id": "invalid/negative-model",
                "name": "Invalid Model",
                "pricing": {
                    "prompt": "-1.0",
                    "completion": "-1.0",
                },
            },
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data

    with patch("httpx.Client.get", return_value=mock_resp):
        prices = fetch_openrouter_prices(timeout=2.0)

    # Verify openai/gpt-4o-2024-11-20
    assert "openai/gpt-4o-2024-11-20" in prices
    assert "gpt-4o-2024-11-20" in prices
    assert prices["gpt-4o-2024-11-20"]["input"] == 2.50
    assert prices["gpt-4o-2024-11-20"]["output"] == 10.00

    # Verify claude-3.5-sonnet and alias claude-3-5-sonnet
    assert "claude-3.5-sonnet" in prices
    assert "claude-3-5-sonnet" in prices
    assert prices["claude-3-5-sonnet"]["input"] == 3.00
    assert prices["claude-3-5-sonnet"]["output"] == 15.00

    # Verify negative rates are ignored
    assert "invalid/negative-model" not in prices


def test_update_pricing_cache_ttl(tmp_path: Path, monkeypatch):
    """Verify cache saving, loading, and TTL behavior."""
    cache_file = tmp_path / "prices_cache.json"
    monkeypatch.setattr("tokenguard.pricing.PRICES_CACHE_DIR", tmp_path)
    monkeypatch.setattr("tokenguard.pricing.PRICES_CACHE_FILE", cache_file)

    # 1. Mock live fetch
    mock_live = {
        "gpt-5-preview": {"input": 5.0, "output": 20.0},
    }
    monkeypatch.setattr("tokenguard.pricing.fetch_openrouter_prices", lambda timeout: mock_live)

    merged, was_updated, msg = update_pricing_cache(force=True)
    assert was_updated is True
    assert "gpt-5-preview" in merged
    assert "gpt-4o" in merged
    assert cache_file.exists()

    # 2. Re-run without force -> should be fresh
    merged2, was_updated2, msg2 = update_pricing_cache(force=False)
    assert was_updated2 is False
    assert "Cache is fresh" in msg2


def test_offline_fallback_on_network_failure(tmp_path: Path, monkeypatch):
    """Verify graceful fallback to offline baseline when OpenRouter is unreachable."""
    cache_file = tmp_path / "non_existent_cache.json"
    monkeypatch.setattr("tokenguard.pricing.PRICES_CACHE_DIR", tmp_path)
    monkeypatch.setattr("tokenguard.pricing.PRICES_CACHE_FILE", cache_file)

    def mock_failing_fetch(timeout):
        raise ConnectionError("Network unreachable")

    monkeypatch.setattr("tokenguard.pricing.fetch_openrouter_prices", mock_failing_fetch)

    merged, was_updated, msg = update_pricing_cache(force=True)
    assert was_updated is False
    assert "Offline baseline" in msg
    assert "gpt-4o" in merged
    assert merged["gpt-4o"]["input"] == 2.50


def test_cli_prices_flags():
    """Verify CLI parser accepts --update and --offline flags and cmd_prices runs."""
    parser = create_parser()

    args_update = parser.parse_args(["prices", "--update"])
    assert args_update.command == "prices"
    assert args_update.update is True

    args_offline = parser.parse_args(["prices", "--offline"])
    assert args_offline.offline is True

    # Test cmd_prices execution with offline flag
    cmd_prices(args_offline)


def test_guard_integration_with_dynamic_prices(tmp_path: Path, monkeypatch):
    """Verify CircuitBreaker calculates cost and avoided cost with dynamic prices."""
    custom_prices = {
        "custom-model": {"input": 4.0, "output": 8.0},
        "default": {"input": 1.0, "output": 2.0},
    }
    custom_path = tmp_path / "custom_prices.json"
    custom_path.write_text(json.dumps(custom_prices))

    guard = CircuitBreaker(prices_path=custom_path)
    assert guard.compute_cost("custom-model", 1000, 500) == (1000 * 4.0 + 500 * 8.0) / 1_000_000.0
