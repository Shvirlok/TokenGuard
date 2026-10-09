"""Dynamic pricing engine for TokenGuard with offline-first OpenRouter registry sync."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger("tokenguard.pricing")

DEFAULT_PRICES_PATH = Path(__file__).resolve().parent / "prices.json"
PRICES_CACHE_DIR = Path.home() / ".tokenguard"
PRICES_CACHE_FILE = PRICES_CACHE_DIR / "prices_cache.json"
CACHE_TTL_SECONDS = 86400.0  # 24 hours
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def load_baseline_prices(path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """Load baseline offline prices from static prices.json."""
    target = path or DEFAULT_PRICES_PATH
    if target and target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load baseline prices from {target}: {e}")

    # Built-in minimal fallback if file is missing
    return {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "deepseek-reasoner": {"input": 0.55, "output": 2.19},
        "default": {"input": 1.00, "output": 2.00},
    }


def load_cached_prices() -> Tuple[Dict[str, Dict[str, float]], float]:
    """Load cached prices from ~/.tokenguard/prices_cache.json.
    
    Returns:
        (prices_dict, last_updated_epoch_timestamp)
    """
    if PRICES_CACHE_FILE.exists():
        try:
            with open(PRICES_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                prices = data.get("prices", {})
                last_updated = float(data.get("last_updated", 0.0))
                return prices, last_updated
        except Exception as e:
            logger.debug(f"Failed to load price cache: {e}")
    return {}, 0.0


def save_cached_prices(prices: Dict[str, Dict[str, float]], last_updated: Optional[float] = None) -> None:
    """Save prices to ~/.tokenguard/prices_cache.json with restricted permissions."""
    try:
        PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ts = last_updated if last_updated is not None else time.time()
        payload = {
            "last_updated": ts,
            "last_updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
            "source": "openrouter.ai",
            "model_count": len(prices),
            "prices": prices,
        }

        temp_file = PRICES_CACHE_DIR / "prices_cache.json.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        temp_file.replace(PRICES_CACHE_FILE)
    except Exception as e:
        logger.warning(f"Failed to write prices cache: {e}")


def fetch_openrouter_prices(timeout: float = 4.0) -> Dict[str, Dict[str, float]]:
    """Fetch live model pricing registry from OpenRouter public API (no API key required).
    
    Returns:
        Dictionary mapping model keys and short names to {"input": float, "output": float} (USD per 1M tokens).
    """
    prices: Dict[str, Dict[str, float]] = {}
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(OPENROUTER_MODELS_URL)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter API returned status {resp.status_code}")

        data = resp.json()
        models = data.get("data", [])
        for item in models:
            model_id = item.get("id")
            if not model_id:
                continue

            pricing = item.get("pricing") or {}
            prompt_str = pricing.get("prompt")
            completion_str = pricing.get("completion")

            if prompt_str is None and completion_str is None:
                continue

            try:
                # OpenRouter pricing is USD per token -> multiply by 1M for USD/1M tokens
                prompt_rate = float(prompt_str or 0.0) * 1_000_000.0
                completion_rate = float(completion_str or 0.0) * 1_000_000.0
                if prompt_rate < 0 or completion_rate < 0:
                    continue
            except (ValueError, TypeError):
                continue

            rate_entry = {
                "input": round(prompt_rate, 4),
                "output": round(completion_rate, 4),
            }

            # Register full ID: e.g. "openai/gpt-4o"
            prices[model_id.lower()] = rate_entry

            # Register short name: e.g. "gpt-4o"
            short_id = model_id.split("/")[-1].lower()
            prices[short_id] = rate_entry

            # Register alias with hyphens instead of dots (e.g. claude-3.5-sonnet -> claude-3-5-sonnet)
            if "." in short_id:
                alt_id = short_id.replace(".", "-")
                prices[alt_id] = rate_entry

    return prices


def update_pricing_cache(
    force: bool = False,
    timeout: float = 4.0,
    baseline_path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, float]], bool, str]:
    """Check TTL and fetch updated live prices from OpenRouter, merging with local baseline.
    
    Returns:
        (merged_prices_dict, was_updated, status_message)
    """
    baseline = load_baseline_prices(baseline_path)
    cached, last_updated = load_cached_prices()
    now = time.time()

    # Check cache freshness
    if not force and cached and (now - last_updated) < CACHE_TTL_SECONDS:
        merged = dict(baseline)
        merged.update(cached)
        age_hours = round((now - last_updated) / 3600.0, 1)
        return merged, False, f"Cache is fresh ({age_hours}h old, {len(merged)} models)"

    # Fetch live rates
    try:
        live_prices = fetch_openrouter_prices(timeout=timeout)
        if live_prices:
            merged = dict(live_prices)
            merged.update(baseline)
            save_cached_prices(merged, last_updated=now)
            return merged, True, f"Successfully fetched live prices ({len(merged)} models synced)"
    except Exception as e:
        logger.debug(f"Dynamic price fetch failed ({e}). Falling back to existing cache or baseline.")

    # Graceful fallback: return cached or baseline
    merged = dict(cached)
    merged.update(baseline)
    if cached:
        return merged, False, f"Offline fallback (using cached pricing with {len(merged)} models)"
    return merged, False, f"Offline baseline (using local prices.json with {len(merged)} models)"


def get_dynamic_pricing_table(
    custom_path: Optional[Path] = None,
    allow_bg_update: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Retrieve the current active pricing table with offline-first fallback and background sync."""
    if custom_path and custom_path.exists():
        return load_baseline_prices(custom_path)

    baseline = load_baseline_prices()
    cached, last_updated = load_cached_prices()
    now = time.time()

    merged = dict(cached)
    merged.update(baseline)

    # If cache is missing or older than TTL, trigger non-blocking background refresh
    if allow_bg_update and ((now - last_updated) >= CACHE_TTL_SECONDS):
        def _bg_worker() -> None:
            try:
                update_pricing_cache(force=True, timeout=3.5)
            except Exception:
                pass

        try:
            t = threading.Thread(target=_bg_worker, daemon=True)
            t.start()
        except Exception:
            pass

    return merged
