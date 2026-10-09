"""Configuration module for TokenGuard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PRICES_PATH = Path(__file__).resolve().parent / "prices.json"
DEFAULT_STATIC_PATH = Path(__file__).resolve().parent / "static"
DEFAULT_DB_PATH = Path("tokenguard.db")


class Settings(BaseSettings):
    """TokenGuard runtime settings with environment variable and CLI support."""

    model_config = SettingsConfigDict(
        env_prefix="TOKENGUARD_",
        env_file=".env",
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1", description="Host to bind the proxy server to")
    port: int = Field(default=8080, description="Port to bind the proxy server to")
    hourly_limit: float = Field(default=5.0, description="Sliding hourly budget limit in USD")
    daily_limit: float = Field(default=50.0, description="Sliding daily budget limit in USD")
    loop_threshold: int = Field(default=3, description="Consecutive identical request hashes before tripping")
    max_repeats: Optional[int] = Field(default=None, description="Alias for loop_threshold (maximum duplicate requests)")
    loop_window_size: int = Field(default=20, description="Size of rolling request hash history")
    loop_window_seconds: float = Field(default=60.0, description="Sliding time window in seconds for consecutive loop matching")
    upstream_url: str = Field(default="https://api.openai.com", description="Upstream OpenAI API base URL")
    anthropic_url: str = Field(default="https://api.anthropic.com", description="Upstream Anthropic API base URL")
    gemini_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta/openai", description="Upstream Google Gemini OpenAI gateway base URL")
    dashscope_url: str = Field(default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", description="Upstream Alibaba DashScope/Qwen base URL")
    deepseek_url: str = Field(default="https://api.deepseek.com", description="Upstream DeepSeek base URL")
    groq_url: str = Field(default="https://api.groq.com/openai/v1", description="Upstream Groq base URL")
    mistral_url: str = Field(default="https://api.mistral.ai/v1", description="Upstream Mistral AI base URL")
    openrouter_url: str = Field(default="https://openrouter.ai/api/v1", description="Upstream OpenRouter base URL")
    db_path: Path = Field(default=DEFAULT_DB_PATH, description="SQLite database file path")
    prices_path: Path = Field(default=DEFAULT_PRICES_PATH, description="Path to prices.json registry")
    static_path: Path = Field(default=DEFAULT_STATIC_PATH, description="Path to dashboard static assets")
    kill_switch: bool = Field(default=False, description="Manual kill switch flag to block all requests")
    timeout: float = Field(default=60.0, description="Upstream HTTP timeout in seconds")
    profile: str = Field(default="careful", description="Active guard profile (careful, standard, passive)")
    passive: bool = Field(default=False, description="Passive observability mode")

    def model_post_init(self, __context) -> None:
        if self.max_repeats is not None:
            self.loop_threshold = self.max_repeats
        if self.passive or self.profile == "passive":
            self.profile = "passive"
            self.loop_threshold = 0


USER_CONFIG_DIR = Path.home() / ".tokenguard"
USER_CONFIG_FILE = USER_CONFIG_DIR / "config.json"

PROFILES = {
    "careful": {
        "name": "Careful",
        "description": "$5/hr cap, 2-loop cutoff, alerts ON",
        "hourly_limit": 5.0,
        "daily_limit": 50.0,
        "loop_threshold": 2,
    },
    "standard": {
        "name": "Standard Agent",
        "description": "$15/hr cap, 4-loop cutoff",
        "hourly_limit": 15.0,
        "daily_limit": 100.0,
        "loop_threshold": 4,
    },
    "passive": {
        "name": "Passive Monitor",
        "description": "No loop blocks, tracking only",
        "hourly_limit": 1000.0,
        "daily_limit": 5000.0,
        "loop_threshold": 0,
    },
}


def load_stored_config() -> dict:
    """Load user configuration from ~/.tokenguard/config.json."""
    if USER_CONFIG_FILE.exists():
        try:
            with open(USER_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_stored_config(data: dict) -> None:
    """Save user configuration to ~/.tokenguard/config.json with restricted permissions (0600)."""
    try:
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(USER_CONFIG_DIR, 0o700)
        except OSError:
            pass

        temp_file = USER_CONFIG_DIR / "config.json.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        try:
            os.chmod(temp_file, 0o600)
        except OSError:
            pass

        temp_file.replace(USER_CONFIG_FILE)
        try:
            os.chmod(USER_CONFIG_FILE, 0o600)
        except OSError:
            pass
    except Exception:
        pass


def update_stored_config(**kwargs) -> dict:
    """Update and persist specific key-value pairs in ~/.tokenguard/config.json."""
    data = load_stored_config()
    data.update(kwargs)
    save_stored_config(data)
    return data


def mask_key(key: Optional[str]) -> str:
    """Return a masked preview of an API key for discreet display."""
    if not key:
        return "None"
    k = key.strip()
    if len(k) <= 8:
        return "••••••••"
    return f"{k[:4]}••••{k[-4:]}"


def read_env_file_keys() -> dict[str, str]:
    """Parse keys from local .env file if it exists in the current directory."""
    keys: dict[str, str] = {}
    env_file = Path(".env")
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k in (
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "GEMINI_API_KEY",
                    "GOOGLE_API_KEY",
                    "DASHSCOPE_API_KEY",
                    "QWEN_API_KEY",
                    "DEEPSEEK_API_KEY",
                    "GROQ_API_KEY",
                    "MISTRAL_API_KEY",
                    "OPENROUTER_API_KEY",
                ) and v:
                    keys[k] = v
        except Exception:
            pass
    return keys


def detect_api_keys() -> dict[str, Optional[str]]:
    """Detect available API keys across environment variables, .env file, and stored config.
    
    Priority order:
    1. os.environ (highest priority)
    2. .env file in CWD
    3. ~/.tokenguard/config.json
    """
    env_file_keys = read_env_file_keys()
    stored = load_stored_config()

    openai_key = (
        os.environ.get("OPENAI_API_KEY")
        or env_file_keys.get("OPENAI_API_KEY")
        or stored.get("openai_api_key")
    )
    anthropic_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or env_file_keys.get("ANTHROPIC_API_KEY")
        or stored.get("anthropic_api_key")
    )
    gemini_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or env_file_keys.get("GEMINI_API_KEY")
        or env_file_keys.get("GOOGLE_API_KEY")
        or stored.get("gemini_api_key")
        or stored.get("google_api_key")
    )
    dashscope_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or env_file_keys.get("DASHSCOPE_API_KEY")
        or env_file_keys.get("QWEN_API_KEY")
        or stored.get("dashscope_api_key")
        or stored.get("qwen_api_key")
    )
    deepseek_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or env_file_keys.get("DEEPSEEK_API_KEY")
        or stored.get("deepseek_api_key")
    )
    groq_key = (
        os.environ.get("GROQ_API_KEY")
        or env_file_keys.get("GROQ_API_KEY")
        or stored.get("groq_api_key")
    )
    mistral_key = (
        os.environ.get("MISTRAL_API_KEY")
        or env_file_keys.get("MISTRAL_API_KEY")
        or stored.get("mistral_api_key")
    )
    openrouter_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or env_file_keys.get("OPENROUTER_API_KEY")
        or stored.get("openrouter_api_key")
    )

    return {
        "OPENAI_API_KEY": openai_key,
        "ANTHROPIC_API_KEY": anthropic_key,
        "GEMINI_API_KEY": gemini_key,
        "DASHSCOPE_API_KEY": dashscope_key,
        "DEEPSEEK_API_KEY": deepseek_key,
        "GROQ_API_KEY": groq_key,
        "MISTRAL_API_KEY": mistral_key,
        "OPENROUTER_API_KEY": openrouter_key,
    }


def get_api_key(provider: str = "openai") -> Optional[str]:
    """Get active API key for a given provider with priority to os.environ."""
    keys = detect_api_keys()
    p = provider.lower().strip()
    if p in ("anthropic", "claude"):
        return keys.get("ANTHROPIC_API_KEY")
    elif p in ("gemini", "google"):
        return keys.get("GEMINI_API_KEY")
    elif p in ("dashscope", "qwen", "qwq"):
        return keys.get("DASHSCOPE_API_KEY")
    elif p in ("deepseek", "deepseek-chat", "deepseek-reasoner", "deepseek-coder"):
        return keys.get("DEEPSEEK_API_KEY")
    elif p in ("groq", "llama"):
        return keys.get("GROQ_API_KEY")
    elif p in ("mistral", "codestral", "pixtral", "ministral"):
        return keys.get("MISTRAL_API_KEY")
    elif p in ("openrouter",):
        return keys.get("OPENROUTER_API_KEY")
    return keys.get("OPENAI_API_KEY")


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the current global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set the global settings instance."""
    global _settings
    _settings = settings

