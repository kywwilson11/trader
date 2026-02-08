"""LLM configuration management — API keys, provider selection, model settings.

Persists to llm_config.json (gitignored). Mirrors the gui_settings.json pattern.
"""

import json
from pathlib import Path

LLM_CONFIG_FILE = Path(__file__).resolve().parent / "llm_config.json"

_DEFAULTS = {
    "provider": "gemini",
    "enabled": True,
    "models": {
        "gemini": {"api_key": "", "model": "gemini-2.5-flash-lite"},
        "claude": {"api_key": "", "model": "claude-sonnet-4-5-20250929"},
        "openai": {"api_key": "", "model": "gpt-4.1"},
    },
    "fmp_api_key": "",
    "max_llm_latency_sec": 15,
    "journal_enabled": True,
}


def load_llm_config() -> dict:
    """Load LLM config from disk, filling in any missing keys with defaults."""
    config = {}
    try:
        if LLM_CONFIG_FILE.exists():
            with open(LLM_CONFIG_FILE) as f:
                config = json.load(f)
    except Exception:
        pass

    # Merge defaults for any missing top-level keys
    for key, default in _DEFAULTS.items():
        if key not in config:
            config[key] = default
        elif key == "models" and isinstance(default, dict):
            # Merge per-provider defaults
            for provider, pdefault in default.items():
                if provider not in config["models"]:
                    config["models"][provider] = pdefault
                else:
                    for pk, pv in pdefault.items():
                        if pk not in config["models"][provider]:
                            config["models"][provider][pk] = pv

    return config


def save_llm_config(config: dict):
    """Persist LLM config to disk."""
    try:
        with open(LLM_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[LLM-CONFIG] Error saving: {e}")
