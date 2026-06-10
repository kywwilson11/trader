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
        # Current model IDs (the old defaults were legacy/dated snapshots)
        "claude": {"api_key": "", "model": "claude-haiku-4-5"},
        "openai": {"api_key": "", "model": "gpt-5.4-nano"},
    },
    "analyst_model_override": None,     # None = use smart routing
    "sentiment_model_override": None,   # None = use smart routing
    "detected_tier": None,              # Auto-detected: 'free' or 'paid'
    "tier_override": None,              # Manual override: 'free', 'paid', or None (auto)
    "fmp_api_key": "",
    # 15s doomed every thinking-model call mid-response; the analyst gate
    # runs every 600s, so a 30-45s budget costs nothing
    "max_llm_latency_sec": 30,
    "journal_enabled": True,
}

# Keys migrated from old format to new
_MIGRATE_KEYS = {
    "analyst_model": "analyst_model_override",
    "sentiment_model": "sentiment_model_override",
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

    # Migrate old keys to new names (e.g. analyst_model -> analyst_model_override)
    migrated = False
    for old_key, new_key in _MIGRATE_KEYS.items():
        if old_key in config and new_key not in config:
            config[new_key] = config.pop(old_key)
            migrated = True
        elif old_key in config and new_key in config:
            # Both exist — remove old key
            del config[old_key]
            migrated = True

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

    if migrated:
        save_llm_config(config)

    return config


def save_llm_config(config: dict):
    """Persist LLM config to disk."""
    try:
        with open(LLM_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[LLM-CONFIG] Error saving: {e}")
