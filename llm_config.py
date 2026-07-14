"""LLM configuration management — API keys, provider selection, model settings.

Persists to llm_config.json (gitignored). Mirrors the gui_settings.json pattern.

Config keys (see _DEFAULTS below for the exact defaults):

  provider                 Legacy single-provider pin: 'auto' (default),
                            'gemini', 'anthropic'/'claude', or 'openai'.
                            Only consulted when selection_mode == 'single'
                            (see below) — the value tells 'single' mode
                            which provider's configured model to use. This
                            is how OLD saved configs (pre-selection_mode,
                            which only ever stored a legacy provider name
                            here) keep working: load_llm_config() only
                            fills MISSING keys, so an old config's
                            "provider": "gemini" is preserved untouched and
                            "selection_mode" gets filled in as 'auto' — the
                            new engine default, not an implicit 'single'.
  selection_mode            How llm_client.resolve_provider_chain() builds
                            its candidate list. One of:
                              'auto'      (default) — try every provider
                                          that has a usable key, in
                                          provider_preference order; each
                                          contributes its primary model
                                          then its own fallback chain, then
                                          enabled endpoints are appended
                                          last. This is the smart, mostly
                                          zero-config default.
                              'single'    — only the provider named by the
                                            'provider' key, only its
                                            configured model (no fallback
                                            chain, no cross-provider
                                            fallback). Backward-compat mode
                                            for a saved Jetson config that
                                            hard-pins one provider.
                              'free-only' — only candidates that cost
                                            nothing: enabled endpoints with
                                            free: true (OpenRouter ':free'
                                            models, Groq's free tier,
                                            keyless local Ollama), plus
                                            Gemini as an always-available
                                            free-tier-budgeted last resort.
                              'best-free' — same candidate set as
                                            'free-only', reordered by a
                                            quality ranking instead of
                                            provider_preference order (see
                                            _FREE_QUALITY_RANK in
                                            llm_client.py).
                            The GUI's future provider checkboxes map onto
                            this field, not onto 'provider'.
  provider_preference       Order 'auto' mode tries native providers in
                            (a provider is skipped if it has no key).
                            Default ['anthropic', 'openai', 'gemini'].
  enabled                   Master on/off switch for the whole LLM stack.
  models.gemini             {api_key, model} — Gemini config + default
                            model (gemini-2.5-flash-lite).
  models.claude             {api_key, model} — Anthropic config + default
                            model (claude-haiku-4-5). api_key falls back to
                            the ANTHROPIC_API_KEY env var when empty.
  models.openai             {api_key, model} — OpenAI config + default
                            model (gpt-5.4-nano). api_key falls back to the
                            OPENAI_API_KEY env var when empty.
  endpoints                 List of OpenAI-compatible endpoint configs for
                            selection_mode 'auto'/'free-only'/'best-free'.
                            Each entry:
                              {
                                "name": str,       # unique label, also the
                                                    # per-provider 429-
                                                    # cooldown/probe key
                                "base_url": str,    # e.g. ".../v1" (no
                                                    # trailing "/chat/
                                                    # completions")
                                "api_key": str,     # optional; if empty,
                                                    # falls back to env var
                                                    # '<NAME>_API_KEY'
                                                    # (name uppercased)
                                "model": str,       # model id to request
                                "free": bool,       # true = eligible under
                                                    # 'free-only'/'best-free'
                                "enabled": bool,    # must be true to be
                                                    # considered at all
                              }
                            Recommended presets to enable (not populated by
                            default — the user opts in per endpoint):
                              openrouter — base_url
                                "https://openrouter.ai/api/v1", env
                                OPENROUTER_API_KEY, use a ":free"-suffixed
                                model id (OpenRouter's free tier), free: true
                              groq       — base_url
                                "https://api.groq.com/openai/v1", env
                                GROQ_API_KEY, free tier, free: true
                              ollama     — base_url
                                "http://localhost:11434/v1", no api_key
                                needed (local), free: true
  analyst_model_override    None = use smart routing; accepts any model
                            name in llm_client.KNOWN_MODELS (Gemini,
                            Claude, or OpenAI).
  sentiment_model_override  None = use smart routing (same model universe
                            as analyst_model_override).
  pricing                   Per-MTok price corrections: {"model": [input,
                            output]} — wins over llm_client's built-in
                            table so price changes never need a code
                            change. Same mechanism covers Gemini, Claude,
                            and OpenAI model pricing.
  detected_tier              Auto-detected: 'free' or 'paid' (Gemini only).
  tier_override               Manual override: 'free', 'paid', or None (auto).
  fmp_api_key                 Financial Modeling Prep key (unrelated to LLM
                            provider selection; kept here for historical
                            reasons).
  max_llm_latency_sec         Per-call timeout budget in seconds.
  journal_enabled              Whether LLM call journaling is on.
"""

import json
import os
from pathlib import Path

LLM_CONFIG_FILE = Path(__file__).resolve().parent / "llm_config.json"

_DEFAULTS = {
    # 'auto' = let the multi-provider selection engine pick (see
    # selection_mode below); an EXISTING saved config's legacy value
    # ('gemini'/'anthropic'/'claude'/'openai') is untouched by this default
    # since load_llm_config() only fills keys that are MISSING.
    "provider": "auto",
    "enabled": True,
    "models": {
        "gemini": {"api_key": "", "model": "gemini-2.5-flash-lite"},
        # Current model IDs (the old defaults were legacy/dated snapshots)
        "claude": {"api_key": "", "model": "claude-haiku-4-5"},
        "openai": {"api_key": "", "model": "gpt-5.4-nano"},
    },
    # Selection engine (see module docstring for the full field reference).
    "selection_mode": "auto",           # auto | single | free-only | best-free
    "provider_preference": ["anthropic", "openai", "gemini"],
    "endpoints": [],                    # OpenAI-compatible endpoints (opt-in)
    "analyst_model_override": None,     # None = use smart routing; accepts
                                        # Gemini, Claude, OR OpenAI model names
    "sentiment_model_override": None,   # None = use smart routing
    # Per-MTok price corrections: {"model": [input, output]} — wins over
    # llm_client's built-in table so price changes never need a code change
    "pricing": {},
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
    except Exception as e:
        # A corrupt config silently reverting the WHOLE LLM stack (provider,
        # keys, overrides) to defaults is a debugging nightmare — say so.
        print(f"[LLM-CONFIG] {LLM_CONFIG_FILE.name} unreadable ({e}) — "
              f"falling back to defaults")

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
    """Persist LLM config to disk (atomic tmp+rename — a crash mid-write
    must not corrupt the file that holds every API key and provider switch)."""
    try:
        tmp = str(LLM_CONFIG_FILE) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(config, f, indent=2)
        os.replace(tmp, LLM_CONFIG_FILE)
    except Exception as e:
        print(f"[LLM-CONFIG] Error saving: {e}")
