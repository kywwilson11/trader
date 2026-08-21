"""Unified LLM client — Gemini + Anthropic (Claude) + OpenAI (+ any
OpenAI-compatible endpoint: OpenRouter/Groq/Ollama/...) via urllib (no SDK
deps).

Reads provider + API keys from llm_config.json (Anthropic/OpenAI keys may
also come from the ANTHROPIC_API_KEY / OPENAI_API_KEY env vars; endpoint
keys from '<NAME>_API_KEY'). Returns raw text or None on failure. Never
blocks trades: all errors result in None return.

Calling modes:
  1. call_llm()    — resolve_provider_chain()'s ordered candidate list,
                     tried in order with per-provider cooldowns/budgets
                     (see resolve_provider_chain's docstring for how
                     config['selection_mode'] — 'auto'/'single'/
                     'free-only'/'best-free' — orders candidates). A dead
                     key on one provider no longer silences the gate as
                     long as another provider or endpoint is usable.
  2. call_gemini() — a specific Gemini model (tiered scoring)
  3. call_claude() — a specific Anthropic model
  4. call_openai() — a specific OpenAI (or OpenAI-compatible, via
                     base_url) model
  5. call_model()  — provider-aware dispatch by model name ('claude-*' ->
                     Anthropic, 'gpt-*'/'o*' -> OpenAI, else Gemini); the
                     analyst/sentiment call sites use this so a config
                     override can point any role at any native provider

resolve_provider_chain(role, config) is the selection engine itself:
given a role ('analyst'/'sentiment'/'backfill') and the loaded config, it
returns an ordered [(provider, model, base_url, api_key), ...] list.
call_llm() consumes it directly; get_recommended_model() consults its head
for analyst/sentiment (backfill stays pinned to Gemini's Batch API).

Schema enforcement parity: Gemini uses responseMimeType+responseSchema;
Anthropic uses FORCED TOOL USE (the schema becomes a tool input_schema and
tool_choice pins the model to it); OpenAI uses response_format={'type':
'json_schema', ..., 'strict': True} — all three return guaranteed-
parseable JSON text, so callers are provider-agnostic.

Smart model routing:
  Selects the best model per role (analyst, sentiment, backfill) based on
  daily cost spend. Progressively downgrades as daily cost increases.

Tier auto-detection:
  Detects free vs paid tier from rate limit headers on first API response.
  Uses tier-appropriate budgets and rate limits.

Daily quota tracking:
  Tracks calls per model since midnight Pacific. Budget limits prevent
  burning through RPD (requests per day) limits.

Rate limiting:
  Client-side sliding window prevents hitting per-minute limits.
"""

import collections
import json
import math
import os
import re
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from llm_config import load_llm_config, save_llm_config

# Gemini models ordered by daily quota (most generous first for fallback)
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

_GEMINI_FALLBACK_CHAIN = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",  # Paid tier: Pro has generous limits, fast enough for fallback
]

# Anthropic (Claude) models. The config's models.claude slot existed for a
# long time with NO implementation behind it — this is that implementation.
# Haiku 4.5 is the analyst-tier workhorse (fast, cheap, schema-capable);
# Sonnet 5 is the quality upgrade via config override. Opus is priced for
# research, not a 600s-cadence sizing gate, so it stays out of the chains.
ANTHROPIC_MODELS = ["claude-sonnet-5", "claude-haiku-4-5"]
_ANTHROPIC_FALLBACK_CHAIN = ["claude-haiku-4-5", "claude-sonnet-5"]
_ANTHROPIC_VERSION = "2023-06-01"

# OpenAI (and OpenAI-compatible: OpenRouter/Groq/Ollama) models. Config's
# models.openai slot existed with no implementation either — mirrors the
# Anthropic addition above. nano is the cheap default; the chain climbs to
# mini/full only on fallback.
OPENAI_MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]
_OPENAI_FALLBACK_CHAIN = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"]

# Every model any provider can route to (override validation)
KNOWN_MODELS = GEMINI_MODELS + ANTHROPIC_MODELS + OPENAI_MODELS


def _provider_for(model: str) -> str:
    m = str(model)
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o"):
        return "openai"
    return "gemini"

# HTTP codes that trigger immediate fallback:
_FALLBACK_CODES = {402, 403, 500, 502, 503, 504}

# 429 retry: parse "retry in Xs" from error body for smart wait
_429_MAX_WAIT_PRIMARY = 10   # paid tier: brief wait usually clears it
_429_MAX_WAIT_FALLBACK = 5   # fallback models: don't wait long

# 429 circuit breaker: when a provider's models all fail with 429, skip that
# PROVIDER for a cooldown. Per-provider (2026-07): a Gemini exhaustion must
# not silence a healthy Claude fallback, and vice versa. Endpoint names
# (OpenRouter/Groq/Ollama/...) aren't known ahead of time — _429_cooled_down
# / _trigger_429_cooldown key off whatever string resolve_provider_chain
# hands back, defaulting to "cooled down" (0.0) for keys not yet present.
_429_COOLDOWN_SEC = 30   # short cooldown — paid tier rarely sustains 429s
_429_cooldown_until: dict[str, float] = {"gemini": 0.0, "anthropic": 0.0, "openai": 0.0}

# --- Tier detection ---
_detected_tier: str | None = None  # 'free', 'paid', or None (unknown)

# Mid-2026 free-tier reality: 2.5 Pro was REMOVED from the free tier
# (May 2026); flash-lite is ~15 RPM / ~1,000 RPD; flash ~10 RPM / ~250 RPD.
_FREE_TIER_BUDGETS = {
    "gemini-2.5-pro": 0,
    "gemini-2.5-flash": 250,
    "gemini-2.5-flash-lite": 1000,
    # Anthropic has no free tier; a present key means a paid account, and
    # the tier detector below is Gemini-specific — so Claude budgets are
    # identical in both tables (the $ daily cost cap is the real governor).
    "claude-haiku-4-5": 5000,
    "claude-sonnet-5": 2000,
    # OpenAI has no free tier either; without these rows get_budget falls
    # back to 50 RPD, which silences an OpenAI-primary analyst mid-day
    # (~288 calls/day cadence). Conservative caps — the $ daily cost cap
    # is the real governor (registry hole flagged by Phase-1/B07.2).
    "gpt-5.4-nano": 5000,
    "gpt-5.4-mini": 2000,
    "gpt-5.4": 1000,
}
_PAID_TIER_BUDGETS = {
    "gemini-2.5-pro": 1000,
    "gemini-2.5-flash": 2000,
    "gemini-2.5-flash-lite": 5000,
    "claude-haiku-4-5": 5000,
    "claude-sonnet-5": 2000,
    # OpenAI rows: same rationale as the free-tier table above — avoid the
    # 50-RPD unknown-model default silencing an OpenAI-primary analyst.
    "gpt-5.4-nano": 5000,
    "gpt-5.4-mini": 2000,
    "gpt-5.4": 1000,
}

# Actual responder of the most recent successful call (for journaling —
# the analysis file used to claim 'pro' produced scores that flash wrote)
_last_model_used: str | None = None


def get_last_model_used() -> str | None:
    return _last_model_used

# --- Sliding-window rate limiter ---
_call_timestamps: collections.deque = collections.deque()

# --- Daily quota tracking (resets at midnight Pacific) ---
_model_calls: dict[str, int] = {}
_quota_reset_date: str = ""

# --- Daily cost tracking (hard cap to prevent runaway spending) ---
_DAILY_COST_LIMIT = 1.00  # ~$30/month (paid tier 1)
_daily_cost: float = 0.0
_cost_reset_date: str = ""
_COST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_cost.json")

# Thread safety for quota/cost tracking
_quota_lock = threading.Lock()

# Inter-PROCESS safety for the shared cost file: the crypto and stock books
# can run as separate processes (plus the sentiment backfill worker), and
# _quota_lock is thread-scoped — two processes could each read $0.98, add a
# few cents, and write back, losing an increment against the HARD daily cap.
# flock serializes the read-modify-write. Lock ordering: _quota_lock (thread)
# always OUTER, file lock INNER.
try:
    import fcntl as _fcntl
except ImportError:  # non-POSIX — thread lock still applies
    _fcntl = None


class _cost_file_lock:
    def __enter__(self):
        self._fh = None
        if _fcntl is not None:
            try:
                self._fh = open(_COST_FILE + ".lock", "w")
                _fcntl.flock(self._fh, _fcntl.LOCK_EX)
            except OSError:
                self._fh = None  # degraded: thread lock only
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            try:
                _fcntl.flock(self._fh, _fcntl.LOCK_UN)
                self._fh.close()
            except OSError:
                pass
        return False

# Per-million-token pricing (input, output) — current Gemini list prices.
# The old table (flash 0.15/0.60, lite 0.075/0.30) understated real spend
# 2-4x, so the $1/day cap tripped far later than intended.
_PRICING = {
    "gemini-2.5-pro":        (1.25, 10.0),
    "gemini-2.5-flash":      (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    # Anthropic list prices per MTok. Prices move — llm_config.json may
    # carry a "pricing": {model: [in, out]} override that wins over this
    # table (see _pricing), so corrections never need a code change.
    "claude-haiku-4-5":      (1.00, 5.00),
    "claude-sonnet-5":       (3.00, 15.00),
    "claude-opus-4-8":       (5.00, 25.00),
    # ⚠️ LOUD WARNING: these OpenAI gpt-5.4 prices are CONSERVATIVE
    # PLACEHOLDERS, not verified against OpenAI's published pricing page.
    # Correct them via config['pricing'] overrides (same mechanism as the
    # Claude/Gemini corrections above) the moment real prices are known —
    # do NOT trust these numbers for real budget/cost-cap decisions.
    "gpt-5.4":               (5.00, 15.00),
    "gpt-5.4-mini":          (1.00, 4.00),
    "gpt-5.4-nano":          (0.25, 1.00),
}

# Models already warned about this process (one loud line per unknown model,
# not one per call) — billing an unknown model at the conservative pro-tier
# fallback silently distorts the $1/day cap either direction.
_unknown_price_warned: set = set()


def _pricing(model: str) -> tuple[float, float]:
    """Per-MTok (input, output) price: config override > table > pro-tier."""
    try:
        override = load_llm_config().get("pricing", {}).get(model)
        if override and len(override) == 2:
            return (float(override[0]), float(override[1]))
    except Exception:
        pass
    if model not in _PRICING and model not in _unknown_price_warned:
        _unknown_price_warned.add(model)
        print(f"[LLM-COST] WARNING: no pricing entry for model '{model}' — "
              f"billing at conservative fallback ($1.25/$10.00 per MTok). "
              f"Add a config['pricing'] entry in llm_config.json.")
    return _PRICING.get(model, (1.25, 10.0))


_CACHE_MULT_DEFAULTS = {"anthropic": (1.25, 0.10), "gemini": (1.00, 0.25)}


def _cache_multipliers(provider: str) -> tuple[float, float]:
    """(write_mult, read_mult) vs input price for cache-billed tokens:
    config['pricing_cache_multipliers'] override > built-in defaults.
    Unknown provider -> (1.0, 0.0) (cache tokens priced as no-ops)."""
    try:
        m = load_llm_config().get("pricing_cache_multipliers", {}).get(provider)
        if m and len(m) == 2:
            return (float(m[0]), float(m[1]))
    except Exception:
        pass
    return _CACHE_MULT_DEFAULTS.get(provider, (1.0, 0.0))

# --- Smart model routing ---
# Cost brackets: {max_daily_cost: {role: model}}
# The analyst gate is a SIZING input, not research: flash-lite with an
# enforced response schema is fast, ~$7/month at this call volume, and —
# unlike the old pro routing — doesn't blow the latency budget on thinking
# tokens and then silently demote to an unschema'd fallback.
_PAID_ROUTING = [
    (0.40, {"analyst": "gemini-2.5-flash-lite", "sentiment": "gemini-2.5-flash-lite", "backfill": "gemini-2.5-flash-lite"}),
    (math.inf, {"analyst": "gemini-2.5-flash-lite", "sentiment": "gemini-2.5-flash-lite", "backfill": "gemini-2.5-flash-lite"}),
]

# Free tier: Pro is no longer available; flash-lite has the only generous RPD
_FREE_ROUTING = [
    (math.inf, {"analyst": "gemini-2.5-flash-lite", "sentiment": "gemini-2.5-flash-lite", "backfill": "gemini-2.5-flash-lite"}),
]


def _get_rate_limit_rpm() -> int:
    """Get rate limit RPM based on detected tier."""
    tier = get_tier()
    return 10 if tier == 'free' else 30


def _get_budgets() -> dict:
    """Get tier-appropriate daily budgets."""
    return _FREE_TIER_BUDGETS if get_tier() == 'free' else _PAID_TIER_BUDGETS


# --- Tier detection ---

def get_tier() -> str:
    """Get current tier. Priority: manual override > detected > cached > 'paid'."""
    config = load_llm_config()

    # 1. Manual override
    override = config.get("tier_override")
    if override in ('free', 'paid'):
        return override

    # 2. Runtime detection
    global _detected_tier
    if _detected_tier:
        return _detected_tier

    # 3. Cached detection from config
    cached = config.get("detected_tier")
    if cached in ('free', 'paid'):
        _detected_tier = cached
        return cached

    # 4. Default to paid (user confirmed they're on paid tier)
    return 'paid'


def _capture_rate_limit_headers(resp, model: str):
    """Check rate limit headers from API response to detect tier.

    Gemini returns x-ratelimit-limit-requests header:
      - Free tier: RPM <= 15
      - Paid tier: RPM >= 30
    """
    global _detected_tier
    if _detected_tier is not None:
        return  # already detected

    try:
        rpm_header = resp.getheader('x-ratelimit-limit-requests')
        if rpm_header is None:
            return

        rpm = int(rpm_header)
        if rpm <= 15:
            _detected_tier = 'free'
        else:
            _detected_tier = 'paid'

        print(f"[LLM] Tier detected: {_detected_tier.upper()} (RPM limit={rpm} for {model})")

        # Persist to config
        config = load_llm_config()
        config['detected_tier'] = _detected_tier
        save_llm_config(config)

    except (TypeError, ValueError):
        pass


def probe_tier() -> str:
    """Make a cheap API call to detect tier from headers. Returns tier string."""
    global _detected_tier
    if _detected_tier:
        return _detected_tier

    config = load_llm_config()
    gemini_key = config.get("models", {}).get("gemini", {}).get("api_key", "")
    if not gemini_key:
        return get_tier()

    try:
        # Minimal call to trigger header capture
        _call_gemini("Hi", "", gemini_key, "gemini-2.5-flash-lite", 10, 10)[0]
    except Exception:
        pass

    return get_tier()


# --- Multi-provider selection engine ---

# 'best-free' quality ranking: lower sorts first. Governs ONLY 'best-free'
# ordering of the free-candidate set that 'free-only' already selects.
# Named free models rank explicitly; endpoints (whose quality varies by
# what the user pointed them at) rank after every named model, in the
# order they appear in config['endpoints'].
_FREE_QUALITY_RANK = {
    "gemini-2.5-pro": 0,
    "gemini-2.5-flash": 1,
    "gemini-2.5-flash-lite": 2,
}
_FREE_ENDPOINT_RANK = 10


def _endpoint_api_key(endpoint: dict) -> str:
    """Endpoint credential: explicit api_key, else env '<NAME>_API_KEY',
    else '' (keyless — e.g. a local Ollama server, which needs no key)."""
    key = endpoint.get("api_key") or ""
    if key:
        return key
    name = str(endpoint.get("name") or "").strip()
    if not name:
        return ""
    return os.environ.get(f"{name.upper()}_API_KEY", "")


def _enabled_endpoints(config: dict, free_only: bool = False) -> list:
    """config['endpoints'] entries with enabled: true (optionally also
    filtered to free: true — 'free-only'/'best-free' selection modes)."""
    out = []
    for ep in config.get("endpoints") or []:
        if not isinstance(ep, dict) or not ep.get("enabled"):
            continue
        if free_only and not ep.get("free"):
            continue
        out.append(ep)
    return out


def resolve_provider_chain(role: str, config: dict):
    """Ordered candidate list for `role`: [(provider, model, base_url, api_key), ...].

    `provider` is 'anthropic', 'gemini', 'openai', or an endpoint's `name`
    (any provider string other than 'anthropic'/'gemini' is dispatched as
    an OpenAI-compatible call by call_llm — see its _dispatch helper).
    `base_url` is None for the three native providers and the endpoint's
    configured base_url otherwise.

    Backfill is PINNED to Gemini regardless of selection_mode — it rides
    the Gemini Batch API, which has no other-provider path wired; every
    call site needing that pin should route through here (or through
    get_recommended_model('backfill'), which special-cases it identically).

    config['selection_mode'] (default 'auto') governs everything else:
      'single'    — just the model configured for config['provider']
                    ('anthropic'/'claude'/'openai'/'gemini' — the legacy
                    field that predates selection_mode; this is how it's
                    still honored). No fallback chain, no cross-provider.
      'auto'      — config['provider_preference'] order (default
                    ['anthropic', 'openai', 'gemini']), skipping any
                    provider with no usable key; each contributing
                    provider adds its primary model then its own fallback
                    chain, then every enabled endpoint is appended last.
      'free-only' — only free candidates: enabled endpoints with
                    free: true (this also covers keyless local endpoints
                    like Ollama — they just have no api_key to check),
                    plus Gemini appended as an always-available free-tier
                    last resort (governed by the existing daily-cost cap
                    and RPD budgets either way).
      'best-free' — the same candidate set as 'free-only', reordered by
                    _FREE_QUALITY_RANK instead of config order.
    """
    if role == "backfill":
        gem_key = config.get("models", {}).get("gemini", {}).get("api_key", "")
        gem_model = (config.get("models", {}).get("gemini", {}).get("model")
                     or "gemini-2.5-flash-lite")
        return [("gemini", gem_model, None, gem_key)] if gem_key else []

    selection_mode = config.get("selection_mode") or "auto"
    provider = str(config.get("provider") or "auto").lower()

    if selection_mode == "single":
        if provider in ("anthropic", "claude"):
            model = (config.get("models", {}).get("claude", {}).get("model")
                     or "claude-haiku-4-5")
            key = _anthropic_key(config)
            return [("anthropic", model, None, key)] if key else []
        if provider == "openai":
            model = (config.get("models", {}).get("openai", {}).get("model")
                     or "gpt-5.4-nano")
            key = _openai_key(config)
            return [("openai", model, None, key)] if key else []
        # 'gemini', 'auto', or anything unrecognized — Gemini is the
        # original default single provider.
        model = (config.get("models", {}).get("gemini", {}).get("model")
                 or "gemini-2.5-flash")
        key = config.get("models", {}).get("gemini", {}).get("api_key", "")
        return [("gemini", model, None, key)] if key else []

    if selection_mode in ("free-only", "best-free"):
        chain = []
        for ep in _enabled_endpoints(config, free_only=True):
            chain.append((ep.get("name") or "endpoint", ep.get("model", ""),
                          ep.get("base_url"), _endpoint_api_key(ep)))
        gem_key = config.get("models", {}).get("gemini", {}).get("api_key", "")
        if gem_key:
            gem_model = (config.get("models", {}).get("gemini", {}).get("model")
                         or "gemini-2.5-flash-lite")
            chain.append(("gemini", gem_model, None, gem_key))
            for fb in _GEMINI_FALLBACK_CHAIN:
                if fb != gem_model:
                    chain.append(("gemini", fb, None, gem_key))
        if selection_mode == "best-free":
            def _rank(entry):
                prov, model, base_url, _key = entry
                if prov == "gemini" and base_url is None:
                    return _FREE_QUALITY_RANK.get(model, _FREE_ENDPOINT_RANK)
                return _FREE_ENDPOINT_RANK
            chain.sort(key=_rank)
        return chain

    # 'auto' (default)
    preference = config.get("provider_preference") or ["anthropic", "openai", "gemini"]
    chain = []
    for prov in preference:
        prov = str(prov).lower()
        if prov in ("anthropic", "claude"):
            key = _anthropic_key(config)
            if not key:
                continue
            primary = (config.get("models", {}).get("claude", {}).get("model")
                       or "claude-haiku-4-5")
            chain.append(("anthropic", primary, None, key))
            for fb in _ANTHROPIC_FALLBACK_CHAIN:
                if fb != primary:
                    chain.append(("anthropic", fb, None, key))
        elif prov == "openai":
            key = _openai_key(config)
            if not key:
                continue
            primary = (config.get("models", {}).get("openai", {}).get("model")
                       or "gpt-5.4-nano")
            chain.append(("openai", primary, None, key))
            for fb in _OPENAI_FALLBACK_CHAIN:
                if fb != primary:
                    chain.append(("openai", fb, None, key))
        elif prov == "gemini":
            key = config.get("models", {}).get("gemini", {}).get("api_key", "")
            if not key:
                continue
            primary = (config.get("models", {}).get("gemini", {}).get("model")
                       or "gemini-2.5-flash")
            chain.append(("gemini", primary, None, key))
            for fb in _GEMINI_FALLBACK_CHAIN:
                if fb != primary:
                    chain.append(("gemini", fb, None, key))
        # unrecognized provider_preference entries are silently skipped —
        # only 'anthropic'/'claude', 'openai', 'gemini' are native
    for ep in _enabled_endpoints(config):
        chain.append((ep.get("name") or "endpoint", ep.get("model", ""),
                      ep.get("base_url"), _endpoint_api_key(ep)))
    return chain


# --- Smart model routing ---

def get_recommended_model(role: str) -> str:
    """Get the recommended model for a role based on daily cost and tier.

    Args:
        role: 'analyst', 'sentiment', or 'backfill'

    Returns:
        Model name string (e.g. 'gemini-2.5-pro')
    """
    config = load_llm_config()

    # 1. Check manual override (either provider's models)
    override_key = f"{role}_model_override"
    override = config.get(override_key)
    if override and override in KNOWN_MODELS:
        return override

    # 1b. Provider selection via the resolve_provider_chain head: when the
    # chain's first candidate is Anthropic or OpenAI (native providers —
    # keyed by model-name prefix so call_model can route to them), send
    # analyst/sentiment there directly. This generalizes the old hardcoded
    # "Anthropic primary" branch to any provider selection_mode/
    # provider_preference produces. Scoped to anthropic/openai (not
    # arbitrary endpoints) because get_recommended_model returns a bare
    # model-name string — call_model dispatches purely by name prefix
    # (_provider_for) and has no base_url to reach a custom endpoint with;
    # endpoint routing works end-to-end through call_llm's own chain loop,
    # just not through this model-name-only path.
    # Backfill is PINNED to Gemini regardless — sentiment_history's
    # backfill rides the Gemini BATCH API, which has no other-provider path
    # wired.
    if role != 'backfill':
        chain = resolve_provider_chain(role, config)
        if chain and chain[0][0] in ("anthropic", "openai"):
            head_model = chain[0][1]
            if head_model:
                return head_model

    # 2. Select routing table based on tier
    tier = get_tier()
    routing = _FREE_ROUTING if tier == 'free' else _PAID_ROUTING

    # 3. Find bracket based on daily cost
    _maybe_reset_quota()
    for threshold, models in routing:
        if _daily_cost < threshold:
            recommended = models.get(role, "gemini-2.5-flash-lite")
            break
    else:
        recommended = "gemini-2.5-flash-lite"

    # 4. Verify recommended model has budget remaining; downgrade if exhausted
    budgets = _get_budgets()
    remaining = budgets.get(recommended, 50) - _model_calls.get(recommended, 0)
    if remaining <= 0:
        # Try downgrading through the model list
        downgrade_order = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
        if recommended == "gemini-2.5-flash":
            downgrade_order = ["gemini-2.5-flash-lite"]
        elif recommended == "gemini-2.5-flash-lite":
            downgrade_order = []

        for fallback in downgrade_order:
            fb_remaining = budgets.get(fallback, 50) - _model_calls.get(fallback, 0)
            if fb_remaining > 0:
                return fallback
        return recommended  # all exhausted, return anyway (call will fail gracefully)

    return recommended


def get_routing_info() -> dict:
    """Get routing info for GUI display."""
    _maybe_reset_quota()
    tier = get_tier()
    budgets = _get_budgets()
    return {
        'tier': tier,
        'daily_cost': round(_daily_cost, 4),
        'daily_limit': _DAILY_COST_LIMIT,
        'analyst_model': get_recommended_model('analyst'),
        'sentiment_model': get_recommended_model('sentiment'),
        'backfill_model': get_recommended_model('backfill'),
        'budgets': {
            model: {
                'used': _model_calls.get(model, 0),
                'total': budgets.get(model, 0),
            }
            for model in GEMINI_MODELS
        },
    }


def _parse_retry_after(http_error) -> float | None:
    """Extract retry delay from a 429 error response body."""
    try:
        body = http_error.read().decode("utf-8", errors="replace")
        if "limit: 0" in body:
            return None  # Daily quota exhausted
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", body, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    # If regex doesn't match, default to exponential backoff
    return 30.0  # Default 30s backoff instead of None


def _rate_limit_ok() -> bool:
    """Check if we're within the per-minute rate limit (tier-aware)."""
    now = time.time()
    cutoff = now - 60.0
    while _call_timestamps and _call_timestamps[0] < cutoff:
        _call_timestamps.popleft()
    rpm = _get_rate_limit_rpm()
    if len(_call_timestamps) >= rpm:
        return False
    _call_timestamps.append(now)
    return True


def _429_cooled_down(provider: str = "gemini") -> bool:
    """Check if the PROVIDER is past its 429 cooldown period."""
    return time.time() >= _429_cooldown_until.get(provider, 0.0)


def _trigger_429_cooldown(provider: str = "gemini"):
    """A provider's models all 429'd — skip that provider for a cooldown."""
    _429_cooldown_until[provider] = time.time() + _429_COOLDOWN_SEC
    print(f"[LLM] {provider}: all models rate-limited, cooling down {_429_COOLDOWN_SEC}s")


def _load_shared_cost():
    """Load daily cost from shared file (cross-process visibility)."""
    global _daily_cost, _cost_reset_date
    try:
        with open(_COST_FILE) as f:
            data = json.load(f)
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
        if data.get("date") == today:
            _daily_cost = data.get("cost", 0.0)
            _cost_reset_date = today
    except (OSError, json.JSONDecodeError, ValueError):
        pass


def _save_shared_cost():
    """Persist daily cost to shared file (cross-process visibility)."""
    today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    data = {"date": today, "cost": round(_daily_cost, 6)}
    try:
        tmp = _COST_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, _COST_FILE)
    except OSError:
        pass


def _maybe_reset_quota():
    """Reset daily quota and cost counters at midnight Pacific."""
    global _quota_reset_date, _cost_reset_date, _daily_cost
    with _quota_lock:
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
        if _quota_reset_date != today:
            _model_calls.clear()
            _quota_reset_date = today
        if _cost_reset_date != today:
            # Load shared cost file first — another process may have spent today
            _load_shared_cost()
            if _cost_reset_date != today:
                # Still not today's date — fresh day
                if _daily_cost > 0:
                    print(f"[LLM] Daily cost reset (yesterday: ${_daily_cost:.4f})")
                _daily_cost = 0.0
                _cost_reset_date = today
                _save_shared_cost()


def _estimate_cost(model: str, prompt_chars: int, response_chars: int) -> float:
    """Estimate API cost from character counts (~4 chars per token).

    Fallback only — usage-based costing (_record_cost with usage metadata)
    is exact and includes thinking tokens, which chars/4 cannot see.
    """
    input_tokens = prompt_chars / 4
    output_tokens = response_chars / 4
    price_in, price_out = _pricing(model)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def _record_cost(model: str, prompt_chars: int, response_chars: int,
                 usage: dict | None = None):
    """Record cost (from API usageMetadata when available) to the shared file.

    Cache-aware (B07.2): Anthropic cache_creation/cache_read tokens are
    reported SEPARATELY from input_tokens and are added at the registry
    write/read multipliers; Gemini's cachedContentTokenCount is INCLUDED in
    promptTokenCount and is credited back to the read-multiplier rate.
    With no cache activity both formulas reduce exactly to the pre-change
    arithmetic. NEVER raises: a costing failure must not discard an
    already-received LLM result (every transport call site invokes this
    inside its own try block) — degrade to the char estimate, then to $0.
    """
    global _daily_cost
    try:
        if usage and usage.get('promptTokenCount') is not None:
            price_in, price_out = _pricing(model)
            in_tok = usage.get('promptTokenCount', 0) or 0
            # candidatesTokenCount excludes thinking tokens; thoughtsTokenCount
            # is billed as output too
            out_tok = ((usage.get('candidatesTokenCount', 0) or 0)
                       + (usage.get('thoughtsTokenCount', 0) or 0))
            cost = (in_tok * price_in + out_tok * price_out) / 1_000_000
            provider = _provider_for(model)
            if provider == "anthropic":
                cw = usage.get('cacheWriteTokenCount', 0) or 0
                cr = usage.get('cacheReadTokenCount', 0) or 0
                if cw or cr:
                    wm, rm = _cache_multipliers("anthropic")
                    cost += (cw * wm + cr * rm) * price_in / 1_000_000
            elif provider == "gemini":
                cached = usage.get('cachedContentTokenCount', 0) or 0
                if cached:
                    _wm, rm = _cache_multipliers("gemini")
                    cost -= cached * (1.0 - rm) * price_in / 1_000_000
        else:
            cost = _estimate_cost(model, prompt_chars, response_chars)
    except Exception as e:
        print(f"[LLM-COST] cost computation failed for {model}: {e} — "
              f"falling back to char estimate")
        try:
            cost = _estimate_cost(model, prompt_chars, response_chars)
        except Exception:
            cost = 0.0
    try:
        with _quota_lock:
            with _cost_file_lock():
                # Re-read under the FILE lock so a concurrent process's spend
                # cannot be lost in this read-modify-write
                _load_shared_cost()
                _daily_cost += cost
                _save_shared_cost()
    except Exception as e:
        print(f"[LLM-COST] ledger write failed: {e} "
              f"(${cost:.6f} may be unrecorded)")


def _cost_ok() -> bool:
    """Check if we're under the daily cost limit."""
    _maybe_reset_quota()
    if _daily_cost >= _DAILY_COST_LIMIT:
        return False
    return True


def get_daily_cost() -> tuple[float, float]:
    """Return (spent_today, daily_limit) for monitoring.

    Reads shared cost file so GUI sees costs from all processes.
    """
    _maybe_reset_quota()
    with _quota_lock:
        _load_shared_cost()
    return _daily_cost, _DAILY_COST_LIMIT


def get_budget(model: str) -> tuple[int, int]:
    """Return (remaining, total) daily budget for a model (tier-aware)."""
    _maybe_reset_quota()
    budgets = _get_budgets()
    total = budgets.get(model, 50)
    used = _model_calls.get(model, 0)
    return max(0, total - used), total


def record_call(model: str):
    """Record that we made an API call to this model."""
    _maybe_reset_quota()
    with _quota_lock:
        _model_calls[model] = _model_calls.get(model, 0) + 1


# --- Public API ---

def call_gemini(prompt: str, system: str = "", model: str = "gemini-2.5-flash",
                max_tokens: int = 2048, json_mode: bool = False,
                json_schema: dict | None = None,
                temperature: float | None = None,
                timeout: float | None = None) -> str | None:
    """Call a specific Gemini model. Returns text or None.

    Used by tiered scoring to target a specific model. Handles 429 with
    retry-after parsing. Does NOT fall back to other models (caller decides).
    """
    global _last_model_used
    if not _429_cooled_down('gemini'):
        return None

    if not _cost_ok():
        print(f"[LLM] Daily cost limit reached (${_daily_cost:.2f}/${_DAILY_COST_LIMIT:.2f})")
        return None

    config = load_llm_config()
    if not config.get("enabled"):
        return None

    gemini_key = config.get("models", {}).get("gemini", {}).get("api_key", "")
    if not gemini_key:
        return None

    if not _rate_limit_ok():
        print(f"[LLM] Rate limit reached, skipping {model}")
        return None

    remaining, total = get_budget(model)
    if remaining <= 0:
        print(f"[LLM] {model}: daily budget exhausted ({total} RPD)")
        return None

    prompt_chars = len(prompt) + len(system)
    if timeout is None:
        timeout = config.get("max_llm_latency_sec", 30)
    start = time.time()

    try:
        result, usage = _call_gemini(prompt, system, gemini_key, model, max_tokens,
                                     timeout, json_mode=json_mode,
                                     json_schema=json_schema,
                                     temperature=temperature)
        elapsed = (time.time() - start) * 1000
        if result:
            record_call(model)
            _record_cost(model, prompt_chars, len(result), usage)
            _last_model_used = model
            print(f"[LLM] {model}: {elapsed:.0f}ms, {len(result)} chars (${_daily_cost:.3f} today)")
        return result

    except urllib.error.HTTPError as e:
        elapsed = (time.time() - start) * 1000
        if e.code == 429:
            wait = _parse_retry_after(e)
            if wait and wait <= _429_MAX_WAIT_PRIMARY:
                print(f"[LLM] {model}: 429, waiting {wait:.0f}s")
                time.sleep(wait)
                try:
                    start2 = time.time()
                    result, usage = _call_gemini(prompt, system, gemini_key, model,
                                                 max_tokens, timeout,
                                                 json_mode=json_mode,
                                                 json_schema=json_schema,
                                                 temperature=temperature)
                    elapsed2 = (time.time() - start2) * 1000
                    if result:
                        record_call(model)
                        _record_cost(model, prompt_chars, len(result), usage)
                        _last_model_used = model
                        print(f"[LLM] {model}: {elapsed2:.0f}ms, {len(result)} chars (after wait)")
                    return result
                except Exception:
                    pass
            print(f"[LLM] {model}: 429 exhausted ({elapsed:.0f}ms)")
        else:
            print(f"[LLM] {model}: HTTP {e.code} ({elapsed:.0f}ms)")
        return None

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        print(f"[LLM] {model}: {e} ({elapsed:.0f}ms)")
        return None


def call_llm(prompt: str, system: str = "", max_tokens: int = 2048,
             json_schema: dict | None = None,
             temperature: float | None = None,
             role: str = "analyst") -> str | None:
    """Send prompt through the resolved provider chain. Returns text or None.

    Generalizes the old hardcoded Gemini-primary / Anthropic-primary
    branching into a single loop over resolve_provider_chain(role, config)
    — see that function's docstring for how selection_mode ('auto' by
    default) orders candidates. Legacy config['provider'] values
    ('gemini'/'anthropic'/'claude'/'openai') are honored only under
    selection_mode='single' (backward compat for a saved Jetson config
    that hard-pins one provider); under the 'auto' default they don't gate
    anything — whichever providers have usable keys are tried in
    provider_preference order, preserving the old cross-provider
    resilience (a dead Gemini key no longer silences the analyst gate, and
    vice versa) now generalized to any number of providers/endpoints.

    The chain now ALSO fires on the dominant real-world failures the old
    code returned None for — socket timeouts, MAX_TOKENS/length
    truncation, and safety blocks — not just on specific HTTP codes.

    role: which resolve_provider_chain role to resolve against. Existing
    call_llm call sites don't distinguish analyst/sentiment/backfill, so
    the default 'analyst' preserves their behavior; role only matters for
    the backfill-pinned-to-Gemini rule (backfill goes through
    get_recommended_model + call_model instead, so no current call_llm
    caller needs to pass it).
    """
    global _last_model_used
    if not _cost_ok():
        print(f"[LLM] Daily cost limit reached (${_daily_cost:.2f}/${_DAILY_COST_LIMIT:.2f})")
        return None

    config = load_llm_config()
    if not config.get("enabled"):
        return None

    if not _rate_limit_ok():
        print("[LLM] Rate limit reached, skipping")
        return None

    chain = resolve_provider_chain(role, config)
    if not chain:
        return None

    timeout = config.get("max_llm_latency_sec", 30)
    prompt_chars = len(prompt) + len(system)

    def _dispatch(provider, model, base_url, api_key):
        if provider == "anthropic":
            return _call_anthropic(prompt, system, api_key, model, max_tokens,
                                   timeout, json_schema=json_schema,
                                   temperature=temperature)
        if provider == "gemini":
            return _call_gemini(prompt, system, api_key, model, max_tokens,
                                timeout, json_schema=json_schema,
                                temperature=temperature)
        # 'openai' or any OpenAI-compatible endpoint name
        return _call_openai(prompt, system, api_key, model, max_tokens,
                            timeout, json_schema=json_schema,
                            temperature=temperature, base_url=base_url)

    for i, (provider, model, base_url, api_key) in enumerate(chain):
        if provider in ("anthropic", "openai", "gemini") and not api_key:
            continue  # endpoints may legitimately be keyless (e.g. Ollama)
        if not _429_cooled_down(provider):
            continue
        remaining, _total = get_budget(model)
        if remaining <= 0:
            continue

        max_wait = _429_MAX_WAIT_PRIMARY if i == 0 else _429_MAX_WAIT_FALLBACK
        start = time.time()
        try:
            result, usage = _dispatch(provider, model, base_url, api_key)
            elapsed = (time.time() - start) * 1000
            if result:
                record_call(model)
                _record_cost(model, prompt_chars, len(result), usage)
                _last_model_used = model
                print(f"[LLM] {provider}/{model}: {elapsed:.0f}ms, {len(result)} chars "
                      f"(${_daily_cost:.3f} today)")
                return result
            print(f"[LLM] {provider}/{model}: empty/truncated, trying next")
            continue
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (_parse_retry_after(e) if provider == "gemini"
                        else _parse_retry_after_anthropic(e))
                if wait and wait <= max_wait:
                    print(f"[LLM] {provider}/{model}: 429, waiting {wait:.0f}s")
                    time.sleep(wait)
                    try:
                        result, usage = _dispatch(provider, model, base_url, api_key)
                        if result:
                            record_call(model)
                            _record_cost(model, prompt_chars, len(result), usage)
                            _last_model_used = model
                            print(f"[LLM] {provider}/{model}: {len(result)} chars (after wait)")
                            return result
                    except Exception:
                        pass
                print(f"[LLM] {provider}/{model}: 429, trying next")
                _trigger_429_cooldown(provider)
                continue
            print(f"[LLM] {provider}/{model}: HTTP {e.code}, trying next")
            continue
        except Exception as e:
            print(f"[LLM] {provider}/{model}: {e}, trying next")
            continue

    print("[LLM] All providers/models in chain exhausted")
    return None


# --- Anthropic (Claude) support ---

def _anthropic_key(config: dict | None = None) -> str:
    """Anthropic key: llm_config models.claude.api_key, else ANTHROPIC_API_KEY env."""
    config = config or load_llm_config()
    key = config.get("models", {}).get("claude", {}).get("api_key", "")
    return key or os.environ.get("ANTHROPIC_API_KEY", "")


def _openai_key(config: dict | None = None) -> str:
    """OpenAI key: llm_config models.openai.api_key, else OPENAI_API_KEY env."""
    config = config or load_llm_config()
    key = config.get("models", {}).get("openai", {}).get("api_key", "")
    return key or os.environ.get("OPENAI_API_KEY", "")


def _parse_retry_after_anthropic(http_error) -> float | None:
    """Anthropic 429s carry a standard retry-after header (seconds)."""
    try:
        ra = http_error.headers.get("retry-after")
        if ra is not None:
            return float(ra)
    except Exception:
        pass
    return 15.0


def _normalize_schema_for_anthropic(schema):
    """Translate a Gemini-dialect schema into standard JSON Schema.

    Callers historically author schemas for Gemini's responseSchema, which
    accepts OpenAPI-style UPPERCASE type names ('OBJECT', 'NUMBER', ...) and
    Gemini-only keys like propertyOrdering. Anthropic's tool input_schema is
    strict JSON Schema — uppercase types are invalid and can fail the whole
    call, silently disabling the analyst gate on a Claude config (fail-open
    masks it). Lowercase every 'type', drop Gemini-only keys, recurse.
    """
    if isinstance(schema, list):
        return [_normalize_schema_for_anthropic(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k == 'propertyOrdering':
            continue  # Gemini-only hint, not JSON Schema
        if k == 'type':
            if isinstance(v, str):
                out[k] = v.lower()
            elif isinstance(v, list):
                out[k] = [t.lower() if isinstance(t, str) else t for t in v]
            else:
                out[k] = v
        elif isinstance(v, (dict, list)):
            out[k] = _normalize_schema_for_anthropic(v)
        else:
            out[k] = v
    return out


def _normalize_schema_for_openai(schema):
    """Translate a (possibly Gemini-dialect) schema into OpenAI's strict
    json_schema form.

    Same lowercase-type + propertyOrdering-drop translation as
    _normalize_schema_for_anthropic, PLUS what OpenAI's strict mode
    additionally requires: 'additionalProperties': false on every object
    schema (recursively — nested 'properties'/'items' objects included).
    Without it, strict schema validation rejects the request outright.
    """
    if isinstance(schema, list):
        return [_normalize_schema_for_openai(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k == 'propertyOrdering':
            continue  # Gemini-only hint, not JSON Schema
        if k == 'type':
            if isinstance(v, str):
                out[k] = v.lower()
            elif isinstance(v, list):
                out[k] = [t.lower() if isinstance(t, str) else t for t in v]
            else:
                out[k] = v
        elif isinstance(v, (dict, list)):
            out[k] = _normalize_schema_for_openai(v)
        else:
            out[k] = v
    type_val = out.get('type')
    is_object = type_val == 'object' or (
        isinstance(type_val, list) and 'object' in type_val)
    if is_object or 'properties' in out:
        out.setdefault('additionalProperties', False)
    return out


def _call_anthropic(prompt, system, api_key, model, max_tokens, timeout,
                    json_mode=False, json_schema=None, temperature=None):
    """Call the Anthropic Messages API. Returns (text|None, usage|None);
    raises urllib errors for the caller's retry/fallback logic.

    json_schema: enforced via FORCED TOOL USE — the schema becomes a tool's
    input_schema and tool_choice pins the model to that tool, so the returned
    tool_use input is schema-validated JSON. It is re-serialized to a JSON
    string so callers parse both providers identically. Usage is normalized
    to Gemini's usageMetadata key names so _record_cost stays provider-
    agnostic. json_mode without a schema is best-effort (prompt discipline).
    """
    url = "https://api.anthropic.com/v1/messages"
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    cache_ttl = ""
    if system:
        # Default-OFF prompt-cache breakpoint (config
        # 'anthropic_cache_system_ttl'; see llm_config.py — B07.2 forbids
        # enabling under the Haiku default). OFF or any config error ->
        # plain-string system, byte-identical to pre-change.
        try:
            cache_ttl = str(load_llm_config().get(
                "anthropic_cache_system_ttl") or "")
        except Exception:
            cache_ttl = ""
        if cache_ttl in ("5m", "1h"):
            cc = {"type": "ephemeral"}
            if cache_ttl == "1h":
                cc["ttl"] = "1h"
            body["system"] = [{"type": "text", "text": system,
                               "cache_control": cc}]
        else:
            body["system"] = system
    if temperature is not None:
        body["temperature"] = temperature
    if json_schema is not None:
        body["tools"] = [{
            "name": "emit_json",
            "description": "Emit the structured answer in the required schema.",
            "input_schema": _normalize_schema_for_anthropic(json_schema),
        }]
        body["tool_choice"] = {"type": "tool", "name": "emit_json"}

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    data = json.loads(resp.read())
    u = data.get("usage") or {}
    usage = {
        "promptTokenCount": u.get("input_tokens", 0),
        "candidatesTokenCount": u.get("output_tokens", 0),
        "thoughtsTokenCount": 0,
    }
    # Cache accounting fields — added ONLY when present-and-nonzero so the
    # normalized dict stays exactly {prompt,candidates,thoughts} on the
    # default no-cache path (exact-equality pins in test_llm_claude.py).
    cw = u.get("cache_creation_input_tokens") or 0
    cr = u.get("cache_read_input_tokens") or 0
    if cw:
        usage["cacheWriteTokenCount"] = cw
    if cr:
        usage["cacheReadTokenCount"] = cr
    if cache_ttl in ("5m", "1h") and not cw and not cr:
        # Flag is ON but the API reported no cache tokens — the prefix is
        # below the model's minimum cacheable length or was invalidated
        # (per-symbol tool schema changed). Loud so the owner sees it.
        print(f"[LLM] Claude {model}: cache_control active but no cache "
              f"tokens reported (prefix below model minimum?)")
    stop = data.get("stop_reason", "unknown")
    text_parts = []
    for block in data.get("content", []) or []:
        if block.get("type") == "tool_use" and json_schema is not None:
            return json.dumps(block.get("input", {})), usage
        if block.get("type") == "text" and block.get("text", "").strip():
            text_parts.append(block["text"])
    if stop == "max_tokens":
        print(f"[LLM] Claude: truncated ({sum(len(t) for t in text_parts)} chars), discarding")
        return None, usage
    if text_parts:
        return "".join(text_parts), usage
    print(f"[LLM] Claude: no usable content (stop={stop})")
    return None, usage


_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _call_openai(prompt, system, api_key, model, max_tokens, timeout,
                 json_schema=None, temperature=None, base_url=None):
    """Call an OpenAI-compatible /chat/completions endpoint. Returns
    (text|None, usage|None); raises urllib errors for the caller's
    retry/fallback logic.

    Works against OpenAI itself (base_url=None -> the default OpenAI API)
    and any OpenAI-compatible endpoint (OpenRouter/Groq/Ollama/...) via the
    base_url override — same request shape, since they all implement the
    Chat Completions wire format.

    json_schema: enforced via response_format={'type': 'json_schema',
    'json_schema': {'name': 'emit_json', 'strict': True, 'schema': ...}} —
    the schema is normalized (_normalize_schema_for_openai) since strict
    mode requires lowercase types and additionalProperties:false on every
    object. Usage is normalized to the Gemini-style dict so _record_cost
    stays provider-agnostic, same as _call_anthropic.
    """
    base = (base_url or _DEFAULT_OPENAI_BASE_URL).rstrip("/")
    url = f"{base}/chat/completions"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        # Native OpenAI rejects 'max_tokens' outright on the gpt-5 family
        # ("Unsupported parameter ... use 'max_completion_tokens'"), while
        # third-party OpenAI-compatible endpoints (Ollama especially) only
        # reliably understand the classic 'max_tokens'. Key the field name
        # on which side we're talking to.
        ("max_completion_tokens" if base_url is None else "max_tokens"): max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "emit_json",
                "strict": True,
                "schema": _normalize_schema_for_openai(json_schema),
            },
        }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    data = json.loads(resp.read())
    u = data.get("usage") or {}
    usage = {
        "promptTokenCount": u.get("prompt_tokens", 0),
        "candidatesTokenCount": u.get("completion_tokens", 0),
        "thoughtsTokenCount": 0,
    }
    choices = data.get("choices") or []
    if not choices:
        print("[LLM] OpenAI: no choices in response")
        return None, usage
    choice = choices[0]
    finish = choice.get("finish_reason", "unknown")
    text = (choice.get("message") or {}).get("content") or ""
    if finish == "length":
        print(f"[LLM] OpenAI: truncated ({len(text)} chars), discarding")
        return None, usage
    if text.strip():
        return text, usage
    print(f"[LLM] OpenAI: no usable content (finish={finish})")
    return None, usage


def call_claude(prompt: str, system: str = "", model: str = "claude-haiku-4-5",
                max_tokens: int = 2048, json_mode: bool = False,
                json_schema: dict | None = None,
                temperature: float | None = None,
                timeout: float | None = None) -> str | None:
    """Call a specific Anthropic model. Returns text or None.

    The Anthropic twin of call_gemini: same cost cap, rate limiter, budget
    and cooldown discipline; single model, no fallback (caller decides).
    """
    global _last_model_used
    if not _429_cooled_down('anthropic'):
        return None
    if not _cost_ok():
        print(f"[LLM] Daily cost limit reached (${_daily_cost:.2f}/${_DAILY_COST_LIMIT:.2f})")
        return None
    config = load_llm_config()
    if not config.get("enabled"):
        return None
    api_key = _anthropic_key(config)
    if not api_key:
        return None
    if not _rate_limit_ok():
        print(f"[LLM] Rate limit reached, skipping {model}")
        return None
    remaining, total = get_budget(model)
    if remaining <= 0:
        print(f"[LLM] {model}: daily budget exhausted ({total} RPD)")
        return None

    prompt_chars = len(prompt) + len(system)
    if timeout is None:
        timeout = config.get("max_llm_latency_sec", 30)

    for attempt in (1, 2):
        start = time.time()
        try:
            result, usage = _call_anthropic(prompt, system, api_key, model,
                                            max_tokens, timeout,
                                            json_mode=json_mode,
                                            json_schema=json_schema,
                                            temperature=temperature)
            elapsed = (time.time() - start) * 1000
            if result:
                record_call(model)
                _record_cost(model, prompt_chars, len(result), usage)
                _last_model_used = model
                print(f"[LLM] {model}: {elapsed:.0f}ms, {len(result)} chars "
                      f"(${_daily_cost:.3f} today)")
            return result
        except urllib.error.HTTPError as e:
            elapsed = (time.time() - start) * 1000
            if e.code == 429 and attempt == 1:
                wait = _parse_retry_after_anthropic(e)
                if wait and wait <= _429_MAX_WAIT_PRIMARY:
                    print(f"[LLM] {model}: 429, waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue
            print(f"[LLM] {model}: HTTP {e.code} ({elapsed:.0f}ms)")
            return None
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            print(f"[LLM] {model}: {e} ({elapsed:.0f}ms)")
            return None
    return None


def call_openai(prompt: str, system: str = "", model: str = "gpt-5.4-nano",
                max_tokens: int = 2048, json_mode: bool = False,
                json_schema: dict | None = None,
                temperature: float | None = None,
                timeout: float | None = None,
                base_url: str | None = None) -> str | None:
    """Call a specific OpenAI (or OpenAI-compatible) model. Returns text or
    None.

    The OpenAI twin of call_claude/call_gemini: same cost cap, rate
    limiter, budget, and cooldown discipline (per-provider cooldown key
    'openai'); single model, no fallback (caller decides). base_url
    defaults to the real OpenAI API; pass an override to hit an
    OpenAI-compatible third-party endpoint directly (normal usage for
    third-party endpoints goes through resolve_provider_chain + call_llm
    instead, which resolves base_url/api_key per-endpoint automatically).
    """
    global _last_model_used
    if not _429_cooled_down('openai'):
        return None
    if not _cost_ok():
        print(f"[LLM] Daily cost limit reached (${_daily_cost:.2f}/${_DAILY_COST_LIMIT:.2f})")
        return None
    config = load_llm_config()
    if not config.get("enabled"):
        return None
    api_key = _openai_key(config)
    if not api_key:
        return None
    if not _rate_limit_ok():
        print(f"[LLM] Rate limit reached, skipping {model}")
        return None
    remaining, total = get_budget(model)
    if remaining <= 0:
        print(f"[LLM] {model}: daily budget exhausted ({total} RPD)")
        return None

    prompt_chars = len(prompt) + len(system)
    if timeout is None:
        timeout = config.get("max_llm_latency_sec", 30)

    for attempt in (1, 2):
        start = time.time()
        try:
            result, usage = _call_openai(prompt, system, api_key, model,
                                         max_tokens, timeout,
                                         json_schema=json_schema,
                                         temperature=temperature,
                                         base_url=base_url)
            elapsed = (time.time() - start) * 1000
            if result:
                record_call(model)
                _record_cost(model, prompt_chars, len(result), usage)
                _last_model_used = model
                print(f"[LLM] {model}: {elapsed:.0f}ms, {len(result)} chars "
                      f"(${_daily_cost:.3f} today)")
            return result
        except urllib.error.HTTPError as e:
            elapsed = (time.time() - start) * 1000
            if e.code == 429 and attempt == 1:
                wait = _parse_retry_after_anthropic(e)  # generic retry-after header parse
                if wait and wait <= _429_MAX_WAIT_PRIMARY:
                    print(f"[LLM] {model}: 429, waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue
            print(f"[LLM] {model}: HTTP {e.code} ({elapsed:.0f}ms)")
            return None
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            print(f"[LLM] {model}: {e} ({elapsed:.0f}ms)")
            return None
    return None


def call_model(prompt: str, system: str = "", model: str = "gemini-2.5-flash-lite",
               max_tokens: int = 2048, json_mode: bool = False,
               json_schema: dict | None = None,
               temperature: float | None = None,
               timeout: float | None = None) -> str | None:
    """Provider-aware single-model call: 'claude-*' -> Anthropic,
    'gpt-*'/'o*' -> OpenAI, else Gemini.

    The analyst/tiered-sentiment call sites use this so a role override or a
    provider switch can point them at any provider without code changes.
    """
    provider = _provider_for(model)
    if provider == "anthropic":
        return call_claude(prompt, system=system, model=model,
                           max_tokens=max_tokens, json_mode=json_mode,
                           json_schema=json_schema, temperature=temperature,
                           timeout=timeout)
    if provider == "openai":
        return call_openai(prompt, system=system, model=model,
                           max_tokens=max_tokens, json_mode=json_mode,
                           json_schema=json_schema, temperature=temperature,
                           timeout=timeout)
    return call_gemini(prompt, system=system, model=model,
                       max_tokens=max_tokens, json_mode=json_mode,
                       json_schema=json_schema, temperature=temperature,
                       timeout=timeout)


def probe_available_models() -> dict:
    """Ask each configured provider (+ every enabled endpoint) what models
    the key can actually see.

    'Are we relying on old models?' becomes a runtime question: every
    native provider AND every OpenAI-compatible endpoint expose a
    model-list endpoint, so new releases show up here without a code
    change (route to them via the config model fields / role overrides,
    and price them via the config "pricing" table). Ops/GUI use only —
    never called in the trading hot path.
    """
    out: dict[str, list] = {"gemini": [], "anthropic": [], "openai": []}
    config = load_llm_config()
    gkey = config.get("models", {}).get("gemini", {}).get("api_key", "")
    if gkey:
        try:
            req = urllib.request.Request(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": gkey})
            data = json.loads(urllib.request.urlopen(req, timeout=10).read())
            out["gemini"] = sorted(
                m["name"].removeprefix("models/")
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", []))
        except Exception as e:
            out["gemini"] = [f"probe failed: {e}"]
    akey = _anthropic_key(config)
    if akey:
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": akey,
                         "anthropic-version": _ANTHROPIC_VERSION})
            data = json.loads(urllib.request.urlopen(req, timeout=10).read())
            out["anthropic"] = sorted(m.get("id", "")
                                      for m in data.get("data", []))
        except Exception as e:
            out["anthropic"] = [f"probe failed: {e}"]
    okey = _openai_key(config)
    if okey:
        try:
            req = urllib.request.Request(
                f"{_DEFAULT_OPENAI_BASE_URL}/models",
                headers={"Authorization": f"Bearer {okey}"})
            data = json.loads(urllib.request.urlopen(req, timeout=10).read())
            out["openai"] = sorted(m.get("id", "")
                                   for m in data.get("data", []))
        except Exception as e:
            out["openai"] = [f"probe failed: {e}"]
    for ep in _enabled_endpoints(config):
        name = ep.get("name") or "endpoint"
        base = (ep.get("base_url") or "").rstrip("/")
        if not base:
            out[name] = ["probe failed: no base_url configured"]
            continue
        key = _endpoint_api_key(ep)
        headers = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            req = urllib.request.Request(f"{base}/models", headers=headers)
            data = json.loads(urllib.request.urlopen(req, timeout=10).read())
            out[name] = sorted(m.get("id", "") for m in data.get("data", []))
        except Exception as e:
            out[name] = [f"probe failed: {e}"]
    return out


# --- Gemini API call ---

def _call_gemini(prompt, system, api_key, model, max_tokens, timeout,
                 json_mode=False, json_schema=None, temperature=None):
    """Call Google Gemini API. Returns (text|None, usage_dict|None);
    raises urllib errors for the caller's retry/fallback logic.

    json_schema: a JSON-schema dict — sets responseMimeType + responseSchema
    so the API GUARANTEES parseable output. This works on ALL current
    Gemini models including 2.5 Pro; the old `"pro" not in model` guard was
    based on a false premise and left the JSON-critical analyst call
    unenforced, which is where the repo's whole repair-parser saga came from.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    contents = [{"role": "user", "parts": [{"text": prompt}]}]

    gen_config = {"maxOutputTokens": max_tokens}
    if temperature is not None:
        # Gemini defaults to 1.0 — far too hot for a sizing gate that
        # should give the same answer to the same inputs
        gen_config["temperature"] = temperature
    if json_schema is not None:
        gen_config["responseMimeType"] = "application/json"
        gen_config["responseSchema"] = json_schema
    elif json_mode:
        gen_config["responseMimeType"] = "application/json"

    body = {
        "contents": contents,
        "generationConfig": gen_config,
    }
    if system:
        # Proper system prompt field (the old fake user/'Understood.' turns
        # weaken instruction adherence and break implicit caching)
        body["systemInstruction"] = {"parts": [{"text": system}]}

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    _capture_rate_limit_headers(resp, model)
    data = json.loads(resp.read())
    usage = data.get("usageMetadata")
    try:
        finish = data["candidates"][0].get("finishReason", "unknown")
        parts = data["candidates"][0]["content"]["parts"]
        # Thinking models: last part with "text" key is the actual output
        # Earlier parts may be thinking/reasoning
        for part in reversed(parts):
            if "text" in part and part["text"].strip():
                if finish == "MAX_TOKENS":
                    print(f"[LLM] Gemini: truncated ({len(part['text'])} chars), discarding")
                    return None, usage  # caller treats as retryable
                if finish != "STOP":
                    print(f"[LLM] Gemini: finish={finish} ({len(part['text'])} chars)")
                return part["text"], usage
        # No text found in any part
        finish = data["candidates"][0].get("finishReason", "unknown")
        print(f"[LLM] Gemini: no text in {len(parts)} parts (finish={finish})")
        return None, usage
    except (KeyError, IndexError):
        # Debug: log what we got so we can fix parsing
        finish = data.get("candidates", [{}])[0].get("finishReason", "unknown") if data.get("candidates") else "no_candidates"
        blocked = data.get("promptFeedback", {}).get("blockReason", "")
        detail = f"finish={finish}"
        if blocked:
            detail += f", blocked={blocked}"
        print(f"[LLM] Gemini: unexpected response ({detail})")
        return None, usage
