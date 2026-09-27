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
  Selects the model per role (analyst, sentiment, backfill) from a daily-cost
  bracket table. NOTE: as shipped, every role in every bracket of both
  _PAID_ROUTING and _FREE_ROUTING maps to gemini-2.5-flash-lite, so the
  cost-bracket downgrade is a no-op placeholder today.

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


# Non-canonical spellings of a Claude id (review L2, 2026-09-26). The
# family/version regex, the price table and the RPD tables are keyed on the
# first-party alias (``claude-opus-4-6``); these all name the SAME model:
#   us.anthropic.claude-opus-4-6 / anthropic.claude-opus-4-6   (Bedrock:
#       region + vendor prefix; older ids also carry ``-v1:0``/``-v2:0``)
#   anthropic/claude-opus-4-6, openrouter/anthropic/...        (router paths)
#   claude-opus-4-5@20251101                                   (Vertex dated)
#   claude-haiku-4-5-20251001, claude-3-7-sonnet-latest        (API snapshots)
#   claude-opus-4.6                                            (dotted version)
#   claude-opus-4-6[1m]                                        (context tag)
# Pre-fix, ``us.anthropic.claude-opus-4-6`` classified as GEMINI and billed
# at $1.25/$10 against a true $5/$25, so the $/day cap tripped late.
# Classification/pricing ONLY — the raw id is still what goes on the wire.
_CLAUDE_BRACKET_TAG_RE = re.compile(r"\[[^\]]*\]$")
_CLAUDE_BEDROCK_VER_RE = re.compile(r"(?:-v\d+(?::\d+)?|:\d+)$")
_CLAUDE_VERTEX_AT_RE = re.compile(r"@[\w.-]*$")
_CLAUDE_LATEST_RE = re.compile(r"-latest$")
_CLAUDE_DATE_RE = re.compile(r"-\d{8}$")
_CLAUDE_DOTTED_VER_RE = re.compile(r"(?<=\d)\.(?=\d)")


def _canonical_claude_id(model) -> str | None:
    """Base first-party alias for any spelling of a Claude id, or None when
    the id is not a Claude id at all. Strips everything before ``claude``
    (region/vendor/router prefixes) and the ``[..]`` tag, Bedrock ``-vN:M``,
    Vertex ``@date``, ``-latest`` and ``-YYYYMMDD`` suffixes; dotted
    versions become dashed (``4.6`` -> ``4-6``). Lower-cased. Never raises."""
    try:
        m = str(model).strip().lower()
    except Exception:
        return None
    i = m.find("claude")
    if i < 0:
        return None
    m = m[i:]
    m = _CLAUDE_BRACKET_TAG_RE.sub("", m)
    m = _CLAUDE_BEDROCK_VER_RE.sub("", m)
    m = _CLAUDE_VERTEX_AT_RE.sub("", m)
    m = _CLAUDE_LATEST_RE.sub("", m)
    m = _CLAUDE_DATE_RE.sub("", m)
    m = _CLAUDE_DOTTED_VER_RE.sub("-", m)
    return m


def _provider_for(model: str) -> str:
    m = str(model)
    # 'claude' ANYWHERE (review L2): us.anthropic.claude-*, anthropic/claude-*
    # are Anthropic models, never the Gemini default.
    if _canonical_claude_id(m) is not None:
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
    # 2026-09 (FIX_G): rows for every id that has a _PRICING row, so a
    # configured/override id no longer silently falls to the 50-RPD
    # unknown default (audit D3). Anthropic/OpenAI rows mirror the paid
    # table (no free tier on either). Gemini 3.x flash-lite: Google
    # publishes per-model free-tier RPD ONLY inside AI Studio
    # (ai.google.dev/gemini-api/docs/rate-limits, fetched 2026-09-26), so
    # 1000 is an UNVERIFIED estimate copied from the 2.5-flash-lite row.
    "gemini-3.5-flash-lite": 1000,
    "gemini-3.1-flash-lite": 1000,
    "claude-haiku-4-5-20251001": 5000,
    "claude-sonnet-4-6": 2000,
    "claude-opus-4-6": 1000,
    "claude-opus-4-7": 1000,
    "claude-opus-4-8": 1000,
    "claude-opus-5": 1000,
    "claude-opus-5-5": 1000,
    "claude-fable-5": 500,
    "claude-fable-5-1": 500,
    "gpt-4.1": 1000,
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
    # 2026-09 (FIX_G): see the free-tier table's comment
    "gemini-3.5-flash-lite": 5000,
    "gemini-3.1-flash-lite": 5000,
    "claude-haiku-4-5-20251001": 5000,
    "claude-sonnet-4-6": 2000,
    "claude-opus-4-6": 1000,
    "claude-opus-4-7": 1000,
    "claude-opus-4-8": 1000,
    "claude-opus-5": 1000,
    "claude-opus-5-5": 1000,
    "claude-fable-5": 500,
    "claude-fable-5-1": 500,
    "gpt-4.1": 1000,
}

# Ids that fell through to the 50-RPD unknown-model default — one loud line
# per id per process (not one per call), mirroring _unknown_price_warned.
_unknown_budget_warned: set = set()

# Actual responder of the most recent successful call (for journaling —
# the analysis file used to claim 'pro' produced scores that flash wrote)
_last_model_used: str | None = None


def get_last_model_used() -> str | None:
    return _last_model_used


# --- Per-attempt transport metadata (INTEL W19, SCOUT_E A2) ---
# Measurement-only: nothing in this module (routing, retry, fallback, cost,
# budget, return values) reads it. `_last_model_used` above is a bare
# unlocked module global written by the calling thread; this slot keeps the
# same "the calling thread is the only writer, no lock" discipline but is
# THREAD-LOCAL, so combined-bot mode's two loop threads (the G4-09 situation
# at _rate_lock below) can never read each other's attempt.
#   .pending  finish/block/status noted by the raw provider parsers
#             (_call_gemini / _call_anthropic(_post) / _call_openai) during
#             the attempt in flight; reset by _meta_begin before each attempt.
#   .last     the completed-attempt dict get_last_call_meta() copies out.
# call_gemini / call_claude / call_openai (hence call_model) reset .last on
# entry, so a call that makes no HTTP attempt (cooldown, cost cap, disabled,
# no key, RPD) leaves None. call_llm deliberately does NOT reset: it is the
# fallback leg, and an attempt-less chain keeps the primary leg's record.
_call_meta_tls = threading.local()


def _meta_clear() -> None:
    try:
        _call_meta_tls.last = None
        _call_meta_tls.pending = None
    except Exception:
        pass


def _meta_begin() -> None:
    try:
        _call_meta_tls.pending = {}
    except Exception:
        pass


def _note_transport(finish_reason=None, block_reason=None, resp=None) -> None:
    """Called by the raw provider parsers; never raises, never alters their
    control flow. Only str reasons / int statuses are kept (JSON-safe)."""
    try:
        pend = getattr(_call_meta_tls, 'pending', None)
        if pend is None:
            pend = {}
            _call_meta_tls.pending = pend
        if resp is not None:
            st = getattr(resp, 'status', None)
            if isinstance(st, int) and not isinstance(st, bool):
                pend['http_status'] = st
        if isinstance(finish_reason, str):
            pend['finish_reason'] = finish_reason
        if isinstance(block_reason, str) and block_reason:
            pend['block_reason'] = block_reason
    except Exception:
        pass


def _meta_end(provider, model, start, attempt_index, fallback_used=False,
              http_status=None) -> None:
    """Record one completed attempt (success or failure). Never raises."""
    try:
        pend = getattr(_call_meta_tls, 'pending', None) or {}
        if not (isinstance(http_status, int)
                and not isinstance(http_status, bool)):
            http_status = pend.get('http_status')
        now = time.time()
        try:
            latency_ms = max(0, int((now - start) * 1000))
        except Exception:
            latency_ms = None
        _call_meta_tls.last = {
            'provider': provider,
            'model': model,
            'finish_reason': pend.get('finish_reason'),
            'block_reason': pend.get('block_reason'),
            'http_status': http_status,
            'latency_ms': latency_ms,
            'attempt_index': int(attempt_index),
            'fallback_used': bool(fallback_used),
            'ts': now,
        }
    except Exception:
        pass


def get_last_call_meta() -> dict | None:
    """READ-ONLY copy of this thread's most recent completed LLM HTTP
    attempt: {provider, model, finish_reason (provider-native string or
    None), block_reason (Gemini promptFeedback.blockReason or None),
    http_status (response status, HTTPError code, or None for a network
    error/timeout), latency_ms, attempt_index (0-based within the public
    call; 429 re-sends count), fallback_used (call_llm only: the attempt was
    on a chain entry after the first), ts (epoch s)}. None when this
    thread's latest call_gemini/call_claude/call_openai/call_model made no
    attempt. Measurement-only (INTEL W19): no gate reads it."""
    try:
        m = getattr(_call_meta_tls, 'last', None)
        return dict(m) if isinstance(m, dict) else None
    except Exception:
        return None

# --- Sliding-window rate limiter ---
_call_timestamps: collections.deque = collections.deque()
# Guards _call_timestamps (G4-09): combined-bot mode reaches _rate_limit_ok
# from two loop threads; the unsynchronised check-then-popleft could raise
# IndexError('pop from an empty deque') and the len/append pair over-admit.
_rate_lock = threading.Lock()

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
    # Gemini paid-tier (Standard) list prices, <=200k-token prompts
    # (ai.google.dev/gemini-api/docs/pricing, fetched 2026-09-26). The
    # 3.x flash-lite ids are stable and "Free of charge" on the free tier;
    # these are their PAID rates (a paid key is what the ledger meters).
    "gemini-2.5-pro":        (1.25, 10.0),
    "gemini-2.5-flash":      (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    # Anthropic first-party list prices per MTok (claude-api skill model
    # table, cached 2026-06-24). Prices move — llm_config.json may carry a
    # "pricing": {model: [in, out]} override that wins over this table
    # (see _pricing), so corrections never need a code change.
    "claude-haiku-4-5":          (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),   # dated snapshot of the above
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-sonnet-5":           (2.00, 10.00),  # was 3/15 (audit D1)
    "claude-opus-4-6":           (5.00, 25.00),
    "claude-opus-4-7":           (5.00, 25.00),
    "claude-opus-4-8":           (5.00, 25.00),
    "claude-opus-5":             (5.00, 25.00),
    "claude-opus-5-5":           (4.00, 20.00),
    "claude-fable-5":            (10.00, 50.00),
    "claude-fable-5-1":          (10.00, 50.00),
    # OpenAI Standard-tier list prices (developers.openai.com/api/docs/
    # pricing, fetched 2026-09-26) — replaces the unverified 5/15, 1/4,
    # 0.25/1 placeholders (audit D2).
    "gpt-5.4":               (2.50, 15.00),
    "gpt-5.4-mini":          (0.75, 4.50),
    "gpt-5.4-nano":          (0.20, 1.25),
    "gpt-4.1":               (2.00, 8.00),
}

# Models already warned about this process (one loud line per unknown model,
# not one per call) — billing an unknown model at the fallback price
# silently distorts the $1/day cap, so the owner must see which id fell
# through.
_unknown_price_warned: set = set()


def _fallback_price(model: str) -> tuple[float, float]:
    """CONSERVATIVE price for an id with no table row: the element-wise max
    (input, output) over every _PRICING row of the same provider family
    (_provider_for), else over the whole table.

    Pre-2026-09 every unknown id billed at a flat $1.25/$10 — cheaper than
    most of the Anthropic line, so an untabled claude-opus-4-6 ($5/$25)
    let the $1 cap admit ~$3 of real spend (audit D3). Billing at the
    family's highest KNOWN tier means the cap can only trip EARLY, never
    late, for any model priced at or below the most expensive tabled
    sibling (today: anthropic $10/$50, openai $2.50/$15, gemini $1.25/$10).
    A same-family model priced above every tabled sibling still needs a
    table row or a config['pricing'] entry.
    """
    fam = _provider_for(model)
    rows = [p for k, p in _PRICING.items() if _provider_for(k) == fam]
    if not rows:
        rows = list(_PRICING.values())
    return (max(r[0] for r in rows), max(r[1] for r in rows))


def _pricing(model: str) -> tuple[float, float]:
    """Per-MTok (input, output) price: config override > table >
    conservative family ceiling (_fallback_price)."""
    canon = _canonical_claude_id(model)
    keys = [model] + ([canon] if canon and canon != model else [])
    try:
        table = load_llm_config().get("pricing", {})
        for k in keys:
            override = table.get(k)
            if override and len(override) == 2:
                return (float(override[0]), float(override[1]))
    except Exception:
        pass
    for k in keys:
        # A region/vendor-prefixed or dated/-latest spelling prices as its
        # base model (review L2); an unknown Claude id still falls through
        # to the ANTHROPIC ceiling below, never the Gemini one.
        if k in _PRICING:
            return _PRICING[k]
    fb = _fallback_price(model)
    if model not in _unknown_price_warned:
        _unknown_price_warned.add(model)
        print(f"[LLM-COST] WARNING: no pricing entry for model '{model}' — "
              f"billing at conservative {_provider_for(model)} ceiling "
              f"(${fb[0]:.2f}/${fb[1]:.2f} per MTok). "
              f"Add a config['pricing'] entry in llm_config.json.")
    return fb


_CACHE_MULT_DEFAULTS = {"anthropic": (1.25, 0.10), "gemini": (1.00, 0.25)}

# Anthropic bills 1-hour-TTL cache WRITES at 2x base input (5-minute writes
# at 1.25x) — claude-api skill, shared/prompt-caching.md "Economics"
# (audit D14: the old code billed both TTLs at 1.25x). Kept separate from
# the (write, read) pair so _cache_multipliers' public 2-tuple is unchanged;
# a config override may carry it as an optional THIRD element:
# pricing_cache_multipliers = {"anthropic": [w5m, read, w1h]}.
_CACHE_WRITE_1H_MULT_DEFAULTS = {"anthropic": 2.0}


def _cache_multipliers(provider: str) -> tuple[float, float]:
    """(write_mult, read_mult) vs input price for cache-billed tokens:
    config['pricing_cache_multipliers'] override > built-in defaults.
    Unknown provider -> (1.0, 0.0) (cache tokens priced as no-ops).
    write_mult is the 5-minute-TTL write rate; see
    _cache_write_1h_multiplier for 1-hour writes."""
    try:
        m = load_llm_config().get("pricing_cache_multipliers", {}).get(provider)
        if m and len(m) >= 2:
            return (float(m[0]), float(m[1]))
    except Exception:
        pass
    return _CACHE_MULT_DEFAULTS.get(provider, (1.0, 0.0))


def _cache_write_1h_multiplier(provider: str) -> float:
    """1-hour-TTL cache-write multiplier vs input price: optional 3rd
    element of the config override > built-in default > the provider's
    5-minute write multiplier."""
    try:
        m = load_llm_config().get("pricing_cache_multipliers", {}).get(provider)
        if m and len(m) >= 3:
            return float(m[2])
    except Exception:
        pass
    if provider in _CACHE_WRITE_1H_MULT_DEFAULTS:
        return _CACHE_WRITE_1H_MULT_DEFAULTS[provider]
    return _cache_multipliers(provider)[0]

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
    # rpm read (may touch llm_config on disk) stays outside the lock; the
    # prune / check / append on the shared deque is one atomic step.
    rpm = _get_rate_limit_rpm()
    with _rate_lock:
        now = time.time()
        cutoff = now - 60.0
        while _call_timestamps and _call_timestamps[0] < cutoff:
            _call_timestamps.popleft()
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


# Daily cost history (INTEL W9, 2026-09-27; measurement-only). The rollover
# below overwrites llm_cost.json with the new day's $0, so before this the
# previous day's spend survived only as a stdout print. One JSON line per
# observed rollover is appended to llm_cost_history.jsonl NEXT TO _COST_FILE
# (derived at call time, so a test that repoints _COST_FILE sandboxes it too).
# Nothing reads it yet (future LLM-spend ledger reader); the ledger file,
# every return value and the rollover itself are unchanged by it.
_COST_HISTORY_MAX_LINE = 200  # bytes; one small O_APPEND write per line


def _cost_history_path() -> str:
    """<_COST_FILE minus .json>_history.jsonl, i.e. llm_cost_history.jsonl."""
    root, _ext = os.path.splitext(_COST_FILE)
    return root + "_history.jsonl"


def _snapshot_prev_ledger(today: str):
    """(date, cost) the shared file holds for a day other than `today` —
    the ledger the rollover is about to overwrite — or None. Never raises."""
    try:
        with open(_COST_FILE) as f:
            data = json.load(f)
        d = data.get("date")
        if isinstance(d, str) and d and d != today:
            return d, float(data.get("cost", 0.0))
    except Exception:
        pass
    return None


def _append_cost_history(prev_file, mem_date: str, mem_cost: float):
    """Append ONE line describing the ledger day that was just rolled over.
    CALLER HOLDS _cost_file_lock (via _rollover_cost_locked), so two
    processes rolling over at once cannot interleave; and since the first
    process to roll over rewrites the file to today under that lock, a
    later process's _load_shared_cost sees today and never reaches here —
    one line per rollover. A duplicate is possible only when the ledger
    write itself failed or the flock degraded; `pid` lets the reader dedupe
    by date. `cost` is the shared file's value for its date (src "file",
    the cross-process total) when readable, else the in-memory ledger
    (src "mem"); `mem_date`/`mem_cost` always carry the in-memory ledger
    (`_daily_cost` after _load_shared_cost — the value the reset print
    shows). No prior ledger day at all (fresh process, no file) -> no line.
    The single os-level write of <= _COST_HISTORY_MAX_LINE bytes goes to an
    'ab' unbuffered handle: POSIX write() with O_APPEND sets the offset to
    EOF with "no intervening file modification operation" between the seek
    and the write (POSIX.1-2017 write(2), XSH). No fsync. FAIL-SOFT: any
    exception is swallowed with one log line."""
    try:
        if prev_file is not None:
            date, cost, src = prev_file[0], prev_file[1], "file"
        elif mem_date:
            date, cost, src = mem_date, mem_cost, "mem"
        else:
            return
        rec = {"date": date, "cost": round(float(cost), 6), "src": src,
               "mem_date": mem_date or None,
               "mem_cost": round(float(mem_cost), 6),
               "reset_at": datetime.now(timezone.utc).isoformat(
                   timespec="seconds"),
               "pid": os.getpid()}
        line = (json.dumps(rec, allow_nan=False) + "\n").encode("utf-8")
        if len(line) > _COST_HISTORY_MAX_LINE:
            raise ValueError(f"history line {len(line)} B > "
                             f"{_COST_HISTORY_MAX_LINE} B")
        with open(_cost_history_path(), "ab", buffering=0) as f:
            f.write(line)
    except Exception as e:
        print(f"[LLM-COST] cost history append failed: {e}")


def _rollover_cost_locked(today: str):
    """Re-sync the in-memory ledger for `today` from the shared file and,
    if the file is still on an older date, start the new day at $0 and
    persist it. CALLER MUST HOLD _quota_lock AND _cost_file_lock — the
    rollover write is a read-modify-write of the shared file like any
    other (audit D10: it used to run outside the flock, so a process
    rolling over could overwrite another process's first spend of the new
    day). After a successful ledger write, the rolled-over day is appended
    to llm_cost_history.jsonl (_append_cost_history; fail-soft)."""
    global _cost_reset_date, _daily_cost
    # Load shared cost file first — another process may have spent today
    _load_shared_cost()
    if _cost_reset_date != today:
        # Still not today's date — fresh day
        prev_file = _snapshot_prev_ledger(today)
        mem_date, mem_cost = _cost_reset_date, _daily_cost
        if _daily_cost > 0:
            print(f"[LLM] Daily cost reset (yesterday: ${_daily_cost:.4f})")
        _daily_cost = 0.0
        _cost_reset_date = today
        _save_shared_cost()
        _append_cost_history(prev_file, mem_date, mem_cost)


def _maybe_reset_quota():
    """Reset daily quota and cost counters at midnight Pacific."""
    global _quota_reset_date
    with _quota_lock:
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
        if _quota_reset_date != today:
            _model_calls.clear()
            _quota_reset_date = today
        if _cost_reset_date != today:
            try:
                with _cost_file_lock():
                    _rollover_cost_locked(today)
            except Exception as e:
                print(f"[LLM-COST] daily rollover failed: {e}")


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
        cost = _cost_of(model, prompt_chars, response_chars, usage)
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
                # cannot be lost in this read-modify-write. A call that
                # straddled midnight PT must not add today's cost onto
                # yesterday's in-memory total (and then stamp that sum with
                # today's date) — roll over first, under the same lock.
                today = datetime.now(
                    ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
                _rollover_cost_locked(today)
                _daily_cost += cost
                _save_shared_cost()
    except Exception as e:
        print(f"[LLM-COST] ledger write failed: {e} "
              f"(${cost:.6f} may be unrecorded)")


def _cost_of(model: str, prompt_chars: int, response_chars: int,
             usage: dict | None = None) -> float:
    """$ cost of one (possibly summed) billed usage dict — the pure half of
    _record_cost (split out 2026-09-26 so the Anthropic validation-retry
    pre-flight can price the still-unrecorded first request, review L1).
    MAY raise; _record_cost owns the never-raise fallbacks."""
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
            # 1-hour-TTL share of the writes (D14): billed at 2x, not
            # the 5-minute 1.25x. Absent -> all writes are 5-minute,
            # i.e. exactly the pre-change arithmetic.
            cw1h = min(usage.get('cacheWrite1hTokenCount', 0) or 0, cw)
            if cw or cr:
                wm, rm = _cache_multipliers("anthropic")
                wm1h = (_cache_write_1h_multiplier("anthropic")
                        if cw1h else wm)
                cost += (((cw - cw1h) * wm + cw1h * wm1h + cr * rm)
                         * price_in / 1_000_000)
        elif provider == "gemini":
            cached = usage.get('cachedContentTokenCount', 0) or 0
            if cached:
                _wm, rm = _cache_multipliers("gemini")
                cost -= cached * (1.0 - rm) * price_in / 1_000_000
    else:
        cost = _estimate_cost(model, prompt_chars, response_chars)
    return cost


def _usage_billed(usage) -> bool:
    """True when a provider usage dict reports any billed token."""
    if not usage:
        return False
    for k in ('promptTokenCount', 'candidatesTokenCount',
              'thoughtsTokenCount', 'cacheWriteTokenCount',
              'cacheReadTokenCount'):
        try:
            if (usage.get(k) or 0) > 0:
                return True
        except (TypeError, AttributeError):
            continue
    return False


def _charge_discarded(model: str, prompt_chars: int, result, usage) -> None:
    """Audit D6: a response the transport DISCARDED (MAX_TOKENS/length
    truncation, safety block, empty or schema-invalid output) was still
    billed by the provider when it carries usage. Charge it to the shared
    ledger so the $1/day cap measures real money. No-op when a result was
    returned (the caller's success branch records that cost) or when no
    billed tokens were reported. Accounting only: the caller's control
    flow and return value are untouched. Never raises (_record_cost
    doesn't)."""
    if result or not _usage_billed(usage):
        return
    # Review L1 (2026-09-26): a billed request is a REQUEST against the
    # model's RPD budget whether or not its answer was usable — count it
    # (success paths count via their own record_call).
    try:
        record_call(model)
    except Exception:
        pass
    _record_cost(model, prompt_chars, 0, usage)
    print(f"[LLM-COST] {model}: billed response discarded — charged to "
          f"ledger (${_daily_cost:.4f} today)")


def _retry_preflight(model: str, pending_usage: dict | None) -> bool:
    """Review L1 (2026-09-26): an in-transport RE-request (the Anthropic
    auto-path validation retry) passes the SAME gates the first request
    passed in call_claude/call_llm — provider 429 cooldown, the $/day cost
    cap, the per-model RPD budget and the per-minute rate limiter — and it
    accounts for the first request, which was billed but is not yet on the
    ledger or the RPD counter (the caller records both after the transport
    returns): the cap check adds that request's cost, the RPD check needs
    room for BOTH requests. The rate limiter goes last because a passing
    _rate_limit_ok() consumes a slot. False -> the caller skips the retry
    and the transport returns None (fail-open, unchanged). Never raises."""
    try:
        if not _429_cooled_down(_provider_for(model)):
            reason = "provider 429 cooldown"
        elif not _cost_ok():
            reason = "daily cost cap"
        else:
            try:
                pending = _cost_of(model, 0, 0, pending_usage) \
                    if pending_usage else 0.0
            except Exception:
                pending = 0.0
            if _daily_cost + pending >= _DAILY_COST_LIMIT:
                reason = "daily cost cap (incl. the first request)"
            elif get_budget(model)[0] <= 1:
                reason = "RPD budget"
            elif not _rate_limit_ok():
                reason = "rate limit"
            else:
                return True
    except Exception as e:
        reason = f"pre-flight error: {e}"
    print(f"[LLM] {model}: validation retry refused ({reason})")
    return False


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
    canon = _canonical_claude_id(model)
    if model not in budgets and canon in budgets:
        # Prefixed/dated spelling of a tabled Claude id (review L2): its
        # base model's RPD row, not the 50-RPD unknown default. The call
        # COUNTER stays keyed on the raw id the caller uses.
        return max(0, budgets[canon] - _model_calls.get(model, 0)), budgets[canon]
    if model not in budgets and model not in _unknown_budget_warned:
        _unknown_budget_warned.add(model)
        print(f"[LLM] WARNING: no RPD budget row for model '{model}' — "
              f"using the conservative 50-RPD unknown-model default "
              f"(per process). Add it to llm_client's budget tables.")
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
    _meta_clear()   # INTEL W19: an attempt-less call leaves no stale meta
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
    _meta_begin()

    try:
        result, usage = _call_gemini(prompt, system, gemini_key, model, max_tokens,
                                     timeout, json_mode=json_mode,
                                     json_schema=json_schema,
                                     temperature=temperature)
        elapsed = (time.time() - start) * 1000
        _meta_end('gemini', model, start, 0)
        _charge_discarded(model, prompt_chars, result, usage)
        if result:
            record_call(model)
            _record_cost(model, prompt_chars, len(result), usage)
            _last_model_used = model
            print(f"[LLM] {model}: {elapsed:.0f}ms, {len(result)} chars (${_daily_cost:.3f} today)")
        return result

    except urllib.error.HTTPError as e:
        elapsed = (time.time() - start) * 1000
        _meta_end('gemini', model, start, 0, http_status=e.code)
        if e.code == 429:
            wait = _parse_retry_after(e)
            if wait and wait <= _429_MAX_WAIT_PRIMARY:
                print(f"[LLM] {model}: 429, waiting {wait:.0f}s")
                time.sleep(wait)
                try:
                    start2 = time.time()
                    _meta_begin()
                    result, usage = _call_gemini(prompt, system, gemini_key, model,
                                                 max_tokens, timeout,
                                                 json_mode=json_mode,
                                                 json_schema=json_schema,
                                                 temperature=temperature)
                    elapsed2 = (time.time() - start2) * 1000
                    _meta_end('gemini', model, start2, 1)
                    _charge_discarded(model, prompt_chars, result, usage)
                    if result:
                        record_call(model)
                        _record_cost(model, prompt_chars, len(result), usage)
                        _last_model_used = model
                        print(f"[LLM] {model}: {elapsed2:.0f}ms, {len(result)} chars (after wait)")
                    return result
                except Exception as _e2:
                    _meta_end('gemini', model, start2, 1,
                              http_status=getattr(_e2, 'code', None))
            print(f"[LLM] {model}: 429 exhausted ({elapsed:.0f}ms)")
        else:
            print(f"[LLM] {model}: HTTP {e.code} ({elapsed:.0f}ms)")
        return None

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        _meta_end('gemini', model, start, 0)
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

    _n_attempts = 0   # INTEL W19 attempt_index (0-based, 429 re-sends count)
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
        _meta_begin()
        _att = _n_attempts
        _n_attempts += 1
        try:
            result, usage = _dispatch(provider, model, base_url, api_key)
            elapsed = (time.time() - start) * 1000
            _meta_end(provider, model, start, _att, i > 0)
            _charge_discarded(model, prompt_chars, result, usage)
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
            _meta_end(provider, model, start, _att, i > 0, http_status=e.code)
            if e.code == 429:
                wait = (_parse_retry_after(e) if provider == "gemini"
                        else _parse_retry_after_anthropic(e))
                if wait and wait <= max_wait:
                    print(f"[LLM] {provider}/{model}: 429, waiting {wait:.0f}s")
                    time.sleep(wait)
                    start_r = time.time()
                    _meta_begin()
                    _att_r = _n_attempts
                    _n_attempts += 1
                    try:
                        result, usage = _dispatch(provider, model, base_url, api_key)
                        _meta_end(provider, model, start_r, _att_r, i > 0)
                        _charge_discarded(model, prompt_chars, result, usage)
                        if result:
                            record_call(model)
                            _record_cost(model, prompt_chars, len(result), usage)
                            _last_model_used = model
                            print(f"[LLM] {provider}/{model}: {len(result)} chars (after wait)")
                            return result
                    except Exception as _e2:
                        _meta_end(provider, model, start_r, _att_r, i > 0,
                                  http_status=getattr(_e2, 'code', None))
                print(f"[LLM] {provider}/{model}: 429, trying next")
                _trigger_429_cooldown(provider)
                continue
            print(f"[LLM] {provider}/{model}: HTTP {e.code}, trying next")
            continue
        except Exception as e:
            _meta_end(provider, model, start, _att, i > 0)
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


# --- Anthropic per-model request-surface gates (claude-api skill, 2026-09) ---
#
# Sampling params: `temperature`/`top_p`/`top_k` return a 400 on Claude
# Fable 5/5.1 (+ Mythos), Opus 5.5, Opus 5, Opus 4.8/4.7 and Sonnet 5; they
# are still accepted on Opus 4.6, Sonnet 4.6, Haiku 4.5 and older (skill
# "Thinking & Effort" table). Audit D4: the old transport always sent the
# analyst's temperature, so every Anthropic model newer than Opus 4.6 —
# including claude-sonnet-5 in _ANTHROPIC_FALLBACK_CHAIN — was a guaranteed
# 400. Removal thresholds are per family, (major, minor); fable/mythos never
# accepted it. Unparseable/unknown ids DROP temperature: omitting it can
# never cause a 400, sending it can.
_SAMPLING_REMOVED_FROM = {"opus": (4, 7), "sonnet": (5, 0), "haiku": (5, 0),
                          "fable": (0, 0), "mythos": (0, 0)}
# Forced tool use (`tool_choice` {"type":"tool"|"any"}) returns a 400 on
# Claude Fable 5.1 / Mythos 5.1 / Opus 5.5 (skill "Forced tool use
# removed"; Fable 5 / Opus 5 / Sonnet 5 / Haiku 4.5 still accept it). Audit
# D5. The sonnet/haiku thresholds are forward guesses (next major) — an id
# at or past a threshold, or an unparseable one, takes the `auto` path,
# which every model accepts.
_FORCED_TOOL_REMOVED_FROM = {"opus": (5, 5), "fable": (5, 1),
                             "mythos": (5, 1), "sonnet": (6, 0),
                             "haiku": (5, 0)}
_CLAUDE_ID_RE = re.compile(
    r"^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?"
    r"(?:-\d{8})?$")
# Appended as a trailing user-turn text block on the `auto` path only (the
# skill's recommended replacement for forcing: auto + an explicit
# instruction naming the tool + strict: true). The user turn, not the
# system prompt, so a system cache prefix is untouched.
_EMIT_JSON_INSTRUCTION = (
    "Respond by calling the emit_json tool exactly once. Its input is your "
    "complete answer; do not answer in plain text.")


def _claude_family_version(model):
    """('opus', (4, 6)) for 'claude-opus-4-6' — and for every non-canonical
    spelling of it (us.anthropic.…, @date, -latest, -vN:M; review L2);
    None if not a recognised Claude id (legacy 'claude-3-…' ids return
    None)."""
    m = _CLAUDE_ID_RE.match(_canonical_claude_id(model) or "")
    if not m:
        return None
    return m.group(1), (int(m.group(2)), int(m.group(3) or 0))


def _anthropic_accepts_sampling(model) -> bool:
    fv = _claude_family_version(model)
    if fv is None:
        # Legacy Claude 2/3 ids accept sampling params; anything else
        # unrecognised is treated as a newer model (drop — never a 400).
        return (_canonical_claude_id(model) or "").startswith(
            ("claude-3", "claude-2"))
    fam, ver = fv
    return ver < _SAMPLING_REMOVED_FROM[fam]


def _anthropic_accepts_forced_tool(model) -> bool:
    fv = _claude_family_version(model)
    if fv is None:
        return (_canonical_claude_id(model) or "").startswith(
            ("claude-3", "claude-2"))
    fam, ver = fv
    return ver < _FORCED_TOOL_REMOVED_FROM[fam]


def _json_type_ok(value, t: str) -> bool:
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    if t == "string":
        return isinstance(value, str)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer())
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "null":
        return value is None
    return True  # unknown type keyword: don't reject


def _json_matches_schema(value, schema) -> bool:
    """Minimal structural validator for the schema subset the callers use
    (type / properties / required / items / enum). Used ONLY on the
    `auto` tool-choice path, where — unlike forced tool use — the API does
    not guarantee a schema-shaped answer exists at all. Lenient on extra
    keys; strict on required keys and types."""
    if not isinstance(schema, dict):
        return True
    t = schema.get("type")
    types = [t] if isinstance(t, str) else (t if isinstance(t, list) else [])
    types = [x.lower() for x in types if isinstance(x, str)]
    if types and not any(_json_type_ok(value, x) for x in types):
        return False
    if "enum" in schema and isinstance(schema["enum"], list) \
            and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for k in schema.get("required") or []:
            if k not in value:
                return False
        for k, v in value.items():
            if k in props and not _json_matches_schema(v, props[k]):
                return False
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return all(_json_matches_schema(v, schema["items"]) for v in value)
    return True


def _sum_usage(a: dict | None, b: dict | None) -> dict | None:
    """Add two normalized usage dicts (both requests were billed)."""
    if not a:
        return b
    if not b:
        return a
    out = dict(a)
    for k, v in b.items():
        try:
            out[k] = (out.get(k) or 0) + (v or 0)
        except TypeError:
            pass
    return out


def _anthropic_post(body: dict, api_key: str, timeout) -> dict:
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    _note_transport(resp=resp)          # INTEL W19 (measurement-only)
    return json.loads(resp.read())


def _anthropic_usage(data: dict, model: str, cache_ttl: str) -> dict:
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
        # 1-hour-TTL share (billed 2x, D14): the API's per-TTL breakdown
        # when present; else infer from the TTL this request asked for.
        cc = u.get("cache_creation")
        if isinstance(cc, dict):
            cw1h = cc.get("ephemeral_1h_input_tokens") or 0
        else:
            cw1h = cw if cache_ttl == "1h" else 0
        if cw1h:
            usage["cacheWrite1hTokenCount"] = min(cw1h, cw)
    if cr:
        usage["cacheReadTokenCount"] = cr
    if cache_ttl in ("5m", "1h") and not cw and not cr:
        # Flag is ON but the API reported no cache tokens — the prefix is
        # below the model's minimum cacheable length or was invalidated
        # (per-symbol tool schema changed). Loud so the owner sees it.
        print(f"[LLM] Claude {model}: cache_control active but no cache "
              f"tokens reported (prefix below model minimum?)")
    return usage


def _call_anthropic(prompt, system, api_key, model, max_tokens, timeout,
                    json_mode=False, json_schema=None, temperature=None):
    """Call the Anthropic Messages API. Returns (text|None, usage|None);
    raises urllib errors for the caller's retry/fallback logic.

    json_schema: on models that accept it, enforced via FORCED TOOL USE —
    the schema becomes a tool's input_schema and tool_choice pins the model
    to that tool, so the returned tool_use input is schema-validated JSON
    (byte-identical request to the pre-2026-09 transport). On models that
    reject forced tool use (Fable 5.1 / Mythos 5.1 / Opus 5.5 — audit D5)
    the equivalent the claude-api skill recommends is used instead:
    tool_choice auto + `strict: true` on the tool + an explicit
    instruction, then the tool input (or, failing that, a JSON text
    answer) is validated against the schema client-side, with ONE
    validation retry — itself gated by _retry_preflight (cost cap incl.
    the first request, RPD, rate limit, 429 cooldown) and counted against
    RPD here (review L1); anything still not schema-shaped returns None
    (-> the analyst's fail-open path). Either way the caller gets the
    tool input re-serialized as a JSON string (so every provider parses
    identically) or None. Usage is normalized to Gemini's usageMetadata
    key names so _record_cost stays provider-agnostic, and on the retry
    path is the SUM of both billed requests. `temperature` is sent only to
    models that accept sampling params (audit D4). json_mode without a
    schema is best-effort (prompt discipline).
    """
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
    if temperature is not None and _anthropic_accepts_sampling(model):
        body["temperature"] = temperature
    forced = True
    if json_schema is not None:
        forced = _anthropic_accepts_forced_tool(model)
        if forced:
            body["tools"] = [{
                "name": "emit_json",
                "description": "Emit the structured answer in the required schema.",
                "input_schema": _normalize_schema_for_anthropic(json_schema),
            }]
            body["tool_choice"] = {"type": "tool", "name": "emit_json"}
        else:
            # Strict tool use requires additionalProperties:false on every
            # object — the same normalization OpenAI strict mode needs.
            body["tools"] = [{
                "name": "emit_json",
                "description": "Emit the structured answer in the required schema.",
                "input_schema": _normalize_schema_for_openai(json_schema),
                "strict": True,
            }]
            body["tool_choice"] = {"type": "auto"}
            body["messages"] = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": _EMIT_JSON_INSTRUCTION},
            ]}]

    data = _anthropic_post(body, api_key, timeout)
    _note_anthropic(data)               # INTEL W19 (measurement-only)
    usage = _anthropic_usage(data, model, cache_ttl)

    if json_schema is not None and not forced:
        check_schema = _normalize_schema_for_anthropic(json_schema)
        for attempt in (1, 2):
            text, stop = _anthropic_extract_validated(data, check_schema)
            if text is not None:
                return text, usage
            if attempt == 2 or stop in ("max_tokens", "refusal"):
                # A truncation/refusal would just repeat — don't pay twice.
                break
            if not _retry_preflight(model, usage):
                # Cap / RPD / rate limit / cooldown refuses a second billed
                # request (review L1): fail open with the first request's
                # usage so the caller still charges + counts it.
                return None, usage
            print(f"[LLM] Claude {model}: no schema-valid emit_json "
                  f"(stop={stop}), one validation retry")
            try:
                data = _anthropic_post(body, api_key, timeout)
            except Exception as e:
                # The first request was billed — return its usage so the
                # caller still charges it (D6), instead of losing it to a
                # raise.
                print(f"[LLM] Claude {model}: validation retry failed: {e}")
                return None, usage
            # The retry is its own billed request: count it against RPD
            # here (the caller's record_call / _charge_discarded counts the
            # first one) — review L1.
            record_call(model)
            _note_anthropic(data)       # INTEL W19: the retry's stop_reason
            usage = _sum_usage(usage, _anthropic_usage(data, model, cache_ttl))
        print(f"[LLM] Claude {model}: no schema-valid answer (stop={stop}), "
              f"discarding")
        return None, usage

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


def _note_anthropic(data) -> None:
    """INTEL W19: Claude's native stop_reason into the attempt meta."""
    try:
        _note_transport(finish_reason=data.get("stop_reason"))
    except Exception:
        pass


def _anthropic_extract_validated(data: dict, schema: dict):
    """`auto`-path extraction: the emit_json tool input if it validates,
    else a JSON text answer if THAT validates, else None.
    Returns (json_text|None, stop_reason)."""
    stop = data.get("stop_reason", "unknown")
    text_parts = []
    for block in data.get("content", []) or []:
        if block.get("type") == "tool_use" and block.get("name") == "emit_json":
            inp = block.get("input")
            if _json_matches_schema(inp, schema):
                return json.dumps(inp), stop
        elif block.get("type") == "text" and block.get("text", "").strip():
            text_parts.append(block["text"])
    if text_parts and stop != "max_tokens":
        raw = "".join(text_parts).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else ""
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3]
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None and _json_matches_schema(parsed, schema):
            return json.dumps(parsed), stop
    return None, stop


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
    _note_openai(resp, data)            # INTEL W19 (measurement-only)
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


def _note_openai(resp, data) -> None:
    """INTEL W19: HTTP status + choices[0].finish_reason (None when the
    response has no choices) into the attempt meta."""
    try:
        choices = data.get("choices") or []
        fr = choices[0].get("finish_reason") if choices else None
        _note_transport(finish_reason=fr, resp=resp)
    except Exception:
        _note_transport(resp=resp)


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
    _meta_clear()   # INTEL W19: an attempt-less call leaves no stale meta
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
        _meta_begin()
        try:
            result, usage = _call_anthropic(prompt, system, api_key, model,
                                            max_tokens, timeout,
                                            json_mode=json_mode,
                                            json_schema=json_schema,
                                            temperature=temperature)
            elapsed = (time.time() - start) * 1000
            _meta_end('anthropic', model, start, attempt - 1)
            _charge_discarded(model, prompt_chars, result, usage)
            if result:
                record_call(model)
                _record_cost(model, prompt_chars, len(result), usage)
                _last_model_used = model
                print(f"[LLM] {model}: {elapsed:.0f}ms, {len(result)} chars "
                      f"(${_daily_cost:.3f} today)")
            return result
        except urllib.error.HTTPError as e:
            elapsed = (time.time() - start) * 1000
            _meta_end('anthropic', model, start, attempt - 1, http_status=e.code)
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
            _meta_end('anthropic', model, start, attempt - 1)
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
    _meta_clear()   # INTEL W19: an attempt-less call leaves no stale meta
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
        _meta_begin()
        try:
            result, usage = _call_openai(prompt, system, api_key, model,
                                         max_tokens, timeout,
                                         json_schema=json_schema,
                                         temperature=temperature,
                                         base_url=base_url)
            elapsed = (time.time() - start) * 1000
            _meta_end('openai', model, start, attempt - 1)
            _charge_discarded(model, prompt_chars, result, usage)
            if result:
                record_call(model)
                _record_cost(model, prompt_chars, len(result), usage)
                _last_model_used = model
                print(f"[LLM] {model}: {elapsed:.0f}ms, {len(result)} chars "
                      f"(${_daily_cost:.3f} today)")
            return result
        except urllib.error.HTTPError as e:
            elapsed = (time.time() - start) * 1000
            _meta_end('openai', model, start, attempt - 1, http_status=e.code)
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
            _meta_end('openai', model, start, attempt - 1)
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
    and price them via the config "pricing" table). No in-repo caller —
    reachable only from a REPL / ops session; never called in the trading
    hot path.
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

def _note_gemini(resp, data) -> None:
    """INTEL W19: HTTP status, candidates[0].finishReason (None when there
    are no candidates) and promptFeedback.blockReason into the attempt
    meta. Reads only; the parse below is untouched."""
    fr = br = None
    try:
        cands = data.get("candidates")
        if isinstance(cands, list) and cands and isinstance(cands[0], dict):
            fr = cands[0].get("finishReason")
    except Exception:
        fr = None
    try:
        pf = data.get("promptFeedback")
        if isinstance(pf, dict):
            br = pf.get("blockReason")
    except Exception:
        br = None
    _note_transport(finish_reason=fr, block_reason=br, resp=resp)


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
    _note_gemini(resp, data)            # INTEL W19 (measurement-only)
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
