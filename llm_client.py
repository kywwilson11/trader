"""Unified LLM client — Gemini API via urllib (no SDK deps).

Reads provider + API key from llm_config.json. Returns raw text or None on failure.
Never blocks trades: all errors result in None return.

Supports two calling modes:
  1. call_llm() — uses config model + automatic fallback chain
  2. call_gemini() — calls a specific Gemini model (for tiered scoring)

Daily quota tracking:
  Tracks calls per model since midnight Pacific. Budget limits prevent
  burning through free-tier RPD (requests per day) limits.

Rate limiting:
  Client-side sliding window (10 RPM) prevents hitting per-minute limits.
"""

import collections
import json
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from llm_config import load_llm_config

# Gemini models ordered by daily quota (most generous first for fallback)
# Free tier RPD: flash-lite≈1000, flash≈250, pro≈100
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

_GEMINI_FALLBACK_CHAIN = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",  # Paid tier: Pro has generous limits, fast enough for fallback
]

# HTTP codes that trigger immediate fallback:
_FALLBACK_CODES = {402, 403, 500, 502, 503, 504}

# 429 retry: parse "retry in Xs" from error body for smart wait
_429_MAX_WAIT_PRIMARY = 10   # paid tier: brief wait usually clears it
_429_MAX_WAIT_FALLBACK = 5   # fallback models: don't wait long

# 429 circuit breaker: when all models fail with 429, skip LLM for a cooldown
_429_COOLDOWN_SEC = 30   # short cooldown — paid tier rarely sustains 429s
_429_cooldown_until: float = 0.0  # timestamp when cooldown expires

# --- Sliding-window rate limiter (10 RPM) ---
_RATE_LIMIT_RPM = 30  # paid tier allows 60+ RPM for pro, 360 for flash
_call_timestamps: collections.deque = collections.deque()

# --- Daily quota tracking (resets at midnight Pacific) ---
_DAILY_BUDGETS = {
    "gemini-2.5-pro": 1000,       # paid tier: 1000 RPD
    "gemini-2.5-flash": 2000,     # paid tier: generous
    "gemini-2.5-flash-lite": 5000, # paid tier: very generous
}
_model_calls: dict[str, int] = {}
_quota_reset_date: str = ""

# --- Daily cost tracking (hard cap to prevent runaway spending) ---
_DAILY_COST_LIMIT = 1.00  # ~$30/month (paid tier 1)
_daily_cost: float = 0.0
_cost_reset_date: str = ""

# Per-million-token pricing (input, output) — conservative estimates
_PRICING = {
    "gemini-2.5-pro":        (1.25, 10.0),
    "gemini-2.5-flash":      (0.15, 0.60),
    "gemini-2.5-flash-lite": (0.075, 0.30),
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
    return None


def _rate_limit_ok() -> bool:
    """Check if we're within the per-minute rate limit."""
    now = time.time()
    cutoff = now - 60.0
    while _call_timestamps and _call_timestamps[0] < cutoff:
        _call_timestamps.popleft()
    if len(_call_timestamps) >= _RATE_LIMIT_RPM:
        return False
    _call_timestamps.append(now)
    return True


def _429_cooled_down() -> bool:
    """Check if we're past the 429 cooldown period."""
    global _429_cooldown_until
    if time.time() < _429_cooldown_until:
        return False
    return True


def _trigger_429_cooldown():
    """All models 429'd — skip LLM calls for a cooldown period."""
    global _429_cooldown_until
    _429_cooldown_until = time.time() + _429_COOLDOWN_SEC
    print(f"[LLM] All models rate-limited, cooling down {_429_COOLDOWN_SEC}s")


def _maybe_reset_quota():
    """Reset daily quota and cost counters at midnight Pacific."""
    global _quota_reset_date, _cost_reset_date, _daily_cost
    today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    if _quota_reset_date != today:
        _model_calls.clear()
        _quota_reset_date = today
    if _cost_reset_date != today:
        if _daily_cost > 0:
            print(f"[LLM] Daily cost reset (yesterday: ${_daily_cost:.4f})")
        _daily_cost = 0.0
        _cost_reset_date = today


def _estimate_cost(model: str, prompt_chars: int, response_chars: int) -> float:
    """Estimate API cost from character counts (~4 chars per token)."""
    input_tokens = prompt_chars / 4
    output_tokens = response_chars / 4
    price_in, price_out = _PRICING.get(model, (1.25, 10.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def _record_cost(model: str, prompt_chars: int, response_chars: int):
    """Record estimated cost for this call."""
    global _daily_cost
    cost = _estimate_cost(model, prompt_chars, response_chars)
    _daily_cost += cost


def _cost_ok() -> bool:
    """Check if we're under the daily cost limit."""
    _maybe_reset_quota()
    if _daily_cost >= _DAILY_COST_LIMIT:
        return False
    return True


def get_daily_cost() -> tuple[float, float]:
    """Return (spent_today, daily_limit) for monitoring."""
    _maybe_reset_quota()
    return _daily_cost, _DAILY_COST_LIMIT


def get_budget(model: str) -> tuple[int, int]:
    """Return (remaining, total) daily budget for a model."""
    _maybe_reset_quota()
    total = _DAILY_BUDGETS.get(model, 50)
    used = _model_calls.get(model, 0)
    return max(0, total - used), total


def record_call(model: str):
    """Record that we made an API call to this model."""
    _maybe_reset_quota()
    _model_calls[model] = _model_calls.get(model, 0) + 1


# --- Public API ---

def call_gemini(prompt: str, system: str = "", model: str = "gemini-2.5-flash",
                max_tokens: int = 2048, json_mode: bool = False) -> str | None:
    """Call a specific Gemini model. Returns text or None.

    Used by tiered scoring to target a specific model. Handles 429 with
    retry-after parsing. Does NOT fall back to other models (caller decides).
    """
    if not _429_cooled_down():
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
    timeout = config.get("max_llm_latency_sec", 30)
    start = time.time()

    try:
        result = _call_gemini(prompt, system, gemini_key, model, max_tokens, timeout,
                              json_mode=json_mode)
        elapsed = (time.time() - start) * 1000
        if result:
            record_call(model)
            _record_cost(model, prompt_chars, len(result))
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
                    result = _call_gemini(prompt, system, gemini_key, model, max_tokens,
                                          timeout, json_mode=json_mode)
                    elapsed2 = (time.time() - start2) * 1000
                    if result:
                        record_call(model)
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


def call_llm(prompt: str, system: str = "", max_tokens: int = 2048) -> str | None:
    """Send prompt to any available Gemini model. Returns text or None.

    Tries the configured model first, then falls back through all models.
    Used by non-sentiment callers (fundamentals, llm_analyst) that don't
    need tiered scoring.
    """
    if not _429_cooled_down():
        return None

    if not _cost_ok():
        print(f"[LLM] Daily cost limit reached (${_daily_cost:.2f}/${_DAILY_COST_LIMIT:.2f})")
        return None

    config = load_llm_config()
    if not config.get("enabled"):
        return None

    if not _rate_limit_ok():
        print("[LLM] Rate limit reached, skipping")
        return None

    gemini_key = config.get("models", {}).get("gemini", {}).get("api_key", "")
    if not gemini_key:
        return None

    model = config.get("models", {}).get("gemini", {}).get("model", "gemini-2.5-flash")
    prompt_chars = len(prompt) + len(system)
    timeout = config.get("max_llm_latency_sec", 30)

    # Try configured model first
    start = time.time()
    try:
        result = _call_gemini(prompt, system, gemini_key, model, max_tokens, timeout)
        elapsed = (time.time() - start) * 1000
        if result:
            record_call(model)
            _record_cost(model, prompt_chars, len(result))
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
                    result = _call_gemini(prompt, system, gemini_key, model, max_tokens, timeout)
                    elapsed2 = (time.time() - start2) * 1000
                    if result:
                        record_call(model)
                        _record_cost(model, prompt_chars, len(result))
                        print(f"[LLM] {model}: {elapsed2:.0f}ms, {len(result)} chars (after wait, ${_daily_cost:.3f} today)")
                    return result
                except Exception:
                    pass
            # Fall through to chain
        elif e.code not in _FALLBACK_CODES:
            print(f"[LLM] {model}: HTTP {e.code} ({elapsed:.0f}ms)")
            return None
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        print(f"[LLM] {model}: {e} ({elapsed:.0f}ms)")
        return None

    # Fallback chain
    print(f"[LLM] {model}: failed, trying fallback chain")
    return _try_gemini_chain(gemini_key, prompt, system, max_tokens, timeout,
                              skip_model=model)


def _try_gemini_chain(api_key, prompt, system, max_tokens, timeout, skip_model=None):
    """Try each Gemini model in fallback order."""
    for model in _GEMINI_FALLBACK_CHAIN:
        if model == skip_model:
            continue

        start = time.time()
        try:
            result = _call_gemini(prompt, system, api_key, model, max_tokens, timeout)
            elapsed = (time.time() - start) * 1000
            if result:
                record_call(model)
                print(f"[LLM] gemini/{model}: {elapsed:.0f}ms, {len(result)} chars")
            return result

        except urllib.error.HTTPError as e:
            elapsed = (time.time() - start) * 1000
            if e.code == 429:
                wait = _parse_retry_after(e)
                if wait and wait <= _429_MAX_WAIT_FALLBACK:
                    print(f"[LLM] gemini/{model}: 429, waiting {wait:.0f}s")
                    time.sleep(wait)
                    try:
                        start2 = time.time()
                        result = _call_gemini(prompt, system, api_key, model, max_tokens, timeout)
                        elapsed2 = (time.time() - start2) * 1000
                        if result:
                            record_call(model)
                            print(f"[LLM] gemini/{model}: {elapsed2:.0f}ms, {len(result)} chars (after wait)")
                        return result
                    except Exception:
                        pass
                print(f"[LLM] gemini/{model}: 429, trying next")
                continue
            print(f"[LLM] gemini/{model}: HTTP {e.code}, trying next")
            continue

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            print(f"[LLM] gemini/{model}: {e}, trying next")
            continue

    print("[LLM] All Gemini models exhausted")
    _trigger_429_cooldown()
    return None


# --- Gemini API call ---

def _call_gemini(prompt, system, api_key, model, max_tokens, timeout,
                 json_mode=False):
    """Call Google Gemini API. Returns text or raises on error."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    contents = []
    if system:
        contents.append({"role": "user", "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    gen_config = {"maxOutputTokens": max_tokens}
    if json_mode and "pro" not in model:
        # Force structured JSON output (not supported by thinking models)
        gen_config["responseMimeType"] = "application/json"

    body = {
        "contents": contents,
        "generationConfig": gen_config,
    }

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
    data = json.loads(resp.read())
    try:
        finish = data["candidates"][0].get("finishReason", "unknown")
        parts = data["candidates"][0]["content"]["parts"]
        # Thinking models: last part with "text" key is the actual output
        # Earlier parts may be thinking/reasoning
        for part in reversed(parts):
            if "text" in part and part["text"].strip():
                if finish != "STOP":
                    print(f"[LLM] Gemini: finish={finish} ({len(part['text'])} chars)")
                return part["text"]
        # No text found in any part
        finish = data["candidates"][0].get("finishReason", "unknown")
        print(f"[LLM] Gemini: no text in {len(parts)} parts (finish={finish})")
        return None
    except (KeyError, IndexError):
        # Debug: log what we got so we can fix parsing
        finish = data.get("candidates", [{}])[0].get("finishReason", "unknown") if data.get("candidates") else "no_candidates"
        blocked = data.get("promptFeedback", {}).get("blockReason", "")
        detail = f"finish={finish}"
        if blocked:
            detail += f", blocked={blocked}"
        print(f"[LLM] Gemini: unexpected response ({detail})")
        return None
