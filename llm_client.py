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

from llm_config import load_llm_config

# Gemini models ordered by daily quota (most generous first for fallback)
# Free tier RPD: flash-lite≈1000, flash≈250, pro≈100
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

_GEMINI_FALLBACK_CHAIN = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# HTTP codes that trigger immediate fallback:
_FALLBACK_CODES = {402, 403, 500, 502, 503, 504}

# 429 retry: parse "retry in Xs" from error body for smart wait
_429_MAX_WAIT_PRIMARY = 45
_429_MAX_WAIT_FALLBACK = 45

# --- Sliding-window rate limiter (10 RPM) ---
_RATE_LIMIT_RPM = 10
_call_timestamps: collections.deque = collections.deque()

# --- Daily quota tracking (resets at midnight Pacific) ---
_DAILY_BUDGETS = {
    "gemini-2.5-pro": 80,         # 100 RPD, leave 20 margin
    "gemini-2.5-flash": 200,      # 250 RPD, leave 50 margin
    "gemini-2.5-flash-lite": 800,  # 1000 RPD, leave 200 margin
}
_model_calls: dict[str, int] = {}
_quota_reset_date: str = ""


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


def _maybe_reset_quota():
    """Reset daily quota counters at midnight Pacific."""
    global _quota_reset_date
    pt = timezone(timedelta(hours=-8))
    today = datetime.now(pt).strftime("%Y-%m-%d")
    if _quota_reset_date != today:
        _model_calls.clear()
        _quota_reset_date = today


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
                max_tokens: int = 2048) -> str | None:
    """Call a specific Gemini model. Returns text or None.

    Used by tiered scoring to target a specific model. Handles 429 with
    retry-after parsing. Does NOT fall back to other models (caller decides).
    """
    config = load_llm_config()
    if not config.get("enabled"):
        return None

    gemini_key = config.get("models", {}).get("gemini", {}).get("api_key", "")
    if not gemini_key:
        return None

    if not _rate_limit_ok():
        print(f"[LLM] Rate limit reached (10 RPM), skipping {model}")
        return None

    remaining, total = get_budget(model)
    if remaining <= 0:
        print(f"[LLM] {model}: daily budget exhausted ({total} RPD)")
        return None

    timeout = config.get("max_llm_latency_sec", 30)
    start = time.time()

    try:
        result = _call_gemini(prompt, system, gemini_key, model, max_tokens, timeout)
        elapsed = (time.time() - start) * 1000
        if result:
            record_call(model)
            print(f"[LLM] {model}: {elapsed:.0f}ms, {len(result)} chars")
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
    config = load_llm_config()
    if not config.get("enabled"):
        return None

    if not _rate_limit_ok():
        print("[LLM] Rate limit reached (10 RPM), skipping")
        return None

    gemini_key = config.get("models", {}).get("gemini", {}).get("api_key", "")
    if not gemini_key:
        return None

    model = config.get("models", {}).get("gemini", {}).get("model", "gemini-2.5-flash")
    timeout = config.get("max_llm_latency_sec", 30)

    # Try configured model first
    start = time.time()
    try:
        result = _call_gemini(prompt, system, gemini_key, model, max_tokens, timeout)
        elapsed = (time.time() - start) * 1000
        if result:
            record_call(model)
            print(f"[LLM] {model}: {elapsed:.0f}ms, {len(result)} chars")
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
                        print(f"[LLM] {model}: {elapsed2:.0f}ms, {len(result)} chars (after wait)")
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
    return None


# --- Gemini API call ---

def _call_gemini(prompt, system, api_key, model, max_tokens, timeout):
    """Call Google Gemini API. Returns text or raises on error."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    contents = []
    if system:
        contents.append({"role": "user", "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    body = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens},
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
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        print(f"[LLM] Gemini: unexpected response shape")
        return None
