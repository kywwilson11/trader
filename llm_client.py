"""Unified LLM REST client — Gemini, Claude, OpenAI via urllib (no SDK deps).

Reads provider + API key from llm_config.json. Returns raw text or None on failure.
Never blocks trades: all errors result in None return.

Fallback chain (Gemini):
  1. Try the configured model (e.g. gemini-2.5-pro)
  2. On 429, parse server "retry in Xs" and wait (up to 45s), then retry once
  3. If still failing, try each remaining Gemini model in priority order
     (pro → flash → flash-lite), each with its own per-model quota
  4. If all Gemini models fail, return None (keyword fallback)

Rate limiting:
  Client-side sliding window (10 RPM) prevents hitting provider limits.
"""

import collections
import json
import re
import time
import urllib.request
import urllib.error

from llm_config import load_llm_config

# Gemini fallback chain — ordered by daily quota (most generous first)
# Free tier RPD: flash-lite=1000, flash=250, pro=100
# Primary model is tried first (user's quality preference), then this chain
_GEMINI_FALLBACK_CHAIN = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# HTTP codes that trigger immediate fallback:
_FALLBACK_CODES = {402, 403, 500, 502, 503, 504}

# 429 retry: parse "retry in Xs" from error body for smart wait
_429_MAX_WAIT_PRIMARY = 45   # wait up to 45s for the primary model
_429_MAX_WAIT_FALLBACK = 45  # wait up to 45s for each fallback model

# --- Sliding-window rate limiter (10 RPM) ---
_RATE_LIMIT_RPM = 10
_call_timestamps: collections.deque = collections.deque()


def _parse_retry_after(http_error) -> float | None:
    """Extract retry delay from a 429 error response body.

    Gemini includes 'Please retry in 38.05s' in the JSON error message.
    Returns seconds to wait, or None if unparseable / daily limit exhausted.
    """
    try:
        body = http_error.read().decode("utf-8", errors="replace")
        # Daily token quota exhausted (limit: 0) — no point retrying
        if "limit: 0" in body:
            return None
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", body, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def _rate_limit_ok() -> bool:
    """Check if we're within the rate limit. Returns True if call is allowed."""
    now = time.time()
    cutoff = now - 60.0
    while _call_timestamps and _call_timestamps[0] < cutoff:
        _call_timestamps.popleft()
    if len(_call_timestamps) >= _RATE_LIMIT_RPM:
        return False
    _call_timestamps.append(now)
    return True


def call_llm(prompt: str, system: str = "", max_tokens: int = 2048) -> str | None:
    """Send prompt to the configured LLM provider. Returns text or None.

    On failure, automatically falls back through all available Gemini models.
    """
    config = load_llm_config()

    if not config.get("enabled"):
        return None

    if not _rate_limit_ok():
        print("[LLM] Rate limit reached (10 RPM), skipping")
        return None

    provider = config.get("provider", "gemini")
    pconfig = config.get("models", {}).get(provider, {})
    api_key = pconfig.get("api_key", "")
    model = pconfig.get("model", "")
    timeout = config.get("max_llm_latency_sec", 15)

    if not api_key:
        # No key for primary — try Gemini fallback chain directly
        return _try_gemini_chain(config, prompt, system, max_tokens, timeout)

    start = time.time()
    try:
        result = _dispatch(provider, prompt, system, api_key, model, max_tokens, timeout)
        elapsed = (time.time() - start) * 1000
        if result:
            print(f"[LLM] {provider}/{model}: {elapsed:.0f}ms, {len(result)} chars")
        return result

    except urllib.error.HTTPError as e:
        elapsed = (time.time() - start) * 1000

        if e.code == 429:
            # Parse "retry in Xs" from error body for smart wait
            wait = _parse_retry_after(e)
            if wait and wait <= _429_MAX_WAIT_PRIMARY:
                print(f"[LLM] {provider}/{model}: HTTP 429 ({elapsed:.0f}ms), "
                      f"waiting {wait:.0f}s")
                time.sleep(wait)
                try:
                    start2 = time.time()
                    result = _dispatch(provider, prompt, system, api_key, model, max_tokens, timeout)
                    elapsed2 = (time.time() - start2) * 1000
                    if result:
                        print(f"[LLM] {provider}/{model}: {elapsed2:.0f}ms, {len(result)} chars (after wait)")
                    return result
                except Exception:
                    pass  # Retry failed, fall through to chain

            # Try other Gemini models (each has its own per-model quota)
            print(f"[LLM] {provider}/{model}: 429, trying other models")
            return _try_gemini_chain(config, prompt, system, max_tokens, timeout,
                                     skip_model=model if provider == "gemini" else None)

        if e.code in _FALLBACK_CODES:
            print(f"[LLM] {provider}/{model}: HTTP {e.code} ({elapsed:.0f}ms), trying other models")
            return _try_gemini_chain(config, prompt, system, max_tokens, timeout,
                                     skip_model=model if provider == "gemini" else None)
        print(f"[LLM] ERROR ({provider}/{model}): HTTP {e.code} ({elapsed:.0f}ms)")
        return None

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        print(f"[LLM] ERROR ({provider}): {e} ({elapsed:.0f}ms)")
        return None


def _try_gemini_chain(config, prompt, system, max_tokens, timeout, skip_model=None):
    """Try each Gemini model in priority order. Returns result or None.

    Each model has its own per-model quota on the free tier, so if one is
    rate-limited, the next may still have capacity.
    """
    gemini_key = config.get("models", {}).get("gemini", {}).get("api_key", "")
    if not gemini_key:
        return None

    for model in _GEMINI_FALLBACK_CHAIN:
        if model == skip_model:
            continue

        start = time.time()
        try:
            result = _call_gemini(prompt, system, gemini_key, model, max_tokens, timeout)
            elapsed = (time.time() - start) * 1000
            if result:
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
                        result = _call_gemini(prompt, system, gemini_key, model, max_tokens, timeout)
                        elapsed2 = (time.time() - start2) * 1000
                        if result:
                            print(f"[LLM] gemini/{model}: {elapsed2:.0f}ms, {len(result)} chars (after wait)")
                        return result
                    except Exception:
                        pass  # This model exhausted, try next
                print(f"[LLM] gemini/{model}: 429 ({elapsed:.0f}ms), trying next")
                continue
            print(f"[LLM] gemini/{model}: HTTP {e.code} ({elapsed:.0f}ms), trying next")
            continue

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            print(f"[LLM] gemini/{model}: {e} ({elapsed:.0f}ms), trying next")
            continue

    print("[LLM] All Gemini models exhausted")
    return None


def _dispatch(provider, prompt, system, api_key, model, max_tokens, timeout):
    """Route to the correct provider."""
    if provider == "gemini":
        return _call_gemini(prompt, system, api_key, model, max_tokens, timeout)
    elif provider == "claude":
        return _call_claude(prompt, system, api_key, model, max_tokens, timeout)
    elif provider == "openai":
        return _call_openai(prompt, system, api_key, model, max_tokens, timeout)
    else:
        print(f"[LLM] Unknown provider: {provider}")
        return None


def _call_gemini(prompt, system, api_key, model, max_tokens, timeout):
    """Call Google Gemini API."""
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


def _call_claude(prompt, system, api_key, model, max_tokens, timeout):
    """Call Anthropic Claude API."""
    url = "https://api.anthropic.com/v1/messages"

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    data = json.loads(resp.read())
    try:
        return data["content"][0]["text"]
    except (KeyError, IndexError):
        print(f"[LLM] Claude: unexpected response shape")
        return None


def _call_openai(prompt, system, api_key, model, max_tokens, timeout):
    """Call OpenAI API."""
    url = "https://api.openai.com/v1/chat/completions"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    data = json.loads(resp.read())
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print(f"[LLM] OpenAI: unexpected response shape")
        return None
