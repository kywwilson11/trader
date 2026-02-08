"""Unified LLM REST client — Gemini, Claude, OpenAI via urllib (no SDK deps).

Reads provider + API key from llm_config.json. Returns raw text or None on failure.
Never blocks trades: all errors result in None return.
"""

import json
import time
import urllib.request
import urllib.error

from llm_config import load_llm_config


def call_llm(prompt: str, system: str = "", max_tokens: int = 2048) -> str | None:
    """Send prompt to the configured LLM provider. Returns text or None."""
    config = load_llm_config()

    if not config.get("enabled"):
        return None

    provider = config.get("provider", "gemini")
    pconfig = config.get("models", {}).get(provider, {})
    api_key = pconfig.get("api_key", "")
    model = pconfig.get("model", "")
    timeout = config.get("max_llm_latency_sec", 15)

    if not api_key:
        return None

    start = time.time()
    try:
        if provider == "gemini":
            result = _call_gemini(prompt, system, api_key, model, max_tokens, timeout)
        elif provider == "claude":
            result = _call_claude(prompt, system, api_key, model, max_tokens, timeout)
        elif provider == "openai":
            result = _call_openai(prompt, system, api_key, model, max_tokens, timeout)
        else:
            print(f"[LLM] Unknown provider: {provider}")
            return None

        elapsed = (time.time() - start) * 1000
        if result:
            print(f"[LLM] {provider}/{model}: {elapsed:.0f}ms, {len(result)} chars")
        return result

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        print(f"[LLM] ERROR ({provider}): {e} ({elapsed:.0f}ms)")
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
