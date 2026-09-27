"""Global market context digest for LLM-enhanced trading.

Fetches broad market data (news, indices, sector ETFs, Fear & Greed, VIX)
and asks the LLM to produce a structured macro narrative. This digest is
injected into per-symbol LLM analysis so each stock/crypto evaluation
has awareness of global forces (geopolitics, monetary policy, sector rotation).

Runs once per hour, caches to global_context.json.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from log_config import get_logger

logger = get_logger(__name__)

_CONTEXT_FILE = Path(__file__).resolve().parent / "global_context.json"
_CACHE_TTL = 3600  # 1 hour

# In-memory cache: (timestamp, context_dict)
_mem_cache: tuple[float, dict] | None = None

# Key indices to snapshot for hard market data
_INDEX_TICKERS = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
    "^VIX": "VIX",
    "DX-Y.NYB": "US Dollar (DXY)",
    "CL=F": "Crude Oil",
    "GC=F": "Gold",
    "^TNX": "10Y Treasury Yield",
    "BTC-USD": "Bitcoin",
}

# Sector ETFs for rotation signal
_SECTOR_ETFS = {
    "XLK": "Technology",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}

_SYSTEM_PROMPT = """\
You are a macro strategist producing a global market context digest. \
Your job is to synthesize raw market data and news into a concise, \
actionable macro narrative that will inform per-symbol trade analysis.

Focus on:
1. REGIME: Is the market risk-on, risk-off, rotational, or in crisis? \
What is driving the current regime?
2. THEMES: What are the 3-5 dominant themes moving global markets right now? \
For each, explain the impact and which sectors are helped (+) or hurt (-).
3. RISK FACTORS: What are the top 3 tail risks to watch?
4. OPPORTUNITIES: Any contrarian signals or sector rotation plays?

Be specific — cite the exact numbers from the data provided (current prices, \
52-week ranges, period returns). The data below is real-time ground truth. \
Avoid vague hedging. The audience is an ML-augmented trading system that \
needs hard context, not general commentary.

Respond with ONLY a raw JSON object (no markdown, no code fences):
{
  "regime": "risk-on|risk-off|rotational|crisis",
  "themes": [
    {
      "theme": "short descriptive title",
      "impact": "2-3 sentences on market impact",
      "sectors_positive": ["sector1", "sector2"],
      "sectors_negative": ["sector3"]
    }
  ],
  "risk_factors": ["risk1", "risk2", "risk3"],
  "opportunities": ["opportunity1", "opportunity2"],
  "summary": "2-3 sentence overall market narrative"
}\
"""


def _price_context(hist) -> str:
    """Build price context string from a 1-year history DataFrame.

    Returns: current price, 52w high/low, 1w/1m/3m/1y % changes.
    """
    if hist is None or hist.empty or len(hist) < 2:
        return ""
    close = hist["Close"]
    cur = close.iloc[-1]
    hi52 = hist["High"].max()
    lo52 = hist["Low"].min()

    parts = [f"{cur:.2f}"]
    parts.append(f"52w: {lo52:.2f}-{hi52:.2f} ({(cur / hi52 - 1) * 100:+.1f}% from high)")
    for label, days in [("1w", 5), ("1m", 21), ("3m", 63), ("1y", len(close) - 1)]:
        if len(close) > days:
            prev = close.iloc[-1 - days]
            parts.append(f"{label}: {(cur / prev - 1) * 100:+.1f}%")
    return " | ".join(parts)


def _fetch_index_snapshot() -> str:
    """Fetch key index levels with 52w range and multi-period returns."""
    try:
        import yfinance as yf
        tickers = yf.Tickers(" ".join(_INDEX_TICKERS.keys()))
        lines = []
        for yf_sym, name in _INDEX_TICKERS.items():
            try:
                tk = tickers.tickers[yf_sym]
                hist = tk.history(period="1y")
                ctx = _price_context(hist)
                if ctx:
                    lines.append(f"  {name}: {ctx}")
            except Exception:
                continue
        return "\n".join(lines) if lines else "  (unavailable)"
    except Exception as e:
        logger.warning("[GLOBAL-CTX] Index snapshot failed: %s", e)
        return "  (unavailable)"


def _fetch_sector_performance() -> str:
    """Fetch sector ETF performance with 52w range and multi-period returns."""
    try:
        import yfinance as yf
        tickers = yf.Tickers(" ".join(_SECTOR_ETFS.keys()))
        lines = []
        for yf_sym, name in _SECTOR_ETFS.items():
            try:
                tk = tickers.tickers[yf_sym]
                hist = tk.history(period="1y")
                ctx = _price_context(hist)
                if ctx:
                    lines.append(f"  {name} ({yf_sym}): {ctx}")
            except Exception:
                continue
        return "\n".join(lines) if lines else "  (unavailable)"
    except Exception as e:
        logger.warning("[GLOBAL-CTX] Sector performance failed: %s", e)
        return "  (unavailable)"


def _fetch_general_news(max_articles: int = 20) -> str:
    """Fetch top general market news from Finnhub."""
    try:
        import finnhub
        import os
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            return "  (no API key)"
        client = finnhub.Client(api_key=api_key)
        articles = client.general_news("general", min_id=0)
        if not articles:
            return "  (no articles)"
        lines = []
        for a in articles[:max_articles]:
            headline = a.get("headline", "")
            source = a.get("source", "")
            if headline:
                prefix = f"[{source}] " if source else ""
                lines.append(f"  - {prefix}{headline}")
        return "\n".join(lines) if lines else "  (no articles)"
    except Exception as e:
        logger.warning("[GLOBAL-CTX] News fetch failed: %s", e)
        return "  (unavailable)"


def _fetch_fear_greed() -> str:
    """Get Fear & Greed values."""
    try:
        from sentiment import get_fear_greed
        fng = get_fear_greed()
        if fng and isinstance(fng, dict):
            val = fng.get("value")
            label = fng.get("label", "")
            if val is not None:
                return f"  Crypto Fear & Greed: {val}/100 ({label})"
        return "  (unavailable)"
    except Exception:
        return "  (unavailable)"


def _fetch_vix_detail() -> str:
    """Get VIX with regime label."""
    try:
        from macro_indicators import fetch_vix
        vix = fetch_vix()
        if vix is not None:
            if vix < 15:
                label = "low/complacent"
            elif vix < 20:
                label = "normal"
            elif vix < 25:
                label = "elevated"
            elif vix < 35:
                label = "high/fearful"
            else:
                label = "extreme/panic"
            return f"  VIX: {vix:.1f} ({label})"
        return "  (unavailable)"
    except Exception:
        return "  (unavailable)"


def _build_prompt(universe: list[str] | None = None) -> str:
    """Build the data-gathering prompt for the global context LLM call."""
    lines = []

    lines.append("## Key Market Indices (latest)")
    lines.append(_fetch_index_snapshot())
    lines.append("")

    lines.append("## Sector Rotation (ETF performance)")
    lines.append(_fetch_sector_performance())
    lines.append("")

    lines.append("## Sentiment Gauges")
    lines.append(_fetch_fear_greed())
    lines.append(_fetch_vix_detail())
    lines.append("")

    lines.append("## Top Market News Headlines")
    lines.append(_fetch_general_news())
    lines.append("")

    if universe:
        # Remove crypto pairs formatting for readability
        clean = [s.replace("/USD", "") for s in universe]
        lines.append(f"## Our Trading Universe ({len(universe)} symbols)")
        lines.append(f"  {', '.join(clean)}")
        lines.append("  Weight themes by relevance to these names, but do NOT "
                      "limit analysis to them — surface any market-moving event.")
        lines.append("")

    lines.append("Produce the global market context digest as JSON.")
    return "\n".join(lines)


def refresh_global_context(universe: list[str] | None = None,
                           force: bool = False) -> dict:
    """Fetch data and call LLM to produce a global context digest.

    Returns the context dict (cached for 1 hour). On failure returns
    the last cached version, or empty dict.
    """
    global _mem_cache

    # Check in-memory cache
    if not force and _mem_cache is not None:
        ts, ctx = _mem_cache
        if time.time() - ts < _CACHE_TTL:
            return ctx

    # Check disk cache
    if not force:
        disk = load_global_context()
        if disk:
            ts_str = disk.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
                age_sec = (datetime.now(timezone.utc) - ts).total_seconds()
                if age_sec < _CACHE_TTL:
                    _mem_cache = (time.time(), disk)
                    return disk
            except (ValueError, TypeError):
                pass

    # Build prompt and call LLM
    logger.info("[GLOBAL-CTX] Refreshing global market context...")

    from llm_client import call_gemini, call_llm, get_recommended_model
    prompt = _build_prompt(universe)

    # Use flash model — this is a summarization task, doesn't need pro
    model = get_recommended_model("analyst")
    response = call_gemini(prompt, system=_SYSTEM_PROMPT,
                           model=model, max_tokens=4096, json_mode=True)
    if not response:
        response = call_llm(prompt, system=_SYSTEM_PROMPT, max_tokens=4096)

    if not response:
        logger.warning("[GLOBAL-CTX] LLM call failed, using stale cache")
        return _mem_cache[1] if _mem_cache else {}

    # Parse response
    ctx = _parse_response(response)
    if not ctx:
        logger.warning("[GLOBAL-CTX] Could not parse LLM response")
        return _mem_cache[1] if _mem_cache else {}

    # Add metadata
    ctx["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ctx["model"] = model

    # Save to disk and memory
    _save_context(ctx)
    _mem_cache = (time.time(), ctx)

    themes = ctx.get("themes", [])
    logger.info("[GLOBAL-CTX] Regime: %s, %d themes, saved to disk",
                ctx.get("regime", "?"), len(themes))
    return ctx


def _parse_response(response: str) -> dict | None:
    """Parse LLM JSON response into context dict."""
    import re
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "regime" in parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Find outermost { ... }
    start = text.find("{")
    if start >= 0:
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _save_context(ctx: dict):
    """Write context to disk."""
    try:
        with open(_CONTEXT_FILE, "w") as f:
            json.dump(ctx, f, indent=2)
    except OSError as e:
        logger.error("[GLOBAL-CTX] Save error: %s", e)


def load_global_context() -> dict:
    """Load cached global context from disk. Returns empty dict on failure."""
    try:
        if _CONTEXT_FILE.exists():
            with open(_CONTEXT_FILE) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def format_context_for_prompt(ctx: dict) -> str:
    """Format the global context dict into a text section for LLM prompts.

    This is what gets injected into per-symbol analysis prompts.
    """
    if not ctx or "regime" not in ctx:
        return ""

    lines = ["## Global Market Context"]

    # Regime
    regime = ctx.get("regime", "unknown")
    lines.append(f"Regime: {regime.upper()}")

    # Themes
    themes = ctx.get("themes", [])
    if themes:
        lines.append("Key themes driving markets:")
        for i, t in enumerate(themes, 1):
            theme_name = t.get("theme", "?")
            impact = t.get("impact", "")
            pos = t.get("sectors_positive", [])
            neg = t.get("sectors_negative", [])
            lines.append(f"  {i}. {theme_name}: {impact}")
            tags = []
            if pos:
                tags.append("+" + ", +".join(pos))
            if neg:
                tags.append("-" + ", -".join(neg))
            if tags:
                lines.append(f"     Sectors: {'; '.join(tags)}")

    # Risks
    risks = ctx.get("risk_factors", [])
    if risks:
        lines.append("Tail risks: " + "; ".join(risks))

    # Opportunities
    opps = ctx.get("opportunities", [])
    if opps:
        lines.append("Opportunities: " + "; ".join(opps))

    # Summary
    summary = ctx.get("summary", "")
    if summary:
        lines.append(f"Summary: {summary}")

    # Age
    ts_str = ctx.get("timestamp", "")
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str)
            age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
            lines.append(f"(Updated {age_min:.0f} minutes ago)")
        except (ValueError, TypeError):
            pass

    lines.append("")
    return "\n".join(lines)


def format_context_for_gui(ctx: dict) -> str:
    """Format the global context for display in the GUI header.

    Returns a compact single-line or short multi-line summary.
    """
    if not ctx or "regime" not in ctx:
        return ""

    regime = ctx.get("regime", "unknown").upper()
    themes = ctx.get("themes", [])
    theme_names = [t.get("theme", "") for t in themes[:4] if t.get("theme")]
    summary = ctx.get("summary", "")

    # Age
    age_str = ""
    ts_str = ctx.get("timestamp", "")
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str)
            age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
            if age_min < 60:
                age_str = f"{age_min:.0f}m ago"
            else:
                age_str = f"{age_min / 60:.0f}h ago"
        except (ValueError, TypeError):
            pass

    parts = [f"Regime: {regime}"]
    if theme_names:
        parts.append(" | ".join(theme_names))
    if age_str:
        parts.append(f"({age_str})")

    return "  ".join(parts)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    if "--view" in sys.argv:
        ctx = load_global_context()
        if not ctx:
            print("No cached global context. Run with --refresh first.")
            sys.exit(1)
        print(format_context_for_prompt(ctx))
        sys.exit(0)

    # Default: refresh and print
    print("Fetching global market context...")
    universe = None
    try:
        from stock_config import load_stock_universe, CRYPTO_SYMBOLS
        universe = load_stock_universe() + list(CRYPTO_SYMBOLS)
    except Exception:
        pass

    ctx = refresh_global_context(universe=universe, force=True)
    if ctx:
        print()
        print(format_context_for_prompt(ctx))
    else:
        print("Failed to generate global context.")
        sys.exit(1)
