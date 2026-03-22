"""Pre-trade LLM qualitative conviction scorer.

One LLM call per trading cycle (all candidates at once). The LLM evaluates
ONLY qualitative factors the ML model cannot see:
  - News events and catalysts
  - Fundamental context (valuations, growth)
  - Macro environment (Fear & Greed, sector rotation)

Returns score (0.0–1.0) with bull/bear reasoning per symbol.
On any failure, returns {} for pass-through (never blocks trades).
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from llm_config import load_llm_config
from llm_client import call_llm, call_gemini, get_recommended_model

_ANALYSIS_FILE = Path(__file__).resolve().parent / "llm_analysis.json"

_SYSTEM_PROMPT = """\
You are a qualitative conviction scorer for a trading system. An ML model \
handles ALL technical analysis (price, momentum, volume, volatility). Your \
job is to evaluate ONLY qualitative factors the ML model cannot see.

IMPORTANT: The ML model has ALREADY decided this is a good technical setup. \
Do NOT second-guess the model's technical analysis. Your role is to adjust \
conviction based on qualitative factors only.

For each symbol, evaluate these factors:
1. PRICE CONTEXT: Where is the price relative to recent history? At 52-week \
highs/lows? Sharp recent move that's extended or has room to run?
2. NEWS IMPACT: Are there breaking events that change the fundamental picture? \
(earnings surprises, partnerships, hacks, fraud, regulatory actions, lawsuits)
3. FUNDAMENTAL CONTEXT: Does the valuation/growth story support or contradict \
the signal? (P/E expansion/compression, revenue acceleration/deceleration)
4. MACRO ENVIRONMENT: Does the broader market context help or hurt? \
(Fear & Greed regime, sector rotation, risk-on/risk-off)

STRUCTURED REASONING — For each symbol, you MUST:
- State the BULL case (1-2 sentences): Why qualitative factors support buying
- State the BEAR case (1-2 sentences): Why qualitative factors argue caution
- Weigh both sides and output your conviction score

SCORING (continuous, use full range):
- 0.00–0.15: VETO — confirmed catastrophic event (hack, fraud, insolvency, delisting)
- 0.15–0.40: Strong negative — material negative news, fundamental deterioration
- 0.40–0.60: Neutral — no significant qualitative signal either way
- 0.60–0.80: Mildly positive — favorable news or fundamental backdrop
- 0.80–1.00: Strong positive — material positive catalyst (earnings beat, major partnership)

DEFAULT TO 0.50 (neutral) when there is no significant qualitative information. \
The ML model's technical signal is the primary driver — your score only adjusts it. \
Do NOT be conservative by default. Only deviate from 0.50 when you have specific \
qualitative evidence.\
"""


def analyze_trades(candidates: list[dict], asset_type: str,
                   equity: float = 0, positions: list[str] = None,
                   fng_value: int = None,
                   model_config: dict = None) -> dict[str, dict]:
    """Batch-analyze trade candidates with LLM.

    Args:
        candidates: list of dicts with keys:
            symbol, pred_return,
            fundamentals_text, news_headlines
        asset_type: 'crypto' or 'stock'
        equity: account equity for context
        positions: list of currently held symbols
        fng_value: current Fear & Greed index value
        model_config: model training config (seq_len, forward_bars, etc.)

    Returns:
        dict mapping symbol -> {'m': float, 's': float, 'r': str,
                                 'bull': str, 'bear': str}
        Empty dict on failure (all symbols get default pass-through).
    """
    config = load_llm_config()
    if not config.get("enabled") or not candidates:
        return {}

    prompt = _build_prompt(candidates, asset_type, equity, positions,
                           fng_value, model_config)

    analyst_model = get_recommended_model('analyst')
    n_syms = len(candidates)
    max_tok = max(4096, n_syms * 400)

    response = call_gemini(prompt, system=_SYSTEM_PROMPT,
                           model=analyst_model, max_tokens=max_tok,
                           json_mode=True)
    if not response:
        response = call_llm(prompt, system=_SYSTEM_PROMPT,
                            max_tokens=max_tok)
    if not response:
        return {}

    result = _parse_response(response, [c["symbol"] for c in candidates])

    # Persist analysis to disk for GUI display
    if result:
        _save_analysis(result, asset_type, analyst_model)

    return result


def load_analysis() -> dict:
    """Load the latest LLM analysis from disk.

    Returns dict with keys: crypto, stock — each mapping
    symbol -> {m, s, r, bull, bear, timestamp, model}.
    """
    try:
        if _ANALYSIS_FILE.exists():
            with open(_ANALYSIS_FILE) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_analysis(result: dict, asset_type: str, model: str):
    """Append analysis results to the JSON file."""
    ts = datetime.now(timezone.utc).isoformat(timespec='seconds')

    # Load existing
    data = load_analysis()

    # Update the asset_type section
    section = data.setdefault(asset_type, {})
    for sym, entry in result.items():
        section[sym] = {
            "m": entry["m"],
            "s": entry.get("s", entry["m"] / 1.5),
            "r": entry.get("r", ""),
            "bull": entry.get("bull", ""),
            "bear": entry.get("bear", ""),
            "timestamp": ts,
            "model": model,
        }

    try:
        with open(_ANALYSIS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        print(f"[LLM-ANALYST] Error saving analysis: {e}")


def _fetch_price_context(symbols):
    """Fetch price performance context for a list of symbols via yfinance."""
    import yfinance as yf

    result = {}
    # Convert crypto symbols for yfinance (BTC/USD → BTC-USD)
    yf_map = {}
    for sym in symbols:
        yf_sym = sym.replace('/', '-') if '/' in sym else sym
        yf_map[yf_sym] = sym

    try:
        tickers = yf.Tickers(list(yf_map.keys()))
        for yf_sym, orig_sym in yf_map.items():
            try:
                h = tickers.tickers[yf_sym].history(period='1y')
                if h.empty or len(h) < 5:
                    continue
                cur = h['Close'].iloc[-1]
                parts = [f"${cur:.2f}"]
                if len(h) >= 5:
                    w1 = h['Close'].iloc[-5]
                    parts.append(f"1w: {(cur / w1 - 1) * 100:+.1f}%")
                if len(h) >= 21:
                    m1 = h['Close'].iloc[-21]
                    parts.append(f"1m: {(cur / m1 - 1) * 100:+.1f}%")
                if len(h) >= 63:
                    m3 = h['Close'].iloc[-63]
                    parts.append(f"3m: {(cur / m3 - 1) * 100:+.1f}%")
                y1 = h['Close'].iloc[0]
                parts.append(f"1y: {(cur / y1 - 1) * 100:+.1f}%")
                hi52 = h['High'].max()
                lo52 = h['Low'].min()
                parts.append(f"52w range: ${lo52:.2f}-${hi52:.2f}"
                             f" ({(cur / hi52 - 1) * 100:+.1f}% from high)")
                result[orig_sym] = " | ".join(parts)
            except Exception:
                pass
    except Exception as e:
        print(f"[LLM-ANALYST] Price context fetch failed: {e}")

    return result


def _build_prompt(candidates, asset_type, equity, positions, fng_value,
                  model_config):
    """Build the user prompt with qualitative and price context."""
    lines = []

    # Market context
    lines.append("## Market Context")
    lines.append(f"- Asset type: {asset_type}")

    if fng_value is not None:
        fng_label = ("Extreme Fear" if fng_value <= 10 else
                     "Fear" if fng_value <= 25 else
                     "Neutral" if fng_value <= 55 else
                     "Greed" if fng_value <= 75 else "Extreme Greed")
        lines.append(f"- Market regime: {fng_label} (Fear & Greed: {fng_value})")

    if equity:
        lines.append(f"- Account equity: ${equity:,.0f}")
    if positions:
        lines.append(f"- Currently holding: {', '.join(positions)}")
    else:
        lines.append("- No open positions")
    lines.append("")

    # Trade memory injection
    try:
        from trade_memory import get_lesson_summary
        has_memory = True
    except ImportError:
        has_memory = False

    # Symbols to evaluate
    lines.append("## Symbols to Evaluate")

    for c in candidates:
        sym = c["symbol"]
        lines.append(f"\n### {sym}")

        # Price context
        price_ctx = c.get("price_context")
        if price_ctx:
            lines.append(f"- Price: {price_ctx}")

        # ML model prediction context
        pred_return = c.get("pred_return")
        if pred_return is not None:
            direction = "bullish" if pred_return > 0 else "bearish"
            lines.append(f"- ML model prediction: {pred_return:+.4f}% ({direction} signal)")

        # News
        headlines = c.get("news_headlines")
        if headlines:
            lines.append("- Recent News:")
            for h in headlines[:5]:
                lines.append(f"  - {h}")

        # Fundamentals
        ft = c.get("fundamentals_text", "")
        if ft:
            lines.append(f"- {ft}")

        # Trade memory
        if has_memory:
            try:
                lesson = get_lesson_summary(sym)
                if lesson:
                    lines.append(f"- Past trade context: {lesson}")
            except Exception:
                pass

    lines.append("")
    lines.append('Respond with ONLY a raw JSON object (no markdown, no code fences).')
    lines.append('For each symbol include: bull (2-3 sentences with specific data points), '
                 'bear (2-3 sentences with specific risks), s (score 0.0-1.0), '
                 'r (1-2 sentence summary referencing current price action).')
    lines.append('Be SPECIFIC — cite price levels, percentage moves, news events, '
                 'dates. Avoid generic statements like "could go up or down."')
    lines.append('Example: {"GLD": {"bull": "Gold at $290, up 48% YoY on central bank buying '
                 'and rate cut expectations. 52w high of $310 suggests room to run if inflation '
                 'stays sticky.", "bear": "Down 12% from highs, sharp 10% weekly drop suggests '
                 'profit-taking. If real yields rise, gold could retest $260 support.", '
                 '"s": 0.55, "r": "Consolidating after strong YoY run. Near-term pullback '
                 'but macro backdrop remains supportive."}}')

    return "\n".join(lines)


def _repair_truncated_json(text: str) -> dict | None:
    """Try to salvage a truncated JSON response.

    If the LLM was cut off mid-response, we close open strings and braces
    so we can still extract any complete symbol entries.
    """
    start = text.find('{')
    if start < 0:
        return None

    fragment = text[start:]
    # Close any open string (odd number of unescaped quotes)
    in_string = False
    for i, ch in enumerate(fragment):
        if ch == '"' and (i == 0 or fragment[i - 1] != '\\'):
            in_string = not in_string
    if in_string:
        fragment += '..."'

    # Close open braces
    depth = 0
    for ch in fragment:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
    fragment += '}' * max(0, depth)

    try:
        parsed = json.loads(fragment)
        if isinstance(parsed, dict):
            print(f"[LLM-ANALYST] Repaired truncated JSON, recovered {len(parsed)} entries")
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_response(response: str, symbols: list[str]) -> dict[str, dict]:
    """Parse LLM JSON response into symbol -> {m, s, r, bull, bear} dict.

    Supports both new format {s, bull, bear, r} and legacy {m, r}.
    """
    # Strip markdown code fences if present
    text = response.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    parsed = None

    # Attempt 1: try parsing the whole cleaned text as JSON
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: find outermost { ... } using brace counting
    if parsed is None:
        start = text.find('{')
        if start >= 0:
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            try:
                parsed = json.loads(text[start:end])
            except (json.JSONDecodeError, ValueError):
                pass

    # Attempt 3: repair truncated JSON (close open strings + braces)
    if parsed is None:
        parsed = _repair_truncated_json(text)

    if not parsed or not isinstance(parsed, dict):
        print(f"[LLM-ANALYST] Could not parse JSON from response ({len(response)} chars): {response[:200]}")
        return {}

    result = {}
    for sym in symbols:
        entry = parsed.get(sym) or parsed.get(sym.replace("/", ""))
        if entry and isinstance(entry, dict):
            # New format: "s" field (0.0-1.0)
            if "s" in entry:
                s = entry.get("s", 0.5)
                try:
                    s = float(s)
                    s = max(0.0, min(1.0, s))
                except (TypeError, ValueError):
                    s = 0.5
                # Map to legacy "m" field: s * 1.5 for backward compat
                m = round(s * 1.5, 2)
            # Legacy format: "m" field (0.0-1.5)
            elif "m" in entry:
                m = entry.get("m", 1.0)
                try:
                    m = float(m)
                    m = max(0.0, min(1.5, m))
                except (TypeError, ValueError):
                    m = 1.0
                # Derive s from m for new consumers
                s = round(m / 1.5, 4)
            else:
                m = 0.75  # 0.5 * 1.5 = neutral
                s = 0.5

            result[sym] = {
                "m": m,
                "s": s,
                "r": entry.get("r", ""),
                "bull": entry.get("bull", ""),
                "bear": entry.get("bear", ""),
            }

    return result


def refresh_all():
    """Analyze ALL symbols in the universe (stocks + crypto).

    Called from GUI 'Refresh All LLM Analysis' button or CLI.
    Batches symbols to avoid exceeding LLM token limits.
    """
    from stock_config import load_stock_universe, CRYPTO_SYMBOLS
    from sentiment import get_recent_headlines, get_fear_greed
    from fundamentals import format_fundamentals_for_llm

    stock_syms = load_stock_universe()
    crypto_syms = list(CRYPTO_SYMBOLS)

    fng = None
    try:
        fng_data = get_fear_greed()
        if isinstance(fng_data, dict):
            fng = fng_data.get('value')
        elif isinstance(fng_data, (int, float)):
            fng = int(fng_data)
    except Exception:
        pass

    BATCH_SIZE = 5  # symbols per LLM call (keep small for token limits)

    for asset_type, syms in [('stock', stock_syms), ('crypto', crypto_syms)]:
        # Batch-fetch price context via yfinance
        price_ctx = _fetch_price_context(syms)

        for i in range(0, len(syms), BATCH_SIZE):
            batch = syms[i:i + BATCH_SIZE]
            candidates = []
            for sym in batch:
                c = {"symbol": sym, "pred_return": None}
                if sym in price_ctx:
                    c["price_context"] = price_ctx[sym]
                try:
                    headlines = get_recent_headlines(sym)
                    if headlines:
                        c["news_headlines"] = headlines[:5]
                except Exception:
                    pass
                try:
                    ft = format_fundamentals_for_llm(sym)
                    if ft:
                        c["fundamentals_text"] = ft
                except Exception:
                    pass
                candidates.append(c)

            print(f"[LLM-ANALYST] Analyzing {asset_type} batch "
                  f"{i // BATCH_SIZE + 1}: {', '.join(batch)}")
            try:
                result = analyze_trades(candidates, asset_type, fng_value=fng)
                n = len(result) if result else 0
                print(f"[LLM-ANALYST] Got {n}/{len(batch)} results")
            except Exception as e:
                print(f"[LLM-ANALYST] Batch failed: {e}")

            # Rate limit between batches
            time.sleep(2)

    print("[LLM-ANALYST] Refresh complete")


if __name__ == "__main__":
    import sys
    if "--refresh-all" in sys.argv:
        refresh_all()
    else:
        print("Usage: python llm_analyst.py --refresh-all")
