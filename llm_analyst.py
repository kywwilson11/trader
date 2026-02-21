"""Pre-trade LLM analysis engine — comprehensive trade evaluation.

One LLM call per trading cycle (all candidates at once). Provides the LLM with:
  - ML model context (architecture, what it was trained on, prediction horizon)
  - Live technical indicator values (the same ones the model sees)
  - Recent news headlines
  - Fundamentals data
  - Fear & Greed index
  - Portfolio context

Returns multiplier (0.0–1.5) and reasoning per symbol.
On any failure, returns {} for pass-through (never blocks trades).
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from llm_config import load_llm_config
from llm_client import call_llm, call_gemini

_ANALYSIS_FILE = Path(__file__).resolve().parent / "llm_analysis.json"

_SYSTEM_PROMPT = """\
You are an independent trading analyst. Evaluate each symbol using the \
technical indicators, news, fundamentals, and market sentiment provided.

Your score directly determines whether a trade happens. Use the FULL \
continuous range from 0.00 to 1.50 — do NOT round to convenient numbers. \
Each symbol should get a DIFFERENT score reflecting its unique setup.

Scale:
- Below 0.30: sell signal (catastrophic risk, strong bearish setup)
- 0.30–0.49: bearish — buys blocked, existing positions reduced
- 0.50–0.79: cautious — small positions only
- 0.80–0.99: favorable — normal position sizing
- 1.00–1.19: bullish — full conviction
- 1.20–1.50: strong buy — boosted position size

For each symbol, analyze:
- Trend: price vs SMA, direction of momentum (RSI, MACD, Stochastic)
- Volatility: Bollinger Band position, ATR, recent returns
- Volume: is it confirming price action or diverging?
- News: catalysts, risks, anything the numbers don't capture
- Setup quality: does this match the strategy's entry criteria?

Be specific in reasoning — cite the actual indicator values driving your score.\
"""


def analyze_trades(candidates: list[dict], asset_type: str,
                   equity: float = 0, positions: list[str] = None,
                   fng_value: int = None,
                   model_config: dict = None) -> dict[str, dict]:
    """Batch-analyze trade candidates with LLM.

    Args:
        candidates: list of dicts with keys:
            symbol, pred_return, snapshot (dict of indicator values),
            fundamentals_text, news_headlines
        asset_type: 'crypto' or 'stock'
        equity: account equity for context
        positions: list of currently held symbols
        fng_value: current Fear & Greed index value
        model_config: model training config (seq_len, forward_bars, etc.)

    Returns:
        dict mapping symbol -> {'m': float, 'r': str}
        Empty dict on failure (all symbols get default 1.0x).
    """
    config = load_llm_config()
    if not config.get("enabled") or not candidates:
        return {}

    prompt = _build_prompt(candidates, asset_type, equity, positions,
                           fng_value, model_config)

    analyst_model = config.get("analyst_model", "gemini-2.5-flash")
    # Gemini 2.5 Flash and Pro are both thinking models — internal reasoning
    # tokens count against maxOutputTokens. Need generous limits.
    n_syms = len(candidates)
    is_pro = "pro" in analyst_model
    max_tok = 16384 if is_pro else 8192

    response = call_gemini(prompt, system=_SYSTEM_PROMPT,
                           model=analyst_model, max_tokens=max_tok,
                           json_mode=True)
    if not response:
        # Fall back to any available model
        response = call_llm(prompt, system=_SYSTEM_PROMPT,
                            max_tokens=max(2048, n_syms * 150))
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
    symbol -> {m, r, timestamp, model}.
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
            "r": entry.get("r", ""),
            "timestamp": ts,
            "model": model,
        }

    try:
        with open(_ANALYSIS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        print(f"[LLM-ANALYST] Error saving analysis: {e}")


def _format_indicators(snapshot: dict) -> str:
    """Format indicator snapshot into readable text for LLM."""
    if not snapshot:
        return "Indicators: unavailable"

    parts = []

    price = snapshot.get('Close')
    if price is not None:
        parts.append(f"Price=${price:,.2f}")

    # Trend
    sma_ratio = snapshot.get('Price_SMA20_Ratio')
    if sma_ratio is not None:
        pct = (sma_ratio - 1) * 100
        label = "above" if pct > 0 else "below"
        parts.append(f"SMA20={abs(pct):.1f}% {label}")

    # Momentum
    rsi = snapshot.get('RSI')
    if rsi is not None:
        zone = ""
        if rsi > 70:
            zone = " (OVERBOUGHT)"
        elif rsi < 30:
            zone = " (OVERSOLD)"
        parts.append(f"RSI={rsi:.1f}{zone}")

    stochk = snapshot.get('STOCHk_14_3_3')
    stochd = snapshot.get('STOCHd_14_3_3')
    if stochk is not None and stochd is not None:
        cross = ""
        if stochk > stochd:
            cross = " K>D(bullish)"
        elif stochk < stochd:
            cross = " K<D(bearish)"
        parts.append(f"Stoch={stochk:.0f}/{stochd:.0f}{cross}")

    # MACD
    macd = snapshot.get('MACD_12_26_9')
    macdh = snapshot.get('MACDh_12_26_9')
    macds = snapshot.get('MACDs_12_26_9')
    if macd is not None and macds is not None:
        signal = "above" if macd > macds else "below"
        parts.append(f"MACD={macd:.4f} ({signal} signal)")
    if macdh is not None:
        parts.append(f"MACD_hist={macdh:+.4f}")

    # Bollinger Bands
    bbp = snapshot.get('BBP_20_2.0')
    bbb = snapshot.get('BBB_20_2.0')
    if bbp is not None:
        zone = ""
        if bbp > 1.0:
            zone = " ABOVE upper band"
        elif bbp < 0.0:
            zone = " BELOW lower band"
        parts.append(f"BB%={bbp:.2f}{zone}")
    if bbb is not None:
        parts.append(f"BB_width={bbb:.2f}")

    # Returns
    ret4h = snapshot.get('Return_4h')
    ret12h = snapshot.get('Return_12h')
    if ret4h is not None:
        parts.append(f"Return_4h={ret4h:+.2f}%")
    if ret12h is not None:
        parts.append(f"Return_12h={ret12h:+.2f}%")

    # Volatility
    vol12h = snapshot.get('Volatility_12h')
    if vol12h is not None:
        parts.append(f"Volatility_12h={vol12h:.2f}%")

    roc = snapshot.get('ROC')
    if roc is not None:
        parts.append(f"ROC={roc:+.2f}")

    # Volume (from last completed bar)
    vol_ratio = snapshot.get('Volume_Ratio')
    if vol_ratio is not None:
        parts.append(f"Vol_ratio={vol_ratio:.2f}x")

    # Sentiment
    sent = snapshot.get('Daily_Sentiment')
    if sent is not None:
        parts.append(f"Daily_Sentiment={sent:.2f}")

    # Cross-asset
    btc_ret = snapshot.get('BTC_Return_1h')
    if btc_ret is not None:
        parts.append(f"BTC_1h={btc_ret:+.2f}%")
    btc_rsi = snapshot.get('BTC_RSI')
    if btc_rsi is not None:
        parts.append(f"BTC_RSI={btc_rsi:.0f}")

    atr_pct = snapshot.get('ATR_Pct')
    if atr_pct is not None:
        parts.append(f"ATR%={atr_pct:.2f}%")

    return "Indicators: " + ", ".join(parts)


_STRATEGY = {
    "crypto": {
        "position_size": "$250 per trade, max $750 per symbol",
        "hold_period": "~24 hours (hourly bars, 24-bar prediction horizon)",
        "stop_loss": "ATR-adaptive, 4% hard stop",
        "take_profit": "2:1 risk-reward ratio, 12% ceiling",
        "style": "Swing trading — buy dips in uptrends, avoid catching falling knives",
    },
    "stock": {
        "position_size": "$2,500 per trade",
        "hold_period": "~1-5 days (hourly bars, 8-24 bar prediction horizon)",
        "stop_loss": "ATR-adaptive, 3% hard stop",
        "take_profit": "2:1 risk-reward ratio, 15% ceiling",
        "style": "Momentum/mean-reversion — buy strong setups with favorable risk/reward",
    },
}


def _build_prompt(candidates, asset_type, equity, positions, fng_value,
                  model_config):
    """Build the user prompt with all candidate data."""
    lines = []

    # Strategy context
    strat = _STRATEGY.get(asset_type, _STRATEGY["crypto"])
    fb = model_config.get('forward_bars', 24) if model_config else 24
    lines.append("## Trading Strategy")
    lines.append(f"- Position size: {strat['position_size']}")
    lines.append(f"- Holding period: {strat['hold_period']}")
    lines.append(f"- Stop-loss: {strat['stop_loss']}")
    lines.append(f"- Take-profit: {strat['take_profit']}")
    lines.append(f"- Style: {strat['style']}")
    lines.append(f"- Evaluate where each symbol is likely to go over the next ~{fb} hours.")
    lines.append("")

    # Portfolio context
    lines.append("## Portfolio Context")
    lines.append(f"- Asset type: {asset_type}")
    if equity:
        lines.append(f"- Account equity: ${equity:,.0f}")
    if positions:
        lines.append(f"- Currently holding: {', '.join(positions)}")
    else:
        lines.append("- No open positions")
    if fng_value is not None:
        fng_label = ("Extreme Fear" if fng_value <= 10 else
                     "Fear" if fng_value <= 25 else
                     "Neutral" if fng_value <= 55 else
                     "Greed" if fng_value <= 75 else "Extreme Greed")
        lines.append(f"- Fear & Greed Index: {fng_value} ({fng_label})")
    lines.append("")

    # Symbols to evaluate
    lines.append("## Symbols to Evaluate")

    for c in candidates:
        sym = c["symbol"]
        lines.append(f"\n### {sym}")

        # Technical indicators
        snapshot = c.get("snapshot")
        if snapshot:
            lines.append(f"- {_format_indicators(snapshot)}")

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

    lines.append("")
    lines.append('Respond with ONLY a raw JSON object (no markdown, no code fences).')
    lines.append('Keep each "r" under 2 sentences. Cite specific indicator values.')
    lines.append('Example: {"BTC/USD": {"m": 0.73, "r": "RSI 42 rising from oversold, but MACD still below signal. Volume weak at 0.6x."}, "ETH/USD": {"m": 1.14, "r": "Stoch K>D crossover at 28, price 2.1% below SMA20 — classic dip setup. ARK accumulation news bullish."}}')

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
    """Parse LLM JSON response into symbol -> {m, r} dict."""
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
            m = entry.get("m", 1.0)
            try:
                m = float(m)
                m = max(0.0, min(1.5, m))
            except (TypeError, ValueError):
                m = 1.0
            result[sym] = {"m": m, "r": entry.get("r", "")}

    return result
