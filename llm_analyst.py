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
You are a research analyst producing trade intelligence that complements an \
ML trading model. The ML model handles pattern recognition on technical \
indicators, but it cannot read news, understand narratives, evaluate \
management quality, or anticipate catalysts. You can see everything — \
use all the data provided to form a complete, informed view.

For each symbol, synthesize:
1. WHY is it moving? What news, events, or macro forces explain the recent \
price action? Be specific — cite events, dates, and magnitudes.
2. FUNDAMENTALS: Is the valuation compelling or stretched given growth? \
What do analyst targets and earnings trajectory imply?
3. CATALYSTS: What upcoming events could move the stock? Earnings dates, \
FDA decisions, product launches, macro events, sector rotation.
4. RISKS: What could go wrong? Crowded positioning, regulatory headwinds, \
competitive threats, deteriorating fundamentals.
5. SYNTHESIS: Given all of the above, what's the risk/reward skew? \
What would you do with this stock today?

SCORING — use precise, continuous values (e.g., 0.37, 0.62, 0.78):
- 0.00–0.15: VETO — confirmed catastrophe (fraud, insolvency, delisting)
- 0.15–0.35: Bearish — material negative catalysts, poor risk/reward
- 0.35–0.50: Lean negative — more headwinds than tailwinds
- 0.50: Neutral — balanced or insufficient information
- 0.50–0.65: Lean positive — modest tailwinds, decent setup
- 0.65–0.85: Bullish — clear catalysts, strong backdrop
- 0.85–1.00: Strong conviction — exceptional, multi-factor opportunity

Use the FULL continuous range. Scores like 0.43 or 0.71 are expected. \
Avoid rounding to 0.05 increments — be precise about your conviction level.\
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


def _build_symbol_profiles(symbols):
    """Build comprehensive data profiles for symbols via yfinance.

    Returns dict[symbol -> str] with structured text blocks containing
    price action, technicals, fundamentals, and news.
    """
    import yfinance as yf
    import numpy as np

    result = {}
    yf_map = {}
    for sym in symbols:
        yf_sym = sym.replace('/', '-') if '/' in sym else sym
        yf_map[yf_sym] = sym

    try:
        tickers = yf.Tickers(list(yf_map.keys()))
    except Exception as e:
        print(f"[LLM-ANALYST] yfinance fetch failed: {e}")
        return result

    for yf_sym, orig_sym in yf_map.items():
        try:
            tk = tickers.tickers[yf_sym]
            info = tk.info or {}
            h = tk.history(period='1y')
            if h.empty or len(h) < 5:
                continue

            lines = []
            close = h['Close']
            vol = h['Volume']
            cur = close.iloc[-1]

            # --- Price Performance ---
            perf = [f"Current: ${cur:.2f}"]
            for label, days in [('1w', 5), ('1m', 21), ('3m', 63), ('1y', len(h) - 1)]:
                if len(h) >= days:
                    prev = close.iloc[-days]
                    perf.append(f"{label}: {(cur / prev - 1) * 100:+.1f}%")
            hi52 = h['High'].max()
            lo52 = h['Low'].min()
            perf.append(f"52w range: ${lo52:.2f}-${hi52:.2f}"
                        f" ({(cur / hi52 - 1) * 100:+.1f}% from high)")
            lines.append("Price: " + " | ".join(perf))

            # --- Technicals ---
            techs = []
            if len(close) >= 20:
                sma20 = close.rolling(20).mean().iloc[-1]
                techs.append(f"SMA20: ${sma20:.2f} ({(cur / sma20 - 1) * 100:+.1f}%)")
            if len(close) >= 50:
                sma50 = close.rolling(50).mean().iloc[-1]
                techs.append(f"SMA50: ${sma50:.2f} ({(cur / sma50 - 1) * 100:+.1f}%)")
            sma200 = info.get('twoHundredDayAverage')
            if sma200:
                techs.append(f"SMA200: ${sma200:.2f} ({(cur / sma200 - 1) * 100:+.1f}%)")

            # RSI
            if len(close) >= 15:
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
                loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
                rsi = 100 - 100 / (1 + gain / loss) if loss > 0 else 50
                rsi_label = ("OVERSOLD" if rsi < 30 else
                             "OVERBOUGHT" if rsi > 70 else "neutral")
                techs.append(f"RSI14: {rsi:.0f} ({rsi_label})")

            # Volume
            if len(vol) >= 20:
                avg_vol = vol.rolling(20).mean().iloc[-1]
                if avg_vol > 0:
                    vol_ratio = vol.iloc[-1] / avg_vol
                    techs.append(f"Volume: {vol_ratio:.1f}x 20d avg")

            # Volatility
            if len(close) >= 21:
                vol20 = close.pct_change().rolling(20).std().iloc[-1] * 100
                techs.append(f"20d volatility: {vol20:.2f}%")

            if techs:
                lines.append("Technicals: " + " | ".join(techs))

            # --- Recent daily closes (last 5 trading days) ---
            recent = []
            for i in range(-5, 0):
                if abs(i) <= len(close):
                    d = h.index[i]
                    c = close.iloc[i]
                    chg = (c / close.iloc[i - 1] - 1) * 100 if abs(i - 1) <= len(close) else 0
                    recent.append(f"{d.strftime('%m/%d')}: ${c:.2f} ({chg:+.1f}%)")
            if recent:
                lines.append("Last 5 days: " + ", ".join(recent))

            # --- Fundamentals ---
            fund = []
            for key, label in [
                ('marketCap', 'MktCap'),
                ('trailingPE', 'P/E'),
                ('forwardPE', 'Fwd P/E'),
                ('priceToBook', 'P/B'),
                ('revenueGrowth', 'RevGrowth'),
                ('earningsGrowth', 'EarningsGrowth'),
                ('beta', 'Beta'),
                ('shortRatio', 'ShortRatio'),
            ]:
                v = info.get(key)
                if v is not None:
                    if key == 'marketCap':
                        if v >= 1e12:
                            fund.append(f"{label}: ${v / 1e12:.1f}T")
                        elif v >= 1e9:
                            fund.append(f"{label}: ${v / 1e9:.1f}B")
                        else:
                            fund.append(f"{label}: ${v / 1e6:.0f}M")
                    elif 'Growth' in key:
                        fund.append(f"{label}: {v * 100:+.1f}%")
                    else:
                        fund.append(f"{label}: {v:.2f}")
            # Analyst targets
            target = info.get('targetMeanPrice')
            n_analysts = info.get('numberOfAnalystOpinions')
            rec = info.get('recommendationKey')
            if target and n_analysts:
                upside = (target / cur - 1) * 100
                fund.append(f"Analyst target: ${target:.2f} ({upside:+.1f}%,"
                            f" {n_analysts} analysts, {rec})")
            if info.get('sector'):
                fund.append(f"Sector: {info['sector']}")

            if fund:
                lines.append("Fundamentals: " + " | ".join(fund))

            # --- News (from yfinance, more relevant than Finnhub) ---
            try:
                news = tk.news or []
                if news:
                    news_lines = []
                    for a in news[:6]:
                        title = a.get('title',
                                      a.get('content', {}).get('title', ''))
                        if title:
                            pub = a.get('publisher',
                                        a.get('content', {}).get('provider',
                                              {}).get('displayName', ''))
                            news_lines.append(f"[{pub}] {title}" if pub
                                              else title)
                    if news_lines:
                        lines.append("Recent news:\n  - " +
                                     "\n  - ".join(news_lines))
            except Exception:
                pass

            result[orig_sym] = "\n".join(lines)
        except Exception as e:
            print(f"[LLM-ANALYST] Profile failed for {orig_sym}: {e}")

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

        # Comprehensive data profile (price, technicals, fundamentals, news)
        profile = c.get("profile")
        if profile:
            lines.append(profile)

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
    lines.append('For each symbol: bull (2-3 sentences with specifics), '
                 'bear (2-3 sentences with risks), '
                 's (precise continuous score like 0.37 or 0.72, NOT rounded to 0.05), '
                 'r (2-3 sentence actionable synthesis).')

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

    BATCH_SIZE = 3  # symbols per LLM call (profiles are data-rich)

    for asset_type, syms in [('stock', stock_syms), ('crypto', crypto_syms)]:
        # Build comprehensive profiles (price, technicals, fundamentals, news)
        print(f"[LLM-ANALYST] Fetching {asset_type} data for {len(syms)} symbols...")
        profiles = _build_symbol_profiles(syms)
        print(f"[LLM-ANALYST] Got profiles for {len(profiles)}/{len(syms)} symbols")

        for i in range(0, len(syms), BATCH_SIZE):
            batch = syms[i:i + BATCH_SIZE]
            candidates = []
            for sym in batch:
                c = {"symbol": sym, "pred_return": None}
                if sym in profiles:
                    c["profile"] = profiles[sym]
                # Supplement with Finnhub headlines if yfinance news was sparse
                try:
                    headlines = get_recent_headlines(sym)
                    if headlines:
                        c["news_headlines"] = headlines[:3]
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
