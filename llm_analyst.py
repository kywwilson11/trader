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
from llm_client import call_llm, call_gemini

_ANALYSIS_FILE = Path(__file__).resolve().parent / "llm_analysis.json"

_SYSTEM_PROMPT = """\
You are a qualitative conviction scorer for a trading system. An ML model \
handles ALL technical analysis (price, momentum, volume, volatility). Your \
job is to evaluate ONLY qualitative factors the ML model cannot see.

IMPORTANT: The ML model has ALREADY decided this is a good technical setup. \
Do NOT second-guess the model's technical analysis. Your role is to adjust \
conviction based on qualitative factors only.

For each symbol, evaluate these qualitative factors:
1. NEWS IMPACT: Are there breaking events that change the fundamental picture? \
(earnings surprises, partnerships, hacks, fraud, regulatory actions, lawsuits)
2. FUNDAMENTAL CONTEXT: Does the valuation/growth story support or contradict \
the technical signal? (P/E expansion/compression, revenue acceleration/deceleration)
3. MACRO ENVIRONMENT: Does the broader market context help or hurt? \
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

    analyst_model = config.get("analyst_model", "gemini-2.5-flash")
    max_tok = 3072

    response = call_gemini(prompt, system=_SYSTEM_PROMPT,
                           model=analyst_model, max_tokens=max_tok,
                           json_mode=True)
    if not response:
        n_syms = len(candidates)
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


def _build_prompt(candidates, asset_type, equity, positions, fng_value,
                  model_config):
    """Build the user prompt with qualitative data only (no technicals)."""
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
    lines.append('For each symbol include: bull (1-2 sentences), bear (1-2 sentences), s (score 0.0-1.0), r (summary under 2 sentences).')
    lines.append('Example: {"BTC/USD": {"bull": "ETF inflows accelerating, institutional adoption growing.", "bear": "No material negative events.", "s": 0.65, "r": "Mild positive from ETF flows."}}')

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
