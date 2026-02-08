"""Pre-trade LLM analysis engine — batch analysis of trade candidates.

One LLM call per trading cycle (all candidates at once). Returns multiplier
(0.0–1.5) and reasoning per symbol. On any failure, returns {} for pass-through.
"""

import json
import re

from llm_config import load_llm_config
from llm_client import call_llm

_SYSTEM_PROMPT = (
    "You are a senior quantitative analyst reviewing trades for an automated "
    "trading system. For each candidate, assess whether the trade should proceed "
    "based on the technical signals, sentiment, and fundamental data provided. "
    "Return a JSON object mapping each symbol to a multiplier and brief reasoning. "
    "Multipliers: 0.0 = block trade, 0.5 = reduce position, 1.0 = neutral/proceed, "
    "1.5 = high conviction increase. Be concise."
)


def analyze_trades(candidates: list[dict], asset_type: str,
                   equity: float = 0, positions: list[str] = None,
                   fng_value: int = None) -> dict[str, dict]:
    """Batch-analyze trade candidates with LLM.

    Args:
        candidates: list of dicts with keys:
            symbol, bull_pred, bear_pred, sentiment_gate, sentiment_reasons,
            fundamentals_text (pre-formatted string)
        asset_type: 'crypto' or 'stock'
        equity: account equity for context
        positions: list of currently held symbols
        fng_value: current Fear & Greed index value

    Returns:
        dict mapping symbol -> {'m': float, 'r': str}
        Empty dict on failure (all symbols get default 1.0x).
    """
    config = load_llm_config()
    if not config.get("enabled") or not candidates:
        return {}

    prompt = _build_prompt(candidates, asset_type, equity, positions, fng_value)

    response = call_llm(prompt, system=_SYSTEM_PROMPT, max_tokens=1024)
    if not response:
        return {}

    return _parse_response(response, [c["symbol"] for c in candidates])


def _build_prompt(candidates, asset_type, equity, positions, fng_value):
    """Build the user prompt with all candidate data."""
    lines = ["## Portfolio Context"]
    lines.append(f"- Asset type: {asset_type}")
    if equity:
        lines.append(f"- Account equity: ${equity:,.0f}")
    if positions:
        lines.append(f"- Current positions: {', '.join(positions)}")
    if fng_value is not None:
        lines.append(f"- Fear & Greed Index: {fng_value}")
    lines.append("")
    lines.append("## Trade Candidates")

    for c in candidates:
        sym = c["symbol"]
        lines.append(f"\n### {sym}")

        bull = c.get("bull_pred")
        bear = c.get("bear_pred")
        if bull is not None:
            lines.append(f"- Technical: Bull={bull:+.4f}%")
        if bear is not None:
            lines.append(f"  Bear={bear:+.4f}%")

        sg = c.get("sentiment_gate")
        sr = c.get("sentiment_reasons")
        if sg is not None:
            reasons_str = ", ".join(sr) if sr else "neutral"
            lines.append(f"- Sentiment gate: {sg:.2f}x ({reasons_str})")

        ft = c.get("fundamentals_text", "")
        if ft:
            lines.append(f"- {ft}")

    lines.append("")
    lines.append('Respond with ONLY a JSON object:')
    lines.append('{"SYMBOL": {"m": 1.0, "r": "reason"}, ...}')

    return "\n".join(lines)


def _parse_response(response: str, symbols: list[str]) -> dict[str, dict]:
    """Parse LLM JSON response into symbol -> {m, r} dict."""
    # Try to extract JSON from the response (LLM might wrap in markdown)
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if not json_match:
        print(f"[LLM-ANALYST] Could not find JSON in response")
        return {}

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"[LLM-ANALYST] JSON parse error: {e}")
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
