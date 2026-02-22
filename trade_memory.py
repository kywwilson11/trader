"""Trade memory — record past trades and inject lessons into LLM prompts.

Stores completed trades with outcomes in a JSON file. Provides per-symbol
history and one-line summaries for LLM prompt injection, helping the system
learn from past mistakes and avoid repeating them.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

_MEMORY_FILE = Path(__file__).resolve().parent / "trade_memory.json"
_MAX_PER_SYMBOL = 50  # Rolling window per symbol


def _load() -> dict:
    """Load trade memory from disk. Returns {symbol: [trade, ...]}."""
    try:
        if _MEMORY_FILE.exists():
            with open(_MEMORY_FILE) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save(data: dict):
    """Save trade memory to disk."""
    try:
        with open(_MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        print(f"[TRADE-MEMORY] Error saving: {e}")


def record_trade(symbol: str, action: str, entry_price: float,
                 exit_price: float, pnl_pct: float,
                 llm_score: float = None, reasoning: str = "",
                 news_context: str = "", exit_reason: str = ""):
    """Record a completed trade for future reference.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USD', 'AAPL')
        action: 'buy' or 'sell'
        entry_price: Entry price
        exit_price: Exit price
        pnl_pct: P&L percentage (positive = profit)
        llm_score: LLM conviction score at entry (0.0-1.0)
        reasoning: LLM reasoning at entry
        news_context: Key news at time of trade
        exit_reason: Why the trade was exited (stop_loss, take_profit, etc.)
    """
    data = _load()
    trades = data.setdefault(symbol, [])

    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec='seconds'),
        "action": action,
        "entry": round(entry_price, 6),
        "exit": round(exit_price, 6),
        "pnl_pct": round(pnl_pct, 4),
        "llm_score": round(llm_score, 4) if llm_score is not None else None,
        "reasoning": reasoning[:200],
        "news": news_context[:200],
        "exit_reason": exit_reason,
    }
    trades.append(record)

    # Rolling window: keep last N
    if len(trades) > _MAX_PER_SYMBOL:
        data[symbol] = trades[-_MAX_PER_SYMBOL:]

    _save(data)


def get_relevant_history(symbol: str, n: int = 3) -> list[dict]:
    """Get N most recent trades for this symbol."""
    data = _load()
    trades = data.get(symbol, [])
    return trades[-n:]


def get_lesson_summary(symbol: str) -> str:
    """One-line summary of patterns for this symbol for LLM injection.

    Returns empty string if no trade history exists.
    """
    data = _load()
    trades = data.get(symbol, [])
    if not trades:
        return ""

    recent = trades[-10:]  # Last 10 trades
    wins = sum(1 for t in recent if t.get("pnl_pct", 0) > 0)
    losses = len(recent) - wins
    avg_pnl = sum(t.get("pnl_pct", 0) for t in recent) / len(recent)

    # Most common exit reason
    exit_reasons = [t.get("exit_reason", "") for t in recent if t.get("exit_reason")]
    most_common = max(set(exit_reasons), key=exit_reasons.count) if exit_reasons else "unknown"

    return (f"Last {len(recent)} trades: {wins}W/{losses}L, "
            f"avg PnL {avg_pnl:+.2f}%, most common exit: {most_common}")
