"""Trade memory — record past trades and inject lessons into LLM prompts.

Stores completed trades with outcomes in a JSON file. Provides one-line
per-symbol summaries for LLM prompt injection, helping the system learn
from past mistakes and avoid repeating them.
"""

import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:  # non-POSIX — thread locking still applies
    fcntl = None

from log_config import get_logger

logger = get_logger(__name__)

_MEMORY_FILE = Path(__file__).resolve().parent / "trade_memory.json"
_MAX_PER_SYMBOL = 50  # Rolling window per symbol

# Both loops write this file: combined-bot mode runs them as two threads in
# one process (run_bots.py), pipeline mode as two processes (run_pipeline.py).
# The thread lock serializes the former, the advisory flock the latter —
# without them concurrent load-modify-saves silently lose records.
_write_lock = threading.Lock()


@contextmanager
def _cross_process_lock():
    """Advisory flock on a sidecar file (os.replace swaps the memory file's
    inode, so the data file itself can't carry the lock). Best-effort: on
    any lock failure the write proceeds unlocked rather than being dropped."""
    if fcntl is None:
        yield
        return
    try:
        fd = open(str(_MEMORY_FILE) + ".lock", "w")
    except OSError:
        yield
        return
    locked = False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
    except OSError:
        pass
    try:
        yield
    finally:
        if locked:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        fd.close()


def _load() -> dict:
    """Load trade memory from disk. Returns {symbol: [trade, ...]}.

    A corrupt or wrong-shape file is quarantined to trade_memory.json.corrupt
    (preserving the bytes for manual recovery) instead of being left in place
    for the next _save to silently overwrite — losing the history starves
    Kelly sizing, monitor_drift's CUSUM and the LLM lesson context.
    """
    try:
        if not _MEMORY_FILE.exists():
            return {}
        with open(_MEMORY_FILE) as f:
            data = json.load(f)
        if not (isinstance(data, dict)
                and all(isinstance(v, list) for v in data.values())):
            raise ValueError(f"expected dict of lists, got {type(data).__name__}")
        return data
    except OSError as e:
        logger.warning("[TRADE-MEMORY] cannot read memory file: %s", e)
        return {}
    except Exception as e:
        logger.warning("[TRADE-MEMORY] corrupt memory file (%s) — "
                       "quarantining to %s.corrupt", e, _MEMORY_FILE.name)
        try:
            os.replace(_MEMORY_FILE, str(_MEMORY_FILE) + ".corrupt")
        except OSError:
            pass
        return {}


def load_all() -> dict:
    """Public read API: the full memory as {symbol: [trade, ...]} ({} when
    missing or corrupt). For external consumers (trading_utils' Kelly
    sizing, monitor_drift) instead of re-reading the file themselves."""
    return _load()


def _save(data: dict) -> None:
    """Save trade memory to disk atomically (write-then-rename).

    The tmp name is unique per writer: with one shared tmp path a writer's
    os.replace can promote another writer's half-written file (or steal its
    tmp out from under it), corrupting or dropping records.
    """
    tmp = f"{_MEMORY_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, str(_MEMORY_FILE))
    except Exception as e:
        logger.warning("[TRADE-MEMORY] Error saving: %s", e)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def record_trade(symbol: str, action: str, entry_price: float,
                 exit_price: float, pnl_pct: float,
                 llm_score: float | None = None, reasoning: str = "",
                 news_context: str = "", exit_reason: str = "",
                 estimated: bool = False) -> None:
    """Record a completed trade for future reference.

    Best-effort journal — never raises: call sites sit in live loops with no
    guard (several fire AFTER a sell filled, one inside the circuit-breaker
    branch BEFORE emergency_flatten), so an exception here aborts the cycle
    and desyncs position tracking.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USD', 'AAPL')
        action: exit side of the round trip (currently always 'sell')
        entry_price: Entry price
        exit_price: Exit price
        pnl_pct: P&L percentage (positive = profit)
        llm_score: LLM conviction score at entry (0.0-1.0)
        reasoning: LLM reasoning at entry
        news_context: Key news at time of trade
        exit_reason: Why the trade was exited (stop_loss, take_profit, etc.)
        estimated: True when exit_price is a quote estimate rather than a
            confirmed fill. Estimated records are excluded from Kelly sizing —
            stop exits have the worst slippage and recording them at
            pre-slippage midpoints inflates the Kelly fraction.
    """
    try:
        with _write_lock, _cross_process_lock():
            data = _load()
            trades = data.setdefault(symbol, [])

            record = {
                "ts": datetime.now(timezone.utc).isoformat(timespec='seconds'),
                "action": action,
                "entry": round(entry_price, 6),
                "exit": round(exit_price, 6),
                "pnl_pct": round(pnl_pct, 4),
                "llm_score": round(llm_score, 4) if llm_score is not None else None,
                "reasoning": str(reasoning or "")[:200],
                "news": str(news_context or "")[:200],
                "exit_reason": exit_reason,
                "estimated": estimated,
            }
            trades.append(record)

            # Rolling window: keep last N
            if len(trades) > _MAX_PER_SYMBOL:
                data[symbol] = trades[-_MAX_PER_SYMBOL:]

            _save(data)
    except Exception as e:
        logger.warning("[TRADE-MEMORY] record failed: %s", e)


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
