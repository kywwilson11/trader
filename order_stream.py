"""TradingStream order-update cache (optional, alpaca-py only).

`manage_order_lifecycle` polls GET /orders every 2 seconds. With the
trade_updates websocket, fills/cancels/rejections arrive as push events —
one shared connection per account replaces thousands of daily polls AND
catches server-side bracket/trailing-stop fills the 30s loop otherwise
misses between cycles.

Design: a background thread runs alpaca-py's TradingStream and writes the
latest known state per order_id into a bounded, lock-guarded dict. The
polling path CONSULTS the cache to decide when to make its single
authoritative REST fetch; correctness never depends on the stream (missed
events just mean we fall back to polling cadence).

Enable with TRADER_ORDER_STREAM=1 (requires alpaca-py installed).
"""

import os
import threading
import time

from log_config import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_states: dict[str, dict] = {}
_MAX_ENTRIES = 500
_started = False

TERMINAL = {'filled', 'canceled', 'expired', 'rejected'}


def get_order_state(order_id: str) -> dict | None:
    """Latest streamed state for an order: {status, filled_qty, filled_avg_price}."""
    with _lock:
        return _states.get(str(order_id))


def _record(order_id: str, status: str, filled_qty, filled_avg_price):
    with _lock:
        _states[str(order_id)] = {
            'status': status,
            'filled_qty': filled_qty,
            'filled_avg_price': filled_avg_price,
            'ts': time.monotonic(),
        }
        if len(_states) > _MAX_ENTRIES:
            # Drop the oldest half
            by_age = sorted(_states.items(), key=lambda kv: kv[1]['ts'])
            for k, _ in by_age[:_MAX_ENTRIES // 2]:
                _states.pop(k, None)


def start_order_stream() -> bool:
    """Start the trade_updates stream thread. Idempotent. Returns success.

    No-ops (returns False) unless TRADER_ORDER_STREAM=1 and alpaca-py is
    importable — the system is fully functional on polling alone.
    """
    global _started
    if _started:
        return True
    if os.environ.get('TRADER_ORDER_STREAM') != '1':
        return False
    try:
        from alpaca.trading.stream import TradingStream
    except ImportError:
        logger.info("[STREAM] alpaca-py not installed — order stream disabled")
        return False

    key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_API_SECRET')
    if not key or not secret:
        return False
    paper = 'paper' in (os.getenv('ALPACA_BASE_URL') or 'paper')

    async def _on_update(data):
        try:
            order = data.order
            status = getattr(getattr(order, 'status', None), 'value',
                             str(getattr(order, 'status', '')))
            _record(str(order.id), status,
                    float(order.filled_qty or 0),
                    float(order.filled_avg_price) if order.filled_avg_price else None)
        except Exception:
            pass

    def _run():
        # TradingStream.run() reconnects internally; if it ever returns or
        # raises, restart with backoff. Missed events are harmless — the
        # polling path remains authoritative.
        backoff = 5
        while True:
            try:
                stream = TradingStream(key, secret, paper=paper)
                stream.subscribe_trade_updates(_on_update)
                logger.info("[STREAM] trade_updates stream connected")
                stream.run()
            except Exception as e:
                logger.warning("[STREAM] order stream died (%s) — retry in %ds", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)

    t = threading.Thread(target=_run, name='order-stream', daemon=True)
    t.start()
    _started = True
    return True
