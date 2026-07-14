"""TradingStream order-update cache (optional, alpaca-py only).

`manage_order_lifecycle` polls GET /orders every 2 seconds. With the
trade_updates websocket, fills/cancels/rejections arrive as push events —
one shared connection per account replaces thousands of daily polls with
fast terminal-state detection during the entry-order lifecycle window
(order_utils.manage_order_lifecycle is the ONLY consumer today).
Server-side bracket/trailing-stop fill events are recorded into the cache
too, but nothing reads them yet — wiring mid-cycle position reconciliation
to those entries is future work.

Design: a background thread runs alpaca-py's TradingStream and writes the
latest known state per order_id into a bounded, lock-guarded dict. The
polling path CONSULTS the cache to decide when to make its single
authoritative REST fetch; correctness never depends on the stream (missed
events just mean we fall back to polling cadence).

Enable with TRADER_ORDER_STREAM=1 (requires alpaca-py installed).
NOTE: Alpaca allows ONE trade_updates connection per account, so only
enable the flag in combined-bots (single-process) mode — two split-mode
processes would perpetually disconnect each other's streams.
"""

import os
import threading
import time

from log_config import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_states: dict[str, dict] = {}
_MAX_ENTRIES = 500
_start_lock = threading.Lock()
_started = False

TERMINAL = {'filled', 'canceled', 'expired', 'rejected'}

# Reconnect backoff: damp rapid failure loops without permanently
# ratcheting a long-lived process toward the cap.
_BACKOFF_INITIAL = 5
_BACKOFF_CAP = 300
_BACKOFF_HEALTHY_SECS = 60


def _next_backoff(backoff: float, connected_secs: float) -> tuple[float, float]:
    """(sleep_now_s, next_backoff_s) after a stream exit.

    A connection that stayed up longer than _BACKOFF_HEALTHY_SECS was
    healthy, so the ladder resets to the initial rung — over weeks of
    uptime each disconnect must not permanently widen the reconnect gap.
    Rapid failures keep the current rung and double the next one, capped.
    """
    if connected_secs > _BACKOFF_HEALTHY_SECS:
        backoff = _BACKOFF_INITIAL
    return backoff, min(backoff * 2, _BACKOFF_CAP)


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

    Alpaca permits ONE trade_updates connection per account: in combined-bots
    mode both loops call this and share the single thread (the whole body is
    serialized under _start_lock so simultaneous callers can't spawn duplicate
    streams); in split-process mode the two processes would fight over the
    connection, so leave the flag unset there.
    """
    global _started
    with _start_lock:
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
            logger.warning("[STREAM] TRADER_ORDER_STREAM=1 but ALPACA_API_KEY/"
                           "SECRET missing — stream disabled")
            return False
        base_url = os.getenv('ALPACA_BASE_URL')
        if not base_url:
            # Fail closed rather than guess: alpaca_compat's trading client
            # resolves an unset base URL differently, and a stream listening
            # on the wrong account silently never sees any order event.
            logger.warning("[STREAM] ALPACA_BASE_URL unset — cannot determine "
                           "paper/live, stream disabled")
            return False
        paper = 'paper' in base_url

        parse_warned = False

        async def _on_update(data):
            nonlocal parse_warned
            try:
                order = data.order
                status = getattr(getattr(order, 'status', None), 'value',
                                 str(getattr(order, 'status', '')))
                _record(str(order.id), status,
                        float(order.filled_qty or 0),
                        float(order.filled_avg_price) if order.filled_avg_price else None)
            except Exception as e:
                # Never raise (the stream must stay non-fatal), but never go
                # mute: a schema change would otherwise leave the cache empty
                # forever while the log claims the stream is healthy.
                if not parse_warned:
                    parse_warned = True
                    logger.warning("[STREAM] failed to parse trade update: %s", e)
                else:
                    logger.debug("[STREAM] failed to parse trade update: %s", e)

        def _run():
            # TradingStream.run() reconnects internally; if it ever returns or
            # raises, restart with backoff. Missed events are harmless — the
            # polling path remains authoritative.
            backoff = _BACKOFF_INITIAL
            while True:
                t0 = time.monotonic()
                err = None
                try:
                    stream = TradingStream(key, secret, paper=paper)
                    stream.subscribe_trade_updates(_on_update)
                    # run() is what actually connects/authenticates — don't
                    # claim 'connected' before it happens.
                    logger.info("[STREAM] starting trade_updates stream (paper=%s)",
                                paper)
                    stream.run()
                except Exception as e:
                    err = e
                sleep_s, backoff = _next_backoff(backoff, time.monotonic() - t0)
                if err is None:
                    logger.info("[STREAM] stream ended — reconnecting in %ds", sleep_s)
                else:
                    logger.warning("[STREAM] order stream died (%s) — retry in %ds",
                                   err, sleep_s)
                time.sleep(sleep_s)

        t = threading.Thread(target=_run, name='order-stream', daemon=True)
        t.start()
        _started = True
        return True
