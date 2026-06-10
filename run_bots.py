"""Combined bot runner — crypto + stock loops as threads in ONE process.

On the Jetson Orin Nano's 8GB, each bot process pays the full
torch+pandas+numba+sklearn import stack (~0.5-0.8GB RSS). Both loops are
I/O-bound (REST calls + 30s sleeps), so running them as threads in a single
process saves a duplicate interpreter stack with no latency cost. Module
state was audited for thread-safety: sentiment_history uses thread-local
SQLite connections (WAL), market_data's bar cache is lock-guarded, and the
correlation cache is keyed per asset class.

Usage:
    python run_bots.py                 # both loops (default)
    python run_bots.py --crypto-only
    python run_bots.py --stock-only

run_pipeline.py uses this as the default bot launch mode; pass
--separate-bots there to keep the old one-process-per-bot layout.
"""
import argparse
import signal
import sys
import threading
import time

from log_config import get_logger

logger = get_logger(__name__)

_shutdown = threading.Event()


def _run_loop(loop_cls, name):
    """Run one trading loop; log fatal errors instead of dying silently."""
    try:
        loop = loop_cls()
        loop.run()
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("[%s] Trading loop crashed", name)
    finally:
        logger.info("[%s] Loop thread exiting", name)
        _shutdown.set()  # one loop dying should surface, not hide


def main():
    ap = argparse.ArgumentParser(description='Run trading bots in one process')
    ap.add_argument('--crypto-only', action='store_true')
    ap.add_argument('--stock-only', action='store_true')
    args = ap.parse_args()

    run_crypto = not args.stock_only
    run_stock = not args.crypto_only

    threads = []
    if run_crypto:
        from crypto_loop import CryptoLoop
        t = threading.Thread(target=_run_loop, args=(CryptoLoop, 'crypto'),
                             name='crypto-loop', daemon=True)
        threads.append(t)
    if run_stock:
        from stock_loop import StockLoop
        t = threading.Thread(target=_run_loop, args=(StockLoop, 'stock'),
                             name='stock-loop', daemon=True)
        threads.append(t)

    if not threads:
        print("Nothing to run")
        return 1

    def _sigterm(_sig, _frm):
        logger.info("[BOTS] SIGTERM received, shutting down")
        _shutdown.set()

    signal.signal(signal.SIGTERM, _sigterm)

    for t in threads:
        t.start()
        time.sleep(5)  # stagger startup: model loads + order cleanup

    logger.info("[BOTS] %d loop(s) running in one process", len(threads))
    try:
        while not _shutdown.is_set():
            time.sleep(5)
            if not any(t.is_alive() for t in threads):
                break
    except KeyboardInterrupt:
        logger.info("[BOTS] Interrupted")
    # Daemon threads die with the process; positions are protected by
    # server-side stops (stocks) and reconstructed on next start.
    return 0


if __name__ == '__main__':
    sys.exit(main())
