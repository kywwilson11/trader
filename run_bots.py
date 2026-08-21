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

Ops thread (c26 T7 / B19): a standalone `python run_bots.py` also starts a
daemon ops thread that polls the Telegram kill switch (/halt /resume
/flatten /status), runs the once-daily PSI drift check, and runs the
once-daily journal-retention pass. Command/drift ops DEFER whenever a live
run_pipeline heartbeat is fresh (<120s) — the pipeline's wait loops already
own them there and two Telegram pollers would race the shared offset file;
rotation runs regardless (a no-op unless TRADER_JOURNAL_ROTATE_DAYS>0).
Disable the whole thread with TRADER_BOTS_OPS=0.
"""
import argparse
import datetime
import os
import signal
import sys
import threading
import time

# Bots are CPU-only BY DESIGN (a CUDA context costs 0.6-1.2GB of unified
# memory on the Jetson; trading_utils.choose_inference_device hard-returns
# 'cpu'). run_pipeline launches bots with these already set (BOT_ENV), but a
# manual `python run_bots.py` used to initialize the CUDA driver and spawn
# default 6-thread torch intra-op pools x 5 prediction workers. setdefault
# preserves any explicit operator override.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TORCH_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

from log_config import get_logger

logger = get_logger(__name__)

_shutdown = threading.Event()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ops thread (c26 T7 / B19): standalone `python run_bots.py` previously had NO
# Telegram kill-switch polling and NO daily drift check (they live in
# run_pipeline's wait loops). Fail-open instrumentation: default ON, but it
# DEFERS whenever a live pipeline heartbeat is fresh — two pollers would race
# notify's shared Telegram offset file and double-run the drift check.
_OPS_ENABLED = os.environ.get('TRADER_BOTS_OPS', '1').strip().lower() not in ('0', 'false', 'no')
_OPS_POLL_SEC = 60
_PIPELINE_STATUS = os.path.join(BASE_DIR, 'pipeline_status.json')
_PIPELINE_FRESH_SEC = 120     # heartbeat re-writes every ~30s (run_pipeline)
_ops_drift_date = None
_ops_rotate_date = None


def _pipeline_alive() -> bool:
    try:
        return (time.time() - os.path.getmtime(_PIPELINE_STATUS)) < _PIPELINE_FRESH_SEC
    except OSError:
        return False


def _ops_handle_commands(threads):
    """Telegram kill switch in standalone mode — mirrors
    run_pipeline._check_telegram_commands semantics (same notify calls,
    texts, and dedupe keys; logger instead of the pipeline log)."""
    from notify import (poll_telegram_commands, set_halt, clear_halt,
                        halt_active, request_flatten, notify)
    for cmd in poll_telegram_commands():
        logger.info("[OPS] telegram command: %s", cmd)
        if cmd == '/halt':
            set_halt('telegram /halt')
            notify("HALT engaged — no new entries until /resume. "
                   "Open positions keep their stops/exits.",
                   level='critical', dedupe_key='tg-halt')
        elif cmd == '/resume':
            clear_halt()
            notify("Halt cleared — entries re-enabled.",
                   level='warning', dedupe_key='tg-resume')
        elif cmd == '/flatten':
            request_flatten('telegram /flatten')
            notify("Flatten requested — each bot liquidates its book "
                   "within one cycle (~30s) and halts.",
                   level='critical', dedupe_key='tg-flatten')
        elif cmd == '/status':
            liveness = ', '.join(f'{t.name}={t.is_alive()}' for t in threads)
            notify(f"standalone bots: {liveness} halted={halt_active()}",
                   level='info', dedupe_key=f'tg-status-{time.time():.0f}')


def _ops_daily_drift(threads):
    """Once-daily PSI drift check (mirrors run_pipeline._maybe_run_drift_check's
    monitor_drift half). Deliberately does NOT call
    shadow.evaluate_and_maybe_promote — promotion decisions stay
    pipeline-owned; standalone run_bots has no retrain machinery."""
    global _ops_drift_date
    today = datetime.date.today().isoformat()
    if _ops_drift_date == today:
        return
    _ops_drift_date = today
    from monitor_drift import run_check
    for prefix, label in (('', 'crypto'), ('stock', 'stock')):
        r = run_check(prefix, label)
        if r is not None:
            logger.info("[OPS] drift %s: PSI=%s (%s, n=%s)",
                        label, r['psi'], r['level'], r['n'])


def _ops_daily_rotation():
    """Once-daily journal retention pass. UNCONDITIONAL of pipeline liveness:
    the pipeline never rotates, and this bot process is the journal writer.
    No-op unless TRADER_JOURNAL_ROTATE_DAYS>0."""
    global _ops_rotate_date
    today = datetime.date.today().isoformat()
    if _ops_rotate_date == today:
        return
    _ops_rotate_date = today
    from trade_journal import rotate_old_journals
    n = rotate_old_journals()
    if n:
        logger.info("[OPS] rotated %d journal file(s) to .gz", n)


def _ops_cycle(threads):
    """One ops pass, fully isolated — a crash here must NEVER touch the
    trading loops."""
    try:
        if not _pipeline_alive():
            _ops_handle_commands(threads)
            _ops_daily_drift(threads)
    except Exception:
        logger.exception("[OPS] standalone ops failed")
    try:
        _ops_daily_rotation()
    except Exception:
        logger.exception("[OPS] journal rotation failed")


def _ops_loop(threads):
    logger.info("[OPS] ops thread running (poll=%ds, defers to live pipeline)",
                _OPS_POLL_SEC)
    while not _shutdown.is_set():
        _ops_cycle(threads)
        _shutdown.wait(_OPS_POLL_SEC)


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

    if _OPS_ENABLED:
        threading.Thread(target=_ops_loop, args=(threads,),
                         name='ops', daemon=True).start()

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
