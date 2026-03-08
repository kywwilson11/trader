#!/usr/bin/env python3
"""Overnight pipeline: train models, start trading, auto-retrain weekly.

Flow:
  1. Initial training (harvest + hypersearch for all models)
  2. Start trading bots (crypto 24/7 + stock during market hours)
  3. Bots run continuously — they hot-reload models when .pth files change
  4. Every Saturday 2 AM: re-harvest data + retrain models in background
     (bots keep trading with current models, swap to new ones automatically)

Writes status to pipeline_status.json for GUI monitoring.
All output logged to pipeline_output.log.

Usage:
    python run_pipeline.py                  # Full pipeline with weekly retrain
    python run_pipeline.py --no-retrain     # One-shot: train once, run bots forever
    python run_pipeline.py --skip-harvest   # Skip data harvest (use existing CSVs)
    python run_pipeline.py --trials 50      # Fewer trials for first run
    python run_pipeline.py --retrain-trials 30  # Fewer trials for weekly retrain
    python run_pipeline.py --bot-only       # Skip training, jump to bots
    python run_pipeline.py --crypto-only    # Crypto only (no stock models/bot)
    python run_pipeline.py --stock-only     # Stock only (no crypto models/bot)
"""

import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

from adaptive_config import (load_adaptive_state, decide_mode, get_trial_count,
                              get_max_forward_bars)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(BASE_DIR, 'pipeline_status.json')
LOG_FILE = os.path.join(BASE_DIR, 'pipeline_output.log')
CRYPTO_BOT_LOG = os.path.join(BASE_DIR, 'crypto_bot_output.log')
STOCK_BOT_LOG = os.path.join(BASE_DIR, 'stock_bot_output.log')
PYTHON = '/home/kyle/miniforge3/envs/jetson/bin/python'
RETRAIN_TRIGGER = os.path.join(BASE_DIR, 'retrain_trigger.json')
# Skip stdout writes when redirected to same log file (avoids doubled lines)
_STDOUT_IS_TTY = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def _print(*args, **kwargs):
    """Print only when stdout is a terminal (avoids doubling when redirected to log)."""
    if _STDOUT_IS_TTY:
        print(*args, **kwargs)


def _check_retrain_trigger():
    """Check for and consume a manual retrain trigger file from the GUI.

    Returns dict {'crypto': bool, 'stock': bool} if trigger found, else None.
    """
    try:
        if os.path.exists(RETRAIN_TRIGGER):
            with open(RETRAIN_TRIGGER) as f:
                trigger = json.load(f)
            os.remove(RETRAIN_TRIGGER)
            if trigger.get('crypto') or trigger.get('stock'):
                return trigger
    except (OSError, json.JSONDecodeError, KeyError):
        # Remove malformed trigger file
        try:
            os.remove(RETRAIN_TRIGGER)
        except OSError:
            pass
    return None


ENV = {
    **os.environ,
    'LD_LIBRARY_PATH': (
        '/home/kyle/miniforge3/envs/jetson/lib:'
        '/home/kyle/miniforge3/envs/jetson/lib/python3.10/site-packages/'
        'nvidia/cusparselt/lib:'
        + os.environ.get('LD_LIBRARY_PATH', '')
    ),
    'LD_PRELOAD': '/home/kyle/miniforge3/envs/jetson/lib/libstdc++.so.6',
    'PYTHONUNBUFFERED': '1',
}

# Throttle JSON writes to avoid excessive disk I/O
_last_status_write = 0
STATUS_WRITE_INTERVAL = 2.0  # seconds

DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                'Friday', 'Saturday', 'Sunday']

# Heartbeat: re-write status every 30s so GUI knows pipeline is alive
_heartbeat_status = None  # Reference to current status dict
_heartbeat_stop = threading.Event()
_heartbeat_lock = threading.Lock()


def _heartbeat_loop():
    """Background thread: re-write status file every 30s."""
    while not _heartbeat_stop.wait(30):
        with _heartbeat_lock:
            if _heartbeat_status is not None:
                write_status(_heartbeat_status, force=True)


def write_status(status, force=False):
    """Write pipeline status to JSON, throttled to every 2 seconds.

    Thread-safe: callers should hold _heartbeat_lock when mutating status
    from background threads.
    """
    global _last_status_write
    now = time.time()
    if not force and (now - _last_status_write) < STATUS_WRITE_INTERVAL:
        return
    _last_status_write = now
    status['updated_at'] = datetime.datetime.now().isoformat()
    elapsed = 0
    try:
        started = datetime.datetime.fromisoformat(status.get('started_at', ''))
        elapsed = (datetime.datetime.now() - started).total_seconds()
    except (ValueError, TypeError):
        pass
    status['elapsed_sec'] = int(elapsed)
    tmp = STATUS_FILE + f'.tmp.{os.getpid()}'
    try:
        with open(tmp, 'w') as f:
            json.dump(status, f, indent=2)
        os.replace(tmp, STATUS_FILE)
    except OSError:
        pass  # Non-fatal — status file is informational only


def run_phase(phase, log_fh, status):
    """Run a single pipeline phase as a subprocess, parsing output in real-time."""
    phase_idx = phase['idx']
    phase_id = phase['id']

    status['phase'] = phase_id
    status['phase_label'] = phase['label']
    status['phase_idx'] = phase_idx
    status['phase_started_at'] = datetime.datetime.now().isoformat()
    status['trial_current'] = 0
    status['trial_prior'] = 0
    status['best_score'] = status.get('best_score', 0.0) if 'search' not in phase_id else 0.0

    if 'search' in phase_id:
        status['trial_total'] = phase.get('trials', 250)
    else:
        status['trial_total'] = 0

    write_status(status, force=True)

    header = (
        f"\n{'='*70}\n"
        f"PHASE {phase_idx + 1}/{status['total_phases']}: {phase['label']}\n"
        f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n"
        f"{'='*70}\n\n"
    )
    log_fh.write(header)
    log_fh.flush()
    _print(header, end='')

    proc = subprocess.Popen(
        phase['cmd'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=ENV,
        cwd=BASE_DIR,
        bufsize=1,
        text=True,
    )

    try:
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            if _STDOUT_IS_TTY:
                sys.stdout.write(line)
                sys.stdout.flush()

            # Parse prior trials: "Resuming from 119 prior trials in bear_study.db"
            m = re.match(r'Resuming from (\d+) prior trials', line)
            if m:
                status['trial_prior'] = int(m.group(1))
                write_status(status, force=True)

            # Parse prior best: "Prior best score=0.396 ..."
            m = re.match(r'Prior best (?:sharpe|score)=(-?\d+\.\d+)', line)
            if m:
                status['best_score'] = float(m.group(1))
                write_status(status, force=True)

            # Parse trial progress: "[  45] score=0.543 ..."
            force = False
            m = re.match(r'\[\s*(\d+)\]', line)
            if m:
                absolute = int(m.group(1))
                status['trial_current'] = absolute - status.get('trial_prior', 0)
                force = True

            # Parse best score on "** BEST **" lines
            if '** BEST **' in line:
                m = re.search(r'score=(-?\d+\.\d+)', line)
                if m:
                    status['best_score'] = float(m.group(1))
                force = True

            write_status(status, force=force)
    except Exception as e:
        _print(f"\n[PIPELINE] Error reading phase output: {e}")
    finally:
        proc.wait()
    status['phase_exit_code'] = proc.returncode

    elapsed = ''
    try:
        started = datetime.datetime.fromisoformat(status['phase_started_at'])
        secs = (datetime.datetime.now() - started).total_seconds()
        elapsed = f" ({secs/60:.1f} min)"
    except (ValueError, TypeError):
        pass

    footer = f"\n--- Phase complete (exit {proc.returncode}){elapsed} ---\n"
    log_fh.write(footer)
    log_fh.flush()
    _print(footer, end='')

    write_status(status, force=True)
    return proc.returncode


# ---------------------------------------------------------------------------
# Bot management helpers
# ---------------------------------------------------------------------------

def _start_bot(cmd, log_path):
    """Start a trading bot as a background process.

    Returns (proc, file_handle) so the caller can close the log FH when done.
    """
    fh = open(log_path, 'a')
    proc = subprocess.Popen(
        cmd, stdout=fh, stderr=subprocess.STDOUT,
        env=ENV, cwd=BASE_DIR,
    )
    return proc, fh


def _check_restart_bots(bots, log_fh):
    """Check for crashed bots and restart them."""
    for i, (name, proc, bot_fh) in enumerate(bots):
        if proc.poll() is not None:
            # Close the old log file handle before opening a new one
            try:
                bot_fh.close()
            except Exception:
                pass
            log_path = CRYPTO_BOT_LOG if name == 'Crypto' else STOCK_BOT_LOG
            cmd = [PYTHON, '-u',
                   'crypto_loop.py' if name == 'Crypto' else 'stock_loop.py']
            new_proc, new_fh = _start_bot(cmd, log_path)
            _all_handles.append(new_fh)
            bots[i] = (name, new_proc, new_fh)
            msg = (f"{name} bot crashed (exit {proc.returncode}),"
                   f" restarted as PID {new_proc.pid}\n")
            log_fh.write(msg)
            log_fh.flush()
            _print(msg, end='')


def _next_retrain_time(retrain_day, retrain_hour):
    """Compute the next retrain datetime (upcoming weekday + hour)."""
    now = datetime.datetime.now()
    days_ahead = retrain_day - now.weekday()
    if days_ahead < 0 or (days_ahead == 0 and now.hour >= retrain_hour):
        days_ahead += 7
    target = now.replace(hour=retrain_hour, minute=0, second=0, microsecond=0)
    target += datetime.timedelta(days=days_ahead)
    return target


# ---------------------------------------------------------------------------
# Phase list builders
# ---------------------------------------------------------------------------

def _get_data_age_hours(prefix):
    """Get age in hours of the newest data file (Parquet or CSV)."""
    stems = {'crypto': 'training_data', 'stock': 'stock_training_data'}
    stem = stems.get(prefix, prefix)
    best_mtime = 0
    for ext in ('.parquet', '.csv'):
        path = os.path.join(BASE_DIR, f'{stem}{ext}')
        if os.path.exists(path):
            best_mtime = max(best_mtime, os.path.getmtime(path))
    if best_mtime == 0:
        return None  # no data file exists
    return (time.time() - best_mtime) / 3600


def _build_harvest_phases(skip_harvest, train_crypto, train_stock):
    """Build harvest phases, skipping if data is fresh."""
    phases = []
    if skip_harvest:
        return phases

    if train_crypto:
        age_h = _get_data_age_hours('crypto')
        if age_h is not None and age_h < 24:
            _print(f"Crypto training data is {age_h:.1f}h old, skipping harvest")
        else:
            phases.append({
                'id': 'crypto_harvest',
                'label': 'Harvesting Crypto Data',
                'cmd': [PYTHON, '-u', os.path.join('scripts', 'harvest_crypto_data.py')],
            })

    if train_stock:
        age_h = _get_data_age_hours('stock')
        if age_h is not None and age_h < 24:
            _print(f"Stock training data is {age_h:.1f}h old, skipping harvest")
        else:
            phases.append({
                'id': 'stock_harvest',
                'label': 'Harvesting Stock Data',
                'cmd': [PYTHON, '-u', os.path.join('scripts', 'harvest_stock_data.py')],
            })

    return phases


def _build_training_phases(trials, train_crypto, train_stock, mode=''):
    """Build model training phases with adaptive mode support."""
    phases = []

    if train_crypto:
        cmd = [PYTHON, '-u', os.path.join('scripts', 'hypersearch_v2.py'),
               '--trials', str(trials), '--preset', 'stationary', '--no-status']
        if mode:
            cmd += ['--mode', mode]
        phases.append({
            'id': 'crypto_search',
            'label': 'Training Crypto Regression Model',
            'cmd': cmd,
            'trials': trials,
        })
    if train_stock:
        cmd = [PYTHON, '-u', os.path.join('scripts', 'hypersearch_v2.py'),
               '--trials', str(trials),
               '--data', 'stock_training_data.csv', '--prefix', 'stock',
               '--preset', 'stationary', '--max-rows', '200000', '--no-status']
        if mode:
            cmd += ['--mode', mode]
        phases.append({
            'id': 'stock_search',
            'label': 'Training Stock Regression Model',
            'cmd': cmd,
            'trials': trials,
        })

    return phases


MAX_PHASE_RETRIES = 3
RETRY_WAIT_SECONDS = 30


def _run_training(phases, log_fh, status, is_retrain):
    """Run all training phases. Returns True if all succeeded."""
    for phase in phases:
        for attempt in range(1, MAX_PHASE_RETRIES + 1):
            rc = run_phase(phase, log_fh, status)
            if rc == 0:
                break
            # Failed — retry with fresh CUDA context (Optuna DB preserves progress)
            if attempt < MAX_PHASE_RETRIES:
                msg = (f"\n[RETRY] {phase['label']} failed (exit {rc}), "
                       f"attempt {attempt}/{MAX_PHASE_RETRIES}. "
                       f"Waiting {RETRY_WAIT_SECONDS}s for memory to clear...\n")
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')
                time.sleep(RETRY_WAIT_SECONDS)
            else:
                msg = (f"\n[FAILED] {phase['label']} failed after "
                       f"{MAX_PHASE_RETRIES} attempts\n")
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')

        # Save final scores
        if phase['id'] == 'crypto_search':
            status['crypto_final_score'] = status.get('best_score', 0)
        elif phase['id'] == 'stock_search':
            status['stock_final_score'] = status.get('best_score', 0)

        if rc != 0:
            if is_retrain:
                msg = (f"\nWARNING: {phase['label']} failed (exit {rc}),"
                       f" bots continue with existing models\n")
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')
                return False
            else:
                msg = f"\nPIPELINE STOPPED: {phase['label']} failed (exit {rc})\n"
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')
                status['phase'] = 'failed'
                status['phase_label'] = f"Failed: {phase['label']}"
                write_status(status, force=True)
                sys.exit(1)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _cleanup_handles(handles):
    """Close all tracked file handles safely."""
    for fh in handles:
        try:
            if fh and not fh.closed:
                fh.close()
        except Exception:
            pass


def _terminate_procs(procs):
    """Terminate all tracked subprocesses."""
    for proc in procs:
        try:
            if proc and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


# Global refs for signal handler cleanup
_all_handles = []
_all_procs = []
_all_bots = []
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Graceful shutdown: stop bots, close handles, exit."""
    global _shutdown_requested
    if _shutdown_requested:
        return  # Prevent re-entry
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name
    _print(f"\n[PIPELINE] {sig_name} received, shutting down...")
    # Stop bot processes
    for name, proc, fh in _all_bots:
        try:
            if proc.poll() is None:
                proc.terminate()
                _print(f"  Stopped {name} bot (PID {proc.pid})")
        except Exception:
            pass
    _terminate_procs(_all_procs)
    _cleanup_handles(_all_handles)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Trading pipeline orchestrator')
    parser.add_argument('--trials', type=int, default=200,
                        help='Trials per model on first run (default: 200)')
    parser.add_argument('--bot-only', action='store_true',
                        help='Skip training, start bots immediately')
    parser.add_argument('--skip-harvest', action='store_true',
                        help='Skip data harvest (use existing CSVs)')
    parser.add_argument('--crypto-only', action='store_true',
                        help='Train and run crypto only (no stock models)')
    parser.add_argument('--stock-only', action='store_true',
                        help='Train and run stocks only (no crypto models)')
    parser.add_argument('--no-retrain', action='store_true',
                        help='Disable weekly retrain (one-shot mode)')
    parser.add_argument('--retrain-day', type=int, default=5,
                        help='Day of week to retrain (0=Mon, 5=Sat, default: 5)')
    parser.add_argument('--retrain-hour', type=int, default=2,
                        help='Hour to start retrain (0-23, default: 2)')
    parser.add_argument('--retrain-trials', type=int, default=100,
                        help='Trials per model for weekly retrain (default: 100)')
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)

    train_crypto = not args.stock_only
    train_stock = not args.crypto_only
    run_crypto = not args.stock_only
    run_stock = not args.crypto_only

    # Preserve final scores from previous runs (e.g. crypto score when restarting stock-only)
    prev_status = {}
    try:
        with open(STATUS_FILE) as f:
            prev_status = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass

    status = {
        'started_at': datetime.datetime.now().isoformat(),
        'phase': 'starting',
        'phase_label': 'Starting Pipeline...',
        'phase_idx': -1,
        'total_phases': 0,
        'trial_current': 0,
        'trial_total': 0,
        'best_score': 0.0,
        'crypto_final_score': prev_status.get('crypto_final_score'),
        'stock_final_score': prev_status.get('stock_final_score'),
        'retrain_cycle': 0,
        'bots_running': False,
    }

    log_fh = open(LOG_FILE, 'a')
    _all_handles.append(log_fh)

    global _heartbeat_status
    _heartbeat_status = status
    hb = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb.start()

    try:
        # =============================================================
        # PHASE A: Initial training (cycle 0)
        # =============================================================
        if not args.bot_only:
            phases = (_build_harvest_phases(args.skip_harvest, train_crypto, train_stock)
                      + _build_training_phases(args.trials, train_crypto, train_stock,
                                               mode='initial'))
            for i, p in enumerate(phases):
                p['idx'] = i

            status['phases'] = [p['id'] for p in phases]
            status['phase_labels'] = {p['id']: p['label'] for p in phases}
            status['total_phases'] = len(phases)
            write_status(status, force=True)

            banner = (
                f"\n{'#'*70}\n"
                f"# PIPELINE STARTED: "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n"
                f"# Phases: {', '.join(p['label'] for p in phases)}\n"
                f"# Trials per model: {args.trials}\n"
                f"{'#'*70}\n"
            )
            log_fh.write(banner)
            log_fh.flush()
            _print(banner, end='')

            # Start background sentiment fetch (runs during training)
            sentiment_proc = None
            sentiment_fh = None
            try:
                sentiment_fh = open(os.path.join(BASE_DIR, 'sentiment_fetch.log'), 'a')
                _all_handles.append(sentiment_fh)
                sentiment_proc = subprocess.Popen(
                    [PYTHON, '-u', 'sentiment_history.py', '--fetch-stocks'],
                    stdout=sentiment_fh,
                    stderr=subprocess.STDOUT,
                    env=ENV, cwd=BASE_DIR,
                )
                _all_procs.append(sentiment_proc)
                msg = f"Background sentiment fetch started (PID {sentiment_proc.pid})\n"
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')
            except Exception as e:
                msg = f"Sentiment fetch failed to start: {e}\n"
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')

            _run_training(phases, log_fh, status, is_retrain=False)

        # =============================================================
        # PHASE B: Start trading bots (run forever)
        # =============================================================
        bots = _all_bots

        if run_crypto:
            proc, bot_fh = _start_bot([PYTHON, '-u', 'crypto_loop.py'], CRYPTO_BOT_LOG)
            bots.append(('Crypto', proc, bot_fh))
            _all_handles.append(bot_fh)
            msg = f"Crypto bot started (PID {proc.pid}, log: crypto_bot_output.log)\n"
            log_fh.write(msg)
            log_fh.flush()
            _print(msg, end='')

        if run_stock:
            proc, bot_fh = _start_bot([PYTHON, '-u', 'stock_loop.py'], STOCK_BOT_LOG)
            bots.append(('Stock', proc, bot_fh))
            _all_handles.append(bot_fh)
            msg = f"Stock bot started (PID {proc.pid}, log: stock_bot_output.log)\n"
            log_fh.write(msg)
            log_fh.flush()
            _print(msg, end='')

        status['phase'] = 'trading'
        status['phase_label'] = 'Trading'
        status['bots_running'] = True
        write_status(status, force=True)

        # Start sentiment backfill worker (LLM-scores historical articles)
        backfill_proc = None
        backfill_fh = None
        try:
            from sentiment_history import set_live_mode
            set_live_mode(True)
            backfill_fh = open(os.path.join(BASE_DIR, 'backfill_output.log'), 'a')
            _all_handles.append(backfill_fh)
            backfill_proc = subprocess.Popen(
                [PYTHON, '-u', 'sentiment_history.py', '--backfill'],
                stdout=backfill_fh,
                stderr=subprocess.STDOUT,
                env=ENV, cwd=BASE_DIR,
            )
            _all_procs.append(backfill_proc)
            msg = f"Backfill worker started (PID {backfill_proc.pid})\n"
            log_fh.write(msg)
            log_fh.flush()
            _print(msg, end='')
        except Exception as e:
            msg = f"Backfill worker failed to start: {e}\n"
            log_fh.write(msg)
            log_fh.flush()
            _print(msg, end='')

        # --- No retrain: wait forever (but still accept manual triggers) ---
        if args.no_retrain:
            msg = "Scheduled retrain disabled, bots running. Manual retrain still available.\n"
            log_fh.write(msg)
            log_fh.flush()
            _print(msg, end='')
            status['phase'] = 'trading'
            status['phase_label'] = 'Trading (no auto-retrain)'
            write_status(status, force=True)
            manual_cycle = 0
            while not _shutdown_requested:
                time.sleep(60)
                _check_restart_bots(bots, log_fh)
                trigger = _check_retrain_trigger()
                if trigger:
                    manual_cycle += 1
                    rt_crypto = trigger.get('crypto', False)
                    rt_stock = trigger.get('stock', False)
                    msg = (f"\n[MANUAL RETRAIN] Triggered from GUI: "
                           f"crypto={rt_crypto}, stock={rt_stock}\n")
                    log_fh.write(msg)
                    log_fh.flush()
                    _print(msg, end='')
                    try:
                        from sentiment_history import set_live_mode
                        set_live_mode(False)
                    except Exception:
                        pass
                    # Build and run retrain phases
                    adaptive_modes = {}
                    adaptive_trial_counts = {}
                    for at in (['crypto'] if rt_crypto else []) + (['stock'] if rt_stock else []):
                        astate = load_adaptive_state(at)
                        amode = decide_mode(astate, astate.get('best_score', 0))
                        atrial = get_trial_count(amode)
                        adaptive_modes[at] = amode
                        adaptive_trial_counts[at] = atrial
                    retrain_trials = max(adaptive_trial_counts.values()) if adaptive_trial_counts else 100
                    retrain_mode = 'explore' if 'explore' in adaptive_modes.values() else 'refine'
                    retrain_phases = (
                        _build_harvest_phases(False, rt_crypto, rt_stock)
                        + _build_training_phases(retrain_trials, rt_crypto, rt_stock,
                                                 mode=retrain_mode)
                    )
                    for i, p in enumerate(retrain_phases):
                        p['idx'] = i
                    status['started_at'] = datetime.datetime.now().isoformat()
                    status['phases'] = [p['id'] for p in retrain_phases]
                    status['phase_labels'] = {p['id']: p['label'] for p in retrain_phases}
                    status['total_phases'] = len(retrain_phases)
                    status['retrain_cycle'] = manual_cycle
                    _run_training(retrain_phases, log_fh, status, is_retrain=True)
                    try:
                        from sentiment_history import set_live_mode
                        set_live_mode(True)
                    except Exception:
                        pass
                    status['phase'] = 'trading'
                    status['phase_label'] = 'Trading (no auto-retrain)'
                    write_status(status, force=True)
            return

        # =============================================================
        # PHASE C: Weekly retrain loop (bots keep running)
        # =============================================================
        cycle = 0
        while not _shutdown_requested:
            cycle += 1
            next_retrain = _next_retrain_time(args.retrain_day, args.retrain_hour)
            status['next_retrain'] = next_retrain.isoformat()
            status['phase'] = 'trading'
            status['phase_label'] = 'Trading'
            status['retrain_cycle'] = cycle
            write_status(status, force=True)

            msg = (f"\nBots running. Next retrain: "
                   f"{DAYS_OF_WEEK[next_retrain.weekday()]} "
                   f"{next_retrain.strftime('%Y-%m-%d %I:%M %p')}\n")
            log_fh.write(msg)
            log_fh.flush()
            _print(msg, end='')

            # Wait until retrain time, auto-restarting crashed bots
            # Also check for manual retrain trigger from GUI
            manual_trigger = None
            while datetime.datetime.now() < next_retrain and not _shutdown_requested:
                time.sleep(60)
                _check_restart_bots(bots, log_fh)
                write_status(status)
                manual_trigger = _check_retrain_trigger()
                if manual_trigger:
                    msg = (f"\n[MANUAL RETRAIN] Triggered from GUI: "
                           f"crypto={manual_trigger.get('crypto', False)}, "
                           f"stock={manual_trigger.get('stock', False)}\n")
                    log_fh.write(msg)
                    log_fh.flush()
                    _print(msg, end='')
                    break

            if _shutdown_requested:
                break

            # --- Retrain (bots keep trading with current models) ---
            # Determine what to retrain: manual trigger overrides defaults
            if manual_trigger:
                rt_crypto = manual_trigger.get('crypto', False)
                rt_stock = manual_trigger.get('stock', False)
            else:
                rt_crypto = train_crypto
                rt_stock = train_stock

            # Pause backfill during retrain to free LLM quota
            try:
                from sentiment_history import set_live_mode
                set_live_mode(False)
            except Exception:
                pass

            # --- Adaptive mode/trial decisions per asset type ---
            # Use the worse-case (more trials) across asset types
            adaptive_modes = {}
            adaptive_trial_counts = {}
            for at in (['crypto'] if rt_crypto else []) + (['stock'] if rt_stock else []):
                astate = load_adaptive_state(at)
                amode = decide_mode(astate, astate.get('best_score', 0))
                atrial = get_trial_count(amode)
                adaptive_modes[at] = amode
                adaptive_trial_counts[at] = atrial
                msg = (f"[ADAPTIVE] {at}: mode={amode}, trials={atrial}, "
                       f"cycles_no_improve={astate.get('cycles_without_improvement', 0)}\n")
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')

            # Use explicit --retrain-trials if not default, else adaptive max
            if args.retrain_trials != 100:
                retrain_trials = args.retrain_trials
                retrain_mode = ''
            else:
                retrain_trials = max(adaptive_trial_counts.values()) if adaptive_trial_counts else 100
                # Use explore if any asset needs it
                retrain_mode = 'explore' if 'explore' in adaptive_modes.values() else 'refine'

            # Check if harvest needed due to forward_bars expansion
            force_harvest = False
            for at in ['crypto', 'stock']:
                if (at == 'crypto' and not rt_crypto) or (at == 'stock' and not rt_stock):
                    continue
                max_fb = get_max_forward_bars(at)
                # Check columns in Parquet (fast) or CSV
                import pandas as pd
                from data_utils import get_data_path
                data_path = get_data_path(at)
                if data_path.exists():
                    if str(data_path).endswith('.parquet'):
                        import pyarrow.parquet as pq
                        cols = pq.read_schema(data_path).names
                    else:
                        cols = pd.read_csv(data_path, nrows=0).columns.tolist()
                    if f'Target_Return_{max_fb}' not in cols:
                        force_harvest = True
                        msg = (f"[ADAPTIVE] {at}: Target_Return_{max_fb} missing from data, "
                               f"forcing re-harvest\n")
                        log_fh.write(msg)
                        log_fh.flush()
                        _print(msg, end='')

            retrain_phases = (
                _build_harvest_phases(not force_harvest, rt_crypto, rt_stock)
                + _build_training_phases(retrain_trials, rt_crypto, rt_stock,
                                         mode=retrain_mode)
            )
            for i, p in enumerate(retrain_phases):
                p['idx'] = i

            status['started_at'] = datetime.datetime.now().isoformat()
            status['phases'] = [p['id'] for p in retrain_phases]
            status['phase_labels'] = {p['id']: p['label'] for p in retrain_phases}
            status['total_phases'] = len(retrain_phases)

            retrain_source = "MANUAL" if manual_trigger else "WEEKLY"
            banner = (
                f"\n{'#'*70}\n"
                f"# {retrain_source} RETRAIN (cycle {cycle}): "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n"
                f"# Bots continue trading — models hot-reload on improvement\n"
                f"# Phases: {', '.join(p['label'] for p in retrain_phases)}\n"
                f"# Adaptive: mode={retrain_mode}, trials={retrain_trials}\n"
                f"{'#'*70}\n"
            )
            log_fh.write(banner)
            log_fh.flush()
            _print(banner, end='')

            _run_training(retrain_phases, log_fh, status, is_retrain=True)

            # Resume live mode after retrain (backfill pauses)
            try:
                from sentiment_history import set_live_mode
                set_live_mode(True)
            except Exception:
                pass

            msg = f"\nRetrain cycle {cycle} complete.\n"
            log_fh.write(msg)
            log_fh.flush()
            _print(msg, end='')

    finally:
        _heartbeat_stop.set()
        _terminate_procs(_all_procs)
        for name, proc, fh in _all_bots:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
        _cleanup_handles(_all_handles)


if __name__ == '__main__':
    main()
