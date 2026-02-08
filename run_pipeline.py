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
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(BASE_DIR, 'pipeline_status.json')
LOG_FILE = os.path.join(BASE_DIR, 'pipeline_output.log')
CRYPTO_BOT_LOG = os.path.join(BASE_DIR, 'crypto_bot_output.log')
STOCK_BOT_LOG = os.path.join(BASE_DIR, 'stock_bot_output.log')
PYTHON = '/home/kyle/miniforge3/envs/jetson/bin/python'
ENV = {
    **os.environ,
    'LD_LIBRARY_PATH': (
        '/home/kyle/miniforge3/envs/jetson/lib/python3.10/site-packages/'
        'nvidia/cusparselt/lib:'
        + os.environ.get('LD_LIBRARY_PATH', '')
    ),
    'PYTHONUNBUFFERED': '1',
}

# Throttle JSON writes to avoid excessive disk I/O
_last_status_write = 0
STATUS_WRITE_INTERVAL = 2.0  # seconds

DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                'Friday', 'Saturday', 'Sunday']


def write_status(status, force=False):
    """Write pipeline status to JSON, throttled to every 2 seconds."""
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
    tmp = STATUS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, STATUS_FILE)


def run_phase(phase, log_fh, status):
    """Run a single pipeline phase as a subprocess, parsing output in real-time."""
    phase_idx = phase['idx']
    phase_id = phase['id']

    status['phase'] = phase_id
    status['phase_label'] = phase['label']
    status['phase_idx'] = phase_idx
    status['phase_started_at'] = datetime.datetime.now().isoformat()
    status['trial_current'] = 0
    status['best_score'] = status.get('best_score', 0.0) if 'search' not in phase_id else 0.0
    status['best_per_class'] = {} if 'search' in phase_id else status.get('best_per_class', {})

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
    print(header, end='')

    proc = subprocess.Popen(
        phase['cmd'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=ENV,
        cwd=BASE_DIR,
        bufsize=1,
        text=True,
    )

    for line in proc.stdout:
        log_fh.write(line)
        log_fh.flush()
        sys.stdout.write(line)
        sys.stdout.flush()

        # Parse trial progress: "[  45] score=0.543 ..."
        force = False
        m = re.match(r'\[\s*(\d+)\]', line)
        if m:
            status['trial_current'] = int(m.group(1))
            force = True

        # Parse best score on "** BEST **" lines
        if '** BEST **' in line:
            m = re.search(r'score=(\d+\.\d+)', line)
            if m:
                status['best_score'] = float(m.group(1))
            m = re.search(r'F1=(\d+\.\d+)', line)
            if m:
                status['best_f1'] = float(m.group(1))
            m = re.search(r'cat=(\d+\.\d+)', line)
            if m:
                status['best_catastrophic'] = float(m.group(1))
            m = re.search(r'B:(\d+)% N:(\d+)% U:(\d+)%', line)
            if m:
                status['best_per_class'] = {
                    'bear': int(m.group(1)) / 100,
                    'neutral': int(m.group(2)) / 100,
                    'bull': int(m.group(3)) / 100,
                }
            force = True

        write_status(status, force=force)

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
    print(footer, end='')

    write_status(status, force=True)
    return proc.returncode


# ---------------------------------------------------------------------------
# Bot management helpers
# ---------------------------------------------------------------------------

def _start_bot(cmd, log_path):
    """Start a trading bot as a background process."""
    fh = open(log_path, 'a')
    return subprocess.Popen(
        cmd, stdout=fh, stderr=subprocess.STDOUT,
        env=ENV, cwd=BASE_DIR,
    )


def _check_restart_bots(bots, log_fh):
    """Check for crashed bots and restart them."""
    for i, (name, proc) in enumerate(bots):
        if proc.poll() is not None:
            log_path = CRYPTO_BOT_LOG if name == 'Crypto' else STOCK_BOT_LOG
            cmd = [PYTHON, '-u',
                   'crypto_loop.py' if name == 'Crypto' else 'stock_loop.py']
            new_proc = _start_bot(cmd, log_path)
            bots[i] = (name, new_proc)
            msg = (f"{name} bot crashed (exit {proc.returncode}),"
                   f" restarted as PID {new_proc.pid}\n")
            log_fh.write(msg)
            log_fh.flush()
            print(msg, end='')


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

def _build_harvest_phases(skip_harvest, train_crypto, train_stock):
    """Build harvest phases, skipping if data is fresh."""
    phases = []
    if skip_harvest:
        return phases

    if train_crypto:
        csv_path = os.path.join(BASE_DIR, 'training_data.csv')
        if os.path.exists(csv_path):
            age_h = (time.time() - os.path.getmtime(csv_path)) / 3600
            if age_h < 24:
                print(f"Crypto training data is {age_h:.1f}h old, skipping harvest")
            else:
                phases.append({
                    'id': 'crypto_harvest',
                    'label': 'Harvesting Crypto Data',
                    'cmd': [PYTHON, '-u', 'harvest_crypto_data.py'],
                })
        else:
            phases.append({
                'id': 'crypto_harvest',
                'label': 'Harvesting Crypto Data',
                'cmd': [PYTHON, '-u', 'harvest_crypto_data.py'],
            })

    if train_stock:
        stock_csv = os.path.join(BASE_DIR, 'stock_training_data.csv')
        if os.path.exists(stock_csv):
            age_h = (time.time() - os.path.getmtime(stock_csv)) / 3600
            if age_h < 24:
                print(f"Stock training data is {age_h:.1f}h old, skipping harvest")
            else:
                phases.append({
                    'id': 'stock_harvest',
                    'label': 'Harvesting Stock Data',
                    'cmd': [PYTHON, '-u', 'harvest_stock_data.py'],
                })
        else:
            phases.append({
                'id': 'stock_harvest',
                'label': 'Harvesting Stock Data',
                'cmd': [PYTHON, '-u', 'harvest_stock_data.py'],
            })

    return phases


def _build_training_phases(trials, train_crypto, train_stock):
    """Build model training phases."""
    phases = []

    if train_crypto:
        phases.append({
            'id': 'bear_search',
            'label': 'Training Crypto Bear Model',
            'cmd': [PYTHON, '-u', 'hypersearch_dual.py',
                    '--target', 'bear', '--trials', str(trials)],
            'trials': trials,
        })
        phases.append({
            'id': 'bull_search',
            'label': 'Training Crypto Bull Model',
            'cmd': [PYTHON, '-u', 'hypersearch_dual.py',
                    '--target', 'bull', '--trials', str(trials)],
            'trials': trials,
        })

    if train_stock:
        phases.append({
            'id': 'stock_bear_search',
            'label': 'Training Stock Bear Model',
            'cmd': [PYTHON, '-u', 'hypersearch_dual.py',
                    '--target', 'bear', '--trials', str(trials),
                    '--data', 'stock_training_data.csv', '--prefix', 'stock'],
            'trials': trials,
        })
        phases.append({
            'id': 'stock_bull_search',
            'label': 'Training Stock Bull Model',
            'cmd': [PYTHON, '-u', 'hypersearch_dual.py',
                    '--target', 'bull', '--trials', str(trials),
                    '--data', 'stock_training_data.csv', '--prefix', 'stock'],
            'trials': trials,
        })

    return phases


def _run_training(phases, log_fh, status, is_retrain):
    """Run all training phases. Returns True if all succeeded."""
    for phase in phases:
        rc = run_phase(phase, log_fh, status)

        # Save final scores
        if phase['id'] == 'bear_search':
            status['bear_final_score'] = status.get('best_score', 0)
        elif phase['id'] == 'bull_search':
            status['bull_final_score'] = status.get('best_score', 0)
        elif phase['id'] == 'stock_bear_search':
            status['stock_bear_final_score'] = status.get('best_score', 0)
        elif phase['id'] == 'stock_bull_search':
            status['stock_bull_final_score'] = status.get('best_score', 0)

        if rc != 0:
            if is_retrain:
                msg = (f"\nWARNING: {phase['label']} failed (exit {rc}),"
                       f" bots continue with existing models\n")
                log_fh.write(msg)
                log_fh.flush()
                print(msg, end='')
                return False
            else:
                msg = f"\nPIPELINE STOPPED: {phase['label']} failed (exit {rc})\n"
                log_fh.write(msg)
                log_fh.flush()
                print(msg, end='')
                status['phase'] = 'failed'
                status['phase_label'] = f"Failed: {phase['label']}"
                write_status(status, force=True)
                sys.exit(1)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Trading pipeline orchestrator')
    parser.add_argument('--trials', type=int, default=250,
                        help='Trials per model on first run (default: 250)')
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
    parser.add_argument('--retrain-trials', type=int, default=50,
                        help='Trials per model for weekly retrain (default: 50)')
    args = parser.parse_args()

    train_crypto = not args.stock_only
    train_stock = not args.crypto_only
    run_crypto = not args.stock_only
    run_stock = not args.crypto_only

    status = {
        'started_at': datetime.datetime.now().isoformat(),
        'phase': 'starting',
        'phase_label': 'Starting Pipeline...',
        'phase_idx': -1,
        'total_phases': 0,
        'trial_current': 0,
        'trial_total': 0,
        'best_score': 0.0,
        'best_per_class': {},
        'bear_final_score': None,
        'bull_final_score': None,
        'stock_bear_final_score': None,
        'stock_bull_final_score': None,
        'retrain_cycle': 0,
        'bots_running': False,
    }

    with open(LOG_FILE, 'a') as log_fh:

        # =============================================================
        # PHASE A: Initial training (cycle 0)
        # =============================================================
        if not args.bot_only:
            phases = (_build_harvest_phases(args.skip_harvest, train_crypto, train_stock)
                      + _build_training_phases(args.trials, train_crypto, train_stock))
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
            print(banner, end='')

            _run_training(phases, log_fh, status, is_retrain=False)

        # =============================================================
        # PHASE B: Start trading bots (run forever)
        # =============================================================
        bots = []

        if run_crypto:
            proc = _start_bot([PYTHON, '-u', 'crypto_loop.py'], CRYPTO_BOT_LOG)
            bots.append(('Crypto', proc))
            msg = f"Crypto bot started (PID {proc.pid}, log: crypto_bot_output.log)\n"
            log_fh.write(msg)
            log_fh.flush()
            print(msg, end='')

        if run_stock:
            proc = _start_bot([PYTHON, '-u', 'stock_loop.py'], STOCK_BOT_LOG)
            bots.append(('Stock', proc))
            msg = f"Stock bot started (PID {proc.pid}, log: stock_bot_output.log)\n"
            log_fh.write(msg)
            log_fh.flush()
            print(msg, end='')

        status['phase'] = 'trading'
        status['phase_label'] = 'Trading'
        status['bots_running'] = True
        write_status(status, force=True)

        # --- No retrain: wait forever ---
        if args.no_retrain:
            msg = "Retrain disabled, bots running until manually stopped.\n"
            log_fh.write(msg)
            log_fh.flush()
            print(msg, end='')
            if bots:
                # Block forever, restarting crashed bots
                while True:
                    time.sleep(60)
                    _check_restart_bots(bots, log_fh)
            return

        # =============================================================
        # PHASE C: Weekly retrain loop (bots keep running)
        # =============================================================
        cycle = 0
        while True:
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
            print(msg, end='')

            # Wait until retrain time, auto-restarting crashed bots
            while datetime.datetime.now() < next_retrain:
                time.sleep(60)
                _check_restart_bots(bots, log_fh)
                write_status(status)

            # --- Retrain (bots keep trading with current models) ---
            retrain_phases = (
                _build_harvest_phases(False, train_crypto, train_stock)
                + _build_training_phases(args.retrain_trials, train_crypto, train_stock)
            )
            for i, p in enumerate(retrain_phases):
                p['idx'] = i

            status['started_at'] = datetime.datetime.now().isoformat()
            status['phases'] = [p['id'] for p in retrain_phases]
            status['phase_labels'] = {p['id']: p['label'] for p in retrain_phases}
            status['total_phases'] = len(retrain_phases)

            banner = (
                f"\n{'#'*70}\n"
                f"# WEEKLY RETRAIN (cycle {cycle}): "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n"
                f"# Bots continue trading — models hot-reload on improvement\n"
                f"# Phases: {', '.join(p['label'] for p in retrain_phases)}\n"
                f"# Trials per model: {args.retrain_trials}\n"
                f"{'#'*70}\n"
            )
            log_fh.write(banner)
            log_fh.flush()
            print(banner, end='')

            _run_training(retrain_phases, log_fh, status, is_retrain=True)

            msg = f"\nRetrain cycle {cycle} complete.\n"
            log_fh.write(msg)
            log_fh.flush()
            print(msg, end='')


if __name__ == '__main__':
    main()
