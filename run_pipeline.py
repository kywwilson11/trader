#!/usr/bin/env python3
"""Overnight pipeline: train models then start trading.

Phases:
  1. Bear model hypersearch (250 trials, ~2-4 hours)
  2. Bull model hypersearch (250 trials, ~2-4 hours)
  3. Start crypto trading bot (runs until killed)

Writes status to pipeline_status.json for GUI monitoring.
All output logged to pipeline_output.log.

Usage:
    python run_pipeline.py                  # Full pipeline
    python run_pipeline.py --skip-harvest   # Skip data harvest (use existing)
    python run_pipeline.py --trials 50      # Fewer trials (faster test)
    python run_pipeline.py --bot-only       # Skip training, jump to bot
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
        status['cycle'] = 0

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

        # Parse trial progress: "[  45] acc=0.543 bear=0.523 ..."
        force = False
        m = re.match(r'\[\s*(\d+)\]', line)
        if m:
            status['trial_current'] = int(m.group(1))
            force = True

        # Parse best score on "** BEST **" lines
        if '** BEST **' in line:
            m = re.search(r'(?:bear|bull)=(\d+\.\d+)', line)
            if m:
                status['best_score'] = float(m.group(1))
            m = re.search(r'B:(\d+)% N:(\d+)% U:(\d+)%', line)
            if m:
                status['best_per_class'] = {
                    'bear': int(m.group(1)) / 100,
                    'neutral': int(m.group(2)) / 100,
                    'bull': int(m.group(3)) / 100,
                }
            force = True

        # Parse crypto bot cycle
        m = re.match(r'--- CYCLE (\d+):', line)
        if m:
            status['cycle'] = int(m.group(1))
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


def main():
    parser = argparse.ArgumentParser(description='Trading pipeline orchestrator')
    parser.add_argument('--trials', type=int, default=250,
                        help='Trials per model (default: 250)')
    parser.add_argument('--bot-only', action='store_true',
                        help='Skip training, start crypto bot immediately')
    parser.add_argument('--skip-harvest', action='store_true',
                        help='Skip data harvest (use existing training_data.csv)')
    args = parser.parse_args()

    # Build phase list
    phases = []
    if not args.bot_only:
        if not args.skip_harvest:
            # Check if training data exists and is recent (< 24h)
            csv_path = os.path.join(BASE_DIR, 'training_data.csv')
            if os.path.exists(csv_path):
                age_h = (time.time() - os.path.getmtime(csv_path)) / 3600
                if age_h < 24:
                    print(f"Training data is {age_h:.1f}h old, skipping harvest")
                else:
                    phases.append({
                        'id': 'harvest',
                        'label': 'Harvesting Crypto Data',
                        'cmd': [PYTHON, '-u', 'harvest_crypto_data.py'],
                    })
            else:
                phases.append({
                    'id': 'harvest',
                    'label': 'Harvesting Crypto Data',
                    'cmd': [PYTHON, '-u', 'harvest_crypto_data.py'],
                })

        phases.append({
            'id': 'bear_search',
            'label': 'Training Bear Model',
            'cmd': [PYTHON, '-u', 'hypersearch_dual.py',
                    '--target', 'bear', '--trials', str(args.trials)],
            'trials': args.trials,
        })
        phases.append({
            'id': 'bull_search',
            'label': 'Training Bull Model',
            'cmd': [PYTHON, '-u', 'hypersearch_dual.py',
                    '--target', 'bull', '--trials', str(args.trials)],
            'trials': args.trials,
        })

    phases.append({
        'id': 'crypto_bot',
        'label': 'Crypto Trading Bot',
        'cmd': [PYTHON, '-u', 'crypto_loop.py'],
    })

    # Assign indices
    for i, p in enumerate(phases):
        p['idx'] = i

    status = {
        'started_at': datetime.datetime.now().isoformat(),
        'phases': [p['id'] for p in phases],
        'phase_labels': {p['id']: p['label'] for p in phases},
        'total_phases': len(phases),
        'phase': 'starting',
        'phase_label': 'Starting Pipeline...',
        'phase_idx': -1,
        'trial_current': 0,
        'trial_total': 0,
        'best_score': 0.0,
        'best_per_class': {},
        'bear_final_score': None,
        'bull_final_score': None,
    }
    write_status(status, force=True)

    with open(LOG_FILE, 'a') as log_fh:
        banner = (
            f"\n{'#'*70}\n"
            f"# PIPELINE STARTED: {datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n"
            f"# Phases: {', '.join(p['label'] for p in phases)}\n"
            f"# Trials per model: {args.trials}\n"
            f"{'#'*70}\n"
        )
        log_fh.write(banner)
        log_fh.flush()
        print(banner, end='')

        for phase in phases:
            rc = run_phase(phase, log_fh, status)

            # Save final scores after search phases
            if phase['id'] == 'bear_search':
                status['bear_final_score'] = status.get('best_score', 0)
            elif phase['id'] == 'bull_search':
                status['bull_final_score'] = status.get('best_score', 0)

            # Stop pipeline on non-zero exit from training phases
            if rc != 0 and phase['id'] != 'crypto_bot':
                msg = f"\nPIPELINE STOPPED: {phase['label']} failed (exit {rc})\n"
                log_fh.write(msg)
                log_fh.flush()
                print(msg, end='')
                status['phase'] = 'failed'
                status['phase_label'] = f"Failed: {phase['label']}"
                write_status(status, force=True)
                sys.exit(1)

        status['phase'] = 'complete'
        status['phase_label'] = 'Pipeline Complete'
        write_status(status, force=True)


if __name__ == '__main__':
    main()
