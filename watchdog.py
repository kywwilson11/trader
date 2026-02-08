#!/usr/bin/env python3
"""Watchdog — monitors and restarts trading loops, emergency liquidation on repeated failure."""

import subprocess
import signal
import sys
import time
import os
import datetime
from collections import deque

from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

from order_utils import emergency_flatten

load_dotenv()

# --- CONFIGURATION ---
PROCESSES = {
    'crypto_loop': {
        'cmd': [sys.executable, 'crypto_loop.py'],
        'log': 'crypto_loop_output.log',
    },
    'stock_loop': {
        'cmd': [sys.executable, 'stock_loop.py'],
        'log': 'stock_loop_output.log',
    },
}

MAX_RESTARTS = 3           # Max restarts within the window before liquidating
RESTART_WINDOW = 3600      # 1 hour window for counting restarts
HEALTH_CHECK_INTERVAL = 30 # Check every 30s


def get_api():
    return tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_API_SECRET'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2',
    )


def start_process(name, config):
    """Start a subprocess, redirecting stdout/stderr to a log file."""
    log_path = config['log']
    log_file = open(log_path, 'a')
    log_file.write(f"\n--- {name} started at {datetime.datetime.now()} ---\n")
    log_file.flush()
    proc = subprocess.Popen(
        config['cmd'],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    print(f"[WATCHDOG] Started {name} (PID {proc.pid}), logging to {log_path}")
    return proc, log_file


def main():
    print(f"[WATCHDOG] Starting at {datetime.datetime.now()}")
    print(f"[WATCHDOG] Monitoring: {', '.join(PROCESSES.keys())}")
    print(f"[WATCHDOG] Max restarts: {MAX_RESTARTS} per {RESTART_WINDOW}s window")

    # State for each process
    procs = {}       # name -> Popen
    log_files = {}   # name -> file handle
    restart_times = {name: deque() for name in PROCESSES}  # name -> deque of restart timestamps
    disabled = set() # processes that exceeded MAX_RESTARTS (manual intervention needed)

    # Start all processes
    for name, config in PROCESSES.items():
        proc, log_file = start_process(name, config)
        procs[name] = proc
        log_files[name] = log_file

    # Handle graceful shutdown
    shutdown = False

    def handle_signal(signum, frame):
        nonlocal shutdown
        print(f"\n[WATCHDOG] Received signal {signum}, shutting down...")
        shutdown = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Main monitoring loop
    while not shutdown:
        time.sleep(HEALTH_CHECK_INTERVAL)

        for name, config in PROCESSES.items():
            if name in disabled:
                continue

            proc = procs.get(name)
            if proc is None:
                continue

            # Check if process is still alive
            ret = proc.poll()
            if ret is not None:
                # Process died
                print(f"[WATCHDOG] {name} exited with code {ret} at {datetime.datetime.now()}")

                # Close old log file
                if name in log_files:
                    log_files[name].close()

                # Prune old restart timestamps outside the window
                now = time.time()
                while restart_times[name] and (now - restart_times[name][0]) > RESTART_WINDOW:
                    restart_times[name].popleft()

                if len(restart_times[name]) >= MAX_RESTARTS:
                    # Too many restarts — emergency liquidation
                    print(f"[WATCHDOG] CRITICAL: {name} crashed {MAX_RESTARTS} times "
                          f"in {RESTART_WINDOW}s, liquidating!")
                    try:
                        api = get_api()
                        emergency_flatten(api)
                    except Exception as e:
                        print(f"[WATCHDOG] Emergency flatten error: {e}")

                    disabled.add(name)
                    procs[name] = None
                    print(f"[WATCHDOG] {name} DISABLED — manual intervention required")
                else:
                    # Restart the process
                    restart_times[name].append(now)
                    recent = len(restart_times[name])
                    print(f"[WATCHDOG] Restarting {name} ({recent}/{MAX_RESTARTS} restarts in window)")
                    proc, log_file = start_process(name, config)
                    procs[name] = proc
                    log_files[name] = log_file

    # Graceful shutdown — terminate children
    print("[WATCHDOG] Stopping child processes...")
    for name, proc in procs.items():
        if proc is not None and proc.poll() is None:
            proc.terminate()
            print(f"[WATCHDOG] Sent SIGTERM to {name} (PID {proc.pid})")

    # Wait for children to exit
    for name, proc in procs.items():
        if proc is not None:
            try:
                proc.wait(timeout=10)
                print(f"[WATCHDOG] {name} exited")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"[WATCHDOG] {name} killed (didn't exit in 10s)")

    # Close log files
    for lf in log_files.values():
        try:
            lf.close()
        except Exception:
            pass

    print("[WATCHDOG] Shutdown complete")


if __name__ == "__main__":
    main()
