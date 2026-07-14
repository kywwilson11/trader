#!/usr/bin/env python3
"""Overnight pipeline: train models, start trading, auto-retrain weekly.

Flow:
  1. Initial training (harvest + hypersearch for all models)
  2. Start trading bots (crypto 24/7 + stock during market hours)
  3. Bots run continuously — they hot-reload models when .pth files change
  4. Every Saturday 2 AM: re-harvest data + retrain models
     (bots stop during training to free GPU memory, restart after)

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
PIPELINE_COMMAND = os.path.join(BASE_DIR, 'pipeline_command.json')
# Skip stdout writes when redirected to same log file (avoids doubled lines)
_STDOUT_IS_TTY = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def _print(*args, **kwargs):
    """Print only when stdout is a terminal (avoids doubling when redirected to log)."""
    if _STDOUT_IS_TTY:
        print(*args, **kwargs)


def _sd_notify(msg: bytes):
    """systemd notify protocol (READY=1 / WATCHDOG=1) — stdlib only.

    No-op outside systemd (NOTIFY_SOCKET unset). Lets trader.service use
    Type=notify + WatchdogSec so a hung pipeline gets auto-restarted.
    """
    sock_path = os.environ.get('NOTIFY_SOCKET')
    if not sock_path:
        return
    try:
        import socket
        if sock_path.startswith('@'):
            sock_path = '\0' + sock_path[1:]
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            s.connect(sock_path)
            s.send(msg)
        finally:
            s.close()
    except OSError:
        pass


def _check_telegram_commands(log_fh, status):
    """Telegram kill switch: /halt /resume /flatten /status."""
    try:
        from notify import (poll_telegram_commands, set_halt, clear_halt,
                            halt_active, request_flatten, notify)
        for cmd in poll_telegram_commands():
            log_fh.write(f"[TELEGRAM] command: {cmd}\n")
            log_fh.flush()
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
                notify(f"phase={status.get('phase')} "
                       f"halted={halt_active()} "
                       f"crypto_bot={status.get('crypto_bot_running')} "
                       f"stock_bot={status.get('stock_bot_running')}",
                       level='info', dedupe_key=f'tg-status-{time.time():.0f}')
    except Exception as e:
        try:
            log_fh.write(f"[TELEGRAM] poll failed: {e}\n")
        except Exception:
            pass


_last_drift_check_date = None


def _maybe_run_drift_check(log_fh):
    """Once-daily in-process PSI drift check. Writes the retrain flags
    that the Phase C wait loop consumes on its next iteration — no
    external cron needed on the Jetson."""
    global _last_drift_check_date
    today = datetime.date.today().isoformat()
    if _last_drift_check_date == today:
        return
    _last_drift_check_date = today
    try:
        from monitor_drift import run_check
        for prefix, label in (('', 'crypto'), ('stock', 'stock')):
            r = run_check(prefix, label)
            if r is not None:
                log_fh.write(f"[DRIFT] {label}: PSI={r['psi']} "
                             f"({r['level']}, n={r['n']})\n")
        log_fh.flush()
    except Exception as e:
        try:
            log_fh.write(f"[DRIFT] daily check failed: {e}\n")
            log_fh.flush()
        except Exception:
            pass
    # Challenger shadow evaluation (DM-HLN promote/discard decisions)
    try:
        from shadow import evaluate_and_maybe_promote
        for prefix, label in (('', 'crypto'), ('stock', 'stock')):
            r = evaluate_and_maybe_promote(prefix, label)
            if r is not None:
                log_fh.write(f"[SHADOW] {label}: n={r['n']} "
                             f"age={r['age_days']:.1f}d p={r['p']} "
                             f"-> {r['decision']}\n")
        log_fh.flush()
    except Exception as e:
        try:
            log_fh.write(f"[SHADOW] daily eval failed: {e}\n")
            log_fh.flush()
        except Exception:
            pass


def _check_drift_trigger():
    """Check for and consume drift-monitor retrain flags.

    monitor_drift.py writes {prefix}retrain_requested.flag after PSI
    exceeds the action level on consecutive days. Same contract as the
    GUI trigger: returns {'crypto': bool, 'stock': bool} or None.
    """
    found = {}
    for key, fname in (('crypto', 'retrain_requested.flag'),
                       ('stock', 'stock_retrain_requested.flag')):
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            try:
                os.remove(path)
                found[key] = True
            except OSError:
                pass
    return found or None


def _check_retrain_trigger():
    """Check for and consume a manual retrain trigger file from the GUI.

    Returns dict {'crypto': bool, 'stock': bool} if trigger found, else None.
    Uses atomic rename-before-read to avoid TOCTOU race conditions.
    """
    if not os.path.exists(RETRAIN_TRIGGER):
        return None
    tmp = str(RETRAIN_TRIGGER) + '.reading'
    try:
        os.rename(str(RETRAIN_TRIGGER), tmp)
        with open(tmp) as f:
            trigger = json.load(f)
        os.remove(tmp)
        if trigger.get('crypto') or trigger.get('stock'):
            return trigger
    except (OSError, json.JSONDecodeError):
        try:
            os.remove(tmp)
        except OSError:
            pass
    return None


def _check_pipeline_command():
    """Check for and consume a pipeline command file from the GUI.

    Returns dict with 'command', 'crypto', 'stock' keys if found, else None.
    Uses atomic rename-before-read to avoid TOCTOU race conditions.
    """
    if not os.path.exists(PIPELINE_COMMAND):
        return None
    tmp = str(PIPELINE_COMMAND) + '.reading'
    try:
        os.rename(str(PIPELINE_COMMAND), tmp)
        with open(tmp) as f:
            cmd = json.load(f)
        os.remove(tmp)
        if cmd.get('command'):
            return cmd
    except (OSError, json.JSONDecodeError):
        try:
            os.remove(tmp)
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

# Bots use CPU-only inference — hide GPU so PyTorch doesn't reserve CUDA memory.
# This frees ~600MB for training (each CUDA context costs ~300MB on Jetson).
# OMP/torch thread caps stop tiny-LSTM inference stealing cores from training.
BOT_ENV = {**ENV, 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '2', 'TORCH_NUM_THREADS': '2'}

# Throttle JSON writes to avoid excessive disk I/O
_last_status_write = 0
STATUS_WRITE_INTERVAL = 2.0  # seconds

DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                'Friday', 'Saturday', 'Sunday']

# Heartbeat: re-write status every 30s so GUI knows pipeline is alive
_heartbeat_status = None  # Reference to current status dict
_heartbeat_stop = threading.Event()
_heartbeat_lock = threading.Lock()

# Main-loop liveness stamp. The heartbeat daemon used to feed the systemd
# watchdog UNCONDITIONALLY, so WatchdogSec=900 only ever caught whole-process
# death — a logically hung main thread (blocked forever in a phase's stdout
# read, a wedged API call) kept getting its lease renewed. The heartbeat now
# forwards WATCHDOG=1 only while the main thread has stamped progress
# recently; a real hang lets the lease expire and systemd restarts us.
_last_progress = time.time()
WATCHDOG_STALL_SEC = 600  # < WatchdogSec=900, > any legitimate quiet gap


def mark_progress():
    """Stamp main-loop liveness (called from phase output + monitor loops)."""
    global _last_progress
    _last_progress = time.time()


def _bounded_thermal_wait(max_temp=70, deadline_sec=1800, poll_interval=30):
    """Poll GPU temp until it drops below max_temp or deadline_sec elapses.

    Replaces a direct hw_monitor.wait_for_cool_gpu() call, which can block
    unboundedly. A deliberate thermal wait IS progress, not a hang, so
    every poll stamps mark_progress() — without this, >WATCHDOG_STALL_SEC
    of cooling would let the systemd watchdog lease lapse and kill an
    otherwise-healthy pipeline mid-cooldown. Fails open (returns
    immediately) on import/read errors or sensor unavailability.
    """
    try:
        from hw_monitor import get_gpu_temp
    except Exception:
        return
    deadline = time.time() + deadline_sec
    while True:
        try:
            temp = get_gpu_temp()
        except Exception:
            return
        mark_progress()
        if temp is None or temp < max_temp:
            return
        if time.time() >= deadline:
            _print(f"[HW] GPU still hot ({temp:.0f}C) after {deadline_sec}s"
                   f" wait, proceeding anyway")
            return
        time.sleep(poll_interval)


def _heartbeat_loop():
    """Background thread: re-write status file + systemd watchdog every 30s."""
    while not _heartbeat_stop.wait(30):
        with _heartbeat_lock:
            if _heartbeat_status is not None:
                try:
                    write_status(_heartbeat_status, force=True)
                except RuntimeError:
                    # json.dump saw the dict mutate mid-iteration (main
                    # thread wrote a key concurrently) — skip this beat,
                    # the next one 30s later self-heals.
                    pass
        if (time.time() - _last_progress) < WATCHDOG_STALL_SEC:
            _sd_notify(b'WATCHDOG=1')


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

    # Thermal gate before GPU-heavy phases: a multi-hour Saturday Optuna run
    # at 25W in an enclosure otherwise starts at whatever temperature the
    # previous phase left the die at. (setup_jetson_system.sh documented this
    # as already wired — it never was; only the bots throttled on temp.)
    if 'search' in phase_id:
        _bounded_thermal_wait(max_temp=70)

    proc = subprocess.Popen(
        phase['cmd'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=ENV,
        cwd=BASE_DIR,
        bufsize=1,
        text=True,
    )
    global _current_phase_proc
    _current_phase_proc = proc

    try:
        for line in proc.stdout:
            mark_progress()
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

            # Check for suspend request from GUI
            global _suspend_requested
            if not _suspend_requested:
                scmd = _check_pipeline_command()
                if scmd and scmd.get('command') == 'suspend_and_start_bot':
                    _suspend_requested = True
                    status['_pending_bot_start'] = {
                        'crypto': scmd.get('crypto', False),
                        'stock': scmd.get('stock', False),
                    }
            if _suspend_requested:
                msg = "\n[SUSPEND] Training suspended by GUI command\n"
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                status['phase_exit_code'] = -99
                write_status(status, force=True)
                return -99

            write_status(status, force=force)
    except Exception as e:
        _print(f"\n[PIPELINE] Error reading phase output: {e}")
    finally:
        proc.wait()
        _current_phase_proc = None
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

def _rotate_log(log_path, max_bytes=20_000_000):
    """One-deep rotation for the append-mode bot stdout logs.

    predict_now prints ~5-8 lines per symbol per cycle, so these files grow
    ~10-20MB/day forever (the RotatingFileHandler in log_config covers logger
    output only, not subprocess stdout). Root-FS disk-full kills the bots AND
    the swapfile lives on the same device — bound it at ~2 x max_bytes.
    """
    try:
        if os.path.exists(log_path) and os.path.getsize(log_path) > max_bytes:
            os.replace(log_path, f"{log_path}.1")
    except OSError:
        pass  # rotation is best-effort; never block a bot launch on it


def _start_bot(cmd, log_path):
    """Start a trading bot as a background process.

    Uses BOT_ENV which hides the GPU (CUDA_VISIBLE_DEVICES='') so bots
    don't reserve ~300MB of GPU memory each via CUDA context init.
    Returns (proc, file_handle) so the caller can close the log FH when done.
    """
    _rotate_log(log_path)
    fh = open(log_path, 'a')
    proc = subprocess.Popen(
        cmd, stdout=fh, stderr=subprocess.STDOUT,
        env=BOT_ENV, cwd=BASE_DIR,
    )
    return proc, fh


def _untrack_handle(fh):
    """Drop a closed file handle from _all_handles.

    Without this, every bot start/stop/crash-restart cycle over a
    long-running pipeline leaves the old (closed) handle in _all_handles
    forever — unbounded growth over weeks of retrain cycles. Closing is
    idempotent so double-close is harmless; only the tracking list needs
    pruning.
    """
    try:
        _all_handles.remove(fh)
    except ValueError:
        pass


def _stop_bots(bots, log_fh):
    """Stop all bot processes to free memory for training.

    On Jetson (8GB unified memory), each bot's PyTorch import reserves
    ~300-500MB.  Stopping them before training frees ~1GB for CUDA.
    """
    for name, proc, fh in bots:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                msg = f"  Stopped {name} bot (PID {proc.pid}) for training\n"
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')
        except Exception as e:
            msg = f"  Warning: failed to stop {name} bot: {e}\n"
            log_fh.write(msg)
            log_fh.flush()
        try:
            fh.close()
        except Exception:
            pass
        _untrack_handle(fh)
    bots.clear()
    # Give OS a moment to reclaim memory from terminated processes
    time.sleep(3)


_COMBINED_BOTS = False  # set from --combined-bots in main()


def _launch_bots(bots, log_fh, run_crypto, run_stock, verb='started'):
    """Launch trading bots — combined single process or one per bot.

    Combined mode (--combined-bots) saves a duplicate torch+pandas import
    stack (~0.5-0.8GB on the Jetson) by running both loops as threads in
    run_bots.py. Per-bot GUI restart commands target the whole 'Bots'
    process in that mode.
    """
    if _COMBINED_BOTS and (run_crypto or run_stock):
        cmd = [PYTHON, '-u', 'run_bots.py']
        if run_crypto and not run_stock:
            cmd.append('--crypto-only')
        elif run_stock and not run_crypto:
            cmd.append('--stock-only')
        proc, bot_fh = _start_bot(cmd, CRYPTO_BOT_LOG)
        bots.append(('Bots', proc, bot_fh))
        _all_handles.append(bot_fh)
        msg = f"Combined bot process {verb} (PID {proc.pid}, log: crypto_bot_output.log)\n"
        log_fh.write(msg)
        log_fh.flush()
        _print(msg, end='')
        return

    if run_crypto:
        proc, bot_fh = _start_bot([PYTHON, '-u', 'crypto_loop.py'], CRYPTO_BOT_LOG)
        bots.append(('Crypto', proc, bot_fh))
        _all_handles.append(bot_fh)
        msg = f"Crypto bot {verb} (PID {proc.pid}, log: crypto_bot_output.log)\n"
        log_fh.write(msg)
        log_fh.flush()
        _print(msg, end='')

    if run_stock:
        proc, bot_fh = _start_bot([PYTHON, '-u', 'stock_loop.py'], STOCK_BOT_LOG)
        bots.append(('Stock', proc, bot_fh))
        _all_handles.append(bot_fh)
        msg = f"Stock bot {verb} (PID {proc.pid}, log: stock_bot_output.log)\n"
        log_fh.write(msg)
        log_fh.flush()
        _print(msg, end='')


def _restart_bots(bots, log_fh, run_crypto, run_stock):
    """Restart bot processes after training completes."""
    _launch_bots(bots, log_fh, run_crypto, run_stock, verb='restarted')


def _stop_single_bot(bots, name, log_fh):
    """Stop a single bot by name. Returns True if found and stopped."""
    for i, (bot_name, proc, fh) in enumerate(bots):
        if bot_name == name:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                msg = f"  Stopped {name} bot (PID {proc.pid}) per GUI command\n"
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')
            except Exception as e:
                msg = f"  Warning: failed to stop {name} bot: {e}\n"
                log_fh.write(msg)
                log_fh.flush()
            try:
                fh.close()
            except Exception:
                pass
            _untrack_handle(fh)
            bots.pop(i)
            return True
    return False


def _start_single_bot(bots, name, log_fh):
    """Start a single bot by name if not already running."""
    for bot_name, proc, fh in bots:
        if bot_name == name and proc.poll() is None:
            return
    if name == 'Crypto':
        cmd = [PYTHON, '-u', 'crypto_loop.py']
        log_path = CRYPTO_BOT_LOG
    elif name == 'Stock':
        cmd = [PYTHON, '-u', 'stock_loop.py']
        log_path = STOCK_BOT_LOG
    else:
        return
    proc, bot_fh = _start_bot(cmd, log_path)
    bots.append((name, proc, bot_fh))
    _all_handles.append(bot_fh)
    msg = f"  {name} bot started (PID {proc.pid}) per GUI command\n"
    log_fh.write(msg)
    log_fh.flush()
    _print(msg, end='')


def _update_per_bot_status(bots, status):
    """Update per-bot running flags in the status dict."""
    crypto_running = any(n == 'Crypto' and p.poll() is None for n, p, _ in bots)
    stock_running = any(n == 'Stock' and p.poll() is None for n, p, _ in bots)
    status['crypto_bot_running'] = crypto_running
    status['stock_bot_running'] = stock_running
    status['bots_running'] = crypto_running or stock_running


def _handle_command(cmd, bots, log_fh, status):
    """Handle a pipeline command from the GUI."""
    global _suspend_requested
    command = cmd.get('command', '')
    want_crypto = cmd.get('crypto', False)
    want_stock = cmd.get('stock', False)

    msg = f"\n[CMD] Received: {command} (crypto={want_crypto}, stock={want_stock})\n"
    log_fh.write(msg)
    log_fh.flush()
    _print(msg, end='')

    if command == 'stop_bot':
        if _COMBINED_BOTS:
            # Combined mode runs both books as threads in ONE process — a
            # per-book stop request has no per-book process to target.
            # Stop the whole thing and say so loudly, instead of silently
            # no-op'ing (the old behavior: _stop_single_bot(bots, 'Crypto'
            # or 'Stock', ...) never matched a bot named 'Bots').
            msg = ("  Combined mode: stopping the WHOLE 'Bots' process"
                   " (both books trade together, or neither does)\n")
            log_fh.write(msg)
            log_fh.flush()
            _print(msg, end='')
            _stop_single_bot(bots, 'Bots', log_fh)
            if want_crypto:
                _manually_stopped.add('Crypto')
            if want_stock:
                _manually_stopped.add('Stock')
        else:
            if want_crypto:
                _stop_single_bot(bots, 'Crypto', log_fh)
                _manually_stopped.add('Crypto')
            if want_stock:
                _stop_single_bot(bots, 'Stock', log_fh)
                _manually_stopped.add('Stock')
        _update_per_bot_status(bots, status)
        write_status(status, force=True)

    elif command == 'start_bot':
        phase = status.get('phase', '')
        if phase not in ('trading', 'idle', 'failed', 'complete', 'suspended', ''):
            msg = "  Cannot start bot: training in progress\n"
            log_fh.write(msg)
            log_fh.flush()
            return
        if _COMBINED_BOTS:
            combined_alive = any(n == 'Bots' and p.poll() is None
                                 for n, p, _ in bots)
            if combined_alive:
                # Starting a per-book loop here would duplicate order flow
                # alongside the already-running combined process.
                msg = ("  Cannot start per-book bot: combined 'Bots' process"
                       " is already running\n")
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')
            else:
                run_crypto, run_stock = _BOT_SCOPE
                _manually_stopped.discard('Crypto')
                _manually_stopped.discard('Stock')
                _launch_bots(bots, log_fh, run_crypto, run_stock, verb='started')
        else:
            if want_crypto:
                _manually_stopped.discard('Crypto')
                _start_single_bot(bots, 'Crypto', log_fh)
            if want_stock:
                _manually_stopped.discard('Stock')
                _start_single_bot(bots, 'Stock', log_fh)
        _update_per_bot_status(bots, status)
        write_status(status, force=True)

    elif command == 'suspend_and_start_bot':
        _suspend_requested = True
        status['_pending_bot_start'] = {
            'crypto': want_crypto, 'stock': want_stock,
        }
        if want_crypto:
            _manually_stopped.discard('Crypto')
        if want_stock:
            _manually_stopped.discard('Stock')
        msg = "  Training suspension requested, will start bots after\n"
        log_fh.write(msg)
        log_fh.flush()
        write_status(status, force=True)


def _check_restart_bots(bots, log_fh):
    """Check for crashed bots and restart them (skips manually stopped)."""
    mark_progress()  # each monitor cycle proves the main loop is alive
    for i, (name, proc, bot_fh) in enumerate(bots):
        if proc.poll() is not None:
            if name in _manually_stopped:
                try:
                    bot_fh.close()
                except Exception:
                    pass
                _untrack_handle(bot_fh)
                bots.pop(i)
                return  # List modified; next cycle will re-check
            # Close the old log file handle before opening a new one
            try:
                bot_fh.close()
            except Exception:
                pass
            _untrack_handle(bot_fh)
            if name == 'Bots':  # combined-mode process
                log_path = CRYPTO_BOT_LOG
                cmd = [PYTHON, '-u', 'run_bots.py']
                run_crypto, run_stock = _BOT_SCOPE
                if run_crypto and not run_stock:
                    cmd.append('--crypto-only')
                elif run_stock and not run_crypto:
                    cmd.append('--stock-only')
            else:
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
            try:
                from notify import notify
                notify(f"{name} bot crashed (exit {proc.returncode}) — "
                       f"auto-restarted as PID {new_proc.pid}",
                       level='warning', dedupe_key=f'bot-crash-{name}')
            except Exception:
                pass


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


def _build_harvest_phases(skip_harvest, train_crypto, train_stock, force=False):
    """Build harvest phases, skipping if data is fresh.

    Args:
        force: harvest even if the data file is <24h old (used when the
            forward_bars expansion needs columns the current file lacks).
    """
    phases = []
    if skip_harvest:
        return phases

    if train_crypto:
        age_h = None if force else _get_data_age_hours('crypto')
        if age_h is not None and age_h < 24:
            _print(f"Crypto training data is {age_h:.1f}h old, skipping harvest")
        else:
            phases.append({
                'id': 'crypto_harvest',
                'label': 'Harvesting Crypto Data',
                'cmd': [PYTHON, '-u', os.path.join('scripts', 'harvest_crypto_data.py')],
            })

    if train_stock:
        age_h = None if force else _get_data_age_hours('stock')
        if age_h is not None and age_h < 24:
            _print(f"Stock training data is {age_h:.1f}h old, skipping harvest")
        else:
            phases.append({
                'id': 'stock_harvest',
                'label': 'Harvesting Stock Data',
                'cmd': [PYTHON, '-u', os.path.join('scripts', 'harvest_stock_data.py')],
            })

    return phases


def _build_training_phases(trials, train_crypto, train_stock, mode='',
                           shadow=False):
    """Build model training phases with adaptive mode support.

    Each training phase is followed by a policy-backtest GATE phase:
    backtest.py replays the real entry/exit stack with real fees over the
    recent window and restores the previous model artifacts if the freshly
    promoted model loses money at the policy level (fit metrics alone are
    not sufficient evidence — see validation.py).
    """
    phases = []
    # Weekly retrains save into the CHALLENGER slot (shadow.py) so a new
    # model must beat the champion on LIVE data before deployment.
    # TRADER_SHADOW_MODE=0 restores immediate promotion.
    shadow = shadow and os.getenv('TRADER_SHADOW_MODE', '1') != '0'

    if train_crypto:
        cmd = [PYTHON, '-u', os.path.join('scripts', 'hypersearch_v2.py'),
               '--trials', str(trials), '--preset', 'stationary', '--no-status']
        if mode:
            cmd += ['--mode', mode]
        if shadow:
            cmd += ['--shadow']
        phases.append({
            'id': 'crypto_search',
            'label': 'Training Crypto Regression Model',
            'cmd': cmd,
            'trials': trials,
        })
        phases.append({
            'id': 'crypto_meta',
            'label': 'Training Crypto Meta-Labeler',
            'cmd': [PYTHON, '-u', 'meta_label.py'],
        })
        phases.append({
            'id': 'crypto_backtest_gate',
            'label': 'Crypto Policy Backtest Gate',
            # 44 days, not 60: the crypto training file spans ~1y, so the
            # untouched holdout is its final ~12% ~= 44 days. A 60d window
            # reached ~16 days INTO the search region (fold-3 validation,
            # LightGBM early-stop slices, meta-label training rows) — ~27%
            # of the gate was in-sample and biased toward passing. The stock
            # gate keeps 60d: its file spans 2021->now, so a 60d window sits
            # entirely inside that book's ~8-month holdout.
            'cmd': [PYTHON, '-u', 'backtest.py', '--days', '44',
                    '--trials', str(max(trials, 10)), '--gate'],
        })
    if train_stock:
        cmd = [PYTHON, '-u', os.path.join('scripts', 'hypersearch_v2.py'),
               '--trials', str(trials),
               '--data', 'stock_training_data.csv', '--prefix', 'stock',
               '--preset', 'stationary', '--max-rows', '200000', '--no-status']
        if mode:
            cmd += ['--mode', mode]
        if shadow:
            cmd += ['--shadow']
        phases.append({
            'id': 'stock_search',
            'label': 'Training Stock Regression Model',
            'cmd': cmd,
            'trials': trials,
        })
        phases.append({
            'id': 'stock_meta',
            'label': 'Training Stock Meta-Labeler',
            'cmd': [PYTHON, '-u', 'meta_label.py', '--prefix', 'stock'],
        })
        phases.append({
            'id': 'stock_backtest_gate',
            'label': 'Stock Policy Backtest Gate',
            'cmd': [PYTHON, '-u', 'backtest.py', '--prefix', 'stock',
                    '--days', '60', '--trials', str(max(trials, 10)), '--gate'],
        })

    return phases


def _resolve_retrain_trials(args, adaptive_trial_counts):
    """Trial-count precedence shared by the --no-retrain manual-trigger
    branch and the Phase-C weekly retrain branch: honor an explicit
    --retrain-trials override (anything != the 100 default) over the
    adaptive per-book trial counts (worst case across active books).
    """
    if args.retrain_trials != 100:
        return args.retrain_trials
    return max(adaptive_trial_counts.values()) if adaptive_trial_counts else 100


def _needs_force_harvest(train_crypto, train_stock):
    """True if an active book's data file is missing the max-forward-bars
    Target_Return column, plus the human-readable reasons (for the caller
    to log). The adaptive forward_bars space can expand between retrains;
    if the on-disk file predates that expansion, training would silently
    run against a stale label horizon until a re-harvest happens.

    Returns (needs_force: bool, reasons: list[str]).
    """
    import pandas as pd
    from data_utils import get_data_path

    reasons = []
    for at, active in (('crypto', train_crypto), ('stock', train_stock)):
        if not active:
            continue
        max_fb = get_max_forward_bars(at)
        data_path = get_data_path(at)
        if not data_path.exists():
            continue
        if str(data_path).endswith('.parquet'):
            import pyarrow.parquet as pq
            cols = pq.read_schema(data_path).names
        else:
            cols = pd.read_csv(data_path, nrows=0).columns.tolist()
        if f'Target_Return_{max_fb}' not in cols:
            reasons.append(f"{at}: Target_Return_{max_fb} missing from data,"
                           f" forcing re-harvest")
    return (len(reasons) > 0, reasons)


MAX_PHASE_RETRIES = 3
RETRY_WAIT_SECONDS = 30


def _phase_book(phase_id):
    """Which book a phase belongs to, from its id prefix ('crypto_...',
    'stock_...'). Used to scope failures to one book (C8) so a crypto
    failure doesn't silently skip the entire stock retrain."""
    if phase_id.startswith('crypto_'):
        return 'crypto'
    if phase_id.startswith('stock_'):
        return 'stock'
    return 'other'


def _run_training(phases, log_fh, status, is_retrain):
    """Run all training phases, grouped by book (crypto/stock).

    A genuine phase failure (after MAX_PHASE_RETRIES retries) skips only
    that book's REMAINING phases and continues with the next book's group
    — a crypto failure no longer silently skips the entire stock retrain.
    A *_backtest_gate phase returning 3 (deterministic policy rejection,
    model already rolled back — see backtest.py) is FINAL immediately: no
    retry, and the book counts as completed-with-rollback, not failed.

    Returns 'suspended' if a GUI suspend request landed mid-phase, else a
    dict {book: bool} of per-book success (a rc==3 gate verdict counts as
    success — the book's model is intact, just rolled back).
    """
    global _suspend_requested
    status.setdefault('phase_results', {})
    failed_books = set()
    book_success = {}
    for phase in phases:
        phase_id = phase['id']
        book = _phase_book(phase_id)
        if book in failed_books:
            msg = (f"\n[SKIP] {phase['label']} skipped — {book} book already"
                   f" failed this run\n")
            log_fh.write(msg)
            log_fh.flush()
            _print(msg, end='')
            continue

        rc = None
        attempts = 0
        gate_rejected = False
        for attempt in range(1, MAX_PHASE_RETRIES + 1):
            attempts = attempt
            rc = run_phase(phase, log_fh, status)
            if rc == -99:
                _suspend_requested = False
                status['phase'] = 'suspended'
                status['phase_label'] = 'Training suspended'
                write_status(status, force=True)
                return 'suspended'
            if rc == 0:
                break
            if phase_id.endswith('_backtest_gate') and rc == 3:
                # Deterministic policy rejection, model already rolled
                # back by backtest.py --gate. Retrying is useless.
                gate_rejected = True
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

        # Save final scores. Tri-state: None (JSON null) when the search
        # itself failed, instead of echoing a stale/zero best_score — the
        # GUI reads via .get and degrades safely on null.
        if phase_id == 'crypto_search':
            status['crypto_final_score'] = (status.get('best_score', 0)
                                            if rc == 0 else None)
        elif phase_id == 'stock_search':
            status['stock_final_score'] = (status.get('best_score', 0)
                                           if rc == 0 else None)

        if gate_rejected:
            outcome = 'gate_failed_rolled_back'
        elif rc == 0:
            outcome = 'ok'
        else:
            outcome = 'failed'
        status['phase_results'][phase_id] = {
            'rc': rc, 'attempts': attempts, 'outcome': outcome,
        }
        write_status(status, force=True)

        if outcome == 'failed':
            failed_books.add(book)
            book_success[book] = False
            tail = ('bots continue with existing models' if is_retrain
                    else 'continuing to bot phase with existing models')
            msg = (f"\nWARNING: {phase['label']} failed (exit {rc}), "
                   f"{tail}; remaining {book} phases skipped this run\n")
            log_fh.write(msg)
            log_fh.flush()
            _print(msg, end='')
        else:
            book_success.setdefault(book, True)
    return book_success


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
    """Terminate all tracked subprocesses and wait to avoid zombies."""
    for proc in procs:
        try:
            if proc and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
    for proc in procs:
        try:
            if proc and proc.poll() is None:
                proc.wait(timeout=5)
        except Exception:
            pass


# Global refs for signal handler cleanup
_all_handles = []
_all_procs = []
_all_bots = []
_shutdown_requested = False
_manually_stopped = set()    # Bot names user explicitly stopped (skip auto-restart)
_suspend_requested = False   # Set True when GUI requests training suspension
# In-flight training-phase subprocess (Optuna search etc). Set by run_phase
# right after Popen, cleared in its finally block. Lets _signal_handler
# terminate a live phase child on Ctrl+C/SIGTERM instead of blocking
# forever in proc.wait() on a child that never got its own signal.
_current_phase_proc = None
# (run_crypto, run_stock) as decided in main() — used by _check_restart_bots
# to rebuild the combined 'Bots' cmd with the correct --crypto-only /
# --stock-only scope flag after a crash-restart.
_BOT_SCOPE = (True, True)


def _signal_handler(signum, frame):
    """Graceful shutdown: stop bots, close handles, exit."""
    global _shutdown_requested
    if _shutdown_requested:
        return  # Prevent re-entry
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name
    _print(f"\n[PIPELINE] {sig_name} received, shutting down...")
    # Terminate an in-flight phase subprocess (e.g. a multi-hour Optuna
    # search) — the main thread is blocked in proc.wait() inside run_phase
    # and that child never receives a signal of its own, so without this
    # a Ctrl+C/SIGTERM during training hangs forever.
    if _current_phase_proc is not None:
        try:
            if _current_phase_proc.poll() is None:
                _current_phase_proc.terminate()
                try:
                    _current_phase_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _current_phase_proc.kill()
                    _current_phase_proc.wait(timeout=5)
        except Exception:
            pass
    # Stop bot processes
    for name, proc, fh in _all_bots:
        try:
            if proc.poll() is None:
                proc.terminate()
                _print(f"  Stopped {name} bot (PID {proc.pid})")
        except Exception:
            pass
    for name, proc, fh in _all_bots:
        try:
            if proc.poll() is None:
                proc.wait(timeout=3)
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
    parser.add_argument('--combined-bots', action='store_true',
                        help='Run both trading loops as threads in ONE process '
                             '(saves ~0.5-0.8GB RAM on the Jetson; per-bot GUI '
                             'restart commands then restart the whole process)')
    args = parser.parse_args()

    global _COMBINED_BOTS
    _COMBINED_BOTS = args.combined_bots

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)

    train_crypto = not args.stock_only
    train_stock = not args.crypto_only
    run_crypto = not args.stock_only
    run_stock = not args.crypto_only

    global _BOT_SCOPE
    _BOT_SCOPE = (run_crypto, run_stock)

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
        'crypto_bot_running': False,
        'stock_bot_running': False,
        # Pre-seeded (not created-then-popped) so the heartbeat thread's
        # concurrent json.dump(status) can never observe a mid-mutation
        # dict-size change ("dict changed size during iteration") — see
        # the read-then-assign-None sites below instead of status.pop(...).
        '_pending_bot_start': None,
        'phase_results': {},
        # gui.py reads this to preserve flags across GUI-triggered restarts.
        'launch_args': sys.argv[1:],
    }

    log_fh = open(LOG_FILE, 'a')
    _all_handles.append(log_fh)

    global _heartbeat_status
    _heartbeat_status = status
    hb = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb.start()
    _sd_notify(b'READY=1')

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
        _launch_bots(bots, log_fh, run_crypto, run_stock, verb='started')

        status['phase'] = 'trading'
        status['phase_label'] = 'Trading'
        _update_per_bot_status(bots, status)
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
                # Poll every 5s for commands, check bots/triggers each 60s cycle
                trigger = None
                for _ in range(12):
                    if _shutdown_requested:
                        break
                    time.sleep(5)
                    cmd = _check_pipeline_command()
                    if cmd:
                        _handle_command(cmd, bots, log_fh, status)
                _check_restart_bots(bots, log_fh)
                _update_per_bot_status(bots, status)
                write_status(status)
                # Mirrors the Phase-C weekly loop: without these, the
                # Telegram /halt //flatten kill switch and the daily PSI
                # drift + shadow-challenger check silently never ran in
                # --no-retrain mode.
                _maybe_run_drift_check(log_fh)
                _check_telegram_commands(log_fh, status)
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
                    # Same precedence as the Phase-C weekly branch (C1 fix:
                    # --no-retrain used to always ignore --retrain-trials).
                    retrain_trials = _resolve_retrain_trials(args, adaptive_trial_counts)
                    retrain_mode = ('' if args.retrain_trials != 100 else
                                    ('explore' if 'explore' in adaptive_modes.values()
                                     else 'refine'))
                    # Same schema check as the Phase-C weekly branch (C1
                    # fix: --no-retrain used to never force a re-harvest
                    # after a forward_bars expansion).
                    force_harvest, force_reasons = _needs_force_harvest(rt_crypto, rt_stock)
                    for r in force_reasons:
                        msg = f"[ADAPTIVE] {r}\n"
                        log_fh.write(msg)
                        log_fh.flush()
                        _print(msg, end='')
                    retrain_phases = (
                        _build_harvest_phases(False, rt_crypto, rt_stock,
                                              force=force_harvest)
                        + _build_training_phases(retrain_trials, rt_crypto, rt_stock,
                                                 mode=retrain_mode, shadow=True)
                    )
                    for i, p in enumerate(retrain_phases):
                        p['idx'] = i
                    status['started_at'] = datetime.datetime.now().isoformat()
                    status['phases'] = [p['id'] for p in retrain_phases]
                    status['phase_labels'] = {p['id']: p['label'] for p in retrain_phases}
                    status['total_phases'] = len(retrain_phases)
                    status['retrain_cycle'] = manual_cycle
                    # Stop bots to free GPU memory for training
                    _stop_bots(bots, log_fh)
                    status['bots_running'] = False
                    _update_per_bot_status(bots, status)
                    write_status(status, force=True)
                    result = _run_training(retrain_phases, log_fh, status, is_retrain=True)
                    if result == 'suspended':
                        # Read-then-assign-None, not pop: the key is
                        # pre-seeded in main() so this never changes the
                        # dict's size out from under the heartbeat
                        # thread's concurrent json.dump(status).
                        pending = status.get('_pending_bot_start') or {}
                        status['_pending_bot_start'] = None
                        if pending.get('crypto'):
                            _start_single_bot(bots, 'Crypto', log_fh)
                        if pending.get('stock'):
                            _start_single_bot(bots, 'Stock', log_fh)
                    else:
                        # Restart all bots after normal training completion
                        _restart_bots(bots, log_fh, run_crypto, run_stock)
                    try:
                        from sentiment_history import set_live_mode
                        set_live_mode(True)
                    except Exception:
                        pass
                    status['phase'] = 'trading'
                    status['phase_label'] = 'Trading (no auto-retrain)'
                    _update_per_bot_status(bots, status)
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
                # Poll every 5s for commands, check bots/triggers each 60s cycle
                for _ in range(12):
                    if _shutdown_requested:
                        break
                    time.sleep(5)
                    cmd = _check_pipeline_command()
                    if cmd:
                        _handle_command(cmd, bots, log_fh, status)
                _check_restart_bots(bots, log_fh)
                _update_per_bot_status(bots, status)
                write_status(status)
                _maybe_run_drift_check(log_fh)
                _check_telegram_commands(log_fh, status)
                manual_trigger = _check_retrain_trigger()
                if manual_trigger:
                    msg = (f"\n[MANUAL RETRAIN] Triggered from GUI: "
                           f"crypto={manual_trigger.get('crypto', False)}, "
                           f"stock={manual_trigger.get('stock', False)}\n")
                    log_fh.write(msg)
                    log_fh.flush()
                    _print(msg, end='')
                    break
                manual_trigger = _check_drift_trigger()
                if manual_trigger:
                    msg = (f"\n[DRIFT RETRAIN] PSI drift monitor requested: "
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
            retrain_trials = _resolve_retrain_trials(args, adaptive_trial_counts)
            retrain_mode = ('' if args.retrain_trials != 100 else
                            # Use explore if any asset needs it
                            ('explore' if 'explore' in adaptive_modes.values()
                             else 'refine'))

            # Check if harvest needed due to forward_bars expansion
            force_harvest, force_reasons = _needs_force_harvest(rt_crypto, rt_stock)
            for r in force_reasons:
                msg = f"[ADAPTIVE] {r}\n"
                log_fh.write(msg)
                log_fh.flush()
                _print(msg, end='')

            # Weekly retrains MUST re-harvest: the docstring always promised
            # "re-harvest + retrain", but skip_harvest=not force_harvest
            # meant every weekly cycle re-ran Optuna on the same frozen CSV,
            # so models drifted months behind the live market. The <24h
            # freshness check inside _build_harvest_phases still prevents
            # back-to-back redundant harvests.
            retrain_phases = (
                _build_harvest_phases(False, rt_crypto, rt_stock,
                                      force=force_harvest)
                + _build_training_phases(retrain_trials, rt_crypto, rt_stock,
                                         mode=retrain_mode, shadow=True)
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
                f"# Bots stop during training, restart after\n"
                f"# Phases: {', '.join(p['label'] for p in retrain_phases)}\n"
                f"# Adaptive: mode={retrain_mode}, trials={retrain_trials}\n"
                f"{'#'*70}\n"
            )
            log_fh.write(banner)
            log_fh.flush()
            _print(banner, end='')

            # Stop bots to free GPU memory for training
            _stop_bots(bots, log_fh)
            status['bots_running'] = False
            _update_per_bot_status(bots, status)
            write_status(status, force=True)

            result = _run_training(retrain_phases, log_fh, status, is_retrain=True)

            if result == 'suspended':
                # Read-then-assign-None, not pop: see the no-retrain
                # branch above — keeps the dict size stable for the
                # heartbeat thread's concurrent json.dump(status).
                pending = status.get('_pending_bot_start') or {}
                status['_pending_bot_start'] = None
                if pending.get('crypto'):
                    _start_single_bot(bots, 'Crypto', log_fh)
                if pending.get('stock'):
                    _start_single_bot(bots, 'Stock', log_fh)
            else:
                # Restart all bots after normal training completion
                _restart_bots(bots, log_fh, run_crypto, run_stock)

            _update_per_bot_status(bots, status)

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
