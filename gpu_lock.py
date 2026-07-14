"""GPU mutex for Jetson Orin Nano — prevents CUDA OOM from concurrent GPU processes.

The Jetson has 8GB unified memory shared between CPU and GPU. Running training
and inference simultaneously causes NvMapMemAllocInternalTagged errors and
unrecoverable CUDA allocator crashes.

This module provides a file-based exclusive lock using fcntl.flock():
  - Training processes MUST acquire the lock before using GPU
    (scripts/hypersearch_v2.py is the production caller)
  - Live inference never consults this lock: the bots are CPU-always by
    design (trading_utils.choose_inference_device() hard-returns 'cpu'
    and run_bots.py sets CUDA_VISIBLE_DEVICES='')
  - is_gpu_free() / gpu_lock_status() / get_lock_info() are ops/debug
    utilities with no production callers (tests + manual use only)
  - The OS automatically releases the lock when a process exits or crashes
    (no stale lock files to clean up)
  - The lock is per-process and non-reentrant: a second
    acquire_for_training() in the same process (any thread) blocks forever

Usage in training scripts:
    with gpu_lock.acquire_for_training("hypersearch_v2"):
        # GPU is exclusively yours
        train_model(...)
"""

import fcntl
import json
import os
import time
from pathlib import Path
from contextlib import contextmanager

_LOCK_FILE = Path(__file__).resolve().parent / '.gpu.lock'
_INFO_FILE = Path(__file__).resolve().parent / '.gpu_lock_info.json'


def _write_info(owner: str):
    """Write lock metadata (who holds it, when, PID). Atomic via tmp +
    os.replace so readers never see a partial file. Non-fatal: metadata
    must never break the lock itself."""
    info = {
        'owner': owner,
        'pid': os.getpid(),
        'acquired_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    try:
        tmp = str(_INFO_FILE) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(info, f)
        os.replace(tmp, _INFO_FILE)
    except OSError as e:
        print(f"[GPU-LOCK] warn: could not write info file: {e}")


def _clear_info():
    """Remove lock metadata. Non-fatal: metadata must never break the lock."""
    try:
        _INFO_FILE.unlink(missing_ok=True)
    except OSError as e:
        print(f"[GPU-LOCK] warn: could not remove info file: {e}")


def get_lock_info() -> dict | None:
    """Read lock metadata. Returns None if no info available."""
    try:
        with open(_INFO_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@contextmanager
def acquire_for_training(owner: str = "training"):
    """Context manager: acquire exclusive GPU lock for training.

    Blocks (possibly forever) until the lock is available. Only one
    training process can hold the lock at a time; non-reentrant within a
    process. The lock is automatically released when the context exits
    (normally or via exception/crash). OSError from open()/flock()
    propagates to the caller.

    Args:
        owner: Label for who holds the lock (e.g. "hypersearch_v2_crypto")
    """
    fd = open(_LOCK_FILE, 'w')  # 'w' creates the lock file if missing
    acquired = False
    try:
        # Try non-blocking first to give a clear message
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            info = get_lock_info()
            holder = info.get('owner', 'unknown') if info else 'unknown'
            holder_pid = info.get('pid', '?') if info else '?'
            print(f"[GPU-LOCK] GPU locked by '{holder}' (PID {holder_pid}), waiting...")
            # Now block until available
            fcntl.flock(fd, fcntl.LOCK_EX)
            print(f"[GPU-LOCK] Lock acquired after waiting")

        acquired = True
        _write_info(owner)
        print(f"[GPU-LOCK] Acquired by '{owner}' (PID {os.getpid()})")
        yield
    finally:
        # Cleanup only if we actually got the lock — a waiter interrupted
        # mid-flock must not delete the real holder's info file or print
        # a false 'Released'.
        if acquired:
            _clear_info()
            fcntl.flock(fd, fcntl.LOCK_UN)
            print(f"[GPU-LOCK] Released by '{owner}'")
        fd.close()


def is_gpu_free() -> bool:
    """Check if GPU is available (no training process holds the lock).

    Non-blocking ops/debug probe (no production callers). Returns True if
    the GPU is free, False if a training process holds the lock. Uses a
    SHARED probe so concurrent probes never report each other as 'busy'.

    Inherently TOCTOU-racy: 'free now' does not mean 'free when you
    allocate' — the answer can be stale by the time the caller acts.
    """
    _LOCK_FILE.touch(exist_ok=True)
    fd = open(_LOCK_FILE, 'r')
    try:
        # LOCK_SH succeeds iff no exclusive (training) holder
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        # Exclusive lock held by someone = GPU is busy
        return False
    finally:
        fd.close()


def gpu_lock_status() -> str:
    """Human-readable lock status (ops/debug utility — no production
    callers). Racy between the free-probe and the info read; tolerates
    incomplete info files."""
    if is_gpu_free():
        return "GPU: free"
    info = get_lock_info()
    if info:
        return (f"GPU: locked by '{info.get('owner', '?')}' "
                f"(PID {info.get('pid', '?')}) "
                f"since {info.get('acquired_at', '?')}")
    return "GPU: locked (unknown holder)"
