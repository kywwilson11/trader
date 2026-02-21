"""GPU mutex for Jetson Orin Nano — prevents CUDA OOM from concurrent GPU processes.

The Jetson has 8GB unified memory shared between CPU and GPU. Running training
and inference simultaneously causes NvMapMemAllocInternalTagged errors and
unrecoverable CUDA allocator crashes.

This module provides a file-based exclusive lock using fcntl.flock():
  - Training processes MUST acquire the lock before using GPU
  - Inference processes CHECK the lock; if held, fall back to CPU
  - The OS automatically releases the lock when a process exits or crashes
    (no stale lock files to clean up)

Usage in training scripts:
    with gpu_lock.acquire_for_training("hypersearch_v2"):
        # GPU is exclusively yours
        train_model(...)

Usage in inference:
    if gpu_lock.is_gpu_free():
        device = 'cuda'
    else:
        device = 'cpu'  # training is running, use CPU
"""

import fcntl
import json
import os
import time
from pathlib import Path
from contextlib import contextmanager

_LOCK_FILE = Path(__file__).resolve().parent / '.gpu.lock'
_INFO_FILE = Path(__file__).resolve().parent / '.gpu_lock_info.json'

# Module-level file descriptor — kept open for the lifetime of the lock
_lock_fd = None


def _write_info(owner: str):
    """Write lock metadata (who holds it, when, PID)."""
    info = {
        'owner': owner,
        'pid': os.getpid(),
        'acquired_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    try:
        with open(_INFO_FILE, 'w') as f:
            json.dump(info, f)
    except OSError:
        pass


def _clear_info():
    """Remove lock metadata."""
    try:
        _INFO_FILE.unlink(missing_ok=True)
    except OSError:
        pass


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

    Blocks until the lock is available. Only one training process can hold
    the lock at a time. The lock is automatically released when the context
    exits (normally or via exception/crash).

    Args:
        owner: Label for who holds the lock (e.g. "hypersearch_v2_crypto")

    Raises:
        RuntimeError: If lock acquisition fails
    """
    global _lock_fd

    # Create lock file if it doesn't exist
    _LOCK_FILE.touch(exist_ok=True)

    fd = open(_LOCK_FILE, 'w')
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

        _lock_fd = fd
        _write_info(owner)
        print(f"[GPU-LOCK] Acquired by '{owner}' (PID {os.getpid()})")
        yield
    finally:
        _clear_info()
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
        _lock_fd = None
        print(f"[GPU-LOCK] Released by '{owner}'")


def try_acquire_for_training(owner: str = "training") -> bool:
    """Non-blocking attempt to acquire GPU lock.

    Returns True if lock acquired (caller MUST call release_training_lock
    when done). Returns False if GPU is busy.
    """
    global _lock_fd

    _LOCK_FILE.touch(exist_ok=True)
    fd = open(_LOCK_FILE, 'w')
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd = fd
        _write_info(owner)
        print(f"[GPU-LOCK] Acquired by '{owner}' (PID {os.getpid()})")
        return True
    except BlockingIOError:
        fd.close()
        return False
    except Exception:
        fd.close()
        raise


def release_training_lock():
    """Release a lock acquired via try_acquire_for_training()."""
    global _lock_fd
    if _lock_fd is not None:
        _clear_info()
        fcntl.flock(_lock_fd, fcntl.LOCK_UN)
        _lock_fd.close()
        _lock_fd = None


def is_gpu_free() -> bool:
    """Check if GPU is available (no training process holds the lock).

    This is a non-blocking check for inference processes. Returns True if
    the GPU is free, False if a training process holds the lock.
    """
    _LOCK_FILE.touch(exist_ok=True)
    fd = open(_LOCK_FILE, 'r')
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Lock acquired = nobody else has it = GPU is free
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        # Lock held by someone = GPU is busy
        return False
    finally:
        fd.close()


def gpu_lock_status() -> str:
    """Human-readable lock status for logging/GUI."""
    if is_gpu_free():
        return "GPU: free"
    info = get_lock_info()
    if info:
        return f"GPU: locked by '{info['owner']}' (PID {info['pid']}) since {info['acquired_at']}"
    return "GPU: locked (unknown holder)"
