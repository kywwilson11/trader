"""Tests for GPU lock coordination."""
import os
import sys
import json
import multiprocessing
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gpu_lock


@pytest.fixture(autouse=True)
def _lock_sandbox(tmp_path, monkeypatch):
    """Keep the lock files out of the repo root.

    gpu_lock._LOCK_FILE / _INFO_FILE are module globals anchored to the
    module's own directory, and both are created at call time, so an
    unsandboxed run leaves <repo>/.gpu.lock behind (and, if a test ever dies
    between acquire and release, <repo>/.gpu_lock_info.json — which is not
    even gitignored). Same pattern as tests/test_review_b19.py::lock_sandbox.
    The child processes below re-import gpu_lock (macOS spawns rather than
    forks), so they are handed these paths explicitly — parent and child must
    contend for ONE file or the cross-process assertions mean nothing.
    """
    monkeypatch.setattr(gpu_lock, '_LOCK_FILE', tmp_path / '.gpu.lock')
    monkeypatch.setattr(gpu_lock, '_INFO_FILE',
                        tmp_path / '.gpu_lock_info.json')
    return tmp_path


def test_is_gpu_free_when_unlocked():
    """GPU should be free when no lock is held."""
    assert gpu_lock.is_gpu_free() is True


def test_acquire_and_release():
    """Lock should be held inside context, free after exit."""
    with gpu_lock.acquire_for_training("test"):
        assert gpu_lock.is_gpu_free() is False
        info = gpu_lock.get_lock_info()
        assert info is not None
        assert info['owner'] == 'test'
        assert info['pid'] == os.getpid()
    assert gpu_lock.is_gpu_free() is True


def _child_acquire(lock_path, info_path, ready_event, release_event):
    """Helper: acquire lock in child process, signal ready, wait for release.

    The spawned child re-imports gpu_lock, so the parent's sandbox paths are
    re-applied here rather than inherited.
    """
    gpu_lock._LOCK_FILE = Path(lock_path)
    gpu_lock._INFO_FILE = Path(info_path)
    with gpu_lock.acquire_for_training("child_process"):
        ready_event.set()
        release_event.wait(timeout=10)


def test_cross_process_lock():
    """Lock should work across processes."""
    ready = multiprocessing.Event()
    release = multiprocessing.Event()

    p = multiprocessing.Process(
        target=_child_acquire,
        args=(str(gpu_lock._LOCK_FILE), str(gpu_lock._INFO_FILE),
              ready, release))
    p.start()

    # Wait for child to acquire
    ready.wait(timeout=5)
    assert gpu_lock.is_gpu_free() is False

    info = gpu_lock.get_lock_info()
    assert info is not None
    assert info['owner'] == 'child_process'

    # Release child
    release.set()
    p.join(timeout=5)
    assert gpu_lock.is_gpu_free() is True


def _child_crash(lock_path):
    """Helper: acquire lock then crash (OS should release flock)."""
    # Acquire lock via context manager, then hard-exit without releasing
    gpu_lock._LOCK_FILE = Path(lock_path)   # spawned child re-imports gpu_lock
    gpu_lock._LOCK_FILE.touch(exist_ok=True)
    import fcntl
    fd = open(gpu_lock._LOCK_FILE, 'w')
    fcntl.flock(fd, fcntl.LOCK_EX)
    os._exit(1)


def test_lock_released_on_crash():
    """Lock should be released when process crashes (OS releases flock)."""
    p = multiprocessing.Process(target=_child_crash,
                                args=(str(gpu_lock._LOCK_FILE),))
    p.start()
    p.join(timeout=5)

    # Give OS a moment to clean up
    time.sleep(0.1)
    assert gpu_lock.is_gpu_free() is True


def test_lock_info_cleared_on_release():
    """Lock info file should be cleaned up after release."""
    with gpu_lock.acquire_for_training("cleanup_test"):
        assert gpu_lock.get_lock_info() is not None
    # Info should be cleared after context exit
    assert gpu_lock.get_lock_info() is None


def test_gpu_lock_status_string():
    """Status string should reflect lock state."""
    status = gpu_lock.gpu_lock_status()
    assert "free" in status

    with gpu_lock.acquire_for_training("status_test"):
        status = gpu_lock.gpu_lock_status()
        assert "status_test" in status
        assert "locked" in status


def test_choose_inference_device_always_cpu():
    """Trading bots should always use CPU for inference."""
    from trading_utils import choose_inference_device
    assert choose_inference_device() == 'cpu'
