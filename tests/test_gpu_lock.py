"""Tests for GPU lock coordination."""
import os
import sys
import json
import multiprocessing
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gpu_lock


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


def test_try_acquire_blocks_second():
    """Second try_acquire should fail when first holds lock."""
    ok = gpu_lock.try_acquire_for_training("first")
    assert ok is True
    assert gpu_lock.is_gpu_free() is False

    ok2 = gpu_lock.try_acquire_for_training("second")
    assert ok2 is False

    gpu_lock.release_training_lock()
    assert gpu_lock.is_gpu_free() is True


def _child_acquire(ready_event, release_event):
    """Helper: acquire lock in child process, signal ready, wait for release."""
    with gpu_lock.acquire_for_training("child_process"):
        ready_event.set()
        release_event.wait(timeout=10)


def test_cross_process_lock():
    """Lock should work across processes."""
    ready = multiprocessing.Event()
    release = multiprocessing.Event()

    p = multiprocessing.Process(target=_child_acquire, args=(ready, release))
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


def _child_crash():
    """Helper: acquire lock then crash (OS should release flock)."""
    gpu_lock.try_acquire_for_training("crash_test")
    # Exit without releasing — OS should release the flock
    os._exit(1)


def test_lock_released_on_crash():
    """Lock should be released when process crashes (OS releases flock)."""
    p = multiprocessing.Process(target=_child_crash)
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
