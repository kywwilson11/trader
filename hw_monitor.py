"""Hardware monitoring utilities for Jetson Orin Nano 8GB.

Provides GPU temperature, RAM usage, and GPU availability checks.

Temperature comes from sysfs thermal zones (microseconds, no subprocess).
The old implementation spawned never-exiting `tegrastats` with timeout=2,
which ALWAYS blocked the full 2s, raised TimeoutExpired, discarded the
output, and fell back to a hardcoded thermal_zone1 anyway — burning ~1.6h
of dead time and ~2,880 forked processes per bot per day for zero data.
"""
import logging
import os
import re
import time

# torch is imported lazily inside is_gpu_available(): importing it at module
# scope forces every consumer of get_gpu_temp() (both bots, the GUI) to pay
# the full torch import even when they never touch the GPU.

_GPU_ZONE_HINTS = ('gpu', 'gpu-thermal', 'gpu_thermal')
_THERMAL_DIR = '/sys/devices/virtual/thermal'

_zone_path_cache: str | None = None
_zone_scan_done = False
_temp_cache: tuple[float, float | None] = (0.0, None)
_TEMP_CACHE_TTL = 20.0  # seconds
_warned_unavailable = False


def _warn_temp_unavailable_once(reason: str) -> None:
    """One-shot per process: base_loop thermal throttling and the
    run_pipeline thermal gate both fail open on None, so losing the
    sensor silently disables thermal protection — the transition must
    leave a trace (but not one line per 30s cycle)."""
    global _warned_unavailable
    if not _warned_unavailable:
        _warned_unavailable = True
        logging.getLogger(__name__).warning(
            "GPU temperature unavailable (%s) — thermal throttling/gate "
            "run without temp data", reason)


def _find_gpu_thermal_zone() -> str | None:
    """Locate the GPU thermal zone by reading each zone's `type` file.

    Zone numbering differs across JetPack releases — hardcoding
    thermal_zone1 silently reads the wrong sensor on some boards.
    Falls back to zone1, then zone0.
    """
    global _zone_path_cache, _zone_scan_done
    if _zone_scan_done:
        return _zone_path_cache
    # _zone_scan_done is set only AFTER the scan populates the cache: a
    # second thread entering mid-scan (combined-bots mode) redoes the
    # idempotent scan instead of reading the not-yet-populated cache as
    # a spurious None.
    found = None
    try:
        for name in sorted(os.listdir(_THERMAL_DIR)):
            if not name.startswith('thermal_zone'):
                continue
            type_path = os.path.join(_THERMAL_DIR, name, 'type')
            try:
                with open(type_path) as f:
                    zone_type = f.read().strip().lower()
                if any(h in zone_type for h in _GPU_ZONE_HINTS):
                    found = os.path.join(_THERMAL_DIR, name, 'temp')
                    break
            except OSError:
                continue
    except OSError:
        pass
    if found is None:
        # Fallbacks: historical default, then zone0 (CPU — close enough for
        # thermal throttling decisions on a shared-die SoC)
        for zone in ('thermal_zone1', 'thermal_zone0'):
            path = os.path.join(_THERMAL_DIR, zone, 'temp')
            if os.path.exists(path):
                found = path
                break
    _zone_path_cache = found
    _zone_scan_done = True
    return _zone_path_cache


def get_gpu_temp():
    """GPU temperature in Celsius from sysfs, cached ~20s.

    Returns float or None if unavailable (e.g. not running on a Jetson);
    the first None logs a one-shot warning.
    """
    global _temp_cache
    now = time.monotonic()
    ts, cached = _temp_cache
    if cached is not None and (now - ts) < _TEMP_CACHE_TTL:
        return cached

    path = _find_gpu_thermal_zone()
    if path is None:
        _warn_temp_unavailable_once("no GPU thermal zone found")
        return None
    try:
        with open(path) as f:
            temp = int(f.read().strip()) / 1000.0
        _temp_cache = (now, temp)
        return temp
    except (ValueError, OSError) as e:
        _warn_temp_unavailable_once(f"zone read failed: {e}")
        return None


def get_ram_usage():
    """Return (used_mb, total_mb) from /proc/meminfo, or (None, None) if
    it is missing/unparsable (e.g. not running on Linux)."""
    try:
        with open('/proc/meminfo') as f:
            info = f.read()
        total = int(re.search(r'MemTotal:\s+(\d+)', info).group(1)) / 1024.0
        available = int(re.search(r'MemAvailable:\s+(\d+)', info).group(1)) / 1024.0
        used = total - available
        return round(used, 1), round(total, 1)
    except (OSError, AttributeError, ValueError):
        return None, None


def is_gpu_available():
    """Try a small CUDA allocation to check if GPU is usable (not OOM).

    Diagnostic only — no production callers; used by `python hw_monitor.py`
    and tests.

    WARNING: a successful check permanently initializes a CUDA context in
    THIS process (~0.6-1.2GB on Jetson unified memory). Bot processes run
    with CUDA_VISIBLE_DEVICES='' so this short-circuits to False without
    paying that cost.
    """
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) == '':
        return False
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(1, device='cuda')
        del t
        return True
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return False


def wait_for_cool_gpu(max_temp=70, poll_interval=30):
    """Block until GPU temperature drops below max_temp.

    Returns the final temperature, or None immediately (fail-open) if the
    sensor is unavailable. poll_interval should exceed _TEMP_CACHE_TTL
    (20s) — a shorter interval would re-read the cached value instead of
    fresh sysfs data.
    """
    while True:
        temp = get_gpu_temp()
        if temp is None:
            print("[HW] Cannot read GPU temp, proceeding anyway")
            return None
        if temp < max_temp:
            return temp
        print(f"[HW] GPU temp {temp:.0f}C > {max_temp}C, waiting {poll_interval}s...")
        time.sleep(poll_interval)


if __name__ == '__main__':
    temp = get_gpu_temp()
    print(f"GPU Temperature: {temp}C" if temp is not None
          else "GPU Temperature: unavailable")

    used, total = get_ram_usage()
    if used is not None:
        print(f"RAM Usage: {used:.0f} MB / {total:.0f} MB ({used/total*100:.0f}%)")
    else:
        print("RAM Usage: unavailable")

    gpu = is_gpu_available()
    print(f"GPU Available: {gpu}")
