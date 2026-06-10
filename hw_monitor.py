"""Hardware monitoring utilities for Jetson Orin Nano 8GB.

Provides GPU temperature, RAM usage, and GPU availability checks.

Temperature comes from sysfs thermal zones (microseconds, no subprocess).
The old implementation spawned never-exiting `tegrastats` with timeout=2,
which ALWAYS blocked the full 2s, raised TimeoutExpired, discarded the
output, and fell back to a hardcoded thermal_zone1 anyway — burning ~1.6h
of dead time and ~2,880 forked processes per bot per day for zero data.
"""
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


def _find_gpu_thermal_zone() -> str | None:
    """Locate the GPU thermal zone by reading each zone's `type` file.

    Zone numbering differs across JetPack releases — hardcoding
    thermal_zone1 silently reads the wrong sensor on some boards.
    Falls back to zone1, then zone0.
    """
    global _zone_path_cache, _zone_scan_done
    if _zone_scan_done:
        return _zone_path_cache
    _zone_scan_done = True
    try:
        for name in sorted(os.listdir(_THERMAL_DIR)):
            if not name.startswith('thermal_zone'):
                continue
            type_path = os.path.join(_THERMAL_DIR, name, 'type')
            try:
                with open(type_path) as f:
                    zone_type = f.read().strip().lower()
                if any(h in zone_type for h in _GPU_ZONE_HINTS):
                    _zone_path_cache = os.path.join(_THERMAL_DIR, name, 'temp')
                    return _zone_path_cache
            except OSError:
                continue
    except OSError:
        pass
    # Fallbacks: historical default, then zone0 (CPU — close enough for
    # thermal throttling decisions on a shared-die SoC)
    for zone in ('thermal_zone1', 'thermal_zone0'):
        path = os.path.join(_THERMAL_DIR, zone, 'temp')
        if os.path.exists(path):
            _zone_path_cache = path
            return _zone_path_cache
    _zone_path_cache = None
    return None


def get_gpu_temp():
    """GPU temperature in Celsius from sysfs, cached ~20s.

    Returns float or None if unavailable (e.g. not running on a Jetson).
    """
    global _temp_cache
    now = time.monotonic()
    ts, cached = _temp_cache
    if cached is not None and (now - ts) < _TEMP_CACHE_TTL:
        return cached

    path = _find_gpu_thermal_zone()
    if path is None:
        return None
    try:
        with open(path) as f:
            temp = int(f.read().strip()) / 1000.0
        _temp_cache = (now, temp)
        return temp
    except (FileNotFoundError, ValueError, OSError):
        return None


def get_ram_usage():
    """Return (used_mb, total_mb) from /proc/meminfo."""
    try:
        with open('/proc/meminfo') as f:
            info = f.read()
        total = int(re.search(r'MemTotal:\s+(\d+)', info).group(1)) / 1024.0
        available = int(re.search(r'MemAvailable:\s+(\d+)', info).group(1)) / 1024.0
        used = total - available
        return round(used, 1), round(total, 1)
    except (FileNotFoundError, AttributeError, ValueError):
        return None, None


def is_gpu_available():
    """Try a small CUDA allocation to check if GPU is usable (not OOM).

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
    Returns the final temperature.
    """
    while True:
        temp = get_gpu_temp()
        if temp is None:
            print(f"[HW] Cannot read GPU temp, proceeding anyway")
            return None
        if temp < max_temp:
            return temp
        print(f"[HW] GPU temp {temp:.0f}C > {max_temp}C, waiting {poll_interval}s...")
        time.sleep(poll_interval)


if __name__ == '__main__':
    temp = get_gpu_temp()
    print(f"GPU Temperature: {temp}C" if temp else "GPU Temperature: unavailable")

    used, total = get_ram_usage()
    if used is not None:
        print(f"RAM Usage: {used:.0f} MB / {total:.0f} MB ({used/total*100:.0f}%)")
    else:
        print("RAM Usage: unavailable")

    gpu = is_gpu_available()
    print(f"GPU Available: {gpu}")
