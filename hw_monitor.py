"""Hardware monitoring utilities for Jetson Orin Nano 8GB.

Provides GPU temperature, RAM usage, and GPU availability checks
via tegrastats (no sudo required).
"""
import subprocess
import re
import time
import torch


def get_gpu_temp():
    """Parse tegrastats output to get GPU temperature in Celsius.
    Returns float or None if unavailable.
    """
    try:
        proc = subprocess.run(
            ['tegrastats', '--interval', '100'],
            capture_output=True, text=True, timeout=2,
        )
        output = proc.stdout + proc.stderr
        # tegrastats format includes e.g. "GPU@55C" or "gpu@55C"
        match = re.search(r'GPU@(\d+(?:\.\d+)?)C', output, re.IGNORECASE)
        if match:
            return float(match.group(1))
        # Fallback: look for thermal zone
        match = re.search(r'tj@(\d+(?:\.\d+)?)C', output, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: read thermal zone directly
    try:
        with open('/sys/devices/virtual/thermal/thermal_zone1/temp') as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError, OSError):
        pass

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
    """Try a small CUDA allocation to check if GPU is usable (not OOM)."""
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
