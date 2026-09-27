#!/usr/bin/env python3
"""Build the C extension for technical indicators.

Usage:
    python c_ext/build.py          # Build in-place (shared lib goes to project root)
    python c_ext/build.py --test   # Build + run quick validation

The compiled .so file is placed in the project root so indicators.py can import it.
"""
import os
import sys
import sysconfig
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
C_SRC = Path(__file__).resolve().parent / "indicators_c.c"


def get_build_cmd():
    """Construct gcc command for building the C extension."""
    python_inc = sysconfig.get_config_var('INCLUDEPY')
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
    output = PROJECT_ROOT / f"indicators_c{ext_suffix}"

    # Get numpy include path
    try:
        import numpy
        numpy_inc = numpy.get_include()
    except ImportError:
        print("ERROR: numpy not found. Install numpy first.")
        sys.exit(1)

    cmd = [
        "gcc",
        "-O3",                          # Maximum optimization
        "-march=native",                # ARM64 NEON auto-vectorization
        "-fno-math-errno",              # Skip errno for math (safe speedup)
        "-shared",
        "-fPIC",
        f"-I{python_inc}",
        f"-I{numpy_inc}",
        str(C_SRC),
        "-o", str(output),
        "-lm",                          # libm for math functions
    ]
    return cmd, output


def build():
    """Compile the C extension."""
    cmd, output = get_build_cmd()
    print(f"Building: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"BUILD FAILED:\n{result.stderr}")
        sys.exit(1)
    print(f"Built: {output} ({output.stat().st_size / 1024:.1f} KB)")
    return output


def test():
    """Quick validation that the C extension produces correct results."""
    sys.path.insert(0, str(PROJECT_ROOT))
    import numpy as np

    # Import C extension
    import indicators_c as ic

    # Test data: random walk with noise (ensures both gains and losses early)
    np.random.seed(42)
    n = 500
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5
    volume = np.random.rand(n) * 1000 + 100

    # RSI
    rsi = ic.rsi(close, 14)
    assert rsi.shape == (n,), f"RSI shape: {rsi.shape}"
    valid_rsi = rsi[~np.isnan(rsi)]
    assert len(valid_rsi) > 0, "RSI all NaN"
    assert np.all((valid_rsi >= 0) & (valid_rsi <= 100)), "RSI out of range"
    print(f"  RSI: OK (range {valid_rsi.min():.1f}-{valid_rsi.max():.1f})")

    # MACD
    macd, hist, sig = ic.macd(close, 12, 26, 9)
    assert macd.shape == (n,), f"MACD shape: {macd.shape}"
    print(f"  MACD: OK")

    # ATR
    atr = ic.atr(high, low, close, 14)
    assert atr.shape == (n,), f"ATR shape: {atr.shape}"
    valid_atr = atr[~np.isnan(atr)]
    assert np.all(valid_atr >= 0), "ATR negative"
    print(f"  ATR: OK (mean={valid_atr.mean():.4f})")

    # BBands
    lo, mid, up, bw, pb = ic.bbands(close, 20, 2.0)
    assert lo.shape == (n,) and up.shape == (n,), "BBands shape"
    print(f"  BBands: OK")

    # Stochastic
    sk, sd = ic.stoch(high, low, close, 14, 3, 3)
    assert sk.shape == (n,), "Stoch shape"
    print(f"  Stoch: OK")

    # OBV
    obv = ic.obv(close, volume)
    assert obv.shape == (n,), "OBV shape"
    print(f"  OBV: OK")

    # Rolling percentile
    rp = ic.rolling_percentile(close, 100)
    assert rp.shape == (n,), "Rolling percentile shape"
    print(f"  Rolling Percentile: OK")

    # Linear slope
    ls = ic.linear_slope(close, 5)
    assert ls.shape == (n,), "Linear slope shape"
    print(f"  Linear Slope: OK")

    # Hurst
    h = ic.hurst(close, 100)
    assert h.shape == (n,), "Hurst shape"
    valid_h = h[~np.isnan(h)]
    assert np.all((valid_h >= 0) & (valid_h <= 1)), f"Hurst out of range: {valid_h.min()}-{valid_h.max()}"
    print(f"  Hurst: OK (mean={valid_h.mean():.3f})")

    print("\nAll tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run validation after build")
    args = parser.parse_args()

    output = build()
    if args.test:
        print("\nRunning validation...")
        test()
