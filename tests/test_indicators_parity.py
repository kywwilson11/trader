"""Kernel-vs-fallback parity harness for indicators.py.

indicators.py has THREE possible backends per function: a C extension
(`indicators_c` — archived 2026-09-26 and opt-in only via
TRADER_INDICATORS_C=1, see archive/README.md), a numba-jitted kernel, and a
pure-pandas/pure-numpy fallback (dispatch priority when the C ext is opted
in: C > numba > pure; by default numba > pure). Train/serve parity across
machines (this dev Mac has neither C ext nor numba; Jetson/CI have numba)
depends on the kernel and the fallback agreeing bit-for-bit-ish on real
inputs. This module asserts that agreement directly, by forcing each
backend in turn (via monkeypatching the module-level `_HAS_C`/`_HAS_NUMBA`
flags that indicators.py's public functions branch on) and comparing
outputs with np.allclose(equal_nan=True).

These tests are only meaningful when numba is actually installed (so the
kernel functions — `_rsi_core`, `_macd_core`, etc. — exist as module
attributes at all; on this Mac the `if _HAS_NUMBA:` block at the top of
indicators.py never executes, so those names don't exist and calling them
would raise AttributeError/NameError, not merely give wrong answers). Every
kernel-comparison test below is therefore individually guarded with
`@pytest.mark.skipif(not _HAS_NUMBA, ...)` rather than an
importorskip('numba') at module scope, so that any future non-numba tests
added to this file would still run on the Mac.

NaN handling is NOT uniform across the kernels — some (`_rolling_mean`,
`_rolling_percentile`) explicitly reset on NaN; `_ewm_span` (used by MACD)
has NO NaN handling at all and would poison every value after the first
gap forever, which the pandas fallback would NOT do (it forward-carries
state through a NaN and recovers). Real OHLCV close/high/low/volume never
has interior gaps in practice (unlike derived features, which commonly
start with a NaN warmup), so the "verified to agree" indicators below are
exercised on a NaN-free OHLCV panel; a *separate* NaN-patched derived
series is used only for the two documented NaN-related divergences
(Stochastic, rolling-percentile).
"""
import platform
import sys

import math
import numpy as np
import pandas as pd
import pytest

import indicators
from indicators import (
    compute_atr,
    compute_bbands,
    compute_hurst,
    compute_linear_slope,
    compute_macd,
    compute_obv,
    compute_rolling_percentile,
    compute_rsi,
    compute_stoch,
    compute_stock_features,
)

_HAS_NUMBA = indicators._HAS_NUMBA
needs_numba = pytest.mark.skipif(
    not _HAS_NUMBA, reason="numba not installed on this machine (dev Mac) — "
    "kernel-vs-fallback parity only meaningful where the kernel exists")


# ── Fixtures ──────────────────────────────────────────────────────────────

def _base_steps(n=600, seed=11):
    """Shared random-walk step generator: a mild AR-free walk PLUS a
    monotonic-rally sub-segment (bars 300-339, strictly non-negative ticks)
    — the "monotonic-rally patch" required by the spec, reused by both the
    clean panel below and (with a NaN patch layered on top) the NaN-bearing
    series used for the documented divergence tests.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.3, n)
    steps[300:340] = np.abs(rng.normal(0.6, 0.15, 40))
    return rng, steps


def _clean_panel(n=600, seed=11):
    """~600-bar seeded synthetic OHLCV panel, NaN-free, with a monotonic
    rally at bars 300-339. Used for the indicators verified to agree
    kernel-vs-fallback on realistic (gap-free) OHLCV data.
    """
    rng, steps = _base_steps(n, seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    close = pd.Series(100 + np.cumsum(steps), index=idx)
    high = close + np.abs(rng.normal(0, 0.2, n))
    low = close - np.abs(rng.normal(0, 0.2, n))
    open_ = close.shift(1).bfill()
    volume = pd.Series(rng.integers(1_000, 50_000, n).astype(float), index=idx)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                          "Close": close, "Volume": volume}, index=idx)


def _nan_patched_series(n=600, seed=11):
    """The SAME base random walk as `_clean_panel`'s Close, with a 5-bar
    NaN patch (bars 150-154) layered on — the "NaN patch" required by the
    spec, used only for the two documented NaN-window divergences.
    """
    rng, steps = _base_steps(n, seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(steps)
    close[150:155] = np.nan
    return pd.Series(close, index=idx)


# ── Kernel-vs-fallback comparison helper ────────────────────────────────

def _kernel_vs_fallback(fn, *args, **kwargs):
    """Call `fn` once forcing the numba kernel path and once forcing the
    pure-pandas/numpy fallback path (C extension disabled in both legs so
    this is strictly kernel-vs-fallback), returning (kernel_out, fallback_out).
    """
    orig_c, orig_nb = indicators._HAS_C, indicators._HAS_NUMBA
    try:
        indicators._HAS_C = False
        indicators._HAS_NUMBA = True
        kernel_out = fn(*args, **kwargs)
        indicators._HAS_NUMBA = False
        fallback_out = fn(*args, **kwargs)
    finally:
        indicators._HAS_C, indicators._HAS_NUMBA = orig_c, orig_nb
    return kernel_out, fallback_out


def _assert_close(a, b, atol=1e-6, rtol=1e-6):
    assert np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                        equal_nan=True, atol=atol, rtol=rtol)


# ── The 7 verified-to-agree pairs (+ RSI on clean data) ─────────────────

@needs_numba
def test_macd_kernel_matches_fallback():
    close = _clean_panel()["Close"]
    (k_line, k_hist, k_sig), (f_line, f_hist, f_sig) = _kernel_vs_fallback(
        compute_macd, close)
    _assert_close(k_line, f_line)
    _assert_close(k_hist, f_hist)
    _assert_close(k_sig, f_sig)


@needs_numba
def test_atr_kernel_matches_fallback():
    df = _clean_panel()
    kernel_out, fallback_out = _kernel_vs_fallback(
        compute_atr, df["High"], df["Low"], df["Close"])
    _assert_close(kernel_out, fallback_out)


@needs_numba
def test_bbands_kernel_matches_fallback():
    close = _clean_panel()["Close"]
    kernel_out, fallback_out = _kernel_vs_fallback(compute_bbands, close)
    for k, f in zip(kernel_out, fallback_out):
        _assert_close(k, f)


@needs_numba
def test_obv_kernel_matches_fallback():
    df = _clean_panel()
    kernel_out, fallback_out = _kernel_vs_fallback(
        compute_obv, df["Close"], df["Volume"])
    _assert_close(kernel_out, fallback_out)


@needs_numba
def test_rolling_percentile_kernel_matches_fallback_clean_data():
    close = _clean_panel()["Close"]
    kernel_out, fallback_out = _kernel_vs_fallback(
        compute_rolling_percentile, close, window=100)
    _assert_close(kernel_out, fallback_out)


@needs_numba
def test_linear_slope_kernel_matches_fallback():
    close = _clean_panel()["Close"]
    kernel_out, fallback_out = _kernel_vs_fallback(
        compute_linear_slope, close, window=5)
    _assert_close(kernel_out, fallback_out)


@needs_numba
def test_hurst_kernel_matches_fallback():
    close = _clean_panel()["Close"]
    kernel_out, fallback_out = _kernel_vs_fallback(
        compute_hurst, close, window=100)
    _assert_close(kernel_out, fallback_out)


@needs_numba
def test_rsi_kernel_matches_fallback_clean_data():
    """Kernel vs fallback RSI agree ONLY from bar 270 on — deliberately.

    This test compares bars 270-299 and nothing earlier. Over the warm-up
    (first valid bar 13 for the kernel vs 14 for the fallback, then up to
    ~10-15 RSI points apart, decaying at (13/14) per bar) the two backends
    DIFFER; that gap is a known, model-facing owner item (the `_rsi_core`
    gain[0]/loss[0]=0 seed vs pandas' NaN diff at index 0), queued for a
    retrain-bundled change, not something this test certifies as equal.
    `test_compute_stock_features_golden_fingerprint` pins the resulting
    per-backend RSI / RSI_Divergence column sums and first-valid indices.
    """
    # Clean-data RSI, after fix #1 (fallback now uses adjust=False). Uses
    # the panel's pre-rally slice (bars 0-299) so avg_loss never hits the
    # exact-zero degenerate case exercised deliberately below.
    #
    # Only the TAIL of the slice is compared. The kernel seeds gain[0]/
    # loss[0] at 0.0 (a real observation baked into `_rsi_core`), while the
    # fallback's `series.diff()`-based gain/loss is NaN at index 0 (pandas
    # excludes it from the EWM entirely) — a one-bar difference in when the
    # `length`-observation warmup is satisfied. That seed mismatch decays
    # geometrically at rate (1 - 1/length) per bar (~20 RSI points at bar
    # 14, below 1e-6 by roughly bar 270 — confirmed by direct simulation of
    # both code paths), so comparing only well past it (bars 270-299) is
    # the correct way to assert "agreement" without re-deriving the kernel
    # (out of scope — DO NOT TOUCH per the spec) or asserting something
    # false about the warmup region itself.
    close = _clean_panel()["Close"].iloc[:300]
    kernel_out, fallback_out = _kernel_vs_fallback(compute_rsi, close, length=14)
    _assert_close(kernel_out.iloc[270:], fallback_out.iloc[270:])


# ── The 3 documented kernel divergences (xfail, not skip, so CI records
#    them without failing the suite) ─────────────────────────────────────

@needs_numba
@pytest.mark.xfail(
    strict=False,
    reason="RSI kernel emits NaN when avg_loss==0 (monotonic rally) while "
    "the fallback/Wilder convention yields 100 — kernel fix is model-facing, "
    "queued for a retrain-bundled change")
def test_rsi_kernel_vs_fallback_monotonic_rally_diverges():
    # avg_loss is an EWM accumulator with memory over the WHOLE history, so
    # a rally mid-series (like `_clean_panel`'s bars 300-339) only nudges an
    # already-nonzero average toward zero asymptotically — it would take
    # thousands of consecutive zero-loss bars to underflow to bit-exact 0.0
    # that way. The degenerate avg_loss==0 case documented in the spec is
    # reached in practice when the EWM *seeds* at zero, i.e. a rally
    # starting at bar 0 of the window fed to compute_rsi: gain[0]/loss[0]
    # seed the kernel's accumulator directly, so a monotonic run from t=0
    # pins avg_loss at exactly 0.0 for as long as it continues.
    close = pd.Series(100.0 + np.arange(60) * 0.75)  # strictly increasing
    kernel_out, fallback_out = _kernel_vs_fallback(compute_rsi, close, length=14)
    _assert_close(kernel_out, fallback_out)


@needs_numba
@pytest.mark.xfail(
    strict=False,
    reason="_rolling_min/_rolling_max skip interior NaNs so Stochastic "
    "diverges from pandas on NaN-bearing windows — kernel fix is "
    "model-facing, queued for a retrain-bundled change")
def test_stoch_kernel_vs_fallback_nan_patch_diverges():
    close = _nan_patched_series()
    high = close + 0.5
    low = close - 0.5
    (k_k, k_d), (f_k, f_d) = _kernel_vs_fallback(compute_stoch, high, low, close)
    _assert_close(k_k, f_k)
    _assert_close(k_d, f_d)


@needs_numba
@pytest.mark.xfail(
    strict=False,
    reason="_rolling_percentile uses a valid-count denominator vs pandas' "
    "min_periods=window on NaN-bearing warmups — kernel fix is model-facing, "
    "queued for a retrain-bundled change")
def test_rolling_percentile_kernel_vs_fallback_nan_patch_diverges():
    series = _nan_patched_series()
    kernel_out, fallback_out = _kernel_vs_fallback(
        compute_rolling_percentile, series, window=100)
    _assert_close(kernel_out, fallback_out)


# ── compute_stock_features golden/regression fingerprint ────────────────
#
# Written BEFORE decomposing compute_stock_features into private helpers
# (spec step 4) so that refactor is provably bit-identical. The golden
# column sums were recorded on the dev Mac's PURE pandas/numpy path; the
# Jetson and both CI legs have numba, so the test now forces each backend
# explicitly (C ext always off) instead of running whatever dispatch picks:
#   - "numba": the PRODUCTION pin (harvest + predict_now run this path).
#     Skipped where numba is absent (dev Mac).
#   - "pure":  the fallback; runs everywhere, the Mac's only leg.
# Exactly three columns differ numba-vs-pure on this fixture — the known,
# model-facing warm-up divergences (rolling-percentile valid-count warm-up
# through ATR's 13-bar NaN lead; RSI gain/loss seeding) — pinned by
# _NUMBA_COL_SUM_OVERRIDES and _WARMUP_PINS below. Everything else must
# match _GOLDEN_COL_SUMS on both backends.
#
# The 64-bit hash_pandas_object fingerprint is ulp- and signed-zero
# sensitive (round(10) does not quantize OBV/Volume-scale columns; a 1-ulp
# input nudge flips it with zero column-sum change), so it is only a
# platform-keyed refactor tripwire: asserted when (platform, machine,
# pandas major, backend) matches a recorded key, otherwise skipped.
# (Evidence: 2026-09-26 Jetson campaign, audit report A.)
#
# If a column sum ever needs regenerating, that means whatever changed was
# NOT equivalent to the previous behavior — don't just update the literals
# to match, go find out why first.

def _golden_stock_frame(days=45, bars_per_day=7, seed=2026):
    """Seeded synthetic hourly stock OHLCV frame (45 business days x 7
    bars) plus an independent SPY close series, for a stable end-to-end
    fingerprint of compute_stock_features' full output.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(end="2026-06-30", periods=days)
    idx = pd.DatetimeIndex(
        [d + pd.Timedelta(hours=14) + pd.Timedelta(hours=h)
         for d in sessions for h in range(bars_per_day)], tz="UTC")
    n = len(idx)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(1e5, 5e5, n)
    df = pd.DataFrame({"Open": openp, "High": np.maximum(high, close),
                        "Low": np.minimum(low, close), "Close": close,
                        "Volume": vol}, index=idx)
    spy_idx = pd.date_range(idx[0], idx[-1], freq="h", tz="UTC")
    spy_close = pd.Series(
        400.0 * np.exp(np.cumsum(rng.normal(0, 0.003, len(spy_idx)))),
        index=spy_idx)
    return df, spy_close


# Platform-keyed fingerprint tripwires:
#   (sys.platform, platform.machine() or None=any, pandas major, backend)
_GOLDEN_FINGERPRINTS = {
    # dev Mac, pure path (the original 2026-07-13 golden; pandas 3.x there).
    ("darwin", None, 3, "pure"): 8972321854121808304,
    # Jetson Orin Nano, numba path (production), numpy 1.26.1 and 2.2.6
    # both give this value with pandas 2.3.3 (verified 2026-09-26).
    ("linux", "aarch64", 2, "numba"): 422055739916008620,
}

_GOLDEN_COL_SUMS = {
    'ATR': 172.762803,
    'ATR_Pct': 164.70732,
    'ATR_Percentile': 109.49,
    'BBB_20_2.0': 9.924699,
    'BBL_20_2.0': 30437.691034,
    'BBP_20_2.0': 172.585286,
    'BBU_20_2.0': 31477.794369,
    'Close': 32953.526409,
    'Day_cos': -25.261039,
    'Day_sin': 110.675842,
    'Gap_Pct': 0.0,
    'High': 33007.793843,
    'Hour_cos': -70.790948,
    'Hour_sin': -264.195416,
    'Hurst': 167.651196,
    'Low': 32900.947727,
    'MACD_12_26_9': 68.00674,
    'MACDh_12_26_9': 2.586753,
    'MACDs_12_26_9': 65.419987,
    'MA_Dist_100d': 0.0,
    'MA_Dist_10d': 189.342229,
    'MA_Dist_200d': 0.0,
    'MA_Dist_20d': 312.328869,
    'MA_Dist_50d': 0.0,
    'MidRange_Gap_20h': -26.66395,
    'MidRange_Gap_60h': -38.046676,
    'Month_cos': -288.305734,
    'Month_sin': 85.624356,
    'OBV': 964355167.473347,
    'ON_Mom_21': 0.0,
    'ON_Mom_252': 0.0,
    'Open': 32943.244784,
    'Pos_Range_20d': 130.793935,
    'Pos_Range_20h': 174.66395,
    'Pos_Range_60d': 0.0,
    'Pos_Range_60h': 166.046676,
    'Price_SMA100_Ratio': 218.524541,
    'Price_SMA20_Ratio': 296.971716,
    'Price_VWAP_Ratio': 315.225636,
    'RM_252_21': 0.0,
    'ROC': 129.706374,
    'ROD_Ret': 36.253666,
    'RR_21': 0.0,
    'RR_5': 0.0,
    'RSI': 16416.688909,
    'RSI_Divergence': -17.204879,
    'RS_vs_SPY': -5490.100301,
    'Ret_21d': 667.252198,
    'Return_12h': 129.706374,
    'Return_4h': 41.570805,
    'SMA_100': 22581.790922,
    'SMA_20': 30957.742701,
    'STOCHd_14_3_3': 16960.614914,
    'STOCHk_14_3_3': 17038.780509,
    'Same_Hour_Mean_40d': 7.631582,
    'TugOfWar_252': 0.0,
    'Turn_of_Month': 49.0,
    'VWAP': 32929.793313,
    'Vol_Price_Confirm': 201.0,
    'Volatility_12h': 123.392667,
    'Volume': 94313579.018395,
    'Volume_Ratio': 295.007594,
    'Volume_SMA_20': 88492904.260532,
}


# numba-path column sums that legitimately differ from the pure golden
# (measured on the Jetson, numba 0.63.1; every other column is identical).
_NUMBA_COL_SUM_OVERRIDES = {
    'ATR_Percentile': 117.441678,
    'RSI': 16356.790603,
    'RSI_Divergence': -9.780238,
}

# Per-backend (NaN count, first-valid index) for the three divergent
# columns. The fixture has no interior NaN, so NaN count == first-valid.
_WARMUP_PINS = {
    "pure": {'ATR_Percentile': (112, 112), 'RSI': (14, 14),
             'RSI_Divergence': (18, 18)},
    "numba": {'ATR_Percentile': (99, 99), 'RSI': (13, 13),
              'RSI_Divergence': (17, 17)},
}


def _golden_result(backend, monkeypatch):
    """Run compute_stock_features on the golden frame with the C ext forced
    off and the requested backend forced on ("numba" skips if absent)."""
    monkeypatch.setattr(indicators, "_HAS_C", False)
    if backend == "numba":
        pytest.importorskip("numba")
        if not indicators._HAS_NUMBA:
            pytest.skip("numba importable but indicators' kernels not built")
        monkeypatch.setattr(indicators, "_HAS_NUMBA", True)
    else:
        monkeypatch.setattr(indicators, "_HAS_NUMBA", False)
    df, spy = _golden_stock_frame()
    return compute_stock_features(df.copy(), spy_close=spy, symbol="GOLDTEST")


@pytest.mark.parametrize("backend", ["numba", "pure"])
def test_compute_stock_features_golden_fingerprint(backend, monkeypatch):
    result = _golden_result(backend, monkeypatch)

    cols = sorted(result.columns.tolist())
    assert cols == sorted(_GOLDEN_COL_SUMS.keys()), (
        "compute_stock_features' column set changed — that's a behavior "
        "change, not a pure structural refactor")

    expected = dict(_GOLDEN_COL_SUMS)
    if backend == "numba":
        expected.update(_NUMBA_COL_SUM_OVERRIDES)
    for c in cols:
        actual = float(result[c].sum(skipna=True))
        # 2026-09-27: compare with a relative tolerance instead of exact 6-dp equality. The goldens were
        # recorded on ARM (Mac / Jetson); a real kernel change moves a column sum by orders of magnitude
        # more than the ~1e-12 relative float noise between ARM and x86 (CI), which exact equality
        # would turn into a permanent CI red. rel 1e-9 on a ~1.6e4 sum is ~1.6e-5 absolute.
        assert math.isclose(actual, expected[c], rel_tol=1e-9, abs_tol=1e-6), (
            f"[{backend}] column {c!r} drifted: {actual!r} != {expected[c]!r}")

    for c, (n_nan, first_valid) in _WARMUP_PINS[backend].items():
        valid = result[c].notna().to_numpy()
        assert int((~valid).sum()) == n_nan, (
            f"[{backend}] {c!r} NaN count {int((~valid).sum())} != {n_nan}")
        assert int(np.flatnonzero(valid)[0]) == first_valid, (
            f"[{backend}] {c!r} first-valid index changed")


@pytest.mark.parametrize("backend", ["numba", "pure"])
def test_compute_stock_features_fingerprint_tripwire(backend, monkeypatch):
    """Bit-level refactor tripwire: asserted only on a recorded
    (platform, machine, pandas major, backend) key, skipped elsewhere."""
    result = _golden_result(backend, monkeypatch)
    cols = sorted(result.columns.tolist())
    fingerprint = int(pd.util.hash_pandas_object(result[cols].round(10)).sum())
    pd_major = int(pd.__version__.split(".")[0])
    for (plat, mach, pmaj, be), golden in _GOLDEN_FINGERPRINTS.items():
        if (plat == sys.platform and mach in (None, platform.machine())
                and pmaj == pd_major and be == backend):
            assert fingerprint == golden, (
                f"[{backend}] bit-level fingerprint changed on the recorded "
                f"platform: {fingerprint} != {golden}")
            return
    pytest.skip(
        f"no recorded fingerprint for ({sys.platform}, {platform.machine()}, "
        f"pandas {pd_major}, {backend}) — got {fingerprint}; the column-sum "
        "and warm-up pins are asserted by "
        "test_compute_stock_features_golden_fingerprint")
