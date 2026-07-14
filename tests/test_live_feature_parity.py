"""Live-path feature parity tests (2026-07 review P0).

The wave-4 daily-window features (MA_Dist_200d, RM_252_21, ...) can never warm
up on a live ~45-day frame, so they are all-NaN by construction there. The live
path used a whole-frame dropna(), so ONE all-NaN column deleted EVERY row and
stock predictions returned None each cycle — silently, with no test exercising
the live feature path. These tests pin the fixed contract:

  fetch-length frame -> compute_stock_features -> fill_warmup_features
      -> dropna(subset=model feature cols) -> enough rows for any seq_len

plus the closed-bar guard (drop_forming_bar) and the Hurst input-mode fix.
Everything here runs without torch/numba (indicators has pure fallbacks).
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indicators
from indicators import (
    compute_features, compute_stock_features, fill_warmup_features,
    WARMUP_FEATURES_ZERO, WARMUP_FEATURES_HALF,
)
from indicator_config import get_preset_features
from market_data import drop_forming_bar

MAX_SEQ_LEN = 64  # adaptive_config hard cap — live must always cover it

rng = np.random.default_rng(7)


def _hourly_stock_frame(days=45, bars_per_day=7):
    """Synthetic RTH-like hourly OHLCV: `days` business days x 7 bars."""
    sessions = pd.bdate_range(end='2026-06-30', periods=days)
    idx = pd.DatetimeIndex(
        [d + pd.Timedelta(hours=14) + pd.Timedelta(hours=h)
         for d in sessions for h in range(bars_per_day)], tz='UTC')
    n = len(idx)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(1e5, 5e5, n)
    return pd.DataFrame({'Open': openp, 'High': np.maximum(high, close),
                         'Low': np.minimum(low, close), 'Close': close,
                         'Volume': vol}, index=idx)


def _crypto_frame(bars=250):
    idx = pd.date_range(end='2026-06-30 23:00', periods=bars, freq='h',
                        tz='UTC')
    n = len(idx)
    close = 50_000.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame({'Open': openp, 'High': np.maximum(high, close),
                         'Low': np.minimum(low, close), 'Close': close,
                         'Volume': vol}, index=idx)


def _present_preset_cols(df, asset='stock'):
    feats = get_preset_features('standard')
    return [c for c in feats if c in df.columns]


# --- The P0 regression: live stock frame must survive the dropna ---

def test_stock_live_frame_survives_subset_dropna():
    raw = _hourly_stock_frame()
    spy = raw['Close'] * rng.uniform(0.9, 1.1)  # crude benchmark series
    df = compute_stock_features(raw.copy(), spy_close=spy, symbol='TEST')

    # Documents the bug class: pre-fill, the long-window daily features are
    # all-NaN on a short frame, so a WHOLE-FRAME dropna deletes every row.
    assert df['MA_Dist_200d'].isna().all()
    assert len(df.dropna()) == 0

    df = fill_warmup_features(df)
    present = _present_preset_cols(df)
    assert present, "standard preset must intersect computed stock columns"
    survivors = df.dropna(subset=present)

    # Enough rows for ANY seq_len the search space can choose
    assert len(survivors) >= MAX_SEQ_LEN, (
        f"only {len(survivors)} usable rows on a live-size stock frame")
    # No NaN remains in any model-consumed column
    assert not survivors[present].isna().any().any()


def test_warmup_fill_values_and_nonneutral_short_windows():
    raw = _hourly_stock_frame()
    spy = raw['Close'] * 1.02
    df = fill_warmup_features(
        compute_stock_features(raw.copy(), spy_close=spy, symbol='TEST'))

    # Long windows can never warm on 45 days -> the harvest's neutral values
    assert (df['MA_Dist_200d'] == 0.0).all()
    assert (df['RM_252_21'] == 0.0).all()
    assert (df['Pos_Range_60d'] == 0.5).all()
    # Short windows DO warm -> must carry real (non-constant) signal
    tail = df['MA_Dist_10d'].tail(50)
    assert (tail != 0.0).any()
    assert df['Pos_Range_20h'].tail(50).between(0, 1).all()


def test_fill_only_touches_nans():
    df = pd.DataFrame({
        'RM_252_21': [np.nan, 1.5, np.nan],
        'Pos_Range_20d': [np.nan, 0.9, 0.1],
        'Other': [np.nan, np.nan, np.nan],
    })
    out = fill_warmup_features(df)
    assert out['RM_252_21'].tolist() == [0.0, 1.5, 0.0]
    assert out['Pos_Range_20d'].tolist() == [0.5, 0.9, 0.1]
    assert out['Other'].isna().all()  # not a warmup col -> untouched
    # The shared lists exist and don't overlap
    assert not set(WARMUP_FEATURES_ZERO) & set(WARMUP_FEATURES_HALF)


def test_crypto_frame_survives_with_seq_len_headroom():
    df = compute_features(_crypto_frame(250))
    present = _present_preset_cols(df, asset='crypto')
    survivors = df.dropna(subset=present)
    assert len(survivors) >= MAX_SEQ_LEN, (
        f"only {len(survivors)} usable crypto rows from a 250-bar fetch")


# --- Closed-bar guard ---

def test_drop_forming_bar_drops_in_progress_hour():
    now = datetime.now(timezone.utc)
    idx = pd.DatetimeIndex([now - timedelta(hours=2, minutes=30),
                            now - timedelta(hours=1, minutes=30),
                            now - timedelta(minutes=30)])
    df = pd.DataFrame({'Close': [1.0, 2.0, 3.0]}, index=idx)
    out = drop_forming_bar(df)
    assert len(out) == 2 and out['Close'].iloc[-1] == 2.0


def test_drop_forming_bar_keeps_closed_history():
    now = datetime.now(timezone.utc)
    idx = pd.DatetimeIndex([now - timedelta(hours=30),
                            now - timedelta(hours=29)])
    df = pd.DataFrame({'Close': [1.0, 2.0]}, index=idx)
    out = drop_forming_bar(df)
    assert len(out) == 2  # weekend/after-hours stock frame loses nothing


def test_drop_forming_bar_handles_naive_and_empty():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    idx = pd.DatetimeIndex([now - timedelta(minutes=10)])  # tz-naive, forming
    df = pd.DataFrame({'Close': [1.0]}, index=idx)
    assert len(drop_forming_bar(df)) == 0
    assert drop_forming_bar(None) is None
    empty = pd.DataFrame({'Close': []},
                         index=pd.DatetimeIndex([], tz='UTC'))
    assert len(drop_forming_bar(empty)) == 0


# --- Hurst input mode ---

def test_hurst_levels_mode_reads_high_on_random_walk():
    # The historical (levels) mode: a pure random walk reads ~0.8, far from
    # the documented 0.5 — which is WHY the live hurst<0.45 gate never fired.
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 800)) + 500.0)
    h = indicators.compute_hurst(walk, window=100).dropna()
    assert h.mean() > 0.65


def test_hurst_returns_mode_reads_near_half_on_random_walk(monkeypatch):
    monkeypatch.setattr(indicators, 'HURST_ON_RETURNS', True)
    df = _crypto_frame(800)
    out = compute_features(df)
    h = out['Hurst'].dropna()
    assert 0.35 < h.mean() < 0.65, (
        f"returns-mode Hurst should read ~0.5 on a random walk, got "
        f"{h.mean():.3f}")
