"""Gate/live parity: the backtest must apply the q10 tail veto."""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import simulate_ticker
from strategy_config import CRYPTO_POLICY


def _tdf(n=200):
    close = np.full(n, 100.0)
    return pd.DataFrame({
        'Close': close, 'High': close + 0.05, 'Low': close - 0.05,
        'Open': close, 'ATR': np.full(n, 1.0),
    }, index=pd.DatetimeIndex([dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
                               + dt.timedelta(hours=h) for h in range(n)]))


def _bullish_preds(n=200):
    p = np.full(n, 5.0)   # far above any threshold/edge floor
    p[:5] = np.nan
    return p


def test_q10_veto_blocks_entries():
    tdf = _tdf()
    preds = _bullish_preds()
    base = simulate_ticker(tdf, preds, 'crypto', 0.5, CRYPTO_POLICY)
    assert len(base) > 0

    q10 = np.full(len(preds), -9.0)   # catastrophic left tail everywhere
    vetoed = simulate_ticker(tdf, preds, 'crypto', 0.5, CRYPTO_POLICY,
                             q10_preds=q10, q10_floor=-2.0)
    assert vetoed == []


def test_q10_above_floor_passes_and_nan_fails_open():
    tdf = _tdf()
    preds = _bullish_preds()
    q10_ok = np.full(len(preds), -1.0)    # tail above the floor
    trades = simulate_ticker(tdf, preds, 'crypto', 0.5, CRYPTO_POLICY,
                             q10_preds=q10_ok, q10_floor=-2.0)
    assert len(trades) > 0

    q10_nan = np.full(len(preds), np.nan)  # no q10 output -> no veto
    trades2 = simulate_ticker(tdf, preds, 'crypto', 0.5, CRYPTO_POLICY,
                              q10_preds=q10_nan, q10_floor=-2.0)
    assert len(trades2) == len(trades)
