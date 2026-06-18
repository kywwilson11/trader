"""Wave-6 Tier-1: per-name EDGE effective-spread cost (liquidity.py).

Removes the train/deploy cost asymmetry — the flat offline spread vs the live
gate's real quoted spread. These tests pin the units (percent), the floor/cap,
the PIT (strictly trailing) property, and the vectorized round-trip helper."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import liquidity


def _ohlc(n=120, spread_frac=0.004, seed=0):
    """Synthetic OHLC with a controllable embedded effective spread."""
    rng = np.random.RandomState(seed)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    half = spread_frac / 2.0
    close = mid * (1 + rng.choice([-half, half], n))  # bid/ask bounce
    high = np.maximum(mid, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(mid, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    op = mid * (1 + rng.normal(0, 0.002, n))
    idx = pd.date_range('2025-01-01', periods=n, freq='h')
    return pd.DataFrame({'Open': op, 'High': high, 'Low': low,
                         'Close': close}, index=idx)


class TestEdgeSpreadSeries:
    def test_returns_percent_in_band(self):
        s = liquidity.edge_spread_series(_ohlc(), window=35)
        assert len(s) == 120
        assert s.notna().all()                       # never NaN
        assert (s >= liquidity.SPREAD_FLOOR_PCT - 1e-9).all()
        assert (s <= liquidity.SPREAD_CAP_PCT + 1e-9).all()

    def test_wider_bounce_gives_wider_estimate(self):
        tight = liquidity.edge_spread_series(_ohlc(spread_frac=0.001, seed=1),
                                             window=35).median()
        wide = liquidity.edge_spread_series(_ohlc(spread_frac=0.012, seed=1),
                                            window=35).median()
        assert wide > tight

    def test_floor_applied(self):
        # zero-spread mid series -> estimate floors, never zero/negative
        s = liquidity.edge_spread_series(_ohlc(spread_frac=0.0, seed=2),
                                         window=35)
        assert (s >= liquidity.SPREAD_FLOOR_PCT - 1e-9).all()

    def test_cap_applied(self):
        s = liquidity.edge_spread_series(_ohlc(spread_frac=0.05, seed=3),
                                         window=20, cap_pct=0.5)
        assert (s <= 0.5 + 1e-9).all()

    def test_case_insensitive_columns(self):
        df = _ohlc().rename(columns=str.lower)
        s = liquidity.edge_spread_series(df, window=35)
        assert len(s) == len(df) and s.notna().all()

    def test_missing_columns_raises(self):
        with pytest.raises(KeyError):
            liquidity.edge_spread_series(pd.DataFrame({'Close': [1, 2, 3]}))

    def test_short_series_returns_floor(self):
        s = liquidity.edge_spread_series(_ohlc(n=3), window=35)
        assert (s == liquidity.SPREAD_FLOOR_PCT).all()

    def test_strictly_trailing_is_pit(self):
        # The estimate at bar t must not change when FUTURE bars change —
        # the property that makes harvest-time stamping look-ahead-free.
        base = _ohlc(n=80, seed=7)
        s_full = liquidity.edge_spread_series(base, window=20)
        # corrupt the last 10 bars; everything up to t=60 must be identical
        future = base.copy()
        future.iloc[-10:, :] *= 1.5
        s_corrupt = liquidity.edge_spread_series(future, window=20)
        np.testing.assert_allclose(s_full.iloc[:60].values,
                                   s_corrupt.iloc[:60].values, rtol=1e-9)


class TestAbdiRanaldoFallback:
    def test_fallback_runs_without_bidask(self):
        df = _ohlc(seed=5)
        frac = liquidity._abdi_ranaldo_rolling(df, window=20)
        finite = frac[np.isfinite(frac)]
        assert finite.size > 0 and (finite >= 0).all()


class TestPerBarRoundTripCost:
    def test_fee_plus_spread(self):
        from fees import round_trip_cost_pct
        spreads = np.array([0.05, 0.20, 0.50])
        cost = liquidity.per_bar_round_trip_cost('stock', spreads)
        # equals the scalar fees path bar-by-bar
        for s, c in zip(spreads, cost):
            assert c == pytest.approx(round_trip_cost_pct('stock', s))

    def test_nan_spread_falls_back_to_flat(self):
        from fees import round_trip_cost_pct
        cost = liquidity.per_bar_round_trip_cost('stock', np.array([np.nan]))
        assert cost[0] == pytest.approx(round_trip_cost_pct('stock', 0.05))

    def test_negative_spread_falls_back(self):
        from fees import round_trip_cost_pct
        cost = liquidity.per_bar_round_trip_cost('crypto', np.array([-1.0]))
        assert cost[0] == pytest.approx(round_trip_cost_pct('crypto', 0.10))

    def test_monotone_in_spread(self):
        cost = liquidity.per_bar_round_trip_cost('stock',
                                                 np.array([0.02, 0.10, 0.40]))
        assert cost[0] < cost[1] < cost[2]
