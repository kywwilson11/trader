"""Wave-6 Tier-1: per-bar effective-spread cost threaded into the backtest.

Verifies the cost choke point in backtest.simulate_ticker uses the per-bar
Eff_Spread_Pct when present and falls back EXACTLY to the flat cost when it is
absent (so the running system is unchanged until a re-harvest stamps spreads).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest
from strategy_config import policy_for


def _trending_frame(n=60, drift=0.004, seed=0):
    """Gently rising bars so a long signal exits profitably (gross > 0)."""
    rng = np.random.RandomState(seed)
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.001, n))
    high = close * 1.002
    low = close * 0.998
    op = close * 0.999
    atr = pd.Series(close).rolling(5, min_periods=1).std().fillna(0.5).values + 0.5
    idx = pd.date_range('2025-03-03 14:00', periods=n, freq='h', tz='UTC')
    return pd.DataFrame({'Open': op, 'High': high, 'Low': low,
                         'Close': close, 'ATR': atr}, index=idx)


def _run(tdf, threshold=0.1):
    preds = np.full(len(tdf), 0.5)  # strong, constant long signal
    return backtest.simulate_ticker(tdf, preds, 'stock', threshold,
                                    policy_for('stock'))


class TestPerBarCostBacktest:
    def test_absent_column_is_flat_fallback(self):
        # No Eff_Spread_Pct -> identical to the legacy flat-cost path.
        tdf = _trending_frame()
        trades = _run(tdf)
        assert trades, "expected at least one trade"
        from fees import round_trip_cost_pct
        flat = round_trip_cost_pct('stock', backtest.SPREAD_PCT['stock'])
        for t in trades:
            assert t['net_pct'] == pytest.approx(t['gross_pct'] - flat, abs=1e-6)

    def test_wide_per_bar_spread_lowers_net(self):
        tdf = _trending_frame()
        base = _run(tdf)
        wide = tdf.copy()
        wide['Eff_Spread_Pct'] = 1.20  # ~120 bps, far above the flat 0.05
        wtrades = _run(wide)
        assert len(base) == len(wtrades)
        # Same gross path, strictly higher cost -> strictly lower net.
        for b, w in zip(base, wtrades):
            assert w['gross_pct'] == pytest.approx(b['gross_pct'], abs=1e-6)
            assert w['net_pct'] < b['net_pct']

    def test_per_bar_matches_helper(self):
        from liquidity import per_bar_round_trip_cost
        tdf = _trending_frame()
        tdf = tdf.copy()
        spreads = np.linspace(0.05, 0.9, len(tdf))
        tdf['Eff_Spread_Pct'] = spreads
        trades = _run(tdf)
        assert trades
        cost = per_bar_round_trip_cost('stock', spreads)
        # Each trade's charged cost must equal the helper at its entry bar.
        # Recover entry bar via entry price match (unique on a monotone series).
        closes = tdf['Close'].values
        for t in trades:
            i = int(np.argmin(np.abs(closes - t['entry'])))
            # gross_pct/net_pct are each rounded to 4dp in the trade dict, so
            # their difference carries up to ~1e-4 rounding error.
            assert (t['gross_pct'] - t['net_pct']) == pytest.approx(
                cost[i], abs=2e-4)

    def test_fees_paid_is_exact_gross_minus_net(self):
        tdf = _trending_frame().copy()
        tdf['Eff_Spread_Pct'] = np.linspace(0.05, 0.9, len(tdf))
        trades = _run(tdf)
        m = backtest.aggregate_metrics(trades, 'stock', span_days=30)
        assert m['fees_paid_pct'] == pytest.approx(
            m['gross_total_pct'] - m['net_total_pct'], abs=0.011)
