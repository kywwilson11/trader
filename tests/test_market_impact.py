"""Wave-8 #6: square-root market-impact cost term.

The offline cost model charges fee + per-name spread but is size-blind: a $100
and a $50k order into a thin name cost identical bps, so impact-driven 'edge' on
illiquid names is certified by the gate. market_impact_pct adds the Almgren/Kyle
sqrt-participation haircut. It is OFF by default — these tests prove both the
math and that the default path is byte-for-byte unchanged.
"""
import numpy as np
import pytest

import liquidity
from liquidity import (
    market_impact_pct,
    per_bar_round_trip_cost,
    impact_inputs_from_df,
    IMPACT_CAP_PCT,
)


def test_zero_when_inputs_unknown_or_nonpositive():
    assert market_impact_pct(0, 1e6, 0.1) == 0.0           # no notional
    assert market_impact_pct(1e4, 0, 0.1) == 0.0           # no ADV
    assert market_impact_pct(1e4, -5, 0.1) == 0.0          # negative ADV
    assert market_impact_pct(1e4, 1e6, 0.0) == 0.0         # no spread
    assert market_impact_pct(np.nan, 1e6, 0.1) == 0.0
    assert market_impact_pct(1e4, np.inf, 0.1) == 0.0


def test_monotone_and_sqrt_scaling_in_notional():
    base = market_impact_pct(10_000, 5e6, 0.1, k=1.0)
    bigger = market_impact_pct(40_000, 5e6, 0.1, k=1.0)
    assert bigger > base > 0.0
    # 4x notional -> 2x impact (square-root law), below the cap.
    assert bigger == pytest.approx(2.0 * base, rel=1e-9)


def test_thinner_name_costs_more():
    liquid = market_impact_pct(25_000, 50e6, 0.05, k=1.0)
    thin = market_impact_pct(25_000, 2e6, 0.30, k=1.0)     # low ADV + wide spread
    assert thin > liquid


def test_round_trip_is_two_sides_and_capped():
    one = market_impact_pct(25_000, 5e6, 0.1, k=1.0, sides=1)
    two = market_impact_pct(25_000, 5e6, 0.1, k=1.0, sides=2)
    assert two == pytest.approx(2.0 * one)
    # Near-zero ADV would explode; the per-side cap binds.
    capped = market_impact_pct(1e9, 1.0, 5.0, k=1.0, sides=1)
    assert capped == pytest.approx(IMPACT_CAP_PCT)


def test_per_bar_default_path_is_unchanged():
    spreads = np.array([0.05, 0.20, np.nan, -1.0, 0.10])
    without = per_bar_round_trip_cost('stock', spreads)
    # passing adv but no notional (or vice versa) must also be a no-op
    half = per_bar_round_trip_cost('stock', spreads, adv_dollar=np.full(5, 5e6))
    assert np.array_equal(without, half)


def test_per_bar_impact_matches_scalar_and_raises_cost():
    sp = np.array([0.10, 0.10, 0.10])
    adv = np.array([5e6, 1e6, 50e6])                       # mid / thin / deep
    N, k = 25_000.0, 1.0
    base = per_bar_round_trip_cost('stock', sp)
    withimp = per_bar_round_trip_cost('stock', sp, adv_dollar=adv, notional=N, impact_k=k)
    assert np.all(withimp >= base)
    # bar-by-bar agreement with the scalar helper (round trip)
    for i in range(3):
        expected = market_impact_pct(N, adv[i], sp[i], k=k, sides=2)
        assert (withimp[i] - base[i]) == pytest.approx(expected, rel=1e-9)
    # thinner ADV (bar 1) pays the most impact
    impact = withimp - base
    assert impact[1] > impact[0] > impact[2]


def test_impact_inputs_off_by_default(monkeypatch):
    import pandas as pd
    df = pd.DataFrame({'Close': [1.0], 'DV30': [5e6]})
    # Default config: disabled -> no impact inputs.
    adv, notional, k = impact_inputs_from_df(df)
    assert adv is None and notional is None and k == 1.0


def test_impact_inputs_enabled_reads_dv30(monkeypatch):
    import pandas as pd
    import strategy_config
    monkeypatch.setattr(strategy_config, 'IMPACT_COST_ENABLED', True, raising=True)
    monkeypatch.setattr(strategy_config, 'IMPACT_K', 1.3, raising=True)
    monkeypatch.setattr(strategy_config, 'IMPACT_TYPICAL_NOTIONAL', 30_000, raising=True)
    df = pd.DataFrame({'Close': [1.0, 2.0], 'DV30': [5e6, 6e6]})
    adv, notional, k = impact_inputs_from_df(df)
    assert list(adv) == [5e6, 6e6] and notional == 30_000.0 and k == 1.3
    # underscore-stamped variant is also recognized
    df2 = pd.DataFrame({'Close': [1.0], '_DV30': [4e6]})
    adv2, _, _ = impact_inputs_from_df(df2)
    assert list(adv2) == [4e6]
    # no DV column -> off even when enabled
    adv3, n3, _ = impact_inputs_from_df(pd.DataFrame({'Close': [1.0]}))
    assert adv3 is None and n3 is None
