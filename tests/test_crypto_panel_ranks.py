"""Wave-9 #6: crypto cross-sectional rank + the soft, cost-neutral size tilt."""
import numpy as np
import pandas as pd
import pytest

from panel_ranks import (
    compute_live_crypto_ranks,
    cs_size_tilt,
    add_crypto_panel_ranks,
    CRYPTO_CS_BASE_COLS,
)


def test_live_ranks_signed_leader_and_laggard():
    vals = {'BTC/USD': 0.05, 'ETH/USD': 0.02, 'SOL/USD': -0.01, 'XRP/USD': -0.04}
    r = compute_live_crypto_ranks(vals)
    assert r['BTC/USD'] == pytest.approx(1.0)        # strongest -> +1
    assert r['XRP/USD'] == pytest.approx(-1.0)       # weakest -> -1
    assert -1 < r['ETH/USD'] < r['BTC/USD']
    # monotone in the underlying value
    ordered = sorted(vals, key=vals.get)
    assert [r[s] for s in ordered] == sorted(r[s] for s in ordered)


def test_live_ranks_degenerate_cases():
    assert compute_live_crypto_ranks({'BTC/USD': 0.01}) == {'BTC/USD': 0.0}
    out = compute_live_crypto_ranks({'BTC/USD': 0.01, 'ETH/USD': np.nan})
    assert out['ETH/USD'] == 0.0 and out['BTC/USD'] == 0.0   # <2 finite -> neutral


def test_cs_size_tilt_bounds_and_neutral():
    assert cs_size_tilt(1.0) == pytest.approx(1.10)
    assert cs_size_tilt(-1.0) == pytest.approx(0.90)
    assert cs_size_tilt(0.0) == pytest.approx(1.0)
    assert cs_size_tilt(None) == 1.0
    assert cs_size_tilt(np.nan) == 1.0
    # monotone + bounded across the range
    grid = np.linspace(-1, 1, 21)
    tilts = [cs_size_tilt(x) for x in grid]
    assert all(0.90 <= t <= 1.10 for t in tilts)
    assert tilts == sorted(tilts)


def test_cs_size_tilt_dispersion_gate_is_a_noop_in_flat_tape():
    # high rank but dispersion below the floor -> no tilt (common-beta hour)
    assert cs_size_tilt(1.0, dispersion=0.001, dispersion_floor=0.01) == 1.0
    # dispersion above the floor -> tilt applies
    assert cs_size_tilt(1.0, dispersion=0.05, dispersion_floor=0.01) == pytest.approx(1.10)


def test_tilt_is_turnover_invariant_by_construction():
    # The tilt only re-weights size within [0.90,1.10]; it can never zero a trade
    # (forgo it) or push one above its base — so the trade SET is unchanged.
    for r in np.linspace(-1, 1, 50):
        t = cs_size_tilt(r)
        assert 0.0 < 0.90 <= t <= 1.10


def test_add_crypto_panel_ranks_on_synthetic_panel():
    idx = pd.to_datetime(['2026-06-18 10:00'] * 6 + ['2026-06-18 11:00'] * 6)
    coins = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'DOGE/USD', 'LINK/USD']
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        'Ticker': coins * 2,
        **{c: rng.normal(size=12) for c in CRYPTO_CS_BASE_COLS},
    }, index=idx)
    out = add_crypto_panel_ranks(df)
    for c in CRYPTO_CS_BASE_COLS:
        col = f'CS_Rank_{c}'
        assert col in out.columns
        # each timestamp's signed ranks span [-1, 1] symmetrically
        for ts in out.index.unique():
            v = out.loc[ts, col].values
            assert v.min() == pytest.approx(-1.0) and v.max() == pytest.approx(1.0)
    assert 'CS_Dispersion' in out.columns
