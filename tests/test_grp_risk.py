"""Cost/risk-kernel group (fees/liquidity/cost_regime/short_cost/borrow_proxy/
portfolio/risk_budget/drawdown) design+scout locks, 2026-07. Mac-green."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import fees
import risk_budget as rb
import short_cost


def test_ar_fallback_inline_comment_matches_code_not_crossproduct():
    src = (REPO / "liquidity.py").read_text()
    assert "form, simplified single-window var" not in src   # old inline rot gone
    assert "SAME-bar squared" in src                          # accurate replacement


def test_required_edge_is_cost_times_multiple():
    for asset in ("crypto", "stock"):
        for sp in (0.0, 0.03, 0.2):
            for m in (1.0, 2.0, 3.5):
                assert fees.required_edge_pct(asset, sp, min_edge=m) == pytest.approx(
                    fees.round_trip_cost_pct(asset, sp) * m)


def test_allocate_book_caps_partitions_total():
    rng = np.random.default_rng(0)
    for _ in range(200):
        vs, vc = rng.uniform(0.01, 1.0, size=2)
        cs, cc = rb.allocate_book_caps(vs, vc, total_cap=0.03)
        assert cs + cc == pytest.approx(0.03)
        assert cs > 0 and cc > 0
    cs, cc = rb.allocate_book_caps(0.0, 0.5, total_cap=0.03)   # fail-open even split
    assert cs == pytest.approx(0.015) and cc == pytest.approx(0.015)


def test_short_round_trip_decomposes_into_long_base_plus_drag():
    base = fees.round_trip_cost_pct("stock", 0.05)
    post = short_cost.short_round_trip_cost_pct("2025-11-01", 0.05, hold_days=5,
                                                likely_etb=True)
    assert post == pytest.approx(base)          # ETB on/after $0-borrow regime
    pre = short_cost.short_round_trip_cost_pct("2024-06-01", 0.05, hold_days=10,
                                               likely_etb=True)
    drag = short_cost.PRE_REGIME_ETB_BPS / 100.0 * 10 / short_cost._DAYS_PER_YEAR
    assert pre == pytest.approx(base + drag)    # pre-regime 30bps schedule
