"""Cross-sectional effective-n tests (2026-07 review).

The DSR gates pooled trades across correlated names as independent draws: on
the 6-coin crypto book a zero-edge model's holdout false-pass rate rises from
~0.2% (IID) to 5-9% at realistic pairwise correlation. These tests pin the
fix: calendar-interval clustering (sample_weights.clustered_effective_n), its
wiring into backtest.aggregate_metrics, and the overlapping-panel deflation in
portfolio_backtest.compare_deflated (which scored DSR ~0.99 on zero-edge
24h-overlap panels before the fwd_bars n_eff was plumbed in).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sample_weights import clustered_effective_n
from backtest import aggregate_metrics
from portfolio_backtest import compare_deflated, top_k


def _ts(hours):
    base = np.datetime64('2026-06-01T00:00')
    return np.array([base + np.timedelta64(int(h * 60), 'm') for h in hours])


# --- clustered_effective_n ---

def test_disjoint_intervals_count_fully():
    entries = _ts([0, 10, 20, 30])
    exits = _ts([5, 15, 25, 35])
    assert clustered_effective_n(entries, exits) == 4


def test_simultaneous_trades_collapse_to_one():
    # six same-hour entries across six names = ONE independent draw
    entries = _ts([0, 0, 0, 0, 0, 0])
    exits = _ts([24, 24, 24, 24, 24, 24])
    assert clustered_effective_n(entries, exits) == 1


def test_chain_overlap_is_one_cluster():
    # a-b overlap and b-c overlap -> one cluster even though a-c don't touch
    entries = _ts([0, 4, 8])
    exits = _ts([5, 9, 12])
    assert clustered_effective_n(entries, exits) == 1


def test_mixed_clusters_and_unsorted_input():
    entries = _ts([50, 0, 2, 100])
    exits = _ts([60, 3, 5, 110])
    assert clustered_effective_n(entries, exits) == 3


def test_float_inputs_and_nan_dropped():
    entries = np.array([0.0, 1.0, np.nan, 50.0])
    exits = np.array([5.0, 2.0, 3.0, 55.0])
    assert clustered_effective_n(entries, exits) == 2
    assert clustered_effective_n(np.array([]), np.array([])) == 0


def test_degenerate_exit_before_entry_is_point():
    entries = np.array([0.0, 10.0])
    exits = np.array([-5.0, 12.0])  # bad span clipped to a point
    assert clustered_effective_n(entries, exits) == 2


def test_never_exceeds_trade_count():
    rng = np.random.default_rng(3)
    e = np.sort(rng.uniform(0, 1000, 200))
    x = e + rng.uniform(0, 30, 200)
    n = clustered_effective_n(e, x)
    assert 1 <= n <= 200


# --- backtest.aggregate_metrics wiring ---

def _fake_trades(n_names, overlapping):
    """n_names trades per hour-slot; overlapping=True stacks them same-hour."""
    trades = []
    rng = np.random.default_rng(11)
    base = pd.Timestamp('2026-06-01', tz='UTC')
    slot = 0
    for i in range(60):
        for j in range(n_names):
            entry = base + pd.Timedelta(hours=slot if overlapping else slot + j * 30)
            trades.append({
                'ticker': f'N{j}', 'entry_time': str(entry),
                'exit_time': str(entry + pd.Timedelta(hours=24)),
                'entry': 100.0, 'exit': 101.0, 'bars_held': 24,
                'gross_pct': float(rng.normal(0.05, 1.0)),
                'net_pct': float(rng.normal(0.0, 1.0)),
                'reason': 'take_profit',
            })
        slot += 30 if overlapping else n_names * 30
    return trades


def test_aggregate_metrics_clusters_cross_name_trades():
    overlapped = aggregate_metrics(_fake_trades(6, overlapping=True),
                                   'crypto', span_days=90)
    spread = aggregate_metrics(_fake_trades(6, overlapping=False),
                               'crypto', span_days=90)
    # 6 names stacked in the same hours -> ~n/6 effective; spread out -> ~n
    assert overlapped['n_eff_clustered'] <= overlapped['n_trades'] // 5
    assert spread['n_eff_clustered'] >= spread['n_trades'] * 0.9
    assert 'dsr' in overlapped and 'n_eff_clustered' in overlapped


# --- compare_deflated overlap deflation ---

def _overlapping_panel(fb=24, n_periods=1500, seed=5):
    """Zero-edge panel whose fwd_return is an fb-bar overlapping sum of
    noise sampled EVERY bar — the exact shape rank_gradient_report feeds it
    (hourly frame, Target_Return_24-style column)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.3, n_periods + fb)
    fwd = np.array([noise[t:t + fb].sum() for t in range(n_periods)])
    return [[{'symbol': 'A', 'signal': 1.0, 'fwd_return': float(fwd[t])}]
            for t in range(n_periods)]


def test_compare_deflated_overlap_ffailopen_is_closed():
    panel = _overlapping_panel()
    pol = {'only': top_k(1)}
    naive = compare_deflated(panel, pol, fwd_bars=1)
    honest = compare_deflated(panel, pol, fwd_bars=24)
    d_naive = naive['results']['only']['dsr']
    d_honest = honest['results']['only']['dsr']
    # zero-edge overlapping panel: the naive DSR is wildly optimistic, the
    # horizon-deflated one must be materially smaller
    assert d_honest < d_naive
    assert d_honest < 0.9, f"honest DSR still fails open: {d_honest}"
    assert honest['results']['only']['n_eff'] <= len(panel) / 24 + 1
    assert honest['results']['only']['fwd_bars'] == 24


def test_compare_deflated_default_unchanged_for_nonoverlap():
    rng = np.random.default_rng(9)
    panel = [[{'symbol': 'A', 'signal': 1.0,
               'fwd_return': float(rng.normal(0, 0.5))}] for _ in range(400)]
    out = compare_deflated(panel, {'only': top_k(1)})
    assert out['results']['only']['fwd_bars'] == 1
    assert out['results']['only']['n_eff'] == 400.0
