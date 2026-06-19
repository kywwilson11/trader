"""Wave-9 #4/#5 gate: rank-gradient Stage-0 verdict (holdout + live)."""
import numpy as np
import pytest

from rank_gradient import rank_gradient_from_panel, rank_gradient_verdict


def _panel(gradient=True, n_periods=200, k=7, seed=1):
    rng = np.random.default_rng(seed)
    panel = []
    for _ in range(n_periods):
        sig = np.sort(rng.normal(size=k))[::-1]
        if gradient:
            fwd = 0.4 * (sig - sig.mean()) + rng.normal(0, 0.2, k)   # edge ~ rank
        else:
            fwd = 0.1 + rng.normal(0, 0.2, k)                        # flat: rank carries nothing
        panel.append([{'symbol': f'S{i}', 'signal': float(sig[i]),
                       'fwd_return': float(fwd[i])} for i in range(k)])
    return panel


def test_from_panel_recovers_bucket_ordering():
    b = rank_gradient_from_panel(_panel(gradient=True))
    assert b['rank_1_3']['mean_net_pct'] > b['rank_6_7']['mean_net_pct']
    assert b['rank_1_3']['n'] > 0 and b['rank_6_7']['n'] > 0


def test_verdict_confirms_real_gradient():
    v = rank_gradient_verdict(rank_gradient_from_panel(_panel(gradient=True)))
    assert v['gradient_exists'] is True
    assert 'CONFIRMED' in v['verdict']
    assert v['ratio_6_7_over_1_3'] < 0.5


def test_verdict_rejects_flat_universe():
    v = rank_gradient_verdict(rank_gradient_from_panel(_panel(gradient=False)))
    assert v['gradient_exists'] is False
    assert 'NEITHER' in v['verdict']


def test_verdict_passes_when_bottom_rank_is_negative():
    buckets = {'rank_1_3': {'mean_net_pct': 0.30}, 'rank_6_7': {'mean_net_pct': -0.05}}
    v = rank_gradient_verdict(buckets)
    assert v['gradient_exists'] is True               # rank_6_7 <= 0 alone is enough


def test_verdict_handles_decision_report_shape_and_missing_buckets():
    # decision_report emits {'rank_1_3': {'n','mean_net_pct','hit_rate'}, ...}
    live = {'rank_1_3': {'n': 50, 'mean_net_pct': 0.4, 'hit_rate': 0.6},
            'rank_6_7': {'n': 40, 'mean_net_pct': 0.05, 'hit_rate': 0.5}}
    assert rank_gradient_verdict(live)['gradient_exists'] is True
    # missing a bucket -> insufficient, not a crash
    assert rank_gradient_verdict({'rank_1_3': {'mean_net_pct': 0.3}})['gradient_exists'] is None
