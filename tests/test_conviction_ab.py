"""Wave-9 #4: conviction A/B evaluator + the multiple-testing deflation fix.

The headline integrity fix: compare() ranked policies by RAW Sharpe with no
correction for the number of policies tried (a mined winner looks real).
compare_deflated deflates by the policy-pool size, so the SAME winner scores
lower DSR when it was the best of MORE tries. Plus: panel PIT construction, the
no-op kill switch, and edge-proportional sizing beating equal-weight on a gradient.
"""
import numpy as np
import pandas as pd
import pytest

from portfolio_backtest import (
    top_k,
    conviction_gated,
    run_policy,
    compare_deflated,
    panel_from_frame,
    edge_proportional_weights,
    run_policy_weighted,
)


def _gradient_panel(n_periods=300, k=7, seed=1):
    """Panel where rank-1 by signal earns the most fwd_return (a real gradient)."""
    rng = np.random.default_rng(seed)
    panel = []
    for _ in range(n_periods):
        sig = np.sort(rng.normal(size=k))[::-1]            # descending signal
        fwd = 0.4 * (sig - sig.mean()) + rng.normal(0, 0.3, k)  # edge ~ signal
        panel.append([{'symbol': f'S{i}', 'signal': float(sig[i]),
                       'fwd_return': float(fwd[i])} for i in range(k)])
    return panel


def test_panel_from_frame_builds_periods_with_pit_lag():
    idx = pd.to_datetime(['2026-06-18 10:00'] * 3 + ['2026-06-18 11:00'] * 3)
    df = pd.DataFrame({
        'Ticker': ['A', 'B', 'C', 'A', 'B', 'C'],
        'sig': [0.3, 0.1, 0.2, 0.5, 0.4, 0.0],
        'fwd': [1.0, -0.5, 0.2, 0.8, 0.1, -0.3],
    }, index=idx)
    panel = panel_from_frame(df, 'sig', 'fwd')
    assert len(panel) == 2 and len(panel[0]) == 3
    assert {c['symbol'] for c in panel[0]} == {'A', 'B', 'C'}
    # signal_lag drops the first period per ticker (no prior signal)
    lagged = panel_from_frame(df, 'sig', 'fwd', signal_lag=1)
    assert len(lagged) == 1                                  # only the 2nd timestamp survives


def test_top_k_recovers_the_injected_gradient():
    panel = _gradient_panel()
    m = run_policy(panel, top_k(3), cost_pct=0.0)
    assert m['net_total'] > 0                                # buying the top names pays


def test_compare_deflated_responds_to_the_policy_pool_size():
    panel = _gradient_panel()
    winner = conviction_gated(k_max=3)                       # the policy we will "select"
    few = compare_deflated(panel, {'base': top_k(7), 'win': winner})
    noise = {f'noise{i}': top_k(int(1 + i % 7)) for i in range(6)}
    many = compare_deflated(panel, {'base': top_k(7), 'win': winner, **noise})
    # Same winning policy, more tries -> a HIGHER expected-max bar -> LOWER DSR.
    assert many['results']['win']['dsr'] <= few['results']['win']['dsr']
    assert many['results']['win']['expected_max_sr'] > few['results']['win']['expected_max_sr']
    # turnover is now a first-class, reported metric
    assert 'turnover' in many['results']['win']


def test_conviction_gate_is_a_true_noop_kill_switch():
    panel = _gradient_panel(seed=2)
    # With CONCENTRATION_ENABLED False the conviction walk == flat top-K exactly.
    incumbent = run_policy(panel, top_k(7))
    no_op = run_policy(panel, conviction_gated(k_max=7))     # no floors applied
    assert incumbent['net_total'] == no_op['net_total']
    assert incumbent['mean_admitted_k'] == no_op['mean_admitted_k']


def test_edge_proportional_weights_properties():
    w = edge_proportional_weights([0.4, 0.2, 0.1])
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]                                # more weight to higher edge
    # no positive edge -> equal weight fallback
    eq = edge_proportional_weights([-1.0, -2.0, -0.5])
    assert np.allclose(eq, 1 / 3)


def test_edge_proportional_beats_equal_weight_on_a_gradient():
    panel = _gradient_panel(seed=3)
    pol = top_k(7)
    equal = run_policy(panel, pol)['net_total']
    weighted = run_policy_weighted(panel, pol, edge_proportional_weights)['net_total']
    assert weighted > equal                                  # concentrating into edge pays
