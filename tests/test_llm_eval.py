"""Wave-8 #3: the incremental-over-pred LLM statistics.

These exercise the pure compute path of llm_eval (no Alpaca, no torch). The key
properties: a redundant LLM that merely ECHOES the ML pred is exposed (high raw
rho, ~0 partial, insignificant b2), genuine orthogonal alpha is RECOVERED even
when the raw rho is ~0, and the verdict ABSTAINS on small samples.
"""
import numpy as np
import pytest

from llm_eval import (
    compute_incremental_report,
    partial_spearman,
    two_by_two_grid,
    _newey_west_se,
    _ols_beta_resid,
    _avg_rank,
    _pearson,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _partial_corr_closed_form(s, realized, pred):
    """First-order partial Spearman via the algebraic identity (independent oracle)."""
    rs, rr, rp = _avg_rank(s), _avg_rank(realized), _avg_rank(pred)
    r_sy = _pearson(rs, rr)
    r_sp = _pearson(rs, rp)
    r_yp = _pearson(rr, rp)
    return (r_sy - r_sp * r_yp) / np.sqrt((1 - r_sp**2) * (1 - r_yp**2))


def test_partial_spearman_matches_closed_form():
    rng = np.random.default_rng(1)
    pred = rng.normal(size=1500)
    g = rng.normal(size=1500)
    realized = 0.8 * pred + 0.5 * g + rng.normal(scale=0.5, size=1500)
    s = _sigmoid(0.6 * g + 0.3 * pred)
    mine, degen = partial_spearman(s, realized, pred)
    assert not degen
    assert mine == pytest.approx(_partial_corr_closed_form(s, realized, pred), abs=1e-9)


def test_redundant_llm_is_exposed_high_raw_low_partial():
    # s is (almost) a monotone function of pred => it ECHOES the model.
    rng = np.random.default_rng(2)
    pred = rng.normal(size=2000)
    realized = 1.2 * pred + rng.normal(scale=0.8, size=2000)
    s = _sigmoid(2.0 * pred + rng.normal(scale=0.05, size=2000))
    rep = compute_incremental_report(list(zip(s, realized, pred)), forward_bars=24, min_n=60)
    assert rep['raw_spearman_s_vs_return'] > 0.30          # looks predictive...
    assert abs(rep['partial_spearman_s_given_pred']) < 0.10  # ...but it's just the echo
    assert rep['echo_gap'] > 0.25
    # b2 should NOT be significantly positive once pred is controlled for.
    enc = rep['encompassing']
    assert not (enc['p_value'] < 0.05 and enc['b2_s'] > 0)
    assert 'ECHO' in rep['verdict'] or 'no measurable incremental' in rep['verdict']


def test_orthogonal_alpha_is_recovered_when_raw_is_weak():
    # pred dominates realized; the LLM tracks an ORTHOGONAL driver g.
    rng = np.random.default_rng(3)
    pred = rng.normal(size=3000)
    g = rng.normal(size=3000)                       # independent of pred
    realized = 2.0 * pred + 0.7 * g + rng.normal(scale=0.5, size=3000)
    s = _sigmoid(1.5 * g + rng.normal(scale=0.05, size=3000))
    rep = compute_incremental_report(list(zip(s, realized, pred)), forward_bars=24, min_n=60)
    raw = abs(rep['raw_spearman_s_vs_return'])
    partial = rep['partial_spearman_s_given_pred']
    assert partial > raw                            # partial recovers what raw understates
    assert partial > 0.10
    enc = rep['encompassing']
    assert enc['b2_s'] > 0 and enc['p_value'] < 0.05
    assert 'INCREMENTAL' in rep['verdict']


def test_null_has_no_incremental_value():
    rng = np.random.default_rng(4)
    pred = rng.normal(size=1500)
    realized = 1.0 * pred + rng.normal(scale=1.0, size=1500)
    s = _sigmoid(rng.normal(size=1500))             # pure noise, unrelated to anything
    rep = compute_incremental_report(list(zip(s, realized, pred)), forward_bars=24, min_n=60)
    assert abs(rep['partial_spearman_s_given_pred']) < 0.10
    enc = rep['encompassing']
    assert not (enc['p_value'] < 0.05 and enc['b2_s'] > 0)


def test_small_sample_abstains():
    rng = np.random.default_rng(5)
    pred = rng.normal(size=40)
    realized = pred + rng.normal(size=40)
    s = _sigmoid(pred)
    rep = compute_incremental_report(list(zip(s, realized, pred)), forward_bars=24, min_n=60)
    assert rep['insufficient_power'] is True
    assert 'insufficient_power' in rep['verdict']


def test_degenerate_pred_flagged_not_crashed():
    rng = np.random.default_rng(6)
    pred = np.full(200, 0.001)                      # constant ML pred
    realized = rng.normal(size=200)
    s = _sigmoid(rng.normal(size=200))
    rep = compute_incremental_report(list(zip(s, realized, pred)), forward_bars=24, min_n=60)
    assert rep['pred_degenerate'] is True
    assert 'pred_degenerate' in rep['verdict']


def test_two_by_two_grid_counts_and_disagreement_cells():
    s = np.array([0.9, 0.9, 0.1, 0.1, 0.8, 0.2])
    pred = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    realized = np.array([1.0, 2.0, -1.0, -2.0, 0.5, -0.5])
    g = two_by_two_grid(s, realized, pred)
    assert g['agree_bull']['n'] == 2          # s>=.5 & pred>0  (idx 0,4)
    assert g['agree_bear']['n'] == 2          # s<.5 & pred<=0  (idx 3,5)
    assert g['llm_bull_ml_bear']['n'] == 1    # idx 1
    assert g['llm_bear_ml_bull']['n'] == 1    # idx 2
    assert g['agree_bull']['avg_fwd_ret_pct'] == pytest.approx(0.75)


def test_timestamp_ordering_used_for_hac():
    # 4-tuples with shuffled t0 must be handled (sorted) without error.
    rng = np.random.default_rng(7)
    n = 300
    pred = rng.normal(size=n)
    realized = pred + rng.normal(size=n)
    s = _sigmoid(pred)
    t0 = rng.permutation(n).astype(float)
    rep = compute_incremental_report(list(zip(s, realized, pred, t0)), forward_bars=12, min_n=60)
    assert rep['n'] == n
    assert rep['encompassing'] is not None


def test_newey_west_matches_statsmodels_oracle():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(8)
    n = 1200
    pred = rng.normal(size=n)
    s = rng.normal(size=n)
    z_s = (s - s.mean()) / s.std()
    realized = 0.5 * pred + 0.3 * z_s + rng.normal(size=n)
    X = np.column_stack([np.ones(n), pred, z_s])
    beta, resid = _ols_beta_resid(X, realized)
    mine = _newey_west_se(X, resid, lag=23)
    res = sm.OLS(realized, X).fit(cov_type='HAC',
                                  cov_kwds={'maxlags': 23, 'use_correction': False})
    np.testing.assert_allclose(beta, res.params, rtol=1e-8)
    np.testing.assert_allclose(mine, res.bse, rtol=1e-4, atol=1e-6)
