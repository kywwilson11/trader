"""Campaign 2026-08 packet V4 — stationary-bootstrap Sharpe diagnostic.

Pins validation.politis_white_block_length (Politis-White 2004 automatic
block length, PPW-2009 D_SB correction) and
validation.stationary_bootstrap_sharpe_pvalue (Politis-Romano 1994
stationary bootstrap, shift-method one-sided p-value): seeded
determinism, no global-RNG contamination, size under the iid null, power
under drifted AR(1) noise, block-length sanity/clamps, and the fail-open
status-dict contract. Runs entirely on the dev Mac (numpy only).
n_boot is reduced below the 1000 production default purely for speed.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import validation as V

RESULT_KEYS = {'p_value', 'sharpe', 'block_len', 'n_boot', 'n_boot_used',
               'ci90', 'n', 'n_dropped', 'status'}


def _series(n=300, seed=0, mu=0.05):
    return np.random.default_rng(seed).normal(mu, 1.0, n)


def _ar1(n, phi, seed, mu=0.0):
    e = np.random.default_rng(seed).normal(0.0, 1.0, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return mu + x


def test_seeded_determinism():
    r = _series()
    a = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=300, seed=7)
    b = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=300, seed=7)
    assert a == b
    assert a['p_value'] == b['p_value']
    assert a['ci90'] == b['ci90']
    assert a['block_len'] == b['block_len']
    c = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=300, seed=8)
    assert c['p_value'] != a['p_value']
    # politis_white_block_length is a pure function
    assert V.politis_white_block_length(r) == V.politis_white_block_length(r)


def test_no_global_rng_state():
    np.random.seed(0)
    baseline = np.random.random()
    np.random.seed(0)
    V.stationary_bootstrap_sharpe_pvalue(_series(), n_boot=100, seed=9)
    assert np.random.random() == baseline


def test_size_under_iid_null():
    ps = []
    for i in range(60):
        d = np.random.default_rng(i).normal(0.0, 1.0, 250)
        res = V.stationary_bootstrap_sharpe_pvalue(d, n_boot=300, seed=i)
        assert res['status'] == 'ok'
        ps.append(res['p_value'])
    ps = np.asarray(ps)
    assert (ps <= 0.05).mean() <= 0.15
    assert 0.35 < ps.mean() < 0.65


def test_power_positive_drift_ar():
    r = _ar1(400, phi=0.3, seed=3, mu=0.2)
    res = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=800, seed=0)
    assert res['status'] == 'ok'
    assert res['sharpe'] > 0
    assert res['p_value'] < 0.05


def test_block_length_sanity():
    n = 2000
    cap = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    rng = np.random.default_rng(42)
    iid = rng.normal(0.0, 1.0, n)
    b_iid = V.politis_white_block_length(iid)
    assert 1.0 <= b_iid <= 6.0
    b_ar = V.politis_white_block_length(_ar1(n, phi=0.9, seed=42))
    assert b_ar > 2.0 * b_iid
    assert b_ar >= 8.0
    assert b_iid <= cap and b_ar <= cap
    # fail-open guards
    assert V.politis_white_block_length(np.ones(10)) == 1.0        # n < 20
    assert V.politis_white_block_length(np.full(100, 3.0)) == 1.0  # constant


def test_auto_equals_explicit_block_len():
    r = _series(seed=1)
    b = V.politis_white_block_length(r)
    auto = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=300, seed=3)
    expl = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=300, seed=3,
                                                block_len=b)
    assert auto == expl
    assert auto['block_len'] == b


def test_block_len_override_and_clamps():
    r = _series(seed=2)
    n = r.size
    p1 = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=400, seed=3,
                                              block_len=1)
    p20 = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=400, seed=3,
                                               block_len=20)
    assert p1['p_value'] != p20['p_value']
    assert p1['block_len'] == 1.0 and p20['block_len'] == 20.0
    big = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=50, seed=3,
                                               block_len=1e9)
    assert big['block_len'] == float(n)
    small = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=50, seed=3,
                                                 block_len=0.3)
    assert small['block_len'] == 1.0
    auto = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=400, seed=3)
    nanb = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=400, seed=3,
                                                block_len=float('nan'))
    assert nanb == auto  # non-finite override falls back to the auto value


def test_fail_open_guards():
    ok = V.stationary_bootstrap_sharpe_pvalue(_series(), n_boot=100, seed=1)
    short = V.stationary_bootstrap_sharpe_pvalue(np.ones(10))
    assert short['status'] == 'insufficient_n'
    assert short['p_value'] is None and short['ci90'] is None
    assert short['block_len'] is None      # skeleton: no block length computed
    assert short['n_boot'] == 1000         # default echoed even on fail-open
    assert short['n_boot_used'] == 0
    assert set(short) == set(ok)
    # The documented floor is 20 (PW bandwidth), not the module's usual 10.
    assert V.stationary_bootstrap_sharpe_pvalue(
        _series(n=19), n_boot=50, seed=1)['status'] == 'insufficient_n'
    assert V.stationary_bootstrap_sharpe_pvalue(
        _series(n=20), n_boot=50, seed=1)['status'] == 'ok'
    const = V.stationary_bootstrap_sharpe_pvalue(np.full(100, 2.5))
    assert const['status'] == 'degenerate'
    assert const['p_value'] is None
    assert const['block_len'] is None  # degenerate before block-length step
    assert set(const) == set(ok)
    withnan = np.concatenate([_series(), [np.nan, np.inf, -np.inf]])
    res = V.stationary_bootstrap_sharpe_pvalue(withnan, n_boot=100, seed=1)
    assert res['status'] == 'ok'
    assert res['n_dropped'] == 3
    assert res['n'] == 300
    two_d = V.stationary_bootstrap_sharpe_pvalue(_series().reshape(30, 10),
                                                 n_boot=100, seed=1)
    assert two_d['status'] == 'ok'  # ravel-flattened
    assert two_d['n'] == 300


def test_docstring_mandates():
    # Spec V4.3: the docstring notes are deliverables — pin them so a
    # later trim cannot silently drop the kill-screen / citation content.
    doc = V.stationary_bootstrap_sharpe_pvalue.__doc__
    assert 'Politis & Romano (1994' in doc          # method citation
    assert 'Ledoit-Wolf (2008)' in doc              # non-studentized, upgrade path
    assert 'KILL_LIST.md:103' in doc                # kill-screen distinction
    assert 'CHRONOLOGICAL' in doc                   # input-order caveat (B02)
    assert 'DIAGNOSTIC ONLY' in doc                 # feeds no promotion verdict
    bdoc = V.politis_white_block_length.__doc__
    assert 'Politis & White (2004' in bdoc
    assert '2009' in bdoc and 'D_SB' in bdoc        # PPW correction named
    assert 'stationary-bootstrap Sharpe p-value' in V.__doc__


def test_result_contract():
    r = _series(seed=5)
    res = V.stationary_bootstrap_sharpe_pvalue(r, n_boot=250, seed=4)
    assert set(res) == RESULT_KEYS
    assert res['status'] == 'ok'
    assert isinstance(res['ci90'], list) and len(res['ci90']) == 2
    assert res['ci90'][0] <= res['ci90'][1]
    assert res['n_boot'] == 250
    assert 0 < res['n_boot_used'] <= res['n_boot']
    assert res['sharpe'] == pytest.approx(r.mean() / r.std())
    assert 0.0 < res['p_value'] <= 1.0
    assert res['block_len'] >= 1.0
    assert res['n'] == r.size and res['n_dropped'] == 0
