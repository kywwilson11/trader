"""R2C-06 measurement kernel suite tests (FR-04, FR-07-A, FR-03, M1, L7).

All synthetic-fixture, Mac-runnable: hand-checked EWMA/vol, the AR(1)
analytic transfer curve, shifted-distribution PSI, seeded entry-timing
probe rows, and the harvest TB-span guard (imported with the same dotenv
stub tests/test_improve_harvest.py already uses).
"""
import datetime
import gzip
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts'))
try:
    import dotenv  # noqa: F401
except ImportError:  # dev-Mac: stub load_dotenv so harvest imports
    _m = types.ModuleType('dotenv')
    _m.load_dotenv = lambda *a, **k: None
    sys.modules['dotenv'] = _m

import naive_baseline as nb              # noqa: E402
import horizon_transfer as ht            # noqa: E402
import stage0_preds                      # noqa: E402
import funding_drift_audit as fda        # noqa: E402
import entry_timing_probe as etp         # noqa: E402
import harvest_stock_data as h           # noqa: E402


def _hourly_ns(n, start='2026-01-05'):
    idx = pd.date_range(start, periods=n, freq='h', tz='UTC')
    return stage0_preds.index_ns(idx), idx


# ===========================================================================
# (a) FR-04 naive baseline — hand-checked EWMA / vol / signal
# ===========================================================================

def test_bar_returns_hand():
    out = nb.bar_returns([100.0, 110.0, 99.0])
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(10.0)
    assert out[2] == pytest.approx((99.0 - 110.0) / 110.0 * 100.0)


def test_bar_returns_bad_closes():
    out = nb.bar_returns([100.0, np.nan, 50.0, 0.0, 25.0])
    assert np.isnan(out[1]) and np.isnan(out[2])  # NaN prev/cur
    assert np.isnan(out[4])                        # zero prev
    assert out[3] == pytest.approx(-100.0)


def test_ewma_momentum_hand():
    # half_life=1 -> alpha=0.5: m = [1, 1.5, 2.75] by hand
    out = nb.ewma_momentum([1.0, 2.0, 4.0], half_life=1.0)
    assert out == pytest.approx([1.0, 1.5, 2.75])


def test_ewma_momentum_nan_seed_and_carry():
    out = nb.ewma_momentum([np.nan, 2.0, np.nan, 4.0], half_life=1.0)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(2.0)   # seeded at first finite
    assert out[2] == pytest.approx(2.0)   # NaN return carries m
    assert out[3] == pytest.approx(3.0)   # 0.5*2 + 0.5*4


def test_trailing_vol_hand():
    out = nb.trailing_vol([1.0, 2.0, 3.0, 5.0], window=2)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(np.std([1.0, 2.0], ddof=1))
    assert out[3] == pytest.approx(np.std([3.0, 5.0], ddof=1))


def test_naive_signal_strictly_trailing():
    rng = np.random.default_rng(7)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    sig_a = nb.naive_signal(closes, half_life=8, vol_window=24)
    tampered = closes.copy()
    tampered[150:] += 500.0  # rewrite the future
    sig_b = nb.naive_signal(tampered, half_life=8, vol_window=24)
    # signal through bar 149 must be identical (uses closes[0..i] only)
    np.testing.assert_allclose(sig_a[:150], sig_b[:150], equal_nan=True)


def test_naive_signal_warmup_and_flat():
    closes = np.full(50, 100.0)  # flat -> vol 0 -> NaN everywhere
    sig = nb.naive_signal(closes, half_life=4, vol_window=10)
    assert np.isnan(sig).all()


def test_lookup_at_times():
    t, _ = _hourly_ns(10)
    v = np.arange(10, dtype=float)
    out = nb.lookup_at_times(t, v, [t[3], t[7], t[9] + 1])
    assert out[0] == 3.0 and out[1] == 7.0
    assert np.isnan(out[2])


def test_long_only_returns_strict_gt():
    p = np.array([0.5, 0.0, -1.0, np.nan, 2.0])
    f = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    out = nb.long_only_returns(p, f, threshold=0.0)
    # 0.0 excluded (strict >), NaNs dropped
    assert out.tolist() == [1.0]


# ===========================================================================
# (b) FR-07-A horizon transfer — analytic fixtures
# ===========================================================================

def test_forward_returns_hand():
    out = ht.forward_returns([100.0, 110.0, 121.0], 2)
    assert out[0] == pytest.approx(21.0)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_iid_null_value():
    assert ht.iid_null(3, 12) == pytest.approx(0.5)


def test_strided_pair_spacing_and_finiteness():
    n = 100
    t, _ = _hourly_ns(n)
    x = np.arange(n, dtype=float)
    y = np.arange(n, dtype=float)
    x[5] = np.nan
    xs, ys, ts_ = ht.strided_pair(x, y, t, stride=10)
    idx = np.searchsorted(t, ts_)
    assert (np.diff(idx) >= 10).all()
    assert np.isfinite(xs).all() and np.isfinite(ys).all()


def _ar1_analytic_rho(phi, d, D):
    """corr(sum of first d AR(1) terms, sum of first D terms), gamma(k)
    proportional to phi^k."""
    g = lambda k: phi ** abs(k)  # / (1-phi^2) cancels in the ratio
    cov = sum(g(i - j) for i in range(d) for j in range(D))
    var_d = sum(g(i - j) for i in range(d) for j in range(d))
    var_D = sum(g(i - j) for i in range(D) for j in range(D))
    return cov / np.sqrt(var_d * var_D)


def _rolling_sum(r, h):
    """forward h-sum anchored at each bar, NaN tail — the additive
    analog of forward_returns for the analytic fixture."""
    out = np.full(len(r), np.nan)
    c = np.concatenate([[0.0], np.cumsum(r)])
    out[:len(r) - h] = c[h:-1] - c[:-h - 1]
    return out


def test_transfer_rho_iid_null():
    rng = np.random.default_rng(11)
    n, d, D = 60_000, 4, 12
    r = rng.normal(0, 1, n)
    t, _ = _hourly_ns(n)
    x, y, ts_ = ht.strided_pair(_rolling_sum(r, d), _rolling_sum(r, D),
                                t, stride=D)
    stat = ht.transfer_stat(x, y, ts_, n_boot=0)
    assert stat['rho'] == pytest.approx(ht.iid_null(d, D), abs=0.03)


def test_transfer_rho_ar1_analytic():
    phi = 0.5
    rng = np.random.default_rng(23)
    n, d, D = 60_000, 4, 12
    r = np.empty(n)
    r[0] = rng.normal()
    eps = rng.normal(0, 1, n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    t, _ = _hourly_ns(n)
    x, y, ts_ = ht.strided_pair(_rolling_sum(r, d), _rolling_sum(r, D),
                                t, stride=D)
    stat = ht.transfer_stat(x, y, ts_, n_boot=0)
    expect = _ar1_analytic_rho(phi, d, D)
    assert expect > ht.iid_null(d, D)  # persistence lifts the curve
    assert stat['rho'] == pytest.approx(expect, abs=0.04)


def test_block_bootstrap_se_sane_and_degenerate():
    rng = np.random.default_rng(3)
    n = 24 * 7 * 10  # 10 weeks hourly
    t, _ = _hourly_ns(n)
    x = rng.normal(0, 1, n)
    y = 0.5 * x + rng.normal(0, 1, n)
    blocks = ht.weekly_block_ids(t)
    se, reps = ht.block_bootstrap_se(
        lambda idx: float(np.corrcoef(x[idx], y[idx])[0, 1]),
        blocks, n_boot=60, seed=1)
    assert se is not None and 0.0 < se < 0.2
    assert reps == 60
    # one block -> not bootstrappable
    se1, r1 = ht.block_bootstrap_se(lambda idx: 1.0,
                                    np.zeros(50, dtype=np.int64))
    assert se1 is None and r1 == 0


def test_transfer_matrix_shapes():
    rng = np.random.default_rng(5)
    n = 24 * 7 * 8
    t, _ = _hourly_ns(n)
    per = {}
    for name in ('A', 'B'):
        r = rng.normal(0, 1, n)
        per[name] = ({4: _rolling_sum(r, 4), 12: _rolling_sum(r, 12)}, t)
    pairs = ht.transfer_matrix(per, n_boot=30, seed=0)
    assert len(pairs) == 1
    p = pairs[0]
    assert p['delta'] == 4 and p['Delta'] == 12
    assert set(p['per_name']) == {'A', 'B'}
    assert p['pooled']['n'] == (p['per_name']['A']['n']
                                + p['per_name']['B']['n'])
    assert p['null'] == pytest.approx(np.sqrt(4 / 12))


# ===========================================================================
# (c) FR-03 funding drift audit kernels
# ===========================================================================

def test_psi_same_distribution_small():
    rng = np.random.default_rng(0)
    psi = fda.psi_from_train_deciles(rng.normal(0, 1, 5000),
                                     rng.normal(0, 1, 5000))
    assert psi is not None and psi < 0.05


def test_psi_shifted_distribution_flags():
    rng = np.random.default_rng(1)
    psi = fda.psi_from_train_deciles(rng.normal(0, 1, 5000),
                                     rng.normal(1.5, 1, 5000))
    assert psi is not None and psi > fda.PSI_FLAG


def test_psi_degenerate_and_small():
    assert fda.psi_from_train_deciles(np.ones(500), np.ones(500)) is None
    assert fda.psi_from_train_deciles(np.ones(10), np.ones(500)) is None


def test_spearman_ci_covers_truth():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 400)
    y = x + rng.normal(0, 1, 400)
    r = fda.spearman_with_ci(x, y)
    assert r is not None
    rho, lo, hi, n = r
    assert lo < rho < hi and n == 400 and rho > 0.5


def test_sign_flip_disjoint():
    pos = (0.4, 0.3, 0.5, 100)
    neg = (-0.4, -0.5, -0.3, 100)
    overlap_neg = (-0.1, -0.35, 0.35, 100)
    assert fda.sign_flip_disjoint(pos, neg)
    assert not fda.sign_flip_disjoint(pos, overlap_neg)  # CIs overlap
    assert not fda.sign_flip_disjoint(pos, pos)
    assert not fda.sign_flip_disjoint(None, neg)


def test_strided_anchor_mask():
    m = fda.strided_anchor_mask(10, 3)
    assert m.tolist() == [True, False, False] * 3 + [True]


def _funding_frame(n=17520, seed=9):
    """Two-year hourly single-name frame: Funding_Z drives fwd + up
    pre-split and - post-split; distribution shifts in the last 90d."""
    rng = np.random.default_rng(seed)
    _, idx = _hourly_ns(n, start='2024-06-01')
    z = rng.normal(0, 1, n)
    t = stage0_preds.index_ns(idx)
    split_ns = int(pd.Timestamp('2026-01-01', tz='UTC').value)
    post = t >= split_ns
    fwd = np.where(post, -3.0 * z, 3.0 * z) + rng.normal(0, 1, n)
    live_cut = t.max() - 90 * 86400 * 10**9
    z_shifted = z + np.where(t >= live_cut, 4.0, 0.0)
    return pd.DataFrame({'Ticker': 'BTC-USD',
                         'Funding_Z': z_shifted,
                         'Target_Return_12': fwd}, index=idx)


def test_audit_frame_flags_shift_and_flip():
    df = _funding_frame()
    rows, meta = fda.audit_frame(df, pd.Timestamp('2026-01-01', tz='UTC'),
                                 trailing_days=90)
    assert meta['fwd_col'] == 'Target_Return_12'
    (row,) = rows
    assert row['column'] == 'Funding_Z'
    assert row['psi'] > fda.PSI_FLAG
    assert row['sign_flip_disjoint_ci'] is True
    assert row['flag'] is True and len(row['reasons']) == 2


def test_audit_frame_stable_no_flag():
    rng = np.random.default_rng(4)
    n = 17520
    _, idx = _hourly_ns(n, start='2024-06-01')
    z = rng.normal(0, 1, n)
    fwd = 3.0 * z + rng.normal(0, 1, n)
    df = pd.DataFrame({'Ticker': 'BTC-USD', 'Funding_Z': z,
                       'Target_Return_12': fwd}, index=idx)
    rows, _ = fda.audit_frame(df, pd.Timestamp('2026-01-01', tz='UTC'),
                              trailing_days=90)
    (row,) = rows
    assert row['flag'] is False and row['reasons'] == []


def test_audit_frame_no_funding_columns():
    _, idx = _hourly_ns(100)
    df = pd.DataFrame({'Ticker': 'X', 'Target_Return_12': 0.0},
                      index=idx)
    rows, meta = fda.audit_frame(df, pd.Timestamp('2026-01-01', tz='UTC'),
                                 90)
    assert rows == [] and 'error' in meta


# ===========================================================================
# (d) M1 entry-timing probe kernels
# ===========================================================================

def test_anchor_returns_hand():
    n, hzn = 12, 3
    t, _ = _hourly_ns(n)
    closes = np.linspace(100.0, 122.0, n)
    q = np.array([t[0], t[4], t[9], t[5] + 1])
    f_train, f_live = etp.anchor_returns(t, closes, q, hzn)
    # row at i=4: train anchor c[4]->c[7], live anchor c[3]->c[6]
    assert f_train[1] == pytest.approx(
        (closes[7] - closes[4]) / closes[4] * 100.0)
    assert f_live[1] == pytest.approx(
        (closes[6] - closes[3]) / closes[3] * 100.0)
    assert np.isnan(f_live[0])       # i=0 has no i-1
    assert np.isfinite(f_train[0])
    assert np.isnan(f_train[2])      # i=9, 9+3 out of range
    assert np.isfinite(f_live[2])    # i-1=8, 8+3=11 in range
    assert np.isnan(f_train[3]) and np.isnan(f_live[3])  # no exact match


def test_ic_anchor_delta_seeded_probe_material():
    rng = np.random.default_rng(42)
    n = 24 * 7 * 12  # 12 weeks of hourly rows
    t, _ = _hourly_ns(n)
    f_live = rng.normal(0, 1, n)
    f_train = rng.normal(0, 1, n)          # independent of pred
    pred = f_live + 0.5 * rng.normal(0, 1, n)
    res = etp.ic_anchor_delta(pred, f_train, f_live, t, n_boot=80,
                              seed=0)
    assert res['ic_live'] > 0.7 and abs(res['ic_train']) < 0.1
    assert res['delta'] == pytest.approx(res['ic_live'] - res['ic_train'])
    assert res['se'] is not None and res['material'] is True


def test_ic_anchor_delta_null_not_material():
    rng = np.random.default_rng(6)
    n = 24 * 7 * 12
    t, _ = _hourly_ns(n)
    fwd = rng.normal(0, 1, n)
    pred = fwd + 0.5 * rng.normal(0, 1, n)
    # both anchors identical -> delta exactly 0
    res = etp.ic_anchor_delta(pred, fwd, fwd, t, n_boot=40, seed=0)
    assert res['delta'] == pytest.approx(0.0)
    assert res['material'] is False


def test_fill_gap_seconds_hand():
    # crypto (:00 close): 10:05:00 UTC -> 300s past the hour
    ts_c = datetime.datetime(2026, 3, 2, 10, 5, 0,
                             tzinfo=datetime.timezone.utc).timestamp()
    assert etp.fill_gap_seconds([ts_c], 0)[0] == pytest.approx(300.0)
    # stock (:30 close): 14:37:20 -> 440s past the half-hour anchor
    ts_s = datetime.datetime(2026, 3, 2, 14, 37, 20,
                             tzinfo=datetime.timezone.utc).timestamp()
    assert etp.fill_gap_seconds([ts_s], 30)[0] == pytest.approx(440.0)


def test_scan_journal_buys(tmp_path):
    rows = [
        {'action': 'buy', 'symbol': 'BTC/USD',
         'ts': '2026-03-02T10:05:00+00:00'},
        {'action': 'sell', 'symbol': 'BTC/USD',
         'ts': '2026-03-02T12:00:00+00:00'},
        {'action': 'buy', 'symbol': 'AAPL',
         'ts': '2026-03-02T14:37:20+00:00'},
    ]
    p = tmp_path / '2026-03-02.jsonl'
    with open(p, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
        f.write('{corrupt line\n')
    gz_rows = [{'action': 'buy', 'symbol': 'ETH/USD',
                'ts': '2026-03-01T09:00:30+00:00'}]
    with gzip.open(tmp_path / '2026-03-01.jsonl.gz', 'wt') as f:
        for r in gz_rows:
            f.write(json.dumps(r) + '\n')
    out = etp.scan_journal_buys(tmp_path)
    syms = sorted(s for s, _ in out)
    assert syms == ['AAPL', 'BTC/USD', 'ETH/USD']
    assert etp.scan_journal_buys(tmp_path / 'nope') == []


# ===========================================================================
# stage0_preds additive 'close' field
# ===========================================================================

def test_build_rows_emits_close():
    n, hzn = 10, 2
    _, idx = _hourly_ns(n)
    closes = np.linspace(50.0, 59.0, n)
    rows = stage0_preds.build_rows(idx, 'BTC-USD', np.ones(n), closes,
                                   hzn, [0, 3])
    assert [r['close'] for r in rows] == [
        pytest.approx(round(closes[0], 6)),
        pytest.approx(round(closes[3], 6))]
    # existing schema preserved alongside
    for r in rows:
        for k in ('ts', 'symbol', 'pred', 'signal', 'fwd_return',
                  'horizon_bars'):
            assert k in r


# ===========================================================================
# (e) L7 harvest TB-span guard
# ===========================================================================

def _idx10():
    return pd.date_range('2026-01-05 14:30', periods=10, freq='h',
                         tz='UTC')


def test_removals_prefix_suffix_ok():
    pre = _idx10()
    ok, n_bad = h._removals_prefix_suffix_only(pre, pre[2:8])
    assert ok and n_bad == 0
    ok, _ = h._removals_prefix_suffix_only(pre, pre)       # no removal
    assert ok
    ok, _ = h._removals_prefix_suffix_only(pre, pre[:0])   # all removed
    assert ok


def test_removals_interior_violation():
    pre = _idx10()
    post = pre[[0, 1, 2, 5, 6]]  # interior hole at 3-4
    ok, n_bad = h._removals_prefix_suffix_only(pre, post)
    assert not ok and n_bad == 1


def test_removals_foreign_rows_violation():
    pre = _idx10()
    foreign = pd.date_range('2027-01-01', periods=3, freq='h', tz='UTC')
    ok, n_bad = h._removals_prefix_suffix_only(pre, foreign)
    assert not ok and n_bad == 3


def test_warn_tb_span_violation_prints(capsys):
    pre = _idx10()
    assert h._warn_tb_span_violation(pre, pre[1:9], 'AAPL', 'dropna')
    assert '[TB-GUARD]' not in capsys.readouterr().out
    assert not h._warn_tb_span_violation(pre, pre[[0, 5]], 'AAPL',
                                         'dropna')
    assert '[TB-GUARD] AAPL' in capsys.readouterr().out


def test_tb_membership_guard_per_ticker(capsys):
    pre_idx = _idx10()
    pre = pd.concat([
        pd.DataFrame({'Ticker': 'AAA'}, index=pre_idx),
        pd.DataFrame({'Ticker': 'BBB'}, index=pre_idx),
    ]).sort_index()
    # AAA loses a prefix (fine); BBB loses interior rows (violation);
    # a fully-removed ticker would simply be absent from post (fine)
    post = pd.concat([
        pd.DataFrame({'Ticker': 'AAA'}, index=pre_idx[3:]),
        pd.DataFrame({'Ticker': 'BBB'}, index=pre_idx[[0, 1, 7]]),
    ]).sort_index()
    ok = h._tb_membership_guard(pre, post)
    out = capsys.readouterr().out
    assert not ok
    assert '[TB-GUARD] BBB' in out and 'AAA' not in out.replace(
        '[TB-GUARD] BBB', '')


def test_warn_tb_span_violation_fail_soft(capsys):
    """A broken CHECK (non-unique per-ticker index makes get_indexer
    raise) must not kill the harvest and must NOT read as a violation —
    it prints its own 'check skipped' line and returns True."""
    dup = _idx10().append(_idx10())  # non-unique pre index
    post = _idx10()[2:8]
    assert h._warn_tb_span_violation(dup, post, 'AAPL', 'dropna') is True
    out = capsys.readouterr().out
    assert 'check skipped' in out and 'NOT a violation' in out


def test_guard_wired_into_harvest_source():
    src = (BASE / 'scripts' / 'harvest_stock_data.py').read_text()
    # per-ticker guard sits after the tradability mask
    i_mask = src.index('_asof_tradability_mask(df, ticker)')
    i_guard = src.index('_warn_tb_span_violation(_tb_stamp_index')
    assert i_guard > i_mask
    # stamped index captured right after TB stamping
    assert src.index('_tb_stamp_index = df.index') > src.index(
        'compute_tb_labels(df, FORWARD_BARS')
    # membership-mask guard sits after the membership mask call in main
    i_member = src.index('_asof_membership_mask(final_df)')
    assert src.index('_tb_membership_guard(_pre_member') > i_member


# ===========================================================================
# driver importability (Mac): the four scripts import without side effects
# ===========================================================================

def test_drivers_import_clean():
    import naive_vs_blend            # noqa: F401
    import horizon_transfer_report   # noqa: F401
    assert callable(naive_vs_blend.main)
    assert callable(horizon_transfer_report.main)
    assert callable(fda.main)
    assert callable(etp.main)


def test_attach_naive_and_threshold_mask():
    import naive_vs_blend as nvb
    n = 200
    t, idx = _hourly_ns(n)
    rng = np.random.default_rng(8)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, n))
    series = {'BTC-USD': (t, closes)}
    rows = [{'symbol': 'BTC/USD', 'ts': str(idx[150]), 'pred': 0.5,
             'fwd_return': 1.0, 'pred_thresh_ratio': 1.2},
            {'symbol': 'BTC/USD', 'ts': str(idx[151]), 'pred': 0.1,
             'fwd_return': -1.0, 'pred_thresh_ratio': 0.8},
            {'symbol': 'ZZZ', 'ts': str(idx[10]), 'pred': 0.2,
             'fwd_return': 0.0}]
    unmatched = nvb.attach_naive(rows, series, half_life=8,
                                 vol_window=24)
    assert unmatched == 1                    # ZZZ has no series
    assert rows[0]['naive'] is not None      # '/'->'-' normalization
    mask = nvb._blend_threshold_mask(rows)
    assert mask.tolist() == [True, False, True]  # ratio>1, ratio<1, pred>0
    assert nvb._cum_trials('crypto', 500) == 500
    # FR-04 'identical stage0 rows': the comparison set is exactly the
    # joined rows — warmup/unmatched rows are excluded from BOTH arms
    joined = nvb.joined_rows(rows)
    assert [r['symbol'] for r in joined] == ['BTC/USD', 'BTC/USD']
    assert all(r['naive'] is not None for r in joined)
    # and main() actually restricts to that set before scoring
    import inspect
    src = inspect.getsource(nvb.main)
    assert 'rows = joined_rows(rows)' in src
