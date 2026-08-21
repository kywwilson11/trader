"""Packet c26_Q1 (D02/B03): effective-n + selection-pressure accounting.

Pins, per the spec:
  A. sample_weights.calendar_effective_n — the AFML ch.4 calendar-hour
     average-uniqueness estimator (single-name reduction, the calendar-tiling
     degenerate case clustered_effective_n collapses on, lockstep stacks,
     disjoint-add exactness + [n/c_max, n] bounds, input-contract parity
     with clustered_effective_n, Kish design-effect softening).
  B. validation — the fail-closed n_eff<10 floor under PROMOTION_GATE_V2,
     OFF-mode byte-identity of the legacy clamp, key-set parity across all
     branches, min_track_record_length.
  C. adaptive_config — cum_trials/trial_history accrual (BEFORE the study-DB
     deletion path), db_deletions audit records, record_trials /
     record_db_deletion round-trips, overlap_weighted_trials, noisy_ratchet.
  D. backtest.aggregate_metrics — OFF-mode byte-pin against an independent
     replication, flag-ON calendar consumption + fail-closed status, key
     parity, Kish wiring, and main()'s --trials pool resolution.

NOTE (deviation from the packet spec, recorded in the report): the spec's
"monotone non-decreasing in added trades" property is mathematically FALSE
for calendar average uniqueness — one long trade blanketing k disjoint
short ones moves n_eff from k to (k+1)/2 — so the property test here pins
what IS true (a DISJOINT added trade adds exactly 1.0; n_eff stays within
[n/max_concurrency, n]) plus the explicit counterexample.
"""
import math
import sqlite3
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import adaptive_config as AC
import backtest
import validation as V
from sample_weights import (average_uniqueness, calendar_effective_n,
                            clustered_effective_n, effective_n)


# ---------------------------------------------------------------------------
# A. calendar_effective_n
# ---------------------------------------------------------------------------

def _ts(hours):
    base = np.datetime64('2026-08-01T00:00')
    return np.array([base + np.timedelta64(int(h * 60), 'm') for h in hours])


def test_single_name_disjoint_counts_fully():
    e = np.array([0.0, 10.0, 20.0, 30.0])
    x = np.array([5.0, 15.0, 25.0, 35.0])
    out = calendar_effective_n(e, x)
    assert out['n_eff'] == pytest.approx(4.0)
    assert out['n_trades'] == 4
    assert out['max_concurrency'] == 1
    assert np.allclose(out['u'], 1.0)


def test_single_name_reduces_to_within_ticker_uniqueness():
    # One ticker's trades expressed as row-offset hold_bars (rows == hours)
    # and as calendar intervals must produce identical per-trade uniqueness.
    rows = np.array([0, 3, 5, 10, 20])
    spans = np.array([4, 2, 6, 3, 5], dtype=float)
    n_rows = int(rows.max() + spans.max()) + 2
    masked = np.full(n_rows, np.nan)
    masked[rows] = spans
    u_block = average_uniqueness(masked)[rows]

    out = calendar_effective_n(rows.astype(float), rows + spans)
    assert np.allclose(out['u'], u_block)
    assert out['n_eff'] == pytest.approx(effective_n(u_block))


def test_calendar_tiling_degenerate_case():
    # 100-trade chain, each overlapping only its neighbours: the pinned
    # connected-components defect collapses it to ONE cluster while the
    # calendar estimator keeps ~2/3 of the trades.
    n = 100
    e = np.array([4.0 * k for k in range(n)])
    x = e + 5.0
    assert clustered_effective_n(e, x) == 1          # the D02 defect, pinned
    out = calendar_effective_n(e, x)
    assert out['n_eff'] >= n / 2 - 1
    assert out['n_eff'] <= n


def test_lockstep_stacks_share_uniqueness():
    # 6 names entering/exiting the same hours in 10 disjoint slots:
    # u_i ~ 1/6 each, n_eff ~ n_slots.
    n_slots, n_names = 10, 6
    e, x = [], []
    for s in range(n_slots):
        for _ in range(n_names):
            e.append(100.0 * s)
            x.append(100.0 * s + 23.0)
    out = calendar_effective_n(np.array(e), np.array(x))
    assert out['max_concurrency'] == 6
    assert np.allclose(out['u'], 1.0 / 6.0)
    assert out['n_eff'] == pytest.approx(float(n_slots))


def test_disjoint_add_and_bounds_property():
    # DEVIATION from the spec'd (false) global-monotonicity claim: pin the
    # true properties — a disjoint added trade adds exactly 1.0 and n_eff
    # always stays within [n/max_concurrency, n].
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = int(rng.integers(2, 40))
        e = rng.uniform(0, 300, n)
        x = e + rng.uniform(0.1, 30, n)
        out = calendar_effective_n(e, x)
        assert out['n_eff'] <= n + 1e-9
        assert out['n_eff'] >= n / out['max_concurrency'] - 1e-9
        # Add one trade disjoint from everything: n_eff rises by exactly 1.
        far = float(np.max(x)) + 10.0
        e2 = np.append(e, far)
        x2 = np.append(x, far + 5.0)
        out2 = calendar_effective_n(e2, x2)
        assert out2['n_eff'] == pytest.approx(out['n_eff'] + 1.0)


def test_nonmonotone_counterexample_documented():
    # k disjoint single-hour trades + one blanket trade: n_eff (k) drops to
    # (k+1)/2 — why the docstring does NOT claim global monotonicity.
    k = 10
    e = np.array([float(i) for i in range(k)])
    x = e.copy()
    base = calendar_effective_n(e, x)['n_eff']
    assert base == pytest.approx(float(k))
    e2 = np.append(e, 0.0)
    x2 = np.append(x, float(k - 1))
    out2 = calendar_effective_n(e2, x2)
    assert out2['n_eff'] == pytest.approx((k + 1) / 2.0)
    assert out2['n_eff'] < base


def test_input_contract_parity_with_clustered():
    # Same contract as clustered_effective_n, pinned.
    with pytest.raises(ValueError):
        calendar_effective_n(np.array([1.0, 2.0]), np.array([3.0]))
    with pytest.raises(TypeError):
        calendar_effective_n(np.array([pd.Timestamp('2026-01-01')],
                                      dtype=object),
                             np.array([pd.Timestamp('2026-01-02')],
                                      dtype=object))
    with pytest.raises(TypeError):
        calendar_effective_n(_ts([0, 1]), np.array([1.0, 2.0]))
    # NaT / NaN pairs dropped
    e = np.array([0.0, 1.0, np.nan, 50.0])
    x = np.array([5.0, 2.0, 3.0, 55.0])
    assert calendar_effective_n(e, x)['n_trades'] == 3
    et = _ts([0, 10])
    xt = et + np.timedelta64(5, 'h')
    xt2 = xt.copy()
    xt2[1] = np.datetime64('NaT')
    assert calendar_effective_n(et, xt2)['n_trades'] == 1
    # exit < entry -> point interval (not an error, still counted)
    out = calendar_effective_n(np.array([0.0, 10.0]),
                               np.array([-5.0, 12.0]))
    assert out['n_trades'] == 2
    assert out['n_eff'] == pytest.approx(2.0)
    # empty
    out = calendar_effective_n(np.array([]), np.array([]))
    assert out['n_eff'] == 0.0 and out['n_trades'] == 0
    assert out['u_bar_mean'] is None and out['max_concurrency'] == 0


def test_datetime_and_numeric_hour_inputs_agree():
    hours_e = [0, 2, 4, 50]
    hours_x = [3, 5, 6, 60]
    d = calendar_effective_n(_ts(hours_e), _ts(hours_x))
    f = calendar_effective_n(np.array(hours_e, dtype=float),
                             np.array(hours_x, dtype=float))
    assert d['n_eff'] == pytest.approx(f['n_eff'])
    assert np.allclose(d['u'], f['u'])


def test_kish_rho_softening():
    e, x = [], []
    for s in range(8):
        for _ in range(6):
            e.append(100.0 * s)
            x.append(100.0 * s + 23.0)
    e, x = np.array(e), np.array(x)
    plain = calendar_effective_n(e, x)
    rho1 = calendar_effective_n(e, x, rho_bar=1.0)
    assert rho1['n_eff'] == pytest.approx(plain['n_eff'])
    assert np.allclose(rho1['u'], plain['u'])
    rho_half = calendar_effective_n(e, x, rho_bar=0.5)
    # Softer than lockstep, but never above the raw count.
    assert plain['n_eff'] < rho_half['n_eff'] < len(e)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            calendar_effective_n(e, x, rho_bar=bad)


# ---------------------------------------------------------------------------
# B. validation: fail-closed floor, OFF-mode pin, MinTRL
# ---------------------------------------------------------------------------

def _good_returns(n=30, seed=21):
    return np.random.default_rng(seed).normal(1.0, 1.0, n)


def test_fail_closed_floor_refuses_to_judge():
    r = _good_returns()
    d = V.dsr_from_trade_returns(r, n_trials=100, n_eff=4.0,
                                 n_eff_source='calendar_uniqueness',
                                 fail_closed_floor=True)
    assert d['dsr'] == 0.0 and d['sr'] == 0.0
    assert d['status'] == 'insufficient_effective_n'
    assert d['n_eff'] == 4.0          # PRE-floor value echoed for audit
    assert d['n_eff_requested'] == 4.0
    assert d['min_trl'] is None
    assert d['n_eff_source'] == 'calendar_uniqueness'


def test_off_mode_legacy_clamp_byte_identical(capsys):
    r = _good_returns()
    a = V.dsr_from_trade_returns(r, n_trials=100, n_eff=4.0)
    b = V.dsr_from_trade_returns(r, n_trials=100, n_eff=4.0,
                                 fail_closed_floor=False)
    assert a == b
    # Loud legacy-clamp warning is instrumentation, printed on both calls
    assert 'silently RAISED' in capsys.readouterr().out
    # Independent replication of the legacy clamp (ne = 10)
    rr = r[np.isfinite(r)]
    sr = float(rr.mean() / rr.std())
    c = rr - rr.mean()
    m2 = float((c ** 2).mean())
    skew = float((c ** 3).mean() / (m2 ** 1.5 + 1e-18))
    kurt = float((c ** 4).mean() / (m2 ** 2 + 1e-18))
    sr0 = V.expected_max_sharpe(100, 1.0 / math.sqrt(10.0))
    assert a['n_eff'] == 10.0
    assert a['sr'] == pytest.approx(sr)
    assert a['expected_max_sr'] == pytest.approx(sr0)
    assert a['dsr'] == pytest.approx(
        V.deflated_sharpe_ratio(sr, sr0, len(rr), skew, kurt, n_eff=10.0))


def test_key_parity_across_all_branches():
    r = _good_returns(300, seed=1)
    keys = set(V.dsr_from_trade_returns(r, 50).keys())
    assert {'status', 'min_trl'} <= keys
    # insufficient_n branch
    assert set(V.dsr_from_trade_returns(r[:5], 50).keys()) == keys
    # degenerate branch (std floor)
    assert set(V.dsr_from_trade_returns(np.full(50, 0.5), 50).keys()) == keys
    # NEW fail-closed branch
    assert set(V.dsr_from_trade_returns(
        r, 50, n_eff=3.0, fail_closed_floor=True).keys()) == keys


def test_success_branch_status_and_min_trl():
    d = V.dsr_from_trade_returns(_good_returns(200, seed=3), n_trials=50)
    assert d['status'] == 'ok'
    assert d['min_trl'] is not None
    # sr >> sr0 here, so the MinTRL is finite
    assert math.isfinite(d['min_trl'])


def test_min_track_record_length():
    assert V.min_track_record_length(0.1, 0.2) == float('inf')
    assert V.min_track_record_length(0.1, 0.1) == float('inf')
    assert V.min_track_record_length(float('nan'), 0.0) == float('inf')
    # Hand-computed reference at sr=0.3, sr0=0.1, skew=0, kurt=3, alpha=.95
    z = V._norm_ppf(0.95)
    expect = 1.0 + (1.0 + 0.5 * 0.3 ** 2) * (z / 0.2) ** 2
    assert V.min_track_record_length(0.3, 0.1) == pytest.approx(expect)
    # Monotone decreasing in sr (bar fixed): higher SR needs fewer trades
    a = V.min_track_record_length(0.2, 0.1)
    b = V.min_track_record_length(0.4, 0.1)
    assert b < a


# ---------------------------------------------------------------------------
# C. adaptive_config: counters, deletion audit, ratchet
# ---------------------------------------------------------------------------

def _base_state(**over):
    s = {
        'asset_type': 'testq1',
        'best_score': 2.0,
        'best_params': {'forward_bars': 24},
        'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
        'mode': 'refine',
        'cycles_without_improvement': 0,
        'expansion_history': [],
        'last_updated': '',
    }
    s.update(over)
    return s


def _mk_study_db(path, k):
    con = sqlite3.connect(str(path))
    con.execute('CREATE TABLE trials (id INTEGER)')
    con.executemany('INSERT INTO trials VALUES (?)', [(i,) for i in range(k)])
    con.commit()
    con.close()


def test_cum_trials_counted_before_db_deletion(tmp_path):
    db = tmp_path / 'v2_study.db'
    _mk_study_db(db, 7)
    state = _base_state(best_params={'forward_bars': 48})
    with mock.patch('adaptive_config.BASE_DIR', tmp_path):
        # forward_bars=48 at the categorical high edge -> expansion ->
        # categoricals_changed -> the DB-deletion path runs.
        result = AC.update_after_search(state, 2.5, {'forward_bars': 48},
                                        study_db_path=str(db),
                                        new_trials_completed=70)
    assert result['cum_trials'] == 70
    assert len(result['trial_history']) == 1
    assert result['trial_history'][0]['n'] == 70
    assert not db.exists()                      # deletion still happened
    dels = result['db_deletions']
    assert len(dels) == 1
    assert dels[0]['trials_lost'] == 7
    assert dels[0]['reason'] == 'categorical_expansion'
    # Increment happened BEFORE the deletion record was written
    assert dels[0]['cum_trials'] == 70
    assert dels[0]['best_score_retained'] == 2.5


def test_legacy_call_shape_unchanged(tmp_path):
    # Calling WITHOUT the new kwargs: pre-existing behavior pinned.
    state = _base_state()
    with mock.patch('adaptive_config.BASE_DIR', tmp_path):
        result = AC.update_after_search(state, 2.5, {'forward_bars': 32})
    assert result['best_score'] == 2.5
    assert result['best_params'] == {'forward_bars': 32}
    assert result['cycles_without_improvement'] == 0
    assert result['cum_trials'] == 0            # nothing accrued
    assert result.get('trial_history', []) == []
    assert result.get('db_deletions', []) == []


def test_store_score_stores_noisy_value_with_real_params(tmp_path):
    state = _base_state()
    with mock.patch('adaptive_config.BASE_DIR', tmp_path):
        result = AC.update_after_search(state, 2.5, {'forward_bars': 32},
                                        store_score=2.61)
    assert result['best_score'] == 2.61          # the noisy stored value
    assert result['best_params'] == {'forward_bars': 32}   # real winner


def test_record_trials_roundtrip(tmp_path):
    with mock.patch('adaptive_config.BASE_DIR', tmp_path):
        s1 = AC.record_trials('testq1', 30)
        assert s1['cum_trials'] == 30
        s2 = AC.record_trials('testq1', 12, event='search_no_update')
        assert s2['cum_trials'] == 42
        assert len(s2['trial_history']) == 2
        assert s2['trial_history'][1]['event'] == 'search_no_update'
        s3 = AC.record_trials('testq1', -5)      # clamped, no record
        assert s3['cum_trials'] == 42
        assert len(s3['trial_history']) == 2
        # persisted on disk
        assert AC.load_adaptive_state('testq1')['cum_trials'] == 42


def test_record_db_deletion_roundtrip(tmp_path, capsys):
    db = tmp_path / 'x_study.db'
    _mk_study_db(db, 3)
    with mock.patch('adaptive_config.BASE_DIR', tmp_path):
        AC.record_db_deletion('testq1', str(db), reason='--fresh')
        state = AC.load_adaptive_state('testq1')
    assert len(state['db_deletions']) == 1
    rec = state['db_deletions'][0]
    assert rec['trials_lost'] == 3 and rec['reason'] == '--fresh'
    assert db.exists()                           # it only LOGS, never deletes
    assert 'study-DB deletion logged' in capsys.readouterr().out
    # never raises, even on garbage input
    AC.record_db_deletion('testq1', None, reason='x')


def test_count_study_trials_fail_soft(tmp_path):
    assert AC._count_study_trials(str(tmp_path / 'missing.db')) is None
    assert AC._count_study_trials(None) is None
    bad = tmp_path / 'bad.db'
    bad.write_text('not a sqlite file')
    assert AC._count_study_trials(str(bad)) is None


def test_overlap_weighted_trials():
    from datetime import datetime, timedelta
    now = datetime(2026, 8, 19, 12, 0, 0)
    hist = [{'date': (now - timedelta(days=7 * k)).isoformat(), 'n': 100}
            for k in range(7)]
    pool = AC.overlap_weighted_trials(hist, now=now)
    assert pool == pytest.approx(364.4, abs=1.0)
    # Older than the 43.8d holdout span -> weight 0
    old = [{'date': (now - timedelta(days=50)).isoformat(), 'n': 1000}]
    assert AC.overlap_weighted_trials(old, now=now) == 0.0
    # Malformed records skipped
    mixed = [{'date': 'garbage', 'n': 100}, {'n': 50}, None,
             {'date': now.isoformat(), 'n': 10}]
    assert AC.overlap_weighted_trials(mixed, now=now) == pytest.approx(10.0)
    assert AC.overlap_weighted_trials([], now=now) == 0.0
    assert AC.overlap_weighted_trials(None, now=now) == 0.0


def test_noisy_ratchet():
    folds = [1.0, 2.0, 3.0]
    sigma_expect = (math.sqrt(2.0 / 3.0)) / math.sqrt(3)
    a = AC.noisy_ratchet(2.5, 2.0, folds, seed=7)
    b = AC.noisy_ratchet(2.5, 2.0, folds, seed=7)
    assert a == b                                 # deterministic per seed
    assert a['sigma'] == pytest.approx(sigma_expect)
    assert a['threshold'] == pytest.approx(2.0 + 2 * a['sigma'] + a['noise'])
    assert a['accept'] == (2.5 > a['threshold'])
    assert a['degraded'] is False
    if a['accept']:
        assert a['store_value'] != 2.5            # noised store
    else:
        assert a['store_value'] == 2.0
    # Degraded (no folds) reduces to the legacy strict comparison
    d = AC.noisy_ratchet(2.0, 2.0, [], seed=1)
    assert d['degraded'] is True and d['sigma'] == 0.0
    assert d['accept'] is False and d['store_value'] == 2.0
    d2 = AC.noisy_ratchet(2.0001, 2.0, None, seed=1)
    assert d2['accept'] is True and d2['store_value'] == 2.0001
    # Non-finite folds degrade too
    d3 = AC.noisy_ratchet(3.0, 2.0, [1.0, float('nan'), 2.0], seed=1)
    assert d3['degraded'] is True and d3['accept'] is True


def test_default_state_carries_new_keys(tmp_path):
    with mock.patch('adaptive_config.BASE_DIR', tmp_path):
        s = AC.load_adaptive_state('brandnew')
    for k in ('cum_trials', 'cum_holdout_gates', 'trial_history',
              'db_deletions'):
        assert k in s
    # Old files on disk are back-filled by the forward-compat loop
    with mock.patch('adaptive_config.BASE_DIR', tmp_path):
        AC.save_adaptive_state(_base_state(asset_type='oldfile'))
        loaded = AC.load_adaptive_state('oldfile')
    assert loaded['cum_trials'] == 0
    assert loaded['trial_history'] == []


# ---------------------------------------------------------------------------
# D. backtest.aggregate_metrics + main() pool resolution
# ---------------------------------------------------------------------------

def _chain_trades(n=100, seed=11, name='N0'):
    """Single-name chain: each trade overlaps only its neighbours —
    clustered collapses to 1, calendar keeps ~2n/3."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp('2026-06-01', tz='UTC')
    trades = []
    for k in range(n):
        entry = base + pd.Timedelta(hours=4 * k)
        trades.append({
            'ticker': name, 'entry_time': str(entry),
            'exit_time': str(entry + pd.Timedelta(hours=5)),
            'entry': 100.0, 'exit': 101.0, 'bars_held': 5,
            'gross_pct': float(rng.normal(0.05, 1.0)),
            'net_pct': float(rng.normal(0.05, 1.0)),
            'reason': 'take_profit',
        })
    return trades


def _stacked_trades(n_names=12, seed=13):
    """n_names trades all sharing ONE calendar window: calendar n_eff ~ 1."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp('2026-06-01', tz='UTC')
    trades = []
    for j in range(n_names):
        trades.append({
            'ticker': f'N{j}', 'entry_time': str(base),
            'exit_time': str(base + pd.Timedelta(hours=24)),
            'entry': 100.0, 'exit': 101.0, 'bars_held': 24,
            'gross_pct': float(rng.normal(0.05, 1.0)),
            'net_pct': float(rng.normal(0.05, 1.0)),
            'reason': 'take_profit',
        })
    return trades


def test_off_mode_aggregate_metrics_byte_pin():
    trades = _chain_trades()
    m = backtest.aggregate_metrics(trades, 'crypto', 90.0, n_search_trials=10)
    assert m['gate_v2_active'] is False
    # Independent replication of the legacy path
    rets = np.array([t['net_pct'] for t in trades])
    ets = pd.to_datetime([t['entry_time'] for t in trades]).values
    xts = pd.to_datetime([t['exit_time'] for t in trades]).values
    n_clusters = clustered_effective_n(ets, xts)
    assert m['n_eff_clustered'] == n_clusters == 1
    ref = V.dsr_from_trade_returns(rets, n_trials=10,
                                   n_eff=float(n_clusters),
                                   n_eff_source='clustered')
    ref_iid = V.dsr_from_trade_returns(rets, n_trials=10, n_eff=None)
    assert m['dsr'] == pytest.approx(round(ref['dsr'], 4))
    assert m['dsr_raw'] == pytest.approx(ref['dsr'])
    assert m['dsr_iid'] == pytest.approx(round(ref_iid['dsr'], 4))
    assert m['dsr_n_eff_used'] == float(ref['n_eff'])
    assert m['n_eff_source'] == 'clustered'
    # New instrumentation keys present with flag OFF
    cal = calendar_effective_n(ets, xts)
    assert m['n_eff_calendar'] == pytest.approx(cal['n_eff'])
    assert m['dsr_status'] == ref['status']
    assert m['dsr_min_trl'] == ref['min_trl']


def test_gate_v2_consumes_calendar_n_eff(monkeypatch):
    monkeypatch.setattr(backtest._strategy_config, 'PROMOTION_GATE_V2', True)
    trades = _chain_trades()
    m = backtest.aggregate_metrics(trades, 'crypto', 90.0, n_search_trials=10)
    assert m['gate_v2_active'] is True
    assert m['n_eff_source'] == 'calendar_uniqueness'
    assert m['n_eff_clustered'] == 1              # reporting-only now
    assert m['n_eff_calendar'] > 40               # ~2n/3 of 100
    # The DSR consumed the calendar value, not the clamped cluster count
    assert m['dsr_n_eff_used'] == pytest.approx(m['n_eff_calendar'], abs=0.01)
    assert m['dsr_status'] == 'ok'


def test_gate_v2_fails_closed_on_crowded_book(monkeypatch):
    monkeypatch.setattr(backtest._strategy_config, 'PROMOTION_GATE_V2', True)
    trades = _stacked_trades(12)                  # calendar n_eff ~ 1 < 10
    m = backtest.aggregate_metrics(trades, 'crypto', 90.0, n_search_trials=10)
    assert m['n_eff_calendar'] < 10
    assert m['dsr'] == 0.0
    assert m['dsr_status'] == 'insufficient_effective_n'
    assert m['dsr_min_trl'] is None


def test_key_parity_empty_vs_populated_both_modes(monkeypatch):
    empty = backtest.aggregate_metrics([], 'crypto', 44.0)
    pop = backtest.aggregate_metrics(_chain_trades(20), 'crypto', 44.0)
    assert set(empty.keys()) == set(pop.keys())
    assert {'n_eff_calendar', 'gate_v2_active', 'dsr_status',
            'dsr_min_trl'} <= set(empty.keys())
    monkeypatch.setattr(backtest._strategy_config, 'PROMOTION_GATE_V2', True)
    empty_on = backtest.aggregate_metrics([], 'crypto', 44.0)
    pop_on = backtest.aggregate_metrics(_chain_trades(20), 'crypto', 44.0)
    assert set(empty_on.keys()) == set(pop_on.keys()) == set(empty.keys())
    assert empty_on['gate_v2_active'] is True


def test_kish_floor_raises_n_eff_but_never_above_n(monkeypatch):
    monkeypatch.setattr(backtest._strategy_config, 'PROMOTION_GATE_V2', True)
    trades = _chain_trades(60)
    plain = backtest.aggregate_metrics(trades, 'crypto', 90.0,
                                       n_search_trials=10)
    monkeypatch.setattr(backtest._strategy_config, 'KISH_NEFF_ENABLED', True)
    kish = backtest.aggregate_metrics(trades, 'crypto', 90.0,
                                      n_search_trials=10)
    assert kish['n_eff_calendar'] > plain['n_eff_calendar']
    assert kish['n_eff_calendar'] <= kish['n_trades']


def test_main_trials_resolution(monkeypatch, capsys):
    seen = {}

    def fake_run(prefix, days, trials):
        seen['trials'] = trials
        return {'n_trades': 20, 'sharpe': 1.0, 'dsr': 0.9}

    monkeypatch.setattr(backtest, 'run_backtest', fake_run)
    # Flag OFF, no --trials -> legacy 100, byte-identical
    monkeypatch.setattr(sys, 'argv', ['backtest.py'])
    assert backtest.main() == 0
    assert seen['trials'] == 100
    assert 'deflation pool' in capsys.readouterr().out
    # Explicit --trials always wins
    monkeypatch.setattr(sys, 'argv', ['backtest.py', '--trials', '55'])
    assert backtest.main() == 0
    assert seen['trials'] == 55

    class _FakeAC:
        @staticmethod
        def load_adaptive_state(asset):
            return {'cum_trials': 730}

    monkeypatch.setitem(sys.modules, 'adaptive_config', _FakeAC)
    # Flag OFF with a persisted cum_trials: STILL legacy 100 (the OFF-mode
    # byte-identity pin — the counter is logged but never consumed).
    monkeypatch.setattr(sys, 'argv', ['backtest.py'])
    assert backtest.main() == 0
    assert seen['trials'] == 100
    assert 'cum_trials=730' in capsys.readouterr().out
    # Flag ON with a persisted cum_trials -> the cumulative pool
    monkeypatch.setattr(backtest._strategy_config, 'PROMOTION_GATE_V2', True)
    assert backtest.main() == 0
    assert seen['trials'] == 730

    class _EmptyAC:
        @staticmethod
        def load_adaptive_state(asset):
            return {'cum_trials': 0}

    # Flag ON but no accumulated pool yet -> legacy 100 fallback
    monkeypatch.setitem(sys.modules, 'adaptive_config', _EmptyAC)
    assert backtest.main() == 0
    assert seen['trials'] == 100
