"""Panel-review batch A (2026-07): risk_budget.py hardening locks — GATE-1
report honesty (rho bracket, missing/unknown/skewed books, ages), write-path
hygiene (anchored path, bounded flock, fsync, thread-unique tmp, non-finite
refusal), simulator fixes (iterator inputs, grid bound, active cross-corr),
and the bit-identical account_risk_budget hoist. Mac-green.
"""
import fcntl
import json
import logging
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import risk_budget as rb
from portfolio import diversified_book_risk


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

def test_registry_default_path_module_anchored():
    assert os.path.isabs(rb.ACCOUNT_RISK_REGISTRY)
    assert os.path.basename(rb.ACCOUNT_RISK_REGISTRY) == 'account_risk_registry.json'
    assert (os.path.dirname(rb.ACCOUNT_RISK_REGISTRY)
            == os.path.dirname(os.path.abspath(rb.__file__)))


def test_constants():
    assert rb.REGISTRY_STALE_AFTER_S == 900.0
    assert rb.CLOCK_SKEW_TOL_S == 60.0
    assert rb.EXPECTED_BOOKS == ('stock', 'crypto')
    assert rb.MAX_SIM_PERIODS == 100_000


# ---------------------------------------------------------------------------
# account_risk_gate1_report — honesty additions
# ---------------------------------------------------------------------------

def test_gate1_degeneracy_pinned_and_indep_bracket():
    rep = rb.account_risk_gate1_report(
        {'stock': {'risk': 0.021, 'rho': 0.6, 'ts': 0.0},
         'crypto': {'risk': 0.017, 'rho': 0.8, 'ts': 0.0}},
        rho_cross=1.0, now=0.0)
    assert rep['account_stop_risk'] == rep['book_sum'] == pytest.approx(0.038)
    assert rep['concentration'] == 1.0
    assert rep['account_stop_risk_indep'] == pytest.approx(
        round(float(np.sqrt(0.021 ** 2 + 0.017 ** 2)), 5))
    assert rep['account_stop_risk_indep'] < rep['book_sum']
    assert rep['stock_rho'] == 0.6 and rep['crypto_rho'] == 0.8


def test_missing_and_unknown_books():
    single_crypto = rb.account_risk_gate1_report(
        {'crypto': {'risk': 0.02, 'rho': 0.7, 'ts': 0.0}}, now=0.0)
    assert single_crypto['missing_books'] == ['stock']
    assert single_crypto['stale_books'] == []
    assert single_crypto['stock_risk'] == 0.0

    unknown_only = rb.account_risk_gate1_report(
        {'options': {'risk': 0.09, 'rho': 0.5, 'ts': 0.0}}, now=0.0)
    assert unknown_only['unknown_books'] == ['options']
    assert unknown_only['account_stop_risk'] == 0.0
    assert unknown_only['stale_books'] == []
    assert unknown_only['missing_books'] == ['crypto', 'stock']

    both_fresh = rb.account_risk_gate1_report(
        {'stock': {'risk': 0.01, 'rho': 0.5, 'ts': 0.0},
         'crypto': {'risk': 0.01, 'rho': 0.5, 'ts': 0.0}}, now=0.0)
    assert both_fresh['missing_books'] == []
    assert both_fresh['unknown_books'] == []

    one_stale = rb.account_risk_gate1_report(
        {'stock': {'risk': 0.01, 'rho': 0.5, 'ts': -1000.0}}, now=0.0)
    assert 'stock' in one_stale['stale_books']
    assert 'stock' not in one_stale['missing_books']


def test_future_ts_two_sided():
    beyond_tol = rb.account_risk_gate1_report(
        {'crypto': {'risk': 0.02, 'rho': 0.5, 'ts': 10000.0}}, now=1000.0)
    assert beyond_tol['crypto_risk'] == 0.0
    assert 'crypto' in beyond_tol['stale_books']
    assert beyond_tol['skewed_books'] == ['crypto']

    within_tol = rb.account_risk_gate1_report(
        {'crypto': {'risk': 0.02, 'rho': 0.5, 'ts': 1030.0}}, now=1000.0)
    assert within_tol['crypto_risk'] == pytest.approx(0.02)
    assert within_tol['skewed_books'] == []

    fresh_edge = rb.account_risk_gate1_report(
        {'crypto': {'risk': 0.02, 'rho': 0.5, 'ts': 0.0}}, now=899.0)
    assert fresh_edge['crypto_risk'] == pytest.approx(0.02)

    stale_edge = rb.account_risk_gate1_report(
        {'crypto': {'risk': 0.02, 'rho': 0.5, 'ts': 0.0}}, now=901.0)
    assert stale_edge['crypto_risk'] == 0.0


def test_book_ages():
    reg = {'stock': {'risk': 0.01, 'rho': 0.5, 'ts': 100.0},
           'crypto': {'risk': 0.02, 'rho': 0.6, 'ts': 900.0}}
    rep = rb.account_risk_gate1_report(reg, now=1000.0, stale_after_s=800.0)
    assert rep['book_ages_s'] == {'stock': 900.0, 'crypto': 100.0}
    assert rep['stock_rho'] is None          # aged out -> never entered `fresh`
    assert rep['crypto_rho'] == 0.6           # fresh -> rho round-trips


def test_non_dict_registry_and_negative_risk():
    rep = rb.account_risk_gate1_report(['stock'])
    assert rep['account_stop_risk'] == 0.0
    assert rep['missing_books'] == ['crypto', 'stock']

    rep2 = rb.account_risk_gate1_report(
        {'stock': {'risk': -0.5, 'ts': 0.0},
         'crypto': {'risk': 0.02, 'rho': 0.4, 'ts': 0.0}}, now=0.0)
    assert rep2['stale_books'] == ['stock']
    assert rep2['book_sum'] == pytest.approx(0.02)
    assert rep2['account_stop_risk'] == pytest.approx(0.02)


def test_nonfinite_ts_dropped_and_bad_entry_ages_recorded():
    # NaN ts parses (json.load accepts the bare NaN token) but must NOT leak
    # a NaN age into the JSON journal; negative-risk entries are dropped but
    # their age IS recorded (diagnostic signal for a bad entry).
    rep = rb.account_risk_gate1_report(
        {'stock': {'risk': 0.01, 'rho': 0.5, 'ts': float('nan')},
         'crypto': {'risk': -0.5, 'rho': 0.5, 'ts': 100.0}}, now=1000.0)
    assert 'stock' not in rep['book_ages_s']       # unparseable ts -> no age
    assert 'stock' in rep['stale_books']
    assert rep['book_ages_s']['crypto'] == 900.0   # dropped, but age recorded
    assert 'crypto' in rep['stale_books']
    assert rep['account_stop_risk'] == 0.0
    json.loads(json.dumps(rep), parse_constant=lambda c: pytest.fail(c))


def test_nan_rho_cross_finite_and_json_strict():
    rep = rb.account_risk_gate1_report(
        {'stock': {'risk': 0.021, 'ts': 100.0},
         'crypto': {'risk': 0.019, 'ts': 100.0}},
        rho_cross=float('nan'), now=100.0)
    assert rep['rho_cross'] == 1.0
    assert math.isfinite(rep['account_stop_risk'])
    assert rep['account_stop_risk'] == pytest.approx(0.04)
    assert rep['over_cap'] is True
    assert json.loads(json.dumps(rep),
                      parse_constant=lambda c: pytest.fail(c)) is not None


# ---------------------------------------------------------------------------
# record_book_risk_and_report — provenance + None-skip
# ---------------------------------------------------------------------------

def test_record_provenance_and_write_ok(tmp_path, monkeypatch):
    p = str(tmp_path / 'reg.json')
    rep = rb.record_book_risk_and_report('crypto', [0.02, 0.02], 0.8,
                                         path=p, now=10.0)
    assert rep['self_risk'] > 0
    assert rep['self_rho'] == 0.8
    assert rep['n_positions'] == 2
    assert rep['registry_write_ok'] is True

    monkeypatch.setattr(rb, '_write_warned', False)
    bad = str(tmp_path / 'no_such_dir' / 'reg.json')
    rep2 = rb.record_book_risk_and_report('stock', [0.02, 0.02], 0.8,
                                          path=bad, now=10.0)
    assert rep2['registry_write_ok'] is False
    assert rep2['self_risk'] > 0
    assert rep2['stock_risk'] > 0


def test_record_none_skips_write(tmp_path):
    p = str(tmp_path / 'reg.json')
    rb.write_book_risk('stock', 0.02, 0.5, path=p, now=0.0)
    before = Path(p).read_text()
    rep = rb.record_book_risk_and_report('stock', None, 0.5, path=p, now=2000.0)
    assert Path(p).read_text() == before
    assert 'stock' in rep['stale_books']
    assert rep['self_risk'] is None
    assert rep['n_positions'] is None
    assert rep['registry_write_ok'] is None


def test_record_accepts_generator(tmp_path):
    p_gen = str(tmp_path / 'gen.json')
    p_list = str(tmp_path / 'list.json')
    rep_gen = rb.record_book_risk_and_report(
        'stock', (r for r in [0.02, 0.02]), 0.5, path=p_gen, now=5.0)
    rep_list = rb.record_book_risk_and_report(
        'stock', [0.02, 0.02], 0.5, path=p_list, now=5.0)
    assert rep_gen['stock_risk'] == pytest.approx(rep_list['stock_risk'])


def test_stale_after_s_passthrough(tmp_path):
    p = str(tmp_path / 'reg.json')
    rb.write_book_risk('crypto', 0.02, 0.5, path=p, now=0.0)

    rep_strict = rb.record_book_risk_and_report(
        'stock', [0.01], 0.5, path=p, stale_after_s=1.0, now=5.0)
    assert 'crypto' in rep_strict['stale_books']

    rep_default = rb.record_book_risk_and_report(
        'stock', [0.01], 0.5, path=p, now=5.0)
    assert 'crypto' not in rep_default['stale_books']
    assert rep_default['crypto_risk'] == pytest.approx(0.02)


def test_full_report_json_strict(tmp_path):
    p = str(tmp_path / 'reg.json')
    rep = rb.record_book_risk_and_report('crypto', [0.02, 0.015], 0.6,
                                         path=p, now=100.0)
    parsed = json.loads(json.dumps(rep),
                        parse_constant=lambda c: pytest.fail(c))
    assert parsed == rep


# ---------------------------------------------------------------------------
# write path hygiene
# ---------------------------------------------------------------------------

def test_write_nonfinite_not_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, '_write_warned', False)
    p = str(tmp_path / 'reg.json')
    reg = rb.write_book_risk('stock', float('nan'), 0.5, path=p, now=1.0)
    assert not os.path.exists(p)
    assert reg == {}

    rb.write_book_risk('stock', 0.01, float('inf'), path=p, now=2.0)
    on_disk = json.loads(Path(p).read_text(),
                         parse_constant=lambda c: pytest.fail(c))
    assert on_disk['stock']['rho'] == 0.0


def test_lock_contention_fails_open_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, '_write_warned', False)
    p = str(tmp_path / 'reg.json')

    def raiser(*a, **kw):
        raise BlockingIOError()

    monkeypatch.setattr(rb.fcntl, 'flock', raiser)
    monkeypatch.setattr(rb.time, 'sleep', lambda s: None)
    reg = rb.write_book_risk('stock', 0.01, 0.5, path=p, now=1.0)
    assert reg['stock']['risk'] == 0.01
    assert not os.path.exists(p)
    assert rb._write_warned is True


def test_lock_contention_real_retry_succeeds(tmp_path):
    p = str(tmp_path / 'reg.json')
    lock_path = p + '.lock'
    held = threading.Event()

    def holder():
        with open(lock_path, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            held.set()
            time.sleep(0.3)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    t = threading.Thread(target=holder)
    t.start()
    assert held.wait(timeout=5.0)
    reg = rb.write_book_risk('stock', 0.03, 0.4, path=p, now=1.0)
    t.join()
    assert reg['stock']['risk'] == 0.03
    assert os.path.exists(p)
    on_disk = json.loads(Path(p).read_text())
    assert on_disk['stock']['risk'] == 0.03


def test_write_warn_resets_after_recovery(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(rb, '_write_warned', False)
    bad = str(tmp_path / 'no_such_dir' / 'reg.json')
    good = str(tmp_path / 'reg.json')
    with caplog.at_level(logging.WARNING, logger='risk_budget'):
        rb.write_book_risk('stock', 0.01, 0.5, path=bad, now=1.0)
        assert rb._write_warned is True
        rb.write_book_risk('stock', 0.01, 0.5, path=good, now=2.0)
        assert rb._write_warned is False
        rb.write_book_risk('stock', 0.01, 0.5, path=bad, now=3.0)
        assert rb._write_warned is True
    warns = [r for r in caplog.records
             if 'registry write failed' in r.getMessage()]
    assert len(warns) == 2


def test_tmp_name_thread_unique_source():
    assert 'threading.get_ident()' in Path(rb.__file__).read_text()


# ---------------------------------------------------------------------------
# simulate_two_books
# ---------------------------------------------------------------------------

def test_simulator_iterator_inputs_match_lists():
    s = [{'exit_period': 0, 'net_pct': 5.0}, {'exit_period': 1, 'net_pct': -4.0}]
    c = [{'exit_period': 1, 'net_pct': -3.0}]
    out_iter = rb.simulate_two_books(iter(s), iter(c))
    out_list = rb.simulate_two_books(s, c)
    assert out_iter == out_list
    assert out_list['combined_total_pct'] == pytest.approx(-2.0)


def test_simulator_weight_none_defaults_to_one():
    with_none = rb.simulate_two_books(
        [{'exit_period': 0, 'net_pct': 2.0, 'weight': None}], [], periods=1)
    without = rb.simulate_two_books(
        [{'exit_period': 0, 'net_pct': 2.0}], [], periods=1)
    assert with_none['combined_total_pct'] == pytest.approx(2.0)
    assert without['combined_total_pct'] == pytest.approx(2.0)

    zero_weight = rb.simulate_two_books(
        [{'exit_period': 0, 'net_pct': 2.0, 'weight': 0.0}], [], periods=1)
    assert zero_weight['combined_total_pct'] == pytest.approx(0.0)


def test_simulator_grid_bound():
    with pytest.raises(ValueError):
        rb.simulate_two_books(
            [{'exit_period': 1_700_000_000, 'net_pct': 1.0}], [])
    with pytest.raises(ValueError):
        rb.simulate_two_books(
            [{'exit_period': 0, 'net_pct': 1.0}], [], periods=10 ** 9)
    out = rb.simulate_two_books([], [], periods=10 ** 9)
    assert out == {'n_periods': 0, 'combined_max_drawdown_pct': 0.0,
                   'combined_sharpe': 0.0, 'combined_total_pct': 0.0,
                   'n_dropped_trades': 0}


def test_simulator_new_keys_and_docstring():
    s = [{'exit_period': 0, 'net_pct': 2.0}, {'exit_period': 1, 'net_pct': -1.0}]
    c = [{'exit_period': 0, 'net_pct': 1.0}]
    out = rb.simulate_two_books(s, c, periods=2)
    assert out['stock_total_pct'] == pytest.approx(1.0)
    assert out['crypto_total_pct'] == pytest.approx(1.0)
    assert out['combined_total_pct'] == pytest.approx(
        out['stock_total_pct'] + out['crypto_total_pct'])
    doc = rb.simulate_two_books.__doc__
    assert 'compound into' not in doc
    assert 'additive' in doc
    assert 't-statistic' in doc


def test_simulator_dd_concentration_none_when_no_drawdown():
    s = [{'exit_period': 0, 'net_pct': 1.0}, {'exit_period': 1, 'net_pct': 1.0}]
    c = [{'exit_period': 0, 'net_pct': 1.0}, {'exit_period': 1, 'net_pct': 1.0}]
    out = rb.simulate_two_books(s, c, periods=2)
    assert out['drawdown_concentration'] is None

    # Reuse the "together" fixture shape from test_risk_budget.py: real,
    # matched drawdowns in both books -> a concrete concentration float.
    together_s = [{'exit_period': i, 'net_pct': p}
                  for i, p in enumerate([1, -3, -3, 2])]
    together_c = [{'exit_period': i, 'net_pct': p}
                  for i, p in enumerate([1, -3, -3, 2])]
    out2 = rb.simulate_two_books(together_s, together_c, periods=4)
    assert isinstance(out2['drawdown_concentration'], float)


def test_simulator_cross_corr_active():
    s = [{'exit_period': i, 'net_pct': 1.0} for i in range(100)]
    c = [{'exit_period': i, 'net_pct': 1.0} for i in range(100, 200)]
    out = rb.simulate_two_books(s, c, periods=200)
    assert out['realized_cross_corr'] == pytest.approx(-1.0)
    assert out['realized_cross_corr_active'] is None
    assert out['n_overlap_periods'] == 0

    s2 = [{'exit_period': i, 'net_pct': (1.0 if i % 2 == 0 else -1.0)}
          for i in range(20)]
    c2 = [{'exit_period': i, 'net_pct': (1.0 if i % 2 == 0 else -1.0)}
          for i in range(20)]
    out2 = rb.simulate_two_books(s2, c2, periods=20)
    assert out2['realized_cross_corr_active'] == pytest.approx(1.0)
    assert out2['n_overlap_periods'] == 20


# ---------------------------------------------------------------------------
# account_risk_budget — validation + bit-identical hoist
# ---------------------------------------------------------------------------

def test_account_risk_budget_validation():
    for bad_book in ('Stock', 'stocks', 'STOCK'):
        with pytest.raises(ValueError):
            rb.account_risk_budget(bad_book, [], [], 0.3, 0.3, 0.5)
    with pytest.raises(ValueError):
        rb.scale_for_account_cap(0.01, 'Stock', [], [], 0.3, 0.3, 0.5)

    assert rb.account_risk_budget('stock', [], [], 0.3, 0.3, 0.5,
                                  max_risk=float('nan')) == 0.0

    val = rb.account_risk_budget('stock', [0.004], [0.006], 0.9, 0.0, 1.0)
    assert val == pytest.approx(0.020336582880581773, abs=1e-12)


def test_account_risk_budget_hoist_bit_identical():
    def ref_account_stop_risk(stock_risks, crypto_risks, rho_stock, rho_crypto,
                              rho_cross):
        r_s = diversified_book_risk(stock_risks, rho_stock)
        r_c = diversified_book_risk(crypto_risks, rho_crypto)
        rx = min(max(float(rho_cross), -1.0), 1.0)
        return float(np.sqrt(max(
            r_s ** 2 + r_c ** 2 + 2.0 * rx * r_s * r_c, 0.0)))

    def ref_account_risk_budget(candidate_book, stock_risks, crypto_risks,
                                rho_stock, rho_crypto, rho_cross,
                                cap=rb.ACCOUNT_RISK_CAP, max_risk=None):
        hi = float(max_risk) if max_risk is not None else float(cap)
        if hi <= 0:
            return 0.0
        base = (list(stock_risks) if candidate_book == 'stock'
                else list(crypto_risks))
        other = (crypto_risks if candidate_book == 'stock' else stock_risks)
        rho_b = rho_stock if candidate_book == 'stock' else rho_crypto
        rho_o = rho_crypto if candidate_book == 'stock' else rho_stock

        def acct_with(rc):
            b = base + [rc]
            if candidate_book == 'stock':
                return ref_account_stop_risk(b, other, rho_b, rho_o, rho_cross)
            return ref_account_stop_risk(other, b, rho_o, rho_b, rho_cross)

        if acct_with(0.0) >= cap:
            return 0.0
        if acct_with(hi) <= cap:
            return hi
        lo, hi2 = 0.0, hi
        for _ in range(40):
            mid = 0.5 * (lo + hi2)
            if acct_with(mid) <= cap:
                lo = mid
            else:
                hi2 = mid
        return lo

    rng = np.random.default_rng(7)
    for _ in range(300):
        n_s = int(rng.integers(0, 4))
        n_c = int(rng.integers(0, 4))
        stock_risks = list(rng.uniform(0.001, 0.03, size=n_s))
        crypto_risks = list(rng.uniform(0.001, 0.03, size=n_c))
        rho_stock = float(rng.uniform(0.0, 1.0))
        rho_crypto = float(rng.uniform(0.0, 1.0))
        rho_cross = float(rng.uniform(-1.0, 1.0))
        max_risk = (None if rng.random() < 0.5
                    else float(rng.uniform(0.001, 0.06)))
        for book in ('stock', 'crypto'):
            got = rb.account_risk_budget(book, stock_risks, crypto_risks,
                                         rho_stock, rho_crypto, rho_cross,
                                         max_risk=max_risk)
            want = ref_account_risk_budget(book, stock_risks, crypto_risks,
                                           rho_stock, rho_crypto, rho_cross,
                                           max_risk=max_risk)
            assert got == want
            if book == 'stock':
                acct = rb.account_stop_risk(stock_risks + [got], crypto_risks,
                                            rho_stock, rho_crypto, rho_cross)
            else:
                acct = rb.account_stop_risk(stock_risks, crypto_risks + [got],
                                            rho_stock, rho_crypto, rho_cross)
            # "Only ever blocks, never enlarges": the post-add account risk
            # stays within cap PROVIDED the book wasn't already over cap
            # before any candidate was added (account_risk_budget's own
            # documented early-exit returns 0.0 in that pre-existing-breach
            # case — it cannot retroactively fix an already-blown cap).
            pre = rb.account_stop_risk(stock_risks, crypto_risks,
                                       rho_stock, rho_crypto, rho_cross)
            if pre < rb.ACCOUNT_RISK_CAP:
                assert acct <= rb.ACCOUNT_RISK_CAP + 1e-9
            else:
                assert got == 0.0


def test_allocate_book_caps_behavior_pinned_and_doc():
    cs, cc = rb.allocate_book_caps(0.01, 1.0, total_cap=0.03, clamp=(0.4, 0.9))
    assert cs == pytest.approx(0.027)
    assert cc == pytest.approx(0.003)

    cs1, cc1 = rb.allocate_book_caps(0.10, 0.40)
    assert cs1 > cc1
    assert cs1 + cc1 == pytest.approx(rb.ACCOUNT_RISK_CAP, abs=1e-9)

    cs2, cc2 = rb.allocate_book_caps(0.001, 10.0)
    assert cs2 == pytest.approx(rb.ACCOUNT_RISK_CAP * 0.75, abs=1e-9)

    assert 'clamp_lo + clamp_hi == 1' in rb.allocate_book_caps.__doc__
