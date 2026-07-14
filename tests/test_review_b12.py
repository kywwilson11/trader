"""Review batch b12: portfolio.py / risk_budget.py / drawdown.py fixes.

Covers: symmetric short-data corr keys and the dead-fallback removal in the
correlation gate; GATE-1 registry robustness (binary/non-dict JSON, string
values, flock-serialized concurrent writes, warn-once fail-open on write
errors); simulate_two_books empty-grid guard + dropped-trade accounting; the
signed-rho_cross semantics account_risk_budget's docstring now states; and
the +inf peak-equity guard that keeps the de-leveraging ladder armed. All
three modules are Mac-importable (numpy-only).
"""
import json
import logging
import threading
from pathlib import Path

import numpy as np
import pytest

import risk_budget as rb
from drawdown import (
    PEAK_SEED,
    drawdown_fraction,
    drawdown_size_multiplier,
    restore_peak_equity,
)
from portfolio import (
    check_portfolio_correlation,
    compute_correlation_matrix,
    get_correlation_sizing_factor,
)

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# portfolio.py
# ---------------------------------------------------------------------------

def test_short_data_pair_writes_both_orderings():
    rng = np.random.default_rng(0)
    returns = {'AAA': rng.normal(size=5), 'BBB': rng.normal(size=40)}
    corr = compute_correlation_matrix(returns)
    assert corr[('AAA', 'BBB')] == 0.0
    assert corr[('BBB', 'AAA')] == 0.0          # mirror key no longer missing
    assert set(corr) == {('AAA', 'BBB'), ('BBB', 'AAA')}


def test_correlation_gate_unchanged_after_dead_fallback_removal():
    m = {('A', 'C'): 0.8, ('C', 'B'): -0.4}
    allowed, avg = check_portfolio_correlation(['A', 'B'], 'C', m)
    assert allowed and avg == pytest.approx(0.6)     # mean of |0.8|, |-0.4|
    assert isinstance(avg, float)                    # plain float now
    rejected, avg2 = check_portfolio_correlation(['A'], 'C', m)
    assert not rejected and avg2 == pytest.approx(0.8)
    # empty book still short-circuits before the removed fallback
    assert check_portfolio_correlation([], 'C', m) == (True, 0.0)


def test_sizing_factor_unchanged_after_dead_fallback_removal():
    m = {('A', 'C'): 0.8, ('C', 'B'): -0.4}
    f = get_correlation_sizing_factor('C', ['A', 'B'], m)
    assert f == pytest.approx(1.0 / np.sqrt(1.0 + 2 * 0.6))
    assert get_correlation_sizing_factor('C', [], m) == 1.0


def test_portfolio_doc_guards():
    import portfolio
    # module docstring covers all three components, not just the corr gate
    assert 'diversified_book_risk' in portfolio.__doc__
    assert 'vol scalar' in portfolio.__doc__.lower()
    # cache contract: symbols is not part of the key -> full universe required
    assert 'full universe' in portfolio.get_correlation_matrix_cached.__doc__
    src = (REPO / 'portfolio.py').read_text()
    assert 'Account-level realized-vol scalar' in src   # not "Book-level"
    assert 'sqrt(252/365)' in src                        # annualization caveat


# ---------------------------------------------------------------------------
# risk_budget.py — registry robustness
# ---------------------------------------------------------------------------

def test_read_registry_binary_garbage_returns_empty(tmp_path):
    p = tmp_path / 'reg.json'
    p.write_bytes(b'\x80\x81\xfe\xff')          # UnicodeDecodeError, not JSON
    assert rb.read_registry(str(p)) == {}


def test_read_registry_non_dict_json_returns_empty(tmp_path):
    p = tmp_path / 'reg.json'
    for payload in ('[]', 'null', '3.14', '"risk"'):
        p.write_text(payload)
        assert rb.read_registry(str(p)) == {}, payload


def test_write_self_heals_non_dict_registry(tmp_path):
    p = tmp_path / 'reg.json'
    p.write_text('[]')                          # parseable, previously fatal
    reg = rb.write_book_risk('stock', 0.01, 0.5, path=str(p), now=1.0)
    assert reg['stock'] == {'risk': 0.01, 'rho': 0.5, 'ts': 1.0}
    assert json.loads(p.read_text())['stock']['rho'] == 0.5


def test_registry_roundtrip_format_and_merge_unchanged(tmp_path):
    p = str(tmp_path / 'reg.json')
    rb.write_book_risk('crypto', 0.024, 0.7, path=p, now=1000.0)
    reg = rb.write_book_risk('stock', 0.01, 0.5, path=p, now=1001.0)
    assert set(reg) == {'crypto', 'stock'}      # other book's entry preserved
    on_disk = json.loads(Path(p).read_text())
    assert on_disk['crypto'] == {'risk': 0.024, 'rho': 0.7, 'ts': 1000.0}
    assert set(on_disk['stock']) == {'risk', 'rho', 'ts'}
    assert not list(Path(p).parent.glob('*.tmp'))        # tmp renamed away
    assert Path(p + '.lock').exists()           # flock sidecar in place


def test_concurrent_writers_never_lose_the_other_book(tmp_path):
    # Two threads (the --combined-bots shape) hammer the registry; the flock
    # totally orders the read-modify-writes, so the final file must hold BOTH
    # books at each book's own last-written ts — the old shared-tmp RMW could
    # drop one.
    p = str(tmp_path / 'reg.json')

    def writer(book, base):
        for i in range(50):
            rb.write_book_risk(book, 0.02, 0.5, path=p, now=float(base + i))

    t1 = threading.Thread(target=writer, args=('crypto', 0))
    t2 = threading.Thread(target=writer, args=('stock', 1000))
    t1.start(); t2.start(); t1.join(); t2.join()
    on_disk = rb.read_registry(p)
    assert set(on_disk) == {'crypto', 'stock'}
    assert on_disk['crypto']['ts'] == 49.0
    assert on_disk['stock']['ts'] == 1049.0


def test_write_failure_fail_open_and_warns_once(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(rb, '_write_warned', False)
    bad = str(tmp_path / 'no_such_dir' / 'reg.json')
    with caplog.at_level(logging.WARNING, logger='risk_budget'):
        reg1 = rb.write_book_risk('crypto', 0.02, 0.5, path=bad, now=1.0)
        reg2 = rb.write_book_risk('crypto', 0.02, 0.5, path=bad, now=2.0)
    assert reg1['crypto']['risk'] == 0.02       # returned registry still usable
    assert reg2['crypto']['ts'] == 2.0
    warns = [r for r in caplog.records
             if 'registry write failed' in r.getMessage()]
    assert len(warns) == 1                      # once per process, not per cycle
    assert rb._write_warned is True


def test_gate1_report_tolerates_non_numeric_entries():
    reg = {'stock': {'risk': 'abc', 'ts': 0.0},          # was: TypeError
           'crypto': {'risk': 0.02, 'ts': 'garbage'}}    # was: ValueError
    rep = rb.account_risk_gate1_report(reg, rho_cross=1.0, now=0.0)
    assert rep['account_stop_risk'] == 0.0
    assert rep['stale_books'] == ['crypto', 'stock']
    # numeric strings coerce instead of crashing
    rep2 = rb.account_risk_gate1_report(
        {'stock': {'risk': '0.02', 'ts': 5.0}}, now=5.0)
    assert rep2['stock_risk'] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# risk_budget.py — simulator + budget semantics
# ---------------------------------------------------------------------------

def test_simulator_zero_or_negative_periods_returns_empty_result():
    trades = [{'exit_period': 0, 'net_pct': 1.0}]
    for periods in (0, -3):                     # was: IndexError at comb_eq[-1]
        out = rb.simulate_two_books(trades, [], periods=periods)
        assert out['n_periods'] == 0
        assert out['combined_total_pct'] == 0.0
        assert out['n_dropped_trades'] == 1


def test_simulator_counts_trades_dropped_by_short_grid():
    s = [{'exit_period': 0, 'net_pct': 1.0}, {'exit_period': 5, 'net_pct': 9.0}]
    c = [{'exit_period': 7, 'net_pct': -2.0}]
    out = rb.simulate_two_books(s, c, periods=2)
    assert out['n_dropped_trades'] == 2         # silent truncation now counted
    assert out['combined_total_pct'] == pytest.approx(1.0)
    full = rb.simulate_two_books(s, c)          # default grid fits everything
    assert full['n_dropped_trades'] == 0
    assert full['n_periods'] == 8
    assert full['combined_total_pct'] == pytest.approx(8.0)


def test_simulator_no_trades_schema_and_sharpe_doc():
    assert rb.simulate_two_books([], []) == {
        'n_periods': 0, 'combined_max_drawdown_pct': 0.0,
        'combined_sharpe': 0.0, 'combined_total_pct': 0.0,
        'n_dropped_trades': 0}
    assert rb.simulate_two_books([], [], periods=5)['n_periods'] == 0
    # combined_sharpe is documented as the sqrt(N)-scaled t-stat it computes
    assert 't-statistic' in rb.simulate_two_books.__doc__


def test_account_budget_negative_rho_blocks_hedge_and_finds_rising_crossing():
    # Early-exit: book already over cap -> 0 even for a perfect hedge
    # (fail-safe: only ever blocks, never enlarges).
    assert rb.account_risk_budget('crypto', [0.04], [], 0.0, 0.0, -1.0) == 0.0
    # U-shape acct(rc) = |0.02 - rc|, cap 0.03: single rising crossing at
    # rc = 0.05 — bisection must land there and never exceed the cap.
    budget = rb.account_risk_budget('crypto', [0.02], [], 0.0, 0.0, -1.0,
                                    cap=0.03, max_risk=0.10)
    assert budget == pytest.approx(0.05, abs=1e-6)
    acct = rb.account_stop_risk([0.02], [budget], 0.0, 0.0, -1.0)
    assert acct <= 0.03 + 1e-9
    assert 'U-shaped' in rb.account_risk_budget.__doc__


# ---------------------------------------------------------------------------
# drawdown.py
# ---------------------------------------------------------------------------

def test_restore_peak_rejects_infinities():
    assert restore_peak_equity(float('inf'), 90_000.0) == PEAK_SEED
    assert restore_peak_equity(float('-inf'), 90_000.0) == PEAK_SEED
    assert restore_peak_equity(float('nan'), 90_000.0) == PEAK_SEED
    assert restore_peak_equity(120_000.0, 90_000.0) == 120_000.0  # normal kept


def test_inf_peak_from_json_state_no_longer_disables_ladder():
    # json round-trips Infinity, so a corrupted position_state.json CAN hand
    # restore a +inf peak; before the fix max() pinned it forever and the
    # ladder froze at 1.0x (drawdown_fraction(inf, e) -> nan -> 0.0).
    saved = json.loads(json.dumps({'peak_equity': float('inf')}))['peak_equity']
    assert saved == float('inf')
    peak = restore_peak_equity(saved, 80_000.0)
    assert peak == PEAK_SEED                    # seed (100k) > current (80k)
    dd = drawdown_fraction(peak, 80_000.0)
    assert dd == pytest.approx(0.20)
    assert drawdown_size_multiplier(dd) == 0.25  # ladder armed, not 1.0x


def test_drawdown_multiplier_docstring_matches_ladder_orientation():
    assert 'shallowest rung' in drawdown_size_multiplier.__doc__
    assert drawdown_size_multiplier(0.05) == 1.0    # below shallowest rung
    assert drawdown_size_multiplier(0.25) == 0.25   # past the richest rung
