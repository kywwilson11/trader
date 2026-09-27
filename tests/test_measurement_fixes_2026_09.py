"""2026-09 measurement-only fixes (Jetson audit D_measurement).

Pins: decision_report counts buys dropped for pred_return=null and states
the true stock replay window; beta_ledger drops spike-and-revert vendor
bad-print days (fold-through of profit_loss); execution_report treats
tactic-less buys as 'unknown' in the notional maker share and writes its
report atomically; sizing_cofire_report --json PATH writes a file and bad
arguments exit non-zero; llm_eval writes via a unique temp name and its
low-n verdict states the real power floor; chart_core distinguishes a
no-rows stale stub from a no-API one; gui's Gap Audit passes only capped
stock sleeve candidates (source-text pin — PySide6 is not importable here).

Mac-safe: numpy/pandas + the pure functions of these modules.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import beta_ledger  # noqa: E402
import chart_core  # noqa: E402
import decision_report  # noqa: E402
import execution_report  # noqa: E402
import llm_eval  # noqa: E402


# --------------------------------------------------------------------------
# 1. decision_report
# --------------------------------------------------------------------------

def test_conviction_counts_null_pred_buys_when_all_null(monkeypatch):
    monkeypatch.setattr(decision_report, '_replay_grouped',
                        lambda *a, **k: ([], 0, 0, 0))
    rows = [{'action': 'buy', 'symbol': 'BTC/USD', 'pred_return': None,
             'ts': '2026-05-01T00:00:00+00:00'} for _ in range(5)]
    out = decision_report.conviction_calibration(rows, api=None)
    assert out['_dropped_null_pred'] == 5


def test_conviction_counts_null_pred_buys_mixed(monkeypatch):
    seen = {}

    def fake_replay(buys, api, **k):
        seen['n'] = len(buys)
        return [(r, 0.1) for r in buys], 0, 0, 0
    monkeypatch.setattr(decision_report, '_replay_grouped', fake_replay)
    rows = ([{'action': 'buy', 'symbol': 'BTC/USD', 'pred_return': None}] * 3
            + [{'action': 'buy', 'symbol': 'AAA', 'pred_return': 0.01 * i}
               for i in range(10)]
            + [{'action': 'skip', 'symbol': 'AAA', 'pred_return': None}])
    out = decision_report.conviction_calibration(rows, api=None)
    assert seen['n'] == 10                  # null-pred buys still not replayed
    assert out['_dropped_null_pred'] == 3   # ...but counted
    assert out['n'] == 10


def _stub_api_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, 'dotenv',
                        types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, 'trading_utils',
                        types.SimpleNamespace(get_api=lambda: object()))


def _seed_journal(tmp_path, monkeypatch, rows):
    jd = tmp_path / 'journals'
    jd.mkdir()
    with open(jd / f'{dt.date.today().isoformat()}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    monkeypatch.setattr(decision_report, 'JOURNAL_DIR', jd)
    monkeypatch.setattr(decision_report, 'BASE_DIR', tmp_path)


def test_run_report_banner_and_quality_count_null_pred(tmp_path, monkeypatch,
                                                         capsys):
    _seed_journal(tmp_path, monkeypatch, [
        {'action': 'buy', 'symbol': 'BTC/USD', 'pred_return': None,
         'ts': dt.datetime.now(dt.timezone.utc).isoformat()}])
    _stub_api_modules(monkeypatch)
    monkeypatch.setattr(decision_report, 'gate_attribution',
                        lambda *a, **k: {'_fetch_failed': 0,
                                         '_horizon_pending': 0,
                                         '_out_of_window': 0})
    monkeypatch.setattr(decision_report, 'signal_exit_audit',
                        lambda *a, **k: {})
    monkeypatch.setattr(decision_report, 'conviction_calibration',
                        lambda *a, **k: {'n': 12, '_fetch_failed': 0,
                                         '_horizon_pending': 0,
                                         '_out_of_window': 0,
                                         '_dropped_null_pred': 88})
    report = decision_report.run_report(days=30)
    out = capsys.readouterr().out
    q = report['quality']
    assert q['dropped_null_pred'] == 88
    assert q['unpriced'] == 88
    assert q['horizon_pending'] == 0
    assert q['representative'] is False          # 88/100 unpriced
    assert '88 buy row(s) dropped' in out
    # the stock-window NOTE fires at the true ~26-30 day threshold
    assert 'last 320 hourly bars' in out
    assert '~45' not in out


def test_stock_window_note_threshold(tmp_path, monkeypatch, capsys):
    assert decision_report.STOCK_FRAME_MIN_DAYS == 26
    _seed_journal(tmp_path, monkeypatch, [
        {'action': 'skip', 'symbol': 'AAA', 'skip_reason': 'x',
         'ts': dt.datetime.now(dt.timezone.utc).isoformat()}])
    _stub_api_modules(monkeypatch)
    for name in ('gate_attribution', 'signal_exit_audit',
                 'conviction_calibration'):
        monkeypatch.setattr(decision_report, name, lambda *a, **k: {})
    decision_report.run_report(days=14)
    assert 'last 320 hourly bars' not in capsys.readouterr().out


# --------------------------------------------------------------------------
# 2. beta_ledger
# --------------------------------------------------------------------------

def _eq(values, start='2026-08-01'):
    idx = pd.date_range(start, periods=len(values), freq='D', tz='UTC')
    return pd.Series(np.asarray(values, dtype=float), index=idx, name='equity')


def test_glitch_spike_that_reverts_is_dropped_and_pl_folded():
    eq = _eq([83000, 83100, 93.63, 82777, 82900])
    pl = pd.Series([0, 100, -83006.37, 82683.37, 123], index=eq.index,
                   name='profit_loss')
    kept, pl_kept, dropped = beta_ledger.drop_glitch_days(eq, pl)
    assert dropped == ['2026-08-03']
    assert len(kept) == 4 and 93.63 not in kept.values
    # folded pl on the next kept day == the true 2-day P&L
    assert pl_kept.loc['2026-08-04'] == pytest.approx(-323.0)
    clean = beta_ledger.clean_returns_from_pl(kept, pl_kept)
    assert abs(clean.loc['2026-08-04']) < 0.01


def test_glitch_nonreverting_crash_is_kept():
    eq = _eq([100000, 100500, 40000, 40100, 40200, 40300])
    kept, _, dropped = beta_ledger.drop_glitch_days(eq)
    assert dropped == []
    assert len(kept) == len(eq)


def test_glitch_upward_spike_and_multi_day_run():
    eq = _eq([1000, 1010, 5000, 5100, 1020, 1030])
    kept, _, dropped = beta_ledger.drop_glitch_days(eq, revert_days=3)
    assert dropped == ['2026-08-03', '2026-08-04']
    # a reversion beyond revert_days is not a glitch
    eq2 = _eq([1000, 1010, 5000, 5100, 5200, 5300, 1020])
    assert beta_ledger.drop_glitch_days(eq2, revert_days=3)[2] == []


def test_glitch_normal_volatility_untouched():
    rng = np.random.default_rng(0)
    eq = _eq(100000 * np.cumprod(1 + rng.normal(0, 0.03, 200)))
    kept, _, dropped = beta_ledger.drop_glitch_days(eq)
    assert dropped == [] and len(kept) == 200


def test_glitch_on_first_day_is_dropped():
    # REVIEW L6: day 0 has no prior good day. Pre-fix it became the
    # reference, no later day "reverted" to it, and it was never dropped.
    eq = _eq([93.63, 83000, 83100, 82900, 83050])
    pl = pd.Series([-82900.0, 82906.37, 100, -200, 150], index=eq.index)
    kept, pl_kept, dropped = beta_ledger.drop_glitch_days(eq, pl)
    assert dropped == ['2026-08-01']
    assert list(kept.values) == [83000, 83100, 82900, 83050]
    # day 0's pl folds into the first kept day (its clean return is NaN
    # anyway: no prior equity), later days untouched
    assert pl_kept.loc['2026-08-02'] == pytest.approx(6.37)
    assert list(pl_kept.values[1:]) == [100, -200, 150]
    clean = beta_ledger.clean_returns_from_pl(kept, pl_kept)
    assert np.isnan(clean.iloc[0])
    assert np.nanmax(np.abs(clean.values)) < 0.01
    # upward first-day spike too
    assert beta_ledger.drop_glitch_days(
        _eq([500000, 1000, 1010, 1020]))[2] == ['2026-08-01']


def test_first_day_glitch_plus_midwindow_glitch():
    eq = _eq([93.63, 83000, 83100, 50.0, 82900, 83050])
    kept, _, dropped = beta_ledger.drop_glitch_days(eq)
    assert dropped == ['2026-08-01', '2026-08-04']
    assert len(kept) == 4


@pytest.mark.parametrize('vals', [
    [100000, 100500, 40000, 40100, 40200],   # level change on day 2: kept
    [0, 0, 100000, 100100, 100200],          # pre-funding zeros: untouched
    [93.63, 83000],                          # too short to confirm day 1
    [93.63, 83000, 200],                     # day 1 not confirmed by day 2
    [100000, 101000, 99000, 100500],         # normal
])
def test_first_day_rule_does_not_fire_without_confirmation(vals):
    eq = _eq(vals)
    kept, _, dropped = beta_ledger.drop_glitch_days(eq)
    assert '2026-08-01' not in dropped


def test_nan_pl_on_dropped_day_carries_as_zero_and_warns(capsys):
    # REVIEW L6: a NaN pl on a dropped day used to make carry NaN and
    # poison the next kept day's pl (and so its clean return).
    eq = _eq([83000, 83100, 93.63, 82777, 82900])
    pl = pd.Series([0, 100, np.nan, 82683.37, 123], index=eq.index)
    kept, pl_kept, dropped = beta_ledger.drop_glitch_days(eq, pl)
    assert dropped == ['2026-08-03']
    assert pl_kept.loc['2026-08-04'] == pytest.approx(82683.37)
    assert np.isfinite(pl_kept.values).all()
    err = capsys.readouterr().err
    assert 'WARNING' in err and 'non-finite profit_loss' in err
    assert '2026-08-03' in err
    # finite pl: no warning
    pl2 = pd.Series([0, 100, -83006.37, 82683.37, 123], index=eq.index)
    beta_ledger.drop_glitch_days(eq, pl2)
    assert 'non-finite' not in capsys.readouterr().err


def test_nan_pl_on_kept_day_is_left_nan():
    # only DROPPED days' non-finite pl is zeroed; a kept day's NaN stays NaN
    eq = _eq([83000, 83100, 93.63, 82777, 82900])
    pl = pd.Series([0, np.nan, -83006.37, 82683.37, 123], index=eq.index)
    _, pl_kept, _ = beta_ledger.drop_glitch_days(eq, pl)
    assert np.isnan(pl_kept.loc['2026-08-02'])
    assert pl_kept.loc['2026-08-04'] == pytest.approx(-323.0)


def test_glitch_bad_max_move_rejected():
    with pytest.raises(ValueError):
        beta_ledger.drop_glitch_days(_eq([1, 2, 3]), max_move=1.5)


def test_beta_ledger_cli_drops_by_default_and_opt_out(tmp_path):
    rng = np.random.default_rng(1)
    n = 80
    idx = pd.date_range('2026-05-01', periods=n, freq='D')
    spy = 400 * np.cumprod(1 + rng.normal(0, 0.01, n))
    btc = 60000 * np.cumprod(1 + rng.normal(0, 0.02, n))
    eq = 100000 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    eq[40] = 93.63
    pd.DataFrame({'date': idx, 'equity': eq}).to_csv(tmp_path / 'eq.csv',
                                                     index=False)
    pd.DataFrame({'date': idx, 'SPY': spy, 'BTC': btc}).to_csv(
        tmp_path / 'b.csv', index=False)
    base = [sys.executable, str(REPO / 'beta_ledger.py'),
            '--equity-csv', str(tmp_path / 'eq.csv'),
            '--benchmarks-csv', str(tmp_path / 'b.csv')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    on = subprocess.run(base + ['--json', str(tmp_path / 'on.json')],
                        capture_output=True, text=True, cwd=str(tmp_path),
                        env=env)
    assert on.returncode == 0, on.stderr
    assert 'dropped 1 glitch day(s)' in on.stdout
    rep_on = json.loads((tmp_path / 'on.json').read_text())
    assert rep_on['data_quality']['glitch_days_dropped'] == [
        str(idx[40].date())]
    off = subprocess.run(base + ['--no-drop-glitch-days',
                                 '--json', str(tmp_path / 'off.json')],
                         capture_output=True, text=True, cwd=str(tmp_path),
                         env=env)
    assert off.returncode == 0, off.stderr
    rep_off = json.loads((tmp_path / 'off.json').read_text())
    assert rep_off['data_quality']['glitch_days_dropped'] == []
    assert rep_off['data_quality']['drop_glitch_days'] is False
    assert (abs(rep_off['strategy']['ann_return'])
            > 10 * abs(rep_on['strategy']['ann_return']))


# --------------------------------------------------------------------------
# 3. execution_report
# --------------------------------------------------------------------------

def _exec_journal(tmp_path, monkeypatch, rows):
    jd = tmp_path / 'journals'
    jd.mkdir()
    with open(jd / f'{dt.date.today().isoformat()}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    monkeypatch.setattr(execution_report, 'JOURNAL_DIR', jd)
    monkeypatch.setattr(execution_report, 'BASE_DIR', tmp_path)


def test_notional_share_excludes_tactic_less_buys(tmp_path, monkeypatch):
    _exec_journal(tmp_path, monkeypatch, [
        {'action': 'buy', 'symbol': 'BTC/USD', 'final_notional': 1000.0,
         'entry_tactic': 'maker_l1'},
        {'action': 'buy', 'symbol': 'ETH/USD', 'final_notional': 1000.0,
         'entry_tactic': 'taker'},
        {'action': 'buy', 'symbol': 'SOL/USD', 'final_notional': 8000.0},
    ])
    rep = execution_report.run_report(days=1)
    assert rep['crypto_maker_notional_share'] == 0.5      # not 0.1
    assert rep['crypto_unknown_tactic_notional'] == 8000.0
    assert rep['crypto_maker_share'] == 0.5               # count block agrees


def test_notional_share_absent_when_all_tactic_less(tmp_path, monkeypatch,
                                                    capsys):
    _exec_journal(tmp_path, monkeypatch, [
        {'action': 'buy', 'symbol': 'SOL/USD', 'final_notional': 8000.0}])
    rep = execution_report.run_report(days=1)
    assert 'crypto_maker_notional_share' not in rep       # no fabricated 0.0%
    assert rep['crypto_unknown_tactic_notional'] == 8000.0
    assert 'n/a' in capsys.readouterr().out


def test_execution_report_write_is_atomic(tmp_path, monkeypatch):
    calls = []
    real_replace = os.replace

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)
    monkeypatch.setattr(execution_report.os, 'replace', spy)
    out = tmp_path / 'execution_report.json'
    execution_report._write_json({'a': 1}, out)
    assert json.loads(out.read_text()) == {'a': 1}
    assert len(calls) == 1 and calls[0][1] == str(out)
    assert Path(calls[0][0]).parent == tmp_path and calls[0][0] != str(out)
    assert [p.name for p in tmp_path.iterdir()] == ['execution_report.json']


# --------------------------------------------------------------------------
# 4. scripts/sizing_cofire_report.py
# --------------------------------------------------------------------------

_SC = str(REPO / 'scripts' / 'sizing_cofire_report.py')


def _sc_journal(tmp_path):
    jd = tmp_path / 'j'
    jd.mkdir()
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    with open(jd / f'{dt.date.today().isoformat()}.jsonl', 'w') as f:
        f.write(json.dumps({'ts': ts, 'action': 'buy', 'symbol': 'BTC/USD',
                            'sizing': {'vix_tilt': 0.7, 'tilt_raw': 0.7,
                                       'tilt': 0.7}}) + '\n')
    return jd


def test_sizing_cofire_json_path_writes_file(tmp_path):
    jd = _sc_journal(tmp_path)
    out = tmp_path / 'rep.json'
    p = subprocess.run([sys.executable, _SC, '--journal-dir', str(jd),
                        '--json', str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(out.read_text())['n_buy_rows'] == 1
    assert not list(tmp_path.glob('rep.json.*.tmp'))


def test_sizing_cofire_bare_json_still_stdout(tmp_path):
    jd = _sc_journal(tmp_path)
    p = subprocess.run([sys.executable, _SC, '--journal-dir', str(jd),
                        '--json', '--book', 'crypto'],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)['n_buy_rows'] == 1


def test_sizing_cofire_bad_arg_exits_nonzero(tmp_path):
    p = subprocess.run([sys.executable, _SC, '--book', 'bogus'],
                       capture_output=True, text=True)
    assert p.returncode == 2
    p = subprocess.run([sys.executable, _SC, '--no-such-flag'],
                       capture_output=True, text=True)
    assert p.returncode == 2
    p = subprocess.run([sys.executable, _SC, '--help'],
                       capture_output=True, text=True)
    assert p.returncode == 0


def test_sizing_cofire_unwritable_json_path_exits_1(tmp_path):
    jd = _sc_journal(tmp_path)
    p = subprocess.run([sys.executable, _SC, '--journal-dir', str(jd),
                        '--json', str(tmp_path / 'nope' / 'rep.json')],
                       capture_output=True, text=True)
    assert p.returncode == 1


# --------------------------------------------------------------------------
# 5. llm_eval
# --------------------------------------------------------------------------

def test_llm_eval_write_uses_unique_tmp_same_dir(tmp_path, monkeypatch):
    srcs = []
    real_replace = os.replace

    def spy(src, dst):
        srcs.append(str(src))
        return real_replace(src, dst)
    monkeypatch.setattr(llm_eval.os, 'replace', spy)
    path = tmp_path / 'llm_eval_report.json'
    llm_eval._write_report(path, {'x': 1})
    llm_eval._write_report(path, {'x': 2})
    assert json.loads(path.read_text()) == {'x': 2}
    assert len(srcs) == 2 and srcs[0] != srcs[1]
    assert all(Path(s).parent == tmp_path for s in srcs)
    assert str(path.with_suffix('.json.tmp')) not in srcs
    assert [p.name for p in tmp_path.iterdir()] == ['llm_eval_report.json']


def test_llm_eval_low_n_verdict_states_real_power_floor():
    rng = np.random.default_rng(0)
    n = 20
    pred = rng.normal(size=n)
    samples = list(zip(rng.uniform(size=n), pred + rng.normal(size=n), pred))
    rep = llm_eval.compute_incremental_report(samples, forward_bars=24)
    assert 'insufficient_power' in rep['verdict']
    assert str(llm_eval.MIN_POWER_T0) in rep['verdict']
    assert str(llm_eval.MIN_EFFECTIVE_N) in rep['verdict']


# --------------------------------------------------------------------------
# 6. chart_core stale reason / gui gap-audit launch site
# --------------------------------------------------------------------------

def test_gate_panel_stale_reason_no_rows_vs_no_api():
    no_rows = chart_core.gate_panel_model(
        {'stale': True, 'api_available': None, 'days': 30})
    assert no_rows['stale_reason'].startswith('no journal rows')
    no_api = chart_core.gate_panel_model(
        {'stale': True, 'api_available': False, 'days': 30})
    assert no_api['stale_reason'].startswith('no API when generated')
    legacy = chart_core.gate_panel_model({'stale': True})
    assert legacy['stale_reason'].startswith('no API when generated')


def test_gui_gap_audit_passes_capped_stock_sleeve_candidates():
    src = (REPO / 'gui.py').read_text()
    start = src.index('def _run_gap_audit_clicked')
    body = src[start:src.index('\n    def ', start + 10)]
    assert "'/' not in" in body                       # no crypto
    assert 'LEVERAGED_ETFS' in body                   # sleeve exclusion
    assert 'OVERNIGHT_SLEEVE_MAX_POSITIONS' in body   # capped at sleeve size
    assert 'ranked[:' in body
