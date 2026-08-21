"""Tests for backtest.py's 2026-07 instrumentation & hardening pass.

Follows tests/test_gate_protocol.py's monkeypatch-the-heavy-seams pattern:
backtest's model-loading and LSTM-prediction functions are monkeypatched at
the seams so run_backtest can be exercised end-to-end on the dev Mac without
torch/lightgbm/joblib. backtest.py's own top level imports only numpy/fees/
strategy_config/validation, so no pytest.importorskip is needed here.

NOTE on test #16 (coverage-split adaptation): run_backtest's missing-features
check is `c not in tdf.columns`, evaluated on a per-ticker slice `tdf =
df[df['Ticker'] == ticker]`. Boolean row-filtering never drops columns, and
pd.concat of heterogeneous per-ticker frames back-fills the column union
with NaN (verified empirically) — so `tdf.columns` is IDENTICAL to
`df.columns` for every ticker, always. A configured feature absent from one
ticker's harvest is therefore absent for every ticker's slice equally; the
missing-features branch cannot fire for one name while a sibling name is
fully evaluated in the same combined frame. test_coverage_short_history_split
and test_coverage_missing_features_all_tickers below exercise the two new
counters against what the code can actually produce, in place of the single
combined 1-evaluated/1-short/1-missing scenario the spec sketch described.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest
from strategy_config import policy_for


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _mk_trades(n, overlapping, net=None, reason='take_profit'):
    """n trade dicts across 6 synthetic names (mirrors
    tests/test_cs_neff.py::_fake_trades). overlapping=True stacks same-hour
    [entry, exit] intervals across the 6 names into one cross-sectional
    cluster per slot; overlapping=False spaces every trade out disjoint in
    calendar time (24h duration, 30h spacing -> zero overlap)."""
    trades = []
    rng = np.random.default_rng(11)
    base = pd.Timestamp('2026-06-01', tz='UTC')
    names = [f'N{j}' for j in range(6)]
    for i in range(n):
        j = i % 6
        if overlapping:
            entry = base + pd.Timedelta(hours=(i // 6) * 30)
        else:
            entry = base + pd.Timedelta(hours=i * 30)
        exit_t = entry + pd.Timedelta(hours=24)
        trades.append({
            'ticker': names[j],
            'entry_time': str(entry), 'exit_time': str(exit_t),
            'entry': 100.0, 'exit': 101.0, 'bars_held': 24,
            'gross_pct': float(rng.normal(0.05, 1.0)),
            'net_pct': (float(rng.normal(0.0, 1.0)) if net is None else net),
            'reason': reason,
        })
    return trades


def _trending_frame(n=60, drift=0.004, seed=0):
    """Gently rising bars so a long signal exits profitably (gross > 0).

    Copied from tests/test_cost_per_bar.py — NO 'Ticker' column (exercises
    simulate_ticker's absent-column ticker guard).
    """
    rng = np.random.RandomState(seed)
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.001, n))
    high = close * 1.002
    low = close * 0.998
    op = close * 0.999
    atr = pd.Series(close).rolling(5, min_periods=1).std().fillna(0.5).values + 0.5
    idx = pd.date_range('2025-03-03 14:00', periods=n, freq='h', tz='UTC')
    return pd.DataFrame({'Open': op, 'High': high, 'Low': low,
                         'Close': close, 'ATR': atr}, index=idx)


def _run_sim(tdf, threshold=0.1):
    preds = np.full(len(tdf), 0.5)  # strong, constant long signal
    return backtest.simulate_ticker(tdf, preds, 'stock', threshold,
                                    policy_for('stock'))


def _one_ticker_frame(ticker='BTC', n=40, seed=0):
    """Harvest-shaped hourly crypto frame with the two harness feature cols."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
    price = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame({
        'Ticker': ticker, 'Open': price, 'High': price * 1.01,
        'Low': price * 0.99, 'Close': price, 'ATR': np.nan,
        'f1': np.linspace(0, 1, n), 'f2': np.linspace(1, 0, n),
    }, index=idx)


def _fake_predict_ticker_factory(actionable_index=20, pred_value=5.0):
    """Mirrors tests/test_gate_protocol.py's factory: one actionable bar."""
    def fake(model, scaler, config, feature_cols, tdf, lgb_model=None,
            q10_model=None):
        n = len(tdf)
        preds = np.full(n, np.nan)
        if n > actionable_index + 1:
            preds[actionable_index] = pred_value
        return preds, None
    return fake


def _harness(monkeypatch, tmp_path, frames, config=None):
    """Wire run_backtest's heavy seams to synthetic frames; returns a
    zero-arg callable that runs run_backtest(prefix='', days=400,
    n_search_trials=10) against the concatenation of `frames`."""
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(
        backtest, '_load_artifacts',
        lambda prefix: (None, None,
                       config or {'seq_len': 5, 'trade_threshold': 0.05},
                       ['f1', 'f2']))
    monkeypatch.setattr(backtest, '_load_lgb', lambda prefix: None)
    monkeypatch.setattr(backtest, '_load_q10', lambda prefix: None)
    monkeypatch.setattr(backtest, '_predict_ticker',
                        _fake_predict_ticker_factory())
    combined = pd.concat(frames)
    monkeypatch.setattr('data_utils.load_training_data',
                        lambda asset: combined)

    def _run():
        return backtest.run_backtest(prefix='', days=400, n_search_trials=10)
    return _run


# ---------------------------------------------------------------------------
# 1. aggregate_metrics: empty/populated schema parity
# ---------------------------------------------------------------------------

def test_empty_and_populated_schema_match():
    empty = backtest.aggregate_metrics([], 'crypto', 44.0)
    populated = backtest.aggregate_metrics(_mk_trades(12, False), 'crypto', 44.0)
    assert set(empty.keys()) == set(populated.keys())


# ---------------------------------------------------------------------------
# 2/3. n_eff_source provenance
# ---------------------------------------------------------------------------

def test_n_eff_source_states():
    overlapped = backtest.aggregate_metrics(_mk_trades(60, True), 'crypto', 90.0)
    assert overlapped['n_eff_source'] == 'clustered'

    spread = backtest.aggregate_metrics(_mk_trades(12, False), 'crypto', 90.0)
    assert spread['n_eff_source'] == 'iid_no_overlap'
    assert spread['n_eff_clustered'] == spread['n_trades']


def test_n_eff_source_unavailable_on_exception(monkeypatch):
    def raiser(*a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr('sample_weights.clustered_effective_n', raiser)
    m = backtest.aggregate_metrics(_mk_trades(30, True), 'crypto', 90.0)
    assert m['n_eff_source'] == 'iid_unavailable'
    assert m['n_eff_clustered'] == m['n_trades']
    assert m['dsr'] == m['dsr_iid']


# ---------------------------------------------------------------------------
# 4. DSR detail fields
# ---------------------------------------------------------------------------

def test_dsr_detail_fields():
    trades = _mk_trades(60, True)
    m = backtest.aggregate_metrics(trades, 'crypto', 90.0, n_search_trials=10)
    expected_n_eff = min(max(m['n_eff_clustered'], 10.0), m['n_trades'])
    assert m['dsr_n_eff_used'] == pytest.approx(expected_n_eff, abs=0.01)
    assert m['dsr_expected_max_sr'] > 0
    assert m['dsr_n_trials'] == 10
    assert 0.0 <= m['dsr_iid'] <= 1.0
    assert m['dsr_raw'] == pytest.approx(m['dsr'], abs=1e-4)


# ---------------------------------------------------------------------------
# 5/6. Warnings printed
# ---------------------------------------------------------------------------

def test_collapse_warning_printed(capsys):
    backtest.aggregate_metrics(_mk_trades(60, True), 'crypto', 90.0)
    assert 'clustering collapsed' in capsys.readouterr().out


def test_nonfinite_trades_warned(capsys):
    trades = _mk_trades(12, False)
    trades[3]['net_pct'] = float('nan')
    m = backtest.aggregate_metrics(trades, 'crypto', 90.0)
    assert m['n_nonfinite_trades'] == 1
    assert m['sharpe'] == 0.0
    assert 'non-finite net_pct' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 7. censored_trade_frac / span_days
# ---------------------------------------------------------------------------

def test_censored_frac_and_span():
    trades = _mk_trades(10, False)
    trades[0]['reason'] = 'end_of_data'
    m = backtest.aggregate_metrics(trades, 'crypto', 55.4)
    assert m['censored_trade_frac'] == pytest.approx(0.1)
    assert m['span_days'] == round(55.4, 2)


# ---------------------------------------------------------------------------
# 8. simulate_ticker ticker key
# ---------------------------------------------------------------------------

def test_trade_ticker_key():
    tdf = _trending_frame()
    tdf_with = tdf.copy()
    tdf_with['Ticker'] = 'XYZ'
    trades = _run_sim(tdf_with)
    assert trades
    assert all(t['ticker'] == 'XYZ' for t in trades)

    trades_without = _run_sim(tdf)
    assert trades_without
    assert all(t['ticker'] == '' for t in trades_without)


# ---------------------------------------------------------------------------
# 9. per-bar cost failure warns + flat-cost fallback
# ---------------------------------------------------------------------------

def test_per_bar_cost_failure_warns(monkeypatch, capsys):
    tdf = _trending_frame().copy()
    tdf['Eff_Spread_Pct'] = np.linspace(0.05, 0.9, len(tdf))

    def _boom(*a, **k):
        raise RuntimeError('per-bar cost regressed')
    monkeypatch.setattr('liquidity.per_bar_round_trip_cost', _boom)

    trades = _run_sim(tdf)
    out = capsys.readouterr().out
    assert '[GATE] per-bar spread cost failed' in out

    from fees import round_trip_cost_pct
    flat = round_trip_cost_pct('stock', backtest.SPREAD_PCT['stock'])
    assert trades
    for t in trades:
        assert t['net_pct'] == pytest.approx(t['gross_pct'] - flat, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. tz-naive entry-window warning (once) + mask semantics preserved
# ---------------------------------------------------------------------------

def test_entry_window_tz_naive_warns_once_mask_unchanged(monkeypatch, capsys):
    from strategy_config import STOCK_ENTRY_WINDOWS_ET

    # Full UTC day so the test doesn't depend on which DST regime the date
    # falls in: whatever the true UTC->ET offset (4h or 5h), the raw-hour
    # and correctly-shifted True sets provably differ (verified below).
    idx_aware = pd.date_range('2026-03-02 00:00', periods=24, freq='h', tz='UTC')
    naive = idx_aware.tz_localize(None)

    monkeypatch.setattr(backtest, '_TZ_NAIVE_WARNED', False)
    m1 = backtest._entry_window_mask(naive)
    out1 = capsys.readouterr().out
    assert out1.count('tz-NAIVE') == 1

    m1_again = backtest._entry_window_mask(naive)
    out2 = capsys.readouterr().out
    assert 'tz-NAIVE' not in out2  # warned once per process, not per call

    # Pre-change semantics: raw clock hour treated as if it were already ET.
    windows = []
    for start_s, end_s in STOCK_ENTRY_WINDOWS_ET:
        sh, sm = map(int, start_s.split(':'))
        eh, em = map(int, end_s.split(':'))
        windows.append((sh * 60 + sm, eh * 60 + em))
    expected = np.zeros(len(naive), dtype=bool)
    for i, t in enumerate(naive):
        minutes = t.hour * 60 + t.minute
        expected[i] = any(s <= minutes < e for s, e in windows)
    np.testing.assert_array_equal(m1, expected)
    np.testing.assert_array_equal(m1_again, expected)

    m_aware = backtest._entry_window_mask(idx_aware)
    assert not np.array_equal(m_aware, m1)


# ---------------------------------------------------------------------------
# 11/12. _load_q10 BASE_DIR anchoring + finite-floor guard
# ---------------------------------------------------------------------------

def test_load_q10_uses_base_dir(monkeypatch, tmp_path):
    import types
    captured = {}

    class FakeBooster:
        def __init__(self, model_file=None):
            captured['model_file'] = model_file

    fake_lgb = types.ModuleType('lightgbm')
    fake_lgb.Booster = FakeBooster
    monkeypatch.setitem(sys.modules, 'lightgbm', fake_lgb)

    (tmp_path / 'lgb_q10.txt').write_text('booster bytes')
    (tmp_path / 'lgb_q10_meta.json').write_text('{"floor": -1.0}')
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)

    other_dir = tmp_path.parent / (tmp_path.name + '_elsewhere')
    other_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(other_dir)

    result = backtest._load_q10('')
    assert result is not None
    booster, floor = result
    assert floor == -1.0
    assert captured['model_file'].startswith(str(tmp_path))


def test_load_q10_nonfinite_floor_rejected(monkeypatch, tmp_path, capsys):
    import types

    class FakeBooster:
        def __init__(self, model_file=None):
            pass

    fake_lgb = types.ModuleType('lightgbm')
    fake_lgb.Booster = FakeBooster
    monkeypatch.setitem(sys.modules, 'lightgbm', fake_lgb)

    (tmp_path / 'lgb_q10.txt').write_text('booster bytes')
    (tmp_path / 'lgb_q10_meta.json').write_text('{"floor": NaN}')
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)

    assert backtest._load_q10('') is None
    assert 'present but failed to load' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 13/14. restore_previous_model manifest-last + partial-failure CRITICAL
# ---------------------------------------------------------------------------

def test_restore_manifest_replaced_last(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    for s in backtest.ARTIFACT_SUFFIXES:
        (tmp_path / f'{s}.prev').write_text('prev')
        (tmp_path / s).write_text('cur')

    recorded = []
    real_replace = os.replace

    def _rec_replace(src, dst):
        recorded.append(Path(dst).name)
        return real_replace(src, dst)
    monkeypatch.setattr(backtest.os, 'replace', _rec_replace)

    assert backtest.restore_previous_model('') is True
    assert recorded[-1] == 'model_v2.manifest.json'
    assert not any(p.name.endswith('.prev') for p in tmp_path.iterdir())


def test_restore_partial_failure_critical(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    for s in backtest.ARTIFACT_SUFFIXES:
        (tmp_path / f'{s}.prev').write_text('prev')
        (tmp_path / s).write_text('cur')

    calls = {'n': 0}
    real_replace = os.replace

    def _flaky_replace(src, dst):
        calls['n'] += 1
        if calls['n'] > 2:
            raise OSError('disk fell over')
        return real_replace(src, dst)
    monkeypatch.setattr(backtest.os, 'replace', _flaky_replace)

    with pytest.raises(OSError):
        backtest.restore_previous_model('')
    assert 'CRITICAL: rollback FAILED partway' in capsys.readouterr().out


def test_restore_partial_failure_reports_planned_denominator(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    # Only the 4 required legs + 1 optional leg have a .prev: the CRITICAL
    # banner's denominator must count the 5 PLANNED restores, not all 11
    # suffixes (most of which never had a .prev to restore).
    for s in backtest.ARTIFACT_SUFFIXES[:4] + [backtest.ARTIFACT_SUFFIXES[5]]:
        (tmp_path / f'{s}.prev').write_text('prev')
        (tmp_path / s).write_text('cur')

    calls = {'n': 0}
    real_replace = os.replace

    def _flaky_replace(src, dst):
        calls['n'] += 1
        if calls['n'] > 2:
            raise OSError('disk fell over')
        return real_replace(src, dst)
    monkeypatch.setattr(backtest.os, 'replace', _flaky_replace)

    with pytest.raises(OSError):
        backtest.restore_previous_model('')
    assert '2/5 artifacts restored' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 15. meta-veto exception warns once, flag stays False
# ---------------------------------------------------------------------------

def test_meta_exception_warns_once_and_flag(monkeypatch, tmp_path, capsys):
    frames = [_one_ticker_frame('BTC'), _one_ticker_frame('ETH', seed=1)]
    run = _harness(monkeypatch, tmp_path, frames)

    def raiser(prefix, tdf, preds):
        raise RuntimeError('meta booster corrupt')
    monkeypatch.setattr('meta_label.predict_meta_array', raiser)

    metrics = run()
    out = capsys.readouterr().out
    assert out.count('meta veto unavailable') == 1
    assert metrics['meta_veto_active'] is False


# ---------------------------------------------------------------------------
# 16. Coverage split counters (adapted — see module docstring NOTE)
# ---------------------------------------------------------------------------

def test_coverage_short_history_split(monkeypatch, tmp_path):
    evaluated = _one_ticker_frame('BTC', n=40)
    idx_short = pd.date_range('2026-01-01', periods=3, freq='h', tz='UTC')
    short_history = pd.DataFrame({
        'Ticker': 'ETH', 'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'ATR': np.nan, 'f1': 0.0, 'f2': 0.0,
    }, index=idx_short)

    run = _harness(monkeypatch, tmp_path, [evaluated, short_history])
    metrics = run()
    assert metrics['n_tickers_evaluated'] == 1
    assert metrics['n_skipped_short_history'] == 1
    assert metrics['n_skipped_missing_features'] == 0
    assert metrics['coverage_frac'] == pytest.approx(0.5)


def test_coverage_missing_features_all_tickers(monkeypatch, tmp_path):
    # tdf.columns == df.columns for every ticker (row-filtering never drops
    # columns), so a feature absent from the harvest is absent for every
    # evaluable-length ticker equally — this fires for BOTH names, not one.
    frames = [_one_ticker_frame('BTC', n=40), _one_ticker_frame('SOL', n=40, seed=2)]
    run = _harness(monkeypatch, tmp_path, frames,
                   config={'seq_len': 5, 'trade_threshold': 0.05})
    monkeypatch.setattr(
        backtest, '_load_artifacts',
        lambda prefix: (None, None,
                       {'seq_len': 5, 'trade_threshold': 0.05},
                       ['f1', 'f2', 'f3']))  # f3 absent from every frame
    metrics = run()
    assert metrics['n_tickers_evaluated'] == 0
    assert metrics['n_skipped_missing_features'] == 2
    assert metrics['n_skipped_short_history'] == 0
    assert metrics['coverage_frac'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 17. Report trades sorted + counted
# ---------------------------------------------------------------------------

def test_report_trades_sorted_and_counts(monkeypatch, tmp_path):
    # AAA (processed first, n=40) exits LATER than BBB (processed second,
    # n=20, so its data — and thus its forced end_of_data exit — ends
    # earlier): the natural insertion order is un-sorted, so this actually
    # exercises the sort rather than passing on already-sorted input.
    first = _one_ticker_frame('AAA', n=40, seed=2)
    second = _one_ticker_frame('BBB', n=20, seed=3)

    def fake_predict(model, scaler, config, feature_cols, tdf, lgb_model=None,
                     q10_model=None):
        n = len(tdf)
        preds = np.full(n, np.nan)
        idx = 25 if tdf['Ticker'].iloc[0] == 'AAA' else 10
        preds[idx] = 5.0
        return preds, None

    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(
        backtest, '_load_artifacts',
        lambda prefix: (None, None,
                       {'seq_len': 5, 'trade_threshold': 0.05}, ['f1', 'f2']))
    monkeypatch.setattr(backtest, '_load_lgb', lambda prefix: None)
    monkeypatch.setattr(backtest, '_load_q10', lambda prefix: None)
    monkeypatch.setattr(backtest, '_predict_ticker', fake_predict)
    combined = pd.concat([first, second])
    monkeypatch.setattr('data_utils.load_training_data',
                        lambda asset: combined)

    backtest.run_backtest(prefix='', days=400, n_search_trials=10)

    report = json.loads((tmp_path / 'backtest_report.json').read_text())
    exit_times = [t['exit_time'] for t in report['trades']]
    assert exit_times == sorted(exit_times)
    assert report['n_trades_total'] == report['n_trades_persisted']
    assert report['n_trades_total'] == len(report['trades'])


# ---------------------------------------------------------------------------
# 18/19. main() gate_block persisted verdict
# ---------------------------------------------------------------------------

def test_gate_block_fail_and_not_rolled_back(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    (tmp_path / 'backtest_report.json').write_text('{"metrics": {}}')
    monkeypatch.setattr(
        backtest, 'run_backtest',
        lambda prefix, days, trials: {'n_trades': 0, 'sharpe': 0.0, 'dsr': 0.0})
    monkeypatch.setattr(backtest, 'restore_previous_model', lambda prefix: False)
    monkeypatch.setattr(sys, 'argv', ['backtest.py', '--gate'])

    rc = backtest.main()
    assert rc == 3
    assert 'NOT ROLLED BACK' in capsys.readouterr().out

    report = json.loads((tmp_path / 'backtest_report.json').read_text())
    assert report['gate']['passed'] is False
    assert report['gate']['restored'] is False
    assert report['gate']['exit_code'] == 3
    assert 'n_trades' in report['gate']['failed_checks']


def test_gate_block_on_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    (tmp_path / 'backtest_report.json').write_text('{"metrics": {}}')
    monkeypatch.setattr(
        backtest, 'run_backtest',
        lambda prefix, days, trials: {'n_trades': 20, 'sharpe': 1.0, 'dsr': 0.9})
    monkeypatch.setattr(sys, 'argv', ['backtest.py', '--gate'])

    assert backtest.main() == 0
    report = json.loads((tmp_path / 'backtest_report.json').read_text())
    assert report['gate']['passed'] is True
    assert report['gate']['restored'] is None
    assert report['gate']['exit_code'] == 0
    assert report['gate']['failed_checks'] == []


def test_patch_gate_block_cleans_tmp_on_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    (tmp_path / 'backtest_report.json').write_text('{"metrics": {}}')

    def _boom(src, dst):
        raise OSError('replace failed')
    monkeypatch.setattr(backtest.os, 'replace', _boom)

    backtest._patch_report_gate_block('', {'passed': True})  # must not raise
    assert 'could not record gate verdict' in capsys.readouterr().out
    assert not (tmp_path / 'backtest_report.json.tmp').exists()
    # Original report left intact when the atomic swap never happened.
    assert json.loads(
        (tmp_path / 'backtest_report.json').read_text()) == {'metrics': {}}


# ---------------------------------------------------------------------------
# 20. --days 0 argparse error
# ---------------------------------------------------------------------------

def test_days_zero_argparse_error(monkeypatch):
    calls = []
    monkeypatch.setattr(backtest, 'restore_previous_model',
                        lambda prefix: calls.append(prefix))
    monkeypatch.setattr(sys, 'argv', ['backtest.py', '--days', '0', '--gate'])
    with pytest.raises(SystemExit) as exc_info:
        backtest.main()
    assert exc_info.value.code == 2
    assert calls == []


# ---------------------------------------------------------------------------
# 21/22. FileNotFoundError handling: late crash vs hermetic "nothing to gate"
# ---------------------------------------------------------------------------

def test_late_fnf_reraises_with_core_artifacts(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    for s in backtest.ARTIFACT_SUFFIXES[:4]:
        (tmp_path / s).write_text('x')

    def raise_late(prefix, days, trials):
        raise FileNotFoundError('late')
    monkeypatch.setattr(backtest, 'run_backtest', raise_late)
    monkeypatch.setattr(sys, 'argv', ['backtest.py', '--gate'])

    with pytest.raises(FileNotFoundError):
        backtest.main()
    assert 'CRASHED before a verdict' in capsys.readouterr().out


def test_missing_artifact_hermetic_returns_0(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)

    def raise_fnf(prefix, days, trials):
        raise FileNotFoundError('no model')
    monkeypatch.setattr(backtest, 'run_backtest', raise_fnf)
    monkeypatch.setattr(sys, 'argv', ['backtest.py', '--gate'])

    assert backtest.main() == 0
    assert 'nothing to gate' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 23. Chunk-row byte budget
# ---------------------------------------------------------------------------

def test_chunk_rows_budget():
    assert backtest._pred_chunk_rows(40, 200) == 1024
    assert backtest._pred_chunk_rows(64, 300) == 64_000_000 // (64 * 300 * 4)
    assert backtest._pred_chunk_rows(1000, 1000) == 64


# ---------------------------------------------------------------------------
# 24. Manifest provenance + newer-challenger warning
# ---------------------------------------------------------------------------

def test_manifest_provenance_and_challenger_warning(monkeypatch, tmp_path, capsys):
    frames = [_one_ticker_frame('BTC')]
    run = _harness(monkeypatch, tmp_path, frames)

    (tmp_path / 'model_v2.manifest.json').write_text(
        json.dumps({'saved_at': '2026-07-01T00:00:00', 'score': 1.0}))
    (tmp_path / 'challenger_model_v2.manifest.json').write_text(
        json.dumps({'saved_at': '2026-07-20T00:00:00', 'score': 1.2}))
    champ_stat = (tmp_path / 'model_v2.manifest.json').stat()
    os.utime(tmp_path / 'challenger_model_v2.manifest.json',
             (champ_stat.st_mtime + 10, champ_stat.st_mtime + 10))

    metrics = run()
    assert metrics['artifact_manifest_saved_at'] == '2026-07-01T00:00:00'
    assert 'NEWER challenger manifest' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 25. policy_values present
# ---------------------------------------------------------------------------

def test_policy_values_present(monkeypatch, tmp_path):
    frames = [_one_ticker_frame('BTC')]
    run = _harness(monkeypatch, tmp_path, frames)
    metrics = run()
    pv = metrics['policy_values']
    assert 'threshold' in pv
    assert 'spread_pct_flat' in pv
    assert 'policy' in pv
    assert 'cooldown_min' in pv['policy']
