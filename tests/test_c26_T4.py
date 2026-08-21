"""Packet T4 (campaign 2026-08, B02): Stage-0 predictions dump + hourly MTM
equity + blend-leg persistence.

A. stage0_preds pure kernels (numpy/pandas only — Mac-testable).
B. Consumer-contract integration: the dump feeds ic_diagnostic.ic_by_name
   (default keys) and portfolio_backtest.panel_from_frame ('signal'/'symbol')
   with zero flags.
C. backtest.run_backtest wiring, via test_backtest_v3's monkeypatched-seam
   harness (backtest.py's top level is heavy-dep-free).
D. predict_now.py source pins (torch-gated module — source text only).

The (preds, q10) return arity of backtest._predict_ticker is pinned by
tests/test_review_b14 (run alongside this file); leg capture rides the
optional legs_out out-param, and run_backtest inspects the live function for
that parameter so legacy monkeypatched fakes keep working.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest
import stage0_preds


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _hourly_index(n, start='2026-01-01', tz='UTC'):
    return pd.date_range(start, periods=n, freq='h', tz=tz)


def _one_ticker_frame(ticker='BTC', n=40, seed=0):
    """Harvest-shaped hourly frame matching test_backtest_v3's harness."""
    rng = np.random.default_rng(seed)
    idx = _hourly_index(n)
    price = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame({
        'Ticker': ticker, 'Open': price, 'High': price * 1.01,
        'Low': price * 0.99, 'Close': price, 'ATR': np.nan,
        'f1': np.linspace(0, 1, n), 'f2': np.linspace(1, 0, n),
    }, index=idx)


def _legacy_fake_predict(actionable_index=20, pred_value=5.0):
    """Fake with the EXACT legacy signature (no legs_out) — mirrors the
    fakes in test_backtest_v3 / test_gate_protocol / test_c26_Q2."""
    def fake(model, scaler, config, feature_cols, tdf, lgb_model=None,
             q10_model=None):
        n = len(tdf)
        preds = np.full(n, np.nan)
        if n > actionable_index + 1:
            preds[actionable_index] = pred_value
        return preds, None
    return fake


def _legs_fake_predict(actionable_index=20, pred_value=5.0,
                       lstm_val=1.25, lgb_val=2.5):
    def fake(model, scaler, config, feature_cols, tdf, lgb_model=None,
             q10_model=None, legs_out=None):
        n = len(tdf)
        preds = np.full(n, np.nan)
        if n > actionable_index + 1:
            preds[actionable_index] = pred_value
        if legs_out is not None:
            legs_out['lstm'] = np.full(n, lstm_val, dtype=np.float64)
            legs_out['lgb'] = np.full(n, lgb_val, dtype=np.float64)
        return preds, None
    return fake


def _harness(monkeypatch, tmp_path, frames, fake_predict=None, config=None):
    """Wire run_backtest's heavy seams (test_backtest_v3 pattern); returns a
    callable forwarding **kwargs to run_backtest."""
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(
        backtest, '_load_artifacts',
        lambda prefix: (None, None,
                        config or {'seq_len': 5, 'trade_threshold': 0.05},
                        ['f1', 'f2']))
    monkeypatch.setattr(backtest, '_load_lgb', lambda prefix: None)
    monkeypatch.setattr(backtest, '_load_q10', lambda prefix: None)
    monkeypatch.setattr(backtest, '_predict_ticker',
                        fake_predict or _legacy_fake_predict())
    combined = pd.concat(frames)
    monkeypatch.setattr('data_utils.load_training_data',
                        lambda asset: combined)

    def _run(**kwargs):
        return backtest.run_backtest(prefix='', days=400,
                                     n_search_trials=10, **kwargs)
    return _run


# ---------------------------------------------------------------------------
# A1. select_row_indices
# ---------------------------------------------------------------------------

def test_select_spacing_and_window():
    n, h = 40, 4
    t_ns = stage0_preds.index_ns(_hourly_index(n))
    preds = np.random.default_rng(0).normal(0, 1, n)
    idx = stage0_preds.select_row_indices(t_ns, preds, h)
    assert idx, 'expected selections on all-finite preds'
    diffs = np.diff(idx)
    assert (diffs >= h).all()
    assert all(i + h <= n - 1 for i in idx)


def test_select_skips_nan_and_keeps_spacing():
    n, h = 30, 3
    t_ns = stage0_preds.index_ns(_hourly_index(n))
    preds = np.full(n, np.nan)
    preds[[2, 4, 10, 11, 20]] = 1.0
    idx = stage0_preds.select_row_indices(t_ns, preds, h)
    assert all(np.isfinite(preds[i]) for i in idx)
    assert (np.diff(idx) >= h).all() if len(idx) > 1 else True
    assert 2 in idx          # first finite pred
    assert 4 not in idx      # within h of 2


def test_select_degenerate():
    t_ns = stage0_preds.index_ns(_hourly_index(5))
    preds = np.ones(5)
    assert stage0_preds.select_row_indices(t_ns, preds, 5) == []
    assert stage0_preds.select_row_indices(np.array([], dtype=np.int64),
                                           np.array([]), 4) == []


# ---------------------------------------------------------------------------
# A2. Cross-name anchor alignment
# ---------------------------------------------------------------------------

def test_anchor_aligns_two_symbols():
    n, h = 32, 4
    idx_a = _hourly_index(n)
    t_ns = stage0_preds.index_ns(idx_a)
    anchor = stage0_preds.global_anchor_ns(t_ns, h)
    rng = np.random.default_rng(1)
    sel_a = stage0_preds.select_row_indices(t_ns, rng.normal(size=n), h,
                                            anchor_ns=anchor)
    # Second symbol: same bar grid, its own preds (first bar NaN — without
    # the anchor its rows would start on a shifted grid).
    preds_b = rng.normal(size=n)
    preds_b[:3] = np.nan
    sel_b = stage0_preds.select_row_indices(t_ns, preds_b, h,
                                            anchor_ns=anchor)
    ts_a = {t_ns[i] for i in sel_a}
    ts_b = {t_ns[i] for i in sel_b}
    assert ts_b <= ts_a  # every B row shares an A timestamp (panel periods)
    assert len(ts_b) > 1
    # all selected bars sit on the anchor grid
    assert ts_a <= set(anchor.tolist())


# ---------------------------------------------------------------------------
# A3. build_rows
# ---------------------------------------------------------------------------

def test_build_rows_schema_and_units():
    n, h = 20, 4
    idx_t = _hourly_index(n)
    closes = np.linspace(100.0, 119.0, n)
    preds = np.full(n, 0.5)
    lstm = np.full(n, 0.7)
    lgb = np.full(n, 0.2)
    meta = np.full(n, 0.61)
    q10 = np.full(n, -1.2)
    sel = [0, 4, 8]
    rows = stage0_preds.build_rows(idx_t, 'BTC', preds, closes, h, sel,
                                   lstm=lstm, lgb=lgb, meta_probs=meta,
                                   q10=q10, threshold=0.25)
    assert len(rows) == 3
    for k, i in zip(rows, sel):
        expected_fwd = (closes[i + h] - closes[i]) / closes[i] * 100.0
        assert k['fwd_return'] == pytest.approx(expected_fwd, abs=1e-6)
        assert k['pred'] == k['signal'] == pytest.approx(0.5)
        assert k['symbol'] == 'BTC'
        assert k['ts'] == str(idx_t[i])
        assert k['horizon_bars'] == h
        assert k['lstm_pred'] == pytest.approx(0.7)
        assert k['lgb_pred'] == pytest.approx(0.2)
        assert k['meta_p'] == pytest.approx(0.61)
        assert k['q10'] == pytest.approx(-1.2)
        assert k['pred_thresh_ratio'] == pytest.approx(0.5 / 0.25)


def test_build_rows_none_handling():
    n, h = 12, 2
    idx_t = _hourly_index(n)
    closes = np.linspace(50.0, 61.0, n)
    preds = np.full(n, 1.0)
    lstm = np.full(n, np.nan)  # provided but NaN -> None
    rows = stage0_preds.build_rows(idx_t, 'ETH', preds, closes, h, [0, 2],
                                   lstm=lstm)
    for r in rows:
        assert r['lstm_pred'] is None       # NaN leg
        assert r['lgb_pred'] is None        # absent array
        assert r['meta_p'] is None
        assert r['q10'] is None
        assert r['pred_thresh_ratio'] is None  # no threshold


def test_build_rows_skips_bad_close():
    n, h = 10, 2
    idx_t = _hourly_index(n)
    closes = np.linspace(10.0, 19.0, n)
    closes[0] = 0.0
    closes[4] = np.nan
    preds = np.ones(n)
    rows = stage0_preds.build_rows(idx_t, 'X', preds, closes, h, [0, 2, 4, 6])
    # i=0: zero entry close; i=2: fwd close (i+h=4) is NaN; i=4: NaN entry
    # close — all skipped, only i=6 survives
    assert [r['ts'] for r in rows] == [str(idx_t[6])]


# ---------------------------------------------------------------------------
# A4. write_rows
# ---------------------------------------------------------------------------

def _tiny_rows():
    idx_t = _hourly_index(8)
    return stage0_preds.build_rows(idx_t, 'BTC', np.ones(8),
                                   np.linspace(100, 107, 8), 2, [0, 2, 4])


def test_write_rows_json_bare_list(tmp_path):
    rows = _tiny_rows()
    path = tmp_path / 'dump.json'
    stage0_preds.write_rows(rows, path)
    loaded = json.loads(path.read_text())
    assert isinstance(loaded, list)  # ic_by_name's contract: a BARE list
    assert loaded == rows
    assert not (tmp_path / 'dump.json.tmp').exists()


def test_write_rows_tmp_cleaned_on_failure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('disk fell over')
    monkeypatch.setattr(stage0_preds.json, 'dump', boom)
    path = tmp_path / 'dump.json'
    with pytest.raises(RuntimeError):
        stage0_preds.write_rows(_tiny_rows(), path)
    assert not (tmp_path / 'dump.json.tmp').exists()
    assert not path.exists()


def test_write_rows_csv_roundtrip(tmp_path):
    rows = _tiny_rows()
    path = tmp_path / 'dump.csv'
    stage0_preds.write_rows(rows, path)
    with open(path, newline='') as f:
        back = list(csv.DictReader(f))
    assert len(back) == len(rows)
    assert back[0]['symbol'] == 'BTC'
    assert float(back[0]['fwd_return']) == pytest.approx(
        rows[0]['fwd_return'])
    # None -> restval '' in CSV
    assert back[0]['lstm_pred'] == ''


# ---------------------------------------------------------------------------
# A5. mtm_equity
# ---------------------------------------------------------------------------

def test_mtm_equity_hand_built():
    idx_t = _hourly_index(10)
    grid = stage0_preds.index_ns(idx_t)
    closes = np.array([100.0, 102.0, 98.0, 104.0, 103.0,
                       103.0, 103.0, 103.0, 103.0, 103.0])
    trades = [{'ticker': 'AAA', 'entry_time': str(idx_t[1]),
               'exit_time': str(idx_t[4]), 'entry': 100.0, 'net_pct': 3.5}]
    out = stage0_preds.mtm_equity(trades, {'AAA': (grid, closes)}, grid)
    eq = out['equity_pct']
    assert out['n_marks'] == 10
    assert out['n_unmarked_trades'] == 0
    assert eq[0] == pytest.approx(0.0)
    assert eq[1] == pytest.approx(2.0)    # marked at 102 gross
    assert eq[2] == pytest.approx(-2.0)   # marked at 98 gross
    assert eq[3] == pytest.approx(4.0)    # marked at 104 gross
    for v in eq[4:]:
        assert v == pytest.approx(3.5)    # closed: net_pct (cost at exit)
    # Invariant: full coverage + all closed by window end -> final == sum net
    assert eq[-1] == pytest.approx(sum(t['net_pct'] for t in trades))
    # Drawdown on the down-then-up path: peak 2.0 at t1 -> -2.0 at t2
    assert out['max_drawdown_pct'] == pytest.approx(4.0)


def test_mtm_equity_empty_and_unknown_ticker():
    idx_t = _hourly_index(6)
    grid = stage0_preds.index_ns(idx_t)
    out = stage0_preds.mtm_equity([], {}, grid)
    assert out['equity_pct'] == [0.0] * 6
    assert out['max_drawdown_pct'] == 0.0
    assert out['n_unmarked_trades'] == 0

    trades = [{'ticker': 'ZZZ', 'entry_time': str(idx_t[1]),
               'exit_time': str(idx_t[3]), 'entry': 50.0, 'net_pct': -1.0}]
    out2 = stage0_preds.mtm_equity(trades, {}, grid)
    assert out2['n_unmarked_trades'] == 1
    # closed step still contributes from the exit mark on
    assert out2['equity_pct'][2] == pytest.approx(0.0)
    assert out2['equity_pct'][3] == pytest.approx(-1.0)
    assert out2['equity_pct'][-1] == pytest.approx(-1.0)


def test_max_drawdown_from_equity():
    assert stage0_preds.max_drawdown_from_equity([]) == 0.0
    # seeded with 0.0: an all-negative path draws down from zero
    assert stage0_preds.max_drawdown_from_equity([-1.0, -3.0]) == 3.0
    assert stage0_preds.max_drawdown_from_equity([1.0, 4.0, 2.0, 5.0]) == 2.0


# ---------------------------------------------------------------------------
# B. Consumer contracts (pure kernels of both Stage-0 consumers)
# ---------------------------------------------------------------------------

def _two_symbol_dump(tmp_path):
    n, h = 40, 4
    idx_t = _hourly_index(n)
    t_ns = stage0_preds.index_ns(idx_t)
    anchor = stage0_preds.global_anchor_ns(t_ns, h)
    rng = np.random.default_rng(7)
    rows = []
    for sym, seed in (('BTC', 1), ('ETH', 2)):
        preds = rng.normal(0, 1, n)
        closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
        sel = stage0_preds.select_row_indices(t_ns, preds, h,
                                              anchor_ns=anchor)
        rows.extend(stage0_preds.build_rows(
            idx_t, sym, preds, closes, h, sel,
            meta_probs=np.full(n, 0.6), threshold=0.15))
    path = tmp_path / 'stage0_preds.json'
    stage0_preds.write_rows(rows, path)
    return path


def test_consumer_ic_by_name_default_keys(tmp_path):
    from ic_diagnostic import ic_by_name
    rows = json.loads(_two_symbol_dump(tmp_path).read_text())
    table = ic_by_name(rows)  # default keys: symbol/pred/fwd_return
    assert set(table) == {'BTC', 'ETH'}
    for name in table:
        assert table[name]['n_finite'] > 0
        assert np.isfinite(table[name]['ic'])


def test_consumer_panel_from_frame(tmp_path):
    from portfolio_backtest import panel_from_frame
    rows = json.loads(_two_symbol_dump(tmp_path).read_text())
    df = pd.DataFrame(rows)
    df = df.set_index(pd.to_datetime(df['ts']))
    stats = {}
    panel = panel_from_frame(df, 'signal', 'fwd_return',
                             ticker_col='symbol',
                             extra_cols=['meta_p', 'pred_thresh_ratio'],
                             stats_out=stats)  # must NOT raise (no dupes)
    assert len(panel) > 0
    # anchored rows: cross-name periods carry BOTH names, not 1 candidate
    assert max(len(p) for p in panel) == 2
    assert stats['n_periods'] == len(panel)


# ---------------------------------------------------------------------------
# C. backtest.run_backtest wiring
# ---------------------------------------------------------------------------

def test_run_backtest_default_on_dump_and_mtm(monkeypatch, tmp_path):
    frames = [_one_ticker_frame('BTC'), _one_ticker_frame('ETH', seed=1)]
    run = _harness(monkeypatch, tmp_path, frames)
    metrics = run()

    dump = tmp_path / 'stage0_preds.json'
    assert dump.exists()
    rows = json.loads(dump.read_text())
    assert isinstance(rows, list) and rows
    assert {r['symbol'] for r in rows} == {'BTC', 'ETH'}
    for r in rows:
        assert r['pred'] == r['signal'] == pytest.approx(5.0)
        assert r['horizon_bars'] == 4  # config lacks forward_bars -> 4
        assert r['lstm_pred'] is None  # legacy fake: no legs captured
        assert r['lgb_pred'] is None

    assert 'mtm_max_drawdown_pct' in metrics
    s0 = metrics['stage0_dump']
    assert s0['n_rows'] == len(rows)
    assert s0['non_overlapping'] is True
    assert s0['units'] == 'percent'
    assert s0['horizon_bars'] == 4
    assert s0['path'] == 'stage0_preds.json'

    report = json.loads((tmp_path / 'backtest_report.json').read_text())
    # pre-existing report keys survive
    for k in ('metrics', 'n_trades_total', 'n_trades_persisted', 'trades'):
        assert k in report
    mtm = report['mtm_equity_hourly']
    assert len(mtm['ts']) == len(mtm['equity_pct']) > 0


def test_run_backtest_legs_fake_lands_in_rows(monkeypatch, tmp_path):
    frames = [_one_ticker_frame('BTC')]
    run = _harness(monkeypatch, tmp_path, frames,
                   fake_predict=_legs_fake_predict())
    run()
    rows = json.loads((tmp_path / 'stage0_preds.json').read_text())
    assert rows
    for r in rows:
        assert r['lstm_pred'] == pytest.approx(1.25)
        assert r['lgb_pred'] == pytest.approx(2.5)


def _assert_no_stage0(tmp_path, metrics):
    assert not (tmp_path / 'stage0_preds.json').exists()
    assert 'mtm_max_drawdown_pct' not in metrics
    assert 'stage0_dump' not in metrics
    report = json.loads((tmp_path / 'backtest_report.json').read_text())
    assert 'mtm_equity_hourly' not in report
    assert set(report) == {'metrics', 'n_trades_total',
                           'n_trades_persisted', 'trades'}


def test_run_backtest_kwarg_off(monkeypatch, tmp_path):
    run = _harness(monkeypatch, tmp_path, [_one_ticker_frame('BTC')])
    metrics = run(stage0_dump=False)
    _assert_no_stage0(tmp_path, metrics)


def test_run_backtest_module_default_off(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest, 'STAGE0_DUMP_DEFAULT', False)
    run = _harness(monkeypatch, tmp_path, [_one_ticker_frame('BTC')])
    metrics = run()  # stage0_dump=None -> module default
    _assert_no_stage0(tmp_path, metrics)


def test_run_backtest_dump_failure_is_fail_soft(monkeypatch, tmp_path,
                                                capsys):
    run = _harness(monkeypatch, tmp_path, [_one_ticker_frame('BTC')])

    def boom(rows, path):
        raise RuntimeError('dump exploded')
    monkeypatch.setattr('stage0_preds.write_rows', boom)

    metrics = run()  # must not raise
    out = capsys.readouterr().out
    assert 'stage0 dump/MTM unavailable' in out
    assert 'sharpe' in metrics
    report = json.loads((tmp_path / 'backtest_report.json').read_text())
    assert 'metrics' in report
    assert 'mtm_equity_hourly' not in report


def test_run_backtest_challenger_slot_dump_name(monkeypatch, tmp_path):
    """Challenger-targeted runs write challenger_stage0_preds.json — never
    clobbering the champion dump (mirrors _report_slot's report naming)."""
    run = _harness(monkeypatch, tmp_path, [_one_ticker_frame('BTC')])
    metrics = run(model_prefix='challenger')
    assert (tmp_path / 'challenger_stage0_preds.json').exists()
    assert not (tmp_path / 'stage0_preds.json').exists()
    assert metrics['stage0_dump']['path'] == 'challenger_stage0_preds.json'
    report = json.loads(
        (tmp_path / 'backtest_challenger_report.json').read_text())
    assert 'mtm_equity_hourly' in report


def test_run_backtest_kwarg_on_overrides_module_default(monkeypatch,
                                                        tmp_path):
    monkeypatch.setattr(backtest, 'STAGE0_DUMP_DEFAULT', False)
    run = _harness(monkeypatch, tmp_path, [_one_ticker_frame('BTC')])
    metrics = run(stage0_dump=True)  # explicit kwarg beats the module default
    assert (tmp_path / 'stage0_preds.json').exists()
    assert 'stage0_dump' in metrics


def test_predict_ticker_real_fn_advertises_legs_out():
    """The real _predict_ticker must expose legs_out (the inspect guard's
    positive branch) while keeping its pinned (preds, q10) return arity —
    the arity itself is pinned by tests/test_review_b14, run alongside."""
    import inspect
    params = inspect.signature(backtest._predict_ticker).parameters
    assert 'legs_out' in params
    assert params['legs_out'].default is None


def test_main_cli_no_stage0_dump_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(backtest, 'STAGE0_DUMP_DEFAULT', True)
    calls = []

    def fake_run(prefix, days, trials):  # pinned 3-positional shape
        calls.append((prefix, days, trials))
        return {'n_trades': 20, 'sharpe': 1.0, 'dsr': 0.9}
    monkeypatch.setattr(backtest, 'run_backtest', fake_run)

    monkeypatch.setattr(sys, 'argv',
                        ['backtest.py', '--no-stage0-dump', '--trials', '10'])
    assert backtest.main() == 0
    assert backtest.STAGE0_DUMP_DEFAULT is False
    assert len(calls) == 1  # the 3-arg call shape was untouched

    # without the flag the default stays ON
    monkeypatch.setattr(backtest, 'STAGE0_DUMP_DEFAULT', True)
    monkeypatch.setattr(sys, 'argv', ['backtest.py', '--trials', '10'])
    assert backtest.main() == 0
    assert backtest.STAGE0_DUMP_DEFAULT is True


# ---------------------------------------------------------------------------
# D. predict_now source pins (torch at module top -> source text only)
# ---------------------------------------------------------------------------

def test_predict_now_blend_leg_source_pins():
    src = (Path(__file__).resolve().parent.parent
           / 'predict_now.py').read_text()
    # leg initialized before the booster branch so it is None when the
    # booster is absent or predict_lgb raised before assignment
    i_init = src.index('lgb_pred = None')
    i_branch = src.index('if lgb_model is not None:')
    assert i_init < i_branch
    # snapshot keys assigned inside the snapshot branch, before the cache put
    i_lstm = src.index("snapshot['LSTM_Pred']")
    i_lgb = src.index("snapshot['LGB_Pred']")
    i_put = src.index('_PRED_CACHE.put(_cache_subkey, _cache_key, '
                      '(predicted_return, snapshot))')
    assert i_lstm < i_put
    assert i_lgb < i_put
    # fail-soft: the assignments sit in a try block
    guard = src[src.index('Blend legs (B02'):i_put]
    assert 'try:' in guard and 'except Exception:' in guard
