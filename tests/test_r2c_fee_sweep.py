"""R2C-08 (FR-16): breakeven fee-multiplier sweep on the policy replay.

Covers the packet gate:
  - default-path byte-identity: no fee_mult -> trade dicts and metrics/report
    key sets unchanged; explicit fee_mult=1.0 -> identical trade values plus
    the self-describing keys;
  - simulate_ticker fee_mult on synthetic OHLC via the pure policy_exits
    fallback: only the CHARGED cost legs scale, entries stay fixed;
  - breakeven_fee_mult linear zero-crossing interpolation unit tests;
  - main() refuses --gate combined with --fee-mult/--fee-sweep (ap.error,
    exit 2, nothing runs), and threads the flags report-only otherwise.

Follows tests/test_backtest_v3.py's monkeypatch-the-heavy-seams pattern:
backtest.py's own top level imports only numpy/fees/strategy_config/
validation, so no importorskip is needed on the dev Mac.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest
from strategy_config import policy_for


# ---------------------------------------------------------------------------
# Shared fixtures / helpers (mirroring tests/test_backtest_v3.py)
# ---------------------------------------------------------------------------

def _trending_frame(n=60, drift=0.004, seed=0):
    """Gently rising bars so a long signal exits profitably (gross > 0)."""
    rng = np.random.RandomState(seed)
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.001, n))
    high = close * 1.002
    low = close * 0.998
    op = close * 0.999
    atr = pd.Series(close).rolling(5, min_periods=1).std().fillna(0.5).values + 0.5
    idx = pd.date_range('2025-03-03 14:00', periods=n, freq='h', tz='UTC')
    return pd.DataFrame({'Open': op, 'High': high, 'Low': low,
                         'Close': close, 'ATR': atr}, index=idx)


def _run_sim(tdf, threshold=0.1, **kw):
    preds = np.full(len(tdf), 0.5)  # strong, constant long signal
    return backtest.simulate_ticker(tdf, preds, 'stock', threshold,
                                    policy_for('stock'), **kw)


def _trending_ticker_frame(ticker='BTC', n=60, drift=0.004, seed=0):
    """Harvest-shaped hourly crypto frame (rising, so gross > 0) with the
    two harness feature cols."""
    tdf = _trending_frame(n=n, drift=drift, seed=seed).copy()
    tdf.index = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
    tdf['Ticker'] = ticker
    tdf['f1'] = np.linspace(0, 1, n)
    tdf['f2'] = np.linspace(1, 0, n)
    return tdf


def _fake_predict_ticker_factory(actionable_index=20, pred_value=5.0):
    def fake(model, scaler, config, feature_cols, tdf, lgb_model=None,
             q10_model=None):
        n = len(tdf)
        preds = np.full(n, np.nan)
        if n > actionable_index + 1:
            preds[actionable_index] = pred_value
        return preds, None
    return fake


def _harness(monkeypatch, tmp_path, frames):
    """Wire run_backtest's heavy seams to synthetic frames."""
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(
        backtest, '_load_artifacts',
        lambda prefix: (None, None,
                        {'seq_len': 5, 'trade_threshold': 0.05},
                        ['f1', 'f2']))
    monkeypatch.setattr(backtest, '_load_lgb', lambda prefix: None)
    monkeypatch.setattr(backtest, '_load_q10', lambda prefix: None)
    monkeypatch.setattr(backtest, '_predict_ticker',
                        _fake_predict_ticker_factory())
    combined = pd.concat(frames)
    monkeypatch.setattr('data_utils.load_training_data',
                        lambda asset: combined)


# ---------------------------------------------------------------------------
# 1. breakeven_fee_mult — linear zero-crossing interpolation
# ---------------------------------------------------------------------------

def test_lambda_crossing_interpolated():
    lam, status = backtest.breakeven_fee_mult([1.0, 2.0, 3.0], [4.0, 2.0, -2.0])
    assert status == 'crossed'
    # crossing between 2 and 3: 2 + 1 * 2/(2 - (-2)) = 2.5
    assert lam == pytest.approx(2.5)


def test_lambda_exact_linear_recovery():
    # net = gross - lam * cost is LINEAR in lam (entries/exits fixed), so a
    # 2-point grid recovers lam* = gross/cost exactly.
    gross, cost = 7.3, 2.9
    mults = [1.0, 6.0]
    nets = [gross - m * cost for m in mults]
    lam, status = backtest.breakeven_fee_mult(mults, nets)
    assert status == 'crossed'
    assert lam == pytest.approx(gross / cost, rel=1e-12)


def test_lambda_zero_at_grid_point():
    lam, status = backtest.breakeven_fee_mult([1.0, 2.0, 3.0], [4.0, 0.0, -4.0])
    assert status == 'crossed'
    assert lam == pytest.approx(2.0)


def test_lambda_below_at_min():
    lam, status = backtest.breakeven_fee_mult([1.0, 2.0], [-0.5, -3.0])
    assert status == 'below_at_min'
    assert lam == 1.0


def test_lambda_no_cross():
    lam, status = backtest.breakeven_fee_mult([1.0, 2.0, 6.0], [9.0, 7.0, 1.0])
    assert status == 'no_cross'
    assert lam is None


def test_lambda_input_validation():
    with pytest.raises(ValueError):
        backtest.breakeven_fee_mult([], [])
    with pytest.raises(ValueError):
        backtest.breakeven_fee_mult([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        backtest.breakeven_fee_mult([2.0, 1.0], [1.0, -1.0])  # not increasing
    with pytest.raises(ValueError):
        backtest.breakeven_fee_mult([1.0, 1.0], [1.0, -1.0])  # not strict
    with pytest.raises(ValueError):
        backtest.breakeven_fee_mult([1.0, 2.0], [np.nan, -1.0])


# ---------------------------------------------------------------------------
# 2. simulate_ticker fee_mult (synthetic OHLC, pure policy_exits fallback)
# ---------------------------------------------------------------------------

def test_sim_default_byte_identity_with_explicit_one():
    tdf = _trending_frame()
    baseline = _run_sim(tdf)
    explicit = _run_sim(tdf, fee_mult=1.0)
    assert baseline  # non-degenerate
    assert explicit == baseline  # full dict equality, every field


def test_sim_fee_mult_scales_only_charged_cost():
    from fees import round_trip_cost_pct
    tdf = _trending_frame()
    flat = round_trip_cost_pct('stock', backtest.SPREAD_PCT['stock'])

    base = _run_sim(tdf)
    stressed = _run_sim(tdf, fee_mult=2.0)
    assert base and stressed
    # Entries/exits fixed: identical trade skeletons...
    assert [t['entry_time'] for t in stressed] == [t['entry_time'] for t in base]
    assert [t['exit_time'] for t in stressed] == [t['exit_time'] for t in base]
    assert [t['gross_pct'] for t in stressed] == [t['gross_pct'] for t in base]
    # ...only the charged cost doubles.
    for t in stressed:
        assert t['net_pct'] == pytest.approx(t['gross_pct'] - 2.0 * flat,
                                             abs=1e-4)


def test_sim_fee_mult_scales_per_bar_cost_leg():
    # Per-bar Eff_Spread_Pct path: cost_i at mult m equals m * cost_i at 1,
    # verified via the paired-run identity net_m = gross - m * (gross - net_1).
    tdf = _trending_frame().copy()
    tdf['Eff_Spread_Pct'] = np.linspace(0.05, 0.9, len(tdf))
    base = _run_sim(tdf)
    stressed = _run_sim(tdf, fee_mult=3.0)
    assert base and stressed
    assert len(base) == len(stressed)
    for b, s in zip(base, stressed):
        cost_1 = b['gross_pct'] - b['net_pct']
        assert s['net_pct'] == pytest.approx(s['gross_pct'] - 3.0 * cost_1,
                                             abs=5e-4)


def test_sim_entries_fixed_under_extreme_stress():
    # A ruinous multiplier must not change ADMISSION (edge_floor unscaled):
    # same trade count/entries, nets simply go deeply negative.
    tdf = _trending_frame()
    base = _run_sim(tdf)
    stressed = _run_sim(tdf, fee_mult=50.0)
    assert len(stressed) == len(base)
    assert [t['entry_time'] for t in stressed] == [t['entry_time'] for t in base]
    assert all(t['net_pct'] < 0 for t in stressed)


def test_sim_fee_mult_keyword_only():
    tdf = _trending_frame()
    preds = np.full(len(tdf), 0.5)
    with pytest.raises(TypeError):
        backtest.simulate_ticker(tdf, preds, 'stock', 0.1, policy_for('stock'),
                                 None, None, None, 2.0)


# ---------------------------------------------------------------------------
# 3. run_backtest threading + D25 report-key convention
# ---------------------------------------------------------------------------

def test_run_backtest_default_has_no_fee_keys(monkeypatch, tmp_path):
    _harness(monkeypatch, tmp_path, [_trending_ticker_frame('BTC')])
    metrics = backtest.run_backtest(prefix='', days=400, n_search_trials=10)
    assert 'fee_mult' not in metrics
    assert 'net_pct_by_name' not in metrics
    report = json.loads((tmp_path / 'backtest_report.json').read_text())
    assert 'fee_mult' not in report['metrics']
    assert 'net_pct_by_name' not in report['metrics']


def test_run_backtest_explicit_one_matches_default(monkeypatch, tmp_path):
    _harness(monkeypatch, tmp_path, [_trending_ticker_frame('BTC')])
    default = backtest.run_backtest(prefix='', days=400, n_search_trials=10)
    stressed = backtest.run_backtest(prefix='', days=400, n_search_trials=10,
                                     fee_mult=1.0)
    # Extra self-describing keys ONLY; every shared value identical.
    assert stressed['fee_mult'] == 1.0
    extra = set(stressed) - set(default)
    assert extra == {'fee_mult', 'net_pct_by_name'}
    for k in default:
        if k != 'generated_at':  # wall-clock stamp
            assert stressed[k] == default[k], k


def test_run_backtest_fee_mult_per_name_totals(monkeypatch, tmp_path):
    frames = [_trending_ticker_frame('BTC'),
              _trending_ticker_frame('ETH', seed=1)]
    _harness(monkeypatch, tmp_path, frames)
    m = backtest.run_backtest(prefix='', days=400, n_search_trials=10,
                              fee_mult=2.0)
    assert m['fee_mult'] == 2.0
    assert set(m['net_pct_by_name']) == {'BTC', 'ETH'}
    assert sum(m['net_pct_by_name'].values()) == pytest.approx(
        m['net_total_pct'], abs=0.02)


def test_run_backtest_stress_report_never_clobbers_book_report(
        monkeypatch, tmp_path):
    # FR-16 hardening: a stressed replay writes its OWN self-describing
    # backtest_stress_report.json; the operational book report the ops
    # surfaces read must stay exactly what the live-cost run wrote.
    _harness(monkeypatch, tmp_path, [_trending_ticker_frame('BTC')])
    backtest.run_backtest(prefix='', days=400, n_search_trials=10)
    book_before = (tmp_path / 'backtest_report.json').read_bytes()

    backtest.run_backtest(prefix='', days=400, n_search_trials=10,
                          fee_mult=3.0)
    # book report byte-identical; stressed report exists and is marked
    assert (tmp_path / 'backtest_report.json').read_bytes() == book_before
    stress = json.loads(
        (tmp_path / 'backtest_stress_report.json').read_text())
    assert stress['metrics']['fee_mult'] == 3.0
    assert 'net_pct_by_name' in stress['metrics']


def test_run_backtest_stress_report_only_file_written(monkeypatch, tmp_path):
    # A stressed run on a slot with NO prior book report must not create one.
    _harness(monkeypatch, tmp_path, [_trending_ticker_frame('BTC')])
    backtest.run_backtest(prefix='', days=400, n_search_trials=10,
                          fee_mult=2.0)
    assert not (tmp_path / 'backtest_report.json').exists()
    assert (tmp_path / 'backtest_stress_report.json').exists()


def test_run_backtest_bad_fee_mult_raises(monkeypatch, tmp_path):
    _harness(monkeypatch, tmp_path, [_trending_ticker_frame('BTC')])
    for bad in (0.0, -2.0, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            backtest.run_backtest(prefix='', days=400, n_search_trials=10,
                                  fee_mult=bad)


# ---------------------------------------------------------------------------
# 4. main(): --gate refusal (ap.error, exit 2, nothing runs)
# ---------------------------------------------------------------------------

def _refusal_recorders(monkeypatch):
    ran, restored = [], []
    monkeypatch.setattr(backtest, 'run_backtest',
                        lambda *a, **k: ran.append((a, k)) or {})
    monkeypatch.setattr(backtest, 'restore_previous_model',
                        lambda prefix: restored.append(prefix))
    return ran, restored


@pytest.mark.parametrize('argv', [
    ['backtest.py', '--gate', '--fee-mult', '2'],
    ['backtest.py', '--gate', '--fee-sweep', '1.0,1.5,2,3,4,6'],
    ['backtest.py', '--fee-mult', '2', '--fee-sweep', '1,2'],  # exclusive
    ['backtest.py', '--fee-mult', '0'],
    ['backtest.py', '--fee-mult', '-1.5'],
    ['backtest.py', '--fee-mult', 'nan'],
    ['backtest.py', '--fee-sweep', '2.0'],       # < 2 distinct points
    ['backtest.py', '--fee-sweep', '1,junk'],    # unparseable
    ['backtest.py', '--fee-sweep', '1,-2'],      # non-positive
])
def test_main_refuses_bad_fee_flags(monkeypatch, argv):
    ran, restored = _refusal_recorders(monkeypatch)
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as exc_info:
        backtest.main()
    assert exc_info.value.code == 2  # argparse ap.error
    assert ran == []
    assert restored == []


def test_main_default_path_stays_three_positional(monkeypatch):
    seen = {}

    def fake_run(*a, **k):
        seen['args'], seen['kwargs'] = a, k
        return {'n_trades': 20, 'sharpe': 1.0, 'dsr': 0.9}
    monkeypatch.setattr(backtest, 'run_backtest', fake_run)
    monkeypatch.setattr(sys, 'argv', ['backtest.py'])
    assert backtest.main() == 0
    assert len(seen['args']) == 3       # the pinned 3-arg positional seam
    assert seen['kwargs'] == {}


def test_main_single_fee_mult_threaded(monkeypatch):
    seen = {}

    def fake_run(prefix, days, trials, *, fee_mult=None, stage0_dump=None,
                 model_prefix=None):
        seen['fee_mult'] = fee_mult
        return {'n_trades': 5, 'sharpe': 0.5, 'dsr': 0.4}
    monkeypatch.setattr(backtest, 'run_backtest', fake_run)
    monkeypatch.setattr(sys, 'argv', ['backtest.py', '--fee-mult', '2.5'])
    assert backtest.main() == 0
    assert seen['fee_mult'] == 2.5


# ---------------------------------------------------------------------------
# 5. main() --fee-sweep: loop protocol, lambda* report, sweep JSON
# ---------------------------------------------------------------------------

def test_main_fee_sweep_loop_and_lambda(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    calls = []

    def fake_run(prefix, days, trials, *, fee_mult=None, stage0_dump=None,
                 model_prefix=None):
        calls.append({'fee_mult': fee_mult, 'stage0_dump': stage0_dump,
                      'model_prefix': model_prefix})
        # Linear book/name nets: book lam*=10/4=2.5; BTC 3.0; ETH 2.0
        return {'n_trades': 8, 'sharpe': 1.0, 'dsr': 0.5,
                'net_total_pct': 10.0 - 4.0 * fee_mult,
                'net_pct_by_name': {'BTC': 6.0 - 2.0 * fee_mult,
                                    'ETH': 4.0 - 2.0 * fee_mult}}
    monkeypatch.setattr(backtest, 'run_backtest', fake_run)
    monkeypatch.setattr(sys, 'argv',
                        ['backtest.py', '--fee-sweep', '1,2,3,4',
                         '--days', '30'])
    assert backtest.main() == 0

    assert [c['fee_mult'] for c in calls] == [1.0, 2.0, 3.0, 4.0]
    # SIG-R2-2: NO sweep pass writes the stage0 dump (pass 1 used to keep
    # the module default and overwrite the weekly --gate dump)
    assert all(c['stage0_dump'] is False for c in calls)
    assert all(c['model_prefix'] is None for c in calls)

    out = capsys.readouterr().out
    assert 'FEE-MULT SWEEP' in out
    assert 'book lambda* = 2.50' in out
    assert 'BTC: 3.00' in out
    assert 'ETH: 2.00' in out

    payload = json.loads((tmp_path / 'backtest_fee_sweep.json').read_text())
    assert payload['mults'] == [1.0, 2.0, 3.0, 4.0]
    assert payload['book_lambda_star'] == pytest.approx(2.5)
    assert payload['book_status'] == 'crossed'
    assert payload['per_name']['BTC']['lambda_star'] == pytest.approx(3.0)
    assert payload['per_name']['ETH']['lambda_star'] == pytest.approx(2.0)


def test_main_fee_sweep_grid_sorted_and_deduped(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    calls = []

    def fake_run(prefix, days, trials, *, fee_mult=None, stage0_dump=None,
                 model_prefix=None):
        calls.append(fee_mult)
        return {'n_trades': 1, 'sharpe': 0.0, 'dsr': 0.0,
                'net_total_pct': 1.0, 'net_pct_by_name': {'X': 1.0}}
    monkeypatch.setattr(backtest, 'run_backtest', fake_run)
    monkeypatch.setattr(sys, 'argv',
                        ['backtest.py', '--fee-sweep', '3, 1.0, 3, 2'])
    assert backtest.main() == 0
    assert calls == [1.0, 2.0, 3.0]


def test_run_fee_sweep_threads_model_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    calls = []

    def fake_run(prefix, days, trials, *, fee_mult=None, stage0_dump=None,
                 model_prefix=None):
        calls.append(model_prefix)
        return {'n_trades': 1, 'sharpe': 0.0, 'dsr': 0.0,
                'net_total_pct': -1.0, 'net_pct_by_name': {'A': -1.0}}
    monkeypatch.setattr(backtest, 'run_backtest', fake_run)
    rc = backtest._run_fee_sweep('', 30, 10, [1.0, 2.0],
                                 model_prefix='challenger')
    assert rc == 0
    assert calls == ['challenger', 'challenger']
    # challenger-slot sweep report keeps its own identity (never clobbers
    # the champion book sweep file)
    payload = json.loads(
        (tmp_path / 'backtest_challenger_fee_sweep.json').read_text())
    assert payload['book_status'] == 'below_at_min'
    assert payload['book_lambda_star'] == 1.0


def test_main_fee_sweep_challenger_slot_dispatch(monkeypatch, tmp_path):
    # Pins main()'s dispatch line: --model-prefix <challenger> with its 4
    # core artifacts present threads model_prefix into every sweep pass and
    # the sweep JSON keeps the challenger identity.
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    for s in backtest.ARTIFACT_SUFFIXES[:4]:
        (tmp_path / f'challenger_{s}').write_bytes(b'x')
    calls = []

    def fake_run(prefix, days, trials, *, fee_mult=None, stage0_dump=None,
                 model_prefix=None):
        calls.append((prefix, model_prefix, fee_mult))
        return {'n_trades': 2, 'sharpe': 0.1, 'dsr': 0.1,
                'net_total_pct': 5.0 - 2.0 * fee_mult,
                'net_pct_by_name': {'BTC': 5.0 - 2.0 * fee_mult}}
    monkeypatch.setattr(backtest, 'run_backtest', fake_run)
    monkeypatch.setattr(sys, 'argv',
                        ['backtest.py', '--fee-sweep', '1,2,4',
                         '--model-prefix', 'challenger'])
    assert backtest.main() == 0
    assert [c[1] for c in calls] == ['challenger'] * 3
    assert [c[0] for c in calls] == [''] * 3  # book data stays the champion's
    payload = json.loads(
        (tmp_path / 'backtest_challenger_fee_sweep.json').read_text())
    assert payload['model_prefix'] == 'challenger'
    assert payload['book_lambda_star'] == pytest.approx(2.5)


def test_main_fee_sweep_fallback_champion_slot(monkeypatch, tmp_path):
    # Missing challenger artifacts -> _resolve_model_slot falls back to the
    # champion slot; the sweep must then run WITHOUT a model_prefix (the
    # champion book sweep), not silently 'sweep' an empty slot.
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    calls = []

    def fake_run(prefix, days, trials, *, fee_mult=None, stage0_dump=None,
                 model_prefix=None):
        calls.append(model_prefix)
        return {'n_trades': 2, 'sharpe': 0.1, 'dsr': 0.1,
                'net_total_pct': 1.0, 'net_pct_by_name': {'X': 1.0}}
    monkeypatch.setattr(backtest, 'run_backtest', fake_run)
    monkeypatch.setattr(sys, 'argv',
                        ['backtest.py', '--fee-sweep', '1,2',
                         '--model-prefix', 'challenger'])
    assert backtest.main() == 0
    assert calls == [None, None]
    assert (tmp_path / 'backtest_fee_sweep.json').exists()


# ---------------------------------------------------------------------------
# 6. End-to-end sweep through the REAL run_backtest (harnessed seams)
# ---------------------------------------------------------------------------

def test_fee_sweep_end_to_end_linearity(monkeypatch, tmp_path):
    _harness(monkeypatch, tmp_path, [_trending_ticker_frame('BTC')])
    rc = backtest._run_fee_sweep('', 400, 10, [1.0, 50.0])
    assert rc == 0
    payload = json.loads((tmp_path / 'backtest_fee_sweep.json').read_text())
    net1, net50 = payload['net_total_pct']
    assert net1 > 0            # trending frame: profitable at live cost
    assert net50 < 0           # ruinous multiplier: net crosses zero
    assert payload['book_status'] == 'crossed'
    # Linearity check: net(m) = gross - m * cost with entries fixed, so
    # cost = (net1 - net50) / 49 and lam* = 1 + net1/cost * (49/(net1-net50))
    cost = (net1 - net50) / 49.0
    lam_expected = 1.0 + net1 / cost
    assert payload['book_lambda_star'] == pytest.approx(lam_expected, rel=1e-6)
    assert 'BTC' in payload['per_name']
