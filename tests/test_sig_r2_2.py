"""SIG-R2-2: --fee-sweep pass 1 overwrote the gate's Stage-0 dump.

backtest._run_fee_sweep forced stage0_dump=False only for passes k > 0, so
pass 1 ran with the module default (STAGE0_DUMP_DEFAULT = True) and
run_backtest's dump writer (`_s0.write_rows(s0_rows, dump_path)`, path
f"{rslot}_stage0_preds.json") REPLACED the weekly --gate replay's dump
(run_pipeline: crypto --days 44, stock --days 60) with the sweep's own
window (180 d in the documented example — mostly inside the search
region). ic_by_name / rank_gradient_report / naive_vs_blend /
evidence_reads then read the wrong dump.

Fix: every sweep pass passes stage0_dump=False. --gate semantics untouched
(--gate refuses --fee-sweep; a plain/gate run still writes the dump).
Measurement-only path — no flag.

Module under test: SIG_R2_BT (default: the repo's backtest.py).
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(os.environ.get('SIG_R2_REPO',
                           Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
BT_PATH = os.environ.get('SIG_R2_BT', str(REPO / 'backtest.py'))

GATE_DUMP = b'[{"sentinel": "weekly --gate dump, --days 44"}]'


@pytest.fixture(scope='module')
def backtest():
    spec = importlib.util.spec_from_file_location('bt_sig_r2_2', BT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frame(ticker='BTC', n=60, drift=0.004, seed=0):
    rng = np.random.RandomState(seed)
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.001, n))
    atr = (pd.Series(close).rolling(5, min_periods=1).std()
           .fillna(0.5).values + 0.5)
    idx = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
    return pd.DataFrame({'Open': close * 0.999, 'High': close * 1.002,
                         'Low': close * 0.998, 'Close': close, 'ATR': atr,
                         'Ticker': ticker, 'f1': np.linspace(0, 1, n),
                         'f2': np.linspace(1, 0, n)}, index=idx)


def _harness(bt, monkeypatch, tmp_path):
    """tests/test_r2c_fee_sweep.py's heavy-seam harness."""
    monkeypatch.setattr(bt, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(bt, 'STAGE0_DUMP_DEFAULT', True)
    monkeypatch.setattr(bt, '_load_artifacts',
                        lambda prefix: (None, None,
                                        {'seq_len': 5,
                                         'trade_threshold': 0.05},
                                        ['f1', 'f2']))
    monkeypatch.setattr(bt, '_load_lgb', lambda prefix: None)
    monkeypatch.setattr(bt, '_load_q10', lambda prefix: None)

    def fake_predict(model, scaler, config, feature_cols, tdf,
                     lgb_model=None, q10_model=None):
        preds = np.full(len(tdf), np.nan)
        preds[20] = 5.0
        return preds, None
    monkeypatch.setattr(bt, '_predict_ticker', fake_predict)
    combined = pd.concat([_frame('BTC'), _frame('ETH', seed=1)])
    monkeypatch.setattr('data_utils.load_training_data',
                        lambda asset: combined)


def test_every_sweep_pass_disables_the_stage0_dump(backtest, monkeypatch,
                                                   tmp_path):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    calls = []

    def fake_run(prefix, days, trials, *, fee_mult=None, stage0_dump=None,
                 model_prefix=None):
        calls.append(stage0_dump)
        return {'n_trades': 3, 'sharpe': 0.1, 'dsr': 0.1,
                'net_total_pct': 5.0 - fee_mult,
                'net_pct_by_name': {'BTC': 5.0 - fee_mult}}
    monkeypatch.setattr(backtest, 'run_backtest', fake_run)
    assert backtest._run_fee_sweep('', 180, 10, [1.0, 2.0, 4.0, 6.0]) == 0
    assert calls == [False, False, False, False]      # pass 1 included


@pytest.mark.parametrize('model_prefix,dump_name', [
    (None, 'stage0_preds.json'),
    ('challenger', 'challenger_stage0_preds.json')])
def test_sweep_never_overwrites_gate_dump(backtest, monkeypatch, tmp_path,
                                         model_prefix, dump_name):
    _harness(backtest, monkeypatch, tmp_path)
    (tmp_path / dump_name).write_bytes(GATE_DUMP)
    rc = backtest._run_fee_sweep('', 400, 10, [1.0, 50.0],
                                 model_prefix=model_prefix)
    assert rc == 0
    assert (tmp_path / dump_name).read_bytes() == GATE_DUMP
    sweep = json.loads(next(tmp_path.glob('*fee_sweep.json')).read_text())
    assert sweep['mults'] == [1.0, 50.0]              # sweep still reports


def test_plain_replay_still_writes_the_dump(backtest, monkeypatch, tmp_path):
    # Positive control (and --gate semantics untouched): the default
    # 3-arg replay — what `--gate` calls — still owns and writes the dump,
    # so the no-overwrite assertion above is not vacuous.
    _harness(backtest, monkeypatch, tmp_path)
    (tmp_path / 'stage0_preds.json').write_bytes(GATE_DUMP)
    m = backtest.run_backtest('', 400, 10)
    assert m['stage0_dump']['path'] == 'stage0_preds.json'
    assert (tmp_path / 'stage0_preds.json').read_bytes() != GATE_DUMP
