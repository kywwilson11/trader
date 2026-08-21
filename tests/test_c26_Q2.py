"""Packet Q2 (campaign 2026-08, defect D03) — challenger-targeted policy gate.

Under the DEFAULT shadow-mode weekly retrain, hypersearch saves the fresh
model into the CHALLENGER slot while backtest --gate replays the CHAMPION:
the model that will actually deploy is never policy-gated and a gate failure
rolls the innocent live champion back to a stale .prev. This packet adds
GATE_TARGETS_CHALLENGER (default OFF) so the weekly gate can score the
challenger slot on the champion's book data, with exit 3 meaning
HOLD-the-challenger (no champion rollback) and the verdict persisted to
{slot}_policy_gate.json.

Groups:
  A. flag default + slot-naming parity with shadow.challenger_prefix
  B. backtest._resolve_model_slot / _report_slot pure decision tables
  C. main() exit-3 action mapping (hold vs rollback vs fallback)
  D. run_backtest slot plumbing (loaders get the model slot, book data and
     report identity stay correct; OFF byte-shape pinned)
  E. run_pipeline gate-phase construction (OFF byte-identity pinned
     literally; ON adds --model-prefix + gate_target; non-shadow never
     targets the challenger)
  F. _run_training rc==3 outcome mapping ('challenger_gate_held')

Mock style follows tests/test_gate_protocol.py: heavy deps are never
imported — loader/predict seams are monkeypatched as single-arg lambdas.
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest
import run_pipeline as rp
import shadow
import strategy_config


CORE = backtest.ARTIFACT_SUFFIXES[:4]


def _touch_core(tmp_path, slot):
    p = f'{slot}_' if slot else ''
    for s in CORE:
        (tmp_path / f'{p}{s}').write_text('x')


FAIL_METRICS = {'n_trades': 0, 'sharpe': -1.0, 'dsr': 0.0}
PASS_METRICS = {'n_trades': 50, 'sharpe': 1.0, 'dsr': 0.9}


# ---------------------------------------------------------------------------
# A — flag default + naming parity
# ---------------------------------------------------------------------------

class TestFlagAndNaming:
    def test_flag_defaults_off(self):
        assert strategy_config.GATE_TARGETS_CHALLENGER is False

    def test_gate_model_prefix_matches_shadow_slot_namer(self):
        # Pin the literal against the REAL slot-namer so drift is impossible.
        assert (rp._gate_model_prefix('', True, True)
                == shadow.challenger_prefix('') == 'challenger')
        assert (rp._gate_model_prefix('stock', True, True)
                == shadow.challenger_prefix('stock') == 'stock_challenger')

    @pytest.mark.parametrize('shadow_active,flag_on', [
        (False, True), (True, False), (False, False)])
    def test_gate_model_prefix_none_unless_both(self, shadow_active, flag_on):
        assert rp._gate_model_prefix('', shadow_active, flag_on) is None
        assert rp._gate_model_prefix('stock', shadow_active, flag_on) is None


# ---------------------------------------------------------------------------
# B — pure decision tables
# ---------------------------------------------------------------------------

class TestResolveModelSlot:
    @pytest.mark.parametrize('data_prefix,requested,core,expected', [
        ('', '', True, ('', 'legacy')),
        ('', '', False, ('', 'legacy')),
        ('stock', 'stock', True, ('stock', 'legacy')),
        ('stock', 'stock', False, ('stock', 'legacy')),
        ('', 'challenger', True, ('challenger', 'challenger')),
        ('stock', 'stock_challenger', True,
         ('stock_challenger', 'challenger')),
        ('', 'challenger', False, ('', 'fallback_champion')),
        ('stock', 'stock_challenger', False, ('stock', 'fallback_champion')),
    ])
    def test_decision_table(self, data_prefix, requested, core, expected):
        assert backtest._resolve_model_slot(
            data_prefix, requested, core) == expected

    def test_report_slot(self):
        assert backtest._report_slot('', '') == ''
        assert backtest._report_slot('', 'challenger') == 'challenger'
        assert backtest._report_slot('stock', 'stock') == 'stock'
        assert (backtest._report_slot('stock', 'stock_challenger')
                == 'stock_challenger')


# ---------------------------------------------------------------------------
# C — main() exit-3 action mapping
# ---------------------------------------------------------------------------

def _wire_main(monkeypatch, tmp_path, metrics, restore_calls):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    if isinstance(metrics, BaseException):
        def rb(prefix, days, trials, **kw):
            raise metrics
    else:
        def rb(prefix, days, trials, **kw):
            return dict(metrics)
    monkeypatch.setattr(backtest, 'run_backtest', rb)

    def spy(prefix):
        restore_calls.append(prefix)
        return True
    monkeypatch.setattr(backtest, 'restore_previous_model', spy)


class TestMainActionMapping:
    def test_challenger_fail_holds_no_rollback_writes_sidecar(
            self, monkeypatch, tmp_path):
        calls = []
        _touch_core(tmp_path, 'challenger')
        _wire_main(monkeypatch, tmp_path, FAIL_METRICS, calls)
        monkeypatch.setattr(sys, 'argv',
                            ['backtest.py', '--gate',
                             '--model-prefix', 'challenger'])
        assert backtest.main() == 3
        assert calls == []  # champion never touched
        sidecar = tmp_path / 'challenger_policy_gate.json'
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text())
        assert payload['passed'] is False
        assert payload['data_prefix'] == ''
        assert payload['gate']['action'] == 'hold_challenger'
        assert payload['gate']['gate_target'] == 'challenger'
        assert payload['gate']['restored'] is False

    def test_challenger_pass_writes_sidecar_passed_true(
            self, monkeypatch, tmp_path):
        calls = []
        _touch_core(tmp_path, 'challenger')
        _wire_main(monkeypatch, tmp_path, PASS_METRICS, calls)
        monkeypatch.setattr(sys, 'argv',
                            ['backtest.py', '--gate',
                             '--model-prefix', 'challenger'])
        assert backtest.main() == 0
        assert calls == []
        payload = json.loads(
            (tmp_path / 'challenger_policy_gate.json').read_text())
        assert payload['passed'] is True

    def test_stock_challenger_fail_sidecar_and_hold(
            self, monkeypatch, tmp_path):
        calls = []
        _touch_core(tmp_path, 'stock_challenger')
        _wire_main(monkeypatch, tmp_path, FAIL_METRICS, calls)
        monkeypatch.setattr(sys, 'argv',
                            ['backtest.py', '--prefix', 'stock', '--gate',
                             '--model-prefix', 'stock_challenger'])
        assert backtest.main() == 3
        assert calls == []  # stock champion never touched
        payload = json.loads(
            (tmp_path / 'stock_challenger_policy_gate.json').read_text())
        assert payload['passed'] is False
        assert payload['data_prefix'] == 'stock'
        assert payload['gate']['gate_target'] == 'stock_challenger'

    def test_challenger_verdict_patches_challenger_report_only(
            self, monkeypatch, tmp_path):
        calls = []
        _touch_core(tmp_path, 'challenger')
        # Manifest present -> its mtime must land in the sidecar: it is the
        # staleness key the future shadow-side promote pre-flight compares
        # against the current challenger manifest (cross-file follow-up).
        man = tmp_path / 'challenger_model_v2.manifest.json'
        man.write_text('{}')
        (tmp_path / 'backtest_challenger_report.json').write_text('{}')
        (tmp_path / 'backtest_report.json').write_text('{}')
        _wire_main(monkeypatch, tmp_path, FAIL_METRICS, calls)
        monkeypatch.setattr(sys, 'argv',
                            ['backtest.py', '--gate',
                             '--model-prefix', 'challenger'])
        assert backtest.main() == 3
        rep = json.loads(
            (tmp_path / 'backtest_challenger_report.json').read_text())
        assert rep['gate']['gate_target'] == 'challenger'
        assert rep['gate']['action'] == 'hold_challenger'
        assert rep['gate']['exit_code'] == 3
        # The champion book report (what the GUI reads) is never patched
        # by a challenger-targeted run.
        assert json.loads(
            (tmp_path / 'backtest_report.json').read_text()) == {}
        payload = json.loads(
            (tmp_path / 'challenger_policy_gate.json').read_text())
        assert payload['challenger_manifest_mtime'] == int(man.stat().st_mtime)

    def test_legacy_fail_rolls_back_and_writes_no_sidecar(
            self, monkeypatch, tmp_path):
        calls = []
        _wire_main(monkeypatch, tmp_path, FAIL_METRICS, calls)
        monkeypatch.setattr(sys, 'argv', ['backtest.py', '--gate'])
        assert backtest.main() == 3
        assert calls == ['']  # legacy rollback of the champion slot
        assert not list(tmp_path.glob('*policy_gate.json'))  # OFF byte-shape

    def test_fallback_champion_when_challenger_slot_empty(
            self, monkeypatch, tmp_path, capsys):
        calls = []
        _touch_core(tmp_path, '')  # champion core present, challenger absent
        _wire_main(monkeypatch, tmp_path, FAIL_METRICS, calls)
        monkeypatch.setattr(sys, 'argv',
                            ['backtest.py', '--gate',
                             '--model-prefix', 'challenger'])
        assert backtest.main() == 3
        # The champion slot holds the fresh (first-deploy) save — a failure
        # must roll it back like the legacy gate.
        assert calls == ['']
        out = capsys.readouterr().out
        assert 'falling back' in out
        assert not (tmp_path / 'challenger_policy_gate.json').exists()

    def test_fallback_with_no_artifacts_anywhere_is_nothing_to_gate(
            self, monkeypatch, tmp_path):
        calls = []
        _wire_main(monkeypatch, tmp_path,
                   FileNotFoundError('no model'), calls)
        monkeypatch.setattr(sys, 'argv',
                            ['backtest.py', '--gate',
                             '--model-prefix', 'challenger'])
        # The FNF core-artifact excuse checks the RESOLVED slot (champion,
        # also empty) -> 'nothing to gate', exit 0, no rollback.
        assert backtest.main() == 0
        assert calls == []


# ---------------------------------------------------------------------------
# D — run_backtest slot plumbing
# ---------------------------------------------------------------------------

def _one_ticker_frame(ticker='BTC', n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
    price = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame({
        'Ticker': ticker, 'Open': price, 'High': price * 1.01,
        'Low': price * 0.99, 'Close': price, 'ATR': np.nan,
        'f1': np.linspace(0, 1, n), 'f2': np.linspace(1, 0, n),
    }, index=idx)


def _fake_predict_ticker(model, scaler, config, feature_cols, tdf,
                         lgb_model=None, q10_model=None):
    n = len(tdf)
    preds = np.full(n, np.nan)
    if n > 21:
        preds[20] = 5.0
    return preds, None


def _wire_run_backtest(monkeypatch, tmp_path, loader_calls, data_calls):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(
        backtest, '_load_artifacts',
        lambda prefix: (loader_calls.append(('artifacts', prefix)) or
                        (None, None,
                         {'seq_len': 5, 'trade_threshold': 0.05},
                         ['f1', 'f2'])))
    monkeypatch.setattr(
        backtest, '_load_lgb',
        lambda prefix: loader_calls.append(('lgb', prefix)))
    monkeypatch.setattr(
        backtest, '_load_q10',
        lambda prefix: loader_calls.append(('q10', prefix)))
    monkeypatch.setattr(backtest, '_predict_ticker', _fake_predict_ticker)
    frame = _one_ticker_frame()
    monkeypatch.setattr(
        'data_utils.load_training_data',
        lambda asset: data_calls.append(asset) or frame)


class TestRunBacktestSlotPlumbing:
    def test_challenger_slot_loaders_book_data_and_report_identity(
            self, monkeypatch, tmp_path):
        loader_calls, data_calls = [], []
        _wire_run_backtest(monkeypatch, tmp_path, loader_calls, data_calls)
        metrics = backtest.run_backtest(prefix='', days=400,
                                        n_search_trials=10,
                                        model_prefix='challenger')
        assert loader_calls == [('artifacts', 'challenger'),
                                ('lgb', 'challenger'),
                                ('q10', 'challenger')]
        assert data_calls == ['crypto']  # book data stays champion-keyed
        assert (tmp_path / 'backtest_challenger_report.json').exists()
        assert not (tmp_path / 'backtest_report.json').exists()
        assert metrics['prefix'] == ''
        assert metrics['model_prefix'] == 'challenger'
        assert metrics['gate_target'] == 'challenger'

    def test_off_shape_no_model_prefix_keys(self, monkeypatch, tmp_path):
        loader_calls, data_calls = [], []
        _wire_run_backtest(monkeypatch, tmp_path, loader_calls, data_calls)
        metrics = backtest.run_backtest(prefix='', days=400,
                                        n_search_trials=10)
        assert loader_calls == [('artifacts', ''), ('lgb', ''), ('q10', '')]
        assert (tmp_path / 'backtest_report.json').exists()
        assert 'model_prefix' not in metrics  # byte-shape pin
        assert 'gate_target' not in metrics


# ---------------------------------------------------------------------------
# E — run_pipeline gate-phase construction
# ---------------------------------------------------------------------------

def _gate_phases(phases):
    return {ph['id']: ph for ph in phases if ph['id'].endswith('_backtest_gate')}


def _spy_print(monkeypatch):
    """rp._print is TTY-gated (silent under pytest) — spy on it directly."""
    lines = []
    monkeypatch.setattr(rp, '_print',
                        lambda *a, **k: lines.append(' '.join(map(str, a))))
    return lines


class TestBuildTrainingPhases:
    def test_flag_off_byte_identity_and_warning(self, monkeypatch):
        monkeypatch.setattr(strategy_config, 'GATE_TARGETS_CHALLENGER', False)
        monkeypatch.setenv('TRADER_SHADOW_MODE', '1')
        printed = _spy_print(monkeypatch)
        phases = rp._build_training_phases(50, True, True, shadow=True)
        gates = _gate_phases(phases)
        # Literal OFF byte-identity pin for both gate cmds.
        assert gates['crypto_backtest_gate']['cmd'] == [
            rp.PYTHON, '-u', 'backtest.py', '--days', '44',
            '--trials', '50', '--gate']
        assert gates['stock_backtest_gate']['cmd'] == [
            rp.PYTHON, '-u', 'backtest.py', '--prefix', 'stock',
            '--days', '60', '--trials', '50', '--gate']
        for ph in phases:
            assert 'gate_target' not in ph
            assert '--model-prefix' not in ph['cmd']
        # D03 warning printed once while the flag is OFF in shadow mode
        assert any('GATE_TARGETS_CHALLENGER is' in ln for ln in printed)

    def test_flag_on_shadow_targets_challenger(self, monkeypatch):
        monkeypatch.setattr(strategy_config, 'GATE_TARGETS_CHALLENGER', True)
        monkeypatch.setenv('TRADER_SHADOW_MODE', '1')
        printed = _spy_print(monkeypatch)
        phases = rp._build_training_phases(50, True, True, shadow=True)
        gates = _gate_phases(phases)
        assert gates['crypto_backtest_gate']['cmd'][-2:] == \
            ['--model-prefix', 'challenger']
        assert gates['stock_backtest_gate']['cmd'][-2:] == \
            ['--model-prefix', 'stock_challenger']
        assert gates['crypto_backtest_gate']['gate_target'] == 'challenger'
        assert gates['stock_backtest_gate']['gate_target'] == 'challenger'
        # Search/meta phases untouched.
        by_id = {ph['id']: ph for ph in phases}
        assert '--model-prefix' not in by_id['crypto_search']['cmd']
        assert by_id['crypto_meta']['cmd'] == [rp.PYTHON, '-u', 'meta_label.py']
        assert by_id['stock_meta']['cmd'] == [
            rp.PYTHON, '-u', 'meta_label.py', '--prefix', 'stock']
        assert not any('GATE_TARGETS_CHALLENGER is' in ln for ln in printed)

    def test_flag_on_but_non_shadow_stays_legacy(self, monkeypatch):
        monkeypatch.setattr(strategy_config, 'GATE_TARGETS_CHALLENGER', True)
        monkeypatch.setenv('TRADER_SHADOW_MODE', '0')
        printed = _spy_print(monkeypatch)
        phases = rp._build_training_phases(50, True, True, shadow=True)
        for ph in phases:
            assert '--model-prefix' not in ph['cmd']
            assert 'gate_target' not in ph
        assert not any('GATE_TARGETS_CHALLENGER is' in ln for ln in printed)


# ---------------------------------------------------------------------------
# F — _run_training rc==3 outcome mapping
# ---------------------------------------------------------------------------

class TestGateRejectOutcome:
    def test_pure_table(self):
        assert rp._gate_reject_outcome({}) == 'gate_failed_rolled_back'
        assert (rp._gate_reject_outcome({'gate_target': 'challenger'})
                == 'challenger_gate_held')

    def test_run_training_maps_challenger_gate_held(self, monkeypatch):
        rc_by_id = {'crypto_search': 0, 'crypto_backtest_gate': 3}
        monkeypatch.setattr(rp, 'run_phase',
                            lambda phase, log_fh, status: rc_by_id[phase['id']])
        monkeypatch.setattr(rp, 'write_status', lambda *a, **k: None)
        monkeypatch.setattr(time, 'sleep', lambda s: None)
        phases = [
            {'id': 'crypto_search', 'label': 'Crypto Search', 'idx': 0},
            {'id': 'crypto_backtest_gate', 'label': 'Crypto Gate', 'idx': 1,
             'gate_target': 'challenger'},
        ]
        status = {'best_score': 0.5}
        result = rp._run_training(phases, io.StringIO(), status,
                                  is_retrain=True)
        entry = status['phase_results']['crypto_backtest_gate']
        assert entry['rc'] == 3
        assert entry['attempts'] == 1  # still no retry
        assert entry['outcome'] == 'challenger_gate_held'
        assert result == {'crypto': True}  # book still counts as success
