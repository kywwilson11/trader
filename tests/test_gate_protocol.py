"""Tests for the backtest.py <-> run_pipeline.py gate exit-code protocol
and the run_pipeline integrity fixes from the 2026-07 deep review:

  A. rc==3 = deterministic policy rejection (model already rolled back);
     run_pipeline treats it as final-but-non-fatal for *_backtest_gate
     phases, and records a per-phase outcome ledger.
  B. backtest.py integrity: orphan optional-artifact cleanup on restore,
     present-but-unloadable vs absent LightGBM/q10 legs, empty-window
     recheck after the --days cutoff, coverage self-description.
  C. run_pipeline.py: shared trial/force-harvest resolution helpers,
     bounded thermal wait, combined-bot scope propagation, the
     _pending_bot_start pre-seed/heartbeat race fix.

Mock style follows tests/test_pipeline.py: heavy deps (torch/joblib/
lightgbm) are never actually imported — backtest's model-loading and
LSTM-prediction functions are monkeypatched at the seams so run_backtest
can be exercised end-to-end on the dev Mac.
"""
import io
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest
import run_pipeline as rp


# ---------------------------------------------------------------------------
# A1 — backtest.main() exit codes
# ---------------------------------------------------------------------------

class TestGateExitCodes:
    def test_gate_failure_returns_3(self, monkeypatch):
        monkeypatch.setattr(
            backtest, 'run_backtest',
            lambda prefix, days, trials: {'n_trades': 20, 'sharpe': -0.5, 'dsr': 0.1})
        monkeypatch.setattr(backtest, 'restore_previous_model', lambda prefix: True)
        monkeypatch.setattr(sys, 'argv', ['backtest.py', '--gate'])
        assert backtest.main() == 3

    def test_gate_pass_returns_0(self, monkeypatch):
        monkeypatch.setattr(
            backtest, 'run_backtest',
            lambda prefix, days, trials: {'n_trades': 20, 'sharpe': 1.0, 'dsr': 0.9})
        monkeypatch.setattr(sys, 'argv', ['backtest.py', '--gate'])
        assert backtest.main() == 0

    def test_no_gate_flag_returns_0_even_on_bad_metrics(self, monkeypatch):
        monkeypatch.setattr(
            backtest, 'run_backtest',
            lambda prefix, days, trials: {'n_trades': 0, 'sharpe': -9.0, 'dsr': 0.0})
        monkeypatch.setattr(sys, 'argv', ['backtest.py'])
        assert backtest.main() == 0

    def test_missing_artifact_returns_0(self, monkeypatch):
        def raise_fnf(prefix, days, trials):
            raise FileNotFoundError('no model')
        monkeypatch.setattr(backtest, 'run_backtest', raise_fnf)
        monkeypatch.setattr(sys, 'argv', ['backtest.py', '--gate'])
        assert backtest.main() == 0


# ---------------------------------------------------------------------------
# B1 — restore_previous_model orphan cleanup
# ---------------------------------------------------------------------------

class TestRestorePreviousModel:
    def test_restore_deletes_orphan_optional_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
        # Required (LSTM) leg: all 4 have a .prev -> restore proceeds.
        for s in backtest.ARTIFACT_SUFFIXES[:4]:
            (tmp_path / f'{s}.prev').write_text('prev')
            (tmp_path / s).write_text('cur')
        # Optional leg WITH a .prev -> restored from .prev.
        gated = backtest.ARTIFACT_SUFFIXES[4]
        (tmp_path / f'{gated}.prev').write_text('prev-content')
        (tmp_path / gated).write_text('cur-content')
        # Optional leg with NO .prev but a current file -> never-gated
        # orphan (e.g. LightGBM trained after the last promoted LSTM) ->
        # must be deleted, not left behind.
        orphan = backtest.ARTIFACT_SUFFIXES[5]
        (tmp_path / orphan).write_text('never gated')

        assert backtest.restore_previous_model('') is True
        assert not (tmp_path / orphan).exists()
        assert (tmp_path / gated).read_text() == 'prev-content'
        assert not (tmp_path / f'{gated}.prev').exists()
        # Required leg restored too.
        assert (tmp_path / backtest.ARTIFACT_SUFFIXES[0]).read_text() == 'prev'

    def test_restore_noop_when_required_prev_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
        # Only 3 of the 4 required .prev files present.
        for s in backtest.ARTIFACT_SUFFIXES[:3]:
            (tmp_path / f'{s}.prev').write_text('prev')
        orphan = backtest.ARTIFACT_SUFFIXES[5]
        (tmp_path / orphan).write_text('should be untouched')

        assert backtest.restore_previous_model('') is False
        assert (tmp_path / orphan).exists()  # untouched — no partial restore


# ---------------------------------------------------------------------------
# B2 — present-but-unloadable vs absent optional legs
# ---------------------------------------------------------------------------

class TestLoadLgbDistinguishesAbsentVsUnloadable:
    def test_present_but_unloadable_logs_and_returns_none(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
        (tmp_path / 'lgb_model.txt').write_text('not a real booster')

        def raise_load(prefix=''):
            raise RuntimeError('corrupt booster')
        monkeypatch.setattr('model_lgb.load_lgb_model', raise_load)

        assert backtest._load_lgb('') is None
        out = capsys.readouterr().out
        assert 'lgb_model.txt present but failed to load' in out
        assert 'evaluating WITHOUT this leg' in out

    def test_absent_is_silent(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)

        def raise_load(prefix=''):
            raise RuntimeError('should not matter — file is absent')
        monkeypatch.setattr('model_lgb.load_lgb_model', raise_load)

        assert backtest._load_lgb('') is None
        out = capsys.readouterr().out
        assert 'present but failed to load' not in out


# ---------------------------------------------------------------------------
# B3/B4 — run_backtest: empty post-cutoff window + coverage fields
# ---------------------------------------------------------------------------

def _fake_predict_ticker_factory(actionable_index=20, pred_value=5.0):
    def fake(model, scaler, config, feature_cols, tdf, lgb_model=None, q10_model=None):
        n = len(tdf)
        preds = np.full(n, np.nan)
        if n > actionable_index + 1:
            preds[actionable_index] = pred_value
        return preds, None
    return fake


class TestRunBacktestIntegrity:
    def test_empty_post_cutoff_window_raises_systemexit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest, '_load_artifacts',
                            lambda prefix: (None, None, {'seq_len': 5}, ['f1']))
        monkeypatch.setattr(backtest, '_load_lgb', lambda prefix: None)
        monkeypatch.setattr(backtest, '_load_q10', lambda prefix: None)
        idx = pd.date_range('2026-01-01', periods=10, freq='h', tz='UTC')
        df = pd.DataFrame({'Ticker': 'BTC', 'Close': 100.0, 'f1': 0.0}, index=idx)
        monkeypatch.setattr('data_utils.load_training_data', lambda asset: df)

        # Negative --days pushes the cutoff PAST the data's own max, so the
        # POST-filter frame is empty even though the pre-filter frame (and
        # the first `if df.empty` check) was not.
        with pytest.raises(SystemExit, match='No training data in the last'):
            backtest.run_backtest(prefix='', days=-5)

    def test_coverage_fields_and_evaluated_vs_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)
        monkeypatch.setattr(
            backtest, '_load_artifacts',
            lambda prefix: (None, None,
                           {'seq_len': 5, 'trade_threshold': 0.05}, ['f1', 'f2']))
        monkeypatch.setattr(backtest, '_load_lgb', lambda prefix: None)
        monkeypatch.setattr(backtest, '_load_q10', lambda prefix: None)
        monkeypatch.setattr(backtest, '_predict_ticker',
                            _fake_predict_ticker_factory())

        rng = np.random.default_rng(0)
        idx_ok = pd.date_range('2026-01-01', periods=40, freq='h', tz='UTC')
        price = 100 + np.cumsum(rng.normal(0, 0.1, len(idx_ok)))
        evaluated = pd.DataFrame({
            'Ticker': 'BTC', 'Open': price, 'High': price * 1.01,
            'Low': price * 0.99, 'Close': price, 'ATR': np.nan,
            'f1': np.linspace(0, 1, len(idx_ok)),
            'f2': np.linspace(1, 0, len(idx_ok)),
        }, index=idx_ok)

        # Too few bars (< seq_len + 10 == 15) -> skipped before prediction.
        idx_short = pd.date_range('2026-01-01', periods=3, freq='h', tz='UTC')
        skipped = pd.DataFrame({
            'Ticker': 'ETH', 'Open': 100.0, 'High': 101.0, 'Low': 99.0,
            'Close': 100.0, 'ATR': np.nan, 'f1': 0.0, 'f2': 0.0,
        }, index=idx_short)

        combined = pd.concat([evaluated, skipped])
        monkeypatch.setattr('data_utils.load_training_data', lambda asset: combined)

        metrics = backtest.run_backtest(prefix='', days=400, n_search_trials=10)

        assert metrics['n_tickers_evaluated'] == 1
        assert metrics['n_tickers_skipped'] == 1
        assert 'ETH' in metrics['skipped_tickers']
        assert isinstance(metrics['skipped_tickers'], list)


# ---------------------------------------------------------------------------
# C1 — shared trial-count / force-harvest helpers
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, retrain_trials=100):
        self.retrain_trials = retrain_trials


class TestResolveRetrainTrials:
    def test_honors_explicit_override(self):
        assert rp._resolve_retrain_trials(_Args(42), {'crypto': 100, 'stock': 150}) == 42

    def test_uses_adaptive_max_when_default(self):
        assert rp._resolve_retrain_trials(_Args(100), {'crypto': 80, 'stock': 120}) == 120

    def test_falls_back_to_100_with_no_adaptive_counts(self):
        assert rp._resolve_retrain_trials(_Args(100), {}) == 100


class TestNeedsForceHarvest:
    def test_detects_missing_target_column(self, tmp_path, monkeypatch):
        csv_path = tmp_path / 'training_data.csv'
        csv_path.write_text('Ticker,Close,Target_Return_24\n')
        monkeypatch.setattr(rp, 'get_max_forward_bars', lambda at: 48)
        monkeypatch.setattr('data_utils.get_data_path', lambda at: csv_path)

        needs, reasons = rp._needs_force_harvest(True, False)
        assert needs is True
        assert any('crypto' in r and 'Target_Return_48' in r for r in reasons)

    def test_false_when_column_present(self, tmp_path, monkeypatch):
        csv_path = tmp_path / 'training_data.csv'
        csv_path.write_text('Ticker,Close,Target_Return_48\n')
        monkeypatch.setattr(rp, 'get_max_forward_bars', lambda at: 48)
        monkeypatch.setattr('data_utils.get_data_path', lambda at: csv_path)

        needs, reasons = rp._needs_force_harvest(True, False)
        assert needs is False
        assert reasons == []

    def test_skips_inactive_books(self):
        needs, reasons = rp._needs_force_harvest(False, False)
        assert needs is False
        assert reasons == []


# ---------------------------------------------------------------------------
# C4 — bounded thermal wait
# ---------------------------------------------------------------------------

class TestBoundedThermalWait:
    def test_calls_mark_progress_and_respects_deadline(self, monkeypatch):
        fake_hw = types.ModuleType('hw_monitor')
        fake_hw.get_gpu_temp = lambda: 99.0  # hot forever
        monkeypatch.setitem(sys.modules, 'hw_monitor', fake_hw)

        progress_calls = []
        monkeypatch.setattr(rp, 'mark_progress', lambda: progress_calls.append(1))

        start = time.time()
        rp._bounded_thermal_wait(max_temp=70, deadline_sec=0.05, poll_interval=0.01)
        elapsed = time.time() - start

        assert len(progress_calls) >= 2
        assert elapsed < 2.0  # bounded — not the real 1800s default

    def test_returns_immediately_when_cool(self, monkeypatch):
        fake_hw = types.ModuleType('hw_monitor')
        fake_hw.get_gpu_temp = lambda: 40.0
        monkeypatch.setitem(sys.modules, 'hw_monitor', fake_hw)
        progress_calls = []
        monkeypatch.setattr(rp, 'mark_progress', lambda: progress_calls.append(1))

        rp._bounded_thermal_wait(max_temp=70, deadline_sec=1800, poll_interval=30)
        assert len(progress_calls) == 1

    def test_fails_open_when_sensor_unavailable(self, monkeypatch):
        fake_hw = types.ModuleType('hw_monitor')
        fake_hw.get_gpu_temp = lambda: None
        monkeypatch.setitem(sys.modules, 'hw_monitor', fake_hw)
        progress_calls = []
        monkeypatch.setattr(rp, 'mark_progress', lambda: progress_calls.append(1))

        rp._bounded_thermal_wait(max_temp=70, deadline_sec=1800, poll_interval=30)
        assert len(progress_calls) == 1


# ---------------------------------------------------------------------------
# C6 — combined-mode crash-restart honors --crypto-only/--stock-only scope
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, returncode):
        self._rc = returncode
        self.returncode = returncode
        self.pid = 12345

    def poll(self):
        return self._rc

    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass


class _FakeFH(io.StringIO):
    pass


class TestCheckRestartBotsScope:
    def test_rebuilds_combined_cmd_with_crypto_only_scope(self, monkeypatch):
        captured = {}

        def fake_start_bot(cmd, log_path):
            captured['cmd'] = cmd
            return _FakeProc(None), _FakeFH()

        monkeypatch.setattr(rp, '_start_bot', fake_start_bot)
        monkeypatch.setattr(rp, '_BOT_SCOPE', (True, False))
        monkeypatch.setattr(rp, '_manually_stopped', set())

        bots = [('Bots', _FakeProc(1), _FakeFH())]  # crashed combined process
        rp._check_restart_bots(bots, _FakeFH())

        assert '--crypto-only' in captured['cmd']
        assert '--stock-only' not in captured['cmd']

    def test_rebuilds_combined_cmd_with_stock_only_scope(self, monkeypatch):
        captured = {}

        def fake_start_bot(cmd, log_path):
            captured['cmd'] = cmd
            return _FakeProc(None), _FakeFH()

        monkeypatch.setattr(rp, '_start_bot', fake_start_bot)
        monkeypatch.setattr(rp, '_BOT_SCOPE', (False, True))
        monkeypatch.setattr(rp, '_manually_stopped', set())

        bots = [('Bots', _FakeProc(1), _FakeFH())]
        rp._check_restart_bots(bots, _FakeFH())

        assert '--stock-only' in captured['cmd']
        assert '--crypto-only' not in captured['cmd']

    def test_full_scope_adds_no_flag(self, monkeypatch):
        captured = {}

        def fake_start_bot(cmd, log_path):
            captured['cmd'] = cmd
            return _FakeProc(None), _FakeFH()

        monkeypatch.setattr(rp, '_start_bot', fake_start_bot)
        monkeypatch.setattr(rp, '_BOT_SCOPE', (True, True))
        monkeypatch.setattr(rp, '_manually_stopped', set())

        bots = [('Bots', _FakeProc(1), _FakeFH())]
        rp._check_restart_bots(bots, _FakeFH())

        assert '--crypto-only' not in captured['cmd']
        assert '--stock-only' not in captured['cmd']


# ---------------------------------------------------------------------------
# A2/A3/C8/C10 — _run_training: phase_results ledger + book scoping
# ---------------------------------------------------------------------------

class TestRunTraining:
    def test_phase_results_ledger_and_gate_rejection_is_final_not_failed(self, monkeypatch):
        rc_by_id = {'crypto_search': 0, 'crypto_backtest_gate': 3, 'stock_search': 0}

        def fake_run_phase(phase, log_fh, status):
            return rc_by_id[phase['id']]
        monkeypatch.setattr(rp, 'run_phase', fake_run_phase)
        monkeypatch.setattr(rp, 'write_status', lambda *a, **k: None)
        monkeypatch.setattr(time, 'sleep', lambda s: None)

        phases = [
            {'id': 'crypto_search', 'label': 'Crypto Search', 'idx': 0},
            {'id': 'crypto_backtest_gate', 'label': 'Crypto Gate', 'idx': 1},
            {'id': 'stock_search', 'label': 'Stock Search', 'idx': 2},
        ]
        status = {'best_score': 0.5}
        result = rp._run_training(phases, io.StringIO(), status, is_retrain=True)

        assert status['phase_results']['crypto_search'] == \
            {'rc': 0, 'attempts': 1, 'outcome': 'ok'}
        gate_entry = status['phase_results']['crypto_backtest_gate']
        assert gate_entry['rc'] == 3
        assert gate_entry['attempts'] == 1  # no retry on a gate rejection
        assert gate_entry['outcome'] == 'gate_failed_rolled_back'
        assert status['phase_results']['stock_search']['outcome'] == 'ok'

        # A rc==3 gate verdict counts the book as rolled-back, not failed.
        assert result == {'crypto': True, 'stock': True}
        assert status['crypto_final_score'] == 0.5
        assert status['stock_final_score'] == 0.5

    def test_genuine_failure_skips_only_that_books_remaining_phases(self, monkeypatch):
        call_log = []

        def fake_run_phase(phase, log_fh, status):
            call_log.append(phase['id'])
            return 1 if phase['id'] == 'crypto_search' else 0
        monkeypatch.setattr(rp, 'run_phase', fake_run_phase)
        monkeypatch.setattr(rp, 'write_status', lambda *a, **k: None)
        monkeypatch.setattr(time, 'sleep', lambda s: None)

        phases = [
            {'id': 'crypto_search', 'label': 'Crypto Search', 'idx': 0},
            {'id': 'crypto_meta', 'label': 'Crypto Meta', 'idx': 1},
            {'id': 'crypto_backtest_gate', 'label': 'Crypto Gate', 'idx': 2},
            {'id': 'stock_search', 'label': 'Stock Search', 'idx': 3},
        ]
        status = {'best_score': 0.0}
        result = rp._run_training(phases, io.StringIO(), status, is_retrain=True)

        # crypto_search retried MAX_PHASE_RETRIES times then gives up —
        # crypto_meta/crypto_backtest_gate never run, stock_search does.
        assert call_log.count('crypto_search') == rp.MAX_PHASE_RETRIES
        assert 'crypto_meta' not in call_log
        assert 'crypto_backtest_gate' not in call_log
        assert 'stock_search' in call_log

        assert result['crypto'] is False
        assert result['stock'] is True
        assert status['crypto_final_score'] is None  # tri-state, not 0.0
        assert status['phase_results']['crypto_search']['outcome'] == 'failed'

    def test_suspend_returns_suspended_sentinel(self, monkeypatch):
        monkeypatch.setattr(rp, 'run_phase', lambda phase, log_fh, status: -99)
        monkeypatch.setattr(rp, 'write_status', lambda *a, **k: None)
        phases = [{'id': 'crypto_search', 'label': 'Crypto Search', 'idx': 0}]
        status = {'best_score': 0.0}
        result = rp._run_training(phases, io.StringIO(), status, is_retrain=True)
        assert result == 'suspended'


# ---------------------------------------------------------------------------
# C5 — _pending_bot_start pre-seed (heartbeat dict-size race)
# ---------------------------------------------------------------------------

class TestPendingBotStartPreseed:
    def test_status_dict_preseeds_key_and_no_pop_remains(self):
        import inspect
        main_src = inspect.getsource(rp.main)
        assert "'_pending_bot_start': None" in main_src
        assert "status.pop('_pending_bot_start'" not in main_src

    def test_no_pop_calls_anywhere_in_module(self):
        src = Path(rp.__file__).read_text()
        assert "status.pop('_pending_bot_start'" not in src
        assert src.count("status['_pending_bot_start'] = None") >= 2

    def test_launch_args_recorded_in_main(self):
        import inspect
        main_src = inspect.getsource(rp.main)
        assert "'launch_args': sys.argv[1:]" in main_src
