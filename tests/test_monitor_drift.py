"""Tests for the PSI drift monitor."""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import monitor_drift
from monitor_drift import (compute_psi, load_recent_predictions,
                           log_predictions, prune_history, run_check)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every monitor_drift file at a temp dir."""
    monkeypatch.setattr(monitor_drift, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(monitor_drift, '_STATE_FILE',
                        tmp_path / 'drift_state.json')
    return tmp_path


# --- PSI math ---

def _edges(values):
    return list(np.percentile(values, np.arange(0, 101, 10)))


def test_psi_same_distribution_near_zero():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    live = rng.normal(0, 1, 2000)
    psi = compute_psi(_edges(ref), live)
    assert psi is not None and psi < 0.05


def test_psi_shifted_distribution_is_action():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    live = rng.normal(1.5, 1, 2000)  # mean shifted 1.5 sigma
    psi = compute_psi(_edges(ref), live)
    assert psi is not None and psi > monitor_drift.PSI_ACTION


def test_psi_widened_distribution_detected():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 5000)
    live = rng.normal(0, 3, 2000)  # same mean, 3x the spread
    psi = compute_psi(_edges(ref), live)
    assert psi is not None and psi > monitor_drift.PSI_WARN


def test_psi_insufficient_or_invalid_input():
    ref = _edges(np.random.default_rng(3).normal(0, 1, 1000))
    assert compute_psi(ref, np.zeros(10)) is None       # too few live
    assert compute_psi([0.0, 1.0], np.zeros(500)) is None  # bad edges


def test_psi_outliers_land_in_end_bins():
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 1, 5000)
    live = np.full(500, 50.0)  # absurd outliers, all one end bin
    psi = compute_psi(_edges(ref), live)
    assert psi is not None and np.isfinite(psi) and psi > 1.0


# --- history log round trip ---

def test_log_load_prune_roundtrip(sandbox):
    log_predictions('', {'BTC/USD': 0.5, 'ETH/USD': -0.2})
    log_predictions('', {'BTC/USD': 0.7})
    vals = load_recent_predictions('')
    assert sorted(vals) == pytest.approx([-0.2, 0.5, 0.7])

    # Inject an old record; prune must drop it, keep recent ones
    old_ts = (dt.datetime.now(dt.timezone.utc)
              - dt.timedelta(days=30)).isoformat()
    with open(monitor_drift.history_file(''), 'a') as f:
        f.write(json.dumps({'ts': old_ts, 'preds': {'BTC/USD': 9.9}}) + '\n')
    assert len(load_recent_predictions('', window_hours=24 * 60)) == 4
    prune_history('')
    assert len(load_recent_predictions('', window_hours=24 * 60)) == 3


def test_load_recent_respects_window(sandbox):
    stale_ts = (dt.datetime.now(dt.timezone.utc)
                - dt.timedelta(hours=30)).isoformat()
    with open(monitor_drift.history_file(''), 'w') as f:
        f.write(json.dumps({'ts': stale_ts, 'preds': {'BTC/USD': 1.0}}) + '\n')
    assert len(load_recent_predictions('', window_hours=24)) == 0


def test_log_predictions_never_raises(sandbox, monkeypatch):
    monkeypatch.setattr(monitor_drift, 'history_file',
                        lambda p: Path('/nonexistent-dir/x.jsonl'))
    log_predictions('', {'BTC/USD': 0.5})  # swallowed, no exception


# --- run_check: consecutive-day trigger ---

def _write_manifest(tmp_path, prefix, deciles):
    p = f'{prefix}_' if prefix else ''
    with open(tmp_path / f'{p}model_v2.manifest.json', 'w') as f:
        json.dump({'holdout': {'pred_deciles': deciles}}, f)


def _fill_history(prefix, values):
    log_predictions(prefix, {f'S{i}': v for i, v in enumerate(values)})


def test_run_check_ok_resets_streak(sandbox):
    rng = np.random.default_rng(5)
    ref = rng.normal(0, 1, 5000)
    _write_manifest(sandbox, '', _edges(ref))
    _fill_history('', list(rng.normal(0, 1, 200)))
    # Seed a prior action streak; an OK day must clear it
    monitor_drift._save_state({'crypto': {'action_days': 1,
                                          'last_action_date': '2026-01-01'}})
    r = run_check('', 'crypto')
    assert r is not None and r['level'] == 'ok'
    assert monitor_drift._load_state()['crypto']['action_days'] == 0
    assert not monitor_drift.retrain_flag_file('').exists()


def test_run_check_first_action_day_no_flag(sandbox):
    rng = np.random.default_rng(6)
    ref = rng.normal(0, 1, 5000)
    _write_manifest(sandbox, '', _edges(ref))
    _fill_history('', list(rng.normal(2.0, 1, 200)))  # heavy drift
    r = run_check('', 'crypto')
    assert r is not None and r['level'] == 'action'
    st = monitor_drift._load_state()['crypto']
    assert st['action_days'] == 1
    assert not monitor_drift.retrain_flag_file('').exists()


def test_run_check_second_consecutive_day_writes_flag(sandbox):
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, 5000)
    _write_manifest(sandbox, '', _edges(ref))
    _fill_history('', list(rng.normal(2.0, 1, 200)))
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    monitor_drift._save_state({'crypto': {'action_days': 1,
                                          'last_action_date': yesterday}})
    r = run_check('', 'crypto')
    assert r is not None and r['level'] == 'action'
    flag = monitor_drift.retrain_flag_file('')
    assert flag.exists()
    assert 'PSI' in json.loads(flag.read_text())['reason']


def test_run_check_gap_day_restarts_streak(sandbox):
    rng = np.random.default_rng(8)
    ref = rng.normal(0, 1, 5000)
    _write_manifest(sandbox, '', _edges(ref))
    _fill_history('', list(rng.normal(2.0, 1, 200)))
    # Last action was 3 days ago -> streak restarts at 1, no flag
    stale = (dt.date.today() - dt.timedelta(days=3)).isoformat()
    monitor_drift._save_state({'crypto': {'action_days': 1,
                                          'last_action_date': stale}})
    run_check('', 'crypto')
    assert monitor_drift._load_state()['crypto']['action_days'] == 1
    assert not monitor_drift.retrain_flag_file('').exists()


def test_run_check_unckeckable_without_manifest(sandbox):
    _fill_history('', [0.1] * 100)
    assert run_check('', 'crypto') is None
