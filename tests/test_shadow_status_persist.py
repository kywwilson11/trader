"""Tests for shadow-status persistence (Phase 2.2 producer side —
research/gui_review_2026-07.md §7 challenger cell / promotion story).

Exercises evaluate_and_maybe_promote's new {prefix}shadow_status.json
write on every decision path (insufficient_n from no rows, insufficient_n
from <10 resolved records, continue, discard, promote) and confirms the
existing report/return-value/control-flow contract is unchanged.

Needs joblib (promote_challenger's config_v2.pkl rewrite) — not installed
on the dev Mac (see CLAUDE.md's two-machine table), so this whole file
skips there and runs for real on Jetson/CI with the full stack.
"""

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('joblib')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shadow
from shadow import challenger_prefix, evaluate_and_maybe_promote, shadow_status_file

EXPECTED_STATUS_KEYS = {
    'ts', 'n', 'min_obs', 'age_days', 'window_days', 'p_value', 'mean_d',
    'dm_stat', 'champ_hit_rate', 'chall_hit_rate', 'decision', 'detail',
}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, 'BASE_DIR', tmp_path)
    return tmp_path


def _mk_challenger(tmp_path, prefix=''):
    man = tmp_path / f'{challenger_prefix(prefix)}_model_v2.manifest.json'
    man.write_text(json.dumps({'saved_at': 'x', 'holdout': {}}))
    return int(man.stat().st_mtime)


def _log_rows(tmp_path, rows):
    with open(tmp_path / 'shadow_preds.jsonl', 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def _fake_closes(start, hours, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=hours, freq='h', tz='UTC')
    prices = 100 * np.cumprod(1 + drift + rng.normal(0, 0.002, hours))
    return pd.Series(prices, index=idx)


def _read_status(tmp_path):
    return json.loads((tmp_path / 'shadow_status.json').read_text())


# --- naming ---

def test_shadow_status_file_naming():
    assert shadow_status_file('').name == 'shadow_status.json'
    assert shadow_status_file('stock').name == 'stock_shadow_status.json'


# --- no challenger at all: nothing to evaluate, nothing written ---

def test_no_challenger_writes_no_status_file(sandbox):
    report = evaluate_and_maybe_promote('', 'crypto', api=object())
    assert report is None
    assert not shadow_status_file('').exists()


# --- challenger exists, zero rows logged yet ---

def test_no_rows_writes_insufficient_n_status(sandbox):
    _mk_challenger(sandbox)
    report = evaluate_and_maybe_promote('', 'crypto', api=object())
    assert report is None  # return contract unchanged

    status = _read_status(sandbox)
    assert set(status.keys()) == EXPECTED_STATUS_KEYS
    assert status['decision'] == 'insufficient_n'
    assert status['n'] == 0
    assert status['min_obs'] == shadow.MIN_OBS
    assert status['age_days'] == 0.0
    assert status['window_days'] == shadow.MIN_SHADOW_DAYS
    assert status['p_value'] is None
    assert status['mean_d'] is None
    assert status['dm_stat'] is None
    assert status['champ_hit_rate'] is None
    assert status['chall_hit_rate'] is None
    assert isinstance(status['detail'], str) and status['detail']
    assert isinstance(status['ts'], (int, float))


# --- rows logged but horizon never resolves (<10 records) ---

def test_insufficient_resolved_records_maps_to_insufficient_n(sandbox, monkeypatch):
    cm = _mk_challenger(sandbox)
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=5)
    closes = _fake_closes(start, 30, seed=6)  # only 30 bars
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    rows = [{'ts': (start + dt.timedelta(hours=20)).isoformat(),
             'sym': 'BTC/USD', 'champ': 0.1, 'chall': 0.1,
             'fb_champ': 24, 'fb_chall': 24, 'cm': cm}]  # 20+24 > 30 bars
    _log_rows(sandbox, rows)

    report = evaluate_and_maybe_promote('', 'crypto', api=object())
    assert report is not None
    assert report['n'] == 0
    assert report['decision'] == 'continue'  # internal control flow unchanged

    status = _read_status(sandbox)
    assert status['decision'] == 'insufficient_n'
    assert status['n'] == 0
    assert status['p_value'] is None
    assert status['mean_d'] is None
    assert status['dm_stat'] is None
    assert status['champ_hit_rate'] is None
    assert status['chall_hit_rate'] is None
    assert status['age_days'] > 0  # real age, computed from row timestamps
    assert status['window_days'] == shadow.MIN_SHADOW_DAYS


# --- accumulating with real (if inconclusive) stats ---

def test_continue_path_persists_real_stats(sandbox, monkeypatch):
    cm = _mk_challenger(sandbox)
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=3)  # age < MIN_SHADOW_DAYS
    closes = _fake_closes(start - dt.timedelta(hours=2), 24 * 7, seed=7)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    fb = 4
    rows = [{'ts': (start + dt.timedelta(hours=i)).isoformat(),
             'sym': 'BTC/USD', 'champ': 0.0, 'chall': 0.1,
             'fb_champ': fb, 'fb_chall': fb, 'cm': cm}
            for i in range(50)]  # >=10 resolved -> real stats computed
    _log_rows(sandbox, rows)

    report = evaluate_and_maybe_promote('', 'crypto', api=object())
    assert report is not None
    assert report['decision'] == 'continue'
    assert report['n'] >= 10

    status = _read_status(sandbox)
    assert status['decision'] == 'continue'
    assert status['n'] == report['n']
    assert status['p_value'] is not None
    assert status['mean_d'] is not None
    assert status['dm_stat'] is not None
    assert status['champ_hit_rate'] is not None
    assert status['chall_hit_rate'] is not None
    assert status['window_days'] == shadow.MIN_SHADOW_DAYS
    assert 0 <= status['age_days'] < shadow.MIN_SHADOW_DAYS


# --- terminal path: discard (with real stats that just don't clear the bar) ---

def test_discard_path_persists_final_decision(sandbox, monkeypatch):
    cm = _mk_challenger(sandbox)
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=29)  # age > MAX_SHADOW_DAYS
    closes = _fake_closes(start - dt.timedelta(hours=2), 24 * 32, seed=8)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    monkeypatch.setattr(shadow, '_notify', lambda *a, **k: None)
    fb = 4
    rows = [{'ts': (start + dt.timedelta(hours=i)).isoformat(),
             'sym': 'BTC/USD', 'champ': 0.0, 'chall': 0.1,
             'fb_champ': fb, 'fb_chall': fb, 'cm': cm}
            for i in range(50)]  # n=50 < MIN_OBS(200) -> forced discard
    _log_rows(sandbox, rows)

    report = evaluate_and_maybe_promote('', 'crypto', api=object())
    assert report is not None
    assert report['decision'] == 'discarded'
    assert not shadow.challenger_manifest('').exists()  # discard cleans up
    assert not shadow.shadow_log_file('').exists()

    status = _read_status(sandbox)
    assert status['decision'] == 'discard'
    assert status['age_days'] >= shadow.MAX_SHADOW_DAYS
    assert status['window_days'] == shadow.MAX_SHADOW_DAYS
    # Real stats WERE computed here (50 resolved records >= 10) -> passed
    # through, not nulled; distinguishes this from the insufficient-data
    # forced-discard case below.
    assert status['p_value'] is not None
    assert status['mean_d'] is not None
    assert status['dm_stat'] is not None
    assert isinstance(status['detail'], str) and 'discard' in status['detail'].lower()


def test_discard_with_insufficient_stats_nulls_placeholders(sandbox, monkeypatch):
    cm = _mk_challenger(sandbox)
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=29)
    closes = _fake_closes(start, 20, seed=12)  # too short: horizon never resolves
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    monkeypatch.setattr(shadow, '_notify', lambda *a, **k: None)
    rows = [{'ts': (start + dt.timedelta(hours=5)).isoformat(),
             'sym': 'BTC/USD', 'champ': 0.1, 'chall': 0.1,
             'fb_champ': 24, 'fb_chall': 24, 'cm': cm}]  # 5+24=29 > 20 bars
    _log_rows(sandbox, rows)

    report = evaluate_and_maybe_promote('', 'crypto', api=object())
    assert report is not None
    assert report['n'] == 0
    assert report['decision'] == 'discarded'  # age>=28 forces discard despite n=0

    status = _read_status(sandbox)
    # The true terminal outcome (discard) wins over the insufficient_n
    # label — the challenger really was deleted, the GUI must see that.
    assert status['decision'] == 'discard'
    assert status['p_value'] is None  # placeholders nulled, not passed through
    assert status['mean_d'] is None
    assert status['dm_stat'] is None


# --- terminal path: promote ---

def test_promote_path_persists_final_decision(sandbox, monkeypatch):
    import joblib
    p = ''
    for suffix in shadow._ARTIFACT_SUFFIXES:
        path = sandbox / suffix
        if suffix == 'config_v2.pkl':
            joblib.dump({'prefix': '', 'forward_bars': 24}, path)
        else:
            path.write_text(f'champion-{suffix}')
    (sandbox / 'model_v2.manifest.json').write_text(json.dumps({'old': True}))

    cp = challenger_prefix(p)
    for suffix in shadow._ARTIFACT_SUFFIXES:
        path = sandbox / f'{cp}_{suffix}'
        if suffix == 'config_v2.pkl':
            joblib.dump({'prefix': cp, 'forward_bars': 24}, path)
        else:
            path.write_text(f'challenger-{suffix}')
    (sandbox / f'{cp}_model_v2.manifest.json').write_text(
        json.dumps({'saved_at': 'now', 'holdout': {'sharpe': 2.0}}))
    cm = int((sandbox / f'{cp}_model_v2.manifest.json').stat().st_mtime)

    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=20)  # age > MIN_SHADOW_DAYS
    fb = 24
    closes = _fake_closes(start - dt.timedelta(hours=1), 24 * 20, seed=11)

    rows = []
    for i in range(250):  # n >= MIN_OBS
        ts = start + dt.timedelta(hours=i)
        idx = int(closes.index.searchsorted(pd.Timestamp(ts)))
        c0, c1 = float(closes.iloc[idx]), float(closes.iloc[idx + fb])
        realized = (c1 - c0) / c0 * 100.0
        rows.append({'ts': ts.isoformat(), 'sym': 'BTC/USD',
                     # champ = no-skill baseline; chall = (near) perfect
                     'champ': 0.0, 'chall': round(realized, 6),
                     'fb_champ': fb, 'fb_chall': fb, 'cm': cm})
    _log_rows(sandbox, rows)

    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    monkeypatch.setattr(shadow, '_notify', lambda *a, **k: None)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: None)

    report = evaluate_and_maybe_promote(p, 'crypto', api=object())
    assert report is not None
    assert report['n'] >= shadow.MIN_OBS
    assert report['decision'] == 'promoted'

    status = _read_status(sandbox)
    assert status['decision'] == 'promote'
    assert status['n'] >= shadow.MIN_OBS
    assert status['p_value'] is not None and status['p_value'] < shadow.EARLY_PROMOTE_P
    assert status['champ_hit_rate'] is not None
    assert status['chall_hit_rate'] is not None
    assert status['age_days'] >= shadow.MIN_SHADOW_DAYS
    assert status['window_days'] == shadow.MAX_SHADOW_DAYS
    assert isinstance(status['detail'], str) and 'promot' in status['detail'].lower()


# --- write failure must never affect the evaluation/promotion outcome ---

def test_status_write_failure_is_swallowed(sandbox, monkeypatch):
    _mk_challenger(sandbox)
    monkeypatch.setattr(
        shadow, 'shadow_status_file',
        lambda prefix: Path('/nonexistent_dir_xyz_shadow/status.json'))
    # Must not raise despite the write target being unwritable, and the
    # evaluation result (None: no rows logged) must be unaffected.
    report = evaluate_and_maybe_promote('', 'crypto', api=object())
    assert report is None
