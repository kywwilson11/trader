"""Tests for challenger shadow mode (DM-HLN test, evaluation, promotion)."""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shadow
from shadow import (challenger_prefix, dm_hln, evaluate_shadow,
                    promote_challenger)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, 'BASE_DIR', tmp_path)
    return tmp_path


# --- naming ---

def test_challenger_prefix_composition():
    assert challenger_prefix('') == 'challenger'
    assert challenger_prefix('stock') == 'stock_challenger'


# --- DM-HLN statistic ---

def test_dm_equal_models_high_p():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 1, 500)  # no systematic difference
    _, p = dm_hln(d, h=24)
    assert p > 0.10


def test_dm_better_challenger_low_p():
    rng = np.random.default_rng(1)
    d = rng.normal(0.5, 1, 500)  # challenger consistently better
    stat, p = dm_hln(d, h=24)
    assert stat > 0 and p < 0.01


def test_dm_worse_challenger_p_near_one():
    rng = np.random.default_rng(2)
    d = rng.normal(-0.5, 1, 500)
    _, p = dm_hln(d, h=24)
    assert p > 0.99


def test_dm_hln_correction_shrinks_stat():
    # Same data, larger horizon -> larger NW lag + HLN penalty -> |stat| falls
    rng = np.random.default_rng(3)
    d = rng.normal(0.3, 1, 300)
    s1, _ = dm_hln(d, h=1)
    s24, _ = dm_hln(d, h=24)
    assert abs(s24) < abs(s1)


def test_dm_degenerate_inputs():
    assert dm_hln(np.array([]), h=24) == (0.0, 1.0)
    assert dm_hln(np.zeros(50), h=24) == (0.0, 1.0)  # zero variance


# --- evaluation over fake bars ---

def _mk_challenger(tmp_path, prefix='', mtime_marker=True):
    man = tmp_path / f'{challenger_prefix(prefix)}_model_v2.manifest.json'
    man.write_text(json.dumps({'saved_at': 'x', 'holdout': {}}))
    return int(man.stat().st_mtime)


def _log_rows(tmp_path, prefix, rows):
    with open(tmp_path / 'shadow_preds.jsonl', 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def _fake_closes(start, hours, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=hours, freq='h', tz='UTC')
    prices = 100 * np.cumprod(1 + drift + rng.normal(0, 0.002, hours))
    return pd.Series(prices, index=idx)


def test_evaluate_shadow_prefers_accurate_model(sandbox, monkeypatch):
    cm = _mk_challenger(sandbox)
    start = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    closes = _fake_closes(start, 800, seed=4)

    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    fb = 24
    rows = []
    for i in range(0, 600):
        ts = start + dt.timedelta(hours=i)
        c0, c1 = float(closes.iloc[i]), float(closes.iloc[i + fb])
        realized = (c1 - c0) / c0 * 100
        rows.append({'ts': ts.isoformat(), 'sym': 'BTC/USD',
                     # challenger = realized + small noise; champ = noise only
                     'champ': round(np.random.default_rng(i).normal(0, 1), 6),
                     'chall': round(realized + np.random.default_rng(i + 1).normal(0, 0.05), 6),
                     'fb_champ': fb, 'fb_chall': fb, 'cm': cm})
    _log_rows(sandbox, '', rows)

    report = evaluate_shadow('', api=object())
    assert report is not None
    assert report['n'] > 500
    assert report['p'] < 0.01            # challenger clearly better
    assert report['hit_chall'] > report['hit_champ']


def test_evaluate_ignores_stale_challenger_rows(sandbox, monkeypatch):
    cm = _mk_challenger(sandbox)
    start = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    closes = _fake_closes(start, 100, seed=5)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    rows = [{'ts': start.isoformat(), 'sym': 'BTC/USD', 'champ': 0.1,
             'chall': 0.1, 'fb_champ': 24, 'fb_chall': 24,
             'cm': cm - 999}]  # previous challenger's row
    _log_rows(sandbox, '', rows)
    assert evaluate_shadow('', api=object()) is None


def test_unresolved_horizons_excluded(sandbox, monkeypatch):
    cm = _mk_challenger(sandbox)
    start = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    closes = _fake_closes(start, 30, seed=6)  # only 30 bars
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    rows = [{'ts': (start + dt.timedelta(hours=20)).isoformat(),
             'sym': 'BTC/USD', 'champ': 0.1, 'chall': 0.1,
             'fb_champ': 24, 'fb_chall': 24, 'cm': cm}]  # 20+24 > 30
    _log_rows(sandbox, '', rows)
    report = evaluate_shadow('', api=object())
    assert report is not None and report['n'] == 0


# --- promotion mechanics ---

def _mk_full_challenger(tmp_path, prefix=''):
    cp = challenger_prefix(prefix)
    import joblib
    for suffix in shadow._ARTIFACT_SUFFIXES:
        path = tmp_path / f'{cp}_{suffix}'
        if suffix == 'config_v2.pkl':
            joblib.dump({'prefix': cp, 'forward_bars': 12}, path)
        else:
            path.write_text(f'challenger-{suffix}')
    (tmp_path / f'{cp}_model_v2.manifest.json').write_text(
        json.dumps({'saved_at': 'now', 'holdout': {'sharpe': 2.0}}))


def test_promote_copies_stack_and_rewrites_prefix(sandbox):
    import joblib
    p = ''
    # Existing champion artifacts
    for suffix in shadow._ARTIFACT_SUFFIXES:
        path = sandbox / f'{suffix}'
        if suffix == 'config_v2.pkl':
            joblib.dump({'prefix': '', 'forward_bars': 24}, path)
        else:
            path.write_text(f'champion-{suffix}')
    (sandbox / 'model_v2.manifest.json').write_text(json.dumps({'old': True}))
    for suffix in shadow._STALE_META_SUFFIXES:
        (sandbox / suffix).write_text('stale-meta')
    _mk_full_challenger(sandbox, p)

    import subprocess
    real_popen = subprocess.Popen
    calls = []
    subprocess.Popen = lambda *a, **k: calls.append(a) or None
    try:
        ok = promote_challenger(p, report={'p': 0.01})
    finally:
        subprocess.Popen = real_popen
    assert ok

    # Champion files replaced with challenger content; .prev backups kept
    assert (sandbox / 'model_v2.pth').read_text() == 'challenger-model_v2.pth'
    assert (sandbox / 'model_v2.pth.prev').read_text() == 'champion-model_v2.pth'
    # Config prefix rewritten to champion namespace
    cfg = joblib.load(sandbox / 'config_v2.pkl')
    assert cfg['prefix'] == '' and cfg['forward_bars'] == 12
    # Manifest carries the promotion record
    man = json.loads((sandbox / 'model_v2.manifest.json').read_text())
    assert 'promoted_from_shadow' in man and man['shadow_report']['p'] == 0.01
    # Stale champion meta deleted (gate fails open until retrain lands)
    for suffix in shadow._STALE_META_SUFFIXES:
        assert not (sandbox / suffix).exists()
    # Meta retrain kicked off; challenger slot cleaned
    assert calls
    assert not (sandbox / 'challenger_model_v2.pth').exists()
    assert not shadow.challenger_manifest(p).exists()


def test_promote_aborts_on_missing_core_artifact(sandbox):
    _mk_full_challenger(sandbox, '')
    (sandbox / 'challenger_scaler_v2.pkl').unlink()
    (sandbox / 'model_v2.pth').write_text('champion')
    assert promote_challenger('') is False
    assert (sandbox / 'model_v2.pth').read_text() == 'champion'  # untouched
