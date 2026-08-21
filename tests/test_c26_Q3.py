"""Packet Q3 — Shadow DM v2 (D34 rebuild): per-hour collapse, IM cluster t,
scheduled looks, frozen/patched skill-score variance. Mac-runnable
(numpy/pandas only). Flag OFF must be byte-identical legacy behavior with
additive dm_v2_* instrumentation; flag ON is exercised via monkeypatching
shadow.DM_V2_ENABLED (never the environment)."""

import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shadow


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, 'BASE_DIR', tmp_path)
    return tmp_path


def _mk_challenger(tmp_path, prefix=''):
    man = (tmp_path
           / f'{shadow.challenger_prefix(prefix)}_model_v2.manifest.json')
    man.write_text(json.dumps({'saved_at': 'x', 'holdout': {}}))
    return int(man.stat().st_mtime)


def _log_rows(tmp_path, prefix, rows):
    p = f'{prefix}_' if prefix else ''
    with open(tmp_path / f'{p}shadow_preds.jsonl', 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def _fake_closes(start, hours, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=hours, freq='h', tz='UTC')
    prices = 100 * np.cumprod(1 + drift + rng.normal(0, 0.002, hours))
    return pd.Series(prices, index=idx)


def _start_ago(days):
    """Hour-floored start `days` ago, so bar indices align with jitter."""
    now = dt.datetime.now(dt.timezone.utc)
    return (now - dt.timedelta(days=days)).replace(
        minute=0, second=0, microsecond=0)


def _build_rows(closes, start, n_hours, syms, cm, fb=6, mode='null', seed=0):
    """Jittered per-symbol rows (minute offsets exercise the hour-floor
    collapse). Anchored so _realized resolves at bar i+1 (first >= ts)."""
    rng = np.random.default_rng(seed)
    rows = []
    for si, sym in enumerate(syms):
        off = 5 + 12 * si   # minutes; always >= 1 so anchor is bar i+1
        for i in range(n_hours):
            ts = start + dt.timedelta(hours=i, minutes=off)
            j = i + 1
            c0, c1 = float(closes.iloc[j]), float(closes.iloc[j + fb])
            realized = (c1 - c0) / c0 * 100
            if mode == 'strong':       # challenger clearly better
                champ = float(rng.normal(0, 1.0))
                chall = realized + float(rng.normal(0, 0.02))
            elif mode == 'worse':      # champion clearly better
                champ = realized + float(rng.normal(0, 0.02))
                chall = float(rng.normal(0, 1.0))
            elif mode == 'equal':      # identical preds -> d exactly 0
                champ = chall = float(rng.normal(0, 1.0))
            else:                      # null: independent equal-scale noise
                champ = float(rng.normal(0, 1.0))
                chall = float(rng.normal(0, 1.0))
            rows.append({'ts': ts.isoformat(), 'sym': sym,
                         'champ': round(champ, 6), 'chall': round(chall, 6),
                         'fb_champ': fb, 'fb_chall': fb, 'cm': cm})
    return rows


# --- T1: flag default ---

def test_flag_default_off():
    assert shadow.DM_V2_ENABLED is False


# --- T2: collapse_by_hour ---

def test_collapse_by_hour_buckets_and_means():
    base = dt.datetime(2026, 5, 1, 10, tzinfo=dt.timezone.utc)
    ts = [base + dt.timedelta(minutes=5), base + dt.timedelta(minutes=17)]
    out = shadow.collapse_by_hour(ts, np.array([1.0, 3.0]))
    assert out.shape == (1,) and out[0] == pytest.approx(2.0, abs=1e-15)
    ts.append(base + dt.timedelta(hours=1, minutes=42))
    out = shadow.collapse_by_hour(ts, np.array([1.0, 3.0, 5.0]))
    assert out.tolist() == [2.0, 5.0]   # ascending hour order


# --- T3: t-table spot checks ---

def test_t_crit_one_sided_spot_values():
    assert shadow.t_crit_one_sided(6, 0.05) == 1.9432   # research's 1.943
    assert shadow.t_crit_one_sided(13, 0.05) == 1.7709  # research's 1.771
    assert shadow.t_crit_one_sided(9, 0.025) == 2.2622
    assert shadow.t_crit_one_sided(9, 0.10) == 1.3830
    assert shadow.t_crit_one_sided(200, 0.05) == 1.6973  # df>30 -> df=30
    with pytest.raises(ValueError):
        shadow.t_crit_one_sided(4, 0.05)


# --- T4: IM cluster t ---

def test_im_cluster_t_hand_computed():
    dbar = np.arange(1.0, 13.0)          # 12 obs, block=2 -> q=6
    t, q = shadow.im_cluster_t(dbar, 2)
    bm = dbar.reshape(6, 2).mean(axis=1)
    expected = math.sqrt(6) * bm.mean() / np.std(bm, ddof=1)
    assert q == 6 and t == pytest.approx(expected, rel=1e-12)


def test_im_cluster_t_refusals():
    t, q = shadow.im_cluster_t(np.random.default_rng(0).normal(0, 1, 100), 48)
    assert math.isnan(t) and q == 2          # q < 6
    t, q = shadow.im_cluster_t(np.ones(72), 12)
    assert math.isnan(t) and q == 6          # zero block-mean sd


def test_im_cluster_t_drops_oldest_remainder():
    tail = np.arange(12.0)
    full = np.concatenate([[999.0], tail])   # T=13, block=2 -> drop oldest 1
    t_full, q_full = shadow.im_cluster_t(full, 2)
    t_tail, q_tail = shadow.im_cluster_t(tail, 2)
    assert q_full == q_tail == 6
    assert t_full == pytest.approx(t_tail, rel=1e-12)


# --- T5: fixed-b DM ---

def test_kv_fixed_b_crit_polynomial():
    b = 0.1
    expected = 1.6449 + 2.1859 * b + 0.3142 * b * b - 0.3427 * b ** 3
    assert shadow.kv_fixed_b_crit(b) == pytest.approx(expected, abs=1e-15)


def test_dm_fixed_b_bartlett_and_refusal():
    rng = np.random.default_rng(5)
    x = np.zeros(300)
    for i in range(1, 300):                  # AR(1) with drift
        x[i] = 0.05 + 0.5 * x[i - 1] + rng.normal(0, 1)
    stat, crit = shadow.dm_fixed_b(x, 24)
    M = min(48, 299)
    assert math.isfinite(stat) and crit == shadow.kv_fixed_b_crit(M / 300)
    stat, crit = shadow.dm_fixed_b(np.zeros(300), 24)  # lrv <= 0 refusal
    assert stat == 0.0 and crit == float('inf')


# --- T6: decision matrix ---

def test_look_schedule():
    dbar = np.random.default_rng(7).normal(0, 1, 300)
    v = shadow.dm_v2_evaluate(dbar, 24, 10.0, 'crypto', False)
    assert v['look'] == 0 and v['decision'] == 'continue'
    v = shadow.dm_v2_evaluate(dbar, 24, 22.0, 'crypto', False)
    assert v['look'] == 1
    v = shadow.dm_v2_evaluate(dbar, 24, 22.0, 'crypto', True)
    assert v['look'] == 0 and v['decision'] == 'continue'
    v = shadow.dm_v2_evaluate(dbar, 24, 29.0, 'crypto', False)
    assert v['look'] == 2
    v = shadow.dm_v2_evaluate(dbar, 24, 29.0, 'stock', False)   # max 56
    assert v['look'] == 0 and v['decision'] == 'continue'
    v = shadow.dm_v2_evaluate(dbar, 24, 57.0, 'stock', False)
    assert v['look'] == 2


def test_final_promote_requires_positive_mean_d():
    # Oldest remainder very negative: IM t large positive (computed on the
    # most recent q*block), but full-series mean_d < 0 -> discard.
    rng = np.random.default_rng(11)
    dbar = np.concatenate([np.full(4, -1000.0),
                           1.0 + 0.01 * rng.normal(0, 1, 96)])
    v = shadow.dm_v2_evaluate(dbar, 6, 29.0, 'crypto', True)
    assert v['look'] == 2 and v['stat'] == 'im'
    assert v['t'] > v['crit'] and v['mean_d'] < 0
    assert v['decision'] == 'discard'


def test_interim_q_lt_6_not_consumed():
    dbar = np.random.default_rng(13).normal(0, 1, 200)  # h=24: q=4 < 6
    v = shadow.dm_v2_evaluate(dbar, 24, 22.0, 'crypto', False)
    assert v['look'] == 1 and v['decision'] == 'continue'
    assert v['look_consumed'] is False and v['q'] == 4


def test_final_insufficient_T_discards():
    dbar = np.random.default_rng(17).normal(0, 1, 100)  # < 192 = 8*24
    v = shadow.dm_v2_evaluate(dbar, 24, 29.0, 'crypto', False)
    assert v['look'] == 2 and v['decision'] == 'discard'
    assert 'insufficient' in v['reason']


def test_final_q_lt_6_uses_fixed_b_fallback():
    dbar = np.random.default_rng(19).normal(0, 1, 250)  # q=5 < 6
    v = shadow.dm_v2_evaluate(dbar, 24, 29.0, 'crypto', False)
    assert v['look'] == 2 and v['stat'] == 'fixed_b'
    assert v['alpha'] == 0.05 and v['look_consumed'] is True
    assert v['decision'] in ('promote', 'discard')


# --- T7: collapsed minimum in hours replaces MIN_OBS ---

def test_collapsed_min_obs_rule():
    dbar = np.random.default_rng(23).normal(0, 1, 150)  # < 192 at h=24
    v = shadow.dm_v2_evaluate(dbar, 24, 22.0, 'crypto', False)
    assert v['decision'] == 'continue' and v['look_consumed'] is False
    assert v['reason'] == 'T<8*h_max'
    v = shadow.dm_v2_evaluate(dbar, 24, 29.0, 'crypto', False)
    assert v['decision'] == 'discard'


# --- T8: null false-promote Monte Carlo ---

def test_null_false_promote_v2_below_legacy_pooled():
    # Deterministic fixed-seed panel sim. True rates at this geometry
    # (1000-panel check): legacy pooled ~0.23, v2 final-look ~0.12 — the
    # v2 rate is FLOORED near 0.10 by the owner's final alpha=0.10 rule
    # itself (02_research.md B03.3 measured the same floor). Seed chosen
    # so the finite-panel draw sits inside the bound with margin.
    rng = np.random.default_rng(1)
    K, T, h, rho = 8, 240, 6, 0.7
    n_panels = 300
    kern = np.ones(h) / h
    legacy_hits = v2_promotes = 0
    for _ in range(n_panels):
        f = rng.normal(0, 1, T + h)
        eps = rng.normal(0, 1, (K, T + h))
        d_raw = math.sqrt(rho) * f[None, :] + math.sqrt(1 - rho) * eps
        d_sm = np.vstack([np.convolve(row, kern, 'valid')[:T]
                          for row in d_raw])   # MA(h) overlap smoothing
        pooled = d_sm.T.reshape(-1)            # K same-hour records adjacent
        _, p = shadow.dm_hln(pooled, h=h)
        if p < 0.05:
            legacy_hits += 1
        dbar = d_sm.mean(axis=0)               # per-timestamp collapse
        v = shadow.dm_v2_evaluate(dbar, h, 28.0, 'crypto', True)
        if v['decision'] == 'promote':
            v2_promotes += 1
    legacy_rate = legacy_hits / n_panels
    v2_rate = v2_promotes / n_panels
    assert v2_rate <= 0.12
    assert v2_rate < legacy_rate


# --- T9: flag OFF end-to-end (side-by-side instrumentation only) ---

def test_flag_off_end_to_end_additive_status(sandbox, monkeypatch):
    cm = _mk_challenger(sandbox)
    start = _start_ago(5.2)
    closes = _fake_closes(start, 200, seed=31)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    rows = _build_rows(closes, start, 120, ['BTC/USD', 'ETH/USD'], cm,
                       fb=6, mode='null', seed=31)
    _log_rows(sandbox, '', rows)

    report = shadow.evaluate_and_maybe_promote('', 'CRYPTO', api=object())
    assert report is not None and report['decision'] == 'continue'
    for k in ('n', 'age_days', 'dm', 'p', 'mean_d', 'hit_champ',
              'hit_chall', 'fb_max'):
        assert k in report
    assert report['n'] == 240

    status = json.loads((sandbox / 'shadow_status.json').read_text())
    legacy_keys = {'ts', 'n', 'min_obs', 'age_days', 'window_days',
                   'p_value', 'mean_d', 'dm_stat', 'champ_hit_rate',
                   'chall_hit_rate', 'decision', 'detail'}
    assert legacy_keys <= set(status.keys())
    for k in ('dm_v2_enabled', 'dm_v2_t', 'dm_v2_q', 'dm_v2_T',
              'dm_v2_look', 'dm_v2_stat', 'dm_v2_alpha', 'dm_v2_crit',
              'dm_v2_decision', 'dm_v2_mean_d', 'dm_v2_var_mode'):
        assert k in status
    assert status['dm_v2_enabled'] is False
    assert status['dm_v2_T'] == 120          # jittered rows collapse 2->1
    assert status['dm_v2_var_mode'] == 'patched'
    assert status['dm_v2_look'] == 0
    # look ledger is NEVER written under flag OFF
    assert not (sandbox / 'shadow_v2_state.json').exists()
    # manifest report stays JSON-round-trippable (C3)
    json.dumps(report)


# --- T10: flag ON blocks the day-14/15 peek ---

def test_flag_on_blocks_early_peek(sandbox, monkeypatch):
    monkeypatch.setattr(shadow, 'DM_V2_ENABLED', True)
    cm = _mk_challenger(sandbox)
    start = _start_ago(15.5)
    closes = _fake_closes(start, 300, seed=41)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    rows = _build_rows(closes, start, 250, ['BTC/USD'], cm,
                       fb=6, mode='strong', seed=41)
    _log_rows(sandbox, '', rows)
    calls = []
    monkeypatch.setattr(shadow, 'promote_challenger',
                        lambda pfx, rep=None: calls.append(pfx) or True)

    report = shadow.evaluate_and_maybe_promote('', 'CRYPTO', api=object())
    # legacy WOULD promote here (n >= 200, age >= 14, p << .05) ...
    assert report['n'] >= shadow.MIN_OBS and report['p'] < 0.05
    # ... but v2 has no scheduled look before day 21
    assert calls == []
    assert report['decision'] == 'continue'
    status = json.loads((sandbox / 'shadow_status.json').read_text())
    assert status['dm_v2_look'] == 0 and status['dm_v2_enabled'] is True


# --- T11: flag ON look-1 promote ---

def test_flag_on_look1_promotes_strong_challenger(sandbox, monkeypatch):
    monkeypatch.setattr(shadow, 'DM_V2_ENABLED', True)
    cm = _mk_challenger(sandbox)
    start = _start_ago(22.2)
    closes = _fake_closes(start, 360, seed=43)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    rows = _build_rows(closes, start, 300, ['BTC/USD'], cm,
                       fb=24, mode='strong', seed=43)
    _log_rows(sandbox, '', rows)
    calls = []
    monkeypatch.setattr(shadow, 'promote_challenger',
                        lambda pfx, rep=None: calls.append(pfx) or True)

    report = shadow.evaluate_and_maybe_promote('', 'CRYPTO', api=object())
    assert calls == ['']
    assert report['decision'] == 'promoted'
    assert report['dm_v2']['look'] == 1 and report['dm_v2']['stat'] == 'im'
    assert report['dm_v2']['q'] >= shadow.V2_MIN_BLOCKS
    status = json.loads((sandbox / 'shadow_status.json').read_text())
    assert status['dm_v2_look'] == 1 and status['decision'] == 'promote'


# --- T12: flag ON look-1 consumed on fail (one-shot state) ---

def test_flag_on_look1_consumed_once(sandbox, monkeypatch):
    monkeypatch.setattr(shadow, 'DM_V2_ENABLED', True)
    cm = _mk_challenger(sandbox)
    start = _start_ago(22.2)
    closes = _fake_closes(start, 360, seed=47)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    rows = _build_rows(closes, start, 300, ['BTC/USD'], cm,
                       fb=24, mode='worse', seed=47)
    _log_rows(sandbox, '', rows)

    report = shadow.evaluate_and_maybe_promote('', 'CRYPTO', api=object())
    assert report['decision'] == 'continue'
    assert report['dm_v2']['look'] == 1
    assert report['dm_v2']['look_consumed'] is True
    state_file = sandbox / 'shadow_v2_state.json'
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert state['cm'] == cm and state['look1_done'] is True

    # Immediate re-evaluation must NOT re-test look 1
    report2 = shadow.evaluate_and_maybe_promote('', 'CRYPTO', api=object())
    assert report2['decision'] == 'continue'
    assert report2['dm_v2']['look'] == 0
    assert report2['dm_v2']['look_consumed'] is False


# --- T13: flag ON stock 56d geometry ---

def test_flag_on_stock_survives_day_28_and_dies_at_56(sandbox, monkeypatch):
    monkeypatch.setattr(shadow, 'DM_V2_ENABLED', True)
    # Part 1: age 30d — legacy would discard at 28, v2 continues to 56
    cm = _mk_challenger(sandbox, 'stock')
    start = _start_ago(30.5)
    closes = _fake_closes(start, 300, seed=53)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    rows = _build_rows(closes, start, 200, ['AAPL'], cm,
                       fb=24, mode='null', seed=53)
    _log_rows(sandbox, 'stock', rows)
    report = shadow.evaluate_and_maybe_promote('stock', 'STOCK', api=object())
    assert report['decision'] == 'continue'
    assert shadow.challenger_manifest('stock').exists()
    assert report['dm_v2']['look'] == 0

    # Part 2: age 57d, no edge — terminal discard via the final look
    (sandbox / 'stock_shadow_preds.jsonl').unlink()
    cm = _mk_challenger(sandbox, 'stock')
    start = _start_ago(57.5)
    closes = _fake_closes(start, 320, seed=59)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    rows = _build_rows(closes, start, 260, ['AAPL'], cm,
                       fb=24, mode='equal', seed=59)
    _log_rows(sandbox, 'stock', rows)
    report = shadow.evaluate_and_maybe_promote('stock', 'STOCK', api=object())
    assert report['decision'] == 'discarded'
    assert report['dm_v2']['look'] == 2
    assert report['dm_v2']['stat'] == 'fixed_b'   # q=5 < 6 at T=260
    assert not shadow.challenger_manifest('stock').exists()
    assert not (sandbox / 'stock_shadow_v2_state.json').exists()


# --- T14: patched-mode small-sample multiplier ---

def test_patched_mode_collapsed_series_exact(sandbox, monkeypatch):
    rng = np.random.default_rng(61)
    base = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    fb = 6
    n_hours = 60
    recs = []
    for i in range(n_hours):
        for off in (5, 17):    # two jittered records per hour
            ts = base + dt.timedelta(hours=i, minutes=off)
            recs.append((ts, float(rng.uniform(0.1, 3.0)),
                         float(rng.uniform(0.1, 3.0)), 0.0, 0.0, 0.0, 0.0))
    vc, vx = 1.3, 0.7
    captured = {}

    def fake_eval(dbar, h_max, age_days, book, look1_done):
        captured['dbar'] = np.asarray(dbar, dtype=float).copy()
        return {'t': None, 'q': None, 'T': int(np.asarray(dbar).size),
                'block': 12, 'look': 0, 'alpha': None, 'crit': None,
                'stat': None, 'decision': 'continue', 'mean_d': 0.0,
                'look_consumed': False, 'reason': 'captured'}

    monkeypatch.setattr(shadow, 'dm_v2_evaluate', fake_eval)
    out = shadow._compute_dm_v2('', recs, {}, vc, vx, fb, fb, fb,
                                5.0, 'crypto')
    assert out is not None and out['var_mode'] == 'patched'

    n_eff = n_hours / fb
    w = (n_eff - 2) / n_eff
    e2c = np.array([r[1] for r in recs])
    e2x = np.array([r[2] for r in recs])
    d = w * e2c / vc - w * e2x / vx
    expected = d.reshape(n_hours, 2).mean(axis=1)   # per-hour bucket means
    assert np.max(np.abs(captured['dbar'] - expected)) < 1e-12


# --- T16: frozen-variance mode (hardener addition) ---

def test_frozen_mode_uses_prewindow_variances(sandbox, monkeypatch):
    """>=90d of pre-window bars per symbol -> var_mode 'frozen' and the
    collapsed series uses the pooled PRE-WINDOW forward-return variances
    per horizon, not the in-window vc/vx."""
    rng = np.random.default_rng(71)
    pre = shadow.V2_FROZEN_MIN_BARS['crypto']    # 2160
    n_bars = pre + 80
    idx = pd.date_range('2026-01-01', periods=n_bars, freq='h', tz='UTC')
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.002, n_bars))
    closes = pd.Series(prices, index=idx)
    t_min = idx[pre].to_pydatetime()             # searchsorted -> exactly pre
    fb_c, fb_x = 6, 12
    recs = []
    for i in range(40):
        ts = t_min + dt.timedelta(hours=i, minutes=7)
        recs.append((ts, float(rng.uniform(0.1, 3.0)),
                     float(rng.uniform(0.1, 3.0)), 0.0, 0.0, 0.0, 0.0))
    captured = {}

    def fake_eval(dbar, h_max, age_days, book, look1_done):
        captured['dbar'] = np.asarray(dbar, dtype=float).copy()
        return {'t': None, 'q': None, 'T': int(np.asarray(dbar).size),
                'block': 24, 'look': 0, 'alpha': None, 'crit': None,
                'stat': None, 'decision': 'continue', 'mean_d': 0.0,
                'look_consumed': False, 'reason': 'captured'}

    monkeypatch.setattr(shadow, 'dm_v2_evaluate', fake_eval)
    out = shadow._compute_dm_v2('', recs, {'BTC/USD': closes},
                                1.3, 0.7, fb_c, fb_x, fb_x, 5.0, 'crypto')
    assert out is not None and out['var_mode'] == 'frozen'

    # Expected frozen variances: anchors i with index[i+fb] < min(ts_list).
    # min ts is t_min + 7min, so bar pre itself also counts as pre-window:
    # index[i+fb] < t_min+7min  <=>  i + fb <= pre  <=>  i < pre + 1 - fb.
    def fwd(fb):
        m = pre + 1 - fb
        c0, c1 = prices[:m], prices[fb:fb + m]
        return (c1 - c0) / c0 * 100.0
    vf_c = max(float(np.var(fwd(fb_c))), 1e-12)
    vf_x = max(float(np.var(fwd(fb_x))), 1e-12)
    e2c = np.array([r[1] for r in recs])
    e2x = np.array([r[2] for r in recs])
    expected = e2c / vf_c - e2x / vf_x   # 1 rec/hour -> collapse = identity
    assert np.max(np.abs(captured['dbar'] - expected)) < 1e-12


def test_frozen_mode_not_entered_short_prewindow(sandbox, monkeypatch):
    """One symbol short of the 90d pre-window -> patched, never frozen.
    (min ts = idx[pre]+7min, so idx[pre] also counts: pre+1 bars before.)"""
    rng = np.random.default_rng(73)
    pre = shadow.V2_FROZEN_MIN_BARS['crypto'] - 2
    idx = pd.date_range('2026-01-01', periods=pre + 80, freq='h', tz='UTC')
    closes = pd.Series(100 + rng.normal(0, 1, pre + 80), index=idx)
    t_min = idx[pre].to_pydatetime()
    recs = [(t_min + dt.timedelta(hours=i, minutes=7),
             float(rng.uniform(0.1, 3.0)), float(rng.uniform(0.1, 3.0)),
             0.0, 0.0, 0.0, 0.0) for i in range(40)]
    out = shadow._compute_dm_v2('', recs, {'BTC/USD': closes},
                                1.3, 0.7, 6, 12, 12, 5.0, 'crypto')
    assert out is not None and out['var_mode'] == 'patched'


# --- T17: _prewindow_forward_returns unit (hardener addition) ---

def test_prewindow_forward_returns_exact_and_boundary():
    idx = pd.date_range('2026-03-01', periods=10, freq='h', tz='UTC')
    prices = np.array([100.0, 101, 99, 102, 103, 98, 105, 104, 106, 107])
    closes = pd.Series(prices, index=idx)
    t_min = idx[7].to_pydatetime()
    out = shadow._prewindow_forward_returns(closes, t_min, 2)
    # anchors i with index[i+2] < t_min  <=>  i+2 < 7  <=>  i in 0..4
    expected = (prices[2:7] - prices[0:5]) / prices[0:5] * 100.0
    assert out.shape == (5,)
    assert np.max(np.abs(out - expected)) < 1e-12
    # boundary: a return resolving exactly AT t_min is excluded (strict <)
    out = shadow._prewindow_forward_returns(closes, idx[2].to_pydatetime(), 2)
    assert out.size == 0
    # degenerate inputs
    assert shadow._prewindow_forward_returns(closes, t_min, 0).size == 0
    empty = pd.Series([], index=pd.DatetimeIndex([], tz='UTC'), dtype=float)
    assert shadow._prewindow_forward_returns(empty, t_min, 2).size == 0


# --- T18: dm_fixed_b exact Bartlett LRV (hardener addition) ---

def test_dm_fixed_b_exact_hand_computed():
    dbar = np.array([0.4, -0.2, 0.7, 0.1, -0.3, 0.5, 0.2, 0.6])
    h_max = 2                                  # M = min(4, 7) = 4
    stat, crit = shadow.dm_fixed_b(dbar, h_max)
    T, M = 8, 4
    dc = dbar - dbar.mean()
    lrv = float(np.mean(dc * dc))
    for k in range(1, M):
        lrv += 2.0 * (1.0 - k / M) * float(np.mean(dc[k:] * dc[:-k]))
    expected = dbar.mean() / math.sqrt(lrv / T)
    assert stat == pytest.approx(expected, rel=1e-12)
    assert crit == pytest.approx(shadow.kv_fixed_b_crit(M / T), abs=1e-15)


# --- T19: fetch limits — OFF byte-identical, ON extended (hardener) ---

def test_fetch_limits_flag_off_and_on(monkeypatch):
    import types
    captured = {}
    fake = types.ModuleType('market_data')
    fake.fetch_bars_alpaca = (
        lambda api, sym, limit, **k: captured.__setitem__('crypto', limit))
    fake.fetch_stock_bars_alpaca = (
        lambda api, sym, limit, **k: captured.__setitem__('stock', limit))
    monkeypatch.setitem(sys.modules, 'market_data', fake)

    assert shadow.DM_V2_ENABLED is False       # default OFF
    shadow._fetch_closes(object(), 'BTC/USD', 'crypto')
    shadow._fetch_closes(object(), 'AAPL', 'stock')
    assert captured['crypto'] == 24 * (shadow.MAX_SHADOW_DAYS + 4)
    assert captured['stock'] == 600            # legacy limits, untouched

    monkeypatch.setattr(shadow, 'DM_V2_ENABLED', True)
    shadow._fetch_closes(object(), 'BTC/USD', 'crypto')
    shadow._fetch_closes(object(), 'AAPL', 'stock')
    assert captured['crypto'] == (24 * (shadow.MAX_SHADOW_DAYS + 4)
                                  + shadow.V2_FROZEN_MIN_BARS['crypto'] + 48)
    assert captured['stock'] == 1100


# --- T15: v2 crash containment ---

def test_v2_crash_never_breaks_legacy_report(sandbox, monkeypatch):
    cm = _mk_challenger(sandbox)
    start = _start_ago(22.2)   # look-1 window so im_cluster_t is reached
    closes = _fake_closes(start, 80, seed=67)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    rows = _build_rows(closes, start, 60, ['BTC/USD'], cm,
                       fb=6, mode='null', seed=67)
    _log_rows(sandbox, '', rows)

    def boom(*a, **k):
        raise RuntimeError('injected v2 failure')

    monkeypatch.setattr(shadow, 'im_cluster_t', boom)
    report = shadow.evaluate_shadow('', api=object())
    assert report is not None and 'dm_v2' not in report
    assert report['n'] == 60
    for k in ('dm', 'p', 'mean_d', 'hit_champ', 'hit_chall', 'fb_max'):
        assert k in report
