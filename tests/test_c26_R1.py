"""Packet R1 — calibration mechanics v2 (D13c, default-OFF CALIBRATION_V2),
ungated-publish closure (D13a/D13b, direct-ship safety), and the shadow
promote pre-flight (Q2 hook, under existing default-OFF GATE_TARGETS_CHALLENGER).

All Mac-runnable: numpy + stdlib only. calibration.py is pure numpy;
meta_label.py's module level imports only stdlib+numpy; shadow.py's touched
helpers are heavy-dep-free. Flag flips are exercised via monkeypatching
strategy_config attributes (call-time reads), never the environment.
"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import strategy_config
import calibration
from calibration import (IsotonicCalibrator, SigmoidCalibrator, _logit, _pava,
                         fit_calibrator)
import meta_label
import shadow


# ===========================================================================
# G1 — CALIBRATION_V2 OFF: byte-identical legacy pins
# ===========================================================================

def test_flag_default_off():
    assert strategy_config.CALIBRATION_V2 is False


def test_default_isotonic_reproduces_legacy_tie_collapse():
    # The pathological case D13c fixes UNDER THE FLAG: flag OFF must keep it.
    cal = IsotonicCalibrator().fit([0.4, 0.5, 0.5, 0.6], [0.0, 0.0, 1.0, 1.0])
    assert float(cal.predict(np.array([0.5]))[0]) == 1.0


def _legacy_isotonic_reference(raw, y):
    """Inline reference of the HEAD algorithm (_pava + searchsorted fit[last])."""
    raw = np.asarray(raw, float)
    y = np.asarray(y, float)
    order = np.argsort(raw, kind='mergesort')
    xs, ys = raw[order], y[order]
    fit = _pava(ys, np.ones_like(ys))
    ux = np.unique(xs)
    last = np.searchsorted(xs, ux, side='right') - 1
    return ux, np.clip(fit[last], 0.0, 1.0)


def test_default_isotonic_matches_head_reference_on_duplicated_scores():
    rng = np.random.default_rng(11)
    raw = rng.choice([0.1, 0.25, 0.4, 0.55, 0.7, 0.85], size=200)
    y = (rng.uniform(size=200) < raw).astype(float)
    cal = IsotonicCalibrator().fit(raw, y)
    ux, uy = _legacy_isotonic_reference(raw, y)
    np.testing.assert_array_equal(cal.x_, ux)
    np.testing.assert_array_equal(cal.y_, uy)


def test_default_sigmoid_and_fit_calibrator_off_equality():
    rng = np.random.default_rng(12)
    x = rng.uniform(size=300)
    y = (rng.uniform(size=300) < x).astype(float)
    b0 = SigmoidCalibrator().fit(x, y).beta_
    b1 = SigmoidCalibrator(platt_v2=False).fit(x, y).beta_
    np.testing.assert_array_equal(b0, b1)
    # default flag (False) == explicit v2=False, prediction-for-prediction
    grid = np.linspace(0.01, 0.99, 25)
    c_flag = fit_calibrator(x, y)            # reads CALIBRATION_V2=False
    c_off = fit_calibrator(x, y, v2=False)
    np.testing.assert_array_equal(c_flag.predict(grid), c_off.predict(grid))
    assert c_flag.calibration_v2_ is False


# ===========================================================================
# G2 — CALIBRATION_V2 ON: isotonic tie pooling (B04.2)
# ===========================================================================

def test_v2_isotonic_pools_ties_to_weighted_mean():
    cal = IsotonicCalibrator(pool_ties=True).fit(
        [0.4, 0.5, 0.5, 0.6], [0.0, 0.0, 1.0, 1.0])
    assert float(cal.predict(np.array([0.5]))[0]) == pytest.approx(0.5)


def test_v2_isotonic_order_independent_legacy_is_not():
    rng = np.random.default_rng(21)
    raw = rng.choice([0.2, 0.4, 0.6, 0.8], size=60)
    y = (rng.uniform(size=60) < 0.5).astype(float)
    v2_fits, legacy_fits = set(), set()
    for seed in range(10):
        perm = np.random.default_rng(seed).permutation(60)
        c_v2 = IsotonicCalibrator(pool_ties=True).fit(raw[perm], y[perm])
        c_le = IsotonicCalibrator().fit(raw[perm], y[perm])
        v2_fits.add((tuple(c_v2.x_), tuple(c_v2.y_)))
        legacy_fits.add((tuple(c_le.x_), tuple(c_le.y_)))
    assert len(v2_fits) == 1              # order-independent under the flag
    assert len(legacy_fits) > 1           # documents the defect (no assertion
    #                                       on legacy VALUES is weakened)


def test_v2_isotonic_true_ten_pct_bucket_regression():
    # 100 rows tied at 0.7 with a TRUE 10% win rate; positives last in input
    # order so the legacy fit[last] path reads the tied bucket as ~1.0.
    raw = np.concatenate([np.full(5, 0.1), np.full(100, 0.7), np.full(5, 0.9)])
    y = np.concatenate([np.zeros(5), np.zeros(90), np.ones(10), np.ones(5)])
    v2 = IsotonicCalibrator(pool_ties=True).fit(raw, y)
    legacy = IsotonicCalibrator().fit(raw, y)
    # pooled-PAVA oracle on the unique grid
    oracle = _pava(np.array([0.0, 0.1, 1.0]), np.array([5.0, 100.0, 5.0]))
    assert float(v2.predict(np.array([0.7]))[0]) == pytest.approx(
        float(oracle[1]), abs=1e-6)
    assert float(v2.predict(np.array([0.7]))[0]) == pytest.approx(0.10, abs=1e-6)
    assert float(legacy.predict(np.array([0.7]))[0]) >= 0.9  # the defect


def test_v2_isotonic_matches_manual_pooled_pava_weighted():
    rng = np.random.default_rng(23)
    raw = rng.choice([0.15, 0.35, 0.55, 0.75], size=80)
    y = (rng.uniform(size=80) < raw).astype(float)
    w = rng.uniform(0.5, 2.0, size=80)
    cal = IsotonicCalibrator(pool_ties=True).fit(raw, y, w=w)
    order = np.argsort(raw, kind='mergesort')
    xs, ys, ws = raw[order], y[order], w[order]
    ux, inv = np.unique(xs, return_inverse=True)
    wsum = np.bincount(inv, weights=ws)
    ysum = np.bincount(inv, weights=ws * ys)
    oracle = np.clip(_pava(ysum / wsum, wsum), 0.0, 1.0)
    np.testing.assert_array_equal(cal.x_, ux)
    np.testing.assert_allclose(cal.y_, oracle, atol=1e-12)


# ===========================================================================
# G3 — CALIBRATION_V2 ON: sigmoid (Platt logit-scale + MAP smoothing)
# ===========================================================================

_SEP_X = np.array([0.1] * 5 + [0.9] * 5)
_SEP_Y = np.array([0.0] * 5 + [1.0] * 5)


def test_v2_sigmoid_bounded_on_separable_fixture():
    cal = SigmoidCalibrator(platt_v2=True).fit(_SEP_X, _SEP_Y)
    assert cal.converged_
    assert np.isfinite(cal.beta_).all()
    p = cal.predict(_SEP_X)
    eps = 1e-3
    lo, hi = 1.0 / (5 + 2.0), (5 + 1.0) / (5 + 2.0)
    assert p.min() >= lo - eps and p.max() <= hi + eps
    # legacy saturates on the same fixture (documents the divergence)
    p_leg = SigmoidCalibrator().fit(_SEP_X, _SEP_Y).predict(_SEP_X)
    assert p_leg.min() < 0.01 and p_leg.max() > 0.99


def _irls_oracle(x, t):
    """In-test twin of the IRLS loop (targets t, unit weights)."""
    x = np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    w = np.ones_like(t)
    for _ in range(1, 101):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = w * p * (1 - p) + 1e-9
        grad = X.T @ (w * (t - p))
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def test_v2_sigmoid_is_irls_on_logit_scores_with_smoothed_targets():
    rng = np.random.default_rng(31)
    x = rng.uniform(0.05, 0.95, size=120)
    y = (rng.uniform(size=120) < x).astype(float)
    cal = SigmoidCalibrator(platt_v2=True).fit(x, y)
    n_pos = float((y == 1.0).sum())
    n_neg = float((y == 0.0).sum())
    t = np.where(y == 1.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    np.testing.assert_allclose(cal.beta_, _irls_oracle(_logit(x), t),
                               rtol=0, atol=1e-12)
    # monotone predict; NaN raw -> NaN out (contract preserved)
    grid = np.linspace(0.01, 0.99, 50)
    p = cal.predict(grid)
    assert np.all(np.diff(p) >= 0)
    assert np.isnan(cal.predict(np.array([np.nan]))[0])


def test_fit_calibrator_v2_plumbing(monkeypatch):
    rng = np.random.default_rng(32)
    x = rng.uniform(size=200)                 # n < 1000 -> sigmoid
    y = (rng.uniform(size=200) < x).astype(float)
    cal = fit_calibrator(x, y, v2=True)
    assert isinstance(cal, SigmoidCalibrator)
    assert cal.platt_v2 is True and cal.calibration_v2_ is True
    # v2=None reads the strategy_config flag at call time
    monkeypatch.setattr(strategy_config, 'CALIBRATION_V2', True)
    cal2 = fit_calibrator(x, y)
    assert cal2.platt_v2 is True and cal2.calibration_v2_ is True
    # explicit kwarg overrides the flag
    cal3 = fit_calibrator(x, y, v2=False)
    assert cal3.platt_v2 is False and cal3.calibration_v2_ is False


def test_fit_calibrator_v2_isotonic_above_min_n():
    rng = np.random.default_rng(33)
    x = rng.choice(np.linspace(0.05, 0.95, 12), size=1500)
    y = (rng.uniform(size=1500) < x).astype(float)
    cal = fit_calibrator(x, y, v2=True)
    assert isinstance(cal, IsotonicCalibrator)
    assert cal.pool_ties is True and cal.calibration_v2_ is True


# ===========================================================================
# G4 — meta_label pure guard helpers + embargo + refusal sidecar
# ===========================================================================

def test_calib_slice_guard_reasons():
    g = meta_label._calib_slice_guard
    assert g([0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 0]) == \
        'calib_slice_too_thin(n=5)'
    ok_scores = np.linspace(0.1, 0.9, 20)
    assert g(ok_scores, np.ones(20)) == 'calib_slice_one_class'
    assert g(np.full(20, 0.3), np.r_[np.zeros(10), np.ones(10)]) == \
        'calib_slice_constant_scores'
    assert g(ok_scores, np.r_[np.zeros(10), np.ones(10)]) is None


def test_publish_guard_reasons():
    g = meta_label._publish_guard_reasons
    spread_p = np.linspace(0.1, 0.9, 50)
    # constant calibrated p
    r = g(np.full(50, 0.42), 0.60, 0.0)
    assert any('constant_calibrated_p' in s for s in r)
    # veto fraction
    r = g(spread_p, 0.60, 0.85)
    assert r == ['frac_below_veto=0.85>=0.8']
    # worse-than-chance AUC
    r = g(spread_p, 0.45, 0.3)
    assert r == ['val_auc=0.450<0.5']
    # val_auc None -> no auc reason; 0.55 skill floor stays a warning only
    assert g(spread_p, None, 0.3) == []
    assert g(spread_p, 0.52, 0.3) == []
    # healthy
    assert g(spread_p, 0.60, 0.3) == []


def test_calibration_embargo_flag(monkeypatch):
    assert meta_label._calibration_embargo() == 0.0
    monkeypatch.setattr(strategy_config, 'CALIBRATION_V2', True)
    assert meta_label._calibration_embargo() == 0.05    # B04.1 binding param
    assert meta_label._calibration_v2_on() is True


def test_write_refusal_atomic_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(meta_label, 'BASE_DIR', tmp_path)
    paths = meta_label._paths('')
    meta_label._write_refusal(paths, ['some_reason'], {'val_auc': 0.4})
    payload = json.loads((tmp_path / 'meta_refused.json').read_text())
    assert payload['reasons'] == ['some_reason']
    assert payload['val_auc'] == 0.4
    assert 'refused_at' in payload
    assert not list(tmp_path.glob('*.tmp.*'))
    # seam dicts without the 'refused' key fall back to the meta-name rewrite
    seam = {'meta': tmp_path / 'stock_meta_meta.json'}
    meta_label._write_refusal(seam, ['r2'])
    assert (tmp_path / 'stock_meta_refused.json').exists()
    # corrupt inputs never raise: unwritable dir + unserializable extra
    bad = {'meta': tmp_path / 'nope' / 'meta_meta.json'}
    meta_label._write_refusal(bad, ['r3'])              # no raise
    meta_label._write_refusal(paths, ['r4'], {'obj': object()})  # no raise


# ===========================================================================
# G5 — promote_staged_meta (guarded staged->live, D13b)
# ===========================================================================

@pytest.fixture
def meta_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(meta_label, 'BASE_DIR', tmp_path)
    meta_label.invalidate_cache()
    yield tmp_path
    meta_label.invalidate_cache()


def _manifest_mtime_ns(tmp_path, prefix=''):
    p = f'{prefix}_' if prefix else ''
    mpath = tmp_path / f'{p}model_v2.manifest.json'
    mpath.write_text(json.dumps({'saved_at': 'x'}))
    return os.stat(mpath).st_mtime_ns


def _stage_triple(tmp_path, prefix='', guards_ok=True, manifest_ns=None,
                  corrupt_meta=False):
    paths = meta_label._paths(prefix)
    payload = {
        'calibration': {'publish_guards_ok': guards_ok, 'guard_reasons': []
                        if guards_ok else ['constant_calibrated_p']},
        'primary': ({'manifest_mtime_ns': manifest_ns}
                    if manifest_ns is not None else None),
    }
    Path(str(paths['model']) + '.staged').write_bytes(b'new-model')
    Path(str(paths['calib']) + '.staged').write_bytes(b'new-calib')
    if corrupt_meta:
        Path(str(paths['meta']) + '.staged').write_text('{not json')
    else:
        Path(str(paths['meta']) + '.staged').write_text(json.dumps(payload))
    return paths


def _live_triple(paths, tag=b'old'):
    paths['model'].write_bytes(tag + b'-model')
    paths['calib'].write_bytes(tag + b'-calib')
    paths['meta'].write_text(json.dumps({'live': tag.decode()}))


def _snapshot_live(paths):
    out = {}
    for k in ('model', 'calib', 'meta'):
        out[k] = paths[k].read_bytes() if paths[k].exists() else None
    return out


def test_promote_staged_meta_happy_path(meta_sandbox):
    ns = _manifest_mtime_ns(meta_sandbox)
    paths = _stage_triple(meta_sandbox, guards_ok=True, manifest_ns=ns)
    _live_triple(paths)
    # shadow's rename-aside leftovers get cleared on a successful promote
    for k in ('model', 'calib', 'meta'):
        Path(str(paths[k]) + '.stale').write_bytes(b'stale')
    assert meta_label.promote_staged_meta('') is True
    assert paths['model'].read_bytes() == b'new-model'
    assert paths['calib'].read_bytes() == b'new-calib'
    assert json.loads(paths['meta'].read_text())['calibration'][
        'publish_guards_ok'] is True
    for k in ('model', 'calib', 'meta'):
        assert not Path(str(paths[k]) + '.staged').exists()
        assert not Path(str(paths[k]) + '.stale').exists()
    # .prev backups of the PRIOR live triple
    assert Path(str(paths['model']) + '.prev').read_bytes() == b'old-model'
    assert Path(str(paths['calib']) + '.prev').read_bytes() == b'old-calib'


def test_promote_staged_meta_first_publish_and_cache_invalidation(meta_sandbox):
    # First-ever publish: NO live triple exists (the fail-open-neutral state
    # after shadow's _stash_stale_meta) — promote must still succeed, and no
    # .prev files appear (there was nothing to back up).
    ns = _manifest_mtime_ns(meta_sandbox)
    paths = _stage_triple(meta_sandbox, guards_ok=True, manifest_ns=ns)
    meta_label._loaded[''] = (('sentinel',), None)   # stale cache entry
    assert meta_label.promote_staged_meta('') is True
    assert paths['model'].read_bytes() == b'new-model'
    for k in ('model', 'calib', 'meta'):
        assert not Path(str(paths[k]) + '.prev').exists()
    # a successful promote invalidates the artifact cache so running bots
    # pick the new triple up without a restart
    assert meta_label._loaded == {}


def test_promote_staged_meta_refusals_leave_live_untouched(meta_sandbox):
    ns = _manifest_mtime_ns(meta_sandbox)

    # 1) missing staged file
    paths = _stage_triple(meta_sandbox, guards_ok=True, manifest_ns=ns)
    _live_triple(paths)
    before = _snapshot_live(paths)
    Path(str(paths['calib']) + '.staged').unlink()
    assert meta_label.promote_staged_meta('') is False
    assert _snapshot_live(paths) == before

    # 2) guards not passed -> refusal sidecar written, live untouched
    _stage_triple(meta_sandbox, guards_ok=False, manifest_ns=ns)
    assert meta_label.promote_staged_meta('') is False
    assert _snapshot_live(paths) == before
    refusal = json.loads((meta_sandbox / 'meta_refused.json').read_text())
    assert refusal['reasons'] == ['staged_guards_not_ok']

    # 3) corrupt staged meta json
    _stage_triple(meta_sandbox, corrupt_meta=True)
    assert meta_label.promote_staged_meta('') is False
    assert _snapshot_live(paths) == before

    # 4) primary manifest changed since staging
    _stage_triple(meta_sandbox, guards_ok=True, manifest_ns=ns + 12345)
    assert meta_label.promote_staged_meta('') is False
    assert _snapshot_live(paths) == before
    # staged triple retained on freshness refusal (operator can inspect)
    assert Path(str(paths['model']) + '.staged').exists()


def test_train_meta_signature_default_publish():
    # run_pipeline/shadow invoke meta_label.py with NO new flags — the default
    # invocation must stage->guard->promote (publish=True).
    import inspect
    sig = inspect.signature(meta_label.train_meta)
    assert sig.parameters['publish'].default is True


# ===========================================================================
# G6 — shadow: Q2 pre-flight + stale-meta stash
# ===========================================================================

@pytest.fixture
def shadow_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, 'BASE_DIR', tmp_path)
    return tmp_path


def _mk_challenger(tmp_path, prefix=''):
    man = (tmp_path
           / f'{shadow.challenger_prefix(prefix)}_model_v2.manifest.json')
    man.write_text(json.dumps({'saved_at': 'x', 'holdout': {}}))
    return int(man.stat().st_mtime)


def _write_gate_sidecar(tmp_path, prefix='', passed=True, mtime=None,
                        corrupt=False):
    path = tmp_path / f'{shadow.challenger_prefix(prefix)}_policy_gate.json'
    if corrupt:
        path.write_text('{broken')
        return path
    path.write_text(json.dumps({
        'passed': passed, 'sharpe': 1.2, 'dsr': 0.7, 'n_trades': 42,
        'challenger_manifest_mtime': mtime,
    }))
    return path


def test_gate_preflight_verdicts(shadow_sandbox):
    cm = _mk_challenger(shadow_sandbox)
    # missing sidecar
    ok, why = shadow._gate_preflight('')
    assert ok is False and 'missing' in why
    # stale (mtime mismatch)
    _write_gate_sidecar(shadow_sandbox, passed=True, mtime=cm - 999)
    ok, why = shadow._gate_preflight('')
    assert ok is False and 'stale' in why
    # gate failed
    _write_gate_sidecar(shadow_sandbox, passed=False, mtime=cm)
    ok, why = shadow._gate_preflight('')
    assert ok is False and 'FAILED' in why
    # corrupt json -> hold, never a raise
    _write_gate_sidecar(shadow_sandbox, corrupt=True)
    ok, why = shadow._gate_preflight('')
    assert ok is False and 'unreadable' in why
    # all good
    _write_gate_sidecar(shadow_sandbox, passed=True, mtime=cm)
    ok, why = shadow._gate_preflight('')
    assert ok is True and why == 'policy gate passed'


def test_stash_stale_meta_renames_and_tolerates_absence(shadow_sandbox):
    # all three present -> renamed with content preserved
    for suffix in shadow._STALE_META_SUFFIXES:
        (shadow_sandbox / suffix).write_bytes(b'content-' + suffix.encode())
    shadow._stash_stale_meta('')
    for suffix in shadow._STALE_META_SUFFIXES:
        assert not (shadow_sandbox / suffix).exists()
        assert (shadow_sandbox / f'{suffix}.stale').read_bytes() == \
            b'content-' + suffix.encode()
    # partial/absent set never raises
    shadow._stash_stale_meta('')
    (shadow_sandbox / 'stock_meta_model.txt').write_bytes(b'x')
    shadow._stash_stale_meta('stock')
    assert (shadow_sandbox / 'stock_meta_model.txt.stale').exists()


# --- promote-shaped evaluate replay (test_c26_Q3 sandbox pattern) ---

def _fake_closes(start, hours, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=hours, freq='h', tz='UTC')
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.002, hours))
    return pd.Series(prices, index=idx)


def _start_ago(days):
    now = dt.datetime.now(dt.timezone.utc)
    return (now - dt.timedelta(days=days)).replace(
        minute=0, second=0, microsecond=0)


def _strong_rows(closes, start, n_hours, cm, fb=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_hours):
        ts = start + dt.timedelta(hours=i, minutes=5)
        j = i + 1
        c0, c1 = float(closes.iloc[j]), float(closes.iloc[j + fb])
        realized = (c1 - c0) / c0 * 100
        rows.append({'ts': ts.isoformat(), 'sym': 'BTC/USD',
                     'champ': round(float(rng.normal(0, 1.0)), 6),
                     'chall': round(realized + float(rng.normal(0, 0.02)), 6),
                     'fb_champ': fb, 'fb_chall': fb, 'cm': cm})
    return rows


def _promote_shaped_sandbox(tmp_path, monkeypatch, seed=43):
    cm = _mk_challenger(tmp_path)
    start = _start_ago(22.2)
    closes = _fake_closes(start, 360, seed=seed)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    rows = _strong_rows(closes, start, 300, cm, fb=24, seed=seed)
    with open(tmp_path / 'shadow_preds.jsonl', 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return cm


def test_flag_off_preflight_never_consulted(shadow_sandbox, monkeypatch):
    assert strategy_config.GATE_TARGETS_CHALLENGER is False   # default
    _promote_shaped_sandbox(shadow_sandbox, monkeypatch)
    calls = []
    monkeypatch.setattr(shadow, '_gate_preflight',
                        lambda pfx: calls.append(pfx) or (False, 'spy'))
    promoted = []
    monkeypatch.setattr(shadow, 'promote_challenger',
                        lambda pfx, rep=None: promoted.append(pfx) or True)
    report = shadow.evaluate_and_maybe_promote('', 'CRYPTO', api=object())
    assert report['decision'] == 'promoted'
    assert promoted == ['']
    assert calls == []            # flag OFF: sidecar never read, path unchanged


def test_flag_on_failing_preflight_holds_promote(shadow_sandbox, monkeypatch):
    _promote_shaped_sandbox(shadow_sandbox, monkeypatch)
    monkeypatch.setattr(strategy_config, 'GATE_TARGETS_CHALLENGER', True)
    promoted = []
    monkeypatch.setattr(shadow, 'promote_challenger',
                        lambda pfx, rep=None: promoted.append(pfx) or True)
    # no sidecar exists -> pre-flight fails -> HOLD, not promote, not discard
    report = shadow.evaluate_and_maybe_promote('', 'CRYPTO', api=object())
    assert report['decision'] == 'continue'
    assert promoted == []
    assert shadow.challenger_manifest('').exists()   # NOT discarded
    status = json.loads((shadow_sandbox / 'shadow_status.json').read_text())
    assert status['decision'] == 'continue'          # pinned GUI label set
    assert 'HELD' in status['detail']


def test_flag_on_passing_preflight_promotes(shadow_sandbox, monkeypatch):
    cm = _promote_shaped_sandbox(shadow_sandbox, monkeypatch)
    monkeypatch.setattr(strategy_config, 'GATE_TARGETS_CHALLENGER', True)
    _write_gate_sidecar(shadow_sandbox, passed=True, mtime=cm)
    promoted = []
    monkeypatch.setattr(shadow, 'promote_challenger',
                        lambda pfx, rep=None: promoted.append(pfx) or True)
    report = shadow.evaluate_and_maybe_promote('', 'CRYPTO', api=object())
    assert report['decision'] == 'promoted'
    assert promoted == ['']
