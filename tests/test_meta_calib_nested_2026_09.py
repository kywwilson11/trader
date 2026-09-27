"""scripts/meta_calib_nested.py — nested (OOF-of-OOF) meta-calibration study.

The input is the SIG-R3-METADUMP frame (meta_label._dump_meta_frame,
meta_label.py:893; schema 'meta_frame_dump/v1', :844). No real dump exists
yet, so `make_dump` below writes a SYNTHETIC one that follows the schema of
that writer column for column (npz: row, ticker, entry_time_ns, exit_time_ns,
entry_e, exit_e, fold_id, split, y, net_pct, raw_score, raw_oof, p_served,
p_legacy, p_purged, X, feature_names; sidecar: schema, prefix, asset_type,
npz, calibration, counts, split_index, kfold, best_iteration, n_iter_purged,
params, features — per landing/SIG-R3-METADUMP/PROOF.md "Dump schema").

Known miscalibration: the true logit is 3*x0 (probabilities near 0/1), but
the booster early-stops after a handful of lr=0.05 rounds, so its raw
probabilities are squashed toward 0.5 — every recalibrating arm (O, V) must
beat raw R on out-of-sample log-loss. (At lr=0.0005 the raw spread is so
small that per-fold intercept shifts swamp the pooled inner OOF scores and O
degrades to ~R — measured 2026-09-27; hence lr=0.05 here.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('lightgbm')
pytest.importorskip('sklearn')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import meta_calib_nested as mcn  # noqa: E402
from meta_label import META_FEATURES, META_FRAME_DUMP_SCHEMA  # noqa: E402


def make_dump(d, n=1200, prefix='', schema=META_FRAME_DUMP_SCHEMA, lr=0.05,
              seed=7):
    rng = np.random.default_rng(seed)
    t0 = 1_767_225_600.0                                   # 2026-01-01 UTC
    entry = np.sort(t0 + rng.uniform(0, 140 * 86400, n))   # ~20 weeks
    exit_ = entry + rng.uniform(4 * 3600, 48 * 3600, n)
    F = len(META_FEATURES)
    X = rng.normal(size=(n, F))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-3.0 * X[:, 0]))).astype(float)
    net = np.where(y > 0, 1.0, -1.0) * rng.uniform(0.2, 2.0, n)
    split = int(n * 0.8)
    nan = np.full(n, np.nan)
    p_served = np.clip(0.5 + 0.1 * X[:, 0], 0, 1)
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    p = f'{prefix}_' if prefix else ''
    npz = d / f'{p}meta_frame.npz'
    np.savez_compressed(
        npz, row=np.arange(n, dtype=np.int64),
        ticker=np.asarray([f'T{i % 4}' for i in range(n)], dtype=np.str_),
        entry_time_ns=(entry * 1e9).astype(np.int64),
        exit_time_ns=(exit_ * 1e9).astype(np.int64),
        entry_e=entry, exit_e=exit_, fold_id=np.full(n, -1, dtype=np.int16),
        split=np.where(np.arange(n) < split, 'train', 'earlystop').astype(np.str_),
        y=y, net_pct=net, raw_score=p_served, raw_oof=nan, p_served=p_served,
        p_legacy=p_served, p_purged=nan, X=X,
        feature_names=np.asarray(META_FEATURES, dtype=np.str_))
    params = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
              'max_depth': 5, 'learning_rate': lr, 'feature_fraction': 0.8,
              'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
              'n_jobs': 4}
    side = {'schema': schema, 'prefix': prefix, 'asset_type': prefix or 'crypto',
            'npz': npz.name, 'artifacts': {},
            'calibration': {'used': 'legacy', 'mode_requested': 'legacy'},
            'counts': {'n_rows': n, 'n_train': split, 'n_earlystop': n - split},
            'split_index': split, 'kfold': {'k': 5, 'embargo': 0.0},
            'best_iteration': 40, 'n_iter_purged': 40, 'params': params,
            'features': list(META_FEATURES)}
    with open(npz.with_suffix('.json'), 'w') as f:
        json.dump(side, f)
    return npz


# --- schema -----------------------------------------------------------------

def test_loader_accepts_v1(tmp_path):
    fr, side = mcn.load_dump(make_dump(tmp_path, n=300))
    assert fr['X'].shape == (300, len(META_FEATURES))
    assert mcn.book_name(side) == 'crypto'
    assert mcn.SCHEMA == META_FRAME_DUMP_SCHEMA


def test_loader_refuses_other_schema_and_main_writes_nothing(tmp_path, capsys):
    npz = make_dump(tmp_path / 'd', n=300, schema='meta_frame_dump/v2')
    with pytest.raises(mcn.SchemaError, match='unsupported dump schema'):
        mcn.load_dump(npz)
    out = tmp_path / 'out'
    rc = mcn.main(['--npz', str(npz), '--out', str(out), '--B', '50'])
    assert rc == 0
    assert '[REFUSED]' in capsys.readouterr().out
    assert not out.exists()


def test_loader_refuses_missing_column(tmp_path):
    npz = make_dump(tmp_path, n=200)
    with np.load(npz) as z:
        cols = {k: z[k] for k in z.files if k != 'raw_oof'}
    np.savez_compressed(npz, **cols)
    with pytest.raises(mcn.SchemaError, match='missing columns'):
        mcn.load_dump(npz)


# --- folds ------------------------------------------------------------------

def _entry_exit(n=600, seed=1):
    rng = np.random.default_rng(seed)
    e = np.sort(rng.uniform(0, 90 * 86400, n))
    return e, e + rng.uniform(3600, 72 * 3600, n)


@pytest.mark.parametrize('embargo', [0.0, 0.05, 7200.0])
def test_outer_purged_folds_have_no_overlap(embargo):
    e, x = _entry_exit()
    folds = mcn.outer_folds_purged(e, x, k=5, embargo=embargo)
    assert len(folds) == 5
    assert np.array_equal(np.sort(np.concatenate([te for _, te in folds])),
                          np.arange(len(e)))
    for tr, te in folds:
        t_start, t_end = e[te].min(), x[te].max()
        span = t_end - t_start
        emb = embargo * span if 0 < embargo < 1 else embargo
        assert len(np.intersect1d(tr, te)) == 0
        # every train label span is disjoint from every test span (+ embargo)
        for j in te[:: max(1, len(te) // 25)]:
            ov = (e[tr] <= x[j] + emb) & (x[tr] >= e[j])
            assert not ov.any()
        assert ((x[tr] < t_start) | (e[tr] > t_end + emb)).all()


def test_forward_chain_folds_are_time_ordered():
    e, x = _entry_exit()
    folds = mcn.outer_folds_forward(e, x, k=5, embargo=0.05)
    assert len(folds) == 5
    prev_first = -1
    for tr, te in folds:
        assert len(tr) > 0
        assert tr.max() < te.min()                   # train strictly before
        gap = 0.05 * (x[te].max() - e[te].min())
        assert (x[tr] < e[te].min() - gap).all()     # labels end before test
        assert te.min() > prev_first
        prev_first = te.min()
    assert 0 not in np.concatenate([te for _, te in folds])   # block 0 unscored


# --- verdict logic (hand-built stats) -----------------------------------------

def _row(ll=0.60, ece=0.02, pnl=10.0, flip=0.02, ci=(-0.02, -0.005)):
    return {'ll': ll, 'd_ll': (ci[0] + ci[1]) / 2 if ci else 0.0,
            'd_ll_ci95': list(ci) if ci else None, 'ece': ece,
            'pnl': pnl, 'flip_rate_vs_L': flip}


L = {'ll': 0.65, 'ece': 0.03, 'pnl': 10.0, 'flip_rate_vs_L': 0.0}


def test_verdict_switch_candidate():
    v, _ = mcn.arm_verdict(_row(), L, n_meta=800)
    assert v == 'SWITCH-CANDIDATE'
    av = {'V': ('INCONCLUSIVE', ''), 'O': (v, '')}
    rows = {'V': _row(ll=0.62), 'O': _row(ll=0.60), 'L': L}
    assert mcn.book_verdict(av, rows, 800)[0] == 'SWITCH-CANDIDATE(O)'


def test_verdict_no_go():
    assert mcn.arm_verdict(_row(ci=(0.001, 0.02)), L, 800)[0] == 'NO-GO'
    assert mcn.arm_verdict(_row(pnl=9.99), L, 800)[0] == 'NO-GO'
    av = {'V': ('NO-GO', ''), 'O': ('NO-GO', '')}
    assert mcn.book_verdict(av, {'V': _row(), 'O': _row()}, 800)[0] == 'NO-GO'


def test_verdict_insufficient():
    assert mcn.arm_verdict(_row(), L, n_meta=499)[0] == 'INSUFFICIENT'
    assert mcn.book_verdict({'O': ('SWITCH-CANDIDATE', '')},
                            {'O': _row()}, 499)[0] == 'INSUFFICIENT'


def test_verdict_inconclusive():
    assert mcn.arm_verdict(_row(ci=(-0.01, 0.004)), L, 800)[0] == 'INCONCLUSIVE'
    assert mcn.arm_verdict(_row(ece=0.0351), L, 800)[0] == 'INCONCLUSIVE'
    assert mcn.arm_verdict(_row(ece=0.0349), L, 800)[0] == 'SWITCH-CANDIDATE'
    assert mcn.arm_verdict(_row(flip=0.10), L, 800)[0] == 'INCONCLUSIVE'
    # forward-chain sign disagreement = drift
    v, why = mcn.arm_verdict(_row(), L, 800, fc_point=+0.004)
    assert v == 'INCONCLUSIVE' and 'drift' in why
    av = {'V': ('NO-GO', ''), 'O': ('INCONCLUSIVE', '')}
    assert mcn.book_verdict(av, {'V': _row(), 'O': _row()}, 800)[0] == 'INCONCLUSIVE'


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(0.3, 1.0, 2000)
    blocks = np.repeat(np.arange(40), 50)
    dist = mcn.block_bootstrap_mean(v, blocks, 2000, np.random.default_rng(1))
    lo, hi = mcn._ci(dist)
    assert lo < v.mean() < hi and lo > 0


def test_equal_mass_ece():
    p = np.linspace(0.05, 0.95, 1000)
    rng = np.random.default_rng(3)
    y_cal = (rng.uniform(size=1000) < p).astype(float)
    assert mcn.ece_equal_mass(p, y_cal) < 0.06
    assert mcn.ece_equal_mass(np.full(1000, 0.9), np.zeros(1000)) == pytest.approx(0.9)


def test_refuses_repo_root():
    assert mcn._refuse_root(ROOT)
    assert mcn._refuse_root(ROOT / 'x.json')
    assert not mcn._refuse_root(ROOT / 'logs' / 'meta_calib_nested' / 'x.json')


# --- dry run + end to end -------------------------------------------------------

def test_dry_run_writes_nothing(tmp_path, capsys):
    npz = make_dump(tmp_path / 'd', n=400)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))
    rc = mcn.main(['--npz', str(npz), '--out', str(tmp_path / 'out'),
                   '--json', str(tmp_path / 'res.json'), '--dry-run',
                   '--forward-chain'])
    assert rc == 0
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))
    assert before == after
    out = capsys.readouterr().out
    assert 'DRY-RUN' in out and 'PRE-REGISTERED RULE' in out and 'INSUFFICIENT' in out
    assert '[FOLD' not in out


def test_end_to_end_recalibration_beats_squashed_raw(tmp_path, capsys):
    npz = make_dump(tmp_path / 'd', n=1200)
    js = tmp_path / 'res.json'
    rc = mcn.main(['--npz', str(npz), '--json', str(js), '--out',
                   str(tmp_path / 'out'), '--B', '200', '--forward-chain',
                   '--arms', 'L,V,O,R'])
    assert rc == 0
    out = capsys.readouterr().out
    assert 'VERDICT[crypto]:' in out and '[ERROR]' not in out
    res = json.loads(js.read_text())['books']['crypto']
    arms = res['score']['arms']
    assert res['n_meta'] >= 1000              # every outer fold was fitted
    # the squashed raw booster is beaten by every recalibrating arm
    assert arms['O']['ll'] < arms['R']['ll'] - 0.02
    assert arms['V']['ll'] < arms['R']['ll'] - 0.02
    assert arms['O']['brier'] < arms['R']['brier']
    assert arms['R']['ece'] > arms['O']['ece']
    assert set(res['arm_verdicts']) == {'V', 'O', 'R'}
    assert res['verdict'].split('(')[0] in ('SWITCH-CANDIDATE', 'NO-GO',
                                             'INCONCLUSIVE')
    assert 'forward_chain' in res and res['forward_chain']['score']['n_scored'] > 0
    for a in ('V', 'O', 'R'):
        lo, hi = arms[a]['d_ll_ci95']
        assert lo <= arms[a]['d_ll'] <= hi
    rows = np.load(tmp_path / 'out' / 'crypto_meta_calib_nested_rows.npz')
    assert {'outer_fold', 'p_L', 'p_O', 'p_fc_O'} <= set(rows.files)
