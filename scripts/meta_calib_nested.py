"""Nested (OOF-of-OOF) meta-calibration study over a METADUMP frame — MEASUREMENT-ONLY.

Question: on rows that NO calibration recipe ever saw, does the shipped legacy
meta calibrator (same-slice isotonic) lose to the CALIBRATION_V2 chooser or to
the purged-OOF calibrator? Design: research/campaign_2026-09_jetson/
research_signal.md, "2026-09-27 R4 scout brief (SCOUT-4)" -> "T2 — An honest
meta-calibration holdout", design (b) OOF-of-OOF. Zero training-path change:
the input is the `TRADER_META_FRAME_DUMP` npz + JSON sidecar written by
meta_label._dump_meta_frame (schema 'meta_frame_dump/v1', meta_label.py:844,
:893); the script refits everything itself and edits nothing.

Outer loop (one book at a time — books are NEVER pooled):
  main       purged k-fold over the dump rows on [entry_e, exit_e] with an
             EXPLICIT embargo (default 0.05 = 5 % of the test fold's span,
             forward-only) — calibration.purged_kfold_indices (:237).
  --forward-chain  sensitivity: k+1 time-ordered blocks, fold i tests block i
             and trains only on rows whose exit < test start - embargo gap
             (train strictly before test; block 0 is never scored).
Inside each outer-TRAIN set (rows stay in time order) the recipes are re-run
exactly as train_meta runs them, and scored on the untouched outer-TEST rows:
  booster  lgb.train(sidecar params, 400 rounds, early_stopping(30)) on the
           first 80 % with the last 20 % as the early-stop slice — mirrors
           meta_label.py:1235-1251 (split, Dataset, params, lgb.train). The
           round count is RE-SELECTED per outer fold (the sidecar's full-sample
           best_iteration is reported, never used — it would leak).
  L  legacy: _calib_slice_guard (:97) then sklearn IsotonicRegression(
     out_of_bounds='clip') on that same 20 % slice — meta_label.py:1303-1314,
     :1332 (the shipped path while CALIBRATION_V2 is False).
  V  the same slice through calibration.fit_calibrator(v2=True) (:191) —
     meta_label.py:1315-1330 (the CALIBRATION_V2 legacy route).
  O  purged_oof: inner calibration.crossfit_oof_predict (:293), k=5, n_iter =
     booster.best_iteration or 200, embargo/v2 = what train_meta would use
     with today's CALIBRATION_V2 (meta_label._calibration_embargo :134), then
     fit_calibrator; a declined fit falls back to L exactly like
     meta_label.py:1267-1300.
  OV (optional) purged_oof with CALIBRATION_V2=True (embargo 0.05, v2 chooser).
  R  raw booster probability (reference only — no config serves it).
Served p = meta_label._calibrated (clip to [0,1], :583).

Metrics on the jointly-finite outer-test rows: per-row log-loss (p clipped to
[eps, 1-eps], --ll-eps, default 1e-4) and Brier; EQUAL-MASS 10-bin ECE (own
implementation, Roelofs et al. 2022 — NOT calibration.expected_calibration_error,
which is equal-width; that one is also reported as ece_width); veto flip rate
vs L at p < META_VETO_PROB (0.30, meta_label.py:58); decision-P&L proxy =
sum of net_pct over rows the arm does NOT veto (plus a sized secondary,
sum 1[p>=0.3]*meta_size_mult(p)*net_pct, meta_label.py:640).
Statistics: paired per-row d = LL(arm) - LL(L) (NEGATIVE favours the arm);
weekly-block bootstrap by entry week (Monday-start UTC weeks, grouped by
--block-weeks), B draws, two-sided 95 % percentile CI.

PRE-REGISTERED RULE (applied mechanically, per book; see RULE_TEXT):
  precondition n_meta >= 500 scored rows, else INSUFFICIENT.
  SWITCH-CANDIDATE(arm) iff CI_hi(d) < 0 AND ECE(arm) <= ECE(L) + 0.005 AND
      P&L(arm) >= P&L(L) AND veto flip rate < 10 % (AND, with
      --forward-chain, no sign disagreement of the forward-chain point d).
  NO-GO iff CI_lo(d) > 0 (the CI favours L) OR P&L(arm) < P&L(L).
  INCONCLUSIVE otherwise (incl. forward-chain sign disagreement = drift).
  Book verdict over the switchable arms (V, O, OV): any SWITCH-CANDIDATE ->
  the one with the lowest mean LL; all NO-GO -> NO-GO; else INCONCLUSIVE.
  With >1 --npz a GLOBAL line applies the design's cross-book rule
  (both flags are global): switch-candidate on >=1 book, and on every other
  book point d <= 0 with the ECE / P&L / flip conditions holding.

Honesty scope: the META layer only — the `pred` feature inside X is the primary
model's in-sample score unless META_OOF_PRED was on for the dumped run.

Writes nothing into the repo root: --json <path> (default <--out>/
meta_calib_nested_<UTC>.json) and <--out>/<book>_meta_calib_nested_rows.npz
(per-row outer-test p per arm); default --out logs/meta_calib_nested/
(gitignored). --dry-run validates the dump, prints the plan and exits without
fitting or writing. Exit code is 0 always; failures are printed.
Jetson: run through the campaign hwlock, CUDA_VISIBLE_DEVICES='', one process,
LightGBM threads = --threads (default 1).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

SCHEMA = 'meta_frame_dump/v1'          # == meta_label.META_FRAME_DUMP_SCHEMA (:844)
REQUIRED_NPZ = ('row', 'ticker', 'entry_time_ns', 'exit_time_ns', 'entry_e',
                'exit_e', 'fold_id', 'split', 'y', 'net_pct', 'raw_score',
                'raw_oof', 'p_served', 'p_legacy', 'p_purged', 'X',
                'feature_names')
WEEK_SECONDS = 7 * 86400
MONDAY_OFFSET = 4 * 86400              # 1970-01-05 (the first Monday) - epoch
VETO_P = 0.30                          # meta_label.META_VETO_PROB (:58)
MIN_N_META = 500
ECE_TOL = 0.005
FLIP_MAX = 0.10
ALL_ARMS = ('L', 'V', 'O', 'OV', 'R')
SWITCHABLE = ('V', 'O', 'OV')
NUM_BOOST_ROUND = 400                  # meta_label.py:1249
EARLY_STOP = 30                        # meta_label.py:1251
INNER_K = 5                            # meta_label.py:1280 (k=5)

RULE_TEXT = (
    "PRE-REGISTERED RULE (per book; d = LL(arm) - LL(L), negative favours arm; "
    "95% weekly-block bootstrap CI):\n"
    f"  precondition n_meta >= {MIN_N_META} scored rows, else INSUFFICIENT\n"
    f"  SWITCH-CANDIDATE(arm) iff CI_hi(d) < 0 AND ECE_eqmass(arm) <= ECE(L) + {ECE_TOL}"
    f" AND P&L(arm) >= P&L(L) AND veto-flip rate < {FLIP_MAX:.0%}"
    " (AND no forward-chain sign disagreement when --forward-chain)\n"
    "  NO-GO iff CI_lo(d) > 0 OR P&L(arm) < P&L(L)\n"
    "  INCONCLUSIVE otherwise (forward-chain sign disagreement = drift)\n"
    "  book verdict over switchable arms V/O/OV; R is a reference arm only")


class SchemaError(ValueError):
    """The dump is not a readable meta_frame_dump/v1 frame."""


# ---------------------------------------------------------------------------
# Loading + validation (pure numpy/json)
# ---------------------------------------------------------------------------

def sidecar_path_for(npz_path):
    p = Path(npz_path)
    return p.with_suffix('.json')


def load_dump(npz_path, sidecar_path=None):
    """Load + validate a METADUMP frame. Refuses anything that is not schema
    'meta_frame_dump/v1' (SchemaError). Returns (frame dict, sidecar dict)."""
    npz_path = Path(npz_path)
    sc_path = Path(sidecar_path) if sidecar_path else sidecar_path_for(npz_path)
    if not sc_path.exists():
        raise SchemaError(f'sidecar {sc_path} missing (the dump writes it LAST '
                          f'as the commit marker — an incomplete dump?)')
    with open(sc_path) as f:
        side = json.load(f)
    schema = side.get('schema') if isinstance(side, dict) else None
    if schema != SCHEMA:
        raise SchemaError(f'unsupported dump schema {schema!r} (this script '
                          f'reads only {SCHEMA!r})')
    with np.load(npz_path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_NPZ if k not in z.files]
        if missing:
            raise SchemaError(f'npz missing columns {missing}')
        fr = {k: z[k] for k in REQUIRED_NPZ}
    n = len(fr['y'])
    for k in REQUIRED_NPZ:
        if k in ('X', 'feature_names'):
            continue
        if len(fr[k]) != n:
            raise SchemaError(f'column {k} has {len(fr[k])} rows, y has {n}')
    X = np.asarray(fr['X'], float)
    if X.ndim != 2 or X.shape[0] != n or X.shape[1] != len(fr['feature_names']):
        raise SchemaError(f'X shape {X.shape} vs n={n}, '
                          f'{len(fr["feature_names"])} feature names')
    y = np.asarray(fr['y'], float)
    if not np.isin(np.unique(y), (0.0, 1.0)).all():
        raise SchemaError('y must be binary 0/1')
    for k in ('entry_e', 'exit_e', 'net_pct'):
        if not np.isfinite(np.asarray(fr[k], float)).all():
            raise SchemaError(f'{k} has non-finite values')
    if np.any(np.asarray(fr['exit_e'], float) < np.asarray(fr['entry_e'], float)):
        raise SchemaError('exit_e < entry_e on some rows')
    if not isinstance(side.get('params'), dict):
        raise SchemaError('sidecar has no params dict')
    fr['X'] = X
    fr['y'] = y
    fr['entry_e'] = np.asarray(fr['entry_e'], float)
    fr['exit_e'] = np.asarray(fr['exit_e'], float)
    fr['net_pct'] = np.asarray(fr['net_pct'], float)
    if n > 1 and np.any(np.diff(fr['entry_e']) < 0):
        # The dump is in train_meta's argsort(ts) order, which is ascending
        # entry; a hand-edited frame is re-sorted (stable) so the folds hold.
        o = np.argsort(fr['entry_e'], kind='stable')
        for k in REQUIRED_NPZ:
            if k != 'feature_names':
                fr[k] = np.asarray(fr[k])[o]
        side = dict(side, _resorted_by_entry=True)
    return fr, side


def book_name(side):
    p = str(side.get('prefix') or '')
    return 'stock' if p.startswith('stock') else (side.get('asset_type') or 'crypto')


# ---------------------------------------------------------------------------
# Outer folds
# ---------------------------------------------------------------------------

def outer_folds_purged(entry, exit_, k=5, embargo=0.05):
    """Main outer loop: calibration.purged_kfold_indices (:237) with an
    explicit embargo (fraction of the test span when in (0,1), else absolute
    seconds)."""
    from calibration import purged_kfold_indices
    return purged_kfold_indices(np.asarray(entry, float),
                                np.asarray(exit_, float), k=k, embargo=embargo)


def outer_folds_forward(entry, exit_, k=5, embargo=0.05):
    """Forward-chaining sensitivity folds: k+1 contiguous time-ordered blocks;
    fold i (1..k) tests block i and trains on rows BEFORE it whose label span
    ends before the test start minus the embargo gap (gap = embargo x test
    span when embargo in (0,1), else absolute). Block 0 is never scored."""
    entry = np.asarray(entry, float)
    exit_ = np.asarray(exit_, float)
    n = len(entry)
    k = max(1, min(int(k), n - 1)) if n > 1 else 0
    if k == 0:
        return []
    edges = np.linspace(0, n, k + 2).astype(int)
    folds = []
    for i in range(1, k + 1):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        test = np.arange(a, b)
        t_start = entry[test].min()
        span = max(exit_[test].max() - t_start, 0.0)
        gap = embargo * span if 0.0 < embargo < 1.0 else embargo
        prior = np.arange(0, a)
        train = prior[exit_[prior] < t_start - gap]
        folds.append((train, test))
    return folds


# ---------------------------------------------------------------------------
# Recipes (lightgbm + sklearn imported lazily)
# ---------------------------------------------------------------------------

def _o_config(which):
    """(embargo, v2) that train_meta's purged branch would use. 'O' = today's
    CALIBRATION_V2 (meta_label._calibration_embargo :134); 'OV' = V2 on."""
    if which == 'OV':
        return 0.05, True
    try:
        import strategy_config as _sc
        v2 = bool(getattr(_sc, 'CALIBRATION_V2', False))
    except Exception:
        v2 = False
    return (0.05 if v2 else 0.0), v2


def lgb_params(side_params, threads=1):
    p = dict(side_params)
    for k in ('n_jobs', 'num_threads', 'nthread', 'num_thread'):
        p.pop(k, None)
    p['num_threads'] = int(threads)
    p['verbose'] = -1
    return p


def fit_fold_arms(Xtr, ytr, etr, xtr, Xte, params, arms, feature_names,
                  min_train=50):
    """Re-run the train_meta recipes on ONE outer-train set and return
    ({arm: p over Xte}, info). An arm that cannot be fitted gives NaN rows."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from calibration import crossfit_oof_predict, fit_calibrator
    from meta_label import _calib_slice_guard, _calibrated

    nte = len(Xte)
    nan = np.full(nte, np.nan)
    out = {a: nan.copy() for a in arms}
    info = {'n_train': int(len(ytr)), 'n_test': int(nte)}
    n = len(ytr)
    s = int(n * 0.8)                       # meta_label.py:1235
    if (n < min_train or np.unique(ytr[:s]).size < 2
            or np.unique(ytr[s:]).size < 2):
        info['skipped'] = 'outer-train too thin or one-class split'
        return out, info
    fn = [str(f) for f in feature_names]
    train_set = lgb.Dataset(Xtr[:s], label=ytr[:s], feature_name=fn)
    val_set = lgb.Dataset(Xtr[s:], label=ytr[s:], reference=train_set)
    booster = lgb.train(params, train_set, num_boost_round=NUM_BOOST_ROUND,
                        valid_sets=[val_set],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    info['best_iteration'] = int(booster.best_iteration or 0)
    raw_val = np.asarray(booster.predict(Xtr[s:]), float)
    raw_te = np.asarray(booster.predict(Xte), float)
    y_sl = np.asarray(ytr[s:], float)
    guard = _calib_slice_guard(raw_val, y_sl)
    info['legacy_guard'] = guard
    p_L = nan.copy()
    if guard is None:
        cal = IsotonicRegression(out_of_bounds='clip').fit(raw_val, y_sl)
        p_L = _calibrated(cal, raw_te)
    if 'L' in out:
        out['L'] = p_L
    if 'R' in out:
        out['R'] = np.clip(raw_te, 0.0, 1.0)
    if 'V' in out and guard is None:
        cal = fit_calibrator(raw_val, y_sl, v2=True)
        if cal is not None:
            out['V'] = _calibrated(cal, raw_te)
        info['V_method'] = getattr(cal, 'method_', None)
    n_iter = int(booster.best_iteration or 200)   # meta_label.py:1272
    oof_cache = {}
    for a in ('O', 'OV'):
        if a not in out:
            continue
        emb, v2 = _o_config(a)
        if emb not in oof_cache:
            def _fp(Xa, ya, Xb):
                ds = lgb.Dataset(Xa, label=ya, feature_name=fn)
                b = lgb.train(params, ds, num_boost_round=n_iter)
                return b.predict(Xb)
            oof_cache[emb] = crossfit_oof_predict(_fp, Xtr, ytr, etr, xtr,
                                                  k=INNER_K, embargo=emb)
        cal = fit_calibrator(oof_cache[emb], ytr, v2=v2)
        if cal is not None:
            out[a] = _calibrated(cal, raw_te)
            info[f'{a}_used'] = 'purged_oof'
            info[f'{a}_method'] = getattr(cal, 'method_', None)
        else:                                  # meta_label.py:1288-1300
            out[a] = p_L.copy()
            info[f'{a}_used'] = 'legacy_fallback'
        info[f'{a}_embargo'] = emb
        info[f'{a}_v2'] = v2
    return out, info


def run_outer(fr, folds, params, arms):
    n = len(fr['y'])
    P = {a: np.full(n, np.nan) for a in arms}
    fold_of = np.full(n, -1, dtype=int)
    infos = []
    for fi, (tr, te) in enumerate(folds):
        fold_of[te] = fi
        p, info = fit_fold_arms(fr['X'][tr], fr['y'][tr], fr['entry_e'][tr],
                                fr['exit_e'][tr], fr['X'][te], params, arms,
                                fr['feature_names'])
        for a in arms:
            P[a][te] = p[a]
        info['fold'] = fi
        infos.append(info)
        print(f"[FOLD {fi}] train={info['n_train']} test={info['n_test']} "
              f"best_iter={info.get('best_iteration')} "
              f"{info.get('skipped') or ''}")
    return P, fold_of, infos


# ---------------------------------------------------------------------------
# Metrics + bootstrap (pure numpy)
# ---------------------------------------------------------------------------

def log_loss_rows(p, y, eps=1e-4):
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def ece_equal_mass(p, y, n_bins=10):
    """Equal-mass (quantile) ECE: rows sorted by p, split into n_bins
    equal-count bins; sum_b n_b/N |mean p_b - mean y_b|."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    N = len(p)
    if N == 0:
        return None
    o = np.argsort(p, kind='stable')
    ece = 0.0
    for idx in np.array_split(o, min(int(n_bins), N)):
        if len(idx):
            ece += len(idx) / N * abs(float(p[idx].mean()) - float(y[idx].mean()))
    return float(ece)


def week_blocks(entry_e, block_weeks=1):
    w = np.floor((np.asarray(entry_e, float) - MONDAY_OFFSET) / WEEK_SECONDS)
    return (w // max(1, int(block_weeks))).astype(np.int64)


def block_bootstrap_mean(v, blocks, B, rng):
    """Bootstrap distribution of mean(v) resampling whole blocks with
    replacement (row-weighted: sum of drawn block sums / drawn row count)."""
    v = np.asarray(v, float)
    ub, inv = np.unique(blocks, return_inverse=True)
    S = np.bincount(inv, weights=v, minlength=len(ub))
    C = np.bincount(inv, minlength=len(ub)).astype(float)
    nb = len(ub)
    draws = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=int(B)).astype(float)
    return (draws @ S) / np.maximum(draws @ C, 1.0)


def block_bootstrap_sum(v, blocks, B, rng):
    """Bootstrap distribution of sum(v) with whole-block resampling."""
    v = np.asarray(v, float)
    ub, inv = np.unique(blocks, return_inverse=True)
    S = np.bincount(inv, weights=v, minlength=len(ub))
    nb = len(ub)
    draws = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=int(B)).astype(float)
    return draws @ S


def _ci(dist):
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return [float(lo), float(hi)]


def score_arms(P, y, net, entry_e, arms, B=5000, block_weeks=1, seed=0,
               eps=1e-4, ece_B=None):
    """Per-arm table + paired statistics vs L on the jointly-finite rows."""
    from calibration import expected_calibration_error
    from meta_label import meta_size_mult
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    net = np.asarray(net, float)
    m = np.ones(len(y), bool)
    for a in arms:
        m &= np.isfinite(P[a])
    idx = np.where(m)[0]
    yy, nn = y[idx], net[idx]
    blocks = week_blocks(np.asarray(entry_e, float)[idx], block_weeks)
    n_blocks = int(len(np.unique(blocks))) if len(idx) else 0
    res = {'n_scored': int(len(idx)), 'n_dropped': int(len(y) - len(idx)),
           'n_blocks': n_blocks, 'block_weeks': int(block_weeks), 'B': int(B),
           'll_eps': eps, 'arms': {}}
    if len(idx) == 0 or 'L' not in arms:
        return res
    size = np.vectorize(meta_size_mult, otypes=[float])
    ll = {a: log_loss_rows(P[a][idx], yy, eps) for a in arms}
    br = {a: (P[a][idx] - yy) ** 2 for a in arms}
    keep = {a: P[a][idx] >= VETO_P for a in arms}
    pnl = {a: nn * keep[a] for a in arms}
    pnl_sized = {a: nn * keep[a] * size(P[a][idx]) for a in arms}
    ece_B = int(min(B, 2000) if ece_B is None else ece_B)
    # ECE is not a mean, so its CI resamples row indices block-wise; draws are
    # generated one at a time (never materialised: B x n indices is ~160 MB
    # at stock scale).
    ub, inv = np.unique(blocks, return_inverse=True)
    members = [np.where(inv == j)[0] for j in range(len(ub))]
    Pi = {a: P[a][idx] for a in arms}
    ece_d = {a: np.empty(ece_B) for a in arms} if n_blocks > 1 else {}
    if n_blocks > 1:
        for t in range(ece_B):
            dd = np.concatenate([members[j] for j in
                                 rng.integers(0, len(ub), len(ub))])
            for a in arms:
                ece_d[a][t] = ece_equal_mass(Pi[a][dd], yy[dd])
    ece_L = ece_equal_mass(P['L'][idx], yy)
    for a in arms:
        pa = P[a][idx]
        row = {
            'll': float(ll[a].mean()), 'brier': float(br[a].mean()),
            'ece': ece_equal_mass(pa, yy),
            'ece_width': expected_calibration_error(pa, yy),
            'veto_rate': float(np.mean(pa < VETO_P)),
            'flip_rate_vs_L': float(np.mean((pa < VETO_P) != (P['L'][idx] < VETO_P))),
            'pnl': float(pnl[a].sum()), 'pnl_sized': float(pnl_sized[a].sum()),
            'p_mean': float(pa.mean()),
        }
        if a != 'L' and n_blocks > 1:
            d = ll[a] - ll['L']
            row['d_ll'] = float(d.mean())
            row['d_ll_ci95'] = _ci(block_bootstrap_mean(d, blocks, B, rng))
            db = br[a] - br['L']
            row['d_brier'] = float(db.mean())
            row['d_brier_ci95'] = _ci(block_bootstrap_mean(db, blocks, B, rng))
            dp = pnl[a] - pnl['L']
            row['d_pnl'] = float(dp.sum())
            row['d_pnl_ci95'] = _ci(block_bootstrap_sum(dp, blocks, B, rng))
            row['d_ece'] = float(row['ece'] - ece_L)
            row['d_ece_ci95'] = _ci(ece_d[a] - ece_d['L'])
        elif a != 'L':
            row['d_ll'] = float((ll[a] - ll['L']).mean())
            row['d_ll_ci95'] = None
        res['arms'][a] = row
    res['ece_B'] = ece_B
    return res


# ---------------------------------------------------------------------------
# Verdict (pure) — the pre-registered rule
# ---------------------------------------------------------------------------

def arm_verdict(arm, L, n_meta, fc_point=None, min_n=MIN_N_META,
                ece_tol=ECE_TOL, flip_max=FLIP_MAX):
    """(verdict, reason) for one arm vs L. `arm`/`L` are score_arms rows;
    fc_point = the forward-chain point d_ll (None = sensitivity not run)."""
    if n_meta < min_n:
        return 'INSUFFICIENT', f'n_meta={n_meta} < {min_n}'
    ci = arm.get('d_ll_ci95')
    if not ci:
        return 'INCONCLUSIVE', 'no bootstrap CI (single block)'
    lo, hi = ci
    ece_ok = (arm['ece'] is not None and L['ece'] is not None
              and arm['ece'] <= L['ece'] + ece_tol)
    pnl_ok = arm['pnl'] >= L['pnl']
    flip_ok = arm['flip_rate_vs_L'] < flip_max
    drift = (fc_point is not None and np.isfinite(fc_point)
             and np.sign(fc_point) * np.sign(arm['d_ll']) < 0)
    if not pnl_ok:
        return 'NO-GO', f"P&L {arm['pnl']:.3f} < L {L['pnl']:.3f}"
    if drift:
        return 'INCONCLUSIVE', (f"drift: forward-chain d={fc_point:+.5f} vs "
                                f"main d={arm['d_ll']:+.5f}")
    if lo > 0:
        return 'NO-GO', f'CI favours L [{lo:+.5f}, {hi:+.5f}]'
    if hi < 0 and ece_ok and flip_ok:
        return 'SWITCH-CANDIDATE', (f'CI [{lo:+.5f}, {hi:+.5f}] < 0, ECE ok, '
                                    f'P&L ok, flip {arm["flip_rate_vs_L"]:.1%}')
    why = []
    if not hi < 0:
        why.append(f'CI spans 0 [{lo:+.5f}, {hi:+.5f}]')
    if not ece_ok:
        why.append(f"ECE {arm['ece']} > L {L['ece']} + {ece_tol}")
    if not flip_ok:
        why.append(f"flip {arm['flip_rate_vs_L']:.1%} >= {flip_max:.0%}")
    return 'INCONCLUSIVE', '; '.join(why)


def book_verdict(arm_verdicts, arm_rows, n_meta, min_n=MIN_N_META):
    """Mechanical book verdict over the switchable arms (V/O/OV)."""
    if n_meta < min_n:
        return 'INSUFFICIENT', f'n_meta={n_meta} < {min_n} (starved, not decided)'
    sw = [a for a in SWITCHABLE if a in arm_verdicts]
    if not sw:
        return 'INCONCLUSIVE', 'no switchable arm measured'
    cands = [a for a in sw if arm_verdicts[a][0] == 'SWITCH-CANDIDATE']
    if cands:
        best = min(cands, key=lambda a: arm_rows[a]['ll'])
        return f'SWITCH-CANDIDATE({best})', f'passing arms {cands}'
    if all(arm_verdicts[a][0] == 'NO-GO' for a in sw):
        return 'NO-GO', 'every switchable arm NO-GO'
    return 'INCONCLUSIVE', 'no arm passes; not all NO-GO — keep legacy'


def global_verdict(books, ece_tol=ECE_TOL, flip_max=FLIP_MAX):
    """Cross-book line (the flags are global): an arm is a GLOBAL
    switch-candidate iff SWITCH-CANDIDATE on >= 1 book and, on every other
    book, point d_ll <= 0 with ECE / P&L / flip conditions holding."""
    out = {}
    names = list(books)
    for a in SWITCHABLE:
        per = {b: books[b] for b in names if a in books[b].get('arm_verdicts', {})}
        if len(per) < 2:
            continue
        passing = [b for b in per if per[b]['arm_verdicts'][a][0] == 'SWITCH-CANDIDATE']
        ok_other = True
        for b in per:
            if b in passing:
                continue
            r = per[b]['score']['arms'][a]
            L = per[b]['score']['arms']['L']
            if not (r.get('d_ll', 1) <= 0 and r['ece'] is not None
                    and L['ece'] is not None and r['ece'] <= L['ece'] + ece_tol
                    and r['pnl'] >= L['pnl'] and r['flip_rate_vs_L'] < flip_max):
                ok_other = False
        out[a] = ('GLOBAL-SWITCH-CANDIDATE' if passing and ok_other
                  else 'NO-GLOBAL-SWITCH')
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _refuse_root(path):
    p = Path(path).expanduser().resolve()
    return p == _ROOT.resolve() or p.parent == _ROOT.resolve()


def run_book(fr, side, args):
    params = lgb_params(side['params'], args.threads)
    n = len(fr['y'])
    folds = outer_folds_purged(fr['entry_e'], fr['exit_e'], args.k, args.embargo)
    t0 = time.time()
    P, fold_of, infos = run_outer(fr, folds, params, args.arms)
    score = score_arms(P, fr['y'], fr['net_pct'], fr['entry_e'], args.arms,
                       B=args.B, block_weeks=args.block_weeks, seed=args.seed,
                       eps=args.ll_eps)
    res = {'n_rows': n, 'n_meta': score['n_scored'], 'folds': infos,
           'score': score, 'sidecar_best_iteration': side.get('best_iteration'),
           'sidecar_calibration_used': (side.get('calibration') or {}).get('used')}
    fc_points = {}
    P_fc = None
    if args.forward_chain:
        ffolds = outer_folds_forward(fr['entry_e'], fr['exit_e'], args.k, args.embargo)
        P_fc, _, finfos = run_outer(fr, ffolds, params, args.arms)
        fsc = score_arms(P_fc, fr['y'], fr['net_pct'], fr['entry_e'], args.arms,
                         B=args.B, block_weeks=args.block_weeks,
                         seed=args.seed + 1, eps=args.ll_eps)
        res['forward_chain'] = {'folds': finfos, 'score': fsc}
        fc_points = {a: r.get('d_ll') for a, r in fsc['arms'].items() if a != 'L'}
    rows = score['arms']
    av = {}
    for a in args.arms:
        if a == 'L' or a not in rows or 'L' not in rows:
            continue
        av[a] = arm_verdict(rows[a], rows['L'], score['n_scored'],
                            fc_point=fc_points.get(a))
    res['arm_verdicts'] = av
    res['verdict'], res['verdict_reason'] = (
        book_verdict(av, rows, score['n_scored']) if rows
        else ('INSUFFICIENT', 'nothing scored'))
    res['wall_s'] = round(time.time() - t0, 1)
    res['_P'] = P
    res['_P_fc'] = P_fc
    res['_fold_of'] = fold_of
    return res


def print_table(book, res):
    sc = res['score']
    print(f"\n[{book}] n_rows={res['n_rows']} scored={sc['n_scored']} "
          f"dropped={sc['n_dropped']} blocks={sc['n_blocks']} B={sc['B']}")
    print(f"  {'arm':<4}{'LL':>9}{'Brier':>9}{'ECEm':>8}{'ECEw':>8}{'veto%':>7}"
          f"{'flip%':>7}{'P&L':>10}{'  d_LL [95% CI]':<34}{'verdict'}")
    for a, r in sc['arms'].items():
        ci = r.get('d_ll_ci95')
        dtxt = (f"  {r['d_ll']:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}]" if ci
                else ('  (reference)' if a == 'L' else f"  {r.get('d_ll', float('nan')):+.5f} [n/a]"))
        v = res['arm_verdicts'].get(a, ('', ''))[0] if a != 'L' else 'baseline'
        if a == 'R' and v:
            v += ' (reference only)'
        ece = r['ece'] if r['ece'] is not None else float('nan')
        ecw = r['ece_width'] if r['ece_width'] is not None else float('nan')
        print(f"  {a:<4}{r['ll']:>9.5f}{r['brier']:>9.5f}{ece:>8.4f}{ecw:>8.4f}"
              f"{r['veto_rate']:>7.1%}{r['flip_rate_vs_L']:>7.1%}{r['pnl']:>10.2f}"
              f"{dtxt:<34}{v}")
    fc = res.get('forward_chain')
    if fc:
        pts = {a: r.get('d_ll') for a, r in fc['score']['arms'].items() if a != 'L'}
        print(f"  forward-chain d_LL points: "
              + ', '.join(f"{a}={v:+.5f}" for a, v in pts.items() if v is not None))


def dry_run(args, dumps):
    print("[DRY-RUN] meta_calib_nested — nothing fitted, nothing written")
    print(RULE_TEXT)
    for path, fr, side, err in dumps:
        if err:
            print(f"[DRY-RUN] {path}: NOT USABLE — {err}")
            continue
        n = len(fr['y'])
        folds = outer_folds_purged(fr['entry_e'], fr['exit_e'], args.k, args.embargo)
        n_o = sum(1 for a in args.arms if a in ('O', 'OV'))
        boosters = len(folds) * (1 + INNER_K * n_o) * (2 if args.forward_chain else 1)
        print(f"[DRY-RUN] {book_name(side)}: n={n} X={fr['X'].shape} "
              f"base_rate={fr['y'].mean():.3f} weeks={len(np.unique(week_blocks(fr['entry_e'])))} "
              f"outer folds={len(folds)} (train sizes {[len(t) for t, _ in folds]}) "
              f"arms={args.arms} ~{boosters} lightgbm fits, B={args.B} "
              f"{'(n < 500 -> INSUFFICIENT)' if n < MIN_N_META else ''}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Nested OOF-of-OOF meta-calibration '
                                             'study over a METADUMP frame (measurement-only)')
    ap.add_argument('--npz', action='append', required=True,
                    help='{p}meta_frame.npz (repeat once per book; sidecar = sibling .json)')
    ap.add_argument('--json', type=str, default=None, help='summary JSON output path')
    ap.add_argument('--out', type=str, default=None,
                    help='output dir (default logs/meta_calib_nested/)')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--embargo', type=float, default=0.05)
    ap.add_argument('--B', type=int, default=5000)
    ap.add_argument('--block-weeks', type=int, default=1)
    ap.add_argument('--arms', type=str, default='L,V,O,R')
    ap.add_argument('--forward-chain', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1, help='LightGBM num_threads')
    ap.add_argument('--ll-eps', type=float, default=1e-4)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    arms = [x.strip().upper() for x in a.arms.split(',') if x.strip()]
    bad = [x for x in arms if x not in ALL_ARMS]
    if bad:
        ap.error(f'unknown arms {bad}; choose from {ALL_ARMS}')
    if 'L' not in arms:
        arms = ['L'] + arms
    a.arms = arms
    return a


def main(argv=None):
    try:
        args = parse_args(argv)
    except SystemExit:
        return 0                     # argparse error/--help: exit 0 always
    try:
        dumps = []
        for pth in args.npz:
            try:
                fr, side = load_dump(pth)
                dumps.append((pth, fr, side, None))
            except Exception as e:
                dumps.append((pth, None, None, f'{type(e).__name__}: {e}'))
        if args.dry_run:
            dry_run(args, dumps)
            return 0
        out_dir = Path(args.out).expanduser() if args.out else _ROOT / 'logs' / 'meta_calib_nested'
        print("meta_calib_nested — design research_signal.md SCOUT-4 T2(b)")
        print(RULE_TEXT)
        books = {}
        for pth, fr, side, err in dumps:
            if err:
                print(f"[REFUSED] {pth}: {err}")
                continue
            b = book_name(side)
            if b in books:
                print(f"[REFUSED] {pth}: second dump for book {b} (books are never pooled)")
                continue
            res = run_book(fr, side, args)
            books[b] = res
            print_table(b, res)
            print(f"VERDICT[{b}]: {res['verdict']} — {res['verdict_reason']}")
        glob = global_verdict(books) if len(books) > 1 else {}
        for a, v in glob.items():
            print(f"GLOBAL[{a}]: {v}")
        if not books:
            return 0
        write_ok = not _refuse_root(out_dir)
        if not write_ok:
            print(f"[REFUSED] --out {out_dir} is the repo root — nothing written")
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        dest = (Path(args.json).expanduser() if args.json
                else out_dir / f'meta_calib_nested_{stamp}.json')
        if _refuse_root(dest):
            print(f"[REFUSED] --json {dest} is in the repo root — JSON not written")
            dest = None
        summary = {
            'tool': 'meta_calib_nested', 'design': 'research_signal.md SCOUT-4 T2(b)',
            'utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'args': vars(args), 'rule': RULE_TEXT, 'global': glob,
            'books': {b: {k: v for k, v in r.items() if not k.startswith('_')}
                      for b, r in books.items()},
        }
        if write_ok:
            out_dir.mkdir(parents=True, exist_ok=True)
            for b, r in books.items():
                cols = {f'p_{a}': r['_P'][a] for a in args.arms}
                if r['_P_fc'] is not None:
                    cols.update({f'p_fc_{a}': r['_P_fc'][a] for a in args.arms})
                rp = out_dir / f'{b}_meta_calib_nested_rows.npz'
                tmp = f'{rp}.tmp.{os.getpid()}'
                with open(tmp, 'wb') as f:
                    np.savez_compressed(f, outer_fold=r['_fold_of'], **cols)
                os.replace(tmp, rp)
                print(f"[OUT] {rp}")
        if dest is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = f'{dest}.tmp.{os.getpid()}'
            with open(tmp, 'w') as f:
                json.dump(summary, f, indent=1,
                          default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
            os.replace(tmp, dest)
            print(f"[OUT] {dest}")
    except Exception as e:  # measurement: never raise past main
        print(f"[ERROR] meta_calib_nested failed: {type(e).__name__}: {e}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
