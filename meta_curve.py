"""B04.3 learning-curve kernel (campaign 2026-08, packet V3) — pure numpy.

Measurement-only math for the meta-label sample-size learning curve:
temporal-block subsampling plans, tie-aware rank AUC, cross-seed veto
flip rates, inverse-power-law fits (Figueroa et al. 2012) and the
empirical "honest floor" criteria from 02_research.md B04.3:

    floor = smallest n with (plateau_AUC - mean_AUC) < 0.01
            AND cross-seed veto flip-rate < 10%

Consumed by scripts/meta_learning_curve.py on the Jetson (which assembles
the real meta-row population via meta_label's own helpers) and by
tests/test_c26_V3.py on the Mac. Everything here is pure numpy + stdlib —
no pandas, no heavy deps — and fail-soft: degenerate inputs return dicts
with ok=False / None fields, never raise (except on caller programming
errors such as shape mismatches).
"""

import json
import math

import numpy as np

DEFAULT_N_GRID = (100, 200, 400, 800, 1600, 3200)
DEFAULT_N_SEEDS = 20
DEFAULT_BLOCK_LEN = 50          # rows per contiguous block (~2 days of hourly trades)
DEFAULT_BASE_SEED = 20260818
AUC_PLATEAU_TOL = 0.01          # B04.3 binding: (plateau_AUC - mean_AUC) < 0.01
FLIP_RATE_TOL = 0.10            # B04.3 binding: cross-seed veto flip-rate < 10%
VETO_PROB = 0.30                # mirror of meta_label.META_VETO_PROB — do NOT
                                # import meta_label here (pure module); the value
                                # is pinned against meta_label.META_VETO_PROB by
                                # tests/test_c26_V3.py, which CAN import it on the
                                # Mac (meta_label's module top is numpy+stdlib).


def _py(obj):
    """Recursively coerce numpy scalars/arrays to plain python so that
    json.dumps(report) always succeeds."""
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, str) else k: _py(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_py(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def resolve_n_grid(n_pool, n_grid=DEFAULT_N_GRID) -> list:
    """Grid values usable against a pool of n_pool rows: sorted unique grid
    entries strictly below n_pool, with n_pool itself appended once as the
    'all' point. Values > n_pool are dropped. Empty pool -> []."""
    n_pool = int(n_pool)
    if n_pool <= 0:
        return []
    keep = sorted({int(v) for v in n_grid if 0 < int(v) < n_pool})
    return keep + [n_pool]


def build_subsample_plan(n_pool, n_grid=None, n_seeds=DEFAULT_N_SEEDS,
                         block_len=DEFAULT_BLOCK_LEN,
                         base_seed=DEFAULT_BASE_SEED) -> list:
    """Deterministic TEMPORAL BLOCK subsampling plan (never iid rows).

    For each resolved n and seed: partition the pool into consecutive
    segments of block_len rows, draw ceil(n/block_len) distinct segments
    with a per-(n, seed) generator, sort them, concatenate their row
    ranges and trim the tail of the LAST block to exactly n rows. The
    n == n_pool 'all' point is arange(n_pool) for every seed (seed
    variation then reaches only the learner). Each draw:
    {'n': int, 'seed': int, 'idx': sorted unique int64 ndarray}.
    """
    n_pool = int(n_pool)
    if n_grid is None:
        n_grid = DEFAULT_N_GRID
    plan = []
    for n in resolve_n_grid(n_pool, n_grid):
        for s in range(int(n_seeds)):
            if n == n_pool:
                idx = np.arange(n_pool, dtype=np.int64)
            else:
                rng = np.random.default_rng(int(base_seed)
                                            + 1_000_003 * s + n)
                n_segments = n_pool // int(block_len)
                m = int(math.ceil(n / float(block_len)))
                if m > n_segments:
                    # Pool too small relative to n for block sampling
                    # (can't happen after resolve_n_grid with
                    # block_len <= n, but guard anyway): one contiguous
                    # run of n rows at a seeded random start.
                    start = int(rng.integers(0, n_pool - n + 1))
                    idx = np.arange(start, start + n, dtype=np.int64)
                else:
                    segs = np.sort(rng.choice(n_segments, size=m,
                                              replace=False))
                    rows = np.concatenate(
                        [np.arange(g * block_len, (g + 1) * block_len,
                                   dtype=np.int64) for g in segs])
                    idx = rows[:n]
            plan.append({'n': int(n), 'seed': int(s), 'idx': idx})
    return plan


def rank_auc(scores, labels):
    """Mann-Whitney AUC with average-rank tie handling, pure numpy.
    Non-finite pairs are dropped. Returns None when a class is absent or
    fewer than 2 usable rows remain."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, float)
    if s.shape != y.shape:
        raise ValueError(f'rank_auc: shape mismatch {s.shape} vs {y.shape}')
    mask = np.isfinite(s) & np.isfinite(y)
    s, y = s[mask], y[mask]
    n = s.size
    if n < 2:
        return None
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    sorted_s = np.sort(s, kind='mergesort')
    # Average (mid) ranks, 1-based: (left+right+1)/2 over each tie group.
    ranks = (np.searchsorted(sorted_s, s, side='left')
             + np.searchsorted(sorted_s, s, side='right') + 1) / 2.0
    rank_sum_pos = float(ranks[y == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def veto_flip_rate(p_matrix, veto_prob=VETO_PROB, groups=None) -> dict:
    """Cross-seed veto-decision flip rate on a FIXED eval set.

    p_matrix: (n_seeds, n_units) calibrated p; a seed row containing any
    non-finite value is a failed seed and is ignored. Requires >= 2 usable
    seeds. Ungrouped ('row' level): a unit flips iff seeds disagree on
    p < veto_prob. With `groups` (len n_units symbol labels, 'symbol'
    level): per seed the group decision is median(p over the group's
    units) < veto_prob, and the flip rate is over groups — B04.3's
    'fraction of symbols whose veto decision flips'.
    """
    P = np.asarray(p_matrix, float)
    if P.ndim != 2:
        raise ValueError(f'veto_flip_rate: p_matrix must be 2-D, got {P.ndim}-D')
    usable = np.isfinite(P).all(axis=1)
    k = int(usable.sum())
    level = 'row' if groups is None else 'symbol'
    if k < 2:
        return {'flip_rate': None, 'n_units': 0, 'n_seeds_used': k,
                'level': level}
    P = P[usable]
    if groups is None:
        dec = P < veto_prob                       # (k, n_units)
    else:
        g = np.asarray(groups)
        if g.shape[0] != P.shape[1]:
            raise ValueError(f'veto_flip_rate: groups len {g.shape[0]} != '
                             f'n_units {P.shape[1]}')
        uniq, inv = np.unique(g, return_inverse=True)
        med = np.empty((P.shape[0], uniq.size), float)
        for gi in range(uniq.size):
            med[:, gi] = np.median(P[:, inv == gi], axis=1)
        dec = med < veto_prob                     # (k, n_groups)
    flips = dec.any(axis=0) & ~dec.all(axis=0)
    n_units = int(dec.shape[1])
    return {'flip_rate': float(flips.mean()) if n_units else None,
            'n_units': n_units, 'n_seeds_used': k, 'level': level}


def fit_power_law(n, err, c_grid_size=64) -> dict:
    """Fit err(n) = a * n^(-b) + c by log-space least squares over a c grid
    (pure numpy — the packet's stated method), picking the c minimizing
    ORIGINAL-space SSE. Fail-soft: ok=False with a reason on degenerate
    input, a non-decreasing curve (b <= 0), or non-finite results."""
    bad = {'ok': False, 'reason': None, 'a': None, 'b': None, 'c': None,
           'r2': None, 'sse': None, 'n_points': 0, 'plateau_err': None}
    n_arr = np.asarray(n, float)
    e_arr = np.asarray(err, float)
    if n_arr.shape != e_arr.shape:
        raise ValueError(f'fit_power_law: shape mismatch {n_arr.shape} '
                         f'vs {e_arr.shape}')
    mask = np.isfinite(n_arr) & np.isfinite(e_arr) & (n_arr > 0) & (e_arr > 0)
    n_arr, e_arr = n_arr[mask], e_arr[mask]
    bad['n_points'] = int(n_arr.size)
    if np.unique(n_arr).size < 3:
        bad['reason'] = f'too_few_points(distinct_n={np.unique(n_arr).size})'
        return bad
    ln_n = np.log(n_arr)
    A = np.column_stack([np.ones_like(ln_n), ln_n])
    ss_tot = float(np.sum((e_arr - e_arr.mean()) ** 2))
    best = None
    for c in np.linspace(0.0, float(e_arr.min()) * (1.0 - 1e-6),
                         int(c_grid_size)):
        z = e_arr - c
        if np.any(z <= 0):
            continue
        coef, *_ = np.linalg.lstsq(A, np.log(z), rcond=None)
        log_a, neg_b = coef
        a = math.exp(float(log_a))
        b = -float(neg_b)
        pred = a * n_arr ** (-b) + c
        sse = float(np.sum((e_arr - pred) ** 2))
        if math.isfinite(sse) and (best is None or sse < best['sse']):
            best = {'a': a, 'b': b, 'c': float(c), 'sse': sse}
    if best is None:
        bad['reason'] = 'no_finite_fit'
        return bad
    if not all(math.isfinite(v) for v in (best['a'], best['b'], best['c'])):
        bad['reason'] = 'nonfinite_params'
        return bad
    if best['b'] <= 1e-9:   # <=0 plus float noise: a flat curve fits b~1e-17
        bad['reason'] = f"b_nonpositive(b={best['b']:.4g})"
        return bad
    r2 = 1.0 - best['sse'] / ss_tot if ss_tot > 0 else None
    return {'ok': True, 'reason': None, 'a': float(best['a']),
            'b': float(best['b']), 'c': float(best['c']),
            'r2': None if r2 is None else float(r2),
            'sse': float(best['sse']), 'n_points': int(n_arr.size),
            'plateau_err': float(best['c'])}


def n_for_target(fit, target_err):
    """Invert err(n) = a*n^(-b) + c: n = (a / (target_err - c)) ** (1/b).
    None when the fit failed, the target is at/below the plateau
    (unreachable), or the inversion is non-finite/non-positive."""
    if not isinstance(fit, dict) or not fit.get('ok'):
        return None
    a, b, c = fit['a'], fit['b'], fit['c']
    try:
        t = float(target_err)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(t) or t <= c:
        return None
    try:
        val = (a / (t - c)) ** (1.0 / b)
    except (OverflowError, ZeroDivisionError):
        return None
    if not math.isfinite(val) or val <= 0:
        return None
    return float(val)


def empirical_floor(grid_stats, plateau_auc, auc_tol=AUC_PLATEAU_TOL,
                    flip_tol=FLIP_RATE_TOL):
    """Smallest n in grid_stats (sorted by n; rows {'n','auc_mean',
    'flip_rate'}) satisfying BOTH B04.3 criteria:
    plateau_auc - auc_mean < auc_tol AND flip_rate < flip_tol.
    None when nothing qualifies (starvation verdict)."""
    if plateau_auc is None:
        return None
    for row in sorted(grid_stats, key=lambda r: r['n']):
        auc_mean = row.get('auc_mean')
        flip = row.get('flip_rate')
        if auc_mean is None or flip is None:
            continue
        if (float(plateau_auc) - float(auc_mean) < auc_tol
                and float(flip) < flip_tol):
            return int(row['n'])
    return None


def assemble_report(records, flip_by_n, meta=None) -> dict:
    """Aggregate per-(n, seed) records + per-n flip results into the
    meta_curve_report/1 schema. All values are plain python (json-safe).

    records: dicts {'n','seed','auc','frac_below_veto','p_q10','p_median',
    'p_q90','calib','error'} (error=str for failed draws, metrics None).
    flip_by_n: {n: veto_flip_rate(...) result}.
    """
    by_n = {}
    for rec in records:
        by_n.setdefault(int(rec['n']), []).append(rec)

    grid = []
    for n in sorted(by_n):
        recs = by_n[n]
        ok_recs = [r for r in recs if r.get('error') is None]
        aucs = np.asarray([r['auc'] for r in ok_recs
                           if r.get('auc') is not None], float)
        aucs = aucs[np.isfinite(aucs)]
        fracs = np.asarray([r['frac_below_veto'] for r in ok_recs
                            if r.get('frac_below_veto') is not None], float)
        fracs = fracs[np.isfinite(fracs)]
        flip = (flip_by_n or {}).get(n, {})
        grid.append({
            'n': int(n),
            'auc_mean': float(aucs.mean()) if aucs.size else None,
            'auc_std': float(aucs.std()) if aucs.size else None,
            'frac_below_veto_mean': float(fracs.mean()) if fracs.size else None,
            'flip_rate': flip.get('flip_rate'),
            'flip_level': flip.get('level'),
            'flip_n_units': flip.get('n_units'),
            'n_ok': len(ok_recs),
            'n_err': len(recs) - len(ok_recs),
        })

    # Power law on err = 1 - mean AUC over n.
    xs = [g['n'] for g in grid if g['auc_mean'] is not None]
    errs = [1.0 - g['auc_mean'] for g in grid if g['auc_mean'] is not None]
    auc_fit = fit_power_law(xs, errs)
    if auc_fit['ok']:
        plateau_auc = 1.0 - auc_fit['c']
    else:
        obs = [g['auc_mean'] for g in grid if g['auc_mean'] is not None]
        plateau_auc = max(obs) if obs else None

    # Second power law on the per-n flip rate.
    xf = [g['n'] for g in grid if g['flip_rate'] is not None]
    ff = [g['flip_rate'] for g in grid if g['flip_rate'] is not None]
    flip_fit = fit_power_law(xf, ff)

    floor_empirical = empirical_floor(grid, plateau_auc)
    # n where the mean AUC comes within AUC_PLATEAU_TOL of the plateau:
    # target_err = c + tol on the (1 - AUC) scale.
    n_extrap_auc = (n_for_target(auc_fit, auc_fit['c'] + AUC_PLATEAU_TOL)
                    if auc_fit['ok'] else None)
    n_extrap_flip = n_for_target(flip_fit, FLIP_RATE_TOL)

    candidates = [v for v in
                  (floor_empirical,
                   None if n_extrap_auc is None else math.ceil(n_extrap_auc),
                   None if n_extrap_flip is None else math.ceil(n_extrap_flip))
                  if v is not None and math.isfinite(v)]
    honest_floor = int(max(candidates)) if candidates else None

    report = {
        'schema': 'meta_curve_report/1',
        'meta': meta or {},
        'grid': grid,
        'records': list(records),
        'powerlaw_auc': auc_fit,
        'powerlaw_flip': flip_fit,
        'plateau_auc': plateau_auc,
        'floor': {
            'empirical_n': floor_empirical,
            'extrapolated_n_auc': n_extrap_auc,
            'extrapolated_n_flip': n_extrap_flip,
            'honest_floor': honest_floor,
            'criteria': {'auc_tol': AUC_PLATEAU_TOL,
                         'flip_tol': FLIP_RATE_TOL,
                         'veto_prob': VETO_PROB},
        },
    }
    report = _py(report)
    json.dumps(report)   # cheap self-check: the report must be serializable
    return report
