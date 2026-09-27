"""FR-03 funding-feature regime-shift audit (R2C-06c, measurement-only).

The Funding_* family (Funding_Rate_Ann / Funding_Z / Funding_Chg_24h — the audit
selects every column whose name contains 'Funding'; the researched
CS_Rank_Funding_Z was never built)
SURVIVES — kill-list survivor #1. It was fit mostly on a positive-funding
world and 2026 funding is persistently negative, so before the next
retrain this script audits, per funding column of the CRYPTO training
store:

  * PSI of the trailing-`--trailing-days` live window against deciles of
    the training window before it (monitor_drift's 10-decile convention,
    outer edges widened to +-inf)
  * two-sample KS (stat + p) on the same two windows
  * split purged IC: pooled per-name Spearman(feature, Target_Return_fb)
    on anchors strided by fb (non-overlapping by construction — that IS
    the overlap adjustment), full sample vs the --split (2026-01-01)
    onward subsample, Fisher CIs.

FLAG = PSI > 0.25 OR an IC sign flip with non-overlapping CIs. A flag
means the scheduled retrain re-fits on the shifted distribution — the
features are NEVER removed (survivor boundary). Output: printed table +
research/funding_drift_2026-08.json attached to the retrain notes.

    python scripts/funding_drift_audit.py            # crypto store
    python scripts/funding_drift_audit.py --split 2026-01-01 \
        --trailing-days 90 --out research/funding_drift_2026-08.json

Small-data honesty (FR-03): ~7 months x 6 names of negative-funding hourly
bars detects a sign flip, not a magnitude — the deliverable is a flag
table, not a refit.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from stage0_preds import index_ns    # noqa: E402

PSI_FLAG = 0.25          # monitor_drift.PSI_ACTION convention
_EPS = 1e-4
MIN_SIDE = 50            # min rows per window for PSI/KS
MIN_IC_N = 20            # min anchors per side for a split IC


def _projected_columns(book, keep):
    """Store columns to request from data_utils.load_training_data: 'Ticker'
    plus every column whose name satisfies keep(name), read from the parquet
    SCHEMA (no data). None (= the full load, today's behaviour) whenever the
    loader would not serve that parquet — absent, CSV fresher
    (data_utils._csv_is_fresher), or no pyarrow — so the CSV path is never
    narrowed by a stale parquet's schema. The DatetimeIndex is restored from
    the parquet's pandas metadata either way, so the loaded frame differs
    only in the columns this script never reads (output sha-identical)."""
    try:
        import contextlib
        import io
        import data_utils
        import pyarrow.parquet as pq
        stem = data_utils._stem(book)
        pqp = data_utils._BASE_DIR / f'{stem}.parquet'
        csvp = data_utils._BASE_DIR / f'{stem}.csv'
        if not pqp.exists():
            return None
        with contextlib.redirect_stdout(io.StringIO()):   # loader re-prints
            if data_utils._csv_is_fresher(pqp, csvp):
                return None
        cols = [c for c in pq.read_schema(pqp).names
                if c == 'Ticker' or keep(c)]
        return cols if 'Ticker' in cols and len(cols) > 1 else None
    except Exception:
        return None


def psi_from_train_deciles(ref, live, eps=_EPS):
    """PSI of `live` against deciles of `ref` — the monitor_drift
    convention (10 equal-mass ref bins, outer edges widened to +-inf so
    live outliers land in the end bins). None when either side is too
    small or ref is degenerate."""
    ref = np.asarray(ref, dtype=np.float64)
    live = np.asarray(live, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    live = live[np.isfinite(live)]
    if len(ref) < MIN_SIDE or len(live) < MIN_SIDE:
        return None
    edges = np.percentile(ref, np.linspace(0, 100, 11))
    if not (np.diff(edges) > 0).any():
        return None  # constant feature — PSI undefined
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(live, bins=edges)
    live_frac = counts / live.size
    # ref bins are equal-mass by construction except at tied edges — use
    # the ACTUAL ref histogram so ties don't fake drift
    ref_counts, _ = np.histogram(ref, bins=edges)
    ref_frac = ref_counts / ref.size
    lf = np.clip(live_frac, eps, None)
    rf = np.clip(ref_frac, eps, None)
    return float(np.sum((lf - rf) * np.log(lf / rf)))


def spearman_with_ci(x, y):
    """(rho, lo, hi, n) — Spearman with a Fisher-z 95% CI, or None when
    degenerate/too small. Callers guarantee non-overlapping anchors, so
    the IID CI is honest."""
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < MIN_IC_N or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return None
    rho = spearmanr(x[m], y[m]).correlation
    if not np.isfinite(rho):
        return None
    rho = float(np.clip(rho, -0.999999, 0.999999))
    z = np.arctanh(rho)
    half = 1.959964 / np.sqrt(n - 3)
    return rho, float(np.tanh(z - half)), float(np.tanh(z + half)), n


def strided_anchor_mask(n, stride):
    """Boolean mask keeping every stride-th row — the non-overlap purge
    for a stride = the label horizon."""
    m = np.zeros(int(n), dtype=bool)
    m[::max(1, int(stride))] = True
    return m


def sign_flip_disjoint(full, sub):
    """True iff the two (rho, lo, hi, n) tuples have opposite IC signs
    AND non-overlapping CIs — the FR-03 sign-flip flag."""
    if full is None or sub is None:
        return False
    r1, lo1, hi1, _ = full
    r2, lo2, hi2, _ = sub
    if r1 == 0.0 or r2 == 0.0 or (r1 > 0) == (r2 > 0):
        return False
    return hi1 < lo2 or hi2 < lo1


def audit_frame(df, split_ts, trailing_days, fwd_bars=None):
    """The audit kernel on a loaded multi-ticker training frame.

    df: DatetimeIndex frame with Ticker, Target_Return_* and Funding
    columns. Returns (rows, meta): one dict per funding column with psi /
    ks / split-IC fields + the flag."""
    from scipy.stats import ks_2samp
    fund_cols = [c for c in df.columns if 'Funding' in c]
    if not fund_cols:
        return [], {'error': 'no Funding columns in the frame'}
    horizons = sorted(int(c.rsplit('_', 1)[1]) for c in df.columns
                      if c.startswith('Target_Return_')
                      and c.rsplit('_', 1)[1].isdigit())
    fb = int(fwd_bars) if fwd_bars else (horizons[0] if horizons else 12)
    fwd_col = f'Target_Return_{fb}'
    if fwd_col not in df.columns:
        return [], {'error': f'{fwd_col} not in the frame'}
    times = index_ns(df.index)
    t_max = int(times.max())
    live_cut = t_max - int(trailing_days) * 86400 * 10**9
    split_ns = int(np.int64(split_ts.value))
    # per-name strided anchors, pooled
    anchor = np.zeros(len(df), dtype=bool)
    for _, sub_idx in df.groupby('Ticker').indices.items():
        sub_idx = np.sort(np.asarray(sub_idx))
        anchor[sub_idx[strided_anchor_mask(len(sub_idx), fb)]] = True
    rows = []
    for col in sorted(fund_cols):
        v = df[col].to_numpy(dtype=np.float64)
        ref = v[times < live_cut]
        live = v[times >= live_cut]
        psi = psi_from_train_deciles(ref, live)
        ks_stat = ks_p = None
        rf, lv = ref[np.isfinite(ref)], live[np.isfinite(live)]
        if len(rf) >= MIN_SIDE and len(lv) >= MIN_SIDE:
            ks = ks_2samp(rf, lv)
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        fwd = df[fwd_col].to_numpy(dtype=np.float64)
        a = anchor
        ic_full = spearman_with_ci(v[a], fwd[a])
        post = a & (times >= split_ns)
        ic_post = spearman_with_ci(v[post], fwd[post])
        flip = sign_flip_disjoint(ic_full, ic_post)
        flag = (psi is not None and psi > PSI_FLAG) or flip
        reasons = []
        if psi is not None and psi > PSI_FLAG:
            reasons.append(f'PSI {psi:.3f} > {PSI_FLAG}')
        if flip:
            reasons.append('IC sign flip w/ disjoint CIs')
        rows.append({
            'column': col,
            'psi': None if psi is None else round(psi, 4),
            'ks_stat': None if ks_stat is None else round(ks_stat, 4),
            'ks_p': None if ks_p is None else round(ks_p, 6),
            'ic_full': _ic_dict(ic_full),
            'ic_post_split': _ic_dict(ic_post),
            'sign_flip_disjoint_ci': bool(flip),
            'flag': bool(flag),
            'reasons': reasons,
        })
    meta = {'fwd_col': fwd_col, 'fwd_bars': fb,
            'n_rows': int(len(df)),
            'n_ref': int((times < live_cut).sum()),
            'n_live': int((times >= live_cut).sum()),
            'split': str(split_ts.date()),
            'trailing_days': int(trailing_days)}
    return rows, meta


def _ic_dict(t):
    if t is None:
        return None
    rho, lo, hi, n = t
    return {'ic': round(rho, 4), 'ci95': [round(lo, 4), round(hi, 4)],
            'n': n}


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='2026-01-01',
                    help='IC subsample split date (default 2026-01-01)')
    ap.add_argument('--trailing-days', type=int, default=90,
                    help='live window for PSI/KS (default 90)')
    ap.add_argument('--fwd-bars', type=int, default=None,
                    help='IC label horizon (default: shortest '
                         'Target_Return_<h> present)')
    ap.add_argument('--out',
                    default=str(BASE_DIR /
                                'research/funding_drift_2026-08.json'))
    args = ap.parse_args()

    from data_utils import load_training_data
    # Column projection (G6 C-1): audit_frame reads only Ticker, *Funding*
    # and Target_Return_* columns.
    df = load_training_data('crypto', columns=_projected_columns(
        'crypto',
        lambda c: 'Funding' in c or c.startswith('Target_Return_')))
    if df.empty:
        sys.exit('no crypto training data — run the harvest first')
    split_ts = pd.Timestamp(args.split)
    if split_ts.tz is None and getattr(df.index, 'tz', None) is not None:
        split_ts = split_ts.tz_localize(df.index.tz)
    rows, meta = audit_frame(df, split_ts, args.trailing_days,
                             args.fwd_bars)
    if not rows:
        sys.exit(f"audit not runnable: {meta.get('error')}")

    print(f"[FR-03] funding drift audit — ref {meta['n_ref']} rows / "
          f"live {meta['n_live']} rows, IC on {meta['fwd_col']}, split "
          f"{meta['split']}")
    print(f"  {'column':<22} {'PSI':>7} {'KS':>7} {'KS p':>9} "
          f"{'IC full':>18} {'IC post-split':>18}  flag")
    for r in rows:
        icf = r['ic_full']
        icp = r['ic_post_split']
        f1 = ('--' if icf is None
              else f"{icf['ic']:+.3f}[{icf['ci95'][0]:+.3f},"
                   f"{icf['ci95'][1]:+.3f}]")
        f2 = ('--' if icp is None
              else f"{icp['ic']:+.3f}[{icp['ci95'][0]:+.3f},"
                   f"{icp['ci95'][1]:+.3f}]")
        psi_s = '     --' if r['psi'] is None else f"{r['psi']:7.3f}"
        ks_s = '     --' if r['ks_stat'] is None else f"{r['ks_stat']:7.3f}"
        ksp_s = '       --' if r['ks_p'] is None else f"{r['ks_p']:9.2e}"
        verdict = ('FLAG: ' + '; '.join(r['reasons']) if r['flag']
                   else 'ok')
        print(f"  {r['column']:<22} {psi_s} {ks_s} {ksp_s} "
              f"{f1:>18} {f2:>18}  {verdict}")
    payload = {'generated': datetime.datetime.now(
                   datetime.timezone.utc).isoformat(),
               'params': meta, 'psi_flag_threshold': PSI_FLAG,
               'rows': rows,
               'note': 'Funding features SURVIVE regardless (kill-list '
                       'survivor #1): a flag means the next retrain '
                       're-fits on the shifted distribution — no feature '
                       'removal.'}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out)
    n_flag = sum(r['flag'] for r in rows)
    print(f"\n[FR-03] {n_flag}/{len(rows)} columns flagged -> {out}")
    print('[FR-03] features SURVIVE regardless — attach this table to '
          'the next retrain notes.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
