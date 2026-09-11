"""FR-04 driver — deployed blend vs the Nagel naive one-liner (R2C-06a).

Measurement-only. Joins per-name closes from the training store to the
stage0 predictions dump (older dumps carry no 'close' field — and the
naive EWMA needs each name's FULL close history anyway, not just the
sparse anchor rows), computes the strictly-trailing naive signal
(naive_baseline.py) at each dump row, and prints side-by-side:

  * purged IC (per-name + pooled Spearman; dump rows are non-overlapping
    by stage0_preds construction, so no further overlap adjustment)
  * DSR on the admitted long-only returns — the blend deflated at the
    campaign's cumulative trial pool (adaptive_state cum_trials, same
    source as backtest --gate), the naive rule at n_trials=1 (it was
    never searched).

Decision rule (FR-04): the blend must beat the naive rule on BOTH IC and
DSR; a within-noise result is an owner report, never an auto-action.

    python scripts/naive_vs_blend.py --preds stage0_preds.json --prefix ''
    python scripts/naive_vs_blend.py --preds stock_stage0_preds.json \
        --prefix stock [--half-life 24 --vol-window 72 --cum-trials N]

Jetson-gated only by DATA (dump + training parquet); pure numpy/pandas/
scipy otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import naive_baseline as nb                      # noqa: E402
from ic_diagnostic import ic_by_name, rank_ic    # noqa: E402
from stage0_preds import index_ns                # noqa: E402
from validation import dsr_from_trade_returns    # noqa: E402


def load_rows(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        sys.exit(f"{path}: expected a BARE JSON list of row dicts")
    return rows


def close_series_by_name(prefix_key):
    """{ticker: (times_ns sorted, closes)} from the training store."""
    from data_utils import load_training_data
    df = load_training_data(prefix_key, columns=['Ticker', 'Close'])
    if df.empty or 'Ticker' not in df.columns:
        sys.exit(f"no training data with a Ticker column for "
                 f"'{prefix_key}' — run the harvest first")
    out = {}
    for tick, sub in df.groupby('Ticker'):
        sub = sub.sort_index()
        out[str(tick)] = (index_ns(sub.index),
                          sub['Close'].to_numpy(dtype=np.float64))
    return out


def attach_naive(rows, series, half_life, vol_window):
    """Stamp row['naive'] (the one-liner at the row's bar) via exact
    timestamp lookup; count unmatched rows honestly."""
    sig_cache = {}
    unmatched = 0
    for r in rows:
        sym = str(r.get('symbol'))
        key = sym if sym in series else sym.replace('/', '-')
        if key not in series:
            r['naive'] = None
            unmatched += 1
            continue
        if key not in sig_cache:
            t, c = series[key]
            sig_cache[key] = (t, nb.naive_signal(
                c, half_life=half_life, vol_window=vol_window))
        t, sig = sig_cache[key]
        q = np.int64(pd.Timestamp(r['ts']).value)
        v = nb.lookup_at_times(t, sig, np.array([q]))[0]
        r['naive'] = float(v) if np.isfinite(v) else None
        if r['naive'] is None:
            unmatched += 1
    return unmatched


def joined_rows(rows):
    """The rows the naive signal actually joined on — the FR-04 comparison
    set. Every side-by-side number (IC AND DSR, both arms) is computed on
    exactly these rows: scoring the blend on rows the naive rule never saw
    (warmup / unmatched names) would not be 'identical stage0 rows'."""
    return [r for r in rows if r.get('naive') is not None]


def _blend_threshold_mask(rows):
    """Blend admission per the hypersearch objective's convention: pred >
    threshold (STRICT; backtest.simulate_ticker and base_loop._execute_buys
    admit pred == threshold), read
    back from pred_thresh_ratio > 1 when the dump carries it, else
    pred > 0 (no threshold recorded)."""
    adm = []
    for r in rows:
        ratio = r.get('pred_thresh_ratio')
        if ratio is not None:
            adm.append(float(ratio) > 1.0)
        else:
            adm.append(float(r.get('pred') or 0.0) > 0.0)
    return np.asarray(adm, dtype=bool)


def _cum_trials(prefix_key, override):
    if override is not None:
        return max(2, int(override))
    try:
        import adaptive_config
        cum = int(adaptive_config.load_adaptive_state(prefix_key)
                  .get('cum_trials', 0) or 0)
    except Exception:
        cum = 0
    if cum < 2:
        print('[WARN] adaptive_state cum_trials unavailable — falling '
              'back to the legacy 100-trial pool (pass --cum-trials)')
        return 100
    return cum


def _fmt_dsr(d):
    return (f"sr/trade={d['sr']:+.4f}  DSR={d['dsr']:.4f}  "
            f"n={d['n']}  n_trials={d['n_trials']}  status={d['status']}"
            if d.get('sr') is not None else f"status={d['status']}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preds', required=True,
                    help='stage0 predictions dump (BARE JSON list)')
    ap.add_argument('--prefix', default='',
                    help="'' = crypto, 'stock' = stock (training store + "
                         'adaptive-state key)')
    ap.add_argument('--half-life', type=float,
                    default=nb.DEFAULT_HALF_LIFE)
    ap.add_argument('--vol-window', type=int,
                    default=nb.DEFAULT_VOL_WINDOW)
    ap.add_argument('--cum-trials', type=int, default=None,
                    help='deflation pool for the BLEND (default: '
                         'adaptive_state cum_trials, legacy 100 fallback)')
    args = ap.parse_args()

    prefix_key = 'stock' if args.prefix == 'stock' else 'crypto'
    rows = load_rows(args.preds)
    if not rows:
        sys.exit('empty dump')
    series = close_series_by_name(prefix_key)
    unmatched = attach_naive(rows, series, args.half_life,
                             args.vol_window)
    n_scored = len(rows) - unmatched
    print(f"[FR-04] {len(rows)} dump rows, naive signal joined on "
          f"{n_scored} ({unmatched} unmatched/warmup)")
    if n_scored < 30:
        sys.exit('too few joined rows to measure anything')
    # FR-04 contract: 'identical stage0 rows' — BOTH arms are scored on
    # the joined subset only (blend numbers on all rows would mix in
    # warmup/unmatched rows the naive rule never saw).
    rows = joined_rows(rows)

    # --- purged IC, per-name + pooled ---------------------------------
    blend_ic = ic_by_name(rows)
    naive_ic = ic_by_name(rows, pred_key='naive')
    pooled_b = rank_ic([r['pred'] for r in rows],
                       [r['fwd_return'] for r in rows])
    pooled_n = rank_ic([r['naive'] for r in rows],
                       [r['fwd_return'] for r in rows])
    print('\n  name             blend IC        naive IC   (n_finite)')
    for name in sorted(blend_ic):
        b, nv = blend_ic[name], naive_ic.get(name, {})
        fb = ('None' if b.get('ic') is None else f"{b['ic']:+.4f}")
        fn = ('None' if nv.get('ic') is None else f"{nv['ic']:+.4f}")
        print(f"  {name:<14} {fb:>10} {fn:>15}   "
              f"({b.get('n_finite', 0)}/{nv.get('n_finite', 0)})")
    fb = 'None' if pooled_b is None else f'{pooled_b:+.4f}'
    fn = 'None' if pooled_n is None else f'{pooled_n:+.4f}'
    print(f"  {'POOLED':<14} {fb:>10} {fn:>15}")

    # --- DSR on admitted long-only returns ----------------------------
    preds = np.array([float(r['pred']) for r in rows])
    naive = np.array([np.nan if r['naive'] is None else r['naive']
                      for r in rows])
    fwd = np.array([float(r['fwd_return']) for r in rows])
    adm = _blend_threshold_mask(rows)
    blend_trades = fwd[adm & np.isfinite(fwd)]
    naive_trades = nb.long_only_returns(naive, fwd, threshold=0.0)
    cum = _cum_trials(prefix_key, args.cum_trials)
    d_b = dsr_from_trade_returns(blend_trades, n_trials=cum,
                                 n_eff_source='iid_nonoverlapping')
    d_n = dsr_from_trade_returns(naive_trades, n_trials=1,
                                 n_eff_source='iid_nonoverlapping')
    print(f"\n  blend  ({int(adm.sum())} admitted): {_fmt_dsr(d_b)}")
    print(f"  naive  ({len(naive_trades)} admitted): {_fmt_dsr(d_n)}")

    # --- verdict (report, never an action) ----------------------------
    ic_win = (pooled_b is not None and pooled_n is not None
              and pooled_b > pooled_n)
    dsr_win = (d_b.get('dsr') is not None and d_n.get('dsr') is not None
               and d_b['dsr'] > d_n['dsr'])
    print(f"\n[FR-04] blend beats naive on IC: {ic_win} | on DSR: "
          f"{dsr_win}")
    if not (ic_win and dsr_win):
        print('[FR-04] blend does NOT dominate the zero-fit one-liner — '
              'owner report (promotion-culture finding); no auto-action.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
