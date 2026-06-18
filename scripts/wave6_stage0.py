#!/usr/bin/env python3
"""Wave-6 Stage-0 measurement (offline, no live data).

The wave-6 plan is integrity-first AND measurement-first: the magnitude of
every de-biasing finding is unknown until measured on the existing offline
artifacts, and the red team explicitly forbids the hand-waved n_eff = n/10.
This script answers Experiment 1 — the realized label-concurrency / effective
sample size, PER BOOK — which gates everything else:

  - If crypto N_eff/N < ~0.30 (average uniqueness u-bar < 0.30), the overlapping
    triple-barrier labels are heavily non-IID: proceed with uniqueness training
    weights AND the effective-n DSR null as Tier-1 work.
  - If stock u-bar > ~0.60 (near-IID because stock TB windows are EOD-capped to
    ~1-7 bars), scope the weighting/DSR effect to crypto and treat the stock
    book as a near-no-op.

It reads ONLY {stock,crypto}.parquet TB_Bars_{fb} columns and reuses the same
average-uniqueness kernel the live gate uses (sample_weights.py), so the number
reported here is exactly the n_eff the Deflated-Sharpe gate will consume.

Usage:
    python scripts/wave6_stage0.py                 # both books, all horizons
    python scripts/wave6_stage0.py --book crypto   # one book
    python scripts/wave6_stage0.py --fb 24         # one horizon
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sample_weights import average_uniqueness, effective_n

FORWARD_BARS = [12, 18, 24, 32, 48]


def _ticker_boundaries(tickers):
    """Half-open (start, end) row ranges per ticker on a Ticker-sorted frame."""
    bounds = {}
    offset = 0
    for tkr, n in tickers.value_counts(sort=False).items():
        bounds[tkr] = (offset, offset + int(n))
        offset += int(n)
    return bounds


def measure_book(prefix, horizons):
    from data_utils import load_training_data
    # Pull only what we need: Ticker + the TB_Bars span columns.
    bar_cols = [f'TB_Bars_{fb}' for fb in horizons]
    try:
        df = load_training_data(prefix, columns=None)
    except Exception as e:
        print(f"[{prefix}] could not load parquet: {e}")
        return None
    if df.empty:
        print(f"[{prefix}] empty parquet — nothing to measure")
        return None
    if 'Ticker' not in df.columns:
        print(f"[{prefix}] no Ticker column — cannot build per-series concurrency")
        return None
    present = [c for c in bar_cols if c in df.columns]
    if not present:
        print(f"[{prefix}] no TB_Bars_* columns harvested — re-run the harvest "
              f"with triple-barrier labels first")
        return None

    df = df.sort_values('Ticker', kind='stable')
    bounds = _ticker_boundaries(df['Ticker'])
    out = {'book': prefix, 'rows': int(len(df)), 'tickers': len(bounds),
           'horizons': {}}
    print(f"\n=== {prefix.upper()} book: {len(df)} rows, {len(bounds)} tickers ===")
    print(f"{'fb':>4} {'labels':>8} {'u_bar_mean':>11} {'u_bar_med':>10} "
          f"{'u_p10':>7} {'u_p90':>7} {'N_eff/N':>8} {'hold_med':>9}")
    for fb in horizons:
        col = f'TB_Bars_{fb}'
        if col not in df.columns:
            continue
        hold = df[col].to_numpy(dtype=np.float64)
        u = average_uniqueness(hold, bounds)
        finite = np.isfinite(u)
        n_labels = int(finite.sum())
        if n_labels == 0:
            continue
        uf = u[finite]
        n_eff = effective_n(u)
        ratio = n_eff / n_labels
        hold_med = float(np.nanmedian(hold[finite]))
        rec = {'n_labels': n_labels, 'u_bar_mean': float(uf.mean()),
               'u_bar_median': float(np.median(uf)),
               'u_p10': float(np.percentile(uf, 10)),
               'u_p90': float(np.percentile(uf, 90)),
               'n_eff': round(n_eff, 1), 'n_eff_over_n': round(ratio, 4),
               'hold_bars_median': hold_med}
        out['horizons'][fb] = rec
        print(f"{fb:>4} {n_labels:>8} {uf.mean():>11.4f} "
              f"{np.median(uf):>10.4f} {np.percentile(uf,10):>7.3f} "
              f"{np.percentile(uf,90):>7.3f} {ratio:>8.3f} {hold_med:>9.1f}")
    return out


def verdict(results):
    print("\n=== STAGE-0 EXP-1 DECISION ===")
    crypto = next((r for r in results if r and r['book'] == 'crypto'), None)
    stock = next((r for r in results if r and r['book'] == 'stock'), None)

    def book_ubar(r):
        if not r or not r['horizons']:
            return None
        return float(np.mean([h['u_bar_mean'] for h in r['horizons'].values()]))

    cu, su = book_ubar(crypto), book_ubar(stock)
    if cu is not None:
        proceed = cu < 0.30
        print(f"crypto mean u-bar = {cu:.3f} -> "
              f"{'NON-IID: ship uniqueness weights + effective-n DSR (Tier-1)' if proceed else 'mild overlap: weighting optional, DSR fix still safe'}")
    if su is not None:
        near_iid = su > 0.60
        print(f"stock  mean u-bar = {su:.3f} -> "
              f"{'near-IID (EOD-capped): scope effect to crypto, stock ~no-op' if near_iid else 'overlap present: include stock in weighting'}")
    print("\nUse the realized per-book u-bar above as the ONLY justified n_eff "
          "input. Do NOT substitute a hand-waved n/10.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', choices=['stock', 'crypto'], default=None,
                    help='measure one book (default: both)')
    ap.add_argument('--fb', type=int, default=None,
                    help='measure one forward-bars horizon (default: all)')
    ap.add_argument('--json', default=None, help='write the summary to this path')
    args = ap.parse_args()

    books = [args.book] if args.book else ['crypto', 'stock']
    horizons = [args.fb] if args.fb else FORWARD_BARS
    results = [measure_book(b, horizons) for b in books]
    verdict(results)

    if args.json:
        import json
        with open(args.json, 'w') as f:
            json.dump([r for r in results if r], f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == '__main__':
    main()
