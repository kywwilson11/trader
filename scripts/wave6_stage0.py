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

It loads the offline training panels via data_utils.load_training_data
(training_data.* for crypto, stock_training_data.* for stock — parquet-first
with CSV fallback) and reuses the same average-uniqueness kernel the DSR gate
uses (sample_weights.py). The gate itself (scripts/hypersearch_v2.py) applies
a trade-mask over realized holdout entries plus cross-sectional clustering ON
TOP of that kernel, so this script shares the kernel, not the final n_eff
number.

Usage:
    python scripts/wave6_stage0.py                 # both books, all horizons
    python scripts/wave6_stage0.py --book crypto   # one book
    python scripts/wave6_stage0.py --fb 24         # one horizon
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sample_weights import average_uniqueness, effective_n

# adaptive_config.DEFAULT_SEARCH_SPACE['forward_bars'] — reference/display only.
# The horizons actually measured are discovered from the data's TB_Bars_*
# columns, because the harvest writes get_forward_bars_list(book) and the
# adaptive search state can mutate the set (8/64/96 are mutation candidates).
FORWARD_BARS = [12, 18, 24, 32, 48]


def _harvested_horizons(columns):
    """Sorted forward-bars horizons present as TB_Bars_* columns."""
    out = []
    for c in columns:
        m = re.fullmatch(r'TB_Bars_(\d+)', str(c))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _ticker_boundaries(tickers):
    """Half-open (start, end) row ranges per ticker on a Ticker-sorted frame."""
    bounds = {}
    offset = 0
    for tkr, n in tickers.value_counts(sort=False).items():
        bounds[tkr] = (offset, offset + int(n))
        offset += int(n)
    return bounds


def measure_book(prefix, horizons=None):
    """Measure one book. horizons=None discovers them from the data itself."""
    from data_utils import load_training_data
    # Full-frame load on purpose: a pruned parquet read (columns=...) raises
    # on missing columns (breaking the graceful diagnostics below), and the
    # horizons are discovered from the columns anyway. The feature columns
    # are dropped right after discovery so the sort copies ~7 columns, not
    # the whole panel.
    try:
        df = load_training_data(prefix, columns=None)
    except Exception as e:
        print(f"[{prefix}] could not load training data: {e}")
        return None
    if df.empty:
        print(f"[{prefix}] no training data — nothing to measure")
        return None
    if 'Ticker' not in df.columns:
        print(f"[{prefix}] no Ticker column — cannot build per-series concurrency")
        return None
    harvested = _harvested_horizons(df.columns)
    if not harvested:
        print(f"[{prefix}] no TB_Bars_* columns harvested — re-run the harvest "
              f"with triple-barrier labels first")
        return None
    if horizons is None:
        use = harvested
    else:
        for fb in horizons:
            if fb not in harvested:
                print(f"[{prefix}] horizon {fb} not in this dataset "
                      f"(available: {harvested})")
        use = [fb for fb in horizons if fb in harvested]
        if not use:
            return None

    df = df[['Ticker'] + [f'TB_Bars_{fb}' for fb in use]]
    df = df.sort_values('Ticker', kind='stable')
    bounds = _ticker_boundaries(df['Ticker'])
    out = {'book': prefix, 'rows': int(len(df)), 'tickers': len(bounds),
           'horizons': {}}
    print(f"\n=== {prefix.upper()} book: {len(df)} rows, {len(bounds)} tickers ===")
    print(f"{'fb':>4} {'labels':>8} {'u_bar_mean':>11} {'u_bar_med':>10} "
          f"{'u_p10':>7} {'u_p90':>7} {'N_eff/N':>8} {'hold_med':>9}")
    for fb in use:
        col = f'TB_Bars_{fb}'
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
    """Print the decision block. Returns False when nothing was measured."""
    print("\n=== STAGE-0 EXP-1 DECISION ===")
    if not any(r and r['horizons'] for r in results):
        print("NOTHING MEASURED — no TB_Bars data found; do not act on this run")
        return False
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
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', choices=['stock', 'crypto'], default=None,
                    help='measure one book (default: both)')
    ap.add_argument('--fb', type=int, default=None,
                    help='measure one forward-bars horizon (default: all '
                         f'harvested; canonical search space {FORWARD_BARS})')
    ap.add_argument('--json', default=None, help='write the summary to this path')
    args = ap.parse_args()

    books = [args.book] if args.book else ['crypto', 'stock']
    horizons = [args.fb] if args.fb else None  # None -> discover from the data
    results = [measure_book(b, horizons) for b in books]
    measured = verdict(results)

    if args.json:
        import json
        with open(args.json, 'w') as f:
            json.dump([r for r in results if r], f, indent=2)
        print(f"\nWrote {args.json}")

    if not measured:
        sys.exit(1)  # zero-measurement runs must be unmistakable to wrappers


if __name__ == '__main__':
    main()
