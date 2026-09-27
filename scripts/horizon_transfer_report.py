"""FR-07-A driver — horizon-transfer curves from harvest labels (R2C-06b).

Measurement-only. Loads the training store's existing multi-horizon
Target_Return_* columns (label parity: the exact arrays training selects
on, not re-derived returns), builds per-name + pooled rho(r^delta,
r^Delta) via horizon_transfer.transfer_matrix (anchors strided by the
longer horizon = non-overlapping; weekly-block-bootstrap SEs), and prints
each pair against the IID null sqrt(delta/Delta).

Reading the table: rho - null > 2*SE (persistence) or < -2*SE (reversal
inside the longer window) = off-diagonal structure -> sequence FR-07-B's
~8 LGB probes (trials counted into cum_trials). A curve on the null kills
the horizon topic cheaply (deferred item 7 of the 06 plan).

    python scripts/horizon_transfer_report.py --prefix ''      # crypto
    python scripts/horizon_transfer_report.py --prefix stock

STOCK CAVEAT (printed): the tradable stock labels are the TB_* columns,
whose EOD barrier fires within one session — every fb >= bars-per-session
yields IDENTICAL stock labels (policy_exits.py horizon-degeneracy caveat).
These curves are computed on the RAW Target_Return_* columns, so for the
stock book they describe the underlying return process, NOT the deployable
TB label family.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import horizon_transfer as ht        # noqa: E402
from stage0_preds import index_ns    # noqa: E402


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


def load_per_name(prefix_key):
    """{name: (fwd_by_h, times_ns)} from the training store's
    Target_Return_{h} columns; also returns the sorted horizon list."""
    from data_utils import load_training_data
    # Column projection (G6 C-1): only Ticker + Target_Return_* are read —
    # the full stock store is ~110 columns / >3 GB in RAM on the 8 GB Jetson.
    df = load_training_data(prefix_key, columns=_projected_columns(
        prefix_key, lambda c: c.startswith('Target_Return_')))
    if df.empty or 'Ticker' not in df.columns:
        sys.exit(f"no training data with a Ticker column for "
                 f"'{prefix_key}' — run the harvest first")
    horizons = sorted(int(c.rsplit('_', 1)[1]) for c in df.columns
                      if c.startswith('Target_Return_')
                      and c.rsplit('_', 1)[1].isdigit())
    if len(horizons) < 2:
        sys.exit('need >= 2 Target_Return_<h> columns for a transfer pair')
    per = {}
    for tick, sub in df.groupby('Ticker'):
        sub = sub.sort_index()
        fwd_by_h = {h: sub[f'Target_Return_{h}'].to_numpy(np.float64)
                    for h in horizons}
        per[str(tick)] = (fwd_by_h, index_ns(sub.index))
    return per, horizons


def _fmt(stat, null):
    if stat['rho'] is None:
        return f"    --      (n={stat['n']})"
    se = stat['se']
    dev = stat['rho'] - null
    flag = ''
    if se is not None and se > 0:
        if dev > 2 * se:
            flag = '  PERSIST>2SE'
        elif dev < -2 * se:
            flag = '  REVERT<-2SE'
    se_s = '   n/a' if se is None else f'{se:.4f}'
    return (f"{stat['rho']:+.4f} ±{se_s} (dev {dev:+.4f}, n={stat['n']},"
            f" wks={stat['n_weeks']}){flag}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prefix', default='',
                    help="'' = crypto, 'stock' = stock")
    ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stride', type=int, default=None,
                    help='anchor stride in bars (default: the pair Delta '
                         '— the non-overlap minimum)')
    ap.add_argument('--per-name', action='store_true',
                    help='print every name (default: pooled only)')
    args = ap.parse_args()

    prefix_key = 'stock' if args.prefix == 'stock' else 'crypto'
    per, horizons = load_per_name(prefix_key)
    print(f"[FR-07-A] {prefix_key}: {len(per)} names, horizons "
          f"{horizons}, stride={'Delta' if args.stride is None else args.stride}")
    if prefix_key == 'stock':
        print('[CAVEAT] stock TB labels are horizon-degenerate above one '
              'session (EOD barrier, policy_exits.py) — these raw '
              'Target_Return curves describe the return process, NOT the '
              'deployable TB label family.')

    pairs = ht.transfer_matrix(per, horizons=horizons,
                               n_boot=args.n_boot, seed=args.seed,
                               stride=args.stride)
    print('\n  delta -> Delta   null      pooled rho')
    off_diag = False
    for p in pairs:
        null = p['null']
        print(f"  {p['delta']:>4} -> {p['Delta']:<5} {null:.4f}   "
              f"{_fmt(p['pooled'], null)}")
        st = p['pooled']
        if (st['rho'] is not None and st['se'] and
                abs(st['rho'] - null) > 2 * st['se']):
            off_diag = True
        if args.per_name:
            for name in sorted(p['per_name']):
                print(f"      {name:<14} "
                      f"{_fmt(p['per_name'][name], null)}")
    print(f"\n[FR-07-A] off-diagonal structure (any pooled |rho-null| > "
          f"2SE): {off_diag}")
    print('[FR-07-A] decision: off-diagonal -> sequence FR-07-B probes '
          '(trials counted); on-null -> record the durable negative.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
