"""Per-name IC diagnostic — the universe-promotion gate (wave-9 #3 activation).

INPUT: per-(symbol, bar) rows {symbol, pred, fwd_return}[, ts] as JSON or CSV.
The dump is produced by backtest.py (B02, default ON): {prefix}stage0_preds.json,
PERCENT units, non-overlapping (pass --time-key ts).
Preconditions the math relies on:

  * rows must be bar-ordered per name — the sub-period ICs and
    positive_consistency are meaningless on unordered rows. If the dump is
    unordered, pass --time-key ts to stable-sort rows by (name, ts) here.
  * fwd_return must be non-overlapping (1-bar, or bars spaced >= the horizon)
    — overlapping multi-bar forward returns on hourly bars inflate the
    t = IC*sqrt(n_finite-1) significance hurdle by ~sqrt(horizon), the same
    overlap double-counting the repo corrects elsewhere (effective-n). If the
    dump overlaps, raise --min-t accordingly or fix the dump.
  * scripts/rank_gradient_report.py reads the same frame but names the
    prediction column 'signal' — pass --pred-key signal to share one dump.

    python scripts/ic_by_name.py --in preds.json --min-ic 0.0 --min-consistency 0.6

Promote into TRADABLE_POOL only the names printed as PROMOTE (positive,
consistent, statistically significant out-of-sample rank-IC); keep the rest
training-only. The breadth alpha is GATED on this — do not promote the whole
pool on a blanket √breadth argument.
"""
import argparse
import json
import math
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ic_diagnostic import ic_by_name, promote_set  # noqa: E402


def _load(path):
    p = Path(path)
    if p.suffix.lower() == '.csv':
        import pandas as pd
        return pd.read_csv(p).to_dict('records')
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', required=True, help='JSON/CSV of rows')
    ap.add_argument('--name-key', default='symbol')
    ap.add_argument('--pred-key', default='pred')
    ap.add_argument('--fwd-key', default='fwd_return')
    ap.add_argument('--time-key', default=None,
                    help='optional timestamp key: stable-sort rows by '
                         '(name, ts) before scoring (for unordered dumps)')
    ap.add_argument('--min-ic', type=float, default=0.0)
    ap.add_argument('--min-consistency', type=float, default=0.6)
    ap.add_argument('--min-t', type=float, default=2.0,
                    help='significance hurdle: IC*sqrt(n_finite-1) >= min-t')
    ap.add_argument('--subperiods', type=int, default=4)
    args = ap.parse_args()

    rows = _load(args.inp)
    if not isinstance(rows, list):
        # ic_by_name expects a list of per-(name,bar) row objects; a JSON object
        # or scalar would iterate its keys/chars and raise a cryptic TypeError
        # deep inside the library (r[name_key] on a str). Fail loud, up front.
        sys.exit(f"{args.inp}: expected a JSON array of row objects, got "
                 f"{type(rows).__name__}")
    keys = [args.name_key, args.pred_key, args.fwd_key]
    if args.time_key is not None:
        keys.append(args.time_key)
    for k in keys:
        # A misspelled pred/fwd key would otherwise yield a silent all-None
        # table (the library uses r.get()) — fail loud instead.
        if rows and not any(k in r for r in rows):
            sys.exit(f"key {k!r} not in rows; available: {sorted(rows[0])}")
    if args.time_key is not None:
        rows = sorted(rows, key=lambda r: (r[args.name_key], r[args.time_key]))

    table = ic_by_name(rows, args.name_key, args.pred_key, args.fwd_key, args.subperiods)
    promoted = set(promote_set(table, args.min_ic, args.min_consistency, args.min_t))

    print(f"\n=== Per-name rank-IC ({len(table)} names) ===")
    print(f"  hurdles: IC > {args.min_ic}, consistency >= {args.min_consistency}, "
          f"t = IC*sqrt(n_finite-1) >= {args.min_t} — ALL required to PROMOTE")
    for name in sorted(table, key=lambda n: (table[n]['ic'] is None,
                                             -(table[n]['ic'] if table[n]['ic'] is not None else 0))):
        m = table[name]
        tag = 'PROMOTE' if name in promoted else 'hold   '
        n_finite = m.get('n_finite', m['n'])
        t = (round(m['ic'] * math.sqrt(max(n_finite - 1, 1)), 2)
             if m['ic'] is not None else None)
        print(f"  {tag}  {name:<8} IC={m['ic']}  t={t}  "
              f"consistency={m['positive_consistency']}  n={m['n']}  "
              f"n_finite={n_finite}")
    print(f"\n  PROMOTE SET ({len(promoted)}): {sorted(promoted)}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
