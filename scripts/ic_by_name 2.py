"""Per-name IC diagnostic — the universe-promotion gate (wave-9 #3 activation).

On the Jetson: run backtest.py over the FULL ticker set (universe + candidate
pool), dump per-(symbol, bar) {symbol, pred, fwd_return} rows to JSON or CSV, then:

    python scripts/ic_by_name.py --in preds.json --min-ic 0.0 --min-consistency 0.6

Promote into TRADABLE_POOL only the names printed as PROMOTE (positive, consistent
out-of-sample rank-IC); keep the rest training-only. The breadth alpha is GATED on
this — do not promote the whole pool on a blanket √breadth argument.
"""
import argparse
import json
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
    ap.add_argument('--min-ic', type=float, default=0.0)
    ap.add_argument('--min-consistency', type=float, default=0.6)
    ap.add_argument('--subperiods', type=int, default=4)
    args = ap.parse_args()

    rows = _load(args.inp)
    table = ic_by_name(rows, args.name_key, args.pred_key, args.fwd_key, args.subperiods)
    promoted = set(promote_set(table, args.min_ic, args.min_consistency))

    print(f"\n=== Per-name rank-IC ({len(table)} names) ===")
    for name in sorted(table, key=lambda n: (table[n]['ic'] is None, -(table[n]['ic'] or -9))):
        m = table[name]
        tag = 'PROMOTE' if name in promoted else 'hold   '
        print(f"  {tag}  {name:<8} IC={m['ic']}  "
              f"consistency={m['positive_consistency']}  n={m['n']}")
    print(f"\n  PROMOTE SET ({len(promoted)}): {sorted(promoted)}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
