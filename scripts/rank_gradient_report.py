"""Rank-gradient Stage-0 gate (wave-9 #4/#5 activation harness).

HOLDOUT side: dump a predictions frame (rows {ts, symbol, signal, fwd_return},
one per symbol-bar from backtest.py over the universe) to JSON/CSV, then:
    python scripts/rank_gradient_report.py --preds preds.json
LIVE side: feed decision_report's rank buckets directly:
    python scripts/rank_gradient_report.py --buckets decision_report.json

PASS on BOTH the holdout panel AND >=20-30d of live journals before shipping the
conviction flagship (CONCENTRATION_ENABLED) or edge-Kelly (EDGE_KELLY_ENABLED).
"""
import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rank_gradient import rank_gradient_from_panel, rank_gradient_verdict  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--preds', help='predictions frame JSON/CSV (ts, symbol, signal, fwd_return)')
    src.add_argument('--buckets', help='decision_report JSON with rank_1_3/rank_6_7')
    ap.add_argument('--signal-lag', type=int, default=0)
    ap.add_argument('--cost-pct', type=float, default=0.0)
    args = ap.parse_args()

    if args.preds:
        import pandas as pd
        from portfolio_backtest import panel_from_frame
        p = Path(args.preds)
        df = (pd.read_csv(p) if p.suffix.lower() == '.csv'
              else pd.DataFrame(json.loads(p.read_text())))
        df = df.set_index(pd.to_datetime(df['ts']))
        panel = panel_from_frame(df, 'signal', 'fwd_return', signal_lag=args.signal_lag)
        buckets = rank_gradient_from_panel(panel, cost_pct=args.cost_pct)
    else:
        buckets = json.loads(Path(args.buckets).read_text())

    v = rank_gradient_verdict(buckets)
    print("\n=== Rank-gradient Stage-0 ===")
    for b in ('rank_1_3', 'rank_4_5', 'rank_6_7'):
        if b in buckets:
            print(f"  {b}: mean_net {buckets[b]['mean_net_pct']}  (n={buckets[b].get('n')})")
    print(f"  ratio 6-7 / 1-3: {v.get('ratio_6_7_over_1_3')}")
    print(f"\n  VERDICT: {v['verdict']}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
