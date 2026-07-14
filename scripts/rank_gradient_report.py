"""Rank-gradient Stage-0 gate (wave-9 #4/#5 activation harness).

HOLDOUT side: dump a predictions frame (rows {ts, symbol, signal, fwd_return},
one per symbol-bar over the universe) to JSON/CSV, then:
    python scripts/rank_gradient_report.py --preds preds.json
The dump step is NOT yet authored — backtest.py emits only its metrics+trades
report JSON, no per-bar prediction rows — so the producer must be written on
the Jetson before the holdout side can run (scripts/ic_by_name.py reads the
same frame; author ONE dump for both). UNITS: signal and fwd_return are
PERCENT, matching harvest Target_Return_* and decision_report's mean_net_pct;
--cost-pct is subtracted raw from fwd_return, so a fractional-return dump with
a percent cost would silently corrupt the bucket means.

LIVE side: feed decision_report's rank buckets directly:
    python scripts/rank_gradient_report.py --buckets decision_report.json

PASS on BOTH the holdout panel AND >=20-30d of live journals before shipping the
conviction flagship (CONCENTRATION_ENABLED) or edge-Kelly (EDGE_KELLY_ENABLED).
Exit status: 0 only when the verdict CONFIRMS the gradient, 1 otherwise, so
scripted use cannot mistake a no-go verdict for success.
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
    ap.add_argument('--cost-pct', type=float, default=0.0,
                    help='flat per-trade cost in the SAME units as fwd_return '
                         '(repo convention: percent, cf. fees.round_trip_cost_pct)')
    args = ap.parse_args()

    if args.preds:
        import pandas as pd
        from portfolio_backtest import panel_from_frame
        p = Path(args.preds)
        df = (pd.read_csv(p) if p.suffix.lower() == '.csv'
              else pd.DataFrame(json.loads(p.read_text())))
        df = df.set_index(pd.to_datetime(df['ts']))
        # The documented frame names the ticker column 'symbol' (shared with
        # ic_by_name); tolerate a repo-internal 'Ticker'-style dump too.
        tcol = 'symbol' if 'symbol' in df.columns else 'Ticker'
        panel = panel_from_frame(df, 'signal', 'fwd_return', ticker_col=tcol,
                                 signal_lag=args.signal_lag)
        buckets = rank_gradient_from_panel(panel, cost_pct=args.cost_pct)
    else:
        buckets = json.loads(Path(args.buckets).read_text())
        # Same unwrap as rank_gradient_verdict: a full decision_report.json
        # nests the rank buckets under 'conviction' — without this, the
        # per-bucket evidence lines below silently print nothing.
        if 'rank_1_3' not in buckets and isinstance(buckets.get('conviction'), dict):
            buckets = buckets['conviction']

    v = rank_gradient_verdict(buckets)
    print("\n=== Rank-gradient Stage-0 ===")
    for b in ('rank_1_3', 'rank_4_5', 'rank_6_7'):
        if b in buckets:
            print(f"  {b}: mean_net {buckets[b]['mean_net_pct']}  (n={buckets[b].get('n')})")
    print(f"  ratio 6-7 / 1-3: {v.get('ratio_6_7_over_1_3')}")
    print(f"\n  VERDICT: {v['verdict']}\n")
    return 0 if v.get('gradient_exists') else 1


if __name__ == '__main__':
    raise SystemExit(main())
