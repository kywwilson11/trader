"""Implementation-shortfall report — what execution actually costs.

Slippage concentrates in exactly the trades that matter (stop-outs in fast
markets), so backtest assumptions must be checked against REALIZED fills.
The loops journal decision_price (quote midpoint when the decision fired)
and fill_price on every confirmed entry and exit; this tool aggregates
them.

Sign convention: positive slippage_bps always means "worse than the
decision price" (buys filled higher, sells filled lower).

Usage:
    python execution_report.py --days 14
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

JOURNAL_DIR = BASE_DIR / "journals"


def _load(days: int):
    rows = []
    today = datetime.now().date()
    for d in range(days + 1):
        path = JOURNAL_DIR / f"{(today - timedelta(days=d)).isoformat()}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get('slippage_bps') is not None:
                    rows.append(e)
    return rows


def run_report(days: int = 14) -> dict:
    rows = _load(days)
    if not rows:
        print("No fills with slippage data yet — the loops journal "
              "decision_price/fill_price on every confirmed fill.")
        return {}

    def crypto(sym):
        return '/' in sym or (sym.endswith('USD') and len(sym) > 5)

    groups = defaultdict(list)
    for e in rows:
        sym = e.get('symbol', '?')
        asset = 'crypto' if crypto(sym) else 'stock'
        action = e.get('action', '?')
        reason = e.get('exit_reason') or 'entry'
        groups[(asset, action, reason)].append(float(e['slippage_bps']))

    print(f"\n=== IMPLEMENTATION SHORTFALL (last {days}d, {len(rows)} fills) ===")
    print(f"{'asset':<8}{'action':<7}{'reason':<14}{'n':>5}"
          f"{'mean bps':>10}{'median':>9}{'p90':>8}{'worst':>9}")
    report = {}
    for (asset, action, reason), vals in sorted(groups.items()):
        a = np.array(vals)
        entry = {
            'n': len(a),
            'mean_bps': round(float(a.mean()), 2),
            'median_bps': round(float(np.median(a)), 2),
            'p90_bps': round(float(np.percentile(a, 90)), 2),
            'worst_bps': round(float(a.max()), 2),
        }
        report[f"{asset}/{action}/{reason}"] = entry
        print(f"{asset:<8}{action:<7}{reason:<14}{entry['n']:>5}"
              f"{entry['mean_bps']:>10.1f}{entry['median_bps']:>9.1f}"
              f"{entry['p90_bps']:>8.1f}{entry['worst_bps']:>9.1f}")

    all_bps = np.array([float(e['slippage_bps']) for e in rows])
    overall = round(float(all_bps.mean()), 2)
    report['overall_mean_bps'] = overall
    print(f"\nOverall mean shortfall: {overall:+.1f} bps per fill")
    print("Compare against the backtest's assumptions (fees.py spread "
          "haircuts: crypto 10 bps, stock 5 bps round trip). If realized "
          "shortfall is persistently higher, the backtest is optimistic — "
          "raise SPREAD_PCT in backtest.py and the entry edge floor.")

    out = BASE_DIR / 'execution_report.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report: {out}")
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Realized execution-cost report')
    ap.add_argument('--days', type=int, default=14)
    args = ap.parse_args()
    run_report(args.days)
