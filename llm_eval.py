"""Measure whether the LLM gate actually predicts returns.

The system spends API budget and applies 0.65-1.5x sizing (plus forced
veto-sells) based on LLM conviction scores — with, until now, zero
evidence the scores correlate with outcomes. This evaluator closes the
loop:

  1. base_loop journals every scored candidate per analysis cycle
     (action='llm_analysis': {symbol: {s, pred}}).
  2. This script replays those entries, fetches the bars that followed,
     and computes each symbol's realized forward return over the model
     horizon.
  3. Report: realized return by s-bucket, rank correlation (Spearman)
     between s and realized return, and the counterfactual P&L of vetoed
     (s < 0.15) candidates — i.e. what the veto saved or cost.

Usage:
    python llm_eval.py --days 14
    python llm_eval.py --days 30 --asset crypto

Interpretation: if high-s buckets don't out-return low-s buckets and the
veto counterfactual is ~0, the LLM gate is decoration — turn it off and
save the spend. Re-run monthly.
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
VETO_THRESHOLD = 0.15


def _load_entries(days: int) -> list[dict]:
    entries = []
    today = datetime.now().date()
    for d in range(days + 1):
        day = today - timedelta(days=d)
        path = JOURNAL_DIR / f"{day.isoformat()}.jsonl"
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
                if e.get("action") == "llm_analysis":
                    entries.append(e)
    return entries


def _bars_lookup(api, symbol: str, asset_type: str, start, end):
    """Hourly closes for [start, end] as a sorted (timestamps, closes)."""
    try:
        if asset_type == 'crypto':
            sym = symbol.replace('-', '/') if '-' in symbol else symbol
            bars = api.get_crypto_bars(sym, '1Hour', start=start.isoformat(),
                                       end=end.isoformat())
        else:
            bars = api.get_bars(symbol, '1Hour', start=start.isoformat(),
                                end=end.isoformat(), adjustment='all')
        ts, closes = [], []
        for b in bars:
            t = b.t
            if hasattr(t, 'to_pydatetime'):
                t = t.to_pydatetime()
            ts.append(t.timestamp())
            closes.append(float(b.c))
        order = np.argsort(ts)
        return np.array(ts)[order], np.array(closes)[order]
    except Exception as e:
        print(f"  [BARS] {symbol}: {e}")
        return np.array([]), np.array([])


def _realized_forward_return(ts_arr, closes, t0: float, horizon_hours: int):
    """Return % from the first bar at/after t0 to ~horizon_hours later."""
    if len(ts_arr) == 0:
        return None
    i0 = int(np.searchsorted(ts_arr, t0))
    if i0 >= len(ts_arr):
        return None
    t_target = ts_arr[i0] + horizon_hours * 3600
    i1 = int(np.searchsorted(ts_arr, t_target))
    if i1 >= len(ts_arr):
        return None  # horizon not yet realized
    if closes[i0] <= 0:
        return None
    return (closes[i1] - closes[i0]) / closes[i0] * 100.0


def _spearman(x, y) -> float | None:
    if len(x) < 5:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def run_eval(days: int = 14, asset_filter: str | None = None) -> dict:
    from trading_utils import get_api

    entries = _load_entries(days)
    if asset_filter:
        entries = [e for e in entries if e.get('asset_type') == asset_filter]
    if not entries:
        print("No llm_analysis journal entries found — the bots journal one "
              "per LLM cycle; run again after some trading.")
        return {}

    api = get_api()

    # Group needed bar ranges by (symbol, asset_type)
    needed = defaultdict(list)  # (symbol, asset) -> [(t0, horizon, s, pred)]
    for e in entries:
        try:
            t0 = datetime.fromisoformat(e['ts']).timestamp()
        except (KeyError, ValueError):
            continue
        horizon = int(e.get('forward_bars', 24) or 24)
        for sym, v in (e.get('scores') or {}).items():
            s = v.get('s')
            if s is None:
                continue
            needed[(sym, e.get('asset_type', 'crypto'))].append(
                (t0, horizon, float(s), v.get('pred')))

    samples = []  # (s, realized, pred)
    for (sym, asset), rows in needed.items():
        t0s = [r[0] for r in rows]
        max_h = max(r[1] for r in rows)
        start = datetime.fromtimestamp(min(t0s)) - timedelta(hours=2)
        end = datetime.fromtimestamp(max(t0s)) + timedelta(hours=max_h + 6)
        ts_arr, closes = _bars_lookup(api, sym, asset, start, end)
        for t0, horizon, s, pred in rows:
            realized = _realized_forward_return(ts_arr, closes, t0, horizon)
            if realized is not None:
                samples.append((s, realized, pred))

    if not samples:
        print("No realized samples yet (horizons may not have elapsed).")
        return {}

    s_vals = np.array([x[0] for x in samples])
    realized = np.array([x[1] for x in samples])

    buckets = [(0.0, 0.15, 'VETO  '), (0.15, 0.35, 'bear  '),
               (0.35, 0.50, 'lean- '), (0.50, 0.65, 'lean+ '),
               (0.65, 0.85, 'bull  '), (0.85, 1.01, 'strong')]
    print(f"\n=== LLM GATE EVALUATION ({len(samples)} scored samples, "
          f"last {days}d) ===")
    print(f"{'bucket':<8}{'n':>5}{'avg fwd ret %':>15}{'hit rate':>10}")
    report = {'n': len(samples), 'buckets': {}}
    for lo, hi, label in buckets:
        mask = (s_vals >= lo) & (s_vals < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg = float(realized[mask].mean())
        hit = float((realized[mask] > 0).mean())
        report['buckets'][label.strip()] = {'n': n, 'avg_fwd_ret_pct': round(avg, 3),
                                            'hit_rate': round(hit, 3)}
        print(f"{label:<8}{n:>5}{avg:>15.3f}{hit:>10.2f}")

    rho = _spearman(s_vals, realized)
    report['spearman_s_vs_return'] = round(rho, 3) if rho is not None else None
    print(f"\nSpearman rank corr (s vs realized): "
          f"{rho:.3f}" if rho is not None else "\nSpearman: n/a")

    veto_mask = s_vals < VETO_THRESHOLD
    if veto_mask.any():
        saved = -float(realized[veto_mask].sum())
        report['veto_counterfactual_pct'] = round(saved, 2)
        print(f"Veto counterfactual: blocking s<{VETO_THRESHOLD} avoided "
              f"{saved:+.2f}% cumulative forward return "
              f"({'saved money' if saved > 0 else 'COST money'})")

    # Sizing value: correlation between (0.5+s) overlay and realized
    if rho is not None:
        verdict = ("LLM gate shows predictive value — keep it."
                   if rho > 0.05 else
                   "LLM gate shows NO predictive value at this sample — "
                   "consider disabling it (set enabled=false) and saving the spend.")
        report['verdict'] = verdict
        print(f"\nVerdict: {verdict}")

    out = BASE_DIR / 'llm_eval_report.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report: {out}")
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Evaluate LLM gate vs realized returns')
    ap.add_argument('--days', type=int, default=14)
    ap.add_argument('--asset', choices=['crypto', 'stock'], default=None)
    args = ap.parse_args()
    run_eval(args.days, args.asset)
