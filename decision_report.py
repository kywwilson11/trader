"""Gate attribution + conviction calibration from the decision journals.

Every veto the system logs is a COUNTERFACTUAL trade: replaying it
through the SAME policy_exits kernel that prices live exits tells us
what that veto actually saved (or cost). A gate whose vetoed entries
would have averaged a positive net return is charging admission, not
providing protection — and the only way to know is to measure.

Two products, both feeding the high-conviction program:

  1. GATE ATTRIBUTION — per skip_reason: veto count, counterfactual
     mean/total net return (after the same round-trip costs live pays).
     "Saved" is the NEGATIVE of counterfactual P&L: a gate that vetoed
     trades averaging -0.8% net saved +0.8% per veto.

  2. CONVICTION CALIBRATION — for TAKEN entries: realized outcome by
     predicted-return decile and by meta-probability bucket. This is
     the red team's "does high conviction select signal or just tail
     noise" experiment, answered from our own logs.

Usage:
    python decision_report.py --days 30
Writes decision_report.json beside the journals.
"""

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

JOURNAL_DIR = BASE_DIR / 'journals'

GATE_REASONS = ['sentiment_block', 'llm_veto', 'meta_veto',
                'q10_tail_veto', 'edgar_event']
MAX_HOLD_BARS = {'crypto': 24, 'stock': 24}   # vertical barrier for replays


def _is_crypto(sym: str) -> bool:
    return '/' in sym


def load_journal(days: int) -> list[dict]:
    rows = []
    today = dt.date.today()
    for d in range(days + 1):
        p = JOURNAL_DIR / f"{(today - dt.timedelta(days=d)).isoformat()}.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _atr14(df) -> np.ndarray:
    """Lightweight ATR(14) — enough for the exit kernel's stop math."""
    import pandas as pd
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=5).mean().values


def _eod_mask(index, asset_type: str) -> np.ndarray:
    if asset_type == 'crypto':
        return np.zeros(len(index), dtype=bool)
    from policy_exits import eod_mask_from_index
    return eod_mask_from_index(index, asset_type)


def replay_entry(bars, ts, asset_type: str) -> float | None:
    """Counterfactual NET % return of an entry at the first bar at/after
    ts, exited by the live policy stack (stops/trail/TP/EOD/vertical —
    no signal exits, since the counterfactual has no later model preds).
    None when the horizon isn't resolvable yet."""
    from policy_exits import exit_walk
    from strategy_config import policy_for
    from fees import round_trip_cost_pct

    idx = bars.index
    i = int(idx.searchsorted(ts))
    if i >= len(bars) - 2:
        return None
    sub = bars.iloc[i:]
    n = len(sub)
    max_hold = MAX_HOLD_BARS.get(asset_type, 24)
    if n < 3:
        return None
    closes = sub['Close'].values.astype(np.float64)
    exit_idx, exit_px, _reason = exit_walk(
        closes,
        sub['High'].values.astype(np.float64),
        sub['Low'].values.astype(np.float64),
        sub['Open'].values.astype(np.float64),
        _atr14(sub),
        _eod_mask(sub.index, asset_type),
        policy_for(asset_type),
        max_hold=max_hold, use_signal_exit=False)
    j = int(exit_idx[0])
    if j <= 0 or j >= n:
        return None
    # Unresolved horizon: vertical barrier coincides with the last bar
    if j == n - 1 and (n - 1) < max_hold and asset_type == 'crypto':
        return None
    gross = (float(exit_px[0]) - closes[0]) / closes[0] * 100.0
    spread = 0.10 if asset_type == 'crypto' else 0.05
    return gross - round_trip_cost_pct(asset_type, spread)


def gate_attribution(rows: list[dict], api) -> dict:
    """Counterfactual P&L per gate from journaled skips."""
    from market_data import fetch_bars_alpaca, fetch_stock_bars_alpaca
    import pandas as pd

    skips = [r for r in rows if r.get('action') == 'skip'
             and r.get('skip_reason') in GATE_REASONS and r.get('symbol')]
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for r in skips:
        by_symbol[r['symbol']].append(r)

    per_gate = defaultdict(list)
    unresolved = 0
    for sym, srows in by_symbol.items():
        asset = 'crypto' if _is_crypto(sym) else 'stock'
        try:
            bars = (fetch_bars_alpaca(api, sym) if asset == 'crypto'
                    else fetch_stock_bars_alpaca(api, sym))
        except Exception:
            bars = None
        if bars is None or len(bars) < 30:
            unresolved += len(srows)
            continue
        if bars.index.tz is None:
            bars = bars.tz_localize('UTC')
        for r in srows:
            try:
                ts = pd.Timestamp(r['ts'])
                if ts.tz is None:
                    ts = ts.tz_localize('UTC')
            except (ValueError, KeyError):
                continue
            net = replay_entry(bars, ts, asset)
            if net is None:
                unresolved += 1
                continue
            per_gate[r['skip_reason']].append(net)

    out = {}
    for gate in GATE_REASONS:
        vals = np.asarray(per_gate.get(gate, []), dtype=float)
        if vals.size == 0:
            continue
        out[gate] = {
            'vetoes_priced': int(vals.size),
            'counterfactual_mean_net_pct': round(float(vals.mean()), 3),
            'counterfactual_hit_rate': round(float((vals > 0).mean()), 3),
            # positive saved = the gate is earning its keep
            'saved_total_pct': round(float(-vals.sum()), 2),
        }
    out['_unresolved'] = unresolved
    return out


def conviction_calibration(rows: list[dict], api) -> dict:
    """Realized counterfactual outcome of TAKEN entries by conviction
    bucket (prediction-magnitude tercile x meta-probability band).

    Uses the same kernel replay as the vetoes so taken and vetoed trades
    are priced identically (journaled realized exits depend on signal
    exits the replay can't see; the kernel replay is the comparable
    yardstick)."""
    from market_data import fetch_bars_alpaca, fetch_stock_bars_alpaca
    import pandas as pd

    buys = [r for r in rows if r.get('action') == 'buy'
            and r.get('pred_return') is not None and r.get('symbol')]
    if not buys:
        return {}

    samples = []   # (pred, meta_prob_or_None, net)
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for r in buys:
        by_symbol[r['symbol']].append(r)
    for sym, srows in by_symbol.items():
        asset = 'crypto' if _is_crypto(sym) else 'stock'
        try:
            bars = (fetch_bars_alpaca(api, sym) if asset == 'crypto'
                    else fetch_stock_bars_alpaca(api, sym))
        except Exception:
            continue
        if bars is None or len(bars) < 30:
            continue
        if bars.index.tz is None:
            bars = bars.tz_localize('UTC')
        for r in srows:
            try:
                ts = pd.Timestamp(r['ts'])
                if ts.tz is None:
                    ts = ts.tz_localize('UTC')
            except (ValueError, KeyError):
                continue
            net = replay_entry(bars, ts, asset)
            if net is None:
                continue
            samples.append((float(r['pred_return']),
                            r.get('meta_prob'), net))

    if len(samples) < 9:
        return {'n': len(samples),
                'note': 'too few resolvable taken entries to bucket'}

    preds = np.array([s[0] for s in samples])
    nets = np.array([s[2] for s in samples])
    out = {'n': len(samples)}

    # Prediction-magnitude terciles: does the top third out-earn?
    qs = np.quantile(preds, [1 / 3, 2 / 3])
    buckets = {'pred_low': preds <= qs[0],
               'pred_mid': (preds > qs[0]) & (preds <= qs[1]),
               'pred_high': preds > qs[1]}
    for name, mask in buckets.items():
        if mask.sum():
            out[name] = {'n': int(mask.sum()),
                         'mean_net_pct': round(float(nets[mask].mean()), 3),
                         'hit_rate': round(float((nets[mask] > 0).mean()), 3)}

    metas = [(s[1], s[2]) for s in samples if s[1] is not None]
    if len(metas) >= 9:
        mp = np.array([m[0] for m in metas])
        mn = np.array([m[1] for m in metas])
        for name, lo, hi in (('meta_0.30_0.45', 0.30, 0.45),
                             ('meta_0.45_0.60', 0.45, 0.60),
                             ('meta_0.60_1.00', 0.60, 1.01)):
            mask = (mp >= lo) & (mp < hi)
            if mask.sum():
                out[name] = {'n': int(mask.sum()),
                             'mean_net_pct': round(float(mn[mask].mean()), 3),
                             'hit_rate': round(float((mn[mask] > 0).mean()), 3)}
    return out


def run_report(days: int = 30) -> dict:
    rows = load_journal(days)
    if not rows:
        print("No journal entries found.")
        return {}
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from trading_utils import get_api
        api = get_api()
    except Exception as e:
        print(f"No API available ({e}) — cannot price counterfactuals.")
        return {}

    gates = gate_attribution(rows, api)
    conviction = conviction_calibration(rows, api)

    print(f"\n=== GATE ATTRIBUTION (last {days}d) ===")
    print(f"{'gate':<18}{'vetoes':>7}{'cf mean':>9}{'cf hit':>8}{'saved':>9}")
    for gate, g in gates.items():
        if gate.startswith('_'):
            continue
        print(f"{gate:<18}{g['vetoes_priced']:>7}"
              f"{g['counterfactual_mean_net_pct']:>8.2f}%"
              f"{g['counterfactual_hit_rate']:>8.0%}"
              f"{g['saved_total_pct']:>8.1f}%")
    print(f"(unresolved/unpriceable vetoes: {gates.get('_unresolved', 0)})")
    print("A gate with NEGATIVE 'cf mean' is earning its keep; a gate "
          "whose vetoed trades average positive is charging admission.")

    print(f"\n=== CONVICTION CALIBRATION (taken entries, kernel-replayed) ===")
    for k, v in conviction.items():
        if isinstance(v, dict) and 'mean_net_pct' in v:
            print(f"{k:<18} n={v['n']:<5} mean={v['mean_net_pct']:+.2f}%  "
                  f"hit={v['hit_rate']:.0%}")
    if conviction.get('n'):
        print(f"(n={conviction['n']} priced entries)")

    report = {'generated': dt.datetime.now().isoformat(), 'days': days,
              'gates': gates, 'conviction': conviction}
    out = BASE_DIR / 'decision_report.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {out}")
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Gate attribution report')
    ap.add_argument('--days', type=int, default=30)
    args = ap.parse_args()
    run_report(args.days)
