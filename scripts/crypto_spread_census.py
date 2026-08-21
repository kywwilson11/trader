"""Crypto venue spread census (B05.1 / NEW-opportunity #1).

Owner evidence for liquidity.py's crypto tier map and the KILL_LIST:90
ruling. Polls Alpaca v1beta3 LATEST crypto quotes for the CRYPTO_POOL over a
configurable window, per venue loc (us / us-1 / us-2 / eu-1), and writes
per-pair spread statistics to JSON. Jetson-run (long window + API keys);
pure helpers are Mac-tested. Measurement-only: nothing reads this file
automatically — liquidity.get_crypto_spread_tier consumes it ONLY under the
dark TRADER_CRYPTO_SPREAD_STAMP flag.

Units note (wave-7 tripwire): bidask/liquidity convention is price FRACTION
x100 = PERCENT. Everything in this script is PERCENT of mid; a 100x
percent-vs-fraction slip lands far outside the sanity_check bands below.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import datetime
import json
import os
import time

import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from stock_config import CRYPTO_POOL

DATA_URL = "https://data.alpaca.markets/v1beta3/crypto/{loc}/latest/quotes"


def quote_spread_pct(bp, ap):
    """Quoted spread as PERCENT of mid, or None for unusable quotes.

    None unless both bid and ask are finite numbers, bp > 0, ap > 0, and
    ap >= bp (crossed/locked-degenerate quotes are counted by the caller,
    never averaged into the stats). A locked book (ap == bp) returns 0.0.
    """
    try:
        b, a = float(bp), float(ap)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(b) and np.isfinite(a)):
        return None
    if b <= 0 or a <= 0 or a < b:
        return None
    return (a - b) / ((a + b) / 2.0) * 100.0


def summarize(samples):
    """Per-pair spread stats from a {sym: [spread_pct, ...]} map.

    Pairs with samples get {'n', 'median_spread_pct', 'p75_spread_pct',
    'p90_spread_pct', 'p95_spread_pct', 'mean_spread_pct'}; pairs with no
    samples get {'n': 0}.
    """
    out = {}
    for sym, vals in samples.items():
        if not vals:
            out[sym] = {'n': 0}
            continue
        arr = np.asarray(vals, dtype=float)
        out[sym] = {
            'n': int(arr.size),
            'median_spread_pct': float(np.median(arr)),
            'p75_spread_pct': float(np.percentile(arr, 75)),
            'p90_spread_pct': float(np.percentile(arr, 90)),
            'p95_spread_pct': float(np.percentile(arr, 95)),
            'mean_spread_pct': float(np.mean(arr)),
        }
    return out


def sanity_check(summary, min_n=30):
    """Units-slip tripwire (B05). Returns a list of violation strings.

    BTC/USD median must sit in [0.005, 0.5]%; every other pair's median in
    [0.01, 5.0]%. Only evaluated for pairs with n >= min_n. A wave-7-style
    100x percent-vs-fraction slip lands far outside these bands.
    """
    violations = []
    for sym, stats in summary.items():
        n = stats.get('n', 0)
        if n < min_n:
            continue
        med = stats.get('median_spread_pct')
        if med is None:
            continue
        lo, hi = (0.005, 0.5) if sym == 'BTC/USD' else (0.01, 5.0)
        if not (lo <= med <= hi):
            violations.append(
                f"{sym}: median {med:.4f}% outside [{lo}, {hi}]% "
                f"(n={n}) — possible units slip or degenerate book")
    return violations


def fetch_latest_quotes(loc, symbols, headers, timeout=10):
    """One poll of the latest-quotes endpoint. Returns the response's
    'quotes' map ({sym: {'bp':..., 'ap':..., ...}}) or {} on any error."""
    try:
        resp = requests.get(DATA_URL.format(loc=loc),
                            params={'symbols': ','.join(symbols)},
                            headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(f"[CENSUS] HTTP {resp.status_code}: {resp.text[:200]}")
            return {}
        return resp.json().get('quotes', {})
    except Exception as e:
        print(f"[CENSUS] poll failed: {type(e).__name__}: {e}")
        return {}


def main():
    ap = argparse.ArgumentParser(description="Crypto venue spread census")
    ap.add_argument('--minutes', type=float, default=60,
                    help="polling window length (default 60)")
    ap.add_argument('--interval', type=float, default=5.0,
                    help="seconds between polls (default 5)")
    ap.add_argument('--loc', default='us',
                    help="Alpaca crypto venue loc: us / us-1 / us-2 / eu-1")
    ap.add_argument('--symbols', default=None,
                    help="comma-separated pairs (default: CRYPTO_POOL)")
    ap.add_argument('--out', default='crypto_spread_census.json')
    args = ap.parse_args()

    symbols = ([s.strip() for s in args.symbols.split(',') if s.strip()]
               if args.symbols else list(CRYPTO_POOL))
    headers = {}
    key, sec = os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET')
    if key and sec:
        # Crypto data endpoints accept keyless at reduced limits; send keys
        # when available.
        headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': sec}

    samples = {s: [] for s in symbols}
    bad = {s: 0 for s in symbols}
    deadline = time.monotonic() + args.minutes * 60.0
    n_polls = 0
    print(f"[CENSUS] loc={args.loc} window={args.minutes}min "
          f"interval={args.interval}s symbols={len(symbols)}")
    try:
        while time.monotonic() < deadline:
            quotes = fetch_latest_quotes(args.loc, symbols, headers)
            n_polls += 1
            for sym in symbols:
                q = quotes.get(sym)
                if q is None:
                    continue
                sp = quote_spread_pct(q.get('bp'), q.get('ap'))
                if sp is None:
                    bad[sym] += 1
                else:
                    samples[sym].append(sp)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("[CENSUS] interrupted — writing partial results")

    summary = summarize(samples)
    violations = sanity_check(summary)
    out = {
        'generated_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'loc': args.loc,
        'window_min': args.minutes,
        'interval_s': args.interval,
        'n_polls': n_polls,
        'pairs': {sym: {**summary[sym], 'n_bad': bad[sym]}
                  for sym in symbols},
        'sanity': {'ok': not violations, 'violations': violations},
    }
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"[CENSUS] wrote {args.out} ({n_polls} polls)")

    print(f"{'pair':<10} {'n':>6} {'median%':>9} {'p90%':>9} {'bad':>5}")
    for sym in symbols:
        st = summary[sym]
        if st.get('n', 0):
            print(f"{sym:<10} {st['n']:>6} {st['median_spread_pct']:>9.4f} "
                  f"{st['p90_spread_pct']:>9.4f} {bad[sym]:>5}")
        else:
            print(f"{sym:<10} {0:>6} {'-':>9} {'-':>9} {bad[sym]:>5}")
    if violations:
        print("[CENSUS] SANITY VIOLATIONS:")
        for v in violations:
            print(f"  {v}")
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
