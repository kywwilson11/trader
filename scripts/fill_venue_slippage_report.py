#!/usr/bin/env python3
"""Crypto fill-venue slippage (X2) + post-fill markout (X4) report — measurement-only.

Research home: research/campaign_2026-09_jetson/research_engine.md, section
"R3 scout — E2/E4 follow-ups" (W12). Nothing here changes how orders are
posted: the killed [wave-7] adverse-selection contrarian-posting guard stays
dead; X4 only MEASURES post-fill returns (research/KILL_LIST.md).

Inputs come from the BROKER, not the decision journals (journals from
2026-02..05 carry no fill price / decision quote; newer buy rows do, but not
the broker order id). A bundle JSON holds:
  orders          {order_id: Alpaca order object}   (GET /v2/orders[/{id}])
  fills           [FILL activity, ...]               (GET /v2/account/activities/FILL)
  arrival_quotes  {"<order_id>|<loc>": {bp, ap, t}}  last quote at/before
                  submitted_at on that crypto data location (us = Alpaca,
                  us-1 = Kraken; docs.alpaca.markets real-time-crypto-pricing-data)
  bars            {"<loc>|<symbol>": {"YYYY-MM-DDTHH:MM": close}}  1-min bars
                  (markout prices; the us-1 close is the reference by default)

Modes:
  --replay BUNDLE   offline, no network (the default way to run this).
  --pull BUNDLE     READ-ONLY GETs to build a bundle (needs ALPACA_API_KEY /
                    ALPACA_API_SECRET in the env or .env; places nothing),
                    then reports on it. Writes only the file you name.
  --json PATH       also write the report dict as JSON ('-' = stdout).

Pre-registered rules (research_engine.md, W12):
  X2  taker fills (type=market: mktfb-*, legacy, exits), n >= --min-n (30): realized slippage is
      measured against the us-1 mid at arrival (common, fresh reference).
      Predictor A = half-spread of the us quote, B = half-spread of the us-1
      quote. The winner is the predictor whose mean absolute error is
      <= (1 - --margin) x the other's (margin 0.25). Else "no_change".
  X4  maker fills (client_order_id 'maker-*') vs taker fills: net markout
      at +H min = (px_ref(t_fill+H) - fill_vwap)/fill_vwap*1e4 - fee_bps.
      TOXIC iff n_maker >= 50, n_taker >= 20 and the 95 % bootstrap CI of
      mean(net_maker) - mean(net_taker) at the primary horizon (30 min)
      lies entirely below 0. Outcome if TOXIC: owner item on the EXISTING
      MAKER_ENTRIES_ENABLED flag only. Legacy (pre-ladder) passive limits are
      reported descriptively and never feed the verdict.
"""

import argparse
import bisect
import datetime as dt
import json
import os
import random
import statistics
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Published Alpaca tier-1 crypto fees (fees.py CRYPTO_MAKER_BPS / CRYPTO_TAKER_BPS;
# docs.alpaca.markets/docs/crypto-trading). Kept literal so this CLI imports no
# repo module; tests/test_fill_venue_slippage_report.py pins them against fees.py.
MAKER_FEE_BPS = 15.0
TAKER_FEE_BPS = 25.0
HORIZONS_MIN = (1, 5, 30)
PRIMARY_H = 30
LOCS = ('us', 'us-1')
DATA_URL = 'https://data.alpaca.markets/v1beta3/crypto/{loc}'


def parse_ts(s):
    """Alpaca RFC-3339 (Z, 0-9 fractional digits) -> naive UTC datetime."""
    s = str(s).replace('Z', '')
    if '+' in s[10:]:
        s = s[:10] + s[10:].split('+')[0]
    if '.' in s:
        head, frac = s.split('.', 1)
        s = head + '.' + (frac + '000000')[:6]
    return dt.datetime.fromisoformat(s)


def classify_tactic(order):
    """Tactic from the client_order_id tag the live code mints
    (order_utils._maker_rung_id 'maker-', make_client_order_id tags
    'mktfb'/'trader'/'csell'/'cstop'/'stop'/'flatten'); untagged (pre-2026-08
    UUID) orders are 'legacy_market' / 'legacy_limit'."""
    cid = str(order.get('client_order_id') or '')
    typ = order.get('type') or order.get('order_type')
    tag = cid.split('-', 1)[0] if '-' in cid else ''
    if tag == 'maker':
        return 'maker'
    if tag == 'mktfb':
        return 'taker_fallback'
    if tag in ('trader', 'csell', 'cstop', 'stop', 'flatten'):
        return tag
    return 'legacy_market' if typ == 'market' else 'legacy_limit'


def is_taker(r):
    """Taker = any market-type order (mktfb-* fallbacks, legacy market orders,
    market exits). Marketable limits are NOT counted: their fill can rest."""
    return r.get('type') == 'market'


def aggregate_fills(fills):
    """Per order_id: qty, notional, vwap, n partial fills, last fill ts."""
    out = {}
    for f in fills:
        if '/' not in str(f.get('symbol', '')):
            continue                      # crypto only (Alpaca pairs carry '/')
        g = out.setdefault(f['order_id'], {'qty': 0.0, 'notional': 0.0,
                                           'n': 0, 'last': ''})
        q, p = float(f['qty']), float(f['price'])
        g['qty'] += q
        g['notional'] += q * p
        g['n'] += 1
        g['last'] = max(g['last'], f['transaction_time'])
    for g in out.values():
        g['vwap'] = g['notional'] / g['qty'] if g['qty'] > 0 else None
    return out


def price_at(series_keys, series, t, max_lag_min=10):
    """Last 1-min close at or before t (within max_lag_min), else None."""
    key = t.strftime('%Y-%m-%dT%H:%M')
    i = bisect.bisect_right(series_keys, key) - 1
    if i < 0:
        return None
    k = series_keys[i]
    if (t - dt.datetime.fromisoformat(k)).total_seconds() > max_lag_min * 60:
        return None
    return series[k]


def build_rows(bundle, ref_loc='us-1'):
    orders = bundle.get('orders', {})
    quotes = bundle.get('arrival_quotes', {})
    bars = {k: (sorted(v), v) for k, v in bundle.get('bars', {}).items()}
    rows = []
    for oid, g in aggregate_fills(bundle.get('fills', [])).items():
        o = orders.get(oid)
        if not o or not o.get('submitted_at') or not g['vwap']:
            continue
        side = 1 if o.get('side') == 'buy' else -1
        vw = g['vwap']
        r = {'order_id': oid, 'symbol': o['symbol'], 'side': o.get('side'),
             'type': o.get('type'), 'tactic': classify_tactic(o),
             'submitted_at': o['submitted_at'], 'filled_at': g['last'],
             'vwap': vw, 'qty': g['qty'], 'n_fills': g['n'],
             'wait_s': (parse_ts(g['last']) - parse_ts(o['submitted_at'])).total_seconds()}
        for loc in LOCS:
            q = quotes.get(f'{oid}|{loc}') or {}
            bp, ap = q.get('bp'), q.get('ap')
            if bp and ap and bp > 0 and ap > 0:
                mid = (bp + ap) / 2.0
                r[f'half_spread_{loc}'] = (ap - bp) / mid * 1e4 / 2.0
                r[f'slip_{loc}'] = side * (vw - mid) / mid * 1e4
                r[f'touch_gap_{loc}'] = side * (vw - (ap if side > 0 else bp)) / vw * 1e4
                if q.get('t'):
                    r[f'quote_age_{loc}'] = (parse_ts(o['submitted_at'])
                                             - parse_ts(q['t'])).total_seconds()
        keys, ser = bars.get(f"{ref_loc}|{o['symbol']}", ([], {}))
        tf = parse_ts(g['last'])
        p0 = price_at(keys, ser, tf) if keys else None
        fee = MAKER_FEE_BPS if r['tactic'] == 'maker' else TAKER_FEE_BPS
        for h in HORIZONS_MIN:
            ph = price_at(keys, ser, tf + dt.timedelta(minutes=h)) if keys else None
            if ph:
                r[f'markout_{h}'] = side * (ph - vw) / vw * 1e4
                r[f'net_markout_{h}'] = r[f'markout_{h}'] - fee
                if p0:
                    r[f'pure_move_{h}'] = side * (ph - p0) / p0 * 1e4
        rows.append(r)
    return rows


def _summ(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return {'n': 0}
    n = len(v)
    return {'n': n, 'median': statistics.median(v), 'mean': statistics.fmean(v),
            'p10': v[int(0.1 * (n - 1))], 'p90': v[int(0.9 * (n - 1))]}


def x2_verdict(rows, min_n=30, margin=0.25, per_symbol_min=10):
    """Which venue's half-spread predicts realized taker slippage (vs the
    us-1 arrival mid)? Returns the pre-registered verdict dict."""
    t = [r for r in rows if is_taker(r)
         and 'slip_us-1' in r and 'half_spread_us' in r and 'half_spread_us-1' in r]

    def mae(rs, loc):
        return statistics.fmean(abs(r['slip_us-1'] - r[f'half_spread_{loc}']) for r in rs)

    out = {'n_taker': len(t), 'min_n': min_n, 'margin': margin}
    if len(t) < min_n:
        out['verdict'] = 'insufficient_n'
        return out
    a, b = mae(t, 'us'), mae(t, 'us-1')
    out.update(mae_half_spread_us=a, mae_half_spread_us1=b)
    out['verdict'] = ('us' if a <= (1 - margin) * b else
                      'us-1' if b <= (1 - margin) * a else 'no_change')
    per = {}
    for sym in sorted({r['symbol'] for r in t}):
        rs = [r for r in t if r['symbol'] == sym]
        if len(rs) >= per_symbol_min:
            per[sym] = {'n': len(rs), 'mae_us': mae(rs, 'us'), 'mae_us1': mae(rs, 'us-1'),
                        'taker_slip_vs_us_mid_median': statistics.median(r['slip_us'] for r in rs),
                        'half_spread_us_median': statistics.median(r['half_spread_us'] for r in rs),
                        'half_spread_us1_median': statistics.median(r['half_spread_us-1'] for r in rs)}
    out['per_symbol'] = per
    out['touch_gap_us'] = _summ([r.get('touch_gap_us') for r in t])
    out['touch_gap_us1'] = _summ([r.get('touch_gap_us-1') for r in t])
    return out


def bootstrap_diff_ci(a, b, n_boot=4000, seed=0, alpha=0.05):
    rng = random.Random(seed)
    d = sorted(statistics.fmean(rng.choices(a, k=len(a)))
               - statistics.fmean(rng.choices(b, k=len(b))) for _ in range(n_boot))
    return d[int(alpha / 2 * n_boot)], d[int((1 - alpha / 2) * n_boot) - 1]


def x4_verdict(rows, min_maker=50, min_taker=20, n_boot=4000, seed=0, side='buy'):
    rs = [r for r in rows if r['side'] == side]
    out = {'side': side, 'primary_horizon_min': PRIMARY_H, 'by_tactic': {}}
    for tac in sorted({r['tactic'] for r in rs}):
        sub = [r for r in rs if r['tactic'] == tac]
        out['by_tactic'][tac] = {f'{k}_{h}': _summ([r.get(f'{k}_{h}') for r in sub])
                                 for h in HORIZONS_MIN
                                 for k in ('markout', 'net_markout', 'pure_move')}
    key = f'net_markout_{PRIMARY_H}'
    mk = [r[key] for r in rs if r['tactic'] == 'maker' and key in r]
    tk = [r[key] for r in rs if is_taker(r) and r['tactic'] != 'maker' and key in r]
    out.update(n_maker=len(mk), n_taker=len(tk))
    if len(mk) < min_maker or len(tk) < min_taker:
        out['verdict'] = 'insufficient_n'
        return out
    lo, hi = bootstrap_diff_ci(mk, tk, n_boot=n_boot, seed=seed)
    out.update(diff_mean=statistics.fmean(mk) - statistics.fmean(tk), ci95=[lo, hi])
    out['verdict'] = 'toxic' if hi < 0 else 'not_toxic'
    return out


def build_report(bundle, min_n=30, margin=0.25, n_boot=4000, seed=0):
    rows = build_rows(bundle)
    return {'n_orders': len(rows),
            'tactics': {t: sum(1 for r in rows if r['tactic'] == t)
                        for t in sorted({r['tactic'] for r in rows})},
            'x2': x2_verdict(rows, min_n=min_n, margin=margin),
            'x4': x4_verdict(rows, n_boot=n_boot, seed=seed)}


def pull_bundle(days, out_path):  # pragma: no cover - network, read-only GETs
    """READ-ONLY: orders, FILL activities, arrival quotes and 1-min bars."""
    import time
    import requests
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, '.env'))
    except ImportError:
        pass
    h = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'],
         'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}
    base = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    base = base[:-3] if base.endswith('/v2') else base
    after = (dt.datetime.utcnow() - dt.timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')

    def get(url, params=None):
        r = requests.get(url, params=params, headers=h, timeout=20)
        time.sleep(0.31)                  # stay under the 200 req/min data limit
        r.raise_for_status()
        return r.json()

    fills, token = [], None
    while True:
        p = {'page_size': 100, 'direction': 'desc', 'after': after}
        if token:
            p['page_token'] = token
        js = get(base + '/v2/account/activities/FILL', p)
        fills += js
        if len(js) < 100:
            break
        token = js[-1]['id']
    orders = {}
    for oid in sorted({f['order_id'] for f in fills if '/' in f.get('symbol', '')}):
        orders[oid] = get(f'{base}/v2/orders/{oid}')
    quotes = {}
    for oid, o in orders.items():
        t = parse_ts(o['submitted_at'])
        for loc in LOCS:
            js = get(DATA_URL.format(loc=loc) + '/quotes',
                     {'symbols': o['symbol'], 'limit': 1, 'sort': 'desc',
                      'start': (t - dt.timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                      'end': t.strftime('%Y-%m-%dT%H:%M:%S.%fZ')})
            q = (js.get('quotes') or {}).get(o['symbol']) or []
            quotes[f'{oid}|{loc}'] = q[0] if q else {}
    bars = {}
    syms = sorted({o['symbol'] for o in orders.values()})
    for day in sorted({f['transaction_time'][:10] for f in fills if f['order_id'] in orders}):
        s = dt.datetime.fromisoformat(day)
        tok = None
        while syms:
            p = {'symbols': ','.join(syms), 'timeframe': '1Min', 'limit': 10000,
                 'start': s.strftime('%Y-%m-%dT%H:%M:%SZ'),
                 'end': (s + dt.timedelta(days=1, minutes=45)).strftime('%Y-%m-%dT%H:%M:%SZ')}
            if tok:
                p['page_token'] = tok
            js = get(DATA_URL.format(loc='us-1') + '/bars', p)
            for sym, bl in (js.get('bars') or {}).items():
                bars.setdefault(f'us-1|{sym}', {}).update({b['t'][:16]: b['c'] for b in bl})
            tok = js.get('next_page_token')
            if not tok:
                break
    bundle = {'orders': orders, 'fills': fills, 'arrival_quotes': quotes, 'bars': bars}
    with open(out_path, 'w') as fh:
        json.dump(bundle, fh)
    return bundle


def _fmt(x):
    return 'n/a' if x is None else f'{x:.2f}'


def print_report(rep):
    print(f"orders with fills: {rep['n_orders']}  tactics: {rep['tactics']}")
    x2 = rep['x2']
    print(f"\n[X2] taker fills n={x2['n_taker']} (min {x2['min_n']}) -> verdict: {x2['verdict']}")
    if 'mae_half_spread_us' in x2:
        print(f"  MAE vs half-spread us={_fmt(x2['mae_half_spread_us'])} bps, "
              f"us-1={_fmt(x2['mae_half_spread_us1'])} bps (margin {x2['margin']:.0%})")
        print(f"  {'symbol':10} {'n':>4} {'mae_us':>7} {'mae_us1':>8} {'slip_med':>9} "
              f"{'hs_us':>6} {'hs_us1':>7}")
        for s, v in x2['per_symbol'].items():
            print(f"  {s:10} {v['n']:4d} {v['mae_us']:7.1f} {v['mae_us1']:8.1f} "
                  f"{v['taker_slip_vs_us_mid_median']:9.1f} {v['half_spread_us_median']:6.1f} "
                  f"{v['half_spread_us1_median']:7.2f}")
    x4 = rep['x4']
    print(f"\n[X4] buys: maker n={x4['n_maker']}, taker n={x4['n_taker']} -> verdict: {x4['verdict']}")
    if 'ci95' in x4:
        print(f"  mean net markout maker-taker @+{PRIMARY_H}m = {x4['diff_mean']:.2f} bps, "
              f"95% CI [{x4['ci95'][0]:.2f}, {x4['ci95'][1]:.2f}]")
    for tac, d in x4['by_tactic'].items():
        cells = ', '.join(f"+{h}m pure {_fmt(d[f'pure_move_{h}'].get('mean'))}"
                          for h in HORIZONS_MIN)
        print(f"  {tac:16} n={d[f'markout_{PRIMARY_H}']['n']:4d}  {cells}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--replay', metavar='BUNDLE', help='offline: report on a saved bundle JSON')
    src.add_argument('--pull', metavar='BUNDLE', help='read-only GETs -> write BUNDLE, then report')
    ap.add_argument('--days', type=int, default=240, help='--pull lookback (default 240)')
    ap.add_argument('--min-n', type=int, default=30, help='X2 minimum taker fills (default 30)')
    ap.add_argument('--margin', type=float, default=0.25, help='X2 MAE win margin (default 0.25)')
    ap.add_argument('--n-boot', type=int, default=4000, help='X4 bootstrap resamples')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', metavar='PATH', help="write the report JSON ('-' = stdout)")
    a = ap.parse_args(argv)
    if a.replay:
        with open(a.replay) as fh:
            bundle = json.load(fh)
    else:
        bundle = pull_bundle(a.days, a.pull)
    rep = build_report(bundle, min_n=a.min_n, margin=a.margin, n_boot=a.n_boot, seed=a.seed)
    if a.json == '-':
        json.dump(rep, sys.stdout, indent=1, default=str)
        print()
    else:
        print_report(rep)
        if a.json:
            with open(a.json, 'w') as fh:
                json.dump(rep, fh, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
