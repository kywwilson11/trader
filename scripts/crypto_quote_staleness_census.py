"""Crypto quote-staleness census (measurement-only, read-only).

Why: base_loop._manage_stops does ``quote = self.get_quote(symbol); if quote
is None: continue`` — a None quote means NO software stop / trail / TP
evaluation for that position that cycle. For crypto, get_quote is
order_utils.get_crypto_quote -> order_utils.get_quote(asset_type='crypto'),
which returns None when (a) the SDK call raises or the symbol is absent,
(b) bid/ask are non-finite or <= 0, or (c) the latest quote's ``t`` is more
than 180 s older than the machine's UTC clock, or (d) ``t`` is absent or
cannot be turned into a finite age (fail-closed since ENGINE r3 W10; attributed
here as 'stale' with age_s None).
A CROSSED quote (ask < bid) is only logged, NOT rejected. This CLI samples the latest quotes the same way
the loop does, attributes every None to a reason, and shows what the None
rate would be under alternative max-age thresholds — the owner's trade-off in
one table. Changing the live threshold is an OWNER decision; this script
changes nothing.

Live path parity: each sample makes ONE SDK call (trading_utils.get_api() —
the same legacy alpaca_trade_api REST client + request timeouts the loop
builds, ``get_latest_crypto_quotes([symbol])`` exactly as get_quote does) and
then feeds that same response object to the REAL order_utils.get_crypto_quote
through a one-shot stub, so the ``live_none`` column IS the live verdict, not
a re-implementation. The pure classifier (``classify``) is the reason
attribution; any disagreement with the live verdict is counted and printed.

Read-only: market-data GETs only; no orders, no account/position writes. No
file is written unless ``--json OUT`` is given. ``--replay FILE`` recomputes
the table from a saved JSON with no network. Imports on the dev Mac without
alpaca/dotenv (the SDK and trading_utils are imported lazily in ``sample``).

Examples (repo root)::

    python scripts/crypto_quote_staleness_census.py --duration 900 --json census.json
    python scripts/crypto_quote_staleness_census.py --replay census.json --thresholds 180,300,600,900
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import datetime
import json
import math
import signal
import time

import numpy as np

# The live rule: order_utils.get_quote rejects `age > 180` (inline literal,
# order_utils.py ~:164). tests/test_crypto_quote_staleness_census.py pins
# this constant against the real function at 179/181 s so it cannot drift.
LIVE_MAX_AGE_S = 180.0

DEFAULT_SYMBOLS = ('BTC/USD', 'ETH/USD', 'SOL/USD', 'LINK/USD', 'XRP/USD',
                   'DOGE/USD')
DEFAULT_THRESHOLDS = (180.0, 300.0, 600.0, 900.0)

# Reasons, in the order get_quote tests them. 'ok' is the only non-None one.
REASONS = ('ok', 'error', 'missing', 'degenerate', 'stale')


# ---------------------------------------------------------------- pure core

def _to_utc(ts):
    """Quote timestamp -> aware UTC datetime, or None if unparseable.

    Mirrors get_quote: pandas Timestamp -> pydatetime; a naive value is UTC
    (Alpaca timestamps are UTC by definition); ISO strings (replay) parsed."""
    if ts is None:
        return None
    try:
        if isinstance(ts, str):
            s = ts.strip()
            if s.endswith('Z'):
                s = s[:-1] + '+00:00'
            # fromisoformat (py3.10) accepts at most microseconds.
            if '.' in s:
                head, rest = s.split('.', 1)
                frac, tz = rest, ''
                for sep in ('+', '-'):
                    if sep in rest:
                        frac, tz = rest.split(sep, 1)
                        tz = sep + tz
                        break
                s = head + '.' + frac[:6] + tz
            ts = datetime.datetime.fromisoformat(s)
        if hasattr(ts, 'to_pydatetime'):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        return ts.astimezone(datetime.timezone.utc)
    except Exception:
        return None


# sample_once's stand-in for a ``t`` whose SDK getattr itself RAISED (legacy
# entity: raw 'garbage' -> pd.Timestamp parse error inside __getattr__).
_UNREADABLE_TS = object()


def _live_age_s(ts, now_utc):
    """(age_s, unparsable) exactly as order_utils.get_quote computes it.

    The value is converted the live way (to_pydatetime; naive = UTC; NO
    string parsing — the SDK hands get_quote a pandas Timestamp); an absent
    (None) timestamp, any TypeError/ValueError/AttributeError/OverflowError
    or a non-finite age -> (None, True): the live rule rejects the quote
    (fail-closed)."""
    try:
        if ts is None or ts is _UNREADABLE_TS:
            raise ValueError('quote t unreadable')
        if hasattr(ts, 'to_pydatetime'):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        age = (now_utc - ts.astimezone(datetime.timezone.utc)).total_seconds()
        if not math.isfinite(age):
            raise ValueError('non-finite age')
        _ = age > 0.0   # an uncomparable age raises TypeError, as live
        return age, False
    except (TypeError, ValueError, AttributeError, OverflowError):
        return None, True


def classify(bid, ask, ts, now_utc, fetch_error=False, missing=False,
             max_age_s=LIVE_MAX_AGE_S):
    """Attribute one sample under the live get_quote rule.

    Returns {'reason', 'age_s', 'spread_bps', 'crossed'}. reason is one of
    REASONS; everything except 'ok' means get_crypto_quote returned None.
    age_s is None when the timestamp is absent or unparsable (the live
    rule rejects it -> 'stale' with age_s None). spread_bps is
    (ask-bid)/mid*1e4 whenever bid/ask are finite and > 0 (crossed quotes
    give a negative value; they are still 'ok' under the live rule)."""
    out = {'reason': 'ok', 'age_s': None, 'spread_bps': None,
           'crossed': False}
    if fetch_error:
        out['reason'] = 'error'
        return out
    if missing:
        out['reason'] = 'missing'
        return out
    try:
        b, a = float(bid), float(ask)
    except (TypeError, ValueError):
        out['reason'] = 'error'      # float(None) raises inside get_quote
        return out
    mid = (a + b) / 2.0
    if (not (math.isfinite(b) and math.isfinite(a))
            or mid <= 0 or b <= 0 or a <= 0):
        out['reason'] = 'degenerate'
        return out
    out['spread_bps'] = (a - b) / mid * 1e4
    out['crossed'] = a < b
    age, unparsable = _live_age_s(ts, now_utc)
    if unparsable:
        out['reason'] = 'stale'      # live: fail-closed, no age
    elif age is not None:
        out['age_s'] = age
        if age > max_age_s:
            out['reason'] = 'stale'
    return out


def none_under(rec, threshold_s):
    """Would this sample be None if the max-age were ``threshold_s``?

    Non-age reasons (error/missing/degenerate) are None at every threshold,
    and so is a 'stale' sample WITHOUT an age (absent/unparsable timestamp — no
    threshold admits it); an age-bearing sample is None iff age_s >
    threshold_s (strict, as live)."""
    if rec['reason'] in ('error', 'missing', 'degenerate'):
        return True
    age = rec.get('age_s')
    if rec['reason'] == 'stale' and age is None:
        return True
    return age is not None and age > threshold_s


def _pct(vals, q):
    return float(np.percentile(np.asarray(vals, dtype=float), q)) if vals else None


def _max_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def build_table(records, thresholds=DEFAULT_THRESHOLDS, symbols=None):
    """Per-symbol summary from sample records (pure; replay uses this).

    records: dicts with at least 'symbol', 'reason', 'age_s', 'spread_bps',
    'crossed', and optionally 'live_none' (the real get_crypto_quote verdict)
    and 'poll' (poll index, for ordering the longest None streak)."""
    thresholds = tuple(float(t) for t in thresholds)
    if symbols is None:
        symbols = []
        for r in records:
            if r['symbol'] not in symbols:
                symbols.append(r['symbol'])
    table = {}
    for sym in symbols:
        rs = sorted((r for r in records if r['symbol'] == sym),
                    key=lambda r: r.get('poll', 0))
        n = len(rs)
        reasons = {k: sum(1 for r in rs if r['reason'] == k) for k in REASONS}
        ages = [r['age_s'] for r in rs if r.get('age_s') is not None]
        spreads = [r['spread_bps'] for r in rs
                   if r.get('spread_bps') is not None]
        live_flags = [r['reason'] != 'ok' for r in rs]
        disagree = sum(1 for r in rs if 'live_none' in r
                       and bool(r['live_none']) != (r['reason'] != 'ok'))
        row = {
            'n': n,
            'reasons': reasons,
            'none_rate_live': (sum(live_flags) / n) if n else None,
            'max_none_streak': _max_run(live_flags),
            'crossed': sum(1 for r in rs if r.get('crossed')),
            'no_ts': sum(1 for r in rs if r['reason'] == 'ok'
                         and r.get('age_s') is None),
            'age_p50_s': _pct(ages, 50), 'age_p90_s': _pct(ages, 90),
            'age_max_s': max(ages) if ages else None,
            'spread_p50_bps': _pct(spreads, 50),
            'spread_p90_bps': _pct(spreads, 90),
            'spread_max_bps': max(spreads) if spreads else None,
            'none_rate_at': {
                str(int(t) if t.is_integer() else t):
                    ((sum(none_under(r, t) for r in rs) / n) if n else None)
                for t in thresholds},
            'live_disagreements': disagree,
        }
        table[sym] = row
    return table


def format_table(table, thresholds=DEFAULT_THRESHOLDS, interval_s=None):
    """Plain-text table (one line per symbol)."""
    ths = [str(int(t) if float(t).is_integer() else t) for t in thresholds]
    f = lambda v, p=0: '-' if v is None else f"{v:.{p}f}"
    pr = lambda v: '-' if v is None else f"{100 * v:.1f}%"
    hdr = (f"{'symbol':<9} {'n':>4} {'None%live':>9} {'stale':>5} {'err':>3} "
           f"{'miss':>4} {'degen':>5} {'xed':>3} {'streak':>6} "
           f"{'age50':>6} {'age90':>6} {'agemax':>6} "
           f"{'sp50bp':>6} {'sp90bp':>6} "
           + ' '.join(f"{'N%@' + t:>8}" for t in ths))
    lines = [hdr]
    for sym, r in table.items():
        rs = r['reasons']
        lines.append(
            f"{sym:<9} {r['n']:>4} {pr(r['none_rate_live']):>9} "
            f"{rs['stale']:>5} {rs['error']:>3} {rs['missing']:>4} "
            f"{rs['degenerate']:>5} {r['crossed']:>3} {r['max_none_streak']:>6} "
            f"{f(r['age_p50_s']):>6} {f(r['age_p90_s']):>6} "
            f"{f(r['age_max_s']):>6} {f(r['spread_p50_bps'], 1):>6} "
            f"{f(r['spread_p90_bps'], 1):>6} "
            + ' '.join(f"{pr(r['none_rate_at'][t]):>8}" for t in ths))
    if interval_s:
        lines.append(f"(streak = longest run of consecutive live-None polls; "
                     f"x {interval_s:g}s = seconds with no software stop "
                     f"evaluation)")
    dis = sum(r['live_disagreements'] for r in table.values())
    lines.append(f"live-verdict disagreements with the classifier: {dis}")
    return '\n'.join(lines)


# ------------------------------------------------------------ live sampling

class _OneShotApi:
    """Feeds one already-fetched SDK response (or its exception) to the REAL
    order_utils.get_crypto_quote, so the live verdict is computed by the
    production function on exactly the object we classified."""

    def __init__(self, resp=None, exc=None):
        self._resp, self._exc = resp, exc

    def get_latest_crypto_quotes(self, symbols, *a, **k):
        if self._exc is not None:
            raise self._exc
        return self._resp


def sample_once(api, symbol, poll_idx, get_crypto_quote):
    """One loop-identical fetch + live verdict + reason attribution."""
    t0 = time.time()
    resp, exc = None, None
    try:
        resp = api.get_latest_crypto_quotes([symbol])
    except Exception as e:          # noqa: BLE001 — attributed, not hidden
        exc = e
    fetch_s = time.time() - t0
    q = None
    missing = False
    if exc is None:
        try:
            q = resp[symbol]
        except Exception:
            missing = True
        if q is None:
            missing = True
    now = datetime.datetime.now(datetime.timezone.utc)
    bid = ask = ts = None
    if q is not None:
        for name in ('bp', 'ap', 't'):
            try:
                val = getattr(q, name, None)
            except Exception:
                # a RAISING t getattr is fail-closed live; bp/ap -> None
                # (float(None) raises inside get_quote -> 'error')
                val = _UNREADABLE_TS if name == 't' else None
            if name == 'bp':
                bid = val
            elif name == 'ap':
                ask = val
            else:
                ts = val
    c = classify(bid, ask, ts, now, fetch_error=exc is not None,
                 missing=missing)
    live = get_crypto_quote(_OneShotApi(resp, exc), symbol)
    # no quote_t_utc for a timestamp the live rule could not parse
    tu = (None if c['reason'] == 'stale' and c['age_s'] is None
          else _to_utc(ts))
    return {
        'poll': poll_idx, 'symbol': symbol,
        'sampled_utc': now.isoformat(),
        'fetch_s': round(fetch_s, 4),
        'bid': None if bid is None else float(bid),
        'ask': None if ask is None else float(ask),
        'quote_t_utc': None if tu is None else tu.isoformat(),
        'error': None if exc is None else f"{type(exc).__name__}: {exc}"[:200],
        **c,
        'live_none': live is None,
    }


def sample(symbols, interval_s, duration_s):
    """Poll every ``interval_s`` for ``duration_s``. Lazy SDK import."""
    from trading_utils import get_api          # dotenv + SDK, Jetson-only
    from order_utils import get_crypto_quote
    api = get_api()
    records = []
    deadline = time.monotonic() + duration_s
    poll = 0
    try:
        while True:
            start = time.monotonic()
            for sym in symbols:
                records.append(sample_once(api, sym, poll, get_crypto_quote))
            nones = [r['symbol'] for r in records[-len(symbols):]
                     if r['reason'] != 'ok']
            print(f"[STALE-CENSUS] poll {poll} "
                  f"{datetime.datetime.now(datetime.timezone.utc):%H:%M:%SZ} "
                  f"None: {nones or '-'}", flush=True)
            poll += 1
            nxt = start + interval_s
            if nxt >= deadline:
                break
            time.sleep(max(0.0, nxt - time.monotonic()))
    except KeyboardInterrupt:
        print("[STALE-CENSUS] interrupted — summarising partial sample")
    return records


def _parse_thresholds(s):
    return tuple(float(x) for x in s.split(',') if x.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Crypto latest-quote staleness census (read-only)")
    ap.add_argument('--symbols', default=','.join(DEFAULT_SYMBOLS),
                    help="comma-separated pairs (default: the six crypto names)")
    ap.add_argument('--interval', type=float, default=30.0,
                    help="seconds between polls (default 30 = LOOP_INTERVAL)")
    ap.add_argument('--duration', type=float, default=900.0,
                    help="sampling window in seconds (default 900)")
    ap.add_argument('--thresholds', default=','.join(
        str(int(t)) for t in DEFAULT_THRESHOLDS),
        help="alternative max-age thresholds, seconds (default 180,300,600,900)")
    ap.add_argument('--json', default=None, metavar='OUT',
                    help="write records + table to OUT (nothing written otherwise)")
    ap.add_argument('--replay', default=None, metavar='FILE',
                    help="recompute the table from a saved --json file (no network)")
    args = ap.parse_args(argv)
    thresholds = _parse_thresholds(args.thresholds)

    if args.replay:
        with open(args.replay) as fh:
            saved = json.load(fh)
        meta = saved.get('meta', {})
        records = saved['records']
        symbols = meta.get('symbols')
        interval = meta.get('interval_s')
    else:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
        interval = args.interval
        # An outer `timeout`/SIGTERM ends sampling like Ctrl-C: the partial
        # sample is still summarised and written (only in live mode).
        def _term(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _term)
        started = datetime.datetime.now(datetime.timezone.utc)
        records = sample(symbols, args.interval, args.duration)
        meta = {'started_utc': started.isoformat(),
                'ended_utc': datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                'symbols': symbols, 'interval_s': args.interval,
                'duration_s': args.duration,
                'live_max_age_s': LIVE_MAX_AGE_S,
                'source': 'trading_utils.get_api().get_latest_crypto_quotes'
                          '([sym]) + order_utils.get_crypto_quote verdict'}

    table = build_table(records, thresholds, symbols)
    print(format_table(table, thresholds, interval))
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'meta': meta, 'thresholds_s': list(thresholds),
                       'table': table, 'records': records}, fh, indent=1)
        print(f"[STALE-CENSUS] wrote {args.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
