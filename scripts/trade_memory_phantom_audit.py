#!/usr/bin/env python3
"""Trade-memory phantom-exit audit (owner item #22) — DRY-RUN REPORT ONLY.

Research home: research/campaign_2026-09_jetson/research_engine.md, section
"R6 scout — E1 follow-up" Q1 (W19): an April-2026 buy -> false "[DESYNC]
Position gone at broker" -> re-buy loop journaled exits that the broker never
filled. Those rows sit in trade_memory.json as ``exit_reason='broker_stop'``
with no ``estimated`` flag, so trading_utils.compute_kelly_fraction counts
them (it drops only ``estimated`` rows, trading_utils.py:233-239).

What this does: for every exit record in trade_memory.json it looks for the
broker execution that should exist — an order of the exit side (``sell`` for
action 'sell', ``buy`` for 'cover') on the same symbol whose fill time is
within the match window of the record — and classifies:

  MATCHED    a same-side execution within +-match_min, one-to-one (each broker
             order backs at most one record), price within price_tol and, when
             the record carries ``qty`` (no row does today), qty within qty_tol.
  PHANTOM    no same-side execution of that symbol within +-ambig_min, and the
             record lies inside the broker-history coverage.
  AMBIGUOUS  everything in between (nearest execution between match_min and
             ambig_min, execution already claimed by a closer record, price/qty
             outside tolerance, or record older than the fetched history).

Output: per-record table, per-reason verdict counts, a Kelly-sample mirror
(before/after the proposed repair) and THE PROPOSED REPAIR — a JSON list of
record ids to set ``estimated=True``, printed or written to ``--out``. It is
NEVER applied: this script has no write mode for trade_memory.json (the file
is only ever opened for reading; ``_write_file`` refuses protected names).

Record id: ``"<symbol>|<ts>|<k>"`` — k = 0-based occurrence among records of
that symbol with that exact ts (records carry no id of their own). The repair
object also carries the sha256 of the trade_memory.json bytes audited, so an
owner-run applier can refuse if the file changed since the audit.

Modes:
  --replay CACHE  offline, no network (tests use this). CACHE holds
                  {"fills": [FILL activity...], "orders": [order...],
                   "coverage_start": iso, "fetched_at": iso}; a raw W19
                  ``ro_history.json`` ({"activities": [...], ...}) also works.
  --fetch         READ-ONLY GETs (/v2/account, /v2/account/activities/FILL,
                  /v2/orders?status=closed); paper host asserted; needs
                  ALPACA_API_KEY/_SECRET in the env or .env. ``--cache FILE``
                  saves the fetched bundle so reruns are offline.
  --json PATH     write the report dict ('-' = stdout).  --out PATH  write
                  only the proposed repair JSON. Nothing is written otherwise.

Examples (repo root)::

    python scripts/trade_memory_phantom_audit.py --fetch --cache /tmp/tm_cache.json
    python scripts/trade_memory_phantom_audit.py --replay /tmp/tm_cache.json --out /tmp/repair.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import datetime as dt
import hashlib
import json
import os
import statistics
import time

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TRADE_MEMORY = BASE_DIR / 'trade_memory.json'

# Reasons that CLAIM a broker-side fill (the only ones whose absence of a
# broker execution makes the row a phantom). 'broker_stop' is the legacy
# (pre-2026-08, never committed) name; today's code journals 'server_stop'
# (base_loop.py:926/:1450, stock_loop.py:1476).
DEFAULT_REASONS = ('broker_stop', 'server_stop')

# Reasons today's code ALWAYS journals with estimated=True (quote-midpoint
# exits): base_loop.py:880 circuit_breaker, :1147 stablecoin_flatten,
# :1634 desync, :3117 remote_flatten; stock_loop.py:850 external_close.
# Legacy rows with these reasons but no flag are reported (not repaired).
ESTIMATED_BY_DESIGN = ('circuit_breaker', 'stablecoin_flatten', 'desync',
                       'remote_flatten', 'external_close')

# Defaults, justified from the 2026-09-27 data (research_engine.md R6 + W21
# report): confirmed software exits sit within 0-4 min of their sell fill and
# the four 2026-03-31 broker_stop rows within 9.3 min; the 63 April/May
# phantom rows have no same-side fill within 5.3 days. Any match window in
# [10 min, 5 d] gives the same phantom set; see the 'sensitivity' block.
MATCH_MIN = 15.0
AMBIG_MIN = 24 * 60.0
PRICE_TOL = 0.05       # |fill vwap - record exit| / record exit
QTY_TOL = 0.02         # only when a record carries 'qty' (none do today)
SENSITIVITY_MATCH_MIN = (5.0, 15.0, 60.0, 240.0)
# strategy_config.KELLY_CAP (literal so this CLI imports no repo module;
# tests/test_trade_memory_phantom_audit.py pins it against strategy_config).
KELLY_CAP = 0.25

# Basenames this script must never write (it writes only --json/--out/--cache).
PROTECTED_NAMES = ('trade_memory.json', 'position_state.json')


# ---------------------------------------------------------------- pure core

def parse_ts(s):
    """RFC-3339 / ISO string (Z or +00:00, 0-9 fractional digits) -> aware UTC."""
    if s is None:
        return None
    s = str(s).strip()
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    tz = ''
    for sep in ('+', '-'):
        j = s.rfind(sep)
        if j > 10:
            s, tz = s[:j], s[j:]
            break
    if '.' in s:
        head, frac = s.split('.', 1)
        s = head + '.' + (frac + '000000')[:6]
    d = dt.datetime.fromisoformat(s + tz)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def norm_sym(s):
    """'BTC/USD' / 'BTCUSD' -> 'BTCUSD' (positions/CFEE drop the slash)."""
    return str(s or '').replace('/', '').upper()


def exit_side(action):
    """The broker side that closes a journaled round trip."""
    return 'buy' if str(action) == 'cover' else 'sell'


def iter_records(data):
    """(rid, symbol, index, record) for every row, in file order."""
    out = []
    for sym, rows in data.items():
        seen = {}
        for i, r in enumerate(rows or []):
            ts = str(r.get('ts', ''))
            k = seen.get(ts, 0)
            seen[ts] = k + 1
            out.append((f'{sym}|{ts}|{k}', sym, i, r))
    return out


def fills_from_cache(cache):
    """FILL activities from a cache ('fills') or a raw W19 dump ('activities')."""
    if 'fills' in cache:
        return list(cache.get('fills') or [])
    return [a for a in cache.get('activities') or []
            if isinstance(a, dict) and a.get('activity_type') == 'FILL']


def build_executions(cache):
    """Broker executions per order: {order_id, sym, side, qty, vwap, times}.

    FILL activities are authoritative; a filled order with no FILL activity
    in the cache (older history) falls back to the order's own filled_at /
    filled_avg_price / filled_qty."""
    ex = {}
    for f in fills_from_cache(cache):
        try:
            oid = f.get('order_id') or f.get('id')
            q, p = float(f['qty']), float(f['price'])
            t = parse_ts(f['transaction_time'])
        except (KeyError, TypeError, ValueError):
            continue
        g = ex.setdefault(oid, {'order_id': oid, 'sym': norm_sym(f.get('symbol')),
                                'side': f.get('side'), 'qty': 0.0, 'notional': 0.0,
                                'times': [], 'source': 'fill'})
        g['qty'] += q
        g['notional'] += q * p
        g['times'].append(t)
    for o in cache.get('orders') or []:
        if not isinstance(o, dict) or o.get('id') in ex:
            continue
        try:
            fq = float(o.get('filled_qty') or 0)
            fp = float(o.get('filled_avg_price') or 0)
            t = parse_ts(o.get('filled_at'))
        except (TypeError, ValueError):
            continue
        if fq > 0 and fp > 0 and t is not None:
            ex[o['id']] = {'order_id': o['id'], 'sym': norm_sym(o.get('symbol')),
                           'side': o.get('side'), 'qty': fq, 'notional': fq * fp,
                           'times': [t], 'source': 'order'}
    out = []
    for g in ex.values():
        g['vwap'] = g['notional'] / g['qty'] if g['qty'] > 0 else None
        g['times'].sort()
        out.append(g)
    return out


def _dt_min(exe, t):
    """Signed minutes from record time t to the execution's closest fill."""
    best = min(exe['times'], key=lambda x: abs((x - t).total_seconds()))
    return (best - t).total_seconds() / 60.0


def ledger_qty_at(executions, sym, t):
    """Net filled qty (buys - sells) of sym with fill time <= t. Ignores
    in-kind CFEE and the broker's own drift (research_engine.md R6 Q1.2)."""
    n = 0.0
    for e in executions:
        if e['sym'] != sym or not e['vwap']:
            continue
        per = e['qty'] / len(e['times'])
        s = 1.0 if e['side'] == 'buy' else -1.0
        n += s * per * sum(1 for x in e['times'] if x <= t)
    return n


def classify(data, executions, match_min=MATCH_MIN, ambig_min=AMBIG_MIN,
             price_tol=PRICE_TOL, qty_tol=QTY_TOL, coverage_start=None):
    """Classify every exit record. Returns a list of row dicts (file order).

    One-to-one: candidate (record, execution) pairs within +-ambig_min on the
    same symbol and exit side are assigned greedily by |dt| (closest first);
    an execution backs at most one record."""
    recs = iter_records(data)
    by_key = {}
    for e in executions:
        by_key.setdefault((e['sym'], e['side']), []).append(e)
    pairs = []
    rows = []
    for rid, sym, idx, r in recs:
        t = parse_ts(r.get('ts'))
        row = {'id': rid, 'symbol': sym, 'index': idx, 'ts': r.get('ts'),
               'action': r.get('action'), 'reason': r.get('exit_reason') or '',
               'exit': r.get('exit'), 'entry': r.get('entry'), 'pnl_pct': r.get('pnl_pct'),
               'qty': r.get('qty'), 'estimated': r.get('estimated'),
               'estimated_key_present': 'estimated' in r, '_t': t}
        cands = by_key.get((norm_sym(sym), exit_side(r.get('action'))), [])
        if t is not None and cands:
            near = min(cands, key=lambda e: abs(_dt_min(e, t)))
            row['nearest_dt_min'] = _dt_min(near, t)
            row['nearest_order_id'] = near['order_id']
            for e in cands:
                d = _dt_min(e, t)
                if abs(d) <= ambig_min:
                    pairs.append((abs(d), len(rows), e['order_id'], d, e))
        else:
            row['nearest_dt_min'] = None
            row['nearest_order_id'] = None
        rows.append(row)
    pairs.sort(key=lambda x: (x[0], x[1], x[2]))
    taken_r, taken_e = set(), set()
    for _, ri, oid, d, e in pairs:
        if ri in taken_r or oid in taken_e:
            continue
        taken_r.add(ri)
        taken_e.add(oid)
        rows[ri].update(match_order_id=oid, match_dt_min=d, match_vwap=e['vwap'],
                        match_qty=e['qty'], match_source=e['source'])
    cov = parse_ts(coverage_start) if isinstance(coverage_start, str) else coverage_start
    for row in rows:
        t = row.pop('_t')
        if t is None:
            row.update(verdict='AMBIGUOUS', why='unparseable_ts')
            continue
        if 'match_order_id' in row:
            why = []
            if abs(row['match_dt_min']) > match_min:
                why.append('dt_outside_match_window')
            try:
                px = float(row['exit'])
                if px > 0 and abs(row['match_vwap'] - px) / px > price_tol:
                    why.append('price_outside_tol')
            except (TypeError, ValueError):
                why.append('no_record_price')
            if row.get('qty') is not None:
                try:
                    q = float(row['qty'])
                    if q <= 0 or abs(row['match_qty'] - q) / q > qty_tol:
                        why.append('qty_outside_tol')
                except (TypeError, ValueError):
                    why.append('bad_record_qty')
            row.update(verdict='AMBIGUOUS' if why else 'MATCHED', why=','.join(why))
            continue
        nd = row.get('nearest_dt_min')
        if nd is not None and abs(nd) <= ambig_min:
            row.update(verdict='AMBIGUOUS', why='fill_claimed_by_closer_record')
        elif cov is not None and t < cov + dt.timedelta(minutes=ambig_min):
            row.update(verdict='AMBIGUOUS', why='before_history_coverage')
        else:
            row.update(verdict='PHANTOM', why='no_fill_within_ambig_window')
    return rows


def kelly_mirror(data, asset_type=None, min_trades=50):
    """Mirror of trading_utils.compute_kelly_fraction (selection :233-239,
    recency :244-245, shrinkage :267-275). Returns (fraction|None, sample)
    where sample = the <=200 most recent admissible records Kelly would use
    (all admissible records when there are fewer than min_trades). Pinned to
    the real function by tests/test_trade_memory_phantom_audit.py."""
    all_trades = []
    for symbol, trades in data.items():
        is_crypto = '/' in symbol
        if asset_type == 'crypto' and not is_crypto:
            continue
        if asset_type == 'stock' and is_crypto:
            continue
        all_trades.extend(t for t in trades if not t.get('estimated'))
    if len(all_trades) < min_trades:
        return None, all_trades
    all_trades.sort(key=lambda t: t.get('ts', ''))
    recent = all_trades[-200:]
    wins = [t for t in recent if t.get('pnl_pct', 0) > 0]
    losses = [t for t in recent if t.get('pnl_pct', 0) < 0]
    if not wins or not losses:
        return None, recent
    win_rate = len(wins) / len(recent)
    avg_win = statistics.fmean(t['pnl_pct'] for t in wins)
    avg_loss = abs(statistics.fmean(t['pnl_pct'] for t in losses))
    if avg_loss == 0:
        return None, recent
    n = len(recent)
    prior_n = 50
    win_rate = (win_rate * n + 0.5 * prior_n) / (n + prior_n)
    wlr = (avg_win / avg_loss * n + 1.0 * prior_n) / (n + prior_n)
    kelly_f = (win_rate * wlr - (1 - win_rate)) / wlr
    return max(0.05, min(0.25, kelly_f / 2)), recent


def kelly_mult(f, kelly_cap=KELLY_CAP):
    """base_loop.py:2586-2591: None -> 1.0x; else clamp(min(f, cap)/0.125, 0.5, 1.5)."""
    if f is None:
        return 1.0
    return max(0.5, min(1.5, min(f, kelly_cap) / 0.125))


def _apply_repair_copy(data, ids):
    """A DEEP COPY of data with estimated=True on ids — for the Kelly 'after'
    column only. The input dict and the file are never modified."""
    cp = json.loads(json.dumps(data))
    want = set(ids)
    for rid, sym, idx, _ in iter_records(cp):
        if rid in want:
            cp[sym][idx]['estimated'] = True
    return cp


def _kelly_block(data, repaired, ids):
    idset = set(ids)
    rid_of = {id(r): rid for rid, _, _, r in iter_records(data)}
    out = {}
    for book in ('crypto', 'stock', None):
        f0, s0 = kelly_mirror(data, book)
        f1, s1 = kelly_mirror(repaired, book)
        out[book or 'pooled'] = {
            'n_admissible_now': len(s0),
            'n_phantom_in_sample_now': sum(1 for r in s0 if rid_of.get(id(r)) in idset),
            'kelly_now': f0, 'kelly_after_repair': f1,
            'kelly_mult_now': kelly_mult(f0), 'kelly_mult_after_repair': kelly_mult(f1),
            'n_sample_after_repair': len(s1)}
    return out


def build_report(data, cache, reasons=DEFAULT_REASONS, match_min=MATCH_MIN,
                 ambig_min=AMBIG_MIN, price_tol=PRICE_TOL, qty_tol=QTY_TOL,
                 source_sha256=None, sensitivity=SENSITIVITY_MATCH_MIN):
    execs = build_executions(cache)
    cov = cache.get('coverage_start')
    if not cov and execs:
        cov = min(min(e['times']) for e in execs)
    rows = classify(data, execs, match_min, ambig_min, price_tol, qty_tol, cov)
    for r in rows:
        if r['verdict'] == 'PHANTOM':
            q = ledger_qty_at(execs, norm_sym(r['symbol']), parse_ts(r['ts']))
            try:
                r['ledger_notional_est'] = max(q, 0.0) * float(r['exit'])
            except (TypeError, ValueError):
                r['ledger_notional_est'] = None
    reasons = tuple(reasons)
    by_reason = {}
    for r in rows:
        d = by_reason.setdefault(r['reason'] or '(none)',
                                 {'n': 0, 'MATCHED': 0, 'PHANTOM': 0, 'AMBIGUOUS': 0})
        d['n'] += 1
        d[r['verdict']] += 1
    scope = [r for r in rows if r['reason'] in reasons]
    phantoms = [r for r in scope if r['verdict'] == 'PHANTOM']
    repair = [r for r in phantoms if not r.get('estimated')]
    ids = [r['id'] for r in repair]
    pnl = [float(r['pnl_pct']) for r in phantoms if r.get('pnl_pct') is not None]
    notional = [r['ledger_notional_est'] for r in phantoms if r.get('ledger_notional_est')]
    sens = {}
    for m in sensitivity:
        rs = [x for x in classify(data, execs, m, max(ambig_min, m), price_tol, qty_tol, cov)
              if x['reason'] in reasons]
        sens[str(m)] = {v: sum(1 for x in rs if x['verdict'] == v)
                        for v in ('MATCHED', 'PHANTOM', 'AMBIGUOUS')}
    legacy_unflagged = {}
    for r in rows:
        if r['reason'] in ESTIMATED_BY_DESIGN and not r.get('estimated'):
            legacy_unflagged[r['reason']] = legacy_unflagged.get(r['reason'], 0) + 1
    rep = {
        'params': {'reasons': list(reasons), 'match_min': match_min, 'ambig_min': ambig_min,
                   'price_tol': price_tol, 'qty_tol': qty_tol},
        'coverage_start': cov.isoformat() if isinstance(cov, dt.datetime) else cov,
        'fetched_at': cache.get('fetched_at'),
        'n_records': len(rows), 'n_executions': len(execs),
        'n_records_estimated_key_absent': sum(1 for r in rows if not r['estimated_key_present']),
        'n_records_estimated_true': sum(1 for r in rows if r.get('estimated')),
        'by_reason': by_reason,
        'in_scope': {'n': len(scope), 'MATCHED': sum(r['verdict'] == 'MATCHED' for r in scope),
                     'PHANTOM': len(phantoms),
                     'AMBIGUOUS': sum(r['verdict'] == 'AMBIGUOUS' for r in scope),
                     'phantom_first_ts': min((r['ts'] for r in phantoms), default=None),
                     'phantom_last_ts': max((r['ts'] for r in phantoms), default=None),
                     'phantom_estimated_set': sum(1 for r in phantoms if r.get('estimated')),
                     'phantom_pnl_pct_mean': statistics.fmean(pnl) if pnl else None,
                     'phantom_wins': sum(1 for x in pnl if x > 0),
                     'phantom_losses': sum(1 for x in pnl if x < 0),
                     'phantom_ledger_notional_est_sum': sum(notional) if notional else None,
                     'phantom_min_abs_nearest_dt_min': min(
                         (abs(r['nearest_dt_min']) for r in phantoms
                          if r.get('nearest_dt_min') is not None), default=None)},
        'legacy_unflagged_estimated_by_design': legacy_unflagged,
        'sensitivity_match_min': sens,
        'kelly': None,
        'rows': rows,
        'repair': {'action': 'set estimated=True (NOT APPLIED — owner decision #22)',
                   'file': 'trade_memory.json', 'source_sha256': source_sha256,
                   'n': len(ids), 'ids': ids,
                   'records': [{'id': r['id'], 'symbol': r['symbol'], 'index': r['index'],
                                'ts': r['ts'], 'exit_reason': r['reason'], 'exit': r['exit'],
                                'pnl_pct': r['pnl_pct'], 'patch': {'estimated': True}}
                               for r in repair]},
    }
    rep['kelly'] = _kelly_block(data, _apply_repair_copy(data, ids), ids)
    return rep


# ---------------------------------------------------------------- I/O

def _write_file(path, text):
    """The ONLY write sink. Refuses the live state files by basename."""
    if os.path.basename(str(path)) in PROTECTED_NAMES:
        raise SystemExit(f'refusing to write protected file {path}')
    with open(path, 'w') as fh:
        fh.write(text)


def read_trade_memory(path):
    with open(path, 'rb') as fh:
        raw = fh.read()
    return json.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


def _alpaca_get_factory():  # pragma: no cover - network
    """READ-ONLY GET helper (stdlib urllib). Paper host asserted."""
    import urllib.parse
    import urllib.request
    env = dict(os.environ)
    envf = BASE_DIR / '.env'
    if envf.exists() and not env.get('ALPACA_API_KEY'):
        for line in envf.read_text().splitlines():
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    base = env.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    base = base[:-3] if base.endswith('/v2') else base
    if 'paper' not in base:
        raise SystemExit(f'refusing non-paper host {base}')
    hdr = {'APCA-API-KEY-ID': env['ALPACA_API_KEY'],
           'APCA-API-SECRET-KEY': env['ALPACA_API_SECRET']}

    def get(path, params=None):
        url = base + path + ('?' + urllib.parse.urlencode(params) if params else '')
        req = urllib.request.Request(url, headers=hdr, method='GET')
        with urllib.request.urlopen(req, timeout=20) as r:
            body = json.loads(r.read())
        time.sleep(0.2)
        return body
    return get


def fetch_cache(since_iso):  # pragma: no cover - network, read-only GETs
    """READ-ONLY: account, every FILL activity, closed orders since since_iso."""
    get = _alpaca_get_factory()
    acct = get('/v2/account')
    fills, token = [], None
    for _ in range(500):
        p = {'page_size': 100, 'direction': 'desc'}
        if token:
            p['page_token'] = token
        page = get('/v2/account/activities/FILL', p)
        if not page:
            break
        fills += page
        if len(page) < 100:
            break
        token = page[-1]['id']
    orders, seen = [], set()
    t = parse_ts(since_iso)
    end = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)
    while t < end:
        u = t + dt.timedelta(days=3)
        page = get('/v2/orders', {'status': 'closed', 'limit': 500, 'direction': 'desc',
                                  'after': t.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                  'until': u.strftime('%Y-%m-%dT%H:%M:%SZ')})
        for o in page:
            if o['id'] not in seen:
                seen.add(o['id'])
                orders.append(o)
        t = u
    return {'fills': fills, 'orders': orders,
            'coverage_start': acct.get('created_at'),
            'fetched_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}


def _f(x, fmt='{:.1f}'):
    return 'n/a' if x is None else fmt.format(x)


def print_report(rep, verbose=False):
    p = rep['params']
    print(f"trade_memory records: {rep['n_records']}  broker executions: {rep['n_executions']}  "
          f"coverage from {rep['coverage_start']}  (fetched {rep['fetched_at']})")
    print(f"params: reasons={p['reasons']} match=±{p['match_min']:g} min "
          f"ambig=±{p['ambig_min']:g} min price_tol={p['price_tol']:.0%}")
    print(f"'estimated' key absent on {rep['n_records_estimated_key_absent']} rows, "
          f"True on {rep['n_records_estimated_true']}")
    print('\nverdicts by exit_reason:')
    for k, v in sorted(rep['by_reason'].items()):
        print(f"  {k:20} n={v['n']:4d}  MATCHED {v['MATCHED']:4d}  PHANTOM {v['PHANTOM']:4d}  "
              f"AMBIGUOUS {v['AMBIGUOUS']:4d}")
    rows = rep['rows'] if verbose else [r for r in rep['rows'] if r['reason'] in p['reasons']]
    print(f"\n{'ts':25} {'symbol':9} {'reason':12} {'qty':>6} {'exit':>12} {'est':>5} "
          f"{'verdict':9} {'nearest_dt_min':>14}  why")
    for r in rows:
        print(f"{r['ts']:25} {r['symbol']:9} {r['reason']:12} {_f(r.get('qty'), '{:g}'):>6} "
              f"{_f(r.get('exit'), '{:g}'):>12} {str(bool(r.get('estimated'))):>5} "
              f"{r['verdict']:9} {_f(r.get('nearest_dt_min')):>14}  {r.get('why', '')}")
    s = rep['in_scope']
    print(f"\nIN SCOPE ({','.join(p['reasons'])}): n={s['n']} MATCHED={s['MATCHED']} "
          f"PHANTOM={s['PHANTOM']} AMBIGUOUS={s['AMBIGUOUS']}")
    print(f"  phantoms {s['phantom_first_ts']} .. {s['phantom_last_ts']}; estimated set on "
          f"{s['phantom_estimated_set']}; pnl mean {_f(s['phantom_pnl_pct_mean'], '{:.2f}')}% "
          f"({s['phantom_wins']} wins / {s['phantom_losses']} losses); ledger notional est "
          f"${_f(s['phantom_ledger_notional_est_sum'], '{:,.0f}')}; closest same-side fill "
          f"{_f(s['phantom_min_abs_nearest_dt_min'], '{:,.0f}')} min")
    print(f"  sensitivity (match_min -> counts): {rep['sensitivity_match_min']}")
    if rep['legacy_unflagged_estimated_by_design']:
        print(f"  legacy rows of always-estimated reasons lacking the flag (not in repair): "
              f"{rep['legacy_unflagged_estimated_by_design']}")
    print('\nKelly mirror (trading_utils.compute_kelly_fraction; drops only estimated rows, :239):')
    for book, k in rep['kelly'].items():
        print(f"  {book:7} sample now {k['n_admissible_now']:4d} (phantoms in it "
              f"{k['n_phantom_in_sample_now']:3d}) kelly now {_f(k['kelly_now'], '{:.4f}')} -> "
              f"after repair {_f(k['kelly_after_repair'], '{:.4f}')} (sample "
              f"{k['n_sample_after_repair']}); sizing mult {k['kelly_mult_now']:.2f}x -> "
              f"{k['kelly_mult_after_repair']:.2f}x")
    print(f"\nPROPOSED REPAIR (NOT APPLIED): {rep['repair']['n']} ids -> estimated=True; "
          f"trade_memory sha256 {rep['repair']['source_sha256']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--replay', metavar='CACHE', help='offline: audit against a saved cache JSON')
    src.add_argument('--fetch', action='store_true', help='read-only GETs (paper host)')
    ap.add_argument('--cache', metavar='FILE', help='--fetch: save the fetched bundle here')
    ap.add_argument('--trade-memory', default=str(DEFAULT_TRADE_MEMORY),
                    help='trade_memory.json to audit (opened read-only)')
    ap.add_argument('--reasons', default=','.join(DEFAULT_REASONS),
                    help='exit reasons that claim a broker fill (default broker_stop,server_stop)')
    ap.add_argument('--match-min', type=float, default=MATCH_MIN,
                    help=f'MATCHED window ± minutes (default {MATCH_MIN:g})')
    ap.add_argument('--ambig-min', type=float, default=AMBIG_MIN,
                    help=f'PHANTOM needs no fill within ± minutes (default {AMBIG_MIN:g})')
    ap.add_argument('--price-tol', type=float, default=PRICE_TOL)
    ap.add_argument('--qty-tol', type=float, default=QTY_TOL)
    ap.add_argument('--verbose', action='store_true', help='print every record, not only in-scope')
    ap.add_argument('--json', metavar='PATH', help="write the report JSON ('-' = stdout)")
    ap.add_argument('--out', metavar='PATH', help='write the proposed repair JSON (never applied)')
    a = ap.parse_args(argv)
    data, sha = read_trade_memory(a.trade_memory)
    if a.replay:
        with open(a.replay) as fh:
            cache = json.load(fh)
    else:
        first = min((str(r.get('ts')) for _, _, _, r in iter_records(data)), default=None)
        since = (parse_ts(first) - dt.timedelta(days=2)).isoformat() if first else \
            '2026-01-01T00:00:00+00:00'
        cache = fetch_cache(since)
        if a.cache:
            _write_file(a.cache, json.dumps(cache))
    rep = build_report(data, cache, reasons=[x for x in a.reasons.split(',') if x],
                       match_min=a.match_min, ambig_min=a.ambig_min, price_tol=a.price_tol,
                       qty_tol=a.qty_tol, source_sha256=sha)
    if a.json == '-':
        json.dump(rep, sys.stdout, indent=1, default=str)
        print()
    else:
        print_report(rep, verbose=a.verbose)
        if a.json:
            _write_file(a.json, json.dumps(rep, indent=1, default=str))
    if a.out:
        _write_file(a.out, json.dumps(rep['repair'], indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
