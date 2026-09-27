#!/usr/bin/env python3
"""Paper-quirk census (experiment X12) — measurement-only, read-only, one snapshot per run.

Research home: research/campaign_2026-09_jetson/research_engine.md, section
"R6 scout — E1 follow-up" Q4 (W19). The Alpaca paper ledger has shown three
quirks on this account: quantities that drift from the fill ledger, positions
that lost their cost basis (``avg_entry_price``/``cost_basis`` = "0"), and all
six crypto positions sitting on an ``asset_id`` none of our orders used (the
S6 "two CUSIPs for one asset" pattern; owner item #23). This CLI takes ONE
snapshot and emits ONE jsonl row with these detectors:

  V  vanish       a symbol held in the last snapshot (or tracked in a
                  *position_state.json) is absent at the broker with no sell
                  FILL since the last snapshot; stays flagged while absent.
  B  basis        qty > 0 with avg_entry_price or cost_basis <= 0 / null.
                  B_new = B on a symbol that was not B in the last snapshot.
  Q  qty drift    |dqty - (sum buy FILL - sum sell FILL + sum in-kind CFEE)|
                  since the last snapshot > max(1e-8, 1e-6 * qty); also in $.
  E  equity~cash  positions exist but equity - cash < 5 % of sum(qty * price).
  A  asset split  position asset_id not among the asset_ids of our open +
                  recent orders for that symbol; per symbol the table shows the
                  position's asset_id vs the asset_id of each resting stop that
                  reserves it, and the current /v2/assets id (READ-ONLY #23 check).
  R  reserve      qty_available < qty with no open sell order behind it, or
                  open sell qty > position qty.

Nothing is scheduled here (gotcha #5): the owner runs it (e.g. every 900 s)
and passes ``--last`` so detectors that need a previous row can fire. No
notification is sent; ``--evaluate DIR`` applies the pre-registered alert
rule (``--alert-rules`` prints it) offline across saved rows.

Read-only: GET /v2/account, /v2/positions, /v2/orders (open + recent),
/v2/assets/{id|sym}, /v2/account/activities (FILL,CFEE since the last row);
paper host asserted; position_state files are opened for reading only.
Nothing is written unless ``--out FILE`` (append the row) or ``--cache FILE``
(the raw fetched bundle, replayable with ``--replay``) is given. Imports on
the dev Mac without alpaca/dotenv (stdlib urllib, lazily in the fetch).

Examples (repo root)::

    python scripts/paper_quirk_census.py --fetch --last logs/paper_census.jsonl --out logs/paper_census.jsonl
    python scripts/paper_quirk_census.py --replay bundle.json --json -
    python scripts/paper_quirk_census.py --evaluate logs/            # offline rule
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import datetime as dt
import glob
import json
import os
import time

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA = 1
RECENT_DAYS = 30
E_FRAC = 0.05              # equity - cash < 5 % of long market value
Q_ALERT_USD = 50.0         # single-row qty-drift alert threshold
X12_DAYS = 30              # evaluation horizon of the experiment
STOP_TYPES = ('stop', 'stop_limit', 'trailing_stop')
OPEN_STATUSES = ('new', 'accepted', 'pending_new', 'partially_filled', 'held',
                 'pending_replace', 'pending_cancel', 'accepted_for_bidding',
                 'calculated', 'stopped', 'suspended')      # still resting / reserving
PROTECTED_NAMES = ('trade_memory.json', 'position_state.json')

ALERT_RULES = """X12 paper-quirk census — pre-registered alert rule (research_engine.md R6 Q4)
  ALERT when, on two CONSECUTIVE snapshot rows (adjacent in time order):
    V  the same symbol is flagged vanished on both rows;
    Q  the same symbol shows unexplained qty drift on both rows;
    E  equity ~ cash (equity - cash < 5 % of long market value) on both rows;
    B  a NEW basis loss (B_new on row k: basis <= 0 on a symbol that had a basis,
       or a newly held symbol without one) is still there on row k+1.
       A basis already missing on the FIRST row of the series is a standing
       state (6/6 crypto on 2026-09-27) and is log-only, like A.
  ALERT on a SINGLE row when Q drift |residual qty| * price >= $50.
  LOG-ONLY (never alerts): A asset-id split (standing 6/6 on 2026-09-27),
    R reservation anomalies, standing B.
  30-day verdict: any V or E event -> owner item for a runtime guard (H1/H2);
    no V/B_new/Q/E event -> close X12 (keep only X3).
  Standing check: the first real exit of an A-flagged name must take the broker
    qty on the OLD asset_id to 0 (compare the next row); otherwise a venue item.
  This CLI never notifies and never schedules; the owner runs it."""


# ---------------------------------------------------------------- pure core

def parse_ts(s):
    """RFC-3339 / ISO string -> aware UTC datetime (None on falsy)."""
    if not s:
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
    return str(s or '').replace('/', '').upper()


def _num(x):
    """Alpaca decimal strings -> float; None/'' / unparseable -> None."""
    if x is None or x == '':
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def trim_position(p):
    return {'symbol': p.get('symbol'), 'asset_id': p.get('asset_id'),
            'qty': _num(p.get('qty')), 'qty_available': _num(p.get('qty_available')),
            'avg_entry_price': _num(p.get('avg_entry_price')),
            'cost_basis': _num(p.get('cost_basis')),
            'current_price': _num(p.get('current_price')),
            'market_value': _num(p.get('market_value')),
            'avg_entry_price_raw': p.get('avg_entry_price')}


def trim_order(o):
    return {'id': o.get('id'), 'symbol': o.get('symbol'), 'asset_id': o.get('asset_id'),
            'type': o.get('type') or o.get('order_type'), 'side': o.get('side'),
            'qty': _num(o.get('qty')), 'filled_qty': _num(o.get('filled_qty')) or 0.0,
            'stop_price': _num(o.get('stop_price')), 'limit_price': _num(o.get('limit_price')),
            'status': o.get('status'), 'submitted_at': o.get('submitted_at')}


def tracked_symbols(states):
    """Symbols the bots track, from *position_state.json 'hwm' keys
    (base_loop._save_position_state writes one hwm per held position)."""
    out = set()
    for st in states or []:
        if isinstance(st, dict):
            out |= {norm_sym(s) for s in (st.get('hwm') or {})}
    return out


def ledger_since(activities, since):
    """Per symbol: {'buy', 'sell', 'cfee'} qty from FILL / in-kind CFEE after since."""
    led = {}
    for a in activities or []:
        typ = a.get('activity_type')
        t = parse_ts(a.get('transaction_time') or a.get('created_at'))
        if since is not None and (t is None or t <= since):
            continue
        sym = norm_sym(a.get('symbol'))
        q = _num(a.get('qty'))
        if not sym or q is None:
            continue
        g = led.setdefault(sym, {'buy': 0.0, 'sell': 0.0, 'cfee': 0.0})
        if typ == 'FILL':
            if a.get('side') == 'buy':
                g['buy'] += q
            else:
                g['sell'] += q
        elif typ == 'CFEE':
            g['cfee'] += q              # in-kind fee qty is signed (negative)
    return led


def detect(bundle, prev_row=None, e_frac=E_FRAC):
    """Compute the snapshot row (dict) from a raw bundle and the last row."""
    now = bundle.get('now')
    acct = bundle.get('account') or {}
    pos = [trim_position(p) for p in bundle.get('positions') or []]
    opn = [trim_order(o) for o in bundle.get('open_orders') or []
           if (o.get('status') in OPEN_STATUSES)]
    rec = [trim_order(o) for o in bundle.get('recent_orders') or []]
    assets = bundle.get('assets') or {}
    pmap = {norm_sym(p['symbol']): p for p in pos if (p['qty'] or 0) != 0}
    prev_pos = {norm_sym(p['symbol']): p for p in (prev_row or {}).get('positions') or []
                if (p.get('qty') or 0) != 0}
    prev_det = (prev_row or {}).get('detectors') or {}
    since = parse_ts((prev_row or {}).get('ts'))
    led = ledger_since(bundle.get('activities'), since) if prev_row else {}
    tracked = tracked_symbols(bundle.get('position_state'))

    det = {}
    # V — vanish (vs last snapshot, carried while absent, vs position_state)
    v = []
    prev_v = {x['sym'] for x in prev_det.get('V') or []}
    for s in sorted((set(prev_pos) | prev_v) - set(pmap)):
        sold = led.get(s, {}).get('sell', 0.0)
        pq = (prev_pos.get(s) or {}).get('qty') or 0.0
        if s in prev_pos and pq > 0 and sold >= pq * (1 - 1e-6):
            continue                        # explained by our own sells
        if s not in prev_pos and sold > 0:
            continue                        # carried vanish closed by a sell
        src = ('last_snapshot' if s in prev_pos else
               'position_state' if s in tracked else 'carried')
        v.append({'sym': s, 'source': src,
                  'prev_qty': pq, 'sold_since': sold})
    for s in sorted(tracked - set(pmap) - {x['sym'] for x in v}):
        v.append({'sym': s, 'source': 'position_state', 'prev_qty': None, 'sold_since': None})
    det['V'] = v
    det['reappeared'] = sorted(prev_v & set(pmap))
    # B — basis lost
    b = []
    for s, p in sorted(pmap.items()):
        avg, cb = p['avg_entry_price'], p['cost_basis']
        if (p['qty'] or 0) > 0 and (avg is None or avg <= 0 or cb is None or cb <= 0):
            b.append({'sym': s, 'avg_entry_price': p['avg_entry_price_raw'], 'cost_basis': cb})
    det['B'] = b
    prev_b = {x['sym'] for x in prev_det.get('B') or []}
    det['B_new'] = sorted(x['sym'] for x in b if prev_row is not None and x['sym'] not in prev_b)
    # Q — qty drift vs last snapshot, net of our fills and in-kind fees
    q = []
    if prev_row is not None:
        for s in sorted(set(pmap) | set(prev_pos)):
            q1 = (pmap.get(s) or {}).get('qty') or 0.0
            q0 = (prev_pos.get(s) or {}).get('qty') or 0.0
            g = led.get(s, {'buy': 0.0, 'sell': 0.0, 'cfee': 0.0})
            expl = g['buy'] - g['sell'] + g['cfee']
            res = (q1 - q0) - expl
            if abs(res) > max(1e-8, 1e-6 * max(abs(q1), abs(q0))):
                px = ((pmap.get(s) or {}).get('current_price')
                      or (prev_pos.get(s) or {}).get('current_price') or 0.0)
                q.append({'sym': s, 'qty_prev': q0, 'qty_now': q1, 'explained': expl,
                          'residual': res, 'usd': abs(res) * px})
    det['Q'] = q
    # E — equity ~ cash while positions exist
    eq, cash = _num(acct.get('equity')), _num(acct.get('cash'))
    lmv = sum((p['qty'] or 0) * (p['current_price'] or 0) for p in pmap.values()
              if (p['qty'] or 0) > 0)
    e_fired = bool(pmap) and lmv > 0 and eq is not None and cash is not None \
        and (eq - cash) < e_frac * lmv
    det['E'] = {'fired': e_fired, 'equity': eq, 'cash': cash, 'long_mv': lmv,
                'gap_frac': ((eq - cash) / lmv) if (lmv > 0 and eq is not None
                                                     and cash is not None) else None}
    # A — asset-id split (+ the #23 per-symbol table)
    ids_by_sym = {}
    for o in opn + rec:
        ids_by_sym.setdefault(norm_sym(o['symbol']), set()).add(o['asset_id'])
    a_tab = []
    for s, p in sorted(pmap.items()):
        stops = [o for o in opn if norm_sym(o['symbol']) == s and o['side'] == 'sell'
                 and o['type'] in STOP_TYPES]
        ours = ids_by_sym.get(s, set())
        a_tab.append({
            'sym': s, 'position_asset_id': p['asset_id'],
            'current_asset_id': assets.get(s),
            'resting_stop_asset_ids': sorted({o['asset_id'] for o in stops}),
            'resting_stop_order_ids': [o['id'] for o in stops],
            'resting_stop_qty': sum((o['qty'] or 0) - (o['filled_qty'] or 0) for o in stops),
            'order_asset_ids': sorted(i for i in ours if i),
            'split': (p['asset_id'] not in ours) if ours else None,
            'stop_on_other_id': any(o['asset_id'] != p['asset_id'] for o in stops),
            'position_on_current_id': (p['asset_id'] == assets.get(s))
            if assets.get(s) else None})
    det['A'] = [r for r in a_tab if r['split']]
    det['A_table'] = a_tab
    # R — reservation anomalies
    r = []
    for s, p in sorted(pmap.items()):
        sells = [o for o in opn if norm_sym(o['symbol']) == s and o['side'] == 'sell']
        open_q = sum((o['qty'] or 0) - (o['filled_qty'] or 0) for o in sells)
        qty, qa = p['qty'] or 0.0, p['qty_available']
        tol = max(1e-8, 1e-6 * abs(qty))
        if qa is not None and qa < qty - tol and not sells:
            r.append({'sym': s, 'kind': 'reserve_without_order', 'qty': qty,
                      'qty_available': qa, 'open_sell_qty': 0.0})
        if open_q > qty + tol:
            r.append({'sym': s, 'kind': 'over_reserve', 'qty': qty,
                      'qty_available': qa, 'open_sell_qty': open_q})
    det['R'] = r
    fired = [k for k in ('V', 'B', 'Q', 'E', 'A', 'R')
             if (det[k]['fired'] if k == 'E' else det[k])]
    return {'schema': SCHEMA, 'ts': now,
            'prev_ts': (prev_row or {}).get('ts'),
            'account': {'equity': eq, 'cash': cash, 'last_equity': _num(acct.get('last_equity'))},
            'positions': pos, 'open_orders': opn,
            'tracked_symbols': sorted(tracked),
            'detectors': det, 'fired': fired}


def _keys(row, code):
    d = (row.get('detectors') or {})
    if code == 'E':
        return {'_account'} if (d.get('E') or {}).get('fired') else set()
    return {x['sym'] for x in d.get(code) or []}


def evaluate(rows, q_alert_usd=Q_ALERT_USD, x12_days=X12_DAYS):
    """Apply the pre-registered rule (ALERT_RULES) to saved rows, offline."""
    rows = sorted((r for r in rows if r.get('ts')), key=lambda r: parse_ts(r['ts']))
    alerts, counts = [], {k: 0 for k in ('V', 'B', 'B_new', 'Q', 'E', 'A', 'R')}
    log_only = {'A': {}, 'R': []}
    b_new_rows = []
    for i, r in enumerate(rows):
        b_now = _keys(r, 'B')
        b_new = set() if i == 0 else b_now - _keys(rows[i - 1], 'B')
        b_new_rows.append(b_new)
        for k in ('V', 'B', 'Q', 'E', 'A', 'R'):
            counts[k] += bool(_keys(r, k))
        counts['B_new'] += bool(b_new)
        for x in (r.get('detectors') or {}).get('Q') or []:
            if (x.get('usd') or 0) >= q_alert_usd:
                alerts.append({'ts': r['ts'], 'rule': 'Q_single_usd', 'keys': [x['sym']],
                               'usd': x['usd']})
        day = r['ts'][:10]
        if _keys(r, 'A'):
            log_only['A'].setdefault(day, sorted(_keys(r, 'A')))
        for x in (r.get('detectors') or {}).get('R') or []:
            log_only['R'].append({'ts': r['ts'], **x})
        if i == 0:
            continue
        p = rows[i - 1]
        gap = (parse_ts(r['ts']) - parse_ts(p['ts'])).total_seconds()
        for k in ('V', 'Q', 'E'):
            both = _keys(p, k) & _keys(r, k)
            if both:
                alerts.append({'ts': r['ts'], 'rule': f'{k}_consecutive',
                               'keys': sorted(both), 'gap_s': gap})
        both = b_new_rows[i - 1] & b_now
        if both:
            alerts.append({'ts': r['ts'], 'rule': 'B_new_consecutive',
                           'keys': sorted(both), 'gap_s': gap})
    span_d = ((parse_ts(rows[-1]['ts']) - parse_ts(rows[0]['ts'])).total_seconds() / 86400
              if len(rows) > 1 else 0.0)
    if span_d < x12_days:
        verdict = 'insufficient_span'
    elif counts['V'] or counts['E']:
        verdict = 'owner_guard_item'
    elif not (counts['B_new'] or counts['Q']):
        verdict = 'close_x12'
    else:
        verdict = 'review'
    return {'n_rows': len(rows), 'first_ts': rows[0]['ts'] if rows else None,
            'last_ts': rows[-1]['ts'] if rows else None, 'span_days': span_d,
            'rows_fired': counts, 'alerts': alerts, 'log_only': log_only,
            'x12_verdict': verdict}


# ---------------------------------------------------------------- I/O

def _write_file(path, text, append=False):
    """The ONLY write sink. Refuses the live state files by basename."""
    base = os.path.basename(str(path))
    if base in PROTECTED_NAMES or base.endswith('position_state.json'):
        raise SystemExit(f'refusing to write protected file {path}')
    with open(path, 'a' if append else 'w') as fh:
        fh.write(text)


def read_last_row(path):
    """Last non-empty JSON line of a jsonl file (or a single JSON object file)."""
    if not path or not os.path.exists(path):
        return None
    last = None
    with open(path) as fh:
        for line in fh:
            if line.strip():
                last = line
    if last is None:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        with open(path) as fh:
            return json.load(fh)


def read_rows(target):
    files = sorted(glob.glob(os.path.join(target, '*.jsonl'))) if os.path.isdir(target) \
        else [target]
    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return rows


def read_states(paths):
    out = []
    for p in paths:
        try:
            with open(p) as fh:
                out.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def default_state_paths():
    return sorted(glob.glob(str(BASE_DIR / '*position_state.json')))


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
    get.quote = urllib.parse.quote
    return get


def fetch_bundle(prev_row, recent_days=RECENT_DAYS):  # pragma: no cover - network
    """READ-ONLY snapshot: account, positions, open + recent orders, assets,
    FILL/CFEE activities since the last row."""
    get = _alpaca_get_factory()
    now = dt.datetime.now(dt.timezone.utc)
    acct = get('/v2/account')
    positions = get('/v2/positions')
    open_orders = get('/v2/orders', {'status': 'open', 'limit': 500, 'nested': 'true'})
    after = (now - dt.timedelta(days=recent_days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    recent = get('/v2/orders', {'status': 'all', 'limit': 500, 'direction': 'desc',
                                'after': after, 'nested': 'true'})
    assets = {}
    for p in positions:
        a = get('/v2/assets/' + p['asset_id'])
        sym = a.get('symbol') or p['symbol']
        cur = get('/v2/assets/' + get.quote(sym, safe=''))
        assets[norm_sym(p['symbol'])] = cur.get('id')
    acts = []
    if prev_row and prev_row.get('ts'):
        token = None
        for _ in range(100):
            q = {'activity_types': 'FILL,CFEE', 'after': prev_row['ts'],
                 'direction': 'asc', 'page_size': 100}
            if token:
                q['page_token'] = token
            page = get('/v2/account/activities', q)
            if not page:
                break
            acts += page
            if len(page) < 100:
                break
            token = page[-1]['id']
    return {'now': now.isoformat(timespec='seconds'), 'account': acct,
            'positions': positions, 'open_orders': open_orders, 'recent_orders': recent,
            'assets': assets, 'activities': acts}


def print_summary(row):
    d = row['detectors']
    a = row['account']
    print(f"snapshot {row['ts']} (prev {row['prev_ts']})  equity {a['equity']} cash {a['cash']}  "
          f"positions {len([p for p in row['positions'] if p['qty']])}  open orders "
          f"{len(row['open_orders'])}  tracked {row['tracked_symbols']}")
    print(f"fired: {row['fired'] or 'none'}")
    print(f"  V vanish: {d['V'] or '-'}  reappeared: {d['reappeared'] or '-'}")
    print(f"  B basis lost: {[x['sym'] for x in d['B']] or '-'}  new: {d['B_new'] or '-'}")
    print(f"  Q qty drift: {d['Q'] or ('-' if row['prev_ts'] else 'n/a (no --last)')}")
    e = d['E']
    print(f"  E equity~cash: {e['fired']} (equity-cash = {e['gap_frac'] if e['gap_frac'] is None else round(e['gap_frac'], 4)} x long mv {round(e['long_mv'], 2)})")
    print(f"  R reserve: {d['R'] or '-'}")
    print('  A asset-id table (#23):')
    print(f"    {'sym':9} {'position asset_id':38} {'resting stop asset_id(s)':38} "
          f"{'current /v2/assets id':38} split stop_other_id")
    for t in d['A_table']:
        print(f"    {t['sym']:9} {str(t['position_asset_id']):38} "
              f"{','.join(t['resting_stop_asset_ids']) or '-':38} {str(t['current_asset_id']):38} "
              f"{str(t['split']):5} {t['stop_on_other_id']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--fetch', action='store_true', help='one read-only snapshot (paper host)')
    src.add_argument('--replay', metavar='BUNDLE', help='offline: compute the row from a saved bundle')
    src.add_argument('--evaluate', metavar='DIR', help='offline: apply the alert rule to *.jsonl rows')
    src.add_argument('--alert-rules', action='store_true', help='print the pre-registered rule')
    ap.add_argument('--last', metavar='FILE', help='previous row (jsonl: last line is used)')
    ap.add_argument('--position-state', metavar='FILE', action='append',
                    help='position_state file(s), read-only (default: repo *position_state.json)')
    ap.add_argument('--recent-days', type=int, default=RECENT_DAYS,
                    help=f'recent-order window for the asset-id set (default {RECENT_DAYS})')
    ap.add_argument('--e-frac', type=float, default=E_FRAC)
    ap.add_argument('--q-alert-usd', type=float, default=Q_ALERT_USD)
    ap.add_argument('--cache', metavar='FILE', help='--fetch: save the raw bundle here')
    ap.add_argument('--out', metavar='FILE', help='append the row (one jsonl line) here')
    ap.add_argument('--json', metavar='PATH', help="write the row / evaluation JSON ('-' = stdout)")
    a = ap.parse_args(argv)
    if a.alert_rules:
        print(ALERT_RULES)
        return 0
    if a.evaluate:
        res = evaluate(read_rows(a.evaluate), q_alert_usd=a.q_alert_usd)
        if a.json == '-':
            print(json.dumps(res, default=str))
        else:
            print(f"rows {res['n_rows']} {res['first_ts']} .. {res['last_ts']} "
                  f"({res['span_days']:.2f} d)  fired {res['rows_fired']}")
            for x in res['alerts']:
                print(f"  ALERT {x}")
            print(f"  log-only A days: {len(res['log_only']['A'])}  R rows: "
                  f"{len(res['log_only']['R'])}  X12 verdict: {res['x12_verdict']}")
            if a.json:
                _write_file(a.json, json.dumps(res, indent=1, default=str))
        return 0
    prev = read_last_row(a.last)
    if a.replay:
        with open(a.replay) as fh:
            bundle = json.load(fh)
    else:
        bundle = fetch_bundle(prev, a.recent_days)
    if 'position_state' not in bundle:
        bundle['position_state'] = read_states(a.position_state or default_state_paths())
    if a.cache:
        _write_file(a.cache, json.dumps(bundle, default=str))
    row = detect(bundle, prev, e_frac=a.e_frac)
    line = json.dumps(row, default=str, separators=(',', ':'))
    if a.json == '-':
        print(line)
    else:
        print_summary(row)
        if a.json:
            _write_file(a.json, json.dumps(row, indent=1, default=str))
    if a.out:
        _write_file(a.out, line + '\n', append=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
