#!/usr/bin/env python3
"""summarize.py — end-of-window report for the harness-launched bots.

Reads monitor.jsonl + bots_stdout.log + the day's journals (+ pred_history,
llm_cost.json) and prints: peak RSS / max temps, cycles per book, non-null
predictions per book, gate decisions by reason, orders placed/filled/cancelled,
distinct Tracebacks (first 3 lines + exception line), LLM spend delta.

Stdlib only (runs under any python3); `--alpaca` adds a READ-ONLY order
query (needs the jetson env: source jenv.sh; $JPY summarize.py --alpaca).

Defaults point at the phase5 files; the time window starts at
phase5/bots_started.json's started_at (written by start_bots.sh).

Examples:
  python3 summarize.py
  python3 summarize.py --journal-date 2026-05-07 --since none \
      --log /home/kyle/trader/stock_bot_output.log --monitor /dev/null
"""
import argparse
import collections
import datetime as dt
import glob
import json
import os
import re
import statistics
import sys

REPO = '/home/kyle/trader'
PHASE5 = '/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/phase5'
PRED_HISTORY = {'crypto': 'pred_history.jsonl', 'stock': 'stock_pred_history.jsonl'}

CYCLE_RE = re.compile(r'--- CYCLE (\d+): ')
WAIT_RE = re.compile(r'\[WAIT\] Market closed')
SLEEP_RE = re.compile(r'\[SLEEP\] Next check')
PRED_LINE_RE = re.compile(r'^Predicted Return:\s*([-+]?\d)')
ORDER_PLACED_RE = re.compile(r'\[ORDER\] (\S+): (buy|sell) (?!LIMIT ERROR)\S+ @ ')
ORDER_ERR_RE = re.compile(r'\[ORDER\] (\S+): (buy|sell) LIMIT ERROR: (.*)'
                          r'|\[ORDER\] (\S+): (bracket) order error: (.*)'
                          r'|\[STOP\] (\S+): ()Sell error: (.*)')
FILLED_RE = re.compile(r'\[LIFECYCLE\] Order \S+ FILLED|\[LIFECYCLE\] Market fallback FILLED'
                       r'|\[MAKER\] \S+: rung \d+ filled|\[LIFECYCLE\] Order filled during cancel'
                       r'|\[LIFECYCLE\] Fully filled during cancel')
CANCEL_RE = re.compile(r'\[LIFECYCLE\] Order \S+ unfilled after .*canceling|\[CANCEL\] ')
GIVEUP_RE = re.compile(r'\[LIFECYCLE\] .*giving up|\[LIFECYCLE\] Giving up')
ERROR_LINE_RE = re.compile(r'^\S+ \S+ \[([^\]]+)\] (ERROR|CRITICAL): (.*)')
TB_RE = re.compile(r'Traceback \(most recent call last\)')
LGB_FATAL_RE = re.compile(r'\[LightGBM\] \[Fatal\] (.*)')
BANNER_RE = re.compile(r'^===== start_bots\.sh ')


def parse_ts(s):
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.astimezone()          # naive journal ts (pre-2026-08 rows) = local time
    return t


def book_of(row):
    b = row.get('book') or row.get('asset_type')
    if b:
        return b
    sym = row.get('symbol') or ''
    if sym:
        return 'crypto' if '/' in sym else 'stock'
    return '?'


def norm_reason(r):
    r = str(r or '?')
    return re.sub(r'\s*\(.*\)\s*$', '', r).strip() or '?'


def pct(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def hdr(t):
    print(f'\n== {t} ' + '=' * max(0, 70 - len(t)))


# --------------------------------------------------------------------------- sections
def load_monitor(path):
    recs = []
    if not path or not os.path.exists(path):
        return recs
    with open(path) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    return recs


def section_monitor(recs):
    hdr('Resources (monitor.jsonl)')
    if not recs:
        print('  no monitor records')
        return
    def col(k):
        return [r[k] for r in recs if isinstance(r.get(k), (int, float))]
    alive = sum(1 for r in recs if r.get('alive'))
    print(f"  ticks: {len(recs)}  ({recs[0].get('ts')} -> {recs[-1].get('ts')}); process alive on {alive}/{len(recs)}")
    for k, lab, fn in (('rss_mb', 'peak RSS MB', max), ('hwm_mb', 'peak VmHWM MB', max),
                       ('swap_mb', 'peak bot VmSwap MB', max),
                       ('mem_available_mb', 'min MemAvailable MB', min),
                       ('swap_used_mb', 'max system swap used MB', max),
                       ('gpu_temp_c', 'max GPU temp C', max), ('cpu_temp_c', 'max CPU temp C', max),
                       ('tj_temp_c', 'max Tj temp C', max), ('cpu_pct', 'max bot CPU % (1 core)', max),
                       ('sys_cpu_pct', 'max system CPU %', max), ('threads', 'max threads', max)):
        v = col(k)
        print(f"  {lab:<26}: {fn(v) if v else 'n/a'}")
    cp = col('cpu_pct')
    if cp:
        print(f"  {'mean bot CPU % (1 core)':<26}: {statistics.mean(cp):.1f}")
    la = [r['loadavg'][0] for r in recs if r.get('loadavg')]
    if la:
        print(f"  {'max 1-min load':<26}: {max(la)}")
    tot = lambda k: sum(r.get(k) or 0 for r in recs)  # noqa: E731
    print(f"  log lines / error lines / tracebacks seen live: {tot('log_new_lines')} / "
          f"{tot('log_err_lines')} / {tot('log_tracebacks')}")
    flags = collections.Counter(k for r in recs for k, v in (r.get('flags') or {}).items() if v)
    print(f"  control flags ever present (ticks): {dict(flags) or 'none'}")
    errs = collections.Counter(k for r in recs for k in r if k.endswith('_err'))
    if errs:
        print(f"  probe errors: {dict(errs)}")
    snaps = [(r['ts'], r['alpaca']) for r in recs if isinstance(r.get('alpaca'), dict)]
    good = [(t, a) for t, a in snaps if 'err' not in a]
    if good:
        (t0, a0), (t1, a1) = good[0], good[-1]
        print(f"  alpaca first {t0}: equity={a0.get('equity')} cash={a0.get('cash')} "
              f"pos={a0.get('n_positions')} open_orders={a0.get('n_open_orders')}")
        print(f"  alpaca last  {t1}: equity={a1.get('equity')} cash={a1.get('cash')} "
              f"pos={a1.get('n_positions')} open_orders={a1.get('n_open_orders')}")
        try:
            print(f"  equity delta: {a1['equity'] - a0['equity']:+.2f}")
        except (KeyError, TypeError):
            pass
    bad = [a['err'] for _, a in snaps if 'err' in a]
    if bad:
        print(f"  alpaca snapshot errors: {len(bad)} (last: {bad[-1][:120]})")


def scan_log(path, last_run):
    out = {'cycles': 0, 'max_cycle': None, 'wait': 0, 'sleep': 0, 'pred_lines': 0,
           'placed': collections.Counter(), 'order_err': collections.Counter(),
           'filled': 0, 'cancel': 0, 'giveup': 0, 'errors': collections.Counter(),
           'lgb_fatal': collections.Counter(), 'tbs': collections.OrderedDict(), 'lines': 0}
    if not path or not os.path.exists(path):
        out['missing'] = True
        return out
    start_off = 0
    if last_run:
        with open(path, 'rb') as f:
            off = 0
            for raw in f:
                if BANNER_RE.match(raw.decode('utf-8', 'replace')):
                    start_off = off
                off += len(raw)
    tb = None
    prev_err = None
    with open(path, 'rb') as f:
        f.seek(start_off)
        for raw in f:
            line = raw.decode('utf-8', 'replace').rstrip('\n')
            out['lines'] += 1
            if tb is not None:
                if line.startswith((' ', '\t')) or line.startswith('During handling') or not line.strip():
                    tb['body'].append(line)
                    continue
                # first non-indented line = exception line (or a chained 'Traceback')
                tb['exc'] = line
                files = [l for l in tb['body'] if l.lstrip().startswith('File ')]
                sig = (line.split(':')[0][:120], files[-1].strip() if files else '')
                ent = out['tbs'].setdefault(sig, {'n': 0, 'ctx': tb['ctx'],
                                                  'first3': ['Traceback (most recent call last):'] + tb['body'][:2],
                                                  'exc': line})
                ent['n'] += 1
                tb = None
                if TB_RE.search(line):
                    tb = {'ctx': None, 'body': []}
                    continue
            if TB_RE.search(line):
                tb = {'ctx': prev_err, 'body': []}
                continue
            m = CYCLE_RE.search(line)
            if m:
                out['cycles'] += 1
                n = int(m.group(1))
                out['max_cycle'] = n if out['max_cycle'] is None else max(out['max_cycle'], n)
            if WAIT_RE.search(line):
                out['wait'] += 1
            if SLEEP_RE.search(line):
                out['sleep'] += 1
            if PRED_LINE_RE.match(line):
                out['pred_lines'] += 1
            m = ORDER_PLACED_RE.search(line)
            if m:
                out['placed'][(('crypto' if '/' in m.group(1) else 'stock'), m.group(2))] += 1
            m = ORDER_ERR_RE.search(line)
            if m:
                g = [x for x in m.groups() if x is not None]
                out['order_err'][(g[1] or 'stop-sell', re.sub(r'\d+(\.\d+)?', '#', g[2])[:80])] += 1
            if FILLED_RE.search(line):
                out['filled'] += 1
            if CANCEL_RE.search(line):
                out['cancel'] += 1
            if GIVEUP_RE.search(line):
                out['giveup'] += 1
            m = LGB_FATAL_RE.search(line)
            if m:
                out['lgb_fatal'][m.group(1)[:100]] += 1
            m = ERROR_LINE_RE.match(line)
            if m:
                msg = re.sub(r'\d+(\.\d+)?', '#', m.group(3))[:110]
                out['errors'][f'[{m.group(1)}] {msg}'] += 1
                prev_err = line[:200]
            elif line.strip():
                prev_err = None
    return out


def load_journal_rows(files, since):
    rows = []
    for p in files:
        opener = open
        if p.endswith('.gz'):
            import gzip
            opener = gzip.open
        try:
            with opener(p, 'rt') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if since is not None:
                        t = parse_ts(r.get('ts'))
                        if t is None or t < since:
                            continue
                    rows.append(r)
        except OSError as e:
            print(f'  (cannot read {p}: {e})')
    return rows


def section_cycles(log, rows):
    hdr('Cycles per book')
    lat = collections.defaultdict(list)
    for r in rows:
        if r.get('action') == 'cycle_latency':
            try:
                lat[book_of(r)].append(float(r.get('total_s') or 0))
            except (TypeError, ValueError):
                pass
    if lat:
        for b, v in sorted(lat.items()):
            print(f"  {b:<7} journal cycle_latency rows: {len(v):>6}   total_s mean {statistics.mean(v):.1f}"
                  f"  p95 {pct(v, 0.95):.1f}  max {max(v):.1f}")
    else:
        print('  no cycle_latency journal rows (pre-2026-08 journals never wrote them)')
    if log.get('missing'):
        print('  log: not found')
    else:
        print(f"  log '--- CYCLE N' lines: {log['cycles']} (max cycle no {log['max_cycle']}) — "
              f"NOT attributable to a book in combined mode (both loops log as [base_loop])")
        print(f"  log [WAIT] market-closed lines: {log['wait']} (stock off-hours; logged every 20th wait)"
              f"   [SLEEP] lines: {log['sleep']}")


def section_preds(log, since, pred_files):
    hdr('Predictions produced (non-null)')
    for b, p in pred_files.items():
        if not os.path.exists(p):
            print(f"  {b:<7} {os.path.basename(p)}: absent")
            continue
        n_lines = n_preds = 0
        syms = collections.Counter()
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if since is not None:
                    t = parse_ts(r.get('ts'))
                    if t is None or t < since:
                        continue
                n_lines += 1
                pr = r.get('preds') or {}
                n_preds += len(pr)
                syms.update(pr.keys())
        print(f"  {b:<7} cycles-with-preds {n_lines:>6}  non-null preds {n_preds:>7}  distinct symbols {len(syms)}")
    if not log.get('missing'):
        print(f"  log 'Predicted Return:' lines (both books, printed by predict_now): {log['pred_lines']}")


def section_gates(rows):
    hdr('Gate decisions (journal skip rows)')
    skips = collections.Counter()
    for r in rows:
        if r.get('action') == 'skip':
            skips[(book_of(r), norm_reason(r.get('skip_reason')))] += 1
    if not skips:
        print('  no skip rows')
    for (b, reason), n in sorted(skips.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f"  {b:<7} {reason:<40} {n:>7}")
    vc = collections.defaultdict(collections.Counter)
    nwin = collections.Counter()
    blocked = collections.Counter()
    admitted = collections.Counter()
    for r in rows:
        if r.get('action') == 'entry_window':
            b = book_of(r)
            nwin[b] += 1
            admitted[b] += int(r.get('admitted_k') or 0)
            if not r.get('buys_allowed', True):
                blocked[b] += 1
            for k, v in (r.get('veto_counts') or {}).items():
                try:
                    vc[b][k] += int(v)
                except (TypeError, ValueError):
                    pass
    for b in sorted(nwin):
        print(f"  {b:<7} entry_window rows {nwin[b]} (buys_allowed=False on {blocked[b]}), admitted total {admitted[b]}")
        for k, v in vc[b].most_common():
            print(f"            veto {k:<32} {v:>7}")
    actions = collections.Counter((book_of(r), r.get('action')) for r in rows)
    print('  all journal rows by (book, action): ' +
          ', '.join(f'{b}:{a}={n}' for (b, a), n in sorted(actions.items(), key=lambda kv: -kv[1])))


def section_orders(log, rows, since, use_alpaca):
    hdr('Orders')
    buys = collections.Counter(book_of(r) for r in rows if r.get('action') == 'buy')
    sells = collections.Counter((book_of(r), r.get('exit_reason') or '?') for r in rows if r.get('action') == 'sell')
    print(f"  journal buy rows (filled entries): {dict(buys) or 0}")
    print(f"  journal sell rows by (book, exit_reason): {dict(sells) or 0}")
    if not log.get('missing'):
        print(f"  log [ORDER] placed by (book, side): {dict(log['placed']) or 0}")
        print(f"  log fills (LIFECYCLE/MAKER FILLED): {log['filled']}   cancels: {log['cancel']}   give-ups: {log['giveup']}")
        if log['order_err']:
            print('  log order ERRORS:')
            for (side, msg), n in log['order_err'].most_common(8):
                print(f"    {n:>6}  {side}: {msg}")
    print('  NOTE: log counts are approximate (several submit_order sites log nothing on success);'
          ' use --alpaca for the authoritative broker-side placed/filled/cancelled counts.')
    if use_alpaca:
        try:
            sys.path.insert(0, REPO)
            from trading_utils import get_api
            api = get_api()
            after = (since or (dt.datetime.now().astimezone() - dt.timedelta(days=1))).isoformat()
            orders = api.list_orders(status='all', after=after, limit=500, nested=False)
            c = collections.Counter((('crypto' if '/' in o.symbol or (o.symbol.endswith('USD') and len(o.symbol) > 3) else 'stock'),
                                     o.side, o.type, o.status) for o in orders)
            print(f"  alpaca orders since {after} (read-only, n={len(orders)}):")
            for k, n in sorted(c.items(), key=lambda kv: -kv[1]):
                print(f"    {n:>5}  {k}")
        except Exception as e:  # noqa: BLE001
            print(f"  alpaca order query failed: {type(e).__name__}: {e}")


def section_errors(log):
    hdr('Errors / Tracebacks (bot log)')
    if log.get('missing'):
        print('  log not found')
        return
    print(f"  lines scanned: {log['lines']}")
    if log['lgb_fatal']:
        print('  LightGBM [Fatal] (missing booster legs => LSTM-only serving):')
        for m, n in log['lgb_fatal'].most_common():
            print(f"    {n:>6}  {m}")
    if log['errors']:
        print('  top ERROR/CRITICAL lines (digits normalised):')
        for m, n in log['errors'].most_common(12):
            print(f"    {n:>6}  {m}")
    if not log['tbs']:
        print('  no Tracebacks')
    else:
        print(f"  distinct Tracebacks: {len(log['tbs'])} (total {sum(e['n'] for e in log['tbs'].values())})")
        for i, (sig, e) in enumerate(log['tbs'].items()):
            if i >= 25:
                print(f"  ... {len(log['tbs']) - 25} more")
                break
            print(f"  [{e['n']}x]" + (f"  after: {e['ctx'][:160]}" if e.get('ctx') else ''))
            for l in e['first3']:
                print(f"      {l[:200]}")
            print(f"      ... {e['exc'][:200]}")


def section_llm(recs, start_file):
    hdr('LLM spend (llm_cost.json, per LA calendar day)')
    series = []
    if start_file and os.path.exists(start_file):
        try:
            series.append(('start', json.load(open(start_file))))
        except ValueError:
            pass
    for r in recs:
        if isinstance(r.get('llm_cost'), dict):
            series.append((r.get('ts'), r['llm_cost']))
    try:
        series.append(('now', json.load(open(os.path.join(REPO, 'llm_cost.json')))))
    except (OSError, ValueError):
        pass
    if len(series) < 2:
        print(f"  not enough readings ({len(series)}); current: {series[-1][1] if series else 'n/a'}")
        return
    base_date, base_cost = series[0][1].get('date'), float(series[0][1].get('cost') or 0)
    by_day = collections.OrderedDict()
    for _, c in series:
        d = c.get('date')
        by_day[d] = max(by_day.get(d, 0.0), float(c.get('cost') or 0))
    delta = 0.0
    for d, mx in by_day.items():
        delta += (mx - base_cost) if d == base_date else mx
    print(f"  baseline {base_date} ${base_cost:.6f}; per-day max: "
          + ', '.join(f'{d}=${v:.6f}' for d, v in by_day.items()))
    print(f"  spend delta over the window: ${delta:.6f}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--monitor', default=os.path.join(PHASE5, 'monitor.jsonl'))
    ap.add_argument('--log', default=os.path.join(PHASE5, 'bots_stdout.log'))
    ap.add_argument('--last-run', action='store_true', help='only the log after the last start_bots banner')
    ap.add_argument('--journal-date', action='append', default=[], help='YYYY-MM-DD (repeatable)')
    ap.add_argument('--journal', action='append', default=[], help='explicit journal file (repeatable)')
    ap.add_argument('--since', default=None,
                    help="ISO ts lower bound for journal/pred rows; 'none' = no filter; "
                         "default = phase5/bots_started.json started_at")
    ap.add_argument('--pred-dir', default=REPO, help='where {,stock_}pred_history.jsonl live')
    ap.add_argument('--alpaca', action='store_true', help='read-only order query (jetson env)')
    args = ap.parse_args()

    started = None
    try:
        with open(os.path.join(PHASE5, 'bots_started.json')) as f:
            started = json.load(f)
    except (OSError, ValueError):
        pass
    if args.since and args.since.lower() == 'none':
        since = None
    elif args.since:
        since = parse_ts(args.since)
    else:
        since = parse_ts(started.get('started_at')) if started else None

    files = list(args.journal)
    dates = list(args.journal_date)
    if not files and not dates:
        d0 = since.astimezone().date() if since else dt.date.today()
        d = d0
        while d <= dt.date.today():
            dates.append(d.isoformat())
            d += dt.timedelta(days=1)
    for d in dates:
        for ext in ('.jsonl', '.jsonl.gz'):
            p = os.path.join(REPO, 'journals', d + ext)
            if os.path.exists(p):
                files.append(p)

    print('trader phase5 observation summary  ' + dt.datetime.now().astimezone().isoformat(timespec='seconds'))
    print(f"  since: {since.isoformat() if since else 'ALL ROWS'}   started: {started or 'n/a'}")
    print(f"  journals: {files or 'none found'}")
    print(f"  log: {args.log}{' (last run only)' if args.last_run else ''}   monitor: {args.monitor}")

    recs = load_monitor(args.monitor)
    log = scan_log(args.log, args.last_run)
    rows = load_journal_rows(files, since)
    pred_files = {b: os.path.join(args.pred_dir, f) for b, f in PRED_HISTORY.items()}

    for fn in (lambda: section_monitor(recs),
               lambda: section_cycles(log, rows),
               lambda: section_preds(log, since, pred_files),
               lambda: section_gates(rows),
               lambda: section_orders(log, rows, since, args.alpaca),
               lambda: section_errors(log),
               lambda: section_llm(recs, os.path.join(PHASE5, 'llm_cost_start.json'))):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f'  !! section failed: {type(e).__name__}: {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
