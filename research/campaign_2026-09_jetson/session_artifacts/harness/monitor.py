#!/usr/bin/env python3
"""monitor.py — observation-window probe loop for the harness-launched bots.

One JSON line per tick -> phase5/monitor.jsonl, plus a one-line human summary
on stdout. Every probe runs in its own try/except; a failing probe records
`"<probe>_err": "..."` and the loop carries on. Nothing here writes to the
repo, and the Alpaca probe is READ-ONLY (get_account / list_positions /
list_orders(status='open')).

Run under the jetson env so the Alpaca probe can import trading_utils:
    source <scratchpad>/jenv.sh
    CUDA_VISIBLE_DEVICES='' $JPY monitor.py --minutes 1440 &

Flags: --minutes N (total; 0 = forever)  --interval S (60)  --pid P
       (default: read phase5/bots.pid each tick)  --out F  --log F
       --alpaca-every K (10; tick 0 always probes)  --no-alpaca
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
import threading
import time
import traceback

CODE_REPO = '/home/kyle/trader'                      # imports (hw_monitor, trading_utils)
REPO = os.environ.get('HARNESS_REPO', CODE_REPO)      # data files; override ONLY for self-test
PHASE5 = '/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/phase5'
CLK_TCK = os.sysconf('SC_CLK_TCK')
NCPU = os.cpu_count() or 1

FLAG_FILES = {                       # confirmed in code:
    'halt': 'trading_halt.flag',     # notify.py:131 _HALT_FLAG
    'flatten_shared': 'flatten_request.flag',   # notify.py:133 _FLATTEN_FLAG
    'flatten_crypto': 'flatten_crypto.flag',    # base_loop._flatten_flag_path
    'flatten_stock': 'flatten_stock.flag',
}
PRED_HISTORY = {'crypto': 'pred_history.jsonl',          # monitor_drift.history_file('')
                'stock': 'stock_pred_history.jsonl'}      # history_file('stock')

ERR_RE = re.compile(r'\] (ERROR|CRITICAL): |Traceback \(most recent call last\)|\[LightGBM\] \[Fatal\]')
TB_RE = re.compile(r'Traceback \(most recent call last\)')
CYCLE_RE = re.compile(r'--- CYCLE (\d+): ')
WAIT_RE = re.compile(r'\[WAIT\] Market closed')
ORDER_RE = re.compile(r'\[ORDER\] \S+: (buy|sell) ')
FILL_RE = re.compile(r'\[LIFECYCLE\] (Order \S+ FILLED|Market fallback FILLED)|\[MAKER\] \S+: rung \d+ filled')


# --------------------------------------------------------------------------- probes
def read_pid(args):
    if args.pid:
        return int(args.pid)
    try:
        with open(os.path.join(PHASE5, 'bots.pid')) as f:
            s = re.sub(r'\D', '', f.read())
        return int(s) if s else None
    except OSError:
        return None


def proc_status(pid):
    out = {}
    with open(f'/proc/{pid}/status') as f:
        for line in f:
            k, _, v = line.partition(':')
            if k in ('VmRSS', 'VmHWM', 'VmSwap', 'Threads', 'State'):
                v = v.strip()
                out[k] = int(v.split()[0]) if k.startswith('Vm') or k == 'Threads' else v
    with open(f'/proc/{pid}/cmdline', 'rb') as f:
        out['cmdline'] = f.read().replace(b'\0', b' ').decode(errors='replace').strip()[:200]
    return out


def proc_ticks(pid):
    with open(f'/proc/{pid}/stat') as f:
        s = f.read()
    rest = s[s.rindex(')') + 2:].split()
    return int(rest[11]) + int(rest[12])            # utime + stime (fields 14,15)


def descendants(pid):
    """All descendant pids (run_bots is expected to have none)."""
    kids = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/stat') as f:
                s = f.read()
            ppid = int(s[s.rindex(')') + 2:].split()[1])
            kids.setdefault(ppid, []).append(int(d))
        except (OSError, ValueError, IndexError):
            continue
    out, stack = [], [pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


def sys_cpu_ticks():
    with open('/proc/stat') as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return sum(v), idle


def meminfo():
    m = {}
    with open('/proc/meminfo') as f:
        for line in f:
            k, _, v = line.partition(':')
            m[k] = int(v.split()[0])
    return {'mem_total_mb': round(m['MemTotal'] / 1024),
            'mem_available_mb': round(m['MemAvailable'] / 1024),
            'swap_used_mb': round((m['SwapTotal'] - m['SwapFree']) / 1024)}


def cpu_temp():
    for z in sorted(glob.glob('/sys/devices/virtual/thermal/thermal_zone*')):
        try:
            with open(z + '/type') as f:
                if f.read().strip() != 'cpu-thermal':
                    continue
            with open(z + '/temp') as f:
                return int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return None


def tj_temp():
    for z in sorted(glob.glob('/sys/devices/virtual/thermal/thermal_zone*')):
        try:
            with open(z + '/type') as f:
                if f.read().strip() != 'tj-thermal':
                    continue
            with open(z + '/temp') as f:
                return int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return None


class Tail:
    """Incremental reader: returns lines appended since the previous call.
    Handles truncation/rotation (size < offset -> restart from 0)."""

    def __init__(self, path, from_start=False):
        self.path = path
        try:
            self.off = 0 if from_start else os.path.getsize(path)
        except OSError:
            self.off = 0

    def read_new(self, max_bytes=64 * 1024 * 1024):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []
        if size < self.off:
            self.off = 0
        if size == self.off:
            return []
        with open(self.path, 'rb') as f:
            f.seek(self.off)
            data = f.read(min(size - self.off, max_bytes))
        # only consume complete lines
        cut = data.rfind(b'\n')
        if cut < 0:
            return []
        self.off += cut + 1
        return data[:cut].decode('utf-8', errors='replace').split('\n')


class JournalTails:
    """Tracks every journals/*.jsonl (by local-date filename; a new day file
    appearing mid-run is read from byte 0)."""

    def __init__(self):
        self.tails = {}
        for p in glob.glob(os.path.join(REPO, 'journals', '*.jsonl')):
            self.tails[p] = Tail(p)              # baseline: existing content not counted

    def read_new(self):
        for p in glob.glob(os.path.join(REPO, 'journals', '*.jsonl')):
            if p not in self.tails:
                self.tails[p] = Tail(p, from_start=True)
        rows = []
        for t in self.tails.values():
            for line in t.read_new():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    rows.append({'action': '<unparseable>'})
        return rows


def book_of(row):
    b = row.get('book') or row.get('asset_type')
    if b:
        return b
    sym = row.get('symbol') or ''
    if sym:
        return 'crypto' if '/' in sym else 'stock'
    return '?'


def alpaca_snapshot(timeout=45):
    """READ-ONLY account snapshot, bounded by a watchdog thread."""
    res = {}

    def _work():
        try:
            if CODE_REPO not in sys.path:
                sys.path.insert(0, CODE_REPO)
            from trading_utils import get_api
            api = get_api()
            a = api.get_account()
            res['equity'] = round(float(a.equity), 2)
            res['cash'] = round(float(a.cash), 2)
            res['buying_power'] = round(float(a.buying_power), 2)
            res['status'] = str(getattr(a, 'status', ''))
            pos = api.list_positions()
            res['n_positions'] = len(pos)
            res['positions_mv'] = round(sum(float(p.market_value) for p in pos), 2)
            res['n_open_orders'] = len(api.list_orders(status='open', limit=500))
        except Exception as e:  # noqa: BLE001
            res['err'] = f'{type(e).__name__}: {e}'[:300]

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        res['err'] = f'timeout after {timeout}s'
    return res


def llm_cost():
    with open(os.path.join(REPO, 'llm_cost.json')) as f:
        return json.load(f)


# --------------------------------------------------------------------------- loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=0, help='total run time (0 = forever)')
    ap.add_argument('--interval', type=float, default=60)
    ap.add_argument('--pid', type=int, default=None, help='override phase5/bots.pid')
    ap.add_argument('--out', default=os.path.join(PHASE5, 'monitor.jsonl'))
    ap.add_argument('--log', default=os.path.join(PHASE5, 'bots_stdout.log'))
    ap.add_argument('--alpaca-every', type=int, default=10)
    ap.add_argument('--no-alpaca', action='store_true')
    ap.add_argument('--log-from-now', action='store_true',
                    help='ignore log lines written before the monitor started '
                         '(default: rewind to the last start_bots.sh banner)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    try:
        sys.path.insert(0, CODE_REPO)
        import hw_monitor                      # stdlib-only module; no torch import
        gpu_temp_fn = hw_monitor.get_gpu_temp
    except Exception as e:  # noqa: BLE001
        print(f'[monitor] hw_monitor import failed: {e}', file=sys.stderr)
        gpu_temp_fn = lambda: None            # noqa: E731

    log_tail = Tail(args.log)
    if not args.log_from_now:
        # count the CURRENT run's startup lines too: rewind to the last
        # start_bots.sh banner (bots are normally started before the monitor)
        try:
            off = last = 0
            with open(args.log, 'rb') as f:
                for raw in f:
                    if raw.startswith(b'===== start_bots.sh '):
                        last = off
                    off += len(raw)
            log_tail.off = last if off else 0
        except OSError:
            pass
    jtails = JournalTails()
    pred_tails = {b: Tail(os.path.join(REPO, f)) for b, f in PRED_HISTORY.items()}
    prev_proc = {}          # pid -> (ticks, wall)
    prev_sys = None
    t_end = time.time() + args.minutes * 60 if args.minutes > 0 else None
    tick = 0

    while True:
        t0 = time.time()
        rec = {'ts': dt.datetime.now().astimezone().isoformat(timespec='seconds'), 'tick': tick}

        # --- bot process
        try:
            pid = read_pid(args)
            rec['pid'] = pid
            if pid is None or not os.path.exists(f'/proc/{pid}'):
                rec['alive'] = False
            else:
                rec['alive'] = True
                st = proc_status(pid)
                rec['rss_mb'] = round(st.get('VmRSS', 0) / 1024, 1)
                rec['hwm_mb'] = round(st.get('VmHWM', 0) / 1024, 1)
                rec['swap_mb'] = round(st.get('VmSwap', 0) / 1024, 1)
                rec['threads'] = st.get('Threads')
                rec['state'] = st.get('State')
                rec['cmdline'] = st.get('cmdline')
                kids = descendants(pid)
                rec['n_children'] = len(kids)
                if kids:
                    tot = 0.0
                    for k in kids:
                        try:
                            tot += proc_status(k).get('VmRSS', 0) / 1024
                        except OSError:
                            pass
                    rec['children_rss_mb'] = round(tot, 1)
        except Exception as e:  # noqa: BLE001
            rec['proc_err'] = repr(e)[:200]

        # --- bot CPU %
        try:
            pid = rec.get('pid')
            if rec.get('alive'):
                ticks, now = proc_ticks(pid), time.monotonic()
                if pid in prev_proc:
                    pt, pw = prev_proc[pid]
                    dw = now - pw
                    if dw > 0:
                        pct = 100.0 * (ticks - pt) / CLK_TCK / dw
                        rec['cpu_pct'] = round(pct, 1)          # % of ONE core
                        rec['cpu_pct_of_box'] = round(pct / NCPU, 1)
                prev_proc = {pid: (ticks, now)}
        except Exception as e:  # noqa: BLE001
            rec['cpu_err'] = repr(e)[:200]

        # --- system CPU %
        try:
            tot, idle = sys_cpu_ticks()
            if prev_sys:
                dt_, di = tot - prev_sys[0], idle - prev_sys[1]
                if dt_ > 0:
                    rec['sys_cpu_pct'] = round(100.0 * (1 - di / dt_), 1)
            prev_sys = (tot, idle)
        except Exception as e:  # noqa: BLE001
            rec['syscpu_err'] = repr(e)[:200]

        # --- memory / swap
        try:
            rec.update(meminfo())
        except Exception as e:  # noqa: BLE001
            rec['mem_err'] = repr(e)[:200]

        # --- temps
        try:
            rec['gpu_temp_c'] = gpu_temp_fn()
            rec['cpu_temp_c'] = cpu_temp()
            rec['tj_temp_c'] = tj_temp()
        except Exception as e:  # noqa: BLE001
            rec['temp_err'] = repr(e)[:200]

        # --- load average
        try:
            rec['loadavg'] = [round(x, 2) for x in os.getloadavg()]
        except Exception as e:  # noqa: BLE001
            rec['load_err'] = repr(e)[:200]

        # --- bot stdout log
        try:
            lines = log_tail.read_new()
            rec['log_new_lines'] = len(lines)
            rec['log_err_lines'] = sum(1 for l in lines if ERR_RE.search(l))
            rec['log_tracebacks'] = sum(1 for l in lines if TB_RE.search(l))
            rec['log_cycles'] = sum(1 for l in lines if CYCLE_RE.search(l))
            rec['log_wait_lines'] = sum(1 for l in lines if WAIT_RE.search(l))
            rec['log_orders'] = sum(1 for l in lines if ORDER_RE.search(l))
            rec['log_fills'] = sum(1 for l in lines if FILL_RE.search(l))
            cyc = [int(m.group(1)) for l in lines for m in [CYCLE_RE.search(l)] if m]
            if cyc:
                rec['log_last_cycle_no'] = max(cyc)
            errs = [l for l in lines if ERR_RE.search(l)]
            if errs:
                rec['log_err_sample'] = errs[-1][:240]
        except Exception as e:  # noqa: BLE001
            rec['log_err'] = repr(e)[:200]

        # --- journals
        try:
            rows = jtails.read_new()
            by = {}
            for r in rows:
                k = f"{book_of(r)}:{r.get('action', '?')}"
                by[k] = by.get(k, 0) + 1
            rec['journal_new_rows'] = len(rows)
            rec['journal_by_action'] = by
        except Exception as e:  # noqa: BLE001
            rec['journal_err'] = repr(e)[:200]

        # --- predictions (monitor_drift pred_history: one line per cycle w/ non-null preds)
        try:
            pn = {}
            for b, t in pred_tails.items():
                n_lines = n_preds = 0
                for line in t.read_new():
                    if not line.strip():
                        continue
                    n_lines += 1
                    try:
                        n_preds += len(json.loads(line).get('preds') or {})
                    except ValueError:
                        pass
                pn[b] = {'cycles_with_preds': n_lines, 'preds': n_preds}
            rec['pred_new'] = pn
        except Exception as e:  # noqa: BLE001
            rec['pred_err'] = repr(e)[:200]

        # --- control flags
        try:
            rec['flags'] = {k: os.path.exists(os.path.join(REPO, f)) for k, f in FLAG_FILES.items()}
        except Exception as e:  # noqa: BLE001
            rec['flags_err'] = repr(e)[:200]

        # --- LLM spend (cumulative for the LA calendar day)
        try:
            rec['llm_cost'] = llm_cost()
        except Exception as e:  # noqa: BLE001
            rec['llm_cost_err'] = repr(e)[:200]

        # --- Alpaca (read-only) every Nth tick
        if not args.no_alpaca and args.alpaca_every > 0 and tick % args.alpaca_every == 0:
            try:
                rec['alpaca'] = alpaca_snapshot()
            except Exception as e:  # noqa: BLE001
                rec['alpaca'] = {'err': repr(e)[:200]}

        rec['probe_s'] = round(time.time() - t0, 2)

        # --- persist + summary line
        try:
            with open(args.out, 'a') as f:
                f.write(json.dumps(rec, default=str) + '\n')
        except Exception as e:  # noqa: BLE001
            print(f'[monitor] write failed: {e}', file=sys.stderr)
        try:
            fl = [k for k, v in (rec.get('flags') or {}).items() if v]
            jb = rec.get('journal_by_action') or {}
            jtxt = ','.join(f'{k}={v}' for k, v in sorted(jb.items())) or '-'
            pn = rec.get('pred_new') or {}
            ptxt = '/'.join(f"{b[0]}{(pn.get(b) or {}).get('preds', 0)}" for b in ('crypto', 'stock'))
            al = rec.get('alpaca')
            atxt = ''
            if al:
                atxt = (f" | acct eq={al.get('equity')} cash={al.get('cash')} "
                        f"pos={al.get('n_positions')} open_ord={al.get('n_open_orders')}"
                        if 'err' not in al else f" | acct ERR {al['err'][:80]}")
            print(f"{rec['ts'][11:19]} #{tick} pid={rec.get('pid')} "
                  f"{'UP' if rec.get('alive') else 'DOWN'} "
                  f"rss={rec.get('rss_mb')}MB hwm={rec.get('hwm_mb')}MB cpu={rec.get('cpu_pct')}% "
                  f"avail={rec.get('mem_available_mb')}MB swap={rec.get('swap_used_mb')}MB "
                  f"gpu={rec.get('gpu_temp_c')}C cpuT={rec.get('cpu_temp_c')}C "
                  f"load={(rec.get('loadavg') or ['?'])[0]} "
                  f"log+{rec.get('log_new_lines')} err+{rec.get('log_err_lines')} "
                  f"cyc+{rec.get('log_cycles')} preds={ptxt} "
                  f"jrn[{jtxt}] flags={fl or '-'}{atxt}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f'[monitor] summary failed: {e}', flush=True)

        tick += 1
        if t_end is not None and time.time() >= t_end:
            break
        sleep_for = args.interval - (time.time() - t0)
        if t_end is not None:
            sleep_for = min(sleep_for, max(0.0, t_end - time.time()))
        if sleep_for > 0:
            try:
                time.sleep(sleep_for)
            except KeyboardInterrupt:
                break
        # (when the sleep lands on t_end, the next iteration is the final tick)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
