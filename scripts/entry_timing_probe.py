"""M1 entry-timing probe (R2C-06d, measurement-only) — MEASURE-FIRST for
the one-bar entry-timing skew between live serving and training/backtest.

Background (06 plan M1, verified): offline windows use offsets
arange(-seq_len, 0) — the prediction at dump row i saw bars up to i-1 and
the certificate scores the label anchored Close[i] -> Close[i+h]
(the "TRAIN anchor": entry one full bar after the last observed close).
Live includes the last closed bar in the window and enters ~immediately
after it closes — for the SAME information set that is an entry at the
last window bar's close, i.e. the label anchored
Close[i-1] -> Close[i-1+h] (the "LIVE anchor").

This probe scores the EXISTING stage0 predictions against BOTH anchors:

    delta = IC(pred, fwd@live) - IC(pred, fwd@train)

per book (run once per prefix), pooled + per-name, with a weekly-block
bootstrap SE. Decision rule (deferred item 1): |delta| > 2*SE = material
-> schedule the WINDOW_INCLUDES_ENTRY_BAR offsets flip at Chain-2; else
record the negative and only the false parity comment gets corrected.

Plus the realized fill-gap join: scan journals/*.jsonl buy rows and report
seconds between each fill and its bar-close anchor (crypto bars close on
the hour; stock hourly bars on the half-hour) — the empirical size of the
timing gap the anchors bracket.

    python scripts/entry_timing_probe.py --preds stage0_preds.json \
        --prefix '' [--journal-dir journals] [--n-boot 500] [--seed 0]

Neither anchor leaks: the prediction used closes through i-1 and both
label windows start at or after Close[i-1].
"""
import argparse
import datetime
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import horizon_transfer as ht        # noqa: E402
from ic_diagnostic import rank_ic    # noqa: E402
from stage0_preds import index_ns    # noqa: E402

# bar-close anchor offset within the hour, minutes past the hour
BAR_ANCHOR_MIN = {'crypto': 0, 'stock': 30}


def anchor_returns(series_times_ns, closes, row_times_ns, horizon):
    """(fwd_train, fwd_live) PERCENT returns for each row time.

    fwd_train[j] = Close[i] -> Close[i+h]   (the certificate convention;
        equals the dump's own fwd_return — used as a join sanity check)
    fwd_live[j]  = Close[i-1] -> Close[i-1+h]  (entry at the last window
        bar's close — the live-timing equivalent)
    where i is the EXACT index of row_times_ns[j] in series_times_ns.
    NaN when the time is absent, i == 0, or the window leaves the series.
    """
    t = np.asarray(series_times_ns, dtype=np.int64)
    c = np.asarray(closes, dtype=np.float64)
    q = np.asarray(row_times_ns, dtype=np.int64)
    h = max(1, int(horizon))
    n = len(t)
    f_train = np.full(len(q), np.nan)
    f_live = np.full(len(q), np.nan)
    if n == 0:
        return f_train, f_live
    pos = np.searchsorted(t, q)
    for j, i in enumerate(pos):
        if i >= n or t[i] != q[j]:
            continue
        if i + h < n:
            c0, c1 = c[i], c[i + h]
            if np.isfinite(c0) and np.isfinite(c1) and c0 != 0.0:
                f_train[j] = (c1 - c0) / c0 * 100.0
        if i >= 1 and i - 1 + h < n:
            c0, c1 = c[i - 1], c[i - 1 + h]
            if np.isfinite(c0) and np.isfinite(c1) and c0 != 0.0:
                f_live[j] = (c1 - c0) / c0 * 100.0
    return f_train, f_live


def ic_anchor_delta(preds, fwd_train, fwd_live, times_ns, n_boot=500,
                    seed=0):
    """{ic_train, ic_live, delta, se, n, material} on the jointly-finite
    rows; SE via the weekly-block bootstrap of the DELTA (blocks resample
    whole weeks so news-sharing rows move together)."""
    p = np.asarray(preds, dtype=np.float64)
    a = np.asarray(fwd_train, dtype=np.float64)
    b = np.asarray(fwd_live, dtype=np.float64)
    t = np.asarray(times_ns, dtype=np.int64)
    m = np.isfinite(p) & np.isfinite(a) & np.isfinite(b)
    p, a, b, t = p[m], a[m], b[m], t[m]
    ic_a = rank_ic(p, a)
    ic_b = rank_ic(p, b)
    out = {'ic_train': ic_a, 'ic_live': ic_b, 'delta': None, 'se': None,
           'n': int(len(p)), 'material': None}
    if ic_a is None or ic_b is None:
        return out
    out['delta'] = ic_b - ic_a

    def _delta(idx):
        x = rank_ic(p[idx], b[idx])
        y = rank_ic(p[idx], a[idx])
        return None if x is None or y is None else x - y

    se, _ = ht.block_bootstrap_se(_delta, ht.weekly_block_ids(t),
                                  n_boot=n_boot, seed=seed)
    out['se'] = se
    if se is not None:  # se == 0.0 (degenerate delta) still verdicts
        out['material'] = bool(abs(out['delta']) > 2 * se)
    return out


def fill_gap_seconds(epoch_ts, anchor_offset_min=0):
    """Seconds elapsed since the most recent bar-close anchor for each
    fill timestamp: (ts - offset*60) mod 3600. anchor_offset_min: minutes
    past the hour where the book's hourly bars close (crypto 0, US-stock
    hourly bars 30)."""
    ts = np.asarray(epoch_ts, dtype=np.float64)
    return np.mod(ts - float(anchor_offset_min) * 60.0, 3600.0)


def scan_journal_buys(journal_dir):
    """[(symbol, epoch_ts), ...] from journals/*.jsonl(.gz) buy rows —
    fail-soft per line (measurement never crashes on a corrupt row)."""
    out = []
    d = Path(journal_dir)
    if not d.is_dir():
        return out
    for path in sorted(list(d.glob('*.jsonl')) +
                       list(d.glob('*.jsonl.gz'))):
        opener = gzip.open if path.suffix == '.gz' else open
        try:
            with opener(path, 'rt') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if row.get('action') != 'buy':
                            continue
                        ts = datetime.datetime.fromisoformat(
                            row['ts']).timestamp()
                        out.append((str(row.get('symbol', '?')), ts))
                    except Exception:
                        continue
        except OSError:
            continue
    return out


def _gap_summary(gaps):
    g = np.asarray(gaps, dtype=np.float64)
    if len(g) == 0:
        return None
    return {'n': int(len(g)), 'median_s': float(np.median(g)),
            'mean_s': float(np.mean(g)),
            'p90_s': float(np.percentile(g, 90))}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preds', required=True,
                    help='stage0 predictions dump (BARE JSON list)')
    ap.add_argument('--prefix', default='',
                    help="'' = crypto, 'stock' = stock")
    ap.add_argument('--journal-dir', default=str(BASE_DIR / 'journals'))
    ap.add_argument('--n-boot', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    prefix_key = 'stock' if args.prefix == 'stock' else 'crypto'
    rows = json.loads(Path(args.preds).read_text())
    if not isinstance(rows, list) or not rows:
        sys.exit(f'{args.preds}: expected a non-empty BARE JSON list')

    from data_utils import load_training_data
    df = load_training_data(prefix_key, columns=['Ticker', 'Close'])
    if df.empty or 'Ticker' not in df.columns:
        sys.exit(f"no training data for '{prefix_key}'")
    series = {}
    for tick, sub in df.groupby('Ticker'):
        sub = sub.sort_index()
        series[str(tick)] = (index_ns(sub.index),
                             sub['Close'].to_numpy(np.float64))

    by_name = {}
    for r in rows:
        by_name.setdefault(str(r['symbol']), []).append(r)
    pooled = {'p': [], 'a': [], 'b': [], 't': []}
    per_name = {}
    n_mismatch = 0
    for sym, rs in sorted(by_name.items()):
        key = sym if sym in series else sym.replace('/', '-')
        if key not in series:
            print(f'  [SKIP] {sym}: no close series in the training store')
            continue
        t_ser, c_ser = series[key]
        q = np.array([np.int64(pd.Timestamp(r['ts']).value) for r in rs])
        h = int(rs[0].get('horizon_bars') or 1)
        p = np.array([float(r['pred']) for r in rs])
        f_dump = np.array([float(r['fwd_return']) for r in rs])
        f_train, f_live = anchor_returns(t_ser, c_ser, q, h)
        # join sanity: the recomputed train anchor must match the dump
        ok = np.isfinite(f_train) & np.isfinite(f_dump)
        n_mismatch += int((np.abs(f_train[ok] - f_dump[ok]) > 1e-3).sum())
        per_name[sym] = ic_anchor_delta(p, f_train, f_live, q,
                                        n_boot=args.n_boot,
                                        seed=args.seed)
        pooled['p'].append(p)
        pooled['a'].append(f_train)
        pooled['b'].append(f_live)
        pooled['t'].append(q)
    if not pooled['p']:
        sys.exit('no rows joined — dump symbols do not match the store')
    if n_mismatch:
        print(f'[WARN] {n_mismatch} recomputed train-anchor returns '
              f'disagree with the dump fwd_return (>1e-3) — check the '
              f'store/dump vintage match')

    res = ic_anchor_delta(np.concatenate(pooled['p']),
                          np.concatenate(pooled['a']),
                          np.concatenate(pooled['b']),
                          np.concatenate(pooled['t']),
                          n_boot=args.n_boot, seed=args.seed)
    print(f"\n[M1] {prefix_key} pooled (n={res['n']}): "
          f"IC@train={_f(res['ic_train'])}  IC@live={_f(res['ic_live'])}"
          f"  delta={_f(res['delta'])} ±{_f(res['se'])}"
          f"  material(|d|>2SE)={res['material']}")
    print('  per-name:')
    for sym in sorted(per_name):
        r = per_name[sym]
        print(f"    {sym:<14} IC@train={_f(r['ic_train'])} "
              f"IC@live={_f(r['ic_live'])} delta={_f(r['delta'])} "
              f"±{_f(r['se'])} n={r['n']}")

    buys = scan_journal_buys(args.journal_dir)
    for book, off in BAR_ANCHOR_MIN.items():
        ts = [t for s, t in buys
              if ('/' in s) == (book == 'crypto')]
        summ = _gap_summary(fill_gap_seconds(ts, off))
        if summ is None:
            print(f'  [FILL-GAP] {book}: no journal buy rows found')
        else:
            print(f"  [FILL-GAP] {book}: n={summ['n']} "
                  f"median={summ['median_s']:.0f}s "
                  f"mean={summ['mean_s']:.0f}s p90={summ['p90_s']:.0f}s "
                  f"past the {off:02d}-min bar close")
    print('\n[M1] decision: material pooled delta -> schedule '
          'WINDOW_INCLUDES_ENTRY_BAR at Chain-2; null -> fix the false '
          'parity comment only (deferred item 1).')
    return 0


def _f(v):
    return '  None' if v is None else f'{v:+.4f}'


if __name__ == '__main__':
    sys.exit(main())
