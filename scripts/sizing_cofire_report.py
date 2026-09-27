#!/usr/bin/env python3
"""Sizing multiplier co-fire report (c26 S3 / defects D10 / 02_research B06).

Measurement-only CLI over journals/*.jsonl BUY rows (the nested 'sizing'
decomposition dict written by base_loop._compute_position_size — see the
trade_journal.py producer contract). Answers the D10 questions with real
fills: which de-risk multipliers fire, how often they co-fire on the same
shared driver, how often the composed product saturates the 0.1 floor or
the TILT_MAX clamp (the floor makes every differentiating input inert), and
— once the DERISK_STACK_V2 shadow keys exist — how the v2 min-family
composition compares against the legacy product before any flag flip.

CONSUMER CONTRACT: read-only over journal rows; keys read here (sizing.*,
sizing.v2.*, skip_reason, ts, symbol) break on a producer rename — keep in
sync with base_loop._compute_position_size and trade_journal.py.

Malformed lines are skipped and counted; empty/missing journal dir prints a
clean 'no rows' result and exits 0. Exit codes (2026-09): 0 = report
produced (or --help); 2 = bad command-line argument (argparse error — no
longer swallowed, so wrappers can detect misuse); 1 = unexpected runtime
failure (e.g. the --json PATH is not writable).

Usage:
  python scripts/sizing_cofire_report.py --days 30 [--book crypto|stock|all]
                                         [--journal-dir DIR] [--json [PATH]]
  --json PATH  writes the report JSON to PATH (atomic tmp + os.replace);
  --json       (bare, or --json -) prints the JSON object to stdout.
"""

import argparse
import datetime as _dt
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    from strategy_config import TILT_MAX
except Exception:
    TILT_MAX = 1.30

# Every multiplier key the producer writes into sizing (base_loop verified
# 2026-08): stats section covers all; the cofire matrix covers the DE-RISK
# subset that shares upstream drivers.
MULT_KEYS = ['kelly_mult', 'vol_mult', 'signal_conf', 'vix_tilt', 'dd_mult',
             'macro_mult', 'corr_mult', 'hmm_mult', 'disagree_mult',
             'sentiment_mult', 'llm_mult', 'meta_mult', 'extra_tilt',
             'book_vol_mult']
DERISK_KEYS = ['vix_tilt', 'macro_mult', 'hmm_mult', 'dd_mult', 'corr_mult',
               'book_vol_mult', 'sentiment_mult', 'llm_mult', 'meta_mult',
               'disagree_mult']


# A naive `ts` is a legacy (pre-2026-08-20) row stamped with
# `datetime.now().isoformat()` — the writer's LOCAL wall clock, not UTC
# (G6-2; same reading as llm_eval / journal_stats / chart_core). None = the
# system zone (`datetime.astimezone()`, DST-aware); tests pin a zone here.
_LOCAL_TZ = None


def _parse_ts(ts):
    try:
        dt = _dt.datetime.fromisoformat(str(ts))
        if dt.tzinfo is None:
            dt = (dt.replace(tzinfo=_LOCAL_TZ) if _LOCAL_TZ is not None
                  else dt.astimezone())
        return dt
    except Exception:
        return None


_TS_MIN = _dt.datetime.min.replace(tzinfo=_dt.timezone.utc)


def _load_rows(journal_dir, days, book):
    """(buy_rows_with_sizing, skip_sizing_zero_rows, n_malformed, n_files,
    n_buy_rows_without_sizing) — the last counts in-window buys that carry no
    'sizing' dict (journals older than the sizing producer), header-only."""
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=days)
    buys, zero_skips, malformed, n_files = [], [], 0, 0
    n_buy_nosizing = 0
    try:
        names = sorted(os.listdir(journal_dir))
    except OSError:
        return buys, zero_skips, malformed, n_files, n_buy_nosizing
    for name in names:
        if not name.endswith('.jsonl'):
            continue
        n_files += 1
        try:
            with open(os.path.join(journal_dir, name), 'r') as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('not an object')
            except Exception:
                malformed += 1
                continue
            ts = _parse_ts(row.get('ts'))
            if ts is None or ts < cutoff:
                continue
            sym = str(row.get('symbol') or '')
            row_book = 'crypto' if '/' in sym else 'stock'
            if book != 'all' and row_book != book:
                continue
            action = row.get('action')
            if action == 'buy' and isinstance(row.get('sizing'), dict):
                buys.append(row)
            elif action == 'buy':
                n_buy_nosizing += 1
            elif action == 'skip' and row.get('skip_reason') == 'sizing_zero':
                zero_skips.append(row)
    return buys, zero_skips, malformed, n_files, n_buy_nosizing


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _stats(vals):
    vals = sorted(vals)
    n = len(vals)
    if not n:
        return None

    def q(p):
        return vals[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {'n_present': n,
            'fire_rate': round(sum(1 for v in vals if v < 1.0) / n, 4),
            'boost_rate': round(sum(1 for v in vals if v > 1.0) / n, 4),
            'mean': round(sum(vals) / n, 4),
            'median': round(q(0.5), 4),
            'p10': round(q(0.10), 4),
            'min': round(vals[0], 4)}


def _median(vals):
    vals = sorted(vals)
    n = len(vals)
    if not n:
        return None
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def build_report(buys, zero_skips, malformed, n_files, days, book,
                 n_buy_rows_without_sizing=0):
    # n_buy_rows = buys WITH a sizing dict (the analysed set; unchanged
    # meaning). n_buy_rows_without_sizing is additive, for the header.
    rep = {'n_buy_rows': len(buys), 'n_files': n_files,
           'n_malformed_lines': malformed, 'days': days, 'book': book,
           'tilt_max': TILT_MAX,
           'n_buy_rows_without_sizing': int(n_buy_rows_without_sizing)}
    sizings = [r['sizing'] for r in buys]

    # 1. per_multiplier
    per_mult = {}
    for key in MULT_KEYS:
        vals = [s[key] for s in sizings if _num(s.get(key))]
        st = _stats(vals)
        if st is not None:
            per_mult[key] = st
    rep['per_multiplier'] = per_mult

    # 2. bind_rates
    raws = [s['tilt_raw'] for s in sizings if _num(s.get('tilt_raw'))]
    rep['bind_rates'] = {
        'n_with_tilt_raw': len(raws),
        'floor_0_1': sum(1 for v in raws if v < 0.1),
        'tilt_max_clamp': sum(1 for v in raws if v > TILT_MAX),
        'degraded': sum(1 for s in sizings if 'degraded_inputs' in s),
        'book_risk_scale_present': sum(
            1 for s in sizings if 'book_risk_scale' in s),
        'sizing_zero_skips': len(zero_skips),
    }

    # 3. cofire matrix over the de-risk keys + P(fires | floor bound)
    pairs = {}
    for i, a in enumerate(DERISK_KEYS):
        for b in DERISK_KEYS[i + 1:]:
            both = sum(1 for s in sizings
                       if _num(s.get(a)) and _num(s.get(b))
                       and s[a] < 1.0 and s[b] < 1.0)
            if both:
                pairs['%s&%s' % (a, b)] = {
                    'count': both,
                    'p_both_fire': round(both / len(sizings), 4)}
    floor_rows = [s for s in sizings
                  if _num(s.get('tilt_raw')) and s['tilt_raw'] < 0.1]
    fires_given_floor = {}
    for key in DERISK_KEYS:
        if floor_rows:
            fires_given_floor[key] = round(
                sum(1 for s in floor_rows
                    if _num(s.get(key)) and s[key] < 1.0) / len(floor_rows), 4)
    rep['cofire'] = {'pairs': pairs,
                     'p_fires_given_floor': fires_given_floor}

    # 4. worst_product
    worst = None
    for r in buys:
        s = r['sizing']
        if _num(s.get('tilt_raw')) and (
                worst is None or s['tilt_raw'] < worst['sizing']['tilt_raw']):
            worst = {'ts': r.get('ts'), 'symbol': r.get('symbol'),
                     'sizing': s}
    rep['worst_product'] = worst

    # 5. marginal_effect: realized (post-floor/clamp) effect of each
    # multiplier on the final tilt — median(tilt / counterfactual tilt
    # with that multiplier removed).
    marginal = {}
    # kelly_mult and vol_mult are applied to `sized` AFTER the tilt product
    # (base_loop._compute_position_size) and never enter tilt_raw, so
    # `raw / m` is meaningless for them — exclude them here.
    for key in [k for k in MULT_KEYS if k not in ('kelly_mult', 'vol_mult')]:
        ratios = []
        for s in sizings:
            m = s.get(key)
            raw, tilt = s.get('tilt_raw'), s.get('tilt')
            if (_num(m) and m != 1.0 and m > 0
                    and _num(raw) and _num(tilt)):
                cf = max(0.1, min(TILT_MAX, raw / m))
                if cf > 0:
                    ratios.append(tilt / cf)
        med = _median(ratios)
        if med is not None:
            marginal[key] = {'n': len(ratios), 'median_effect': round(med, 4)}
    rep['marginal_effect'] = marginal

    # 6. v2_shadow (only when the S3 shadow keys exist)
    v2_rows = [r for r in buys if isinstance(r['sizing'].get('v2'), dict)
               and r['sizing']['v2']]
    if v2_rows:
        leg = [r['sizing']['tilt'] for r in v2_rows
               if _num(r['sizing'].get('tilt'))]
        v2t = [r['sizing']['v2']['tilt'] for r in v2_rows
               if _num(r['sizing']['v2'].get('tilt'))]
        ratios = [r['sizing']['v2']['tilt'] / r['sizing']['tilt']
                  for r in v2_rows
                  if _num(r['sizing'].get('tilt'))
                  and _num(r['sizing']['v2'].get('tilt'))
                  and r['sizing']['tilt'] > 0]
        min_src = {}
        for r in v2_rows:
            src = r['sizing']['v2'].get('min_src')
            if src:
                min_src[src] = min_src.get(src, 0) + 1
        # Hysteresis instrumentation: row-over-row flips of the v2 VIX tier
        # and the BTC-RV state, per day.
        flips = {'vix_tier_total': 0, 'btc_rv_state_total': 0, 'per_day': {}}
        # Chronological, not lexicographic: ISO strings with different UTC
        # offsets (DST fall-back hour) do not sort in time order as text.
        ordered = sorted(v2_rows,
                         key=lambda r: _parse_ts(r.get('ts')) or _TS_MIN)
        prev = {}
        for r in ordered:
            day = str(r.get('ts'))[:10]
            v2d = r['sizing']['v2']
            cur = {'vix_tier': (v2d.get('family') or {}).get('vix'),
                   'btc_rv_state': v2d.get('btc_rv_state')}
            for field in ('vix_tier', 'btc_rv_state'):
                if (cur[field] is not None and prev.get(field) is not None
                        and cur[field] != prev[field]):
                    flips[field + '_total'] += 1
                    dd = flips['per_day'].setdefault(
                        day, {'vix_tier': 0, 'btc_rv_state': 0})
                    dd[field] += 1
                if cur[field] is not None:
                    prev[field] = cur[field]
        rep['v2_shadow'] = {
            'n_rows': len(v2_rows),
            'stacks_applied': {
                k: sum(1 for r in v2_rows if r['sizing'].get('stack') == k)
                for k in ('legacy', 'v2')},
            'legacy_tilt': _stats(leg),
            'v2_tilt': _stats(v2t),
            'median_v2_over_legacy': (round(_median(ratios), 4)
                                      if ratios else None),
            'min_src_histogram': min_src,
            'floor_saturation': {
                'legacy': sum(1 for r in v2_rows
                              if _num(r['sizing'].get('tilt_raw'))
                              and r['sizing']['tilt_raw'] < 0.1),
                'v2': sum(1 for r in v2_rows
                          if _num(r['sizing']['v2'].get('tilt_raw'))
                          and r['sizing']['v2']['tilt_raw'] < 0.1)},
            'flip_counts': flips,
        }
    else:
        rep['v2_shadow'] = None
    return rep


def _print_table(title, rows, cols):
    print('\n== %s ==' % title)
    if not rows:
        print('  (none)')
        return
    widths = [max(len(str(c)), max((len(str(r[i])) for r in rows),
                                   default=0)) for i, c in enumerate(cols)]
    print('  ' + '  '.join(str(c).ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print('  ' + '  '.join(str(v).ljust(w) for v, w in zip(r, widths)))


def print_report(rep):
    n_nosz = int(rep.get('n_buy_rows_without_sizing') or 0)
    print('sizing co-fire report — last %sd, book=%s: %d buy rows, %d with '
          'sizing (%d files, %d malformed lines skipped)'
          % (rep['days'], rep['book'], rep['n_buy_rows'] + n_nosz,
             rep['n_buy_rows'], rep['n_files'], rep['n_malformed_lines']))
    if not rep['n_buy_rows']:
        if n_nosz:
            print('no rows with sizing — the %d buy row(s) in the window '
                  'pre-date the sizing producer (base_loop, 2026-07)' % n_nosz)
        else:
            print('no rows — nothing to report')
        return
    _print_table('per-multiplier', [
        (k, s['n_present'], s['fire_rate'], s['boost_rate'], s['mean'],
         s['median'], s['p10'], s['min'])
        for k, s in rep['per_multiplier'].items()],
        ['key', 'n', 'fire<1', 'boost>1', 'mean', 'median', 'p10', 'min'])
    _print_table('bind rates',
                 [(k, v) for k, v in rep['bind_rates'].items()],
                 ['bind', 'count'])
    _print_table('co-fire pairs (both < 1.0)', [
        (k, v['count'], v['p_both_fire'])
        for k, v in rep['cofire']['pairs'].items()],
        ['pair', 'count', 'p_both'])
    _print_table('P(fires | tilt_raw < 0.1)', [
        (k, v) for k, v in rep['cofire']['p_fires_given_floor'].items()],
        ['key', 'p'])
    w = rep['worst_product']
    if w:
        print('\n== worst product ==\n  %s %s tilt_raw=%s\n  sizing=%s'
              % (w['ts'], w['symbol'], w['sizing'].get('tilt_raw'),
                 json.dumps(w['sizing'], default=str)))
    _print_table('marginal effect (median tilt/tilt-without)', [
        (k, v['n'], v['median_effect'])
        for k, v in rep['marginal_effect'].items()],
        ['key', 'n', 'median'])
    v2 = rep['v2_shadow']
    if v2:
        print('\n== v2 shadow (DERISK_STACK_V2) ==')
        print('  rows=%d stacks_applied=%s median_v2/legacy=%s'
              % (v2['n_rows'], v2['stacks_applied'],
                 v2['median_v2_over_legacy']))
        print('  legacy_tilt=%s' % json.dumps(v2['legacy_tilt']))
        print('  v2_tilt=%s' % json.dumps(v2['v2_tilt']))
        print('  min_src=%s floor_saturation=%s'
              % (v2['min_src_histogram'], v2['floor_saturation']))
        print('  flip_counts=%s' % json.dumps(v2['flip_counts']))
    else:
        print('\n== v2 shadow ==\n  (no sizing.v2 rows yet — pre-S3 journals)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--journal-dir',
                    default=os.path.join(_REPO, 'journals'))
    ap.add_argument('--book', choices=['crypto', 'stock', 'all'],
                    default='all')
    ap.add_argument('--json', nargs='?', const='-', default=None,
                    metavar='PATH',
                    help='write the report as JSON to PATH; bare --json '
                         '(or --json -) prints one JSON object to stdout')
    args = ap.parse_args(argv)
    buys, zero_skips, malformed, n_files, n_nosz = _load_rows(
        args.journal_dir, args.days, args.book)
    rep = build_report(buys, zero_skips, malformed, n_files,
                       args.days, args.book, n_buy_rows_without_sizing=n_nosz)
    if args.json is None:
        print_report(rep)
    elif args.json == '-':
        print(json.dumps(rep, default=str))
    else:
        tmp = '%s.%d.tmp' % (args.json, os.getpid())
        try:
            with open(tmp, 'w') as f:
                json.dump(rep, f, default=str, indent=2)
            os.replace(tmp, args.json)
        finally:
            # a failed replace (e.g. PATH is a directory) must not leak tmp
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass
        print_report(rep)
        print('\n[sizing_cofire_report] wrote %s' % args.json)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise         # argparse: --help exits 0, a bad argument exits 2
    except Exception as e:
        print('sizing_cofire_report failed: %s' % e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
