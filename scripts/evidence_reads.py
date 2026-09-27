#!/usr/bin/env python3
"""One-command runbook evidence reads + a readiness table (measurement-only).

Runs the measurement shelf the activation runbook asks for
(research/campaign_2026-08/03_jetson_runbook.md Phase 0 step 5 + Phase 1)
with the right --days, captures every instrument's stdout/stderr, snapshots
the JSON each one writes, extracts the few numbers that say whether the read
has enough data to be believed, and prints a READY / NOT YET / NO DATA /
FAILED / SKIPPED table.

    python scripts/evidence_reads.py                    # all default steps
    python scripts/evidence_reads.py --days 45 --only llm_eval,decision_report
    python scripts/evidence_reads.py --dry-run          # print the commands, write nothing

Steps run SEQUENTIALLY (never in parallel), each as `sys.executable <script>`
with cwd = repo root, CUDA_VISIBLE_DEVICES='' and a per-step timeout. Output
lands in --out (default logs/evidence_reads/<UTC timestamp>/; logs/ is
gitignored). --days applies to the journal-based reads only; beta_ledger
reads the Alpaca equity curve and keeps the runbook's --days 90.

REPORT REDIRECT: decision_report.py, execution_report.py and llm_eval.py
(plain, --asset stock and --advisor) default to writing their report JSON in
the repo root (decision_report.json, execution_report.json,
llm_eval_report.json, llm_advisor_report.json — gitignored runtime files the
GUI reads). This wrapper passes each of them `--out <out>/<step name>` (one
subdir per step, so llm_eval's plain and --asset stock runs no longer share
one file), so an evidence run leaves the root untouched. The old
copy-a-fresh-root-file fallback is kept: if an instrument ever writes its
repo-root file during its step anyway, that file is still copied into --out
and listed under root_files_written (normally empty now).

Conditional steps: scripts/reliability_report.py runs only when
calib_holdout.json exists in the repo root (no repo code writes it — it is a
Jetson hand-made dump); ic_by_name / rank_gradient_report run per Stage-0
dump that exists and holds >= 1 row (stage0_preds.json = crypto,
stock_stage0_preds.json = stock; backtest.py writes them).

READINESS ETA (measurement-only): every readiness row also carries
`eta_days` (float >= 0, or null) and `eta_basis`. For a NOT YET row the
wrapper reads PRIOR runs' summary.json files under --history-dir (default:
the parent of --out, i.e. logs/evidence_reads/), orders them by the
`generated_at` recorded inside each file, keeps the last --eta-window runs
(default 5) in which the step reported the count, adds this run, and fits a
straight line through the earliest and the latest point: rate =
d(count)/d(days), eta_days = (threshold - count) / rate. With several short
checks (llm_eval) the slowest one binds. eta_basis is
"linear/<k> runs over <d> d", "no history" (no prior point, or the window
spans < 6 h), "no accrual" (rate <= 0), "ready" (READY) or "n/a" (FAILED,
SKIPPED, PARSE FAILED). Journal-window counts are compared only across runs
with the same --days (beta_ledger: the same beta_days). A linear fit on a
rolling-window count is a rough guide, not a forecast. An unreadable history
file is skipped with a warning on stderr; history never fails the run.

Exit status: 0 = every selected step ran (its instrument exited with a code
that means "ran", see Step.ok_exits/nodata_exits) regardless of readiness;
1 = at least one step failed or timed out; 2 = usage error.

Stdlib only — dev-Mac safe; imports nothing from the repo.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DAYS = 30        # runbook 03:44,46 (decision_report / llm_eval --days 30)
BETA_DAYS = 90           # runbook 03:41 (beta_ledger --days 90)
DEFAULT_TIMEOUT = 600.0

# Stage-0 dumps: (step-name suffix, file name, book). backtest.py:910-911
# writes f"{rslot + '_' if rslot else ''}stage0_preds.json" in the repo root.
STAGE0_DUMPS = (('', 'stage0_preds.json', 'crypto'),
                ('_stock', 'stock_stage0_preds.json', 'stock'))

# rank_gradient_report --cost-pct (PERCENT, same units as the dump's
# fwd_return — stage0_preds.py:26-28). The runbook (03:58-59) says only
# "--cost-pct <fees>", so this is a PROVISIONAL evidence_reads default:
# fees.round_trip_cost_pct(book, fees.FLAT_SPREAD_PCT[book]) at static
# (taker, live=False) pricing — crypto (25+25) bps + 0.10 % spread = 0.60 %,
# stock (0.3 + 2*3) bps + 0.05 % spread = 0.113 %. Pinned against fees.py by
# tests/test_evidence_reads_2026_09.py.
COST_PCT_DEFAULT = {'crypto': 0.60, 'stock': 0.113}
COST_PCT_SOURCE = ('provisional, evidence_reads default: '
                   'fees.round_trip_cost_pct(book, FLAT_SPREAD_PCT[book]), '
                   'static taker pricing')

CALIB_INPUT = 'calib_holdout.json'   # scripts/reliability_report.py:7

_PROV = 'provisional, evidence_reads default'

# Pre-registered readiness rules. checks = [(parsed key, minimum)]; the
# FIRST check is the primary count (0 => NO DATA). Every threshold names its
# source; anything no doc fixes is labelled provisional, not given authority.
_LLM_RULE = {
    'rule': 'n >= 60 AND n_clusters >= 120 AND effective_n >= 20',
    'checks': [('n', 60), ('n_clusters', 120), ('effective_n', 20)],
    'source': ('runbook 03:48 (>=120 distinct hourly t0 clusters, n_eff >= 20 '
               '= keep/kill-LLM-spend read); n >= 60 = llm_eval.MIN_POWER_N '
               '(llm_eval.py:73-77)'),
}
_STAGE0_RULE = {
    'rule': 'Stage-0 dump present with >= 1 row',
    'checks': [('dump_rows', 1)],
    'source': 'runbook 03:56-60 (dump lands on the next weekly backtest)',
}
READINESS_RULES = {
    'beta_ledger': {
        'rule': 'n_obs_used >= 60 daily equity obs',
        'checks': [('n_obs_used', 60)],
        'source': _PROV + ' (runbook 03:41-43 names the keys, not a count)'},
    'decision_report': {
        'rule': 'quality.priced >= 30',
        'checks': [('priced', 30)],
        'source': _PROV},
    'llm_eval': _LLM_RULE,
    'llm_eval_stock': _LLM_RULE,
    'llm_advisor': _LLM_RULE,
    'execution_report': {
        'rule': 'buys carrying slippage_bps >= 30',
        'checks': [('n_buys_with_slippage', 30)],
        'source': _PROV},
    'sizing_cofire': {
        'rule': 'buy rows with sizing >= 30 (>= 1 for any read)',
        'checks': [('n_buy_rows', 30)],
        'source': _PROV + ' (co-fire matrix; runbook 03:54-55 gives none)'},
    'reliability': {
        'rule': 'calib_holdout.json present with >= 1 holdout row',
        'checks': [('n', 1)],
        'source': 'scripts/reliability_report.py:7 (input), no power floor'},
    'ic_by_name': _STAGE0_RULE,
    'rank_gradient': _STAGE0_RULE,
    'ic_by_name_stock': _STAGE0_RULE,
    'rank_gradient_stock': _STAGE0_RULE,
}

# Readiness ETA (see module docstring). Only count-valued checks get an ETA.
ETA_WINDOW_DEFAULT = 5
ETA_MIN_SPAN_H = 6.0
ETA_COUNT_KEYS = frozenset({'n', 'n_clusters', 'effective_n', 'priced',
                            'n_obs_used', 'n_buy_rows', 'n_buys_with_slippage',
                            'dump_rows'})
# Steps whose count is taken over the journal window (--days) or the beta
# window (beta_days); a count is only comparable across runs with the same
# window. Stage-0 dumps and reliability are window-free.
_ETA_WINDOW_KEY = {n: 'days' for n in ('decision_report', 'llm_eval',
                                       'llm_eval_stock', 'llm_advisor',
                                       'execution_report', 'sizing_cofire')}
_ETA_WINDOW_KEY['beta_ledger'] = 'beta_days'


# ---------------------------------------------------------------- parsers
# Pure: (report JSON or None, stdout text) -> dict of numbers. They raise
# on a missing required key; _safe_parse turns that into '_parse_error'.

def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def parse_beta(js, stdout=''):
    period = js['period']
    joint = js.get('joint') or {}
    n_used = period.get('n_obs_used', period['n_days'])
    return {'n_obs_used': _num(n_used),
            'n_days': _num(period.get('n_days')),
            'obs_per_year_grid': _num(period.get('obs_per_year_grid')),
            'contamination_delta': _num(joint.get('contamination_delta')),
            'alpha_t_corrected': _num(joint.get('alpha_t_corrected'))}


def parse_decision(js, stdout=''):
    if 'quality' not in js and js.get('stale'):
        return {'no_data': 'stale report (api_available=%r)'
                % js.get('api_available'), 'priced': 0}
    q = js['quality']
    out = {k: _num(q.get(k)) for k in ('priced', 'unpriced', 'out_of_window',
                                         'dropped_null_pred', 'unpriced_rate')}
    out['priced'] = _num(q['priced'])
    out['stale'] = bool(js.get('stale'))
    return out


def parse_llm(js, stdout=''):
    if 'incremental' not in js:
        if js.get('verdict') == 'no_data':
            return {'no_data': 'no_data stub (%s)' % js.get('reason'), 'n': 0}
        raise KeyError('incremental')
    inc = js['incremental']
    return {'n': _num(inc['n']),
            'n_clusters': _num(inc.get('n_clusters')),
            'effective_n': _num(inc.get('effective_n_hint')),
            'verdict': js.get('verdict', inc.get('verdict'))}


_RE_SKIPPED = re.compile(r'shortfall section skipped: (\d+)/(\d+) buys')


def parse_execution(js, stdout=''):
    if 'generated_at' not in js:
        raise KeyError('generated_at')
    groups = {k: v for k, v in js.items()
              if isinstance(v, dict) and k.count('/') == 2 and 'n' in v}
    n_fills = sum(int(v['n']) for v in groups.values())
    n_buy_fills = sum(int(v['n']) for k, v in groups.items()
                      if k.split('/')[1] == 'buy')
    out = {'n_buys_with_slippage': n_buy_fills, 'n_fills_with_slippage': n_fills,
           'n_buys': None}
    m = _RE_SKIPPED.search(stdout or '')
    if m:
        out['n_buys'] = int(m.group(2))
    return out


def parse_sizing(js, stdout=''):
    return {'n_buy_rows': _num(js['n_buy_rows']),
            'n_buy_rows_without_sizing': _num(js.get('n_buy_rows_without_sizing'))}


_RE_REL_N = re.compile(r'calibration: legacy vs purged-OOF \(n=(\d+)\)')
_RE_VERDICT = re.compile(r'VERDICT: (.+)')


def parse_reliability(js, stdout=''):
    m = _RE_REL_N.search(stdout or '')
    if not m:
        raise KeyError('(n=N) header line')
    v = _RE_VERDICT.search(stdout)
    return {'n': int(m.group(1)), 'verdict': v.group(1).strip() if v else None}


_RE_IC_NAMES = re.compile(r'Per-name rank-IC \((\d+) names\)')
_RE_PROMOTE = re.compile(r'PROMOTE SET \((\d+)\)')


def parse_ic(js, stdout=''):
    m = _RE_IC_NAMES.search(stdout or '')
    p = _RE_PROMOTE.search(stdout or '')
    if not (m and p):
        raise KeyError('Per-name rank-IC / PROMOTE SET lines')
    return {'n_names': int(m.group(1)), 'n_promote': int(p.group(1))}


_RE_RATIO = re.compile(r'ratio 6-7 / 1-3: (.+)')


def parse_rank(js, stdout=''):
    v = _RE_VERDICT.search(stdout or '')
    if not v:
        raise KeyError('VERDICT line')
    r = _RE_RATIO.search(stdout)
    return {'verdict': v.group(1).strip(),
            'ratio_6_7_over_1_3': r.group(1).strip() if r else None}


PARSERS = {'beta': parse_beta, 'decision': parse_decision, 'llm': parse_llm,
           'execution': parse_execution, 'sizing': parse_sizing,
           'reliability': parse_reliability, 'ic': parse_ic, 'rank': parse_rank}
_JSON_PARSERS = {'beta', 'decision', 'llm', 'execution', 'sizing'}


def _safe_parse(kind, js, stdout):
    if kind in _JSON_PARSERS and js is None:
        return {'_parse_error': 'no report JSON produced'}
    try:
        return PARSERS[kind](js, stdout)
    except Exception as e:  # a parse failure is recorded, never raised
        return {'_parse_error': '%s: %s' % (type(e).__name__, e)}


# ---------------------------------------------------------------- readiness

def readiness_verdict(name, step):
    """-> (observed str, verdict str) for one step result dict."""
    rule = READINESS_RULES[name]
    status = step.get('status')
    parsed = step.get('parsed') or {}
    if status == 'skipped':
        return '-', 'SKIPPED (%s)' % step.get('skip_reason')
    if status == 'timeout':
        return '-', 'FAILED (timeout after %ss)' % step.get('timeout')
    if status == 'failed':
        return '-', 'FAILED (exit %s)' % step.get('exit_code')
    if status == 'nodata':
        return '-', 'NO DATA (exit %s: input unusable)' % step.get('exit_code')
    if status == 'planned':
        return '-', 'NOT RUN (dry-run)'
    if '_parse_error' in parsed:
        return '-', 'PARSE FAILED (%s)' % parsed['_parse_error']
    vals = [(k, need, _num(parsed.get(k))) for k, need in rule['checks']]
    observed = ', '.join('%s=%s' % (k, 'n/a' if v is None else _fmt(v))
                         for k, _, v in vals)
    k0, _, v0 = vals[0]
    if parsed.get('no_data'):
        return observed, 'NO DATA (%s)' % parsed['no_data']
    if v0 is None:
        return observed, 'PARSE FAILED (missing %s)' % k0
    if v0 <= 0:
        return observed, 'NO DATA (have %s=%s)' % (k0, _fmt(v0))
    short = ['%s %s < %s' % (k, _fmt(v), need) for k, need, v in vals
             if v is not None and v < need]
    missing = [k for k, _, v in vals if v is None]
    if short:
        return observed, 'NOT YET (%s)' % '; '.join(
            short + ['%s n/a' % k for k in missing])
    if missing:
        return observed, 'PARSE FAILED (missing %s)' % ', '.join(missing)
    return observed, 'READY'


def _fmt(v):
    if isinstance(v, float):
        return ('%.3g' % v) if abs(v) < 1e4 else ('%.0f' % v)
    return str(v)


# ---------------------------------------------------------------- readiness ETA

def _parse_ts(s):
    """ISO-8601 text -> aware UTC datetime, or None."""
    if not isinstance(s, str):
        return None
    try:
        t = dt.datetime.fromisoformat(s.strip().replace('Z', '+00:00'))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t


def load_history(hist_dir, exclude=(), before=None):
    """Prior runs under hist_dir/<run>/summary.json -> [(ts, summary)]
    sorted by the generated_at recorded INSIDE each file (never by directory
    name). Absent files are ignored; unreadable ones are skipped with a
    warning on stderr. Files in `exclude` (this run's own) and runs stamped
    at or after `before` are dropped. Never raises."""
    out = []
    try:
        hist_dir = Path(hist_dir)
        if not hist_dir.is_dir():
            return out
        children = sorted(hist_dir.iterdir())
    except OSError as e:
        print('evidence_reads: WARNING: cannot list history dir %s (%s)'
              % (hist_dir, e), file=sys.stderr)
        return out
    skip = set()
    for x in exclude:
        try:
            skip.add(Path(x).resolve())
        except (OSError, RuntimeError):
            pass
    for d in children:
        p = d / 'summary.json'
        try:
            if not p.is_file() or p.resolve() in skip:
                continue
            s = json.loads(p.read_text())
            if not isinstance(s, dict):
                raise ValueError('not a JSON object')
            ts = _parse_ts(s.get('generated_at'))
            if ts is None:
                raise ValueError('no parseable generated_at')
            if not isinstance(s.get('steps'), list):
                raise ValueError('no steps list')
        except Exception as e:
            print('evidence_reads: WARNING: skipping history file %s (%s: %s)'
                  % (p, type(e).__name__, e), file=sys.stderr)
            continue
        if before is not None and ts >= before:
            continue
        out.append((ts, s))
    out.sort(key=lambda x: x[0])
    return out


def _hist_value(summary, name, key, window):
    """The numeric `key` a prior run parsed for step `name`, or None (also
    None when that run used a different journal/beta window)."""
    try:
        wk = _ETA_WINDOW_KEY.get(name)
        if wk is not None and summary.get(wk) != window.get(wk):
            return None
        for st in summary['steps']:
            if isinstance(st, dict) and st.get('name') == name:
                parsed = st.get('parsed')
                if isinstance(parsed, dict) and '_parse_error' not in parsed:
                    return _num(parsed.get(key))
                return None
    except Exception:
        return None
    return None


def _eta_for(name, key, need, cur, now, history, window, eta_window):
    """-> (eta_days or None, basis) for one count check."""
    pts = [(ts, v) for ts, v in
           ((ts, _hist_value(s, name, key, window)) for ts, s in history)
           if v is not None]
    pts = pts[-eta_window:] + [(now, cur)]
    span_d = (pts[-1][0] - pts[0][0]).total_seconds() / 86400.0
    if len(pts) < 2 or span_d * 24.0 < ETA_MIN_SPAN_H:
        return None, 'no history'
    rate = (pts[-1][1] - pts[0][1]) / span_d
    if not rate > 0:
        return None, 'no accrual'
    eta = max(0.0, (need - cur) / rate)
    return round(eta, 2), 'linear/%d runs over %.1f d' % (len(pts), span_d)


def readiness_eta(name, step, verdict, now, history, window,
                  eta_window=ETA_WINDOW_DEFAULT):
    """-> (eta_days, eta_basis) for one readiness row (never raises)."""
    if verdict == 'READY':
        return None, 'ready'
    nodata = verdict.startswith('NO DATA')
    if not (nodata or verdict.startswith('NOT YET')):
        return None, 'n/a'
    rule = READINESS_RULES[name]
    parsed = step.get('parsed') or {}
    if nodata:
        k0, need0 = rule['checks'][0]
        if k0 not in ETA_COUNT_KEYS:
            return None, 'n/a'
        cur = _num(parsed.get(k0))
        _, basis = _eta_for(name, k0, need0, cur if cur is not None else 0,
                            now, history, window, eta_window)
        # a zero count cannot have a positive slope ending at it
        return None, 'no history' if basis == 'no history' else 'no accrual'
    best, nulls = None, []
    for k, need in rule['checks']:
        v = _num(parsed.get(k))
        if v is None or not v < need:
            continue        # met, or n/a (not reported yet): does not bind
        if k not in ETA_COUNT_KEYS:
            return None, 'n/a'
        eta, basis = _eta_for(name, k, need, v, now, history, window,
                              eta_window)
        if eta is None:
            nulls.append(basis)
        elif best is None or eta > best[0]:
            best = (eta, basis)
    if nulls:
        return None, 'no history' if 'no history' in nulls else 'no accrual'
    if best is None:
        return None, 'n/a'
    return best


def _fmt_eta(rd):
    e = rd.get('eta_days')
    return '-' if e is None else '%.1fd' % e


# ---------------------------------------------------------------- steps

def _stage0_rows(path):
    """-> (rows or None, skip reason or None) for a Stage-0 dump."""
    if not path.exists():
        return None, 'no %s (lands on the next weekly backtest)' % path.name
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return None, 'unparseable %s (%s)' % (path.name, type(e).__name__)
    if not isinstance(data, list):
        return None, '%s is not a JSON list' % path.name
    if not data:
        return 0, 'empty %s (0 rows)' % path.name
    return len(data), None


def _calib_rows(path):
    if not path.exists():
        return None, 'no %s in repo root (hand-made Jetson dump)' % path.name
    return None, None


def build_steps(root, out, days=DEFAULT_DAYS):
    """The default step table, in run order. Each step is a dict:
    name, argv (script-relative, no interpreter), parse kind, out_json
    (written via the instrument's own flag), root_json (the repo-root file
    the instrument writes by DEFAULT — with the `--out` passed here it should
    stay untouched; kept as a pollution check + copy fallback), ok_exits /
    nodata_exits,
    skip_reason, extra parsed values from the precheck."""
    root, out = Path(root), Path(out)
    d = str(int(days))
    steps = [
        dict(name='beta_ledger', parse='beta',
             argv=['beta_ledger.py', '--days', str(BETA_DAYS),
                   '--json', str(out / 'beta_report.json')],
             out_json=out / 'beta_report.json'),
        dict(name='decision_report', parse='decision',
             argv=['decision_report.py', '--days', d,
                   '--out', str(out / 'decision_report')],
             out_json=out / 'decision_report' / 'decision_report.json',
             root_json='decision_report.json'),
        dict(name='llm_eval', parse='llm',
             argv=['llm_eval.py', '--days', d, '--out', str(out / 'llm_eval')],
             out_json=out / 'llm_eval' / 'llm_eval_report.json',
             root_json='llm_eval_report.json'),
        dict(name='llm_eval_stock', parse='llm',
             argv=['llm_eval.py', '--days', d,
                   '--out', str(out / 'llm_eval_stock'), '--asset', 'stock'],
             out_json=out / 'llm_eval_stock' / 'llm_eval_report.json',
             root_json='llm_eval_report.json'),
        dict(name='llm_advisor', parse='llm',
             argv=['llm_eval.py', '--days', d,
                   '--out', str(out / 'llm_advisor'), '--advisor'],
             out_json=out / 'llm_advisor' / 'llm_advisor_report.json',
             root_json='llm_advisor_report.json'),
        dict(name='execution_report', parse='execution',
             argv=['execution_report.py', '--days', d,
                   '--out', str(out / 'execution_report')],
             out_json=out / 'execution_report' / 'execution_report.json',
             root_json='execution_report.json'),
        dict(name='sizing_cofire', parse='sizing',
             argv=['scripts/sizing_cofire_report.py', '--days', d,
                   '--json', str(out / 'sizing_cofire.json')],
             out_json=out / 'sizing_cofire.json'),
    ]
    _, why = _calib_rows(root / CALIB_INPUT)
    steps.append(dict(name='reliability', parse='reliability',
                      argv=['scripts/reliability_report.py', '--in', CALIB_INPUT],
                      nodata_exits=(2,), skip_reason=why))
    for suffix, fname, book in STAGE0_DUMPS:
        rows, why = _stage0_rows(root / fname)
        extra = {'dump_rows': rows}
        steps.append(dict(name='ic_by_name' + suffix, parse='ic',
                          argv=['scripts/ic_by_name.py', '--in', fname,
                                '--time-key', 'ts'],
                          skip_reason=why, extra=extra))
        steps.append(dict(name='rank_gradient' + suffix, parse='rank',
                          argv=['scripts/rank_gradient_report.py', '--preds', fname,
                                '--fwd-bars', '1',
                                '--cost-pct', repr(COST_PCT_DEFAULT[book]),
                                '--extra-cols', 'meta_p,pred_thresh_ratio'],
                          # rank_gradient_report: 0 CONFIRMED, 1 ran-but-no-go,
                          # 2 input unusable (scripts/rank_gradient_report.py:47-52)
                          ok_exits=(0, 1), nodata_exits=(2,),
                          skip_reason=why, extra=extra,
                          note='--cost-pct %s = %s' % (COST_PCT_DEFAULT[book],
                                                       COST_PCT_SOURCE)))
    for s in steps:
        s.setdefault('ok_exits', (0,))
        s.setdefault('nodata_exits', ())
        s.setdefault('skip_reason', None)
        s.setdefault('extra', {})
        s.setdefault('out_json', None)
        s.setdefault('root_json', None)
        s.setdefault('note', None)
    return steps


STEP_NAMES = tuple(READINESS_RULES)


# ---------------------------------------------------------------- runner

def _maxrss_mb(ru):
    # ru_maxrss: kilobytes on Linux, bytes on macOS.
    div = 1024.0 * 1024.0 if sys.platform == 'darwin' else 1024.0
    return round(ru.ru_maxrss / div, 1)


def run_step(step, root, out, python, timeout):
    """Run one step sequentially; returns the result dict (never raises)."""
    name = step['name']
    argv = [python] + step['argv']
    res = {'name': name, 'argv': argv, 'cwd': str(root), 'exit_code': None,
           'status': None, 'seconds': 0.0, 'peak_rss_mb': None,
           'stdout_path': None, 'stderr_path': None, 'json_path': None,
           'root_json_written': None, 'skip_reason': step['skip_reason'],
           'note': step['note'], 'timeout': timeout, 'parsed': {}}
    if step['skip_reason']:
        res['status'] = 'skipped'
        res['parsed'] = dict(step['extra'])
        return res
    so, se = out / ('%s.stdout.txt' % name), out / ('%s.stderr.txt' % name)
    res['stdout_path'], res['stderr_path'] = str(so), str(se)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''
    t0_wall = time.time()
    t0 = time.monotonic()
    status = rusage = None
    with open(so, 'w') as fo, open(se, 'w') as fe:
        try:
            proc = subprocess.Popen(argv, cwd=str(root), env=env, stdout=fo,
                                    stderr=fe, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        except OSError as e:
            fe.write('evidence_reads: could not start: %s\n' % e)
            res['status'], res['exit_code'] = 'failed', None
            return res
        timed_out = False
        while True:
            pid, st, ru = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                status, rusage = st, ru
                break
            if time.monotonic() - t0 > timeout:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                _, status, rusage = os.wait4(proc.pid, 0)
                break
            time.sleep(0.1)
        proc.returncode = os.waitstatus_to_exitcode(status)
    res['seconds'] = round(time.monotonic() - t0, 1)
    res['exit_code'] = proc.returncode
    res['peak_rss_mb'] = _maxrss_mb(rusage) if rusage is not None else None
    if timed_out:
        res['status'] = 'timeout'
    elif proc.returncode in step['ok_exits']:
        res['status'] = 'ok'
    elif proc.returncode in step['nodata_exits']:
        res['status'] = 'nodata'
    else:
        res['status'] = 'failed'

    js = None
    jpath = None
    rp = None
    if step['root_json']:
        # Pollution check, independent of --out: only a repo-root file
        # (re)written by THIS step counts (mtime >= its start).
        cand = root / step['root_json']
        if cand.exists() and cand.stat().st_mtime >= t0_wall - 1.0:
            rp = cand
            res['root_json_written'] = str(rp)
    if step['out_json'] is not None and Path(step['out_json']).exists():
        jpath = Path(step['out_json'])
    elif rp is not None:
        # Fallback: the instrument ignored --out and wrote the root file.
        jpath = out / ('%s.json' % name)
        shutil.copy2(rp, jpath)
    if jpath is not None:
        res['json_path'] = str(jpath)
        try:
            js = json.loads(jpath.read_text())
        except Exception as e:
            res['parsed'] = {'_parse_error': 'unreadable JSON: %s' % e}
    if res['status'] == 'ok' and not res['parsed']:
        try:
            stdout = so.read_text(errors='replace')
        except OSError:
            stdout = ''
        res['parsed'] = _safe_parse(step['parse'], js, stdout)
    for k, v in step['extra'].items():
        res['parsed'].setdefault(k, v)
    return res


def _select(names, only, skip):
    """-> (selected names, error or None)."""
    def split(s):
        return [x.strip() for x in s.split(',') if x.strip()] if s else []
    o, k = split(only), split(skip)
    bad = sorted(set(o + k) - set(names))
    if bad:
        return None, 'unknown step name(s): %s (known: %s)' % (
            ', '.join(bad), ', '.join(names))
    sel = [n for n in names if (not o or n in o) and n not in k]
    return sel, None


def _print_commands(steps):
    print('evidence_reads — command table (cwd = repo root, '
          "CUDA_VISIBLE_DEVICES='')")
    for s in steps:
        cmd = ' '.join(s['argv'])
        tail = '  [SKIP: %s]' % s['skip_reason'] if s['skip_reason'] else ''
        print('  %-20s %s%s' % (s['name'], cmd, tail))
        if s['root_json']:
            print('  %-20s   -> writes %s via --out (repo-root %s untouched)'
                  % ('', s['out_json'], s['root_json']))
        if s['note']:
            print('  %-20s   (%s)' % ('', s['note']))


def _print_table(results, readiness):
    print('\n=== evidence reads: readiness ===')
    print('%-20s %-8s %6s %8s %8s  %s' % ('read', 'status', 'secs', 'rss_mb',
                                         'ETA', 'verdict'))
    for r, rd in zip(results, readiness):
        rss = '-' if r['peak_rss_mb'] is None else '%.0f' % r['peak_rss_mb']
        print('%-20s %-8s %6.1f %8s %8s  %s' % (r['name'], r['status'],
                                               r['seconds'], rss, _fmt_eta(rd),
                                               rd['verdict']))
        if rd['observed'] != '-':
            print('%-20s   observed: %s  [rule: %s]  [eta: %s]'
                  % ('', rd['observed'], rd['rule'], rd.get('eta_basis')))


def main(argv=None, root=None, python=None):
    root = Path(root) if root is not None else REPO_ROOT
    python = python or sys.executable
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--days', type=int, default=DEFAULT_DAYS,
                    help='window for the journal-based reads (default %d; '
                         'beta_ledger keeps --days %d)' % (DEFAULT_DAYS, BETA_DAYS))
    ap.add_argument('--out', default=None,
                    help='output dir (default logs/evidence_reads/<UTC ts>/)')
    ap.add_argument('--only', default='', help='comma-separated step names')
    ap.add_argument('--skip', default='', help='comma-separated step names')
    ap.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                    help='per-step timeout in seconds (default %d)' % DEFAULT_TIMEOUT)
    ap.add_argument('--dry-run', action='store_true',
                    help='print the command table and exit 0; writes nothing')
    ap.add_argument('--json', default=None,
                    help='summary JSON path (default <out>/summary.json)')
    ap.add_argument('--history-dir', default=None,
                    help='dir of prior runs (<dir>/<run>/summary.json) for the '
                         'readiness ETA (default: the parent of --out)')
    ap.add_argument('--eta-window', type=int, default=ETA_WINDOW_DEFAULT,
                    help='prior runs used for the ETA slope (default %d)'
                         % ETA_WINDOW_DEFAULT)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    if args.days < 1:
        print('evidence_reads: --days must be >= 1', file=sys.stderr)
        return 2
    if not args.timeout > 0:
        print('evidence_reads: --timeout must be > 0', file=sys.stderr)
        return 2
    if args.eta_window < 1:
        print('evidence_reads: --eta-window must be >= 1', file=sys.stderr)
        return 2
    selected, err = _select(STEP_NAMES, args.only, args.skip)
    if err:
        print('evidence_reads: ' + err, file=sys.stderr)
        return 2

    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(args.out) if args.out else root / 'logs' / 'evidence_reads' / stamp
    if not out.is_absolute():
        out = Path.cwd() / out
    steps = [s for s in build_steps(root, out, args.days) if s['name'] in selected]

    if args.dry_run:
        _print_commands(steps)
        return 0

    out.mkdir(parents=True, exist_ok=True)
    _print_commands(steps)
    results, readiness = [], []
    for s in steps:     # strictly sequential — one instrument at a time
        print('[evidence_reads] %s ...' % s['name'], flush=True)
        r = run_step(s, root, out, python, args.timeout)
        results.append(r)
        observed, verdict = readiness_verdict(s['name'], r)
        rule = READINESS_RULES[s['name']]
        readiness.append({'read': s['name'], 'rule': rule['rule'],
                          'source': rule['source'], 'observed': observed,
                          'verdict': verdict})
    spath = Path(args.json) if args.json else out / 'summary.json'
    now = dt.datetime.now(dt.timezone.utc)
    hdir = Path(args.history_dir) if args.history_dir else out.parent
    try:
        history = load_history(hdir, exclude=(spath, out / 'summary.json'),
                               before=now)
    except Exception as e:      # belt and braces: history never fails a run
        print('evidence_reads: WARNING: history ignored (%s)' % e,
              file=sys.stderr)
        history = []
    window = {'days': args.days, 'beta_days': BETA_DAYS}
    for s, r, rd in zip(steps, results, readiness):
        try:
            rd['eta_days'], rd['eta_basis'] = readiness_eta(
                s['name'], r, rd['verdict'], now, history, window,
                args.eta_window)
        except Exception as e:
            print('evidence_reads: WARNING: ETA for %s failed (%s)'
                  % (s['name'], e), file=sys.stderr)
            rd['eta_days'], rd['eta_basis'] = None, 'n/a'
    _print_table(results, readiness)

    failed = [r['name'] for r in results if r['status'] in ('failed', 'timeout')]
    polluted = sorted({r['root_json_written'] for r in results
                       if r['root_json_written']})
    if polluted:
        print('\nWARNING: repo-root report files (re)written by the '
              'instruments despite --out (copies are in --out): '
              + ', '.join(polluted))
    summary = {'generated_at': now.isoformat(),
               'repo_root': str(root), 'out_dir': str(out), 'days': args.days,
               'beta_days': BETA_DAYS, 'timeout_s': args.timeout,
               'cost_pct': COST_PCT_DEFAULT, 'cost_pct_source': COST_PCT_SOURCE,
               'steps': results, 'readiness': readiness,
               'root_files_written': polluted, 'failed_steps': failed,
               'exit_code': 1 if failed else 0}
    tmp = spath.with_name('%s.%d.tmp' % (spath.name, os.getpid()))
    try:
        tmp.write_text(json.dumps(summary, indent=2, default=str))
        os.replace(tmp, spath)
    finally:
        if tmp.exists():
            tmp.unlink()
    print('\nsummary: %s' % spath)
    return summary['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
