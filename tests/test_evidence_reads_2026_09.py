"""scripts/evidence_reads.py — one-command runbook evidence reads (2026-09).

Dev-Mac safe: stdlib + pytest only. The runner is exercised against a FAKE
repo root in tmp_path whose instruments are tiny stub scripts (write a known
JSON / print known lines / exit with a chosen code / sleep past the timeout).
The last tests read the REAL instrument sources (regex only, no import) to pin
that every default argv names an existing script and only flags its argparse
accepts.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))

import evidence_reads as er  # noqa: E402

INSTRUMENTS = ['beta_ledger.py', 'decision_report.py', 'llm_eval.py',
               'execution_report.py', 'scripts/sizing_cofire_report.py',
               'scripts/reliability_report.py', 'scripts/ic_by_name.py',
               'scripts/rank_gradient_report.py']

# One generic stub: identifies itself by (relative script, --asset/--advisor),
# logs argv/cwd/env, then acts per _behavior.json.
STUB = r'''
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[%(depth)d]
rel = %(rel)r
key = rel
if '--asset' in sys.argv:
    key += '|' + sys.argv[sys.argv.index('--asset') + 1]
if '--advisor' in sys.argv:
    key += '|advisor'
with open(ROOT / '_calls.jsonl', 'a') as f:
    f.write(json.dumps({'key': key, 'argv': sys.argv[1:], 'cwd': os.getcwd(),
                        'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\n')
b = json.loads((ROOT / '_behavior.json').read_text()).get(key, {})
time.sleep(b.get('sleep', 0))
if 'json' in b:
    to = b['json_to']
    if to == 'flag':
        p = Path(sys.argv[sys.argv.index('--json') + 1])
    elif '--out' in sys.argv and not b.get('ignore_out'):
        # the real CLIs' `--out DIR`: DIR/<same file name>, DIR created
        p = Path(sys.argv[sys.argv.index('--out') + 1]) / to
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p = ROOT / to
    p.write_text(json.dumps(b['json']))
sys.stdout.write(b.get('stdout', ''))
sys.stderr.write(b.get('stderr', ''))
sys.exit(b.get('exit', 0))
'''

LLM_READY = {'meta': {}, 'n': 400, 'verdict': 'LLM adds signal',
             'incremental': {'n': 400, 'n_clusters': 150,
                             'effective_n_hint': 25.0, 'verdict': 'x'}}
REL_OUT = ('\n=== Meta-label calibration: legacy vs purged-OOF (n=500) ===\n'
           '  Brier  legacy 0.2  ->  purged 0.19  (better)\n'
           '\n  VERDICT: safe to flip\n')
IC_OUT = ('\n=== Per-name rank-IC (6 names) ===\n  PROMOTE  BTC/USD IC=0.1\n'
          '\n  PROMOTE SET (2): [\'BTC/USD\', \'ETH/USD\']\n')
RANK_OUT = ('\n=== Rank-gradient Stage-0 ===\n  ratio 6-7 / 1-3: 0.4\n'
            '\n  VERDICT: gradient CONFIRMED\n')


def happy_behaviors():
    return {
        'beta_ledger.py': {'json_to': 'flag', 'json': {
            'period': {'n_days': 89, 'n_obs_used': 88, 'obs_per_year_grid': 365.25},
            'joint': {'contamination_delta': -0.01, 'alpha_t_corrected': 1.2}}},
        'decision_report.py': {'json_to': 'decision_report.json', 'json': {
            'quality': {'priced': 40, 'unpriced': 3, 'out_of_window': 1,
                        'dropped_null_pred': 0, 'unpriced_rate': 0.07}}},
        'llm_eval.py': {'json_to': 'llm_eval_report.json', 'json': LLM_READY},
        'llm_eval.py|stock': {'json_to': 'llm_eval_report.json', 'json': LLM_READY},
        'llm_eval.py|advisor': {'json_to': 'llm_advisor_report.json',
                                'json': LLM_READY},
        'execution_report.py': {'json_to': 'execution_report.json', 'json': {
            'generated_at': 'x', 'window_days': 30,
            'crypto/buy/entry': {'n': 20, 'mean_bps': 3.0},
            'stock/buy/entry': {'n': 15, 'mean_bps': 2.0},
            'stock/sell/hard_stop': {'n': 9, 'mean_bps': 5.0}}},
        'scripts/sizing_cofire_report.py': {'json_to': 'flag', 'json': {
            'n_buy_rows': 31, 'n_buy_rows_without_sizing': 4}},
        'scripts/reliability_report.py': {'stdout': REL_OUT},
        'scripts/ic_by_name.py': {'stdout': IC_OUT},
        'scripts/rank_gradient_report.py': {'stdout': RANK_OUT},
    }


def make_repo(tmp_path, behaviors=None, stage0=('stage0_preds.json',),
              calib=True):
    root = tmp_path / 'repo'
    (root / 'scripts').mkdir(parents=True)
    for rel in INSTRUMENTS:
        (root / rel).write_text(STUB % {'depth': rel.count('/'), 'rel': rel})
    (root / '_behavior.json').write_text(json.dumps(
        happy_behaviors() if behaviors is None else behaviors))
    for name in stage0:
        (root / name).write_text(json.dumps(
            [{'ts': '2026-01-01', 'symbol': 'A', 'pred': 1, 'signal': 1,
              'fwd_return': 0.1}] * 3))
    if calib:
        (root / 'calib_holdout.json').write_text(
            json.dumps({'p_legacy': [0.5], 'p_purged': [0.5], 'y': [1]}))
    return root


def calls(root):
    p = root / '_calls.jsonl'
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def run(root, *args):
    return er.main(list(args), root=root, python=sys.executable)


def load_summary(out):
    return json.loads((Path(out) / 'summary.json').read_text())


def by_name(summary):
    return ({s['name']: s for s in summary['steps']},
            {r['read']: r for r in summary['readiness']})


# ------------------------------------------------------------ step table

def test_default_step_order_and_days_propagation(tmp_path):
    root = make_repo(tmp_path, stage0=('stage0_preds.json', 'stock_stage0_preds.json'))
    out = tmp_path / 'out'
    steps = er.build_steps(root, out, days=45)
    assert [s['name'] for s in steps] == list(er.STEP_NAMES)
    argv = {s['name']: s['argv'] for s in steps}
    assert argv['beta_ledger'] == ['beta_ledger.py', '--days', '90', '--json',
                                   str(out / 'beta_report.json')]
    for n in ('decision_report', 'llm_eval', 'llm_eval_stock', 'llm_advisor',
              'execution_report', 'sizing_cofire'):
        i = argv[n].index('--days')
        assert argv[n][i + 1] == '45', n
    assert argv['llm_eval_stock'][-2:] == ['--asset', 'stock']
    assert argv['llm_advisor'][-1] == '--advisor'
    # the four root-default instruments are redirected, one subdir per step
    for n in ('decision_report', 'llm_eval', 'llm_eval_stock', 'llm_advisor',
              'execution_report'):
        i = argv[n].index('--out')
        assert argv[n][i + 1] == str(out / n), n
    assert argv['sizing_cofire'][-2:] == ['--json', str(out / 'sizing_cofire.json')]
    assert argv['ic_by_name'] == ['scripts/ic_by_name.py', '--in',
                                  'stage0_preds.json', '--time-key', 'ts']
    assert argv['rank_gradient_stock'] == [
        'scripts/rank_gradient_report.py', '--preds', 'stock_stage0_preds.json',
        '--fwd-bars', '1', '--cost-pct', '0.113',
        '--extra-cols', 'meta_p,pred_thresh_ratio']
    assert '0.6' in argv['rank_gradient']


def test_cost_pct_defaults_match_fees():
    fees = pytest.importorskip('fees')
    for book, v in er.COST_PCT_DEFAULT.items():
        want = fees.round_trip_cost_pct(book, fees.FLAT_SPREAD_PCT[book])
        assert v == pytest.approx(want, abs=1e-9), book


def test_select_only_skip_and_unknown():
    names = er.STEP_NAMES
    sel, err = er._select(names, 'llm_eval,beta_ledger', '')
    assert err is None and sel == ['beta_ledger', 'llm_eval']   # table order
    sel, err = er._select(names, '', 'llm_eval, reliability')
    assert err is None and 'llm_eval' not in sel and 'reliability' not in sel
    assert len(sel) == len(names) - 2
    sel, err = er._select(names, 'nope', '')
    assert sel is None and 'nope' in err


def test_only_runs_just_the_selected_steps(tmp_path):
    root = make_repo(tmp_path)
    out = tmp_path / 'out'
    assert run(root, '--out', str(out), '--only', 'decision_report,llm_eval_stock',
               '--days', '12') == 0
    c = calls(root)
    assert [x['key'] for x in c] == ['decision_report.py', 'llm_eval.py|stock']
    assert all(x['argv'][x['argv'].index('--days') + 1] == '12' for x in c)
    s = load_summary(out)
    assert [st['name'] for st in s['steps']] == ['decision_report', 'llm_eval_stock']


def test_skip_excludes_steps(tmp_path):
    root = make_repo(tmp_path)
    out = tmp_path / 'out'
    assert run(root, '--out', str(out), '--skip',
               'beta_ledger,llm_eval,llm_eval_stock,llm_advisor') == 0
    keys = {x['key'] for x in calls(root)}
    assert not any(k.startswith('llm_eval') or k.startswith('beta') for k in keys)
    assert 'decision_report.py' in keys


# ------------------------------------------------------------ usage / dry-run

@pytest.mark.parametrize('args', [['--days', '0'], ['--timeout', '0'],
                                  ['--only', 'bogus'], ['--skip', 'x,y'],
                                  ['--no-such-flag']])
def test_usage_errors_exit_2(tmp_path, args, capsys):
    root = make_repo(tmp_path)
    assert run(root, '--out', str(tmp_path / 'out'), *args) == 2
    assert not (tmp_path / 'out').exists()
    assert calls(root) == []


def test_dry_run_prints_and_writes_nothing(tmp_path, capsys):
    root = make_repo(tmp_path)
    before = sorted(p.name for p in root.iterdir())
    assert run(root, '--dry-run') == 0
    assert run(root, '--dry-run', '--out', str(tmp_path / 'o')) == 0
    txt = capsys.readouterr().out
    assert 'decision_report.py --days 30' in txt
    assert ('writes %s via --out (repo-root decision_report.json untouched)'
            % (tmp_path / 'o' / 'decision_report' / 'decision_report.json')) in txt
    assert not (tmp_path / 'o').exists()
    assert not (root / 'logs').exists()
    assert calls(root) == []
    assert sorted(p.name for p in root.iterdir()) == before


# ------------------------------------------------------------ full runs

def test_happy_run_ready_parsed_and_captured(tmp_path, capsys):
    root = make_repo(tmp_path)
    assert run(root) == 0
    outs = list((root / 'logs' / 'evidence_reads').iterdir())
    assert len(outs) == 1 and re.fullmatch(r'\d{8}T\d{6}Z', outs[0].name)
    out = outs[0]
    s = load_summary(out)
    steps, rd = by_name(s)
    ran = [n for n in er.STEP_NAMES if n not in ('ic_by_name_stock',
                                                 'rank_gradient_stock')]
    for n in ran:
        st = steps[n]
        assert st['status'] == 'ok' and st['exit_code'] == 0, n
        assert Path(st['stdout_path']).parent == out
        assert Path(st['stdout_path']).exists() and Path(st['stderr_path']).exists()
        assert rd[n]['verdict'] == 'READY', (n, rd[n])
        assert st['argv'][0] == sys.executable
    # every child saw cwd=root and CUDA_VISIBLE_DEVICES=''
    for c in calls(root):
        assert Path(c['cwd']).resolve() == root.resolve() and c['cuda'] == ''
    p = {n: steps[n]['parsed'] for n in steps}
    assert p['beta_ledger'] == {'n_obs_used': 88, 'n_days': 89,
                                'obs_per_year_grid': 365.25,
                                'contamination_delta': -0.01,
                                'alpha_t_corrected': 1.2}
    assert p['decision_report']['priced'] == 40
    assert p['decision_report']['unpriced'] == 3
    assert p['llm_eval']['n_clusters'] == 150 and p['llm_eval']['effective_n'] == 25.0
    assert p['execution_report']['n_buys_with_slippage'] == 35
    assert p['execution_report']['n_fills_with_slippage'] == 44
    assert p['sizing_cofire'] == {'n_buy_rows': 31, 'n_buy_rows_without_sizing': 4}
    assert p['reliability'] == {'n': 500, 'verdict': 'safe to flip'}
    assert p['ic_by_name'] == {'n_names': 6, 'n_promote': 2, 'dump_rows': 3}
    assert p['rank_gradient']['verdict'] == 'gradient CONFIRMED'
    # root-default instruments now write under --out/<step>; the root stays
    # clean and the copy-a-fresh-root-file fallback finds nothing
    assert steps['decision_report']['json_path'] == str(
        out / 'decision_report' / 'decision_report.json')
    assert (out / 'llm_eval' / 'llm_eval_report.json').exists()
    assert (out / 'llm_eval_stock' / 'llm_eval_report.json').exists()
    assert (out / 'llm_advisor' / 'llm_advisor_report.json').exists()
    assert (out / 'execution_report' / 'execution_report.json').exists()
    assert s['root_files_written'] == []
    for f in ('decision_report.json', 'execution_report.json',
              'llm_advisor_report.json', 'llm_eval_report.json'):
        assert not (root / f).exists(), f
    for n in ('decision_report', 'llm_eval', 'llm_eval_stock', 'llm_advisor',
              'execution_report'):
        assert steps[n]['root_json_written'] is None, n
        assert not (out / ('%s.json' % n)).exists(), n
    assert steps['beta_ledger']['json_path'] == str(out / 'beta_report.json')
    assert steps['beta_ledger']['root_json_written'] is None
    # the absent stock dump is a SKIPPED conditional step, never run
    assert steps['ic_by_name_stock']['status'] == 'skipped'
    assert rd['rank_gradient_stock']['verdict'].startswith(
        'SKIPPED (no stock_stage0_preds.json')
    assert 'scripts/ic_by_name.py|' not in {c['key'] for c in calls(root)}
    assert sum(c['key'] == 'scripts/ic_by_name.py' for c in calls(root)) == 1
    assert s['exit_code'] == 0 and s['failed_steps'] == []
    txt = capsys.readouterr().out
    assert '=== evidence reads: readiness ===' in txt and '\x1b[' not in txt


def test_instrument_ignoring_out_is_still_copied_and_flagged(tmp_path, capsys):
    """Fallback kept: an instrument that writes its repo-root file anyway is
    snapshotted into --out and listed under root_files_written."""
    b = happy_behaviors()
    b['decision_report.py']['ignore_out'] = True
    root = make_repo(tmp_path, b)
    out = tmp_path / 'out'
    assert run(root, '--out', str(out), '--only',
               'decision_report,execution_report') == 0
    s = load_summary(out)
    steps, rd = by_name(s)
    assert s['root_files_written'] == [str(root / 'decision_report.json')]
    assert steps['decision_report']['json_path'] == str(out / 'decision_report.json')
    assert rd['decision_report']['verdict'] == 'READY'
    assert steps['execution_report']['root_json_written'] is None
    assert 'WARNING: repo-root report files' in capsys.readouterr().out


def test_custom_summary_path(tmp_path):
    root = make_repo(tmp_path)
    j = tmp_path / 'sum.json'
    assert run(root, '--out', str(tmp_path / 'o'), '--only', 'sizing_cofire',
               '--json', str(j)) == 0
    assert json.loads(j.read_text())['steps'][0]['name'] == 'sizing_cofire'
    assert not (tmp_path / 'o' / 'summary.json').exists()


def test_failed_step_exit_1_and_verdict(tmp_path):
    b = happy_behaviors()
    b['decision_report.py'] = {'exit': 3, 'stderr': 'boom'}
    root = make_repo(tmp_path, b)
    out = tmp_path / 'out'
    assert run(root, '--out', str(out), '--only', 'decision_report,sizing_cofire') == 1
    s = load_summary(out)
    steps, rd = by_name(s)
    assert steps['decision_report']['status'] == 'failed'
    assert rd['decision_report']['verdict'] == 'FAILED (exit 3)'
    assert Path(steps['decision_report']['stderr_path']).read_text() == 'boom'
    assert rd['sizing_cofire']['verdict'] == 'READY'      # later steps still ran
    assert s['failed_steps'] == ['decision_report'] and s['exit_code'] == 1


def test_timeout_kills_step_and_exits_1(tmp_path):
    b = happy_behaviors()
    b['execution_report.py'] = {'sleep': 30}
    root = make_repo(tmp_path, b)
    out = tmp_path / 'out'
    t = time.monotonic()
    assert run(root, '--out', str(out), '--only', 'execution_report',
               '--timeout', '1') == 1
    assert time.monotonic() - t < 15
    steps, rd = by_name(load_summary(out))
    assert steps['execution_report']['status'] == 'timeout'
    assert rd['execution_report']['verdict'] == 'FAILED (timeout after 1.0s)'


def test_parse_failure_is_recorded_not_raised(tmp_path):
    b = happy_behaviors()
    b['llm_eval.py'] = {'json_to': 'llm_eval_report.json', 'json': {'n': 3}}
    b['scripts/ic_by_name.py'] = {'stdout': 'garbage'}
    root = make_repo(tmp_path, b)
    out = tmp_path / 'out'
    assert run(root, '--out', str(out), '--only', 'llm_eval,ic_by_name') == 0
    steps, rd = by_name(load_summary(out))
    assert rd['llm_eval']['verdict'].startswith('PARSE FAILED (KeyError')
    assert rd['ic_by_name']['verdict'].startswith('PARSE FAILED')


def test_no_json_written_is_parse_failure(tmp_path):
    b = happy_behaviors()
    b['decision_report.py'] = {'stdout': 'nothing written'}
    root = make_repo(tmp_path, b)
    out = tmp_path / 'out'
    assert run(root, '--out', str(out), '--only', 'decision_report') == 0
    steps, rd = by_name(load_summary(out))
    assert steps['decision_report']['json_path'] is None
    assert rd['decision_report']['verdict'] == 'PARSE FAILED (no report JSON produced)'


def test_stale_root_report_is_not_attributed_to_the_step(tmp_path):
    b = happy_behaviors()
    b['decision_report.py'] = {'stdout': 'wrote nothing'}
    root = make_repo(tmp_path, b)
    old = root / 'decision_report.json'
    old.write_text(json.dumps({'quality': {'priced': 99}}))
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    out = tmp_path / 'out'
    assert run(root, '--out', str(out), '--only', 'decision_report') == 0
    steps, _ = by_name(load_summary(out))
    assert steps['decision_report']['root_json_written'] is None
    assert not (out / 'decision_report.json').exists()


def test_no_data_and_not_yet_via_stubs(tmp_path):
    b = happy_behaviors()
    b['llm_eval.py'] = {'json_to': 'llm_eval_report.json', 'json': {
        'meta': {}, 'n': 0, 'verdict': 'no_data', 'reason': 'no_journal_entries'}}
    b['scripts/sizing_cofire_report.py'] = {'json_to': 'flag', 'json': {
        'n_buy_rows': 0, 'n_buy_rows_without_sizing': 169}}
    b['decision_report.py'] = {'json_to': 'decision_report.json', 'json': {
        'quality': {'priced': 2, 'unpriced': 167, 'out_of_window': 15,
                    'dropped_null_pred': 152, 'unpriced_rate': 0.988}}}
    b['execution_report.py'] = {
        'json_to': 'execution_report.json',
        'json': {'generated_at': 'x', 'window_days': 30},
        'stdout': 'shortfall section skipped: 0/169 buys carry slippage_bps (x)\n'}
    root = make_repo(tmp_path, b)
    out = tmp_path / 'out'
    assert run(root, '--out', str(out), '--only',
               'llm_eval,sizing_cofire,decision_report,execution_report') == 0
    steps, rd = by_name(load_summary(out))
    assert rd['llm_eval']['verdict'] == 'NO DATA (no_data stub (no_journal_entries))'
    assert rd['sizing_cofire']['verdict'] == 'NO DATA (have n_buy_rows=0)'
    assert steps['sizing_cofire']['parsed']['n_buy_rows_without_sizing'] == 169
    assert rd['decision_report']['verdict'] == 'NOT YET (priced 2 < 30)'
    assert rd['execution_report']['verdict'] == 'NO DATA (have n_buys_with_slippage=0)'
    assert steps['execution_report']['parsed']['n_buys'] == 169


def test_stage0_conditional_steps(tmp_path):
    # crypto dump empty -> SKIPPED (0 rows); stock dump present -> both run.
    root = make_repo(tmp_path, stage0=('stock_stage0_preds.json',))
    (root / 'stage0_preds.json').write_text('[]')
    out = tmp_path / 'out'
    assert run(root, '--out', str(out), '--only',
               'ic_by_name,rank_gradient,ic_by_name_stock,rank_gradient_stock') == 0
    steps, rd = by_name(load_summary(out))
    assert rd['ic_by_name']['verdict'] == 'SKIPPED (empty stage0_preds.json (0 rows))'
    assert steps['rank_gradient']['parsed'] == {'dump_rows': 0}
    assert rd['ic_by_name_stock']['verdict'] == 'READY'
    assert rd['rank_gradient_stock']['observed'] == 'dump_rows=3'
    ran = [c for c in calls(root)]
    assert [c['argv'][:2] for c in ran] == [['--in', 'stock_stage0_preds.json'],
                                            ['--preds', 'stock_stage0_preds.json']]


def test_rank_gradient_exit_codes(tmp_path):
    b = happy_behaviors()
    b['scripts/rank_gradient_report.py'] = {'exit': 1, 'stdout': RANK_OUT.replace(
        'gradient CONFIRMED', 'no gradient')}
    root = make_repo(tmp_path, b)
    out = tmp_path / 'out'
    assert run(root, '--out', str(out), '--only', 'rank_gradient') == 0   # ran, said no
    steps, rd = by_name(load_summary(out))
    assert steps['rank_gradient']['status'] == 'ok'
    assert steps['rank_gradient']['parsed']['verdict'] == 'no gradient'
    assert rd['rank_gradient']['verdict'] == 'READY'
    b['scripts/rank_gradient_report.py'] = {'exit': 2}
    (root / '_behavior.json').write_text(json.dumps(b))
    out2 = tmp_path / 'out2'
    assert run(root, '--out', str(out2), '--only', 'rank_gradient') == 0
    _, rd = by_name(load_summary(out2))
    assert rd['rank_gradient']['verdict'] == 'NO DATA (exit 2: input unusable)'


def test_reliability_skipped_without_input(tmp_path):
    root = make_repo(tmp_path, calib=False)
    out = tmp_path / 'out'
    assert run(root, '--out', str(out), '--only', 'reliability') == 0
    steps, rd = by_name(load_summary(out))
    assert steps['reliability']['status'] == 'skipped'
    assert rd['reliability']['verdict'].startswith('SKIPPED (no calib_holdout.json')
    assert calls(root) == []


# ------------------------------------------------------------ pure parsers

def test_parse_beta_real_shape_and_missing_optional():
    js = {'period': {'n_days': 61, 'n_obs_used': 60, 'obs_per_year_grid': 365.25},
          'joint': {'alpha_t_corrected': 0.5}}
    p = er.parse_beta(js)
    assert p['n_obs_used'] == 60 and p['contamination_delta'] is None
    assert er.parse_beta({'period': {'n_days': 12}})['n_obs_used'] == 12
    assert '_parse_error' in er._safe_parse('beta', {'joint': {}}, '')


def test_parse_decision_quality_and_stale():
    js = {'quality': {'rows_loaded': 11984, 'priced': 2, 'unpriced': 167,
                      'out_of_window': 15, 'dropped_null_pred': 152,
                      'unpriced_rate': 0.988, 'representative': False}}
    p = er.parse_decision(js)
    assert (p['priced'], p['unpriced'], p['out_of_window'],
            p['dropped_null_pred']) == (2, 167, 15, 152)
    s = er.parse_decision({'stale': True, 'api_available': False})
    assert s['priced'] == 0 and 'stale' in s['no_data']


def test_parse_llm_keys_and_early_return_shape():
    p = er.parse_llm({'incremental': {'n': 3, 'min_n': 60,
                                      'verdict': 'insufficient_power'}})
    assert p['n'] == 3 and p['n_clusters'] is None
    full = er.parse_llm(LLM_READY)      # llm_eval.py:497-502 key names
    assert (full['n'], full['n_clusters'], full['effective_n']) == (400, 150, 25.0)
    assert er.parse_llm({'n': 0, 'verdict': 'no_data', 'reason': 'r'})['n'] == 0


def test_parse_execution_and_sizing_and_stdout_parsers():
    js = {'generated_at': 'x', 'crypto/buy/entry': {'n': 4},
          'crypto/sell/signal': {'n': 2}, 'entry_slippage_by_quote_age': {'a': {'n': 9}},
          'llm_analysis': {'n_calls': 7}}
    p = er.parse_execution(js, '')
    assert p['n_buys_with_slippage'] == 4 and p['n_fills_with_slippage'] == 6
    assert '_parse_error' in er._safe_parse('execution', {'x': 1}, '')
    assert '_parse_error' in er._safe_parse('sizing', {'n_rows': 1}, '')
    assert '_parse_error' in er._safe_parse('reliability', None, 'no header')
    assert er.parse_ic(None, IC_OUT) == {'n_names': 6, 'n_promote': 2}
    assert er.parse_rank(None, RANK_OUT)['ratio_6_7_over_1_3'] == '0.4'


def test_deliberately_renamed_key_fails_the_parse():
    # The sizing JSON key is n_buy_rows; a renamed key must not read as 0.
    assert '_parse_error' in er._safe_parse(
        'sizing', {'n_buy_rows_with_sizing': 40}, '')


# ------------------------------------------------------------ readiness branches

def _step(status='ok', parsed=None, **kw):
    d = {'status': status, 'parsed': parsed or {}, 'exit_code': 0,
         'timeout': 600.0, 'skip_reason': None}
    d.update(kw)
    return d


@pytest.mark.parametrize('name,step,want', [
    ('llm_eval', _step(parsed={'n': 400, 'n_clusters': 150, 'effective_n': 25.0}),
     'READY'),
    ('llm_eval', _step(parsed={'n': 400, 'n_clusters': 119, 'effective_n': 25.0}),
     'NOT YET (n_clusters 119 < 120)'),
    ('llm_eval', _step(parsed={'n': 30, 'n_clusters': None, 'effective_n': None}),
     'NOT YET (n 30 < 60; n_clusters n/a; effective_n n/a)'),
    ('llm_eval', _step(parsed={'n': 400, 'n_clusters': 150}),
     'PARSE FAILED (missing effective_n)'),
    ('llm_eval', _step(parsed={'n': 0, 'no_data': 'stub'}), 'NO DATA (stub)'),
    ('decision_report', _step(parsed={'priced': 0}), 'NO DATA (have priced=0)'),
    ('decision_report', _step(parsed={}), 'PARSE FAILED (missing priced)'),
    ('decision_report', _step(parsed={'_parse_error': 'KeyError: q'}),
     'PARSE FAILED (KeyError: q)'),
    ('beta_ledger', _step(parsed={'n_obs_used': 59}), 'NOT YET (n_obs_used 59 < 60)'),
    ('beta_ledger', _step(parsed={'n_obs_used': 60}), 'READY'),
    ('sizing_cofire', _step(parsed={'n_buy_rows': 5}), 'NOT YET (n_buy_rows 5 < 30)'),
    ('beta_ledger', _step('failed', exit_code=1), 'FAILED (exit 1)'),
    ('beta_ledger', _step('timeout', timeout=5.0), 'FAILED (timeout after 5.0s)'),
    ('rank_gradient', _step('nodata', exit_code=2), 'NO DATA (exit 2: input unusable)'),
    ('ic_by_name', _step('skipped', skip_reason='no dump'), 'SKIPPED (no dump)'),
])
def test_readiness_verdict_branches(name, step, want):
    assert er.readiness_verdict(name, step)[1] == want


def test_every_step_has_a_sourced_rule():
    for name, rule in er.READINESS_RULES.items():
        assert rule['checks'] and rule['source'] and rule['rule'], name
    llm = er.READINESS_RULES['llm_eval']
    assert dict(llm['checks']) == {'n': 60, 'n_clusters': 120, 'effective_n': 20}
    for n in ('decision_report', 'sizing_cofire', 'beta_ledger', 'execution_report'):
        assert er.READINESS_RULES[n]['source'].startswith('provisional'), n


# ------------------------------------------------------------ real repo contract

def _accepted_flags(src):
    return set(re.findall(r"add_argument\(\s*'(--[a-z0-9-]+)'", src))


def test_default_argv_names_real_instruments_and_accepted_flags(tmp_path):
    out = tmp_path / 'out'
    steps = er.build_steps(REPO, out)
    assert {s['name'] for s in steps} == set(er.STEP_NAMES)
    for s in steps:
        script = REPO / s['argv'][0]
        assert script.is_file(), s['argv'][0]
        flags = _accepted_flags(script.read_text())
        used = [a for a in s['argv'][1:] if a.startswith('--')]
        assert used and set(used) <= flags, (s['name'], set(used) - flags)


def test_root_json_names_match_the_instruments_output_code():
    """Every root_json the wrapper snapshots is the literal the instrument
    writes under BASE_DIR (so a rename there breaks this, not the read)."""
    for s in er.build_steps(REPO, Path('/nonexistent')):
        if s['root_json']:
            src = (REPO / s['argv'][0]).read_text()
            assert re.search(r"BASE_DIR / ['\"]%s['\"]" % re.escape(s['root_json']),
                             src), (s['name'], s['root_json'])


def test_stage0_dump_names_match_backtest():
    src = (REPO / 'backtest.py').read_text()
    assert "stage0_preds.json\")" in src or "stage0_preds.json')" in src
    assert [f for _, f, _ in er.STAGE0_DUMPS] == ['stage0_preds.json',
                                                   'stock_stage0_preds.json']


def test_script_imports_stdlib_only():
    src = (REPO / 'scripts' / 'evidence_reads.py').read_text()
    mods = set(re.findall(r'^(?:import|from) ([a-zA-Z_]+)', src, re.M))
    assert mods <= {'argparse', 'datetime', 'json', 'os', 're', 'shutil',
                    'signal', 'subprocess', 'sys', 'time', 'pathlib'}, mods


# ------------------------------------------------------------ readiness ETA (W12)
# Each readiness row carries eta_days (float >= 0 or null) + eta_basis, fit
# from PRIOR runs' summary.json files under --history-dir (default: parent
# of --out), ordered by the generated_at INSIDE each file.

import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402

_UTC = _dt.timezone.utc


def _ago(days=0.0, hours=0.0):
    return _dt.datetime.now(_UTC) - _dt.timedelta(days=days, hours=hours)


def write_hist(hdir, dirname, ts, parsed_by_step, days=30, beta_days=90):
    """One synthetic prior run: hdir/dirname/summary.json."""
    d = Path(hdir) / dirname
    d.mkdir(parents=True, exist_ok=True)
    steps = [{'name': n, 'status': 'ok', 'parsed': p}
             for n, p in parsed_by_step.items()]
    (d / 'summary.json').write_text(json.dumps(
        {'generated_at': ts.isoformat(), 'days': days, 'beta_days': beta_days,
         'steps': steps, 'readiness': []}))
    return d


def _decision_behaviors(priced):
    b = happy_behaviors()
    b['decision_report.py'] = {'json_to': 'decision_report.json', 'json': {
        'quality': {'priced': priced, 'unpriced': 1, 'out_of_window': 0,
                    'dropped_null_pred': 0, 'unpriced_rate': 0.1}}}
    return b


def _run_decision(tmp_path, priced, *extra):
    root = tmp_path / 'repo'
    if root.exists():       # second run in one test: same stub root
        (root / '_behavior.json').write_text(json.dumps(_decision_behaviors(priced)))
    else:
        root = make_repo(tmp_path, _decision_behaviors(priced))
    out = tmp_path / 'runs' / 'current'
    rc = run(root, '--out', str(out), '--only', 'decision_report', *extra)
    return rc, by_name(load_summary(out))[1]['decision_report']


def test_eta_zero_prior_runs_is_no_history(tmp_path):
    rc, row = _run_decision(tmp_path, 10)
    assert rc == 0 and row['verdict'] == 'NOT YET (priced 10 < 30)'
    assert row['eta_days'] is None and row['eta_basis'] == 'no history'


def test_eta_one_prior_run_linear(tmp_path):
    write_hist(tmp_path / 'runs', 'a', _ago(days=2), {'decision_report': {'priced': 4}})
    _, row = _run_decision(tmp_path, 10)
    # rate = (10 - 4) / 2 d = 3/d ; eta = (30 - 10) / 3 = 6.67 d
    assert row['eta_days'] == pytest.approx(20 / 3, abs=0.02)
    assert row['eta_basis'] == 'linear/2 runs over 2.0 d'


def test_eta_two_prior_runs_uses_earliest_and_latest(tmp_path):
    h = tmp_path / 'runs'
    write_hist(h, 'a', _ago(days=4), {'decision_report': {'priced': 2}})
    write_hist(h, 'b', _ago(days=1), {'decision_report': {'priced': 9}})  # ignored mid-point
    _, row = _run_decision(tmp_path, 10)
    # rate = (10 - 2) / 4 = 2/d ; eta = 20 / 2 = 10 d
    assert row['eta_days'] == pytest.approx(10.0, abs=0.02)
    assert row['eta_basis'] == 'linear/3 runs over 4.0 d'


def test_eta_five_prior_runs_and_window(tmp_path):
    h = tmp_path / 'runs'
    for i, (d, v) in enumerate([(10, 0), (8, 1), (5, 4), (4, 5), (3, 6), (2, 7), (1, 8)]):
        write_hist(h, 'r%d' % i, _ago(days=d), {'decision_report': {'priced': v}})
    # default window 5: prior points at 5..1 d + this run -> 6 points, 5 d
    _, row = _run_decision(tmp_path, 9)
    assert row['eta_basis'] == 'linear/6 runs over 5.0 d'
    assert row['eta_days'] == pytest.approx((30 - 9) / ((9 - 4) / 5.0), abs=0.05)
    # --eta-window 7 reaches the 10-day-old run
    rc, row = _run_decision(tmp_path, 9, '--eta-window', '7')
    assert rc == 0 and row['eta_basis'] == 'linear/8 runs over 10.0 d'
    assert row['eta_days'] == pytest.approx((30 - 9) / 0.9, abs=0.1)


def test_eta_exactly_five_prior_runs(tmp_path):
    h = tmp_path / 'runs'
    for i, d in enumerate([5, 4, 3, 2, 1]):
        write_hist(h, 'r%d' % i, _ago(days=d), {'decision_report': {'priced': 10 - d}})
    _, row = _run_decision(tmp_path, 10)
    assert row['eta_basis'] == 'linear/6 runs over 5.0 d'
    assert row['eta_days'] == pytest.approx(20.0, abs=0.05)


@pytest.mark.parametrize('prior', [10, 14])      # flat, decreasing
def test_eta_flat_or_decreasing_is_no_accrual(tmp_path, prior):
    write_hist(tmp_path / 'runs', 'a', _ago(days=3), {'decision_report': {'priced': prior}})
    _, row = _run_decision(tmp_path, 10)
    assert row['eta_days'] is None and row['eta_basis'] == 'no accrual'


def test_eta_history_under_6h_is_no_history(tmp_path):
    h = tmp_path / 'runs'
    write_hist(h, 'a', _ago(hours=5.5), {'decision_report': {'priced': 1}})
    write_hist(h, 'b', _ago(hours=2), {'decision_report': {'priced': 5}})
    _, row = _run_decision(tmp_path, 10)
    assert row['eta_days'] is None and row['eta_basis'] == 'no history'


def test_eta_corrupt_and_absent_history_skipped_with_warning(tmp_path, capsys):
    h = tmp_path / 'runs'
    write_hist(h, 'good', _ago(days=2), {'decision_report': {'priced': 4}})
    (h / 'bad').mkdir()
    (h / 'bad' / 'summary.json').write_text('{not json')
    (h / 'nots').mkdir()
    (h / 'nots' / 'summary.json').write_text(json.dumps({'generated_at': 'yesterday',
                                                         'steps': []}))
    (h / 'empty_dir').mkdir()                         # absent summary: silent
    (h / 'stray.txt').write_text('x')
    rc, row = _run_decision(tmp_path, 10)
    assert rc == 0
    assert row['eta_basis'] == 'linear/2 runs over 2.0 d'
    err = capsys.readouterr().err
    assert 'skipping history file %s' % (h / 'bad' / 'summary.json') in err
    assert 'skipping history file %s' % (h / 'nots' / 'summary.json') in err
    assert 'empty_dir' not in err


def test_eta_orders_by_internal_timestamp_not_dirname(tmp_path):
    h = tmp_path / 'runs'
    # dir names sort opposite to the recorded time
    write_hist(h, 'z_old', _ago(days=4), {'decision_report': {'priced': 2}})
    write_hist(h, 'a_new', _ago(days=1), {'decision_report': {'priced': 20}})
    _, row = _run_decision(tmp_path, 10, '--eta-window', '1')
    # window 1 = the LATEST prior run by internal time (1 d ago, 20) -> falling
    assert row['eta_basis'] == 'no accrual'
    _, row = _run_decision(tmp_path, 10, '--eta-window', '2')
    assert row['eta_basis'] == 'linear/3 runs over 4.0 d'


def test_eta_ignores_other_window_runs_own_file_and_future(tmp_path):
    h = tmp_path / 'runs'
    write_hist(h, 'd200', _ago(days=3), {'decision_report': {'priced': 1}}, days=200)
    write_hist(h, 'future', _ago(days=-1), {'decision_report': {'priced': 1}})
    # a stale summary already in this run's own --out dir is not "history"
    write_hist(h, 'current', _ago(days=5), {'decision_report': {'priced': 0}})
    _, row = _run_decision(tmp_path, 10)
    assert row['eta_days'] is None and row['eta_basis'] == 'no history'


def test_eta_history_dir_flag_and_default_parent(tmp_path):
    other = tmp_path / 'elsewhere'
    write_hist(other, 'a', _ago(days=2), {'decision_report': {'priced': 4}})
    _, row = _run_decision(tmp_path, 10)
    assert row['eta_basis'] == 'no history'          # default = parent of --out
    _, row = _run_decision(tmp_path, 10, '--history-dir', str(other))
    assert row['eta_basis'] == 'linear/2 runs over 2.0 d'


def test_eta_ready_nodata_failed_skipped_rows(tmp_path):
    b = happy_behaviors()
    b['scripts/sizing_cofire_report.py'] = {'json_to': 'flag', 'json': {
        'n_buy_rows': 0, 'n_buy_rows_without_sizing': 9}}
    b['execution_report.py'] = {'exit': 3}
    root = make_repo(tmp_path, b, calib=False)
    h = tmp_path / 'runs'
    write_hist(h, 'a', _ago(days=2), {'sizing_cofire': {'n_buy_rows': 3},
                                      'beta_ledger': {'n_obs_used': 10}})
    out = h / 'cur'
    assert run(root, '--out', str(out), '--only',
               'beta_ledger,sizing_cofire,execution_report,reliability') == 1
    rd = by_name(load_summary(out))[1]
    assert (rd['beta_ledger']['eta_days'], rd['beta_ledger']['eta_basis']) == (None, 'ready')
    assert rd['sizing_cofire']['verdict'].startswith('NO DATA')
    assert (rd['sizing_cofire']['eta_days'], rd['sizing_cofire']['eta_basis']) == (
        None, 'no accrual')
    assert (rd['execution_report']['eta_days'],
            rd['execution_report']['eta_basis']) == (None, 'n/a')
    assert (rd['reliability']['eta_days'], rd['reliability']['eta_basis']) == (None, 'n/a')


def test_eta_nodata_without_history_is_no_history():
    now = _dt.datetime.now(_UTC)
    step = {'status': 'nodata', 'exit_code': 2, 'parsed': {'dump_rows': 3}}
    v = er.readiness_verdict('rank_gradient', step)[1]
    assert er.readiness_eta('rank_gradient', step, v, now, [], {}) == (None, 'no history')


def test_eta_llm_slowest_short_check_binds():
    now = _dt.datetime.now(_UTC)
    hist = [(now - _dt.timedelta(days=10),
             {'days': 30, 'steps': [{'name': 'llm_eval', 'parsed': {
                 'n': 20, 'n_clusters': 40, 'effective_n': 5.0}}]})]
    step = {'status': 'ok', 'parsed': {'n': 70, 'n_clusters': 80, 'effective_n': 10.0}}
    v = er.readiness_verdict('llm_eval', step)[1]
    assert v.startswith('NOT YET (n_clusters 80 < 120; effective_n 10 < 20')
    eta, basis = er.readiness_eta('llm_eval', step, v, now, hist, {'days': 30})
    # n met; clusters 4/d -> 10 d; eff_n 0.5/d -> 20 d -> 20 binds
    assert eta == pytest.approx(20.0) and basis == 'linear/2 runs over 10.0 d'
    # one binding check without accrual -> the row has no ETA
    hist[0][1]['steps'][0]['parsed']['effective_n'] = 12.0
    assert er.readiness_eta('llm_eval', step, v, now, hist, {'days': 30}) == (
        None, 'no accrual')


def test_eta_window_usage_error(tmp_path):
    root = make_repo(tmp_path)
    assert run(root, '--out', str(tmp_path / 'o'), '--eta-window', '0') == 2
    assert not (tmp_path / 'o').exists()


def test_eta_column_printed(tmp_path, capsys):
    write_hist(tmp_path / 'runs', 'a', _ago(days=2), {'decision_report': {'priced': 4}})
    _run_decision(tmp_path, 10)
    txt = capsys.readouterr().out
    head = [l for l in txt.splitlines() if l.startswith('read ')][0]
    assert head.split() == ['read', 'status', 'secs', 'rss_mb', 'ETA', 'verdict']
    line = [l for l in txt.splitlines() if l.startswith('decision_report ')][-1]
    assert ' 6.7d  NOT YET (priced 10 < 30)' in line
    assert '[eta: linear/2 runs over 2.0 d]' in txt


# Golden: the pre-W12 wrapper's summary.json (volatile fields dropped, paths
# normalised) for a fixed stub root, sha256-pinned. Generated once from the
# pre-edit script; with no history the new wrapper must reproduce it exactly
# once eta_days / eta_basis are removed from the readiness rows.
_GOLDEN_SHA256 = "ffa470b88eb415fd5f0ee871843e57ba29ed40a1f05dfaf2bc54d8784706623c"


def _golden_repo(tmp_path):
    b = happy_behaviors()
    b['decision_report.py'] = {'json_to': 'decision_report.json', 'json': {
        'quality': {'priced': 2, 'unpriced': 167, 'out_of_window': 15,
                    'dropped_null_pred': 152, 'unpriced_rate': 0.988}}}
    b['llm_eval.py|stock'] = {'json_to': 'llm_eval_report.json', 'json': {
        'incremental': {'n': 5, 'min_n': 60, 'verdict': 'insufficient_power'}}}
    b['llm_eval.py'] = {'json_to': 'llm_eval_report.json', 'json': {
        'meta': {}, 'n': 0, 'verdict': 'no_data', 'reason': 'no_journal_entries'}}
    b['scripts/sizing_cofire_report.py'] = {'json_to': 'flag', 'json': {
        'n_buy_rows': 0, 'n_buy_rows_without_sizing': 169}}
    b['scripts/rank_gradient_report.py'] = {'exit': 3, 'stderr': 'boom'}
    return make_repo(tmp_path, b, calib=False)


def _golden_norm(summary, tmp_path):
    s = json.loads(json.dumps(summary))
    s.pop('generated_at')
    for st in s['steps']:
        st.pop('seconds')
        st.pop('peak_rss_mb')
    for r in s['readiness']:
        r.pop('eta_days', None)
        r.pop('eta_basis', None)
    txt = json.dumps(s, sort_keys=True, indent=1)
    return txt.replace(sys.executable, '<PY>').replace(str(tmp_path), '<TMP>')


def test_golden_summary_unchanged_minus_eta_keys(tmp_path):
    root = _golden_repo(tmp_path)
    out = tmp_path / 'runs' / 'r1'
    assert run(root, '--out', str(out), '--days', '30') == 1   # rank_gradient failed
    s = load_summary(out)
    basis = {r['read']: (r['verdict'].split(' ')[0], r['eta_days'], r['eta_basis'])
             for r in s['readiness']}
    for name, (v, eta, b) in basis.items():
        assert eta is None, name
        want = {'READY': 'ready', 'NOT': 'no history', 'NO': 'no history'}.get(v, 'n/a')
        assert b == want, (name, v, b)
    assert basis['decision_report'][2] == 'no history'        # NOT YET row
    assert basis['sizing_cofire'][2] == 'no history'          # NO DATA row
    norm = _golden_norm(s, tmp_path)
    assert hashlib.sha256(norm.encode()).hexdigest() == _GOLDEN_SHA256, norm
