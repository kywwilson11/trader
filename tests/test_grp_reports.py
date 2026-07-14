"""Gate-Report scripts: robustness / verdict-clarity / arg-validation tests.

Covers the four measurement-only activation-gate CLIs:
    scripts/cscv_audit.py         (CSCV PBO audit)
    scripts/ic_by_name.py         (per-name rank-IC promotion gate)
    scripts/reliability_report.py (meta calibration legacy-vs-purged)
    scripts/rank_gradient_report.py (rank-gradient Stage-0 gate)

These are pure-numpy/scipy consumers (validation, ic_diagnostic, calibration,
rank_gradient, portfolio_backtest) so the whole suite runs on the dev Mac with no
torch/lightgbm/sklearn. Each test drives the CLI end-to-end as a subprocess and
asserts on exit status + operator-facing text, which is the contract that matters
for scripted `... && enable-flag` use.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / 'scripts'


def _run(script, *args):
    """Run scripts/<script> with args; return (returncode, stdout, stderr)."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


# --------------------------------------------------------------------------- #
# rank_gradient_report.py — the headline: REFUSE stale decision_report input
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('api_available', [False, None])
def test_rgr_refuses_stale_decision_report(tmp_path, api_available):
    """A decision_report placeholder marked stale carries no live buckets; the
    gate must refuse it (exit 2, REFUSED) rather than misreport it as merely
    'insufficient rank coverage' (which is indistinguishable from a real no-go).
    """
    stale = {'generated': '2026-07-13T00:00:00', 'days': 30,
             'api_available': api_available, 'stale': True}
    p = _write(tmp_path, 'stale.json', stale)
    rc, out, err = _run('rank_gradient_report.py', '--buckets', p)
    assert rc == 2, (rc, out, err)
    assert 'REFUSED' in err and 'STALE' in err
    assert f'api_available={api_available!r}' in err
    # Must NOT emit the generic coverage verdict for a stale report.
    assert 'insufficient rank coverage' not in out


def test_rgr_rejects_nondict_buckets(tmp_path):
    """A JSON list/scalar would AttributeError on .get(); expect a named error."""
    p = _write(tmp_path, 'list.json', [1, 2, 3])
    rc, out, err = _run('rank_gradient_report.py', '--buckets', p)
    assert rc == 2, (rc, out, err)
    assert 'must be a JSON object' in err
    assert 'Traceback' not in err


def test_rgr_confirms_gradient_on_nested_conviction(tmp_path):
    """Full decision_report.json (buckets nested under 'conviction') must unwrap,
    print the per-bucket evidence lines, and CONFIRM a real gradient (exit 0)."""
    report = {'generated': '2026-07-13', 'days': 30, 'conviction': {
        'rank_1_3': {'mean_net_pct': 1.20, 'n': 50, 'hit_rate': 0.6},
        'rank_4_5': {'mean_net_pct': 0.30, 'n': 40, 'hit_rate': 0.5},
        'rank_6_7': {'mean_net_pct': -0.10, 'n': 30, 'hit_rate': 0.4}}}
    p = _write(tmp_path, 'dr.json', report)
    rc, out, err = _run('rank_gradient_report.py', '--buckets', p)
    assert rc == 0, (rc, out, err)
    assert 'CONFIRMED' in out
    # regression lock: the per-bucket evidence lines print for nested input
    assert 'rank_1_3: mean_net 1.2' in out
    assert 'rank_4_5: mean_net 0.3' in out


def test_rgr_no_gradient_exits_one(tmp_path):
    """A flat panel (top barely beats bottom) is a ran-but-no-go: exit 1."""
    report = {'conviction': {'rank_1_3': {'mean_net_pct': 0.10, 'n': 50},
                             'rank_6_7': {'mean_net_pct': 0.09, 'n': 30}}}
    p = _write(tmp_path, 'flat.json', report)
    rc, out, err = _run('rank_gradient_report.py', '--buckets', p)
    assert rc == 1, (rc, out, err)
    assert 'NO rank gradient' in out


def test_rgr_bare_buckets_still_supported(tmp_path):
    """A bare (un-nested) rank-bucket dict — rank_gradient_from_panel's own shape
    — must keep working without the 'conviction' wrapper."""
    bare = {'rank_1_3': {'mean_net_pct': 1.0, 'n': 30},
            'rank_6_7': {'mean_net_pct': -0.2, 'n': 30}}
    p = _write(tmp_path, 'bare.json', bare)
    rc, out, err = _run('rank_gradient_report.py', '--buckets', p)
    assert rc == 0, (rc, out, err)
    assert 'CONFIRMED' in out


def test_rgr_preds_holdout_side(tmp_path):
    """Holdout side (--preds) on the documented 'symbol' frame must build the
    panel (ticker_col='symbol') and CONFIRM a monotone gradient — regression lock
    on the KeyError-['Ticker'] fix."""
    rows = []
    for t in range(12):
        ts = f'2026-07-01T{t:02d}:00:00'
        for k in range(8):
            rows.append({'ts': ts, 'symbol': f'S{k}',
                         'signal': float(8 - k), 'fwd_return': float((8 - k) * 0.2)})
    p = _write(tmp_path, 'preds.json', rows)
    rc, out, err = _run('rank_gradient_report.py', '--preds', p)
    assert rc == 0, (rc, out, err)
    assert 'CONFIRMED' in out
    assert 'rank_1_3: mean_net' in out


def test_rgr_requires_one_source(tmp_path):
    """--preds and --buckets are a required mutually-exclusive group."""
    rc, out, err = _run('rank_gradient_report.py')
    assert rc == 2, (rc, out, err)  # argparse usage error


# --------------------------------------------------------------------------- #
# ic_by_name.py — non-list guard, key validation, promotion
# --------------------------------------------------------------------------- #

def test_icbyname_rejects_nonlist(tmp_path):
    """A JSON object (not the required array of rows) must fail loud, not raise a
    cryptic 'string indices must be integers' deep in the library."""
    p = _write(tmp_path, 'obj.json', {'symbol': 'AAA', 'pred': 1, 'fwd_return': 2})
    rc, out, err = _run('ic_by_name.py', '--in', p)
    assert rc != 0, (rc, out, err)
    assert 'expected a JSON array' in err
    assert 'string indices' not in err


def test_icbyname_missing_key_named(tmp_path):
    """A misspelled/absent pred key exits naming the key + available keys, rather
    than silently producing an all-None table."""
    rows = [{'symbol': 'AAA', 'prediction': 1.0, 'fwd_return': 1.0}]
    p = _write(tmp_path, 'rows.json', rows)
    rc, out, err = _run('ic_by_name.py', '--in', p)
    assert rc != 0, (rc, out, err)
    assert "'pred'" in err and 'available' in err


def test_icbyname_promotes_positive_consistent_name(tmp_path):
    """A name whose signal perfectly orders forward return (IC=1, t>>2, full
    consistency) promotes; its perfect-inverse twin holds."""
    rows = []
    for i in range(20):
        rows.append({'symbol': 'AAA', 'pred': float(i), 'fwd_return': float(i)})
        rows.append({'symbol': 'BBB', 'pred': float(i), 'fwd_return': float(-i)})
    p = _write(tmp_path, 'rows.json', rows)
    rc, out, err = _run('ic_by_name.py', '--in', p)
    assert rc == 0, (rc, out, err)
    assert 'PROMOTE  AAA' in out
    assert 'hold     BBB' in out
    assert "PROMOTE SET (1): ['AAA']" in out


def test_icbyname_min_t_hurdle_blocks_thin_name(tmp_path):
    """The t = IC*sqrt(n_finite-1) hurdle must block a positive, consistent, but
    statistically thin name — 3 rows can't clear t>=2 even at IC=1."""
    rows = [{'symbol': 'AAA', 'pred': float(i), 'fwd_return': float(i)}
            for i in range(3)]
    p = _write(tmp_path, 'thin.json', rows)
    rc, out, err = _run('ic_by_name.py', '--in', p)
    assert rc == 0, (rc, out, err)
    assert 'PROMOTE SET (0)' in out


# --------------------------------------------------------------------------- #
# reliability_report.py — payload validation, empty guard, tie semantics
# --------------------------------------------------------------------------- #

def test_reliability_rejects_empty_arrays(tmp_path):
    """Empty arrays pass allclose() as 'identical' and would print a vacuous n=0
    report with a falsely reassuring 'tied' verdict; refuse (exit 2) instead."""
    p = _write(tmp_path, 'empty.json', {'p_legacy': [], 'p_purged': [], 'y': []})
    rc, out, err = _run('reliability_report.py', '--in', p)
    assert rc == 2, (rc, out, err)
    assert 'empty arrays' in err
    assert 'VERDICT' not in out


def test_reliability_rejects_nondict(tmp_path):
    p = _write(tmp_path, 'list.json', [0.1, 0.2, 0.3])
    rc, out, err = _run('reliability_report.py', '--in', p)
    assert rc == 2, (rc, out, err)
    assert 'JSON object' in err


def test_reliability_length_mismatch(tmp_path):
    p = _write(tmp_path, 'mismatch.json',
               {'p_legacy': [0.1, 0.2], 'p_purged': [0.3], 'y': [0, 1]})
    rc, out, err = _run('reliability_report.py', '--in', p)
    assert rc == 2, (rc, out, err)
    assert 'length mismatch' in err


def test_reliability_bins_lt_two_rejected(tmp_path):
    p = _write(tmp_path, 'calib.json',
               {'p_legacy': [0.2, 0.8], 'p_purged': [0.2, 0.8], 'y': [0, 1]})
    rc, out, err = _run('reliability_report.py', '--in', p, '--bins', '1')
    assert rc == 2, (rc, out, err)  # argparse ap.error
    assert 'bins' in err


def test_reliability_identical_reports_tied(tmp_path):
    """Identical legacy/purged arrays carry no comparative evidence: verdict must
    override to 'tied — do NOT flip' with a loud stderr warning."""
    p = [0.2, 0.4, 0.6, 0.8] * 10
    payload = {'p_legacy': p, 'p_purged': p, 'y': [0, 1, 0, 1] * 10}
    fp = _write(tmp_path, 'id.json', payload)
    rc, out, err = _run('reliability_report.py', '--in', fp)
    assert rc == 0, (rc, out, err)
    assert 'tied' in out and 'do NOT flip' in out
    assert 'WARNING' in err


def test_reliability_runs_on_discriminating_dump(tmp_path):
    """A genuinely different pair produces a full report (header + Brier/ECE
    lines) and exits 0 whatever the verdict."""
    y = [i % 2 for i in range(200)]
    p_legacy = [0.5 for _ in y]                       # uninformative
    p_purged = [0.85 if yi else 0.15 for yi in y]     # sharp + correct
    fp = _write(tmp_path, 'disc.json',
                {'p_legacy': p_legacy, 'p_purged': p_purged, 'y': y})
    rc, out, err = _run('reliability_report.py', '--in', fp)
    assert rc == 0, (rc, out, err)
    assert 'Meta-label calibration' in out
    assert 'Brier' in out and 'ECE' in out


# --------------------------------------------------------------------------- #
# cscv_audit.py — arg validation + PBO / n-a paths
# --------------------------------------------------------------------------- #

def test_cscv_odd_ngroups_rejected(tmp_path):
    p = _write(tmp_path, 'blk.json', [[0.1] * 8, [0.2] * 8])
    rc, out, err = _run('cscv_audit.py', '--blocks', p, '--n-groups', '3')
    assert rc == 2, (rc, out, err)
    assert 'n-groups' in err


def test_cscv_returns_nondivisible_nblocks_rejected(tmp_path):
    """--n-blocks not divisible by --n-groups would silently drop trailing blocks
    of every trial; reject up front (before any file read)."""
    p = _write(tmp_path, 'ret.json', [[0.01] * 20, [0.02] * 20])
    rc, out, err = _run('cscv_audit.py', '--returns', p,
                        '--n-blocks', '8', '--n-groups', '6')
    assert rc == 2, (rc, out, err)
    assert 'n-blocks' in err


def test_cscv_na_when_too_few_trials(tmp_path):
    """One trial cannot form the CSCV symmetric split: report n/a, exit 0 (gate
    stays on DSR), no fabricated PBO."""
    p = _write(tmp_path, 'blk1.json', [[1, 2, 3, 4, 5, 6, 7, 8]])
    rc, out, err = _run('cscv_audit.py', '--blocks', p)
    assert rc == 0, (rc, out, err)
    assert 'PBO: n/a' in out


def test_cscv_computes_pbo_on_valid_blocks(tmp_path):
    """Enough distinct non-constant width-8 trials yield a real PBO + verdict."""
    blocks = [
        [0.10, -0.20, 0.30, -0.10, 0.05, -0.05, 0.15, -0.15],
        [-0.05, 0.25, -0.15, 0.10, -0.20, 0.30, -0.10, 0.20],
        [0.20, -0.10, 0.05, -0.25, 0.15, -0.05, 0.10, -0.20],
        [-0.15, 0.10, -0.05, 0.20, -0.10, 0.25, -0.20, 0.05],
    ]
    p = _write(tmp_path, 'blk4.json', blocks)
    rc, out, err = _run('cscv_audit.py', '--blocks', p)
    assert rc == 0, (rc, out, err)
    assert 'Probability of Backtest Overfitting' in out
    assert 'PBO' in out and 'verdict' in out


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
