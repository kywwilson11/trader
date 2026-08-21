"""Campaign Wave A / packet P2 — measurement-stack integrity.

Covers:
  - monitor_drift.py (D32): the retrain flag is (re)written only on a day
    whose OWN level is 'action' (a warn day must not re-create a trigger
    run_pipeline already consumed), and a model-manifest content change
    fences pred_history (truncates it) and resets the action streak so
    post-retrain PSI never scores the old model's predictions against the
    new manifest's deciles. The FIRST time a label is ever checked the
    manifest hash is stamped without fencing (bootstrap — there is no prior
    model to fence against), which is required for the existing pinned
    tests in tests/test_monitor_drift.py (seeded state with no
    manifest_hash) to keep passing.
  - portfolio_backtest.py conviction_gated(strict=...) (D35a): the default
    (strict=False) fail-open behaviour on an ABSENT floored field is pinned
    elsewhere (tests/test_portfolio_backtest_v3.py) and reproduced here for
    completeness; strict=True instead raises ValueError naming extra_cols,
    while a present-but-NaN field still fails closed under strict (no
    exception, just excluded).
  - rank_gradient.py (D35b): rank_gradient_from_panel now also emits ci90 /
    n_eff per bucket (fwd_bars-aware), and rank_gradient_verdict grows an
    opt-in strict mode (min_bucket_n + require_ci) that returns
    INSUFFICIENT EVIDENCE / NOT ESTABLISHED verdicts a point estimate alone
    cannot produce. The historical point-estimate default is unchanged.
  - scripts/rank_gradient_report.py: --fwd-bars / --extra-cols / --strict
    wiring, additive only (default output byte-identical except the new
    ci90= evidence suffix, which lands strictly AFTER the existing
    '(n=...)' substring the b22/grp_reports tests pin).
  - notify.py: a failed CRITICAL send gets one bounded retry; every other
    level stays one-shot (failure-path only, no trading-path change).
"""

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import monitor_drift  # noqa: E402
import notify  # noqa: E402
import portfolio_backtest as pb  # noqa: E402
import rank_gradient  # noqa: E402
from monitor_drift import log_predictions, run_check  # noqa: E402

SCRIPTS = REPO / 'scripts'


# ===========================================================================
# A. monitor_drift.py (D32)
# ===========================================================================

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every monitor_drift file at a temp dir (mirrors
    tests/test_monitor_drift.py's fixture)."""
    monkeypatch.setattr(monitor_drift, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(monitor_drift, '_STATE_FILE',
                        tmp_path / 'drift_state.json')
    return tmp_path


def _edges(values):
    return list(np.percentile(values, np.arange(0, 101, 10)))


def _write_manifest(tmp_path, prefix, deciles):
    p = f'{prefix}_' if prefix else ''
    with open(tmp_path / f'{p}model_v2.manifest.json', 'w') as f:
        json.dump({'holdout': {'pred_deciles': deciles}}, f)


def _fill_history(prefix, values):
    log_predictions(prefix, {f'S{i}': v for i, v in enumerate(values)})


def test_warn_day_does_not_recreate_consumed_flag(sandbox, monkeypatch):
    monkeypatch.setattr(monitor_drift, 'check_drift',
                        lambda prefix: {'psi': 0.18, 'n': 100, 'level': 'warn'})
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    monitor_drift._save_state({'crypto': {'action_days': 2,
                                          'last_action_date': yesterday}})
    r = run_check('', 'crypto')
    assert r is not None and r['level'] == 'warn'
    assert not monitor_drift.retrain_flag_file('').exists()
    st = monitor_drift._load_state()['crypto']
    assert st['action_days'] == 2   # warn neither extends nor resets


def test_action_day_with_streak_still_writes_flag(sandbox, monkeypatch):
    monkeypatch.setattr(monitor_drift, 'check_drift',
                        lambda prefix: {'psi': 0.30, 'n': 100, 'level': 'action'})
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    monitor_drift._save_state({'crypto': {'action_days': 1,
                                          'last_action_date': yesterday}})
    r = run_check('', 'crypto')
    assert r is not None and r['level'] == 'action'
    flag = monitor_drift.retrain_flag_file('')
    assert flag.exists()
    st = monitor_drift._load_state()['crypto']
    assert st['action_days'] == 2


def test_manifest_change_fences_history_and_resets_streak(sandbox):
    rng = np.random.default_rng(11)
    ref_a = rng.normal(0, 1, 5000)
    _write_manifest(sandbox, '', _edges(ref_a))
    mh_a = monitor_drift._manifest_hash('')
    assert mh_a is not None

    _fill_history('', list(rng.normal(0, 1, 200)))
    assert monitor_drift.load_recent_predictions('').size > 0

    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    monitor_drift._save_state({'crypto': {'manifest_hash': mh_a,
                                          'action_days': 1,
                                          'last_action_date': yesterday}})

    # Deploy a new model: different holdout deciles -> different manifest content.
    ref_b = rng.normal(3, 2, 5000)
    _write_manifest(sandbox, '', _edges(ref_b))
    mh_b = monitor_drift._manifest_hash('')
    assert mh_b != mh_a

    r = run_check('', 'crypto')
    assert r is None   # history fenced -> below MIN_LIVE_SAMPLES
    assert monitor_drift.load_recent_predictions('').size == 0

    st = monitor_drift._load_state()['crypto']
    assert st['action_days'] == 0
    assert st['manifest_hash'] == mh_b
    assert 'last_action_date' not in st
    assert not monitor_drift.retrain_flag_file('').exists()


def test_first_run_stamps_hash_without_fencing(sandbox):
    rng = np.random.default_rng(12)
    ref = rng.normal(0, 1, 5000)
    _write_manifest(sandbox, '', _edges(ref))
    _fill_history('', list(rng.normal(0, 1, 200)))   # in-distribution -> 'ok'

    r = run_check('', 'crypto')   # no stored manifest_hash -> bootstrap, no fence
    assert r is not None
    assert r['level'] == 'ok'
    assert monitor_drift.load_recent_predictions('').size > 0   # NOT truncated

    st = monitor_drift._load_state()['crypto']
    assert st['manifest_hash'] == monitor_drift._manifest_hash('')


def test_manifest_hash_helper(sandbox):
    assert monitor_drift._manifest_hash('') is None   # no manifest yet
    _write_manifest(sandbox, '', [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    h1 = monitor_drift._manifest_hash('')
    h1_again = monitor_drift._manifest_hash('')
    assert h1 == h1_again
    _write_manifest(sandbox, '', [0.0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    h2 = monitor_drift._manifest_hash('')
    assert h2 != h1


# ===========================================================================
# B. portfolio_backtest.conviction_gated(strict=...) (D35a)
# ===========================================================================

def test_strict_raises_on_absent_meta_p():
    cands = [{'symbol': 'A', 'signal': 0.9, 'fwd_return': 1.0}]
    with pytest.raises(ValueError, match='extra_cols'):
        pb.conviction_gated(7, meta_floor=0.6, strict=True)(cands)


def test_strict_raises_on_absent_ratio_field():
    cands = [{'symbol': 'A', 'signal': 0.9, 'fwd_return': 1.0}]
    with pytest.raises(ValueError, match='extra_cols'):
        pb.conviction_gated(7, ratio_floor=1.0, strict=True)(cands)


def test_strict_nan_still_fails_closed_no_raise():
    cands = [{'symbol': 'A', 'signal': 0.9, 'fwd_return': 1.0,
              'meta_p': float('nan')}]
    admitted = pb.conviction_gated(7, meta_floor=0.6, strict=True)(cands)
    assert admitted == []


def test_default_fail_open_unchanged():
    cands = [{'symbol': 'A', 'signal': 0.9, 'fwd_return': 1.0},
             {'symbol': 'B', 'signal': 0.8, 'fwd_return': 1.0}]
    admitted = pb.conviction_gated(7, meta_floor=0.99)(cands)
    assert [c['symbol'] for c in admitted] == ['A', 'B']


def test_strict_equals_default_when_fields_present():
    cands = [{'symbol': 'A', 'signal': 0.9, 'fwd_return': 1.0,
              'meta_p': 0.7, 'pred_thresh_ratio': 1.2},
             {'symbol': 'B', 'signal': 0.8, 'fwd_return': 1.0,
              'meta_p': 0.4, 'pred_thresh_ratio': 0.9}]
    default = pb.conviction_gated(7, meta_floor=0.5, ratio_floor=1.0)(cands)
    strict = pb.conviction_gated(7, meta_floor=0.5, ratio_floor=1.0,
                                 strict=True)(cands)
    assert [c['symbol'] for c in default] == [c['symbol'] for c in strict] == ['A']


# ===========================================================================
# C. rank_gradient.py — ci90/n_eff + strict verdict (D35b)
# ===========================================================================

def _make_panel(gradient=True, n_periods=200, k=7, seed=1):
    rng = np.random.default_rng(seed)
    panel = []
    for _ in range(n_periods):
        sig = np.sort(rng.normal(size=k))[::-1]
        if gradient:
            fwd = 0.4 * (sig - sig.mean()) + rng.normal(0, 0.2, k)
        else:
            fwd = 0.1 + rng.normal(0, 0.2, k)
        panel.append([{'symbol': f'S{i}', 'signal': float(sig[i]),
                       'fwd_return': float(fwd[i])} for i in range(k)])
    return panel


def test_from_panel_emits_ci90_and_neff():
    panel = _make_panel()
    buckets = rank_gradient.rank_gradient_from_panel(panel)
    for label, stat in buckets.items():
        assert 'ci90' in stat and len(stat['ci90']) == 2
        lo, hi = stat['ci90']
        assert lo <= stat['mean_net_pct'] <= hi
        assert stat['n_eff'] == stat['n']   # fwd_bars defaults to 1

    buckets3 = rank_gradient.rank_gradient_from_panel(panel, fwd_bars=3)
    for label, stat in buckets3.items():
        assert stat['n_eff'] == round(stat['n'] / 3, 1)


def test_ci_widens_with_fwd_bars():
    panel = _make_panel()
    b1 = rank_gradient.rank_gradient_from_panel(panel, fwd_bars=1)['rank_1_3']
    b24 = rank_gradient.rank_gradient_from_panel(panel, fwd_bars=24)['rank_1_3']
    half1 = b1['ci90'][1] - b1['ci90'][0]
    half24 = b24['ci90'][1] - b24['ci90'][0]
    assert half24 > half1


def test_strict_verdict_insufficient_n():
    buckets = {'rank_1_3': {'mean_net_pct': 0.4, 'n': 10,
                            'ci90': [0.2, 0.6]},
              'rank_6_7': {'mean_net_pct': 0.05, 'n': 10,
                          'ci90': [-0.1, 0.2]}}
    v = rank_gradient.rank_gradient_verdict(
        buckets, min_bucket_n=rank_gradient.MIN_BUCKET_N, require_ci=True)
    assert v['gradient_exists'] is None
    assert 'INSUFFICIENT EVIDENCE' in v['verdict']
    assert 'min_bucket_n' in v['verdict']


def test_strict_verdict_missing_ci():
    buckets = {'rank_1_3': {'mean_net_pct': 0.4, 'n': 50},
              'rank_6_7': {'mean_net_pct': 0.05, 'n': 50}}
    v = rank_gradient.rank_gradient_verdict(
        buckets, min_bucket_n=rank_gradient.MIN_BUCKET_N, require_ci=True)
    assert v['gradient_exists'] is None
    assert 'ci90' in v['verdict']


def test_strict_verdict_ci_overlap_not_confirmed():
    buckets = {'rank_1_3': {'mean_net_pct': 0.4, 'n': 50,
                            'ci90': [-0.1, 0.9]},
              'rank_6_7': {'mean_net_pct': 0.05, 'n': 50}}
    v = rank_gradient.rank_gradient_verdict(buckets, require_ci=True)
    assert v['gradient_exists'] is False
    assert 'NOT ESTABLISHED' in v['verdict']
    assert 'ship NEITHER' in v['verdict']


def test_strict_verdict_confirms_with_ci():
    buckets = {'rank_1_3': {'mean_net_pct': 0.4, 'n': 50,
                            'ci90': [0.2, 0.6]},
              'rank_6_7': {'mean_net_pct': 0.05, 'n': 50,
                          'ci90': [-0.1, 0.2]}}
    v = rank_gradient.rank_gradient_verdict(
        buckets, min_bucket_n=rank_gradient.MIN_BUCKET_N, require_ci=True)
    assert v['gradient_exists'] is True
    assert 'CONFIRMED' in v['verdict']


def test_default_verdict_byte_compatible():
    buckets = {'rank_1_3': {'mean_net_pct': 0.3},
              'rank_6_7': {'mean_net_pct': -0.05}}
    v = rank_gradient.rank_gradient_verdict(buckets)
    assert v['gradient_exists'] is True
    assert 'CONFIRMED' in v['verdict']


# ===========================================================================
# D. scripts/rank_gradient_report.py wiring
# ===========================================================================

def _run_script(*args):
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / 'rank_gradient_report.py'), *map(str, args)],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _monotone_rows(n_periods, k, with_meta_p=False):
    rows = []
    for t in range(n_periods):
        ts = f'2026-07-01T{t:02d}:00:00'
        for i in range(k):
            row = {'ts': ts, 'symbol': f'S{i}', 'signal': float(k - i),
                   'fwd_return': float((k - i) * 0.2)}
            if with_meta_p:
                row['meta_p'] = 0.6
            rows.append(row)
    return rows


def test_report_strict_small_n_exits_1(tmp_path):
    rows = _monotone_rows(n_periods=3, k=7)   # n_1_3 = 9
    p = tmp_path / 'small.json'
    p.write_text(json.dumps(rows))

    rc, out, err = _run_script('--preds', p, '--strict')
    assert rc == 1, (rc, out, err)
    assert 'INSUFFICIENT EVIDENCE' in out

    rc0, out0, err0 = _run_script('--preds', p)
    assert rc0 == 0, (rc0, out0, err0)
    assert 'CONFIRMED' in out0


def test_report_strict_large_n_confirms(tmp_path):
    rows = _monotone_rows(n_periods=16, k=8)   # n_1_3=48, n_6_7=32 (both >=30)
    p = tmp_path / 'large.json'
    p.write_text(json.dumps(rows))

    rc, out, err = _run_script('--preds', p, '--strict')
    assert rc == 0, (rc, out, err)
    assert 'CONFIRMED' in out
    assert 'ci90=' in out


def _ci90_width(stdout, bucket):
    m = re.search(rf'{bucket}: .*ci90=\[([-\d.]+), ([-\d.]+)\]', stdout)
    assert m, stdout
    return float(m.group(2)) - float(m.group(1))


def test_report_extra_cols_and_fwd_bars_plumbed(tmp_path):
    rows = _monotone_rows(n_periods=12, k=8, with_meta_p=True)
    p = tmp_path / 'meta.json'
    p.write_text(json.dumps(rows))

    rc, out, err = _run_script('--preds', p, '--extra-cols', 'meta_p',
                               '--fwd-bars', 3)
    assert rc == 0, (rc, out, err)
    # --fwd-bars must actually reach rank_gradient_from_panel: the ci90 in
    # the evidence line is wider than the default (n_eff = n/3 < n).
    rc1, out1, _ = _run_script('--preds', p, '--extra-cols', 'meta_p')
    assert rc1 == 0
    assert _ci90_width(out, 'rank_1_3') > _ci90_width(out1, 'rank_1_3')

    # In-process companion: the wired dump must also satisfy strict
    # conviction floors once meta_p is carried into the panel.
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df.set_index(pd.to_datetime(df['ts']))
    panel = pb.panel_from_frame(df, 'signal', 'fwd_return', ticker_col='symbol',
                                extra_cols=['meta_p'])
    admitted = pb.conviction_gated(3, meta_floor=0.5, strict=True)(
        sorted(panel[0], key=lambda c: c['signal'], reverse=True))
    assert len(admitted) > 0


# ===========================================================================
# E. notify.py critical retry (failure-path only)
# ===========================================================================

def _isolate_webhook_only(monkeypatch):
    monkeypatch.setenv('TRADER_WEBHOOK_URL', 'http://example.invalid/hook')
    monkeypatch.delenv('TRADER_TELEGRAM_BOT_TOKEN', raising=False)
    monkeypatch.delenv('TRADER_TELEGRAM_CHAT_ID', raising=False)
    monkeypatch.setattr(notify, '_CRITICAL_RETRY_DELAY_SEC', 0.0)


def test_critical_retry_once_then_success(monkeypatch):
    _isolate_webhook_only(monkeypatch)
    calls = []

    def fake_post(url, payload):
        calls.append(1)
        if len(calls) == 1:
            raise OSError('transient')

    monkeypatch.setattr(notify, '_post', fake_post)
    notify._send('x', 'critical')
    assert len(calls) == 2


def test_critical_retry_bounded(monkeypatch):
    _isolate_webhook_only(monkeypatch)
    calls = []

    def fake_post(url, payload):
        calls.append(1)
        raise OSError('down')

    monkeypatch.setattr(notify, '_post', fake_post)
    notify._send('x', 'critical')   # must not raise
    assert len(calls) == 2


def test_warning_not_retried(monkeypatch):
    _isolate_webhook_only(monkeypatch)
    calls = []

    def fake_post(url, payload):
        calls.append(1)
        raise OSError('down')

    monkeypatch.setattr(notify, '_post', fake_post)
    notify._send('x', 'warning')
    assert len(calls) == 1


def test_info_success_single_call(monkeypatch):
    _isolate_webhook_only(monkeypatch)
    calls = []

    def fake_post(url, payload):
        calls.append(1)

    monkeypatch.setattr(notify, '_post', fake_post)
    notify._send('x', 'info')
    assert len(calls) == 1
