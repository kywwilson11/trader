"""INTEL W17 (Scout B F5/E3, 2026-09): decision_report day-cluster CI.

Report-only, additive: beside every existing `ci90` decision_report now
carries `ci90_dayclust` (cluster bootstrap resampling CALENDAR DAYS), `n_days`,
`dayclust_few_days`, `ci90_dayclust_reason` (when null) and — where a verdict
exists — `verdict_dayclust` from the SAME verdict function; plus the
report-level `verdict_disagreement_rate` printed as ONE new line carrying
the pre-registered E3 rule. Existing ci90 / verdict / insufficient_n keys
and every pre-existing printed line must stay byte-identical (golden test).

Pure numpy/pandas/stdlib; market_data fetches and replay_entry are stubbed
so no policy/fee/strategy_config value can move the golden numbers.
"""

import contextlib
import datetime as dt
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import decision_report
import market_data
from decision_report import (
    _bootstrap_ci, _bucket_stats, _day_cluster_ci, _dayclust_fields,
    _dedup_first_per_day, _disagreement_line, _gate_verdict, _row_day,
    _signal_exit_verdict, _verdict_key, verdict_disagreement,
    conviction_calibration, gate_attribution, signal_exit_audit,
    E3_RULE, MIN_VERDICT_N,
)


RULE_TEXT = ('if ≥20% on the first post-retrain 30-day report → owner: switch '
             'the GUI verdict source to the day-cluster CI and raise '
             'MIN_VERDICT_N to 20')

# Keys the day-cluster addition introduces (stripped for the golden compare).
NEW_ROW_KEYS = ('ci90_dayclust', 'n_days', 'dayclust_few_days',
                'ci90_dayclust_reason', 'verdict_dayclust')
NEW_REPORT_KEYS = ('verdict_disagreement_rate', 'verdict_disagreement')


# ===========================================================================
# Shared synthetic priced set (also used to capture the golden from the
# PRE-edit module — hence `mod` is a parameter, never the import above)
# ===========================================================================

_DAY_SHIFT = [1.2, 0.8, -0.9, 1.5, 0.4, -0.3, 1.1, 0.9, -0.6, 1.3, 0.2, 0.7]


def _scenario_bars():
    idx = pd.date_range('2026-06-01', periods=24 * 14, freq='h', tz='UTC')
    return pd.DataFrame({'Open': 100.0, 'High': 101.0, 'Low': 99.0,
                         'Close': 100.0, 'Volume': 1.0}, index=idx)


def _scenario_rows(bars):
    rows = []
    for d in range(12):
        for sym, h in (('AAA/USD', 5), ('BBB/USD', 9)):
            rows.append({'action': 'skip', 'skip_reason': 'meta_veto',
                         'symbol': sym, 'ts': str(bars.index[d * 24 + h])})
    for d in range(10):
        rows.append({'action': 'skip', 'skip_reason': 'cost_floor',
                     'symbol': 'AAA/USD', 'spread_pct': 0.1,
                     'ts': str(bars.index[d * 24 + 7])})
    for d in range(4):
        rows.append({'action': 'skip', 'skip_reason': 'llm_veto',
                     'symbol': 'BBB/USD', 'ts': str(bars.index[d * 24 + 11])})
    for d in range(12):
        rows.append({'action': 'buy', 'symbol': 'AAA/USD',
                     'pred_return': 0.001 * (d + 1), 'meta_p': 0.35 + 0.02 * d,
                     'entry_rank': 1 + (d % 3),
                     'ts': str(bars.index[d * 24 + 13])})
    for d in range(10):
        rows.append({'action': 'sell', 'exit_reason': 'signal_sell',
                     'symbol': 'BBB/USD', 'pnl_pct': 0.1 * d - 0.3,
                     'ts': str(bars.index[d * 24 + 15])})
    rows.append({'action': 'entry_window', 'asset_type': 'crypto',
                 'admitted_k': 2, 'n_candidates': 5,
                 'veto_counts': {'meta_veto': 2, 'cooldown': 1}})
    return rows


def _fake_net(sym, ts):
    """Deterministic net per (symbol, ts): a day-level shift (shared by
    every name that day) plus a per-symbol and per-hour offset."""
    d = (ts - pd.Timestamp('2026-06-01', tz='UTC')).days
    return (_DAY_SHIFT[d % len(_DAY_SHIFT)]
            + (0.3 if sym == 'AAA/USD' else -0.2) + 0.01 * ts.hour)


def _ts_to_sym(rows):
    out = {}
    for r in rows:
        if r.get('symbol') and r.get('ts'):
            v = pd.Timestamp(r['ts']).value
            assert v not in out
            out[v] = r['symbol']
    return out


def _run_scenario(mod, tmp_path, monkeypatch):
    """Run `mod.run_report(days=1)` on the synthetic priced set; return
    (stdout with tmp_path normalised, report dict)."""
    bars = _scenario_bars()
    rows = _scenario_rows(bars)
    journal_dir = Path(tmp_path) / 'journals'
    journal_dir.mkdir(parents=True, exist_ok=True)
    with open(journal_dir / f'{dt.date.today().isoformat()}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    monkeypatch.setattr(mod, 'JOURNAL_DIR', journal_dir)
    monkeypatch.setattr(mod, 'BASE_DIR', Path(tmp_path))
    monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                        lambda api, s, **k: bars)
    monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                        lambda api, s, **k: bars)

    # replay_entry gets (bars, ts, asset) only; recover the symbol from a
    # UTC-ns -> symbol map (every scenario ts is unique across symbols)
    ts_to_sym = _ts_to_sym(rows)

    def replay_by_ts(b, ts, asset, **k):
        return _fake_net(ts_to_sym[ts.value], ts)
    monkeypatch.setattr(mod, 'replay_entry', replay_by_ts)

    import strategy_config
    monkeypatch.setattr(strategy_config, 'CONVICTION_JOURNAL_ENABLED', True)
    fake = types.ModuleType('dotenv')
    fake.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'dotenv', fake)
    pre_existing = 'trading_utils' in sys.modules
    import trading_utils
    monkeypatch.setattr(trading_utils, 'get_api', lambda: object())
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            report = mod.run_report(days=1)
    finally:
        if not pre_existing:
            sys.modules.pop('trading_utils', None)
    return buf.getvalue().replace(str(tmp_path), '<TMP>'), report


def _strip_new(obj):
    """Deep copy of a report with the day-cluster keys removed."""
    if isinstance(obj, dict):
        return {k: _strip_new(v) for k, v in obj.items()
                if k not in NEW_ROW_KEYS and k not in NEW_REPORT_KEYS}
    if isinstance(obj, list):
        return [_strip_new(v) for v in obj]
    return obj


# Captured 2026-09-27 on the Jetson from the PRE-edit decision_report.py
# (INTEL W17 scratch copy) on the synthetic priced set above; the POST-edit
# module reproduced both exactly (minus the one new line / new keys).
GOLDEN_STDOUT = (
    'methodology 2026-07b: per-episode dedup + bootstrap CIs + full-frame ATR stops + out-of-window exclusion — numbers not comparable to earlier reports\n'
    '\n'
    '=== ADMITTED-K DISTRIBUTION (last 1d) ===\n'
    'crypto: 1 windows, mean k=2.0, P(k>=6)=0%, P(k=0)=0%\n'
    'If P(k>=6) is tiny, the gate stack already concentrates and a top-K cap adds little.\n'
    '\n'
    '=== GATE ATTRIBUTION (last 1d) ===\n'
    'Unpriced gates observed this window (entry_window rollups): cooldown=1\n'
    'fail-closed vetoes: no_pred=0 no_quote=0 (nonzero no_pred = the model returned None for that many candidate-cycles)\n'
    'llm_veto           n=4    (raw=4   ) mean=+0.56% [-0.39, +1.24] hit=75% saved=-2.2%  -> insufficient n (n=4 < 9) — no verdict\n'
    'meta_veto          n=24   (raw=24  ) mean=+0.65% [+0.37, +0.91] hit=79% saved=-15.5%  -> REVIEW (charging admission — CI excludes zero)\n'
    'cost_floor         n=10   (raw=10  ) mean=+0.91% [+0.47, +1.31] hit=80% saved=-9.1%  -> REVIEW (charging admission — CI excludes zero)\n'
    '(fetch failures: 0, horizon-pending: 0, out-of-window: 0, unresolved total: 0)\n'
    'Verdict key: REVIEW = 90% bootstrap CI excludes zero on the positive side (charging admission); OK = CI excludes zero on the negative side (earning its keep); cannot conclude = CI spans zero.\n'
    '\n'
    '=== CONVICTION CALIBRATION (taken entries, kernel-replayed) ===\n'
    'pred_low           n=4     mean=+1.08% [+0.13, +1.75]  hit=75%\n'
    'pred_mid           n=4     mean=+0.95% [+0.48, +1.38]  hit=100%\n'
    'pred_high          n=4     mean=+0.83% [+0.30, +1.43]  hit=75%\n'
    'rank_1_3           n=12    mean=+0.95% [+0.61, +1.30]  hit=83%\n'
    'meta_0.30_0.45     n=6     mean=+0.88% [+0.31, +1.41]  hit=83%\n'
    'meta_0.45_0.60     n=6     mean=+1.03% [+0.56, +1.43]  hit=83%\n'
    '(n=12 priced entries; fetch failures=0, horizon-pending=0, out-of-window=0)\n'
    '\n'
    '=== SIGNAL-EXIT AUDIT (last 1d) ===\n'
    'signal_sells=10 episodes=10 priced=10 cf mean=+0.49% [+0.05, +0.89] cf hit=70% given up=+4.9%\n'
    'Verdict: CHANGE — apply 2-reading confirmation to signal exits (CI excludes zero)\n'
    'Audit bias disclosure: this counterfactual models signal exits fully DISABLED (stops/trail/TP/EOD manage the whole hold) — a STRONGER intervention than the 2-reading confirmation actually under decision, so a CHANGE verdict here is an upper bound on the case for change, not proof the milder fix earns as much.\n'
    '\n'
    'Report: <TMP>/decision_report.json\n'
    ''
)
GOLDEN_JSON = json.loads(r'''
{
 "days": 1,
 "gates": {
  "llm_veto": {
   "vetoes_priced": 4,
   "vetoes_raw": 4,
   "counterfactual_mean_net_pct": 0.56,
   "counterfactual_hit_rate": 0.75,
   "saved_total_pct": -2.24,
   "ci90": [
    -0.39,
    1.235
   ],
   "verdict": "insufficient n (n=4 < 9) \u2014 no verdict",
   "insufficient_n": true
  },
  "meta_veto": {
   "vetoes_priced": 24,
   "vetoes_raw": 24,
   "counterfactual_mean_net_pct": 0.645,
   "counterfactual_hit_rate": 0.792,
   "saved_total_pct": -15.48,
   "ci90": [
    0.373,
    0.909
   ],
   "verdict": "REVIEW (charging admission \u2014 CI excludes zero)",
   "insufficient_n": false
  },
  "cost_floor": {
   "vetoes_priced": 10,
   "vetoes_raw": 10,
   "counterfactual_mean_net_pct": 0.91,
   "counterfactual_hit_rate": 0.8,
   "saved_total_pct": -9.1,
   "ci90": [
    0.47,
    1.31
   ],
   "verdict": "REVIEW (charging admission \u2014 CI excludes zero)",
   "insufficient_n": false
  },
  "_fetch_failed": 0,
  "_horizon_pending": 0,
  "_out_of_window": 0,
  "_unresolved": 0,
  "_cost_floor_spread_coverage": 1.0,
  "_cost_floor_flat_spread": false
 },
 "conviction": {
  "n": 12,
  "_fetch_failed": 0,
  "_horizon_pending": 0,
  "_out_of_window": 0,
  "_unresolved": 0,
  "_malformed_pred_return": 0,
  "_dropped_null_pred": 0,
  "pred_low": {
   "n": 4,
   "mean_net_pct": 1.08,
   "hit_rate": 0.75,
   "ci90": [
    0.13,
    1.755
   ],
   "insufficient_n": true
  },
  "pred_mid": {
   "n": 4,
   "mean_net_pct": 0.955,
   "hit_rate": 1.0,
   "ci90": [
    0.48,
    1.38
   ],
   "insufficient_n": true
  },
  "pred_high": {
   "n": 4,
   "mean_net_pct": 0.83,
   "hit_rate": 0.75,
   "ci90": [
    0.305,
    1.43
   ],
   "insufficient_n": true
  },
  "rank_coverage": {
   "n_total": 12,
   "n_with_rank": 12,
   "stock_with_rank": 0,
   "crypto_with_rank": 12
  },
  "rank_1_3": {
   "n": 12,
   "mean_net_pct": 0.955,
   "hit_rate": 0.833,
   "ci90": [
    0.613,
    1.297
   ],
   "insufficient_n": false
  },
  "meta_0.30_0.45": {
   "n": 6,
   "mean_net_pct": 0.88,
   "hit_rate": 0.833,
   "ci90": [
    0.312,
    1.413
   ],
   "insufficient_n": true
  },
  "meta_0.45_0.60": {
   "n": 6,
   "mean_net_pct": 1.03,
   "hit_rate": 0.833,
   "ci90": [
    0.563,
    1.43
   ],
   "insufficient_n": true
  }
 },
 "admitted_k": {
  "crypto": {
   "windows": 1,
   "mean_admitted_k": 2.0,
   "pct_windows_k_ge_6": 0.0,
   "pct_windows_zero": 0.0,
   "pct_windows_zero_note": "conditional on >=1 evaluatable candidate; cycles where every symbol dropped before evaluation journal NO entry_window row",
   "mean_n_candidates": 5.0,
   "fail_closed": {
    "no_pred": 0,
    "no_quote": 0
   },
   "_malformed_admitted_k": 0,
   "admitted_k_hist": {
    "0": 0,
    "1": 0,
    "2": 1,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8+": 0
   },
   "total_vetoes_by_reason": {
    "meta_veto": 2,
    "cooldown": 1
   }
  }
 },
 "signal_exit": {
  "n_signal_sells": 10,
  "episodes": 10,
  "priced": 10,
  "_fetch_failed": 0,
  "_horizon_pending": 0,
  "_out_of_window": 0,
  "_unresolved": 0,
  "counterfactual_mean_net_pct": 0.49,
  "counterfactual_hit_rate": 0.7,
  "given_up_total_pct": 4.9,
  "ci90": [
   0.049,
   0.89
  ],
  "realized_mean_pnl_pct": 0.15,
  "verdict": "CHANGE \u2014 apply 2-reading confirmation to signal exits (CI excludes zero)",
  "insufficient_n": false
 },
 "quality": {
  "rows_loaded": 61,
  "priced": 60,
  "unpriced": 0,
  "fetch_failed": 0,
  "horizon_pending": 0,
  "out_of_window": 0,
  "dropped_null_pred": 0,
  "unpriced_rate": 0.0,
  "representative": true
 },
 "journal_flags": {
  "conviction_journal_enabled": true,
  "llm_journal_enabled": true
 }
}
''')


# ===========================================================================
# 1. golden byte-identity of every pre-existing printed line + key
# ===========================================================================

def test_golden_existing_output_byte_identical(tmp_path, monkeypatch):
    out, report = _run_scenario(decision_report, tmp_path, monkeypatch)
    new_lines = [ln for ln in out.split('\n')
                 if ln.startswith('verdict_disagreement_rate')]
    assert len(new_lines) == 1                       # exactly ONE new line
    assert RULE_TEXT in new_lines[0]
    old_view = '\n'.join(ln for ln in out.split('\n')
                         if not ln.startswith('verdict_disagreement_rate'))
    assert old_view == GOLDEN_STDOUT
    rep = dict(report)
    rep.pop('generated')
    assert json.loads(json.dumps(_strip_new(rep))) == GOLDEN_JSON
    # and the new keys ARE there, beside every ci90
    for name, g in report['gates'].items():
        if not name.startswith('_'):
            assert 'ci90_dayclust' in g and 'verdict_dayclust' in g
            assert 'n_days' in g
    assert 'verdict_dayclust' in report['signal_exit']
    assert 'verdict_disagreement_rate' in report


# ===========================================================================
# 2. iid rows: the day-cluster CI agrees with the iid CI
# ===========================================================================

def test_one_row_per_day_reproduces_iid_ci():
    vals = np.random.default_rng(7).normal(0.1, 1.0, 40)
    days = [f'2026-06-{i:02d}' if i <= 30 else f'2026-07-{i - 30:02d}'
            for i in range(1, 41)]
    ci_dc, n_days, reason = _day_cluster_ci(vals, days)
    assert n_days == 40 and reason is None
    iid = _bootstrap_ci(vals)
    assert ci_dc[0] == pytest.approx(iid[0], abs=1e-3)
    assert ci_dc[1] == pytest.approx(iid[1], abs=1e-3)


def test_iid_rows_many_per_day_agree_within_mc_tolerance():
    rng = np.random.default_rng(11)
    vals = rng.normal(0.1, 1.0, 240)
    days = [str(dt.date(2026, 3, 1) + dt.timedelta(days=i // 4))
            for i in range(240)]                     # 60 days x 4 rows
    ci_dc, n_days, _ = _day_cluster_ci(vals, days)
    iid = _bootstrap_ci(vals)
    assert n_days == 60
    w_iid = iid[1] - iid[0]
    w_dc = ci_dc[1] - ci_dc[0]
    assert 0.75 < w_dc / w_iid < 1.33
    assert abs((ci_dc[0] + ci_dc[1]) / 2 - (iid[0] + iid[1]) / 2) < 0.25 * w_iid


# ===========================================================================
# 3. strong same-day correlation: day-cluster CI is materially wider
# ===========================================================================

def test_same_day_correlation_widens_ci():
    rng = np.random.default_rng(3)
    vals, days = [], []
    for d, shift in zip(('2026-06-01', '2026-06-02', '2026-06-03'),
                        (-1.0, 0.0, 1.0)):
        for _ in range(10):
            vals.append(shift + rng.normal(0.0, 0.1))
            days.append(d)
    ci_dc, n_days, reason = _day_cluster_ci(vals, days)
    iid = _bootstrap_ci(vals)
    assert n_days == 3 and reason is None
    assert (ci_dc[1] - ci_dc[0]) / (iid[1] - iid[0]) > 1.3
    f = _dayclust_fields(vals, days, verdict_fn=_gate_verdict)
    assert f['dayclust_few_days'] is False           # 3 days: not < 3


# ===========================================================================
# 4. disagreement-rate arithmetic + the printed line
# ===========================================================================

def _g(n, v, vdc):
    return {'vetoes_priced': n, 'verdict': v, 'verdict_dayclust': vdc}


def test_disagreement_rate_arithmetic():
    rev = 'REVIEW (charging admission — CI excludes zero)'
    ok = 'OK (earning its keep — CI excludes zero)'
    cc = 'cannot conclude (CI spans zero)'
    cc1 = ('cannot conclude (single distinct day — day-cluster bootstrap '
           'undefined (one cluster has zero resampling spread)) — no '
           'day-cluster verdict')
    gates = {
        'meta_veto': _g(20, rev, cc),                 # disagree
        'llm_veto': _g(9, ok, ok),                    # agree (n == floor)
        'cost_floor': _g(15, cc, cc1),                # agree (both cannot)
        'q10_tail_veto': _g(12, ok, rev),             # disagree
        'earnings': _g(8, rev, cc),                   # n < 9: excluded
        '_fetch_failed': 3, '_unresolved': 3,         # counters ignored
    }
    chg = 'CHANGE — apply 2-reading confirmation to signal exits (CI excludes zero)'
    sig = {'priced': 11, 'verdict': chg, 'verdict_dayclust': chg}
    d = verdict_disagreement(gates, sig)
    assert d['n_compared'] == 5
    assert d['n_disagree'] == 2
    assert d['disagreeing'] == ['meta_veto', 'q10_tail_veto']
    assert d['rate'] == 0.4
    assert d['min_verdict_n'] == MIN_VERDICT_N == 9
    assert d['rule'] == E3_RULE == RULE_TEXT
    line = _disagreement_line(d)
    assert '\n' not in line
    assert line.startswith('verdict_disagreement_rate (iid vs day-cluster CI): 40% (2/5')
    assert RULE_TEXT in line


def test_disagreement_empty_is_na():
    d = verdict_disagreement({'_fetch_failed': 0}, {'n_signal_sells': 0})
    assert d['rate'] is None and d['n_compared'] == 0
    line = _disagreement_line(d)
    assert 'n/a (0 verdicts with n>=9)' in line and RULE_TEXT in line
    assert verdict_disagreement({}, {})['rate'] is None
    # an insufficient-n signal exit (n < 9) never enters the compared set
    sig = {'priced': 3, 'verdict': _signal_exit_verdict(3, (0.1, 0.2)),
           'verdict_dayclust': _signal_exit_verdict(3, (0.1, 0.2))}
    assert verdict_disagreement({}, sig)['n_compared'] == 0


def test_verdict_key_classes():
    assert _verdict_key('REVIEW (charging admission — CI excludes zero)') == 'REVIEW'
    assert _verdict_key('NO CHANGE — the flip is saving money (CI excludes zero)') == 'NO CHANGE'
    assert _verdict_key('CHANGE — apply 2-reading confirmation to signal exits '
                        '(CI excludes zero)') == 'CHANGE'
    assert _verdict_key(f'insufficient n (n=3 < {MIN_VERDICT_N}) — no verdict') == 'insufficient n'


# ===========================================================================
# 5. degenerate branches
# ===========================================================================

def test_single_day_is_null_with_reason():
    vals = [0.5, 0.7, 0.9, 1.1, 0.4, 0.6, 0.8, 1.0, 1.2, 0.3, 0.2, 0.9]
    ci, n_days, reason = _day_cluster_ci(vals, ['2026-06-01'] * 12)
    assert ci is None and n_days == 1 and 'single distinct day' in reason
    f = _dayclust_fields(vals, ['2026-06-01'] * 12, verdict_fn=_gate_verdict)
    assert f['ci90_dayclust'] is None
    assert f['dayclust_few_days'] is True
    assert 'single distinct day' in f['ci90_dayclust_reason']
    assert f['verdict_dayclust'].startswith('cannot conclude (')
    json.dumps(f)                                     # JSON-serialisable (null)
    # below the verdict floor the verdict function's own refusal is used
    f5 = _dayclust_fields(vals[:5], ['2026-06-01'] * 5, verdict_fn=_gate_verdict)
    assert f5['verdict_dayclust'] == _gate_verdict(5, (0.0, 0.0))


def test_two_days_flagged_few_days_and_nan_dropped():
    vals = [0.1, float('nan'), 0.3, 0.2]
    days = ['2026-06-01', '2026-06-02', '2026-06-02', '2026-06-01']
    ci, n_days, reason = _day_cluster_ci(vals, days)
    assert n_days == 2 and reason is None and ci is not None
    assert ci[0] <= ci[1]
    f = _dayclust_fields(vals, days)
    assert f['dayclust_few_days'] is True and 'verdict_dayclust' not in f
    # a day whose only row is NaN disappears with it
    assert _day_cluster_ci([0.1, float('nan')], ['a', 'b'])[1] == 1
    assert _day_cluster_ci([], []) == (None, 0, 'no priced rows')
    with pytest.raises(ValueError):
        _day_cluster_ci([0.1, 0.2], ['a'])
    # day None = its own singleton cluster
    assert _day_cluster_ci([0.1, 0.2, 0.3], [None, None, 'a'])[1] == 3


def test_row_day_matches_dedup_bucket():
    assert _row_day({'ts': '2026-06-01T23:30:00-05:00'}) == '2026-06-01'
    assert _row_day({'ts': '2026-06-01T23:30:00'}) == '2026-06-01'
    assert _row_day({'ts': '2026-06-02 04:30:00+00:00'}) == '2026-06-02'
    assert _row_day({'ts': 'garbage'}) is None
    assert _row_day({}) is None
    # same convention as _dedup_first_per_day: rows sharing _row_day collapse
    rows = [{'symbol': 'X', 'skip_reason': 'r', 'ts': '2026-06-01T23:30:00-05:00'},
            {'symbol': 'X', 'skip_reason': 'r', 'ts': '2026-06-01T01:00:00-05:00'}]
    assert _row_day(rows[0]) == _row_day(rows[1])
    assert len(_dedup_first_per_day(rows, ['symbol', 'skip_reason'])) == 1


def test_bucket_stats_without_days_key_set_unchanged():
    assert set(_bucket_stats([0.1, 0.2, -0.3])) == {
        'n', 'mean_net_pct', 'hit_rate', 'ci90', 'insufficient_n'}


# ===========================================================================
# 6. wiring: every ci90-bearing dict carries the additive keys
# ===========================================================================

def test_sections_carry_dayclust_keys(tmp_path, monkeypatch):
    bars = _scenario_bars()
    rows = _scenario_rows(bars)
    monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                        lambda api, s, **k: bars)
    monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                        lambda api, s, **k: bars)
    ts_to_sym = _ts_to_sym(rows)
    monkeypatch.setattr(decision_report, 'replay_entry',
                        lambda b, ts, asset, **k: _fake_net(ts_to_sym[ts.value], ts))
    g = gate_attribution(rows, api=object())
    mv = g['meta_veto']
    assert mv['vetoes_priced'] == 24 and mv['n_days'] == 12
    assert mv['verdict'] == _gate_verdict(24, tuple(mv['ci90']))
    assert mv['verdict_dayclust'] == _gate_verdict(24, tuple(mv['ci90_dayclust']))
    assert g['llm_veto']['insufficient_n'] is True
    assert g['llm_veto']['n_days'] == 4
    se = signal_exit_audit(rows, api=object())
    assert se['n_days'] == 10 and 'verdict_dayclust' in se
    conv = conviction_calibration(rows, api=object())
    for k, v in conv.items():
        if isinstance(v, dict) and 'ci90' in v:
            assert 'ci90_dayclust' in v and 'n_days' in v
            assert 'verdict_dayclust' not in v        # buckets carry no verdict
    json.dumps(g), json.dumps(se), json.dumps(conv)


def test_run_report_adds_exactly_one_line_and_report_keys(tmp_path, monkeypatch):
    out, report = _run_scenario(decision_report, tmp_path, monkeypatch)
    hits = [ln for ln in out.split('\n') if 'verdict_disagreement_rate' in ln]
    assert len(hits) == 1 and RULE_TEXT in hits[0]
    d = report['verdict_disagreement']
    assert report['verdict_disagreement_rate'] == d['rate']
    assert d['n_compared'] >= 2                       # meta_veto + cost_floor (+ signal exit)
    on_disk = json.loads((tmp_path / 'decision_report.json').read_text())
    assert on_disk['verdict_disagreement_rate'] == d['rate']
