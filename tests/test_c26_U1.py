"""Packet c26 U1 — GUI decision truth + measurement reconciliation (B22).

Three sections, all dev-Mac-runnable (numpy/pandas/stdlib only):

A — beta_ledger additive keys: HAC d.o.f.-corrected alpha t, PIT lagged
    trend-conditional betas, median-spacing obs_per_year_grid, and the
    transfer-clean (*_clean) parallel keys — with the core guarantee that
    a plain beta_report() is byte-identical to before.
B — chart_core additions: obs_per_year, artifact_freshness,
    sizing_stack_summary, gate/meta render models, report formatters.
C — gui.py source contracts (AST inspection, PySide6-free — same pattern
    as tests/test_gui_contracts.py).
"""
import ast
import copy
import datetime as dt
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import beta_ledger
import chart_core

REPO = Path(__file__).resolve().parent.parent
GUI_SRC = (REPO / "gui.py").read_text()
GUI_TREE = ast.parse(GUI_SRC)
GUI_LINES = GUI_SRC.splitlines()


def _method_source(name):
    """Source text of a function/method by name, wherever it's nested."""
    for node in ast.walk(GUI_TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return "\n".join(GUI_LINES[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"method {name!r} not found in gui.py")


def _req(a, b):
    """Recursive equality with nan == nan."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_req(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_req(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    return a == b


def _strip_clean(rep):
    """Remove the additive *_clean keys so the remainder can be compared to
    a plain run."""
    rep = copy.deepcopy(rep)
    rep.pop('joint_clean', None)
    for section in ('joint', 'strategy'):
        d = rep.get(section)
        if isinstance(d, dict):
            for k in list(d):
                if k.endswith('_clean') or k == 'contamination_delta':
                    d.pop(k)
    return rep


def _mk_fixture(n=121, seed=0):
    """(equity, bench_prices) on a shared calendar-daily grid."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2026-01-01', periods=n, freq='D')
    eq = pd.Series(100000.0 * np.cumprod(1 + rng.normal(0.0003, 0.004, n)),
                   index=idx, name='equity')
    spy = pd.Series(400.0 * np.cumprod(1 + rng.normal(0.0002, 0.008, n)),
                    index=idx, name='SPY')
    return eq, pd.DataFrame({'SPY': spy})


# ===========================================================================
# Section A — beta_ledger additive keys
# ===========================================================================

def test_a1_alpha_t_corrected_scaling():
    eq, bench = _mk_fixture(121)
    strat_ret = eq.pct_change().dropna()
    bench_ret = bench.pct_change().reindex(strat_ret.index)
    out = beta_ledger.lagged_beta_regression(strat_ret, bench_ret, lags=1)
    n, k = out['n_obs'], 3  # intercept + SPY lag0 + SPY lag1
    for key in ('dof_scale', 'alpha_se_corrected', 'alpha_t_corrected'):
        assert key in out
    scale = math.sqrt(n / (n - k))
    assert out['dof_scale'] == pytest.approx(scale, abs=1e-12)
    assert out['alpha_se_corrected'] == pytest.approx(
        out['alpha_se'] * scale, abs=1e-9)
    assert out['alpha_t_corrected'] == pytest.approx(
        out['alpha_t'] * math.sqrt((n - k) / n), abs=1e-9)
    # OLD alpha_t/alpha_se unchanged vs an independent ols_hac recompute.
    df = pd.concat([strat_ret.rename('_y'), bench_ret], axis=1).dropna()
    cols = [df['SPY'].shift(0).values, df['SPY'].shift(1).values]
    X = np.column_stack([np.ones(len(df))] + cols)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(df['_y'].values)
    beta, se, _ = beta_ledger.ols_hac(X[valid], df['_y'].values[valid])
    assert out['alpha_t'] == pytest.approx(float(beta[0] / se[0]), abs=1e-12)
    assert out['alpha_se'] == pytest.approx(float(se[0]), abs=1e-12)


def test_a2_clean_keys_are_purely_additive():
    eq, bench = _mk_fixture(121)
    plain = beta_ledger.beta_report(eq, bench, lags=1)
    clean = eq.pct_change().rename('clean_ret')
    with_clean = beta_ledger.beta_report(eq, bench, lags=1, clean_ret=clean)
    # plain run carries NO clean keys anywhere
    assert 'joint_clean' not in plain
    assert not any(k.endswith('_clean') or k == 'contamination_delta'
                   for k in plain['joint'])
    assert not any(k.endswith('_clean') for k in plain['strategy'])
    # clean run got them
    assert 'joint_clean' in with_clean
    assert 'alpha_annual_clean' in with_clean['joint']
    # stripping them yields the plain report exactly (nan == nan)
    assert _req(_strip_clean(with_clean), plain)


def test_a3_trend_conditional_lagged_pit():
    n = 320
    idx = pd.date_range('2025-01-01', periods=n, freq='D')
    px = np.empty(n)
    px[:270] = 100.0 - 0.05 * np.arange(270)   # declining => below own SMA200
    px[270:] = 200.0                            # jump above => single crossing
    spy = pd.Series(px, index=idx)
    rng = np.random.default_rng(1)
    eq = pd.Series(100000.0 * np.cumprod(1 + rng.normal(0.0002, 0.003, n)),
                   index=idx, name='equity')
    rep = beta_ledger.beta_report(eq, pd.DataFrame({'SPY': spy}), lags=1)
    diag = rep['SPY']
    assert 'trend_conditional' in diag
    assert 'trend_conditional_lagged' in diag
    tc, tcl = diag['trend_conditional'], diag['trend_conditional_lagged']
    # same key structure
    assert set(tc) == set(tcl)
    # unlagged block equals an independent conditional_betas with the
    # unshifted state (old numbers untouched)
    strat_ret = eq.pct_change().dropna()
    bench_ret = spy.pct_change().reindex(strat_ret.index)
    sma200 = spy.rolling(200).mean()
    state = (spy > sma200).where(sma200.notna()).reindex(
        strat_ret.index, method='ffill')
    indep = beta_ledger.conditional_betas(strat_ret, bench_ret, state,
                                          state_name='above_200d')
    tc_wo = {k: v for k, v in tc.items() if k != 'n_state_undefined'}
    assert _req(tc_wo, indep)
    # PIT: the single False->True crossing day moves into the prior day's
    # bucket => TRUE-bucket n_obs differs by exactly 1
    assert (tc['above_200d_true']['n_obs']
            - tcl['above_200d_true']['n_obs']) == 1
    assert tcl['n_state_undefined'] == tc['n_state_undefined'] + 1


def test_a4_obs_per_year_grid():
    eq, bench = _mk_fixture(101)
    rep = beta_ledger.beta_report(eq, bench, lags=1)
    assert abs(rep['period']['obs_per_year_grid'] - 365.25) < 0.5
    assert rep['period']['obs_per_year'] > 330   # span-based key untouched

    # every-2-days grid
    rng = np.random.default_rng(2)
    idx = pd.date_range('2026-01-01', periods=60, freq='2D')
    eq2 = pd.Series(100000.0 * np.cumprod(1 + rng.normal(0, 0.004, 60)),
                    index=idx)
    spy2 = pd.Series(400.0 * np.cumprod(1 + rng.normal(0, 0.008, 60)),
                     index=idx)
    rep2 = beta_ledger.beta_report(eq2, pd.DataFrame({'SPY': spy2}), lags=1)
    assert abs(rep2['period']['obs_per_year_grid'] - 365.25 / 2) < 0.5


def test_a5_clean_returns_from_pl():
    n = 60
    idx = pd.date_range('2026-03-01', periods=n, freq='D')
    vals = np.full(n, 100000.0)
    vals = vals * np.cumprod(1 + np.full(n, 0.001))
    vals[30:] *= 1.5           # +50% deposit jump on day 30
    eq = pd.Series(vals, index=idx)
    pl = eq.diff()
    pl.iloc[30] = 10.0         # profit_loss excludes the deposit
    pl.iloc[45] = np.nan       # missing pl day
    raw = eq.pct_change()
    clean = beta_ledger.clean_returns_from_pl(eq, pl)
    assert raw.iloc[30] > 0.15                       # raw fabricates a jump
    assert abs(clean.iloc[30]) < 0.001               # clean shows ~0
    assert math.isnan(clean.iloc[0])                 # first row NaN
    assert math.isnan(clean.iloc[45])                # missing pl -> NaN
    # prior equity <= 0 -> NaN
    eq2 = eq.copy()
    eq2.iloc[40] = -5.0
    clean2 = beta_ledger.clean_returns_from_pl(eq2, pl)
    # eq2 dropna keeps -5 (not NaN); prev.where(prev > 0) masks the NEXT day
    assert math.isnan(clean2.iloc[41])


def test_a6_beta_report_clean_keys():
    eq, bench = _mk_fixture(121, seed=3)
    vals = eq.values.copy()
    vals[30:] *= 1.5                    # deposit on day 30
    vals[10] = vals[9]                  # one flat day => joint_active emitted
    eq = pd.Series(vals, index=eq.index, name='equity')
    pl = eq.diff()
    pl.iloc[30] = 10.0
    clean = beta_ledger.clean_returns_from_pl(eq, pl)
    rep = beta_ledger.beta_report(eq, bench, lags=1, clean_ret=clean)
    j = rep['joint']
    for key in ('alpha_annual_clean', 'alpha_t_clean', 'n_obs_clean',
                'contamination_delta'):
        assert key in j
    assert j['contamination_delta'] == pytest.approx(
        j['alpha_annual_clean'] - j['alpha_annual'], abs=1e-9)
    s = rep['strategy']
    for key in ('sharpe_clean', 'ann_return_clean', 'ann_vol_clean'):
        assert key in s
    jc = rep['joint_clean']
    assert isinstance(jc, dict) and 'betas' in jc and 'alpha_annual' in jc
    # B1's dof-corrected keys ride along in EVERY lagged_beta_regression
    # output: joint, joint_active (flat day engineered above), joint_clean.
    assert 'joint_active' in rep
    for block in (j, rep['joint_active'], jc):
        for key in ('dof_scale', 'alpha_se_corrected', 'alpha_t_corrected'):
            assert key in block
    # RFC-8259 round trip
    json.dumps(beta_ledger._json_safe(rep), allow_nan=False, default=str)


def test_a7_clean_too_few_observations():
    eq, bench = _mk_fixture(121, seed=4)
    plain = beta_ledger.beta_report(eq, bench, lags=1)
    sparse = pd.Series(np.nan, index=eq.index, name='clean_ret')
    sparse.iloc[5:10] = 0.001          # only 5 finite values
    rep = beta_ledger.beta_report(eq, bench, lags=1, clean_ret=sparse)
    assert 'joint_clean' not in rep
    assert not any(k.endswith('_clean') or k == 'contamination_delta'
                   for k in rep['joint'])
    assert not any(k.endswith('_clean') for k in rep['strategy'])
    extra = [w for w in rep['warnings'] if 'clean-return' in w]
    assert len(extra) == 1
    stripped = copy.deepcopy(rep)
    stripped['warnings'] = [w for w in stripped['warnings']
                            if 'clean-return' not in w]
    assert _req(stripped, plain)


def _with_fake_trading_utils(fake_hist, fn):
    prior = sys.modules.get('trading_utils')
    fake_api = types.SimpleNamespace(get_portfolio_history=lambda **kw: fake_hist)
    sys.modules['trading_utils'] = types.SimpleNamespace(get_api=lambda: fake_api)
    try:
        return fn()
    finally:
        if prior is not None:
            sys.modules['trading_utils'] = prior
        else:
            sys.modules.pop('trading_utils', None)


def test_a8_load_equity_alpaca_with_pl():
    # (i) hist WITHOUT profit_loss attr — default contract byte-identical,
    # with_pl=True returns (eq, None)
    hist = types.SimpleNamespace(
        timestamp=[1700000000, 1700086400, 1700172800, 1700259200],
        equity=[None, None, 100.0, '101.0'],
    )
    eq = _with_fake_trading_utils(hist, lambda: beta_ledger.load_equity_alpaca(90))
    assert list(eq.values) == [100.0, 101.0]
    res = _with_fake_trading_utils(
        hist, lambda: beta_ledger.load_equity_alpaca(90, with_pl=True))
    assert isinstance(res, tuple) and len(res) == 2
    eq2, pl = res
    assert list(eq2.values) == [100.0, 101.0]
    assert pl is None

    # (ii) hist WITH profit_loss (incl. a None) — pl indexed like eq,
    # NaN at the None slot
    hist2 = types.SimpleNamespace(
        timestamp=[1700000000, 1700086400, 1700172800, 1700259200],
        equity=[None, None, 100.0, '101.0'],
        profit_loss=[1.0, 2.0, None, 4.0],
    )
    eq3, pl3 = _with_fake_trading_utils(
        hist2, lambda: beta_ledger.load_equity_alpaca(90, with_pl=True))
    assert pl3 is not None
    assert list(pl3.index) == list(eq3.index)
    assert math.isnan(pl3.iloc[0])
    assert pl3.iloc[1] == 4.0


# ===========================================================================
# Section B — chart_core additions
# ===========================================================================

def test_b1_obs_per_year():
    daily = np.arange(50, dtype=float) * 86400.0
    v = chart_core.obs_per_year(daily)
    assert abs(v - 365.25) < 0.5
    hourly = np.arange(100, dtype=float) * 3600.0
    v = chart_core.obs_per_year(hourly)
    assert abs(v - 365.25 * 24) < 5
    assert chart_core.obs_per_year([1.0]) is None
    assert chart_core.obs_per_year([]) is None
    assert chart_core.obs_per_year([np.nan, np.nan]) is None
    # unsorted input handled
    shuffled = daily.copy()
    rng = np.random.default_rng(5)
    rng.shuffle(shuffled)
    assert abs(chart_core.obs_per_year(shuffled) - 365.25) < 0.5


def test_b2_obs_per_year_matches_beta_ledger_grid():
    eq, bench = _mk_fixture(101)
    rep = beta_ledger.beta_report(eq, bench, lags=1)
    epochs = np.array([ts.timestamp() for ts in eq.index])
    v = chart_core.obs_per_year(epochs)
    grid = rep['period']['obs_per_year_grid']
    assert abs(v - grid) / grid < 1e-6


def test_b3_artifact_freshness(tmp_path):
    now = time.time()
    fresh = tmp_path / "fresh.json"
    fresh.write_text("{}")
    old = tmp_path / "old.jsonl"
    old.write_text("{}")
    os.utime(old, (now - 10 * 86400, now - 10 * 86400))
    missing = tmp_path / "missing.json"
    items = [
        ("fresh", fresh),
        ("old", old, 2 * 86400),
        ("missing", missing),
        ("ledger", old, None),      # None threshold: never age-stale
        ("garbage", None),          # garbage path: no raise
    ]
    rows = chart_core.artifact_freshness(items, now=now)
    assert [r['name'] for r in rows] == \
        ['fresh', 'old', 'missing', 'ledger', 'garbage']
    assert rows[0]['exists'] and not rows[0]['stale'] and rows[0]['age_s'] < 60
    assert rows[1]['exists'] and rows[1]['stale']
    assert abs(rows[1]['age_s'] - 10 * 86400) < 60
    assert not rows[2]['exists'] and rows[2]['age_s'] is None and rows[2]['stale']
    assert rows[3]['exists'] and not rows[3]['stale']
    assert not rows[4]['exists'] and rows[4]['stale']


def _write_journal(tmp_path):
    jdir = tmp_path / "journals"
    jdir.mkdir()
    today = dt.date.today().isoformat()
    rows = [
        {'action': 'buy', 'symbol': 'BTC/USD',
         'sizing': {'tilt': 0.8, 'stack': 'legacy'}},
        {'action': 'buy', 'symbol': 'ETH/USD',
         'sizing': {'tilt': 0.9, 'stack': 'legacy',
                    'v2': {'tilt': 0.6, 'min_src': 'vix'}}},
        {'action': 'buy', 'symbol': 'SOL/USD',
         'sizing': {'tilt': 1.0, 'stack': 'v2',
                    'v2': {'tilt': 0.7, 'min_src': 'vix'}}},
        {'action': 'sell', 'symbol': 'BTC/USD', 'pnl_pct': 1.0},
        {'action': 'buy', 'symbol': 'LTC/USD'},
    ]
    lines = [json.dumps(r) for r in rows] + ['{corrupt json']
    (jdir / f"{today}.jsonl").write_text("\n".join(lines) + "\n")
    return jdir


def test_b4_sizing_stack_summary(tmp_path):
    jdir = _write_journal(tmp_path)
    s = chart_core.sizing_stack_summary(str(jdir), time.time() - 86400)
    assert s['n_buy_rows'] == 4
    assert s['n_with_sizing'] == 3
    assert s['n_with_v2'] == 2
    assert s['stack_counts'] == {'legacy': 2, 'v2': 1}
    assert s['legacy_tilt_median'] == pytest.approx(0.9)
    assert s['v2_tilt_median'] == pytest.approx(0.65)
    assert s['tilt_divergence_median'] == pytest.approx(-0.3)
    assert s['v2_min_src_counts'] == {'vix': 2}
    # missing dir => zeroed shape, no raise
    z = chart_core.sizing_stack_summary(str(tmp_path / "nope"),
                                        time.time() - 86400)
    assert z['n_buy_rows'] == 0 and z['n_with_sizing'] == 0
    assert z['legacy_tilt_median'] is None
    assert z['stack_counts'] == {} and z['v2_min_src_counts'] == {}


@pytest.mark.parametrize("verdict,insufficient,expected", [
    ("REVIEW (charging admission — CI excludes zero)", False, 'review'),
    ("OK (earning its keep — CI excludes zero)", False, 'ok'),
    ("cannot conclude (CI spans zero)", False, 'inconclusive'),
    ("insufficient n (n=2 < 8) — no verdict", True, 'insufficient'),
    ("CHANGE — apply 2-reading confirmation to signal exits (CI excludes zero)",
     False, 'change'),
    ("NO CHANGE — the flip is saving money (CI excludes zero)", False,
     'no_change'),
    (None, False, 'inconclusive'),
])
def test_b5_gate_verdict_class(verdict, insufficient, expected):
    assert chart_core.gate_verdict_class(verdict, insufficient) == expected


def test_b6_gate_panel_model():
    rep = {
        'generated': '2026-08-19T10:00:00',
        'days': 30,
        'gates': {
            'llm': {'vetoes_priced': 12, 'vetoes_raw': 15,
                    'counterfactual_mean_net_pct': 0.5, 'ci90': [0.1, 0.9],
                    'counterfactual_hit_rate': 0.6, 'saved_total_pct': -6.0,
                    'verdict': 'REVIEW (charging admission — CI excludes zero)',
                    'insufficient_n': False},
            'cost': {'vetoes_priced': 10, 'vetoes_raw': 10,
                     'counterfactual_mean_net_pct': -0.4, 'ci90': [-0.8, -0.1],
                     'counterfactual_hit_rate': 0.3, 'saved_total_pct': 4.0,
                     'verdict': 'OK (earning its keep — CI excludes zero)',
                     'insufficient_n': False},
            'meta': {'vetoes_priced': 9, 'vetoes_raw': 9,
                     'counterfactual_mean_net_pct': 0.05, 'ci90': [-0.2, 0.3],
                     'counterfactual_hit_rate': 0.5, 'saved_total_pct': 0.4,
                     'verdict': 'cannot conclude (CI spans zero)',
                     'insufficient_n': False},
            'sent': {'vetoes_priced': 2, 'vetoes_raw': 2,
                     'counterfactual_mean_net_pct': 1.2, 'ci90': [None, None],
                     'counterfactual_hit_rate': 1.0, 'saved_total_pct': -2.0,
                     'verdict': 'insufficient n (n=2 < 8) — no verdict',
                     'insufficient_n': True},
            '_private': {'skip': True},
            'garbage': "not-a-dict",
        },
        'signal_exit': {
            'n_signal_sells': 14, 'priced': 11, 'ci90': [0.1, 0.6],
            'verdict': 'CHANGE — apply 2-reading confirmation to signal '
                       'exits (CI excludes zero)',
            'insufficient_n': False},
        'quality': {'priced': 33, 'unpriced': 20, 'unpriced_rate': 0.377,
                    'fetch_failed': 3, 'representative': False},
    }
    m = chart_core.gate_panel_model(rep)
    assert m['stale'] is False
    assert m['representative'] is False
    assert m['quality_line'] is not None
    assert 'priced 33' in m['quality_line']
    names = [g['name'] for g in m['gates']]
    assert names == ['llm', 'cost', 'meta', 'sent']   # ordered, filtered
    classes = {g['name']: g['verdict_class'] for g in m['gates']}
    assert classes == {'llm': 'review', 'cost': 'ok',
                       'meta': 'inconclusive', 'sent': 'insufficient'}
    assert m['signal_exit']['verdict_class'] == 'change'
    assert m['signal_exit']['priced'] == 11

    # _write_stale_report stub => default no-API reason, gates [], no exit
    stub = {'generated': '2026-08-19T10:00:00', 'days': 30,
            'api_available': False, 'stale': True}
    ms = chart_core.gate_panel_model(stub)
    assert ms['stale'] is True
    assert ms['stale_reason'] == ('no API when generated — counterfactuals '
                                  'not priced')
    assert ms['gates'] == [] and ms['signal_exit'] is None

    # explicit stale_reason passed through verbatim
    mr = chart_core.gate_panel_model(
        {'stale': True, 'stale_reason': 'analysis error: boom'})
    assert mr['stale_reason'] == 'analysis error: boom'

    # {} and garbage never raise
    assert chart_core.gate_panel_model({})['gates'] == []
    assert chart_core.gate_panel_model(None)['stale'] is False
    assert chart_core.gate_panel_model({'gates': 'nope'})['gates'] == []


def test_b7_meta_panel_model():
    meta = {'pred_source': 'oof', 'trained_at': '2026-08-18T02:00:00',
            'val_auc': 0.57, 'n_trades': 400, 'base_win_rate': 0.51,
            'oof': {'status': 'ok', 'fallback_reason': None},
            'replay_parity': None}
    refused = {'refused_at': '2026-08-18T02:05:00',
               'reasons': ['a', 'b', 'c', 'd']}
    m = chart_core.meta_panel_model(meta, refused)
    assert m['present'] and m['refused']
    assert m['pred_source'] == 'oof'
    assert m['val_auc'] == 0.57 and m['n_trades'] == 400
    assert m['refused_reasons'] == ['a', 'b', 'c']       # truncated to 3
    assert m['refused_at'] == '2026-08-18T02:05:00'
    # meta only
    m2 = chart_core.meta_panel_model(meta, None)
    assert m2['present'] and not m2['refused']
    # neither
    m3 = chart_core.meta_panel_model(None, None)
    assert not m3['present'] and not m3['refused']
    # non-list reasons coerced without raising
    m4 = chart_core.meta_panel_model(None, {'reasons': 'just-a-string'})
    assert m4['refused'] and m4['refused_reasons'] == ['just-a-string']


def test_b8_formatters():
    llm_rep = {
        'verdict': 'LLM adds signal INCREMENTAL to the ML pred',
        'n': 80, 'n_with_pred': 74,
        'incremental': {
            'n': 74,
            'encompassing': {'estimator': 'driscoll_kraay',
                             'b2_s': 0.031, 'p_value': 0.021},
            'legacy_b2': {'b2_s': 0.028, 'p_value': 0.04,
                          'estimator': 'newey_west_rows'},
        },
        'spend_ledger': {'daily_cost_usd': 0.42, 'daily_cost_limit_usd': 2.0,
                         'window_journaled_cost_usd': 4.1,
                         'n_entries_with_cost': 61,
                         'llm_tilt_bps_per_trade': 3.2,
                         'veto_avoided_ret_pct_sum': 1.4,
                         'cost_read_ok': True},
        'veto_counterfactual_pct': -0.8,
        'meta': {'generated_at': '2026-08-19T09:00:00', 'days': 14},
    }
    txt = chart_core.format_llm_eval_summary(llm_rep)
    assert 'INCREMENTAL' in txt
    assert 'driscoll_kraay' in txt
    assert 'legacy' in txt
    assert '0.42' in txt
    stub = chart_core.format_llm_eval_summary(
        {'verdict': 'no_data', 'reason': 'x'})
    assert 'no data' in stub and 'x' in stub
    assert chart_core.format_llm_eval_summary({})

    adv = {'verdict': 'llm_score_degenerate', 'n_total': 40,
           'n_calibratable': 30, 'signal_source': 'p_up',
           'p_up_present_frac': 0.95, 'n_dedup_hit': 12,
           'dedup_hit_frac': 0.4, 'n_unique_llm_calls': 18,
           'by_model': {'m1': {}, 'm2': {}},
           'incremental_p_up_only': {'verdict': 'insufficient_power'},
           'meta': {'generated_at': '2026-08-19T09:00:00'}}
    atxt = chart_core.format_advisor_summary(adv)
    assert 'p_up' in atxt and 'dedup' in atxt
    assert chart_core.format_advisor_summary({'verdict': 'no_data',
                                              'reason': 'y'})
    assert chart_core.format_advisor_summary({})

    ex = {'generated_at': '2026-08-19T09:00:00', 'window_days': 14,
          'overall_mean_bps': 3.4,
          'crypto/buy/entry': {'n': 20, 'mean_bps': 2.5, 'median_bps': 2.0,
                               'p90_bps': 6.0, 'worst_bps': 11.0}}
    etxt = chart_core.format_execution_summary(ex)
    assert 'crypto/buy/entry' in etxt and '2.5' in etxt
    empty = chart_core.format_execution_summary(
        {'generated_at': 'x', 'window_days': 14})
    assert 'no fills' in empty

    # sizing formatter round-trips B4's summary and the zeroed shape
    full = {'n_buy_rows': 4, 'n_with_sizing': 3, 'n_with_v2': 2,
            'stack_counts': {'legacy': 2, 'v2': 1},
            'legacy_tilt_median': 0.9, 'v2_tilt_median': 0.65,
            'tilt_divergence_median': -0.3,
            'v2_min_src_counts': {'vix': 2}}
    ftxt = chart_core.format_sizing_stack(full)
    assert 'sizing stack' in ftxt and 'vix' in ftxt and 'legacy' in ftxt
    ztxt = chart_core.format_sizing_stack(
        {'n_buy_rows': 0, 'n_with_sizing': 0, 'n_with_v2': 0,
         'stack_counts': {}, 'legacy_tilt_median': None,
         'v2_tilt_median': None, 'tilt_divergence_median': None,
         'v2_min_src_counts': {}})
    assert 'no sizing decomposition' in ztxt
    assert 'no sizing decomposition' in chart_core.format_sizing_stack(None)


# ===========================================================================
# Section C — gui.py source contracts (AST; PySide6-free)
# ===========================================================================

def test_c1_gate_attribution_verdict_first():
    body = _method_source('_refresh_gate_attribution')
    for needle in ('gate_panel_model', 'stale_reason', 'representative',
                   'verdict_class'):
        assert needle in body, f"_refresh_gate_attribution must use {needle}"
    assert 'mean < 0' not in body, \
        "raw-mean-sign coloring must be gone (verdict-first)"


def test_c2_models_tab_buttons_and_strip():
    body = _method_source('_build_models_tab')
    assert '"llm_eval.py", "--days", "14"' in body
    assert '"--advisor"' in body
    assert '"execution_report.py", "--days", "14"' in body
    assert '_reports_fresh_label' in body
    assert 'Meta gate' in body
    # U5 legacy command-string pins survive
    assert '"decision_report.py", "--days", "30"' in body
    assert '"beta_ledger.py", "--days", "90"' in body
    assert '"indicator_leadlag.py", "--data", "crypto_training_data.parquet"' \
        in body


def test_c3_run_report_persistent_beta_json():
    body = _method_source('_run_report_clicked')
    assert 'BETA_REPORT_FILE' in body
    assert '_report_json_persistent' in body
    assert 'subprocess.PIPE' not in body


def test_c4_check_report_run_summaries():
    body = _method_source('_check_report_run')
    for needle in ('format_llm_eval_summary', 'format_advisor_summary',
                   'format_execution_summary', '_report_json_persistent',
                   '_refresh_reports_freshness'):
        assert needle in body, f"_check_report_run must reference {needle}"


def test_c5_shadow_panel_v2_and_gate():
    body = _method_source('_refresh_shadow_panel')
    assert 'dm_v2_decision' in body
    assert 'POLICY_GATE_FILES' in body


def test_c6_new_refreshers_and_constants():
    # methods exist
    _method_source('_refresh_meta_panel')
    _method_source('_refresh_reports_freshness')
    assert 'REPORT_FRESHNESS_ITEMS' in GUI_SRC
    assert 'promotion_ledger.jsonl' in GUI_SRC
    # the 60s piggyback method that calls _refresh_drift_panel() also calls
    # both new refreshers (inside try/except)
    hosts = []
    for node in ast.walk(GUI_TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            src = "\n".join(GUI_LINES[node.lineno - 1:node.end_lineno])
            if 'self._refresh_drift_panel()' in src:
                hosts.append(src)
    assert hosts, "no method calls _refresh_drift_panel()"
    assert any('self._refresh_meta_panel()' in src
               and 'self._refresh_reports_freshness()' in src
               for src in hosts), \
        "the 60s piggyback must also call both new refreshers"


def test_c7_journal_worker_payload():
    assert 'sizing_stack_summary' in _method_source('_refresh_journal_analytics')
    assert 'format_sizing_stack' in _method_source('_on_journal_stats_ready')


def test_c8_meta_and_freshness_render_bodies():
    """G4 render semantics: the meta panel goes through the pure model and
    amber-flags the in-sample diagnostic; the freshness strip goes through
    artifact_freshness and renders the meta_refused sidecars with INVERTED
    (presence == alarm) semantics."""
    meta_body = _method_source('_refresh_meta_panel')
    assert 'meta_panel_model' in meta_body
    assert "'in_sample'" in meta_body
    assert 'META REFUSED' in meta_body
    fresh_body = _method_source('_refresh_reports_freshness')
    assert 'artifact_freshness' in fresh_body
    assert 'REPORT_FRESHNESS_ITEMS' in fresh_body
    assert 'META_REFUSED_FILES' in fresh_body
    assert 'PRESENT' in fresh_body


def test_c9_beta_summary_additive_lines():
    """G10: _format_beta_summary surfaces every additive beta_ledger key
    (all .get()-guarded, so old reports still render)."""
    body = _method_source('_format_beta_summary')
    for needle in ('obs_per_year_grid', 'alpha_t_corrected',
                   'alpha_annual_clean', 'contamination_delta',
                   'sharpe_clean', 'trend_conditional_lagged'):
        assert needle in body, f"_format_beta_summary must surface {needle}"
