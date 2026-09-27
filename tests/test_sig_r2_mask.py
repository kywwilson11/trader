"""SIG-R2-MASK: OBJECTIVE_SESSION_MASK — score stock LONG entries only on
rows the live book can enter.

The trainer's objective (fold / pruning / regime / holdout / threshold-
reselect scorers) walks long entries on EVERY row of the extended-session
stock panel (04:00-19:00 ET open-times), while stock_loop enters only with
the market clock open AND inside STOCK_ENTRY_WINDOWS_ET
(stock_loop._in_entry_window). Default-OFF flag (strategy_config
OBJECTIVE_SESSION_MASK / env TRADER_OBJECTIVE_SESSION_MASK): ON, stock
book only, long_veto = ~entry_ok per row. OFF: byte-identical.

Module under test: SIG_R2_MASK_HS (default: the repo's
scripts/hypersearch_v2.py); SIG_R2_LIVE_HS (optional) = a pre-patch copy
for the OFF before/after pin.
"""
import ast
import datetime as _dt
import importlib.util
import os
import sys
import types
import zoneinfo
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sklearn')
pytest.importorskip('optuna')

REPO = Path(os.environ.get('SIG_R2_MASK_REPO',
                           Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
HS_PATH = os.environ.get('SIG_R2_MASK_HS',
                         str(REPO / 'scripts' / 'hypersearch_v2.py'))
LIVE_HS = os.environ.get('SIG_R2_LIVE_HS')

import strategy_config  # noqa: E402

ET = zoneinfo.ZoneInfo('America/New_York')
FB = 4
H = 3000
# 2026-03-02 00:00 UTC (a Monday); the panel crosses the 2026-03-08 DST
# switch, weekends included (the pure-mask weekday rule is exercised).
T0 = 1_772_409_600
WINDOWS = [('09:45', '11:00'), ('14:30', '15:30')]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def hs():
    return _load(HS_PATH, 'hs_sig_r2_mask')


def _flag(monkeypatch, on):
    if on:
        monkeypatch.setenv('TRADER_OBJECTIVE_SESSION_MASK', '1')
    else:
        monkeypatch.delenv('TRADER_OBJECTIVE_SESSION_MASK', raising=False)
        monkeypatch.setattr(strategy_config, 'OBJECTIVE_SESSION_MASK',
                            False, raising=False)


def _pin_env(hs_mod, monkeypatch):
    monkeypatch.setattr(hs_mod, '_objective_v3', lambda: False)
    monkeypatch.setattr(hs_mod, '_objective_long_only', lambda: True)
    monkeypatch.setattr(hs_mod, '_training_repairs', lambda: False)
    monkeypatch.setattr(hs_mod, '_fixed_holdout_days', lambda: None)
    monkeypatch.setattr(strategy_config, 'PROMOTION_GATE_V2', False)
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', False,
                        raising=False)
    monkeypatch.setattr(strategy_config, 'ENTRY_WINDOWS_ENABLED', True)
    monkeypatch.setattr(strategy_config, 'STOCK_ENTRY_WINDOWS_ET',
                        list(WINDOWS))
    monkeypatch.delenv('TRADER_HOLDOUT_SPAN_BY_TARGET', raising=False)


def _live_rule(epoch_s):
    """Independent replica of stock_loop._in_entry_window on a bar's
    open-time + the market-clock weekday gate."""
    local = _dt.datetime.fromtimestamp(int(epoch_s), _dt.timezone.utc
                                       ).astimezone(ET)
    m = local.hour * 60 + local.minute
    ok = any(int(a[:2]) * 60 + int(a[3:]) <= m < int(b[:2]) * 60 + int(b[3:])
             for a, b in WINDOWS)
    return ok and local.weekday() < 5


# ---------------------------------------------------------------- pure mask
def test_mask_matches_live_entry_window_minute_by_minute(monkeypatch):
    """Feed IDENTICAL minutes to stock_loop._in_entry_window (the engine
    method, via a stub self) and to objective_utils.session_entry_mask."""
    stock_loop = pytest.importorskip('stock_loop')
    from objective_utils import session_entry_mask
    monkeypatch.setattr(strategy_config, 'ENTRY_WINDOWS_ENABLED', True)
    monkeypatch.setattr(strategy_config, 'STOCK_ENTRY_WINDOWS_ET',
                        list(WINDOWS))
    # Every minute of Mon 2026-03-09 (first EDT weekday) + Mon 2026-03-02
    # (EST): 2 x 1440 minutes, with seconds that must be dropped (:59).
    ts = []
    for day0 in (1_772_409_600, 1_773_014_400):
        ts.extend(day0 + np.arange(1440 * 2) * 60 + 59)
    ts = np.asarray(ts, dtype=np.int64)
    got = session_entry_mask(ts, WINDOWS)
    want = np.array([
        stock_loop.StockLoop._in_entry_window(types.SimpleNamespace(
            _get_eastern_now=lambda t=t: _dt.datetime.fromtimestamp(
                int(t), _dt.timezone.utc).astimezone(ET)))
        for t in ts])
    assert got.sum() > 0
    np.testing.assert_array_equal(got, want)


def test_mask_matches_backtest_entry_window_mask(monkeypatch):
    """Same rows as backtest._entry_window_mask (the policy replay / meta
    parity mask) on hourly open-times across the DST switch (weekdays)."""
    import pandas as pd
    backtest = pytest.importorskip('backtest')
    from objective_utils import session_entry_mask
    monkeypatch.setattr(strategy_config, 'ENTRY_WINDOWS_ENABLED', True)
    monkeypatch.setattr(strategy_config, 'STOCK_ENTRY_WINDOWS_ET',
                        list(WINDOWS))
    ts = (T0 + np.arange(24 * 21) * 3600 + 1800 * (np.arange(24 * 21) % 2)
          ).astype(np.int64)
    idx = pd.to_datetime(ts, unit='s', utc=True)
    wd = np.asarray(idx.tz_convert(ET).weekday) < 5
    want = backtest._entry_window_mask(idx) & wd
    np.testing.assert_array_equal(session_entry_mask(ts, WINDOWS), want)
    np.testing.assert_array_equal(session_entry_mask(ts, WINDOWS),
                                  np.array([_live_rule(t) for t in ts]))


def test_mask_weekend_and_empty():
    from objective_utils import session_entry_mask
    sat_10et = 1_772_897_400  # Sat 2026-03-07 10:30 EST (15:30 UTC)
    assert not session_entry_mask([sat_10et], WINDOWS)[0]
    assert session_entry_mask([sat_10et], WINDOWS, weekdays_only=False)[0]
    assert session_entry_mask([], WINDOWS).shape == (0,)


# ---------------------------------------------------------- flag resolution
def test_resolver_off_is_none_and_crypto_never_masked(hs, monkeypatch):
    t = T0 + np.arange(48) * 3600
    _flag(monkeypatch, False)
    assert hs._session_entry_ok(t, 'stock') is None
    _flag(monkeypatch, True)
    assert hs._session_entry_ok(t, 'crypto') is None
    ok = hs._session_entry_ok(t, 'stock')
    assert ok is not None and ok.dtype == bool and 0 < ok.sum() < len(t)
    monkeypatch.setenv('TRADER_OBJECTIVE_SESSION_MASK', 'off')
    monkeypatch.setattr(strategy_config, 'OBJECTIVE_SESSION_MASK', True,
                        raising=False)
    assert hs._session_entry_ok(t, 'stock') is None     # env wins
    monkeypatch.delenv('TRADER_OBJECTIVE_SESSION_MASK')
    assert hs._session_entry_ok(t, 'stock') is not None  # constant ON


def test_windows_disabled_falls_back_to_rth(hs, monkeypatch):
    _flag(monkeypatch, True)
    monkeypatch.setattr(strategy_config, 'ENTRY_WINDOWS_ENABLED', False)
    t = T0 + 14 * 3600 + np.arange(0, 1440, 30) * 60  # Mon 09:00 EST + 30m
    ok = hs._session_entry_ok(t, 'stock')
    local = [(_dt.datetime.fromtimestamp(int(x), _dt.timezone.utc)
              .astimezone(ET)) for x in t]
    want = [570 <= d.hour * 60 + d.minute < 960 for d in local]
    np.testing.assert_array_equal(ok, want)


# ------------------------------------------------------------- call wiring
def test_every_trade_scorer_call_passes_long_veto(hs):
    """Source pin: every compute_sharpe / simulate_trades /
    compute_regime_sharpes CALL in hypersearch_v2 forwards a long_veto
    keyword (a scorer that forgets it would score off-window entries)."""
    tree = ast.parse(Path(HS_PATH).read_text())
    names = {'compute_sharpe', 'simulate_trades', 'compute_regime_sharpes'}
    missing = []
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            nm = f.id if isinstance(f, ast.Name) else getattr(f, 'attr', None)
            if nm in names:
                n += 1
                if not any(k.arg == 'long_veto' for k in node.keywords):
                    missing.append((nm, node.lineno))
    assert n >= 8, n
    assert not missing, missing


# --------------------------------------------------------- fold / regime
def _panel(seed=3, n=900):
    rng = np.random.default_rng(seed)
    t = (T0 + np.arange(n) * 3600).astype(np.int64)
    preds = rng.normal(0.0, 1.0, n)
    y = rng.normal(0.05, 1.0, n)
    return t, preds, y


def test_fold_and_regime_scores_off_equal_legacy(hs, monkeypatch):
    """OFF: sv / regime veto are None -> the scorers see exactly the
    legacy call (long_veto=None)."""
    _pin_env(hs, monkeypatch)
    _flag(monkeypatch, False)
    t, p, y = _panel()
    sv = hs._session_entry_ok(t, 'stock')
    assert sv is None
    a = hs.compute_sharpe(p, y, 0.3, forward_bars=FB, asset_type='stock',
                          block_ids=None, long_veto=sv)
    b = hs.compute_sharpe(p, y, 0.3, forward_bars=FB, asset_type='stock')
    assert a == b
    ra = hs.compute_regime_sharpes(p, y * 3, 0.3, forward_bars=FB,
                                   asset_type='stock', long_veto=None)
    rb = hs.compute_regime_sharpes(p, y * 3, 0.3, forward_bars=FB,
                                   asset_type='stock')
    assert ra == rb


def test_fold_and_regime_scores_on_use_window_rows_only(hs, monkeypatch):
    from objective_utils import simulate_trades_core
    _pin_env(hs, monkeypatch)
    _flag(monkeypatch, True)
    t, p, y = _panel(n=3000)
    ok = hs._session_entry_ok(t, 'stock')
    ret_off, ent_off = hs.simulate_trades(p, y, 0.0, FB, 0.11,
                                          return_entries=True)
    ret_on, ent_on = hs.simulate_trades(p, y, 0.0, FB, 0.11,
                                        return_entries=True, long_veto=~ok)
    assert 0 < len(ent_on) < len(ent_off)
    assert ok[ent_on].all() and not ok[ent_off].all()
    want, _ = simulate_trades_core(p, y, 0.0, FB, 0.11, long_only=True,
                                   long_veto=~ok)
    np.testing.assert_array_equal(ret_on, want)
    # regime scorer subsets the veto with each regime mask
    reg = hs.compute_regime_sharpes(p, y * 3, 0.0, forward_bars=FB,
                                    asset_type='stock', long_veto=~ok)
    reg_off = hs.compute_regime_sharpes(p, y * 3, 0.0, forward_bars=FB,
                                        asset_type='stock')
    assert reg != reg_off


# ------------------------------------------------------------ holdout
def _holdout(hs_mod, monkeypatch, flag):
    from sklearn.preprocessing import RobustScaler
    _pin_env(hs_mod, monkeypatch)
    _flag(monkeypatch, flag)
    rng = np.random.default_rng(11)
    tickers = ['A', 'B']
    bounds = {'A': (0, H), 'B': (H, 2 * H)}
    times = np.concatenate([T0 + np.arange(H) * 3600,
                            T0 + np.arange(H) * 3600]).astype(np.int64)
    feats = rng.normal(size=(2 * H, 3)).astype(np.float32)
    ret = rng.normal(0.2, 1.0, size=2 * H).astype(np.float32)
    ret[H - FB:H] = np.nan
    ret[2 * H - FB:] = np.nan
    returns = {FB: ret}
    spans = {FB: np.full(2 * H, float(FB), dtype=np.float32)}
    cfg = {'seq_len': 8, 'forward_bars': FB, 'trade_threshold': -1e9,
           'target_kind': 'raw', 'hidden_dim': 8, 'num_layers': 1,
           'dropout': 0.0, 'n_heads': 2}
    torch.manual_seed(0)
    state = hs_mod.RegressionLSTM(3, 8, 1, 0.0, 2).state_dict()
    rep = hs_mod.evaluate_on_holdout(state, RobustScaler().fit(feats), cfg,
                                     feats, returns, times, tickers, bounds,
                                     3, 'stock', n_trials=10,
                                     all_tb_bars_by_fb=spans)
    hidx = hs_mod.get_holdout_indices(times, tickers, bounds, cfg['seq_len'])
    hidx = hidx[~np.isnan(ret[hidx])]
    return rep, hidx, times, ret


def test_holdout_off_report_unchanged(hs, monkeypatch):
    rep, *_ = _holdout(hs, monkeypatch, False)
    assert 'session_mask' not in rep and 'session_rows_ok' not in rep
    monkeypatch.setattr(hs, '_session_entry_ok', lambda *a, **k: None)
    rep2, *_ = _holdout(hs, monkeypatch, False)
    assert rep == rep2


@pytest.mark.skipif(not LIVE_HS or Path(LIVE_HS).resolve()
                    == Path(HS_PATH).resolve(),
                    reason='SIG_R2_LIVE_HS (pre-patch copy) not provided')
def test_holdout_off_report_byte_identical_to_pre_patch(hs, monkeypatch):
    live = _load(LIVE_HS, 'hs_sig_r2_mask_live')
    rep_new, *_ = _holdout(hs, monkeypatch, False)
    rep_old, *_ = _holdout(live, monkeypatch, False)
    assert rep_new == rep_old


def test_holdout_on_certifies_only_window_entries(hs, monkeypatch):
    from objective_utils import simulate_trades_core
    rep_off, hidx, times, ret = _holdout(hs, monkeypatch, False)
    rep_on, _, _, _ = _holdout(hs, monkeypatch, True)
    ok = np.array([_live_rule(t) for t in times[hidx]])
    assert rep_on['session_mask'] is True
    assert rep_on['session_rows_ok'] == int(ok.sum())
    tr, ent = simulate_trades_core(np.ones(len(hidx)), ret[hidx], -1e9, FB,
                                   0.11, long_only=True, long_veto=~ok)
    assert ok[ent].all()
    assert rep_on['n_trades'] == len(tr) < rep_off['n_trades']
    np.testing.assert_allclose(rep_on['trade_returns'],
                               np.round(tr, 6), atol=1e-6)
    # q10 veto is OR'd, never replaced: no-q10 path here -> key absent
    assert 'q10_vetoed' not in rep_on


# ------------------------------------------- holdout A/B driver (measure)
AB_PATH = REPO / 'scripts' / 'session_mask_holdout_ab.py'


@pytest.fixture(scope='module')
def ab():
    if not AB_PATH.exists():
        pytest.skip('scripts/session_mask_holdout_ab.py not present')
    return _load(str(AB_PATH), 'session_mask_holdout_ab_t')


def test_ab_decision_rule(ab):
    base = {'sharpe': 1.0, 'dsr': 0.9}
    # unmasked certified, masked DSR below 0.60 -> owner escalation
    d = ab.decide(base, {'dsr': 0.3, 'n_eff_v2': 40.0}, 0.0, 0.1, 0.60)
    assert d['verdict'] == 'ESCALATE'
    d = ab.decide(base, {'dsr': 0.7, 'n_eff_v2': 40.0}, -0.05, 0.1, 0.60)
    assert d['verdict'] == 'PROPOSE'
    d = ab.decide(base, {'dsr': 0.7, 'n_eff_v2': 9.9}, 0.0, 0.1, 0.60)
    assert d['verdict'] == 'HOLD' and not d['c1_masked_neff_ge_10']
    d = ab.decide(base, {'dsr': 0.7, 'n_eff_v2': 40.0}, -0.15, 0.1, 0.60)
    assert d['verdict'] == 'HOLD' and not d['c2_edge_within_1se']


def test_ab_weekly_block_se(ab):
    rng = np.random.default_rng(5)
    t = T0 + np.sort(rng.integers(0, 70 * 86400, 400))
    r = rng.normal(0.1, 1.0, 400)
    diff, se, k = ab.weekly_block_se_diff(r, t, r[::2], t[::2], n_boot=500)
    assert k == 10 and np.isfinite(se) and se > 0
    assert diff == pytest.approx(r[::2].mean() - r.mean())
    d0, se0, _ = ab.weekly_block_se_diff(r, t, r, t, n_boot=200)
    assert d0 == 0.0 and se0 == 0.0          # identical series: paired


def test_ab_driver_runs_real_holdout_off_on(hs, ab, monkeypatch):
    """run() drives the REAL evaluate_on_holdout twice and restores the
    env + module attributes it wraps."""
    from sklearn.preprocessing import RobustScaler
    _pin_env(hs, monkeypatch)
    monkeypatch.delenv('TRADER_OBJECTIVE_SESSION_MASK', raising=False)
    rng = np.random.default_rng(11)
    bounds = {'A': (0, H), 'B': (H, 2 * H)}
    times = np.concatenate([T0 + np.arange(H) * 3600] * 2).astype(np.int64)
    feats = rng.normal(size=(2 * H, 3)).astype(np.float32)
    ret = rng.normal(0.2, 1.0, size=2 * H).astype(np.float32)
    ret[H - FB:H] = np.nan
    ret[2 * H - FB:] = np.nan
    data = (feats, {FB: ret}, times, times, ['A', 'B'], bounds, ['f0', 'f1',
            'f2'], 3, 'x', True, {FB: np.full(2 * H, float(FB), np.float32)})
    cfg = {'seq_len': 8, 'forward_bars': FB, 'trade_threshold': -1e9,
           'target_kind': 'raw', 'hidden_dim': 8, 'num_layers': 1,
           'dropout': 0.0, 'n_heads': 2}
    torch.manual_seed(0)
    state = hs.RegressionLSTM(3, 8, 1, 0.0, 2).state_dict()
    sim, hid = hs.simulate_trades, hs.get_holdout_indices
    res = ab.run(hs, cfg, state, RobustScaler().fit(feats), data, 10,
                 n_boot=200)
    assert hs.simulate_trades is sim and hs.get_holdout_indices is hid
    assert 'TRADER_OBJECTIVE_SESSION_MASK' not in os.environ
    assert 'session_mask' not in res['off']
    assert res['on']['session_mask'] is True
    assert res['on']['n_trades'] < res['off']['n_trades']
    assert res['decision']['verdict'] in ('PROPOSE', 'HOLD', 'ESCALATE')
    assert np.isfinite(res['decision']['edge_se'])
