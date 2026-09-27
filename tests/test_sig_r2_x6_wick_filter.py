"""SIG-R2-X6 — default-OFF harvest wick-print guard (WICK_PRINT_FILTER).

OFF pins (pass on the live tree AND the patched tree): the real
harvest_crypto_data.prepare_data feeds compute_features the fetched bars
unchanged and its output is identical for env unset / '0'.
ON tests (fail on the live tree, pass patched): the repair rule, the
flag reader, census/guard agreement, and the ON path through prepare_data.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts'))
try:
    import dotenv  # noqa: F401
except ImportError:  # dev-Mac: stub load_dotenv so harvest modules import
    _m = types.ModuleType('dotenv')
    _m.load_dotenv = lambda *a, **k: None
    sys.modules['dotenv'] = _m

import harvest_crypto_data as hc   # noqa: E402

FB = [3, 6]
WICK_I, HIGH_I = 60, 90


def _raw(seed=7, n=160, wicks=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2026-01-05', periods=n, freq='h', tz='UTC')
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.006, n)))
    open_ = np.r_[close[0], close[:-1]]
    sp = np.abs(rng.normal(0.0, 0.003, n)) * close + 0.01
    df = pd.DataFrame({'Open': open_, 'High': np.maximum(open_, close) + sp,
                       'Low': np.minimum(open_, close) - sp, 'Close': close,
                       'Volume': 5e3}, index=idx)
    if wicks:
        o = df['Open'].iat[WICK_I]
        df.iloc[WICK_I, df.columns.get_loc('Close')] = o * 1.002
        df.iloc[WICK_I, df.columns.get_loc('High')] = o * 1.004
        df.iloc[WICK_I, df.columns.get_loc('Low')] = o * 0.60     # -40 %
        o = df['Open'].iat[HIGH_I]
        df.iloc[HIGH_I, df.columns.get_loc('Close')] = o * 0.99   # 1 % body
        df.iloc[HIGH_I, df.columns.get_loc('High')] = o * 1.30    # +30 %
        df.iloc[HIGH_I, df.columns.get_loc('Low')] = o * 0.985
    return df


def _env(monkeypatch, raw, seen):
    monkeypatch.setattr(hc, 'FORWARD_BARS', list(FB))

    def feats(ohlcv, btc_close=None):
        seen.append(ohlcv.copy())
        df = ohlcv.copy()
        tr = pd.concat([df['High'] - df['Low'],
                        (df['High'] - df['Close'].shift()).abs(),
                        (df['Low'] - df['Close'].shift()).abs()],
                       axis=1).max(axis=1)
        df['ATR'] = tr.rolling(5).mean()
        df['Volume_Ratio'] = df['Close'].pct_change()
        return df
    monkeypatch.setattr(hc, 'compute_features', feats)
    monkeypatch.setattr(hc, 'fetch_with_fallback', lambda *a, **k: raw.copy())
    fa = types.ModuleType('funding_archive')
    fa.funding_features_for_index = lambda s, idx: None
    monkeypatch.setitem(sys.modules, 'funding_archive', fa)
    oa = types.ModuleType('oi_archive')
    for f in ('oi_features_for_index', 'ls_features_for_index',
              'taker_features_for_index'):
        setattr(oa, f, lambda s, idx: None)
    monkeypatch.setitem(sys.modules, 'oi_archive', oa)
    monkeypatch.setitem(sys.modules, 'liquidity',
                        types.ModuleType('liquidity'))
    cr = types.ModuleType('cost_regime')
    cr.stamp_cost_regime_features = lambda df, at: df
    monkeypatch.setitem(sys.modules, 'cost_regime', cr)


# ---------------------------------------------------------------- OFF pins
@pytest.mark.parametrize('val', [None, '0', 'false'])
def test_off_feeds_features_the_fetched_bars_unchanged(monkeypatch, val):
    if val is None:
        monkeypatch.delenv('TRADER_WICK_PRINT_FILTER', raising=False)
    else:
        monkeypatch.setenv('TRADER_WICK_PRINT_FILTER', val)
    import strategy_config
    monkeypatch.delattr(strategy_config, 'WICK_PRINT_FILTER', raising=False)
    raw, seen = _raw(), []
    _env(monkeypatch, raw, seen)
    out = hc.prepare_data('BTC-USD', api=None)
    assert len(seen) == 1
    pd.testing.assert_frame_equal(seen[0], raw, check_exact=True,
                                  check_freq=False)
    # the wick survives into the stored row (legacy behaviour)
    assert out['Low'].loc[raw.index[WICK_I]] == raw['Low'].iat[WICK_I]


def test_off_output_identical_unset_vs_zero(monkeypatch):
    raw = _raw()
    outs = []
    for val in (None, '0'):
        if val is None:
            monkeypatch.delenv('TRADER_WICK_PRINT_FILTER', raising=False)
        else:
            monkeypatch.setenv('TRADER_WICK_PRINT_FILTER', val)
        _env(monkeypatch, raw, [])
        outs.append(hc.prepare_data('BTC-USD', api=None))
    pd.testing.assert_frame_equal(outs[0], outs[1], check_exact=True)


# ---------------------------------------------------------------- ON path
def test_flag_reader(monkeypatch):
    from data_utils import wick_print_filter_enabled
    import strategy_config
    monkeypatch.delenv('TRADER_WICK_PRINT_FILTER', raising=False)
    monkeypatch.delattr(strategy_config, 'WICK_PRINT_FILTER', raising=False)
    assert wick_print_filter_enabled() is False          # absent -> OFF
    monkeypatch.setattr(strategy_config, 'WICK_PRINT_FILTER', True,
                        raising=False)
    assert wick_print_filter_enabled() is True
    monkeypatch.setenv('TRADER_WICK_PRINT_FILTER', '0')   # env wins
    assert wick_print_filter_enabled() is False
    for v in ('1', 'true', 'YES', 'on'):
        monkeypatch.setenv('TRADER_WICK_PRINT_FILTER', v)
        assert wick_print_filter_enabled() is True


def test_repair_rule_exact():
    from data_utils import repair_wick_prints, WICK_PRINT_BODY_FRAC
    assert WICK_PRINT_BODY_FRAC == 0.03
    idx = pd.date_range('2025-01-01', periods=10, freq='h', tz='UTC')
    g = pd.DataFrame({'Open': 100.0, 'High': 101.0, 'Low': 99.5,
                      'Close': 100.5, 'Volume': 1.0}, index=idx)
    g.iloc[7] = [100.0, 100.6, 80.0, 100.2, 1.0]    # low wick, 0.2 % body
    g.iloc[8] = [100.0, 130.0, 99.8, 100.0, 1.0]    # high wick
    g.iloc[9] = [100.0, 125.0, 99.0, 104.0, 1.0]    # 4 % body: untouched
    logs = []
    r = repair_wick_prints(g, 'T', log=logs.append)
    assert r['Low'].iat[7] == pytest.approx(98.5)    # min(O,C) - medTR 1.5
    assert r['High'].iat[8] == pytest.approx(101.5)  # max(O,C) + medTR 1.5
    keep = [i for i in range(10) if i not in (7, 8)]
    pd.testing.assert_frame_equal(r.iloc[keep], g.iloc[keep])
    for c in ('Open', 'Close', 'Volume'):
        assert (r[c] == g[c]).all()
    assert len(r) == len(g) and r.index.equals(g.index)
    assert sum('[WICK-GUARD]' in m for m in logs) == 3   # 2 repairs + total
    # nothing flagged -> the SAME object back, nothing logged
    clean = g.iloc[:7]
    logs.clear()
    assert repair_wick_prints(clean, 'T', log=logs.append) is clean
    assert logs == []


def test_repair_fallback_scale_when_history_thin():
    from data_utils import repair_wick_prints
    idx = pd.date_range('2025-01-01', periods=3, freq='h', tz='UTC')
    g = pd.DataFrame({'Open': [100.0, 100.0, 100.0],
                      'High': [101.0, 100.4, 102.0],
                      'Low': [99.0, 60.0, 99.0],
                      'Close': [100.0, 100.1, 101.0], 'Volume': 1.0},
                     index=idx)
    r = repair_wick_prints(g, log=None)
    # TR = [2.0, 40.4, 3.0] -> whole-series median 3.0 (prior bars < 5)
    assert r['Low'].iat[1] == pytest.approx(100.0 - 3.0)


def test_census_and_guard_agree():
    import bad_print_census as bp
    import data_utils as du
    g = _raw(seed=3, n=300)
    for i, side, d in ((40, 'low', 0.3), (41, 'low', 0.25), (200, 'high', 0.2)):
        o = g['Open'].iat[i]
        g.iloc[i, g.columns.get_loc('Close')] = o * 1.02   # 2 % body
        col = 'Low' if side == 'low' else 'High'
        g.iloc[i, g.columns.get_loc(col)] = o * (1 - d if side == 'low'
                                                  else 1 + d)
    lo, hi = bp.flag_wick_prints(g['Open'], g['High'], g['Low'], g['Close'],
                                 du.WICK_PRINT_LOW_FRAC,
                                 du.WICK_PRINT_BODY_FRAC)
    lo2, hi2 = du.flag_wick_prints(g['Open'], g['High'], g['Low'], g['Close'])
    # 40, 41 + the fixture's row 60 (low); 90 + 200 (high)
    assert (lo == lo2).all() and (hi == hi2).all()
    assert lo.sum() == 3 and hi.sum() == 2
    assert bp.MED_TR_WINDOW == du.WICK_PRINT_MED_TR_WINDOW
    assert bp.MED_TR_MIN == du.WICK_PRINT_MED_TR_MIN
    pd.testing.assert_frame_equal(bp.repair_frame(g, lo, hi),
                                  du.repair_wick_prints(g, log=None))


def test_on_repairs_before_features_keeps_rows(monkeypatch, capsys):
    from data_utils import repair_wick_prints
    monkeypatch.setenv('TRADER_WICK_PRINT_FILTER', '1')
    raw, seen = _raw(), []
    _env(monkeypatch, raw, seen)
    out_on = hc.prepare_data('BTC-USD', api=None)
    fed = seen[0]
    expect = repair_wick_prints(raw, log=None)
    pd.testing.assert_frame_equal(fed, expect, check_freq=False)
    o, c = raw['Open'].iat[WICK_I], raw['Close'].iat[WICK_I]
    assert fed['Low'].iat[WICK_I] > min(o, c) * 0.95        # wick gone
    assert fed['High'].iat[HIGH_I] < raw['High'].iat[HIGH_I] * 0.85
    diff = (fed != raw).any(axis=1)
    assert set(np.flatnonzero(diff)) == {WICK_I, HIGH_I}
    assert '[WICK-GUARD] BTC-USD' in capsys.readouterr().out
    monkeypatch.setenv('TRADER_WICK_PRINT_FILTER', '0')
    _env(monkeypatch, raw, [])
    out_off = hc.prepare_data('BTC-USD', api=None)
    assert out_on.index.equals(out_off.index)       # no row dropped
    assert out_on['Low'].loc[raw.index[WICK_I]] > out_off['Low'].loc[
        raw.index[WICK_I]]


def test_on_without_prints_equals_off(monkeypatch):
    raw = _raw(wicks=False)
    outs = []
    for val in ('1', '0'):
        monkeypatch.setenv('TRADER_WICK_PRINT_FILTER', val)
        _env(monkeypatch, raw, [])
        outs.append(hc.prepare_data('BTC-USD', api=None))
    pd.testing.assert_frame_equal(outs[0], outs[1], check_exact=True)
    from data_utils import repair_wick_prints   # ON-only symbol
    assert repair_wick_prints(raw, log=None) is raw
