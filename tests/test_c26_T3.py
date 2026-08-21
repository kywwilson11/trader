"""Packet T3 tests — cost truth (B05 + B21 + D40).

Covers: TRADER_SPREAD_FILL_V2 (D40 no-estimate fill + inf ordering),
crypto spread tiers + census consumption (dark, KILL_LIST:90-adjacent),
minute-bar EDGE daily stamp, impact vol-scale re-base, cost-regime harvest
wiring (Option B), the pct_change pin sweep, and the census script's pure
helpers. Mac-runnable: numpy/pandas/pytest/bidask only; census module
imported with the T2 dotenv-stub pattern.
"""
import re
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
try:
    import dotenv  # noqa: F401
except ImportError:  # dev-Mac: stub load_dotenv so scripts import
    _m = types.ModuleType('dotenv')
    _m.load_dotenv = lambda *a, **k: None
    sys.modules['dotenv'] = _m

import cost_regime
import liquidity

PANDAS_MAJOR = int(pd.__version__.split('.')[0])


def _ohlc(n=60, seed=3, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    idx = pd.date_range('2025-01-06', periods=n, freq='h')
    return pd.DataFrame({'Open': openp, 'High': high, 'Low': low,
                         'Close': close}, index=idx)


def _constant_ohlc(n=120, px=50.0):
    idx = pd.date_range('2025-01-06', periods=n, freq='h')
    return pd.DataFrame({'Open': px, 'High': px, 'Low': px, 'Close': px},
                        index=idx)


# =====================================================================
# A. TRADER_SPREAD_FILL_V2 (D40)
# =====================================================================

class TestSpreadFillV2:
    def test_flag_off_default_and_byte_identity(self):
        assert liquidity.SPREAD_FILL_V2 is False
        # Zero-range (constant) frame: every bar has no EDGE estimate and is
        # stamped at the floor — the pinned pre-c26 behavior.
        pytest.importorskip('bidask')
        s = liquidity.edge_spread_series(_constant_ohlc())
        assert (s == liquidity.SPREAD_FLOOR_PCT).all()

    def test_finalize_v1_inf_and_nan_all_floor(self):
        pct = pd.Series([np.inf, -np.inf, np.nan, 0.5])
        out, raw_nan, at_floor, at_cap = liquidity._finalize_spread_pct(
            pct, 0.02, 1.50, 0.02)
        # V1: inf -> NaN first, so +inf/-inf/NaN ALL land at the floor.
        assert list(out) == [0.02, 0.02, 0.02, 0.5]
        assert raw_nan == 3

    def test_finalize_v2_ordering_fix(self, monkeypatch):
        monkeypatch.setattr(liquidity, 'SPREAD_FILL_V2', True)
        pct = pd.Series([np.inf, -np.inf, np.nan, 0.5])
        out, raw_nan, at_floor, at_cap = liquidity._finalize_spread_pct(
            pct, 0.02, 1.50, 0.05)
        # V2: clip BEFORE inf handling — +inf at the CAP, -inf at the floor,
        # NaN at the flat fill.
        assert list(out) == [1.50, 0.02, 0.05, 0.5]
        assert raw_nan == 1
        assert at_cap == 1

    def test_v2_short_frame_flat_fill_per_asset(self, monkeypatch):
        monkeypatch.setattr(liquidity, 'SPREAD_FILL_V2', True)
        df = _ohlc(n=10)
        s = liquidity.edge_spread_series(df, asset_type='stock')
        assert (s == 0.05).all()
        c = liquidity.edge_spread_series(df, asset_type='crypto')
        assert (c == 0.10).all()

    def test_v2_zero_range_flat_not_floor(self, monkeypatch):
        pytest.importorskip('bidask')
        monkeypatch.setattr(liquidity, 'SPREAD_FILL_V2', True)
        s = liquidity.edge_spread_series(_constant_ohlc(), asset_type='stock')
        assert (s == 0.05).all()
        assert not (s == liquidity.SPREAD_FLOOR_PCT).any()

    def test_no_estimate_fill_pct(self, monkeypatch):
        assert liquidity._no_estimate_fill_pct(0.02, 'stock') == 0.02
        assert liquidity._no_estimate_fill_pct(0.02, 'crypto') == 0.02
        monkeypatch.setattr(liquidity, 'SPREAD_FILL_V2', True)
        assert liquidity._no_estimate_fill_pct(0.02, 'stock') == 0.05
        assert liquidity._no_estimate_fill_pct(0.02, 'crypto') == 0.10


# =====================================================================
# B. Crypto spread tiers (dark, KILL_LIST:90-adjacent)
# =====================================================================

class TestCryptoTier:
    def test_tier_defaults_and_normalization(self, monkeypatch):
        monkeypatch.setattr(liquidity, '_CENSUS_MEMO', {})  # force defaults
        assert liquidity.get_crypto_spread_tier('BTC/USD') == 0.05
        assert liquidity.get_crypto_spread_tier('LINK/USD') == 0.10
        assert liquidity.get_crypto_spread_tier('DOT/USD') == 0.25
        assert (liquidity.get_crypto_spread_tier('ZZZ/USD')
                == liquidity.CRYPTO_TIER_FALLBACK_PCT)
        # dash form normalizes to the slash pair
        assert liquidity.get_crypto_spread_tier('BTC-USD') == 0.05

    def test_census_file_consumed(self, monkeypatch, tmp_path):
        p = tmp_path / 'census.json'
        p.write_text('{"pairs": {"BTC/USD": {"median_spread_pct": 0.031}}}')
        monkeypatch.setattr(liquidity, 'CRYPTO_CENSUS_FILE', str(p))
        monkeypatch.setattr(liquidity, '_CENSUS_MEMO', None)
        assert liquidity.get_crypto_spread_tier('BTC/USD') == pytest.approx(0.031)
        # census miss falls back to defaults
        assert liquidity.get_crypto_spread_tier('ETH/USD') == 0.05

    def test_census_out_of_range_clipped(self, monkeypatch, tmp_path):
        p = tmp_path / 'census.json'
        p.write_text('{"pairs": {"BTC/USD": {"median_spread_pct": 9.9}}}')
        monkeypatch.setattr(liquidity, 'CRYPTO_CENSUS_FILE', str(p))
        monkeypatch.setattr(liquidity, '_CENSUS_MEMO', None)
        assert (liquidity.get_crypto_spread_tier('BTC/USD')
                == liquidity.SPREAD_CAP_PCT)

    def test_census_corrupt_file_falls_back(self, monkeypatch, tmp_path):
        p = tmp_path / 'census.json'
        p.write_text('not json {{')
        monkeypatch.setattr(liquidity, 'CRYPTO_CENSUS_FILE', str(p))
        monkeypatch.setattr(liquidity, '_CENSUS_MEMO', None)
        assert liquidity.get_crypto_spread_tier('BTC/USD') == 0.05  # no raise

    def test_stamp_flag_off_same_object(self):
        assert liquidity.CRYPTO_SPREAD_STAMP is False
        df = _ohlc(n=20)
        out = liquidity.stamp_crypto_spreads(df, 'BTC/USD')
        assert out is df
        assert 'Eff_Spread_Pct' not in out.columns

    def test_stamp_flag_on_constant_tier(self, monkeypatch):
        monkeypatch.setattr(liquidity, 'CRYPTO_SPREAD_STAMP', True)
        monkeypatch.setattr(liquidity, '_CENSUS_MEMO', {})
        df = _ohlc(n=20)
        out = liquidity.stamp_crypto_spreads(df, 'BTC/USD')
        assert out is not df
        assert 'Eff_Spread_Pct' not in df.columns   # input not mutated
        assert (out['Eff_Spread_Pct'] == 0.05).all()


# =====================================================================
# C. Minute-bar EDGE daily stamp
# =====================================================================

def _make_minute(days=6, s=0.002, wide_day=None, wide_s=0.01, short_day=None):
    """Seeded bid-ask-bounce minute bars: per-bar trade prices sit at
    mid*(1 +/- s/2), highs/lows touch both quotes. Deterministic per day."""
    frames = []
    dates = pd.bdate_range('2025-03-03', periods=days)
    for di, d in enumerate(dates):
        rng = np.random.default_rng(100 + di)
        n = 390 if (short_day is None or di != short_day) else 10
        sp = wide_s if (wide_day is not None and di == wide_day) else s
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=30),
                            periods=n, freq='min')
        mid = 100.0 * np.exp(np.cumsum(rng.normal(0, 2e-4, n)))
        close = mid * (1 + rng.choice([1.0, -1.0], n) * sp / 2)
        openp = mid * (1 + rng.choice([1.0, -1.0], n) * sp / 2)
        high = mid * (1 + sp / 2) * (1 + np.abs(rng.normal(0, 1e-4, n)))
        low = mid * (1 - sp / 2) * (1 - np.abs(rng.normal(0, 1e-4, n)))
        frames.append(pd.DataFrame({'Open': openp, 'High': high,
                                    'Low': low, 'Close': close}, index=idx))
    return pd.concat(frames), dates


def _hourly_index(dates):
    return pd.DatetimeIndex([d + pd.Timedelta(hours=h)
                             for d in dates for h in range(10, 16)])


class TestMinuteEdge:
    def test_shape_lag_and_band(self):
        pytest.importorskip('bidask')
        mdf, dates = _make_minute()
        hidx = _hourly_index(dates)
        out = liquidity.edge_spread_daily_from_minute(mdf, hidx, min_days=1)
        assert len(out) == len(hidx)
        day = out.index.normalize()
        # day 1: no PRIOR covered day after the shift -> NaN
        assert out[day == dates[0]].isna().all()
        later = out[day > dates[0]]
        assert later.notna().all()
        assert ((later >= 0.02) & (later <= 0.8)).all()

    def test_strictly_trailing_wide_last_day(self):
        pytest.importorskip('bidask')
        mdf, dates = _make_minute()
        mdf_w, _ = _make_minute(wide_day=5)
        hidx = _hourly_index(dates)
        base = liquidity.edge_spread_daily_from_minute(mdf, hidx, min_days=1)
        wide = liquidity.edge_spread_daily_from_minute(mdf_w, hidx, min_days=1)
        m6 = base.index.normalize() == dates[5]
        # widening ONLY day 6 must not move day-6 stamps (they see <= day 5)
        np.testing.assert_allclose(base[m6].values, wide[m6].values)

    def test_short_day_skipped(self):
        pytest.importorskip('bidask')
        mdf, dates = _make_minute(short_day=2)
        hidx = _hourly_index(dates)
        out = liquidity.edge_spread_daily_from_minute(mdf, hidx, min_days=1)
        day = out.index.normalize()
        # the skipped day is absent from the daily series -> its own hourly
        # rows map to NaN; later days still get finite trailing values
        assert out[day == dates[2]].isna().all()
        assert out[day == dates[4]].notna().all()

    def test_estimator_failure_all_nan(self, monkeypatch):
        bidask = pytest.importorskip('bidask')
        def _boom(*a, **k):
            raise RuntimeError('census boom')
        monkeypatch.setattr(bidask, 'edge', _boom)
        mdf, dates = _make_minute(days=3)
        hidx = _hourly_index(dates)
        out = liquidity.edge_spread_daily_from_minute(mdf, hidx, min_days=1)
        assert out.isna().all() and len(out) == len(hidx)


# =====================================================================
# D. Impact vol-scale re-base (doubly dark)
# =====================================================================

class TestImpactVolscale:
    def test_volscale_k_constants_and_form(self):
        assert liquidity.volscale_impact_k('stock') == 20.0
        assert liquidity.volscale_impact_k('crypto') == 18.0
        assert liquidity.volscale_impact_k(
            'stock', sigma_daily_pct=2.0, spread_pct=0.05) == pytest.approx(
            0.5 * 2.0 / 0.05)
        # bad inputs fail open to the per-book constant
        assert liquidity.volscale_impact_k('stock', sigma_daily_pct=np.nan,
                                           spread_pct=0.05) == 20.0
        assert liquidity.volscale_impact_k('crypto', sigma_daily_pct='x',
                                           spread_pct=0.05) == 18.0
        assert liquidity.volscale_impact_k('stock', sigma_daily_pct=2.0,
                                           spread_pct=0.0) == 20.0

    def test_real_config_disabled_both_flag_states(self, monkeypatch):
        try:
            import strategy_config
        except Exception:
            pytest.skip('strategy_config not importable on this machine')
        if strategy_config.IMPACT_COST_ENABLED:
            pytest.skip('IMPACT_COST_ENABLED unexpectedly True')
        df = pd.DataFrame({'DV30': [1e6] * 5})
        assert liquidity.impact_inputs_from_df(df) == (None, None, 1.0)
        monkeypatch.setattr(liquidity, 'IMPACT_VOLSCALE', True)
        assert liquidity.impact_inputs_from_df(df) == (None, None, 1.0)

    def _fake_cfg(self, monkeypatch):
        fake = types.ModuleType('strategy_config')
        fake.IMPACT_COST_ENABLED = True
        fake.IMPACT_K = 1.0
        fake.IMPACT_TYPICAL_NOTIONAL = 25000
        monkeypatch.setitem(sys.modules, 'strategy_config', fake)

    def test_flag_off_k_unchanged(self, monkeypatch):
        self._fake_cfg(monkeypatch)
        df = pd.DataFrame({'DV30': [1e6] * 5, 'Ticker': ['BTC/USD'] * 5})
        adv, notional, k = liquidity.impact_inputs_from_df(df)
        assert k == 1.0 and notional == 25000.0
        np.testing.assert_array_equal(adv, df['DV30'].values)

    def test_flag_on_rebased_per_book(self, monkeypatch):
        self._fake_cfg(monkeypatch)
        monkeypatch.setattr(liquidity, 'IMPACT_VOLSCALE', True)
        crypto = pd.DataFrame({'DV30': [1e6] * 3, 'Ticker': ['BTC/USD'] * 3})
        assert liquidity.impact_inputs_from_df(crypto)[2] == 18.0
        stock = pd.DataFrame({'DV30': [1e6] * 3, 'Ticker': ['AAPL'] * 3})
        assert liquidity.impact_inputs_from_df(stock)[2] == 20.0
        # no Ticker column -> conservative stock (higher) k
        bare = pd.DataFrame({'DV30': [1e6] * 3})
        assert liquidity.impact_inputs_from_df(bare)[2] == 20.0


# =====================================================================
# E. Cost-regime harvest wiring (B21 Option B, dark)
# =====================================================================

def _hourly_frame(n=200):
    rng = np.random.default_rng(11)
    idx = pd.date_range('2025-01-06', periods=n, freq='h')
    close = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    vol = rng.integers(1_000, 50_000, n).astype(float)
    return pd.DataFrame({'Close': close, 'Volume': vol}, index=idx)


def _vix_daily():
    rng = np.random.default_rng(5)
    idx = pd.date_range('2024-08-01', periods=300, freq='D')
    return pd.Series(18.0 + np.cumsum(rng.normal(0, 0.3, 300)).clip(-5, 20),
                     index=idx)


class TestCostRegimeStamp:
    def test_flag_off_same_object(self):
        assert cost_regime.COST_REGIME_FEATURES is False
        df = _hourly_frame()
        out = cost_regime.stamp_cost_regime_features(df, 'stock')
        assert out is df
        assert 'VIX_Level' not in out.columns
        assert 'Amihud_Illiq' not in out.columns

    def test_flag_on_adds_exact_columns(self, monkeypatch):
        monkeypatch.setattr(cost_regime, 'COST_REGIME_FEATURES', True)
        df = _hourly_frame()
        vix = _vix_daily()
        out = cost_regime.stamp_cost_regime_features(df, 'stock',
                                                     vix_daily=vix)
        assert out is not df
        added = set(out.columns) - set(df.columns)
        assert added == {'VIX_Level', 'VIX_Regime', 'VIX_Pctile',
                         'Amihud_Illiq'}
        # Amihud is ffill+0.0-filled so the harvest dropna() can't eat rows
        assert out['Amihud_Illiq'].notna().all()
        # values match the direct vix_features_for_index call
        feats = cost_regime.vix_features_for_index(vix, df.index)
        np.testing.assert_array_equal(out['VIX_Level'].values,
                                      feats['VIX_Level'])
        # PIT: a bar on day D carries the day-(D-1) VIX close
        bar = out.index[30]                      # 2025-01-07T06
        d_minus_1 = (bar.normalize() - pd.Timedelta(days=1))
        assert out.loc[bar, 'VIX_Level'] == pytest.approx(vix.loc[d_minus_1])

    def test_flag_on_no_vix_history_amihud_still_added(self, monkeypatch):
        monkeypatch.setattr(cost_regime, 'COST_REGIME_FEATURES', True)
        monkeypatch.setattr(cost_regime, 'fetch_fred_vixcls', lambda: None)
        df = _hourly_frame()
        out = cost_regime.stamp_cost_regime_features(df, 'crypto')
        assert 'VIX_Level' not in out.columns
        assert 'Amihud_Illiq' in out.columns
        assert out['Amihud_Illiq'].notna().all()

    def test_amihud_pin_no_pad_fabrication(self):
        n = 120
        rng = np.random.default_rng(9)
        idx = pd.date_range('2025-01-06', periods=n, freq='h')
        close = pd.Series(50.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                          index=idx)
        vol = pd.Series(rng.integers(1_000, 9_000, n).astype(float),
                        index=idx)
        gap = 50
        close_nan = close.copy()
        close_nan.iloc[gap] = np.nan
        new = cost_regime.amihud_illiq(close_nan, vol)
        # padded reference == the pre-pin pandas-2 pad semantics
        ret_pad = close_nan.ffill().pct_change(fill_method=None).abs()
        dv = close_nan * vol
        dollar_vol = dv.where(dv > 0)
        daily = ((ret_pad / dollar_vol) * 1e6).where(
            lambda x: np.isfinite(x))
        ref = daily.rolling(21, min_periods=10).mean()
        # identical before the gap...
        pd.testing.assert_series_equal(new.iloc[:gap], ref.iloc[:gap])
        # ...but the pad fabricates a return at gap+1 the pin excludes
        after = slice(gap + 1, gap + 22)
        assert not np.allclose(new.iloc[after].values,
                               ref.iloc[after].values, equal_nan=True)
        # clean series: pin is value-identical to the legacy call
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            ret_legacy = close.pct_change().abs()
        dv2 = close * vol
        daily2 = ((ret_legacy / dv2.where(dv2 > 0)) * 1e6).where(
            lambda x: np.isfinite(x))
        ref2 = daily2.rolling(21, min_periods=10).mean()
        pd.testing.assert_series_equal(cost_regime.amihud_illiq(close, vol),
                                       ref2)


# =====================================================================
# F. pct_change pin sweep
# =====================================================================

SWEEP_FILES = ['indicators.py', 'llm_analyst.py', 'beta_ledger.py',
               'oi_archive.py', 'portfolio.py', 'indicator_leadlag.py',
               'cost_regime.py']


class TestPctChangePins:
    def test_pattern_value_identity(self):
        s = pd.Series([1.0, np.nan, 2.0, np.nan, np.nan, 3.0, 4.0])
        pinned = s.ffill().pct_change(fill_method=None)
        p = s.ffill()
        manual_pad = p / p.shift(1) - 1
        pd.testing.assert_series_equal(pinned, manual_pad)
        if PANDAS_MAJOR < 3:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                legacy = s.pct_change()
            pd.testing.assert_series_equal(pinned, legacy)

    def test_source_guard_every_call_pinned(self):
        for fname in SWEEP_FILES:
            text = (REPO / fname).read_text()
            for i, line in enumerate(text.splitlines(), 1):
                if re.search(r'\.pct_change\(', line):
                    assert 'fill_method' in line, (
                        f'{fname}:{i} unpinned pct_change: {line.strip()}')


# =====================================================================
# G. requirements pin
# =====================================================================

class TestRequirements:
    def test_jetson_pandas_upper_bound(self):
        text = (REPO / 'requirements-jetson.txt').read_text()
        pandas_lines = [ln for ln in text.splitlines()
                        if ln.strip().startswith('pandas')]
        assert pandas_lines and ',<3' in pandas_lines[0]


# =====================================================================
# H. Census script pure helpers
# =====================================================================

import crypto_spread_census as census  # noqa: E402


class TestCensusHelpers:
    def test_quote_spread_pct(self):
        assert census.quote_spread_pct(100.0, 101.0) == pytest.approx(
            1.0 / 100.5 * 100.0)
        assert census.quote_spread_pct(101.0, 100.0) is None   # crossed
        assert census.quote_spread_pct(0.0, 100.0) is None
        assert census.quote_spread_pct(100.0, 0.0) is None
        assert census.quote_spread_pct(-1.0, 100.0) is None
        assert census.quote_spread_pct(np.nan, 100.0) is None
        assert census.quote_spread_pct('x', 100.0) is None
        assert census.quote_spread_pct(None, 100.0) is None
        assert census.quote_spread_pct(100.0, 100.0) == 0.0    # locked

    def test_summarize(self):
        out = census.summarize({'A': [0.1, 0.2, 0.3], 'B': []})
        assert out['A']['n'] == 3
        assert out['A']['median_spread_pct'] == pytest.approx(0.2)
        assert out['A']['p90_spread_pct'] == pytest.approx(
            np.percentile([0.1, 0.2, 0.3], 90))
        assert out['A']['mean_spread_pct'] == pytest.approx(0.2)
        assert out['B'] == {'n': 0}

    def test_sanity_check(self):
        bad = census.sanity_check(
            {'BTC/USD': {'n': 100, 'median_spread_pct': 2.0}})
        assert len(bad) == 1 and 'BTC/USD' in bad[0]
        ok = census.sanity_check(
            {'BTC/USD': {'n': 100, 'median_spread_pct': 0.03},
             'SOL/USD': {'n': 100, 'median_spread_pct': 0.15}})
        assert ok == []
        # under min_n the pair is not evaluated
        assert census.sanity_check(
            {'BTC/USD': {'n': 5, 'median_spread_pct': 2.0}}) == []
        assert census.sanity_check({'DOGE/USD': {'n': 0}}) == []


# =====================================================================
# I. Harvest wiring stays Mac-importable (T2 stub pattern)
# =====================================================================

class TestHarvestWiringInert:
    def test_harvest_modules_import(self):
        import harvest_crypto_data  # noqa: F401
        import harvest_stock_data as h
        assert callable(h._minute_edge_overlay)
        assert h.MINUTE_EDGE_DAYS == 120


# =====================================================================
# J. Hardening pass (c26-T3 verifier): gap-closing pins
# =====================================================================

class TestT3Hardening:
    def test_all_dark_flags_default_off(self):
        # byte-identity insurance: every T3 flag reads OFF in a clean env
        assert liquidity.SPREAD_FILL_V2 is False
        assert liquidity.CRYPTO_SPREAD_STAMP is False
        assert liquidity.STOCK_MINUTE_EDGE is False
        assert liquidity.IMPACT_VOLSCALE is False
        assert cost_regime.COST_REGIME_FEATURES is False

    def test_census_flat_map_schema(self, monkeypatch, tmp_path):
        # _load_census accepts a flat {sym: pct} map (no 'pairs' wrapper),
        # and get_crypto_spread_tier floats a scalar entry
        p = tmp_path / 'flat.json'
        p.write_text('{"BTC/USD": 0.042, "ETH/USD": {"median_spread_pct": 0.06}}')
        monkeypatch.setattr(liquidity, 'CRYPTO_CENSUS_FILE', str(p))
        monkeypatch.setattr(liquidity, '_CENSUS_MEMO', None)
        assert liquidity.get_crypto_spread_tier('BTC/USD') == pytest.approx(0.042)
        assert liquidity.get_crypto_spread_tier('ETH/USD') == pytest.approx(0.06)

    def test_census_nonpositive_value_falls_back(self, monkeypatch, tmp_path):
        # a 0/negative census median is not evidence — defaults win
        p = tmp_path / 'zero.json'
        p.write_text('{"pairs": {"BTC/USD": {"median_spread_pct": 0.0}}}')
        monkeypatch.setattr(liquidity, 'CRYPTO_CENSUS_FILE', str(p))
        monkeypatch.setattr(liquidity, '_CENSUS_MEMO', None)
        assert liquidity.get_crypto_spread_tier('BTC/USD') == 0.05

    def test_stamp_crypto_spreads_fail_open(self, monkeypatch):
        # any internal error returns the input frame unchanged (fail-open)
        monkeypatch.setattr(liquidity, 'CRYPTO_SPREAD_STAMP', True)
        def _boom(sym):
            raise RuntimeError('tier boom')
        monkeypatch.setattr(liquidity, 'get_crypto_spread_tier', _boom)
        df = _ohlc(n=10)
        out = liquidity.stamp_crypto_spreads(df, 'BTC/USD')
        assert out is df
        assert 'Eff_Spread_Pct' not in out.columns

    def test_minute_edge_missing_ohlc_all_nan(self):
        # malformed minute frame -> all-NaN (hourly stamp retained), no raise
        dates = pd.bdate_range('2025-03-03', periods=3)
        hidx = _hourly_index(dates)
        bad = pd.DataFrame({'Close': [1.0, 2.0]},
                           index=pd.date_range('2025-03-03', periods=2,
                                               freq='min'))
        out = liquidity.edge_spread_daily_from_minute(bad, hidx)
        assert len(out) == len(hidx) and out.isna().all()

    def test_minute_edge_min_days_warmup(self):
        # default min_days=3: first finite smoothed value on covered day 3,
        # shift(1) -> first finite HOURLY stamp on day 4
        pytest.importorskip('bidask')
        mdf, dates = _make_minute()
        hidx = _hourly_index(dates)
        out = liquidity.edge_spread_daily_from_minute(mdf, hidx)  # min_days=3
        day = out.index.normalize()
        assert out[day <= dates[2]].isna().all()
        assert out[day >= dates[3]].notna().all()
