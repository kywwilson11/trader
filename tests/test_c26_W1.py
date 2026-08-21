"""Wave C-3 packet W1 tests — daily-bars cache + stock feature restoration
(D11) + HAR-RV daily feed (D30).

Mac-runnable: imports market_data, indicators, volatility, pandas/numpy only
(predict_now imports torch and gets py_compile only). Baselines are
reconstructed via `git show HEAD:<file>` (never git stash).
"""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import indicators
import market_data
import volatility


# --- helpers -----------------------------------------------------------------

def _load_head_module(rel_path, name, tmp_path):
    """importlib-load the HEAD version of a repo file (read-only baseline)."""
    src = subprocess.run(['git', 'show', f'HEAD:{rel_path}'], cwd=str(REPO),
                         capture_output=True, text=True, check=True).stdout
    p = tmp_path / f'{name}.py'
    p.write_text(src)
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rth_frame(n_calendar_days, seed=0, bars_per_day=7, start='2024-01-02'):
    """Synthetic RTH hourly stock frame (weekdays only, tz-aware UTC)."""
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    px = 100.0
    for d in range(n_calendar_days):
        day = pd.Timestamp(start) + pd.Timedelta(days=d)
        if day.weekday() >= 5:
            continue
        for h in range(bars_per_day):
            r = rng.normal(0, 0.01 / np.sqrt(bars_per_day))
            o = px
            px = px * (1 + r)
            hi = max(o, px) * (1 + abs(rng.normal(0, 0.002)))
            lo = min(o, px) * (1 - abs(rng.normal(0, 0.002)))
            idx.append(pd.Timestamp(f'{day.date()} {13 + h}:30', tz='UTC'))
            rows.append((o, hi, lo, px, 1e6))
    return pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close',
                                       'Volume'],
                        index=pd.DatetimeIndex(idx))


def _crypto_frame(n_days, seed=0, start='2025-06-01', head_day_bars=24,
                  tz=None):
    """Synthetic 24/7 hourly crypto frame (tz-naive by default)."""
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    px = 50000.0
    for d in range(n_days):
        day = pd.Timestamp(start) + pd.Timedelta(days=d)
        nb = head_day_bars if d == 0 else 24
        for h in range(24 - nb, 24):
            r = rng.normal(0, 0.02 / np.sqrt(24))
            o = px
            px = px * (1 + r)
            hi = max(o, px) * (1 + abs(rng.normal(0, 0.003)))
            lo = min(o, px) * (1 - abs(rng.normal(0, 0.003)))
            ts = day + pd.Timedelta(hours=h)
            idx.append(ts.tz_localize(tz) if tz else ts)
            rows.append((o, hi, lo, px, 1e6))
    return pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close',
                                       'Volume'],
                        index=pd.DatetimeIndex(idx))


def _harvest_daily(full):
    """Daily bars aggregated EXACTLY like the harvest's resample('1D')."""
    d = pd.DataFrame({
        'Open': full['Open'].resample('1D').first(),
        'High': full['High'].resample('1D').max(),
        'Low': full['Low'].resample('1D').min(),
        'Close': full['Close'].resample('1D').last(),
        'Volume': full['Volume'].resample('1D').sum(),
    }).dropna(subset=['Close'])
    return d


class _Bar:
    def __init__(self, t, o, h, l, c, v):
        self.t, self.o, self.h, self.l, self.c, self.v = t, o, h, l, c, v


class _StubAPI:
    def __init__(self, bars, raise_exc=None):
        self.bars = bars
        self.raise_exc = raise_exc
        self.calls = 0

    def get_bars(self, symbol, timeframe, start=None, adjustment=None):
        self.calls += 1
        assert timeframe == '1Day'
        assert adjustment == 'all'
        if self.raise_exc is not None:
            raise self.raise_exc
        return list(self.bars)


@pytest.fixture
def daily_cache(tmp_path, monkeypatch):
    """Isolated market_data daily cache (tmp file, fresh in-memory state)."""
    monkeypatch.setattr(market_data, '_DAILY_CACHE_FILE',
                        str(tmp_path / 'daily_bars_cache.json'))
    monkeypatch.setattr(market_data, '_daily_cache',
                        {'loaded': False, 'symbols': {}})
    return tmp_path


@pytest.fixture
def har_store(tmp_path, monkeypatch):
    """Isolated volatility HAR RRV store + crypto_rv file + caches."""
    monkeypatch.setattr(volatility, '_HAR_RRV_FILE',
                        str(tmp_path / 'har_rrv_history.json'))
    monkeypatch.setattr(volatility, '_CRYPTO_RV_FILE',
                        str(tmp_path / 'crypto_rv_history.json'))
    monkeypatch.setattr(volatility, '_har_rrv_store',
                        {'loaded': False, 'symbols': {}})
    volatility._har_cache.clear()
    volatility._har_gap_logged.clear()
    yield tmp_path
    volatility._har_cache.clear()
    volatility._har_gap_logged.clear()


# --- T1: flag readers --------------------------------------------------------

class TestFlags:
    @pytest.mark.parametrize('env,reader', [
        ('TRADER_DAILY_FEATURE_RESTORE', market_data.daily_feature_restore_enabled),
        ('TRADER_HAR_DAILY_FEED', market_data.har_daily_feed_enabled),
        ('TRADER_HAR_DAILY_FEED', volatility.har_daily_feed_enabled),
    ])
    def test_default_off_and_call_time(self, monkeypatch, env, reader):
        monkeypatch.delenv(env, raising=False)
        assert reader() is False
        monkeypatch.setenv(env, '0')
        assert reader() is False
        monkeypatch.setenv(env, '1')
        assert reader() is True
        monkeypatch.setenv(env, 'true')
        assert reader() is True
        monkeypatch.setenv(env, 'no')
        assert reader() is False


# --- T2: refactor byte-parity (the D11 flag-OFF pin) -------------------------

class TestRefactorParity:
    def test_compute_stock_features_byte_identical(self, tmp_path):
        base = _load_head_module('indicators.py', 'indicators_head', tmp_path)
        frame = _rth_frame(45, seed=7)
        spy = _rth_frame(45, seed=8)['Close']
        old = base.compute_stock_features(frame.copy(), spy_close=spy,
                                          symbol='TSLA')
        new = indicators.compute_stock_features(frame.copy(), spy_close=spy,
                                                symbol='TSLA')
        pd.testing.assert_frame_equal(old, new, check_exact=True)


# --- T3: constant-set regression --------------------------------------------

class TestConstantSet:
    def test_exact_constant_columns_on_45d_frame(self):
        frame = _rth_frame(45, seed=7)
        spy = _rth_frame(45, seed=8)['Close']
        df = indicators.compute_stock_features(frame.copy(), spy_close=spy,
                                               symbol='TSLA')
        warm = (list(indicators.WARMUP_FEATURES_ZERO)
                + list(indicators.WARMUP_FEATURES_HALF))
        all_nan = {c for c in warm if c in df.columns and df[c].isna().all()}
        assert all_nan == set(indicators.DAILY_RESTORE_COLUMNS)
        n_const, n_present = indicators.count_warmup_constant_columns(df)
        assert n_const == 9
        assert n_present == 20   # SVR_21/SVR_Z live in short_flow, absent here
        for col in ('Same_Hour_Mean_40d', 'Ret_21d', 'Pos_Range_20d',
                    'ROD_Ret', 'ON_Mom_21', 'MA_Dist_20d'):
            assert not df[col].isna().all(), col


# --- T4: daily-bars cache ----------------------------------------------------

def _stub_daily_bars(n_days=30, include_today=True):
    from datetime import datetime, timedelta, timezone
    today = datetime.now(timezone.utc).date()
    bars = []
    for i in range(n_days, 0, -1):
        d = today - timedelta(days=i)
        px = 100.0 + i
        bars.append(_Bar(pd.Timestamp(d), px, px * 1.01, px * 0.99,
                         px * 1.005, 1e6))
    if include_today:
        bars.append(_Bar(pd.Timestamp(today), 99.0, 99.5, 98.5, 99.2, 5e5))
    return bars, today


class TestDailyCache:
    def test_refresh_stores_complete_days_only(self, daily_cache):
        bars, today = _stub_daily_bars(30, include_today=True)
        api = _StubAPI(bars)
        assert market_data.refresh_daily_bars(api, 'TSLA') is True
        df = market_data.load_daily_bars('TSLA')
        assert df is not None and len(df) == 30
        assert df.index[-1].date() < today          # today excluded
        assert df.index.is_monotonic_increasing
        assert str(df.index.tz) == 'UTC'
        assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        # OHLCV round-trip on the last complete day
        last = bars[-2]
        assert df['Open'].iloc[-1] == last.o
        assert df['Volume'].iloc[-1] == last.v
        assert market_data.daily_bars_fetched_at('TSLA') is not None

    def test_refresh_ttl_throttles_api(self, daily_cache):
        bars, _ = _stub_daily_bars(10)
        api = _StubAPI(bars)
        assert market_data.refresh_daily_bars(api, 'TSLA') is True
        assert market_data.refresh_daily_bars(api, 'TSLA') is True
        assert api.calls == 1                       # second within TTL: no call

    def test_refresh_failure_keeps_previous_entry(self, daily_cache):
        bars, _ = _stub_daily_bars(10)
        assert market_data.refresh_daily_bars(_StubAPI(bars), 'TSLA') is True
        before = market_data.load_daily_bars('TSLA')
        # force TTL expiry, then a raising API
        market_data._daily_cache['symbols']['TSLA']['fetched_at'] = 0.0
        bad = _StubAPI([], raise_exc=RuntimeError('boom'))
        assert market_data.refresh_daily_bars(bad, 'TSLA') is False
        after = market_data.load_daily_bars('TSLA')
        pd.testing.assert_frame_equal(before, after)

    def test_atomic_write_no_tmp_residue_and_roundtrip(self, daily_cache):
        bars, _ = _stub_daily_bars(10)
        market_data.refresh_daily_bars(_StubAPI(bars), 'TSLA')
        assert not os.path.exists(market_data._DAILY_CACHE_FILE + '.tmp')
        assert os.path.exists(market_data._DAILY_CACHE_FILE)
        # a fresh process (cleared in-memory state) reads the same bars
        saved = market_data.load_daily_bars('TSLA')
        market_data._daily_cache['loaded'] = False
        market_data._daily_cache['symbols'] = {}
        pd.testing.assert_frame_equal(market_data.load_daily_bars('TSLA'),
                                      saved)

    def test_max_rows_cap(self, daily_cache):
        bars, _ = _stub_daily_bars(400)
        market_data.refresh_daily_bars(_StubAPI(bars), 'TSLA')
        df = market_data.load_daily_bars('TSLA')
        assert len(df) == market_data._DAILY_CACHE_MAX_ROWS

    def test_load_missing_symbol_is_none(self, daily_cache):
        assert market_data.load_daily_bars('NOPE') is None
        assert market_data.daily_bars_fetched_at('NOPE') is None


# --- T5: builder parity (Mac analogue of the Jetson bit-parity check) --------

class TestBuilderParity:
    @pytest.fixture(scope='class')
    def parity_setup(self):
        full = _rth_frame(400, seed=11)
        spy_full = _rth_frame(400, seed=12)['Close']
        harvest = indicators.compute_stock_features(full.copy(),
                                                    spy_close=spy_full,
                                                    symbol='TSLA')
        tail = full[full.index >= full.index[-1] - pd.Timedelta(days=45)]
        spy_tail = spy_full.reindex(tail.index)
        last_norm = full.index[-1].normalize()
        daily = _harvest_daily(full)
        cache_daily = daily[daily.index < last_norm]     # complete days only
        spy_daily = spy_full.resample('1D').last().dropna()
        spy_daily = spy_daily[spy_daily.index < last_norm]
        return full, harvest, tail, spy_tail, cache_daily, spy_daily

    def test_restore_matches_harvest_values(self, parity_setup):
        full, harvest, tail, spy_tail, cache_daily, spy_daily = parity_setup
        live = indicators.compute_stock_features(tail.copy(),
                                                 spy_close=spy_tail,
                                                 symbol='TSLA')
        # pre-restore: the 9 columns are all-NaN on the live frame
        for col in indicators.DAILY_RESTORE_COLUMNS:
            assert live[col].isna().all(), col
        live, n_restored, n_left = indicators.apply_daily_restore(
            live, cache_daily, spy_daily, 'TSLA')
        assert n_restored == 9 and n_left == 0
        for col in indicators.DAILY_RESTORE_COLUMNS:
            got = live[col]
            exp = harvest[col].reindex(tail.index)
            pd.testing.assert_series_equal(got, exp, check_names=False,
                                           check_exact=True)
            # includes the current-day rows via the tail-extension mapping
            assert np.isfinite(got.iloc[-1])

    def test_stale_cache_degrades_to_warmup_fill(self, parity_setup):
        full, harvest, tail, spy_tail, cache_daily, spy_daily = parity_setup
        flag_off = indicators.fill_warmup_features(
            indicators.compute_stock_features(tail.copy(), spy_close=spy_tail,
                                              symbol='TSLA'))
        live = indicators.compute_stock_features(tail.copy(),
                                                 spy_close=spy_tail,
                                                 symbol='TSLA')
        stale = cache_daily.iloc[:-5]                # 5 sessions behind
        live, n_restored, _ = indicators.apply_daily_restore(
            live, stale, spy_daily, 'TSLA')
        assert n_restored == 0                       # last row NaN -> skipped
        live = indicators.fill_warmup_features(live)
        pd.testing.assert_frame_equal(live, flag_off, check_exact=True)

    def test_no_spy_daily_skips_rm_rr_only(self, parity_setup):
        full, harvest, tail, spy_tail, cache_daily, spy_daily = parity_setup
        live = indicators.compute_stock_features(tail.copy(),
                                                 spy_close=spy_tail,
                                                 symbol='TSLA')
        live, n_restored, n_left = indicators.apply_daily_restore(
            live, cache_daily, None, 'TSLA')
        assert n_restored == 6 and n_left == 3       # RM_252_21, RR_5, RR_21
        for col in ('RM_252_21', 'RR_5', 'RR_21'):
            assert live[col].isna().all(), col

    def test_apply_daily_restore_never_raises(self):
        frame = _rth_frame(45, seed=7)
        df = indicators.compute_stock_features(frame.copy(), symbol='TSLA')
        out, n_restored, n_left = indicators.apply_daily_restore(
            df, 'not a dataframe', None, 'TSLA')
        assert out is df and n_restored == 0
        assert n_left == len(indicators.DAILY_RESTORE_COLUMNS)


# --- T6: HAR math ------------------------------------------------------------

class TestHarMath:
    def _rrv_series(self, n_days, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.DatetimeIndex([pd.Timestamp('2024-01-01')
                                + pd.Timedelta(days=i) for i in range(n_days)])
        vals = np.exp(rng.normal(-9.0, 0.5, n_days))
        return pd.Series(vals, index=idx)

    def test_matches_head_har_forecast_sigma(self, tmp_path):
        base = _load_head_module('volatility.py', 'volatility_head', tmp_path)
        bars = _rth_frame(220, seed=3)
        rrv = volatility.daily_realized_range(bars)
        expected = base.har_forecast_sigma(bars, 'stock')
        assert expected is not None
        assert volatility._har_sigma_from_rrv(rrv, 'stock') == expected
        assert volatility.har_forecast_sigma(bars, 'stock') == expected

    def test_sixty_complete_days_passes_guard(self):
        rrv = self._rrv_series(60, seed=1)
        # m22 leaves 38 regression rows; 38 < (60-22) is False -> fit runs
        assert volatility._har_sigma_from_rrv(rrv, 'stock') is not None

    def test_below_min_days_returns_none(self):
        assert volatility._har_sigma_from_rrv(
            self._rrv_series(59, seed=1), 'stock') is None

    def test_shrink_changes_small_n_output(self):
        rrv = self._rrv_series(60, seed=2)
        plain = volatility._har_sigma_from_rrv(rrv, 'stock', shrink=False)
        shrunk = volatility._har_sigma_from_rrv(rrv, 'stock', shrink=True)
        assert plain is not None and shrunk is not None
        assert plain != shrunk

    def test_shrink_formula_exact(self):
        # Deterministic pin of the B11 shrinkage algebra: lam = n/(n+120),
        # prior slopes (0.40, 0.30, 0.25), prior intercept from the sample
        # means — replicated here and compared exactly.
        rrv = self._rrv_series(90, seed=3)
        got = volatility._har_sigma_from_rrv(rrv, 'stock', shrink=True)
        assert got is not None

        r = rrv[rrv > 0].tail(volatility._HAR_WINDOW)
        m5 = r.rolling(5, min_periods=5).mean()
        m22 = r.rolling(22, min_periods=22).mean()
        df = pd.DataFrame({'y': np.log(r).shift(-1), 'x1': np.log(r),
                           'x2': np.log(m5), 'x3': np.log(m22)}).dropna()
        X = np.column_stack([np.ones(len(df)), df['x1'], df['x2'], df['x3']])
        beta, *_ = np.linalg.lstsq(X, df['y'].values, rcond=None)
        n = len(df)
        lam = n / (n + 120.0)
        prior_intercept = (float(df['y'].mean()) - 0.40 * float(df['x1'].mean())
                           - 0.30 * float(df['x2'].mean())
                           - 0.25 * float(df['x3'].mean()))
        beta = lam * beta + (1.0 - lam) * np.array(
            [prior_intercept, 0.40, 0.30, 0.25])
        resid = df['y'].values - X @ beta
        sig2 = float(np.var(resid, ddof=4))
        x_now = np.array([1.0, np.log(r.iloc[-1]), np.log(m5.iloc[-1]),
                          np.log(m22.iloc[-1])])
        rrv_hat = float(np.exp(x_now @ beta + 0.5 * sig2))
        rrv_hat = min(max(rrv_hat, float(r.min())), float(r.max()))
        expected = float(np.sqrt(rrv_hat) / np.sqrt(6.5))
        assert got == expected

    def test_c_scale_applied_after_clamp(self):
        rrv = self._rrv_series(120, seed=4)
        s1 = volatility._har_sigma_from_rrv(rrv, 'stock', c_scale=1.0)
        s2 = volatility._har_sigma_from_rrv(rrv, 'stock', c_scale=2.0)
        assert s2 == pytest.approx(s1 * np.sqrt(2.0), rel=1e-12)

    def _seed_stock_cache(self, monkeypatch, closes, sym='XYZ'):
        bars = {}
        for i, c in enumerate(closes):
            d = (pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)).date()
            # tiny true range so close-to-close variance dominates
            bars[d.isoformat()] = [c, c * 1.0001, c * 0.9999, c, 1e6]
        monkeypatch.setattr(market_data, '_daily_cache', {
            'loaded': True,
            'symbols': {sym: {'fetched_at': time.time(), 'bars': bars}}})

    def test_hansen_lunde_c_clamps(self, monkeypatch, har_store):
        captured = []

        def spy_sigma(rrv, asset_type, shrink=False, c_scale=1.0):
            captured.append(c_scale)
            return 0.001

        monkeypatch.setattr(volatility, '_har_sigma_from_rrv', spy_sigma)
        # (a) big close moves vs tiny ranges -> c clamped at 2.5
        closes = [100.0 * (1.05 if i % 2 else 0.95) for i in range(80)]
        self._seed_stock_cache(monkeypatch, closes)
        assert volatility._har_feed_sigma('XYZ', None, 'stock') == 0.001
        assert captured[-1] == 2.5
        # (b) constant closes -> c floors at 1.0
        volatility._har_cache.clear()
        self._seed_stock_cache(monkeypatch, [100.0] * 80)
        assert volatility._har_feed_sigma('XYZ', None, 'stock') == 0.001
        assert captured[-1] == 1.0

    def test_merge_excludes_partial_last_day(self, har_store):
        bars = _crypto_frame(10, seed=5)
        history = {}
        volatility._merge_complete_day_rrvs(history, bars, 250)
        last_day = bars.index[-1].normalize().date().isoformat()
        assert last_day not in history
        assert len(history) == 9


# --- T7: get_sigma flag-OFF byte-identity + feed path ------------------------

class TestGetSigmaFeed:
    def test_flag_off_identical_to_head_behavior(self, monkeypatch, har_store):
        monkeypatch.delenv('TRADER_HAR_DAILY_FEED', raising=False)
        calls = []
        real = volatility.har_forecast_sigma

        def counting(bars, at):
            calls.append(1)
            return real(bars, at)

        monkeypatch.setattr(volatility, 'har_forecast_sigma', counting)
        bars = _rth_frame(220, seed=3)
        s1 = volatility.get_sigma('ABC', np.zeros(10), bars=bars,
                                  asset_type='stock')
        s2 = volatility.get_sigma('ABC', np.zeros(10), bars=bars,
                                  asset_type='stock')
        assert s1 == s2 and s1 is not None
        assert len(calls) == 1
        # legacy cache key: the frame's LAST (forming) day, as a date
        assert volatility._har_cache['ABC'][0] == bars.index[-1].date()
        # the HAR RRV store is never touched with the flag off
        assert not os.path.exists(volatility._HAR_RRV_FILE)

    def test_flag_on_crypto_feed(self, monkeypatch, har_store):
        monkeypatch.setenv('TRADER_HAR_DAILY_FEED', '1')
        sym = 'SOL/USD'
        seed_days = {}
        rng = np.random.default_rng(9)
        for i in range(80):
            d = (pd.Timestamp('2025-03-01') + pd.Timedelta(days=i)).date()
            seed_days[d.isoformat()] = float(np.exp(rng.normal(-8.0, 0.4)))
        Path(volatility._HAR_RRV_FILE).write_text(json.dumps({sym: seed_days}))

        def garch_must_not_run(s, r):
            raise AssertionError('GARCH fit must not be reached')

        monkeypatch.setattr(volatility, 'get_cached_sigma', garch_must_not_run)
        bars = _crypto_frame(10, seed=5)
        sigma = volatility.get_sigma(sym, np.zeros(10), bars=bars,
                                     asset_type='crypto')
        assert sigma is not None and sigma > 0
        # cache keyed on the last COMPLETE day, not the forming one
        expected_key = pd.Timestamp(
            (bars.index[-1].normalize() - pd.Timedelta(days=1)).date())
        assert volatility._har_cache[sym][0] == expected_key
        # second call: served from the day cache, same value
        assert volatility.get_sigma(sym, np.zeros(10), bars=bars,
                                    asset_type='crypto') == sigma
        # store persisted atomically with the merged complete days
        assert os.path.exists(volatility._HAR_RRV_FILE)
        assert not os.path.exists(volatility._HAR_RRV_FILE + '.tmp')
        stored = json.loads(Path(volatility._HAR_RRV_FILE).read_text())
        assert len(stored[sym]) == 80 + 9            # 9 new complete days

    def test_flag_on_thin_history_falls_back(self, monkeypatch, har_store):
        monkeypatch.setenv('TRADER_HAR_DAILY_FEED', '1')
        monkeypatch.setattr(volatility, 'get_cached_sigma',
                            lambda s, r: 0.0123)
        bars = _crypto_frame(10, seed=6)
        # 10-day frame alone (~9 complete days) < 60 -> feed returns None,
        # legacy HAR also thin -> GARCH fallback
        out = volatility.get_sigma('DOGE/USD', np.zeros(10), bars=bars,
                                   asset_type='crypto')
        assert out == 0.0123

    def test_store_capped_at_har_window(self, monkeypatch, har_store):
        monkeypatch.setenv('TRADER_HAR_DAILY_FEED', '1')
        sym = 'ETH/USD'
        seed_days = {}
        rng = np.random.default_rng(10)
        for i in range(260):
            d = (pd.Timestamp('2024-06-01') + pd.Timedelta(days=i)).date()
            seed_days[d.isoformat()] = float(np.exp(rng.normal(-8.0, 0.4)))
        Path(volatility._HAR_RRV_FILE).write_text(json.dumps({sym: seed_days}))
        bars = _crypto_frame(10, seed=7, start='2025-06-01')
        sigma = volatility.get_sigma(sym, np.zeros(300), bars=bars,
                                     asset_type='crypto')
        assert sigma is not None and sigma > 0
        assert len(volatility._har_rrv_store['symbols'][sym]) <= \
            volatility._HAR_WINDOW

    def test_merge_min_bars_none_matches_b3_inline(self, har_store):
        bars = _crypto_frame(12, seed=8)
        new_hist = {'2000-01-01': 1e-4}              # pre-existing entry
        volatility._merge_complete_day_rrvs(new_hist, bars, 365)
        # replicate the original Wave B-3 inline logic verbatim
        old_hist = {'2000-01-01': 1e-4}
        rrv = volatility.daily_realized_range(bars)
        last_day = bars.index[-1].normalize()
        for day, val in rrv.items():
            if day != last_day and np.isfinite(val) and val > 0:
                old_hist[day.date().isoformat()] = float(val)
        if len(old_hist) > 365:
            for k in sorted(old_hist)[:-365]:
                del old_hist[k]
        assert new_hist == old_hist

    def test_merge_min_bars_skips_thin_head_day(self, har_store):
        bars = _crypto_frame(10, seed=9, head_day_bars=5)
        history = {}
        changed = volatility._merge_complete_day_rrvs(history, bars, 250,
                                                      min_bars=20)
        assert changed is True
        head_day = bars.index[0].normalize().date().isoformat()
        assert head_day not in history               # 5-bar day skipped
        assert len(history) == 8                     # 9 complete - 1 thin

    def test_btc_seeds_from_crypto_rv_history(self, har_store):
        rrv = {}
        rng = np.random.default_rng(11)
        for i in range(100):
            d = (pd.Timestamp('2025-01-01') + pd.Timedelta(days=i)).date()
            rrv[d.isoformat()] = float(np.exp(rng.normal(-8.0, 0.4)))
        Path(volatility._CRYPTO_RV_FILE).write_text(json.dumps({
            'rrv': rrv, 'state': 'normal', 'exit_count': 0,
            'last_bar_ts': None}))
        volatility._har_rrv_load()
        assert len(volatility._har_rrv_store['symbols']['BTC/USD']) == 100

    def test_update_crypto_rv_state_unchanged(self, har_store, tmp_path,
                                              monkeypatch):
        # B-3 pin: the factored merge leaves update_crypto_rv_state's
        # history contents identical to the inline original.
        volatility._reset_crypto_rv_state()
        bars = _crypto_frame(12, seed=12)
        volatility.update_crypto_rv_state('BTC/USD', bars)
        history = dict(volatility._crypto_rv['history'])
        expected = {}
        rrv = volatility.daily_realized_range(bars)
        last_day = bars.index[-1].normalize()
        for day, val in rrv.items():
            if day != last_day and np.isfinite(val) and val > 0:
                expected[day.date().isoformat()] = float(val)
        assert history == expected
        volatility._reset_crypto_rv_state()


# --- Hardening pass: rebind safety, column invariance, stock feed e2e --------

class TestHardening:
    def _parity_inputs(self, seed):
        full = _rth_frame(400, seed=seed)
        daily = _harvest_daily(full)
        last_norm = full.index[-1].normalize()
        cache_daily = daily[daily.index < last_norm]
        tail = full[full.index >= full.index[-1] - pd.Timedelta(days=45)]
        return full, tail, cache_daily

    def test_map_global_restored_after_failure(self):
        # A daily frame missing 'Open' raises INSIDE the rebound region
        # (_session_momentum's daily_bars['Open'] arg) — the finally must
        # put the original _map_daily_to_hourly back.
        orig = indicators._map_daily_to_hourly
        full, tail, cache_daily = self._parity_inputs(21)
        broken = cache_daily.drop(columns=['Open'])
        live = indicators.compute_stock_features(tail.copy(), symbol='TSLA')
        out, n_restored, n_left = indicators.apply_daily_restore(
            live, broken, None, 'TSLA')
        assert n_restored == 0
        assert n_left == len(indicators.DAILY_RESTORE_COLUMNS)
        assert indicators._map_daily_to_hourly is orig

    def test_map_global_restored_after_success(self):
        orig = indicators._map_daily_to_hourly
        full, tail, cache_daily = self._parity_inputs(22)
        feats = indicators.build_daily_restore_features(tail, cache_daily,
                                                        None, 'TSLA')
        assert feats
        assert indicators._map_daily_to_hourly is orig

    def test_foreign_thread_gets_unextended_mapping(self, monkeypatch):
        # Combined-bots hazard: another thread computing stock features
        # mid-build must see the ORIGINAL mapping, not the tail-extended
        # wrapper. Probe: a constant daily series covering every cache day;
        # the frame's final (beyond-tail) date maps to a real value only
        # under the extension.
        import threading
        full, tail, cache_daily = self._parity_inputs(23)
        probe_series = cache_daily['Close'] * 0 + 1.0
        probe_dates = pd.Series(tail.index.normalize(), index=tail.index)
        # sanity: the plain mapping leaves the final rows NaN
        plain = indicators._map_daily_to_hourly(probe_series, probe_dates)
        assert np.isnan(plain[-1])

        real_ma = indicators._ma_distances
        results = {}

        def instrumented_ma(daily_close, daily_dates):
            th = threading.Thread(target=lambda: results.__setitem__(
                'foreign',
                indicators._map_daily_to_hourly(probe_series, probe_dates)))
            th.start()
            th.join()
            results['building'] = indicators._map_daily_to_hourly(
                probe_series, probe_dates)
            return real_ma(daily_close, daily_dates)

        monkeypatch.setattr(indicators, '_ma_distances', instrumented_ma)
        indicators.build_daily_restore_features(tail, cache_daily, None,
                                                'TSLA')
        assert np.isfinite(results['building'][-1])   # extended for builder
        assert np.isnan(results['foreign'][-1])       # untouched elsewhere

    def test_apply_daily_restore_leaves_other_columns_untouched(self):
        # The dropped-D11 invariant: short-window daily features already
        # carry real live values and MUST NOT be overwritten by the restore.
        full = _rth_frame(400, seed=24)
        spy_full = _rth_frame(400, seed=25)['Close']
        daily = _harvest_daily(full)
        last_norm = full.index[-1].normalize()
        cache_daily = daily[daily.index < last_norm]
        spy_daily = spy_full.resample('1D').last().dropna()
        spy_daily = spy_daily[spy_daily.index < last_norm]
        tail = full[full.index >= full.index[-1] - pd.Timedelta(days=45)]
        spy_tail = spy_full.reindex(tail.index)
        live = indicators.compute_stock_features(tail.copy(),
                                                 spy_close=spy_tail,
                                                 symbol='TSLA')
        before = live.copy(deep=True)
        out, n_restored, _ = indicators.apply_daily_restore(
            live, cache_daily, spy_daily, 'TSLA')
        assert n_restored == 9
        other = [c for c in before.columns
                 if c not in indicators.DAILY_RESTORE_COLUMNS]
        pd.testing.assert_frame_equal(out[other], before[other],
                                      check_exact=True)

    def test_flag_on_stock_feed_via_daily_cache(self, monkeypatch, har_store):
        # End-to-end stock feed: get_sigma -> _har_feed_sigma ->
        # market_data.load_daily_bars, GARCH never reached, cache keyed on
        # the last COMPLETE cached day (tz-aware), not the hourly frame's
        # forming day.
        monkeypatch.setenv('TRADER_HAR_DAILY_FEED', '1')
        bars = {}
        rng = np.random.default_rng(30)
        px = 100.0
        for i in range(80):
            d = (pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)).date()
            o = px
            px *= float(np.exp(rng.normal(0, 0.015)))
            hi = max(o, px) * float(1 + abs(rng.normal(0, 0.004)))
            lo = min(o, px) * float(1 - abs(rng.normal(0, 0.004)))
            bars[d.isoformat()] = [o, hi, lo, px, 1e6]
        monkeypatch.setattr(market_data, '_daily_cache', {
            'loaded': True,
            'symbols': {'TSLA': {'fetched_at': time.time(), 'bars': bars}}})

        def garch_must_not_run(s, r):
            raise AssertionError('GARCH fit must not be reached')

        monkeypatch.setattr(volatility, 'get_cached_sigma', garch_must_not_run)
        hourly = _rth_frame(10, seed=31)
        sigma = volatility.get_sigma('TSLA', np.zeros(10), bars=hourly,
                                     asset_type='stock')
        assert sigma is not None and sigma > 0
        key, cached = volatility._har_cache['TSLA']
        assert cached == sigma
        assert key == pd.Timestamp(sorted(bars)[-1], tz='UTC')
        assert key != hourly.index[-1].date()         # not the forming day
        # second call: served from the complete-day cache
        assert volatility.get_sigma('TSLA', np.zeros(10), bars=hourly,
                                    asset_type='stock') == sigma
