"""Tests for wave 4: HAR-RV, feature suite, shorting flow, sleeve blend."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))


def _ohlc(n_days, day_vol, seed=0, bars_per_day=7):
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    px = 100.0
    for d in range(n_days):
        day = pd.Timestamp('2025-01-02') + pd.Timedelta(days=d)
        if day.weekday() >= 5:
            continue
        for h in range(bars_per_day):
            r = rng.normal(0, day_vol / np.sqrt(bars_per_day))
            o = px
            px = px * (1 + r)
            hi = max(o, px) * (1 + abs(rng.normal(0, day_vol / 4)))
            lo = min(o, px) * (1 - abs(rng.normal(0, day_vol / 4)))
            idx.append(pd.Timestamp(f'{day.date()} {13 + h}:30', tz='UTC'))
            rows.append((o, hi, lo, px))
    df = pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close'],
                      index=pd.DatetimeIndex(idx))
    df['Volume'] = 1e6
    return df


class TestHARRV:
    def test_sigma_tracks_vol_regime(self):
        from volatility import har_forecast_sigma
        calm = har_forecast_sigma(_ohlc(150, 0.005, seed=1), 'stock')
        wild = har_forecast_sigma(_ohlc(150, 0.030, seed=1), 'stock')
        assert calm is not None and wild is not None
        assert wild > 2 * calm
        assert 0 < calm < 0.05  # sane per-bar decimal

    def test_insufficient_history_returns_none(self):
        from volatility import har_forecast_sigma
        assert har_forecast_sigma(_ohlc(30, 0.01), 'stock') is None

    def test_insanity_filter_clamps(self):
        from volatility import har_forecast_sigma, daily_realized_range
        bars = _ohlc(150, 0.01, seed=2)
        sigma = har_forecast_sigma(bars, 'stock')
        rrv = daily_realized_range(bars)
        hi = np.sqrt(rrv.max() / 6.5)
        lo = np.sqrt(rrv[rrv > 0].min() / 6.5)
        assert lo * 0.99 <= sigma <= hi * 1.01

    def test_get_sigma_falls_back_without_bars(self, monkeypatch):
        import volatility
        monkeypatch.setattr(volatility, 'get_cached_sigma',
                            lambda s, r: 0.0123)
        out = volatility.get_sigma('XYZ', np.random.normal(0, 1, 300),
                                   bars=None, asset_type='stock')
        assert out == 0.0123

    def test_get_sigma_prefers_har_and_caches(self, monkeypatch):
        import volatility
        volatility._har_cache.clear()
        calls = []
        real = volatility.har_forecast_sigma

        def counting(bars, at):
            calls.append(1)
            return real(bars, at)

        monkeypatch.setattr(volatility, 'har_forecast_sigma', counting)
        bars = _ohlc(150, 0.01, seed=3)
        s1 = volatility.get_sigma('ABC', np.zeros(10), bars=bars,
                                  asset_type='stock')
        s2 = volatility.get_sigma('ABC', np.zeros(10), bars=bars,
                                  asset_type='stock')
        assert s1 == s2 and s1 is not None
        assert len(calls) == 1  # second call served from the day cache


class TestWave4Features:
    def _features(self, n_days=320, seed=4, on_drift=0.0):
        from indicators import compute_stock_features
        rng = np.random.default_rng(seed)
        rows, idx = [], []
        prev_close = 100.0
        for d in range(n_days):
            day = pd.Timestamp('2025-01-02') + pd.Timedelta(days=d)
            if day.weekday() >= 5:
                continue
            o = prev_close * (1 + on_drift + rng.normal(0, 0.002))
            closes = o * np.cumprod(1 + rng.normal(0, 0.003, 7))
            px = o
            for h, c in enumerate(closes):
                idx.append(pd.Timestamp(f'{day.date()} {13 + h}:30', tz='UTC'))
                rows.append((px, max(px, c) * 1.001, min(px, c) * 0.999, c))
                px = c
            prev_close = closes[-1]
        df = pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close'],
                          index=pd.DatetimeIndex(idx))
        df['Volume'] = 1e6
        spy = df['Close'] * 5 + np.random.default_rng(9).normal(0, 1, len(df))
        return compute_stock_features(df, spy_close=spy)

    def test_columns_exist_and_bounded(self):
        out = self._features()
        for col in ('RR_5', 'RR_21', 'MA_Dist_50d', 'MA_Dist_200d',
                    'ON_Mom_21', 'ON_Mom_252', 'TugOfWar_252',
                    'Pos_Range_20h', 'MidRange_Gap_20h', 'Pos_Range_20d'):
            assert col in out.columns, col
            assert out[col].notna().sum() > 0, col
        pr = out['Pos_Range_20h'].dropna()
        assert ((pr >= 0) & (pr <= 1)).all()
        mg = out['MidRange_Gap_20h'].dropna()
        assert ((mg >= -1) & (mg <= 1)).all()

    def test_overnight_drift_shows_in_on_mom(self):
        # Persistent positive overnight gap -> ON_Mom_21 ~ drift in %
        out = self._features(on_drift=0.004)
        on = out['ON_Mom_21'].dropna()
        assert on.iloc[-1] == pytest.approx(0.4, abs=0.15)

    def test_daily_features_constant_within_day(self):
        out = self._features()
        last_day = out.index.normalize()[-1]
        day = out[out.index.normalize() == last_day]
        for col in ('MA_Dist_50d', 'ON_Mom_252', 'RR_5', 'Pos_Range_20d'):
            assert day[col].nunique() <= 1, col  # PIT: from completed days


class TestWarmupFill:
    def test_long_warmup_features_survive_dropna(self):
        from harvest_stock_data import _fill_warmup_features
        idx = pd.date_range('2021-01-04', periods=50, freq='h', tz='UTC')
        df = pd.DataFrame({'Close': 100.0, 'RSI': 50.0}, index=idx)
        df['RM_252_21'] = np.nan       # 273-day warmup
        df['Pos_Range_20d'] = np.nan
        out = _fill_warmup_features(df).dropna()
        assert len(out) == 50
        assert (out['RM_252_21'] == 0.0).all()
        assert (out['Pos_Range_20d'] == 0.5).all()  # mid-range neutral


class TestShortFlow:
    HEADER = 'Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market'

    def test_parse_aggregates_venues(self):
        import short_flow
        text = '\n'.join([
            self.HEADER,
            '20260610|NVDA|100|0|300|B',
            '20260610|NVDA|50|0|100|Q',
            '20260610|ZZZZ|9|0|10|B',     # not in panel -> dropped
        ])
        out = short_flow._parse_file(text, keep={'NVDA'})
        assert len(out) == 1
        assert out.iloc[0]['short_vol'] == 150
        assert out.iloc[0]['total_vol'] == 400

    def _mk_archive(self, tmp_path, monkeypatch, n_days=320, ratio=0.4,
                    spike_last=0):
        import short_flow
        monkeypatch.setattr(short_flow, 'ARCHIVE_FILE',
                            tmp_path / 'sf.parquet')
        days = pd.bdate_range('2025-01-02', periods=n_days)
        sv = np.full(n_days, ratio * 1e6)
        if spike_last:
            sv[-spike_last:] = 0.8e6
        arc = pd.DataFrame({'date': days, 'symbol': 'NVDA',
                            'short_vol': sv, 'total_vol': 1e6})
        arc.to_parquet(tmp_path / 'sf.parquet')

    def test_svr_math_and_z_spike(self, tmp_path, monkeypatch):
        import short_flow
        self._mk_archive(tmp_path, monkeypatch, spike_last=25)
        svr, z = short_flow.svr_series('NVDA')
        assert svr.iloc[-1] == pytest.approx(0.8, abs=0.01)
        assert z.iloc[-1] > 2    # sustained shorting-flow surge

    def test_features_shift_one_day(self, tmp_path, monkeypatch):
        import short_flow
        self._mk_archive(tmp_path, monkeypatch)
        last_day = pd.bdate_range('2025-01-02', periods=320)[-1]
        bars = pd.date_range(last_day + pd.Timedelta(hours=14), periods=3,
                             freq='h', tz='UTC')
        out = short_flow.svr_features_for_index('NVDA', bars)
        assert out is not None
        # Bars ON the last print day see the PRIOR day's value
        svr, _ = short_flow.svr_series('NVDA')
        assert out['SVR_21'][0] == pytest.approx(svr.iloc[-2])

    def test_live_uses_latest_completed(self, tmp_path, monkeypatch):
        import short_flow
        self._mk_archive(tmp_path, monkeypatch)
        live = short_flow.live_svr_features('NVDA')
        assert live['SVR_21'] == pytest.approx(0.4, abs=0.01)

    def test_missing_name_none(self, tmp_path, monkeypatch):
        import short_flow
        self._mk_archive(tmp_path, monkeypatch)
        assert short_flow.svr_series('AAPL') is None
