"""Packet T2 tests — data-store integrity (D39 sidecar, D08 provenance,
B15 merge guard/exit codes, D38 closed-bar enforcement).

Mac-runnable: numpy/pandas/pytest/unittest.mock only; all fetches mocked.
"""
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
try:
    import dotenv  # noqa: F401
except ImportError:  # dev-Mac: stub load_dotenv so harvest modules import
    _m = types.ModuleType('dotenv')
    _m.load_dotenv = lambda *a, **k: None
    sys.modules['dotenv'] = _m

import data_utils
import data_sources
import market_data
import panel_ranks
import harvest_stock_data as h
import harvest_crypto_data as hc


# =====================================================================
# data_utils — sidecar kernels
# =====================================================================

def _ohlcv(start, periods, close=100.0, ticker=None, src=None, freq='h'):
    idx = pd.date_range(start, periods=periods, freq=freq, tz='UTC')
    df = pd.DataFrame({
        'Open': close, 'High': close * 1.01, 'Low': close * 0.99,
        'Close': [close + 0.01 * i for i in range(periods)],
        'Volume': 1e6,
    }, index=idx)
    if ticker is not None:
        df['Ticker'] = ticker
    if src is not None:
        df['Src'] = src
    return df


class TestMergeRawOhlcv:
    def test_empty_existing_returns_new_sorted(self):
        new = _ohlcv('2026-01-05', 5).iloc[::-1]  # reversed order
        out = data_utils.merge_raw_ohlcv(pd.DataFrame(), new, 'BTC-USD')
        assert list(out['Ticker'].unique()) == ['BTC-USD']
        assert out.index.is_monotonic_increasing
        assert len(out) == 5

    def test_keep_last_on_ticker_ts(self):
        raw = data_utils.merge_raw_ohlcv(
            pd.DataFrame(), _ohlcv('2026-01-05', 10, close=100.0), 'AAA')
        # Overlap the last 4 bars with NEW closes; extend 3 more
        new = _ohlcv('2026-01-05 06:00', 7, close=555.0)
        out = data_utils.merge_raw_ohlcv(raw, new, 'AAA')
        assert len(out) == 13
        # Overlapping timestamps carry the NEW closes (keep='last')
        ts = pd.Timestamp('2026-01-05 06:00', tz='UTC')
        assert out.loc[ts, 'Close'] == pytest.approx(555.0)

    def test_other_tickers_untouched(self):
        raw = data_utils.merge_raw_ohlcv(
            pd.DataFrame(), _ohlcv('2026-01-05', 5, close=50.0), 'AAA')
        out = data_utils.merge_raw_ohlcv(
            raw, _ohlcv('2026-01-05', 5, close=99.0), 'BBB')
        a = out[out['Ticker'] == 'AAA']
        assert len(a) == 5
        assert a['Close'].iloc[0] == pytest.approx(50.0)
        assert len(out) == 10
        assert out.index.is_monotonic_increasing

    def test_ticker_column_overwritten(self):
        new = _ohlcv('2026-01-05', 3, ticker='WRONG')
        out = data_utils.merge_raw_ohlcv(pd.DataFrame(), new, 'RIGHT')
        assert set(out['Ticker']) == {'RIGHT'}

    def test_keeps_src_drops_foreign_columns(self):
        new = _ohlcv('2026-01-05', 3, src='alpaca')
        new['Garbage'] = 1.0
        out = data_utils.merge_raw_ohlcv(pd.DataFrame(), new, 'AAA')
        assert 'Src' in out.columns
        assert 'Garbage' not in out.columns


class TestLatestRawTs:
    def test_none_on_empty_or_missing(self):
        assert data_utils.latest_raw_ts(pd.DataFrame(), 'AAA') is None
        no_tick = _ohlcv('2026-01-05', 3)
        assert data_utils.latest_raw_ts(no_tick, 'AAA') is None
        raw = _ohlcv('2026-01-05', 3, ticker='BBB')
        assert data_utils.latest_raw_ts(raw, 'AAA') is None

    def test_per_ticker_max(self):
        raw = pd.concat([_ohlcv('2026-01-05', 3, ticker='AAA'),
                         _ohlcv('2026-01-01', 8, ticker='BBB')])
        ts = data_utils.latest_raw_ts(raw, 'AAA')
        assert ts == pd.Timestamp('2026-01-05 02:00', tz='UTC')


class TestOverlapCloseDivergence:
    def test_identical_overlap(self):
        a = _ohlcv('2026-01-05', 10)
        div, n = data_utils.overlap_close_divergence(a, a.copy())
        assert div == pytest.approx(0.0)
        assert n == 10

    def test_two_pct_shift_detected(self):
        a = _ohlcv('2026-01-05', 10)
        b = a.copy()
        b['Close'] = b['Close'] * 1.02
        div, n = data_utils.overlap_close_divergence(a, b)
        assert n == 10
        assert div > data_utils.OVERLAP_DIVERGENCE_MAX
        assert div == pytest.approx(0.02, rel=1e-6)

    def test_constant_pinned_and_below_threshold_passes(self):
        # Pin the guard constant (wiring refuses strictly ABOVE it) and the
        # sub-threshold path: 0.9% drift must NOT trip the >1% refusal.
        assert data_utils.OVERLAP_DIVERGENCE_MAX == 0.01
        a = _ohlcv('2026-01-05', 10)
        b = a.copy()
        b['Close'] = b['Close'] * 1.009
        div, n = data_utils.overlap_close_divergence(a, b)
        assert n == 10
        assert div < data_utils.OVERLAP_DIVERGENCE_MAX

    def test_disjoint_returns_zero(self):
        a = _ohlcv('2026-01-05', 5)
        b = _ohlcv('2026-02-05', 5)
        assert data_utils.overlap_close_divergence(a, b) == (0.0, 0)

    def test_duplicate_index_deduped(self):
        a = _ohlcv('2026-01-05', 3)
        dup = pd.concat([a.iloc[[0]].assign(Close=100.0), a.iloc[1:]])
        dup = pd.concat([a.iloc[[0]].assign(Close=999.0), dup])  # stale first
        b = a.copy()
        b.loc[b.index[0], 'Close'] = 100.0
        div, n = data_utils.overlap_close_divergence(dup, b)
        assert n == 3
        assert div < data_utils.OVERLAP_DIVERGENCE_MAX  # keep='last' won


class TestFindInteriorGaps:
    def test_crypto_6h_hole_found(self):
        idx = pd.date_range('2026-01-05', periods=24, freq='h', tz='UTC')
        holed = idx.delete(slice(10, 15))  # 6h jump 09:00 -> 15:00
        gaps = data_utils.find_interior_gaps(holed, 'crypto')
        assert len(gaps) == 1
        g0, g1 = gaps[0]
        assert g1 == pd.Timestamp('2026-01-05 15:00', tz='UTC')
        assert g1 - g0 == pd.Timedelta(hours=6)

    def test_contiguous_no_gaps(self):
        idx = pd.date_range('2026-01-05', periods=48, freq='h', tz='UTC')
        assert data_utils.find_interior_gaps(idx, 'crypto') == []

    def test_stock_weekend_not_flagged(self):
        fri = pd.date_range('2026-01-09 14:00', periods=7, freq='h', tz='UTC')
        mon = pd.date_range('2026-01-12 14:00', periods=7, freq='h', tz='UTC')
        idx = fri.append(mon)
        assert data_utils.find_interior_gaps(idx, 'stock') == []

    def test_stock_two_busday_hole_flagged(self):
        mon = pd.date_range('2026-01-05 14:00', periods=7, freq='h', tz='UTC')
        thu = pd.date_range('2026-01-08 14:00', periods=7, freq='h', tz='UTC')
        idx = mon.append(thu)
        gaps = data_utils.find_interior_gaps(idx, 'stock')
        assert len(gaps) == 1
        assert gaps[0][1] == pd.Timestamp('2026-01-08 14:00', tz='UTC')

    def test_max_windows_cap(self):
        pieces = [pd.date_range(f'2026-01-{5 + d} 00:00', periods=4,
                                freq='h', tz='UTC') for d in range(7)]
        idx = pieces[0]
        for p in pieces[1:]:
            idx = idx.append(p)   # 20h holes between the pieces
        gaps = data_utils.find_interior_gaps(idx, 'crypto', max_windows=3)
        assert len(gaps) == 3

    def test_short_index_empty(self):
        assert data_utils.find_interior_gaps(
            pd.DatetimeIndex([]), 'crypto') == []


class TestSidecarIO:
    def test_paths(self):
        assert data_utils.raw_sidecar_path('crypto').name == 'raw_ohlcv.parquet'
        assert data_utils.raw_sidecar_path('stock').name == \
            'stock_raw_ohlcv.parquet'

    def test_load_missing_file_empty(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(data_utils, '_BASE_DIR', tmp_path)
        out = data_utils.load_raw_ohlcv('stock')
        assert out.empty
        assert '[SIDECAR]' in capsys.readouterr().out

    def test_save_failure_graceful_false(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(data_utils, '_BASE_DIR', tmp_path)
        monkeypatch.setattr(data_utils, '_atomic_to_disk',
                            lambda df, p: False)
        assert data_utils.save_raw_ohlcv(_ohlcv('2026-01-05', 3),
                                         'stock') is False
        assert 'WARNING' in capsys.readouterr().out

    def test_save_success_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_utils, '_BASE_DIR', tmp_path)

        def fake_write(df, p):
            p.write_bytes(b'x')
            return True
        monkeypatch.setattr(data_utils, '_atomic_to_disk', fake_write)
        assert data_utils.save_raw_ohlcv(_ohlcv('2026-01-05', 3),
                                         'crypto') is True

    def test_flag_parsing(self, monkeypatch):
        monkeypatch.delenv('TRADER_RAW_SIDECAR', raising=False)
        assert data_utils.raw_sidecar_enabled() is False
        monkeypatch.setenv('TRADER_RAW_SIDECAR', '1')
        assert data_utils.raw_sidecar_enabled() is True
        monkeypatch.setenv('TRADER_RAW_SIDECAR', 'true')
        assert data_utils.raw_sidecar_enabled() is True
        monkeypatch.setenv('TRADER_RAW_SIDECAR', '0')
        assert data_utils.raw_sidecar_enabled() is False


# =====================================================================
# market_data — closed_only / D38 flag / end_date
# =====================================================================

def _fake_api(n=30, base_close=100.0):
    """Fake Alpaca api serving n hourly bars whose LAST bar opens at the
    current hour (still forming). Gently varying prices dodge the
    bad-print filter."""
    now_h = pd.Timestamp.now(tz='UTC').floor('h')
    bars = []
    for i in range(n):
        c = base_close + 0.5 * i
        # Range widens with i so ATR actually depends on the last bar
        w = 0.3 + 0.05 * i
        bars.append(SimpleNamespace(
            o=c - 0.1, h=c + w, l=c - w, c=c, v=1000.0,
            t=now_h - pd.Timedelta(hours=n - 1 - i)))
    api = SimpleNamespace()
    api.get_crypto_bars = lambda symbol, tf, start=None: list(bars)
    api.get_bars = lambda symbol, tf, start=None, adjustment=None: list(bars)
    return api


@pytest.fixture(autouse=True)
def _clean_caches():
    market_data._bar_cache.clear()
    panel_ranks._live_cache = None
    yield
    market_data._bar_cache.clear()
    panel_ranks._live_cache = None


class TestClosedOnly:
    def test_crypto_default_keeps_forming_bar(self, monkeypatch):
        monkeypatch.delenv('TRADER_CLOSED_BARS_V2', raising=False)
        api = _fake_api(30)
        df = market_data.fetch_bars_alpaca(api, 'BTC/USD')
        assert len(df) == 30

    def test_crypto_closed_only_drops_forming_bar(self):
        api = _fake_api(30)
        df = market_data.fetch_bars_alpaca(api, 'BTC/USD', closed_only=True)
        assert len(df) == 29

    def test_stock_closed_only(self):
        api = _fake_api(30)
        raw = market_data.fetch_stock_bars_alpaca(api, 'TSLA')
        market_data._bar_cache.clear()
        closed = market_data.fetch_stock_bars_alpaca(api, 'TSLA',
                                                     closed_only=True)
        assert len(raw) == 30
        assert len(closed) == 29

    def test_cache_stores_raw(self):
        api = _fake_api(30)
        first = market_data.fetch_bars_alpaca(api, 'ETH/USD',
                                              closed_only=True)
        assert len(first) == 29
        # Same key, cached path, closed_only=False -> FULL frame back
        api.get_crypto_bars = lambda *a, **k: []  # cache must serve it
        full = market_data.fetch_bars_alpaca(api, 'ETH/USD')
        assert len(full) == 30
        again_closed = market_data.fetch_bars_alpaca(api, 'ETH/USD',
                                                     closed_only=True)
        assert len(again_closed) == 29

    def test_spy_passthrough(self):
        api = _fake_api(30)
        closed = market_data.fetch_spy_bars_alpaca(api, closed_only=True)
        assert len(closed) == 29

    def test_flag_parsing(self, monkeypatch):
        monkeypatch.delenv('TRADER_CLOSED_BARS_V2', raising=False)
        assert market_data.closed_bars_v2_enabled() is False
        monkeypatch.setenv('TRADER_CLOSED_BARS_V2', '1')
        assert market_data.closed_bars_v2_enabled() is True
        monkeypatch.setenv('TRADER_CLOSED_BARS_V2', 'no')
        assert market_data.closed_bars_v2_enabled() is False


class TestGetLiveAtr:
    def test_flag_off_uses_raw_frame(self, monkeypatch):
        from indicators import compute_atr
        monkeypatch.delenv('TRADER_CLOSED_BARS_V2', raising=False)
        api = _fake_api(40)
        full = market_data.fetch_bars_alpaca(api, 'BTC/USD')
        expected = float(compute_atr(full['High'], full['Low'],
                                     full['Close'], 14).dropna().iloc[-1])
        got = market_data.get_live_atr(api, 'BTC/USD', asset_type='crypto')
        assert got == pytest.approx(expected)

    def test_flag_on_uses_closed_frame(self, monkeypatch):
        from indicators import compute_atr
        api = _fake_api(40)
        full = market_data.fetch_bars_alpaca(api, 'BTC/USD')
        closed = full.iloc[:-1]
        expected = float(compute_atr(closed['High'], closed['Low'],
                                     closed['Close'], 14).dropna().iloc[-1])
        monkeypatch.setenv('TRADER_CLOSED_BARS_V2', '1')
        got = market_data.get_live_atr(api, 'BTC/USD', asset_type='crypto')
        assert got == pytest.approx(expected)
        # And it genuinely differs from the forming-bar ATR here
        full_atr = float(compute_atr(full['High'], full['Low'],
                                     full['Close'], 14).dropna().iloc[-1])
        assert got != pytest.approx(full_atr)


class TestFetchHistoricalEndDate:
    def _run(self, monkeypatch, **kwargs):
        captured = []

        def fake_chunk(api, symbol, start_iso, end_iso, asset_type,
                       max_retries=4):
            captured.append((start_iso, end_iso))
            return []
        monkeypatch.setattr(market_data, '_fetch_chunk', fake_chunk)
        monkeypatch.setattr(market_data.time, 'sleep', lambda s: None)
        market_data.fetch_historical_bars(None, 'TSLA', '2021-01-01',
                                          asset_type='stock', **kwargs)
        return captured

    def test_end_date_bounds_chunks(self, monkeypatch):
        captured = self._run(monkeypatch, end_date='2021-04-01')
        assert captured, 'no chunks fetched'
        last_end = datetime.fromisoformat(captured[-1][1])
        assert last_end == datetime(2021, 4, 1, tzinfo=timezone.utc)
        assert captured[0][0].startswith('2021-01-01')

    def test_default_none_reaches_now(self, monkeypatch):
        captured = self._run(monkeypatch)
        last_end = datetime.fromisoformat(captured[-1][1])
        assert (datetime.now(timezone.utc) - last_end).total_seconds() < 3600


# =====================================================================
# panel_ranks — D38 flag plumbing
# =====================================================================

class TestPanelClosedOnly:
    def _capture(self, monkeypatch):
        seen = []

        def fake_fetch(api, sym, limit=320, closed_only=False):
            seen.append(closed_only)
            return None
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                            fake_fetch)
        return seen

    def test_flag_off_passes_false(self, monkeypatch):
        monkeypatch.delenv('TRADER_CLOSED_BARS_V2', raising=False)
        seen = self._capture(monkeypatch)
        out = panel_ranks.compute_live_panel_ranks(api=None)
        assert out == {}
        assert seen and all(v is False for v in seen)

    def test_flag_on_passes_true(self, monkeypatch):
        monkeypatch.setenv('TRADER_CLOSED_BARS_V2', '1')
        seen = self._capture(monkeypatch)
        out = panel_ranks.compute_live_panel_ranks(api=None)
        assert out == {}
        assert seen and all(v is True for v in seen)


# =====================================================================
# data_sources — Src provenance + TRADER_YF_WINDOW_SLICE
# =====================================================================

def _yf_module(frame):
    m = mock.MagicMock()
    m.download.return_value = frame
    return m


class TestSrcStamp:
    def test_alpaca_wins_collisions_and_src_carried(self, monkeypatch):
        monkeypatch.delenv('TRADER_YF_WINDOW_SLICE', raising=False)
        monkeypatch.setattr(data_sources.time, 'sleep', lambda s: None)
        dates = pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC')
        alpaca_df = pd.DataFrame({'Open': 100.0, 'High': 101.0, 'Low': 99.0,
                                  'Close': 100.5, 'Volume': 1000.0},
                                 index=dates)
        yf_dates = pd.date_range('2024-01-01', periods=8, freq='h', tz='UTC')
        yf_df = pd.DataFrame({'Open': 200.0, 'High': 201.0, 'Low': 199.0,
                              'Close': 200.5, 'Volume': 2000.0},
                             index=yf_dates)
        with mock.patch('market_data.fetch_historical_bars',
                        return_value=alpaca_df), \
             mock.patch.dict('sys.modules', {'yfinance': _yf_module(yf_df)}), \
             mock.patch('market_data.flatten_yfinance_columns',
                        side_effect=lambda d: d), \
             mock.patch('data_sources.fetch_cryptocompare_hourly',
                        return_value=None):
            df = data_sources.fetch_with_fallback(
                'BTC-USD', '2024-01-01', api=mock.MagicMock(),
                asset_type='crypto')
        assert df is not None and len(df) == 8
        assert 'Src' in df.columns
        assert (df['Src'].iloc[:5] == 'alpaca').all()   # collisions -> alpaca
        assert (df['Src'].iloc[5:] == 'yfinance').all()
        assert (df['Close'].iloc[:5] == 100.5).all()

    def test_yf_window_slice_flag(self, monkeypatch):
        monkeypatch.setattr(data_sources.time, 'sleep', lambda s: None)
        yf_dates = pd.date_range('2023-12-30', periods=96, freq='h', tz='UTC')
        yf_df = pd.DataFrame({'Open': 200.0, 'High': 201.0, 'Low': 199.0,
                              'Close': 200.5, 'Volume': 2000.0},
                             index=yf_dates)
        start = '2024-01-01'
        n_after = int((yf_dates >= pd.Timestamp(start, tz='UTC')).sum())
        assert 0 < n_after < 96

        def run():
            with mock.patch.dict('sys.modules',
                                 {'yfinance': _yf_module(yf_df.copy())}), \
                 mock.patch('market_data.flatten_yfinance_columns',
                            side_effect=lambda d: d), \
                 mock.patch('data_sources.fetch_cryptocompare_hourly',
                            return_value=None):
                return data_sources.fetch_with_fallback(
                    'BTC-USD', start, api=None, asset_type='crypto')

        monkeypatch.delenv('TRADER_YF_WINDOW_SLICE', raising=False)
        off = run()
        assert off is not None and len(off) == 96   # flag OFF: full range

        monkeypatch.setenv('TRADER_YF_WINDOW_SLICE', '1')
        on = run()
        assert on is not None and len(on) == n_after
        assert on.index.min() >= pd.Timestamp(start, tz='UTC')

    def test_flag_parsing(self, monkeypatch):
        monkeypatch.delenv('TRADER_YF_WINDOW_SLICE', raising=False)
        assert data_sources._yf_window_slice_enabled() is False
        monkeypatch.setenv('TRADER_YF_WINDOW_SLICE', 'yes')
        assert data_sources._yf_window_slice_enabled() is True


# =====================================================================
# harvest_stock_data — guard, Src strip, exit codes, sidecar wiring
# =====================================================================

def _stub_pipeline(monkeypatch):
    """Neutralize the heavy feature steps inside prepare_stock_data."""
    monkeypatch.setattr(h, 'compute_stock_features',
                        lambda ohlcv, spy_close=None, symbol=None:
                        ohlcv.copy())
    import policy_exits
    monkeypatch.setattr(policy_exits, 'compute_tb_labels',
                        lambda df, fbs, at: {})
    sf = types.ModuleType('short_flow')
    sf.svr_features_for_index = lambda t, idx: None
    sf.sync = lambda: None
    monkeypatch.setitem(sys.modules, 'short_flow', sf)


def _big_ohlcv(start='2026-01-05', periods=120, close=50.0, src=None):
    # Volume large enough to clear the $5M/30d tradability floor
    df = _ohlcv(start, periods, close=close, src=src)
    df['Volume'] = 5e6
    return df


class TestPrepareStockData:
    def test_src_stripped_and_totals_accumulated(self, monkeypatch):
        _stub_pipeline(monkeypatch)
        new = _big_ohlcv(src='alpaca')
        new.loc[new.index[-3:], 'Src'] = 'yfinance'
        monkeypatch.setattr(h, 'fetch_with_fallback',
                            lambda *a, **k: new.copy())
        totals = {}
        out = h.prepare_stock_data('TST', None, api=None,
                                   src_totals=totals)
        assert out is not None
        assert 'Src' not in out.columns
        assert totals == {'alpaca': 117, 'yfinance': 3}

    def test_divergence_guard_refuses_merge(self, monkeypatch, capsys):
        _stub_pipeline(monkeypatch)
        existing = _big_ohlcv(periods=120)
        # New frame overlaps the last 48 bars with 5%-shifted closes
        new = existing.iloc[-48:].copy()
        new['Close'] = new['Close'] * 1.05
        extra = _big_ohlcv(start=str(existing.index[-1]
                                     + pd.Timedelta(hours=1)), periods=5)
        new = pd.concat([new, extra])
        monkeypatch.setattr(h, 'fetch_with_fallback',
                            lambda *a, **k: new.copy())
        raw_out = {}
        out = h.prepare_stock_data('TST', None, api=None,
                                   existing_ohlcv=existing,
                                   start_date='2026-01-08',
                                   raw_out=raw_out)
        assert out is not None
        # Refused: raw capture holds ONLY the existing rows
        assert raw_out['TST'].index.max() == existing.index.max()
        assert len(raw_out['TST']) == len(existing)
        assert '[MERGE-GUARD]' in capsys.readouterr().out

    def test_identical_overlap_merges(self, monkeypatch, capsys):
        _stub_pipeline(monkeypatch)
        existing = _big_ohlcv(periods=120)
        new = existing.iloc[-48:].copy()
        extra = _big_ohlcv(start=str(existing.index[-1]
                                     + pd.Timedelta(hours=1)), periods=5)
        new = pd.concat([new, extra])
        monkeypatch.setattr(h, 'fetch_with_fallback',
                            lambda *a, **k: new.copy())
        raw_out = {}
        out = h.prepare_stock_data('TST', None, api=None,
                                   existing_ohlcv=existing,
                                   start_date='2026-01-08',
                                   raw_out=raw_out)
        assert out is not None
        assert raw_out['TST'].index.max() == extra.index.max()
        assert len(raw_out['TST']) == len(existing) + 5
        captured = capsys.readouterr().out
        assert '[MERGE-GUARD]' not in captured
        assert '5 new bars' in captured


def _stub_main(monkeypatch, tickers, prepare):
    monkeypatch.setattr(h, 'STOCK_TICKERS', tickers)
    monkeypatch.setattr(h, '_get_alpaca_api', lambda: None)
    monkeypatch.setattr(h, 'fetch_spy_close', lambda api=None: None)
    monkeypatch.setattr(h, 'prepare_stock_data', prepare)
    monkeypatch.setattr(h, 'validate_training_data', lambda df, at: {})
    sf = types.ModuleType('short_flow')
    sf.sync = lambda: None
    monkeypatch.setitem(sys.modules, 'short_flow', sf)
    sh = types.ModuleType('sentiment_history')
    sh.fetch_stock_sentiment_history = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, 'sentiment_history', sh)


def _feature_df():
    df = _ohlcv('2026-01-05', 6, close=50.0)
    df['_DV30'] = 1e9
    return df


class TestMainStock:
    def test_no_data_exits_1(self, monkeypatch):
        monkeypatch.delenv('TRADER_RAW_SIDECAR', raising=False)
        _stub_main(monkeypatch, ['AAA'], lambda *a, **k: None)
        monkeypatch.setattr(h, 'load_training_data',
                            lambda prefix: pd.DataFrame())
        with pytest.raises(SystemExit) as ei:
            h.main()
        assert ei.value.code == 1

    def test_save_failure_exits_1(self, monkeypatch):
        monkeypatch.delenv('TRADER_RAW_SIDECAR', raising=False)
        _stub_main(monkeypatch, ['AAA'],
                   lambda *a, **k: _feature_df())
        monkeypatch.setattr(h, 'load_training_data',
                            lambda prefix: pd.DataFrame())
        monkeypatch.setattr(h, 'save_training_data',
                            lambda df, prefix: False)
        with pytest.raises(SystemExit) as ei:
            h.main()
        assert ei.value.code == 1

    def test_sidecar_forces_full_refetch_and_saves_raw(self, monkeypatch):
        monkeypatch.setenv('TRADER_RAW_SIDECAR', '1')
        starts = {}
        saved = {}

        def prepare(t, spy_close, api=None, existing_ohlcv=None,
                    start_date=None, src_totals=None, raw_out=None):
            starts[t] = start_date
            assert existing_ohlcv is None   # empty sidecar -> no state
            if raw_out is not None:
                raw_out[t] = _ohlcv('2026-01-05', 4, src='alpaca')
            return _feature_df()

        _stub_main(monkeypatch, ['AAA', 'BBB'], prepare)
        # NON-empty feature store proves sidecar mode ignores it
        monkeypatch.setattr(
            h, 'load_training_data',
            lambda prefix: _ohlcv('2026-01-05', 8, ticker='AAA'))
        monkeypatch.setattr(h, 'load_raw_ohlcv',
                            lambda prefix: pd.DataFrame())
        monkeypatch.setattr(h, 'save_raw_ohlcv',
                            lambda df, prefix: saved.update(
                                {'prefix': prefix, 'df': df}) or True)
        monkeypatch.setattr(h, 'save_training_data',
                            lambda df, prefix: True)
        h.main()
        assert starts == {'AAA': h.ALPACA_START, 'BBB': h.ALPACA_START}
        assert saved['prefix'] == 'stock'
        assert 'Ticker' in saved['df'].columns
        assert set(saved['df']['Ticker']) == {'AAA', 'BBB'}
        assert 'Src' in saved['df'].columns   # provenance kept in sidecar

    def test_flag_off_never_touches_sidecar(self, monkeypatch):
        monkeypatch.delenv('TRADER_RAW_SIDECAR', raising=False)
        _stub_main(monkeypatch, ['AAA'],
                   lambda *a, **k: _feature_df())
        monkeypatch.setattr(h, 'load_training_data',
                            lambda prefix: pd.DataFrame())
        monkeypatch.setattr(h, 'save_training_data', lambda df, prefix: True)

        def boom(*a, **k):
            raise AssertionError('sidecar IO must not run flag-OFF')
        monkeypatch.setattr(h, 'load_raw_ohlcv', boom)
        monkeypatch.setattr(h, 'save_raw_ohlcv', boom)
        h.main()   # no exception = sidecar untouched


# =====================================================================
# harvest_crypto_data — Src-drop + src_totals parity
# =====================================================================

class TestPrepareCryptoData:
    def test_src_stripped_and_totals(self, monkeypatch):
        monkeypatch.setattr(hc, 'compute_features',
                            lambda ohlcv, btc_close=None: ohlcv.copy())
        import policy_exits
        monkeypatch.setattr(policy_exits, 'compute_tb_labels',
                            lambda df, fbs, at: {})
        fa = types.ModuleType('funding_archive')
        fa.funding_features_for_index = lambda s, idx: None
        fa.sync = lambda: None
        monkeypatch.setitem(sys.modules, 'funding_archive', fa)
        oa = types.ModuleType('oi_archive')
        oa.oi_features_for_index = lambda s, idx: None
        oa.ls_features_for_index = lambda s, idx: None
        oa.taker_features_for_index = lambda s, idx: None
        oa.sync = lambda: None
        monkeypatch.setitem(sys.modules, 'oi_archive', oa)

        new = _ohlcv('2026-01-05', 120, close=50.0, src='alpaca')
        new.loc[new.index[-7:], 'Src'] = 'yfinance'
        monkeypatch.setattr(hc, 'fetch_with_fallback',
                            lambda *a, **k: new.copy())
        totals = {}
        raw_out = {}
        out = hc.prepare_data('BTC-USD', api=None, src_totals=totals,
                              raw_out=raw_out)
        assert out is not None
        assert 'Src' not in out.columns
        assert totals == {'alpaca': 113, 'yfinance': 7}
        assert 'Src' in raw_out['BTC-USD'].columns


# =====================================================================
# main() sidecar wiring — NON-empty raw store: incremental start comes
# from the sidecar (max ts - 48h) + bounded interior-gap repair (F1)
# =====================================================================

class TestSidecarGapRepair:
    def test_stock_main_incremental_start_and_gap_repair(self, monkeypatch,
                                                         capsys):
        monkeypatch.setenv('TRADER_RAW_SIDECAR', '1')
        # Mon 2026-01-05 RTH bars, then Mon 2026-01-12 — the week between
        # is a >=2-busday interior hole (a weekend alone would not flag).
        raw = pd.concat([
            _ohlcv('2026-01-05 14:00', 7, ticker='AAA', src='alpaca'),
            _ohlcv('2026-01-12 14:00', 7, ticker='AAA', src='alpaca'),
        ])
        seen = {}
        fetches = []

        def prepare(t, spy_close, api=None, existing_ohlcv=None,
                    start_date=None, src_totals=None, raw_out=None):
            seen['existing'] = existing_ohlcv
            seen['start'] = start_date
            if raw_out is not None:
                raw_out[t] = existing_ohlcv
            return _feature_df()

        patch = _ohlcv('2026-01-07 14:00', 3)   # inside the hole, no Src

        def fake_hist(api, symbol, start_date, asset_type='crypto',
                      chunk_months=6, end_date=None):
            fetches.append((symbol, start_date, asset_type, end_date))
            return patch.copy()

        _stub_main(monkeypatch, ['AAA'], prepare)
        monkeypatch.setattr(h, '_get_alpaca_api', lambda: object())
        monkeypatch.setattr(h, 'fetch_historical_bars', fake_hist)
        monkeypatch.setattr(h, 'load_raw_ohlcv', lambda prefix: raw.copy())
        saved = {}
        monkeypatch.setattr(
            h, 'save_raw_ohlcv',
            lambda df, prefix: saved.update({'df': df}) or True)
        monkeypatch.setattr(h, 'save_training_data', lambda df, prefix: True)
        h.main()

        # Incremental start = sidecar max ts (Jan 12 20:00) minus 48h
        assert seen['start'] == '2026-01-10'
        # Gap repair: ONE bounded refetch covering exactly the hole
        assert fetches == [('AAA', '2026-01-05', 'stock', '2026-01-13')]
        # existing_ohlcv = 14 sidecar bars + 3 patched bars, Src kept
        assert len(seen['existing']) == 17
        assert 'Src' in seen['existing'].columns
        assert (pd.Timestamp('2026-01-07 15:00', tz='UTC')
                in seen['existing'].index)
        assert '[GAP-REPAIR] AAA' in capsys.readouterr().out
        # The repaired rows reach the persisted sidecar
        assert len(saved['df']) == 17

    def test_crypto_main_symbol_conversion_and_bounds(self, monkeypatch):
        monkeypatch.setenv('TRADER_RAW_SIDECAR', '1')
        raw = pd.concat([
            _ohlcv('2026-01-05 00:00', 10, ticker='BTC-USD', src='alpaca'),
            _ohlcv('2026-01-05 16:00', 10, ticker='BTC-USD', src='alpaca'),
        ])  # 09:00 -> 16:00 = 7h interior hole (> 3h crypto threshold)
        seen = {}
        fetches = []

        def prepare(t, btc_close=None, api=None, existing_ohlcv=None,
                    start_date=None, src_totals=None, raw_out=None):
            seen['existing'] = existing_ohlcv
            seen['start'] = start_date
            return _feature_df()

        def fake_hist(api, symbol, start_date, asset_type='crypto',
                      chunk_months=6, end_date=None):
            fetches.append((symbol, start_date, asset_type, end_date))
            return None   # nothing appended — the bounds are the assertion

        monkeypatch.setattr(hc, 'CRYPTO_TICKERS', ['BTC-USD'])
        monkeypatch.setattr(hc, '_get_alpaca_api', lambda: object())
        monkeypatch.setattr(hc, 'fetch_btc_close', lambda api=None: None)
        monkeypatch.setattr(hc, 'prepare_data', prepare)
        monkeypatch.setattr(hc, 'fetch_historical_bars', fake_hist)
        monkeypatch.setattr(hc, 'load_raw_ohlcv', lambda prefix: raw.copy())
        monkeypatch.setattr(hc, 'save_raw_ohlcv', lambda df, prefix: True)
        monkeypatch.setattr(hc, 'save_training_data', lambda df, prefix: True)
        monkeypatch.setattr(hc, 'validate_training_data', lambda df, at: {})
        for name in ('funding_archive', 'oi_archive'):
            m = types.ModuleType(name)
            m.sync = lambda: None
            monkeypatch.setitem(sys.modules, name, m)
        sh = types.ModuleType('sentiment_history')
        sh.fetch_crypto_sentiment_history = lambda *a, **k: {}
        monkeypatch.setitem(sys.modules, 'sentiment_history', sh)
        hc.main()

        # Alpaca symbol conversion + tight window around the hole
        assert fetches == [('BTC/USD', '2026-01-05', 'crypto', '2026-01-06')]
        # Incremental start = sidecar max (Jan 6 01:00) minus 48h
        assert seen['start'] == '2026-01-04'
        assert len(seen['existing']) == 20
