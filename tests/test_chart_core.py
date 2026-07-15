"""Synthetic-data unit tests for chart_core.py — pure numpy/stdlib, runs on
the dev Mac (no PySide6/pyqtgraph/pandas anywhere in this module)."""

import datetime
import json

import numpy as np
import pytest

import chart_core as cc


# 1) lttb_indices
class TestLTTB:
    def test_passthrough_when_len_le_n_out(self):
        idx = cc.lttb_indices(np.arange(5), np.arange(5), 10)
        assert list(idx) == [0, 1, 2, 3, 4]
    def test_endpoints_always_present(self):
        x = np.arange(1000, dtype=float)
        y = np.sin(x / 10)
        idx = cc.lttb_indices(x, y, 100)
        assert idx[0] == 0
        assert idx[-1] == 999
    def test_result_length_matches_n_out(self):
        idx = cc.lttb_indices(np.arange(1000), np.zeros(1000), 250)
        assert len(idx) == 250
    def test_indices_strictly_increasing(self):
        idx = cc.lttb_indices(np.arange(2000), np.random.RandomState(1).randn(2000), 300)
        assert np.all(np.diff(idx) > 0)
    def test_spike_survives_downsample(self):
        n = 10000
        x = np.arange(n, dtype=float)
        y = np.zeros(n)
        spike_at = 6234
        y[spike_at] = 1e6
        idx = cc.lttb_indices(x, y, 500)
        assert spike_at in idx
# 2) coerce_xy
class TestCoerceXY:
    def test_nan_rows_dropped_jointly(self):
        t, ys, note = cc.coerce_xy([1, 2, 3, 4], [10, np.nan, 30, 40], [1, 2, np.nan, 4])
        assert list(t) == [1, 4]
        assert list(ys[0]) == [10, 40]
        assert list(ys[1]) == [1, 4]
    def test_length_mismatch_truncates_and_notes(self):
        t, ys, note = cc.coerce_xy([1, 2, 3], [1, 2])
        assert len(t) == 2
        assert 'truncated to 2' in note
    def test_unsorted_timestamps_sorted_with_ys_permuted(self):
        t, ys, note = cc.coerce_xy([3, 1, 2], [30, 10, 20])
        assert list(t) == [1, 2, 3]
        assert list(ys[0]) == [10, 20, 30]
    def test_non_numeric_input_returns_empty(self):
        t, ys, note = cc.coerce_xy(['a', 'b'], [1, 2])
        assert len(t) == 0
        assert ys == []
        assert 'non-numeric' in note
# 3) perf_stats / compute_hwm
class TestPerfStats:
    def test_hand_built_equity_series(self):
        equity = [100, 110, 105, 120, 90]
        hwm = cc.compute_hwm(equity)
        assert list(hwm) == [100, 110, 110, 120, 120]
        stats = cc.perf_stats(equity, [5, -3, 2])
        assert stats['max_dd_pct'] == pytest.approx(25.0)
        assert stats['total_return'] == pytest.approx(4.0)
        assert stats['best_day'] == pytest.approx(5.0)
        assert stats['worst_day'] == pytest.approx(-3.0)
    def test_empty_pnl_gives_zero_stats(self):
        stats = cc.perf_stats([100, 110], [])
        assert stats['total_return'] == 0.0
        assert stats['best_day'] == 0.0
        assert stats['worst_day'] == 0.0
# 4) bar_widths
class TestBarWidths:
    def test_irregular_gaps_no_overlap(self):
        t = [0, 1, 10]
        w = cc.bar_widths(t)
        assert w[0] <= 0.8 * 1
        assert w[1] <= 0.8 * min(1, 9)
        assert w[2] <= 0.8 * 9
    def test_single_element_default_width(self):
        w = cc.bar_widths([5])
        assert w[0] == pytest.approx(0.8 * 86400.0)
    def test_empty_array(self):
        w = cc.bar_widths([])
        assert len(w) == 0
# 5) ohlc_aggregate
class TestOHLCAggregate:
    def test_factor_3_on_10_bars(self):
        t = list(range(10))
        o = list(range(10))
        h = [x + 1 for x in range(10)]
        l = [x - 1 for x in range(10)]
        c = [x + 0.5 for x in range(10)]
        v = [1] * 10
        t2, o2, h2, l2, c2, v2 = cc.ohlc_aggregate(t, o, h, l, c, v, 3)
        assert len(t2) == 4  # 3 full buckets + 1 partial
        assert list(t2) == [0, 3, 6, 9]
        assert list(o2) == [0, 3, 6, 9]
        assert list(h2) == [3, 6, 9, 10]
        assert list(l2) == [-1, 2, 5, 8]
        assert list(v2) == [3, 3, 3, 1]
    def test_factor_1_is_passthrough(self):
        t2, o2, h2, l2, c2, v2 = cc.ohlc_aggregate([1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], 1)
        assert list(t2) == [1, 2]
# 6) build_equity_view
class TestBuildEquityView:
    def test_empty_history_gives_empty_status(self):
        view = cc.build_equity_view({})
        assert view.status.status == cc.EMPTY
        assert view.status.message
    def test_equal_length_payload_ok_and_hwm_monotone(self):
        hist = {'equity': [100, 105, 95, 120], 'timestamp': [1, 2, 3, 4],
                'profit_loss': [5, -10, 25, 5]}
        view = cc.build_equity_view(hist, now=10)
        assert view.status.status == cc.OK
        assert np.all(view.hwm >= view.equity - 1e-9)
    def test_pnl_shorter_than_timestamp_is_partial_not_dropped(self):
        hist = {'equity': [100, 105, 95, 120, 130], 'timestamp': [1, 2, 3, 4, 5],
                'profit_loss': [5, -10]}
        view = cc.build_equity_view(hist, now=10)
        assert view.status.status == cc.PARTIAL
        assert 'truncated' in view.status.note
        assert len(view.pnl) == 2  # old GUI would have dropped all bars
    def test_20000_point_series_downsampled(self):
        n = 20000
        rng = np.random.RandomState(0)
        hist = {'equity': list(np.cumsum(rng.randn(n)) + 1000),
                'timestamp': list(range(n)),
                'profit_loss': list(rng.randn(n))}
        view = cc.build_equity_view(hist, now=n)
        assert len(view.ts) <= 1500
        assert len(view.hwm) == len(view.ts)
    def test_fingerprint_stable_and_changes(self):
        hist = {'equity': [1, 2, 3], 'timestamp': [1, 2, 3], 'profit_loss': [1, 1, 1]}
        v1 = cc.build_equity_view(hist, now=10)
        v2 = cc.build_equity_view(hist, now=10)
        assert v1.fingerprint == v2.fingerprint
        hist2 = dict(hist)
        hist2['equity'] = [1, 2, 4]
        v3 = cc.build_equity_view(hist2, now=10)
        assert v1.fingerprint != v3.fingerprint
# 7) build_price_view
class TestBuildPriceView:
    def test_error_payload(self):
        view = cc.build_price_view({'error': 'boom'}, '1M', now=1000)
        assert view.status.status == cc.ERROR
        assert view.mode == 'none'
    def test_closes_only_backward_compat_line_mode(self):
        now = 100000.0
        n = 50
        ts = [now - 86400 * (n - i) for i in range(n)]
        data = {'closes': list(range(n)), 'timestamps': ts}
        view = cc.build_price_view(data, '1M', now=now)
        assert view.mode == 'line'
        assert view.status.status == cc.OK
    def test_full_ohlcv_aggregates_to_candles(self):
        now = 100000.0
        n = 2000
        ts = [now - 3600 * (n - i) for i in range(n)]
        opens = list(np.linspace(100, 200, n))
        highs = [x + 1 for x in opens]
        lows = [x - 1 for x in opens]
        closes = [x + 0.5 for x in opens]
        vols = [10] * n
        data = {'closes': closes, 'timestamps': ts, 'opens': opens,
                'highs': highs, 'lows': lows, 'volumes': vols}
        view = cc.build_price_view(data, '1Y', now=now)
        assert view.mode == 'candles'
        assert len(view.t) <= 300
        assert 'aggregated' in view.status.note
        assert np.array_equal(view.up, view.c >= view.o)
    def test_all_data_older_than_window_is_empty(self):
        now = 1_000_000.0
        old_ts = [now - 10 * 86400] * 5
        data = {'closes': [1, 2, 3, 4, 5], 'timestamps': old_ts}
        view = cc.build_price_view(data, '1D', now=now)
        assert view.status.status == cc.EMPTY
        assert view.status.message == 'no data in window'
    def test_markers_filtered_to_window(self):
        now = 1_000_000.0
        n = 40
        ts = [now - 86400 * (n - i) for i in range(n)]
        data = {
            'closes': list(range(n)), 'timestamps': ts,
            'markers': {
                'entry_t': [now - 100 * 86400, now - 5 * 86400],
                'entry_p': [1.0, 2.0],
                'exit_t': [now - 200 * 86400],
                'exit_p': [3.0],
                'exit_pnl': [1.5],
            },
        }
        view = cc.build_price_view(data, '1M', now=now)
        assert len(view.markers['entry_t']) == 1
        assert len(view.markers['exit_t']) == 0
# 8) palette
THEME_DARK = {
    'green': '#4caf50', 'red': '#f44336', 'yellow': '#ffd700', 'white': '#e0e0e0',
    'muted': '#888888', 'bg_dark': '#0a0a0a', 'bg_card': '#1a1a1a', 'bg_table': '#141414',
    'accent': '#ffd700', 'bg_header': '#222222', 'bg_border': '#3a3a3a',
    'bg_hover': '#2a2a2a', 'bg_log': '#050505',
}
THEME_LIGHT = dict(THEME_DARK, bg_dark='#f5f5f5', white='#111111')
THEME_PATHOLOGICAL = dict(THEME_DARK, green='#404040', red='#3f3f41', bg_dark='#3a3a3a')


class TestPalette:
    @pytest.mark.parametrize('theme', [THEME_DARK, THEME_LIGHT, THEME_PATHOLOGICAL])
    def test_all_keys_present_and_valid_hex(self, theme):
        pal = cc.derive_chart_palette(theme)
        expected = {'bg', 'fg', 'grid', 'up', 'down', 'equity', 'hwm', 'dd_fill',
                    'vol_up', 'vol_down', 'crosshair', 'marker_entry', 'marker_exit',
                    'title_warn', 'title_err', 'tile_neutral'}
        assert expected <= set(pal.keys())
        import re
        rx = re.compile(r'^#[0-9a-f]{6}([0-9a-f]{2})?$')
        for k, v in pal.items():
            assert rx.match(v), f'{k}={v} not valid hex'
    @pytest.mark.parametrize('theme', [THEME_DARK, THEME_LIGHT, THEME_PATHOLOGICAL])
    def test_contrast_guarantees_hold(self, theme):
        pal = cc.derive_chart_palette(theme)
        assert cc.contrast_ratio(pal['up'], pal['bg']) >= 2.5
        assert cc.contrast_ratio(pal['down'], pal['bg']) >= 2.5
        assert cc.contrast_ratio(pal['equity'], pal['bg']) >= 2.5
        assert pal['up'] != pal['down']
# 9) heatmap_style
class TestHeatmapStyle:
    def test_contrast_and_neutral(self):
        pal = cc.derive_chart_palette(THEME_DARK)
        for chg in [-10, -5, -1, 0, 1, 5, 10]:
            bg, text = cc.heatmap_style(chg, pal)
            assert cc.contrast_ratio(text, bg) >= 3.0
        bg0, _ = cc.heatmap_style(0, pal)
        assert bg0 == pal['tile_neutral']
        bg_pos, _ = cc.heatmap_style(3, pal)
        bg_neg, _ = cc.heatmap_style(-3, pal)
        assert bg_pos != bg_neg
# 10) load_trade_markers
class TestLoadTradeMarkers:
    def _write_day(self, tmp_path, day, rows):
        fp = tmp_path / f"{day.isoformat()}.jsonl"
        with open(fp, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
            f.write('{not valid json\n')  # corrupt trailing line
            f.write('\n')  # blank line
        return fp
    def test_corrupt_and_missing_symbol_rows_excluded(self, tmp_path):
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        now_dt = datetime.datetime.now().astimezone()
        yest_dt = now_dt - datetime.timedelta(days=1)
        rows_today = [
            {'symbol': 'BTC/USD', 'action': 'buy', 'fill_price': 100.0, 'ts': now_dt.isoformat()},
            {'symbol': 'BTC/USD', 'action': 'sell', 'fill_price': 110.0, 'pnl_pct': 10.0,
             'ts': now_dt.isoformat()},
            {'symbol': 'ETH/USD', 'action': 'buy', 'fill_price': 50.0, 'ts': now_dt.isoformat()},
            {'symbol': 'BTC/USD', 'action': 'skip', 'ts': now_dt.isoformat()},
        ]
        rows_yest = [
            {'symbol': 'BTC/USD', 'action': 'buy', 'fill_price': 90.0, 'ts': yest_dt.isoformat()},
        ]
        self._write_day(tmp_path, today, rows_today)
        self._write_day(tmp_path, yesterday, rows_yest)
        since_ts = (now_dt - datetime.timedelta(days=5)).timestamp()
        markers = cc.load_trade_markers('BTC/USD', tmp_path, since_ts)
        assert len(markers['entry_t']) == 2  # today + yesterday buys, not ETH/skip
        assert len(markers['exit_t']) == 1
        assert markers['exit_pnl'][0] == pytest.approx(10.0)
    def test_since_ts_filter_excludes_older_row(self, tmp_path):
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        now_dt = datetime.datetime.now().astimezone()
        yest_dt = now_dt - datetime.timedelta(days=1)
        self._write_day(tmp_path, today,
                         [{'symbol': 'BTC/USD', 'action': 'buy', 'fill_price': 100.0,
                           'ts': now_dt.isoformat()}])
        self._write_day(tmp_path, yesterday,
                         [{'symbol': 'BTC/USD', 'action': 'buy', 'fill_price': 90.0,
                           'ts': yest_dt.isoformat()}])
        since_today = now_dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        markers = cc.load_trade_markers('BTC/USD', tmp_path, since_today)
        assert len(markers['entry_t']) == 1
    def test_missing_directory_returns_empty_without_raising(self, tmp_path):
        markers = cc.load_trade_markers('BTC/USD', tmp_path / 'does_not_exist', 0.0)
        assert len(markers['entry_t']) == 0
        assert len(markers['exit_t']) == 0
    def test_repeat_call_hits_mtime_cache(self, tmp_path):
        today = datetime.date.today()
        now_dt = datetime.datetime.now().astimezone()
        self._write_day(tmp_path, today,
                         [{'symbol': 'BTC/USD', 'action': 'buy', 'fill_price': 100.0,
                           'ts': now_dt.isoformat()}])
        since_ts = (now_dt - datetime.timedelta(days=1)).timestamp()
        m1 = cc.load_trade_markers('BTC/USD', tmp_path, since_ts)
        m2 = cc.load_trade_markers('BTC/USD', tmp_path, since_ts)
        assert np.array_equal(m1['entry_t'], m2['entry_t'])
        assert np.array_equal(m1['entry_p'], m2['entry_p'])
# 11) ChartStatus / format_age
class TestChartStatus:
    def test_ok_no_note_empty_suffix(self):
        s = cc.ChartStatus(status=cc.OK)
        assert s.title_suffix(100) == ''
    def test_empty_with_message_and_age(self):
        s = cc.ChartStatus(status=cc.EMPTY, message='no history from Alpaca', updated_at=90)
        suffix = s.title_suffix(now=95)
        assert 'no history from Alpaca' in suffix
        assert 'ago' in suffix
    def test_format_age_units(self):
        assert cc.format_age(5) == '5s'
        assert cc.format_age(300) == '5m'
        assert cc.format_age(7200) == '2h'
        assert cc.format_age(200000) == '2d'