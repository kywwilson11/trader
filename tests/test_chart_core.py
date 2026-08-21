"""Synthetic-data unit tests for chart_core.py — pure numpy/stdlib, runs on
the dev Mac (no PySide6/pyqtgraph/pandas anywhere in this module)."""

import ast
import datetime
import json
import math
from pathlib import Path

import numpy as np
import pytest

import chart_core as cc


# Real GUI themes parsed straight out of gui.py's source (no PySide6 import),
# same ast technique as tests/test_design_tokens.py. Each QColor(r, g, b)
# literal becomes a '#rrggbb' hex so derive_chart_palette can consume it.
_GUI_PATH = Path(__file__).resolve().parent.parent / "gui.py"


def _parse_gui_themes():
    tree = ast.parse(_GUI_PATH.read_text())
    node = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "THEMES" for t in n.targets):
            node = n.value
            break
    assert node is not None, "THEMES not found in gui.py"
    out = {}
    for name_node, body in zip(node.keys, node.values):
        roles = {}
        for rk, cv in zip(body.keys, body.values):
            r, g, b = (a.value for a in cv.args)
            roles[rk.value] = f"#{r:02x}{g:02x}{b:02x}"
        out[name_node.value] = roles
    return out


GUI_THEMES = _parse_gui_themes()


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


def _orig_perf_stats(equity, pnl):
    """perf_stats exactly as it existed before the sharpe/sortino/volatility/
    win_rate/cagr extension — copied verbatim as a byte-compat pin. See
    TestPerfStatsExtended.test_byte_compat_old_keys_vs_original{,_fuzz}."""
    equity = np.asarray(equity, dtype=float)
    pnl = np.asarray(pnl, dtype=float)
    finite_pnl = pnl[np.isfinite(pnl)]
    total_return = float(np.nansum(pnl)) if len(pnl) else 0.0
    best_day = float(np.nanmax(finite_pnl)) if len(finite_pnl) else 0.0
    worst_day = float(np.nanmin(finite_pnl)) if len(finite_pnl) else 0.0
    max_dd = 0.0
    if len(equity):
        peak = equity[0]
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak else 0.0
            if dd > max_dd:
                max_dd = dd
    return {
        'total_return': total_return,
        'best_day': best_day,
        'worst_day': worst_day,
        'max_dd_pct': float(max_dd),
    }


class TestPerfStatsExtended:
    """sharpe/sortino/volatility/win_rate/cagr, layered on top of the
    original total/best/worst/max_dd_pct covered by TestPerfStats above."""

    def test_new_keys_present_alongside_old(self):
        stats = cc.perf_stats([100, 110, 105], [5, -3])
        assert {'total_return', 'best_day', 'worst_day', 'max_dd_pct',
                'sharpe', 'sortino', 'volatility', 'win_rate', 'cagr'} == set(stats.keys())

    def test_byte_compat_old_keys_vs_original(self):
        equity = [100, 110, 105, 120, 90, 130, 80, 150]
        pnl = [10, -5, 15, -30, 40, -50, 70, -20]
        old = _orig_perf_stats(equity, pnl)
        new_no_t = cc.perf_stats(equity, pnl)
        # t must not perturb the old keys at all, even when supplied.
        new_with_t = cc.perf_stats(equity, pnl, t=list(range(len(equity))))
        for k in old:
            assert new_no_t[k] == pytest.approx(old[k])
            assert new_with_t[k] == pytest.approx(old[k])

    def test_byte_compat_old_keys_vs_original_fuzz(self):
        rng = np.random.RandomState(7)
        for _ in range(20):
            n = rng.randint(0, 30)
            equity = rng.uniform(-50, 500, size=n)
            n_bad = max(n // 6, 0)
            if n_bad:
                bad = rng.choice(n, size=n_bad, replace=False)
                equity[bad] = np.nan
            pnl = rng.uniform(-100, 100, size=max(n - 1, 0))
            old = _orig_perf_stats(equity, pnl)
            new = cc.perf_stats(equity, pnl)
            for k in old:
                assert new[k] == pytest.approx(old[k], nan_ok=True)

    def test_empty_and_single_and_two_point_series_give_none_not_nan(self):
        # 0, 1, and 2 equity points (0 or 1 usable returns) can't support a
        # variance-based stat — None, not a crash and not NaN.
        for equity, pnl in [([], []), ([100], []), ([100], [5]), ([100, 110], [])]:
            stats = cc.perf_stats(equity, pnl)
            for k in ('sharpe', 'sortino', 'volatility', 'cagr'):
                assert stats[k] is None, f'{k} should be None for equity={equity} pnl={pnl}'

    def test_flat_equity_volatility_zero_but_sharpe_and_sortino_none(self):
        # Zero variance is itself a real, computable volatility (0.0); a
        # ratio with zero in the denominator is not (None) — pinned.
        stats = cc.perf_stats([100.0, 100.0, 100.0], [0, 0])
        assert stats['volatility'] == pytest.approx(0.0)
        assert stats['sharpe'] is None
        assert stats['sortino'] is None
        assert stats['win_rate'] is None  # no nonzero pnl entries

    def test_constant_growth_equity_positive_sharpe_near_zero_vol_exact_cagr(self):
        equity = [100.0]
        for _ in range(100):
            equity.append(equity[-1] * 1.001)  # ~0.1%/step; float noise, not exactly 0 variance
        t = [86400.0 * i for i in range(101)]  # daily spacing
        stats = cc.perf_stats(equity, [], t=t)
        assert stats['sharpe'] is not None and math.isfinite(stats['sharpe'])
        assert stats['sharpe'] > 0
        assert stats['volatility'] == pytest.approx(0.0, abs=1e-6)
        expected_cagr = (equity[-1] / equity[0]) ** ((365.25 * 86400) / (t[-1] - t[0])) - 1.0
        assert stats['cagr'] == pytest.approx(expected_cagr, rel=1e-6)

    def test_alternating_gain_loss_exact_win_rate_and_finite_sortino(self):
        pnl = [10, -5, 10, -5, 10, -5]
        equity = [1000]
        for p in pnl:
            equity.append(equity[-1] + p)
        stats = cc.perf_stats(equity, pnl)
        assert stats['win_rate'] == pytest.approx(0.5)
        assert stats['sortino'] is not None and math.isfinite(stats['sortino'])

    def test_all_losses_win_rate_zero_sortino_pinned_negative(self):
        # Definition pin: sortino uses mean(all returns)/downside-deviation,
        # so "all losses" (with >=2 varying downside obs) is a well-defined
        # NEGATIVE finite ratio, not None — None is reserved for the
        # zero-downside-observations case (see test_alternating_* above has
        # some upside, this case has none).
        equity = [1000, 950, 870, 800, 700]
        pnl = [-50, -80, -70, -100]
        stats = cc.perf_stats(equity, pnl)
        assert stats['win_rate'] == pytest.approx(0.0)
        assert stats['sortino'] == pytest.approx(-43.764807580376775, rel=1e-6)
        assert stats['sortino'] < 0

    def test_t_none_fallback_vs_t_provided_annualization_differ(self):
        rng = np.random.RandomState(42)
        equity = (1000 + np.cumsum(rng.randn(50))).tolist()
        t = [3600.0 * i for i in range(50)]  # hourly spacing
        stats_t = cc.perf_stats(equity, [], t=t)
        stats_none = cc.perf_stats(equity, [])
        assert stats_t['volatility'] != pytest.approx(stats_none['volatility'])
        ann_t = math.sqrt(365.25 * 86400.0 / 3600.0)
        ann_none = math.sqrt(252.0)
        # Same underlying returns in both calls -> ratio is exactly the
        # ratio of the two annualization factors.
        assert stats_t['volatility'] == pytest.approx(
            stats_none['volatility'] * ann_t / ann_none, rel=1e-9)
        assert stats_t['sharpe'] == pytest.approx(
            stats_none['sharpe'] * ann_t / ann_none, rel=1e-9)

    def test_cagr_none_without_t_or_with_non_positive_equity_or_elapsed(self):
        equity = [100, 105, 110]
        t = [0, 86400, 172800]
        assert cc.perf_stats(equity, [])['cagr'] is None  # t omitted (positional)
        assert cc.perf_stats(equity, [], t=None)['cagr'] is None  # t explicitly None
        assert cc.perf_stats([-5, 105, 110], [], t=t)['cagr'] is None  # first equity <= 0
        assert cc.perf_stats([100, 105, -1], [], t=t)['cagr'] is None  # last equity <= 0
        assert cc.perf_stats([100, 105, 110], [], t=[5, 5, 5])['cagr'] is None  # elapsed <= 0

    def test_cagr_overflow_from_tiny_elapsed_is_none_not_a_crash(self):
        # Two snapshots a couple seconds apart (e.g. a freshly-started bot)
        # blow the annualization exponent up enough that a naive ** would
        # raise OverflowError instead of just saturating to inf.
        stats = cc.perf_stats([100, 150], [], t=[0, 2])
        assert stats['cagr'] is None

    def test_build_equity_view_wires_timestamps_into_perf_stats(self):
        hist = {'equity': [100, 105, 95, 120], 'timestamp': [0, 86400, 172800, 259200],
                'profit_loss': [5, -10, 25, 5]}
        view = cc.build_equity_view(hist, now=1e12)
        # Only reachable if build_equity_view is passing t= through to perf_stats.
        assert view.stats['cagr'] is not None
        assert view.stats['sharpe'] is not None
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
        # Floor raised 2.5 -> 3.0 (WCAG-AA graphical) with the contrast change.
        pal = cc.derive_chart_palette(theme)
        assert cc.contrast_ratio(pal['up'], pal['bg']) >= 3.0
        assert cc.contrast_ratio(pal['down'], pal['bg']) >= 3.0
        assert cc.contrast_ratio(pal['equity'], pal['bg']) >= 3.0
        assert pal['up'] != pal['down']

    @pytest.mark.parametrize('theme', [THEME_DARK, THEME_LIGHT, THEME_PATHOLOGICAL])
    def test_luminance_gap_always_on_for_cvd(self, theme):
        # Direction must be legible by BRIGHTNESS, not hue alone (color-blind
        # safety) — always-on, not just the rare hue-collapse case.
        pal = cc.derive_chart_palette(theme)
        gap = abs(cc.rel_luminance(pal['up']) - cc.rel_luminance(pal['down']))
        assert gap >= 0.08, f"up/down luminance gap {gap:.3f} < 0.08"


class TestSeparateLuminance:
    """The always-on CVD brightness separator, direct unit contract."""

    def test_opens_gap_while_holding_contrast_on_dark_bg(self):
        # Inputs already >= 3.0 vs bg (as derive_chart_palette always passes)
        # but near-identical luminance -> a gap opens, both stay >= 3.0.
        up, down = cc.separate_luminance('#4c8a4c', '#7a7a4c', '#0a0a0a')
        gap = abs(cc.rel_luminance(up) - cc.rel_luminance(down))
        assert gap >= 0.08
        assert cc.contrast_ratio(up, '#0a0a0a') >= 3.0
        assert cc.contrast_ratio(down, '#0a0a0a') >= 3.0

    def test_already_separated_is_left_alone(self):
        up, down = '#00ff66', '#880000'  # luminance gap already huge
        out_up, out_down = cc.separate_luminance(up, down, '#0a0a0a')
        assert (out_up, out_down) == (up, down)

    def test_light_bg_opens_gap_by_darkening_down(self):
        # On a light bg the gap opens mainly by darkening down (its contrast
        # grows); both endpoints stay >= 3.0.
        up, down = cc.separate_luminance('#2f9a55', '#3a9a6a', '#f5f5f5')
        gap = abs(cc.rel_luminance(up) - cc.rel_luminance(down))
        assert gap >= 0.08
        assert cc.contrast_ratio(up, '#f5f5f5') >= 3.0
        assert cc.contrast_ratio(down, '#f5f5f5') >= 3.0


class TestPaletteRealGuiThemes:
    """Every shipped gui.py theme (parsed from source) must yield legible,
    CVD-safe chart colors: contrast >= 3.0 vs bg AND a >= 0.08 luminance gap
    between up and down. Covers the two Phase-4 additions (Terminal/Paper,
    incl. the light theme) automatically via source parsing."""

    @pytest.mark.parametrize('theme_name', sorted(GUI_THEMES))
    def test_contrast_floor_3_and_cvd_gap(self, theme_name):
        pal = cc.derive_chart_palette(GUI_THEMES[theme_name])
        bg = pal['bg']
        assert cc.contrast_ratio(pal['up'], bg) >= 3.0
        assert cc.contrast_ratio(pal['down'], bg) >= 3.0
        assert cc.contrast_ratio(pal['equity'], bg) >= 3.0
        gap = abs(cc.rel_luminance(pal['up']) - cc.rel_luminance(pal['down']))
        assert gap >= 0.08, f"{theme_name}: up/down luminance gap {gap:.3f} < 0.08"
        assert pal['up'] != pal['down']

    def test_at_least_the_ten_base_themes_present(self):
        assert len(GUI_THEMES) >= 10
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
# 12) Price overlays — trailing_sma / wilder_atr / build_price_view(overlays=...)
def _make_ohlcv(n, now, spacing=3600.0):
    """Same shape as TestBuildPriceView's full-OHLCV fixtures above (strictly
    increasing opens so up/down + h/l are unambiguous)."""
    ts = [now - spacing * (n - i) for i in range(n)]
    opens = list(np.linspace(100, 200, n))
    highs = [x + 1 for x in opens]
    lows = [x - 1 for x in opens]
    closes = [x + 0.5 for x in opens]
    vols = [10] * n
    return {'closes': closes, 'timestamps': ts, 'opens': opens,
            'highs': highs, 'lows': lows, 'volumes': vols}


class TestPriceOverlays:
    # -- trailing_sma: hand-computed exactness + warmup NaNs --
    def test_trailing_sma_hand_computed_with_warmup_nans(self):
        sma = cc.trailing_sma([1, 2, 3, 4, 5, 6], 3)
        assert np.isnan(sma[0]) and np.isnan(sma[1])
        assert list(sma[2:]) == pytest.approx([2, 3, 4, 5])
    def test_trailing_sma_fewer_bars_than_n_is_all_nan(self):
        sma = cc.trailing_sma([1, 2], 3)
        assert len(sma) == 2
        assert np.all(np.isnan(sma))
    def test_trailing_sma_empty_input(self):
        assert len(cc.trailing_sma([], 20)) == 0

    # -- wilder_atr: Wilder recurrence exactness on a hand-built fixture --
    def test_wilder_atr_recurrence_exact_on_small_fixture(self):
        # TR (by hand from H/L/C below) = [1, 2, 2, 2, 3, 2]; with length=3
        # the seed is mean(TR[:3]) and every later value follows Wilder's
        # own recurrence atr[i] = (atr[i-1]*(n-1) + tr[i]) / n — NOT a
        # plain rolling mean of TR (that would give a different i=3.. path).
        high = [10, 11, 12, 11, 13, 14]
        low = [9, 9, 10, 9, 11, 12]
        close = [9.5, 10, 11, 10, 12, 13]
        atr = cc.wilder_atr(high, low, close, length=3)
        assert np.isnan(atr[0]) and np.isnan(atr[1])
        expected = [5 / 3, 16 / 9, 59 / 27, 172 / 81]
        assert list(atr[2:]) == pytest.approx(expected, rel=1e-12)
    def test_wilder_atr_length_1_reduces_to_true_range(self):
        # Degenerate but sharp pin on the recurrence: with length=1 the
        # seed IS tr[0] and every step is (prev*0 + tr[i])/1 == tr[i], so
        # ATR(1) must equal the true-range series exactly.
        high = [10, 11, 12, 11, 13, 14]
        low = [9, 9, 10, 9, 11, 12]
        close = [9.5, 10, 11, 10, 12, 13]
        tr = [1, 2, 2, 2, 3, 2]
        atr = cc.wilder_atr(high, low, close, length=1)
        assert list(atr) == pytest.approx(tr)
    def test_wilder_atr_fewer_bars_than_length_is_all_nan(self):
        atr = cc.wilder_atr([1, 2], [0, 1], [0.5, 1.5], length=14)
        assert len(atr) == 2
        assert np.all(np.isnan(atr))
    def test_wilder_atr_empty_input(self):
        assert len(cc.wilder_atr([], [], [])) == 0

    # -- build_price_view(overlays=...) wiring --
    def test_overlays_default_empty_tuple_gives_empty_dict(self):
        now = 100000.0
        data = _make_ohlcv(50, now)
        view = cc.build_price_view(data, '1M', now=now)
        assert view.overlays == {}
    def test_overlay_length_matches_view_t_aggregated_path(self):
        # Mirrors TestBuildPriceView.test_full_ohlcv_aggregates_to_candles
        # (crypto-1Y, n=2000 > max_candles=300 -> aggregation kicks in).
        now = 100000.0
        data = _make_ohlcv(2000, now)
        view = cc.build_price_view(data, '1Y', now=now,
                                    overlays=('sma20', 'sma50', 'atr_band'))
        assert 'aggregated' in view.status.note
        assert len(view.t) <= 300
        assert view.overlays['sma20'].shape == view.t.shape
        assert view.overlays['sma50'].shape == view.t.shape
        upper, lower = view.overlays['atr_band']
        assert upper.shape == view.t.shape
        assert lower.shape == view.t.shape
    def test_overlay_length_matches_view_t_non_aggregated_path(self):
        # Same OHLCV shape, but n <= max_candles so no aggregation fires —
        # the "non-aggregated path" companion to the test above.
        now = 100000.0
        data = _make_ohlcv(120, now)
        view = cc.build_price_view(data, '1M', now=now,
                                    overlays=('sma20', 'sma50', 'atr_band'))
        assert 'aggregated' not in view.status.note
        assert len(view.t) == 120
        assert view.overlays['sma20'].shape == view.t.shape
        assert view.overlays['sma50'].shape == view.t.shape
        upper, lower = view.overlays['atr_band']
        assert upper.shape == view.t.shape
        assert lower.shape == view.t.shape
    def test_atr_band_matches_direct_wilder_atr_call_default_and_custom_mult(self):
        now = 100000.0
        data = _make_ohlcv(120, now)
        view = cc.build_price_view(data, '1M', now=now, overlays=('atr_band',))
        atr = cc.wilder_atr(view.h, view.l, view.c, 14)
        expected_upper = view.c + 2.0 * atr
        expected_lower = view.c - 2.0 * atr
        got_upper, got_lower = view.overlays['atr_band']
        assert np.allclose(got_upper, expected_upper, equal_nan=True)
        assert np.allclose(got_lower, expected_lower, equal_nan=True)
        view2 = cc.build_price_view(data, '1M', now=now, overlays=('atr_band',), atr_mult=3.5)
        got_upper2, got_lower2 = view2.overlays['atr_band']
        assert np.allclose(got_upper2, view.c + 3.5 * atr, equal_nan=True)
        assert np.allclose(got_lower2, view.c - 3.5 * atr, equal_nan=True)
    def test_sma_overlays_match_direct_trailing_sma_call(self):
        now = 100000.0
        data = _make_ohlcv(120, now)
        view = cc.build_price_view(data, '1M', now=now, overlays=('sma20', 'sma50'))
        assert np.allclose(view.overlays['sma20'], cc.trailing_sma(view.c, 20), equal_nan=True)
        assert np.allclose(view.overlays['sma50'], cc.trailing_sma(view.c, 50), equal_nan=True)
    def test_requesting_only_one_overlay_yields_only_that_key(self):
        now = 100000.0
        data = _make_ohlcv(120, now)
        view = cc.build_price_view(data, '1M', now=now, overlays=('sma20',))
        assert set(view.overlays.keys()) == {'sma20'}
    def test_unknown_overlay_name_ignored_not_a_crash(self):
        now = 100000.0
        data = _make_ohlcv(120, now)
        view = cc.build_price_view(data, '1M', now=now, overlays=('bogus', 'sma20'))
        assert set(view.overlays.keys()) == {'sma20'}
    def test_line_mode_gives_empty_overlays_even_if_requested(self):
        # Closes-only backward-compat path: no h/l to compute atr_band from,
        # and view.t there is pre-downsample (not what's actually drawn —
        # line_t/line_c are) so overlays are intentionally not computed.
        now = 100000.0
        n = 50
        ts = [now - 86400 * (n - i) for i in range(n)]
        data = {'closes': list(range(n)), 'timestamps': ts}
        view = cc.build_price_view(data, '1M', now=now,
                                    overlays=('sma20', 'sma50', 'atr_band'))
        assert view.mode == 'line'
        assert view.overlays == {}
# 13) align_benchmark
class TestAlignBenchmark:
    def test_exact_normalization_on_synthetic_full_overlap(self):
        t_equity = [0, 1, 2, 3, 4]
        t_bench = [0, 1, 2, 3, 4]
        bench_close = [50, 55, 60, 50, 45]
        equity = [1000, 1010, 1005, 1030, 1000]
        result = cc.align_benchmark(t_equity, t_bench, bench_close, equity)
        assert result == pytest.approx([1000, 1100, 1200, 1000, 900])
    def test_non_overlapping_ranges_all_nan_no_crash(self):
        t_equity = [100, 200, 300]
        t_bench = [1, 2, 3]
        bench_close = [10, 20, 30]
        equity = [500, 510, 520]
        result = cc.align_benchmark(t_equity, t_bench, bench_close, equity)
        assert len(result) == 3
        assert np.all(np.isnan(result))
    def test_interp_correctness_at_midpoint(self):
        t_bench = [0, 10]
        bench_close = [100, 200]
        t_equity = [0, 5, 10]
        equity = [1000, 1000, 1000]
        result = cc.align_benchmark(t_equity, t_bench, bench_close, equity)
        # i0=0 -> scale = equity[0] / interp(0) = 1000/100 = 10
        assert result[0] == pytest.approx(1000.0)
        assert result[1] == pytest.approx(1500.0)  # midpoint of 100,200 -> 150 * 10
        assert result[2] == pytest.approx(2000.0)
    def test_partial_overlap_mixes_nan_and_values(self):
        # Equity history runs longer than the cached benchmark window (the
        # realistic GUI case) -> points before benchmark coverage are NaN,
        # the rest are interpolated+scaled.
        t_equity = [0, 1, 2, 3, 4, 5]
        equity = [100, 101, 102, 103, 104, 105]
        t_bench = [2, 3, 4]
        bench_close = [10, 20, 30]
        result = cc.align_benchmark(t_equity, t_bench, bench_close, equity)
        assert np.isnan(result[0]) and np.isnan(result[1])
        assert np.all(np.isfinite(result[2:5]))
        # index 5 (t=5) is past t_bench's last point (4) -> NaN too.
        assert np.isnan(result[5])
        assert result[2] == pytest.approx(102.0)  # anchor point == equity[2]
    def test_nonpositive_or_nonfinite_anchor_price_returns_all_nan(self):
        t_equity = [0, 1, 2]
        equity = [100, 110, 120]
        t_bench = [0, 1, 2]
        for bad_bench in ([0, 10, 20], [-5, 10, 20]):
            result = cc.align_benchmark(t_equity, t_bench, bad_bench, equity)
            assert np.all(np.isnan(result))
    def test_empty_inputs_never_crash(self):
        assert len(cc.align_benchmark([], [1, 2], [10, 20], [])) == 0
        r1 = cc.align_benchmark([1, 2], [], [], [100, 110])
        assert len(r1) == 2 and np.all(np.isnan(r1))
        r2 = cc.align_benchmark([1, 2], [1, 2], [10, 20], [])
        assert len(r2) == 2 and np.all(np.isnan(r2))
    def test_short_equity_array_anchor_out_of_bounds_no_crash(self):
        # Overlap doesn't start until t_equity index 2, but `equity` only
        # has 2 entries -> the anchor index (2) is out of bounds for it.
        # Must degrade to all-NaN, never raise/index-error.
        t_equity = [-100, -50, 1, 2, 3, 4]
        result = cc.align_benchmark(t_equity, [1, 2, 3, 4], [10, 20, 30, 40], [100, 100])
        assert len(result) == len(t_equity)
        assert np.all(np.isnan(result))
    def test_non_numeric_t_equity_no_crash(self):
        result = cc.align_benchmark(['a', 'b'], [1, 2], [10, 20], [100, 110])
        assert len(result) == 2
        assert np.all(np.isnan(result))
# 14) format_si
class TestFormatSi:
    def test_known_cases(self):
        cases = [
            (999, '999'),
            (1000, '1K'),
            (1.25e6, '1.2M'),
            (0, '0'),
            (-500, '-500'),
            (-2500, '-2.5K'),
            (5.6e9, '5.6B'),
        ]
        for value, expected in cases:
            assert cc.format_si(value) == expected, f'format_si({value})'
    def test_non_numeric_and_non_finite_never_crash(self):
        assert cc.format_si('abc') == '0'
        assert cc.format_si(float('nan')) == '0'
        assert cc.format_si(float('inf')) == '0'
        assert cc.format_si(None) == '0'