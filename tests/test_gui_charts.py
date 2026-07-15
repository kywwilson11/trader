"""Chart-overhaul source contracts: pure source inspection, PySide6-free.

gui.py cannot be imported on the dev Mac (no PySide6/pyqtgraph), so these
tests parse the source text/AST and assert the shape of the Markets- and
Performance-tab chart overhaul (chart_core-backed):

- module wiring: antialias on, chart_core imported, theme->color derivation
  only through _chart_palette() (single choke point)
- price area: candles + x-linked volume pane sharing ONE time axis — no
  dual y-axis anywhere in gui.py
- CVD-safe direction encoding (palette-validator mandated): up-candles are
  HOLLOW (surface fill), down-candles FILLED — direction is never encoded
  by green/red hue alone; volume is a single recessive hue (magnitude job,
  no green/red double-encoding); P&L bar sign is carried by position vs the
  zero baseline (safe)
- trade markers: shape-coded (t1 = entry, t = exit), surface-colored ring,
  re-themed in _restyle
- robustness: mode-'none' PriceViews (error/empty/no-data-in-window) have
  an empty markers dict — the scatter update must tolerate missing keys;
  _restyle owns the entire widget restyle (regression guard: the staleness
  timer slot must not contain restyle code or reference the local `t`
  theme alias) and clears the fingerprint memo so a theme flip repaints
  data-colored items (candles, P&L bars)
- performance tab: HWM curve + FillBetweenItem drawdown wash + legend;
  per-bar P&L widths from chart_core (no constant ts[1]-ts[0] width)
- fetcher: chart payloads carry full OHLCV; journal IO (trade markers)
  happens only in the DataFetcher thread, never in a paint/update path
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GUI_PATH = REPO / "gui.py"
SRC = GUI_PATH.read_text()
TREE = ast.parse(SRC)
SRC_LINES = SRC.splitlines()


def _node_source(node):
    return "\n".join(SRC_LINES[node.lineno - 1:node.end_lineno])


def _method_source(name):
    """Source text of a function/method by name, wherever it's nested."""
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return _node_source(node)
    raise AssertionError(f"method {name!r} not found in gui.py")


def _class_source(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return _node_source(node)
    raise AssertionError(f"class {name!r} not found in gui.py")


# ---------------------------------------------------------------------------
# Module wiring
# ---------------------------------------------------------------------------

class TestModuleWiring:
    def test_antialias_configured(self):
        assert re.search(r"pg\.setConfigOptions\(\s*antialias\s*=\s*True", SRC), \
            "pg.setConfigOptions(antialias=True) missing"

    def test_chart_core_imported(self):
        assert any(isinstance(n, ast.Import)
                   and any(a.name == "chart_core" for a in n.names)
                   for n in TREE.body), "module-level 'import chart_core' missing"

    def test_palette_single_choke_point(self):
        # theme -> chart colors is derived in exactly one place
        assert SRC.count("derive_chart_palette(") == 1
        assert "derive_chart_palette(" in _method_source("_chart_palette")


# ---------------------------------------------------------------------------
# Price area: candles + volume pane, one time axis
# ---------------------------------------------------------------------------

class TestPriceArea:
    def test_volume_pane_xlinked(self):
        body = _method_source("_build_stocks_tab")
        assert "setXLink" in body, "volume pane must be x-linked to price pane"

    def test_no_dual_axis_anywhere(self):
        assert "showAxis('right'" not in SRC
        assert 'showAxis("right"' not in SRC

    def test_price_pane_hides_bottom_tick_values(self):
        # one visible time axis (the volume pane's) for the linked stack
        body = _method_source("_build_stocks_tab")
        assert "setStyle(showValues=False)" in body


# ---------------------------------------------------------------------------
# CVD-safe direction encoding (validator-mandated)
# ---------------------------------------------------------------------------

class TestDirectionEncoding:
    def test_candlestick_supports_hollow_up(self):
        body = _class_source("CandlestickItem")
        assert "bg_color" in body, \
            "CandlestickItem.set_data must accept a surface fill for up-candles"

    def test_apply_chart_zoom_passes_surface_fill(self):
        body = _method_source("_apply_chart_zoom")
        assert re.search(r"bg_color\s*=\s*pal\['bg'\]", body), \
            "candles branch must pass bg_color=pal['bg'] (hollow up-candles)"

    def test_volume_is_single_recessive_hue(self):
        body = _method_source("_apply_chart_zoom")
        assert "vol_down" not in body, \
            "volume bars must not repeat the up/down direction encoding"
        assert "vol_up" not in body

    def test_pnl_bar_sign_by_position(self):
        # sign encoded by position vs zero baseline (two BarGraphItems)
        body = _method_source("_apply_perf_data")
        assert "_pnl_bars_pos" in body and "_pnl_bars_neg" in body


# ---------------------------------------------------------------------------
# Trade markers
# ---------------------------------------------------------------------------

class TestTradeMarkers:
    def test_shape_coded_markers(self):
        body = _method_source("_build_stocks_tab")
        assert "symbol='t1'" in body and "symbol='t'" in body
        assert body.count("size=11") >= 2

    def test_surface_ring_not_hardcoded_black(self):
        body = _method_source("_build_stocks_tab")
        assert "'#000000'" not in body and '"#000000"' not in body
        assert re.search(r"mkPen\(chart_pal\['bg'\],\s*width=2\)", body), \
            "marker ring must be the surface color at width 2"

    def test_marker_ring_rethemed(self):
        body = _method_source("_restyle")
        assert "_entry_scatter.setPen" in body and "_exit_scatter.setPen" in body


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_markers_tolerate_mode_none_views(self):
        # error/empty PriceViews carry markers={} — KeyError guard required
        body = _method_source("_apply_chart_zoom")
        assert "markers['entry_t']" not in body
        assert ".get('entry_t'" in body and ".get('exit_t'" in body

    def test_restyle_owns_the_whole_widget_restyle(self):
        # regression: an earlier merge left the tail of _restyle (zoom
        # buttons .. clock) inside _refresh_chart_staleness, where the
        # theme alias `t` is undefined -> NameError every 30s
        restyle = _method_source("_restyle")
        stale = _method_source("_refresh_chart_staleness")
        for needle in ("Zoom buttons", "_clock_label_right", "_log_display"):
            assert needle in restyle, f"_restyle lost its tail ({needle})"
            assert needle not in stale, f"{needle} leaked into the staleness slot"
        assert "t['" not in stale, "staleness slot references undefined theme alias t"

    def test_theme_flip_repaints_data_colored_items(self):
        # fingerprint memo must be cleared before cached data is re-applied
        body = _method_source("_restyle")
        clear_pos = body.find("_chart_fp.clear()")
        assert clear_pos != -1, "_restyle must clear the chart fingerprint memo"
        reapply_pos = body.find("_apply_perf_data")
        assert reapply_pos != -1 and clear_pos < reapply_pos, \
            "memo must be cleared BEFORE cached data is re-applied"

    def test_every_chart_state_titled(self):
        # empty/error/stale states surface via _set_chart_status, not blanks
        assert "_set_chart_status" in _method_source("_apply_chart_zoom")
        assert "_set_chart_status" in _method_source("_apply_perf_data")
        assert "build_price_view" in _method_source("_apply_chart_zoom")
        assert "build_equity_view" in _method_source("_apply_perf_data")


# ---------------------------------------------------------------------------
# Performance tab
# ---------------------------------------------------------------------------

class TestPerformanceTab:
    def test_drawdown_wash_and_legend(self):
        body = _method_source("_build_performance_tab")
        assert "FillBetweenItem" in body, "drawdown fill between HWM and equity"
        assert "addLegend" in body, "2 series on one plot requires a legend"
        assert "_equity_hwm" in body

    def test_hwm_is_dashed_level(self):
        body = _method_source("_restyle")
        assert re.search(r"_equity_hwm\.setPen\(.*DashLine", body), \
            "HWM must be a dashed reference level"

    def test_pnl_bars_use_per_bar_widths(self):
        body = _method_source("_apply_perf_data")
        assert "pnl_widths" in body
        assert "ts_arr[1] - ts_arr[0]" not in body, \
            "constant first-interval bar width (weekend-gap overlap bug)"

    def test_pnl_plot_xlinked_to_equity(self):
        body = _method_source("_build_performance_tab")
        assert "setXLink" in body


# ---------------------------------------------------------------------------
# Crosshair
# ---------------------------------------------------------------------------

class TestCrosshair:
    def test_rate_limited_signal_proxy(self):
        body = _class_source("ChartCrosshair")
        assert "SignalProxy" in body
        assert "sigMouseMoved" in body
        assert "rateLimit" in body

    def test_snaps_to_real_bars(self):
        body = _class_source("ChartCrosshair")
        assert "nearest_index" in body, "crosshair must snap to actual data"


# ---------------------------------------------------------------------------
# DataFetcher: read-only widening, IO stays off the paint path
# ---------------------------------------------------------------------------

class TestFetcher:
    def test_fetch_chart_emits_full_ohlcv(self):
        body = _method_source("fetch_chart")
        for key in ("'opens'", "'highs'", "'lows'", "'volumes'"):
            assert key in body, f"fetch_chart payload missing {key}"

    def test_journal_io_only_in_fetcher_thread(self):
        assert "load_trade_markers" in _method_source("fetch_chart")
        for painter in ("_apply_chart_zoom", "_apply_perf_data", "_restyle",
                        "_refresh_chart_staleness"):
            assert "load_trade_markers" not in _method_source(painter), \
                f"journal IO leaked into paint path {painter}"
