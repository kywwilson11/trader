"""GUI fixes from the 2026-09 Jetson headless audit (F_gui D1/D2/D3/D4/D5/D10).

gui.py imports PySide6, which is absent on the dev Mac (and in the Jetson
`jetson` env), so every test here is PySide6-free: the pure helpers and the
NumericTableItem class are extracted from the gui.py AST and exec'd against
stubs; layout/chart fixes are pinned as source-text contracts.

- D1  the orders-pagination `until` cursor is RFC3339 UTC ('...T..:..:..Z').
      str(pandas.Timestamp) ('2026-04-07 19:45:17.419822+00:00') was rejected
      by Alpaca, so page 2 always failed and the whole orders stream died.
- D2  NumericTableItem.__lt__ never calls super().__lt__ (PySide6 6.8 re-enters
      the Python override -> RecursionError on every text-column sort).
- D3  the empty ATR FillBetweenItem no longer pins the price y-axis at 0.
- D4/D5  Performance / Markets / Models pages scroll instead of clipping.
- D10 the engine-subprocess LD_LIBRARY_PATH has no empty element (cwd).
"""
import ast
import datetime as dt
import os
import re
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = (REPO / "gui.py").read_text()
TREE = ast.parse(SRC)


def _node(name, kind=(ast.FunctionDef, ast.ClassDef)):
    for node in ast.walk(TREE):
        if isinstance(node, kind) and node.name == name:
            return node
    raise AssertionError(f"{name!r} not found in gui.py")


def _source(name, kind=(ast.FunctionDef, ast.ClassDef)):
    return ast.get_source_segment(SRC, _node(name, kind))


def _load(name, ns):
    exec(compile(_source(name), f"gui.py::{name}", "exec"), ns)
    return ns[name]


# ---------------------------------------------------------------------------
# D1 — RFC3339 `until` cursor
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def alpaca_until():
    return _load("_alpaca_until", {"dt": dt, "re": re})


RFC3339_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class TestAlpacaUntil:
    def test_pandas_timestamp_str_form_rounds_up(self, alpaca_until):
        # the exact value Alpaca rejected in the audit
        out = alpaca_until("2026-04-07 19:45:17.419822+00:00")
        assert out == "2026-04-07T19:45:18Z"
        assert RFC3339_Z.match(out)

    def test_pandas_timestamp_object(self, alpaca_until):
        pd = pytest.importorskip("pandas")
        ts = pd.Timestamp("2026-04-07 19:45:17.419822", tz="UTC")
        assert alpaca_until(ts) == "2026-04-07T19:45:18Z"
        # a pure-nanosecond fraction still counts as sub-second -> round up
        ts_ns = pd.Timestamp("2026-04-07 19:45:17.000000500", tz="UTC")
        assert alpaca_until(ts_ns) == "2026-04-07T19:45:18Z"
        whole = pd.Timestamp("2026-04-07 19:45:17", tz="UTC")
        assert alpaca_until(whole) == "2026-04-07T19:45:17Z"

    def test_whole_second_unchanged(self, alpaca_until):
        d = dt.datetime(2026, 4, 7, 19, 45, 17, tzinfo=dt.timezone.utc)
        assert alpaca_until(d) == "2026-04-07T19:45:17Z"
        assert alpaca_until("2026-04-07T19:45:17Z") == "2026-04-07T19:45:17Z"

    def test_non_utc_offset_converted(self, alpaca_until):
        cst = dt.timezone(dt.timedelta(hours=-5))
        d = dt.datetime(2026, 4, 7, 14, 45, 17, tzinfo=cst)
        assert alpaca_until(d) == "2026-04-07T19:45:17Z"
        assert alpaca_until("2026-04-07T14:45:17-05:00") == "2026-04-07T19:45:17Z"

    def test_naive_taken_as_utc(self, alpaca_until):
        assert alpaca_until(dt.datetime(2026, 4, 7, 19, 45, 17)) == "2026-04-07T19:45:17Z"

    def test_round_up_crosses_midnight(self, alpaca_until):
        d = dt.datetime(2026, 12, 31, 23, 59, 59, 5, tzinfo=dt.timezone.utc)
        assert alpaca_until(d) == "2027-01-01T00:00:00Z"

    @pytest.mark.parametrize("s,want", [
        ("2026-04-07T19:45:17.419822Z", "2026-04-07T19:45:18Z"),
        ("2026-04-07T19:45:17.419822123Z", "2026-04-07T19:45:18Z"),  # ns digits
        ("2026-04-07T19:45:17.4Z", "2026-04-07T19:45:18Z"),          # 1 digit
        ("2026-04-07T19:45:17.000Z", "2026-04-07T19:45:17Z"),
    ])
    def test_string_fractions(self, alpaca_until, s, want):
        assert alpaca_until(s) == want

    @pytest.mark.parametrize("bad", [None, "", "   ", "not a date", "2026-13-45"])
    def test_unparseable_is_none(self, alpaca_until, bad):
        assert alpaca_until(bad) is None

    def test_fetch_orders_uses_helper(self):
        # M3 (2026-09-26) moved the page walk into _walk_orders_pages; the
        # contract applies to the walk, and fetch_orders must use the walk.
        assert "self._walk_orders_pages(" in _source("fetch_orders",
                                                      ast.FunctionDef)
        assert "list_orders(" not in _source("fetch_orders", ast.FunctionDef)
        body = _source("_walk_orders_pages", ast.FunctionDef)
        assert "_alpaca_until(oldest.submitted_at)" in body
        assert "new_until = str(" not in body
        # every until= passed to the API is the cursor built by the helper
        assert re.findall(r"until=(\w+)", body) == ["until"]
        # dedupe by order id absorbs the one-second round-up overlap
        assert "seen_ids" in body


# ---------------------------------------------------------------------------
# D2 — NumericTableItem.__lt__ without super() recursion
# ---------------------------------------------------------------------------
USER_ROLE = 256


class _StubItem:
    """Mimics the PySide6 6.8 failure: the base-class __lt__ dispatches back
    into the most-derived Python override (so super().__lt__ recurses)."""
    def __init__(self, text=""):
        self._text = text
        self._data = {}

    def setFont(self, _f):
        pass

    def setData(self, role, v):
        self._data[role] = v

    def data(self, role):
        return self._data.get(role)

    def text(self):
        return self._text

    def __lt__(self, other):
        return type(self).__lt__(self, other)


def _ns():
    return {"QTableWidgetItem": _StubItem,
            "Qt": types.SimpleNamespace(UserRole=USER_ROLE),
            "QFont": lambda fam: fam,
            "design_tokens": types.SimpleNamespace(NUMERIC_FAMILY="Plex Mono")}


@pytest.fixture(scope="module")
def Item():
    return _load("NumericTableItem", _ns())


def test_stub_reproduces_the_pyside68_recursion():
    """Sanity: the stub is faithful — the pre-fix `return super().__lt__(other)`
    fallback recurses forever against it."""
    old = '''
class OldItem(QTableWidgetItem):
    def __lt__(self, other):
        v1 = self.data(Qt.UserRole)
        v2 = other.data(Qt.UserRole) if other else None
        if v1 is not None and v2 is not None:
            return float(v1) < float(v2)
        return super().__lt__(other)
'''
    ns = _ns()
    exec(old, ns)
    a, b = ns["OldItem"]("BTC/USD"), ns["OldItem"]("AAPL")
    with pytest.raises(RecursionError):
        _ = a < b


class TestNumericTableItemLt:
    def test_text_compare_without_payload(self, Item):
        a, b = Item("BTC/USD"), Item("AAPL")
        assert (b < a) is True
        assert (a < b) is False
        syms = ["XRP/USD", "SOL/USD", "LINK/USD", "ETH/USD", "DOGE/USD", "BTC/USD"]
        assert [i.text() for i in sorted(Item(s) for s in syms)] == sorted(syms)

    def test_numeric_payloads_compare_numerically(self, Item):
        a, b = Item("$9.00"), Item("$10.00")
        a.setData(USER_ROLE, 9.0)
        b.setData(USER_ROLE, 10.0)
        assert a < b and not (b < a)  # text compare would say "$10" < "$9"

    def test_mixed_payload_falls_back_to_text(self, Item):
        a, b = Item("b"), Item("a")
        a.setData(USER_ROLE, 1.0)  # only one side numeric
        assert (b < a) is True and (a < b) is False

    def test_non_numeric_payload_falls_back_to_text(self, Item):
        a, b = Item("a"), Item("b")
        a.setData(USER_ROLE, "x")
        b.setData(USER_ROLE, "y")
        assert a < b

    def test_none_other_and_empty_text(self, Item):
        a = Item("")
        assert (a < None) is False
        assert (Item("") < Item("x")) is True

    def test_source_never_calls_super_lt(self):
        body = ast.get_source_segment(SRC, _node("NumericTableItem", ast.ClassDef))
        lt = body[body.index("def __lt__"):]
        code = "\n".join(l for l in lt.splitlines() if not l.strip().startswith("#"))
        assert "super()" not in code


# ---------------------------------------------------------------------------
# D10 — no empty LD_LIBRARY_PATH element in engine subprocess env
# ---------------------------------------------------------------------------
@pytest.fixture
def engine_env(tmp_path):
    fake_py = tmp_path / "bin" / "python"
    fake_py.parent.mkdir()
    fake_py.write_text("")
    ns = {"os": os, "_JETSON_PY": str(fake_py), "_JETSON_PREFIX": str(tmp_path)}
    return _load("_engine_env", ns), str(tmp_path)


def _elements(v):
    return v.split(":")


def test_engine_env_no_trailing_colon_when_parent_unset(engine_env, monkeypatch):
    fn, pre = engine_env
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    v = fn()["LD_LIBRARY_PATH"]
    assert v == pre + "/lib"
    assert "" not in _elements(v)
    v2 = fn(cusparselt=True)["LD_LIBRARY_PATH"]
    assert "" not in _elements(v2) and v2.endswith("/nvidia/cusparselt/lib")


def test_engine_env_keeps_inherited_but_drops_empties(engine_env, monkeypatch):
    fn, pre = engine_env
    monkeypatch.setenv("LD_LIBRARY_PATH", ":/a::/b:")
    v = fn()["LD_LIBRARY_PATH"]
    assert v == pre + "/lib:/a:/b"


def test_engine_env_empty_parent_value(engine_env, monkeypatch):
    fn, pre = engine_env
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    assert fn()["LD_LIBRARY_PATH"] == pre + "/lib"


# ---------------------------------------------------------------------------
# D3 — ATR fill excluded from price-axis autorange, hidden while empty
# ---------------------------------------------------------------------------
def test_atr_fill_ignored_by_autorange_and_hidden_when_empty():
    build = _source("_build_stocks_tab", ast.FunctionDef)
    assert "addItem(self._atr_fill, ignoreBounds=True)" in build
    assert "self._atr_fill.hide()" in build
    assert "self._stock_chart.addItem(self._atr_fill)\n" not in build
    zoom = _source("_apply_chart_zoom", ast.FunctionDef)
    assert "self._atr_fill.setVisible(" in zoom
    # the symbol-switch / empty-state clear hides the fill with the band lines
    clear = _source("_clear_price_items", ast.FunctionDef)
    assert "ln.setData([], [])" in clear and "self._atr_fill.hide()" in clear


# ---------------------------------------------------------------------------
# D4/D5 — Performance / Markets / Models pages hosted in a QScrollArea
# ---------------------------------------------------------------------------
def test_scroll_wrap_helper_matches_settings_recipe():
    body = _source("_scroll_wrap", ast.FunctionDef)
    for needle in ("QScrollArea()", "setWidgetResizable(True)",
                   "setFrameShape(QFrame.NoFrame)", "setWidget(inner)"):
        assert needle in body


@pytest.mark.parametrize("builder,label", [
    ("_build_performance_tab", "Performance"),
    ("_build_models_tab", "Models"),
])
def test_tab_added_through_scroll_wrap(builder, label):
    body = _source(builder, ast.FunctionDef)
    assert f'self.tabs.addTab(self._scroll_wrap(tab), "{label}")' in body
    assert f'self.tabs.addTab(tab, "{label}")' not in body


def test_markets_tab_index_points_at_scroll_page():
    body = _source("_build_stocks_tab", ast.FunctionDef)
    assert "markets_page = self._scroll_wrap(tab)" in body
    assert 'self.tabs.addTab(markets_page, "Markets")' in body
    # the index must be of the widget actually added, else indexOf -> -1
    assert "self._markets_tab_index = self.tabs.indexOf(markets_page)" in body


def test_models_report_buttons_wrap_to_two_rows():
    """One row of eight report buttons needed ~1300 px (> the 1280 default
    window), which clipped the labels / forced a horizontal scrollbar."""
    body = _source("_build_models_tab", ast.FunctionDef)
    assert "reports_row2 = QHBoxLayout()" in body
    assert "(reports_layout if i < 4 else reports_row2).addWidget(btn)" in body
    assert "reports_v.addLayout(reports_row2)" in body


def test_performance_plots_have_a_usable_floor():
    body = _source("_build_performance_tab", ast.FunctionDef)
    assert "self._equity_plot.setMinimumHeight(" in body
    assert "self._pnl_plot.setMinimumHeight(" in body


# ---------------------------------------------------------------------------
# M3 (review 2026-09-26) — orders stream: full walk at boot + every
# ORDERS_FULL_WALK_SEC, page-1 refresh merged by id in between
# ---------------------------------------------------------------------------
def _module_const(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"module constant {name!r} not found in gui.py")


def test_orders_cadence_constants():
    assert _module_const("ORDERS_FULL_WALK_SEC") == 600
    assert _module_const("ORDERS_REFRESH_PAGES") == 1


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


class _Order:
    def __init__(self, i, ts, status="filled"):
        self.id = f"o{i}"
        self.symbol = "AAA"
        self.side = "buy"
        self.qty = "1"
        self.type = "market"
        self.status = status
        self.submitted_at = ts
        self.filled_at = ts
        self.filled_avg_price = "10"
        self.notional = None
        self.filled_qty = "1"


class _API:
    """list_orders over an in-memory, newest-first order book with Alpaca's
    exclusive RFC3339 `until` cursor; counts requests."""
    T0 = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)

    def __init__(self, n):
        self.orders = []   # newest first
        self.next_i = 0
        self.calls = 0
        self.fail = False
        self.add(n)

    def add(self, n):
        new = []
        for _ in range(n):
            ts = self.T0 + dt.timedelta(minutes=self.next_i)
            new.append(_Order(self.next_i, ts))
            self.next_i += 1
        self.orders = list(reversed(new)) + self.orders

    def list_orders(self, status, limit, after, until, direction):
        self.calls += 1
        if self.fail:
            raise RuntimeError("HTTP 429")
        rows = self.orders
        if until is not None:
            cut = dt.datetime.strptime(until, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=dt.timezone.utc)
            rows = [o for o in rows if o.submitted_at < cut]
        return rows[:limit]


class _Sig:
    def __init__(self):
        self.calls = []

    def emit(self, *a):
        self.calls.append(a)


@pytest.fixture()
def fetcher(tmp_path):
    clock = _Clock()
    ns = {"dt": dt, "re": re, "time": clock, "BASE_DIR": tmp_path,
          "ORDERS_FULL_WALK_SEC": _module_const("ORDERS_FULL_WALK_SEC"),
          "ORDERS_REFRESH_PAGES": _module_const("ORDERS_REFRESH_PAGES")}
    _load("_alpaca_until", ns)
    fetch = _load("fetch_orders", ns)
    walk = _load("_walk_orders_pages", ns)
    f = types.SimpleNamespace(api=_API(250), orders_updated=_Sig(),
                              error_occurred=_Sig(), results=[], clock=clock,
                              slate=tmp_path / ".clean_slate")
    f._stream_result = lambda stream, ok: f.results.append((stream, ok))
    f.fetch_orders = types.MethodType(fetch, f)
    f._walk_orders_pages = types.MethodType(walk, f)
    return f


def _last_emit(f):
    orders, truncated = f.orders_updated.calls[-1]
    return orders, truncated


def test_orders_boot_is_full_walk(fetcher):
    fetcher.fetch_orders()
    orders, truncated = _last_emit(fetcher)
    assert fetcher.api.calls == 3            # 100 + 100 + 50 (short page)
    assert len(orders) == 250 and truncated is False
    assert len({o["id"] for o in orders}) == 250
    assert orders[0]["id"] == "o249"         # newest first
    assert fetcher.results == [("orders", True)]


def test_orders_refresh_is_one_page_and_merges_by_id(fetcher):
    fetcher.fetch_orders()
    fetcher.api.calls = 0
    fetcher.api.add(3)                        # three new orders
    fetcher.api.orders[5].status = "canceled"  # a status change on page 1
    fetcher.clock.t += 30
    fetcher.fetch_orders()
    orders, _ = _last_emit(fetcher)
    assert fetcher.api.calls == 1
    assert len(orders) == 253
    assert len({o["id"] for o in orders}) == 253
    assert [o["id"] for o in orders[:3]] == ["o252", "o251", "o250"]
    by_id = {o["id"]: o for o in orders}
    assert by_id[fetcher.api.orders[5].id]["status"] == "canceled"
    # order is still strictly newest-first across the page/cache seam
    ts = [o["submitted_at"] for o in orders]
    assert ts == sorted(ts, reverse=True)


def test_orders_full_walk_repeats_on_cadence(fetcher):
    fetcher.fetch_orders()
    fetcher.api.calls = 0
    fetcher.clock.t += 599
    fetcher.fetch_orders()
    assert fetcher.api.calls == 1
    fetcher.clock.t += 1                      # 600 s since the boot walk
    fetcher.fetch_orders()
    assert fetcher.api.calls == 1 + 3


def test_orders_gap_forces_full_walk(fetcher):
    fetcher.fetch_orders()
    fetcher.api.calls = 0
    fetcher.api.add(150)                      # > one refresh page since last tick
    fetcher.clock.t += 30
    fetcher.fetch_orders()
    orders, _ = _last_emit(fetcher)
    # refresh page + the 400-order walk (4 full pages + the empty 5th)
    assert fetcher.api.calls == 1 + 5
    assert len(orders) == 400
    assert len({o["id"] for o in orders}) == 400


def test_orders_error_backs_off_and_keeps_cache(fetcher):
    fetcher.fetch_orders()
    n_emits = len(fetcher.orders_updated.calls)
    fetcher.api.fail = True
    fetcher.clock.t += 30
    fetcher.fetch_orders()
    assert fetcher.results[-1] == ("orders", False)
    assert fetcher.error_occurred.calls[-1][0] == "orders"
    assert len(fetcher.orders_updated.calls) == n_emits  # nothing emitted
    fetcher.api.fail = False
    fetcher.api.calls = 0
    fetcher.clock.t += 30
    fetcher.fetch_orders()
    assert fetcher.api.calls == 1             # cache survived -> refresh only
    assert len(_last_emit(fetcher)[0]) == 250


def test_orders_failed_scheduled_walk_does_not_retry_every_tick(fetcher):
    fetcher.fetch_orders()
    fetcher.clock.t += 600
    fetcher.api.fail = True
    fetcher.fetch_orders()                    # the scheduled full walk fails
    fetcher.api.fail = False
    fetcher.api.calls = 0
    fetcher.clock.t += 30
    fetcher.fetch_orders()
    assert fetcher.api.calls == 1             # refresh, not another walk


def test_orders_boot_failure_retries_full_walk(fetcher):
    fetcher.api.fail = True
    fetcher.fetch_orders()
    fetcher.api.fail = False
    fetcher.api.calls = 0
    fetcher.clock.t += 30
    fetcher.fetch_orders()
    assert fetcher.api.calls == 3 and len(_last_emit(fetcher)[0]) == 250


def test_orders_clean_slate_change_forces_full_walk(fetcher):
    fetcher.fetch_orders()
    fetcher.api.calls = 0
    fetcher.slate.write_text("2026-09-01T02:00:00Z")
    fetcher.clock.t += 30
    fetcher.fetch_orders()
    assert fetcher.api.calls == 3             # full walk under the new cutoff


def test_orders_truncated_flag_carried_on_refresh(fetcher):
    fetcher.api.add(1000)                     # 1250 orders > 1000 cap
    fetcher.fetch_orders()
    orders, truncated = _last_emit(fetcher)
    assert truncated is True and len(orders) == 1000
    fetcher.api.calls = 0
    fetcher.clock.t += 30
    fetcher.fetch_orders()
    assert fetcher.api.calls == 1
    assert _last_emit(fetcher)[1] is True


def test_orders_request_budget_over_three_minutes(fetcher):
    """The review's load: ~11 pages per 30 s tick. Now: one walk + one
    page per tick (boot + 6 ticks in 3 min at the 30 s default)."""
    fetcher.api.add(838)                      # 1088 orders, as observed live
    fetcher.fetch_orders()
    walk_pages = fetcher.api.calls
    assert walk_pages == 10                   # 1000-order cap
    for _ in range(6):
        fetcher.clock.t += 30
        fetcher.fetch_orders()
    assert fetcher.api.calls == walk_pages + 6   # was 7 * 10 = 70
