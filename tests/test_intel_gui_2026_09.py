"""INTEL W3 operator-console fixes (2026-09 Jetson, gui.py).

gui.py imports PySide6 (absent on the dev Mac and in the Jetson `jetson` env),
so this file is PySide6-free: layout changes are pinned by parsing the gui.py
AST, and the best-score helpers / methods are extracted from the AST and
exec'd against stubs (the tests/test_g8_fixes_2026_09.py pattern).

- A  the Cockpit and Trading pages are hosted in `_scroll_wrap` (scroll
     instead of clip below ~1280x800), their tables/feeds get a usable
     minimum height, the Trading index is taken of the widget actually added,
     and the long positions headers are two-line.
- B  optuna.load_study (1.2-1.6 s at window build, ~160 ms per cold refresh)
     no longer runs on the UI thread: `_refresh_models_tab` only peeks the
     (path, mtime, size)-keyed cache; misses load on a daemon worker whose
     result is routed back by Signal into `_on_best_scores_ready`.
"""
import ast
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = (REPO / "gui.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def _node(name, kind=(ast.FunctionDef, ast.ClassDef)):
    for node in ast.walk(TREE):
        if isinstance(node, kind) and node.name == name:
            return node
    raise AssertionError(f"{name!r} not found in gui.py")


def _source(name, kind=(ast.FunctionDef, ast.ClassDef)):
    return ast.get_source_segment(SRC, _node(name, kind))


def _load(names, ns):
    for name in names:
        exec(compile(_source(name), f"gui.py::{name}", "exec"), ns)
    return ns


def _is_self_call(node, attr):
    """node is `self.<attr>(...)`."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self")


def _is_tabs_call(node, meth):
    """node is `self.tabs.<meth>(...)`."""
    f = getattr(node, "func", None)
    return (isinstance(node, ast.Call) and isinstance(f, ast.Attribute)
            and f.attr == meth and isinstance(f.value, ast.Attribute)
            and f.value.attr == "tabs")


def _add_tab(builder, label):
    """The first argument of `self.tabs.addTab(<page>, label)` in builder."""
    for node in ast.walk(_node(builder, ast.FunctionDef)):
        if (_is_tabs_call(node, "addTab") and len(node.args) == 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == label):
            return node.args[0]
    raise AssertionError(f"addTab(..., {label!r}) not found in {builder}")


def _resolves_to_scroll_wrap(builder, page):
    """page is `self._scroll_wrap(tab)`, or a local name assigned from it."""
    if _is_self_call(page, "_scroll_wrap"):
        return True
    if isinstance(page, ast.Name):
        for node in ast.walk(_node(builder, ast.FunctionDef)):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == page.id
                            for t in node.targets)
                    and _is_self_call(node.value, "_scroll_wrap")):
                return True
    return False


# ---------------------------------------------------------------------------
# A — Cockpit / Trading scroll instead of clip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("builder,label", [
    ("_build_dashboard_tab", "Cockpit"),
    ("_build_trading_tab", "Trading"),
    # the FIX_F pages stay wrapped (regression guard for the shared helper)
    ("_build_performance_tab", "Performance"),
    ("_build_stocks_tab", "Markets"),
    ("_build_models_tab", "Models"),
])
def test_page_added_through_scroll_wrap(builder, label):
    assert _resolves_to_scroll_wrap(builder, _add_tab(builder, label)), (
        f"{label} page must be added via self._scroll_wrap(...)")


def test_every_tab_index_is_taken_of_the_widget_actually_added():
    """indexOf(inner) after addTab(scroll_page) is -1: Trading's lazy first
    paint (_on_tab_changed) would silently never fire again."""
    checked = 0
    for node in ast.walk(TREE):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("_build_")):
            continue
        added = [ast.dump(n.args[0]) for n in ast.walk(node)
                 if _is_tabs_call(n, "addTab") and n.args]
        for n in ast.walk(node):
            if _is_tabs_call(n, "indexOf"):
                checked += 1
                assert ast.dump(n.args[0]) in added, (
                    f"{node.name}: indexOf({ast.unparse(n.args[0])}) is not "
                    f"the widget passed to addTab")
    assert checked >= 4  # trading, news, markets, logs


def test_trading_index_uses_the_scroll_page():
    body = _source("_build_trading_tab", ast.FunctionDef)
    assert "self._trading_tab_index = self.tabs.indexOf(trading_page)" in body
    assert "self.tabs.indexOf(tab)" not in body


def _min_heights(builder):
    out = {}
    for n in ast.walk(_node(builder, ast.FunctionDef)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "setMinimumHeight"
                and isinstance(n.func.value, ast.Attribute)
                and isinstance(n.func.value.value, ast.Name)
                and n.func.value.value.id == "self"
                and n.args and isinstance(n.args[0], ast.Constant)):
            out[n.func.value.attr] = n.args[0].value
    return out


@pytest.mark.parametrize("builder,widget,floor", [
    # header (~38 px two-line) + ~5 rows of 30 px
    ("_build_dashboard_tab", "_positions_table", 180),
    ("_build_dashboard_tab", "_alerts_list", 60),
    ("_build_dashboard_tab", "_last_actions_list", 60),
    ("_build_trading_tab", "_open_orders_table", 100),
    ("_build_trading_tab", "_fills_table", 180),
    ("_build_trading_tab", "_journal_view", 80),
])
def test_wrapped_page_widgets_have_a_usable_floor(builder, widget, floor):
    """Inside a widget-resizable QScrollArea a table collapses to its ~70 px
    hint (Recent Fills showed 1 row at 1280x800) unless it has a floor."""
    got = _min_heights(builder)
    assert widget in got, f"{builder}: self.{widget}.setMinimumHeight missing"
    assert got[widget] >= floor


def test_positions_headers_fit_a_narrow_stretch_column():
    """At 1280x800 each of the 11 stretch columns gets ~60-70 px, which
    clipped 'Current Price' / 'Unrealized P&L' — every header LINE must stay
    short (<= 10 chars, ~66 px in the header font)."""
    body = _node("_build_dashboard_tab", ast.FunctionDef)
    labels = None
    for n in ast.walk(body):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "setHorizontalHeaderLabels"
                and ast.unparse(n.func.value) == "self._positions_table"):
            labels = ast.literal_eval(n.args[0])
    assert labels is not None and len(labels) == 12
    assert max(len(line) for lab in labels for line in lab.split("\n")) <= 10
    # same 12 columns, same meaning (only line breaks were added)
    assert [lab.replace("\n", " ") for lab in labels] == [
        "Symbol", "Qty", "Side", "Avg Entry", "Current Price", "Mkt Value",
        "Unrealized P&L", "P&L %", "Stop", "TP", "%→Stop", ""]


# ---------------------------------------------------------------------------
# B — best-score cache helpers (pure, exec'd from the AST)
# ---------------------------------------------------------------------------
class _FakeStudy:
    def __init__(self, value):
        self.best_value = value


def _fake_optuna(calls, value=8.154, exc=None, delay=0.0):
    m = types.ModuleType("optuna")
    m.logging = types.SimpleNamespace(WARNING=30, set_verbosity=lambda v: None)

    def load_study(study_name, storage):
        calls.append((study_name, storage, threading.current_thread().name))
        if delay:
            time.sleep(delay)
        if exc is not None:
            raise exc
        return _FakeStudy(value)
    m.load_study = load_study
    return m


@pytest.fixture
def bs(tmp_path):
    db = tmp_path / "stock_v2_study.db"
    db.write_bytes(b"x" * 64)
    ns = {"STUDY_DBS": {"Stock": ("stock_v2_study.db", "stock_v2_search"),
                        "Crypto": ("v2_study.db", "v2_search")},
          "BASE_DIR": tmp_path, "_BEST_SCORE_CACHE": {},
          "BEST_SCORE_PENDING": "…"}
    _load(["_best_score_sig", "_peek_best_score", "_load_best_score",
           "_store_best_score", "_get_best_score", "_best_score_display"], ns)
    ns["db"] = db
    return ns


def test_pending_glyph_constant():
    for node in TREE.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "BEST_SCORE_PENDING"
                        for t in node.targets)):
            assert ast.literal_eval(node.value) == "…"
            return
    raise AssertionError("BEST_SCORE_PENDING not defined")


def test_missing_db_is_known_none_and_never_loads(bs, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "optuna", _fake_optuna(calls))
    assert bs["_peek_best_score"]("Crypto") == (True, None)   # no v2_study.db
    assert bs["_peek_best_score"]("Nope") == (True, None)     # no STUDY_DBS row
    assert bs["_get_best_score"]("Crypto") is None
    assert calls == []


def test_peek_miss_then_load_store_hit(bs, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "optuna", _fake_optuna(calls))
    assert bs["_peek_best_score"]("Stock") == (False, None)
    sig, value = bs["_load_best_score"]("Stock")
    st = bs["db"].stat()
    assert sig == (str(bs["db"]), st.st_mtime, st.st_size)    # same key shape
    assert value == 8.154
    # storage string unchanged (rw sqlite URL, not a mode=ro URI)
    assert calls == [("stock_v2_search", f"sqlite:///{bs['db']}", calls[0][2])]
    assert bs["_BEST_SCORE_CACHE"] == {}      # load never writes the cache
    assert bs["_peek_best_score"]("Stock") == (False, None)
    bs["_store_best_score"]("Stock", sig, value)
    assert bs["_peek_best_score"]("Stock") == (True, 8.154)
    assert len(calls) == 1                    # the peek never loads


@pytest.mark.parametrize("mutate", ["mtime", "size"])
def test_cache_invalidates_when_the_db_stat_signature_moves(bs, monkeypatch, mutate):
    calls = []
    monkeypatch.setitem(sys.modules, "optuna", _fake_optuna(calls))
    assert bs["_get_best_score"]("Stock") == 8.154
    assert bs["_get_best_score"]("Stock") == 8.154
    assert len(calls) == 1
    if mutate == "mtime":
        st = bs["db"].stat()
        os.utime(bs["db"], ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    else:
        with open(bs["db"], "ab") as f:
            f.write(b"y")
    assert bs["_peek_best_score"]("Stock") == (False, None)
    assert bs["_get_best_score"]("Stock") == 8.154
    assert len(calls) == 2


def test_failures_are_not_cached_so_the_next_refresh_retries(bs, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "optuna",
                        _fake_optuna(calls, exc=KeyError("Record does not exist.")))
    sig, value = bs["_load_best_score"]("Stock")
    assert sig is not None and value is None
    bs["_store_best_score"]("Stock", sig, value)
    assert bs["_BEST_SCORE_CACHE"] == {}
    assert bs["_get_best_score"]("Stock") is None
    assert bs["_get_best_score"]("Stock") is None
    assert len(calls) == 3


def test_display_rules(bs):
    disp = bs["_best_score_display"]
    shown = {}
    assert disp(False, None, shown, "Stock") == ("…", None)   # first paint
    assert disp(True, None, shown, "Stock") == ("—", None)    # no db / failed
    assert disp(True, 8.15378, shown, "Stock") == ("8.154", 8.15378)
    shown["Stock"] = 6.8267                                         # reload in flight
    assert disp(False, None, shown, "Stock") == ("6.827", 6.8267)
    shown["Stock"] = None                                           # last load failed
    assert disp(False, None, shown, "Stock") == ("—", None)


# ---------------------------------------------------------------------------
# B — routing: UI-thread refresh never loads; worker -> Signal -> slot
# ---------------------------------------------------------------------------
def _calls_in(fn_name):
    names = set()
    for n in ast.walk(_node(fn_name, ast.FunctionDef)):
        if isinstance(n, ast.Call):
            f = n.func
            names.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", None))
    return names


def test_refresh_models_tab_never_blocks_on_optuna():
    called = _calls_in("_refresh_models_tab")
    assert "_get_best_score" not in called and "_load_best_score" not in called
    assert "load_study" not in called
    assert {"_peek_best_score", "_best_score_display",
            "_start_best_score_load"} <= called


def test_signal_connected_before_the_first_refresh():
    init = None
    for node in ast.walk(_node("TradingDashboard", ast.ClassDef)):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            init = ast.get_source_segment(SRC, node)
            break
    conn = init.index("self._best_scores_ready.connect(self._on_best_scores_ready)")
    first = init.index("self._refresh_models_tab()")
    assert conn < first
    assert init.index("self._best_score_shown = {}") < first
    cls = _source("TradingDashboard", ast.ClassDef)
    assert "_best_scores_ready = Signal(object)" in cls


class _Signal:
    def __init__(self):
        self.calls = []
        self.done = threading.Event()

    def emit(self, *a):
        self.calls.append((a, threading.current_thread().name))
        self.done.set()


class _Item:
    def __init__(self, text=""):
        self._text = text
        self.fg = None
        self.align = None

    def text(self):
        return self._text

    def setTextAlignment(self, a):
        self.align = a

    def setForeground(self, c):
        self.fg = c


class _Table:
    def __init__(self, names):
        self.cells = {(r, 0): _Item(n) for r, n in enumerate(names)}

    def rowCount(self):
        return len({r for r, _ in self.cells})

    def item(self, r, c):
        return self.cells.get((r, c))

    def setItem(self, r, c, it):
        self.cells[(r, c)] = it


def _dash(bs):
    ns = dict(bs)
    ns.update({"QTableWidgetItem": _Item,
               "Qt": types.SimpleNamespace(AlignCenter=4),
               "T": {"green": "GREEN", "yellow": "YELLOW"},
               "Slot": lambda *a, **k: (lambda f: f)})
    _load(["_start_best_score_load", "_on_best_scores_ready"], ns)
    self = types.SimpleNamespace(
        _best_score_busy=False, _best_score_shown={},
        _best_scores_ready=_Signal(), _model_table=_Table(["Crypto", "Stock"]))
    start = types.MethodType(ns["_start_best_score_load"], self)
    ready = types.MethodType(ns["_on_best_scores_ready"], self)
    return ns, self, start, ready


def test_worker_loads_off_the_calling_thread_and_routes_results(bs, monkeypatch):
    calls = []
    gate = threading.Event()
    fake = _fake_optuna(calls)
    real_load = fake.load_study

    def gated(study_name, storage):
        gate.wait(5)
        return real_load(study_name=study_name, storage=storage)
    fake.load_study = gated
    monkeypatch.setitem(sys.modules, "optuna", fake)
    ns, self, start, ready = _dash(bs)

    t0 = time.perf_counter()
    start(["Stock"])
    assert time.perf_counter() - t0 < 0.5      # returns while the load blocks
    assert self._best_score_busy is True
    start(["Stock"])                           # one load in flight: no-op
    gate.set()
    assert self._best_scores_ready.done.wait(5)
    assert len(self._best_scores_ready.calls) == 1
    (payload,), thread_name = self._best_scores_ready.calls[0]
    assert thread_name == "best-score" != threading.current_thread().name
    assert calls[0][2] == "best-score"         # load_study ran on the worker
    assert len(calls) == 1
    sig, value = payload["Stock"]
    assert value == 8.154 and sig[0] == str(bs["db"])
    assert ns["_BEST_SCORE_CACHE"] == {}       # worker never writes the cache

    ready(payload)                             # queued slot, UI thread
    assert self._best_score_busy is False
    assert ns["_BEST_SCORE_CACHE"]["Stock"] == (sig, 8.154)
    assert ns["_peek_best_score"]("Stock") == (True, 8.154)
    assert self._best_score_shown == {"Stock": 8.154}
    cell = self._model_table.item(1, 2)        # the Stock row only
    assert cell.text() == "8.154" and cell.fg == "GREEN" and cell.align == 4
    assert self._model_table.item(0, 2) is None


def test_failed_worker_load_shows_dash_and_is_not_cached(bs, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "optuna",
                        _fake_optuna(calls, exc=RuntimeError("locked")))
    ns, self, start, ready = _dash(bs)
    start(["Stock"])
    assert self._best_scores_ready.done.wait(5)
    (payload,), _ = self._best_scores_ready.calls[0]
    ready(payload)
    assert ns["_BEST_SCORE_CACHE"] == {}
    assert self._best_score_shown == {"Stock": None}
    cell = self._model_table.item(1, 2)
    assert cell.text() == "—" and cell.fg is None
    assert self._best_score_busy is False
    # next refresh: still stale -> retried, but the cell keeps '—' (no '…')
    assert ns["_peek_best_score"]("Stock") == (False, None)
    assert ns["_best_score_display"](False, None, self._best_score_shown,
                                     "Stock") == ("—", None)
