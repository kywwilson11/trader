"""INTEL W11 (2026-09-27): Cockpit alarm hierarchy + readiness-ETA column.

PySide6-free: chart_core is imported directly (numpy + stdlib); the gui.py
methods are exec'd from the gui.py AST against stubs (the
tests/test_intel_gui2_2026_09.py pattern).

- chart_core.alert_priority: every gui.py `_push_alert(` kind maps to a
  priority with a TEXT tag (WCAG SC 1.4.1: colour is never the only channel).
- chart_core.AlertLedger: per-kind+text dedupe over 10 min (xN counter),
  flood collapse (>10 / 10 min), ack (row kept, highlight gone), shelve
  (kind hidden 30 min, excluded from the flood rate), cap 100.
- gui.py wiring: byte-pin — no collision and < 10 alerts => the row text is
  the pre-W11 '<ts>  <text>' with only '[Pn] ' prefixed; 30 mixed alerts in
  one minute; click/double-click/shelve handlers; cockpit tick re-renders.
- chart_core.format_eta / evidence_readiness_rows(with_eta=True) and the
  Models-tab ETA column, with and without W12's eta_days/eta_basis fields.
"""
import ast
import html
import json
import os
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import chart_core  # noqa: E402

SRC = (REPO / "gui.py").read_text()
TREE = ast.parse(SRC)


def _node(name):
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"{name!r} not found in gui.py")


def _class_src(*names):
    # get_source_segment drops only the FIRST line's 4-space method indent
    return "class Win:\n" + "\n".join(
        "    " + ast.get_source_segment(SRC, _node(n)) for n in names)


# ---------------------------------------------------------------------------
# alert_priority
# ---------------------------------------------------------------------------
def _gui_alert_kinds():
    out = []
    for n in ast.walk(TREE):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_push_alert"):
            assert len(n.args) == 2 and not n.keywords   # signature unchanged
            assert isinstance(n.args[0], ast.Constant)
            out.append((n.args[0].value, n.lineno))
    return out


def test_every_gui_alert_kind_has_a_documented_priority():
    kinds = _gui_alert_kinds()
    assert len(kinds) >= 11
    doc = chart_core.alert_priority.__doc__
    for kind, _line in kinds:          # line numbers drift: not pinned here
        assert kind in chart_core.ALERT_PRIORITY, kind
        p, tag = chart_core.alert_priority(kind, "x")
        assert tag == f"[{p}]" and p in ("P1", "P2", "P3")
        assert kind in doc and "gui.py:" in doc


def test_alert_priority_mapping():
    P = lambda k, t="": chart_core.alert_priority(k, t)[0]   # noqa: E731
    assert [P(k) for k in ("halt", "flatten", "order-error", "rejected")] == ["P1"] * 4
    assert [P(k) for k in ("heartbeat", "stream", "stale")] == ["P2"] * 3
    assert [P(k) for k in ("resume", "flatten-complete")] == ["P3"] * 2
    assert P("novel", "fetch ERROR") == "P2" and P("novel", "x failing") == "P2"
    assert P("novel", "hello") == "P3" and P(None, None) == "P3"


# ---------------------------------------------------------------------------
# AlertLedger (pure, injected clock)
# ---------------------------------------------------------------------------
def _labels(led, now):
    return [r["label"] for r in led.rows(now)]


def test_dedupe_counter_window_and_realarm():
    led = chart_core.AlertLedger()
    led.push("stale", "pipeline status stale (>2 min)", 1000, "12:00:00")
    led.push("halt", "Trading halted — entries blocked", 1010, "12:00:10")
    assert _labels(led, 1010) == ["[P1] 12:00:10  Trading halted — entries blocked",
                                  "[P2] 12:00:00  pipeline status stale (>2 min)"]
    led.ack(("stale", "pipeline status stale (>2 min)"), 1011)
    led.push("stale", "pipeline status stale (>2 min)", 1300, "12:05:00")
    led.push("stale", "pipeline status stale (>2 min)", 1500, "12:08:20")
    rows = led.rows(1500)
    assert len(rows) == 2                                  # not 4 rows
    assert rows[0]["label"] == "[P2] 12:08:20  pipeline status stale (>2 min) ×3"
    assert rows[0]["acked"] is False                       # re-alarm clears ack
    # 10-min window measured from the LAST occurrence: 1500+601 -> a new row
    led.push("stale", "pipeline status stale (>2 min)", 2101, "12:18:21")
    assert _labels(led, 2101)[0] == "[P2] 12:18:21  pipeline status stale (>2 min)"
    assert len(led.rows(2101)) == 3
    # same kind, different text is a different row
    led.push("stale", "other", 2102, "t")
    assert len(led.rows(2102)) == 4


def _thirty(led, t0=10_000.0):
    """30 alerts of mixed kinds inside one minute: 12 distinct kind+text."""
    kinds = [("halt", "Trading halted — entries blocked"),
             ("order-error", "AAPL order error: boom"),
             ("order-error", "MSFT close error: boom"),
             ("heartbeat", "crypto heartbeat stale"),
             ("stream", "quotes stream failing (x3)"),
             ("stale", "pipeline status stale (>2 min)"),
             ("resume", "Entries resumed"),
             ("flatten", "Flatten requested"),
             ("flatten-complete", "Flatten complete — trading halted"),
             ("rejected", "command RETRAIN REJECTED: busy"),
             ("heartbeat", "stock heartbeat stale"),
             ("stream", "bars stream failing (x3)")]
    seq = [kinds[i % 12] for i in range(30)]
    for i, (k, t) in enumerate(seq):
        led.push(k, t, t0 + 2 * i, "12:00:%02d" % (2 * i))
    return seq, t0 + 58


def test_flood_collapse_30_alerts_in_one_minute():
    led = chart_core.AlertLedger()
    led.push("resume", "old one", 1_000.0, "11:00:00")         # outside window
    seq, now = _thirty(led)
    n, c = led.flood_counts(now)
    exp = {p: sum(chart_core.alert_priority(k)[0] == p for k, _ in seq)
           for p in ("P1", "P2", "P3")}
    assert n == 30 and c == exp == {"P1": 13, "P2": 13, "P3": 4}
    rows = led.rows(now)
    assert len(rows) == 2                                 # collapse + old row
    top = rows[0]
    assert top["key"] == chart_core.AlertLedger.COLLAPSED_KEY and top["prio"] == "P1"
    assert top["label"] == ("[P1] 12:00:58  30 alerts collapsed "
                            "(P1 13, P2 13, P3 4) [+]")
    assert top["detail"].startswith("12 rows — click to expand\n")
    assert top["detail"].count("\n") == 12                # + 12 hidden rows
    assert "×3" in top["detail"] and "×2" in top["detail"]
    assert rows[1]["label"] == "[P3] 11:00:00  old one"
    led.expanded = True
    rows = led.rows(now)
    assert len(rows) == 1 + 12 + 1 and rows[0]["label"].endswith("[−]")
    assert rows[0]["detail"].startswith("12 rows — click to collapse")
    counts = sorted(int(r["label"].rsplit("×", 1)[1]) if "×" in r["label"] else 1
                    for r in rows[1:13])
    assert counts == [2] * 6 + [3] * 6 and sum(counts) == 30
    # the flood decays once the window passes
    led.expanded = False
    assert len(led.rows(now + 700)) == 13


def test_no_flood_at_ten_or_single_row():
    led = chart_core.AlertLedger()
    for i in range(10):
        led.push("stream", f"s{i}", 100 + i, "t")
    assert len(led.rows(110)) == 10                       # exactly 10: no flood
    led2 = chart_core.AlertLedger()
    for i in range(25):
        led2.push("stale", "same", 100 + i, "t")
    assert _labels(led2, 125) == ["[P2] t  same ×25"]     # one row: nothing to fold


def test_ack_keeps_row_and_collapse_ack_acks_all():
    led = chart_core.AlertLedger()
    led.push("halt", "h", 1, "t1")
    led.push("resume", "r", 2, "t2")
    assert led.ack(("halt", "h"), 3) == 1
    rows = led.rows(3)
    assert [r["label"] for r in rows] == ["[P3] t2  r", "[P1] t1  h (ack)"]
    assert [r["acked"] for r in rows] == [False, True]
    seq, now = _thirty(led)
    assert led.rows(now)[0]["acked"] is False
    assert led.ack(chart_core.AlertLedger.COLLAPSED_KEY, now) == 12
    top = led.rows(now)[0]
    assert top["acked"] is True and top["label"].endswith("(ack)")
    led.ack_all()
    assert all(r["acked"] for r in led.rows(now))


def test_shelve_hides_kind_for_30_min_and_leaves_flood_rate():
    led = chart_core.AlertLedger()
    led.push("heartbeat", "crypto heartbeat stale", 0, "t0")
    led.push("halt", "h", 1, "t1")
    led.shelve("heartbeat", 2)
    led.push("heartbeat", "crypto heartbeat stale", 3, "t3")  # counted, hidden
    rows = led.rows(4)
    assert [r["label"] for r in rows] == [
        "[P1] t1  h", "[shelved] heartbeat — 30 min left (click to unshelve)"]
    assert rows[1]["kind"] is None and rows[1]["key"] == ("__shelved__", "heartbeat")
    for i in range(12):
        led.push("heartbeat", f"x{i}", 5 + i, "t")
    assert led.flood_counts(20) == (1, {"P1": 1, "P2": 0, "P3": 0})
    assert not any("collapsed" in lb for lb in _labels(led, 20))
    # expiry after 30 min: the hidden row (×2) is back, the shelf row is gone
    labels = _labels(led, 2 + 1800)
    assert "[P2] t3  crypto heartbeat stale ×2" in labels
    assert not any(lb.startswith("[shelved]") for lb in labels)
    led.shelve("halt", 3000)
    led.unshelve("halt")
    assert not any(lb.startswith("[shelved]") for lb in _labels(led, 3001))


def test_cap_100_rows():
    led = chart_core.AlertLedger(flood_n=10 ** 9)
    for i in range(150):
        led.push("stream", f"s{i}", i, "t")
    rows = led.rows(150)
    assert len(rows) == 100 and rows[0]["label"] == "[P2] t  s149"


# ---------------------------------------------------------------------------
# gui.py wiring (AST-exec against stubs)
# ---------------------------------------------------------------------------
class _Color:
    def __init__(self, n):
        self._n = n

    def name(self):
        return self._n


class _Font:
    def __init__(self):
        self.bold = False

    def setBold(self, b):
        self.bold = b


class _Item:
    def __init__(self, text):
        self._text, self._data, self.fg, self.tip = text, {}, None, None
        self._font = _Font()

    def text(self):
        return self._text

    def setData(self, role, v):
        self._data[role] = v

    def data(self, role):
        return self._data.get(role)

    def setForeground(self, c):
        self.fg = c

    def font(self):
        return self._font

    def setFont(self, f):
        self._font = f

    def setToolTip(self, t):
        self.tip = t


class _List:
    def __init__(self):
        self.items, self.n_clear = [], 0

    def clear(self):
        self.items, self.n_clear = [], self.n_clear + 1

    def addItem(self, it):
        self.items.append(it)

    def count(self):
        return len(self.items)

    def item(self, i):
        return self.items[i] if 0 <= i < len(self.items) else None


class _Clock:
    def __init__(self, t):
        self.t = t

    def time(self):
        return self.t


def _alert_win():
    clock = _Clock(50_000.0)

    class _Now:
        def strftime(self, fmt):
            assert fmt == "%H:%M:%S"
            s = int(clock.t) % 86400
            return "%02d:%02d:%02d" % (s // 3600, s // 60 % 60, s % 60)

    dt_stub = types.SimpleNamespace(datetime=types.SimpleNamespace(
        now=lambda tz: _Now()))
    T = {k: _Color(v) for k, v in {"red": "#ff0000", "yellow": "#ffff00",
                                   "green": "#00ff00", "white": "#ffffff",
                                   "muted": "#888888"}.items()}
    ns = {"T": T, "time": clock, "dt": dt_stub, "TZ_CENTRAL": None,
          "chart_core": chart_core, "QListWidgetItem": _Item,
          "Qt": types.SimpleNamespace(UserRole=256)}
    exec(compile(_class_src(
        "_alert_color", "_alert_ledger_get", "_push_alert", "_render_alerts",
        "_alert_row_of", "_on_alert_clicked", "_on_alert_double_clicked",
        "_alert_shelve", "_alert_ack_all"), "gui.py::W11", "exec"), ns)
    win = ns["Win"]()
    win._alerts_list = _List()
    return win, clock, ns


def _texts(win):
    return [it.text() for it in win._alerts_list.items]


def test_gui_byte_pin_old_text_plus_tag():
    """No dedupe collision and < 10 alerts: every row == the pre-W11 row text
    (f"{ts}  {text}", gui.py _push_alert) with only '[Pn] ' prefixed."""
    win, clock, _ = _alert_win()
    pushes = [("stale", "pipeline status stale (>2 min)"),
              ("halt", "Trading halted — entries blocked"),
              ("resume", "Entries resumed"),
              ("order-error", "AAPL order error: 403 forbidden"),
              ("heartbeat", "crypto heartbeat stale")]
    old = []
    for i, (k, t) in enumerate(pushes):
        clock.t = 50_000.0 + 7 * i
        win._push_alert(k, t)
        s = int(clock.t) % 86400
        ts = "%02d:%02d:%02d" % (s // 3600, s // 60 % 60, s % 60)
        old.insert(0, (chart_core.alert_priority(k, t)[1], f"{ts}  {t}"))
    assert _texts(win) == [f"{tag} {o}" for tag, o in old]
    it = win._alerts_list.items
    assert [x.fg.name() for x in it] == ["#ffff00", "#ff0000", "#00ff00",
                                         "#ff0000", "#ffff00"]
    assert [x._font.bold for x in it] == [False, True, False, True, False]
    assert all(x.tip is None for x in it)


def test_gui_identical_repeat_now_counts_instead_of_skipping():
    win, clock, _ = _alert_win()
    win._push_alert("stale", "pipeline status stale (>2 min)")
    clock.t += 30
    win._push_alert("stale", "pipeline status stale (>2 min)")
    assert _texts(win) == ["[P2] 13:53:50  pipeline status stale (>2 min) ×2"]
    n = win._alerts_list.n_clear
    win._render_alerts()                                  # unchanged: no rebuild
    assert win._alerts_list.n_clear == n


def test_gui_thirty_alerts_collapse_ack_shelve():
    win, clock, _ = _alert_win()
    led = chart_core.AlertLedger()
    seq, _ = _thirty(led)
    for i, (k, t) in enumerate(seq):
        clock.t = 50_000.0 + 2 * i
        win._push_alert(k, t)
    assert len(_texts(win)) == 1
    top = win._alerts_list.items[0]
    assert top.text() == ("[P1] 13:54:18  30 alerts collapsed "
                          "(P1 13, P2 13, P3 4) [+]")
    assert top.fg.name() == "#ff0000" and top._font.bold
    assert top.tip.count("\n") == 12
    win._on_alert_clicked(top)                            # expand
    assert len(_texts(win)) == 13
    assert sum(("×" in t) for t in _texts(win)) == 12
    row = win._alert_row_of(win._alerts_list.items[1])
    win._on_alert_double_clicked(win._alerts_list.items[1])   # ack one row
    acked = [it for it in win._alerts_list.items if it.text().endswith("(ack)")]
    assert len(acked) == 1 and acked[0].fg.name() == "#888888"
    assert not acked[0]._font.bold and len(_texts(win)) == 13
    win._alert_shelve(row["kind"])
    t = _texts(win)
    assert any(x.startswith(f"[shelved] {row['kind']} — 30 min left") for x in t)
    shelf = [it for it in win._alerts_list.items if it.text().startswith("[shelved]")][0]
    win._on_alert_double_clicked(shelf)                   # no-op on a shelf row
    win._on_alert_clicked(shelf)                          # unshelve
    assert not any(x.startswith("[shelved]") for x in _texts(win))
    win._on_alert_clicked(win._alerts_list.items[0])      # collapse again
    assert len(_texts(win)) == 1
    win._alert_ack_all()
    assert _texts(win)[0].endswith("(ack)")
    assert win._alerts_list.items[0].fg.name() == "#888888"


def test_gui_no_list_is_a_noop():
    win, _, _ = _alert_win()
    del win._alerts_list
    win._push_alert("halt", "x")                          # no raise
    win._render_alerts()


def test_gui_cockpit_tick_rerenders_alerts_and_menu_wired():
    body = ast.get_source_segment(SRC, _node("_refresh_cockpit"))
    assert "self._render_alerts" in body
    build = ast.get_source_segment(SRC, _node("_build_dashboard_tab"))
    for s in ("itemClicked.connect(self._on_alert_clicked)",
              "itemDoubleClicked.connect",
              "customContextMenuRequested.connect"):
        assert s in build
    menu = ast.get_source_segment(SRC, _node("_on_alert_menu"))
    assert "Acknowledge all" in menu and "for 30 min" in menu


# ---------------------------------------------------------------------------
# Task B: readiness ETA
# ---------------------------------------------------------------------------
def test_format_eta():
    f = chart_core.format_eta
    assert f(12.4, "NOT YET") == "~12 d" and f(0.3, "NOT YET") == "<1 d"
    assert f(0, "NOT YET") == "<1 d" and f(99.6, "NO DATA") == "~100 d"
    assert f(3.0, "READY") == "n/a (READY)" and f(None, "READY") == "n/a (READY)"
    for bad in (None, float("nan"), float("inf"), -1, "x", True, [1]):
        assert f(bad, "NOT YET") == "—"


SUMMARY_OLD = {"generated_at": "2026-09-27T05:26:36+00:00", "readiness": [
    {"read": "beta_ledger", "rule": "n_obs_used >= 60", "source": "provisional",
     "observed": "n_obs_used=87", "verdict": "READY"},
    {"read": "decision_report", "rule": "quality.priced >= 30",
     "source": "runbook 03:12", "observed": "priced=2",
     "verdict": "NOT YET (priced 2 < 30)"}]}
SUMMARY_ETA = {"generated_at": "2026-09-27T05:26:36+00:00", "readiness": [
    dict(SUMMARY_OLD["readiness"][0], eta_days=None, eta_basis="ready"),
    dict(SUMMARY_OLD["readiness"][1], eta_days=11.7,
         eta_basis="linear/3 runs over 4.0 d"),
    {"read": "llm_eval", "rule": "n >= 60", "source": "runbook 03:48",
     "observed": "n=0", "verdict": "NOT YET (n 0 < 60)", "eta_days": None,
     "eta_basis": "no accrual"},
    {"read": "execution", "rule": "n_buys_with_slippage >= 30", "source": "d",
     "observed": "n_buys_with_slippage=1", "verdict": "NOT YET",
     "eta_days": 4.2}]}


def test_readiness_rows_with_eta():
    base = chart_core.evidence_readiness_rows(SUMMARY_ETA)
    assert all(len(r) == 6 for r in base)                  # default shape kept
    assert base == chart_core.evidence_readiness_rows(SUMMARY_OLD) + [
        r for r in base[2:]]
    rows = chart_core.evidence_readiness_rows(SUMMARY_ETA, with_eta=True)
    assert [r[:6] for r in rows] == base
    assert [r[6:] for r in rows] == [("n/a (READY)", "ready"),
                                     ("~12 d", "linear/3 runs over 4.0 d"),
                                     ("—", "no accrual"), ("~4 d", "")]
    old = chart_core.evidence_readiness_rows(SUMMARY_OLD, with_eta=True)
    assert [r[6:] for r in old] == [("n/a (READY)", None), ("—", None)]
    assert chart_core.evidence_readiness_rows(None, with_eta=True) == []


class _Label:
    def __init__(self):
        self.text, self.tip = None, None

    def setText(self, t):
        self.text = t

    def setToolTip(self, t):
        self.tip = t


def _panel(tmp_path, summary):
    T = {k: _Color(v) for k, v in {"green": "#00ff00", "yellow": "#ffff00",
                                   "red": "#ff0000", "white": "#ffffff",
                                   "muted": "#888888"}.items()}
    ns = {"T": T, "html": html, "os": os, "json": json, "time": time,
          "chart_core": chart_core,
          "EVIDENCE_READS_DIR": tmp_path / "logs" / "evidence_reads"}
    exec(compile(_class_src("_refresh_evidence_panel"), "gui.py::W11b",
                 "exec"), ns)
    win = ns["Win"]()
    win._evidence_label = _Label()
    run = ns["EVIDENCE_READS_DIR"] / "20260927T052636Z"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(summary))
    win._refresh_evidence_panel()
    return win._evidence_label


def test_gui_eta_column_with_fields(tmp_path):
    lab = _panel(tmp_path, SUMMARY_ETA)
    t = lab.text
    assert "<th align='left'>ETA</th>" in t
    assert "no ETA projection" not in t and "ETA = accrual projection" in t
    assert t.count("<tr>") == 1 + 4
    for cell in ("n/a (READY)", "~12 d", "—", "~4 d"):
        assert f"<td>{html.escape(cell)}&nbsp;&nbsp;</td>" in t
    assert "decision_report: runbook 03:12 · ETA basis: linear/3 runs over 4.0 d" in lab.tip
    assert "execution: d" in lab.tip.splitlines()


def test_gui_eta_column_without_fields(tmp_path):
    lab = _panel(tmp_path, SUMMARY_OLD)
    t = lab.text
    assert "<th align='left'>ETA</th>" in t and "no ETA projection" in t
    assert "<td>n/a (READY)&nbsp;&nbsp;</td>" in t
    assert "<td>—&nbsp;&nbsp;</td>" in t
    assert "ETA basis" not in lab.tip
