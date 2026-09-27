"""G8 operator-console fixes (2026-09 Jetson hunt, gui.py + tax_lots.py).

gui.py imports PySide6 (absent on the dev Mac and in the Jetson `jetson` env),
so every gui.py piece tested here is a pure helper (or a method / small class
exec'd against stubs) extracted from the gui.py AST. tax_lots.py is pure
stdlib and imported directly.

- G8-1  llm_analysis.json {crypto, stock} flatten keeps the NEWEST timestamp per
        symbol (a stale crypto copy in the stock section used to win).
- G8-2  Positions-table exit levels: 'BTCUSD' (Alpaca's slash-less position
        symbol) gets the crypto policy and the 'BTC/USD' trailing state.
- G8-3  canceled/expired/partially_filled orders with filled_qty > 0 are fills
        (tax lots + Recent Fills; the lot arithmetic is in
        tests/test_tax_lots.py::TestG8FillsOnNonFilledStatus).
- G8-4  first view of a log reads a bounded tail window, byte-identical buffer.
- G8-5  gui_settings.json / news_cache.json are written atomically.
- G8-6  long-term = sold after the calendar anniversary (29 Feb aware)
        -> tests/test_tax_lots.py::TestG8LeapYearLongTerm.
- G8-7  LogTailer reads a byte range and decodes it (no char/byte mix-up).
- G8-8  float dust (|remaining| < 1e-9) is a matched sell
        -> tests/test_tax_lots.py::TestG8FloatDust.
"""
import ast
import datetime as dt
import errno
import json
import os
import pathlib
import types
from pathlib import Path

import pytest

from tax_lots import order_has_fill

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


def _load(names, ns):
    for name in names:
        exec(compile(_source(name), f"gui.py::{name}", "exec"), ns)
    return ns


UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# G8-1 — newest-timestamp-wins section merge
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def merge():
    ns = _load(["_llm_entry_ts", "_merge_llm_sections"], {"dt": dt})
    return ns["_merge_llm_sections"]


def _real_shape():
    """The shape of the real llm_analysis.json on the Jetson (May 2026): the
    crypto section holds the fresh BTC/USD entry, the stock section (written
    later in file order) still holds a stale copy from a March batch."""
    fresh = {"m": 1, "s": "bullish", "timestamp": "2026-05-05T04:29:52+00:00",
             "model": "gemini-3.1-pro-preview"}
    stale = {"m": 0, "s": "neutral", "timestamp": "2026-03-30T23:49:10+00:00",
             "model": "gemini-3-flash-preview"}
    eth = {"s": "bearish", "timestamp": "2026-05-05T04:30:00+00:00"}
    dash = {"s": "bullish", "timestamp": "2026-05-06T14:00:00+00:00"}
    doc = {"crypto": {"BTC/USD": fresh, "ETH/USD": eth},
           "stock": {"DASH": dash, "BTC/USD": stale}}
    return doc, fresh, stale, eth, dash


def test_merge_keeps_newest_duplicate(merge):
    doc, fresh, stale, eth, dash = _real_shape()
    out = merge(doc)
    assert out["BTC/USD"] is fresh
    assert set(out) == {"BTC/USD", "ETH/USD", "DASH"}
    assert out["ETH/USD"] is eth and out["DASH"] is dash


def test_merge_newest_wins_regardless_of_section_order(merge):
    doc, fresh, stale, _, _ = _real_shape()
    flipped = {"stock": doc["stock"], "crypto": doc["crypto"]}
    assert merge(flipped)["BTC/USD"] is fresh
    # and a genuinely newer stock-section copy DOES win
    newer = dict(stale, timestamp="2026-06-01T00:00:00Z")
    doc["stock"]["BTC/USD"] = newer
    assert merge(doc)["BTC/USD"] is newer


def test_merge_identical_to_naive_for_non_duplicates(merge):
    doc, *_ = _real_shape()
    doc["stock"].pop("BTC/USD")
    naive = {}
    for section in doc.values():
        naive.update(section)
    out = merge(doc)
    assert out == naive and all(out[k] is naive[k] for k in naive)


def test_merge_missing_or_bad_timestamp_loses(merge):
    good = {"timestamp": "2026-01-01T00:00:00+00:00"}
    for bad in ({}, {"timestamp": None}, {"timestamp": "not-a-date"}, "junk"):
        assert merge({"crypto": {"X": good}, "stock": {"X": bad}})["X"] is good
        assert merge({"crypto": {"X": bad}, "stock": {"X": good}})["X"] is good


def test_merge_ties_keep_file_order(merge):
    a = {"timestamp": "2026-01-01T00:00:00+00:00", "id": "a"}
    b = {"timestamp": "2026-01-01T00:00:00Z", "id": "b"}
    assert merge({"crypto": {"X": a}, "stock": {"X": b}})["X"] is b
    na, nb = {"id": "na"}, {"id": "nb"}
    assert merge({"crypto": {"X": na}, "stock": {"X": nb}})["X"] is nb


def test_merge_naive_timestamp_is_utc_and_z_suffix_parses(merge):
    older_naive = {"timestamp": "2026-01-01T10:00:00"}
    newer_z = {"timestamp": "2026-01-01T10:00:01Z"}
    assert merge({"stock": {"X": newer_z}, "crypto": {"X": older_naive}})["X"] is newer_z


def test_merge_skips_non_dict_sections_and_docs(merge):
    assert merge([1, 2]) == {} and merge(None) == {}
    out = merge({"meta": "v2", "crypto": {"A": {"timestamp": "2026-01-01"}}})
    assert list(out) == ["A"]


def test_merge_on_real_llm_analysis_file_if_present(merge):
    path = REPO / "llm_analysis.json"
    if not path.exists():
        pytest.skip("no llm_analysis.json on this machine")
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        pytest.skip("llm_analysis.json unreadable / mid-write")
    ts = _load(["_llm_entry_ts"], {"dt": dt})["_llm_entry_ts"]
    out = merge(doc)
    for sym, entry in out.items():
        stamps = [ts(sec[sym]) for sec in doc.values()
                  if isinstance(sec, dict) and sym in sec]
        stamps = [t for t in stamps if t is not None]
        if stamps:
            assert ts(entry) == max(stamps), sym


def test_both_readers_use_the_shared_helper():
    fetch = _source("fetch_stocks")
    reload_ = _source("_reload_llm_from_disk")
    for body in (fetch, reload_):
        assert "_merge_llm_sections(raw)" in body
        assert ".update(section)" not in body


# ---------------------------------------------------------------------------
# G8-2 — slash-insensitive crypto classification + position-state lookup
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def exit_ns():
    from strategy_config import CRYPTO_POLICY, STOCK_POLICY
    ns = {"CRYPTO_POLICY": CRYPTO_POLICY, "STOCK_POLICY": STOCK_POLICY}
    _load(["_build_crypto_symbol_set"], ns)
    ns["CRYPTO_SYMBOL_SET"] = ns["_build_crypto_symbol_set"]()
    _load(["_is_crypto_symbol", "_lookup_position_state", "_exit_levels_raw"], ns)
    return ns


def test_is_crypto_symbol_both_spellings(exit_ns):
    f = exit_ns["_is_crypto_symbol"]
    for s in ("BTC/USD", "BTCUSD", "btcusd", "ETHUSD", "DOGE/USD"):
        assert f(s), s
    for s in ("AAPL", "SOXL", "", None, "FOOUSD"):
        assert not f(s), s


def test_lookup_position_state_slash_insensitive(exit_ns):
    f = exit_ns["_lookup_position_state"]
    st = {"hwm": 90000.0, "trailing": True}
    pstates = {"BTC/USD": st, "AAPL": {"hwm": 200.0, "trailing": False}}
    assert f(pstates, "BTC/USD") is st
    assert f(pstates, "BTCUSD") is st
    assert f(pstates, "btcusd") is st
    assert f(pstates, "AAPL") is pstates["AAPL"]
    assert f(pstates, "ETHUSD") == {}
    assert f({}, "BTCUSD") == {} and f(None, "BTCUSD") == {}


def test_btcusd_gets_crypto_policy_and_btc_slash_usd_state(exit_ns):
    """The hunt's proof case: entry 80 000, hwm 90 000, trailing on. The table
    path ('BTCUSD') used to show the STOCK floor with no ratchet (79 200 /
    81 600); it must now equal the chart path ('BTC/USD')."""
    raw = exit_ns["_exit_levels_raw"]
    cp = exit_ns["CRYPTO_POLICY"]
    pstates = {"BTC/USD": {"hwm": 90000.0, "trailing": True}}
    table = raw(None, "BTCUSD", 80000.0, 85000.0, pstates)
    chart = raw(None, "BTC/USD", 80000.0, 85000.0, pstates)
    assert table == chart
    floor, rr = cp["stop_floor_pct"], cp["tp_rr"]
    entry, stop, tp = table
    assert entry == 80000.0
    assert stop == pytest.approx(max(80000 * (1 - floor), 90000 * (1 - floor)))
    assert stop == pytest.approx(90000 * (1 - floor))   # the trailing ratchet applied
    assert tp == pytest.approx(80000 * (1 + rr * floor))


def test_stock_symbols_unchanged(exit_ns):
    raw = exit_ns["_exit_levels_raw"]
    sp = exit_ns["STOCK_POLICY"]
    entry, stop, tp = raw(None, "AAPL", 200.0, 210.0, {})
    assert stop == pytest.approx(200 * (1 - sp["stop_floor_pct"]))
    assert tp == pytest.approx(200 * (1 + sp["tp_rr"] * sp["stop_floor_pct"]))
    # a stock row never borrows a crypto state
    e2, s2, _ = raw(None, "AAPL", 200.0, 210.0,
                    {"BTC/USD": {"hwm": 1e9, "trailing": True}})
    assert s2 == pytest.approx(stop)


# ---------------------------------------------------------------------------
# G8-3 — fills on canceled / expired / partially_filled orders count
# ---------------------------------------------------------------------------
def _o(sym, side, qty, px, when, status="filled", filled_qty=None):
    return {"id": f"{sym}{side}{when}", "symbol": sym, "side": side,
            "qty": str(qty), "type": "limit", "status": status,
            "submitted_at": when, "filled_at": when,
            "filled_avg_price": None if px is None else str(px),
            "notional": None,
            "filled_qty": str(qty if filled_qty is None else filled_qty)}


def test_order_has_fill_predicate():
    assert order_has_fill(_o("A", "buy", 1, 10, "t"))
    for st in ("canceled", "expired", "partially_filled", "done_for_day"):
        assert order_has_fill(_o("A", "buy", 5, 10, "t", status=st, filled_qty=2))
        assert not order_has_fill(_o("A", "buy", 5, 10, "t", status=st, filled_qty=0))
        no_fq = _o("A", "buy", 5, 10, "t", status=st)
        no_fq["filled_qty"] = None
        assert not order_has_fill(no_fq)
        assert not order_has_fill(_o("A", "buy", 5, None, "t", status=st, filled_qty=2))
    # statuses that cannot carry a real fill stay excluded (pinned by
    # tests/test_tax_lots.py::test_non_filled_orders_are_ignored)
    for st in ("new", "accepted", "pending_new", "rejected", "replaced"):
        assert not order_has_fill(_o("A", "buy", 5, 10, "t", status=st, filled_qty=5))
    assert not order_has_fill(_o("A", "buy", 5, 10, "t", status="canceled",
                                 filled_qty="garbage"))
    assert order_has_fill(types.SimpleNamespace(
        status="expired", filled_qty="0.5", filled_avg_price="100"))


def test_recent_fills_table_uses_the_shared_predicate():
    body = _source("_apply_trade_filter")
    assert "order_has_fill(o)" in body
    assert 'o["status"] == "filled"' not in body
    assert "from tax_lots import estimate_taxes, order_has_fill" in SRC


# ---------------------------------------------------------------------------
# G8-4 — bounded tail read, byte-identical buffer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tail_ns():
    ns = {"os": os, "LOG_BUFFER_MAXLEN": 200_000}
    return _load(["_trim_to_newline", "_decode_log_bytes", "_read_text_tail",
                  "_read_log_tail"], ns)


def _full(p):
    return p.read_text(encoding="utf-8", errors="replace")


# 4-byte, 3-byte, 2-byte chars + CRLF + a lone CR + an invalid byte, so a cut at
# ANY window offset lands inside a multibyte sequence for some pad value.
_CHUNK = ("\U0001F680 sizing — 0.25× cap → ok “q”\r\n"
          "\U0001F4C8\U0001F4C9\U0001F4B0éè\n").encode("utf-8") + b"bad\xff\xfe\rline\n"


@pytest.mark.parametrize("pad", range(8))
@pytest.mark.parametrize("n", [1, 7, 64, 333])
def test_read_text_tail_equals_read_text_suffix(tmp_path, tail_ns, pad, n):
    p = tmp_path / "bot.log"
    p.write_bytes(b"x" * pad + _CHUNK * 60)
    got = tail_ns["_read_text_tail"](p, n)
    full = _full(p)
    assert got[-n:] == full[-n:]
    assert (len(got) > n) == (len(full) > n)


def test_read_text_tail_worst_case_all_four_byte_chars(tmp_path, tail_ns):
    for pad in range(4):
        p = tmp_path / f"emoji{pad}.log"
        p.write_bytes(b"a" * pad + ("\U0001F600" * 50 + "\n").encode() * 40)
        for n in (1, 9, 50, 51, 52, 200):
            got = tail_ns["_read_text_tail"](p, n)
            assert got[-n:] == _full(p)[-n:], (pad, n)
            assert len(got) > n


def test_read_text_tail_small_file_is_whole_file(tmp_path, tail_ns):
    p = tmp_path / "s.log"
    p.write_bytes("a—b\r\nc\rd\n".encode())
    assert tail_ns["_read_text_tail"](p, 10_000) == _full(p)
    p.write_bytes(b"")
    assert tail_ns["_read_text_tail"](p, 10) == ""


def test_read_log_tail_byte_identical_on_a_large_multibyte_log(tmp_path, tail_ns):
    """> 800 KB (the 4*LOG_BUFFER_MAXLEN window) so the bounded path really
    seeks; compare against the old `_trim_to_newline(read_text())` exactly."""
    trim = tail_ns["_trim_to_newline"]
    for pad in range(4):
        p = tmp_path / f"big{pad}.log"
        p.write_bytes(b"z" * pad + _CHUNK * 13_000)
        assert p.stat().st_size > 4 * 200_000 + 8
        assert tail_ns["_read_log_tail"](p) == trim(_full(p))


def test_on_log_selected_uses_bounded_read():
    body = _source("_on_log_selected")
    assert "_read_log_tail(path)" in body
    assert "read_text(" not in body


# ---------------------------------------------------------------------------
# G8-5 — atomic tmp + os.replace for gui_settings.json / news_cache.json
# ---------------------------------------------------------------------------
@pytest.fixture
def settings_ns(tmp_path):
    ns = {"os": os, "json": json, "dt": dt, "Path": Path,
          "GUI_SETTINGS_FILE": tmp_path / "gui_settings.json",
          "NEWS_CACHE_FILE": tmp_path / "news_cache.json",
          "NEWS_CACHE_MAX_AGE_DAYS": 7}
    return _load(["_atomic_write_json", "_load_gui_settings", "_save_gui_settings",
                  "_load_news_cache", "_save_news_cache"], ns)


def _enospc_json(real):
    fake = types.SimpleNamespace(**{k: getattr(real, k) for k in dir(real)
                                    if not k.startswith("__")})

    def dump(obj, fp, **kw):   # the disk fills after the first 10 bytes
        fp.write(real.dumps(obj, **kw)[:10])
        fp.flush()
        raise OSError(errno.ENOSPC, "No space left on device")
    fake.dump = dump
    return fake


def test_settings_roundtrip_same_bytes_as_before(settings_ns):
    good = {"theme": "Paper", "cadences": {"orders": 60}, "ov_sma20": True}
    settings_ns["_save_gui_settings"](good)
    path = settings_ns["GUI_SETTINGS_FILE"]
    assert path.read_text() == json.dumps(good, indent=2)
    assert settings_ns["_load_gui_settings"]() == good
    assert sorted(x.name for x in path.parent.iterdir()) == ["gui_settings.json"]


def test_failed_settings_write_keeps_previous_file(settings_ns):
    good = {"theme": "Paper", "cadences": {"orders": 60, "stocks": 120},
            "chart_default_zoom": "3M"}
    settings_ns["_save_gui_settings"](good)
    settings_ns["json"] = _enospc_json(json)
    settings_ns["_save_gui_settings"](dict(good, ov_sma50=True))  # swallowed
    settings_ns["json"] = json
    assert settings_ns["_load_gui_settings"]() == good
    # no tmp residue left beside the file
    path = settings_ns["GUI_SETTINGS_FILE"]
    assert sorted(x.name for x in path.parent.iterdir()) == ["gui_settings.json"]


def test_failed_news_cache_write_keeps_previous_file(settings_ns):
    now = dt.datetime.now().timestamp()
    arts = [{"headline": "h", "datetime": now - 60, "_sentiment": 0.4}]
    settings_ns["_save_news_cache"](arts, {"value": 50})
    before = settings_ns["NEWS_CACHE_FILE"].read_text()
    settings_ns["json"] = _enospc_json(json)
    settings_ns["_save_news_cache"](arts * 3, {"value": 10})
    settings_ns["json"] = json
    assert settings_ns["NEWS_CACHE_FILE"].read_text() == before
    loaded = settings_ns["_load_news_cache"]()
    assert loaded["fng"] == {"value": 50} and len(loaded["articles"]) == 1


def test_atomic_writer_tmp_name_is_per_writer():
    body = _source("_atomic_write_json")
    assert "os.replace(tmp, path)" in body
    assert "os.getpid()" in body and "threading.get_ident()" in body
    for fn in ("_save_gui_settings", "_save_news_cache"):
        b = _source(fn)
        assert "_atomic_write_json(" in b and "open(" not in b


# ---------------------------------------------------------------------------
# G8-7 — LogTailer: byte range read + decode, no over-read / duplicate
# ---------------------------------------------------------------------------
class _StubSignal:
    def __init__(self, *types_):
        self.calls = []

    def emit(self, *a):
        self.calls.append(a)


def _tailer_ns(log_files):
    ns = {"QObject": object, "Signal": _StubSignal,
          "Slot": lambda *a, **k: (lambda f: f), "QTimer": None,
          "LOG_FILES": log_files}
    _load(["_utf8_complete_len", "_decode_log_bytes", "_read_log_increment",
           "LogTailer"], ns)
    return ns


class _RacyPath(type(pathlib.Path())):
    """A bot appends one more line right after the tailer's stat()."""
    armed = False

    def stat(self, *a, **k):
        st = super().stat(*a, **k)
        if _RacyPath.armed:
            _RacyPath.armed = False
            with open(self, "a", encoding="utf-8") as f:
                f.write("LINE-C written during the tick\n")
        return st


def test_logtailer_hunter_interleave_no_corrupted_line(tmp_path):
    p = _RacyPath(tmp_path, "crypto_bot_output.log")
    p.write_text("boot\n", encoding="utf-8")
    ns = _tailer_ns({"Crypto Bot": p})
    t = ns["LogTailer"]()
    t.new_lines = _StubSignal()
    t._positions = {"Crypto Bot": p.stat().st_size}
    with open(p, "a", encoding="utf-8") as f:
        f.write("LINE-A sizing — 0.25× cap → ok\n")
        f.write("LINE-B “quoted” ✓\n")
    _RacyPath.armed = True
    t.check_logs()      # tick 1: writer appends LINE-C between stat and read
    t.check_logs()      # tick 2
    shown = "".join(text for _, text in t.new_lines.calls).splitlines()
    assert shown == p.read_text(encoding="utf-8").splitlines()[1:]
    assert t._positions["Crypto Bot"] == p.stat().st_size


def test_logtailer_split_multibyte_char_is_held_back(tmp_path):
    p = tmp_path / "b.log"
    p.write_bytes(b"")
    ns = _tailer_ns({"B": p})
    t = ns["LogTailer"]()
    t.new_lines = _StubSignal()
    t._positions = {"B": 0}
    enc = "— dash \U0001F680\n".encode("utf-8")
    for cut in range(1, len(enc)):
        p.write_bytes(b"")
        t._positions = {"B": 0}
        t.new_lines.calls.clear()
        with open(p, "ab") as f:
            f.write(enc[:cut])
        t.check_logs()
        with open(p, "ab") as f:
            f.write(enc[cut:])
        t.check_logs()
        joined = "".join(text for _, text in t.new_lines.calls)
        # a whitespace-only final chunk ("\n") is not emitted (text.strip()
        # guard, unchanged) — compare the visible text
        assert joined.rstrip("\n") == enc.decode("utf-8").rstrip("\n"), cut
        assert "�" not in joined


def test_logtailer_truncation_resets_and_invalid_bytes_replace(tmp_path):
    p = tmp_path / "c.log"
    p.write_bytes(b"x" * 100)
    ns = _tailer_ns({"C": p})
    t = ns["LogTailer"]()
    t.new_lines = _StubSignal()
    t._positions = {"C": 100}
    p.write_bytes(b"new \xff line\r\n")          # rotated/truncated
    t.check_logs()
    assert t.new_lines.calls == [("C", "new � line\n")]
    assert t._positions["C"] == p.stat().st_size


@pytest.mark.parametrize("data,keep", [
    (b"", 0), (b"abc", 3), ("—".encode(), 3), ("—".encode()[:1], 0),
    ("—".encode()[:2], 0), (b"ab" + "\U0001F680".encode()[:3], 2),
    (b"ab" + "é".encode()[:1], 2), (b"\x80\x80", 2), (b"a\xff", 2),
    (b"a\xc0", 2),
])
def test_utf8_complete_len(data, keep):
    ns = _load(["_utf8_complete_len"], {})
    assert ns["_utf8_complete_len"](data) == keep


def test_check_logs_source_is_byte_based():
    body = _source("check_logs")
    assert "_read_log_increment(path, last_pos, size)" in body
    assert 'open(path, "r"' not in body
