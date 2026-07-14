"""Review-batch b06 tests — fundamentals.py + edgar_events.py fixes.

Covers:
  - all urllib urlopen calls are context-managed (socket hygiene)
  - fundamentals docstrings state the SEC filing path's runtime reality
  - corrupt-but-fresh edgar_tickers.json falls through to refetch
    (JSONDecodeError no longer escapes to the fail-open handler)
  - ticker map memoized in-process on file mtime
  - fail-open path emits a rate-limited WARNING (once/hour)
  - missing-CIK symbols are logged instead of silently skipped
  - atomic writers use pid-unique tmp names (bot + manual CLI can't
    interleave into one tmp file)

Pure stdlib — runs on the dev Mac (fundamentals/edgar_events import only
json/urllib/log_config/llm_config at module level).
"""

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import edgar_events as ee
import fundamentals as fnd


class _FakeResp:
    """Context-manager-only HTTP response stand-in: read() works, and the
    test can assert the code actually entered/exited the with-block."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def read(self):
        return self._payload


class _LogRecorder:
    def __init__(self):
        self.calls = {"debug": [], "info": [], "warning": []}

    def debug(self, msg, *args):
        self.calls["debug"].append(msg % args if args else msg)

    def info(self, msg, *args):
        self.calls["info"].append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.calls["warning"].append(msg % args if args else msg)


# --- urlopen hygiene -------------------------------------------------------

def test_all_urlopen_calls_context_managed():
    for name in ("fundamentals.py", "edgar_events.py"):
        src = (ROOT / name).read_text()
        n_calls = src.count("urlopen(")
        n_with = src.count("with urllib.request.urlopen(")
        assert n_calls > 0, name
        assert n_calls == n_with, f"{name}: bare urlopen() without `with`"


def test_fmp_fetch_context_manages_response(monkeypatch):
    fnd._cache.pop("fmp_TESTX", None)
    fake = _FakeResp(json.dumps([{"peRatio": 12.5, "pbRatio": 2.0}]).encode())
    monkeypatch.setattr(fnd, "load_llm_config", lambda: {"fmp_api_key": "k"})
    monkeypatch.setattr(fnd.urllib.request, "urlopen",
                        lambda req, timeout=None: fake)
    out = fnd._fetch_fmp_metrics("TESTX")
    assert fake.entered and fake.exited   # response closed via with-block
    assert out["pe_ratio"] == 12.5


def test_get_sec_filings_context_manages_and_field_reality(monkeypatch):
    fnd._cache.pop("sec_TESTY", None)
    payload = {"hits": {"hits": [{
        "_id": "0000320193-26-000011:aapl-20260430.htm",
        "_source": {"file_type": "8-K", "file_date": "2026-04-30",
                    "display_names": ["Apple Inc.  (AAPL)"],
                    "ciks": ["0000320193"]},
    }]}}
    fake = _FakeResp(json.dumps(payload).encode())
    monkeypatch.setattr(fnd.urllib.request, "urlopen",
                        lambda req, timeout=None: fake)
    out = fnd.get_sec_filings("TESTY")
    assert fake.entered and fake.exited
    assert len(out) == 1
    assert out[0]["filed_date"] == "2026-04-30"
    assert out[0]["cik"] == "0000320193"
    # Locks in documented current behavior: EFTS responses carry the form
    # under 'file_type' (there is no 'form_type' field), so form_type comes
    # through "". Update together with the EFTS repair AND the
    # get_sec_filings/get_filing_summary docstrings.
    assert out[0]["form_type"] == ""


# --- fundamentals doc reality ----------------------------------------------

def test_fundamentals_docs_state_runtime_reality():
    assert "INOPERATIVE" in fnd.__doc__
    d = fnd.get_sec_filings.__doc__.lower()
    assert "relevance" in d and "inoperative" in d
    assert "RUNTIME REALITY" in fnd.get_filing_summary.__doc__
    src = (ROOT / "fundamentals.py").read_text()
    assert "# Find most recent 10-K or 10-Q" not in src  # false comment gone


# --- edgar_events: corrupt ticker map heals ---------------------------------

def test_corrupt_fresh_ticker_map_refetches(tmp_path, monkeypatch):
    map_file = tmp_path / "edgar_tickers.json"
    map_file.write_text("{corrupt json!")   # fresh mtime, invalid JSON
    monkeypatch.setattr(ee, "_TICKER_MAP_FILE", map_file)
    monkeypatch.setattr(ee, "_map_memo", None)
    monkeypatch.setattr(
        ee, "_get_json",
        lambda url, timeout=5: {"0": {"ticker": "nvda", "cik_str": 1045810}})
    replaced = []
    real_replace = os.replace

    def rec_replace(src, dst):
        replaced.append(str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(ee.os, "replace", rec_replace)
    out = ee._ticker_cik_map()
    assert out == {"NVDA": "0001045810"}
    # corrupt file healed via refetch (pre-fix: JSONDecodeError escaped and
    # the veto failed open for a week without ever refetching)
    assert json.loads(map_file.read_text()) == out
    assert replaced and replaced[0].endswith(f".{os.getpid()}.tmp")


def test_ticker_map_memoized_on_mtime(tmp_path, monkeypatch):
    map_file = tmp_path / "edgar_tickers.json"
    map_file.write_text(json.dumps({"NVDA": "0001045810"}))
    monkeypatch.setattr(ee, "_TICKER_MAP_FILE", map_file)
    monkeypatch.setattr(ee, "_map_memo", None)

    def no_network(url, timeout=5):
        raise AssertionError("network hit — disk cache/memo bypassed")

    monkeypatch.setattr(ee, "_get_json", no_network)
    first = ee._ticker_cik_map()
    assert first == {"NVDA": "0001045810"}
    # Corrupt the bytes but restore the exact mtime: a memo hit must return
    # the already-parsed map without re-reading the (now corrupt) file
    st = map_file.stat()
    map_file.write_text("{corrupt")
    os.utime(map_file, ns=(st.st_atime_ns, st.st_mtime_ns))
    second = ee._ticker_cik_map()
    assert second is first


# --- edgar_events: fail-open visibility -------------------------------------

def test_fail_open_warns_rate_limited(tmp_path, monkeypatch):
    monkeypatch.setattr(ee, "_CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(ee, "_last_fail_open_warn", -float("inf"))
    rec = _LogRecorder()
    monkeypatch.setattr(ee, "logger", rec)

    def boom():
        raise RuntimeError("edgar down")

    monkeypatch.setattr(ee, "_ticker_cik_map", boom)
    assert ee.entry_blocked("NVDA") == (False, None)
    assert ee.entry_blocked("NVDA") == (False, None)
    assert len(rec.calls["warning"]) == 1        # second call rate-limited
    assert len(rec.calls["debug"]) == 2          # per-call debug kept
    assert "failing open" in rec.calls["warning"][0]
    # once the interval has elapsed the warning fires again
    monkeypatch.setattr(ee, "_last_fail_open_warn",
                        time.monotonic() - ee._FAIL_OPEN_WARN_SEC - 1)
    assert ee.entry_blocked("NVDA") == (False, None)
    assert len(rec.calls["warning"]) == 2


def test_missing_cik_logged_and_cached(tmp_path, monkeypatch):
    cache_file = tmp_path / "cache.json"
    monkeypatch.setattr(ee, "_CACHE_FILE", cache_file)
    rec = _LogRecorder()
    monkeypatch.setattr(ee, "logger", rec)
    monkeypatch.setattr(ee, "_ticker_cik_map", lambda: {})
    assert ee.entry_blocked("ARKK") == (False, None)
    assert any("no CIK" in m for m in rec.calls["debug"])
    assert not rec.calls["warning"]              # not an error condition
    saved = json.loads(cache_file.read_text())
    assert saved["ARKK"]["blocked"] is False
    assert saved["ARKK"]["date"] == dt.date.today().isoformat()


# --- edgar_events: pid-unique atomic writes ---------------------------------

def test_save_cache_pid_unique_tmp(tmp_path, monkeypatch):
    cache_file = tmp_path / "edgar_cache.json"
    monkeypatch.setattr(ee, "_CACHE_FILE", cache_file)
    replaced = []
    real_replace = os.replace

    def rec_replace(src, dst):
        replaced.append(str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(ee.os, "replace", rec_replace)
    ee._save_cache({"NVDA": {"date": "2026-07-02", "blocked": False,
                             "reason": None}})
    assert replaced and replaced[0].endswith(f".{os.getpid()}.tmp")
    assert json.loads(cache_file.read_text())["NVDA"]["blocked"] is False
    assert not list(tmp_path.glob("*.tmp"))      # tmp consumed by replace


# --- edgar_events: docstring matches MA_FORMS -------------------------------

def test_edgar_docstring_covers_ma_forms_and_acquirer_note():
    doc = ee.__doc__
    for form in ee.MA_FORMS:
        assert form in doc, f"MA form {form} missing from module docstring"
    assert "acquirer" in doc.lower()
