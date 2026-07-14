"""Review batch b18 — trade_journal.py, trade_memory.py, notify.py fixes.

Covers: journal ts stamped offset-aware from ONE clock read (decision_report's
pd.Timestamp and llm_eval's fromisoformat().timestamp() now agree on the
epoch) without mutating the caller's dict; log_decision never raises into a
trading loop (config loader, mkdir and stamping all guarded) and warns once
when journaling is disabled; get_journal_summary tolerates corrupt lines
per-line; a canonical iter_journal_rows reader. trade_memory: record_trade
never raises, corrupt files are quarantined (not silently wiped by the next
save), wrong-shape/non-UTF8 files handled, unique-per-writer tmp names +
thread/flock locking so concurrent writers lose nothing, dead
get_relevant_history removed, public load_all(). notify: _send/poll failures
visible (warning, with continuous-failure escalation for the kill switch),
notify() gates on a COMPLETE telegram channel (token+chat), dedupe/prune and
heartbeat paths unit-tested, urlopen responses context-managed."""

import datetime
import inspect
import io
import json
import re
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

import notify
import trade_journal
import trade_memory


class _RecordingLogger:
    def __init__(self):
        self.warnings = []
        self.debugs = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def debug(self, msg, *args):
        self.debugs.append(msg % args if args else msg)


class _SyncThread:
    """threading.Thread stand-in that runs the target synchronously."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None,
                 name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target:
            self._target(*self._args, **self._kwargs)


@pytest.fixture
def journal_dir(tmp_path, monkeypatch):
    jdir = tmp_path / "journals"
    jdir.mkdir()
    monkeypatch.setattr(trade_journal, "JOURNAL_DIR", jdir)
    monkeypatch.setattr(trade_journal, "load_llm_config",
                        lambda: {"journal_enabled": True})
    return jdir


@pytest.fixture
def memory_file(tmp_path, monkeypatch):
    f = tmp_path / "trade_memory.json"
    monkeypatch.setattr(trade_memory, "_MEMORY_FILE", f)
    return f


# --- trade_journal: ts stamping ---

class TestJournalTsStamping:
    def _one_row(self, jdir, entry=None):
        trade_journal.log_decision(entry if entry is not None
                                   else {"action": "skip", "symbol": "X"})
        files = list(jdir.glob("*.jsonl"))
        assert len(files) == 1
        return files[0], json.loads(files[0].read_text().splitlines()[0])

    def test_ts_offset_aware_and_both_stage0_parsers_agree(self, journal_dir):
        _, row = self._one_row(journal_dir)
        ts = row["ts"]
        parsed = datetime.datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None            # offset written

        # decision_report path: tz-is-None branch must never trigger
        ts_dr = pd.Timestamp(ts)
        assert ts_dr.tz is not None
        epoch_dr = ts_dr.timestamp()
        # llm_eval path
        epoch_le = datetime.datetime.fromisoformat(ts).timestamp()
        # pre-fix these disagreed by the UTC offset (7h on a UTC-7 box)
        assert epoch_dr == pytest.approx(epoch_le, abs=1e-6)
        assert epoch_le == pytest.approx(time.time(), abs=60)

    def test_filename_and_ts_from_single_clock_read(self, journal_dir):
        filepath, row = self._one_row(journal_dir)
        ts_date = datetime.datetime.fromisoformat(row["ts"]).date()
        assert filepath.name == f"{ts_date.isoformat()}.jsonl"
        # the second clock read (date.today()) is gone from log_decision
        src = inspect.getsource(trade_journal.log_decision)
        assert "date.today()" not in src

    def test_caller_dict_not_mutated(self, journal_dir):
        entry = {"action": "buy", "symbol": "ETH/USD"}
        _, row = self._one_row(journal_dir, entry)
        assert "ts" not in entry                    # caller's dict untouched
        assert "ts" in row                          # but the row is stamped


# --- trade_journal: never raises into a trading loop ---

class TestJournalNeverRaises:
    def test_no_raise_when_config_loader_raises(self, journal_dir,
                                                monkeypatch):
        # pathological llm_config.json (JSON array top level) makes
        # load_llm_config raise TypeError — must not escape log_decision
        def boom():
            raise TypeError("list indices must be integers")
        monkeypatch.setattr(trade_journal, "load_llm_config", boom)
        rec = _RecordingLogger()
        monkeypatch.setattr(trade_journal, "logger", rec)
        trade_journal.log_decision({"action": "buy", "symbol": "TSLA"})
        assert not list(journal_dir.glob("*.jsonl"))
        assert len(rec.warnings) == 1

    def test_no_raise_when_journal_dir_is_a_regular_file(self, tmp_path,
                                                         monkeypatch):
        blocked = tmp_path / "journals"
        blocked.write_text("not a directory")
        monkeypatch.setattr(trade_journal, "JOURNAL_DIR", blocked)
        monkeypatch.setattr(trade_journal, "load_llm_config",
                            lambda: {"journal_enabled": True})
        rec = _RecordingLogger()
        monkeypatch.setattr(trade_journal, "logger", rec)
        trade_journal.log_decision({"action": "buy", "symbol": "TSLA"})
        assert len(rec.warnings) == 1

    def test_disabled_journal_warns_once_not_per_row(self, journal_dir,
                                                     monkeypatch):
        monkeypatch.setattr(trade_journal, "load_llm_config",
                            lambda: {"journal_enabled": False})
        monkeypatch.setattr(trade_journal, "_disabled_warned", False)
        rec = _RecordingLogger()
        monkeypatch.setattr(trade_journal, "logger", rec)
        trade_journal.log_decision({"action": "buy", "symbol": "A"})
        trade_journal.log_decision({"action": "sell", "symbol": "B"})
        assert not list(journal_dir.glob("*.jsonl"))    # rows dropped
        disabled = [w for w in rec.warnings if "disabled" in w]
        assert len(disabled) == 1                       # once per process


# --- trade_journal: corrupt-row tolerance ---

class TestJournalReaders:
    def test_summary_keeps_rows_after_a_corrupt_line(self, journal_dir):
        today = datetime.date.today().isoformat()
        (journal_dir / f"{today}.jsonl").write_text(
            json.dumps({"action": "buy", "symbol": "A"}) + "\n"
            + '{"action": "sell", "sym' + "\n"           # torn line
            + json.dumps({"action": "skip", "skip_reason": "llm_veto"}) + "\n")
        s = trade_journal.get_journal_summary(today)
        assert s["total"] == 2                           # both valid rows
        assert s["buys"] == 1 and s["skips"] == 1        # incl. post-corrupt
        assert s["llm_blocks"] == 1
        assert s["skipped_lines"] == 1

    def test_iter_journal_rows_window_order_and_corrupt_skip(self,
                                                             journal_dir):
        today = datetime.date.today()
        d1 = today - datetime.timedelta(days=1)
        d3 = today - datetime.timedelta(days=3)
        (journal_dir / f"{d3.isoformat()}.jsonl").write_text(
            json.dumps({"n": "too-old"}) + "\n")
        (journal_dir / f"{d1.isoformat()}.jsonl").write_text(
            json.dumps({"n": 1}) + "\nGARBAGE-LINE\n"
            + json.dumps({"n": 2}) + "\n")
        (journal_dir / f"{today.isoformat()}.jsonl").write_text(
            "\n" + json.dumps({"n": 3}) + "\n")          # blank line skipped
        rows = list(trade_journal.iter_journal_rows(days=2))
        assert [r["n"] for r in rows] == [1, 2, 3]       # oldest-first, no
        #                                                # d3, corrupt skipped


# --- trade_memory: robustness / never raises ---

class TestTradeMemoryRobustness:
    def test_round_trip_and_none_text_coercion(self, memory_file):
        # base_loop passes llm_info.get('r', '') which is None when the
        # stored analysis has an explicit null r — must not TypeError
        trade_memory.record_trade("BTC/USD", "sell", 100.0, 101.0, 1.0,
                                  reasoning=None, news_context=None,
                                  exit_reason="take_profit")
        data = trade_memory.load_all()
        assert list(data) == ["BTC/USD"]
        rec = data["BTC/USD"][0]
        assert rec["reasoning"] == "" and rec["news"] == ""
        assert rec["pnl_pct"] == 1.0 and rec["estimated"] is False
        assert datetime.datetime.fromisoformat(rec["ts"]).tzinfo is not None
        assert (memory_file.parent / (memory_file.name + ".lock")).exists()

    def test_corrupt_file_quarantined_not_silently_wiped(self, memory_file,
                                                         monkeypatch):
        rec = _RecordingLogger()
        monkeypatch.setattr(trade_memory, "logger", rec)
        memory_file.write_text('{"BTC/USD": [{"pnl_pct": 1.0}')  # truncated
        assert trade_memory._load() == {}
        corrupt = memory_file.parent / (memory_file.name + ".corrupt")
        assert corrupt.exists()                          # evidence preserved
        assert corrupt.read_text().startswith('{"BTC/USD"')
        assert not memory_file.exists()
        assert any("quarantin" in w for w in rec.warnings)
        # next record starts a fresh file; the old bytes are still around
        trade_memory.record_trade("ETH/USD", "sell", 10, 11, 10.0)
        assert list(trade_memory.load_all()) == ["ETH/USD"]
        assert corrupt.exists()

    def test_valid_json_wrong_shape_treated_as_corrupt(self, memory_file):
        memory_file.write_text("[1, 2, 3]")              # list top level
        assert trade_memory._load() == {}                # no AttributeError
        memory_file.write_text('{"BTC/USD": {"not": "a list"}}')
        assert trade_memory._load() == {}

    def test_non_utf8_bytes_do_not_raise(self, memory_file):
        memory_file.write_bytes(b"\xff\xfe\x00garbage")
        assert trade_memory._load() == {}                # no UnicodeDecodeError

    def test_record_trade_never_raises(self, memory_file):
        # non-JSON-serializable field -> _save TypeError, contained
        trade_memory.record_trade("A", "sell", 1.0, 1.0, 0.0,
                                  exit_reason=object())
        # non-numeric price -> round TypeError, contained
        trade_memory.record_trade("A", "sell", None, None, None)
        assert trade_memory.load_all() == {}             # nothing half-written
        assert not list(memory_file.parent.glob("*.tmp"))  # no stale tmp

    def test_concurrent_writers_lose_no_records(self, memory_file):
        # pre-fix: 2-thread stress lost ~50% to load-modify-save races and
        # shared-tmp stealing
        def writer(sym):
            for i in range(40):
                trade_memory.record_trade(sym, "sell", 100.0, 100.0 + i,
                                          float(i), exit_reason="tp")
        threads = [threading.Thread(target=writer, args=(s,))
                   for s in ("AAA", "BBB")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        data = trade_memory.load_all()
        assert len(data["AAA"]) == 40
        assert len(data["BBB"]) == 40

    def test_rolling_window_still_caps_at_50(self, memory_file):
        for i in range(55):
            trade_memory.record_trade("AAA", "sell", 1.0, 1.0, float(i))
        trades = trade_memory.load_all()["AAA"]
        assert len(trades) == trade_memory._MAX_PER_SYMBOL == 50
        assert trades[-1]["pnl_pct"] == 54.0             # newest kept

    def test_api_surface(self, memory_file):
        # dead code removed; the used API intact
        assert not hasattr(trade_memory, "get_relevant_history")
        trade_memory.record_trade("AAA", "sell", 1.0, 2.0, 5.0,
                                  exit_reason="take_profit")
        assert "1W/0L" in trade_memory.get_lesson_summary("AAA")
        assert trade_memory.get_lesson_summary("NOPE") == ""
        src = inspect.getsource(trade_memory._save)
        assert "os.getpid()" in src and "threading.get_ident()" in src


# --- notify: dedupe / channel gate ---

@pytest.fixture
def notify_sandbox(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "_send",
                        lambda message, level: sent.append((message, level)))
    monkeypatch.setattr(notify, "threading",
                        type("T", (), {"Thread": _SyncThread}))
    monkeypatch.setattr(notify, "_last_sent", {})
    for var in ("TRADER_WEBHOOK_URL", "TRADER_TELEGRAM_BOT_TOKEN",
                "TRADER_TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(var, raising=False)
    return sent


class TestNotifyDedupe:
    def test_dedupe_suppresses_repeat_key_within_window(self, notify_sandbox,
                                                        monkeypatch):
        monkeypatch.setenv("TRADER_WEBHOOK_URL", "http://hook")
        notify.notify("breaker tripped", dedupe_key="cb")
        notify.notify("breaker tripped", dedupe_key="cb")   # suppressed
        notify.notify("flatten failed", dedupe_key="fl")    # different key
        assert len(notify_sandbox) == 2
        assert notify_sandbox[0][0] == "breaker tripped"
        assert notify_sandbox[1][0] == "flatten failed"

    def test_prune_drops_expired_keys_past_200(self, notify_sandbox,
                                               monkeypatch):
        monkeypatch.setenv("TRADER_WEBHOOK_URL", "http://hook")
        now = time.monotonic()
        expired = {f"old-{i}": now - 700 for i in range(201)}
        monkeypatch.setattr(notify, "_last_sent", dict(expired))
        notify.notify("fresh", dedupe_key="fresh")
        assert len(notify_sandbox) == 1
        assert set(notify._last_sent) == {"fresh"}       # expired pruned

    def test_noop_without_any_channel(self, notify_sandbox):
        notify.notify("nobody home", dedupe_key="x")
        assert notify_sandbox == []
        assert notify._last_sent == {}                   # no dedupe entry

    def test_token_without_chat_is_not_a_channel(self, notify_sandbox,
                                                 monkeypatch):
        # pre-fix: dedupe entry recorded + a thread spawned that provably
        # sent nothing (both _send legs need token AND chat)
        monkeypatch.setenv("TRADER_TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setattr(notify, "_tg_misconfig_warned", False)
        rec = _RecordingLogger()
        monkeypatch.setattr(notify, "logger", rec)
        notify.notify("lost alert", dedupe_key="k")
        notify.notify("lost alert 2", dedupe_key="k2")
        assert notify_sandbox == []
        assert notify._last_sent == {}
        assert len(rec.warnings) == 1                    # warned once only

    def test_webhook_alone_still_sends(self, notify_sandbox, monkeypatch):
        monkeypatch.setenv("TRADER_WEBHOOK_URL", "http://hook")
        notify.notify("hi", dedupe_key="w")
        assert len(notify_sandbox) == 1


# --- notify: heartbeat ---

class TestHeartbeat:
    def _cleanup(self, name):
        p = REPO / f"{name}_heartbeat"
        if p.exists():
            p.unlink()

    def test_writes_file_and_rate_limits_per_name(self, monkeypatch):
        name = "b18hbtest"
        monkeypatch.setattr(notify, "_hb_last", {})
        monkeypatch.delenv("TRADER_HEALTHCHECK_URL", raising=False)
        monkeypatch.delenv(f"TRADER_HEALTHCHECK_URL_{name.upper()}",
                           raising=False)
        path = REPO / f"{name}_heartbeat"
        try:
            notify.ping_heartbeat(name)
            assert path.exists()
            float(path.read_text())                      # mtime payload
            path.write_text("sentinel")
            notify.ping_heartbeat(name)                  # <60s -> rate-limited
            assert path.read_text() == "sentinel"
            notify._hb_last[name] = -1e9                 # window expired
            notify.ping_heartbeat(name)
            assert path.read_text() != "sentinel"
        finally:
            self._cleanup(name)

    def test_per_name_url_beats_shared_url(self, monkeypatch):
        name = "b18hburl"
        pinged = []
        monkeypatch.setattr(notify, "_hb_last", {})
        monkeypatch.setattr(notify, "threading",
                            type("T", (), {"Thread": _SyncThread}))
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda url, timeout=10: pinged.append(url) or io.BytesIO(b""))
        monkeypatch.setenv(f"TRADER_HEALTHCHECK_URL_{name.upper()}",
                           "http://per-name")
        monkeypatch.setenv("TRADER_HEALTHCHECK_URL", "http://shared")
        try:
            notify.ping_heartbeat(name)
            assert pinged == ["http://per-name"]
        finally:
            self._cleanup(name)

    def test_outer_failure_logged_not_raised(self, monkeypatch):
        rec = _RecordingLogger()
        monkeypatch.setattr(notify, "logger", rec)
        monkeypatch.setattr(notify, "_hb_last", None)    # .get -> AttributeError
        notify.ping_heartbeat("b18hbboom")               # must not raise
        assert any("heartbeat failed" in d for d in rec.debugs)


# --- notify: kill-switch poll escalation ---

class TestTelegramPollEscalation:
    @pytest.fixture
    def poll_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADER_TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TRADER_TELEGRAM_CHAT_ID", "42")
        monkeypatch.setattr(notify, "_TG_OFFSET_FILE",
                            str(tmp_path / "off.json"))
        monkeypatch.setattr(notify, "_tg_fail_since", None)
        monkeypatch.setattr(notify, "_tg_last_warn", -1e9)
        rec = _RecordingLogger()
        monkeypatch.setattr(notify, "logger", rec)
        return rec

    def _boom(self, monkeypatch):
        def boom(req, timeout=10):
            raise OSError("401 unauthorized")
        monkeypatch.setattr("urllib.request.urlopen", boom)

    def test_first_failure_stays_debug(self, poll_env, monkeypatch):
        self._boom(monkeypatch)
        assert notify.poll_telegram_commands() == []
        assert poll_env.warnings == []                   # transient blip
        assert len(poll_env.debugs) == 1
        assert notify._tg_fail_since is not None         # tracking started

    def test_continuous_failure_warns_at_most_hourly(self, poll_env,
                                                     monkeypatch):
        self._boom(monkeypatch)
        # failing for >10 min already
        monkeypatch.setattr(notify, "_tg_fail_since",
                            time.monotonic() - 700)
        assert notify.poll_telegram_commands() == []     # never raises
        assert len(poll_env.warnings) == 1
        assert "kill switch" in poll_env.warnings[0]
        assert notify.poll_telegram_commands() == []
        assert len(poll_env.warnings) == 1               # hourly gate

    def test_success_resets_failure_tracking(self, poll_env, monkeypatch):
        monkeypatch.setattr(notify, "_tg_fail_since",
                            time.monotonic() - 700)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=10: io.BytesIO(b'{"result": []}'))
        assert notify.poll_telegram_commands() == []
        assert notify._tg_fail_since is None
        assert poll_env.warnings == []


# --- source guards ---

class TestSourceGuards:
    def test_notify_urlopen_responses_context_managed(self):
        src = (REPO / "notify.py").read_text()
        total = len(re.findall(r"urllib\.request\.urlopen\(", src))
        managed = len(re.findall(r"with urllib\.request\.urlopen\(", src))
        assert total == managed == 3                     # _post, _ping, poll

    def test_journal_writer_offset_aware_source(self):
        src = inspect.getsource(trade_journal.log_decision)
        assert ".astimezone()" in src
        # the caller's dict is copied, not stamped in place
        assert 'entry["ts"] =' not in src
