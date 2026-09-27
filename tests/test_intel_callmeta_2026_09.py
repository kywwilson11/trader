"""INTEL W19 (2026-09-27): llm_client.get_last_call_meta + the
_parse_response OverflowError crash-path fix.

TASK 1 (SCOUT_E A2, measurement-only): every completed HTTP attempt made by
call_gemini / call_claude / call_openai / call_llm (hence call_model and the
OpenAI-compatible endpoints) records a small dict in a THREAD-LOCAL slot —
{provider, model, finish_reason, block_reason, http_status, latency_ms,
attempt_index, fallback_used, ts} — exposed as a COPY by
get_last_call_meta(). llm_analyst's W15 `llm_call` row now populates
finish_reason / block_reason / http_status (hence `transport_discard`) from it.
Byte-pins: every wrapper / raw-parser return value is unchanged (goldens with
stubbed HTTP), and the W15 row key set is unchanged.

TASK 2 (class A, crash-path only): float() of a huge JSON integer literal
(10**400) raised OverflowError out of _parse_response and so out of
analyze_trades (breaking its fail-open contract). It now takes the existing
non-finite path: s -> 0.5 (flagged s_defaulted), p_up -> None. Finite inputs
are byte-identical.

All network I/O is faked (urllib.request.urlopen / call_model stubs); the cost
ledger is sandboxed (conftest autouse + the lc fixture); every journal write
goes to tmp_path. Zero network, zero LLM spend.
"""
import json
import threading
import urllib.error
import urllib.request

import pytest

llm_client = pytest.importorskip("llm_client")
llm_analyst = pytest.importorskip("llm_analyst")


META_KEYS = {"provider", "model", "finish_reason", "block_reason",
             "http_status", "latency_ms", "attempt_index", "fallback_used",
             "ts"}

# W15's `llm_call` row key set — must be unchanged by W19.
CALL_ROW_KEYS = {"ts", "action", "asset_type", "requested_model",
                 "model_used", "path", "outcome", "n_symbols_sent",
                 "n_symbols_returned", "latency_ms", "response_chars",
                 "max_tokens", "temperature", "prompt_sha256", "advisor_v2",
                 "dedup_hit", "fence_stripped", "n_s_defaulted",
                 "n_nonfinite", "n_out_of_range", "transport_errors",
                 "finish_reason", "block_reason", "usage_in", "usage_out",
                 "http_status", "cost_usd"}

HUGE = "1" + "0" * 400          # a JSON integer literal float() rejects


# --------------------------------------------------------------------------- #
# Fixtures / fakes
# --------------------------------------------------------------------------- #

def _cfg(gem_key="g", claude_key="c", openai_key="o", enabled=True):
    return {
        "provider": "auto", "selection_mode": "auto", "enabled": enabled,
        "models": {
            "gemini": {"api_key": gem_key, "model": "gemini-2.5-flash"},
            "claude": {"api_key": claude_key, "model": "claude-haiku-4-5"},
            "openai": {"api_key": openai_key, "model": "gpt-5.4-nano"},
        },
        "provider_preference": ["anthropic", "openai", "gemini"],
        "endpoints": [], "max_llm_latency_sec": 5, "pricing": {},
    }


@pytest.fixture()
def lc(monkeypatch, tmp_path):
    """llm_client with in-place state reset (never importlib.reload — same
    pattern as tests/test_llm_providers.py) and a clean call-meta slot."""
    mod = llm_client
    state = {"cfg": _cfg()}
    monkeypatch.setattr(mod, "_COST_FILE", str(tmp_path / "cost.json"))
    monkeypatch.setattr(mod, "save_llm_config", lambda cfg: None)
    monkeypatch.setattr(mod, "load_llm_config", lambda: state["cfg"])
    monkeypatch.setattr(mod, "_daily_cost", 0.0)
    monkeypatch.setattr(mod, "_cost_reset_date", "")
    monkeypatch.setattr(mod, "_quota_reset_date", "")
    monkeypatch.setattr(mod, "_detected_tier", None)
    monkeypatch.setattr(mod, "_last_model_used", None)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mod._model_calls.clear()
    mod._call_timestamps.clear()
    mod._429_cooldown_until.update({"gemini": 0.0, "anthropic": 0.0,
                                    "openai": 0.0})
    mod._meta_clear()
    monkeypatch.setattr(mod, "_w19_state", state, raising=False)
    yield mod
    mod._meta_clear()
    mod._model_calls.clear()
    mod._call_timestamps.clear()
    mod._429_cooldown_until.update({"gemini": 0.0, "anthropic": 0.0,
                                    "openai": 0.0})


class FakeResp:
    def __init__(self, payload, status=200):
        self._p = json.dumps(payload).encode()
        self.status = status

    def read(self):
        return self._p

    def getheader(self, name):
        return None


def _http_error(code, retry_after=None):
    hdrs = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return urllib.error.HTTPError("https://x", code, "err", hdrs, None)


def _serve(monkeypatch, *responses):
    """urlopen fake: each call pops the next item (a payload dict -> 200
    FakeResp, a FakeResp, or an exception instance to raise)."""
    queue = list(responses)
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.full_url)
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, FakeResp) else FakeResp(item)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


GEMINI_OK = {"candidates": [{"content": {"parts": [{"text": '{"s": 0.4}'}]},
                             "finishReason": "STOP"}],
             "usageMetadata": {"promptTokenCount": 10,
                               "candidatesTokenCount": 5}}
GEMINI_MAXTOK = {"candidates": [{"content": {"parts": [{"text": '{"s": 0.'}]},
                                 "finishReason": "MAX_TOKENS"}],
                 "usageMetadata": {"promptTokenCount": 10,
                                   "candidatesTokenCount": 4096}}
GEMINI_BLOCKED = {"promptFeedback": {"blockReason": "SAFETY"},
                  "usageMetadata": {"promptTokenCount": 10}}
CLAUDE_OK = {"content": [{"type": "text", "text": "hello"}],
             "usage": {"input_tokens": 10, "output_tokens": 2},
             "stop_reason": "end_turn"}
OPENAI_OK = {"choices": [{"message": {"content": "hi there"},
                          "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
OPENAI_LENGTH = {"choices": [{"message": {"content": '{"s": 0.'},
                              "finish_reason": "length"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 512}}


def _assert_meta(meta, **want):
    assert meta is not None
    assert set(meta) == META_KEYS
    for k, v in want.items():
        assert meta[k] == v, (k, meta[k], v)
    assert isinstance(meta["latency_ms"], int) and meta["latency_ms"] >= 0
    assert isinstance(meta["ts"], float)


# --------------------------------------------------------------------------- #
# TASK 1 — one record per provider parser, return values byte-pinned
# --------------------------------------------------------------------------- #

def test_gemini_success_meta_and_return_golden(lc, monkeypatch):
    _serve(monkeypatch, GEMINI_OK)
    out = lc.call_gemini("p", model="gemini-2.5-flash")
    assert out == '{"s": 0.4}'                          # golden
    assert lc.get_last_model_used() == "gemini-2.5-flash"
    _assert_meta(lc.get_last_call_meta(), provider="gemini",
                 model="gemini-2.5-flash", finish_reason="STOP",
                 block_reason=None, http_status=200, attempt_index=0,
                 fallback_used=False)


def test_gemini_max_tokens_meta_return_none(lc, monkeypatch):
    _serve(monkeypatch, GEMINI_MAXTOK)
    assert lc.call_gemini("p", model="gemini-2.5-flash") is None   # golden
    _assert_meta(lc.get_last_call_meta(), provider="gemini",
                 finish_reason="MAX_TOKENS", block_reason=None,
                 http_status=200)


def test_gemini_blocked_prompt_records_block_reason(lc, monkeypatch):
    _serve(monkeypatch, GEMINI_BLOCKED)
    assert lc.call_gemini("p", model="gemini-2.5-flash") is None   # golden
    _assert_meta(lc.get_last_call_meta(), provider="gemini",
                 finish_reason=None, block_reason="SAFETY", http_status=200)


def test_raw_gemini_parser_return_unchanged(lc, monkeypatch):
    _serve(monkeypatch, GEMINI_BLOCKED, GEMINI_OK)
    assert lc._call_gemini("p", "", "k", "gemini-2.5-flash", 64, 5) == (
        None, {"promptTokenCount": 10})
    assert lc._call_gemini("p", "", "k", "gemini-2.5-flash", 64, 5) == (
        '{"s": 0.4}', {"promptTokenCount": 10, "candidatesTokenCount": 5})


def test_claude_success_meta_and_return_golden(lc, monkeypatch):
    _serve(monkeypatch, CLAUDE_OK)
    assert lc.call_claude("p", model="claude-haiku-4-5") == "hello"
    _assert_meta(lc.get_last_call_meta(), provider="anthropic",
                 model="claude-haiku-4-5", finish_reason="end_turn",
                 block_reason=None, http_status=200, attempt_index=0,
                 fallback_used=False)


def test_claude_max_tokens_meta(lc, monkeypatch):
    _serve(monkeypatch, dict(CLAUDE_OK, stop_reason="max_tokens"))
    assert lc.call_claude("p", model="claude-haiku-4-5") is None
    _assert_meta(lc.get_last_call_meta(), provider="anthropic",
                 finish_reason="max_tokens")


def test_openai_length_truncation_meta(lc, monkeypatch):
    _serve(monkeypatch, OPENAI_LENGTH)
    assert lc.call_openai("p", model="gpt-5.4-nano") is None       # golden
    _assert_meta(lc.get_last_call_meta(), provider="openai",
                 model="gpt-5.4-nano", finish_reason="length",
                 block_reason=None, http_status=200, attempt_index=0)


def test_raw_openai_parser_return_unchanged(lc, monkeypatch):
    _serve(monkeypatch, OPENAI_OK, OPENAI_LENGTH, {"choices": []})
    usage = {"promptTokenCount": 10, "candidatesTokenCount": 3,
             "thoughtsTokenCount": 0}
    assert lc._call_openai("p", "", "k", "gpt-5.4-nano", 64, 5) == (
        "hi there", usage)
    assert lc._call_openai("p", "", "k", "gpt-5.4-nano", 64, 5) == (
        None, dict(usage, candidatesTokenCount=512))
    assert lc._call_openai("p", "", "k", "gpt-5.4-nano", 64, 5) == (
        None, {"promptTokenCount": 0, "candidatesTokenCount": 0,
               "thoughtsTokenCount": 0})


def test_http_error_records_status_no_reason(lc, monkeypatch):
    _serve(monkeypatch, _http_error(500))
    assert lc.call_openai("p", model="gpt-5.4-nano") is None
    _assert_meta(lc.get_last_call_meta(), provider="openai",
                 finish_reason=None, block_reason=None, http_status=500,
                 attempt_index=0)


def test_network_error_records_none_status(lc, monkeypatch):
    _serve(monkeypatch, urllib.error.URLError("timed out"))
    assert lc.call_gemini("p", model="gemini-2.5-flash") is None
    _assert_meta(lc.get_last_call_meta(), provider="gemini",
                 finish_reason=None, block_reason=None, http_status=None)


def test_429_resend_is_attempt_index_1(lc, monkeypatch):
    _serve(monkeypatch, _http_error(429, retry_after=1), CLAUDE_OK)
    assert lc.call_claude("p", model="claude-haiku-4-5") == "hello"
    _assert_meta(lc.get_last_call_meta(), provider="anthropic",
                 finish_reason="end_turn", http_status=200, attempt_index=1)


def test_gemini_429_resend_is_attempt_index_1(lc, monkeypatch):
    monkeypatch.setattr(lc, "_parse_retry_after", lambda e: 1.0)
    _serve(monkeypatch, _http_error(429), GEMINI_OK)
    assert lc.call_gemini("p", model="gemini-2.5-flash") == '{"s": 0.4}'
    _assert_meta(lc.get_last_call_meta(), provider="gemini",
                 finish_reason="STOP", attempt_index=1)


# --- call_llm: fallback_used / attempt_index / OpenAI-compatible -----------

def _chain(monkeypatch, lc, chain):
    monkeypatch.setattr(lc, "resolve_provider_chain",
                        lambda role, config: list(chain))


def test_call_llm_primary_success_not_fallback(lc, monkeypatch):
    _chain(monkeypatch, lc, [("anthropic", "claude-haiku-4-5", None, "c"),
                             ("openai", "gpt-5.4-nano", None, "o")])
    _serve(monkeypatch, CLAUDE_OK)
    assert lc.call_llm("p") == "hello"
    _assert_meta(lc.get_last_call_meta(), provider="anthropic",
                 attempt_index=0, fallback_used=False,
                 finish_reason="end_turn")


def test_call_llm_fallback_used_and_attempt_index(lc, monkeypatch):
    _chain(monkeypatch, lc, [("anthropic", "claude-haiku-4-5", None, "c"),
                             ("openai", "gpt-5.4-nano", None, "o")])
    _serve(monkeypatch, _http_error(500), OPENAI_OK)
    assert lc.call_llm("p") == "hi there"                          # golden
    assert lc.get_last_model_used() == "gpt-5.4-nano"
    _assert_meta(lc.get_last_call_meta(), provider="openai",
                 model="gpt-5.4-nano", attempt_index=1, fallback_used=True,
                 finish_reason="stop", http_status=200)


def test_call_llm_skipped_entries_are_not_attempts(lc, monkeypatch):
    # keyless native provider is skipped (no attempt) -> the answering entry
    # is attempt 0 but still a fallback (chain position 1)
    _chain(monkeypatch, lc, [("anthropic", "claude-haiku-4-5", None, ""),
                             ("openai", "gpt-5.4-nano", None, "o")])
    _serve(monkeypatch, OPENAI_OK)
    assert lc.call_llm("p") == "hi there"
    _assert_meta(lc.get_last_call_meta(), attempt_index=0,
                 fallback_used=True)


def test_call_llm_openai_compatible_endpoint(lc, monkeypatch):
    _chain(monkeypatch, lc, [("ollama", "llama3",
                              "http://localhost:11434/v1", "")])
    seen = _serve(monkeypatch, OPENAI_LENGTH)
    assert lc.call_llm("p") is None                                # golden
    assert seen == ["http://localhost:11434/v1/chat/completions"]
    _assert_meta(lc.get_last_call_meta(), provider="ollama", model="llama3",
                 finish_reason="length", attempt_index=0,
                 fallback_used=False)


def test_call_llm_all_fail_keeps_last_attempt(lc, monkeypatch):
    _chain(monkeypatch, lc, [("anthropic", "claude-haiku-4-5", None, "c"),
                             ("gemini", "gemini-2.5-flash", None, "g")])
    _serve(monkeypatch, _http_error(503), GEMINI_BLOCKED)
    assert lc.call_llm("p") is None
    _assert_meta(lc.get_last_call_meta(), provider="gemini",
                 block_reason="SAFETY", attempt_index=1, fallback_used=True)


# --- reset / copy / thread isolation ---------------------------------------

def test_attemptless_call_resets_to_none(lc, monkeypatch):
    _serve(monkeypatch, OPENAI_OK)
    assert lc.call_openai("p", model="gpt-5.4-nano") == "hi there"
    assert lc.get_last_call_meta() is not None
    lc._w19_state["cfg"] = _cfg(enabled=False)
    assert lc.call_openai("p", model="gpt-5.4-nano") is None
    assert lc.get_last_call_meta() is None


def test_call_model_routes_and_records(lc, monkeypatch):
    _serve(monkeypatch, CLAUDE_OK)
    assert lc.call_model("p", model="claude-haiku-4-5") == "hello"
    assert lc.get_last_call_meta()["provider"] == "anthropic"


def test_accessor_returns_a_copy(lc, monkeypatch):
    _serve(monkeypatch, GEMINI_OK)
    lc.call_gemini("p", model="gemini-2.5-flash")
    m1 = lc.get_last_call_meta()
    m1["finish_reason"] = "TAMPERED"
    m1["extra"] = 1
    m2 = lc.get_last_call_meta()
    assert m2["finish_reason"] == "STOP" and "extra" not in m2
    assert m1 is not m2


def test_meta_is_thread_local(lc, monkeypatch):
    _serve(monkeypatch, GEMINI_OK, OPENAI_LENGTH)
    lc.call_gemini("p", model="gemini-2.5-flash")
    main_before = lc.get_last_call_meta()
    seen = {}

    def worker():
        seen["initial"] = lc.get_last_call_meta()
        lc.call_openai("p", model="gpt-5.4-nano")
        seen["after"] = lc.get_last_call_meta()

    t = threading.Thread(target=worker)
    t.start()
    t.join(10)
    assert seen["initial"] is None                  # nothing leaks in
    assert seen["after"]["provider"] == "openai"
    assert seen["after"]["finish_reason"] == "length"
    assert lc.get_last_call_meta() == main_before   # nothing leaks out


def test_meta_is_json_safe(lc, monkeypatch):
    _serve(monkeypatch, GEMINI_BLOCKED)
    lc.call_gemini("p", model="gemini-2.5-flash")
    json.dumps(lc.get_last_call_meta(), allow_nan=False)


def test_note_transport_ignores_non_native_types(lc):
    lc._meta_begin()

    class Weird:
        status = "200"      # not an int -> dropped

    lc._note_transport(finish_reason=123, block_reason="", resp=Weird())
    lc._meta_end("gemini", "m", lc.time.time(), 0)
    m = lc.get_last_call_meta()
    assert m["finish_reason"] is None and m["block_reason"] is None
    assert m["http_status"] is None


# --------------------------------------------------------------------------- #
# TASK 2 — _parse_response OverflowError -> existing non-finite path
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tok", [HUGE, "-" + HUGE])
def test_parse_response_huge_int_score_is_neutral(tok):
    out = llm_analyst._parse_response(
        '{"AAA": {"s": %s, "r": "x"}, "BBB": {"s": 0.8}}' % tok,
        ["AAA", "BBB"])
    assert out["AAA"]["s"] == 0.5 and out["AAA"]["m"] == 0.75
    assert out["BBB"]["s"] == 0.8
    # identical to the missing / NaN handling (G4-07 contract)
    nan = llm_analyst._parse_response('{"AAA": {"s": NaN, "r": "x"}}',
                                      ["AAA"])
    assert llm_analyst._parse_response(
        '{"AAA": {"s": %s, "r": "x"}}' % tok, ["AAA"]) == nan


def test_parse_response_huge_int_extended_fields():
    out = llm_analyst._parse_response(
        '{"AAA": {"s": 0.6, "p_up": %s, "conviction": %s}}' % (HUGE, HUGE),
        ["AAA"], extended=True)
    assert out["AAA"]["s"] == 0.6
    assert out["AAA"]["p_up"] is None            # non-finite -> missing
    assert out["AAA"]["conviction"] == 5         # int() never overflowed: unchanged


def test_parse_response_finite_goldens_unchanged():
    out = llm_analyst._parse_response(
        '{"A": {"s": 1.7}, "B": {"s": -0.2}, "C": {"s": "0.3"}, '
        '"D": {"s": 7}, "E": {"s": 0}}', list("ABCDE"))
    assert [out[k]["s"] for k in "ABCDE"] == [1.0, 0.0, 0.3, 1.0, 0.0]
    ext = llm_analyst._parse_response(
        '{"A": {"s": 0.6, "p_up": 1.4, "conviction": 9}, '
        '"B": {"s": 0.2, "p_up": 0.25, "conviction": "3"}}', ["A", "B"],
        extended=True)
    assert (ext["A"]["p_up"], ext["A"]["conviction"]) == (1.0, 5)
    assert (ext["B"]["p_up"], ext["B"]["conviction"]) == (0.25, 3)


def test_parse_diagnostics_huge_int_flags_defaulted():
    d = llm_analyst._parse_diagnostics('{"AAA": {"s": %s}}' % HUGE, ["AAA"])
    fl = d["parse_flags"]["AAA"]
    assert fl["s_defaulted"] is True and fl["s_nonfinite"] is True
    assert fl["s_clamped"] is False


# --- end-to-end through analyze_trades (journal to tmp_path) ---------------

@pytest.fixture
def env(monkeypatch, tmp_path):
    cfg = {"enabled": True, "advisor_v2_enabled": False,
           "analyst_dedup_ttl_sec": 0, "replay_capture_enabled": True}
    monkeypatch.setattr(llm_analyst, "load_llm_config", lambda: cfg)
    monkeypatch.setattr(llm_analyst, "_ANALYSIS_FILE",
                        tmp_path / "llm_analysis.json")
    monkeypatch.setattr(llm_analyst, "_REPLAY_DIR", tmp_path / "llm_replay")
    monkeypatch.setattr(llm_analyst, "get_recommended_model",
                        lambda role: "gemini-2.5-flash")
    monkeypatch.setattr(llm_analyst, "_LAST_CALL_META", {})
    monkeypatch.setattr(llm_client, "get_routing_info",
                        lambda: {"daily_cost": 0.0})
    llm_analyst._DEDUP_CACHE.clear()
    llm_client._meta_clear()
    yield {"call_dir": tmp_path / "llm_calls",
           "replay_dir": tmp_path / "llm_replay"}
    llm_analyst._DEDUP_CACHE.clear()
    llm_client._meta_clear()


def _rows(d):
    out = []
    if d.exists():
        for f in sorted(d.glob("*.jsonl")):
            out += [json.loads(x) for x in f.read_text().splitlines()
                    if x.strip()]
    return out


def _cands(*syms):
    return [{"symbol": s, "pred_return": 0.33, "fundamentals_text": "P/E=20",
             "news_headlines": ["hi"]} for s in syms]


@pytest.mark.parametrize("syms,outcome", [(("AAA",), "ok"),
                                          (("AAA", "BBB"), "partial")])
def test_analyze_trades_huge_int_does_not_raise_and_journals(
        env, monkeypatch, syms, outcome):
    monkeypatch.setattr(llm_analyst, "call_model",
                        lambda *a, **k: '{"AAA": {"s": %s, "r": "r", '
                                        '"bull": "b", "bear": "be"}}' % HUGE)
    monkeypatch.setattr(llm_analyst, "call_llm", lambda *a, **k: None)
    monkeypatch.setattr(llm_analyst, "get_last_model_used",
                        lambda: "gemini-2.5-flash")
    out = llm_analyst.analyze_trades(_cands(*syms), "stock")
    assert out == {"AAA": {"m": 0.75, "s": 0.5, "r": "r", "bull": "b",
                           "bear": "be"}}
    row, = _rows(env["call_dir"])
    assert set(row) == CALL_ROW_KEYS
    assert row["outcome"] == outcome
    assert row["n_s_defaulted"] == 1 and row["n_nonfinite"] == 1
    # per-symbol flags live on the replay record (the call row keeps counts)
    rec, = _rows(env["replay_dir"])
    assert rec["parse_flags"]["AAA"]["s_defaulted"] is True
    assert rec["parse_flags"]["AAA"]["s_nonfinite"] is True
    assert (llm_analyst.get_last_analysis_meta()["parse_flags"]["AAA"]
            ["s_defaulted"] is True)


def test_analyze_trades_huge_int_persist_false_does_not_raise(
        env, monkeypatch):
    monkeypatch.setattr(llm_analyst, "call_model",
                        lambda *a, **k: '{"AAA": {"s": %s}}' % HUGE)
    monkeypatch.setattr(llm_analyst, "call_llm", lambda *a, **k: None)
    out = llm_analyst.analyze_trades(_cands("AAA"), "stock", persist=False)
    assert out["AAA"]["s"] == 0.5
    assert _rows(env["call_dir"]) == []


# --- W15 row wiring: real call_model -> stubbed HTTP -> populated fields ---

def test_row_transport_discard_from_real_client(env, lc, monkeypatch):
    _serve(monkeypatch, GEMINI_MAXTOK)
    monkeypatch.setattr(llm_analyst, "call_llm", lambda *a, **k: None)
    assert llm_analyst.analyze_trades(_cands("AAA"), "stock") == {}
    row, = _rows(env["call_dir"])
    assert set(row) == CALL_ROW_KEYS
    assert row["outcome"] == "transport_discard"
    assert row["finish_reason"] == "MAX_TOKENS"
    assert row["block_reason"] is None and row["http_status"] == 200
    assert row["usage_in"] is None and row["usage_out"] is None
    assert row["path"] == "call_llm" and row["transport_errors"] == []


def test_row_blocked_prompt_from_real_client(env, lc, monkeypatch):
    _serve(monkeypatch, GEMINI_BLOCKED)
    monkeypatch.setattr(llm_analyst, "call_llm", lambda *a, **k: None)
    assert llm_analyst.analyze_trades(_cands("AAA"), "stock") == {}
    row, = _rows(env["call_dir"])
    assert row["outcome"] == "transport_discard"
    assert row["block_reason"] == "SAFETY" and row["finish_reason"] is None


def test_row_success_from_real_client_return_golden(env, lc, monkeypatch):
    _serve(monkeypatch, {"candidates": [{"content": {"parts": [{"text": json.dumps(
        {"AAA": {"s": 0.72, "r": "r", "bull": "b", "bear": "be"}})}]},
        "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 1}})
    monkeypatch.setattr(llm_analyst, "call_llm", lambda *a, **k: None)
    out = llm_analyst.analyze_trades(_cands("AAA"), "stock")
    assert out == {"AAA": {"m": round(0.72 * 1.5, 2), "s": 0.72, "r": "r",
                           "bull": "b", "bear": "be"}}
    row, = _rows(env["call_dir"])
    assert set(row) == CALL_ROW_KEYS
    assert row["outcome"] == "ok" and row["path"] == "call_model"
    assert row["finish_reason"] == "STOP" and row["http_status"] == 200
    assert row["model_used"] == "gemini-2.5-flash"


def test_row_http_error_status_from_real_client(env, lc, monkeypatch):
    _serve(monkeypatch, _http_error(500))
    monkeypatch.setattr(llm_analyst, "call_llm", lambda *a, **k: None)
    assert llm_analyst.analyze_trades(_cands("AAA"), "stock") == {}
    row, = _rows(env["call_dir"])
    assert row["outcome"] == "empty"            # no finish/block reason
    assert row["http_status"] == 500
