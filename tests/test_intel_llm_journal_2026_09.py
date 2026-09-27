"""INTEL W15 (2026-09-27): llm_analyst attempt journal — SCOUT_E spec A1-A4.

Pins the INTEL half of the failed-LLM-call journal fields:
  * A1  every non-dedup analyze_trades attempt (success OR failure) appends
        one `llm_call` row to journals/llm_calls/YYYY-MM-DD.jsonl (sibling of
        the replay dir), with outcome / path / n_symbols_* / latency_ms /
        prompt_sha256 / counters. A failed call used to leave NO trace.
  * A2  finish_reason / block_reason / http_status come from the READ-ONLY
        llm_client.get_last_call_meta() (INTEL W19; usage_* stay None); the
        transport_discard outcome engages only when it reports a reason.
  * A3  per-symbol parse flags (raw_s, s_defaulted, s_nonfinite, s_clamped,
        extended p_up/conviction flags) on the replay record, and via
        get_last_analysis_meta()['parse_flags'] for base_loop (ENGINE half).
  * A4  replay record gains prompt_sha256 / latency_ms / dedup_hit.
Invariants: the legacy replay-record keys/values and analyze_trades' return
value are byte-identical; persist=False and replay_capture_enabled=False
write nothing; the writer never raises.

Every write is redirected to tmp_path (_REPLAY_DIR / _ANALYSIS_FILE), the
transport is stubbed, and llm_client.get_routing_info is stubbed so the cost
ledger is never touched. Zero network, zero LLM spend.
"""

import hashlib
import json
import math
import re

import pytest

llm_analyst = pytest.importorskip("llm_analyst")
llm_client = pytest.importorskip("llm_client")

LEGACY_REPLAY_KEYS = ["ts", "asset_type", "forward_bars", "equity",
                      "positions", "position_details", "fng", "candidates",
                      "live_scores", "live_model"]
NEW_REPLAY_KEYS = {"prompt_sha256", "latency_ms", "dedup_hit",
                   "fence_stripped", "parse_flags"}
CALL_ROW_KEYS = {"ts", "action", "asset_type", "requested_model",
                 "model_used", "path", "outcome", "n_symbols_sent",
                 "n_symbols_returned", "latency_ms", "response_chars",
                 "max_tokens", "temperature", "prompt_sha256", "advisor_v2",
                 "dedup_hit", "fence_stripped", "n_s_defaulted",
                 "n_nonfinite", "n_out_of_range", "transport_errors",
                 "finish_reason", "block_reason", "usage_in", "usage_out",
                 "http_status", "cost_usd"}


def _cfg(**over):
    cfg = {"enabled": True, "advisor_v2_enabled": False,
           "analyst_dedup_ttl_sec": 0, "replay_capture_enabled": True}
    cfg.update(over)
    return cfg


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated analyze_trades: tmp dirs, stubbed routing/ledger/model."""
    state = {"cfg": _cfg(), "calls": []}
    monkeypatch.setattr(llm_analyst, "load_llm_config", lambda: state["cfg"])
    monkeypatch.setattr(llm_analyst, "_ANALYSIS_FILE",
                        tmp_path / "llm_analysis.json")
    monkeypatch.setattr(llm_analyst, "_REPLAY_DIR", tmp_path / "llm_replay")
    monkeypatch.setattr(llm_analyst, "get_recommended_model",
                        lambda role: "model-x")
    monkeypatch.setattr(llm_analyst, "get_last_model_used", lambda: "model-x")
    monkeypatch.setattr(llm_analyst, "_LAST_CALL_META", {})
    monkeypatch.setattr(llm_client, "get_routing_info",
                        lambda: {"daily_cost": 0.0})
    # order-independent: never read another test's thread-local attempt meta
    monkeypatch.setattr(llm_client, "get_last_call_meta", lambda: None,
                        raising=False)
    llm_analyst._DEDUP_CACHE.clear()
    state["replay_dir"] = tmp_path / "llm_replay"
    state["call_dir"] = tmp_path / "llm_calls"
    yield state
    llm_analyst._DEDUP_CACHE.clear()


def _transport(monkeypatch, state, response, llm_response=None,
               model_exc=None, llm_exc=None):
    def fake_call_model(prompt, system="", model="", max_tokens=2048,
                        json_schema=None, temperature=None, timeout=None,
                        **k):
        state["calls"].append({"prompt": prompt, "system": system})
        if model_exc is not None:
            raise model_exc
        return response

    def fake_call_llm(prompt, system="", max_tokens=2048, json_schema=None,
                      temperature=None, **k):
        if llm_exc is not None:
            raise llm_exc
        return llm_response

    monkeypatch.setattr(llm_analyst, "call_model", fake_call_model)
    monkeypatch.setattr(llm_analyst, "call_llm", fake_call_llm)


def _rows(d):
    out = []
    if d.exists():
        for f in sorted(d.glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    out.append(json.loads(line))
    return out


def _cands(*syms, headline="hi"):
    return [{"symbol": s, "pred_return": 0.33, "fundamentals_text": "P/E=20",
             "news_headlines": [headline]} for s in (syms or ("TSLA",))]


# --------------------------------------------------------------------------- #
# Success path: legacy replay record byte-identical + new keys; call row
# --------------------------------------------------------------------------- #

def test_success_replay_record_legacy_keys_unchanged_new_keys_added(
        env, monkeypatch):
    _transport(monkeypatch, env, json.dumps(
        {"TSLA": {"s": 0.72, "bull": "b", "bear": "be", "r": "r"}}))
    cands = _cands("TSLA")
    result = llm_analyst.analyze_trades(
        cands, "stock", equity=1000, positions=["TSLA"],
        position_details={"TSLA": {"qty": 1, "entry_price": 10}},
        fng_value=40, model_config={"forward_bars": 24})
    # return value byte-identical to the pre-W15 contract
    assert result == {"TSLA": {"m": round(0.72 * 1.5, 2), "s": 0.72,
                               "r": "r", "bull": "b", "bear": "be"}}
    recs = _rows(env["replay_dir"])
    assert len(recs) == 1
    rec = recs[0]
    keys = list(rec)
    # legacy keys first, same order, then ONLY the additive keys
    assert keys[:len(LEGACY_REPLAY_KEYS)] == LEGACY_REPLAY_KEYS
    assert set(keys[len(LEGACY_REPLAY_KEYS):]) == NEW_REPLAY_KEYS
    golden = {"asset_type": "stock", "forward_bars": 24, "equity": 1000,
              "positions": ["TSLA"],
              "position_details": {"TSLA": {"qty": 1, "entry_price": 10}},
              "fng": 40, "candidates": cands,
              "live_scores": {"TSLA": 0.72}, "live_model": "model-x"}
    assert {k: rec[k] for k in LEGACY_REPLAY_KEYS if k != "ts"} == golden
    assert rec["dedup_hit"] is False
    assert rec["fence_stripped"] is False
    assert isinstance(rec["latency_ms"], int) and rec["latency_ms"] >= 0
    assert rec["parse_flags"] == {"TSLA": {"raw_s": 0.72,
                                           "s_defaulted": False,
                                           "s_nonfinite": False,
                                           "s_clamped": False}}


def test_success_writes_one_llm_call_row(env, monkeypatch):
    _transport(monkeypatch, env, json.dumps(
        {"TSLA": {"s": 0.6, "bull": "b", "bear": "be", "r": "r"}}))
    llm_analyst.analyze_trades(_cands("TSLA"), "stock")
    rows = _rows(env["call_dir"])
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == CALL_ROW_KEYS
    assert list(row)[:2] == ["ts", "action"]
    assert row["action"] == "llm_call"
    assert row["outcome"] == "ok"
    assert row["path"] == "call_model"
    assert row["n_symbols_sent"] == 1 and row["n_symbols_returned"] == 1
    assert row["model_used"] == "model-x"
    assert row["requested_model"] == "model-x"
    assert row["transport_errors"] == []
    assert row["advisor_v2"] is False and row["dedup_hit"] is False
    assert row["max_tokens"] == 4096
    assert row["temperature"] == llm_analyst._ANALYST_TEMPERATURE
    # A2: stubbed transport makes no HTTP attempt -> client meta None
    for k in ("finish_reason", "block_reason", "usage_in", "usage_out",
              "http_status"):
        assert row[k] is None
    # sha is the exact system+NUL+prompt the transport saw, and joins the
    # replay record and the llm_analysis meta
    sent = env["calls"][0]
    want = hashlib.sha256(
        (sent["system"] + "\x00" + sent["prompt"]).encode()).hexdigest()
    assert row["prompt_sha256"] == want
    assert _rows(env["replay_dir"])[0]["prompt_sha256"] == want
    assert llm_analyst.get_last_analysis_meta()["prompt_sha256"] == want


def test_last_analysis_meta_legacy_keys_plus_parse_flags(env, monkeypatch):
    _transport(monkeypatch, env, json.dumps(
        {"TSLA": {"r": "r", "bull": "b", "bear": "be"}}))   # s omitted
    llm_analyst.analyze_trades(_cands("TSLA"), "stock")
    meta = llm_analyst.get_last_analysis_meta()
    assert {"model", "prompt_sha256", "dedup_hit", "latency_ms",
            "parse_flags"} <= set(meta)
    assert meta["parse_flags"]["TSLA"]["s_defaulted"] is True
    assert meta["parse_flags"]["TSLA"]["raw_s"] is None


# --------------------------------------------------------------------------- #
# Failure rows: written, right outcome, never raising, no replay record
# --------------------------------------------------------------------------- #

def test_transport_error_writes_failure_row_no_exception(env, monkeypatch):
    _transport(monkeypatch, env, None, model_exc=RuntimeError("down"),
               llm_exc=ValueError("also down"))
    assert llm_analyst.analyze_trades(_cands("TSLA", "AAPL"), "stock") == {}
    assert _rows(env["replay_dir"]) == []
    rows = _rows(env["call_dir"])
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == CALL_ROW_KEYS
    assert row["outcome"] == "transport_error"
    assert row["transport_errors"] == ["call_model:RuntimeError",
                                       "call_llm:ValueError"]
    assert row["path"] == "call_llm"
    assert row["n_symbols_sent"] == 2 and row["n_symbols_returned"] == 0
    assert row["model_used"] is None and row["response_chars"] == 0
    assert re.fullmatch(r"[0-9a-f]{64}", row["prompt_sha256"])
    # a failure must not touch the success-only meta
    assert llm_analyst.get_last_analysis_meta() == {}


def test_empty_response_outcome_empty(env, monkeypatch):
    _transport(monkeypatch, env, None, llm_response="")
    assert llm_analyst.analyze_trades(_cands("TSLA"), "stock") == {}
    row, = _rows(env["call_dir"])
    assert row["outcome"] == "empty" and row["transport_errors"] == []


def test_transport_discard_when_client_exposes_finish_reason(
        env, monkeypatch):
    monkeypatch.setattr(llm_client, "get_last_call_meta",
                        lambda: {"finish_reason": "MAX_TOKENS",
                                 "block_reason": None, "usage_out": 4096},
                        raising=False)
    _transport(monkeypatch, env, None, llm_response=None)
    assert llm_analyst.analyze_trades(_cands("TSLA"), "stock") == {}
    row, = _rows(env["call_dir"])
    assert row["outcome"] == "transport_discard"
    assert row["finish_reason"] == "MAX_TOKENS" and row["usage_out"] == 4096


@pytest.mark.parametrize("text,outcome", [("this is not json", "parse_fail"),
                                          ("[1, 2]", "not_object"),
                                          ('{"ZZZ": {"s": 0.4}}', "partial")])
def test_parse_level_failures_write_row_only(env, monkeypatch, text, outcome):
    _transport(monkeypatch, env, text)
    assert llm_analyst.analyze_trades(_cands("TSLA"), "stock") == {}
    assert _rows(env["replay_dir"]) == []          # no scoreless cycle
    assert not (env["replay_dir"].parent / "llm_analysis.json").exists()
    row, = _rows(env["call_dir"])
    assert row["outcome"] == outcome
    assert row["n_symbols_returned"] == 0
    assert row["response_chars"] == len(text)
    assert row["model_used"] == "model-x"


def test_partial_response_keeps_replay_record(env, monkeypatch):
    _transport(monkeypatch, env, json.dumps(
        {"TSLA": {"s": 0.6, "r": "", "bull": "", "bear": ""}}))
    out = llm_analyst.analyze_trades(_cands("TSLA", "AAPL"), "stock")
    assert set(out) == {"TSLA"}
    row, = _rows(env["call_dir"])
    assert row["outcome"] == "partial"
    assert (row["n_symbols_sent"], row["n_symbols_returned"]) == (2, 1)
    assert len(_rows(env["replay_dir"])) == 1


def test_call_row_writer_never_raises(env, monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")          # a FILE where the journal dirs would go
    monkeypatch.setattr(llm_analyst, "_REPLAY_DIR", blocker / "llm_replay")
    _transport(monkeypatch, env, None, model_exc=RuntimeError("down"),
               llm_exc=RuntimeError("down"))
    assert llm_analyst.analyze_trades(_cands("TSLA"), "stock") == {}
    _transport(monkeypatch, env, json.dumps({"TSLA": {"s": 0.6}}))
    assert llm_analyst.analyze_trades(_cands("TSLA"), "stock")["TSLA"]["s"] == 0.6
    llm_analyst._write_llm_call_row({"bad": object()}, None)   # default=str
    llm_analyst._write_llm_call_row(None, None)                # not a dict


def test_build_call_row_internal_error_returns_none():
    assert llm_analyst._build_call_row(
        asset_type="stock", requested_model="m", path="call_model",
        response=object(), result={}, diag=None, n_sent=1, latency_ms=0,
        max_tokens=1, prompt_sha256=None, advisor_v2=False,
        transport_errors=[], cost0=None) is None


# --------------------------------------------------------------------------- #
# Gates: persist=False / replay_capture_enabled=False / dedup hit
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fail", [False, True])
def test_persist_false_writes_nothing(env, monkeypatch, fail):
    if fail:
        _transport(monkeypatch, env, "not json", llm_response=None)
    else:
        _transport(monkeypatch, env, json.dumps({"TSLA": {"s": 0.6}}))
    llm_analyst.analyze_trades(_cands("TSLA"), "stock", persist=False)
    assert not env["replay_dir"].exists()
    assert not env["call_dir"].exists()
    assert llm_analyst.get_last_analysis_meta() == {}


@pytest.mark.parametrize("fail", [False, True])
def test_replay_capture_disabled_writes_nothing(env, monkeypatch, fail):
    env["cfg"] = _cfg(replay_capture_enabled=False)
    if fail:
        _transport(monkeypatch, env, None, model_exc=RuntimeError("x"),
                   llm_exc=RuntimeError("y"))
    else:
        _transport(monkeypatch, env, json.dumps({"TSLA": {"s": 0.6}}))
    llm_analyst.analyze_trades(_cands("TSLA"), "stock")
    assert not env["replay_dir"].exists()
    assert not env["call_dir"].exists()


def test_dedup_hit_writes_no_call_row(env, monkeypatch):
    env["cfg"] = _cfg(analyst_dedup_ttl_sec=1800)
    _transport(monkeypatch, env, json.dumps({"TSLA": {"s": 0.7}}))
    llm_analyst.analyze_trades(_cands("TSLA"), "stock")
    llm_analyst.analyze_trades(_cands("TSLA"), "stock")
    assert len(env["calls"]) == 1                    # second was a cache hit
    assert len(_rows(env["call_dir"])) == 1


def test_journal_replay_without_call_meta_is_legacy(env):
    llm_analyst._journal_replay([{"symbol": "TSLA"}], "stock", 0, [], None,
                                None, None, {"TSLA": {"s": 0.5}}, "m")
    rec, = _rows(env["replay_dir"])
    assert list(rec) == LEGACY_REPLAY_KEYS
    assert not env["call_dir"].exists()


# --------------------------------------------------------------------------- #
# prompt_sha256 stability
# --------------------------------------------------------------------------- #

def test_prompt_sha256_stable_and_distinct(env, monkeypatch):
    _transport(monkeypatch, env, json.dumps({"TSLA": {"s": 0.6}}))
    llm_analyst.analyze_trades(_cands("TSLA", headline="h1"), "stock")
    llm_analyst.analyze_trades(_cands("TSLA", headline="h1"), "stock")
    llm_analyst.analyze_trades(_cands("TSLA", headline="h2"), "stock")
    shas = [r["prompt_sha256"] for r in _rows(env["call_dir"])]
    assert len(shas) == 3
    assert shas[0] == shas[1] != shas[2]
    assert llm_analyst._prompt_sha256("s", "p") == hashlib.sha256(
        b"s\x00p").hexdigest()
    assert llm_analyst._prompt_sha256(None, "p") is None


# --------------------------------------------------------------------------- #
# Non-finite / out-of-range / defaulted counters (A3)
# --------------------------------------------------------------------------- #

def test_counters_and_flags_end_to_end(env, monkeypatch):
    text = ('{"AAA": {"s": NaN}, "BBB": {"s": 1.7}, "CCC": {"r": "x"}, '
            '"DDD": {"s": "abc"}, "EEE": {"s": 0.4}}')
    _transport(monkeypatch, env, text)
    out = llm_analyst.analyze_trades(
        _cands("AAA", "BBB", "CCC", "DDD", "EEE"), "stock")
    # scores unchanged by the instrumentation
    assert {k: v["s"] for k, v in out.items()} == {
        "AAA": 0.5, "BBB": 1.0, "CCC": 0.5, "DDD": 0.5, "EEE": 0.4}
    row, = _rows(env["call_dir"])
    assert row["outcome"] == "ok"
    assert row["n_s_defaulted"] == 3          # NaN, missing, non-numeric
    assert row["n_nonfinite"] == 1
    assert row["n_out_of_range"] == 1
    flags = _rows(env["replay_dir"])[0]["parse_flags"]
    assert flags["AAA"] == {"raw_s": "nan", "s_defaulted": True,
                            "s_nonfinite": True, "s_clamped": False}
    assert flags["BBB"]["s_clamped"] is True and flags["BBB"]["raw_s"] == 1.7
    assert flags["DDD"]["raw_s"] == "abc" and flags["DDD"]["s_defaulted"]


def test_parse_diagnostics_extended_and_fence():
    d = llm_analyst._parse_diagnostics(
        '```json\n{"A": {"s": 0.5, "p_up": Infinity, "conviction": 9}, '
        '"B": {"s": 0.5, "p_up": 1.5, "conviction": NaN}, '
        '"C": {"s": 0.5, "p_up": 0.4, "conviction": Infinity}}\n```',
        ["A", "B", "C", "D"], extended=True)
    assert d["parse_outcome"] == "ok" and d["fence_stripped"] is True
    a, b, c = (d["parse_flags"][k] for k in "ABC")
    assert "D" not in d["parse_flags"]
    assert a["p_up_nonfinite"] and a["raw_p_up"] == "inf"
    assert a["conviction_clamped"] and not a["conviction_nonfinite"]
    assert b["p_up_clamped"] and b["conviction_nonfinite"]
    assert c["conviction_nonfinite"] and not c["p_up_nonfinite"]
    # everything journaled is strict JSON (no NaN/Infinity tokens)
    json.dumps(d, allow_nan=False)


def test_parse_diagnostics_outcomes_and_never_raises():
    pd_ = llm_analyst._parse_diagnostics
    assert pd_("nope", ["A"])["parse_outcome"] == "parse_fail"
    assert pd_("[1]", ["A"])["parse_outcome"] == "not_object"
    assert pd_(None, ["A"])["parse_outcome"] == "error"      # no raise
    assert pd_('{"BTCUSD": {"s": 0.3}}', ["BTC/USD"])[
        "parse_flags"]["BTC/USD"]["raw_s"] == 0.3            # slash alias


def test_journal_scalar_bounds():
    js = llm_analyst._journal_scalar
    assert js(None) is None and js(True) is True and js(0.25) == 0.25
    assert js(float("nan")) == "nan" and js(float("-inf")) == "-inf"
    assert js("x" * 100) == "x" * 32
    assert js([1]) == "<list>" and js({"a": 1}) == "<dict>"
    assert isinstance(js(10 ** 400), str) and len(js(10 ** 400)) <= 32
    assert not math.isnan(js(1.0))


def test_outcome_vocabulary_is_the_spec_set():
    assert set(llm_analyst.LLM_CALL_OUTCOMES) == {
        "ok", "partial", "parse_fail", "not_object", "empty",
        "transport_discard", "transport_error"}
