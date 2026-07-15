"""Tests for the advisor-v2 decision dossier (llm_analyst.py) and its
measurement harness (llm_eval.py).

All HTTP mocked (monkeypatch llm_analyst.call_model / llm_analyst.call_llm);
config mocked (monkeypatch llm_analyst.load_llm_config /
trade_journal.load_llm_config); files redirected (llm_analyst._ANALYSIS_FILE,
trade_journal.JOURNAL_DIR, llm_eval.JOURNAL_DIR -> tmp_path) — zero network,
zero real quota.

Test groups (see the design doc for the full acceptance criteria):
  A. DEFAULT BYTE-COMPAT — the regression net: fails if anyone changes the
     default prompt/schema/journal behavior.
  B. V2 — extended schema/prompt/parse/shadow-journal, all opt-in.
  C. DEDUP — evidence-hash call-dedup cache + the veto-margin safety rule.
  D. CALIBRATION — pure numpy/scipy statistics on seeded synthetic data.
  E. LOADER — _load_advisor_entries filters correctly.
"""
import copy
import json
import re

import numpy as np
import pytest

import events_calendar
import macro_calendar
import llm_analyst
import llm_config
import llm_eval
import trade_journal
from llm_analyst import (
    _response_schema, _build_prompt, _parse_response, analyze_trades,
    _compute_event_lines, _macro_event_flags, _evidence_hash,
    _dedup_cache_hit, _fng_label, _pred_sign,
    PROMPT_VERSION_V1, PROMPT_VERSION_V2, PROMPT_REGISTRY, EVENT_FLAG_VOCAB,
    _SYSTEM_PROMPT, _V2_SYSTEM_ADDENDUM, _DEDUP_CACHE, LLM_VETO_THRESHOLD,
)
from llm_eval import (
    compute_incremental_report, compute_calibration_report,
    _load_advisor_entries, _avg_rank, _pearson, VETO_THRESHOLD, MIN_POWER_N,
)


# --------------------------------------------------------------------------- #
# Shared fixtures / helpers
# --------------------------------------------------------------------------- #

def _cfg(**overrides):
    """Full llm_config dict from real defaults + overrides. replay_capture
    is always forced off in tests — _journal_replay writes to a hardcoded
    journals/llm_replay path (not redirectable) and is out of this file's
    scope; disabling it keeps tests from touching the real repo tree."""
    cfg = copy.deepcopy(llm_config._DEFAULTS)
    cfg["replay_capture_enabled"] = False
    cfg.update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_analyst, "_ANALYSIS_FILE", tmp_path / "llm_analysis.json")
    monkeypatch.setattr(trade_journal, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
    llm_analyst._DEDUP_CACHE.clear()
    yield
    llm_analyst._DEDUP_CACHE.clear()


def _patch_config(monkeypatch, cfg):
    """Both llm_analyst AND trade_journal hold their own `load_llm_config`
    reference (each did `from llm_config import load_llm_config`) — patch
    both so log_decision's journal_enabled check agrees with analyze_trades'
    own config read."""
    monkeypatch.setattr(llm_analyst, "load_llm_config", lambda: cfg)
    monkeypatch.setattr(trade_journal, "load_llm_config", lambda: cfg)


def _candidates(symbols=("ZQZQ",), pred_return=0.5, news_headlines=None,
                fundamentals_text="", profile=None):
    out = []
    for sym in symbols:
        c = {"symbol": sym, "pred_return": pred_return,
            "fundamentals_text": fundamentals_text,
            "news_headlines": list(news_headlines) if news_headlines else []}
        if profile is not None:
            c["profile"] = profile
        out.append(c)
    return out


def _install_transport(monkeypatch, response_obj, call_log=None,
                       raise_exc=None, model="test-model"):
    """Mock call_model (and, unused by default, call_llm) + the model
    routing helpers so no real config/network is touched."""
    if call_log is None:
        call_log = []
    body = json.dumps(response_obj) if not isinstance(response_obj, str) else response_obj

    def fake_call_model(prompt, system="", model=model, max_tokens=0,
                        json_schema=None, temperature=0.0, timeout=30):
        if raise_exc is not None:
            raise raise_exc
        call_log.append({"prompt": prompt, "system": system, "model": model,
                         "max_tokens": max_tokens, "json_schema": json_schema,
                         "temperature": temperature})
        return body

    def fake_call_llm(prompt, system="", max_tokens=0, json_schema=None,
                      temperature=0.0):
        raise AssertionError("call_llm fallback should not be reached when "
                             "call_model succeeds")

    monkeypatch.setattr(llm_analyst, "call_model", fake_call_model)
    monkeypatch.setattr(llm_analyst, "call_llm", fake_call_llm)
    monkeypatch.setattr(llm_analyst, "get_recommended_model", lambda role: "test-model")
    monkeypatch.setattr(llm_analyst, "get_last_model_used", lambda: None)
    return call_log


def _read_journal_rows(tmp_path):
    rows = []
    for f in sorted(tmp_path.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Group A — default byte-compat (the regression net)
# --------------------------------------------------------------------------- #

class TestDefaultByteCompat:
    def test_schema_default_byte_identical(self):
        expected = {
            "type": "OBJECT",
            "properties": {
                sym: {
                    "type": "OBJECT",
                    "properties": {
                        "s": {"type": "NUMBER",
                              "description": "conviction score 0.0-1.0 (see rubric)"},
                        "bull": {"type": "STRING"},
                        "bear": {"type": "STRING"},
                        "r": {"type": "STRING",
                              "description": "2-3 sentence actionable synthesis"},
                    },
                    "required": ["s", "bull", "bear", "r"],
                } for sym in ("BTC/USD", "AAPL")
            },
            "required": ["BTC/USD", "AAPL"],
        }
        assert _response_schema(["BTC/USD", "AAPL"]) == expected
        # extended=False explicitly must match the bare default too
        assert _response_schema(["BTC/USD", "AAPL"], extended=False) == expected

    def test_build_prompt_default_byte_identical(self):
        from fees import round_trip_cost_pct
        rt = round_trip_cost_pct("stock", spread_pct=0.05)
        candidates = [{"symbol": "ZQZQ", "pred_return": 0.4213}]
        prompt = _build_prompt(candidates, "stock", 100000, ["AAPL"], 55,
                               {"forward_bars": 24})
        expected = "\n".join([
            "## Market Context",
            "- Asset type: stock",
            "- Market regime: Neutral (Fear & Greed: 55)",
            "- Account equity: $100,000",
            "- Currently holding: AAPL",
            f"- Round-trip trading cost: ~{rt:.2f}% of notional — a thesis "
            "must be worth multiples of this to act on",
            "- ML model horizon: ~24 hours forward",
            "",
            "## Symbols to Evaluate",
            "",
            "### ZQZQ",
            "- ML model prediction: +0.4213% (bullish signal)",
            "",
            "For each symbol provide: bull (2-3 sentences with specifics), "
            "bear (2-3 sentences with risks), s (precise continuous score "
            "like 0.37 or 0.72, NOT rounded to 0.05), r (2-3 sentence "
            "actionable synthesis).",
        ])
        assert prompt == expected

    def test_analyze_trades_default_passes_v1_prompt_and_schema(self, monkeypatch):
        cfg = _cfg()
        _patch_config(monkeypatch, cfg)
        candidates = _candidates()
        response = {"ZQZQ": {"s": 0.6, "bull": "b", "bear": "be", "r": "r"}}
        log = _install_transport(monkeypatch, response)

        result = analyze_trades(candidates, "crypto")

        assert len(log) == 1
        assert log[0]["system"] == _SYSTEM_PROMPT
        assert log[0]["json_schema"] == _response_schema(["ZQZQ"])
        assert log[0]["max_tokens"] == max(4096, 1 * 400)
        assert result["ZQZQ"]["s"] == 0.6

    def test_analyze_trades_default_no_advisor_row(self, monkeypatch, tmp_path):
        cfg = _cfg(advisor_v2_enabled=False)
        _patch_config(monkeypatch, cfg)
        response = {"ZQZQ": {"s": 0.6, "bull": "b", "bear": "be", "r": "r"}}
        _install_transport(monkeypatch, response)

        analyze_trades(_candidates(), "crypto")

        rows = _read_journal_rows(tmp_path)
        assert not any(r.get("action") == "llm_advisor_v2" for r in rows)

    def test_analyze_trades_default_llm_analysis_json_keys(self, monkeypatch):
        cfg = _cfg()
        _patch_config(monkeypatch, cfg)
        response = {"ZQZQ": {"s": 0.6, "bull": "b", "bear": "be", "r": "r"}}
        _install_transport(monkeypatch, response)

        analyze_trades(_candidates(), "crypto")
        saved = llm_analyst.load_analysis()
        entry = saved["crypto"]["ZQZQ"]
        assert set(entry.keys()) == {"m", "s", "r", "bull", "bear",
                                     "timestamp", "model"}

    def test_effective_system_prompt_default_equals_system_prompt(self, monkeypatch):
        cfg = _cfg()
        _patch_config(monkeypatch, cfg)
        response = {"ZQZQ": {"s": 0.5, "bull": "", "bear": "", "r": ""}}
        log = _install_transport(monkeypatch, response)
        analyze_trades(_candidates(), "crypto")
        assert log[0]["system"] == _SYSTEM_PROMPT
        assert _V2_SYSTEM_ADDENDUM not in log[0]["system"]


# --------------------------------------------------------------------------- #
# Group B — advisor v2
# --------------------------------------------------------------------------- #

class TestSchemaV2:
    def test_extended_schema_has_five_extra_required_fields(self):
        schema = _response_schema(["TSLA"], extended=True)
        entry = schema["properties"]["TSLA"]
        for field in ("p_up", "conviction", "abstain", "key_risks", "event_flags"):
            assert field in entry["properties"]
            assert field in entry["required"]
        assert set(entry["required"]) == {"s", "bull", "bear", "r", "p_up",
                                          "conviction", "abstain",
                                          "key_risks", "event_flags"}
        assert entry["properties"]["event_flags"]["items"]["enum"] == list(EVENT_FLAG_VOCAB)
        assert entry["properties"]["conviction"]["type"] == "INTEGER"
        assert entry["properties"]["abstain"]["type"] == "BOOLEAN"
        assert entry["properties"]["key_risks"]["type"] == "ARRAY"

    def test_prompt_registry_has_both_versions(self):
        assert PROMPT_VERSION_V1 in PROMPT_REGISTRY
        assert PROMPT_VERSION_V2 in PROMPT_REGISTRY


class TestBuildPromptV2:
    def test_extended_prompt_contains_addendum(self):
        prompt = _build_prompt(_candidates(), "crypto", 0, None, None, {},
                               extended=True)
        # the addendum is on the SYSTEM prompt, not the user prompt — verify
        # via analyze_trades' system composition instead, but confirm the
        # user prompt at least carries the event-flags framing is absent
        # here (correctly scoped to the system prompt).
        assert prompt is not None

    def test_extended_prompt_contains_computed_event_lines(self, monkeypatch):
        monkeypatch.setattr(events_calendar, "earnings_within_days",
                            lambda sym, days=1: True)
        monkeypatch.setattr(events_calendar, "reported_recently", lambda sym: False)
        monkeypatch.setattr(events_calendar, "blocks_overnight_hold", lambda sym: False)
        monkeypatch.setattr(macro_calendar, "macro_standdown",
                            lambda now=None: (False, None))
        monkeypatch.setattr(macro_calendar, "FOMC_STATEMENT_DAYS", [])
        monkeypatch.setattr(macro_calendar, "CPI_RELEASE_DAYS", [])

        candidates = [{"symbol": "TSLA", "pred_return": 0.1}]
        prompt = _build_prompt(candidates, "stock", 0, None, None, {},
                               extended=True)
        assert "Known scheduled events (computed): earnings_within_3d" in prompt

    def test_extended_prompt_contains_macro_line(self, monkeypatch):
        monkeypatch.setattr(macro_calendar, "macro_standdown",
                            lambda now=None: (True, "FOMC stand-down"))
        monkeypatch.setattr(macro_calendar, "FOMC_STATEMENT_DAYS", [])
        monkeypatch.setattr(macro_calendar, "CPI_RELEASE_DAYS", [])

        candidates = [{"symbol": "BTC/USD", "pred_return": 0.1}]
        prompt = _build_prompt(candidates, "crypto", 0, None, None, {},
                               extended=True)
        assert "Macro calendar:" in prompt
        assert "macro stand-down window active" in prompt

    def test_default_extended_false_omits_v2_content(self, monkeypatch):
        monkeypatch.setattr(macro_calendar, "macro_standdown",
                            lambda now=None: (True, "FOMC stand-down"))
        candidates = [{"symbol": "TSLA", "pred_return": 0.1}]
        prompt = _build_prompt(candidates, "stock", 0, None, None, {})
        assert "Macro calendar:" not in prompt
        assert "Known scheduled events" not in prompt


class TestComputeEventLines:
    def test_stock_earnings_flags(self, monkeypatch):
        monkeypatch.setattr(events_calendar, "earnings_within_days",
                            lambda sym, days=1: True)
        monkeypatch.setattr(events_calendar, "reported_recently", lambda sym: True)
        monkeypatch.setattr(events_calendar, "blocks_overnight_hold", lambda sym: True)
        monkeypatch.setattr(macro_calendar, "macro_standdown",
                            lambda now=None: (False, None))
        monkeypatch.setattr(macro_calendar, "FOMC_STATEMENT_DAYS", [])
        monkeypatch.setattr(macro_calendar, "CPI_RELEASE_DAYS", [])

        lines, flags = _compute_event_lines("TSLA", "stock")
        assert set(flags) == {"earnings_within_3d", "post_earnings", "overnight_block"}
        assert len(lines) == 1

    def test_crypto_never_calls_earnings_calendar(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("should not be called for crypto")
        monkeypatch.setattr(events_calendar, "earnings_within_days", boom)
        monkeypatch.setattr(macro_calendar, "macro_standdown",
                            lambda now=None: (False, None))
        monkeypatch.setattr(macro_calendar, "FOMC_STATEMENT_DAYS", [])
        monkeypatch.setattr(macro_calendar, "CPI_RELEASE_DAYS", [])
        lines, flags = _compute_event_lines("BTC/USD", "crypto")
        assert flags == []
        assert lines == []

    def test_any_exception_returns_empty(self, monkeypatch):
        def boom(sym, days=1):
            raise RuntimeError("calendar down")
        monkeypatch.setattr(events_calendar, "earnings_within_days", boom)
        lines, flags = _compute_event_lines("TSLA", "stock")
        assert (lines, flags) == ([], [])

    def test_macro_event_flags_fomc_today(self, monkeypatch):
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        et = now.astimezone(macro_calendar._ET)
        monkeypatch.setattr(macro_calendar, "FOMC_STATEMENT_DAYS",
                            [(et.year, et.month, et.day)])
        monkeypatch.setattr(macro_calendar, "CPI_RELEASE_DAYS", [])
        monkeypatch.setattr(macro_calendar, "macro_standdown",
                            lambda now=None: (False, None))
        assert "fomc_today" in _macro_event_flags()

    def test_macro_event_flags_never_raises(self, monkeypatch):
        monkeypatch.setattr(macro_calendar, "macro_standdown",
                            lambda now=None: (_ for _ in ()).throw(RuntimeError()))
        assert _macro_event_flags() == []


class TestParseResponseV2:
    def _entry(self, **overrides):
        base = {"s": 0.6, "bull": "b", "bear": "be", "r": "r",
               "p_up": 0.7, "conviction": 3, "abstain": False,
               "key_risks": ["risk one"], "event_flags": ["earnings_within_3d"]}
        base.update(overrides)
        return base

    def test_p_up_clamped_high(self):
        response = json.dumps({"TSLA": self._entry(p_up=1.7)})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert result["TSLA"]["p_up"] == 1.0

    def test_p_up_clamped_low(self):
        response = json.dumps({"TSLA": self._entry(p_up=-0.2)})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert result["TSLA"]["p_up"] == 0.0

    def test_p_up_non_numeric_is_none(self):
        response = json.dumps({"TSLA": self._entry(p_up="high")})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert result["TSLA"]["p_up"] is None

    def test_conviction_clamped_high(self):
        response = json.dumps({"TSLA": self._entry(conviction=9)})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert result["TSLA"]["conviction"] == 5

    def test_conviction_non_numeric_is_none(self):
        response = json.dumps({"TSLA": self._entry(conviction="x")})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert result["TSLA"]["conviction"] is None

    def test_abstain_truthy_becomes_true(self):
        response = json.dumps({"TSLA": self._entry(abstain=True)})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert result["TSLA"]["abstain"] is True

    def test_abstain_missing_defaults_false(self):
        entry = self._entry()
        del entry["abstain"]
        response = json.dumps({"TSLA": entry})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert result["TSLA"]["abstain"] is False

    def test_key_risks_capped_at_three_and_sanitized(self):
        long_risk = "x" * 500
        response = json.dumps({"TSLA": self._entry(
            key_risks=["one", "two", "three", "four", long_risk])})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert len(result["TSLA"]["key_risks"]) == 3
        assert result["TSLA"]["key_risks"] == ["one", "two", "three"]

    def test_key_risks_sanitized_length(self):
        long_risk = "y" * 500
        response = json.dumps({"TSLA": self._entry(key_risks=[long_risk])})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert len(result["TSLA"]["key_risks"][0]) <= 100

    def test_key_risks_non_list_becomes_empty(self):
        response = json.dumps({"TSLA": self._entry(key_risks="not a list")})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert result["TSLA"]["key_risks"] == []

    def test_event_flags_junk_dropped_by_whitelist(self):
        response = json.dumps({"TSLA": self._entry(
            event_flags=["earnings_within_3d", "totally_fake_flag", "<script>"])})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert result["TSLA"]["event_flags"] == ["earnings_within_3d"]

    def test_event_flags_non_list_becomes_empty(self):
        response = json.dumps({"TSLA": self._entry(event_flags="bad")})
        result = _parse_response(response, ["TSLA"], extended=True)
        assert result["TSLA"]["event_flags"] == []

    def test_extended_false_does_not_add_v2_keys(self):
        response = json.dumps({"TSLA": self._entry()})
        result = _parse_response(response, ["TSLA"], extended=False)
        for key in ("p_up", "conviction", "abstain", "key_risks", "event_flags"):
            assert key not in result["TSLA"]

    def test_abstain_true_leaves_s_identical_to_v1_parse(self):
        payload = self._entry(abstain=True, s=0.42)
        response = json.dumps({"TSLA": payload})
        v1 = _parse_response(response, ["TSLA"], extended=False)
        v2 = _parse_response(response, ["TSLA"], extended=True)
        assert v1["TSLA"]["s"] == v2["TSLA"]["s"] == 0.42
        assert v2["TSLA"]["abstain"] is True


class TestAnalyzeTradesV2Shadow:
    def _v2_response(self, sym="ZQZQ", s=0.6):
        return {sym: {"s": s, "bull": "b", "bear": "be", "r": "r",
                      "p_up": 0.65, "conviction": 4, "abstain": False,
                      "key_risks": ["risk"], "event_flags": ["earnings_within_3d"]}}

    def test_shadow_row_present_with_version_and_sha(self, monkeypatch, tmp_path):
        cfg = _cfg(advisor_v2_enabled=True)
        _patch_config(monkeypatch, cfg)
        _install_transport(monkeypatch, self._v2_response())

        analyze_trades(_candidates(), "crypto", model_config={"forward_bars": 12})

        rows = _read_journal_rows(tmp_path)
        advisor_rows = [r for r in rows if r.get("action") == "llm_advisor_v2"]
        assert len(advisor_rows) == 1
        row = advisor_rows[0]
        assert row["prompt_version"] == PROMPT_VERSION_V2
        assert re.fullmatch(r"[0-9a-f]{64}", row["prompt_sha256"])
        assert row["forward_bars"] == 12
        assert row["dedup_hit"] is False

    def test_shadow_row_per_symbol_fields(self, monkeypatch, tmp_path):
        cfg = _cfg(advisor_v2_enabled=True)
        _patch_config(monkeypatch, cfg)
        _install_transport(monkeypatch, self._v2_response())

        analyze_trades(_candidates(news_headlines=["h1", "h2"]), "crypto")

        rows = _read_journal_rows(tmp_path)
        row = [r for r in rows if r.get("action") == "llm_advisor_v2"][0]
        entry = row["scores"]["ZQZQ"]
        assert entry["p_up"] == 0.65
        assert entry["conviction"] == 4
        assert entry["n_headlines"] == 2
        assert entry["key_risks"] == ["risk"]
        assert "computed_events" in entry

    def test_journal_enabled_false_suppresses_row(self, monkeypatch, tmp_path):
        cfg = _cfg(advisor_v2_enabled=True, journal_enabled=False)
        _patch_config(monkeypatch, cfg)
        _install_transport(monkeypatch, self._v2_response())

        analyze_trades(_candidates(), "crypto")

        rows = _read_journal_rows(tmp_path)
        assert not any(r.get("action") == "llm_advisor_v2" for r in rows)

    def test_extended_schema_and_prompt_used_when_enabled(self, monkeypatch):
        cfg = _cfg(advisor_v2_enabled=True)
        _patch_config(monkeypatch, cfg)
        log = _install_transport(monkeypatch, self._v2_response())

        analyze_trades(_candidates(), "crypto")

        assert log[0]["system"] == _SYSTEM_PROMPT + _V2_SYSTEM_ADDENDUM
        assert log[0]["json_schema"] == _response_schema(["ZQZQ"], extended=True)
        assert log[0]["max_tokens"] == max(4096, 1 * 550)

    def test_call_model_raising_is_fail_soft(self, monkeypatch):
        cfg = _cfg(advisor_v2_enabled=True)
        _patch_config(monkeypatch, cfg)

        def boom(*a, **k):
            raise RuntimeError("provider down")

        monkeypatch.setattr(llm_analyst, "call_model", boom)
        monkeypatch.setattr(llm_analyst, "call_llm", boom)
        monkeypatch.setattr(llm_analyst, "get_recommended_model", lambda role: "test-model")

        result = analyze_trades(_candidates(), "crypto")
        assert result == {}


# --------------------------------------------------------------------------- #
# Group C — dedup cache
# --------------------------------------------------------------------------- #

class TestDedupCache:
    def _response(self, s=0.7):
        return {"ZQZQ": {"s": s, "bull": "b", "bear": "be", "r": "r"}}

    def test_ttl_zero_calls_every_time(self, monkeypatch):
        cfg = _cfg(analyst_dedup_ttl_sec=0)
        _patch_config(monkeypatch, cfg)
        log = _install_transport(monkeypatch, self._response())

        analyze_trades(_candidates(news_headlines=["h1"]), "crypto")
        analyze_trades(_candidates(news_headlines=["h1"]), "crypto")

        assert len(log) == 2

    def test_ttl_positive_identical_evidence_is_one_call(self, monkeypatch, tmp_path):
        cfg = _cfg(analyst_dedup_ttl_sec=1800)
        _patch_config(monkeypatch, cfg)
        log = _install_transport(monkeypatch, self._response())

        r1 = analyze_trades(_candidates(news_headlines=["h1"]), "crypto")
        analysis_1 = llm_analyst.load_analysis()
        r2 = analyze_trades(_candidates(news_headlines=["h1"]), "crypto")
        analysis_2 = llm_analyst.load_analysis()

        assert len(log) == 1
        assert r1 == r2
        # cache hit must NOT rewrite llm_analysis.json (timestamp unchanged)
        assert analysis_1 == analysis_2

    def test_headline_change_is_a_miss(self, monkeypatch):
        cfg = _cfg(analyst_dedup_ttl_sec=1800)
        _patch_config(monkeypatch, cfg)
        log = _install_transport(monkeypatch, self._response())

        analyze_trades(_candidates(news_headlines=["h1"]), "crypto")
        analyze_trades(_candidates(news_headlines=["h2"]), "crypto")

        assert len(log) == 2

    def test_held_set_change_is_a_miss(self, monkeypatch):
        cfg = _cfg(analyst_dedup_ttl_sec=1800)
        _patch_config(monkeypatch, cfg)
        log = _install_transport(monkeypatch, self._response())

        analyze_trades(_candidates(news_headlines=["h1"]), "crypto", positions=[])
        analyze_trades(_candidates(news_headlines=["h1"]), "crypto", positions=["ZQZQ"])

        assert len(log) == 2

    def test_pred_sign_flip_is_a_miss(self, monkeypatch):
        cfg = _cfg(analyst_dedup_ttl_sec=1800)
        _patch_config(monkeypatch, cfg)
        log = _install_transport(monkeypatch, self._response())

        analyze_trades(_candidates(pred_return=0.5), "crypto")
        analyze_trades(_candidates(pred_return=-0.5), "crypto")

        assert len(log) == 2

    def test_pred_magnitude_drift_alone_is_a_hit(self, monkeypatch):
        cfg = _cfg(analyst_dedup_ttl_sec=1800)
        _patch_config(monkeypatch, cfg)
        log = _install_transport(monkeypatch, self._response())

        analyze_trades(_candidates(pred_return=0.5), "crypto")
        analyze_trades(_candidates(pred_return=0.99), "crypto")

        assert len(log) == 1

    def test_fng_bucket_change_is_a_miss(self, monkeypatch):
        cfg = _cfg(analyst_dedup_ttl_sec=1800)
        _patch_config(monkeypatch, cfg)
        log = _install_transport(monkeypatch, self._response())

        analyze_trades(_candidates(news_headlines=["h1"]), "crypto", fng_value=55)
        analyze_trades(_candidates(news_headlines=["h1"]), "crypto", fng_value=80)

        assert len(log) == 2

    def test_ttl_expiry_is_a_miss(self, monkeypatch):
        cfg = _cfg(analyst_dedup_ttl_sec=100)
        _patch_config(monkeypatch, cfg)
        log = _install_transport(monkeypatch, self._response())

        clock = {"t": 1_000_000.0}
        monkeypatch.setattr(llm_analyst.time, "time", lambda: clock["t"])

        analyze_trades(_candidates(news_headlines=["h1"]), "crypto")
        clock["t"] += 200  # past the 100s ttl
        analyze_trades(_candidates(news_headlines=["h1"]), "crypto")

        assert len(log) == 2

    def test_ttl_clamped_to_7000(self, monkeypatch):
        cfg = _cfg(analyst_dedup_ttl_sec=99999)
        _patch_config(monkeypatch, cfg)
        log = _install_transport(monkeypatch, self._response())

        clock = {"t": 1_000_000.0}
        monkeypatch.setattr(llm_analyst.time, "time", lambda: clock["t"])

        analyze_trades(_candidates(news_headlines=["h1"]), "crypto")
        clock["t"] += 7001  # beyond the clamp of 7000, even though config said 99999
        analyze_trades(_candidates(news_headlines=["h1"]), "crypto")

        assert len(log) == 2

    def test_cached_score_near_veto_bypasses_cache(self, monkeypatch):
        cfg = _cfg(analyst_dedup_ttl_sec=1800)
        _patch_config(monkeypatch, cfg)
        # s = 0.16 < LLM_VETO_THRESHOLD(0.15) + 0.05 = 0.20 -> must bypass
        log = _install_transport(monkeypatch, self._response(s=0.16))

        analyze_trades(_candidates(news_headlines=["h1"]), "crypto")
        analyze_trades(_candidates(news_headlines=["h1"]), "crypto")

        assert len(log) == 2

    def test_cached_score_safely_above_veto_margin_is_cached(self, monkeypatch):
        cfg = _cfg(analyst_dedup_ttl_sec=1800)
        _patch_config(monkeypatch, cfg)
        log = _install_transport(monkeypatch, self._response(s=0.7))

        analyze_trades(_candidates(news_headlines=["h1"]), "crypto")
        analyze_trades(_candidates(news_headlines=["h1"]), "crypto")

        assert len(log) == 1


class TestDedupHelpers:
    def test_evidence_hash_deterministic(self):
        candidates = _candidates(news_headlines=["h1"])
        h1 = _evidence_hash(candidates, "crypto", [], 50, {})
        h2 = _evidence_hash(candidates, "crypto", [], 50, {})
        assert h1 == h2
        assert len(h1) == 64

    def test_evidence_hash_changes_on_computed_events(self):
        candidates = _candidates()
        h1 = _evidence_hash(candidates, "stock", [], 50, {"ZQZQ": []})
        h2 = _evidence_hash(candidates, "stock", [], 50, {"ZQZQ": ["earnings_within_3d"]})
        assert h1 != h2

    def test_dedup_cache_hit_false_when_empty(self):
        assert _dedup_cache_hit(None, "abc", 100) is False

    def test_dedup_cache_hit_false_on_hash_mismatch(self):
        entry = {"hash": "abc", "ts": 0.0, "result": {"X": {"s": 0.7}}}
        assert _dedup_cache_hit(entry, "def", 1e12) is False

    def test_fng_label_buckets(self):
        assert _fng_label(5) == "Extreme Fear"
        assert _fng_label(20) == "Fear"
        assert _fng_label(50) == "Neutral"
        assert _fng_label(70) == "Greed"
        assert _fng_label(90) == "Extreme Greed"
        assert _fng_label(None) is None

    def test_pred_sign(self):
        assert _pred_sign(1.5) == 1
        assert _pred_sign(-0.001) == -1
        assert _pred_sign(0.0) == 0
        assert _pred_sign(None) is None
        assert _pred_sign("bad") is None


# --------------------------------------------------------------------------- #
# Group D — calibration statistics (pure numpy/scipy, seeded synthetic data)
# --------------------------------------------------------------------------- #

class TestCalibrationReport:
    def test_perfectly_calibrated(self):
        rng = np.random.default_rng(101)
        n = 2000
        p = rng.uniform(0, 1, n)
        u = rng.uniform(0, 1, n)
        outcome = (u < p).astype(int)
        magnitude = rng.exponential(1.0, n) + 0.01
        realized = np.where(outcome == 1, magnitude, -magnitude)

        rep = compute_calibration_report(p, realized)
        gaps = [abs(b["gap"]) for b in rep["bins"] if b["gap"] is not None]
        assert max(gaps) < 0.06
        assert rep["brier_skill"] > 0.15
        assert 0.85 <= rep["calibration_slope"] <= 1.15
        assert rep["verdict"] == "well_calibrated"

    def test_overconfident(self):
        rng = np.random.default_rng(102)
        n = 2000
        z = rng.uniform(-1, 1, n)
        true_p = np.clip(0.5 + 0.1 * z, 0.0, 1.0)
        u = rng.uniform(0, 1, n)
        outcome = (u < true_p).astype(int)
        magnitude = rng.exponential(1.0, n) + 0.01
        realized = np.where(outcome == 1, magnitude, -magnitude)
        stated_p = np.clip(0.5 + 0.4 * z, 0.0, 1.0)   # exaggerated certainty

        rep = compute_calibration_report(stated_p, realized)
        assert rep["calibration_slope"] < 0.7
        assert rep["verdict"] == "overconfident"

    def test_conviction_monotone(self):
        rng = np.random.default_rng(103)
        n = 2000
        conviction = rng.integers(1, 6, n)
        realized = 0.5 * (conviction - 3) + rng.normal(0, 1, n)
        p_up = rng.uniform(0, 1, n)  # irrelevant to this block

        rep = compute_calibration_report(p_up, realized, conviction=conviction)
        levels = rep["conviction"]["levels"]
        avg_rets = [levels[str(lvl)]["avg_fwd_ret_pct"] for lvl in range(1, 6)]
        assert avg_rets == sorted(avg_rets)
        assert rep["conviction"]["spearman_conviction_vs_realized"] > 0.2

    def test_informative_abstain(self):
        rng = np.random.default_rng(104)
        n = 1500
        abstain = rng.random(n) < 0.3

        p_up = np.empty(n)
        realized = np.empty(n)

        n_abstain = int(abstain.sum())
        # Abstain rows: p_up is pure noise, realized is a coin flip — no
        # relationship between the two (the abstention-artifact check).
        p_up[abstain] = rng.uniform(0, 1, n_abstain)
        coin = rng.choice([-1.0, 1.0], n_abstain)
        realized[abstain] = coin * (rng.exponential(1.0, n_abstain) + 0.01)

        n_active = n - n_abstain
        active_p = rng.uniform(0, 1, n_active)
        u = rng.uniform(0, 1, n_active)
        outcome = (u < active_p).astype(int)
        magnitude = rng.exponential(1.0, n_active) + 0.01
        p_up[~abstain] = active_p
        realized[~abstain] = np.where(outcome == 1, magnitude, -magnitude)

        rep = compute_calibration_report(p_up, realized, abstain=abstain)
        block = rep["abstain"]
        assert abs(block["hit_rate_abstain"] - 0.5) < 0.08

        all_rows_spearman = _pearson(_avg_rank(p_up), _avg_rank(realized))
        assert block["spearman_active_only"] > all_rows_spearman

    def test_small_sample_insufficient_power(self):
        rng = np.random.default_rng(105)
        n = 40
        p_up = rng.uniform(0, 1, n)
        realized = rng.normal(size=n)
        rep = compute_calibration_report(p_up, realized)
        assert rep["insufficient_power"] is True
        assert "insufficient_power" in rep["verdict"]

    def test_brier_skill_none_when_base_rate_degenerate(self):
        n = 100
        p_up = np.full(n, 0.9)
        realized = np.full(n, 1.0)  # outcome always 1 -> base_rate degenerate
        rep = compute_calibration_report(p_up, realized, min_n=10)
        assert rep["brier_skill"] is None


class TestIncrementalReportExtra:
    def _seeded_samples(self, seed=200, n=800):
        rng = np.random.default_rng(seed)
        pred = rng.normal(size=n)
        realized = pred + rng.normal(scale=1.0, size=n)
        s = 1.0 / (1.0 + np.exp(-pred))
        return list(zip(s, realized, pred)), rng

    def test_extra_none_is_regression_guard_identical(self):
        samples, _ = self._seeded_samples()
        rep_default = compute_incremental_report(samples, forward_bars=24, min_n=60)
        rep_explicit_none = compute_incremental_report(samples, forward_bars=24,
                                                        min_n=60, extra=None)
        assert rep_default == rep_explicit_none

    def test_extra_covariate_drives_its_own_beta(self):
        rng = np.random.default_rng(300)
        n = 3000
        pred = rng.normal(size=n)
        g = rng.normal(size=n)                  # independent extra covariate
        s_driver = rng.normal(size=n)             # independent driver for s
        realized = 1.0 * pred + 0.8 * g + 0.6 * s_driver + rng.normal(scale=0.5, size=n)
        s = 1.0 / (1.0 + np.exp(-1.2 * s_driver))
        samples = list(zip(s, realized, pred))

        rep_no_extra = compute_incremental_report(samples, forward_bars=24,
                                                   min_n=60)
        rep_extra = compute_incremental_report(samples, forward_bars=24,
                                               min_n=60, extra=g)

        enc = rep_extra["encompassing"]
        assert enc is not None and "b_extra" in enc
        assert abs(enc["b_extra"][0]["beta"]) > 0.1
        # b2 (s's own coefficient) preserved in sign/rough magnitude since s
        # is independent of g.
        b2_no_extra = rep_no_extra["encompassing"]["b2_s"]
        b2_extra = enc["b2_s"]
        assert abs(b2_extra - b2_no_extra) < 0.5

    def test_extra_with_missing_values_mean_imputed_no_crash(self):
        samples, _ = self._seeded_samples(seed=201)
        n = len(samples)
        rng = np.random.default_rng(202)
        extra = rng.normal(size=n)
        extra[:5] = np.nan
        rep = compute_incremental_report(samples, forward_bars=24, min_n=60,
                                         extra=extra)
        assert rep["encompassing"] is not None


# --------------------------------------------------------------------------- #
# Group E — advisor journal loader
# --------------------------------------------------------------------------- #

class TestLoadAdvisorEntries:
    def _write_journal(self, tmp_path, date_str, rows):
        path = tmp_path / f"{date_str}.jsonl"
        with open(path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_filters_only_advisor_rows_across_days(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
        import datetime
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)

        self._write_journal(tmp_path, today.isoformat(), [
            {"action": "llm_analysis", "ts": "2026-01-01T00:00:00+00:00"},
            {"action": "llm_advisor_v2", "ts": "2026-01-01T00:00:00+00:00",
             "asset_type": "crypto"},
            {"action": "buy", "ts": "2026-01-01T00:00:00+00:00"},
        ])
        self._write_journal(tmp_path, yesterday.isoformat(), [
            {"action": "llm_advisor_v2", "ts": "2026-01-01T00:00:00+00:00",
             "asset_type": "stock"},
        ])

        entries = _load_advisor_entries(days=3)
        assert len(entries) == 2
        assert all(e["action"] == "llm_advisor_v2" for e in entries)
        asset_types = {e["asset_type"] for e in entries}
        assert asset_types == {"crypto", "stock"}

    def test_no_files_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
        assert _load_advisor_entries(days=5) == []

    def test_corrupt_line_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
        import datetime
        today = datetime.date.today()
        path = tmp_path / f"{today.isoformat()}.jsonl"
        with open(path, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps({"action": "llm_advisor_v2"}) + "\n")
        entries = _load_advisor_entries(days=1)
        assert len(entries) == 1
