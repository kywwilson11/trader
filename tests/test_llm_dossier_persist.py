"""Tests for llm_analyst.py's Phase 2.1 fix (2026-07 GUI review §5/§11):
the advisor-v2 decision dossier (p_up, conviction, abstain, key_risks,
event_flags) computed by analyze_trades used to be discarded at
_save_analysis — persisting only m/s/r/bull/bear/timestamp/model — so the
GUI could never show the advisor's actual decision payload. This pins the
producer-side fix: _save_analysis now carries the dossier onto disk when
analyze_trades supplied it, byte-identically omits it when it didn't (the
default advisor_v2_enabled=False path), and bounds list/string growth.

llm_analyst.py imports no torch/lightgbm/etc at module level and happens
to import cleanly on this Mac today, but pytest.importorskip guards it
anyway per repo convention (see tests/test_command_ack.py), so this module
SKIPS (not errors) if that ever stops being true here or in CI.
"""
import copy
import json

import pytest

llm_analyst = pytest.importorskip("llm_analyst")
llm_config = pytest.importorskip("llm_config")


def _patch_path(monkeypatch, tmp_path):
    target = tmp_path / "llm_analysis.json"
    monkeypatch.setattr(llm_analyst, "_ANALYSIS_FILE", target)
    return target


# --------------------------------------------------------------------------- #
# _bounded_str_list — the file-growth-bound helper
# --------------------------------------------------------------------------- #

class TestBoundedStrList:
    def test_caps_element_count_and_string_length(self):
        items = ["x" * 500 for _ in range(9)]  # 9 oversized "risks"
        out = llm_analyst._bounded_str_list(items)
        assert len(out) == 5
        assert all(len(x) == 300 for x in out)

    def test_short_list_passes_through_unchanged(self):
        assert llm_analyst._bounded_str_list(["a", "b"]) == ["a", "b"]

    def test_dict_input_persists_its_values(self):
        # event_flags is documented as "list[str] or dict" by the task spec
        # even though _parse_response today only ever produces a list —
        # defend the persistence layer against that shape too.
        out = llm_analyst._bounded_str_list({"earnings": "earnings_within_3d"})
        assert out == ["earnings_within_3d"]

    def test_none_and_other_garbage_returns_empty_list(self):
        assert llm_analyst._bounded_str_list(None) == []
        assert llm_analyst._bounded_str_list(42) == []
        assert llm_analyst._bounded_str_list("not-a-list") == []


# --------------------------------------------------------------------------- #
# _save_analysis — the producer-side persistence fix
# --------------------------------------------------------------------------- #

class TestSaveAnalysisDossierPersistence:
    """Unit-level: drive _save_analysis directly with hand-built `result`
    dicts, mirroring the existing TestSaveAnalysisAtomicWrite /
    TestSaveAnalysisDefensiveEntryAccess classes in test_llm_analyst.py."""

    def test_v2_fields_persisted_when_present_on_entry(self, monkeypatch, tmp_path):
        target = _patch_path(monkeypatch, tmp_path)
        result = {
            "TSLA": {
                "m": 0.9, "s": 0.6, "r": "synthesis",
                "bull": "bull case", "bear": "bear case",
                "p_up": 0.63, "conviction": 4, "abstain": False,
                "key_risks": ["competition", "valuation"],
                "event_flags": ["earnings_within_3d"],
            },
        }
        llm_analyst._save_analysis(result, "stock", "test-model")

        with open(target) as f:
            data = json.load(f)
        entry = data["stock"]["TSLA"]
        # Original 7 keys untouched
        assert entry["m"] == 0.9 and entry["s"] == 0.6
        assert entry["bull"] == "bull case" and entry["bear"] == "bear case"
        # New dossier fields now reach disk
        assert entry["p_up"] == 0.63
        assert entry["conviction"] == 4
        assert entry["abstain"] is False
        assert entry["key_risks"] == ["competition", "valuation"]
        assert entry["event_flags"] == ["earnings_within_3d"]
        assert entry["prompt_version"] == llm_analyst.PROMPT_VERSION_V2

    def test_v1_path_keeps_exact_legacy_seven_keys(self, monkeypatch, tmp_path):
        """Byte-compat: an entry with no p_up (the default/v1 schema shape)
        must persist with EXACTLY the original 7 keys — no v2 key appears
        even as None. Companion to
        test_llm_advisor.py::test_analyze_trades_default_llm_analysis_json_keys
        (which exercises this through the full analyze_trades path)."""
        target = _patch_path(monkeypatch, tmp_path)
        result = {
            "AAPL": {"m": 0.9, "s": 0.6, "r": "r", "bull": "b", "bear": "b"},
        }
        llm_analyst._save_analysis(result, "stock", "test-model")

        with open(target) as f:
            data = json.load(f)
        entry = data["stock"]["AAPL"]
        assert set(entry.keys()) == {"m", "s", "r", "bull", "bear",
                                     "timestamp", "model"}

    def test_unparseable_v2_field_persists_as_none_never_fabricated(
            self, monkeypatch, tmp_path):
        """A field present-but-unparseable upstream (_parse_response already
        reduced it to None) must be persisted as None — never invented,
        never silently dropped either (the key stays present)."""
        target = _patch_path(monkeypatch, tmp_path)
        result = {
            "ZQZQ": {
                "m": 0.9, "s": 0.6, "r": "r", "bull": "b", "bear": "b",
                "p_up": None, "conviction": None, "abstain": True,
                "key_risks": [], "event_flags": [],
            },
        }
        llm_analyst._save_analysis(result, "crypto", "test-model")

        with open(target) as f:
            data = json.load(f)
        entry = data["crypto"]["ZQZQ"]
        assert entry["p_up"] is None
        assert entry["conviction"] is None
        assert entry["abstain"] is True
        assert entry["key_risks"] == []
        assert entry["event_flags"] == []

    def test_event_flags_capped_at_five_key_risks_string_capped_at_300(
            self, monkeypatch, tmp_path):
        """Defense-in-depth: _parse_response already caps key_risks at 3
        entries of <=100 chars, but event_flags has no upstream length cap
        beyond the 12-token vocabulary, so a >5 list is the realistic case
        this guards against file growth."""
        target = _patch_path(monkeypatch, tmp_path)
        many_flags = ["earnings_within_3d", "post_earnings", "overnight_block",
                      "fomc_today", "cpi_today", "macro_standdown", "regulatory"]
        assert len(many_flags) > 5
        result = {
            "BTC/USD": {
                "m": 0.9, "s": 0.6, "r": "r", "bull": "b", "bear": "b",
                "p_up": 0.5, "conviction": 3, "abstain": False,
                "key_risks": ["r" * 500],
                "event_flags": many_flags,
            },
        }
        llm_analyst._save_analysis(result, "crypto", "test-model")

        with open(target) as f:
            data = json.load(f)
        entry = data["crypto"]["BTC/USD"]
        assert len(entry["event_flags"]) == 5
        assert len(entry["key_risks"][0]) == 300

    def test_persisted_dossier_is_json_serializable_round_trip(
            self, monkeypatch, tmp_path):
        """Guards the atomic-write path itself: a bad type in a new field
        must not raise inside json.dump and corrupt the whole file for
        every OTHER symbol in the same section."""
        target = _patch_path(monkeypatch, tmp_path)
        result = {
            "GOOD": {"m": 0.9, "s": 0.6, "r": "r", "bull": "b", "bear": "b"},
            "ALSO_GOOD": {
                "m": 0.9, "s": 0.6, "r": "r", "bull": "b", "bear": "b",
                "p_up": 0.4, "conviction": 2, "abstain": False,
                "key_risks": ["thin liquidity"], "event_flags": [],
            },
        }
        llm_analyst._save_analysis(result, "stock", "test-model")

        with open(target) as f:
            data = json.load(f)
        assert "GOOD" in data["stock"] and "ALSO_GOOD" in data["stock"]


# --------------------------------------------------------------------------- #
# End-to-end wiring: analyze_trades -> _parse_response -> _save_analysis
# --------------------------------------------------------------------------- #

class TestAnalyzeTradesEndToEndDossierWiring:
    """One integration test proving the dossier actually flows through the
    real call chain (not just _save_analysis exercised in isolation above)
    — this is the regression guard if a future change alters the `entry`
    dict shape _save_analysis relies on ("p_up" in entry) to detect a v2
    call."""

    def test_advisor_v2_dossier_reaches_disk_end_to_end(self, monkeypatch, tmp_path):
        target = _patch_path(monkeypatch, tmp_path)
        cfg = copy.deepcopy(llm_config._DEFAULTS)
        cfg["advisor_v2_enabled"] = True
        cfg["replay_capture_enabled"] = False  # _journal_replay writes a
        # real, non-redirectable journals/llm_replay path — keep it off
        # (matches tests/test_llm_advisor.py's _cfg() convention).
        monkeypatch.setattr(llm_analyst, "load_llm_config", lambda: cfg)
        monkeypatch.setattr(llm_analyst, "get_recommended_model",
                            lambda role: "test-model")
        monkeypatch.setattr(llm_analyst, "get_last_model_used", lambda: None)
        # The shadow journal row is a separate concern from disk
        # persistence (this file's scope) and calls trade_journal.log_decision
        # under a config reference this test doesn't otherwise patch —
        # no-op it so this test can never write into the real journals/ dir.
        monkeypatch.setattr(llm_analyst, "_journal_advisor_shadow",
                            lambda *a, **k: None)

        response_obj = {
            "ZQZQ": {"s": 0.6, "bull": "b", "bear": "be", "r": "r",
                     "p_up": 0.71, "conviction": 5, "abstain": False,
                     "key_risks": ["thin volume"],
                     "event_flags": ["earnings_within_3d"]},
        }
        body = json.dumps(response_obj)

        def fake_call_model(prompt, system="", model="test-model",
                            max_tokens=0, json_schema=None,
                            temperature=0.0, timeout=30):
            return body

        monkeypatch.setattr(llm_analyst, "call_model", fake_call_model)

        candidates = [{"symbol": "ZQZQ", "pred_return": 0.5,
                      "fundamentals_text": "", "news_headlines": []}]
        result = llm_analyst.analyze_trades(candidates, "crypto")

        # Sanity: the v2 dossier really is on the in-memory return value.
        assert result["ZQZQ"]["p_up"] == 0.71

        with open(target) as f:
            data = json.load(f)
        entry = data["crypto"]["ZQZQ"]
        assert entry["p_up"] == 0.71
        assert entry["conviction"] == 5
        assert entry["abstain"] is False
        assert entry["key_risks"] == ["thin volume"]
        assert entry["event_flags"] == ["earnings_within_3d"]
        assert entry["prompt_version"] == llm_analyst.PROMPT_VERSION_V2

    def test_default_config_omits_dossier_end_to_end(self, monkeypatch, tmp_path):
        """Companion negative case: with advisor_v2_enabled at its shipped
        default (False), the full analyze_trades path persists exactly the
        legacy 7 keys — no plumbing change altered the default path."""
        target = _patch_path(monkeypatch, tmp_path)
        cfg = copy.deepcopy(llm_config._DEFAULTS)
        cfg["replay_capture_enabled"] = False
        monkeypatch.setattr(llm_analyst, "load_llm_config", lambda: cfg)
        monkeypatch.setattr(llm_analyst, "get_recommended_model",
                            lambda role: "test-model")
        monkeypatch.setattr(llm_analyst, "get_last_model_used", lambda: None)

        response_obj = {"ZQZQ": {"s": 0.6, "bull": "b", "bear": "be", "r": "r"}}
        body = json.dumps(response_obj)

        def fake_call_model(prompt, system="", model="test-model",
                            max_tokens=0, json_schema=None,
                            temperature=0.0, timeout=30):
            return body

        monkeypatch.setattr(llm_analyst, "call_model", fake_call_model)

        candidates = [{"symbol": "ZQZQ", "pred_return": 0.5,
                      "fundamentals_text": "", "news_headlines": []}]
        llm_analyst.analyze_trades(candidates, "crypto")

        with open(target) as f:
            data = json.load(f)
        entry = data["crypto"]["ZQZQ"]
        assert set(entry.keys()) == {"m", "s", "r", "bull", "bear",
                                     "timestamp", "model"}
