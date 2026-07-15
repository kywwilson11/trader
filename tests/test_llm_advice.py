"""Tests for the LLM conviction-gate rich-context + offline prompt A/B
harness additions:

  - llm_analyst.build_compact_evidence / rich_context_enabled
  - llm_analyst._build_prompt(include_pred=...)
  - llm_analyst.analyze_trades(system_prompt=, include_pred=, persist=,
    model_override=) plumbing + _journal_replay
  - events_calendar.next_earnings_date (cache-read only, no network)
  - llm_eval.realize_scored_rows
  - scripts/prompt_ab.py pure parts (loader/dedup/resumability/pairing/verdict)
  - llm_config.py flag defaults

Mac-runnable: stdlib + numpy only, no torch/alpaca imports.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import llm_analyst
from llm_analyst import (build_compact_evidence, rich_context_enabled,
                         _build_prompt, analyze_trades)
import events_calendar
import llm_eval
import llm_config
import prompt_ab


# --------------------------------------------------------------------------- #
# 1. build_compact_evidence
# --------------------------------------------------------------------------- #

class TestBuildCompactEvidence:
    FULL_SNAPSHOT = {
        'Close': 123.45, 'Return_4h': 1.2, 'Return_12h': -0.5,
        'Volatility_12h': 2.3, 'ATR_Pct': 1.1, 'RSI': 28.0,
        'Price_SMA20_Ratio': 1.02, 'BBP_20_2.0': 0.15,
        'Volume_Ratio': 1.8, 'Hurst': 0.55, 'RS_vs_SPY': 0.8,
    }
    FULL_FUND = {
        'pe_ratio': 25.0, 'pb_ratio': 5.0, 'market_cap': 3.0e12,
        'revenue_growth': 0.12, 'beta': 1.3, 'sector': 'Technology',
        'week52_high': 150.0, 'week52_low': 100.0,
    }

    @pytest.fixture(autouse=True)
    def _no_earnings_cache(self, monkeypatch):
        # Deterministic: no earnings-calendar entries, and never hits disk.
        monkeypatch.setattr(events_calendar, '_load_cache', lambda: {})

    def test_full_block_contains_expected_fields(self):
        block = build_compact_evidence(
            'AAPL', self.FULL_SNAPSHOT, self.FULL_FUND,
            position={'qty': 10, 'entry_price': 100.0}, asset_type='stock')
        assert block is not None
        assert block.startswith("Quantitative snapshot (last CLOSED hourly bar):")
        assert 'Ret4h +1.20%' in block
        assert 'Vol12h 2.30%' in block
        assert 'RSI 28' in block
        assert 'OVERSOLD' in block
        assert 'RS vs SPY +0.80%' in block
        assert 'P/E 25.0' in block
        assert 'P/B 5.0' in block
        assert 'MktCap $3.0T' in block
        assert 'RevGrowth +12.0%' in block
        assert 'Beta 1.30' in block
        assert 'Sector Technology' in block
        assert '52w-pos' in block
        assert 'OPEN POSITION' in block
        assert '+23.45% unrealized' in block
        # Deliberately never restates the ML prediction — must not become
        # a second echo channel (findings 1/9 of the review).
        assert 'ML model' not in block
        assert 'prediction' not in block.lower()

    def test_empty_snapshot_returns_none(self):
        assert build_compact_evidence('AAPL', None, self.FULL_FUND) is None
        assert build_compact_evidence('AAPL', {}, self.FULL_FUND) is None

    def test_crypto_all_none_fundamentals_no_valuation_no_crash(self):
        crypto_fund = {
            "pe_ratio": None, "pb_ratio": None, "market_cap": None,
            "revenue_growth": None, "eps": None, "dividend_yield": None,
            "week52_high": None, "week52_low": None, "sector": None,
            "beta": None, "avg_volume": None,
        }
        snapshot = {'Close': 50000.0, 'BTC_RSI': 60.0, 'BTC_SMA_Ratio': 1.01,
                   'BTC_Return_1h': 0.3}
        block = build_compact_evidence('BTC/USD', snapshot, crypto_fund,
                                       asset_type='crypto')
        assert block is not None
        assert 'P/E' not in block
        assert 'MktCap' not in block
        assert 'Sector' not in block
        assert 'BTC RSI 60' in block

    def test_nan_fields_skipped(self):
        snapshot = {'Close': 100.0, 'RSI': float('nan'), 'Return_4h': 1.0}
        block = build_compact_evidence('X', snapshot, None, asset_type='stock')
        assert block is not None
        assert 'RSI' not in block
        assert 'Ret4h' in block

    def test_no_fundamentals_and_no_position_still_works(self):
        block = build_compact_evidence('X', {'Close': 10.0}, None,
                                       position=None, asset_type='stock')
        assert block is not None
        assert 'OPEN POSITION' not in block

    def test_length_capped_at_600(self):
        block = build_compact_evidence(
            'AAPL', self.FULL_SNAPSHOT, self.FULL_FUND,
            position={'qty': 100, 'entry_price': 90.0}, asset_type='stock')
        assert block is not None
        assert len(block) <= 600


# --------------------------------------------------------------------------- #
# 2. _build_prompt(include_pred=...)
# --------------------------------------------------------------------------- #

class TestBuildPromptIncludePred:
    def test_include_pred_false_omits_ml_line(self):
        candidates = [{"symbol": "TSLA", "pred_return": 0.42}]
        prompt = _build_prompt(candidates, "stock", 0, None, None, {},
                               include_pred=False)
        assert "ML model prediction" not in prompt
        assert "+0.4200" not in prompt

    def test_include_pred_default_true_includes_line(self):
        """Guards against regressing the pre-existing behavior (also pinned
        by tests/test_llm_analyst.py::test_includes_ml_prediction)."""
        candidates = [{"symbol": "TSLA", "pred_return": 0.42}]
        prompt = _build_prompt(candidates, "stock", 0, None, None, {})
        assert "ML model prediction" in prompt
        assert "+0.4200" in prompt


# --------------------------------------------------------------------------- #
# 3. analyze_trades plumbing
# --------------------------------------------------------------------------- #

class TestAnalyzeTradesPlumbing:
    @staticmethod
    def _config(**overrides):
        cfg = {
            "enabled": True,
            "advisor_v2_enabled": False,
            "analyst_dedup_ttl_sec": 0,
            "replay_capture_enabled": True,
        }
        cfg.update(overrides)
        return cfg

    def test_system_prompt_and_model_override_honored(self, monkeypatch, tmp_path):
        captured = {}

        def fake_call_model(prompt, system="", model="", max_tokens=2048,
                            json_schema=None, temperature=None, timeout=None,
                            json_mode=False):
            captured['system'] = system
            captured['model'] = model
            return json.dumps({"TSLA": {"s": 0.6, "bull": "b", "bear": "b", "r": "r"}})

        monkeypatch.setattr(llm_analyst, "call_model", fake_call_model)
        monkeypatch.setattr(llm_analyst, "load_llm_config",
                            lambda: self._config())
        monkeypatch.setattr(llm_analyst, "_ANALYSIS_FILE",
                            tmp_path / "llm_analysis.json")
        monkeypatch.setattr(llm_analyst, "_REPLAY_DIR",
                            tmp_path / "llm_replay")

        candidates = [{"symbol": "TSLA", "pred_return": 0.1}]
        result = analyze_trades(candidates, "stock",
                                system_prompt="CUSTOM SYSTEM TEXT",
                                model_override="claude-haiku-4-5")
        assert result["TSLA"]["s"] == 0.6
        assert captured['system'] == "CUSTOM SYSTEM TEXT"
        assert captured['model'] == "claude-haiku-4-5"

    def test_persist_false_leaves_analysis_file_untouched_and_no_replay(
            self, monkeypatch, tmp_path):
        def fake_call_model(*a, **k):
            return json.dumps({"TSLA": {"s": 0.6, "bull": "b", "bear": "b", "r": "r"}})

        monkeypatch.setattr(llm_analyst, "call_model", fake_call_model)
        monkeypatch.setattr(llm_analyst, "load_llm_config",
                            lambda: self._config())
        analysis_file = tmp_path / "llm_analysis.json"
        replay_dir = tmp_path / "llm_replay"
        monkeypatch.setattr(llm_analyst, "_ANALYSIS_FILE", analysis_file)
        monkeypatch.setattr(llm_analyst, "_REPLAY_DIR", replay_dir)

        candidates = [{"symbol": "TSLA", "pred_return": 0.1}]
        result = analyze_trades(candidates, "stock", persist=False)

        assert result["TSLA"]["s"] == 0.6
        assert not analysis_file.exists()
        assert not replay_dir.exists() or not list(replay_dir.glob("*.jsonl"))

    def test_persist_true_writes_one_roundtrippable_replay_record(
            self, monkeypatch, tmp_path):
        def fake_call_model(*a, **k):
            return json.dumps({"TSLA": {"s": 0.72, "bull": "b", "bear": "b", "r": "r"}})

        monkeypatch.setattr(llm_analyst, "call_model", fake_call_model)
        monkeypatch.setattr(llm_analyst, "load_llm_config",
                            lambda: self._config(replay_capture_enabled=True))
        analysis_file = tmp_path / "llm_analysis.json"
        replay_dir = tmp_path / "llm_replay"
        monkeypatch.setattr(llm_analyst, "_ANALYSIS_FILE", analysis_file)
        monkeypatch.setattr(llm_analyst, "_REPLAY_DIR", replay_dir)

        candidates = [{"symbol": "TSLA", "pred_return": 0.33,
                      "fundamentals_text": "P/E=20", "news_headlines": ["hi"]}]
        result = analyze_trades(
            candidates, "stock", equity=1000, positions=["TSLA"],
            position_details={"TSLA": {"qty": 1, "entry_price": 10}},
            fng_value=40, model_config={"forward_bars": 24}, persist=True)

        assert result["TSLA"]["s"] == 0.72
        assert analysis_file.exists()

        files = list(replay_dir.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["asset_type"] == "stock"
        assert record["forward_bars"] == 24
        assert record["live_scores"] == {"TSLA": 0.72}
        assert record["candidates"][0]["symbol"] == "TSLA"
        assert record["positions"] == ["TSLA"]

        # Round-trippable by the harness loader.
        cycles = prompt_ab.load_replay_cycles(days=3650, replay_dir=replay_dir)
        assert len(cycles) == 1
        assert cycles[0]["live_scores"]["TSLA"] == 0.72

    def test_replay_capture_disabled_by_config(self, monkeypatch, tmp_path):
        def fake_call_model(*a, **k):
            return json.dumps({"TSLA": {"s": 0.6, "bull": "b", "bear": "b", "r": "r"}})

        monkeypatch.setattr(llm_analyst, "call_model", fake_call_model)
        monkeypatch.setattr(llm_analyst, "load_llm_config",
                            lambda: self._config(replay_capture_enabled=False))
        replay_dir = tmp_path / "llm_replay"
        monkeypatch.setattr(llm_analyst, "_ANALYSIS_FILE",
                            tmp_path / "llm_analysis.json")
        monkeypatch.setattr(llm_analyst, "_REPLAY_DIR", replay_dir)

        candidates = [{"symbol": "TSLA", "pred_return": 0.1}]
        analyze_trades(candidates, "stock", persist=True)
        assert not replay_dir.exists() or not list(replay_dir.glob("*.jsonl"))


# --------------------------------------------------------------------------- #
# 4. events_calendar.next_earnings_date
# --------------------------------------------------------------------------- #

class TestNextEarningsDate:
    def test_returns_nearest_future_date(self, monkeypatch):
        import datetime as dt
        today = dt.date.today()
        cache = {
            'by_symbol': {
                'AAPL': [
                    {'date': (today - dt.timedelta(days=5)).isoformat(), 'hour': 'amc'},
                    {'date': (today + dt.timedelta(days=10)).isoformat(), 'hour': 'bmo'},
                    {'date': (today + dt.timedelta(days=3)).isoformat(), 'hour': 'amc'},
                ]
            }
        }
        monkeypatch.setattr(events_calendar, '_load_cache', lambda: cache)
        result = events_calendar.next_earnings_date('AAPL')
        assert result == (today + dt.timedelta(days=3)).isoformat()

    def test_no_future_dates_returns_none(self, monkeypatch):
        import datetime as dt
        today = dt.date.today()
        cache = {'by_symbol': {'AAPL': [
            {'date': (today - dt.timedelta(days=5)).isoformat()}]}}
        monkeypatch.setattr(events_calendar, '_load_cache', lambda: cache)
        assert events_calendar.next_earnings_date('AAPL') is None

    def test_unknown_symbol_returns_none(self, monkeypatch):
        monkeypatch.setattr(events_calendar, '_load_cache', lambda: {'by_symbol': {}})
        assert events_calendar.next_earnings_date('ZZZZ') is None

    def test_no_network_call(self, monkeypatch):
        """Cache-read only — must never call refresh_if_stale (which can fetch)."""
        monkeypatch.setattr(events_calendar, '_load_cache', lambda: {'by_symbol': {}})

        def _boom():
            raise AssertionError("refresh_if_stale must not be called")

        monkeypatch.setattr(events_calendar, 'refresh_if_stale', _boom)
        assert events_calendar.next_earnings_date('AAPL') is None


# --------------------------------------------------------------------------- #
# 5. scripts/prompt_ab.py pure parts
# --------------------------------------------------------------------------- #

class TestPromptAbPureParts:
    def test_load_replay_cycles_dedup_and_days_filter(self, tmp_path):
        from datetime import datetime, timezone, timedelta
        replay_dir = tmp_path / "llm_replay"
        replay_dir.mkdir()
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=100)).isoformat()
        new_ts = now.isoformat()
        recs = [
            {"ts": new_ts, "asset_type": "stock", "candidates": [{"symbol": "A"}]},
            {"ts": new_ts, "asset_type": "stock", "candidates": [{"symbol": "A"}]},  # dup
            {"ts": old_ts, "asset_type": "stock", "candidates": [{"symbol": "B"}]},  # too old
        ]
        f = replay_dir / f"{now.date().isoformat()}.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")

        cycles = prompt_ab.load_replay_cycles(days=14, replay_dir=replay_dir)
        assert len(cycles) == 1
        assert cycles[0]["ts"] == new_ts

    def test_load_replay_cycles_asset_filter(self, tmp_path):
        from datetime import datetime, timezone
        replay_dir = tmp_path / "llm_replay"
        replay_dir.mkdir()
        now = datetime.now(timezone.utc)
        recs = [
            {"ts": now.isoformat(), "asset_type": "stock", "candidates": []},
            {"ts": now.isoformat(), "asset_type": "crypto", "candidates": []},
        ]
        f = replay_dir / f"{now.date().isoformat()}.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")

        cycles = prompt_ab.load_replay_cycles(days=1, asset_filter="crypto",
                                              replay_dir=replay_dir)
        assert len(cycles) == 1
        assert cycles[0]["asset_type"] == "crypto"

    def test_load_replay_cycles_missing_dir_returns_empty(self, tmp_path):
        assert prompt_ab.load_replay_cycles(days=14,
                                            replay_dir=tmp_path / "nope") == []

    def test_load_existing_pairs_resumability(self, tmp_path):
        out = tmp_path / "scores.jsonl"
        rows = [{"t0": 1.0, "symbol": "A"}, {"t0": 2.0, "symbol": "B"}]
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        pairs = prompt_ab.load_existing_pairs(out)
        assert pairs == {(1.0, "A"), (2.0, "B")}

    def test_load_existing_pairs_missing_file_returns_empty(self, tmp_path):
        assert prompt_ab.load_existing_pairs(tmp_path / "nope.jsonl") == set()

    def test_pair_variant_samples(self):
        rows = [
            {"symbol": "A", "s_a": 0.6, "s_b": 0.65},
            {"symbol": "B", "s_a": 0.1, "s_b": 0.6},   # flips across 0.5; s_a<veto
            {"symbol": "C", "s_a": 0.5, "s_b": None},  # b missing -> not paired
        ]
        realized_tuples = [
            (None, 1.0, 0.3, 100.0),
            (None, -2.0, -0.1, 101.0),
            (None, 0.5, 0.0, 102.0),
        ]
        built = prompt_ab.pair_variant_samples(rows, realized_tuples,
                                               veto_threshold=0.15)
        assert len(built["samples_a"]) == 3
        assert len(built["samples_b"]) == 2
        paired = built["paired"]
        assert paired["n_paired"] == 2
        assert paired["n_flips_across_0.5"] == 1
        assert paired["veto_rate_a"] == pytest.approx(1 / 3, abs=1e-4)
        assert paired["veto_rate_b"] == 0.0
        assert paired["mean_abs_delta_s"] == pytest.approx(0.275, abs=1e-9)

    def test_build_variant_b_candidates_copies_without_profile_warning(self, capsys):
        candidates = [{"symbol": "A", "pred_return": 0.1}]
        warn_once = []
        out = prompt_ab.build_variant_b_candidates(candidates, True, warn_once)
        assert out == candidates
        assert out is not candidates
        captured = capsys.readouterr()
        assert "no captured profile" in captured.out
        assert len(warn_once) == 1

    def test_build_variant_b_candidates_keeps_captured_profile(self, capsys):
        candidates = [{"symbol": "A", "pred_return": 0.1, "profile": "Quant snapshot"}]
        warn_once = []
        out = prompt_ab.build_variant_b_candidates(candidates, True, warn_once)
        assert out[0]["profile"] == "Quant snapshot"
        captured = capsys.readouterr()
        assert "no captured profile" not in captured.out

    def _report(self, n, b2=None, p=None, echo=None, insufficient=False):
        r = {"n": n}
        if insufficient:
            r["insufficient_power"] = True
            return r
        r["encompassing"] = {"b2_s": b2, "p_value": p}
        r["echo_gap"] = echo
        return r

    def test_decide_adopt_insufficient_power_abstains(self):
        a = self._report(30, 0.1, 0.01, 0.05)
        b = self._report(30, 0.2, 0.01, 0.02)
        verdict = prompt_ab.decide_adopt(a, b, min_n=60)
        assert "insufficient_power" in verdict

    def test_decide_adopt_adopts_b(self):
        a = self._report(100, 0.05, 0.5, 0.10)   # not significant
        b = self._report(100, 0.20, 0.01, 0.05)  # significant, better, echo not worse
        verdict = prompt_ab.decide_adopt(a, b, min_n=60)
        assert verdict.startswith("ADOPT B")

    def test_decide_adopt_keeps_a_when_b_not_significant(self):
        a = self._report(100, 0.05, 0.5, 0.10)
        b = self._report(100, 0.02, 0.5, 0.10)
        verdict = prompt_ab.decide_adopt(a, b, min_n=60)
        assert verdict.startswith("KEEP A")

    def test_decide_adopt_keeps_a_when_echo_gap_worsens(self):
        a = self._report(100, 0.05, 0.5, 0.05)   # not significant
        b = self._report(100, 0.20, 0.01, 0.20)  # significant but worse echo
        verdict = prompt_ab.decide_adopt(a, b, min_n=60)
        assert verdict.startswith("KEEP A")

    def test_decide_adopt_keeps_a_when_a_already_significant_and_better(self):
        a = self._report(100, 0.30, 0.001, 0.02)  # already significant, higher b2
        b = self._report(100, 0.20, 0.01, 0.02)
        verdict = prompt_ab.decide_adopt(a, b, min_n=60)
        assert verdict.startswith("KEEP A")


class TestPromptAbRunFailOpen:
    """cmd_run must not burn (t0, symbol) resumability pairs when
    analyze_trades fails OPEN by returning {} for both variants — those
    cycles have to stay retryable on the next run."""

    @staticmethod
    def _write_cycle(replay_dir):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        rec = {"ts": now.isoformat(), "asset_type": "stock",
               "forward_bars": 24, "equity": 1000, "positions": [],
               "position_details": {}, "fng": 50,
               "candidates": [{"symbol": "AAPL", "pred_return": 0.1}]}
        (replay_dir / f"{now.date().isoformat()}.jsonl").write_text(
            json.dumps(rec) + "\n")

    @staticmethod
    def _args(out_path):
        from types import SimpleNamespace
        return SimpleNamespace(days=14, asset=None, system_b=None,
                               hide_pred_b=False, rich_context_b=False,
                               model="pinned", max_cycles=None, sleep_sec=0.0,
                               out=str(out_path), dry_run=False)

    def test_empty_both_variants_writes_nothing(self, tmp_path, monkeypatch):
        replay_dir = tmp_path / "llm_replay"
        replay_dir.mkdir()
        self._write_cycle(replay_dir)
        monkeypatch.setattr(prompt_ab, "REPLAY_DIR", replay_dir)
        monkeypatch.setattr(llm_analyst, "analyze_trades",
                            lambda *a, **k: {})  # provider fail-open
        out_path = tmp_path / "scores.jsonl"
        prompt_ab.cmd_run(self._args(out_path))
        assert not out_path.exists() or out_path.read_text() == ""

    def test_scored_cycle_writes_rows_with_persist_false(self, tmp_path,
                                                         monkeypatch):
        replay_dir = tmp_path / "llm_replay"
        replay_dir.mkdir()
        self._write_cycle(replay_dir)
        monkeypatch.setattr(prompt_ab, "REPLAY_DIR", replay_dir)
        persist_seen = set()

        def stub(candidates, asset_type, **kw):
            persist_seen.add(kw.get("persist"))
            return {c["symbol"]: {"s": 0.6, "m": 1.0, "r": "r",
                                  "bull": "b", "bear": "b"}
                    for c in candidates}

        monkeypatch.setattr(llm_analyst, "analyze_trades", stub)
        out_path = tmp_path / "scores.jsonl"
        prompt_ab.cmd_run(self._args(out_path))
        rows = [json.loads(l) for l in out_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["s_a"] == 0.6 and rows[0]["s_b"] == 0.6
        assert rows[0]["model"] == "pinned"
        # The harness must NEVER touch live state.
        assert persist_seen == {False}


# --------------------------------------------------------------------------- #
# 6. llm_eval.realize_scored_rows
# --------------------------------------------------------------------------- #

class TestRealizeScoredRows:
    def test_grouping_and_tuples(self, monkeypatch):
        import numpy as np

        calls = []

        def fake_bars_lookup(api, symbol, asset_type, start, end):
            calls.append((symbol, asset_type))
            base = 1_700_000_000.0
            ts = base + np.arange(0, 100) * 3600.0
            closes = 100.0 + np.arange(100) * 0.5
            return ts, closes

        monkeypatch.setattr(llm_eval, "_bars_lookup", fake_bars_lookup)

        base = 1_700_000_000.0
        rows = [
            {"symbol": "AAPL", "asset_type": "stock", "t0": base + 3600,
             "horizon": 24, "s": 0.6, "pred": 0.1},
            {"symbol": "AAPL", "asset_type": "stock", "t0": base + 7200,
             "horizon": 24, "s": 0.4, "pred": -0.2},
            {"symbol": "MSFT", "asset_type": "stock", "t0": base + 3600,
             "horizon": 24, "s": 0.7, "pred": 0.3},
        ]
        out = llm_eval.realize_scored_rows(rows, api=object())

        assert len(out) == 3
        # One _bars_lookup call per (symbol, asset_type) group.
        assert sorted(calls) == [("AAPL", "stock"), ("MSFT", "stock")]
        assert len(calls) == 2

        for (s, realized, pred, t0), row in zip(out, rows):
            assert s == row["s"]
            assert pred == row["pred"]
            assert t0 == row["t0"]
            assert realized is not None

    def test_missing_bars_yields_none_realized(self, monkeypatch):
        import numpy as np

        def empty_bars_lookup(api, symbol, asset_type, start, end):
            return np.array([]), np.array([])

        monkeypatch.setattr(llm_eval, "_bars_lookup", empty_bars_lookup)
        rows = [{"symbol": "AAPL", "asset_type": "stock", "t0": 1_700_000_000.0,
                "horizon": 24, "s": 0.5, "pred": 0.0}]
        out = llm_eval.realize_scored_rows(rows, api=object())
        assert out == [(0.5, None, 0.0, 1_700_000_000.0)]


# --------------------------------------------------------------------------- #
# 7. Flag defaults
# --------------------------------------------------------------------------- #

class TestFlagDefaults:
    def test_rich_context_and_replay_capture_defaults(self, tmp_path, monkeypatch):
        fake_path = tmp_path / "nonexistent_llm_config.json"
        monkeypatch.setattr(llm_config, "LLM_CONFIG_FILE", fake_path)
        cfg = llm_config.load_llm_config()
        assert cfg["rich_context_enabled"] is False
        assert cfg["replay_capture_enabled"] is True

    def test_defaults_dict_has_the_keys(self):
        assert llm_config._DEFAULTS["rich_context_enabled"] is False
        assert llm_config._DEFAULTS["replay_capture_enabled"] is True

    def test_rich_context_enabled_accessor_fail_soft(self, monkeypatch):
        def _boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(llm_analyst, "load_llm_config", _boom)
        assert rich_context_enabled() is False

    def test_rich_context_enabled_reads_config(self, monkeypatch):
        monkeypatch.setattr(llm_analyst, "load_llm_config",
                            lambda: {"rich_context_enabled": True})
        assert rich_context_enabled() is True
