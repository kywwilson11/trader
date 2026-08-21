"""Tests for the llm_eval.py v3 hardening pass:

  - W1 loader consolidation (_load_entries_by_action) + OSError hardening
    + the needle prefilter's superset property.
  - W2/W3 _realized_forward_return's NaN guard + 4-tuple diagnostics, and
    realize_scored_rows' tz-aware bar window + diag_out plumbing.
  - W4/W5 run_eval's api injection + max(horizons) forward_bars selection.
  - W6 compute_incremental_report additions: n_input/n_dropped,
    time_ordered on mixed tuple arity, s_degenerate + ANTI-predictive
    verdict branches, echo_gap_abs/n_s_exactly_half, and the
    pseudo-replication caveat.
  - W7 compute_calibration_report's n_bins honoring, the n<5 vs n<min_n
    split (descriptive stats survive below the power floor), and the
    three-way length contract for conviction/abstain.
  - W8/W9 run_eval/advisor_report's coverage/meta/dedup/model/veto
    additions.

Mac-runnable: stdlib + numpy/scipy only, no Alpaca, no dotenv, no torch.
"""
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("scipy")

import llm_eval


# --------------------------------------------------------------------------- #
# Shared fixtures / helpers
# --------------------------------------------------------------------------- #

def _write_journal(tmp_path, date_str, rows):
    path = tmp_path / f"{date_str}.jsonl"
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _wide_hourly_ts(center_ts, days_before=3, days_after=10):
    """A clean hourly grid spanning [center_ts - days_before,
    center_ts + days_after] so any journaled t0/horizon within that window
    realizes against the stub."""
    ts_start = center_ts - days_before * 86400
    ts_end = center_ts + days_after * 86400
    n_bars = int((ts_end - ts_start) // 3600) + 1
    return ts_start + np.arange(n_bars) * 3600.0


# --------------------------------------------------------------------------- #
# 1. Loader (W1)
# --------------------------------------------------------------------------- #

def test_loader_oserror_skips_file(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    today_path = tmp_path / f"{today.isoformat()}.jsonl"
    yesterday_path = tmp_path / f"{yesterday.isoformat()}.jsonl"
    today_path.write_text(json.dumps(
        {"action": "llm_analysis", "ts": "2026-01-01T00:00:00+00:00"}) + "\n")
    yesterday_path.write_text(json.dumps(
        {"action": "llm_analysis", "ts": "2026-01-02T00:00:00+00:00"}) + "\n")

    os.chmod(today_path, 0o000)
    try:
        if os.access(today_path, os.R_OK):
            pytest.skip("chmod 0o000 did not block reads (root/CI quirk)")
        entries = llm_eval._load_entries(days=2)
        assert len(entries) == 1
        assert entries[0]["ts"] == "2026-01-02T00:00:00+00:00"
    finally:
        os.chmod(today_path, 0o644)


def test_loader_prefilter_is_superset(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
    today = datetime.now().date()
    rows = [
        # Contains the literal quoted needle '"llm_analysis"' (as a nested
        # JSON key) but action=='buy' -> the prefilter is a SUPERSET check
        # (it lets this line through), the exact e.get("action") test must
        # still exclude it.
        {"action": "buy", "ts": "2026-01-01T00:00:00+00:00",
         "notes": {"llm_analysis": "referenced but not this action"}},
        {"action": "llm_analysis", "ts": "2026-01-01T00:00:00+00:00",
         "scores": {}},
    ]
    _write_journal(tmp_path, today.isoformat(), rows)

    entries = llm_eval._load_entries(days=0)
    assert len(entries) == 1
    assert entries[0]["action"] == "llm_analysis"


# --------------------------------------------------------------------------- #
# 2. run_eval: api injection + max(horizons) + coverage/meta (W4/W5/W8)
# --------------------------------------------------------------------------- #

def test_forward_bars_uses_max(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(llm_eval, "BASE_DIR", tmp_path)

    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    _write_journal(tmp_path, today.isoformat(), [
        {"action": "llm_analysis", "ts": base.isoformat(), "asset_type": "crypto",
         "forward_bars": 24, "scores": {"BTC/USD": {"s": 0.6, "pred": 0.1}}}])
    _write_journal(tmp_path, yesterday.isoformat(), [
        {"action": "llm_analysis", "ts": base.isoformat(), "asset_type": "crypto",
         "forward_bars": 12, "scores": {"ETH/USD": {"s": 0.55, "pred": -0.05}}}])

    ts_full = _wide_hourly_ts(base.timestamp())
    closes_full = 100.0 + np.arange(len(ts_full)) * 0.001

    def fake_bars_lookup(api, symbol, asset_type, start, end):
        return ts_full, closes_full

    monkeypatch.setattr(llm_eval, "_bars_lookup", fake_bars_lookup)

    report = llm_eval.run_eval(days=2, api=object())
    assert report != {}
    assert report["meta"]["forward_bars_used"] == 24
    assert report["meta"]["forward_bars_seen"] == [12, 24]


def test_run_eval_empty_journal_no_crash_and_writes_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(llm_eval, "BASE_DIR", tmp_path)

    result = llm_eval.run_eval(days=1, api=object())
    assert result == {}

    out = tmp_path / "llm_eval_report.json"
    assert out.exists()
    stub = json.loads(out.read_text())
    assert stub["verdict"] == "no_data"


def test_bucket_edge_tracks_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_eval, "VETO_THRESHOLD", 0.25)
    monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(llm_eval, "BASE_DIR", tmp_path)

    today = datetime.now().date()
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    _write_journal(tmp_path, today.isoformat(), [
        {"action": "llm_analysis", "ts": base.isoformat(), "asset_type": "crypto",
         "forward_bars": 24, "scores": {"BTC/USD": {"s": 0.17, "pred": 0.05}}}])

    ts_full = _wide_hourly_ts(base.timestamp())
    closes_full = 100.0 + np.arange(len(ts_full)) * 0.001

    def fake_bars_lookup(api, symbol, asset_type, start, end):
        return ts_full, closes_full

    monkeypatch.setattr(llm_eval, "_bars_lookup", fake_bars_lookup)

    report = llm_eval.run_eval(days=1, api=object())
    assert report != {}
    assert "VETO" in report["buckets"]
    assert report["buckets"]["VETO"]["n"] == 1
    assert report["meta"]["veto_threshold"] == 0.25


def test_veto_counterfactual_mean_and_n(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(llm_eval, "BASE_DIR", tmp_path)

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    symbols = [f"SYM{i}" for i in range(10)]
    scores = {sym: {"s": 0.05, "pred": 0.1} for sym in symbols}
    today = datetime.now().date()
    _write_journal(tmp_path, today.isoformat(), [
        {"action": "llm_analysis", "ts": base.isoformat(), "asset_type": "crypto",
         "forward_bars": 24, "scores": scores}])

    t0 = base.timestamp()
    ts_full = _wide_hourly_ts(t0)
    closes_full = np.full(len(ts_full), 100.0)
    i0 = int(round((t0 - ts_full[0]) / 3600.0))
    closes_full[i0 + 24] = 99.0   # exactly -1.0% over the 24h horizon

    def fake_bars_lookup(api, symbol, asset_type, start, end):
        return ts_full, closes_full.copy()

    monkeypatch.setattr(llm_eval, "_bars_lookup", fake_bars_lookup)

    report = llm_eval.run_eval(days=1, api=object())
    assert report != {}
    assert report["veto_counterfactual"]["n"] == 10
    assert report["veto_counterfactual"]["avg_fwd_ret_pct"] == -1.0
    assert report["veto_counterfactual_pct"] == 10.0   # legacy key preserved


# --------------------------------------------------------------------------- #
# 3. realize_scored_rows / _realized_forward_return (W2/W3)
# --------------------------------------------------------------------------- #

def test_realize_window_is_utc_aware(monkeypatch):
    captured = {}

    def fake_bars_lookup(api, symbol, asset_type, start, end):
        captured["start"] = start
        captured["end"] = end
        return np.array([]), np.array([])

    monkeypatch.setattr(llm_eval, "_bars_lookup", fake_bars_lookup)

    min_t0 = 1_700_000_000.0
    rows = [{"symbol": "AAPL", "asset_type": "stock", "t0": min_t0,
            "horizon": 24, "s": 0.5, "pred": 0.1}]
    llm_eval.realize_scored_rows(rows, api=object())

    start = captured["start"]
    assert start.tzinfo is not None
    assert abs(start.timestamp() - (min_t0 - 7200)) < 1


def test_realize_empty_rows_needs_no_api():
    # Empty input must return [] WITHOUT resolving api=None via
    # trading_utils.get_api() (Jetson/Alpaca-gated — would raise on this
    # Mac), and must leave diag_out untouched.
    diag = []
    assert llm_eval.realize_scored_rows([], api=None, diag_out=diag) == []
    assert diag == []


def test_realized_return_nan_close_is_none():
    ts = np.array([0.0, 3600.0, 7200.0, 24 * 3600.0, 25 * 3600.0])
    closes = np.array([100.0, 101.0, 102.0, np.nan, 105.0])
    result = llm_eval._realized_forward_return(ts, closes, 0.0, 24)
    assert result[0] is None


def test_realized_return_weekend_gap_elapsed():
    # Friday hourly bars, then a ~64h weekend gap, then Monday bars — pins
    # the elapsed_hours diagnostic (RTH bars_spanned != wall-clock hours),
    # NOT a rejection: the return is still computed.
    fri = np.arange(4) * 3600.0
    mon_start = fri[-1] + 64 * 3600.0
    mon = mon_start + np.arange(30) * 3600.0
    ts = np.concatenate([fri, mon])
    closes = 100.0 + np.arange(len(ts)) * 0.1

    ret, elapsed, lag, bars = llm_eval._realized_forward_return(ts, closes, fri[0], 24)
    assert ret is not None
    assert elapsed > 60


# --------------------------------------------------------------------------- #
# 4. compute_incremental_report additions (W6)
# --------------------------------------------------------------------------- #

def test_s_degenerate_verdict():
    rng = np.random.default_rng(42)
    n = 120
    pred = rng.normal(size=n)
    realized = pred + rng.normal(scale=0.5, size=n)
    s = np.full(n, 0.5)
    t0 = np.arange(n) * 3600.0
    samples = list(zip(s, realized, pred, t0))

    rep = llm_eval.compute_incremental_report(samples, forward_bars=24, min_n=60)
    assert rep["s_degenerate"] is True
    assert "llm_score_degenerate" in rep["verdict"]
    assert "candidate to disable" not in rep["verdict"]


def test_anti_predictive_verdict():
    rng = np.random.default_rng(7)
    n = 1500
    pred = rng.normal(size=n)
    s_driver = rng.normal(size=n)               # independent of pred
    s = 1.0 / (1.0 + np.exp(-s_driver))
    z_s = (s - s.mean()) / s.std()
    realized = 1.5 * pred - 0.6 * z_s + rng.normal(scale=0.3, size=n)
    samples = list(zip(s, realized, pred))

    rep = llm_eval.compute_incremental_report(samples, forward_bars=24, min_n=60)
    enc = rep["encompassing"]
    assert enc is not None
    assert enc["b2_s"] < 0
    assert enc["p_value"] < 0.05
    assert "ANTI" in rep["verdict"]


def test_mixed_tuple_arity_no_crash():
    samples = [(0.5, 1.0, 0.1, 100.0), (0.6, -1.0, 0.2)] * 40
    rep = llm_eval.compute_incremental_report(samples, forward_bars=24, min_n=60)
    assert rep["time_ordered"] is False


def test_pseudo_replication_flags():
    rng = np.random.default_rng(11)
    n_symbols = 6
    n_t0 = 144 * 2
    t0_base = 1_700_000_000.0
    t0_values = t0_base + np.arange(n_t0) * 600.0

    rows = []
    for t0 in t0_values:
        pred_common = rng.normal()
        for _ in range(n_symbols):
            pred = pred_common + rng.normal(scale=0.1)
            s = 1.0 / (1.0 + np.exp(-rng.normal()))
            realized = pred + rng.normal(scale=1.0)
            rows.append((s, realized, pred, t0))

    rep = llm_eval.compute_incremental_report(rows, forward_bars=24, min_n=60)
    assert rep["rows_per_t0"] >= 5
    assert rep["pseudo_replication"] is True
    assert rep["n_distinct_t0"] < rep["n"]
    assert "unreliable" in rep["verdict"]


def test_n_input_and_dropped_pred():
    rng = np.random.default_rng(12)
    n = 100
    s = rng.uniform(0, 1, n)
    realized = rng.normal(size=n)
    pred = [None] * 90 + list(rng.normal(size=10))
    samples = list(zip(s, realized, pred))

    rep = llm_eval.compute_incremental_report(samples, forward_bars=24, min_n=60)
    assert rep["n"] == 10
    assert rep["n_input"] == 100
    assert rep["n_dropped"]["pred_none"] == 90


def test_echo_gap_abs_and_half_count():
    rng = np.random.default_rng(21)
    n = 2000
    pred = rng.normal(size=n)
    realized = -1.2 * pred + rng.normal(scale=0.8, size=n)      # negative raw driver
    s = 1.0 / (1.0 + np.exp(-(2.0 * pred + rng.normal(scale=0.05, size=n))))

    extra_pred = rng.normal(size=40)
    extra_realized = rng.normal(size=40)
    extra_s = np.full(40, 0.5)

    s_all = np.concatenate([s, extra_s])
    realized_all = np.concatenate([realized, extra_realized])
    pred_all = np.concatenate([pred, extra_pred])
    samples = list(zip(s_all, realized_all, pred_all))

    rep = llm_eval.compute_incremental_report(samples, forward_bars=24, min_n=60)
    raw = rep["raw_spearman_s_vs_return"]
    partial = rep["partial_spearman_s_given_pred"]
    assert raw < 0                                    # "negative raw" per spec
    # echo_gap/echo_gap_abs are computed internally from full-precision raw/
    # partial, then rounded once; raw/partial here are ALREADY 4-decimal
    # rounded, so recomputing from them is a double-rounding and can differ
    # by 1 in the last decimal place — assert the relationship with a small
    # tolerance rather than bit-exact equality.
    assert rep["echo_gap"] == pytest.approx(raw - partial, abs=1e-3)  # raw - partial, unchanged sign
    expected_abs = abs(raw) - abs(partial)
    assert rep["echo_gap_abs"] == pytest.approx(expected_abs, abs=1e-3)
    assert rep["n_s_exactly_half"] == 40


# --------------------------------------------------------------------------- #
# 5. compute_calibration_report restructure (W7)
# --------------------------------------------------------------------------- #

def test_calibration_abstain_with_missing_p_up():
    rng = np.random.default_rng(13)
    p_up = np.concatenate([np.full(40, np.nan), rng.uniform(0, 1, 80)])
    abstain = np.array([True] * 40 + [False] * 80)
    realized = rng.normal(size=120)

    rep = llm_eval.compute_calibration_report(p_up, realized, abstain=abstain)
    assert rep["abstain"]["n_abstain"] == 40
    assert rep["abstain"]["hit_rate_abstain"] is not None


def test_calibration_below_floor_still_descriptive():
    rng = np.random.default_rng(14)
    n = 30
    p = rng.uniform(0, 1, n)
    u = rng.uniform(0, 1, n)
    outcome = (u < p).astype(int)
    magnitude = rng.exponential(1.0, n) + 0.01
    realized = np.where(outcome == 1, magnitude, -magnitude)
    abstain = rng.random(n) < 0.3

    rep = llm_eval.compute_calibration_report(p, realized, abstain=abstain)
    assert rep["insufficient_power"] is True
    assert "bins" in rep
    assert rep["abstain"]["abstain_rate"] is not None


def test_calibration_length_mismatch_marker():
    rng = np.random.default_rng(15)
    n = 100
    p_up = rng.uniform(0, 1, n)
    realized = rng.normal(size=n)
    conviction = rng.integers(1, 6, 90)   # length mismatch: 90 vs 100

    rep = llm_eval.compute_calibration_report(p_up, realized, conviction=conviction)
    assert "error" in rep["conviction"]
    assert "90" in rep["conviction"]["error"]
    assert "100" in rep["conviction"]["error"]


def test_n_bins_honored_and_default_identical():
    rng = np.random.default_rng(16)
    n = 200
    p = rng.uniform(0, 1, n)
    u = rng.uniform(0, 1, n)
    outcome = (u < p).astype(int)
    magnitude = rng.exponential(1.0, n) + 0.01
    realized = np.where(outcome == 1, magnitude, -magnitude)

    rep10 = llm_eval.compute_calibration_report(p, realized, n_bins=10, min_n=10)
    assert len(rep10["bins"]) == 10

    rep5 = llm_eval.compute_calibration_report(p, realized, min_n=10)
    assert [b["lo"] for b in rep5["bins"]] == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert rep5["bins"][-1]["hi"] == 1.0


# --------------------------------------------------------------------------- #
# 6. advisor_report additions (W9)
# --------------------------------------------------------------------------- #

def test_advisor_dedup_and_model_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(llm_eval, "BASE_DIR", tmp_path)

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows_data = [
        ("BTC/USD", True, "model-A"),
        ("ETH/USD", True, "model-A"),
        ("SOL/USD", False, "model-B"),
    ]
    rows = []
    for i, (sym, dedup_hit, model) in enumerate(rows_data):
        ts = (base + timedelta(minutes=i)).isoformat()
        rows.append({
            "action": "llm_advisor_v2", "asset_type": "crypto", "forward_bars": 24,
            "ts": ts, "prompt_version": "v2", "prompt_sha256": "shared_sha_abc",
            "model": model, "dedup_hit": dedup_hit, "fng_value": 50,
            "scores": {sym: {"s": 0.6, "pred": 0.1, "p_up": 0.6, "conviction": 3,
                             "abstain": False, "event_flags": [],
                             "computed_events": [], "n_headlines": 2}},
        })
    today = datetime.now().date()
    _write_journal(tmp_path, today.isoformat(), rows)

    ts_full = _wide_hourly_ts(base.timestamp())
    closes_full = 100.0 + np.arange(len(ts_full)) * 0.001

    def fake_bars_lookup(api, symbol, asset_type, start, end):
        return ts_full, closes_full

    monkeypatch.setattr(llm_eval, "_bars_lookup", fake_bars_lookup)

    report = llm_eval.advisor_report(days=1, api=object())
    assert report != {}
    assert report["n_dedup_hit"] == 2
    assert report["n_unique_llm_calls"] == 1
    assert {"model-A", "model-B"} <= set(report["by_model"].keys())
    assert report["by_prompt_version"]
    for stats in report["by_prompt_version"].values():
        assert "avg_fwd_ret_pct" in stats


def test_advisor_p_up_fallback_disclosed(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(llm_eval, "BASE_DIR", tmp_path)

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    symbols = ["A1", "A2", "A3", "A4", "A5"]
    scores = {}
    for i, sym in enumerate(symbols):
        entry = {"s": 0.5 + i * 0.02, "pred": 0.1 * (i - 2), "conviction": 3,
                 "abstain": False, "event_flags": [], "computed_events": [],
                 "n_headlines": 1}
        entry["p_up"] = None if i >= 3 else 0.5 + i * 0.05   # A4, A5 lack p_up
        scores[sym] = entry

    today = datetime.now().date()
    _write_journal(tmp_path, today.isoformat(), [
        {"action": "llm_advisor_v2", "asset_type": "crypto", "forward_bars": 24,
         "ts": base.isoformat(), "prompt_version": "v2", "prompt_sha256": "sha1",
         "model": "model-A", "dedup_hit": False, "fng_value": 50, "scores": scores}])

    ts_full = _wide_hourly_ts(base.timestamp())
    closes_full = 100.0 + np.arange(len(ts_full)) * 0.001

    def fake_bars_lookup(api, symbol, asset_type, start, end):
        return ts_full, closes_full

    monkeypatch.setattr(llm_eval, "_bars_lookup", fake_bars_lookup)

    report = llm_eval.advisor_report(days=1, api=object())
    assert report != {}
    assert report["signal_source"] == "mixed"
    assert report["n_p_up_fallback_to_s"] == 2
    assert "incremental_p_up_only" in report
