"""c26 packet S1: llm_eval inference rebuild + journal-consumer integrity +
spend ledger (D09 + D33-consumer + B07).

  - D09.a: Driscoll-Kraay (1998) SEs clustered by t0-hour become the PRIMARY
    encompassing estimator (legacy rows-HAC kept alongside as 'legacy_b2'
    for one release), plus B07 hard power gates (MIN_POWER_T0=120 clusters,
    MIN_EFFECTIVE_N=20 span/horizon).
  - D09.b: _realized_forward_return steps the horizon in BARS over the
    symbol's bar index (policy_exits vertical-barrier semantics) instead of
    consuming forward_bars as wall-clock hours (~3.4x RTH stock mismatch).
  - D33-consumer: run_eval/advisor_report count s-null skips and collapse
    dedup_hit re-serves of an already-kept (prompt_sha256, symbol).
  - B07: spend-vs-benefit ledger (fail-soft llm_client daily-cost read +
    journaled cost_usd + deployed-tilt benefit in bps).
  - prompt_ab single-sources VETO_THRESHOLD / MIN_POWER_N from llm_eval.

Mac-runnable: stdlib + numpy/scipy only, no Alpaca, no dotenv, no torch.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import llm_eval
import prompt_ab


# --------------------------------------------------------------------------- #
# Shared fixtures / helpers (mirrors tests/test_llm_eval_v3.py style)
# --------------------------------------------------------------------------- #

def _write_journal(tmp_path, date_str, rows):
    path = tmp_path / f"{date_str}.jsonl"
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _wide_hourly_ts(center_ts, days_before=3, days_after=10):
    ts_start = center_ts - days_before * 86400
    ts_end = center_ts + days_after * 86400
    n_bars = int((ts_end - ts_start) // 3600) + 1
    return ts_start + np.arange(n_bars) * 3600.0


def _null_panel_trial(rng, G=150, n_sym=6, h=6):
    """One null-panel simulation per D09/B07: G hourly t0-clusters x n_sym
    symbols, a common per-t0 shock with MA(h) overlap in realized, within-t0
    correlated conviction noise s, and NO true b2 (s independent of
    realized). Geometry passes the hard gates: n_clusters=G>=120,
    effective_n ~ (G-1)/h >= 20."""
    eps = rng.normal(size=G + h)
    R = np.array([eps[k:k + h].sum() for k in range(G)])
    samples = []
    for k in range(G):
        t0 = k * 3600.0
        z_common = rng.normal()
        for _ in range(n_sym):
            pred = rng.normal()
            realized = R[k] + rng.normal(scale=0.3)
            s = 1.0 / (1.0 + np.exp(-(z_common + 0.3 * rng.normal())))
            samples.append((s, realized, pred, t0))
    return samples


# --------------------------------------------------------------------------- #
# 1-2. Driscoll-Kraay estimator (E3/E6)
# --------------------------------------------------------------------------- #

def test_dk_nests_nw_singleton_clusters():
    # One row per hourly cluster (rows time-sorted) => DK degenerates to the
    # rows-NW sandwich, modulo the G/(G-1) small-sample factor.
    rng = np.random.default_rng(101)
    G = 200
    pred = rng.normal(size=G)
    s = rng.uniform(0, 1, G)
    z_s = (s - s.mean()) / s.std()
    realized = 0.4 * pred + 0.2 * z_s + rng.normal(size=G)
    X = np.column_stack([np.ones(G), pred, z_s])
    beta, resid = llm_eval._ols_beta_resid(X, realized)

    cluster_ids = np.arange(G, dtype=np.int64)
    lag = 5
    se_dk, g = llm_eval._driscoll_kraay_se(X, resid, cluster_ids, lag=lag)
    se_nw = llm_eval._newey_west_se(X, resid, lag=lag)
    assert g == G
    np.testing.assert_allclose(se_dk, se_nw * np.sqrt(G / (G - 1.0)),
                               rtol=1e-9, atol=1e-12)


def test_dk_tames_null_false_keep():
    # Null panel: legacy rows-HAC over-rejects (false "keep the LLM spend");
    # DK-by-cluster stays at/below ~nominal size. b2 identical across both
    # (same X/resid), only SE/p differ.
    rng = np.random.default_rng(2026)
    trials = 100
    dk_rej = leg_rej = 0
    for _ in range(trials):
        rep = llm_eval.compute_incremental_report(
            _null_panel_trial(rng), forward_bars=6, min_n=60)
        enc = rep['encompassing']
        leg = rep['legacy_b2']
        assert enc['estimator'] == 'driscoll_kraay'
        assert enc['g_clusters'] == 150
        assert leg['estimator'] == 'newey_west_rows'
        assert leg['b2_s'] == enc['b2_s']
        assert 'insufficient_power' not in rep['verdict']  # gates pass here
        if enc['p_value'] is not None and enc['p_value'] < 0.05:
            dk_rej += 1
        if leg['p_value'] is not None and leg['p_value'] < 0.05:
            leg_rej += 1
    assert dk_rej / trials <= 0.10
    assert dk_rej < leg_rej


# --------------------------------------------------------------------------- #
# 3-4. B07 hard power gates (E7)
# --------------------------------------------------------------------------- #

def test_hard_gate_effective_n():
    rng = np.random.default_rng(103)
    n = 300
    t0 = np.arange(n) * 3600.0            # 300 hourly clusters
    pred = rng.normal(size=n)
    s = rng.uniform(0, 1, n)
    realized = pred + rng.normal(size=n)
    rep = llm_eval.compute_incremental_report(
        list(zip(s, realized, pred, t0)), forward_bars=24, min_n=60)
    assert rep['n_clusters'] == 300       # >= MIN_POWER_T0
    assert rep['effective_n_hint'] < llm_eval.MIN_EFFECTIVE_N   # 299/24 ~ 12.5
    assert rep['insufficient_power'] is True
    assert 'insufficient_power' in rep['verdict']
    assert 'unreliable' in rep['verdict']
    # The gate abstains on the VERDICT only — the numbers are still shown.
    assert rep['encompassing'] is not None
    assert 'legacy_b2' in rep


def test_hard_gate_min_clusters():
    rng = np.random.default_rng(104)
    n = 60                                 # == min_n, passes the n floor
    t0 = np.arange(n) * 3600.0             # 60 clusters < MIN_POWER_T0
    pred = rng.normal(size=n)
    s = rng.uniform(0, 1, n)
    realized = pred + rng.normal(size=n)
    rep = llm_eval.compute_incremental_report(
        list(zip(s, realized, pred, t0)), forward_bars=2, min_n=60)
    assert rep['n_clusters'] == 60
    assert rep['effective_n_hint'] >= llm_eval.MIN_EFFECTIVE_N  # 59/2 = 29.5
    assert rep['insufficient_power'] is True
    assert 'n_clusters=60' in rep['verdict']
    assert 'unreliable' in rep['verdict']


# --------------------------------------------------------------------------- #
# 5. has_ts=False path stays byte-identical legacy NW (E6)
# --------------------------------------------------------------------------- #

def test_no_ts_path_identical_legacy():
    rng = np.random.default_rng(105)
    n = 400
    pred = rng.normal(size=n)
    s = rng.uniform(0, 1, n)
    realized = 0.5 * pred + rng.normal(size=n)
    rep = llm_eval.compute_incremental_report(
        list(zip(s, realized, pred)), forward_bars=24, min_n=60)
    enc = rep['encompassing']
    assert enc['estimator'] == 'newey_west_rows'
    assert enc['dof'] == n - 3
    assert 'legacy_b2' not in rep
    # Manual recompute of the legacy path — must match exactly.
    z_s = (s - s.mean()) / s.std()
    X = np.column_stack([np.ones(n), pred, z_s])
    beta, resid = llm_eval._ols_beta_resid(X, realized)
    se = llm_eval._newey_west_se(X, resid, lag=23)
    assert enc['b2_s'] == round(float(beta[2]), 5)
    assert enc['se_hac'] == round(float(se[2]), 5)


# --------------------------------------------------------------------------- #
# 6. Bar-stepped horizon (E2)
# --------------------------------------------------------------------------- #

def test_bar_stepped_stock_horizon():
    # RTH-like grid: 7 bars/day x 6 days with overnight gaps. horizon=24
    # must span 24 BARS (the journaled meaning), with the >3x wall-clock
    # stretch SURFACED via elapsed_hours, not hidden.
    ts = np.concatenate([d * 86400.0 + np.arange(7) * 3600.0
                         for d in range(6)])
    closes = 100.0 + np.arange(len(ts))
    ret, elapsed, lag, bars = llm_eval._realized_forward_return(ts, closes, 0.0, 24)
    assert ret is not None
    assert bars == 24
    assert elapsed > 48        # ts[24] is day 3 bar 3 -> 75 wall-clock hours

    # Contiguous hourly (24/7 crypto) grid: identical to the old wall-clock
    # formula (exact searchsorted hit at i0+24).
    ts_c = np.arange(200) * 3600.0
    closes_c = 100.0 + np.arange(200) * 0.5
    i0 = 10
    ret_c, el_c, lag_c, bars_c = llm_eval._realized_forward_return(
        ts_c, closes_c, i0 * 3600.0, 24)
    expected = (closes_c[i0 + 24] - closes_c[i0]) / closes_c[i0] * 100.0
    assert ret_c == pytest.approx(expected)
    assert bars_c == 24
    assert el_c == 24.0


# --------------------------------------------------------------------------- #
# 7. run_eval D33-consumer counters + dedup collapse (E8)
# --------------------------------------------------------------------------- #

def test_run_eval_s_null_and_dedup_collapse(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(llm_eval, "BASE_DIR", tmp_path)
    monkeypatch.setattr(llm_eval, "_read_daily_cost", lambda: None)

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [
        # (a) fresh serve: AAA kept (registers (shaX, AAA)); BBB s-null.
        {"action": "llm_analysis", "ts": base.isoformat(),
         "asset_type": "crypto", "forward_bars": 24,
         "prompt_sha256": "shaX", "dedup_hit": False, "cost_usd": 0.02,
         "scores": {"AAA": {"s": 0.6, "pred": 0.1},
                    "BBB": {"s": None, "pred": 0.2}}},
        # (b) dedup re-serve of (shaX, AAA) -> collapsed.
        {"action": "llm_analysis",
         "ts": (base + timedelta(minutes=5)).isoformat(),
         "asset_type": "crypto", "forward_bars": 24,
         "prompt_sha256": "shaX", "dedup_hit": True,
         "scores": {"AAA": {"s": 0.6, "pred": 0.1}}},
        # (c) dedup hit with NO sha -> unattributable, KEPT.
        {"action": "llm_analysis",
         "ts": (base + timedelta(minutes=10)).isoformat(),
         "asset_type": "crypto", "forward_bars": 24, "dedup_hit": True,
         "scores": {"CCC": {"s": 0.4, "pred": -0.1}}},
    ]
    today = datetime.now().date()
    _write_journal(tmp_path, today.isoformat(), rows)

    ts_full = _wide_hourly_ts(base.timestamp())
    closes_full = 100.0 + np.arange(len(ts_full)) * 0.001
    monkeypatch.setattr(
        llm_eval, "_bars_lookup",
        lambda api, symbol, asset_type, start, end: (ts_full, closes_full))

    report = llm_eval.run_eval(days=1, api=object())
    assert report != {}
    cov = report["coverage"]
    assert cov["n_s_null"] == 1
    assert cov["n_dedup_collapsed"] == 1
    assert cov["n_dedup_unattributable"] == 1
    assert cov["n_rows_scored"] == 2          # AAA (a) + CCC (c)
    # Spend ledger present even when the daily-cost read fails (fail-soft).
    ledger = report["spend_ledger"]
    assert ledger["cost_read_ok"] is False
    assert ledger["daily_cost_usd"] is None
    assert ledger["window_journaled_cost_usd"] == 0.02
    assert ledger["n_entries_with_cost"] == 1
    assert ledger["n_realized_trades"] == 2


# --------------------------------------------------------------------------- #
# 8. Spend-ledger pure math + fail-soft (E9)
# --------------------------------------------------------------------------- #

def test_spend_ledger_math_and_failsoft():
    # s=1.0 -> tilt = (0.5+1.0)-1.0 = 0.5; realized 2.0% -> 0.5*2.0 = 1.0
    # (%*mult) -> *100 = 100.0 bps/trade.
    block = llm_eval._spend_ledger_block(
        [1.0], [2.0], np.array([False]),
        [{"cost_usd": 0.01}, {"cost_usd": "x"}, {}], days=14, cost=None)
    assert block["llm_tilt_bps_per_trade"] == 100.0
    assert block["cost_read_ok"] is False
    assert block["daily_cost_usd"] is None
    assert block["daily_cost_limit_usd"] is None
    assert block["window_journaled_cost_usd"] == 0.01
    assert block["n_entries_with_cost"] == 1
    assert block["veto_avoided_ret_pct_sum"] == 0.0
    assert block["days"] == 14
    assert block["n_realized_trades"] == 1
    assert block["sizing_formula"] == "llm_mult = 0.5 + s"

    # Veto sign: a vetoed -2% forward return is an AVOIDED loss -> +2.0.
    block2 = llm_eval._spend_ledger_block(
        [0.05], [-2.0], np.array([True]), [], days=7, cost=(1.25, 5.0))
    assert block2["veto_avoided_ret_pct_sum"] == 2.0
    assert block2["cost_read_ok"] is True
    assert block2["daily_cost_usd"] == 1.25
    assert block2["daily_cost_limit_usd"] == 5.0
    assert block2["window_journaled_cost_usd"] == 0.0
    assert block2["n_entries_with_cost"] == 0


# --------------------------------------------------------------------------- #
# 9. Ibragimov-Muller cross-check keys (E4) — report-only
# --------------------------------------------------------------------------- #

def test_im_block_key_present():
    rng = np.random.default_rng(109)
    rep = llm_eval.compute_incremental_report(
        _null_panel_trial(rng), forward_bars=6, min_n=60)
    enc = rep['encompassing']
    assert 'b2_im_p' in enc
    assert 'b2_im_mean' in enc
    assert 'im_blocks_used' in enc
    assert enc['im_blocks_used'] >= 4


# --------------------------------------------------------------------------- #
# 10. prompt_ab single-sources its constants from llm_eval (E14/E15)
# --------------------------------------------------------------------------- #

def test_prompt_ab_single_source():
    assert prompt_ab.VETO_THRESHOLD == llm_eval.VETO_THRESHOLD
    assert prompt_ab.MIN_POWER_N_DEFAULT == llm_eval.MIN_POWER_N == 60
    # decide_adopt abstains whenever a report carries insufficient_power=True
    # — which the new hard gates also set, so the three floors are mirrored
    # by construction.
    a = {"n": 500, "insufficient_power": True}
    b = {"n": 500}
    verdict = prompt_ab.decide_adopt(a, b, min_n=60)
    assert "insufficient_power" in verdict


# --------------------------------------------------------------------------- #
# 11. _meta_block additive keys (E12)
# --------------------------------------------------------------------------- #

def test_meta_block_additive():
    meta = llm_eval._meta_block(14, None, horizons=[24, 12])
    # New additive keys.
    assert meta["min_power_t0"] == llm_eval.MIN_POWER_T0 == 120
    assert meta["min_effective_n"] == llm_eval.MIN_EFFECTIVE_N == 20
    # Existing keys untouched.
    assert meta["min_power_n"] == 60
    assert meta["veto_threshold"] == llm_eval.VETO_THRESHOLD
    assert meta["days"] == 14
    assert meta["asset_filter"] is None
    assert meta["forward_bars_used"] == 24
    assert meta["forward_bars_seen"] == [12, 24]


# --------------------------------------------------------------------------- #
# 12. advisor_report D33-consumer parity — an ACTUAL collapse (E11)
#     (test_advisor_dedup_and_model_accounting's rows have distinct symbols,
#     so it never exercises the collapse branch — this one does.)
# --------------------------------------------------------------------------- #

def test_advisor_s_null_and_dedup_collapse(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_eval, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(llm_eval, "BASE_DIR", tmp_path)

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    common = {"action": "llm_advisor_v2", "asset_type": "crypto",
              "forward_bars": 24, "prompt_version": "v2", "model": "model-A",
              "fng_value": 50}
    rows = [
        # (a) fresh serve: DDD kept (registers (shaY, DDD)); EEE s-null.
        {**common, "ts": base.isoformat(), "prompt_sha256": "shaY",
         "dedup_hit": False,
         "scores": {"DDD": {"s": 0.7, "pred": 0.2, "p_up": 0.65},
                    "EEE": {"s": None, "pred": 0.1}}},
        # (b) dedup re-serve of (shaY, DDD) -> collapsed.
        {**common, "ts": (base + timedelta(minutes=5)).isoformat(),
         "prompt_sha256": "shaY", "dedup_hit": True,
         "scores": {"DDD": {"s": 0.7, "pred": 0.2, "p_up": 0.65}}},
        # (c) dedup hit with NO sha -> unattributable, KEPT.
        {**common, "ts": (base + timedelta(minutes=10)).isoformat(),
         "dedup_hit": True,
         "scores": {"FFF": {"s": 0.3, "pred": -0.1, "p_up": 0.35}}},
    ]
    today = datetime.now().date()
    _write_journal(tmp_path, today.isoformat(), rows)

    ts_full = _wide_hourly_ts(base.timestamp())
    closes_full = 100.0 + np.arange(len(ts_full)) * 0.001
    monkeypatch.setattr(
        llm_eval, "_bars_lookup",
        lambda api, symbol, asset_type, start, end: (ts_full, closes_full))

    report = llm_eval.advisor_report(days=1, api=object())
    assert report != {}
    cov = report["coverage"]
    assert cov["n_s_null"] == 1
    assert cov["n_dedup_collapsed"] == 1
    assert cov["n_dedup_unattributable"] == 1
    assert cov["n_rows_scored"] == 2          # DDD (a) + FFF (c)
    # Dedup accounting is over KEPT rows only: (c) is the sole dedup-hit row.
    assert report["n_dedup_hit"] == 1
    assert report["n_unique_llm_calls"] == 1  # shaY is the only sha seen


# --------------------------------------------------------------------------- #
# 13. _read_daily_cost never raises (E9 fail-soft — get_daily_cost blows up)
# --------------------------------------------------------------------------- #

def test_read_daily_cost_failsoft(monkeypatch):
    import types

    broken = types.ModuleType("llm_client")

    def _boom():
        raise RuntimeError("ledger unreadable")

    broken.get_daily_cost = _boom
    monkeypatch.setitem(sys.modules, "llm_client", broken)
    assert llm_eval._read_daily_cost() is None

    ok = types.ModuleType("llm_client")
    ok.get_daily_cost = lambda: (1.5, 5.0)
    monkeypatch.setitem(sys.modules, "llm_client", ok)
    assert llm_eval._read_daily_cost() == (1.5, 5.0)
