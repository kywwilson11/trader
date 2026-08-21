"""module-improve-v3 batch B4: portfolio_backtest.py hardening + instrumentation.

Covers the fail-loud guards (non-finite/missing signal, duplicate symbols
within a period, duplicate (timestamp, ticker) rows, weight_fn shape/
finiteness), the nan-sharpe fix on NaN-contaminated nets, the new metric
surface (hit_rate_invested, n_invested_periods, n_nonfinite_periods,
weight_turnover, admitted_k_hist, pct_periods_k_ge_6, periods_per_year,
avg_gross_exposure), the compare()/compare_deflated() baseline-resolution +
generator-materialization fix, compare_deflated's expanded provenance
(n_eff_used, n_eff_floored, n_dropped, sr_per_period, n_trials, fwd_bars
sentinel, weight_fns dispatch, k_delta/avg_entry_fraction_delta), the
panel_from_frame vectorized rebuild (value-identity across mixed dtypes and
row order), and the shared BARS_PER_YEAR constant. numpy/pandas only — no
heavy deps, so no importorskip is needed and no new baseline-failure name
can appear.
"""
import ast
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import portfolio_backtest as pb

REPO = Path(__file__).resolve().parent.parent


def _period(*triples):
    # triples of (symbol, signal, fwd_return) [+ optional meta_p]
    out = []
    for t in triples:
        d = {'symbol': t[0], 'signal': t[1], 'fwd_return': t[2]}
        if len(t) > 3:
            d['meta_p'] = t[3]
        out.append(d)
    return out


def _gradient_panel(n_periods=300, k=7, seed=1):
    """Panel where rank-1 by signal earns the most fwd_return (a real
    gradient) — same construction as tests/test_conviction_ab.py."""
    rng = np.random.default_rng(seed)
    panel = []
    for _ in range(n_periods):
        sig = np.sort(rng.normal(size=k))[::-1]
        fwd = 0.4 * (sig - sig.mean()) + rng.normal(0, 0.3, k)
        panel.append([{'symbol': f'S{i}', 'signal': float(sig[i]),
                       'fwd_return': float(fwd[i])} for i in range(k)])
    return panel


# ---------------------------------------------------------------------------
# 1. BARS_PER_YEAR constant
# ---------------------------------------------------------------------------

def test_bars_per_year_matches_sibling_copies():
    """Pin: BARS_PER_YEAR must stay byte-identical across portfolio_backtest,
    volatility.py, backtest.py, scripts/hypersearch_v2.py — drift would
    silently de-sync promotion-gate annualization from live vol targeting."""
    pat = re.compile(r"BARS_PER_YEAR\s*=\s*(\{[^}]*\})")
    expected = {'crypto': 8760, 'stock': 1638}
    for rel in ('volatility.py', 'backtest.py', 'scripts/hypersearch_v2.py'):
        m = pat.search((REPO / rel).read_text())
        assert m, f"BARS_PER_YEAR assignment not found in {rel}"
        assert ast.literal_eval(m.group(1)) == expected, f"{rel} drifted"
    assert pb.BARS_PER_YEAR == expected
    assert pb.DEFAULT_PERIODS_PER_YEAR == 1638.0


# ---------------------------------------------------------------------------
# 2. periods_per_year echoed + scales sharpe
# ---------------------------------------------------------------------------

def test_periods_per_year_echoed_and_scales_sharpe():
    """Pin: run_policy echoes periods_per_year verbatim, and switching to the
    crypto annualization scales sharpe by sqrt(8760/1638) — the crypto/stock
    BARS_PER_YEAR split actually reaches the Sharpe computation."""
    rng = np.random.default_rng(7)
    panel = [[{'symbol': 'A', 'signal': 1.0,
              'fwd_return': float(rng.normal(0, 1))}] for _ in range(30)]
    r1 = pb.run_policy(panel, pb.top_k(1))
    assert r1['periods_per_year'] == 1638.0
    r2 = pb.run_policy(panel, pb.top_k(1), periods_per_year=8760.0)
    assert r2['periods_per_year'] == 8760.0
    assert r2['sharpe'] == pytest.approx(r1['sharpe'] * math.sqrt(8760 / 1638), rel=1e-3)


# ---------------------------------------------------------------------------
# 3-5. _sorted_desc fail-loud
# ---------------------------------------------------------------------------

def test_sorted_desc_missing_signal_raises():
    """Pin: a candidate dict missing 'signal' entirely fails loud (KeyError)
    instead of silently defaulting to 0.0 and mis-ranking."""
    with pytest.raises(KeyError):
        pb._sorted_desc([{'symbol': 'X', 'fwd_return': 1.0}])


def test_sorted_desc_nan_signal_raises():
    """Pin: a NaN signal raises ValueError (a NaN signal makes the ranking
    row-order-dependent) both directly and through run_policy, and
    regardless of the NaN row's position within the period."""
    per = [{'symbol': 'A', 'signal': 0.5, 'fwd_return': -1.0},
           {'symbol': 'B', 'signal': float('nan'), 'fwd_return': 0.0},
           {'symbol': 'C', 'signal': 0.9, 'fwd_return': 5.0}]
    with pytest.raises(ValueError):
        pb._sorted_desc(per)
    with pytest.raises(ValueError):
        pb.run_policy([per], pb.top_k(1))
    with pytest.raises(ValueError):
        pb._sorted_desc(list(reversed(per)))


def test_sorted_desc_finite_order_unchanged():
    """Pin: the happy path (all-finite signals) still ranks descending —
    byte-identity of the pre-existing behavior."""
    per = [{'symbol': 'A', 'signal': 0.1, 'fwd_return': 1.0},
           {'symbol': 'B', 'signal': 0.9, 'fwd_return': 1.0},
           {'symbol': 'C', 'signal': 0.5, 'fwd_return': 1.0}]
    assert [c['symbol'] for c in pb._sorted_desc(per)] == ['B', 'C', 'A']


# ---------------------------------------------------------------------------
# 6. conviction_gated missing-field fail-open (owner decision, pinned)
# ---------------------------------------------------------------------------

def test_missing_conviction_field_passes_set_floor_pinned():
    """Pin (OPEN OWNER DECISION): a candidate dict missing 'meta_p' /
    'pred_thresh_ratio' entirely currently PASSES that floor (missing != NaN
    — a present-but-NaN field fails-closed, but an absent field fail-opens).
    Docs updated; semantics deliberately left to the owner."""
    cands = [{'symbol': 'A', 'signal': 0.9, 'fwd_return': 1.0},
             {'symbol': 'B', 'signal': 0.8, 'fwd_return': 1.0}]
    admitted = pb.conviction_gated(7, meta_floor=0.99)(cands)
    assert [c['symbol'] for c in admitted] == ['A', 'B']
    admitted2 = pb.conviction_gated(7, ratio_floor=1.0)(cands)
    assert [c['symbol'] for c in admitted2] == ['A', 'B']


# ---------------------------------------------------------------------------
# 7. duplicate symbol within a period
# ---------------------------------------------------------------------------

def test_duplicate_symbol_in_period_raises():
    """Pin: a period listing the same symbol twice raises in BOTH run_policy
    and run_policy_weighted, instead of silently double-counting a book slot
    (previously: top_k(3) reported k=3 for a 2-name book)."""
    per = [{'symbol': 'A', 'signal': 0.9, 'fwd_return': 1.0},
           {'symbol': 'A', 'signal': 0.8, 'fwd_return': 1.0},
           {'symbol': 'B', 'signal': 0.5, 'fwd_return': 1.0}]
    with pytest.raises(ValueError, match='duplicate'):
        pb.run_policy([per], pb.top_k(3))
    with pytest.raises(ValueError, match='duplicate'):
        pb.run_policy_weighted([per], pb.top_k(3), pb.equal_weights)


# ---------------------------------------------------------------------------
# 8-9. panel_from_frame guards
# ---------------------------------------------------------------------------

def test_panel_from_frame_duplicate_ts_symbol_raises():
    """Pin: two rows sharing the same (timestamp, ticker) raise — a
    predictions dump must carry each symbol at most once per bar."""
    idx = pd.to_datetime(['2026-01-01 10:00', '2026-01-01 10:00'])
    df = pd.DataFrame({'Ticker': ['A', 'A'], 'sig': [1.0, 2.0],
                       'fwd': [0.1, 0.2]}, index=idx)
    with pytest.raises(ValueError, match='duplicate'):
        pb.panel_from_frame(df, 'sig', 'fwd')


def test_panel_from_frame_extra_cols_collision_raises():
    """Pin: extra_cols naming the same column as signal_col (or any
    positional column) raises rather than silently building a malformed
    candidate dict with a clobbered key."""
    idx = pd.to_datetime(['2026-01-01 10:00'])
    df = pd.DataFrame({'Ticker': ['A'], 'sig': [1.0], 'fwd': [0.1]}, index=idx)
    with pytest.raises(ValueError, match='duplicate column'):
        pb.panel_from_frame(df, 'sig', 'fwd', extra_cols=['sig'])


# ---------------------------------------------------------------------------
# 10. panel_from_frame stats_out
# ---------------------------------------------------------------------------

def test_panel_from_frame_stats_out():
    """Pin: stats_out (caller-supplied dict) receives coverage counters so a
    silently-shrunken panel (e.g. an unexpected dropna wipeout) is
    detectable."""
    idx = pd.to_datetime(['2026-01-01 10:00'] * 6)
    df = pd.DataFrame({
        'Ticker': ['A', 'B', 'C', 'D', 'E', 'F'],
        'sig': [1.0] + [float('nan')] * 5,
        'fwd': [0.1] * 6,
    }, index=idx)
    stats = {}
    panel = pb.panel_from_frame(df, 'sig', 'fwd', stats_out=stats)
    assert stats == {'rows_in': 6, 'rows_dropped': 5, 'n_periods': 1,
                     'mean_candidates_per_period': 1.0}
    assert len(panel) == 1 and len(panel[0]) == 1


# ---------------------------------------------------------------------------
# 11. panel_from_frame vectorized build: value identity, mixed dtypes
# ---------------------------------------------------------------------------

def test_panel_from_frame_value_identity_mixed_dtypes():
    """Pin: the vectorized boundary-scan build (replacing the old per-group
    iterrows loop) is value-identical to the documented semantics —
    including row order on same-timestamp ties, resolved by a stable
    timestamp sort — on a shuffled frame with mixed extra_cols dtypes (int +
    str) and a NaN-signal row, under both signal_lag=0 and signal_lag=1.
    Also pins that extras survive as JSON-serializable python scalars."""
    idx = pd.to_datetime(['2026-02-01 11:00', '2026-02-01 10:00',
                          '2026-02-01 10:00', '2026-02-01 10:00',
                          '2026-02-01 11:00', '2026-02-01 11:00'])
    # rows in this (shuffled) order: B@T2, A@T1, C@T1, B@T1(NaN sig), C@T2, A@T2
    df = pd.DataFrame({
        'Ticker':   ['B',   'A',   'C',   'B',          'C',   'A'],
        'sig':      [13.0,  10.0,  14.0,  float('nan'), 15.0,  11.0],
        'fwd':      [0.04,  0.01,  0.05,  0.03,         0.06,  0.02],
        'vol_rank': [103,   100,   104,   102,          105,   101],
        'regime':   ['r2',  'r1',  'r3',  'r2',         'r3',  'r1'],
    }, index=idx)

    panel0 = pb.panel_from_frame(df, 'sig', 'fwd', extra_cols=['vol_rank', 'regime'],
                                 signal_lag=0)
    expected0 = [
        [{'symbol': 'A', 'signal': 10.0, 'fwd_return': 0.01, 'vol_rank': 100, 'regime': 'r1'},
         {'symbol': 'C', 'signal': 14.0, 'fwd_return': 0.05, 'vol_rank': 104, 'regime': 'r3'}],
        [{'symbol': 'B', 'signal': 13.0, 'fwd_return': 0.04, 'vol_rank': 103, 'regime': 'r2'},
         {'symbol': 'C', 'signal': 15.0, 'fwd_return': 0.06, 'vol_rank': 105, 'regime': 'r3'},
         {'symbol': 'A', 'signal': 11.0, 'fwd_return': 0.02, 'vol_rank': 101, 'regime': 'r1'}],
    ]
    assert panel0 == expected0
    assert json.dumps(panel0)   # python-scalar extras, not numpy types

    panel1 = pb.panel_from_frame(df, 'sig', 'fwd', extra_cols=['vol_rank', 'regime'],
                                 signal_lag=1)
    # lag=1: A@T1 and C@T1 shift to NaN (nothing precedes them); B@T1's
    # ORIGINAL NaN shifts forward and also poisons B@T2. Only C@T2 (lagged
    # from C@T1's original 14.0) and A@T2 (lagged from A@T1's original 10.0)
    # survive, both landing in the single T2 period.
    expected1 = [
        [{'symbol': 'C', 'signal': 14.0, 'fwd_return': 0.06, 'vol_rank': 105, 'regime': 'r3'},
         {'symbol': 'A', 'signal': 10.0, 'fwd_return': 0.02, 'vol_rank': 101, 'regime': 'r1'}],
    ]
    assert panel1 == expected1
    assert json.dumps(panel1)


# ---------------------------------------------------------------------------
# 12-13. hit_rate / nan-sharpe additive keys
# ---------------------------------------------------------------------------

def test_hit_rate_cash_pinned_and_invested_added():
    """Pin: hit_rate keeps its CALENDAR denominator (an all-cash period
    counts as a non-hit); hit_rate_invested/n_invested_periods are the new
    k>0-only view (None when never invested)."""
    pol = pb.conviction_gated(1, signal_floor=0.5)
    panel = [_period(('A', 0.1, 5.0)), _period(('A', 0.1, 5.0)),
             _period(('A', 0.9, 5.0)), _period(('A', 0.1, 5.0))]
    r = pb.run_policy(panel, pol)
    assert r['hit_rate'] == 0.25
    assert r['pct_periods_cash'] == 0.75
    assert r['hit_rate_invested'] == 1.0
    assert r['n_invested_periods'] == 1

    all_cash = [_period(('A', 0.1, 5.0)) for _ in range(4)]
    rc = pb.run_policy(all_cash, pol)
    assert rc['hit_rate'] == 0.0
    assert rc['hit_rate_invested'] is None
    assert rc['n_invested_periods'] == 0


def test_nan_fwd_return_nan_sharpe_and_count():
    """Pin: a NaN-contaminated nets series reports sharpe/net_total as NaN
    (never a confident 0.0) and n_nonfinite_periods counts the
    contamination; a clean panel has n_nonfinite_periods == 0 and a finite
    sharpe."""
    rng = np.random.default_rng(5)
    periods = []
    for i in range(20):
        fwd = float('nan') if i == 10 else float(rng.normal(0, 1))
        periods.append(_period(('A', 1.0, fwd)))
    r = pb.run_policy(periods, pb.top_k(1))
    assert math.isnan(r['sharpe'])
    assert math.isnan(r['net_total'])
    assert r['n_nonfinite_periods'] == 1

    clean = [_period(('A', 1.0, float(rng.normal(0, 1)))) for _ in range(20)]
    rc = pb.run_policy(clean, pb.top_k(1))
    assert rc['n_nonfinite_periods'] == 0
    assert not math.isnan(rc['sharpe'])


# ---------------------------------------------------------------------------
# 14. weight_turnover: reported vs charged cost basis
# ---------------------------------------------------------------------------

def test_weight_turnover_reports_uncharged_rebalance():
    """Pin: run_policy's charged cost (entry-set only) and weight_turnover
    (full weight-basis turnover) diverge on a K-shrink; run_policy_weighted
    charges the FULL weight-turnover basis instead — both costs on the SAME
    shrink are pinned exactly so a cost-basis change is a loud test edit."""
    names = ['A', 'B', 'C', 'D', 'E']
    p1 = _period(*[(n, 1.0, 0.0) for n in names])
    p2 = _period(*([(n, 1.0, 0.0) for n in names[:3]] + [(n, 0.0, 0.0) for n in names[3:]]))
    pol = pb.conviction_gated(5, signal_floor=0.5)

    rp = pb.run_policy([p1, p2], pol, cost_pct=0.10)
    assert rp['net_total'] == pytest.approx(-0.10)          # pins today's charge
    assert rp['weight_turnover'] == pytest.approx(1.4, abs=1e-9)  # 1.0 entry + 0.4 uncharged rebalance

    rw = pb.run_policy_weighted([p1, p2], pol, pb.equal_weights, cost_pct=0.10)
    assert rw['net_total'] == pytest.approx(-0.14)          # the divergence is pinned

    # Constant-K sanity: full-churn top_k(1) over 2 periods -> weight_turnover
    # == 2.0 (a full 1/k swap out then in), entries == 2 (X then Y).
    churn = [_period(('X', 0.9, 1.0)), _period(('Y', 0.9, 1.0))]
    rc = pb.run_policy(churn, pb.top_k(1))
    assert rc['weight_turnover'] == pytest.approx(2.0)
    assert rc['entries'] == 2


def test_equal_weights_matches_run_policy_at_zero_cost():
    """Pin: equal_weights routed through run_policy_weighted reproduces
    run_policy's equal-weight net_total exactly when cost_pct=0 (both
    average the same admitted fwd_returns)."""
    panel = [_period(('A', 0.9, 1.0), ('B', 0.8, -0.5), ('C', 0.7, 0.2)),
             _period(('A', 0.6, 0.3), ('B', 0.5, 0.4), ('C', 0.4, -0.1))]
    rw = pb.run_policy_weighted(panel, pb.top_k(2), pb.equal_weights)
    rp = pb.run_policy(panel, pb.top_k(2))
    assert rw['net_total'] == pytest.approx(rp['net_total'])


# ---------------------------------------------------------------------------
# 16-18. compare(): empty policies, unknown baseline, generator materialization
# ---------------------------------------------------------------------------

def test_empty_policies_raises():
    """Pin: an empty policies dict raises rather than an opaque bare
    IndexError from names[0] on a length-0 list."""
    panel = [_period(('A', 1.0, 1.0))]
    with pytest.raises(ValueError):
        pb.compare(panel, {})
    with pytest.raises(ValueError):
        pb.compare_deflated(panel, {})


def test_unknown_baseline_raises_default_unchanged():
    """Pin: an unknown explicit baseline raises (previously: silently
    substituted the first key); the implicit default (baseline=None) still
    resolves to the first key."""
    panel = [_period(('A', 1.0, 1.0))]
    policies = {'x': pb.top_k(1), 'y': pb.top_k(1)}
    with pytest.raises(ValueError, match='baseline'):
        pb.compare(panel, policies, baseline='typo')
    out = pb.compare(panel, policies)
    assert out['baseline'] == 'x'


def test_generator_panel_materialized():
    """Pin: a generator/iterator panel is materialized ONCE so every named
    policy sees the full panel (previously: the second+ policy in the dict
    comprehension consumed an already-exhausted iterator and got 0
    periods)."""
    per = _period(('A', 1.0, 1.0))
    out = pb.compare(iter([per] * 20), {'first': pb.top_k(1), 'second': pb.top_k(1)})
    assert out['results']['first']['n_periods'] == 20
    assert out['results']['second']['n_periods'] == 20
    assert all(v == 0.0 for v in out['deltas']['second'].values())


# ---------------------------------------------------------------------------
# 19-22. compare_deflated: provenance, fwd_bars sentinel, n_trials, deltas
# ---------------------------------------------------------------------------

def test_compare_deflated_provenance():
    """Pin: the new provenance keys (n_eff_used, n_eff_floored, n_dropped,
    sr_per_period, n_trials) surface exactly what
    validation.dsr_from_trade_returns consumed — including the 10-sample
    floor kicking in on an 8.3-effective panel, and a 1-policy pool clamped
    to n_trials=2."""
    rng = np.random.default_rng(42)
    panel = [[{'symbol': 'A', 'signal': 1.0,
              'fwd_return': float(rng.normal(0, 0.5))}] for _ in range(200)]
    out = pb.compare_deflated(panel, {'only': pb.top_k(1)}, fwd_bars=24)
    m = out['results']['only']
    assert m['n_eff'] == 8.3
    assert m['n_eff_used'] == 10.0
    assert m['n_eff_floored'] is True
    assert m['n_dropped'] == 0
    assert m['n_trials'] == 2
    assert m['sr_per_period'] == pytest.approx(m['sharpe'] / math.sqrt(1638.0), abs=1e-3)

    rng2 = np.random.default_rng(9)
    panel2 = [[{'symbol': 'A', 'signal': 1.0,
               'fwd_return': float(rng2.normal(0, 0.5))}] for _ in range(400)]
    out2 = pb.compare_deflated(panel2, {'only': pb.top_k(1)})
    m2 = out2['results']['only']
    assert m2['n_eff'] == 400.0
    assert m2['n_eff_used'] == 400.0
    assert m2['n_eff_floored'] is False


def test_n_eff_floored_no_false_positives():
    """Pin (hardening fix): n_eff_floored means validation's 10-sample floor
    ACTUALLY raised the request. It must NOT trip on (a) the round-to-2 echo
    exceeding the raw request (400 finite periods at fwd_bars=24 -> request
    16.6667, echo 16.67 — no floor applied) or (b) the fail-closed n<10 path
    (8 finite periods: dsr=0.0, n_eff echoes the raw count, floor never
    applied)."""
    rng = np.random.default_rng(11)
    panel = [[{'symbol': 'A', 'signal': 1.0,
              'fwd_return': float(rng.normal(0, 0.5))}] for _ in range(400)]
    m = pb.compare_deflated(panel, {'only': pb.top_k(1)}, fwd_bars=24)['results']['only']
    assert m['n_eff'] == 16.7
    assert m['n_eff_used'] == 16.67          # rounded echo > raw request...
    assert m['n_eff_floored'] is False       # ...but the floor never applied

    ms = pb.compare_deflated(panel[:8], {'only': pb.top_k(1)}, fwd_bars=24)['results']['only']
    assert ms['dsr'] == 0.0                  # fail-closed, nothing deflated
    assert ms['n_eff_used'] == 8.0           # raw count echoed, not floored
    assert ms['n_eff_floored'] is False


def test_metric_surface_parity():
    """Pin: run_policy and run_policy_weighted share ONE result surface (via
    the shared _metrics_summary builder) — run_policy_weighted adds exactly
    avg_gross_exposure and nothing else, so compare_deflated's weight_fns
    dispatch can mix the two engines in one results table."""
    panel = [_period(('A', 0.9, 1.0), ('B', 0.8, -0.5)),
             _period(('A', 0.6, 0.3), ('B', 0.5, 0.4))]
    rp = pb.run_policy(panel, pb.top_k(2))
    rw = pb.run_policy_weighted(panel, pb.top_k(2), pb.equal_weights)
    assert set(rw) - set(rp) == {'avg_gross_exposure'}
    assert set(rp) < set(rw)


def test_fwd_bars_defaulted_flag():
    """Pin: an omitted fwd_bars defaults to 1 AND is flagged
    fwd_bars_defaulted=True (an assumed, not declared, horizon); an explicit
    fwd_bars=1 gives byte-identical downstream numbers but defaulted=False."""
    rng = np.random.default_rng(3)
    panel = [[{'symbol': 'A', 'signal': 1.0,
              'fwd_return': float(rng.normal(0, 0.5))}] for _ in range(50)]
    default = pb.compare_deflated(panel, {'only': pb.top_k(1)})
    m = default['results']['only']
    assert m['fwd_bars'] == 1
    assert m['fwd_bars_defaulted'] is True

    explicit = pb.compare_deflated(panel, {'only': pb.top_k(1)}, fwd_bars=1)
    me = explicit['results']['only']
    assert me['fwd_bars_defaulted'] is False
    assert me['dsr'] == m['dsr']
    assert me['n_eff'] == m['n_eff']


def test_n_trials_override_raises_bar():
    """Pin: n_trials is a real override, not just len(policies) — a larger
    explicit pool raises the expected-max-Sharpe bar and cannot improve DSR;
    two default calls are byte-equal (fully deterministic)."""
    rng = np.random.default_rng(3)
    panel = [[{'symbol': f'S{i}', 'signal': float(7 - i),
              'fwd_return': float(7 - i) / 10 + rng.normal(0, 0.05)}
             for i in range(7)] for _ in range(50)]
    policies = {'base': pb.top_k(7), 'conv': pb.conviction_gated(7, signal_floor=3.5)}

    d1 = pb.compare_deflated(panel, policies)
    d2 = pb.compare_deflated(panel, policies)
    assert d1['results']['base']['dsr'] == d2['results']['base']['dsr']
    assert d1['results']['base']['expected_max_sr'] == d2['results']['base']['expected_max_sr']

    override = pb.compare_deflated(panel, policies, n_trials=60)
    assert override['results']['base']['expected_max_sr'] > d1['results']['base']['expected_max_sr']
    assert override['results']['base']['dsr'] <= d1['results']['base']['dsr']


def test_deflated_deltas_k_and_entry_fraction():
    """Pin: compare_deflated's deltas gain k_delta (matching compare()'s
    value exactly, on the same panel/policies/baseline) and
    avg_entry_fraction_delta."""
    rng = np.random.default_rng(3)
    panel = [[{'symbol': f'S{i}', 'signal': float(7 - i),
              'fwd_return': float(7 - i) / 10 + rng.normal(0, 0.05)}
             for i in range(7)] for _ in range(50)]
    policies = {'base': pb.top_k(7), 'conv': pb.conviction_gated(7, signal_floor=3.5)}
    cd = pb.compare_deflated(panel, policies, baseline='base')
    c = pb.compare(panel, policies, baseline='base')
    assert 'k_delta' in cd['deltas']['conv']
    assert cd['deltas']['conv']['k_delta'] < 0
    assert cd['deltas']['conv']['k_delta'] == c['deltas']['conv']['k_delta']
    assert 'avg_entry_fraction_delta' in cd['deltas']['conv']


# ---------------------------------------------------------------------------
# 23-24. weight_fns dispatch + weight_fn validation
# ---------------------------------------------------------------------------

def test_weighted_arm_deflatable_full_surface():
    """Pin: weight_fns={name: weight_fn} routes that arm through
    run_policy_weighted inside compare_deflated, carrying the full metric
    surface (dsr/turnover/n_eff/hit_rate/mean_admitted_k) plus
    avg_gross_exposure/weight_turnover; an unknown weight_fns name raises."""
    panel = _gradient_panel()
    out = pb.compare_deflated(panel, {'equal': pb.top_k(7), 'edge': pb.top_k(7)},
                              weight_fns={'edge': pb.edge_proportional_weights})
    for name in ('equal', 'edge'):
        row = out['results'][name]
        for key in ('dsr', 'turnover', 'n_eff', 'hit_rate', 'mean_admitted_k'):
            assert key in row
    assert out['results']['edge']['avg_gross_exposure'] == pytest.approx(1.0)
    assert 'weight_turnover' in out['results']['edge']
    with pytest.raises(ValueError):
        pb.compare_deflated(panel, {'equal': pb.top_k(7)},
                            weight_fns={'typo': pb.edge_proportional_weights})


def test_weight_fn_validation_raises():
    """Pin: run_policy_weighted validates the weight_fn's return shape and
    finiteness rather than silently broadcasting/truncating (previously: a
    length-1 vector on a 2-name book gave net_total=2.0 — silently
    levered)."""
    per = [{'symbol': 'A', 'signal': 1.0, 'fwd_return': 1.0},
           {'symbol': 'B', 'signal': 0.5, 'fwd_return': 1.0}]
    with pytest.raises(ValueError, match='shape'):
        pb.run_policy_weighted([per], pb.top_k(2), lambda s: np.array([1.0]))
    with pytest.raises(ValueError, match='non-finite'):
        pb.run_policy_weighted([per], pb.top_k(2), lambda s: [float('nan'), 1.0])


# ---------------------------------------------------------------------------
# 25. admitted_k_hist / pct_periods_k_ge_6
# ---------------------------------------------------------------------------

def test_admitted_k_hist():
    """Pin: admitted_k_hist/pct_periods_k_ge_6 mirror decision_report.py's
    live-side shape/naming so the holdout and live sides diff mechanically."""
    periods = []
    for i in range(14):
        sig = 1.0 if i % 2 == 0 else 0.1
        periods.append(_period(*[(f'S{j}', sig, 0.0) for j in range(7)]))
    pol = pb.conviction_gated(7, signal_floor=0.5)
    r = pb.run_policy(periods, pol)
    assert r['mean_admitted_k'] == 3.5
    assert r['pct_periods_k_ge_6'] == 0.5
    assert r['admitted_k_hist']['0'] == 7
    assert r['admitted_k_hist']['7'] == 7

    constant = [_period(('A', 1.0, 0.0), ('B', 1.0, 0.0), ('C', 1.0, 0.0))] * 5
    rc = pb.run_policy(constant, pb.top_k(3))
    assert rc['pct_periods_k_ge_6'] == 0.0


# ---------------------------------------------------------------------------
# 26. edge_proportional_weights docstring pins (no code change)
# ---------------------------------------------------------------------------

def test_edge_prop_negative_floor_nan_pinned():
    """Pin (OPEN OWNER DECISION): a non-finite signal is zeroed BEFORE the
    floor subtraction, so a NEGATIVE floor gives a NaN name positive
    weight."""
    w = pb.edge_proportional_weights([float('nan'), 0.1], floor=-0.5)
    assert list(w) == pytest.approx([0.4545, 0.5455], abs=1e-4)
    w2 = pb.edge_proportional_weights([float('nan'), 0.1])
    assert list(w2) == pytest.approx([0.0, 1.0])


# ---------------------------------------------------------------------------
# 27. panel_from_frame signal_lag: row-based, not time-based
# ---------------------------------------------------------------------------

def test_signal_lag_row_based_gap_pinned():
    """Pin: signal_lag is a ROW shift, not a time-based shift — a ticker
    with a gapped bar reaches back across the gap using the previous ROW
    (here, 4 hours), not a strict signal_lag-hours lookback. A future
    time-based lag would be a deliberate test edit."""
    idx = pd.to_datetime(['2026-01-01 10:00', '2026-01-01 11:00', '2026-01-01 15:00'])
    df = pd.DataFrame({'Ticker': ['Z', 'Z', 'Z'], 'sig': [1.0, 2.0, 3.0],
                       'fwd': [0.1, 0.2, 0.3]}, index=idx)
    panel = pb.panel_from_frame(df, 'sig', 'fwd', signal_lag=1)
    assert panel == [
        [{'symbol': 'Z', 'signal': 1.0, 'fwd_return': 0.2}],   # 11:00 <- 10:00's signal
        [{'symbol': 'Z', 'signal': 2.0, 'fwd_return': 0.3}],   # 15:00 <- 11:00's signal (4h gap)
    ]


# ---------------------------------------------------------------------------
# 28. terminal exits convention
# ---------------------------------------------------------------------------

def test_terminal_exits_convention_pinned():
    """Pin: entries/exits count only within-panel transitions — a
    buy-and-hold panel counts the initial entries and zero exits (the
    terminal book is never flushed)."""
    panel = [_period(('A', 0.9, 1.0), ('B', 0.8, 1.0), ('C', 0.7, 1.0))
             for _ in range(10)]
    r = pb.run_policy(panel, pb.top_k(3))
    assert r['entries'] == 3
    assert r['exits'] == 0


# ---------------------------------------------------------------------------
# 29. run_policy_weighted hand-computed
# ---------------------------------------------------------------------------

def test_run_policy_weighted_hand_computed():
    """Pin: run_policy_weighted's expanded metric surface against a
    hand-computed 2-period custom-weight book (locks net_total, entries,
    exits, mean_admitted_k, hit_rate, pct_periods_cash)."""
    p1 = _period(('A', 1.0, 0.10), ('B', 1.0, -0.20))
    p2 = _period(('A', 1.0, 0.05), ('C', 1.0, 0.30))

    def wf(signals):
        n = len(signals)
        return np.array([0.7] + [0.3 / (n - 1)] * (n - 1)) if n > 1 else np.array([1.0])

    r = pb.run_policy_weighted([p1, p2], pb.top_k(2), wf, cost_pct=0.10)
    # period1: w=[.7,.3] over {A,B}; gross=.7*.10+.3*-.20=.01;
    #   turnover (cash -> book) = .7+.3 = 1.0; net = .01-.10*1.0 = -.09
    # period2: w=[.7,.3] over {A,C}; gross=.7*.05+.3*.30=.125;
    #   turnover = max(.7-.7,0)[A] + max(.3-0,0)[C] + max(0-.3,0)[B] = .3;
    #   net = .125-.10*.3 = .095
    assert r['net_total'] == pytest.approx(-0.09 + 0.095)
    assert r['entries'] == 3          # A,B enter period1; C enters period2 (A held)
    assert r['exits'] == 1            # B exits at period2
    assert r['mean_admitted_k'] == 2.0
    assert r['hit_rate'] == 0.5       # period2 net positive, period1 negative
    assert r['pct_periods_cash'] == 0.0
