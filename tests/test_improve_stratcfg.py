"""Stage-3 improvement batch — strategy_config.py comment/docstring pins.

Pure, Mac-runnable: imports only strategy_config + stdlib (math, re, pathlib).
Cross-file checks (cooldown derivation, wave-9 flag readers) are done by
reading SOURCE TEXT of other modules, never importing them — several of those
modules (meta_label, backtest, base_loop, crypto_loop, stock_loop,
predict_now, portfolio) pull in torch/lightgbm/numba/sklearn/dotenv, which
are not installed on this dev Mac. Mirrors the source-text-read pattern in
tests/test_review_b10.py.
"""

import re
from pathlib import Path

import strategy_config as sc

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# 1. Policy schema
# --------------------------------------------------------------------------

def test_policy_schemas_match():
    expected = {
        'atr_stop_mult', 'atr_trail_mult', 'trail_activate_pct',
        'stop_floor_pct', 'stop_ceil_pct', 'tp_rr', 'tp_ceil_pct',
        'stop_fallback_pct', 'trail_fallback_pct', 'cooldown_min',
        'lockout_hours',
    }
    assert set(sc.CRYPTO_POLICY) == set(sc.STOCK_POLICY) == expected


# --------------------------------------------------------------------------
# 2. Policy value invariants
# --------------------------------------------------------------------------

def test_policy_value_invariants():
    for policy in (sc.CRYPTO_POLICY, sc.STOCK_POLICY):
        assert 0 < policy['stop_floor_pct'] < policy['stop_ceil_pct']
        assert policy['tp_rr'] > 0
        assert policy['tp_ceil_pct'] > 0
        assert policy['cooldown_min'] > 0
        assert policy['lockout_hours'] > 0
        assert policy['atr_stop_mult'] > 0
        assert policy['atr_trail_mult'] > 0
        assert policy['stop_floor_pct'] <= policy['stop_fallback_pct'] <= policy['stop_ceil_pct']
        assert policy['trail_activate_pct'] > 0


# --------------------------------------------------------------------------
# 3. policy_for dispatch
# --------------------------------------------------------------------------

def test_policy_for_dispatch():
    assert sc.policy_for('crypto') is sc.CRYPTO_POLICY
    assert sc.policy_for('stock') is sc.STOCK_POLICY
    # documents the current 'else -> STOCK' semantics
    assert sc.policy_for('anything_else') is sc.STOCK_POLICY


# --------------------------------------------------------------------------
# 4. Sizing bounds
# --------------------------------------------------------------------------

def test_sizing_bounds():
    assert 0 < sc.RISK_PCT_PER_TRADE < sc.MAX_BOOK_RISK_PCT < 1
    assert 0 < sc.KELLY_CAP <= 1
    assert sc.TILT_MIN < sc.TILT_MAX
    assert sc.TILT_MAX > 1.0
    assert set(sc.PORTFOLIO_VOL_TARGET) == {'crypto', 'stock'}
    assert all(v > 0 for v in sc.PORTFOLIO_VOL_TARGET.values())
    assert sc.MIN_ORDER_NOTIONAL > 0
    assert set(sc.MAX_TRADES_PER_SYMBOL_PER_DAY) == {'crypto', 'stock'}
    assert all(
        isinstance(v, int) and v > 0
        for v in sc.MAX_TRADES_PER_SYMBOL_PER_DAY.values()
    )


# --------------------------------------------------------------------------
# 5. IOC cap ordering
# --------------------------------------------------------------------------

def test_ioc_cap_ordering():
    assert set(sc.IOC_CAP_BPS) == set(sc.IOC_EXIT_CAP_BPS) == {'mega', 'mid', 'spec'}
    assert sc.IOC_CAP_BPS['mega'] < sc.IOC_CAP_BPS['mid'] < sc.IOC_CAP_BPS['spec']
    for k in sc.IOC_CAP_BPS:
        assert sc.IOC_EXIT_CAP_BPS[k] >= sc.IOC_CAP_BPS[k]


# --------------------------------------------------------------------------
# 6. Entry windows shape
# --------------------------------------------------------------------------

def test_entry_windows_shape():
    windows = sc.STOCK_ENTRY_WINDOWS_ET
    assert isinstance(windows, list) and len(windows) > 0
    lo, hi = '09:30', '16:00'
    prev_end = None
    for w in windows:
        assert isinstance(w, tuple) and len(w) == 2
        start, end = w
        assert isinstance(start, str) and isinstance(end, str)
        assert start < end
        assert lo <= start <= hi
        assert lo <= end <= hi
        if prev_end is not None:
            assert prev_end < start  # ascending, non-overlapping
        prev_end = end


# --------------------------------------------------------------------------
# 7. Wave-9 flags default-off
# --------------------------------------------------------------------------

def test_wave9_flags_default_off():
    # These constants have no production reader yet (see the RESERVED banner
    # in strategy_config.py); if this test fails because you WIRED one,
    # update it in the same change — never flip a flag without its wiring.
    assert sc.EDGE_KELLY_ENABLED is False
    assert sc.CRYPTO_CS_RANK_ENABLED is False
    assert sc.CRYPTO_TREND_GATE_ENABLED is False
    assert sc.CONCENTRATION_ENABLED is False
    assert sc.TIER_SIZING_ENABLED is False
    assert sc.CONVICTION_SIGNAL_FLOOR is None
    assert sc.CONVICTION_META_FLOOR is None
    assert sc.CONVICTION_RATIO_FLOOR is None
    assert sc.CONVICTION_K_MIN <= sc.TIER_A_K <= sc.CONVICTION_K_MAX
    assert sc.META_CALIBRATION_MODE == 'legacy'


# --------------------------------------------------------------------------
# 8. Wave-9 flags have no production reader (tripwire for silent wiring)
# --------------------------------------------------------------------------

def test_wave9_flags_have_no_production_reader():
    modules = [
        'base_loop.py', 'crypto_loop.py', 'stock_loop.py', 'predict_now.py',
        'portfolio.py', 'backtest.py', 'meta_label.py',
    ]
    tokens = [
        'EDGE_KELLY_ENABLED', 'CONCENTRATION_ENABLED', 'TIER_SIZING_ENABLED',
        'CRYPTO_TREND_GATE_ENABLED',
    ]
    for mod in modules:
        text = (REPO / mod).read_text()
        for token in tokens:
            assert not re.search(r'\b' + re.escape(token) + r'\b', text), (
                f"{mod} now references {token} — wiring landed; update "
                f"test_wave9_flags_default_off consciously in the same change."
            )


# --------------------------------------------------------------------------
# 9. cooldown derivation in sync between meta_label.py and backtest.py
# --------------------------------------------------------------------------

def test_cooldown_derivation_in_sync():
    pattern = re.compile(
        r"max\(1,\s*int\(math\.ceil\(policy\['cooldown_min'\]\s*/\s*60\)\)\)"
    )
    for mod in ('meta_label.py', 'backtest.py'):
        text = (REPO / mod).read_text()
        matches = pattern.findall(text)
        assert len(matches) == 1, (
            f"{mod}: expected exactly one occurrence of the verbatim cooldown "
            f"derivation, found {len(matches)}"
        )


# --------------------------------------------------------------------------
# 10. Default-off offline flags
# --------------------------------------------------------------------------

def test_default_off_offline_flags():
    # Same rationale as test 7 — these flip only via the documented Jetson
    # procedures (see the comments above each constant in strategy_config.py).
    assert sc.IMPACT_COST_ENABLED is False
    assert sc.UNIQUENESS_WEIGHTS_ENABLED is False
    assert sc.OBJECTIVE_LONG_ONLY is False
    assert sc.PREDICTION_CACHE_ENABLED is False
    assert sc.CROSS_BOOK_RHO == 1.0
