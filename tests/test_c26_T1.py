"""Packet T1 — model-fit honesty (D22/D23/D24-part/D25/D05-threshold, B12).

Mac-runnable: pure numpy + fees + source pins. NEVER imports
scripts/hypersearch_v2 (torch) or lightgbm.

[A] blend_fit.fit_blend_weight_v2 (NNLS + overlap-corrected SE gate)
[B] blend_fit.smooth_across_retrains
[C] EXISTING fit_blend_weight regression pins (additive-only promise)
[D] objective_utils.v3_trade_threshold_range (fees-anchored)
[E] objective_utils.ticker_block_ids
[F] objective_utils.simulate_trades_core (legacy-parity fuzz + extensions)
[G] objective_utils.refit_epoch_budget
[H] strategy_config default-OFF flag pins
[I] sparing source-structure asserts on scripts/hypersearch_v2.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from blend_fit import (fit_blend_weight, fit_blend_weight_v2,
                       smooth_across_retrains)
from objective_utils import (refit_epoch_budget, simulate_trades_core,
                             ticker_block_ids, v3_trade_threshold_range)

HS = (REPO / 'scripts' / 'hypersearch_v2.py').read_text()


# ---------------------------------------------------------------------------
# [A] fit_blend_weight_v2
# ---------------------------------------------------------------------------

class TestFitBlendWeightV2:
    def test_strong_signal_recovery(self):
        rng = np.random.default_rng(10)
        a = rng.normal(size=2000)
        y = a + rng.normal(0, 0.05, 2000)
        b = rng.normal(size=2000)
        fit = fit_blend_weight_v2(a, b, y, forward_bars=1)
        assert fit['significant'] is True
        assert fit['w_raw'] == pytest.approx(1.0, abs=0.05)
        # significant -> shrink 0.5/0.5: w = 0.5*clip(w_raw,0,1) + 0.25
        assert fit['w'] == pytest.approx(
            min(max(0.5 * min(max(fit['w_raw'], 0.0), 1.0) + 0.25, 0.0), 1.0))
        assert fit['w'] == pytest.approx(0.75, abs=0.02)
        assert fit['n'] == 2000 and fit['n_eff'] == 2000.0

    def test_leg_swap_symmetry(self):
        rng = np.random.default_rng(10)
        a = rng.normal(size=2000)
        y = a + rng.normal(0, 0.05, 2000)
        b = rng.normal(size=2000)
        w = fit_blend_weight_v2(a, b, y, forward_bars=1)['w']
        w_sw = fit_blend_weight_v2(b, a, y, forward_bars=1)['w']
        assert w + w_sw == pytest.approx(1.0, abs=1e-9)

    def test_pure_noise_is_exactly_half(self):
        rng = np.random.default_rng(11)
        y = rng.normal(size=2000)
        a = rng.normal(size=2000)
        b = rng.normal(size=2000)
        fit = fit_blend_weight_v2(a, b, y, forward_bars=1)
        assert fit['significant'] is False
        assert fit['w'] == 0.5  # EXACTLY the simple average

    def test_thin_input_fails_safe(self):
        fit = fit_blend_weight_v2([1, 2, 3], [3, 2, 1], [0, 1, 0])
        assert fit == {'w': 0.5, 'w_raw': None, 'se': None,
                       'significant': False, 'n': 3, 'n_eff': None}

    def test_identical_legs_fail_safe(self):
        rng = np.random.default_rng(5)
        a = rng.normal(size=100)
        fit = fit_blend_weight_v2(a, a.copy(), rng.normal(size=100))
        assert fit['w'] == 0.5 and fit['w_raw'] is None
        assert fit['significant'] is False

    def test_overlap_correction_monotonicity(self):
        # Borderline case: significant at fb=1, se grows by sqrt(48) at
        # fb=48 and the SAME unshrunk estimate flips to insignificant.
        rng = np.random.default_rng(12)
        n = 3000
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        y = 0.7 * a + 0.3 * b + rng.normal(0, 1.5, n)
        f1 = fit_blend_weight_v2(a, b, y, forward_bars=1)
        f48 = fit_blend_weight_v2(a, b, y, forward_bars=48)
        assert f48['w_raw'] == f1['w_raw']          # estimator unchanged
        assert f48['se'] == pytest.approx(f1['se'] * np.sqrt(48), rel=1e-9)
        assert f48['n_eff'] == pytest.approx(n / 48)
        assert f1['significant'] is True and f48['significant'] is False
        assert f48['w'] == 0.5

    def test_nan_rows_dropped(self):
        rng = np.random.default_rng(6)
        a = rng.normal(size=300)
        y = a + rng.normal(0, 0.05, 300)
        b = rng.normal(size=300)
        a[:50] = np.nan
        fit = fit_blend_weight_v2(a, b, y, forward_bars=1)
        assert fit['n'] == 250
        assert np.isfinite(fit['w'])

    def test_significant_path_bounded_quarter_three_quarters(self):
        # With shrink_to=0.5/shrink_lambda=0.5 the significant branch is
        # 0.5*clip(w_raw,0,1)+0.25 in [0.25, 0.75]; insignificant is 0.5.
        rng = np.random.default_rng(7)
        for _ in range(50):
            n = int(rng.integers(20, 400))
            a = rng.normal(size=n)
            b = rng.normal(size=n)
            y = rng.uniform(-1, 2) * a + rng.uniform(-1, 2) * b \
                + rng.normal(0, rng.uniform(0.01, 2.0), n)
            fit = fit_blend_weight_v2(a, b, y,
                                      forward_bars=int(rng.integers(1, 49)))
            assert 0.25 <= fit['w'] <= 0.75


# ---------------------------------------------------------------------------
# [B] smooth_across_retrains
# ---------------------------------------------------------------------------

class TestSmoothAcrossRetrains:
    def test_no_prev_clamps_new(self):
        assert smooth_across_retrains(0.6) == 0.6
        assert smooth_across_retrains(0.9) == 0.75
        assert smooth_across_retrains(0.1) == 0.25
        assert smooth_across_retrains(0.6, w_prev=float('nan')) == 0.6

    def test_with_prev_clamps_mean(self):
        assert smooth_across_retrains(0.6, 0.4) == pytest.approx(0.5)
        assert smooth_across_retrains(0.75, 0.95) == 0.75   # clamp hi
        assert smooth_across_retrains(0.30, 0.10) == 0.25   # clamp lo


# ---------------------------------------------------------------------------
# [C] fit_blend_weight regression pins (function untouched by T1)
# ---------------------------------------------------------------------------

class TestFitBlendWeightUntouched:
    def test_seeded_outputs_pinned(self):
        rng = np.random.default_rng(42)
        y = rng.normal(size=500)
        lgb = y + rng.normal(0, 0.5, 500)
        lstm = 0.5 * y + rng.normal(0, 1.0, 500)
        assert fit_blend_weight(lstm, lgb, y, objective='nnls',
                                shrink_lambda=0.0) \
            == pytest.approx(0.18262816432659776, abs=1e-12)
        assert fit_blend_weight(lstm, lgb, y, objective='sharpe',
                                threshold=0.2) \
            == pytest.approx(0.365, abs=1e-12)
        assert fit_blend_weight(lstm, lgb, y, objective='nnls') \
            == pytest.approx(0.3413140821632989, abs=1e-12)


# ---------------------------------------------------------------------------
# [D] v3_trade_threshold_range
# ---------------------------------------------------------------------------

class TestV3TradeThresholdRange:
    def test_computed_live_from_fees(self):
        from fees import required_edge_pct, FLAT_SPREAD_PCT
        for asset in ('crypto', 'stock'):
            floor = required_edge_pct(asset,
                                      spread_pct=FLAT_SPREAD_PCT[asset])
            lo, hi = v3_trade_threshold_range(asset)
            assert lo == round(0.8 * floor, 2)
            assert hi == round(min(2.5 * floor, 2.0), 2)
            assert hi > lo

    def test_today_values_sanity(self):
        # Values float automatically with the fee schedule; today:
        assert v3_trade_threshold_range('crypto') == [0.96, 2.0]
        assert v3_trade_threshold_range('stock') == [0.18, 0.57]
        # unknown asset types price as stock (fees convention)
        assert v3_trade_threshold_range('etf') \
            == v3_trade_threshold_range('stock')

    def test_default_search_space_untouched(self):
        from adaptive_config import DEFAULT_SEARCH_SPACE
        assert DEFAULT_SEARCH_SPACE['trade_threshold'] == [0.05, 1.0]


# ---------------------------------------------------------------------------
# [E] ticker_block_ids
# ---------------------------------------------------------------------------

class TestTickerBlockIds:
    BOUNDS = {'AAA': (0, 30), 'BBB': (30, 55), 'CCC': (55, 100)}

    def test_dict_and_list_forms_agree(self):
        rows = np.array([0, 29, 30, 54, 55, 99])
        ids_d = ticker_block_ids(rows, self.BOUNDS)
        ids_l = ticker_block_ids(rows, list(self.BOUNDS.values()))
        assert (ids_d == ids_l).all()
        assert list(ids_d) == [0, 0, 1, 1, 2, 2]

    def test_subset_invariance_under_masks(self):
        rng = np.random.default_rng(3)
        rows = np.sort(rng.integers(0, 100, size=40))
        ids = ticker_block_ids(rows, self.BOUNDS)
        mask = rng.random(40) < 0.5
        assert (ids[mask] == ticker_block_ids(rows[mask], self.BOUNDS)).all()


# ---------------------------------------------------------------------------
# [F] simulate_trades_core
# ---------------------------------------------------------------------------

def _legacy_simulate_trades(predictions, actual_returns, threshold,
                            forward_bars, txn_cost_pct, long_only):
    """Verbatim reference copy of the pre-T1 hypersearch_v2.simulate_trades
    loop body (the delegated-away legacy walk)."""
    n = len(predictions)
    trade_returns = []
    entries = []
    i = 0
    while i < n:
        p = predictions[i]
        r = actual_returns[i]
        if p > threshold and np.isfinite(r):
            trade_returns.append(r - txn_cost_pct)
            entries.append(i)
            i += forward_bars
        elif (not long_only) and p < -threshold and np.isfinite(r):
            trade_returns.append(-r - txn_cost_pct)
            entries.append(i)
            i += forward_bars
        else:
            i += 1
    return (np.asarray(trade_returns, dtype=np.float64),
            np.asarray(entries, dtype=np.int64))


class TestSimulateTradesCore:
    def test_flag_off_parity_fuzz(self):
        # ~200 seeded random cases: block_ids=None/long_veto=None must be
        # EXACTLY the legacy walk (returns AND entries).
        rng = np.random.default_rng(2026)
        for case in range(200):
            n = int(rng.integers(1, 400))
            preds = rng.normal(0, 1.0, n)
            rets = rng.normal(0, 2.0, n)
            rets[rng.random(n) < 0.05] = np.nan       # NaN labels
            threshold = float(rng.uniform(0.0, 1.5))
            fb = int(rng.integers(1, 49))
            cost = float(rng.uniform(0.0, 1.0))
            long_only = bool(rng.integers(0, 2))
            ref_r, ref_e = _legacy_simulate_trades(preds, rets, threshold,
                                                   fb, cost, long_only)
            got_r, got_e = simulate_trades_core(preds, rets, threshold, fb,
                                                cost, long_only=long_only)
            np.testing.assert_array_equal(got_e, ref_e)
            np.testing.assert_array_equal(got_r, ref_r)

    def test_boundary_reset_takes_next_block_signal(self):
        n = 60
        block_ids = np.repeat([0, 1], 30)
        preds = np.zeros(n)
        preds[28] = 1.0     # entry 2 bars before block 0 ends
        preds[30] = 1.0     # profitable signal at the NEXT block's first row
        rets = np.full(n, 0.5)
        # WITH block reset: the hold stops at the boundary, scan resumes at
        # row 30 and its signal IS taken.
        _, e_blk = simulate_trades_core(preds, rets, 0.5, 24, 0.1,
                                        block_ids=block_ids)
        assert list(e_blk) == [28, 30]
        # WITHOUT block ids the fb=24 hold swallows row 30 (legacy defect).
        _, e_leg = simulate_trades_core(preds, rets, 0.5, 24, 0.1)
        assert list(e_leg) == [28]

    def test_hold_never_spans_a_boundary(self):
        rng = np.random.default_rng(9)
        n = 200
        block_ids = np.repeat([0, 1, 2, 3], 50)
        preds = rng.normal(0, 1.0, n)
        rets = rng.normal(0, 1.0, n)
        fb = 24
        _, entries = simulate_trades_core(preds, rets, 0.3, fb, 0.1,
                                          block_ids=block_ids)
        for e1, e2 in zip(entries[:-1], entries[1:]):
            hold_end = min(e1 + fb, (e1 // 50 + 1) * 50)  # block end
            assert e2 >= hold_end

    def test_long_veto_blocks_longs_only(self):
        preds = np.zeros(10)
        preds[2] = 1.0     # long signal, vetoed
        preds[5] = -1.0    # short signal, "vetoed" index — must still trade
        rets = np.full(10, 0.5)
        veto = np.zeros(10, dtype=bool)
        veto[2] = True
        veto[5] = True
        r, e = simulate_trades_core(preds, rets, 0.5, 2, 0.1,
                                    long_veto=veto)
        assert list(e) == [5]                      # short taken, long blocked
        assert r[0] == pytest.approx(-0.5 - 0.1)   # short leg payoff
        # long_veto=None is inert (== all-False)
        r0, e0 = simulate_trades_core(preds, rets, 0.5, 2, 0.1)
        r1, e1 = simulate_trades_core(preds, rets, 0.5, 2, 0.1,
                                      long_veto=np.zeros(10, dtype=bool))
        np.testing.assert_array_equal(e0, e1)
        np.testing.assert_array_equal(r0, r1)
        assert list(e0) == [2, 5]


# ---------------------------------------------------------------------------
# [G] refit_epoch_budget
# ---------------------------------------------------------------------------

class TestRefitEpochBudget:
    def test_median(self):
        assert refit_epoch_budget([7, 12, 20]) == 12

    def test_empty_and_invalid(self):
        assert refit_epoch_budget([]) is None
        assert refit_epoch_budget(None) is None
        assert refit_epoch_budget([np.nan, -1]) is None
        assert refit_epoch_budget([np.nan, -1, 8]) == 8

    def test_clamps(self):
        assert refit_epoch_budget([0, 0, 0]) == 1
        assert refit_epoch_budget([500, 900], max_epochs=60) == 60


# ---------------------------------------------------------------------------
# [H] strategy_config default pins
# ---------------------------------------------------------------------------

def test_flags_default_off():
    import strategy_config
    assert strategy_config.HYPERSEARCH_V3 is False
    assert strategy_config.OBJECTIVE_V3 is False


# ---------------------------------------------------------------------------
# [I] sparing source-structure asserts (hypersearch_v2 not importable here)
# ---------------------------------------------------------------------------

class TestHypersearchSourcePins:
    def test_new_machinery_present(self):
        assert 'def final_refit' in HS
        assert 'purge_val_labels' in HS
        assert 'extra_artifacts' in HS

    def test_evaluate_on_holdout_signature_extended(self):
        sig = HS.split('def evaluate_on_holdout', 1)[1][:400]
        assert 'lgb_booster=None' in sig
        assert 'lstm_weight=None' in sig

    def test_pinned_trade_threshold_literal_verbatim(self):
        # test_grp_training regex-pins this exact fallback literal; the V3
        # override must come AFTER it, never replace it.
        assert "_space.get('trade_threshold', [0.05, 1.0])" in HS

    def test_legacy_post_save_lgb_call_flag_guarded(self):
        # The guard must sit directly on the legacy post-save LGB call
        # (train_lgb_ensemble(save_prefix, best_scaler, ...) verbatim).
        k = HS.index('if not _v3:')
        window = HS[k:k + 200]
        assert 'train_lgb_ensemble(save_prefix, best_scaler' in window

    def test_flag_on_gate_ordering_lgb_before_holdout(self):
        # Under HYPERSEARCH_V3 the LGB legs train BEFORE the holdout gate
        # (save=False call site precedes the evaluate_on_holdout call).
        assert HS.index('save=False)') \
            < HS.index('holdout_report = evaluate_on_holdout')

    def test_manifest_last_invariant_with_extras(self):
        # save_model_atomically: the merged artifacts+extras tmp+rename
        # loop runs BEFORE the OOF npz write, which runs BEFORE the
        # manifest write (B-2 manifest-LAST invariant, strengthened: with
        # extras present ALL artifacts incl. boosters are on disk before
        # the manifest appears).
        body = HS.split('def save_model_atomically', 1)[1]
        body = body.split('\ndef ', 1)[0]
        i_extras = body.index("{**artifacts, **(extra_artifacts or {})}.items()")
        i_npz = body.index('write_oof_npz')
        i_manifest = body.index("manifest = {")
        assert i_extras < i_npz < i_manifest
