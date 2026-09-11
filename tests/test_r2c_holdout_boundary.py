"""R2C-05 (2026-08 R2-C wave): FR-01 fixed-calendar holdout boundary +
FR-02 training-window cutoff / A/B runner.

Pure-kernel tests for objective_utils.holdout_boundary / window_cutoff run
real math on synthetic timestamps. scripts/hypersearch_v2.py and
scripts/window_ab.py import torch/lightgbm and cannot be imported on the
dev Mac, so their wiring (the get_holdout_boundary choke-point delegation,
the load_data window mask, the FR-01 instrumentation line, and window_ab's
no-Optuna / no-ratchet / no-save contract) is pinned with source-structure
asserts — including the flag-OFF byte-pin: FIXED_HOLDOUT_DAYS defaults to
None and the None path reproduces the legacy 0.12-quantile expression
bit-for-bit.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from objective_utils import holdout_boundary, window_cutoff

REPO = Path(__file__).resolve().parent.parent
HS = (REPO / 'scripts' / 'hypersearch_v2.py').read_text()
WA = (REPO / 'scripts' / 'window_ab.py').read_text()

BASE = 1_600_000_000  # any epoch anchor


def hourly(n, start=BASE):
    return start + 3600 * np.arange(n, dtype=np.int64)


# ---------------------------------------------------------------------------
# holdout_boundary — legacy quantile path (fixed_days=None)
# ---------------------------------------------------------------------------

class TestHoldoutBoundaryLegacy:
    def test_byte_identical_to_legacy_expression(self):
        # The exact legacy formula: int(np.quantile(t, 1.0 - 0.12))
        for n in (100, 1000, 8760):
            t = hourly(n)
            assert holdout_boundary(t) == int(np.quantile(t, 1.0 - 0.12))
            assert holdout_boundary(t, fixed_days=None,
                                    holdout_fraction=0.12) == \
                int(np.quantile(t, 1.0 - 0.12))

    def test_custom_fraction_forwarded(self):
        t = hourly(500)
        assert holdout_boundary(t, holdout_fraction=0.25) == \
            int(np.quantile(t, 0.75))

    def test_order_independent(self):
        # Pooled multi-ticker arrays are per-ticker concatenations, not
        # globally sorted — both rules must not care.
        rng = np.random.default_rng(7)
        t = hourly(2000)
        shuf = rng.permutation(t)
        assert holdout_boundary(t) == holdout_boundary(shuf)
        assert holdout_boundary(t, fixed_days=60) == \
            holdout_boundary(shuf, fixed_days=60)

    def test_returns_int(self):
        assert isinstance(holdout_boundary(hourly(100)), int)
        assert isinstance(holdout_boundary(hourly(100), fixed_days=10), int)

    def test_proportional_width_scales_with_span(self):
        # The FR-01 defect: the legacy holdout WIDTH depends on the span.
        t_full = hourly(4 * 8760)
        t_1y = hourly(8760)
        w_full = int(t_full.max()) - holdout_boundary(t_full)
        w_1y = int(t_1y.max()) - holdout_boundary(t_1y)
        assert w_full > 3 * w_1y  # ~4x on a 4x span


# ---------------------------------------------------------------------------
# holdout_boundary — fixed-days path
# ---------------------------------------------------------------------------

class TestHoldoutBoundaryFixed:
    def test_exact_arithmetic(self):
        t = hourly(8760)
        assert holdout_boundary(t, fixed_days=60) == \
            int(t.max()) - 60 * 86400

    def test_fractional_days(self):
        t = hourly(100)
        assert holdout_boundary(t, fixed_days=0.5) == \
            int(t.max()) - 43200

    def test_fixed_width_is_span_invariant(self):
        # The FR-01 repair: every arm gets the SAME holdout width.
        t_full = hourly(4 * 8760)
        t_1y = hourly(8760)
        w_full = int(t_full.max()) - holdout_boundary(t_full, fixed_days=60)
        w_1y = int(t_1y.max()) - holdout_boundary(t_1y, fixed_days=60)
        assert w_full == w_1y == 60 * 86400

    def test_holdout_bar_count_on_hourly_grid(self):
        # Rows with t > boundary on a continuous hourly grid: exactly
        # days*24 bars (t > max - days*86400 admits the last days*24 bars,
        # excluding the boundary bar itself).
        t = hourly(8760)
        b = holdout_boundary(t, fixed_days=10)
        assert int((t > b).sum()) == 10 * 24

    def test_purge_interaction(self):
        # final_refit / fold purge semantics are unchanged: a train row
        # survives iff its LABEL window completes on/before the boundary
        # (label_times <= boundary). With fb=24 forward bars the last
        # 10*24 + 24 rows are excluded from training under a fixed 10d
        # holdout (10*24 holdout rows + 24 purged label-crossers), and no
        # surviving label time exceeds the boundary.
        n, fb = 2000, 24
        t = hourly(n)
        label_idx = np.minimum(np.arange(n) + fb, n - 1)
        label_t = t[label_idx]
        b = holdout_boundary(t, fixed_days=10)
        train = np.flatnonzero(label_t <= b)
        assert label_t[train].max() <= b
        assert len(train) == n - (10 * 24 + fb)
        # zero leakage across the boundary in either direction
        holdout = np.flatnonzero(t > b)
        assert len(np.intersect1d(train, holdout)) == 0

    def test_cross_arm_comparability(self):
        # Full-history vs a windowed arm sharing the same panel end: the
        # fixed rule yields identical holdout row sets on the shared
        # region; the proportional rule does not.
        t_full = hourly(8760)
        t_win = t_full[-2000:]  # trailing-window arm
        bf_full = holdout_boundary(t_full, fixed_days=30)
        bf_win = holdout_boundary(t_win, fixed_days=30)
        assert bf_full == bf_win
        assert int((t_full > bf_full).sum()) == int((t_win > bf_win).sum())
        assert holdout_boundary(t_full) != holdout_boundary(t_win)


# ---------------------------------------------------------------------------
# window_cutoff — FR-02 per-ticker trailing mask helper
# ---------------------------------------------------------------------------

class TestWindowCutoff:
    def test_none_means_no_mask(self):
        assert window_cutoff(hourly(100), None) is None

    def test_exact_cutoff(self):
        t = hourly(8760)
        assert window_cutoff(t, 365) == float(t.max()) - 365 * 86400.0

    def test_boundary_row_kept_semantics(self):
        # Contract: rows with t >= cutoff survive — a bar exactly at the
        # cutoff is kept, so a 1-day window on an hourly grid keeps 25
        # bars (24 strictly after + the boundary bar).
        t = hourly(100)
        cut = window_cutoff(t, 1)
        assert int((t >= cut).sum()) == 25

    def test_per_ticker_anchoring(self):
        # Each name's cutoff anchors to its OWN final bar.
        a = hourly(1000)
        b = hourly(900)  # ends 100h earlier
        ca, cb = window_cutoff(a, 30), window_cutoff(b, 30)
        assert ca - cb == 100 * 3600

    def test_degenerate_inputs_disable_mask(self):
        t = hourly(10)
        assert window_cutoff(np.array([], dtype=np.int64), 30) is None
        assert window_cutoff(t, 0) is None
        assert window_cutoff(t, -5) is None
        assert window_cutoff(t, float('nan')) is None
        assert window_cutoff(t, float('inf')) is None

    def test_window_longer_than_history_keeps_everything(self):
        t = hourly(48)  # 2 days of data, 30-day window
        cut = window_cutoff(t, 30)
        assert int((t >= cut).sum()) == len(t)

    def test_window_then_fixed_holdout_composition(self):
        # FR-02 arm mechanics end-to-end on synthetic timestamps: mask to
        # a trailing 90d window, then a fixed 30d holdout — training
        # region is exactly the 60d between cutoff and boundary.
        t = hourly(8760)
        cut = window_cutoff(t, 90)
        kept = t[t >= cut]
        b = holdout_boundary(kept, fixed_days=30)
        train = kept[kept <= b]
        assert len(kept) == 90 * 24 + 1
        assert int((kept > b).sum()) == 30 * 24
        assert (train.max() - train.min()) == 60 * 86400


# ---------------------------------------------------------------------------
# strategy_config flag — default-OFF pin
# ---------------------------------------------------------------------------

class TestFlagDefault:
    def test_fixed_holdout_days_defaults_none(self):
        from strategy_config import FIXED_HOLDOUT_DAYS
        assert FIXED_HOLDOUT_DAYS is None


# ---------------------------------------------------------------------------
# hypersearch_v2 wiring — source-structure pins (torch: not importable here)
# ---------------------------------------------------------------------------

class TestHypersearchWiring:
    def test_choke_point_delegates_to_pure_helper(self):
        body = HS.split('def get_holdout_boundary(', 1)[1]
        body = body.split('\ndef ', 1)[0]
        assert 'holdout_boundary(all_times' in body
        assert '_fixed_holdout_days()' in body
        assert 'holdout_fraction=HOLDOUT_FRACTION' in body
        # the legacy inline formula must NOT survive in the choke point
        assert 'np.quantile' not in body

    def test_legacy_fraction_constant_unchanged(self):
        assert re.search(r'^HOLDOUT_FRACTION = 0\.12', HS, re.M)

    def test_fixed_days_reader_env_and_config(self):
        body = HS.split('def _fixed_holdout_days(', 1)[1]
        body = body.split('\ndef ', 1)[0]
        assert "os.environ.get('TRADER_FIXED_HOLDOUT_DAYS')" in body
        assert 'from strategy_config import FIXED_HOLDOUT_DAYS' in body
        # fail-open to legacy None on any read problem
        assert 'return None' in body

    def test_no_other_boundary_rule_exists(self):
        # every consumer must inherit through the ONE choke point: no
        # second occurrence of the quantile-boundary formula anywhere
        assert HS.count('np.quantile(all_times') == 0

    def test_load_data_window_days_parameter(self):
        sig = HS.split('def load_data(', 1)[1].split('):', 1)[0]
        assert 'window_days=None' in sig
        assert 'if window_days is not None:' in HS
        assert 'window_cutoff(t, window_days)' in HS

    def test_window_mask_before_max_rows_cap(self):
        # the cap's rows/ticker arithmetic must see the windowed panel
        assert HS.index('if window_days is not None:') < \
            HS.index('# Apply --max-rows cap')

    def test_fr01_instrumentation_prints_both_boundaries(self):
        assert 'FR-01 boundary quantile=' in HS
        seg = HS.split('FR-01 boundary quantile=', 1)[1][:600]
        assert 'fixed-' in seg
        assert 'trades=' in seg

    def test_helpers_imported_from_objective_utils(self):
        m = re.search(r'from objective_utils import \(([^)]+)\)', HS)
        assert m and 'holdout_boundary' in m.group(1)
        assert 'window_cutoff' in m.group(1)


# ---------------------------------------------------------------------------
# window_ab runner — contract pins (torch/lightgbm: not importable here)
# ---------------------------------------------------------------------------

class TestWindowAbContract:
    def test_no_optuna(self):
        assert re.search(r'^\s*(import optuna|from optuna)', WA, re.M) \
            is None

    def test_no_ratchet_no_saves_no_state_writes(self):
        assert 'save_model_atomically' not in WA
        assert 'update_after_search' not in WA
        assert 'noisy_ratchet' not in WA
        assert 'save_adaptive_state' not in WA
        assert 'record_trials' not in WA

    def test_lgb_legs_never_touch_disk(self):
        assert 'save=False' in WA

    def test_calls_the_four_pipeline_steps_directly(self):
        for fn in ('final_refit', 'train_lgb_ensemble',
                   'fit_blend_weight_v2', 'evaluate_on_holdout'):
            assert fn in WA, fn

    def test_identical_deflation_pool_per_arm(self):
        assert 'n_trials=n_trials' in WA

    def test_fixed_seed_via_derive_seed(self):
        assert 'derive_seed(args.seed' in WA

    def test_stage0_dump_and_summary(self):
        assert 'write_rows' in WA
        assert '_stage0' in WA
        assert 'window_ab_summary' in WA

    def test_no_cross_retrain_weight_smoothing(self):
        # each arm's blend w must come from its own data only
        assert 'smooth_across_retrains' not in WA

    def test_warns_when_fixed_holdout_inactive(self):
        assert '_fixed_holdout_days()' in WA
        assert 'NOT comparable' in WA

    def test_load_data_receives_window_days(self):
        assert 'window_days=window_days' in WA

    def test_stock_prefix_defaults_stock_data(self):
        # load_data derives the book from the DATA PATH — a stock prefix
        # with the crypto default --data must be re-pointed, never allowed
        # to silently load the crypto panel against the stock champion.
        assert 'stock_training_data.csv' in WA

    def test_fold_sharpe_matches_objective_scoring(self):
        # the "fold-objective wins" half of the FR-02 decision rule must
        # score with the objective's own rule (block-boundary resets when
        # OBJECTIVE_V3 is on)
        assert 'ticker_block_ids' in WA
        assert 'block_ids=_bids' in WA

    def test_one_full_panel_transform_per_arm(self):
        # Jetson perf: exactly one scaler.transform of the full panel per
        # arm (run_arm's precompute) plus the standalone fallback inside
        # _predict_rows — every per-fold/holdout call must reuse `scaled`.
        assert WA.count('scaler.transform(') == 2
        assert 'scaled=scaled' in WA

    def test_holdout_q10_scored_in_same_pass(self):
        # q10 preds ride the same window gather as the mean booster —
        # no second full LSTM/gather pass over the holdout
        assert 'q10_booster=q10b' in WA
