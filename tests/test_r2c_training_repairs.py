"""R2C-04 (2026-08 R2-C wave): training-loop repairs + seed plumbing +
config truth — L1, L2, L3, L5, L6, L8.

Pure-kernel tests for the objective_utils helpers (derive_seed,
lagged_regime_series, embargo_end_time) run real math on synthetic data.
scripts/hypersearch_v2.py imports torch and cannot be imported on the dev
Mac, so its wiring (TRAINING_REPAIRS_V1 flag guards, TRAINER_SEED
threading, --preset default None) is pinned with source-structure asserts,
including the flag-OFF byte-pin: every legacy expression must remain
verbatim on the default path.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from objective_utils import derive_seed, embargo_end_time, lagged_regime_series

REPO = Path(__file__).resolve().parent.parent
HS = (REPO / 'scripts' / 'hypersearch_v2.py').read_text()
RP = (REPO / 'run_pipeline.py').read_text()


# ---------------------------------------------------------------------------
# derive_seed (L3 — the FR-02 / FR-13 determinism prerequisite)
# ---------------------------------------------------------------------------

class TestDeriveSeed:
    def test_deterministic(self):
        a = derive_seed(42, 'v2_search', 17, 2)
        b = derive_seed(42, 'v2_search', 17, 2)
        assert a == b

    def test_distinct_across_every_token(self):
        base = derive_seed(42, 'v2_search', 17, 2)
        assert derive_seed(43, 'v2_search', 17, 2) != base
        assert derive_seed(42, 'stock_v2_search', 17, 2) != base
        assert derive_seed(42, 'v2_search', 18, 2) != base
        assert derive_seed(42, 'v2_search', 17, 1) != base

    def test_folds_trials_sampler_refit_get_disjoint_streams(self):
        seeds = {derive_seed(7, 'v2_search', t, f)
                 for t in range(50) for f in range(3)}
        seeds.add(derive_seed(7, 'v2_search', 'sampler'))
        seeds.add(derive_seed(7, 'v2_search', 'refit'))
        assert len(seeds) == 50 * 3 + 2  # no collisions on realistic keys

    def test_range_fits_every_consumer(self):
        # torch.manual_seed, np.random.default_rng and optuna TPESampler
        # (numpy RandomState) all accept [0, 2**32)
        for parts in (('x',), ('v2_search', 0, 0), ('refit',)):
            s = derive_seed(123456789, *parts)
            assert isinstance(s, int)
            assert 0 <= s < 2 ** 32

    def test_seeds_numpy_rng(self):
        rng1 = np.random.default_rng(derive_seed(1, 'a', 0))
        rng2 = np.random.default_rng(derive_seed(1, 'a', 0))
        assert np.array_equal(rng1.permutation(100), rng2.permutation(100))


# ---------------------------------------------------------------------------
# lagged_regime_series (L2 — regime-penalty look-ahead repair)
# ---------------------------------------------------------------------------

class TestLaggedRegimeSeries:
    def test_warmup_rows_are_nan(self):
        out = lagged_regime_series(np.ones(200), forward_bars=12, window=50)
        assert np.all(np.isnan(out[:50 - 1 + 12]))
        assert np.all(np.isfinite(out[50 - 1 + 12:]))

    def test_hand_computed_values(self):
        # window=3, fb=2: out[t] = mean(r[t-4], r[t-3], r[t-2]) * 3/2
        r = np.arange(10, dtype=float)
        out = lagged_regime_series(r, forward_bars=2, window=3)
        assert np.all(np.isnan(out[:4]))
        for t in range(4, 10):
            expected = np.mean(r[t - 4:t - 1]) * (3 / 2)
            assert out[t] == pytest.approx(expected)

    def test_constant_returns_scale_by_window_over_fb(self):
        out = lagged_regime_series(np.full(300, 0.5), forward_bars=24,
                                   window=50)
        valid = out[np.isfinite(out)]
        assert len(valid) > 0
        assert np.allclose(valid, 0.5 * 50 / 24)

    def test_no_look_ahead(self):
        # The value at t must use only returns COMPLETED by t — returns
        # stamped at rows <= t - fb. Perturbing any later return (the
        # legacy defect region: the mask at t embedded returns through
        # t+fb) must not move out[t].
        rng = np.random.default_rng(0)
        r = rng.normal(size=300)
        fb, w = 12, 50
        base = lagged_regime_series(r, forward_bars=fb, window=w)
        t = 200
        r2 = r.copy()
        r2[t - fb + 1:] = 99.0  # everything not yet completed at t
        pert = lagged_regime_series(r2, forward_bars=fb, window=w)
        assert pert[t] == pytest.approx(base[t])

    def test_differs_from_legacy_forward_series(self):
        # Legacy: trailing mean over rows t-49..t of FORWARD returns —
        # out_legacy[t] depends on r[t] itself. Repaired must not.
        rng = np.random.default_rng(1)
        r = rng.normal(size=300)
        fb, w = 12, 50
        base = lagged_regime_series(r, forward_bars=fb, window=w)
        r2 = r.copy()
        r2[200] += 100.0
        pert = lagged_regime_series(r2, forward_bars=fb, window=w)
        assert pert[200] == pytest.approx(base[200])  # own row: no effect
        # ...but the row that HAS completed it (200 + fb) does move
        assert pert[200 + fb] != pytest.approx(base[200 + fb])

    def test_nonfinite_returns_treated_as_zero(self):
        r = np.full(200, 1.0)
        r[100] = np.nan
        out = lagged_regime_series(r, forward_bars=2, window=3)
        # Rows whose window covers index 100 average with a zero in place
        t = 100 + 2 + 1  # window at t=103 covers rows 99,100,101
        assert out[t] == pytest.approx((1.0 + 0.0 + 1.0) / 3 * (3 / 2))

    def test_short_and_empty_inputs(self):
        assert np.all(np.isnan(lagged_regime_series(np.ones(10),
                                                    forward_bars=12,
                                                    window=50)))
        assert lagged_regime_series(np.array([]), 12, 50).size == 0


# ---------------------------------------------------------------------------
# embargo_end_time (L6 — embargo denominated in bars)
# ---------------------------------------------------------------------------

class TestEmbargoEndTime:
    def _hourly(self, n, start=0):
        return start + 3600 * np.arange(n, dtype=np.int64)

    def test_hourly_grid_matches_legacy_seconds_rule(self):
        # Continuous hourly grid, on-grid boundary: the n-th distinct bar
        # strictly after the boundary IS boundary + n*3600 — identical to
        # the legacy `t >= boundary + n_bars*3600` admission point.
        times = self._hourly(500)
        boundary = int(times[100])
        for n_bars in (1, 12, 40):
            assert embargo_end_time(times, boundary, n_bars) == \
                boundary + n_bars * 3600

    def test_rth_grid_counts_trading_bars_not_calendar_hours(self):
        # Stock RTH grid: 7 hourly bars/day, 5 days/week (weekend gaps).
        # A 40-bar embargo must span ~6 trading days, NOT 40 calendar
        # hours (which the legacy rule shrank to ~11 RTH bars).
        bars = []
        t = 0
        for day in range(20):
            if day % 7 in (5, 6):  # weekend
                continue
            day_start = day * 86400
            bars.extend(day_start + 3600 * h for h in range(7))
        times = np.asarray(bars, dtype=np.int64)
        boundary = int(times[10])  # mid first week
        end = embargo_end_time(times, boundary, 40)
        # exactly the 40th bar strictly after the boundary
        after = times[times > boundary]
        assert end == float(after[39])
        # and far beyond the legacy 40-calendar-hour point
        assert end > boundary + 40 * 3600

    def test_duplicate_timestamps_collapse_to_one_bar(self):
        # Pooled multi-ticker rows share bar timestamps — 6 tickers on the
        # same grid must count each bar ONCE.
        base = self._hourly(200)
        pooled = np.concatenate([base] * 6)
        boundary = int(base[50])
        assert embargo_end_time(pooled, boundary, 10) == \
            embargo_end_time(base, boundary, 10)

    def test_zero_or_negative_bars_returns_boundary(self):
        times = self._hourly(50)
        assert embargo_end_time(times, 7200, 0) == 7200.0
        assert embargo_end_time(times, 7200, -3) == 7200.0

    def test_empty_grid_returns_boundary(self):
        assert embargo_end_time(np.array([], dtype=np.int64), 123, 5) == 123.0

    def test_embargo_beyond_grid_swallows_region(self):
        times = self._hourly(20)
        assert embargo_end_time(times, int(times[15]), 10) == float('inf')

    def test_off_grid_boundary_counts_strictly_after(self):
        times = self._hourly(50)
        # boundary between bar 1 (3600) and bar 2 (7200): bars strictly
        # after are 7200, 10800, ... -> 3rd is 14400
        assert embargo_end_time(times, 5000, 3) == 14400.0


# ---------------------------------------------------------------------------
# strategy_config flags (default-OFF / default-None pins)
# ---------------------------------------------------------------------------

class TestFlagDefaults:
    def test_trainer_seed_defaults_none(self):
        import strategy_config
        assert strategy_config.TRAINER_SEED is None

    def test_training_repairs_defaults_off(self):
        import strategy_config
        assert strategy_config.TRAINING_REPAIRS_V1 is False


# ---------------------------------------------------------------------------
# Source-structure pins on scripts/hypersearch_v2.py (torch — not
# importable on the dev Mac). Flag-OFF byte-pin: the legacy expressions
# must survive verbatim on the default path.
# ---------------------------------------------------------------------------

class TestHypersearchSourceWiring:
    # --- flag-OFF byte-pin: legacy expressions still present verbatim ---
    def test_legacy_val_loss_expression_retained(self):
        assert ('nn.functional.huber_loss(vo, yvb).item() * xvb.size(0)'
                in HS)

    def test_legacy_embargo_expression_retained(self):
        assert 'embargo_seconds = seq_len * EMBARGO_MULTIPLIER * 3600' in HS
        assert '(t >= t_train_end + embargo_seconds)' in HS

    def test_legacy_regime_series_retained(self):
        assert 'trailing_mean * (window / max(forward_bars, 1))' in HS

    def test_legacy_unseeded_permutation_retained_in_both_loops(self):
        # fold loop + final_refit both keep the ambient-RNG shuffle as the
        # None-seed branch
        assert HS.count('np.random.permutation(n_train)') == 2

    # --- TRAINING_REPAIRS_V1 wiring (L1/L2/L5/L6) ---
    def test_repairs_flag_reader_defined_default_false(self):
        body = HS.split('def _training_repairs', 1)[1][:800]
        assert 'TRAINING_REPAIRS_V1' in body
        assert 'return False' in body

    def test_repaired_val_loss_uses_trial_criterion(self):
        # L1: repaired branch scores with the trial's criterion + weights
        k = HS.index('v_raw = criterion(vo, yvb)')
        window = HS[k:k + 400]
        assert 'torch.clamp(torch.abs(yvb) + 1.0' in window
        assert '(v_raw * v_w).sum().item()' in window

    def test_regime_repair_calls_pure_helper_and_masks_warmup(self):
        seg = HS.split('def compute_regime_sharpes', 1)[1]
        seg = seg.split('\ndef ', 1)[0]
        assert 'lagged_regime_series(actual_returns, forward_bars' in seg
        assert 'np.isfinite(rolling_ret)' in seg  # NaN warmup joins NO regime
        assert '_training_repairs()' in seg

    def test_oom_probe_restore_present_in_fold_loop_and_refit(self):
        # L5: pristine-init snapshot + restore around BOTH memory probes
        assert HS.count('_init_snap = {k: v.detach().cpu().clone()') == 2
        assert HS.count('model.load_state_dict(_init_snap)') == 2

    def test_embargo_bars_branch_present(self):
        seg = HS.split('def get_walk_forward_folds', 1)[1]
        seg = seg.split('\ndef ', 1)[0]
        assert 'embargo_end_time(search_times, t_train_end' in seg
        assert '_training_repairs()' in seg

    # --- TRAINER_SEED wiring (L3) ---
    def test_seed_reader_env_over_config(self):
        body = HS.split('def _trainer_seed', 1)[1][:800]
        assert 'TRADER_TRAINER_SEED' in body
        assert 'TRAINER_SEED' in body
        assert 'return None' in body

    def test_fold_seed_derived_from_study_trial_fold(self):
        assert ('derive_seed(base_seed, study_name, trial.number,\n'
                '                                        fold_idx)' in HS
                or re.search(r'derive_seed\(base_seed,\s*study_name,\s*'
                             r'trial\.number,\s*fold_idx\)', HS))
        assert 'torch.manual_seed(fold_seed)' in HS

    def test_final_refit_gains_seed_kwarg(self):
        sig = HS.split('def final_refit', 1)[1][:300]
        assert 'seed=None' in sig
        body = HS.split('def final_refit', 1)[1]
        assert 'torch.manual_seed(int(seed))' in body
        assert 'refit_rng.permutation(n_train)' in body

    def test_refit_call_site_passes_derived_seed(self):
        assert re.search(r"derive_seed\(_seed_base,\s*study_name,\s*'refit'\)",
                         HS)

    def test_sampler_seed_threaded_default_none(self):
        seg = HS.split('optuna.samplers.TPESampler(', 1)[1][:300]
        assert re.search(r"derive_seed\(_seed_base,\s*study_name,\s*'sampler'\)",
                         seg)
        assert 'if _seed_base is not None else None' in seg

    def test_create_objective_receives_study_name(self):
        sig = HS.split('def create_objective', 1)[1][:400]
        assert "study_name=''" in sig
        assert 'study_name=study_name' in HS  # main() passes it

    # --- L8: --preset config truth ---
    def test_preset_cli_default_is_none(self):
        m = re.search(r"add_argument\('--preset',\s*type=str,\s*default=(\S+?),",
                      HS)
        assert m, '--preset argument not found'
        assert m.group(1) == 'None'

    def test_load_data_resolves_preset_from_indicator_config(self):
        assert ('preset_name = preset_override or '
                'load_indicator_config()["preset"]' in HS)

    def test_run_pipeline_still_passes_stationary_explicitly(self):
        # behavior preserved on the production path: run_pipeline pins the
        # preset itself and never relies on the CLI default
        assert RP.count("'--preset', 'stationary'") >= 2


class TestFlagGuardStructure:
    """Hardening pins: the repaired branches must be GUARDED by the flag
    with the legacy expression as the default (else / None) path — not
    merely coexist with it. Each regex demands if-repair...else-legacy in
    that order, exactly once."""

    def test_val_loss_repair_guarded_legacy_is_else(self):
        # L1: if _repairs -> trial criterion; else -> legacy huber
        pat = re.compile(
            r'if _repairs:(?:(?!\belse\b).)*?'
            r'v_raw = criterion\(vo, yvb\)(?:(?!\belse\b).)*?'
            r'else:\s*\n\s*val_loss_sum \+= '
            r'nn\.functional\.huber_loss\(vo, yvb\)\.item\(\) \* '
            r'xvb\.size\(0\)',
            re.S)
        assert len(pat.findall(HS)) == 1

    def test_embargo_repair_guarded_legacy_is_else(self):
        # L6: if _repairs -> bar-counted _val_start (n_bars WITHOUT *3600);
        # else -> legacy calendar-seconds admission
        pat = re.compile(
            r'if _repairs:(?:(?!\belse\b).)*?'
            r'_val_start = embargo_end_time\(search_times, t_train_end,\s*'
            r'seq_len \* EMBARGO_MULTIPLIER\)(?:(?!\belse\b).)*?'
            r'else:(?:(?!\bif\b).)*?'
            r'\(t >= t_train_end \+ embargo_seconds\)',
            re.S)
        assert len(pat.findall(HS)) == 1

    def test_regime_repair_guarded_legacy_is_else(self):
        # L2: if _training_repairs() -> lagged kernel + NaN exclusion;
        # else -> legacy forward-embedding series
        pat = re.compile(
            r'if _training_repairs\(\):(?:(?!\belse\b).)*?'
            r'lagged_regime_series\(actual_returns, forward_bars'
            r'(?:(?!\belse\b).)*?'
            r'else:(?:(?!\bdef\b).)*?'
            r'trailing_mean \* \(window / max\(forward_bars, 1\)\)',
            re.S)
        assert len(pat.findall(HS)) == 1

    def test_oom_probe_restore_guarded_both_sites(self):
        # L5: every _init_snap snapshot AND restore sits under if _repairs
        for anchor in ('_init_snap = {k: v.detach().cpu().clone()',
                       'model.load_state_dict(_init_snap)'):
            starts = [m.start() for m in re.finditer(re.escape(anchor), HS)]
            assert len(starts) == 2  # fold loop + final_refit
            for s in starts:
                assert 'if _repairs:' in HS[max(0, s - 500):s]

    def test_seeding_guarded_none_never_calls_manual_seed(self):
        # L3: flag-OFF (seed None) must never touch torch.manual_seed —
        # every call sits directly under a not-None guard
        assert HS.count('torch.manual_seed(') == 2
        assert ('if fold_seed is not None:\n'
                '                        torch.manual_seed(fold_seed)') in HS
        assert ('if seed is not None:\n'
                '                    torch.manual_seed(int(seed))') in HS

    def test_batch_shuffle_ternaries_default_to_ambient_rng(self):
        # L3: the None branch of both shuffle ternaries is the verbatim
        # legacy ambient-RNG call
        pat = re.compile(
            r'perm = \((\w+_rng)\.permutation\(n_train\)\s*\n'
            r'\s*if \1 is not None\s*\n'
            r'\s*else np\.random\.permutation\(n_train\)\)')
        assert len(pat.findall(HS)) == 2
