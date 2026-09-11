"""Packet R2C-03 — LGB full refit + honest q10 floor (M3, M4-floor).

Mac-runnable: pure numpy + source pins. NEVER imports scripts/hypersearch_v2
(torch) or lightgbm — the refit wiring is pinned by sparing source-structure
asserts; the cap/purge index math lives in objective_utils and is tested
directly on synthetic data.

[A] objective_utils.lgb_refit_indices (final_refit purge + NaN filter +
    most-recent-first byte-budget cap, in that order)
[B] objective_utils.fixed_boost_rounds (round-count resolution)
[C] strategy_config default-OFF pin (LGB_REFIT_FULL)
[D] q10 meta json byte-compat replica (flag OFF -> legacy bytes)
[E] sparing source-structure asserts on scripts/hypersearch_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from objective_utils import fixed_boost_rounds, lgb_refit_indices

HS = (REPO / 'scripts' / 'hypersearch_v2.py').read_text()
SEG = HS.split('def train_lgb_ensemble', 1)[1].split('\ndef ', 1)[0]


# ---------------------------------------------------------------------------
# [A] lgb_refit_indices — purge / NaN / cap index math
# ---------------------------------------------------------------------------

def _panel(n=200, seed=7, n_nan=0):
    """Two contiguous 'ticker blocks' whose GLOBAL index is not time-sorted
    across blocks (the repo's per-ticker concatenated layout)."""
    rng = np.random.default_rng(seed)
    # block A rows 0..99 at times 1000..1099, block B rows 100..199 at
    # times 1000..1099 too -> global argsort(times) interleaves blocks.
    times = np.concatenate([np.arange(1000, 1100), np.arange(1000, 1100)])
    label_times = times + 24  # label completes 24 "hours" later
    returns = rng.normal(size=n)
    if n_nan:
        returns[rng.choice(n, size=n_nan, replace=False)] = np.nan
    valid = np.arange(n, dtype=np.int64)
    return valid, label_times, times, returns


class TestLgbRefitIndices:
    def test_purge_is_inclusive_on_label_time(self):
        valid, lt, t, r = _panel()
        # boundary exactly at one row's label completion -> row KEPT
        # (final_refit's rule: all_label_times[valid] <= boundary).
        boundary = int(lt[50])
        idx = lgb_refit_indices(valid, lt, t, r, boundary)
        assert 50 in idx
        assert np.all(lt[idx] <= boundary)
        # the very next label time is excluded
        assert 51 not in idx or lt[51] <= boundary

    def test_matches_final_refit_purge_replica(self):
        valid, lt, t, r = _panel(n_nan=17)
        boundary = int(np.quantile(t, 0.88))
        # the exact final_refit expression, replicated
        ref = valid[lt[valid] <= boundary]
        ref = ref[~np.isnan(r[ref])]
        got = lgb_refit_indices(valid, lt, t, r, boundary, max_rows=None)
        assert np.array_equal(got, ref)

    def test_nan_labels_dropped(self):
        valid, lt, t, r = _panel()
        r[[3, 40, 150]] = np.nan
        idx = lgb_refit_indices(valid, lt, t, r, int(lt.max()))
        assert not set([3, 40, 150]) & set(idx.tolist())
        assert not np.isnan(r[idx]).any()

    def test_cap_keeps_most_recent_by_time_across_blocks(self):
        valid, lt, t, r = _panel()
        idx = lgb_refit_indices(valid, lt, t, r, int(lt.max()),
                                max_rows=20)
        assert len(idx) == 20
        # newest 20 bars by TIME = the last 10 timestamps of EACH block
        # (both blocks share the same time range) — the cap must sort by
        # bar time over the GLOBAL index, not just truncate positionally.
        assert set(t[idx].tolist()) == set(range(1090, 1100))
        assert set(idx.tolist()) & set(range(90, 100))  # block A tail
        assert set(idx.tolist()) & set(range(190, 200))  # block B tail

    def test_cap_matches_fold_path_replica(self):
        # The fold path caps as train_idx[np.argsort(all_times[train_idx])]
        # [-max_rows:] — the helper must reproduce that exact selection.
        valid, lt, t, r = _panel(n_nan=9)
        boundary = int(np.quantile(t, 0.9))
        surv = valid[lt[valid] <= boundary]
        surv = surv[~np.isnan(r[surv])]
        ref = surv[np.argsort(t[surv])][-25:]
        got = lgb_refit_indices(valid, lt, t, r, boundary, max_rows=25)
        assert np.array_equal(got, ref)

    def test_cap_runs_after_purge_and_nan(self):
        # A recent row with a NaN label or a purged label must not consume
        # cap slots: with max_rows=5 all five survivors must be usable.
        valid, lt, t, r = _panel()
        r[190:200] = np.nan  # newest block-B rows unusable
        boundary = int(lt[95])  # purges the very newest block-A labels
        idx = lgb_refit_indices(valid, lt, t, r, boundary, max_rows=5)
        assert len(idx) == 5
        assert not np.isnan(r[idx]).any()
        assert np.all(lt[idx] <= boundary)

    def test_no_cap_when_max_rows_none_or_bigger(self):
        valid, lt, t, r = _panel()
        full = lgb_refit_indices(valid, lt, t, r, int(lt.max()),
                                 max_rows=None)
        assert len(full) == len(valid)
        assert np.array_equal(
            full, lgb_refit_indices(valid, lt, t, r, int(lt.max()),
                                    max_rows=10_000))

    def test_empty_valid(self):
        got = lgb_refit_indices(np.array([], dtype=np.int64),
                                np.arange(10), np.arange(10),
                                np.ones(10), 5, max_rows=3)
        assert got.size == 0 and got.dtype == np.int64

    def test_returns_int64_indices(self):
        valid, lt, t, r = _panel()
        idx = lgb_refit_indices(list(range(200)), lt, t, r, int(lt.max()),
                                max_rows=7)
        assert idx.dtype == np.int64 and len(idx) == 7


# ---------------------------------------------------------------------------
# [B] fixed_boost_rounds
# ---------------------------------------------------------------------------

class TestFixedBoostRounds:
    def test_best_iteration_wins(self):
        assert fixed_boost_rounds(137, 500) == 137

    def test_zero_best_falls_to_current(self):
        # lightgbm leaves best_iteration <= 0 when early stopping never
        # fired — the total trained iteration count is the honest budget.
        assert fixed_boost_rounds(0, 500) == 500
        assert fixed_boost_rounds(None, 321) == 321
        assert fixed_boost_rounds(-1, 42) == 42

    def test_neither_usable_returns_none(self):
        assert fixed_boost_rounds(0, 0) is None
        assert fixed_boost_rounds(None, None) is None
        assert fixed_boost_rounds(-3, -1) is None

    def test_garbage_input_falls_through(self):
        assert fixed_boost_rounds('nope', 88) == 88
        assert fixed_boost_rounds(object(), object()) is None

    def test_numpy_ints_accepted(self):
        assert fixed_boost_rounds(np.int64(64), np.int64(500)) == 64
        assert isinstance(fixed_boost_rounds(np.int64(64), 500), int)


# ---------------------------------------------------------------------------
# [C] strategy_config default-OFF pin
# ---------------------------------------------------------------------------

def test_lgb_refit_full_default_off():
    import strategy_config
    assert strategy_config.LGB_REFIT_FULL is False
    # The V3 save path this flag lives inside stays OFF too.
    assert strategy_config.HYPERSEARCH_V3 is False


# ---------------------------------------------------------------------------
# [D] q10 meta json byte-compat replica
# ---------------------------------------------------------------------------

def _meta_writer_replica(floor, n_val, extra):
    """Mirror of both meta writers (in-function save=True and the atomic
    _write_q10_meta) — the source pins in [E] keep this replica honest."""
    m = {'alpha': 0.10, 'floor': round(floor, 6), 'val_rows': int(n_val or 0)}
    m.update(extra or {})
    return json.dumps(m)


class TestQ10MetaByteCompat:
    def test_flag_off_bytes_identical_to_legacy(self):
        legacy = json.dumps({'alpha': 0.10, 'floor': round(-0.8123456, 6),
                             'val_rows': 12345})
        assert _meta_writer_replica(-0.8123456, 12345, None) == legacy
        assert _meta_writer_replica(-0.8123456, 12345, {}) == legacy

    def test_refit_caveat_appends_after_legacy_keys(self):
        extra = {'refit_full': True,
                 'floor_source': 'refit_q10_on_fold_val',
                 'floor_in_sample': True, 'refit_rows': 90000,
                 'fixed_rounds_mean': 137, 'fixed_rounds_q10': 88}
        out = json.loads(_meta_writer_replica(-0.5, 100, extra))
        # legacy keys unchanged, caveat present
        assert out['alpha'] == 0.10 and out['val_rows'] == 100
        assert out['floor_in_sample'] is True
        assert out['floor_source'] == 'refit_q10_on_fold_val'
        assert list(out)[:3] == ['alpha', 'floor', 'val_rows']


# ---------------------------------------------------------------------------
# [E] sparing source-structure asserts on scripts/hypersearch_v2.py
# ---------------------------------------------------------------------------

class TestHypersearchSourcePins:
    def test_flag_helper_reads_strategy_config(self):
        i = HS.index('def _lgb_refit_full():')
        assert 'from strategy_config import LGB_REFIT_FULL' in HS[i:i + 300]

    def test_mean_save_deferred_under_flag(self):
        # Flag OFF: the mean booster saves exactly where legacy did.
        # Flag ON: the save runs AFTER the refit block.
        i_off = SEG.index('if save and not _refit_full:')
        i_refit = SEG.index('if _refit_full:')
        i_on = SEG.index('if save and _refit_full:')
        assert i_off < i_refit < i_on

    def test_refit_block_between_fold_q10_and_floor(self):
        # Ordering: fold q10 training -> refit -> q10_val prediction on the
        # ORIGINAL fold-val rows -> percentile-15 floor. That ordering IS
        # the M4-floor fix (the floor comes from the refit q10).
        i_fold_q10 = SEG.index("params={'objective': 'quantile', 'alpha': "
                               "0.10,")
        i_refit = SEG.index('if _refit_full:')
        i_pred = SEG.index('q10_val = q10.predict(X_val)')
        i_floor = SEG.index('float(np.percentile(q10_val, 15))')
        assert i_fold_q10 < i_refit < i_pred < i_floor

    def test_refit_uses_pure_helpers_with_fold_cap(self):
        # Purge boundary + the SAME max_rows the fold path computed.
        assert 'lgb_refit_indices(' in SEG
        assert 'get_holdout_boundary(all_times), max_rows)' in SEG
        assert SEG.count('fixed_boost_rounds(') == 2  # mean + q10

    def test_refit_trainings_flag_guarded_and_no_early_stopping(self):
        # Every num_iterations refit call sits INSIDE the `if _refit_full:`
        # guard, and no refit call passes a validation set (no val set ->
        # train_lgb attaches no early-stopping callback).
        i_guard = SEG.index('if _refit_full:')
        assert SEG.index("'num_iterations'") > i_guard
        assert SEG.count("'num_iterations'") == 2
        assert 'X_refit, y_refit,' in SEG
        assert 'X_refit, y_refit, X_val' not in SEG

    def test_refit_fail_soft(self):
        assert 'full refit failed (non-fatal)' in SEG
        # the deferred mean save is recovered even when the q10 path dies
        assert 'not _mean_saved' in SEG

    def test_refit_rebind_is_both_or_nothing(self):
        # Hardener pin: the refit trainings land in temporaries and
        # booster/q10 are rebound TOGETHER only after BOTH succeeded — a
        # q10-refit failure (e.g. OOM on the second Dataset copy) must
        # ship the intact FOLD pair, never refit-mean + fold-q10.
        i_mean = SEG.index('_new_mean = train_lgb(')
        i_q10 = SEG.index('_new_q10 = train_lgb(')
        i_bind = SEG.index('booster, q10 = _new_mean, _new_q10')
        assert i_mean < i_q10 < i_bind
        # no direct rebind of booster/q10 from train_lgb inside the guard
        i_guard = SEG.index('if _refit_full:')
        i_end = SEG.index('q10_val = q10.predict(X_val)')
        guard_body = SEG[i_guard:i_end]
        assert ' booster = train_lgb(' not in guard_body
        assert ' q10 = train_lgb(' not in guard_body  # ' ' excludes _new_q10
        # the meta caveat is attached only after the rebind (so it always
        # describes the pair that actually ships)
        assert i_bind < SEG.index('q10._r2c03_meta_extra = {')

    def test_in_function_meta_writer_merges_caveat(self):
        # The save=True writer (legacy path) must merge the SAME caveat
        # the atomic writer does — flag OFF leaves it None -> legacy
        # bytes; the [D] replica mirrors this exact expression.
        assert ("_meta.update(getattr(q10, '_r2c03_meta_extra', None) "
                "or {})") in SEG
        assert 'json.dump(_meta, f)' in SEG

    def test_save_false_tuple_unchanged(self):
        assert 'return (booster, q10, q10_floor_val, n_q10_val)' in SEG

    def test_caveat_travels_on_the_booster_not_the_tuple(self):
        assert '_r2c03_meta_extra' in SEG
        # ... and the atomic-save meta writer merges it (flag OFF -> None
        # -> byte-identical legacy json).
        i_writer = HS.index('def _write_q10_meta')
        w = HS[i_writer:i_writer + 700]
        assert "_r2c03_meta_extra" in w
        assert "_m.update(extra or {})" in w

    def test_atomic_save_path_unchanged(self):
        # R2C-03 must not touch the atomic-save mechanics.
        assert 'save_model_atomically(save_prefix, ship_state, best_cfg,' \
            in HS
        assert 'extra_artifacts=extra_artifacts)' in HS

    def test_legacy_floor_and_cap_expressions_survive(self):
        # The fold path's cap and the floor formula stay verbatim (the
        # replicas in [A]/[D] mirror them).
        assert ('train_idx[np.argsort(all_times[train_idx])][-max_rows:]'
                in SEG)
        assert "{'alpha': 0.10, 'floor': round(floor, 6)," in SEG

    def test_shared_cleanup_still_valid_after_refit(self):
        # The refit branch deletes X_train early; the shared cleanup at the
        # function tail still references it, so the branch must leave a
        # rebindable name behind.
        i_del = SEG.index('del X_train\n')
        assert 'X_train = None' in SEG[i_del:i_del + 200]
        assert 'del X_train, X_val, all_scaled' in SEG


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
