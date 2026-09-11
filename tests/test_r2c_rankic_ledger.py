"""R2C-07 (2026-08 R2-C wave): FR-05 cross-sectional rank-IC certificate
lines + FR-08 retrain-gain ledger.

Mac-runnable: pure numpy/pandas + source pins. NEVER imports
scripts/hypersearch_v2 (torch) or lightgbm; retrain_ledger's heavy
scorer functions (_load_stack/_score_stack/record_retrain_gain) keep
their torch/joblib/lightgbm imports lazy, so importing the module here
is safe and only the pure kernels are executed.

R2C-07 ships DIRECT (measurement/instrumentation) — no new flag; the
D25 flag-OFF report key convention is pinned at source level instead:
'cs_rank_ic' enters the holdout report ONLY inside the `if blended:`
block, leaving the flag-OFF report key-for-key identical.

[A] objective_utils.cs_rank_ic — kernel property tests
[B] retrain_ledger.paired_scores — paired kernel
[C] retrain_ledger.stack_valid_rows / trailing_purged_rows
[D] retrain_ledger.append_ledger_row — capped round-trip
[E] retrain_ledger.incumbent_paths — the --shadow slot correction
[F] source-structure pins on scripts/hypersearch_v2.py
[G] retrain_ledger.window_gather_plan — compact windowed-gather plan
[H] record_retrain_gain end-to-end (heavy internals monkeypatched) —
    incl. the state-threading fix: main's update_after_search re-saves
    its own adaptive_state object AFTER the ledger call, so the row must
    be appended into THAT object, or the later save clobbers it.
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from objective_utils import cs_rank_ic, _average_ranks
from retrain_ledger import (paired_scores, stack_valid_rows,
                            trailing_purged_rows, append_ledger_row,
                            incumbent_paths, window_gather_plan,
                            record_retrain_gain, LEDGER_CAP, LEDGER_KEY,
                            MIN_LEDGER_BARS)

HS = (REPO / 'scripts' / 'hypersearch_v2.py').read_text()

BASE = 1_700_000_000


def make_panel(n_groups, group_size, ic_sign=+1, seed=0, start=BASE):
    """Synthetic per-timestamp panel: y iid, preds = rank-monotone
    (ic_sign=+1) or anti-monotone (-1) transform of y within each group."""
    rng = np.random.default_rng(seed)
    gids, preds, ys = [], [], []
    for k in range(n_groups):
        y = rng.normal(size=group_size)
        # strictly monotone (anti-)transform of y => per-group Spearman +/-1
        p = ic_sign * (np.exp(y) + 3.0)
        gids.append(np.full(group_size, start + 3600 * k))
        preds.append(p)
        ys.append(y)
    return (np.concatenate(preds), np.concatenate(ys),
            np.concatenate(gids))


# ---------------------------------------------------------------------------
# [A] cs_rank_ic
# ---------------------------------------------------------------------------

class TestCsRankIcProperties:
    def test_perfect_rank_is_one(self):
        p, y, g = make_panel(8, 6, ic_sign=+1, seed=1)
        r = cs_rank_ic(p, y, g)
        assert r['n_groups'] == 8
        assert r['n_skipped'] == 0
        assert r['mean'] == pytest.approx(1.0)
        # per-group ICs all exactly 1 => zero dispersion
        assert r['se'] == pytest.approx(0.0)

    def test_anti_rank_is_minus_one(self):
        p, y, g = make_panel(8, 6, ic_sign=-1, seed=2)
        r = cs_rank_ic(p, y, g)
        assert r['mean'] == pytest.approx(-1.0)
        assert r['n_groups'] == 8

    def test_thin_groups_skipped(self):
        # groups of 4 (< default min_group=5) are counted skipped, never
        # scored; a 5-name group just clears the bar
        p4, y4, g4 = make_panel(3, 4, seed=3)
        p5, y5, g5 = make_panel(2, 5, seed=4, start=BASE + 10 * 86400)
        r = cs_rank_ic(np.r_[p4, p5], np.r_[y4, y5], np.r_[g4, g5])
        assert r['n_groups'] == 2
        assert r['n_skipped'] == 3
        assert r['mean'] == pytest.approx(1.0)

    def test_min_group_override(self):
        p, y, g = make_panel(3, 4, seed=5)
        r = cs_rank_ic(p, y, g, min_group=4)
        assert r['n_groups'] == 3
        assert r['n_skipped'] == 0

    def test_nan_rows_shrink_group_membership(self):
        # 6-name group with 2 NaN preds -> 4 finite rows -> skipped
        p, y, g = make_panel(1, 6, seed=6)
        p = p.copy()
        p[[0, 3]] = np.nan
        r = cs_rank_ic(p, y, g)
        assert r['n_groups'] == 0
        assert r['n_skipped'] == 1
        assert r['mean'] is None

    def test_degenerate_constant_preds_skipped(self):
        y = np.arange(6, dtype=float)
        p = np.ones(6)
        g = np.full(6, BASE)
        r = cs_rank_ic(p, y, g)
        assert r['n_groups'] == 0
        assert r['n_skipped'] == 1

    def test_empty_input(self):
        r = cs_rank_ic([], [], [])
        assert r['mean'] is None and r['se'] is None
        assert r['n_groups'] == 0 and r['n_skipped'] == 0
        assert len(r['splits']) == 4
        assert all(s['mean'] is None and s['n_groups'] == 0
                   for s in r['splits'])

    def test_matches_pandas_spearman(self):
        # independent reference: per-group pandas method='spearman'
        rng = np.random.default_rng(7)
        p, y, g = [], [], []
        for k in range(12):
            m = int(rng.integers(5, 11))
            p.append(rng.normal(size=m))
            y.append(rng.normal(size=m))
            g.append(np.full(m, BASE + 3600 * k))
        p, y, g = np.concatenate(p), np.concatenate(y), np.concatenate(g)
        r = cs_rank_ic(p, y, g)
        ref = []
        for gid in np.unique(g):
            m = g == gid
            ref.append(pd.Series(p[m]).corr(pd.Series(y[m]),
                                            method='spearman'))
        assert r['n_groups'] == 12
        assert r['mean'] == pytest.approx(np.mean(ref), abs=1e-12)
        assert r['se'] == pytest.approx(np.std(ref, ddof=1) / np.sqrt(12),
                                        abs=1e-12)

    def test_ties_use_midranks(self):
        # hand case with tied preds; reference via pandas spearman
        p = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        y = np.array([0.5, 1.0, 3.0, 2.0, 4.0])
        g = np.full(5, BASE)
        r = cs_rank_ic(p, y, g)
        ref = pd.Series(p).corr(pd.Series(y), method='spearman')
        assert r['mean'] == pytest.approx(ref, abs=1e-12)

    def test_row_order_invariance(self):
        p, y, g = make_panel(6, 7, seed=8)
        rng = np.random.default_rng(9)
        perm = rng.permutation(len(p))
        r1 = cs_rank_ic(p, y, g)
        r2 = cs_rank_ic(p[perm], y[perm], g[perm])
        assert r1['mean'] == pytest.approx(r2['mean'])
        assert r1['splits'] == r2['splits']

    def test_splits_are_chronological_quarters(self):
        # first 5 groups perfect, last 5 anti — the split means must
        # follow the array_split layout [3,3,2,2] over sorted group ids
        pa, ya, ga = make_panel(5, 6, ic_sign=+1, seed=10, start=BASE)
        pb, yb, gb = make_panel(5, 6, ic_sign=-1, seed=11,
                                start=BASE + 100 * 3600)
        r = cs_rank_ic(np.r_[pa, pb], np.r_[ya, yb], np.r_[ga, gb])
        means = [s['mean'] for s in r['splits']]
        sizes = [s['n_groups'] for s in r['splits']]
        assert sizes == [3, 3, 2, 2]
        assert means[0] == pytest.approx(1.0)
        assert means[1] == pytest.approx((1.0 + 1.0 - 1.0) / 3.0)
        assert means[2] == pytest.approx(-1.0)
        assert means[3] == pytest.approx(-1.0)

    def test_json_serializable(self):
        p, y, g = make_panel(4, 6, seed=12)
        json.dumps(cs_rank_ic(p, y, g))

    def test_average_ranks_midrank_contract(self):
        x = np.array([3.0, 1.0, 3.0, 2.0])
        # sorted: 1(0), 2(1), 3(2), 3(3) -> ties at positions 2,3 -> 2.5
        assert list(_average_ranks(x)) == [2.5, 0.0, 2.5, 1.0]


# ---------------------------------------------------------------------------
# [B] paired_scores
# ---------------------------------------------------------------------------

class TestPairedScores:
    def test_hand_computed(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        a = y + 1.0          # mse 1, ic 1
        b = np.array([4.0, 3.0, 2.0, 1.0])  # anti: ic -1
        s = paired_scores(a, b, y)
        assert s['n'] == 4
        assert s['mse_a'] == pytest.approx(1.0)
        assert s['mse_b'] == pytest.approx(np.mean((b - y) ** 2))
        assert s['ic_a'] == pytest.approx(1.0)
        assert s['ic_b'] == pytest.approx(-1.0)

    def test_pairing_mask_is_shared(self):
        # a NaN in EITHER prediction removes the row from BOTH scores
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        b = np.array([2.0, 2.0, 3.0, np.nan, 5.0])
        s = paired_scores(a, b, y)
        assert s['n'] == 3
        keep = [0, 2, 4]
        assert s['mse_a'] == pytest.approx(
            np.mean((a[keep] - y[keep]) ** 2))
        assert s['mse_b'] == pytest.approx(
            np.mean((b[keep] - y[keep]) ** 2))

    def test_nan_y_drops_row(self):
        y = np.array([1.0, np.nan, 3.0])
        s = paired_scores(y * 0 + 1, y * 0 + 2, y)
        assert s['n'] == 2

    def test_empty_all_none(self):
        s = paired_scores([], [], [])
        assert s == {'mse_a': None, 'mse_b': None, 'ic_a': None,
                     'ic_b': None, 'n': 0}

    def test_small_n_no_ic(self):
        s = paired_scores([1.0, 2.0], [2.0, 1.0], [1.0, 2.0])
        assert s['n'] == 2
        assert s['mse_a'] is not None
        assert s['ic_a'] is None and s['ic_b'] is None

    def test_zero_variance_pred_no_ic(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        s = paired_scores(np.ones(4), y.copy(), y)
        assert s['ic_a'] is None
        assert s['ic_b'] == pytest.approx(1.0)

    def test_json_serializable(self):
        json.dumps(paired_scores([1.0, 2.0, 3.0], [3.0, 2.0, 1.0],
                                 [1.0, 2.0, 4.0]))


# ---------------------------------------------------------------------------
# [C] row-selection kernels
# ---------------------------------------------------------------------------

class TestStackValidRows:
    def test_mirrors_valid_indices(self):
        b = {'A': (0, 100), 'B': (100, 130), 'C': (130, 135)}
        rows = stack_valid_rows(b, 30)
        expect = np.r_[np.arange(30, 100),   # A
                       np.arange(130, 130)]  # B has exactly 30 -> dropped
        # B: end - start == 30 == seq_len -> NOT > seq_len -> dropped;
        # C: 5 bars -> dropped
        assert np.array_equal(rows, np.arange(30, 100))
        assert rows.dtype == np.int64
        assert len(expect) == 70  # sanity on the hand construction

    def test_accepts_pair_iterable(self):
        rows = stack_valid_rows([(0, 10), (10, 25)], 4)
        assert np.array_equal(rows, np.r_[np.arange(4, 10),
                                          np.arange(14, 25)])

    def test_empty(self):
        assert stack_valid_rows({}, 10).size == 0
        assert stack_valid_rows({'A': (0, 5)}, 10).size == 0


class TestTrailingPurgedRows:
    def test_window_and_purge(self):
        n = 24 * 30  # 30 days hourly
        t = BASE + 3600 * np.arange(n, dtype=np.int64)
        fb = 24
        mask = trailing_purged_rows(t, days=7.0, forward_bars=fb)
        t_end = t.max()
        kept = t[mask]
        assert kept.min() > t_end - 7 * 86400
        assert kept.max() <= t_end - fb * 3600
        # exact count: 7*24 window minus the fb purged bars
        assert mask.sum() == 7 * 24 - fb

    def test_purge_dominates_short_history(self):
        t = BASE + 3600 * np.arange(10, dtype=np.int64)
        mask = trailing_purged_rows(t, days=7.0, forward_bars=24)
        assert mask.sum() == 0

    def test_empty(self):
        m = trailing_purged_rows([])
        assert m.dtype == bool and m.size == 0

    def test_purge_counts_bars_not_calendar_hours_on_rth_grid(self):
        # Stock RTH grid: 7 hourly bars/day. forward_bars=24 must purge
        # the last 24 TRADING bars (~3.4 trading days), not 24 calendar
        # hours (~7 bars) — the L6 lesson applied to the ledger window.
        t = np.array([BASE + d * 86400 + h * 3600
                      for d in range(30) for h in range(7)], dtype=np.int64)
        fb = 24
        mask = trailing_purged_rows(t, days=7.0, forward_bars=fb)
        u = np.unique(t)
        kept = t[mask]
        # nothing kept inside the final fb bars
        assert kept.max() == u[u.size - fb - 1]
        # trailing 7 calendar days = last 49 bars; minus the 24 purged
        assert mask.sum() == 49 - fb
        # the old calendar-hours rule would have kept far more rows
        calendar_kept = ((t > t.max() - 7 * 86400)
                         & (t <= t.max() - fb * 3600)).sum()
        assert calendar_kept > mask.sum()

    def test_identical_on_continuous_hourly_grid(self):
        # crypto's 24/7 hourly grid: bar-count purge == calendar purge
        t = BASE + 3600 * np.arange(24 * 20, dtype=np.int64)
        fb = 24
        mask = trailing_purged_rows(t, days=7.0, forward_bars=fb)
        legacy = (t > t.max() - 7 * 86400) & (t <= t.max() - fb * 3600)
        assert np.array_equal(mask, legacy)

    def test_pooled_duplicate_timestamps_share_bar_grid(self):
        # two tickers on the same grid: distinct-bar counting must not
        # treat duplicates as extra bars
        t1 = BASE + 3600 * np.arange(24 * 20, dtype=np.int64)
        pooled = np.r_[t1, t1]
        m1 = trailing_purged_rows(t1, days=7.0, forward_bars=24)
        mp = trailing_purged_rows(pooled, days=7.0, forward_bars=24)
        assert np.array_equal(mp, np.r_[m1, m1])


# ---------------------------------------------------------------------------
# [D] ledger append round-trip
# ---------------------------------------------------------------------------

class TestAppendLedgerRow:
    def row(self, i):
        return {'date': f'2026-08-{i:02d}', 'mse_inc': 0.1, 'mse_new': 0.09,
                'ic_inc': 0.01, 'ic_new': 0.02, 'n_bars': 144}

    def test_creates_key_and_appends(self):
        state = {'asset_type': 'crypto'}
        append_ledger_row(state, self.row(1))
        append_ledger_row(state, self.row(2))
        assert [r['date'] for r in state[LEDGER_KEY]] == \
            ['2026-08-01', '2026-08-02']

    def test_cap_keeps_newest(self):
        state = {}
        for i in range(1, 12):
            append_ledger_row(state, self.row(i), cap=5)
        assert len(state[LEDGER_KEY]) == 5
        assert state[LEDGER_KEY][0]['date'] == '2026-08-07'
        assert state[LEDGER_KEY][-1]['date'] == '2026-08-11'

    def test_default_cap_is_two_years_weekly(self):
        assert LEDGER_CAP == 104

    def test_row_copied_not_aliased(self):
        state = {}
        r = self.row(1)
        append_ledger_row(state, r)
        r['mse_inc'] = 999
        assert state[LEDGER_KEY][0]['mse_inc'] == 0.1

    def test_json_round_trip(self):
        state = {'best_score': 1.2}
        append_ledger_row(state, self.row(3))
        back = json.loads(json.dumps(state))
        assert back[LEDGER_KEY] == state[LEDGER_KEY]
        assert back['best_score'] == 1.2


# ---------------------------------------------------------------------------
# [E] incumbent slot resolution (the --shadow slot correction)
# ---------------------------------------------------------------------------

class TestIncumbentPaths:
    def test_challenger_save_reads_champion_artifacts(self):
        paths, src = incumbent_paths('', 'challenger_')
        assert src == 'champion'
        assert paths['model_v2.pth'] == 'model_v2.pth'
        assert paths['config_v2.pkl'] == 'config_v2.pkl'
        paths, src = incumbent_paths('stock_', 'stock_challenger_')
        assert src == 'champion'
        assert paths['scaler_v2.pkl'] == 'stock_scaler_v2.pkl'

    def test_same_slot_save_reads_prev(self):
        for p in ('', 'stock_'):
            paths, src = incumbent_paths(p, p)
            assert src == 'prev'
            assert paths['model_v2.pth'] == f'{p}model_v2.pth.prev'
            assert paths['feature_cols_v2.pkl'] == \
                f'{p}feature_cols_v2.pkl.prev'

    def test_all_core_artifacts_covered(self):
        paths, _ = incumbent_paths('', '')
        assert set(paths) == {'model_v2.pth', 'config_v2.pkl',
                              'scaler_v2.pkl', 'feature_cols_v2.pkl'}


# ---------------------------------------------------------------------------
# [F] source-structure pins on scripts/hypersearch_v2.py
# ---------------------------------------------------------------------------

class TestHypersearchSourcePins:
    def test_cs_rank_ic_imported_from_objective_utils(self):
        assert 'cs_rank_ic)' in HS.split('_STATUS_FILE')[0]

    def test_lstm_leg_kept_before_blend_overwrite(self):
        i_copy = HS.index('lstm_preds = preds.copy()')
        i_over = HS.index('preds = blend_preds\n                blended = True')
        assert i_copy < i_over

    def test_d25_key_convention_cs_rank_ic_only_when_blended(self):
        # The base report literal must stay key-for-key identical when
        # not blended: 'cs_rank_ic' never appears between the report
        # literal and the blended guard, and exactly one report
        # assignment exists, after `if blended:`.
        i_report = HS.index("report = {'sharpe'")
        i_blended = HS.index('# ONLY when blended', i_report)
        assert 'cs_rank_ic' not in HS[i_report:i_blended]
        i_guard = HS.index('if blended:', i_blended)
        assert HS.count("report['cs_rank_ic']") == 1
        assert HS.index("report['cs_rank_ic']") > i_guard

    def test_certificate_ic_lines_fail_soft(self):
        seg = HS[HS.index('# ONLY when blended'):
                 HS.index('def save_model_atomically')]
        assert 'FR-05 cs-rank-IC failed (non-fatal)' in seg
        # crypto wide-CI caveat printed on the certificate
        assert 'FR-05 caveat: crypto cross-sections' in seg

    def test_gate_time_per_fold_print_present_and_fail_soft(self):
        i = HS.index('[FR-05] fold')
        # inside a try with a non-fatal except
        assert 'per-fold cs-rank-IC failed (non-fatal)' in HS
        # runs BEFORE the final holdout gate call
        assert i < HS.index('holdout_report = evaluate_on_holdout')
        # crypto caveat on the per-fold read too
        assert '[FR-05] caveat: crypto cross-sections' in HS

    def test_ledger_call_site_after_atomic_save(self):
        i_save = HS.index('save_model_atomically(save_prefix')
        i_ledger = HS.index('from retrain_ledger import record_retrain_gain')
        assert i_save < i_ledger
        # after the legacy post-save LGB training so boosters exist in
        # both modes
        i_legacy_lgb = HS.index('train_lgb_ensemble(save_prefix')
        assert i_legacy_lgb < i_ledger
        # fail-soft wrapper
        seg = HS[i_ledger - 600:i_ledger + 900]
        assert 'try:' in seg
        assert 'retrain-gain recording failed' in seg

    def test_no_new_flag_minted(self):
        # R2C-07 ships direct: no strategy_config flag reads were added
        # for the rank-IC lines or the ledger.
        assert 'RANK_IC' not in HS
        assert 'RETRAIN_LEDGER' not in HS

    def test_ledger_call_threads_live_adaptive_state(self):
        # The clobber guard: update_after_search re-saves main's
        # adaptive_state object AFTER the ledger call — the row must be
        # appended into that very object (state=adaptive_state), not a
        # freshly loaded copy the later save would overwrite.
        i = HS.index('from retrain_ledger import record_retrain_gain')
        seg = HS[i:i + 700]
        assert 'state=adaptive_state' in seg
        # and update_after_search really does run after the call site
        assert HS.index('update_after_search(adaptive_state', i) > i

    def test_record_retrain_gain_accepts_state_kwarg(self):
        p = inspect.signature(record_retrain_gain).parameters
        assert 'state' in p and p['state'].default is None


# ---------------------------------------------------------------------------
# [G] window_gather_plan — compact windowed gather
# ---------------------------------------------------------------------------

class TestWindowGatherPlan:
    def test_equivalent_to_full_panel_gather(self):
        rng = np.random.default_rng(20)
        M = rng.normal(size=(300, 4))
        rows = np.r_[np.arange(40, 60), np.arange(200, 230)]
        sl = 16
        src, remap = window_gather_plan(rows, sl)
        offsets = np.arange(-sl, 0)
        direct = M[rows[:, None] + offsets[None, :]]
        assert np.array_equal(M[src][remap], direct)

    def test_src_is_sorted_unique_and_much_smaller(self):
        rows = np.arange(50, 80)   # 30 contiguous rows, sl=8
        src, remap = window_gather_plan(rows, 8)
        assert np.array_equal(src, np.unique(src))
        # contiguous windows overlap (and the row itself is excluded):
        # sources 42..78 = 37 unique rows, not 30*8 = 240
        assert np.array_equal(src, np.arange(42, 79))
        assert remap.shape == (30, 8)

    def test_empty_rows(self):
        src, remap = window_gather_plan(np.array([], dtype=np.int64), 12)
        assert src.size == 0
        assert remap.shape == (0, 12)

    def test_excludes_the_row_itself(self):
        # gather_windows offsets convention: r consumes r-sl..r-1 only
        src, _ = window_gather_plan(np.array([100]), 5)
        assert np.array_equal(src, np.arange(95, 100))


# ---------------------------------------------------------------------------
# [H] record_retrain_gain end-to-end (heavy internals monkeypatched)
# ---------------------------------------------------------------------------

class TestRecordRetrainGain:
    """Full flow with _load_stack/_score_stack faked (no torch/joblib) and
    adaptive_config.BASE_DIR pointed at tmp_path — covers row selection,
    the paired row schema, the MIN_LEDGER_BARS floor, and the
    state-threading clobber guard."""

    N_PER = 200          # hourly bars per ticker
    SEQ_LEN = 16
    FB = 24

    def _setup(self, monkeypatch, tmp_path, fb=None):
        import retrain_ledger as rl
        import adaptive_config
        monkeypatch.setattr(adaptive_config, 'BASE_DIR', tmp_path)
        fb = fb if fb is not None else self.FB
        tA = BASE + 3600 * np.arange(self.N_PER, dtype=np.int64)
        env = {
            'all_times': np.r_[tA, tA],
            'boundaries': {'A': (0, self.N_PER),
                           'B': (self.N_PER, 2 * self.N_PER)},
        }
        rng = np.random.default_rng(30)
        env['returns'] = rng.normal(size=2 * self.N_PER)
        env['feats'] = rng.normal(size=(2 * self.N_PER, 3)).astype(np.float32)
        stacks = {
            'fresh': {'config': {'forward_bars': fb, 'target_kind': 'raw',
                                 'seq_len': self.SEQ_LEN},
                      'booster': object()},
            'incumbent': {'config': {'forward_bars': fb,
                                     'target_kind': 'raw',
                                     'seq_len': self.SEQ_LEN},
                          'booster': None},
        }
        monkeypatch.setattr(rl, '_load_stack',
                            lambda paths, lgb, label: dict(stacks[label]))
        # incumbent (no booster): y+0.1 -> mse 0.01; fresh: y+0.05 -> 0.0025
        monkeypatch.setattr(
            rl, '_score_stack',
            lambda stack, feats, cols, rows:
                env['returns'][rows] + (0.1 if stack['booster'] is None
                                        else 0.05))
        return rl, env

    def _call(self, rl, env, state, save_prefix='challenger_'):
        return rl.record_retrain_gain(
            '', save_prefix, 'crypto',
            env['feats'], ['a', 'b', 'c'], {self.FB: env['returns']},
            env['all_times'], ['A', 'B'], env['boundaries'], state=state)

    def test_row_schema_and_paired_values(self, monkeypatch, tmp_path):
        rl, env = self._setup(monkeypatch, tmp_path)
        state = {'asset_type': 'crypto'}
        row = self._call(rl, env, state)
        assert row is not None
        assert row['incumbent_source'] == 'champion'  # --shadow slot fix
        assert row['slot_new'] == 'challenger_'
        assert row['blend_new'] is True and row['blend_inc'] is False
        assert row['fb_inc'] == self.FB and row['fb_new'] == self.FB
        # trailing 7d window on a 200-bar hourly grid: bars 32..175 kept
        # (168-bar window minus the 24-bar purge) x 2 tickers
        assert row['n_bars'] == 144 * 2
        assert row['n_bars'] >= MIN_LEDGER_BARS
        assert row['mse_inc'] == pytest.approx(0.01)
        assert row['mse_new'] == pytest.approx(0.0025)
        assert row['ic_inc'] == pytest.approx(1.0)
        assert row['ic_new'] == pytest.approx(1.0)

    def test_row_appended_to_callers_state_and_persisted(
            self, monkeypatch, tmp_path):
        rl, env = self._setup(monkeypatch, tmp_path)
        state = {'asset_type': 'crypto'}
        row = self._call(rl, env, state)
        # the CALLER'S object carries the row (update_after_search will
        # re-save this object later — the row survives that save)
        assert state[LEDGER_KEY] == [row]
        saved = json.loads(
            (tmp_path / 'adaptive_state_crypto.json').read_text())
        assert saved[LEDGER_KEY] == [row]

    def test_state_none_loads_and_saves_directly(self, monkeypatch,
                                                 tmp_path):
        rl, env = self._setup(monkeypatch, tmp_path)
        row = self._call(rl, env, state=None)
        assert row is not None
        saved = json.loads(
            (tmp_path / 'adaptive_state_crypto.json').read_text())
        assert saved[LEDGER_KEY] == [row]

    def test_min_bars_floor_writes_nothing(self, monkeypatch, tmp_path):
        # fb=190 on a 200-bar grid: the purge swallows everything below
        # the seq_len-valid region -> no scorable rows -> no row, state
        # untouched, no state file written
        rl, env = self._setup(monkeypatch, tmp_path, fb=190)
        state = {'asset_type': 'crypto'}
        row = rl.record_retrain_gain(
            '', 'challenger_', 'crypto',
            env['feats'], ['a', 'b', 'c'], {self.FB: env['returns']},
            env['all_times'], ['A', 'B'], env['boundaries'], state=state)
        assert row is None
        assert LEDGER_KEY not in state
        assert not (tmp_path / 'adaptive_state_crypto.json').exists()

    def test_fail_soft_returns_none(self, monkeypatch, tmp_path):
        rl, env = self._setup(monkeypatch, tmp_path)

        def _boom(paths, lgb, label):
            raise FileNotFoundError('incumbent stack incomplete')
        monkeypatch.setattr(rl, '_load_stack', _boom)
        state = {'asset_type': 'crypto'}
        assert self._call(rl, env, state) is None
        assert LEDGER_KEY not in state

    def test_same_slot_save_uses_prev(self, monkeypatch, tmp_path):
        rl, env = self._setup(monkeypatch, tmp_path)
        state = {'asset_type': 'crypto'}
        row = self._call(rl, env, state, save_prefix='')
        assert row['incumbent_source'] == 'prev'
