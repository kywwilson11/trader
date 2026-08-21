"""c26 packet V3 — meta-label learning-curve harness (B04.3), Mac-runnable.

Pins the pure kernel meta_curve.py: temporal-block subsample plans
(deterministic, contiguous, never iid), tie-aware rank AUC vs a brute-force
oracle, cross-seed veto flip rates (row + symbol level), inverse-power-law
recovery/degenerate handling/inversion, the B04.3 empirical floor, the
end-to-end report assembly (json-safe), the VETO_PROB pin against
meta_label.META_VETO_PROB, and the CLI's lazy-import discipline (--help
must succeed on a machine with no lightgbm).

numpy + pytest + stdlib only.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

import meta_curve  # noqa: E402


# ---------------------------------------------------------------------------
# resolve_n_grid
# ---------------------------------------------------------------------------

class TestResolveNGrid:
    def test_drops_over_pool_and_appends_all_point(self):
        got = meta_curve.resolve_n_grid(1000)
        assert got == [100, 200, 400, 800, 1000]
        assert got[-1] == 1000  # 'all' point appended once

    def test_grid_value_equal_to_pool_dedupes(self):
        got = meta_curve.resolve_n_grid(3200)
        assert got == [100, 200, 400, 800, 1600, 3200]
        assert got.count(3200) == 1

    def test_dedup_and_sort_of_custom_grid(self):
        got = meta_curve.resolve_n_grid(500, n_grid=(400, 100, 400, 100, 900))
        assert got == [100, 400, 500]

    def test_tiny_pool(self):
        assert meta_curve.resolve_n_grid(50) == [50]
        assert meta_curve.resolve_n_grid(0) == []


# ---------------------------------------------------------------------------
# build_subsample_plan
# ---------------------------------------------------------------------------

class TestSubsamplePlan:
    POOL, BLOCK, SEEDS = 4000, 50, 5

    def _plan(self):
        return meta_curve.build_subsample_plan(
            self.POOL, n_seeds=self.SEEDS, block_len=self.BLOCK)

    @staticmethod
    def _runs(idx):
        return 1 + int((np.diff(idx) != 1).sum())

    def test_contiguity_and_determinism(self):
        plan = self._plan()
        expected_ns = [100, 200, 400, 800, 1600, 3200, 4000]
        assert sorted({d['n'] for d in plan}) == expected_ns
        assert len(plan) == len(expected_ns) * self.SEEDS
        for d in plan:
            idx = d['idx']
            assert idx.dtype == np.int64
            assert len(idx) == d['n']
            diffs = np.diff(idx)
            assert (diffs > 0).all()          # strictly increasing => unique
            assert idx[0] >= 0 and idx[-1] < self.POOL
            m = math.ceil(d['n'] / self.BLOCK)
            assert self._runs(idx) <= m
        # deterministic per (n, seed)
        plan2 = self._plan()
        for d1, d2 in zip(plan, plan2):
            assert d1['n'] == d2['n'] and d1['seed'] == d2['seed']
            assert np.array_equal(d1['idx'], d2['idx'])

    def test_seeds_differ_below_pool_and_all_point_is_arange(self):
        plan = self._plan()
        by_n = {}
        for d in plan:
            by_n.setdefault(d['n'], []).append(d)
        for n, draws in by_n.items():
            if n == self.POOL:
                for d in draws:
                    assert np.array_equal(d['idx'],
                                          np.arange(self.POOL, dtype=np.int64))
            else:
                distinct = {d['idx'].tobytes() for d in draws}
                assert len(distinct) > 1, f'all seeds identical at n={n}'

    def test_never_iid(self):
        # An iid 400-of-4000 sample has ~360+ contiguous runs with
        # overwhelming probability; block sampling is bounded at
        # ceil(400/50) = 8 runs — a deterministic bound, no flakiness.
        plan = self._plan()
        for d in plan:
            if d['n'] == 400:
                assert self._runs(d['idx']) <= 8

    def test_base_seed_changes_draws(self):
        kw = dict(n_grid=(400,), n_seeds=1, block_len=50)
        d1 = meta_curve.build_subsample_plan(4000, base_seed=1, **kw)[0]
        d2 = meta_curve.build_subsample_plan(4000, base_seed=2, **kw)[0]
        assert d1['n'] == d2['n'] == 400
        assert not np.array_equal(d1['idx'], d2['idx'])

    def test_fallback_when_blocks_dont_fit(self):
        # pool 120, block_len 100 -> n_segments=1; n=110 needs m=2 blocks
        # -> contiguous-run fallback.
        plan = meta_curve.build_subsample_plan(
            120, n_grid=(110,), n_seeds=3, block_len=100)
        draws = [d for d in plan if d['n'] == 110]
        assert len(draws) == 3
        for d in draws:
            assert len(d['idx']) == 110
            assert self._runs(d['idx']) == 1


# ---------------------------------------------------------------------------
# rank_auc
# ---------------------------------------------------------------------------

class TestRankAuc:
    def test_perfect_and_inverted_and_tied(self):
        y = np.array([0, 0, 1, 1])
        assert meta_curve.rank_auc([0.1, 0.2, 0.8, 0.9], y) == 1.0
        assert meta_curve.rank_auc([0.9, 0.8, 0.2, 0.1], y) == 0.0
        assert meta_curve.rank_auc([0.5, 0.5, 0.5, 0.5], y) == 0.5

    def test_one_class_and_thin(self):
        assert meta_curve.rank_auc([0.1, 0.2], [1, 1]) is None
        assert meta_curve.rank_auc([0.1], [1]) is None
        assert meta_curve.rank_auc([], []) is None
        # all scores non-finite -> nothing usable
        assert meta_curve.rank_auc([np.nan, np.nan, np.nan], [0, 1, 1]) is None

    def test_nonfinite_pairs_dropped(self):
        s = np.array([0.1, np.nan, 0.9, 0.2])
        y = np.array([0, 1, 1, 0])
        assert meta_curve.rank_auc(s, y) == 1.0

    def test_matches_brute_force_oracle(self):
        rng = np.random.default_rng(7)
        s = np.round(rng.random(200), 1)   # heavy ties
        y = (rng.random(200) < 0.4).astype(float)
        pos = s[y == 1]
        neg = s[y == 0]
        wins = 0.0
        for p in pos:
            wins += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
        oracle = wins / (len(pos) * len(neg))
        assert abs(meta_curve.rank_auc(s, y) - oracle) < 1e-12


# ---------------------------------------------------------------------------
# fit_power_law / n_for_target
# ---------------------------------------------------------------------------

class TestPowerLaw:
    def test_recovery(self):
        n = np.array(meta_curve.DEFAULT_N_GRID, float)
        rng = np.random.default_rng(3)
        err = 0.9 * n ** (-0.5) + 0.03 + rng.normal(0, 1e-4, n.size)
        fit = meta_curve.fit_power_law(n, err)
        assert fit['ok'] is True
        assert 0.4 <= fit['b'] <= 0.6
        assert abs(fit['c'] - 0.03) < 0.01
        assert fit['r2'] > 0.99
        assert abs(fit['plateau_err'] - 0.03) < 0.01
        assert fit['n_points'] == n.size

    def test_constant_err_declines(self):
        fit = meta_curve.fit_power_law([100, 400, 1600], [0.2, 0.2, 0.2])
        assert fit['ok'] is False
        assert fit['reason'] is not None

    def test_too_few_points(self):
        fit = meta_curve.fit_power_law([100, 400], [0.3, 0.2])
        assert fit['ok'] is False
        assert 'too_few' in fit['reason']

    def test_nan_and_negative_handled_without_raising(self):
        fit = meta_curve.fit_power_law([100, 200, 400, 800],
                                       [np.nan, -0.1, 0.2, 0.1])
        assert fit['ok'] is False   # only 2 usable points remain

    def test_n_for_target_analytic_roundtrip(self):
        fit = {'ok': True, 'a': 2.0, 'b': 0.5, 'c': 0.05}
        n0 = 800.0
        err0 = 2.0 * n0 ** (-0.5) + 0.05
        back = meta_curve.n_for_target(fit, err0)
        assert abs(back - n0) / n0 < 1e-9

    def test_n_for_target_fitted_roundtrip(self):
        n = np.array(meta_curve.DEFAULT_N_GRID, float)
        err = 1.0 * n ** (-0.5) + 0.05
        fit = meta_curve.fit_power_law(n, err)
        assert fit['ok']
        back = meta_curve.n_for_target(fit, 1.0 * 800.0 ** (-0.5) + 0.05)
        assert back is not None and abs(back - 800.0) / 800.0 < 0.15

    def test_n_for_target_none_cases(self):
        fit = {'ok': True, 'a': 2.0, 'b': 0.5, 'c': 0.05}
        assert meta_curve.n_for_target(fit, 0.05) is None    # at plateau
        assert meta_curve.n_for_target(fit, 0.01) is None    # below plateau
        assert meta_curve.n_for_target({'ok': False}, 0.1) is None


# ---------------------------------------------------------------------------
# veto_flip_rate
# ---------------------------------------------------------------------------

class TestVetoFlipRate:
    # units:      0     1     2     3     4     5
    P = np.array([[0.1, 0.5, 0.1, 0.5, 0.9, 0.2],
                  [0.5, 0.1, 0.1, 0.5, 0.9, 0.2],
                  [0.1, 0.5, 0.1, 0.5, 0.9, 0.2]])

    def test_row_level(self):
        out = meta_curve.veto_flip_rate(self.P)
        assert out['flip_rate'] == pytest.approx(2 / 6)
        assert out['n_units'] == 6
        assert out['n_seeds_used'] == 3
        assert out['level'] == 'row'

    def test_nan_seed_row_ignored(self):
        P4 = np.vstack([self.P, np.full(6, np.nan)])
        out = meta_curve.veto_flip_rate(P4)
        assert out['flip_rate'] == pytest.approx(2 / 6)
        assert out['n_seeds_used'] == 3

    def test_symbol_level(self):
        # groups A={0,2}, B={1,3}, C={4,5}: A's median flips
        # (median(0.1,0.1)=0.1 veto vs median(0.5,0.1)=0.3 no-veto),
        # B stays no-veto (medians 0.5/0.3/0.5), C stays no-veto (0.55).
        groups = np.array(['A', 'B', 'A', 'B', 'C', 'C'])
        out = meta_curve.veto_flip_rate(self.P, groups=groups)
        assert out['flip_rate'] == pytest.approx(1 / 3)
        assert out['n_units'] == 3
        assert out['level'] == 'symbol'

    def test_too_few_usable_seeds(self):
        P = np.vstack([self.P[0], np.full(6, np.nan)])
        out = meta_curve.veto_flip_rate(P)
        assert out['flip_rate'] is None
        assert out['n_units'] == 0
        assert out['n_seeds_used'] == 1


# ---------------------------------------------------------------------------
# empirical_floor
# ---------------------------------------------------------------------------

class TestEmpiricalFloor:
    def test_flip_criterion_binds_later(self):
        grid = [
            {'n': 400,  'auc_mean': 0.650, 'flip_rate': 0.30},
            {'n': 800,  'auc_mean': 0.695, 'flip_rate': 0.15},
            {'n': 1600, 'auc_mean': 0.696, 'flip_rate': 0.08},
            {'n': 3200, 'auc_mean': 0.699, 'flip_rate': 0.05},
        ]
        # AUC criterion passes from 800 (0.70-0.695 < 0.01); flip from 1600.
        assert meta_curve.empirical_floor(grid, 0.70) == 1600

    def test_nothing_qualifies(self):
        grid = [{'n': 400, 'auc_mean': 0.60, 'flip_rate': 0.5},
                {'n': 800, 'auc_mean': 0.62, 'flip_rate': 0.4}]
        assert meta_curve.empirical_floor(grid, 0.70) is None

    def test_none_fields_skipped(self):
        grid = [{'n': 400, 'auc_mean': None, 'flip_rate': 0.01},
                {'n': 800, 'auc_mean': 0.699, 'flip_rate': None},
                {'n': 1600, 'auc_mean': 0.699, 'flip_rate': 0.01}]
        assert meta_curve.empirical_floor(grid, 0.70) == 1600
        assert meta_curve.empirical_floor(grid, None) is None


# ---------------------------------------------------------------------------
# assemble_report end-to-end
# ---------------------------------------------------------------------------

class TestAssembleReport:
    def test_end_to_end(self):
        rng = np.random.default_rng(11)
        ns = list(meta_curve.DEFAULT_N_GRID)
        seeds = 5
        records = []
        flip_by_n = {}
        for n in ns:
            for s in range(seeds):
                auc = 0.70 - 0.5 * n ** (-0.5) + rng.normal(0, 1e-3)
                records.append({
                    'n': n, 'seed': s, 'auc': float(auc),
                    'frac_below_veto': float(0.6 * n ** (-0.1)),
                    'p_q10': 0.2, 'p_median': 0.5, 'p_q90': 0.8,
                    'calib': 'SigmoidCalibrator', 'error': None,
                })
            flip_by_n[n] = {'flip_rate': float(0.8 * n ** (-0.4)),
                            'n_units': 40, 'n_seeds_used': seeds,
                            'level': 'symbol'}
        # one failed draw: must count as n_err, not poison aggregates
        records.append({'n': ns[0], 'seed': seeds, 'auc': None,
                        'frac_below_veto': None, 'p_q10': None,
                        'p_median': None, 'p_q90': None, 'calib': None,
                        'error': 'RuntimeError: boom'})

        report = meta_curve.assemble_report(records, flip_by_n,
                                            meta={'prefix': ''})
        assert report['schema'] == 'meta_curve_report/1'
        assert report['meta'] == {'prefix': ''}
        assert [g['n'] for g in report['grid']] == ns
        g0 = report['grid'][0]
        assert g0['n_ok'] == seeds and g0['n_err'] == 1
        assert report['powerlaw_auc']['ok'] is True
        assert report['powerlaw_flip']['ok'] is True
        assert abs(report['plateau_auc'] - 0.70) < 0.02
        floor = report['floor']
        hf = floor['honest_floor']
        assert isinstance(hf, int) and hf > 0
        assert floor['criteria'] == {
            'auc_tol': meta_curve.AUC_PLATEAU_TOL,
            'flip_tol': meta_curve.FLIP_RATE_TOL,
            'veto_prob': meta_curve.VETO_PROB}
        # honest_floor = max of the finite candidates
        cands = [floor['empirical_n'],
                 None if floor['extrapolated_n_auc'] is None
                 else math.ceil(floor['extrapolated_n_auc']),
                 None if floor['extrapolated_n_flip'] is None
                 else math.ceil(floor['extrapolated_n_flip'])]
        assert hf == max(c for c in cands if c is not None)
        json.dumps(report)   # plain-python coercion holds

    def test_numpy_typed_records_coerced(self):
        # The CLI emits python floats, but _py must keep the report json-safe
        # even when numpy scalars (or NaN) leak into records/flip/meta.
        records = [{'n': np.int64(100), 'seed': np.int64(s),
                    'auc': np.float64(0.6 + 0.01 * s),
                    'frac_below_veto': np.float64(0.4),
                    'p_q10': np.float32(0.2), 'p_median': np.float64('nan'),
                    'p_q90': np.float64(0.8), 'calib': 'raw', 'error': None}
                   for s in range(3)]
        flip = {100: {'flip_rate': np.float64(0.2), 'n_units': np.int64(5),
                      'n_seeds_used': np.int64(3), 'level': 'row'}}
        report = meta_curve.assemble_report(records, flip,
                                            meta={'seeds': np.int64(3)})
        json.dumps(report)   # would raise on any surviving numpy scalar
        g0 = report['grid'][0]
        assert type(g0['n']) is int and g0['n'] == 100
        assert type(g0['auc_mean']) is float
        assert type(g0['flip_rate']) is float
        assert report['records'][0]['p_median'] is None   # NaN -> None
        assert report['meta']['seeds'] == 3

    def test_degenerate_records_still_serializable(self):
        records = [{'n': 100, 'seed': 0, 'auc': None,
                    'frac_below_veto': None, 'p_q10': None, 'p_median': None,
                    'p_q90': None, 'calib': None, 'error': 'X'}]
        report = meta_curve.assemble_report(records, {})
        assert report['powerlaw_auc']['ok'] is False
        assert report['plateau_auc'] is None
        assert report['floor']['honest_floor'] is None
        json.dumps(report)


# ---------------------------------------------------------------------------
# constants pin + CLI lazy-import discipline
# ---------------------------------------------------------------------------

def test_veto_prob_pinned_to_meta_label():
    # meta_label's module top is numpy + stdlib only — Mac-importable.
    import meta_label
    assert meta_curve.VETO_PROB == meta_label.META_VETO_PROB


def test_cli_compiles_and_help_without_lightgbm():
    cli = REPO / 'scripts' / 'meta_learning_curve.py'
    r = subprocess.run([sys.executable, '-m', 'py_compile', str(cli)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # --help must succeed on a machine with NO lightgbm — proves every heavy
    # import lives inside main() after argparse.
    r = subprocess.run([sys.executable, str(cli), '--help'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert '--prefix' in r.stdout and '--grid' in r.stdout
