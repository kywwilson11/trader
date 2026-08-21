"""Panel-review v3 hardening tests for liquidity.py (2026-07).

Pins: input validation, column handling, degenerate-input behavior (documented
as-is where it is an open owner decision), estimator-health instrumentation,
market_impact_pct fail-open parity, per_bar out-of-band warnings, impact-term
independent fail-open, and the bidask dependency declaration.
Mac-green; tests that need the real bidask estimator importorskip it inline.
"""
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import liquidity
from fees import round_trip_cost_pct, FLAT_SPREAD_PCT


def _ohlc(n=120, spread_frac=0.004, seed=0):
    rng = np.random.RandomState(seed)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    half = spread_frac / 2.0
    close = mid * (1 + rng.choice([-half, half], n))
    high = np.maximum(mid, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(mid, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    op = mid * (1 + rng.normal(0, 0.002, n))
    idx = pd.date_range('2025-01-01', periods=n, freq='h')
    return pd.DataFrame({'Open': op, 'High': high, 'Low': low,
                         'Close': close}, index=idx)


class TestValidation:
    def test_bad_window_raises(self):
        for w in (0, 1, 2, -5, 2.5, '2D'):
            with pytest.raises(ValueError):
                liquidity.edge_spread_series(_ohlc(60), window=w)

    def test_bad_floor_cap_raises(self):
        with pytest.raises(ValueError):
            liquidity.edge_spread_series(_ohlc(120), floor_pct=1.0, cap_pct=0.1)
        with pytest.raises(ValueError):
            liquidity.edge_spread_series(_ohlc(120), floor_pct=float('nan'))
        with pytest.raises(ValueError):
            liquidity.edge_spread_series(_ohlc(120), cap_pct=float('inf'))

    def test_valid_path_values_match_raw_estimator(self):
        bidask = pytest.importorskip('bidask')
        df = _ohlc(120)
        got = liquidity.edge_spread_series(df, window=35)
        raw = bidask.edge_rolling(df[['Open', 'High', 'Low', 'Close']],
                                  window=35)
        exp = pd.Series(np.asarray(raw, dtype=float) * 100.0, index=df.index)
        exp = exp.replace([np.inf, -np.inf], np.nan)
        exp = exp.clip(lower=liquidity.SPREAD_FLOOR_PCT,
                       upper=liquidity.SPREAD_CAP_PCT)
        exp = exp.fillna(liquidity.SPREAD_FLOOR_PCT)
        np.testing.assert_array_equal(got.values, exp.values)


class TestColumnHandling:
    def test_case_variant_duplicate_raises_keyerror(self):
        df = _ohlc(60)
        df['close'] = df['Close']
        with pytest.raises(KeyError):
            liquidity.edge_spread_series(df, window=20)

    def test_literal_duplicate_raises_keyerror(self):
        df = _ohlc(60)
        dup = pd.concat([df, df[['Close']]], axis=1)
        with pytest.raises(KeyError):
            liquidity.edge_spread_series(dup, window=20)

    def test_multiindex_columns_raise_keyerror(self):
        df = _ohlc(60)
        df.columns = pd.MultiIndex.from_product(
            [['Open', 'High', 'Low', 'Close'], ['AAPL']])
        with pytest.raises(KeyError):
            liquidity.edge_spread_series(df, window=20)

    def test_junk_and_wide_frames_equal_subset(self):
        df = _ohlc(60)
        wide = df.copy()
        wide[123] = 1.0                      # non-string junk label tolerated
        rng = np.random.RandomState(9)
        for i in range(50):
            wide[f'F{i}'] = rng.normal(size=len(wide))
        a = liquidity.edge_spread_series(wide, window=20)
        b = liquidity.edge_spread_series(df, window=20)
        np.testing.assert_array_equal(a.values, b.values)


class TestShortAndDegenerateFrames:
    def test_short_frames_all_floor_and_warn(self, caplog):
        for n in (5, 17, 25, 34):
            with caplog.at_level(logging.WARNING, logger='liquidity'):
                s = liquidity.edge_spread_series(_ohlc(n=n), window=35)
            assert (s == liquidity.SPREAD_FLOOR_PCT).all()
        msgs = [r.getMessage() for r in caplog.records if r.name == 'liquidity']
        assert any('floor' in m for m in msgs)

    def test_at_window_not_all_floor(self):
        pytest.importorskip('bidask')
        s = liquidity.edge_spread_series(_ohlc(n=35), window=35)
        assert not (s == liquidity.SPREAD_FLOOR_PCT).all()

    def test_constant_frame_all_floor_and_warns(self, caplog):
        # Documents TODAY's behavior: a halted/constant name is stamped at the
        # 2 bps floor (CHEAPER than the flat fallback) — open owner decision.
        pytest.importorskip('bidask')
        idx = pd.date_range('2025-01-01', periods=120, freq='h')
        const = pd.DataFrame({'Open': 50.0, 'High': 50.0, 'Low': 50.0,
                              'Close': 50.0}, index=idx)
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            s = liquidity.edge_spread_series(const)
        assert s.notna().all() and (s == liquidity.SPREAD_FLOOR_PCT).all()
        msgs = [r.getMessage() for r in caplog.records if r.name == 'liquidity']
        assert any('fabricated' in m for m in msgs)

    def test_healthy_frame_stays_warning_silent(self, caplog):
        # Mirrors test_review_b10::test_bidask_success_stays_silent.
        pytest.importorskip('bidask')
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            liquidity.edge_spread_series(_ohlc(n=60), window=20)
        assert not [r for r in caplog.records if r.name == 'liquidity']

    def test_healthy_frame_info_summary(self, caplog):
        pytest.importorskip('bidask')
        with caplog.at_level(logging.INFO, logger='liquidity'):
            liquidity.edge_spread_series(_ohlc(n=60), window=20)
        msgs = [r.getMessage() for r in caplog.records if r.name == 'liquidity']
        assert any('median' in m and 'raw-NaN' in m for m in msgs)

    def test_nan_block_and_zero_low_never_nan(self):
        pytest.importorskip('bidask')
        df = _ohlc(n=200, seed=11)
        df.iloc[50:60, :] = np.nan
        s = liquidity.edge_spread_series(df)
        assert s.notna().all()
        z = _ohlc(n=200, seed=11)
        z.iloc[100, z.columns.get_loc('Low')] = 0.0
        sz = liquidity.edge_spread_series(z)
        assert sz.notna().all()
        # one bad print floors at least a full trailing window of bars
        assert int((sz == liquidity.SPREAD_FLOOR_PCT).sum()) >= 35


class TestMonotonicIndexWarning:
    def test_shuffled_index_warns(self, caplog):
        pytest.importorskip('bidask')
        df = _ohlc(n=200).sample(frac=1.0, random_state=2)
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            s = liquidity.edge_spread_series(df)
        assert len(s) == 200 and s.notna().all()
        msgs = [r.getMessage() for r in caplog.records if r.name == 'liquidity']
        assert any('monoton' in m for m in msgs)

    def test_sorted_index_silent(self, caplog):
        pytest.importorskip('bidask')
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            liquidity.edge_spread_series(_ohlc(n=200))
        assert not [r for r in caplog.records if r.name == 'liquidity']


class TestMarketImpactGuards:
    def test_none_inputs_return_zero(self):
        assert liquidity.market_impact_pct(None, 1e6, 0.1) == 0.0
        assert liquidity.market_impact_pct(1e4, None, 0.1) == 0.0
        assert liquidity.market_impact_pct(1e4, 1e6, None) == 0.0

    def test_bad_k_sides_cap_return_zero(self):
        assert liquidity.market_impact_pct(1e4, 1e6, 0.1, k=np.nan) == 0.0
        assert liquidity.market_impact_pct(1e4, 1e6, 0.1, k=np.inf) == 0.0
        assert liquidity.market_impact_pct(1e4, 1e6, 0.1, k=-2.0) == 0.0
        assert liquidity.market_impact_pct(1e4, 1e6, 0.1, sides=np.nan) == 0.0
        assert liquidity.market_impact_pct(1e4, 1e6, 0.1, sides=-2) == 0.0
        assert liquidity.market_impact_pct(1e5, 1e5, 1.0,
                                           cap_pct=np.nan) == 0.0

    def test_finite_path_unchanged(self):
        import math as _m
        got = liquidity.market_impact_pct(25_000, 5e6, 0.1, k=1.0, sides=2)
        exp = min(1.0 * 0.1 * _m.sqrt(25_000 / 5e6),
                  liquidity.IMPACT_CAP_PCT) * 2
        assert got == pytest.approx(exp, rel=1e-12)

    def test_scalar_vector_parity_on_degenerate_k(self):
        sp = np.array([0.05, 0.10])
        adv = np.array([5e6, 1e6])
        base = liquidity.per_bar_round_trip_cost('stock', sp)
        for k in (np.nan, np.inf, -2.0, 0.0, 1.3):
            got = liquidity.per_bar_round_trip_cost(
                'stock', sp, adv_dollar=adv, notional=25_000.0, impact_k=k)
            for i in range(2):
                exp = liquidity.market_impact_pct(25_000.0, adv[i], sp[i],
                                                  k=k, sides=2)
                assert got[i] - base[i] == pytest.approx(exp, abs=1e-12)


class TestPerBarWarnings:
    def test_above_cap_warns_values_unchanged(self, caplog):
        fee_const = round_trip_cost_pct('stock', 0.0)
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            cost = liquidity.per_bar_round_trip_cost(
                'stock', np.array([0.05, 99999.0]))
        np.testing.assert_allclose(cost,
                                   [fee_const + 0.05, fee_const + 99999.0])
        msgs = [r.getMessage() for r in caplog.records if r.name == 'liquidity']
        assert any('SPREAD_CAP_PCT' in m for m in msgs)

    def test_zero_spread_warns_value_unchanged(self, caplog):
        # Documents TODAY's behavior: a literal 0.0 is accepted as valid
        # (charges fee only) — treating it as missing is an owner decision.
        fee_const = round_trip_cost_pct('stock', 0.0)
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            cost = liquidity.per_bar_round_trip_cost('stock', np.array([0.0]))
        assert cost[0] == pytest.approx(fee_const)
        msgs = [r.getMessage() for r in caplog.records if r.name == 'liquidity']
        assert any('0.0' in m for m in msgs)

    def test_inf_mixed_with_above_cap(self, caplog):
        # +inf is corrupt -> flat-substituted (designed, silent path); only
        # the FINITE above-cap bar is counted, and the reported max must be
        # that bar, not the inf.
        fee_const = round_trip_cost_pct('stock', 0.0)
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            cost = liquidity.per_bar_round_trip_cost(
                'stock', np.array([2.0, np.inf]))
        np.testing.assert_allclose(
            cost, [fee_const + 2.0, fee_const + FLAT_SPREAD_PCT['stock']])
        msgs = [r.getMessage() for r in caplog.records if r.name == 'liquidity']
        assert any('1 bars above' in m and '2.000' in m and 'inf' not in m
                   for m in msgs)

    def test_in_band_and_designed_fallbacks_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            liquidity.per_bar_round_trip_cost('stock', np.array([0.05, 1.2]))
            liquidity.per_bar_round_trip_cost('stock',
                                              np.array([np.nan, -1.0]))
        assert not [r for r in caplog.records if r.name == 'liquidity']


class TestImpactRobustness:
    def test_object_adv_retains_spread_only(self, caplog):
        sp = np.array([0.05, 0.10])
        base = liquidity.per_bar_round_trip_cost('stock', sp)
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            got = liquidity.per_bar_round_trip_cost(
                'stock', sp, adv_dollar=np.array(['n/a', '1e6'], dtype=object),
                notional=25_000.0)
        np.testing.assert_array_equal(got, base)
        msgs = [r.getMessage() for r in caplog.records if r.name == 'liquidity']
        assert any('impact term skipped' in m for m in msgs)

    def test_shape_mismatch_retains_spread_only(self, caplog):
        sp = np.array([0.05, 0.10])
        base = liquidity.per_bar_round_trip_cost('stock', sp)
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            got = liquidity.per_bar_round_trip_cost(
                'stock', sp, adv_dollar=np.array([5e6, 5e6, 5e6]),
                notional=25_000.0)
        np.testing.assert_array_equal(got, base)
        msgs = [r.getMessage() for r in caplog.records if r.name == 'liquidity']
        assert any('impact term skipped' in m for m in msgs)

    def test_zero_adv_no_runtime_warning(self):
        import math as _m
        sp = np.array([0.05] * 3)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            got = liquidity.per_bar_round_trip_cost(
                'stock', sp, adv_dollar=np.array([5e6, 0.0, np.nan]),
                notional=25_000.0)
        base = round_trip_cost_pct('stock', 0.05)
        imp0 = min(1.0 * 0.05 * _m.sqrt(25_000 / 5e6),
                   liquidity.IMPACT_CAP_PCT)
        np.testing.assert_allclose(got, [base + 2 * imp0, base, base])


class TestSourceAndDeclarationPins:
    def test_floor_comment_and_docstring_truth(self):
        src = (REPO / 'liquidity.py').read_text()
        assert 'can go slightly negative' not in src
        assert 'sign=False' in src
        assert 'SAME-bar squared' in src                       # test_grp_risk pin
        assert 'form, simplified single-window var' not in src  # test_grp_risk pin
        assert 'harvest_stock_data.py' in liquidity.__doc__     # stock-only scope
        assert 'crypto' in liquidity.__doc__
        doc = liquidity._abdi_ranaldo_rolling.__doc__
        assert 'UPWARD' in doc and 'UPPER BOUND' in doc         # b10 pin

    def test_bidask_declared_everywhere(self):
        for rel in ('requirements.txt', 'requirements-ci.txt',
                    'requirements-jetson.txt', '.github/workflows/ci.yml'):
            assert 'bidask' in (REPO / rel).read_text(), rel
