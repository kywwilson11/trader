"""Wave-6 Tier-2: cost-regime META features (cost_regime).

Amihud ILLIQ math, FRED CSV parsing (legacy + current headers, '.' missing),
the fixed VIX regime buckets, and the PIT VIX alignment (1-day lag, trailing
percentile, no look-ahead)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cost_regime as cr


class TestAmihud:
    def test_illiquid_name_has_higher_illiq(self):
        # same |returns|, but lower dollar-volume -> higher Amihud
        close = pd.Series(100 * np.cumprod(1 + np.array([0.0] + [0.02, -0.02] * 15)))
        liquid = cr.amihud_illiq(close, pd.Series([1e8] * len(close)), window=10)
        illiquid = cr.amihud_illiq(close, pd.Series([1e5] * len(close)), window=10)
        assert illiquid.iloc[-1] > liquid.iloc[-1]

    def test_zero_volume_is_nan_not_inf(self):
        close = pd.Series([100.0, 101.0, 102.0, 103.0])
        out = cr.amihud_illiq(close, pd.Series([0, 0, 0, 0]), window=2)
        assert not np.isinf(out.to_numpy()).any()

    def test_trailing_window_is_pit(self):
        close = pd.Series(100 + np.arange(30) * 0.5)
        vol = pd.Series([1e6] * 30)
        out = cr.amihud_illiq(close, vol, window=10)
        # first rows (insufficient history) are NaN -> no look-ahead leakage
        assert out.iloc[:4].isna().all()


class TestFredParse:
    def test_legacy_header_with_missing(self):
        csv = "DATE,VIXCLS\n2024-01-02,13.2\n2024-01-03,.\n2024-01-04,14.0\n"
        s = cr.parse_fred_vixcls(csv)
        assert len(s) == 2  # the '.' row dropped
        assert s.iloc[0] == pytest.approx(13.2)

    def test_current_observation_date_header(self):
        csv = "observation_date,VIXCLS\n2024-06-01,18.5\n2024-06-02,19.1\n"
        s = cr.parse_fred_vixcls(csv)
        assert len(s) == 2 and s.iloc[-1] == pytest.approx(19.1)


class TestVixRegime:
    def test_buckets(self):
        assert cr.vix_regime_code(12.0) == 0
        assert cr.vix_regime_code(20.0) == 1
        assert cr.vix_regime_code(30.0) == 2
        assert cr.vix_regime_code(float('nan')) == 1


class TestVixFeaturesPIT:
    def test_lagged_no_lookahead(self):
        # daily VIX; a bar on day D must see day D-1's close, not D's.
        days = pd.date_range('2024-01-01', periods=40, freq='D')
        vix = pd.Series(np.linspace(10, 40, 40), index=days)
        # hourly bars on 2024-01-10
        idx = pd.date_range('2024-01-10 14:00', periods=3, freq='h', tz='UTC')
        out = cr.vix_features_for_index(vix, idx, pct_window=30)
        assert out is not None
        # value seen on the 10th == the 9th's VIX (shift 1)
        expected = vix.loc['2024-01-09']
        assert out['VIX_Level'][0] == pytest.approx(expected)

    def test_regime_and_pctile_present(self):
        days = pd.date_range('2024-01-01', periods=60, freq='D')
        vix = pd.Series(np.concatenate([np.full(50, 12.0), np.full(10, 30.0)]),
                        index=days)
        idx = pd.date_range('2024-02-25', periods=2, freq='h', tz='UTC')
        out = cr.vix_features_for_index(vix, idx, pct_window=40)
        assert out['VIX_Regime'][0] == 2          # in the stress stretch
        assert 0.0 <= out['VIX_Pctile'][0] <= 1.0

    def test_too_short_returns_none(self):
        vix = pd.Series([12.0, 13.0], index=pd.date_range('2024-01-01', periods=2))
        idx = pd.date_range('2024-01-05', periods=1, freq='h', tz='UTC')
        assert cr.vix_features_for_index(vix, idx) is None
