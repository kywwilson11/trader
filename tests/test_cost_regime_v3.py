"""Pins the adjudicated 2026-07 cost_regime.py panel fixes (module-improve-v3).

VIX-side tz symmetry (VIX-side tz was previously stripped asymmetrically vs
the bar side), the sub-daily/duplicate-date same-day lag leak (collapse to
one obs/day BEFORE the shift(1) lag), numeric/string vix_daily index
handling, VIX_Regime/VIX_Pctile dtype stability (float64 regardless of
calendar / nullable input), the rolling().rank() swap for the old Python-
lambda percentile (with a cross-pandas-version equivalence guard), the new
leading-edge coverage warnings, the FRED CSV parse try/except contract, the
pd.NA-safe vix_regime_code, and fetch_fred_vixcls's per-process memo +
bounded read.

Pure numpy/pandas/urllib/pytest — no heavy deps, collects cleanly on the dev
Mac, the Jetson, and CI.
"""

import io
import sys
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cost_regime as cr


@pytest.fixture
def vix40():
    return pd.Series(np.linspace(10, 40, 40),
                     index=pd.date_range('2024-01-01', periods=40, freq='D'))


@pytest.fixture
def bars3():
    return pd.date_range('2024-01-10 14:00', periods=3, freq='h', tz='UTC')


# --------------------------------------------------------------------------
# VIX-side tz symmetry
# --------------------------------------------------------------------------

class TestVixTzSymmetry:
    def test_tz_aware_vix_matches_naive(self, vix40, bars3):
        with redirect_stdout(io.StringIO()):
            base = cr.vix_features_for_index(vix40, bars3, pct_window=30)
            for tz in ('UTC', 'America/New_York'):
                out = cr.vix_features_for_index(vix40.tz_localize(tz), bars3,
                                                 pct_window=30)
                for k in base:
                    assert np.array_equal(base[k], out[k], equal_nan=True)

    def test_tz_aware_vix_naive_bars(self, vix40, bars3):
        with redirect_stdout(io.StringIO()):
            base = cr.vix_features_for_index(vix40, bars3, pct_window=30)
            out = cr.vix_features_for_index(vix40.tz_localize('UTC'),
                                             bars3.tz_localize(None),
                                             pct_window=30)
            for k in base:
                assert np.array_equal(base[k], out[k], equal_nan=True)

    def test_daily_pit_pin_unchanged(self, vix40, bars3):
        with redirect_stdout(io.StringIO()):
            out = cr.vix_features_for_index(vix40, bars3, pct_window=30)
        assert out['VIX_Level'][0] == pytest.approx(vix40.loc['2024-01-09'])


# --------------------------------------------------------------------------
# One-observation-per-day collapse BEFORE the shift(1) lag
# --------------------------------------------------------------------------

class TestVixOnePerDayCollapse:
    def test_duplicate_date_is_not_a_same_day_read(self):
        v = pd.Series([10., 20., 99., 21., 22., 23.],
                      index=pd.to_datetime(['2024-01-01', '2024-01-02',
                                            '2024-01-02', '2024-01-03',
                                            '2024-01-04', '2024-01-05']))
        bar = pd.DatetimeIndex([pd.Timestamp('2024-01-02 12:00', tz='UTC')])
        with redirect_stdout(io.StringIO()):
            out = cr.vix_features_for_index(v, bar, pct_window=5)
        # the 01-01 close, NOT the same-day 20.0 (or the later dup 99.0)
        assert out['VIX_Level'][0] == pytest.approx(10.0)

    def test_intraday_input_lags_a_full_day(self):
        rows = []
        for i in range(1, 7):
            day = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i - 1)
            rows.append((day + pd.Timedelta(hours=9), 10 + i))
            rows.append((day + pd.Timedelta(hours=16), 50 + i))
        idx, vals = zip(*rows)
        v = pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=float)
        bar = pd.DatetimeIndex([pd.Timestamp('2024-01-06 00:30', tz='UTC')])
        with redirect_stdout(io.StringIO()):
            out = cr.vix_features_for_index(v, bar, pct_window=5)
        # 01-05's LAST obs (55.0), NOT the old leaked same-day 16.0
        assert out['VIX_Level'][0] == pytest.approx(55.0)


# --------------------------------------------------------------------------
# vix_daily index validation
# --------------------------------------------------------------------------

class TestVixInputValidation:
    def test_numeric_index_returns_none_and_warns(self, bars3, capsys):
        assert cr.vix_features_for_index(
            list(np.linspace(10, 40, 40)), bars3, pct_window=30) is None
        assert '[COST-REGIME]' in capsys.readouterr().out
        assert cr.vix_features_for_index(
            np.linspace(10, 40, 40), bars3, pct_window=30) is None
        assert '[COST-REGIME]' in capsys.readouterr().out

    def test_string_index_is_coerced(self):
        sv = pd.Series([10., 11., 12., 13., 14., 15.],
                       index=['2024-01-05', '2024-1-8', '2024-01-09',
                             '2024-01-10', '2024-01-11', '2024-01-12'])
        bar = pd.DatetimeIndex([pd.Timestamp('2024-01-10 12:00', tz='UTC')])
        with redirect_stdout(io.StringIO()):
            out = cr.vix_features_for_index(sv, bar, pct_window=5)
        assert out['VIX_Level'][0] == pytest.approx(12.0)

    def test_too_short_after_collapse_returns_none(self):
        # 6 intraday observations spanning only 3 calendar days
        idx = pd.to_datetime(['2024-01-01 09:00', '2024-01-01 16:00',
                              '2024-01-02 09:00', '2024-01-02 16:00',
                              '2024-01-03 09:00', '2024-01-03 16:00'])
        v = pd.Series([10., 11., 12., 13., 14., 15.], index=idx)
        bar = pd.DatetimeIndex([pd.Timestamp('2024-01-04 10:00', tz='UTC')])
        with redirect_stdout(io.StringIO()):
            out = cr.vix_features_for_index(v, bar, pct_window=5)
        assert out is None


# --------------------------------------------------------------------------
# dtype stability
# --------------------------------------------------------------------------

class TestVixDtypeStability:
    def test_regime_is_float64_on_both_calendars(self):
        vv = pd.Series(np.full(40, 30.0),
                       index=pd.date_range('2024-03-01', periods=40,
                                           freq='D'))
        inside_bars = pd.date_range('2024-03-20', periods=3, freq='h',
                                    tz='UTC')
        straddle_bars = pd.date_range('2024-02-25', periods=3, freq='D',
                                      tz='UTC')
        with redirect_stdout(io.StringIO()):
            inside = cr.vix_features_for_index(vv, inside_bars,
                                               pct_window=10)
            straddle = cr.vix_features_for_index(vv, straddle_bars,
                                                 pct_window=10)
        for out in (inside, straddle):
            for k in ('VIX_Level', 'VIX_Regime', 'VIX_Pctile'):
                assert isinstance(out[k], np.ndarray)
                assert out[k].dtype == np.float64
        assert inside['VIX_Regime'][0] == pytest.approx(2.0)

    def test_nullable_float64_input_yields_plain_ndarrays(self):
        days = pd.date_range('2024-01-01', periods=60, freq='D')
        vv = pd.Series(np.linspace(10, 40, 60), index=days).astype('Float64')
        bars = pd.date_range('2024-02-25 14:00', periods=2, freq='h',
                             tz='UTC')
        with redirect_stdout(io.StringIO()):
            out = cr.vix_features_for_index(vv, bars, pct_window=30)
        for k in ('VIX_Level', 'VIX_Regime', 'VIX_Pctile'):
            assert isinstance(out[k], np.ndarray)
            assert out[k].dtype == np.float64
        assert np.isfinite(out['VIX_Level']).all()


# --------------------------------------------------------------------------
# rolling().rank() vs the old Python-lambda percentile
# --------------------------------------------------------------------------

class TestVixRankEquivalence:
    def test_rank_max_matches_old_lambda_tie_heavy(self):
        s = pd.Series(
            np.round(np.random.default_rng(7).normal(20, 5, 1500), 2),
            index=pd.bdate_range('2018-01-01', periods=1500))
        for w in (5, 20, 252):
            mp = min(w, max(2, min(20, w)))
            ref = s.rolling(w, min_periods=mp).apply(
                lambda x: (x[-1] >= x).mean(), raw=True)
            new = s.rolling(w, min_periods=mp).rank(method='max', pct=True)
            assert np.array_equal(ref.values, new.values, equal_nan=True)


# --------------------------------------------------------------------------
# coverage warnings (leading-edge + short-history pctile)
# --------------------------------------------------------------------------

class TestVixCoverageWarnings:
    def test_pre_history_bars_warn_and_stay_nan(self, vix40, capsys):
        bars = pd.date_range('2019-05-01', periods=5, freq='h', tz='UTC')
        out = cr.vix_features_for_index(vix40, bars, pct_window=30)
        captured = capsys.readouterr().out
        for k in ('VIX_Level', 'VIX_Regime', 'VIX_Pctile'):
            assert np.isnan(out[k]).all()
        assert '[COST-REGIME]' in captured

    def test_short_history_pctile_all_nan_warns(self, capsys):
        days = pd.date_range('2024-01-01', periods=10, freq='D')
        vix = pd.Series(np.linspace(10, 19, 10), index=days)
        bars = pd.date_range('2024-01-08 10:00', periods=2, freq='h',
                             tz='UTC')
        out = cr.vix_features_for_index(vix, bars, pct_window=252)
        captured = capsys.readouterr().out
        assert np.isnan(out['VIX_Pctile']).all()
        assert np.isfinite(out['VIX_Level']).all()
        assert 'VIX_Pctile' in captured

    def test_full_coverage_stays_silent(self, vix40, capsys):
        bars = pd.date_range('2024-02-05 14:00', periods=2, freq='h',
                             tz='UTC')
        cr.vix_features_for_index(vix40, bars, pct_window=30)
        assert '[COST-REGIME]' not in capsys.readouterr().out


# --------------------------------------------------------------------------
# FRED CSV parse contract: Series or None, never raises
# --------------------------------------------------------------------------

class TestFredParseContract:
    def test_empty_string_returns_none(self):
        assert cr.parse_fred_vixcls('') is None

    def test_newline_only_returns_none(self):
        assert cr.parse_fred_vixcls('\n') is None

    def test_ragged_body_returns_none(self):
        csv = "DATE,VIXCLS\n2024-01-02,13.2\nbroken,row,extra\n"
        assert cr.parse_fred_vixcls(csv) is None

    def test_valid_csv_still_parses(self):
        csv = "DATE,VIXCLS\n2024-01-02,13.2\n2024-01-03,14.0\n"
        s = cr.parse_fred_vixcls(csv)
        assert s.iloc[0] == pytest.approx(13.2)


# --------------------------------------------------------------------------
# vix_regime_code scalar contract
# --------------------------------------------------------------------------

class TestVixRegimeCodeScalar:
    def test_buckets_incl_boundaries_and_missing(self):
        assert cr.vix_regime_code(12.0) == 0
        assert cr.vix_regime_code(15.0) == 1
        assert cr.vix_regime_code(20.0) == 1
        assert cr.vix_regime_code(25.0) == 2
        assert cr.vix_regime_code(30.0) == 2
        for missing in (None, float('nan'), np.float64('nan'), pd.NA):
            assert cr.vix_regime_code(missing) == 1


# --------------------------------------------------------------------------
# Amihud guards
# --------------------------------------------------------------------------

def _amihud_clean_series(n=30):
    idx = pd.date_range('2025-01-01', periods=n, freq='h')
    c = pd.Series(100 + np.arange(n) * 0.5, index=idx)
    v = pd.Series(np.full(n, 1e6), index=idx)
    return c, v


class TestAmihudGuards:
    def test_formula_identity_on_clean_data(self):
        c, v = _amihud_clean_series()
        ref = ((c.pct_change().abs() / (c * v)) * 1e6).rolling(
            10, min_periods=5).mean()
        out = cr.amihud_illiq(c, v, window=10)
        assert out.equals(ref)

    def test_warmup_is_half_window(self):
        c, v = _amihud_clean_series()
        out = cr.amihud_illiq(c, v, window=21)
        assert out.iloc[:10].isna().all()
        assert out.notna().iloc[10]

    def test_window_one_no_longer_crashes(self):
        idx = pd.date_range('2025-01-01', periods=3, freq='h')
        c = pd.Series([100.0, 100.5, 101.0], index=idx)
        v = pd.Series([1e6, 1e6, 1e6], index=idx)
        out = cr.amihud_illiq(c, v, window=1)
        assert np.isfinite(out.to_numpy()[-1])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cr.amihud_illiq(np.linspace(100, 130, 30), np.full(25, 1e6),
                            window=5)

    def test_negative_volume_never_negative_illiq(self):
        close = pd.Series([100., 100.01, 102., 103., 104., 105.])
        vol = pd.Series([1e6, 1e6, -1e2, 1e6, 1e6, 1e6])
        out = cr.amihud_illiq(close, vol, window=3)
        assert (out.dropna() >= 0).all()

    def test_zero_volume_still_nan_not_inf(self):
        close = pd.Series([100.0, 101.0, 102.0, 103.0])
        out = cr.amihud_illiq(close, pd.Series([0, 0, 0, 0]), window=2)
        assert not np.isinf(out.to_numpy()).any()

    def test_misaligned_volume_warns(self, capsys):
        c, v = _amihud_clean_series()
        v2 = v.copy()
        v2.index = v2.index + pd.Timedelta('3h')
        cr.amihud_illiq(c, v2, window=10)
        assert 'POSITIONALLY' in capsys.readouterr().out

    def test_panel_non_unique_index_warns(self, capsys):
        c, v = _amihud_clean_series(n=4)
        df1 = pd.DataFrame({'Close': c, 'Volume': v})
        df2 = pd.DataFrame({'Close': c + 1, 'Volume': v})
        panel = pd.concat([df1, df2]).sort_index()
        cr.amihud_illiq(panel['Close'], panel['Volume'], window=2)
        assert 'non-unique' in capsys.readouterr().out

    def test_aligned_single_name_stays_silent(self, capsys):
        c, v = _amihud_clean_series()
        cr.amihud_illiq(c, v, window=10)
        assert capsys.readouterr().out == ''


# --------------------------------------------------------------------------
# fetch_fred_vixcls: memo + context manager
# --------------------------------------------------------------------------

class _FakeResp:
    """Fake urlopen() response: context-managed, read(n) returns the body."""

    def __init__(self, body, log):
        self._body = body
        self._log = log

    def read(self, n=-1):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._log['exited'] = True
        return False


class TestFredFetch:
    def test_success_memo_and_context_manager(self, monkeypatch):
        body = b"DATE,VIXCLS\n2024-01-02,13.2\n2024-01-03,14.0\n"
        log = {}
        calls = {'n': 0}

        def fake_urlopen(req, timeout=30):
            calls['n'] += 1
            return _FakeResp(body, log)

        monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
        monkeypatch.setattr(cr, '_VIXCLS_MEMO', None)

        s1 = cr.fetch_fred_vixcls()
        s2 = cr.fetch_fred_vixcls()

        assert len(s1) == 2
        assert s1.iloc[0] == pytest.approx(13.2)
        assert s2.equals(s1)
        assert calls['n'] == 1
        assert log.get('exited') is True
        assert s1 is not s2

    def test_failure_returns_none_and_prints(self, monkeypatch, capsys):
        def boom(req, timeout=30):
            raise OSError('network is down')

        monkeypatch.setattr(urllib.request, 'urlopen', boom)
        monkeypatch.setattr(cr, '_VIXCLS_MEMO', None)

        assert cr.fetch_fred_vixcls() is None
        assert '[COST-REGIME]' in capsys.readouterr().out

    def test_degenerate_body_returns_none_not_memoized(self, monkeypatch):
        body = b"DATE,VIXCLS\n"
        log = {}

        def fake_urlopen(req, timeout=30):
            return _FakeResp(body, log)

        monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
        monkeypatch.setattr(cr, '_VIXCLS_MEMO', None)

        assert cr.fetch_fred_vixcls() is None
        assert cr._VIXCLS_MEMO is None
