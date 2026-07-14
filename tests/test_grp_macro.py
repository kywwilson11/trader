"""Group-macro doc/comment hygiene regression tests.

Covers three fixes:
  1. macro_indicators.fetch_cape docstring said "1.5 adjustment factor" while
     the code (and module header) has always used 1.6 -> docstring corrected,
     numeric behavior unchanged.
  2. volatility.py's EGARCH comment claimed "captures asymmetry: crashes
     increase vol more than rallies", but arch_model(..., vol='EGARCH', ...)
     is called with no o= kwarg (arch default o=0 -> symmetric) -> comment
     corrected to state the model is symmetric here, code unchanged.
  3. volatility.get_sigma silently fell through to GARCH/cached-sigma whenever
     HAR-RV had fewer than _HAR_MIN_DAYS (60) daily observations (the
     production-live case per the module review). A once-per-symbol INFO log
     now surfaces the gap; the fall-through return value is unchanged.

Mac-runnable: no network, no arch/torch/lightgbm/numba/sklearn; only
numpy/pandas/scipy-tier deps, mirroring tests/test_review_b07.py and
tests/test_wave4.py conventions.
"""

import logging
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import macro_indicators as mi
import volatility as vol

REPO = Path(__file__).resolve().parent.parent


def _src(name):
    return (REPO / name).read_text()


def _logs(caplog, needle, level=logging.INFO):
    return [r for r in caplog.records
            if r.levelno >= level and needle in r.getMessage()]


def _ohlc(n_days, day_vol, seed=0, bars_per_day=7):
    """Synthetic intraday OHLC frame (business days only), mirroring the
    helper in tests/test_wave4.py so HAR-RV has realistic Parkinson ranges."""
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    px = 100.0
    for d in range(n_days):
        day = pd.Timestamp('2025-01-02') + pd.Timedelta(days=d)
        if day.weekday() >= 5:
            continue
        for h in range(bars_per_day):
            r = rng.normal(0, day_vol / np.sqrt(bars_per_day))
            o = px
            px = px * (1 + r)
            hi = max(o, px) * (1 + abs(rng.normal(0, day_vol / 4)))
            lo = min(o, px) * (1 - abs(rng.normal(0, day_vol / 4)))
            idx.append(pd.Timestamp(f'{day.date()} {13 + h}:30', tz='UTC'))
            rows.append((o, hi, lo, px))
    df = pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close'],
                      index=pd.DatetimeIndex(idx))
    df['Volume'] = 1e6
    return df


class _Ticker:
    def __init__(self, info):
        self.info = info


def _yfinance_with(info=None):
    mod = types.ModuleType('yfinance')
    mod.Ticker = lambda sym: _Ticker(info or {})
    return mod


@pytest.fixture(autouse=True)
def _clean_module_state():
    """Isolate module-level caches between tests."""
    mi._cache.clear()
    vol._har_cache.clear()
    vol._har_gap_logged.clear()
    yield
    mi._cache.clear()
    vol._har_cache.clear()
    vol._har_gap_logged.clear()


# ---------------------------------------------------------------------------
# EDIT 1: macro_indicators.fetch_cape docstring 1.5 -> 1.6
# ---------------------------------------------------------------------------

class TestFetchCapeDocFix:
    def test_docstring_says_1_6_not_1_5(self):
        doc = mi.fetch_cape.__doc__ or ''
        assert '1.6' in doc
        assert '1.5' not in doc

    def test_source_has_no_stale_1_5_docstring_text(self):
        src = _src('macro_indicators.py')
        assert '1.5 adjustment factor' not in src

    def test_fetch_cape_numeric_behavior_unchanged(self, monkeypatch):
        # Locks the 1.6 multiplier the corrected docstring must match
        # (docstring-only edit; behavior already covered by
        # tests/test_review_b07.py::test_fetch_cape_happy_path).
        monkeypatch.setitem(sys.modules, 'yfinance',
                             _yfinance_with(info={'trailingPE': 25.0}))
        assert mi.fetch_cape() == pytest.approx(40.0)  # 25 * 1.6


# ---------------------------------------------------------------------------
# EDIT 2: volatility.py EGARCH comment corrected (symmetric, o=0)
# ---------------------------------------------------------------------------

class TestEGARCHCommentFix:
    def test_comment_no_longer_claims_asymmetry(self):
        src = _src('volatility.py')
        assert 'captures asymmetry' not in src
        assert 'crashes increase vol more than rallies' not in src

    def test_comment_now_documents_symmetric_model(self):
        src = _src('volatility.py')
        assert 'symmetric' in src.lower()

    def test_egarch_call_still_passes_no_o_kwarg(self):
        # Guards the code/comment from silently diverging again: the fixed
        # comment claims "symmetric (o=0)" — verify the call site agrees.
        src = _src('volatility.py')
        m = re.search(r"arch_model\(returns, vol='EGARCH'[^)]*\)", src)
        assert m is not None
        assert 'o=' not in m.group(0)


# ---------------------------------------------------------------------------
# EDIT 3: get_sigma once-per-symbol INFO gap-log on thin HAR history
# ---------------------------------------------------------------------------

class TestHARGapLog:
    def test_har_gap_logged_once_per_symbol(self, caplog):
        caplog.set_level(logging.INFO)
        bars = _ohlc(30, 0.01, seed=7)  # ~21 biz days, well under _HAR_MIN_DAYS=60
        returns = np.random.default_rng(1).normal(0, 1, 150)

        vol.get_sigma('THINSYM', returns, bars=bars, asset_type='stock')
        assert len(_logs(caplog, 'THINSYM: HAR-RV unavailable')) == 1

        caplog.clear()
        vol.get_sigma('THINSYM', returns, bars=bars, asset_type='stock')
        assert len(_logs(caplog, 'THINSYM: HAR-RV unavailable')) == 0  # deduped

    def test_har_gap_log_is_per_symbol_not_global(self, caplog):
        caplog.set_level(logging.INFO)
        bars = _ohlc(30, 0.01, seed=8)
        returns = np.random.default_rng(2).normal(0, 1, 150)

        vol.get_sigma('SYMA', returns, bars=bars, asset_type='stock')
        caplog.clear()
        vol.get_sigma('SYMB', returns, bars=bars, asset_type='stock')
        assert len(_logs(caplog, 'SYMB: HAR-RV unavailable')) == 1

    def test_har_gap_log_mentions_min_days_threshold(self, caplog):
        caplog.set_level(logging.INFO)
        bars = _ohlc(30, 0.01, seed=11)
        returns = np.random.default_rng(5).normal(0, 1, 150)

        vol.get_sigma('THRESH', returns, bars=bars, asset_type='stock')
        matches = _logs(caplog, 'THRESH: HAR-RV unavailable')
        assert len(matches) == 1
        assert '60' in matches[0].getMessage()

    def test_har_gap_log_not_emitted_when_har_succeeds(self, caplog):
        caplog.set_level(logging.INFO)
        bars = _ohlc(150, 0.01, seed=9)  # plenty of history -> HAR succeeds
        returns = np.random.default_rng(3).normal(0, 1, 150)

        sigma = vol.get_sigma('FULLHIST', returns, bars=bars, asset_type='stock')
        assert sigma is not None
        assert len(_logs(caplog, 'HAR-RV unavailable')) == 0

    def test_har_gap_log_not_emitted_without_bars(self, caplog, monkeypatch):
        caplog.set_level(logging.INFO)
        monkeypatch.setattr(vol, 'get_cached_sigma', lambda s, r: 0.02)
        returns = np.random.default_rng(6).normal(0, 1, 150)

        out = vol.get_sigma('NOBARS', returns, bars=None, asset_type='stock')
        assert out == 0.02
        assert len(_logs(caplog, 'HAR-RV unavailable')) == 0

    def test_fallthrough_sigma_value_unchanged_by_the_log(self, monkeypatch):
        """The log is purely additive: get_sigma's return value on the
        insufficient-history path must equal get_cached_sigma's output,
        exactly as before this edit."""
        monkeypatch.setattr(vol, 'get_cached_sigma', lambda s, r: 0.0321)
        bars = _ohlc(30, 0.01, seed=10)
        returns = np.random.default_rng(4).normal(0, 1, 150)

        out = vol.get_sigma('ANY', returns, bars=bars, asset_type='stock')
        assert out == 0.0321

    def test_har_gap_logged_set_exists_and_is_a_set(self):
        assert isinstance(vol._har_gap_logged, set)
