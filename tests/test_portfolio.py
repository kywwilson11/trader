"""Tests for the portfolio layer: variance-correct correlation tilt,
equicorrelation book-risk cap, and the EWMA book-vol scalar."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio import (avg_book_correlation, book_risk_budget,
                       diversified_book_risk, ewma_annualized_vol,
                       get_book_vol_scalar_cached,
                       get_correlation_sizing_factor)
import portfolio


# --- get_correlation_sizing_factor: 1/sqrt(1 + n*rho) ---

def test_corr_factor_empty_book_is_neutral():
    assert get_correlation_sizing_factor('BTC/USD', [], {}) == 1.0


def test_corr_factor_uncorrelated_is_neutral():
    corr = {('ETH/USD', 'BTC/USD'): 0.0}
    assert get_correlation_sizing_factor('BTC/USD', ['ETH/USD'], corr) == 1.0


def test_corr_factor_matches_formula():
    # 3 positions, all corr 0.8 with candidate -> 1/sqrt(1 + 3*0.8)
    corr = {('A', 'X'): 0.8, ('B', 'X'): 0.8, ('C', 'X'): 0.8}
    got = get_correlation_sizing_factor('X', ['A', 'B', 'C'], corr)
    assert got == pytest.approx(max(0.4, 1 / math.sqrt(1 + 3 * 0.8)))


def test_corr_factor_scales_down_with_book_size():
    # Same avg corr, bigger book -> SMALLER factor (the old linear
    # formula failed this: the 5th clone got the 1st clone's haircut)
    corr = {(s, 'X'): 0.7 for s in 'ABCDE'}
    f1 = get_correlation_sizing_factor('X', ['A'], corr)
    f5 = get_correlation_sizing_factor('X', list('ABCDE'), corr)
    assert f5 < f1 < 1.0


def test_corr_factor_floor():
    corr = {(s, 'X'): 1.0 for s in 'ABCDEFGH'}
    assert get_correlation_sizing_factor('X', list('ABCDEFGH'), corr) == 0.4


# --- diversified_book_risk: equicorrelation model ---

def test_book_risk_independent_is_rss():
    # rho=0 -> sqrt of sum of squares
    risks = [0.005, 0.005, 0.005, 0.005]
    expected = math.sqrt(4 * 0.005 ** 2)
    assert diversified_book_risk(risks, 0.0) == pytest.approx(expected)


def test_book_risk_lockstep_is_sum():
    # rho=1 -> plain sum
    risks = [0.005, 0.003, 0.002]
    assert diversified_book_risk(risks, 1.0) == pytest.approx(0.010)


def test_book_risk_monotone_in_rho():
    risks = [0.005] * 5
    vals = [diversified_book_risk(risks, r) for r in (0.0, 0.3, 0.6, 0.9)]
    assert vals == sorted(vals)


def test_book_risk_ignores_nonpositive():
    assert diversified_book_risk([0.005, 0.0, -0.1], 1.0) == pytest.approx(0.005)
    assert diversified_book_risk([], 0.5) == 0.0


# --- book_risk_budget: closed-form headroom ---

def test_budget_empty_book_is_full_cap():
    assert book_risk_budget([], 0.7, 0.025) == pytest.approx(0.025)


def test_budget_solves_the_cap_equation():
    # Adding exactly the budget must land the book exactly on the cap
    existing = [0.005, 0.004, 0.006]
    for rho in (0.0, 0.4, 0.8, 1.0):
        b = book_risk_budget(existing, rho, 0.025)
        assert b > 0
        total = diversified_book_risk(existing + [b], rho)
        assert total == pytest.approx(0.025, abs=1e-12)


def test_budget_zero_when_cap_used_up():
    # 5 lockstep positions at 0.5% each = 2.5% book risk -> no headroom
    assert book_risk_budget([0.005] * 5, 1.0, 0.025) == 0.0


def test_budget_larger_for_uncorrelated_books():
    existing = [0.005] * 3
    assert (book_risk_budget(existing, 0.0, 0.025)
            > book_risk_budget(existing, 0.9, 0.025))


# --- avg_book_correlation ---

def test_avg_book_correlation():
    corr = {('A', 'B'): 0.8, ('A', 'C'): -0.4, ('B', 'C'): 0.6}
    # mean(|0.8|, |-0.4|, |0.6|) = 0.6
    assert avg_book_correlation(['A', 'B', 'C'], corr) == pytest.approx(0.6)
    assert avg_book_correlation(['A'], corr) == 0.0
    assert avg_book_correlation(['A', 'B'], {}) == 0.0


# --- EWMA book vol ---

def test_ewma_vol_flat_curve_is_zero():
    assert ewma_annualized_vol([100_000.0] * 30) == pytest.approx(0.0)


def test_ewma_vol_scales_with_volatility():
    rng = np.random.default_rng(3)
    base = 100_000 * np.cumprod(1 + rng.normal(0, 0.01, 60))
    calm = ewma_annualized_vol(list(base))
    wild = ewma_annualized_vol(list(100_000 * np.cumprod(1 + rng.normal(0, 0.04, 60))))
    assert calm is not None and wild is not None
    assert wild > calm > 0


def test_ewma_vol_insufficient_data():
    assert ewma_annualized_vol([100.0] * 5) is None
    assert ewma_annualized_vol(None) is None


def _fake_api(equity_curve):
    return SimpleNamespace(
        get_portfolio_history=lambda **kw: SimpleNamespace(
            equity=equity_curve, timestamp=list(range(len(equity_curve)))))


def test_book_vol_scalar_derisk_only():
    portfolio._book_vol_cache.clear()
    # ~63% annualized realized vol vs 35% crypto target -> ~0.55x
    rng = np.random.default_rng(11)
    curve = list(100_000 * np.cumprod(1 + rng.normal(0, 0.04, 90)))
    s = get_book_vol_scalar_cached(_fake_api(curve), 'crypto')
    assert 0.5 <= s < 1.0

    portfolio._book_vol_cache.clear()
    # Calm curve -> clamped at 1.0, NEVER boosts
    calm = list(100_000 * np.cumprod(1 + rng.normal(0, 0.001, 90)))
    assert get_book_vol_scalar_cached(_fake_api(calm), 'crypto') == 1.0


def test_book_vol_scalar_neutral_on_failure():
    portfolio._book_vol_cache.clear()
    api = SimpleNamespace(get_portfolio_history=None)  # not callable
    assert get_book_vol_scalar_cached(api, 'stock') == 1.0


def test_book_vol_scalar_cached_per_asset():
    portfolio._book_vol_cache.clear()
    calls = []

    def hist(**kw):
        calls.append(1)
        return SimpleNamespace(equity=[100.0] * 30, timestamp=[])

    api = SimpleNamespace(get_portfolio_history=hist)
    get_book_vol_scalar_cached(api, 'crypto')
    get_book_vol_scalar_cached(api, 'crypto')
    assert len(calls) == 1  # second hit served from cache
