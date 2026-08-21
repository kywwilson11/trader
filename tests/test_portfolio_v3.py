"""portfolio.py panel-review v3: instrumentation, estimator sentinel, fail-closed
guards. Pure numpy/pandas — Mac-runnable; _LedoitWolf is forced to False
(corrcoef) by the autouse fixture so values are deterministic on machines
WITH sklearn too."""

import logging
import math
import sys
import time
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import portfolio
from portfolio import (
    avg_book_correlation,
    book_risk_budget,
    check_portfolio_correlation,
    compute_correlation_matrix,
    correlation_gate_stats,
    diversified_book_risk,
    ewma_annualized_vol,
    get_book_vol_scalar_cached,
    get_correlation_matrix_cached,
    get_correlation_sizing_factor,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    portfolio._book_vol_cache.clear()
    portfolio._corr_cache.clear()
    portfolio.correlation_gate_stats(reset=True)
    portfolio._bookcorr_warn_ts = 0.0
    monkeypatch.setattr(portfolio, '_LedoitWolf', False)
    yield


def _fake_api(equity_curve):
    return SimpleNamespace(
        get_portfolio_history=lambda **kw: SimpleNamespace(
            equity=equity_curve, timestamp=list(range(len(equity_curve)))))


# ---------------------------------------------------------------------------
# ewma_annualized_vol / _ewma_vol_diag
# ---------------------------------------------------------------------------

def test_ewma_accepts_array_like():
    rng = np.random.default_rng(7)
    curve = list(100_000 * np.cumprod(1 + rng.normal(0, 0.01, 60)))
    ref = ewma_annualized_vol(curve)
    assert ref is not None
    assert ewma_annualized_vol(np.asarray(curve)) == pytest.approx(ref)
    assert ewma_annualized_vol(tuple(curve)) == pytest.approx(ref)
    assert ewma_annualized_vol(pd.Series(curve)) == pytest.approx(ref)
    assert ewma_annualized_vol(None) is None
    assert ewma_annualized_vol([100.0] * 5) is None


def test_ewma_diag_outliers():
    clean = list(100_000 * np.cumprod(
        1 + np.random.default_rng(5).normal(0, 0.004, 63)))
    vol, d = portfolio._ewma_vol_diag(clean)
    assert d['n_outliers'] == 0 and vol < 0.15

    deposit = clean[:30] + [x * 1.5 for x in clean[30:]]
    vol2, d2 = portfolio._ewma_vol_diag(deposit)
    assert d2['n_outliers'] >= 1
    assert d2['max_abs_ret'] > 0.15
    # The single deposit-shaped jump inflates realized vol by an order of
    # magnitude relative to the clean series (not pinned to an absolute
    # ">1.0" bound — the EWMA recursion is seeded+decayed over the whole
    # window, so a lone late-window outlier is damped, not explosive).
    assert vol2 > 5 * vol
    assert vol2 > 0.3


# ---------------------------------------------------------------------------
# get_book_vol_scalar_cached
# ---------------------------------------------------------------------------

def test_book_vol_deposit_warning_and_floor(caplog):
    clean = list(100_000 * np.cumprod(
        1 + np.random.default_rng(5).normal(0, 0.004, 63)))
    deposit = clean[:30] + [x * 1.5 for x in clean[30:]]

    with caplog.at_level(logging.DEBUG, logger='portfolio'):
        s = get_book_vol_scalar_cached(_fake_api(deposit), 'stock')
    assert s == 0.5
    assert any('deposit/withdrawal' in r.getMessage()
               for r in caplog.records if r.levelname == 'WARNING')

    portfolio._book_vol_cache.clear()
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger='portfolio'):
        s2 = get_book_vol_scalar_cached(_fake_api(clean), 'stock')
    assert s2 == 1.0
    assert not any('deposit/withdrawal' in r.getMessage()
                   for r in caplog.records)
    assert any('series=portfolio_history(3M,1D)' in r.getMessage()
               for r in caplog.records if r.levelname == 'INFO')


def test_book_vol_failure_warns_and_still_caches(caplog):
    calls = []

    def h(**kw):
        calls.append(1)
        raise RuntimeError('boom')

    api = SimpleNamespace(get_portfolio_history=h)
    with caplog.at_level(logging.DEBUG, logger='portfolio'):
        s1 = get_book_vol_scalar_cached(api, 'crypto')
    assert s1 == 1.0
    assert any('portfolio history unavailable' in r.getMessage()
               for r in caplog.records if r.levelname == 'WARNING')

    s2 = get_book_vol_scalar_cached(api, 'crypto')
    assert s2 == 1.0
    assert len(calls) == 1  # failure result is still cached (deferred item)


def test_book_vol_missing_equity_warns(caplog):
    api = SimpleNamespace(get_portfolio_history=lambda **kw: SimpleNamespace())
    with caplog.at_level(logging.DEBUG, logger='portfolio'):
        s = get_book_vol_scalar_cached(api, 'crypto')
    assert s == 1.0
    assert any('insufficient equity history' in r.getMessage()
               for r in caplog.records if r.levelname == 'WARNING')


def test_book_vol_unknown_asset_type_warns(caplog):
    curve = list(100_000 * np.cumprod(
        1 + np.random.default_rng(9).normal(0, 0.02, 90)))
    realized = ewma_annualized_vol(curve)
    expected = min(max(0.25 / realized, 0.5), 1.0)
    with caplog.at_level(logging.DEBUG, logger='portfolio'):
        s = get_book_vol_scalar_cached(_fake_api(curve), 'stocks')
    assert s == pytest.approx(expected)
    assert any('unknown asset_type' in r.getMessage()
               for r in caplog.records if r.levelname == 'WARNING')


# ---------------------------------------------------------------------------
# compute_correlation_matrix: estimator sentinel + diagnostics
# ---------------------------------------------------------------------------

def test_corr_matrix_identical_to_corrcoef():
    rng = np.random.default_rng(2)
    rd = {s: rng.normal(size=40) for s in 'ABCDE'}
    out = compute_correlation_matrix(rd, window=30)
    symbols = sorted(rd.keys())
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            expected = float(np.corrcoef(rd[a][-30:], rd[b][-30:])[0, 1])
            assert out[(a, b)] == pytest.approx(expected)
            assert out[(a, b)] == out[(b, a)]
    assert len(out) == 5 * 4
    assert all(k[0] != k[1] for k in out)
    assert portfolio._last_corr_diag['estimator'] == 'corrcoef'


def test_ledoitwolf_branch_and_fallback(monkeypatch):
    class StubLW:
        def fit(self, X):
            self.covariance_ = np.array([[1.0, 0.3], [0.3, 1.0]])
            return self

    rng = np.random.default_rng(4)
    a = rng.normal(size=40)
    b = a + rng.normal(scale=0.01, size=40)   # near-clone of a
    rd = {'A': a, 'B': b}

    monkeypatch.setattr(portfolio, '_LedoitWolf', StubLW)
    out = compute_correlation_matrix(rd)
    # Stub covariance pins 0.3 -- corrcoef on these near-identical series
    # would read ~1.0, so this also pins non-interchangeability.
    assert out[('A', 'B')] == pytest.approx(0.3)
    assert out[('B', 'A')] == pytest.approx(0.3)
    assert portfolio._last_corr_diag['estimator'] == 'ledoit-wolf'

    class StubRaises:
        def fit(self, X):
            raise RuntimeError('fit failed')

    monkeypatch.setattr(portfolio, '_LedoitWolf', StubRaises)
    out2 = compute_correlation_matrix(rd)
    expected = float(np.corrcoef(a[-30:], b[-30:])[0, 1])
    assert out2[('A', 'B')] == pytest.approx(expected)
    assert portfolio._last_corr_diag['n_fallback'] == \
        portfolio._last_corr_diag['n_pairs']
    assert portfolio._last_corr_diag['n_fallback'] == 1


def test_degenerate_pair_warns_and_no_runtimewarning(caplog):
    rd = {'A': np.random.default_rng(1).normal(size=40), 'FLAT': np.zeros(40)}
    with caplog.at_level(logging.DEBUG, logger='portfolio'):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = compute_correlation_matrix(rd)  # must not raise
    assert out[('A', 'FLAT')] == 0.0
    assert portfolio._last_corr_diag['n_degenerate'] == 1
    assert any('zero-variance' in r.getMessage() for r in caplog.records)


def test_rebuild_coverage_and_stale_age(monkeypatch, caplog):
    syms = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    good = set(syms[:2])
    rng = np.random.default_rng(3)

    def _make_df():
        return pd.DataFrame(
            {'Close': rng.normal(100, 1, 40).cumsum() + 500})

    def fetch_stock_bars_alpaca(api, sym, **k):
        if sym in good:
            return _make_df()
        raise RuntimeError('no data')

    def fetch_bars_alpaca(api, sym, **k):
        if sym in good:
            return _make_df()
        raise RuntimeError('no data')

    m = types.ModuleType('market_data')
    m.fetch_stock_bars_alpaca = fetch_stock_bars_alpaca
    m.fetch_bars_alpaca = fetch_bars_alpaca
    m.closed_bars_v2_enabled = lambda: False   # real module exports it (c26 T2)
    monkeypatch.setitem(sys.modules, 'market_data', m)

    with caplog.at_level(logging.DEBUG, logger='portfolio'):
        corr = get_correlation_matrix_cached(object(), syms, 'stock')
    assert len(corr) == 2
    assert any('only 2/6 symbols' in r.getMessage()
               for r in caplog.records if r.levelname == 'WARNING')
    assert any('corr rebuild' in r.getMessage()
               for r in caplog.records if r.levelname == 'INFO')

    # Force the cached entry stale (older than _CORR_CACHE_TTL) so the next
    # call attempts a real rebuild instead of serving the cache hit.
    portfolio._corr_cache['stock'] = (corr, time.monotonic() - 4000)

    def _always_raise(api, sym, **k):
        raise RuntimeError('no data')

    m.fetch_stock_bars_alpaca = _always_raise
    m.fetch_bars_alpaca = _always_raise

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger='portfolio'):
        corr2 = get_correlation_matrix_cached(object(), syms, 'stock')
    assert corr2 == {}
    assert any('returned nothing' in r.getMessage()
               for r in caplog.records if r.levelname == 'WARNING')


# ---------------------------------------------------------------------------
# gate/sizing shared average + counters
# ---------------------------------------------------------------------------

def test_gate_stats_and_rejection_log(caplog):
    m = {('A', 'C'): 0.9, ('B', 'C'): 0.2}
    with caplog.at_level(logging.DEBUG, logger='portfolio'):
        r1 = check_portfolio_correlation(['A'], 'C', m)
        assert r1[0] is False and r1[1] == pytest.approx(0.9)
        assert any('gate: 1 rejected / 1 checked' in r.getMessage()
                   for r in caplog.records)

        r2 = check_portfolio_correlation(['B'], 'C', m)
        assert r2 == (True, pytest.approx(0.2))

        stats = correlation_gate_stats()
        assert stats['n_checked'] == 2
        assert stats['n_rejected'] == 1
        assert stats['max_avg_corr_seen'] == pytest.approx(0.9)

        caplog.clear()
        check_portfolio_correlation(['A'], 'C', {('A', 'C'): 0.6})
        assert any('near the corr cap' in r.getMessage()
                   for r in caplog.records)

    # reset=True must return the PRE-reset snapshot (read-and-reset pattern
    # for a journal/GUI poller), then zero the counters.
    snap = correlation_gate_stats(reset=True)
    assert snap['n_checked'] == 3 and snap['n_rejected'] == 1
    assert correlation_gate_stats()['n_checked'] == 0


def test_gate_and_sizing_share_one_average():
    m = {('C', 'A'): 0.8, ('C', 'B'): 0.4}   # reverse-ordered keys only
    allowed, avg = check_portfolio_correlation(['A', 'B'], 'C', m)
    assert avg == pytest.approx(0.6)
    assert isinstance(avg, float)
    assert get_correlation_sizing_factor('C', ['A', 'B'], m) == pytest.approx(
        1.0 / math.sqrt(1 + 2 * 0.6))


def test_avg_book_corr_coverage_warning_and_pyramid_silence(caplog):
    m = {('X', 'Y'): 0.9, ('Y', 'X'): 0.9}
    with caplog.at_level(logging.DEBUG, logger='portfolio'):
        assert avg_book_correlation(['A', 'B'], m) == 0.0
    assert any('pairs missing' in r.getMessage() for r in caplog.records)

    caplog.clear()
    portfolio._bookcorr_warn_ts = 0.0
    with caplog.at_level(logging.DEBUG, logger='portfolio'):
        assert avg_book_correlation(['X', 'Y'], m) == pytest.approx(0.9)
        # Diagonal duplicate from a pyramiding candidate list must not be
        # counted as a missing pair.
        assert avg_book_correlation(['X', 'Y', 'Y'], m) == pytest.approx(0.9)
    assert not any('pairs missing' in r.getMessage() for r in caplog.records)


def test_timestamp_diag_direct():
    # Direct coverage for the spacing/weekend instrumentation that settles
    # the parked 252-vs-365 annualization question from prod logs.
    from datetime import datetime, timezone
    base = datetime(2026, 7, 23, 21, 0, tzinfo=timezone.utc).timestamp()
    week = [base + i * 86400 for i in range(7)]      # Thu..Wed spans Sat+Sun
    spacing, has_weekend = portfolio._timestamp_diag(week)
    assert spacing == pytest.approx(1.0)
    assert has_weekend is True

    mon = datetime(2026, 7, 20, 21, 0, tzinfo=timezone.utc).timestamp()
    weekdays = [mon + i * 86400 for i in range(3)]   # Mon, Tue, Wed
    spacing2, has_weekend2 = portfolio._timestamp_diag(weekdays)
    assert spacing2 == pytest.approx(1.0)
    assert has_weekend2 is False

    # Non-epoch small values: spacing still computed, weekday call skipped.
    spacing3, has_weekend3 = portfolio._timestamp_diag([0, 1, 2, 3])
    assert spacing3 == pytest.approx(1.0 / 86400.0)
    assert has_weekend3 is None

    # Best-effort contract: junk / too-short / None never raise.
    assert portfolio._timestamp_diag(['x', 'y', 'z']) == (None, None)
    assert portfolio._timestamp_diag([base, base + 86400]) == (None, None)
    assert portfolio._timestamp_diag(None) == (None, None)


# ---------------------------------------------------------------------------
# ENB kernel guards
# ---------------------------------------------------------------------------

def test_book_risk_budget_cap_guard():
    assert book_risk_budget([0.005], 0.5, -0.025) == 0.0
    assert book_risk_budget([0.005], 0.5, float('nan')) == 0.0
    assert book_risk_budget([0.005], 0.5, 0.0) == 0.0
    for rho in (0.0, 0.4, 0.8):
        b = book_risk_budget([0.005, 0.004, 0.006], rho, 0.025)
        total = diversified_book_risk([0.005, 0.004, 0.006] + [b], rho)
        assert total == pytest.approx(0.025, abs=1e-12)


def test_nan_rho_fails_closed():
    v = diversified_book_risk([0.005, 0.005], float('nan'))
    assert np.isfinite(v) and v == pytest.approx(0.010)
    b = book_risk_budget([0.005] * 5, float('nan'), 0.025)
    assert np.isfinite(b) and b == 0.0


# ---------------------------------------------------------------------------
# doc/source guards + unreachability fact
# ---------------------------------------------------------------------------

def test_docstring_and_source_guards_v3():
    d = portfolio.get_correlation_sizing_factor.__doc__
    assert 'AVERAGE share' in d
    assert 'marginal variance ~ (1 + n' not in d

    assert 'sample variance' in portfolio.ewma_annualized_vol.__doc__

    src = Path(portfolio.__file__).read_text()
    assert 'Account-level realized-vol scalar' in src   # not "Book-level"
    assert 'sqrt(252/365)' in src                        # annualization caveat

    # No TOP-LEVEL sklearn import (only the indented one inside
    # _resolve_ledoit_wolf). Checked by statement-prefix, not bare
    # substring: several top-level comments legitimately say "sklearn"
    # in prose (documenting the lazy-sentinel design) without importing it.
    assert all(
        not line.lstrip().startswith(('import sklearn', 'from sklearn'))
        or line.startswith((' ', '\t'))
        for line in src.splitlines())


def test_min_len_short_branch_unreachable_fact():
    # get_returns_for_symbols admits a symbol only when len(df) > 10, which
    # always yields >= 10 returns after pct_change().dropna() -- so the
    # compute_correlation_matrix min_len<10 branch is defensive-only from
    # the production caller.
    for n in range(11, 16):
        s = pd.Series(range(n), dtype=float).pct_change().dropna()
        assert len(s) >= 10
