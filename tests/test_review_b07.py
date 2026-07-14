"""Review-batch b07 regression tests: macro_indicators, regime_detector,
adaptive_config.

Covers the b07 review fixes: silent-failure logging (SPY trend, VIX/stress/
CAPE fetchers, blind regime), stablecoin total-outage not cached as all-clear,
regime-detector failure visibility (hmmlearn import warn-once, rate-limited
fit warnings), dead-output removal (threshold_mult/stop_mult), raw-fit logging
before smoothing, adaptive-config dead config removal, PID-unique tmp writes,
fail-closed corrupt-state load, and study-DB sidecar cleanup.

All tests are Mac-runnable: network and hmmlearn are stubbed via sys.modules /
monkeypatch; heavy deps are never imported.
"""

import json
import logging
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import adaptive_config as ac
import macro_indicators as mi
import regime_detector as rd

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean_module_state():
    """Isolate module-level caches/counters between tests."""
    mi._cache.clear()
    rd._hmm_cache.clear()
    rd._last_regime.clear()
    rd._fail_counts.clear()
    yield
    mi._cache.clear()
    rd._hmm_cache.clear()
    rd._last_regime.clear()
    rd._fail_counts.clear()


def _warnings(caplog, needle):
    return [r for r in caplog.records
            if r.levelno >= logging.WARNING and needle in r.getMessage()]


# ---------------------------------------------------------------------------
# Stubs (no network, no hmmlearn)
# ---------------------------------------------------------------------------

class _Quote:
    def __init__(self, bp, ap):
        self.bp, self.ap = bp, ap


class _PegAPI:
    """Stub Alpaca api for check_stablecoin_pegs."""

    def __init__(self, quotes=None, fail=False):
        self.quotes = quotes or {}
        self.fail = fail
        self.calls = 0

    def get_latest_crypto_quotes(self, symbols):
        self.calls += 1
        if self.fail:
            raise ConnectionError('quote outage')
        sym = symbols[0]
        if sym not in self.quotes:
            raise KeyError(sym)
        return {sym: self.quotes[sym]}


class _Bar:
    def __init__(self, c):
        self.c = c


class _BarsAPI:
    """Stub Alpaca api for get_spy_trend_ok."""

    def __init__(self, closes=None, fail=False):
        self.closes = closes or []
        self.fail = fail

    def get_bars(self, symbol, timeframe, start=None, adjustment=None):
        if self.fail:
            raise ConnectionError('bars outage')
        return [_Bar(c) for c in self.closes]


def _broken_yfinance():
    m = types.ModuleType('yfinance')

    def _ticker(sym):
        raise ConnectionError('yahoo down')

    m.Ticker = _ticker
    return m


def _yfinance_with(info=None, hist=None):
    m = types.ModuleType('yfinance')

    class _T:
        def __init__(self, sym):
            self.info = info or {}

        def history(self, period='5d'):
            return hist

    m.Ticker = _T
    return m


def _requests_stub(status_code=200, text='', exc=None):
    m = types.ModuleType('requests')

    class _Resp:
        pass

    _Resp.status_code = status_code
    _Resp.text = text

    def get(url, timeout=10):
        if exc is not None:
            raise exc
        return _Resp()

    m.get = get
    return m


class _FakeHTTPResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body.encode()


def _urlopen_returning(body):
    def _fake(req, timeout=10):
        return _FakeHTTPResponse(body)
    return _fake


def _urlopen_raising(req, timeout=10):
    raise ConnectionError('fred down')


# ---------------------------------------------------------------------------
# macro_indicators: fetchers warn on total failure (fail-open, but visible)
# ---------------------------------------------------------------------------

class TestMacroFetchers:
    def test_fetch_vix_total_failure_warns_and_returns_none(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, 'yfinance', _broken_yfinance())
        monkeypatch.setattr('urllib.request.urlopen', _urlopen_raising)
        with caplog.at_level(logging.DEBUG, logger='macro_indicators'):
            assert mi.fetch_vix() is None
        assert _warnings(caplog, 'VIX unavailable from ALL sources')

    def test_fetch_vix_yfinance_happy_path(self, monkeypatch, caplog):
        import pandas as pd
        hist = pd.DataFrame({'Close': [16.0, 17.0]})
        monkeypatch.setitem(sys.modules, 'yfinance', _yfinance_with(hist=hist))
        with caplog.at_level(logging.DEBUG, logger='macro_indicators'):
            assert mi.fetch_vix() == 17.0
        assert mi._get_cached('vix', 3600) == 17.0
        assert not _warnings(caplog, 'VIX unavailable')

    def test_fetch_vix_fred_fallback_skips_dot_tail(self, monkeypatch):
        """FRED CSV parser scans backwards past '.' placeholders and header."""
        monkeypatch.setitem(sys.modules, 'yfinance', _broken_yfinance())
        body = "DATE,VIXCLS\n2026-06-29,16.8\n2026-06-30,17.5\n2026-07-01,."
        monkeypatch.setattr('urllib.request.urlopen', _urlopen_returning(body))
        assert mi.fetch_vix() == 17.5
        assert mi._get_cached('vix', 3600) == 17.5

    def test_fetch_stress_happy_path(self, monkeypatch):
        text = "DATE,STLFSI2\n2026-06-19,-0.213\n2026-06-26,1.25"
        monkeypatch.setitem(sys.modules, 'requests', _requests_stub(text=text))
        assert mi.fetch_financial_stress() == 1.25

    def test_fetch_stress_failure_warns(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, 'requests',
                            _requests_stub(exc=ConnectionError('fred down')))
        with caplog.at_level(logging.DEBUG, logger='macro_indicators'):
            assert mi.fetch_financial_stress() is None
        assert _warnings(caplog, 'STLFSI2')

    def test_fetch_cape_missing_pe_warns(self, monkeypatch, caplog):
        """trailingPE=None raises no exception — the no-log gap that was fixed."""
        monkeypatch.setitem(sys.modules, 'yfinance', _yfinance_with(info={}))
        with caplog.at_level(logging.DEBUG, logger='macro_indicators'):
            assert mi.fetch_cape() is None
        assert _warnings(caplog, 'CAPE')

    def test_fetch_cape_happy_path(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'yfinance',
                            _yfinance_with(info={'trailingPE': 25.0}))
        assert mi.fetch_cape() == pytest.approx(40.0)  # 25 * 1.6


# ---------------------------------------------------------------------------
# macro_indicators: stablecoin peg checks
# ---------------------------------------------------------------------------

class TestStablecoinPegs:
    def test_total_outage_not_cached_and_warns(self, monkeypatch, caplog):
        api = _PegAPI(fail=True)
        with caplog.at_level(logging.DEBUG, logger='macro_indicators'):
            r1 = mi.check_stablecoin_pegs(api)
        assert r1 == {'depegged': False, 'emergency': False, 'deviations': {}}
        assert 'stablecoins' not in mi._cache          # failure NOT cached
        assert _warnings(caplog, 'peg status UNKNOWN')
        calls_after_first = api.calls
        mi.check_stablecoin_pegs(api)                  # retries immediately
        assert api.calls == calls_after_first + len(mi._STABLECOINS)

    def test_success_is_cached(self):
        api = _PegAPI(quotes={'USDT/USD': _Quote(0.999, 1.001),
                              'USDC/USD': _Quote(0.998, 1.000)})
        r1 = mi.check_stablecoin_pegs(api)
        calls = api.calls
        r2 = mi.check_stablecoin_pegs(api)
        assert api.calls == calls                      # served from cache
        assert r2 == r1
        assert not r1['depegged']
        assert len(r1['deviations']) == 2

    def test_partial_success_still_cached(self):
        """One live quote is real data — only the ALL-failed case skips cache."""
        api = _PegAPI(quotes={'USDT/USD': _Quote(0.999, 1.001)})
        r = mi.check_stablecoin_pegs(api)
        assert len(r['deviations']) == 1
        assert 'stablecoins' in mi._cache

    def test_warn_tier_deviation(self):
        api = _PegAPI(quotes={'USDT/USD': _Quote(0.992, 0.994),
                              'USDC/USD': _Quote(0.999, 1.001)})
        r = mi.check_stablecoin_pegs(api)
        assert r['depegged'] is True
        assert r['emergency'] is False

    def test_emergency_tier_deviation(self):
        api = _PegAPI(quotes={'USDT/USD': _Quote(0.960, 0.970),
                              'USDC/USD': _Quote(0.999, 1.001)})
        r = mi.check_stablecoin_pegs(api)
        assert r['emergency'] is True
        assert r['depegged'] is True


# ---------------------------------------------------------------------------
# macro_indicators: SPY 200d trend filter
# ---------------------------------------------------------------------------

class TestSpyTrend:
    def test_fetch_failure_warns_and_fails_open(self, caplog):
        with caplog.at_level(logging.DEBUG, logger='macro_indicators'):
            assert mi.get_spy_trend_ok(_BarsAPI(fail=True)) is None
        assert _warnings(caplog, 'SPY trend fetch failed')

    def test_short_history_warns_and_fails_open(self, caplog):
        with caplog.at_level(logging.DEBUG, logger='macro_indicators'):
            assert mi.get_spy_trend_ok(_BarsAPI(closes=[100.0] * 50)) is None
        assert _warnings(caplog, '<200')

    def test_above_trend_true_and_cached(self):
        closes = [float(100 + i * 0.5) for i in range(250)]  # rising
        assert mi.get_spy_trend_ok(_BarsAPI(closes=closes)) is True
        assert mi._get_cached('spy_trend', 3600) is True

    def test_below_trend_false(self):
        closes = [float(300 - i * 0.5) for i in range(250)]  # falling
        assert mi.get_spy_trend_ok(_BarsAPI(closes=closes)) is False


# ---------------------------------------------------------------------------
# macro_indicators: get_macro_regime composition
# ---------------------------------------------------------------------------

def _patch_fetchers(monkeypatch, vix=None, stress=None, cape=None):
    monkeypatch.setattr(mi, 'fetch_vix', lambda: vix)
    monkeypatch.setattr(mi, 'fetch_financial_stress', lambda: stress)
    monkeypatch.setattr(mi, 'fetch_cape', lambda: cape)


class TestMacroRegime:
    @pytest.mark.parametrize('vix,mult,label', [
        (10.0, 1.0, 'normal'),
        (16.0, 0.8, 'caution'),
        (26.0, 0.5, 'defensive'),
        (36.0, 0.3, 'crisis'),
    ])
    def test_vix_tiers(self, monkeypatch, vix, mult, label):
        _patch_fetchers(monkeypatch, vix=vix)
        r = mi.get_macro_regime(api=None, asset_type='crypto')
        assert r.sizing_mult == pytest.approx(mult)
        assert label in r.regime_label

    def test_high_stress_halves_sizing_tightens_stops(self, monkeypatch):
        _patch_fetchers(monkeypatch, vix=10.0, stress=1.5)
        r = mi.get_macro_regime(api=None, asset_type='crypto')
        assert r.sizing_mult == pytest.approx(0.5)
        assert r.stop_mult == pytest.approx(0.8)
        assert 'high_stress' in r.regime_label

    def test_cape_overvaluation_stocks_only(self, monkeypatch):
        _patch_fetchers(monkeypatch, vix=10.0, cape=40.0)  # z = 1.875 > 1.5
        r = mi.get_macro_regime(api=None, asset_type='stock')
        assert r.sizing_mult == pytest.approx(0.7)
        assert 'overvalued' in r.regime_label

    def test_cape_not_fetched_for_crypto(self, monkeypatch):
        _patch_fetchers(monkeypatch, vix=10.0, cape=40.0)
        r = mi.get_macro_regime(api=None, asset_type='crypto')
        assert r.cape is None
        assert 'overvalued' not in r.regime_label

    def test_stablecoin_emergency_zeroes_sizing(self, monkeypatch):
        _patch_fetchers(monkeypatch, vix=10.0)
        monkeypatch.setattr(mi, 'check_stablecoin_pegs',
                            lambda api: {'depegged': True, 'emergency': True,
                                         'deviations': {'USDT/USD': 0.03}})
        r = mi.get_macro_regime(api=object(), asset_type='crypto')
        assert r.sizing_mult == 0.0
        assert r.stablecoin_alert is True
        assert 'stablecoin_emergency' in r.regime_label

    def test_stablecoin_warning_tightens_stops(self, monkeypatch):
        _patch_fetchers(monkeypatch, vix=10.0)
        monkeypatch.setattr(mi, 'check_stablecoin_pegs',
                            lambda api: {'depegged': True, 'emergency': False,
                                         'deviations': {'USDT/USD': 0.007}})
        r = mi.get_macro_regime(api=object(), asset_type='crypto')
        assert r.stop_mult == pytest.approx(0.7)
        assert 'stablecoin_warning' in r.regime_label

    def test_blind_regime_warns_but_stays_normal(self, monkeypatch, caplog):
        """All inputs None: label stays 'normal' (unchanged behavior) but a
        WARNING now distinguishes blindness from a genuinely calm market."""
        _patch_fetchers(monkeypatch)
        with caplog.at_level(logging.DEBUG, logger='macro_indicators'):
            r = mi.get_macro_regime(api=None, asset_type='crypto')
        assert r.regime_label == 'normal'
        assert r.sizing_mult == pytest.approx(1.0)
        assert _warnings(caplog, 'WITHOUT VIX')

    def test_elevated_vix_pops_cache_for_faster_refetch(self, monkeypatch):
        _patch_fetchers(monkeypatch, vix=30.0)
        mi._set_cached('vix', 30.0)
        mi.get_macro_regime(api=None, asset_type='crypto')
        assert 'vix' not in mi._cache

    def test_calm_vix_keeps_cache(self, monkeypatch):
        _patch_fetchers(monkeypatch, vix=15.0)
        mi._set_cached('vix', 15.0)
        mi.get_macro_regime(api=None, asset_type='crypto')
        assert 'vix' in mi._cache

    def test_cache_ttl_helpers(self, monkeypatch):
        mi._set_cached('k', 42)
        assert mi._get_cached('k', 1000) == 42
        monkeypatch.setattr(mi.time, 'time',
                            lambda _real=time.time: _real() + 2000)
        assert mi._get_cached('k', 1000) is None       # expired


# ---------------------------------------------------------------------------
# regime_detector: failure visibility
# ---------------------------------------------------------------------------

class TestRegimeDetectorFailures:
    def test_missing_hmmlearn_warns_once_and_fails_neutral(self, monkeypatch, caplog):
        # Force ImportError deterministically (even where hmmlearn exists)
        monkeypatch.setitem(sys.modules, 'hmmlearn', None)
        monkeypatch.setitem(sys.modules, 'hmmlearn.hmm', None)
        monkeypatch.setattr(rd, '_import_warned', False)
        returns = np.random.randn(250)
        with caplog.at_level(logging.DEBUG, logger='regime_detector'):
            assert rd.fit_hmm(returns) == (None, None)
            assert rd.fit_hmm(returns) == (None, None)
        assert len(_warnings(caplog, 'hmmlearn unavailable')) == 1  # warn-once

    def test_fit_failures_warn_rate_limited(self, monkeypatch, caplog):
        fake_hmm = types.ModuleType('hmmlearn.hmm')

        class _BoomHMM:
            def __init__(self, **kwargs):
                pass

            def fit(self, X):
                raise RuntimeError('degenerate fit')

        fake_hmm.GaussianHMM = _BoomHMM
        fake_root = types.ModuleType('hmmlearn')
        fake_root.hmm = fake_hmm
        monkeypatch.setitem(sys.modules, 'hmmlearn', fake_root)
        monkeypatch.setitem(sys.modules, 'hmmlearn.hmm', fake_hmm)

        returns = np.random.randn(250)
        n_calls = rd._WARN_EVERY + 1
        with caplog.at_level(logging.DEBUG, logger='regime_detector'):
            for _ in range(n_calls):
                assert rd.fit_hmm(returns) == (None, None)
        # WARNING on the 1st and on the _WARN_EVERY-th; DEBUG in between
        assert len(_warnings(caplog, 'HMM fit failed')) == 2
        assert rd._fail_counts['fit'] == n_calls

    def test_insufficient_data_still_quiet_none(self):
        assert rd.fit_hmm(np.random.randn(50)) == (None, None)


# ---------------------------------------------------------------------------
# regime_detector: regime dict shape and sizing branches (no hmmlearn needed)
# ---------------------------------------------------------------------------

_STATE_LABELS = {
    0: {'label': 'bear', 'mean': -1.0, 'vol': 1.0},
    1: {'label': 'neutral', 'mean': 0.0, 'vol': 1.0},
    2: {'label': 'bull', 'mean': 1.0, 'vol': 1.0},
}


class _FakeHMM:
    """Duck-typed fitted model: constant Viterbi state, one-hot posteriors."""

    def __init__(self, state, n_states=3):
        self._state, self._n = state, n_states

    def predict(self, X):
        return np.full(len(X), self._state, dtype=int)

    def predict_proba(self, X):
        out = np.zeros((len(X), self._n))
        out[:, self._state] = 1.0
        return out


class TestRegimeDict:
    def test_default_regime_shape(self):
        r = rd._default_regime()
        assert set(r) == {'label', 'probabilities', 'sizing_mult'}
        assert r['sizing_mult'] == 1.0
        assert r['label'] == 'unknown'

    def test_dead_outputs_removed(self):
        r = rd.get_current_regime(_FakeHMM(2), _STATE_LABELS, np.zeros(20))
        assert 'threshold_mult' not in r
        assert 'stop_mult' not in r

    @pytest.mark.parametrize('state,label,mult', [
        (2, 'bull', 1.2),
        (0, 'bear', 0.3),
        (1, 'neutral', 1.0),
    ])
    def test_sizing_branches(self, state, label, mult):
        r = rd.get_current_regime(_FakeHMM(state), _STATE_LABELS, np.zeros(20))
        assert r['label'] == label
        assert r['sizing_mult'] == pytest.approx(mult)
        assert r['probabilities'][label] == pytest.approx(1.0)

    def test_high_vol_state_relabeled(self):
        labels = {
            0: {'label': 'bear', 'mean': -1.0, 'vol': 1.0},
            1: {'label': 'neutral', 'mean': 0.0, 'vol': 5.0},  # >1.5x median
            2: {'label': 'bull', 'mean': 1.0, 'vol': 1.0},
        }
        r = rd.get_current_regime(_FakeHMM(1), labels, np.zeros(20))
        assert r['label'] == 'high_vol'
        assert r['sizing_mult'] == pytest.approx(0.5)

    def test_none_model_neutral(self):
        assert rd.get_current_regime(None, {}, np.zeros(20)) == rd._default_regime()

    def test_short_returns_neutral(self):
        r = rd.get_current_regime(_FakeHMM(2), _STATE_LABELS, np.zeros(5))
        assert r == rd._default_regime()


class TestRegimeSmoothing:
    _BULL = {'label': 'bull', 'probabilities': {'bull': 0.9}, 'sizing_mult': 1.2}
    _BEAR = {'label': 'bear', 'probabilities': {'bear': 0.9}, 'sizing_mult': 0.3}

    def test_first_observation_forced_neutral(self):
        r = rd._smooth_regime('S', dict(self._BULL))
        assert r == rd._default_regime()
        assert rd._last_regime['S'] == ('bull', 1)

    def test_same_label_passes_and_counts(self):
        rd._last_regime['S'] = ('bull', 2)
        r = rd._smooth_regime('S', dict(self._BULL))
        assert r['label'] == 'bull'
        assert rd._last_regime['S'] == ('bull', 3)

    def test_flicker_suppressed(self):
        rd._last_regime['S'] = ('bull', 1)   # not persistent yet
        r = rd._smooth_regime('S', dict(self._BEAR))
        assert r == rd._default_regime()

    def test_persistent_switch_allowed(self):
        rd._last_regime['S'] = ('bull', 5)   # >= _REGIME_PERSISTENCE
        r = rd._smooth_regime('S', dict(self._BEAR))
        assert r['label'] == 'bear'

    def test_raw_fitted_regime_logged_before_smoothing(self, monkeypatch, caplog):
        """The refit INFO line must carry the RAW fitted label/probs even though
        first-call smoothing returns 'unknown' to the caller."""
        monkeypatch.setattr(rd, 'fit_hmm',
                            lambda returns, n_states=3: (object(), _STATE_LABELS))
        monkeypatch.setattr(rd, 'get_current_regime',
                            lambda m, l, r: dict(self._BULL))
        with caplog.at_level(logging.INFO, logger='regime_detector'):
            regime = rd.get_cached_regime('TESTSYM', np.zeros(300))
        assert regime['label'] == 'unknown'  # smoothing unchanged
        refit_lines = [r.getMessage() for r in caplog.records
                       if '[REGIME]' in r.getMessage()]
        assert refit_lines and 'fitted bull' in refit_lines[0]


# ---------------------------------------------------------------------------
# adaptive_config
# ---------------------------------------------------------------------------

class TestAdaptiveConfig:
    def test_post_explore_removed_and_fallback_safe(self):
        assert 'post_explore' not in ac.TRIAL_COUNTS
        # get_trial_count falls back to refine for unknown modes
        assert ac.get_trial_count('post_explore') == ac.TRIAL_COUNTS['refine']

    def test_decide_mode_score_arg_is_inert(self):
        state = {
            'mode': 'refine',
            'best_score': 2.0,
            'best_params': {'forward_bars': 24},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
            'cycles_without_improvement': 0,
        }
        # documented-unused parameter: result cannot depend on it
        assert ac.decide_mode(state, 0.0) == ac.decide_mode(state, 999.0) == 'refine'

    def test_save_uses_pid_unique_tmp_and_leaves_no_litter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ac, 'BASE_DIR', tmp_path)
        captured = {}
        real_replace = os.replace

        def spy(src, dst):
            captured['src'] = str(src)
            return real_replace(src, dst)

        monkeypatch.setattr(ac.os, 'replace', spy)
        ac.save_adaptive_state(ac._default_state('b07test'))
        assert captured['src'].endswith(f'.tmp.{os.getpid()}')
        assert (tmp_path / 'adaptive_state_b07test.json').exists()
        assert list(tmp_path.glob('*.tmp*')) == []

    def test_load_corrupt_state_fails_closed_naming_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ac, 'BASE_DIR', tmp_path)
        bad = tmp_path / 'adaptive_state_bad.json'
        bad.write_text('{"asset_type": "bad", TRUNCATED')
        with pytest.raises(ValueError) as excinfo:
            ac.load_adaptive_state('bad')
        assert 'adaptive_state_bad.json' in str(excinfo.value)

    def test_load_roundtrip_still_works(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ac, 'BASE_DIR', tmp_path)
        ac.save_adaptive_state(ac._default_state('b07rt'))
        state = ac.load_adaptive_state('b07rt')
        assert state['asset_type'] == 'b07rt'
        assert state['search_space'] == ac.DEFAULT_SEARCH_SPACE

    def test_update_after_search_removes_db_and_sidecars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ac, 'BASE_DIR', tmp_path)
        db = tmp_path / 'v2_study.db'
        sidecars = [Path(str(db) + s) for s in ('-wal', '-shm', '-journal')]
        for f in [db] + sidecars:
            f.write_text('x')
        state = {
            'asset_type': 'b07db',
            'best_score': 0.0,
            'best_params': {},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
            'mode': 'refine',
            'cycles_without_improvement': 0,
            'expansion_history': [],
            'last_updated': '',
        }
        # forward_bars at high edge -> categorical expansion -> DB reset
        result = ac.update_after_search(state, 1.0, {'forward_bars': 48},
                                        study_db_path=str(db))
        assert not db.exists()
        for f in sidecars:
            assert not f.exists()
        assert 64 in result['search_space']['forward_bars']

    def test_update_after_search_no_categorical_change_keeps_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ac, 'BASE_DIR', tmp_path)
        db = tmp_path / 'v2_study.db'
        db.write_text('x')
        state = {
            'asset_type': 'b07keep',
            'best_score': 0.0,
            'best_params': {},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48],
                             'dropout': [0.10, 0.40]},
            'mode': 'refine',
            'cycles_without_improvement': 0,
            'expansion_history': [],
            'last_updated': '',
        }
        # dropout (range param) at high edge: expansion but NOT categorical
        ac.update_after_search(state, 1.0, {'dropout': 0.39},
                               study_db_path=str(db))
        assert db.exists()


# ---------------------------------------------------------------------------
# Source hygiene (dead code / doc rot fixed and kept out)
# ---------------------------------------------------------------------------

def _src(name):
    return (REPO / name).read_text()


class TestSourceHygiene:
    def test_macro_dead_imports_removed(self):
        src = _src('macro_indicators.py')
        assert 'SimpleNamespace' not in src
        assert 'from dataclasses import dataclass' not in src

    def test_macro_stale_comments_fixed(self):
        src = _src('macro_indicators.py')
        assert 'Shiller data via free API' not in src
        assert 'free JSON API' not in src
        assert '15-min natural throttle' not in src

    def test_regime_attribution_corrected(self):
        src = _src('regime_detector.py')
        assert 'Hamilton' in src
        assert 'Sargent' not in src

    def test_regime_dead_outputs_gone_from_source(self):
        src = _src('regime_detector.py')
        assert 'threshold_mult' not in src

    def test_regime_dead_guards_gone(self):
        src = _src('regime_detector.py')
        assert 'rank < len(labels_list)' not in src
        assert 'state_id < len(vols)' not in src

    def test_viterbi_posterior_distinction_documented(self):
        assert 'Viterbi' in _src('regime_detector.py')
