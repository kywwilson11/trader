"""c26 packet S3 — de-risk multiplier stack consolidation (D10 + D29 + B06).

Mac-runnable (numpy/pandas + the Wave A two-stub base_loop import pattern).
Pins:
  A. DERISK_STACK_V2 default OFF + the B06 CRYPTO_RV_* constants.
  B. macro_indicators.vix_tier_mult_v2 hysteresis + regime_family_mults_v2.
  C. volatility BTC trailing-RV state machine + persistence (tmp file).
  D. portfolio D29 outlier exclusion (default path byte-identical).
  E. base_loop legacy-vs-v2 sizing compositions (flag OFF byte-equality,
     monotonicity, crisis differentiation, hard floors in BOTH modes,
     single vol-target scope, v2 fail-open).
  F. scripts/sizing_cofire_report.py CLI over synthetic journals.
"""

import datetime as _dt
import json
import logging
import os
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# base_loop's only Mac-unimportable links are predict_now (joblib/torch)
# and trading_utils's `from dotenv import load_dotenv`. Stub exactly those
# two, import, then RESTORE sys.modules (same pattern as
# tests/test_c26_base_loop_functional.py — replicated, NOT imported).
_dv = types.ModuleType('dotenv'); _dv.load_dotenv = lambda *a, **k: None
_pn = types.ModuleType('predict_now')
_pn.load_models = lambda *a, **k: (None, None, {}, None)
sys.modules['dotenv'] = _dv
sys.modules['predict_now'] = _pn
try:
    import base_loop
finally:
    for _m in ('dotenv', 'predict_now', 'trading_utils',
               'base_loop', 'stock_loop', 'crypto_loop'):
        sys.modules.pop(_m, None)

import macro_indicators
import market_data
import portfolio
import strategy_config
import volatility
from types_mod import MacroRegime

TILT_MAX = strategy_config.TILT_MAX


# ---------------------------------------------------------------------------
# A. Flag + constants
# ---------------------------------------------------------------------------

def test_flag_default_off():
    assert strategy_config.DERISK_STACK_V2 is False


def test_crypto_rv_constants_b06_values():
    assert strategy_config.CRYPTO_RV_ENTER_HIGH_PCT == 80.0
    assert strategy_config.CRYPTO_RV_ENTER_CRISIS_PCT == 95.0
    assert strategy_config.CRYPTO_RV_EXIT_HIGH_PCT == 65.0
    assert strategy_config.CRYPTO_RV_EXIT_CRISIS_PCT == 90.0
    assert strategy_config.CRYPTO_RV_EXIT_HOLD_EVALS == 12
    assert strategy_config.CRYPTO_RV_MIN_HISTORY_DAYS == 90
    assert strategy_config.CRYPTO_RV_HIGH_MULT == 0.5
    assert strategy_config.CRYPTO_RV_CRISIS_MULT == 0.3


# ---------------------------------------------------------------------------
# B. vix_tier_mult_v2 + regime_family_mults_v2
# ---------------------------------------------------------------------------

@pytest.fixture
def vix_state():
    macro_indicators._reset_vix_tier_state()
    yield
    macro_indicators._reset_vix_tier_state()


def test_vix_tier_mapping_modal_neutral(vix_state):
    assert macro_indicators.vix_tier_mult_v2(18.0) == 1.0   # modal -> 1.0 (c)
    macro_indicators._reset_vix_tier_state()
    assert macro_indicators.vix_tier_mult_v2(26.0) == 0.5
    macro_indicators._reset_vix_tier_state()
    assert macro_indicators.vix_tier_mult_v2(36.0) == 0.3


def test_vix_tier_hysteresis_sequence(vix_state):
    f = macro_indicators.vix_tier_mult_v2
    assert f(24.0) == 1.0
    assert f(26.0) == 0.5      # enter defensive
    assert f(23.0) == 0.5      # holds above exit 22
    assert f(21.0) == 1.0      # exits below 22
    assert f(36.0) == 0.3      # enter crisis
    assert f(32.0) == 0.3      # holds above exit 31
    assert f(30.0) == 0.5      # crisis -> defensive
    assert f(21.0) == 1.0      # cascading two-tier drop in one call
    macro_indicators._reset_vix_tier_state()
    assert f(21.0) == 1.0
    assert f(36.0) == 0.3      # jump straight to crisis in one call


def test_vix_tier_none_fails_open_state_untouched(vix_state):
    f = macro_indicators.vix_tier_mult_v2
    assert f(26.0) == 0.5
    assert f(None) == 1.0                                    # fail-open
    assert macro_indicators._vix_tier_state['tier'] == 1     # unmutated
    assert f(23.0) == 0.5                                    # state survived


def _regime(stress=0.0, vix=18.0, sizing=1.0):
    return MacroRegime(stress_level=stress, vix=vix, cape=None,
                       regime_label='x', sizing_mult=sizing, stop_mult=1.0)


def test_family_stock_has_vix_and_stress(vix_state):
    fam = macro_indicators.regime_family_mults_v2(_regime(vix=18.0), 'stock')
    assert fam == {'vix': 1.0, 'stress': 1.0}
    macro_indicators._reset_vix_tier_state()
    fam = macro_indicators.regime_family_mults_v2(
        _regime(stress=1.5, vix=27.0), 'stock')
    assert fam == {'vix': 0.5, 'stress': 0.5}


def test_family_crypto_stress_only(vix_state):
    fam = macro_indicators.regime_family_mults_v2(_regime(vix=40.0), 'crypto')
    assert fam == {'stress': 1.0}
    fam = macro_indicators.regime_family_mults_v2(
        _regime(stress=1.5, vix=40.0), 'crypto')
    assert fam == {'stress': 0.5}


def test_family_stress_boundary_and_none(vix_state):
    fam = macro_indicators.regime_family_mults_v2(_regime(stress=1.0), 'crypto')
    assert fam['stress'] == 1.0        # strictly > 1.0 fires (legacy rule)
    fam = macro_indicators.regime_family_mults_v2(
        MacroRegime(stress_level=None, vix=None, cape=None,
                    regime_label='x'), 'crypto')
    assert fam['stress'] == 1.0


def test_family_none_regime_and_cape_never_present(vix_state):
    assert macro_indicators.regime_family_mults_v2(None, 'stock') == {}
    fam = macro_indicators.regime_family_mults_v2(_regime(), 'stock')
    assert 'cape' not in fam


def test_family_cape_exclusion_announced_once(vix_state, monkeypatch, caplog):
    monkeypatch.setattr(macro_indicators, '_cape_exclusion_logged', False)
    with caplog.at_level(logging.WARNING, logger='macro_indicators'):
        macro_indicators.regime_family_mults_v2(_regime(), 'stock',
                                                announce=True)
        macro_indicators.regime_family_mults_v2(_regime(), 'stock',
                                                announce=True)
    hits = [r for r in caplog.records if 'pseudo-CAPE' in r.getMessage()]
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# C. BTC trailing-RV state (volatility)
# ---------------------------------------------------------------------------

@pytest.fixture
def rv_env(tmp_path, monkeypatch):
    monkeypatch.setattr(volatility, '_CRYPTO_RV_FILE',
                        str(tmp_path / 'crypto_rv_history.json'))
    volatility._reset_crypto_rv_state()
    yield tmp_path
    volatility._reset_crypto_rv_state()


_DAY0 = pd.Timestamp('2025-01-01')


def _bars(day_ranges):
    """day_ranges: iterable of (day_offset, per-bar log-range, n_hours)."""
    idx, hi, lo = [], [], []
    for d, a, hours in day_ranges:
        for h in range(hours):
            idx.append(_DAY0 + pd.Timedelta(days=d, hours=h))
            hi.append(100.0 * np.exp(a / 2.0))
            lo.append(100.0 * np.exp(-a / 2.0))
    return pd.DataFrame({'High': hi, 'Low': lo},
                        index=pd.DatetimeIndex(idx))


def _a_for_pctile(p):
    """Log-range whose daily RRV sits at ~percentile p of the seeded
    history (seed a_d linear 0.001..0.01 over merged days 0..364)."""
    return 0.001 + 0.009 * (p / 100.0)


def _seed_frame():
    # days 0..364 with ascending ranges (merged); day 365 calm (current)
    rows = [(d, 0.001 + 0.009 * d / 364.0, 24) for d in range(365)]
    rows.append((365, 0.001, 24))
    return _bars(rows)


def _eval_frame(day, a):
    """One complete calm day (day-1) + 24 bars of range `a` on `day`."""
    return _bars([(day - 1, 0.001, 24), (day, a, 24)])


def test_rv_short_history_unknown_fails_open(rv_env):
    frame = _bars([(d, 0.002, 24) for d in range(10)])
    volatility.update_crypto_rv_state('BTC/USD', frame)
    assert volatility._crypto_rv['state'] == 'unknown'
    mult, state, pct = volatility.get_crypto_rv_mult()
    assert mult == 1.0
    assert state == 'unknown'
    assert pct is None


def test_rv_seed_normal_then_crisis(rv_env):
    volatility.update_crypto_rv_state('BTC/USD', _seed_frame())
    assert len(volatility._crypto_rv['history']) == 365
    mult, state, pct = volatility.get_crypto_rv_mult()
    assert (mult, state) == (1.0, 'normal')
    # 24 huge-range bars -> pctile ~100 > 95 -> crisis (enter is immediate)
    volatility.update_crypto_rv_state('BTC/USD', _eval_frame(366, 0.05))
    mult, state, pct = volatility.get_crypto_rv_mult()
    assert (mult, state) == (0.3, 'crisis')
    assert pct > 95


def test_rv_crisis_to_high_immediate_then_slow_normal(rv_env):
    volatility.update_crypto_rv_state('BTC/USD', _seed_frame())
    volatility.update_crypto_rv_state('BTC/USD', _eval_frame(366, 0.05))
    assert volatility._crypto_rv['state'] == 'crisis'
    day = 367
    calm = _a_for_pctile(30)    # ~30th pctile: < exit_crisis 90, < exit_high 65
    # (3) crisis -> high on the first calm NEW bar (pctile < 90)
    volatility.update_crypto_rv_state('BTC/USD', _eval_frame(day, calm))
    mult, state, _ = volatility.get_crypto_rv_mult()
    assert (mult, state) == (0.5, 'high')
    # (4) high -> normal only after 12 calm NEW-bar evaluations
    for i in range(5):
        day += 1
        volatility.update_crypto_rv_state('BTC/USD', _eval_frame(day, calm))
    assert volatility._crypto_rv['state'] == 'high'
    assert volatility._crypto_rv['exit_count'] == 5
    # same-timestamp re-eval does not count
    volatility.update_crypto_rv_state('BTC/USD', _eval_frame(day, calm))
    assert volatility._crypto_rv['exit_count'] == 5
    # one intervening pctile >= 65 resets the counter
    day += 1
    volatility.update_crypto_rv_state(
        'BTC/USD', _eval_frame(day, _a_for_pctile(72)))
    assert volatility._crypto_rv['state'] == 'high'
    assert volatility._crypto_rv['exit_count'] == 0
    for i in range(11):
        day += 1
        volatility.update_crypto_rv_state('BTC/USD', _eval_frame(day, calm))
    assert volatility._crypto_rv['state'] == 'high'   # 11 evals: not yet
    assert volatility._crypto_rv['exit_count'] == 11
    day += 1
    volatility.update_crypto_rv_state('BTC/USD', _eval_frame(day, calm))
    mult, state, _ = volatility.get_crypto_rv_mult()
    assert (mult, state) == (1.0, 'normal')           # 12th: released
    assert volatility._crypto_rv['exit_count'] == 0


def test_rv_persistence_roundtrip(rv_env):
    volatility.update_crypto_rv_state('BTC/USD', _seed_frame())
    volatility.update_crypto_rv_state('BTC/USD', _eval_frame(366, 0.05))
    assert volatility._crypto_rv['state'] == 'crisis'
    n_hist = len(volatility._crypto_rv['history'])
    volatility._reset_crypto_rv_state()
    assert volatility._crypto_rv['history'] is None
    # Same frame again (same last bar ts -> not a new bar): the reloaded
    # state must survive — enters don't fire (calm current would not
    # enter), exits need a NEW bar.
    volatility.update_crypto_rv_state('BTC/USD', _eval_frame(366, 0.05))
    assert volatility._crypto_rv['state'] == 'crisis'
    assert len(volatility._crypto_rv['history']) == n_hist


def test_rv_non_source_symbol_noop(rv_env):
    volatility.update_crypto_rv_state('ETH/USD', _seed_frame())
    assert volatility._crypto_rv['history'] is None
    assert volatility._crypto_rv['state'] == 'unknown'
    assert not os.path.exists(volatility._CRYPTO_RV_FILE)


def test_rv_corrupt_file_fresh_start_no_raise(rv_env):
    with open(volatility._CRYPTO_RV_FILE, 'w') as f:
        f.write('not json{')
    frame = _bars([(d, 0.002, 24) for d in range(10)])
    volatility.update_crypto_rv_state('BTC/USD', frame)   # must not raise
    assert volatility._crypto_rv['state'] == 'unknown'


def test_rv_mult_stale_fails_open(rv_env, monkeypatch):
    volatility.update_crypto_rv_state('BTC/USD', _seed_frame())
    volatility.update_crypto_rv_state('BTC/USD', _eval_frame(366, 0.05))
    assert volatility.get_crypto_rv_mult()[0] == 0.3
    monkeypatch.setattr(
        volatility.time, 'monotonic',
        lambda: volatility._crypto_rv['updated_mono']
        + volatility._CRYPTO_RV_STALE_SEC + 1)
    mult, state, _ = volatility.get_crypto_rv_mult()
    assert (mult, state) == (1.0, 'stale')


# ---------------------------------------------------------------------------
# D. portfolio D29 — outlier exclusion from the book-vol EWMA
# ---------------------------------------------------------------------------

def _alternating_curve(n=70, r=0.0315, spike_at=None, spike=0.40):
    eq = [100_000.0]
    for i in range(n):
        step = spike if (spike_at is not None and i == spike_at) else (
            r if i % 2 == 0 else -r)
        eq.append(eq[-1] * (1.0 + step))
    return eq


def _legacy_ref(curve, lam=portfolio.EWMA_LAMBDA):
    eq = np.asarray([e for e in curve if e], dtype=float)
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    var = float(np.var(rets))
    for x in rets:
        var = lam * var + (1.0 - lam) * x * x
    return float(np.sqrt(var * 252))


def test_d29_default_path_pinned_byte_identical():
    curve = _alternating_curve(spike_at=60)
    v_default = portfolio.ewma_annualized_vol(curve)
    v_explicit = portfolio.ewma_annualized_vol(curve, exclude_outliers=False)
    assert v_default == v_explicit == pytest.approx(_legacy_ref(curve))


def test_d29_exclusion_drops_deposit_day():
    curve = _alternating_curve(spike_at=60)
    v_off, diag_off = portfolio._ewma_vol_diag(curve)
    v_on, diag_on = portfolio._ewma_vol_diag(curve, exclude_outliers=True)
    assert v_on < v_off
    assert diag_on['n_excluded'] == 1
    assert 'n_excluded' not in diag_off
    # the always-on contamination diagnostics keep seeing the FULL series
    assert diag_on['n_outliers'] == 1
    assert diag_on['max_abs_ret'] == pytest.approx(0.40)


def test_d29_finite_positive_filter_matches_prefiltered():
    clean = _alternating_curve(n=40)
    dirty = list(clean)
    dirty.insert(20, float('nan'))
    dirty.insert(30, -5.0)
    v_dirty, diag = portfolio._ewma_vol_diag(dirty, exclude_outliers=True)
    v_clean, _ = portfolio._ewma_vol_diag(clean, exclude_outliers=True)
    assert v_dirty == pytest.approx(v_clean)
    assert np.isfinite(v_dirty)


def test_d29_book_scalar_uses_flag_and_warning_always_fires(
        monkeypatch, caplog):
    curve = _alternating_curve(spike_at=60)
    api = types.SimpleNamespace(
        get_portfolio_history=lambda period, timeframe:
        types.SimpleNamespace(equity=curve, timestamp=None))
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', False)
    portfolio._book_vol_cache.clear()
    with caplog.at_level(logging.WARNING, logger='portfolio'):
        scalar_off = portfolio.get_book_vol_scalar_cached(api, 'crypto')
    assert any('deposit/withdrawal' in r.getMessage()
               for r in caplog.records)
    caplog.clear()
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', True)
    portfolio._book_vol_cache.clear()
    with caplog.at_level(logging.WARNING, logger='portfolio'):
        scalar_on = portfolio.get_book_vol_scalar_cached(api, 'crypto')
    warns = [r.getMessage() for r in caplog.records
             if 'deposit/withdrawal' in r.getMessage()]
    assert len(warns) == 1                      # warning ALWAYS fires
    assert 'excluded from the EWMA recursion' in warns[0]
    portfolio._book_vol_cache.clear()
    # deposit spike no longer pins the scalar at its floor
    assert scalar_off == 0.5
    assert scalar_on > scalar_off
    assert scalar_on == pytest.approx(0.7, abs=0.05)


# ---------------------------------------------------------------------------
# E. base_loop compositions (real _compute_position_size)
# ---------------------------------------------------------------------------

class _Loop(base_loop.BaseTradingLoop):
    MODEL_PREFIX = ''

    def get_symbol_universe(self):
        return ['BTC/USD']

    def check_market_hours(self):
        return True

    def get_asset_type(self):
        return 'crypto'

    def get_quote(self, symbol):
        return {'midpoint': 100.0, 'spread_pct': 0.02}

    def place_buy_order(self, *a, **k):
        return None

    def place_sell_order(self, *a, **k):
        return None

    def get_benchmark_close(self):
        return None

    def get_headlines(self, symbol):
        return []

    def flatten_before_close(self):
        pass

    def write_prediction_cache(self, preds, **kwargs):
        pass


def _mk(**over):
    inst = object.__new__(_Loop)
    inst.api = None
    inst.trade_threshold = 0.15
    inst.positions = {}
    inst.macro_regime = None
    inst.corr_matrix = {}
    inst._equity = 100_000.0
    inst._peak_equity = 100_000.0
    inst._leveraged_etfs = {}
    inst._last_sizing_detail = None
    for k, v in over.items():
        setattr(inst, k, v)
    return inst


def _macro(vix=12.0, sizing=1.0, stress=0.0):
    return MacroRegime(stress_level=stress, vix=vix, cape=None,
                       regime_label='x', sizing_mult=sizing, stop_mult=1.0)


def _bars_150():
    return pd.DataFrame({'Close': np.linspace(100.0, 115.0, 150)})


def _seams(monkeypatch, kelly=0.125, bars='full', bookvol=1.0, sigma=None,
           rv=(1.0, 'normal', None)):
    monkeypatch.setattr(market_data, 'get_live_atr', lambda *a, **k: None)
    monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                        lambda *a, **k: (_bars_150() if bars == 'full'
                                         else None))
    monkeypatch.setattr(base_loop, 'compute_kelly_fraction',
                        lambda *a, **k: kelly)
    monkeypatch.setattr(portfolio, 'get_book_vol_scalar_cached',
                        lambda api, at: bookvol)
    monkeypatch.setattr(volatility, 'get_sigma', lambda *a, **k: sigma)
    monkeypatch.setattr(volatility, 'get_crypto_rv_mult', lambda: rv)


_QUOTE = {'midpoint': 100.0, 'spread_pct': 0.02}


def _full_info_inst(**over):
    kw = dict(macro_regime=_macro(), corr_matrix={'ETH/USD': {}})
    kw.update(over)
    return _mk(**kw)


def test_off_modal_byte_identical_with_v2_shadow(monkeypatch):
    inst = _full_info_inst(macro_regime=_macro(vix=18.0, sizing=0.8))
    _seams(monkeypatch)
    result = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE))
    assert result == 559           # legacy int(1000 * (0.7*0.8)) = 559 (fp)
    d = inst._last_sizing_detail
    assert d['stack'] == 'legacy'
    assert d['vix_tilt'] == 0.7
    assert d['macro_mult'] == 0.8
    assert d['v2']['tilt'] == 1.0              # v2 shadow: no cut at modal VIX
    assert d['v2']['family'] == {'stress': 1.0, 'btc_rv': 1.0, 'bookvol': 1.0}
    assert d['v2']['btc_rv_state'] == 'normal'


def test_off_full_info_expectations_unchanged(monkeypatch):
    _seams(monkeypatch)
    inst = _full_info_inst()
    assert inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE)) == 1000
    assert inst._last_sizing_detail['v2']['tilt'] == 1.0
    inst = _full_info_inst()
    assert inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE),
                                       llm_mult=1.5) == 1300
    d = inst._last_sizing_detail
    assert d['tilt'] == TILT_MAX
    assert d['v2']['tilt'] == TILT_MAX         # boost clamp shared
    inst = _full_info_inst()
    assert inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE),
                                       sentiment_mult=0.15,
                                       llm_mult=0.65) == 100
    d = inst._last_sizing_detail
    assert d['v2']['tilt'] == 0.1              # 0.1 floor applies in v2 too


def test_on_modal_sizes_at_full(monkeypatch):
    _seams(monkeypatch)
    inst = _full_info_inst(macro_regime=_macro(vix=18.0, sizing=0.8))
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', True)
    result = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE))
    assert result == 1000                      # >= the 559 legacy cut
    d = inst._last_sizing_detail
    assert d['stack'] == 'v2'
    assert d['tilt'] == 0.56                   # legacy still journaled


@pytest.mark.parametrize('vix', [12.0, 18.0, 30.0, 40.0])
@pytest.mark.parametrize('bookvol', [0.6, 1.0])
@pytest.mark.parametrize('meta', [0.5, 1.0])
def test_v2_monotone_vs_legacy_in_calm_vix(monkeypatch, vix, bookvol, meta):
    _seams(monkeypatch, bookvol=bookvol)
    inst = _full_info_inst(macro_regime=_macro(vix=vix))
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', False)
    off = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE),
                                      meta_mult=meta)
    inst = _full_info_inst(macro_regime=_macro(vix=vix))
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', True)
    on = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE),
                                     meta_mult=meta)
    if vix <= 25.0:            # modal/normal tier
        assert on >= off


def test_on_crisis_differentiates_meta(monkeypatch):
    for meta, want_off, want_on in ((0.5, 100, 150), (1.0, 100, 300)):
        _seams(monkeypatch, rv=(0.3, 'crisis', 97.5))
        inst = _full_info_inst(macro_regime=_macro(vix=40.0, sizing=0.3))
        monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', False)
        off = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE),
                                          meta_mult=meta)
        assert off == want_off                 # floor-saturated: no gradient
        inst = _full_info_inst(macro_regime=_macro(vix=40.0, sizing=0.3))
        monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', True)
        on = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE),
                                         meta_mult=meta)
        assert on == want_on                   # min=0.3 preserves the gradient
        assert inst._last_sizing_detail['v2']['min_src'] == 'btc_rv'


class _Sentinel:
    def __getattr__(self, name):
        raise AssertionError(f'api touched ({name}) despite emergency zero')


@pytest.mark.parametrize('flag', [False, True])
def test_d26_emergency_zero_both_modes(monkeypatch, flag):
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', flag)
    inst = _mk(macro_regime=_macro(sizing=0.0), api=_Sentinel())
    assert inst._compute_position_size('BTC/USD', 0.5,
                                       {'midpoint': 100.0}) == 0


@pytest.mark.parametrize('flag', [False, True])
def test_dust_returns_zero_both_modes(monkeypatch, flag):
    _seams(monkeypatch, kelly=None, bars='none', rv=(1.0, 'stale', None))
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', flag)
    inst = _mk()
    inst.NOTIONAL_PER_SYMBOL = 150
    assert inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE)) == 0


@pytest.mark.parametrize('flag', [False, True])
def test_degraded_clamp_caps_both_tilts(monkeypatch, flag):
    _seams(monkeypatch, kelly=None, bars='none', rv=(1.0, 'stale', None))
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', flag)
    inst = _mk()                               # 4 missing advisory inputs
    result = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE))
    assert result == 500                       # 1000 * 0.5 in BOTH modes
    d = inst._last_sizing_detail
    assert d['degraded_inputs'] == 4
    assert d['v2']['degraded'] == 4
    assert d['v2']['tilt'] == 0.5


def test_on_single_vol_target_scope(monkeypatch):
    # sigma small enough that the per-position ratio clamps at 1.5x
    _seams(monkeypatch, sigma=0.001)
    inst = _full_info_inst(macro_regime=_macro(vix=18.0, sizing=0.8))
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', False)
    off = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE))
    assert off == 839              # int(1000*1.5*(0.7*0.8)) — double count
    assert inst._last_sizing_detail['vol_mult'] == 1.5
    inst = _full_info_inst(macro_regime=_macro(vix=18.0, sizing=0.8))
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', True)
    on = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE))
    assert on == 1000                          # vol_mult composes at 1.0
    assert inst._last_sizing_detail['vol_mult'] == 1.5   # still journaled


def test_on_v2_failure_falls_back_to_legacy(monkeypatch, caplog):
    _seams(monkeypatch)
    inst = _full_info_inst(macro_regime=_macro(vix=18.0, sizing=0.8))
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', True)
    monkeypatch.setattr(
        macro_indicators, 'regime_family_mults_v2',
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    with caplog.at_level(logging.WARNING, logger='base_loop'):
        result = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE))
    assert result == 559                       # legacy product retained
    d = inst._last_sizing_detail
    assert d['stack'] == 'legacy'
    assert any('legacy product retained' in r.getMessage()
               for r in caplog.records)


class _StockLoop(_Loop):
    def get_asset_type(self):
        return 'stock'


def test_off_stock_book_v2_shadow_tier_map_no_btc_rv(monkeypatch, caplog,
                                                     vix_state):
    """Stock book, flag OFF: v2 family uses the ONE tier map (no btc_rv),
    get_crypto_rv_mult is never touched, and the CAPE-exclusion announce
    stays SILENT in shadow mode (it fires only when the flag is ON)."""
    _seams(monkeypatch)
    monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                        lambda *a, **k: _bars_150())
    # If the stock path ever called this, the v2 try would abort before
    # v2['family'] is set — asserted below.
    monkeypatch.setattr(volatility, 'get_crypto_rv_mult',
                        lambda: (_ for _ in ()).throw(
                            AssertionError('crypto RV read on stock book')))
    monkeypatch.setattr(macro_indicators, '_cape_exclusion_logged', False)
    inst = object.__new__(_StockLoop)
    for k, v in dict(api=None, trade_threshold=0.15, positions={},
                     macro_regime=_macro(vix=26.0), corr_matrix={'MSFT': {}},
                     _equity=100_000.0, _peak_equity=100_000.0,
                     _leveraged_etfs={}, _last_sizing_detail=None).items():
        setattr(inst, k, v)
    with caplog.at_level(logging.WARNING, logger='macro_indicators'):
        result = inst._compute_position_size('NVDA', 0.15, dict(_QUOTE))
    assert result == 500                       # legacy: vix 26 -> 0.5 tilt
    d = inst._last_sizing_detail
    assert d['stack'] == 'legacy'
    assert d['vix_tilt'] == 0.5
    assert d['v2']['family'] == {'vix': 0.5, 'stress': 1.0, 'bookvol': 1.0}
    assert d['v2']['min_src'] == 'vix'
    assert 'btc_rv_state' not in d['v2']
    assert not any('pseudo-CAPE' in r.getMessage() for r in caplog.records)


def test_on_activation_warning_logged_once(monkeypatch, caplog):
    _seams(monkeypatch)
    monkeypatch.setattr(strategy_config, 'DERISK_STACK_V2', True)
    monkeypatch.setattr(base_loop, '_derisk_v2_logged', False)
    with caplog.at_level(logging.WARNING, logger='base_loop'):
        for _ in range(2):
            inst = _full_info_inst()
            inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE))
    hits = [r for r in caplog.records
            if '[DERISK-V2] ACTIVE' in r.getMessage()]
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# F. scripts/sizing_cofire_report.py (subprocess)
# ---------------------------------------------------------------------------

_SCRIPT = str(REPO / 'scripts' / 'sizing_cofire_report.py')


def _run_report(journal_dir, *extra):
    proc = subprocess.run(
        [sys.executable, _SCRIPT, '--journal-dir', str(journal_dir),
         '--days', '30', *extra],
        capture_output=True, text=True, cwd=str(REPO))
    return proc


def _ts(minute):
    return (_dt.datetime.now(_dt.timezone.utc)
            - _dt.timedelta(hours=1)
            + _dt.timedelta(minutes=minute)).isoformat()


def _write_journal(tmp_path):
    base = {'kelly_mult': 1.0, 'vol_mult': 1.0, 'dd_mult': 1.0,
            'sentiment_mult': 1.0, 'llm_mult': 1.0, 'meta_mult': 1.0}
    rows = [
        # A: modal co-fire (vix ladder + macro both < 1)
        {'ts': _ts(0), 'action': 'buy', 'symbol': 'BTC/USD',
         'sizing': {**base, 'vix_tilt': 0.7, 'macro_mult': 0.8,
                    'tilt_raw': 0.56, 'tilt': 0.56}},
        # B: floor-saturated crisis row (worst product)
        {'ts': _ts(1), 'action': 'buy', 'symbol': 'BTC/USD',
         'sizing': {**base, 'vix_tilt': 0.3, 'macro_mult': 0.3,
                    'llm_mult': 0.5, 'tilt_raw': 0.045, 'tilt': 0.1}},
        # C/D: crypto v2-shadow rows (one btc_rv_state flip)
        {'ts': _ts(2), 'action': 'buy', 'symbol': 'BTC/USD',
         'sizing': {**base, 'meta_mult': 0.5, 'tilt_raw': 0.56, 'tilt': 0.56,
                    'stack': 'legacy',
                    'v2': {'family': {'stress': 1.0, 'btc_rv': 1.0,
                                      'bookvol': 1.0},
                           'f_regime_min': 1.0, 'min_src': 'stress',
                           'tilt_raw': 1.0, 'tilt': 1.0,
                           'btc_rv_state': 'normal'}}},
        {'ts': _ts(3), 'action': 'buy', 'symbol': 'BTC/USD',
         'sizing': {**base, 'tilt_raw': 0.5, 'tilt': 0.5, 'stack': 'legacy',
                    'v2': {'family': {'stress': 1.0, 'btc_rv': 0.5,
                                      'bookvol': 1.0},
                           'f_regime_min': 0.5, 'min_src': 'btc_rv',
                           'tilt_raw': 0.5, 'tilt': 0.5,
                           'btc_rv_state': 'high'}}},
        # E/F: stock v2-shadow rows (one vix-tier flip)
        {'ts': _ts(4), 'action': 'buy', 'symbol': 'NVDA',
         'sizing': {**base, 'vix_tilt': 0.7, 'tilt_raw': 0.7, 'tilt': 0.7,
                    'stack': 'legacy',
                    'v2': {'family': {'vix': 1.0, 'stress': 1.0,
                                      'bookvol': 1.0},
                           'f_regime_min': 1.0, 'min_src': 'vix',
                           'tilt_raw': 1.0, 'tilt': 1.0}}},
        {'ts': _ts(5), 'action': 'buy', 'symbol': 'NVDA',
         'sizing': {**base, 'vix_tilt': 0.5, 'tilt_raw': 0.35, 'tilt': 0.35,
                    'stack': 'legacy',
                    'v2': {'family': {'vix': 0.5, 'stress': 1.0,
                                      'bookvol': 1.0},
                           'f_regime_min': 0.5, 'min_src': 'vix',
                           'tilt_raw': 0.5, 'tilt': 0.5}}},
        # skip row + malformed line
        {'ts': _ts(6), 'action': 'skip', 'symbol': 'BTC/USD',
         'skip_reason': 'sizing_zero'},
    ]
    path = tmp_path / f'{_dt.date.today().isoformat()}.jsonl'
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
        f.write('not json{\n')
    return path


def test_report_json_sections(tmp_path):
    _write_journal(tmp_path)
    proc = _run_report(tmp_path, '--json')
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(proc.stdout)
    assert rep['n_buy_rows'] == 6
    assert rep['n_malformed_lines'] == 1
    # 1. per-multiplier
    vt = rep['per_multiplier']['vix_tilt']
    assert vt['n_present'] == 4
    assert vt['fire_rate'] == 1.0
    assert vt['min'] == 0.3
    # 2. bind rates
    assert rep['bind_rates']['floor_0_1'] == 1
    assert rep['bind_rates']['sizing_zero_skips'] == 1
    assert rep['bind_rates']['degraded'] == 0
    # 3. co-fire
    assert rep['cofire']['pairs']['vix_tilt&macro_mult']['count'] == 2
    assert rep['cofire']['p_fires_given_floor']['vix_tilt'] == 1.0
    # 4. worst product
    assert rep['worst_product']['symbol'] == 'BTC/USD'
    assert rep['worst_product']['sizing']['tilt_raw'] == 0.045
    # 5. marginal effect: llm 0.5 on the floored row is fully absorbed
    #    (0.045/0.5 -> floored 0.1; 0.1/0.1 = 1.0); meta 0.5 on the
    #    unfloored row has full effect (0.56 / (0.56/0.5) = 0.5)
    assert rep['marginal_effect']['llm_mult']['median_effect'] == 1.0
    assert rep['marginal_effect']['meta_mult']['median_effect'] == 0.5
    # 6. v2 shadow
    v2 = rep['v2_shadow']
    assert v2['n_rows'] == 4
    assert v2['stacks_applied'] == {'legacy': 4, 'v2': 0}
    assert v2['min_src_histogram'] == {'stress': 1, 'btc_rv': 1, 'vix': 2}
    assert v2['floor_saturation'] == {'legacy': 0, 'v2': 0}
    assert v2['flip_counts']['btc_rv_state_total'] == 1
    assert v2['flip_counts']['vix_tier_total'] == 1
    ratios = sorted([1.0 / 0.56, 0.5 / 0.5, 1.0 / 0.7, 0.5 / 0.35])
    expected_median = (ratios[1] + ratios[2]) / 2.0
    assert v2['median_v2_over_legacy'] == pytest.approx(expected_median,
                                                        abs=1e-3)


def test_report_book_filter(tmp_path):
    _write_journal(tmp_path)
    proc = _run_report(tmp_path, '--json', '--book', 'stock')
    rep = json.loads(proc.stdout)
    assert rep['n_buy_rows'] == 2
    assert rep['v2_shadow']['flip_counts']['btc_rv_state_total'] == 0


def test_report_empty_dir_clean_exit(tmp_path):
    proc = _run_report(tmp_path / 'missing')
    assert proc.returncode == 0
    assert 'no rows' in proc.stdout


def test_report_human_output_runs(tmp_path):
    _write_journal(tmp_path)
    proc = _run_report(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert 'per-multiplier' in proc.stdout
    assert 'v2 shadow' in proc.stdout
