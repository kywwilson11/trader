"""ENGINE-R2 (2026-09 Jetson campaign) — two default-OFF flag wirings.

1. O8 — strategy_config.CRYPTO_QUOTE_MAX_AGE_SEC (default None): the crypto
   quote-staleness limit in order_utils.get_quote. None -> the legacy literal
   180 s (byte-identical); a number -> that many seconds for CRYPTO only.
   Read at call time (order_utils._quote_max_age_sec).
2. BARS_PER_YEAR lockstep — the EXISTING flag strategy_config.
   BARS_PER_YEAR_MEASURED now also moves volatility.py's two read sites
   (compute_vol_adjusted_size per-bar target; HAR daily -> per-bar sigma)
   via bars_calendar.bars_per_year / bars_per_day. OFF byte-identical.
"""
import datetime
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import order_utils                     # noqa: E402
import volatility                      # noqa: E402
import bars_calendar                   # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LEGACY_BPY = {'crypto': 8760, 'stock': 1638}
LEGACY_BPD = {'crypto': 24.0, 'stock': 6.5}
UTC = datetime.timezone.utc


# =====================================================================
# 1. CRYPTO_QUOTE_MAX_AGE_SEC
# =====================================================================

class _Api:
    """Stub SDK: a quote whose feed timestamp is `age` seconds old."""

    def __init__(self, age):
        self.age = age

    def _q(self):
        t = datetime.datetime.now(UTC) - datetime.timedelta(seconds=self.age)
        return SimpleNamespace(bp=100.0, ap=100.1, t=t)

    def get_latest_crypto_quotes(self, symbols):
        return {symbols[0]: self._q()}

    def get_latest_quote(self, symbol):
        return self._q()


def _set_age_flag(monkeypatch, value):
    import strategy_config
    if value == 'absent':
        monkeypatch.delattr(strategy_config, 'CRYPTO_QUOTE_MAX_AGE_SEC',
                            raising=False)
    else:
        monkeypatch.setattr(strategy_config, 'CRYPTO_QUOTE_MAX_AGE_SEC',
                            value, raising=False)


def test_flag_default_is_none():
    import strategy_config
    assert strategy_config.CRYPTO_QUOTE_MAX_AGE_SEC is None


# Ages 179/181/299/301 s: the legacy rule flips at 180, the ON(300) rule at 300.
OFF_EXPECT = {179: False, 181: True, 299: True, 301: True}      # is None?
ON300_EXPECT = {179: False, 181: False, 299: False, 301: True}


@pytest.mark.parametrize('state', ['absent', None])
@pytest.mark.parametrize('age', sorted(OFF_EXPECT))
def test_off_real_get_crypto_quote_legacy_180(monkeypatch, state, age):
    _set_age_flag(monkeypatch, state)
    out = order_utils.get_crypto_quote(_Api(age), 'ETH/USD')
    assert (out is None) is OFF_EXPECT[age]


@pytest.mark.parametrize('state', ['absent', None])
def test_off_threshold_is_the_legacy_int_literal(monkeypatch, state):
    _set_age_flag(monkeypatch, state)
    for asset in ('crypto', 'stock'):
        v = order_utils._quote_max_age_sec(asset)
        assert v == 180 and type(v) is int


@pytest.mark.parametrize('age', sorted(ON300_EXPECT))
def test_on_300_real_get_crypto_quote(monkeypatch, age):
    _set_age_flag(monkeypatch, 300)
    out = order_utils.get_crypto_quote(_Api(age), 'ETH/USD')
    assert (out is None) is ON300_EXPECT[age]
    if out is not None:
        assert out['bid'] == 100.0 and out['ask'] == 100.1


@pytest.mark.parametrize('age', sorted(OFF_EXPECT))
def test_on_flag_leaves_stocks_at_180(monkeypatch, age):
    _set_age_flag(monkeypatch, 300)
    out = order_utils.get_stock_quote(_Api(age), 'AAPL')
    assert (out is None) is OFF_EXPECT[age]


def test_on_600_threshold(monkeypatch):
    _set_age_flag(monkeypatch, 600)
    assert order_utils.get_crypto_quote(_Api(486), 'SOL/USD') is not None
    assert order_utils.get_crypto_quote(_Api(601), 'SOL/USD') is None


@pytest.mark.parametrize('bad', ['abc', float('nan'), float('inf'), 0, -5,
                                 True, [300]])
def test_invalid_flag_value_falls_back_to_180(monkeypatch, bad):
    # Must never raise (it runs inside get_quote's staleness try-block,
    # where a raise would silently skip the age check = fail open).
    _set_age_flag(monkeypatch, bad)
    assert order_utils._quote_max_age_sec('crypto') == 180
    assert order_utils.get_crypto_quote(_Api(181), 'ETH/USD') is None
    assert order_utils.get_crypto_quote(_Api(179), 'ETH/USD') is not None


def test_numeric_string_flag_is_accepted(monkeypatch):
    _set_age_flag(monkeypatch, '300')
    assert order_utils._quote_max_age_sec('crypto') == 300.0
    assert order_utils.get_crypto_quote(_Api(299), 'ETH/USD') is not None


def test_flag_read_at_call_time(monkeypatch):
    api = _Api(250)
    _set_age_flag(monkeypatch, None)
    assert order_utils.get_crypto_quote(api, 'ETH/USD') is None
    _set_age_flag(monkeypatch, 300)
    assert order_utils.get_crypto_quote(api, 'ETH/USD') is not None
    _set_age_flag(monkeypatch, None)
    assert order_utils.get_crypto_quote(api, 'ETH/USD') is None


# =====================================================================
# 2a. bars_calendar.bars_per_day (new helper)
# =====================================================================

@pytest.fixture
def bpy_off(monkeypatch):
    import strategy_config
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', False,
                        raising=False)


@pytest.fixture
def bpy_on(monkeypatch):
    import strategy_config
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', True,
                        raising=False)


@pytest.mark.parametrize('state', ['absent', 'false'])
def test_bars_per_day_off(monkeypatch, state):
    import strategy_config
    if state == 'absent':
        monkeypatch.delattr(strategy_config, 'BARS_PER_YEAR_MEASURED',
                            raising=False)
    else:
        monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', False,
                            raising=False)
    bpd = bars_calendar.bars_per_day
    assert bpd('stock') == 6.5 and bpd('crypto') == 24.0
    assert bpd('fx') == 6.5 and bpd('fx', LEGACY_BPD, 1.0) == 1.0
    own = {'stock': 7}
    assert bpd('stock', own) == 7 and type(bpd('stock', own)) is int
    assert bars_calendar.LEGACY_BARS_PER_DAY == LEGACY_BPD


def test_bars_per_day_on(bpy_on):
    bpd = bars_calendar.bars_per_day
    assert bpd('stock') == 3827 / 252
    assert bpd('stock', LEGACY_BPD) == 3827 / 252
    assert abs(bpd('stock') - 15.19) < 0.01
    v = bpd('crypto', LEGACY_BPD)
    assert v == 24.0 and type(v) is float          # crypto unchanged
    assert bpd('fx', LEGACY_BPD, 6.5) == 6.5


@pytest.mark.parametrize('state', [False, True])
def test_per_day_times_days_equals_per_year(monkeypatch, state):
    import strategy_config
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', state,
                        raising=False)
    for asset, days in bars_calendar.TRADING_DAYS_PER_YEAR.items():
        assert (bars_calendar.bars_per_day(asset) * days
                == bars_calendar.bars_per_year(asset))


# =====================================================================
# 2b. volatility.py read sites under BARS_PER_YEAR_MEASURED
# =====================================================================

def _legacy_vol_adjusted(base, sigma, asset):
    """compute_vol_adjusted_size's pre-edit body, verbatim, literal table."""
    if not np.isfinite(sigma) or sigma <= 0:
        return base
    from strategy_config import PORTFOLIO_VOL_TARGET
    annual_target = PORTFOLIO_VOL_TARGET.get(asset, 0.25)
    target_per_bar = annual_target / np.sqrt(LEGACY_BPY.get(asset, 8760))
    ratio = target_per_bar / sigma
    ratio = max(0.5, min(1.5, ratio))
    return base * ratio


SIGMAS = [float(s) for s in np.geomspace(1e-4, 5e-2, 25)] + [0.0, -1.0,
                                                             float('nan')]


@pytest.mark.parametrize('state', ['absent', 'false'])
@pytest.mark.parametrize('asset', ['crypto', 'stock', 'fx'])
def test_vol_adjusted_off_byte_identical(monkeypatch, state, asset):
    import strategy_config
    if state == 'absent':
        monkeypatch.delattr(strategy_config, 'BARS_PER_YEAR_MEASURED',
                            raising=False)
    else:
        monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', False,
                            raising=False)
    for base in (1.0, 1234.5):
        for s in SIGMAS:
            got = volatility.compute_vol_adjusted_size(base, s, asset)
            exp = _legacy_vol_adjusted(base, s, asset)
            assert got == exp or (math.isnan(got) and math.isnan(exp)), (s,)


def test_vol_adjusted_on_stock_target_uses_measured(bpy_on):
    from strategy_config import PORTFOLIO_VOL_TARGET
    annual = PORTFOLIO_VOL_TARGET.get('stock', 0.25)
    target = annual / np.sqrt(3827)
    for s in SIGMAS[:25]:
        exp = max(0.5, min(1.5, target / s))
        assert volatility.compute_vol_adjusted_size(1.0, s, 'stock') == exp
    # crypto unchanged under ON
    for s in SIGMAS:
        got = volatility.compute_vol_adjusted_size(1.0, s, 'crypto')
        exp = _legacy_vol_adjusted(1.0, s, 'crypto')
        assert got == exp or (math.isnan(got) and math.isnan(exp))


def test_vol_adjusted_on_actually_moves_stock(monkeypatch):
    import strategy_config
    s = 0.004                                   # unclamped both ways
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', False,
                        raising=False)
    off = volatility.compute_vol_adjusted_size(1.0, s, 'stock')
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', True)
    on = volatility.compute_vol_adjusted_size(1.0, s, 'stock')
    assert 0.5 < on < off < 1.5
    assert on == pytest.approx(off * math.sqrt(1638 / 3827), rel=1e-12)


def _rrv_series(n_days, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n_days, freq='D')
    return pd.Series(np.exp(rng.normal(-9.0, 0.5, n_days)), index=idx)


def _hourly_bars(n_days, bars_per_day, seed):
    rng = np.random.default_rng(seed)
    idx, rows, px = [], [], 100.0
    for d in range(n_days):
        day = pd.Timestamp('2024-01-02', tz='UTC') + pd.Timedelta(days=d)
        for h in range(bars_per_day):
            o = px
            px = px * (1 + rng.normal(0, 0.004))
            rows.append((o, max(o, px) * (1 + abs(rng.normal(0, 0.002))),
                         min(o, px) * (1 - abs(rng.normal(0, 0.002))), px))
            idx.append(day + pd.Timedelta(hours=13 + h))
    return pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close'],
                        index=pd.DatetimeIndex(idx))


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_har_off_equals_import_fallback_legacy_expression(monkeypatch, seed):
    """OFF == the ImportError branch, which IS the pre-edit expression
    `BARS_PER_DAY.get(asset_type, 6.5)` on the legacy table."""
    import strategy_config
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', False,
                        raising=False)
    rrv = _rrv_series(200, seed)
    bars = _hourly_bars(90, 7, seed)
    off = {a: (volatility._har_sigma_from_rrv(rrv, a),
               volatility._har_sigma_from_rrv(rrv, a, shrink=True,
                                              c_scale=1.3),
               volatility.har_forecast_sigma(bars, a))
           for a in ('stock', 'crypto', 'fx')}
    monkeypatch.setitem(sys.modules, 'bars_calendar', None)  # -> ImportError
    for a in ('stock', 'crypto', 'fx'):
        fb = (volatility._har_sigma_from_rrv(rrv, a),
              volatility._har_sigma_from_rrv(rrv, a, shrink=True,
                                             c_scale=1.3),
              volatility.har_forecast_sigma(bars, a))
        assert fb == off[a] and None not in fb
        assert volatility._bars_per_day(a) == LEGACY_BPD.get(a, 6.5)
        assert volatility._bars_per_year(a) == LEGACY_BPY.get(a, 8760)


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_har_on_stock_uses_measured_bars_per_day(monkeypatch, seed):
    import strategy_config
    rrv = _rrv_series(200, seed)
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', False,
                        raising=False)
    off_s = volatility._har_sigma_from_rrv(rrv, 'stock')
    off_c = volatility._har_sigma_from_rrv(rrv, 'crypto')
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', True)
    on_s = volatility._har_sigma_from_rrv(rrv, 'stock')
    on_c = volatility._har_sigma_from_rrv(rrv, 'crypto')
    # sigma_daily is unchanged; only the per-bar divisor moves 6.5 -> 3827/252
    assert on_s == pytest.approx(off_s * math.sqrt(6.5 / (3827 / 252)),
                                 rel=1e-12)
    assert on_c == off_c                        # crypto byte-identical


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_lockstep_har_sizing_ratio_invariant(monkeypatch, seed):
    """With bpd = bpy / 252 the HAR-sourced vol ratio
    annual/sqrt(bpy) / (sigma_daily/sqrt(bpd)) = annual/(sqrt(252) sigma_daily)
    does not depend on the calendar: ONE flip rescales target and sigma
    together (only GARCH-sourced per-bar sigmas re-scale)."""
    import strategy_config
    rrv = _rrv_series(200, seed)
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', False,
                        raising=False)
    s_off = volatility._har_sigma_from_rrv(rrv, 'stock')
    r_off = volatility.compute_vol_adjusted_size(1.0, s_off, 'stock')
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', True)
    s_on = volatility._har_sigma_from_rrv(rrv, 'stock')
    r_on = volatility.compute_vol_adjusted_size(1.0, s_on, 'stock')
    assert 0.5 < r_off < 1.5                    # unclamped -> a real check
    assert r_on == pytest.approx(r_off, rel=1e-12)


def test_measured_numbers_have_one_home():
    src = (REPO / 'volatility.py').read_text()
    assert '3827' not in src and '15.18' not in src and '15.19' not in src
    # the legacy literal tables stay (sibling-copy source pins)
    assert "BARS_PER_YEAR = {'crypto': 8760, 'stock': 1638}" in src
    assert "BARS_PER_DAY = {'crypto': 24.0, 'stock': 6.5}" in src
    # both read sites route through the call-time helpers; the legacy
    # `.get` survives only as each helper's ImportError fallback
    assert src.count('BARS_PER_DAY.get(') == 1
    assert src.count('BARS_PER_YEAR.get(') == 1
    assert 'np.sqrt(_bars_per_day(asset_type))' in src
    assert 'np.sqrt(_bars_per_year(asset_type))' in src
