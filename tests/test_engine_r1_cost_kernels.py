"""ENGINE round-1 W2 (2026-09-27): cost-model property / invariant tests and
the G2-2 crypto price-tick design test.

Plain pytest with SEEDED random sweeps + edge grids (hypothesis is not
installed on the Jetson). Pure: no network, no journals (the maker-share
reader is monkeypatched), so the module runs on the dev Mac and the Jetson.

Scope — the cost model every gate shares:
  fees.py            round_trip_cost_pct / required_edge_pct / crypto_entry_fee_bps
  liquidity.py       per_bar_round_trip_cost / market_impact_pct / edge_spread_series
  cost_regime.py     amihud_illiq / vix_regime_code (harvest META features)
  order_utils.py     get_quote (spread_pct producer), should_trade (live gate),
                     compute_limit_price / _round_price_band / maker rung pricing
UNIT CONVENTION under test: every cost is a PERCENT of notional (0.5 == 0.5%
== 50 bps); spreads are the FULL bid/ask spread as a percent of the MIDPOINT.
"""

import logging
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fees  # noqa: E402
import liquidity  # noqa: E402
import cost_regime  # noqa: E402
import order_utils  # noqa: E402

SEED = 20260927


@pytest.fixture(autouse=True)
def _no_journal_share(monkeypatch):
    """Isolate every test from the on-disk journals: the live maker-share
    blend reads None (full taker) unless a test sets it explicitly."""
    monkeypatch.setattr(fees, 'realized_crypto_maker_share',
                        lambda *a, **k: None)
    monkeypatch.setattr(order_utils, 'MAKER_SHARE_NOTIONAL_ENABLED', False)
    monkeypatch.setattr(liquidity, 'SPREAD_FILL_V2', False)


def _share(monkeypatch, value):
    monkeypatch.setattr(fees, 'realized_crypto_maker_share',
                        lambda *a, **k: value)


def _spread_grid(rng, n=200):
    """Valid (finite, >= 0) spreads in percent: edge values + log-uniform."""
    edge = [0.0, 1e-9, 0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 1.0, 1.5, 5.0]
    return sorted(edge + [10 ** rng.uniform(-4, 1) for _ in range(n)])


# ===========================================================================
# 1. Unit consistency (bps vs fraction vs percent vs dollars)
# ===========================================================================

class TestUnitContract:
    def test_crypto_taker_round_trip_in_dollars(self):
        # $10k crypto trade, 25 bps taker per side, zero spread:
        # round_trip_cost_pct is PERCENT -> $10k * 0.50 / 100 = $50 = 2 x $25.
        notional = 10_000.0
        rt_pct = fees.round_trip_cost_pct('crypto', 0.0)
        assert rt_pct == pytest.approx(2 * fees.CRYPTO_TAKER_BPS / 100.0)
        assert notional * rt_pct / 100.0 == pytest.approx(50.0, abs=1e-9)
        one_way_dollars = notional * fees.CRYPTO_TAKER_BPS / 1e4
        assert one_way_dollars == pytest.approx(25.0)
        # A bps/fraction mix-up would give $0.50 or $5000 — both far off.

    def test_spread_charged_once_in_percent(self):
        # 10 bps full spread on $10k = $10 charged once per round trip.
        base = fees.round_trip_cost_pct('crypto', 0.0)
        with_sp = fees.round_trip_cost_pct('crypto', 0.10)
        assert 10_000 * (with_sp - base) / 100.0 == pytest.approx(10.0)
        base_s = fees.round_trip_cost_pct('stock', 0.0)
        # stock: 3 bps slippage per side x2 + 0.3 bps regulatory = 6.3 bps
        assert 10_000 * base_s / 100.0 == pytest.approx(6.3)

    def test_quote_to_gate_chain_units(self):
        """get_quote -> spread_pct (percent of MID) -> should_trade ->
        required_edge_pct: the whole live chain agrees on percent."""
        import datetime as _dt

        class _Q:
            # fresh t (ENGINE r3 W10: a missing timestamp now fails closed)
            bp, ap = 99.95, 100.05
            t = _dt.datetime.now(_dt.timezone.utc)

        class _Api:
            def get_latest_crypto_quotes(self, syms):
                return {syms[0]: _Q()}

            def get_latest_quote(self, sym):
                return _Q()

        q = order_utils.get_quote(_Api(), 'BTC/USD', 'crypto')
        assert q['spread_pct'] == pytest.approx(0.10)          # 10 bps, percent
        floor = fees.required_edge_pct('crypto', q['spread_pct'])
        assert floor == pytest.approx(2.0 * (0.50 + 0.10))       # 1.20 %
        assert order_utils.should_trade(floor + 1e-9, q['spread_pct']) is True
        assert order_utils.should_trade(floor - 1e-9, q['spread_pct']) is False
        qs = order_utils.get_quote(_Api(), 'AAPL', 'stock')
        fs = fees.required_edge_pct('stock', qs['spread_pct'])
        assert fs == pytest.approx(2.0 * (0.063 + 0.10))
        assert order_utils.should_trade(fs + 1e-9, qs['spread_pct'],
                                        asset_type='stock') is True
        assert order_utils.should_trade(fs - 1e-9, qs['spread_pct'],
                                        asset_type='stock') is False

    def test_market_impact_units_percent(self):
        # k=1, spread 0.1 %, notional == ADV -> one side 0.1 % -> RT 0.2 %.
        assert liquidity.market_impact_pct(1e6, 1e6, 0.1, k=1.0) == \
            pytest.approx(0.2)
        # $25k into $2.5M ADV: sqrt(0.01)=0.1 -> 0.1*0.1*2 = 0.02 % = $5.
        imp = liquidity.market_impact_pct(25_000, 2.5e6, 0.1, k=1.0)
        assert 25_000 * imp / 100.0 == pytest.approx(5.0)

    def test_edge_stamp_fraction_to_percent(self, monkeypatch):
        """edge_spread_series scales bidask's FRACTION output x100 exactly
        once (a fake bidask returning 0.001 must stamp 0.10 %)."""
        fake = types.ModuleType('bidask')
        fake.edge_rolling = lambda df, window: pd.Series(
            np.full(len(df), 0.001), index=df.index)
        monkeypatch.setitem(sys.modules, 'bidask', fake)
        df = _ohlc(np.random.default_rng(SEED), 60)
        s = liquidity.edge_spread_series(df)
        assert np.allclose(s.values, 0.10)

    def test_short_cost_base_is_long_cost(self):
        import short_cost
        rng = random.Random(SEED)
        for sp in _spread_grid(rng, 50):
            got = short_cost.short_round_trip_cost_pct('2025-06-01', sp,
                                                       hold_days=0.0)
            assert got == pytest.approx(fees.round_trip_cost_pct('stock', sp))

    def test_execution_report_fee_identity(self, monkeypatch):
        # execution_report prints RT = taker - (taker-maker)*share (entry)
        # + taker (exit), in bps. Must equal fees' live RT at spread 0 x100.
        taker, maker = fees.CRYPTO_TAKER_BPS, fees.CRYPTO_MAKER_BPS
        for share in np.linspace(0.0, 1.0, 21):
            _share(monkeypatch, float(share))
            report_rt_bps = (taker - (taker - maker) * share) + taker
            fees_rt_bps = fees.round_trip_cost_pct('crypto', 0.0, live=True) * 100
            assert fees_rt_bps == pytest.approx(report_rt_bps)

    def test_backtest_charges_percent_crypto(self):
        """backtest.simulate_ticker: gross_pct - net_pct == the percent RT
        cost; on $10k that is $60 at the flat 0.10 % crypto spread."""
        backtest = pytest.importorskip('backtest')
        from strategy_config import policy_for
        rng = np.random.RandomState(1)
        n = 80
        close = 100 * np.cumprod(1 + 0.004 + rng.normal(0, 0.001, n))
        idx = pd.date_range('2025-03-03', periods=n, freq='h', tz='UTC')
        tdf = pd.DataFrame({'Open': close * 0.999, 'High': close * 1.002,
                            'Low': close * 0.998, 'Close': close,
                            'ATR': np.full(n, 0.6)}, index=idx)
        trades = backtest.simulate_ticker(tdf, np.full(n, 5.0), 'crypto',
                                          0.1, policy_for('crypto'))
        assert trades
        for t in trades:
            charged = t['gross_pct'] - t['net_pct']
            assert 10_000 * charged / 100.0 == pytest.approx(60.0, abs=0.02)


# ===========================================================================
# 2. Monotonicity
# ===========================================================================

class TestMonotonicity:
    @pytest.mark.parametrize('asset', ['crypto', 'stock'])
    @pytest.mark.parametrize('maker', [False, True])
    def test_rt_nondecreasing_in_spread(self, asset, maker):
        rng = random.Random(SEED)
        grid = _spread_grid(rng)
        costs = [fees.round_trip_cost_pct(asset, s, maker=maker) for s in grid]
        assert all(b >= a for a, b in zip(costs, costs[1:]))
        floors = [fees.required_edge_pct(asset, s, maker=maker) for s in grid]
        assert all(b >= a for a, b in zip(floors, floors[1:]))

    def test_rt_nondecreasing_in_taker_fee(self, monkeypatch):
        rng = random.Random(SEED)
        for s in _spread_grid(rng, 20):
            prev = -math.inf
            for taker in sorted([0.0, 1.0, 15.0, 25.0, 40.0, 100.0]):
                monkeypatch.setattr(fees, 'CRYPTO_TAKER_BPS', taker)
                c = fees.round_trip_cost_pct('crypto', s)
                assert c >= prev
                prev = c

    def test_rt_nonincreasing_in_maker_share(self, monkeypatch):
        rng = random.Random(SEED)
        shares = sorted([0.0, 1.0] + [rng.random() for _ in range(50)])
        static = fees.round_trip_cost_pct('crypto', 0.1)
        maker_only = fees.round_trip_cost_pct('crypto', 0.1, maker=True)
        prev = math.inf
        for sh in shares:
            _share(monkeypatch, sh)
            c = fees.round_trip_cost_pct('crypto', 0.1, live=True)
            assert c <= prev + 1e-15
            assert maker_only - 1e-12 <= c <= static + 1e-12
            prev = c

    def test_live_share_clamped_out_of_range(self, monkeypatch):
        for sh, want in [(-3.0, fees.CRYPTO_TAKER_BPS),
                         (7.0, fees.CRYPTO_MAKER_BPS)]:
            _share(monkeypatch, sh)
            assert fees.crypto_entry_fee_bps(live=True) == want

    def test_fee_schedule_has_no_volume_tier_lookup(self):
        # The task's "fee tier monotone in 30-day volume" invariant is
        # vacuous today: fees.py prices ONE tier (tier 1). Pin that, so a
        # future tier function lands with its own monotonicity test.
        tierish = [n for n in dir(fees)
                   if 'tier' in n.lower() or 'volume' in n.lower()]
        assert tierish == []

    @pytest.mark.parametrize('asset', ['crypto', 'stock'])
    def test_per_bar_monotone_in_spread_notional_adv_k(self, asset):
        rng = np.random.default_rng(SEED)
        sp = np.sort(np.concatenate([[0.0, 0.02, 1.5, 3.0],
                                     10 ** rng.uniform(-3, 0.5, 200)]))
        base = liquidity.per_bar_round_trip_cost(asset, sp)
        assert np.all(np.diff(base) >= 0)
        adv = np.full(sp.shape, 5e6)
        prev = base
        for notional in [1.0, 1e3, 1e4, 1e5, 1e6, 1e8]:
            c = liquidity.per_bar_round_trip_cost(asset, sp, adv_dollar=adv,
                                                  notional=notional)
            assert np.all(c >= prev - 1e-12)
            prev = c
        prev = np.full(sp.shape, np.inf)
        for a in [1e3, 1e5, 1e7, 1e9]:
            c = liquidity.per_bar_round_trip_cost(
                asset, sp, adv_dollar=np.full(sp.shape, a), notional=25_000)
            assert np.all(c <= prev + 1e-12)
            prev = c
        prev = base
        for k in [0.0, 0.5, 1.0, 5.0, 50.0]:
            c = liquidity.per_bar_round_trip_cost(
                asset, sp, adv_dollar=adv, notional=25_000, impact_k=k)
            assert np.all(c >= prev - 1e-12)
            # impact capped per side
            assert np.all(c - base <= 2 * liquidity.IMPACT_CAP_PCT + 1e-12)
            prev = c

    def test_scalar_impact_monotone_and_capped(self):
        rng = random.Random(SEED)
        for _ in range(200):
            adv = 10 ** rng.uniform(3, 9)
            sp = 10 ** rng.uniform(-3, 0.3)
            ns = sorted(10 ** rng.uniform(0, 9) for _ in range(6))
            vals = [liquidity.market_impact_pct(n, adv, sp) for n in ns]
            assert all(b >= a for a, b in zip(vals, vals[1:]))
            assert all(0.0 <= v <= 2 * liquidity.IMPACT_CAP_PCT for v in vals)

    def test_amihud_nonnegative_and_degenerate(self):
        rng = np.random.default_rng(SEED)
        c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))))
        v = pd.Series(rng.uniform(0, 1e5, 100))
        v.iloc[[3, 50]] = 0.0
        c.iloc[70] = np.nan
        out = cost_regime.amihud_illiq(c, v)
        fin = out.dropna()
        assert (fin >= 0).all() and np.isfinite(fin).all()
        const = cost_regime.amihud_illiq(pd.Series(np.full(30, 5.0)),
                                         pd.Series(np.full(30, 100.0)))
        assert (const.dropna() == 0.0).all()
        z = cost_regime.amihud_illiq(pd.Series(np.zeros(30)),
                                     pd.Series(np.full(30, 100.0)))
        assert not np.isinf(z).any() and not (z < 0).any()

    def test_vix_regime_code_monotone(self):
        levels = np.linspace(5, 80, 400)
        codes = [cost_regime.vix_regime_code(x) for x in levels]
        assert all(b >= a for a, b in zip(codes, codes[1:]))
        assert cost_regime.vix_regime_code(float('nan')) == 1
        assert cost_regime.vix_regime_code(None) == 1


# ===========================================================================
# 2b. Spread estimators on degenerate inputs (both bidask and fallback paths)
# ===========================================================================

def _ohlc(rng, n, px=100.0, vol=0.003):
    c = px * np.exp(np.cumsum(rng.normal(0, vol, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.date_range('2025-01-02', periods=n, freq='h')
    return pd.DataFrame({'Open': o, 'High': h, 'Low': lo, 'Close': c},
                        index=idx)


def _degenerate_frames():
    rng = np.random.default_rng(SEED)
    idx = pd.date_range('2025-01-02', periods=60, freq='h')
    const = pd.DataFrame({k: np.full(60, 50.0) for k in
                          ('Open', 'High', 'Low', 'Close')}, index=idx)
    zeros = const * 0.0
    base = _ohlc(rng, 60)
    with_nan = base.copy()
    with_nan.iloc[40, :] = np.nan
    with_zero = base.copy()
    with_zero.iloc[40, :] = 0.0
    with_neg = base.copy()
    with_neg.iloc[45, 3] = -1.0
    return {
        'one_bar': base.iloc[:1], 'two_bars': base.iloc[:2],
        'constant': const, 'all_zero': zeros, 'nan_row': with_nan,
        'zero_row': with_zero, 'neg_close': with_neg,
        'empty': base.iloc[:0],
    }


@pytest.mark.parametrize('path', ['bidask', 'fallback'])
@pytest.mark.parametrize('name', sorted(_degenerate_frames()))
@pytest.mark.parametrize('v2', [False, True])
def test_edge_series_never_negative_inf_or_nan(monkeypatch, path, name, v2):
    if path == 'fallback':
        monkeypatch.setitem(sys.modules, 'bidask', None)   # force ImportError
    else:
        pytest.importorskip('bidask')
    monkeypatch.setattr(liquidity, 'SPREAD_FILL_V2', v2)
    df = _degenerate_frames()[name]
    with np.errstate(all='ignore'):
        s = liquidity.edge_spread_series(df, asset_type='stock')
    assert len(s) == len(df)
    vals = s.to_numpy(dtype=float)
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= liquidity.SPREAD_FLOOR_PCT - 1e-12)
    assert np.all(vals <= liquidity.SPREAD_CAP_PCT + 1e-12)


def test_abdi_ranaldo_raw_never_negative():
    rng = np.random.default_rng(SEED)
    for name, df in _degenerate_frames().items():
        if len(df) < 3:
            continue
        with np.errstate(all='ignore'):
            raw = liquidity._abdi_ranaldo_rolling(df, 35)
        fin = raw[np.isfinite(raw)]
        assert np.all(fin >= 0.0), name
    df = _ohlc(rng, 200)
    raw = liquidity._abdi_ranaldo_rolling(df, 35)
    assert np.all(np.isnan(raw[:34])) and np.all(raw[34:] > 0)


@pytest.mark.xfail(strict=False, reason=(
    'OWNER item W2-F1: _abdi_ranaldo_rolling maps a NaN variance (zero/NaN/'
    'negative price in the window) to 0.0, not NaN, so under '
    'TRADER_SPREAD_FILL_V2 a corrupt window is stamped at the 0.02 floor '
    'instead of FLAT_SPREAD_PCT (V2 contract). Default V1 values identical.'))
def test_fallback_corrupt_window_pays_flat_under_v2(monkeypatch):
    monkeypatch.setitem(sys.modules, 'bidask', None)
    monkeypatch.setattr(liquidity, 'SPREAD_FILL_V2', True)
    df = _degenerate_frames()['zero_row']
    with np.errstate(all='ignore'):
        s = liquidity.edge_spread_series(df, asset_type='stock')
    # rows 40..59 have the zero print inside their 35-bar window
    assert np.allclose(s.values[40:], fees.FLAT_SPREAD_PCT['stock'])


# ===========================================================================
# 3. Symmetry / identity / branch selection
# ===========================================================================

class TestIdentity:
    def test_zero_fee_zero_spread_is_exactly_zero(self, monkeypatch):
        for name in ('CRYPTO_TAKER_BPS', 'CRYPTO_MAKER_BPS',
                     'STOCK_REGULATORY_BPS', 'STOCK_SLIPPAGE_BPS_PER_SIDE'):
            monkeypatch.setattr(fees, name, 0.0)
        for asset in ('crypto', 'stock'):
            for maker in (False, True):
                assert fees.round_trip_cost_pct(asset, 0.0, maker=maker) == 0.0
                assert fees.required_edge_pct(asset, 0.0, maker=maker) == 0.0
        assert liquidity.market_impact_pct(1e4, 1e6, 0.0) == 0.0

    def test_crypto_round_trip_is_twice_one_way_taker(self):
        rt_fee = fees.round_trip_cost_pct('crypto', 0.0)
        assert rt_fee == pytest.approx(2 * fees.crypto_entry_fee_bps() / 100)
        mk = fees.round_trip_cost_pct('crypto', 0.0, maker=True)
        assert mk == pytest.approx(
            (fees.CRYPTO_MAKER_BPS + fees.CRYPTO_TAKER_BPS) / 100)

    def test_linearity_contract_random(self):
        """fee_const + spread, exactly, for every valid spread (the
        contract per_bar_round_trip_cost vectorizes on)."""
        rng = random.Random(SEED)
        for asset in ('crypto', 'stock'):
            for maker in (False, True):
                fc = fees.round_trip_cost_pct(asset, 0.0, maker=maker)
                for s in _spread_grid(rng, 100):
                    assert fees.round_trip_cost_pct(asset, s, maker=maker) == \
                        pytest.approx(fc + s, rel=0, abs=1e-12)

    def test_branch_selection_by_asset_type(self, caplog):
        c = fees.round_trip_cost_pct('crypto', 0.0)
        s = fees.round_trip_cost_pct('stock', 0.0)
        assert c == pytest.approx(0.50) and s == pytest.approx(0.063)
        assert c != s
        # Unknown / mis-cased labels price as STOCK and WARN (not silent).
        for bad in ('Crypto', 'CRYPTO', 'option', ''):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger='fees'):
                assert fees.round_trip_cost_pct(bad, 0.0) == s
            assert any('unknown asset_type' in r.getMessage()
                       for r in caplog.records), bad
        # Every flat-spread selector in liquidity maps labels identically.
        for lab in ('crypto', 'stock', 'Crypto', 'x'):
            want = fees.FLAT_SPREAD_PCT['crypto' if lab == 'crypto' else 'stock']
            got = liquidity.per_bar_round_trip_cost(lab, np.array([np.nan]))[0]
            assert got == pytest.approx(
                fees.round_trip_cost_pct(lab, 0.0) + want)

    def test_stock_gate_never_reads_crypto_maker_share(self, monkeypatch):
        base = fees.required_edge_pct('stock', 0.05, live=True)
        _share(monkeypatch, 1.0)
        assert fees.required_edge_pct('stock', 0.05, live=True) == base
        monkeypatch.setattr(order_utils, 'MAKER_SHARE_NOTIONAL_ENABLED', True)
        monkeypatch.setattr(order_utils, 'realized_crypto_maker_share_notional',
                            lambda: 1.0)
        assert order_utils.should_trade(base + 1e-9, 0.05,
                                        asset_type='stock') is True
        assert order_utils.should_trade(base - 1e-9, 0.05,
                                        asset_type='stock') is False

    def test_should_trade_is_direction_blind(self):
        # Documented: abs(pred). Pinned so a sign change is deliberate.
        f = fees.required_edge_pct('crypto', 0.1)
        assert order_utils.should_trade(-(f + 1e-6), 0.1) is True


# ===========================================================================
# 4. Parity of duplicated cost arithmetic
# ===========================================================================

class TestParity:
    def test_per_bar_equals_scalar_random(self):
        rng = np.random.default_rng(SEED)
        for asset in ('crypto', 'stock'):
            for maker in (False, True):
                sp = 10 ** rng.uniform(-3, 0.5, 300)
                vec = liquidity.per_bar_round_trip_cost(asset, sp, maker=maker)
                ref = [fees.round_trip_cost_pct(asset, float(x), maker=maker)
                       for x in sp]
                assert np.allclose(vec, ref, rtol=0, atol=1e-12)

    def test_vector_impact_equals_scalar_random(self):
        rng = np.random.default_rng(SEED + 1)
        sp = 10 ** rng.uniform(-3, 0.3, 300)
        adv = 10 ** rng.uniform(2, 9, 300)
        for notional in (1.0, 25_000.0, 1e7):
            for k in (0.0, 1.0, 18.0):
                vec = liquidity.per_bar_round_trip_cost(
                    'stock', sp, adv_dollar=adv, notional=notional, impact_k=k)
                base = liquidity.per_bar_round_trip_cost('stock', sp)
                ref = [liquidity.market_impact_pct(notional, a, s, k=k)
                       for a, s in zip(adv, sp)]
                assert np.allclose(vec - base, ref, rtol=0, atol=1e-10)

    @pytest.mark.parametrize('count_share', [None, 0.0, 0.3, 1.0])
    def test_should_trade_notional_delta_matches_fees_blend(
            self, monkeypatch, count_share):
        """order_utils.should_trade re-derives the notional-share threshold
        as a DELTA on fees' count-share threshold; it must equal fees'
        blend evaluated directly at the notional share, on a grid."""
        _share(monkeypatch, count_share)
        monkeypatch.setattr(order_utils, 'MAKER_SHARE_NOTIONAL_ENABLED', True)
        rng = random.Random(SEED)
        for sh in [0.0, 0.25, 0.5, 1.0, -0.5, 1.7] + \
                [rng.random() for _ in range(10)]:
            monkeypatch.setattr(order_utils,
                                'realized_crypto_maker_share_notional',
                                lambda sh=sh: sh)
            for spread in (0.0, 0.05, 0.3):
                for m in (1.0, 2.0, 3.5):
                    c = min(max(sh, 0.0), 1.0)
                    entry = (fees.CRYPTO_MAKER_BPS * c
                             + fees.CRYPTO_TAKER_BPS * (1 - c))
                    want = m * ((entry + fees.CRYPTO_TAKER_BPS) / 100 + spread)
                    assert order_utils.should_trade(want + 1e-9, spread,
                                                    min_edge=m) is True
                    assert order_utils.should_trade(want - 1e-9, spread,
                                                    min_edge=m) is False


# ===========================================================================
# 5. G2-2 — crypto price rounding vs Alpaca's price_increment (OWNER item)
# ===========================================================================
# Fixture: Alpaca GET /v2/assets/{symbol}, fetched READ-ONLY on the Jetson
# 2026-09-27T05:20:53Z (legacy SDK api.get_asset). Tonight's quotes from
# api.get_latest_crypto_quotes (2026-09-27 ~01:20-01:25 EDT).
ALPACA_CRYPTO_ASSETS = {
    'BTC/USD':  {'price_increment': 1e-9, 'min_order_size': 0.000011834,
                 'min_trade_increment': 1e-9},
    'ETH/USD':  {'price_increment': 1e-9, 'min_order_size': 0.000370014,
                 'min_trade_increment': 1e-9},
    'SOL/USD':  {'price_increment': 1e-9, 'min_order_size': 0.008224698,
                 'min_trade_increment': 1e-9},
    'LINK/USD': {'price_increment': 1e-9, 'min_order_size': 0.070521861,
                 'min_trade_increment': 1e-9},
    'XRP/USD':  {'price_increment': 1e-9, 'min_order_size': 0.655737704,
                 'min_trade_increment': 1e-9},
    'DOGE/USD': {'price_increment': 1e-9, 'min_order_size': 10.374520178,
                 'min_trade_increment': 1e-9},
}
TONIGHT_QUOTES = {   # (bid, ask)
    'BTC/USD': (84359.613, 84388.707), 'ETH/USD': (2694.2, 2695.005),
    'SOL/USD': (120.554, 120.7), 'LINK/USD': (14.1146, 14.1458),
    'XRP/USD': (1.51194, 1.5177), 'DOGE/USD': (0.0959143, 0.096205),
}


def _on_grid(px, inc):
    return round(round(px / inc) * inc, 12)


def _price_samples(sym, n=300):
    """Prices ON the asset's increment grid around tonight's quotes."""
    rng = random.Random(SEED)
    inc = ALPACA_CRYPTO_ASSETS[sym]['price_increment']
    bid, ask = TONIGHT_QUOTES[sym]
    out = [bid, ask]
    for _ in range(n):
        out.append(_on_grid(bid * (1 + rng.uniform(-0.05, 0.05)), inc))
    return out


def _maker_rung_limit(monkeypatch, bid, ask):
    """Drive order_utils.place_maker_buy through ONE rung with a fake API
    (no network, nothing is sent) and return the submitted limit_price."""
    sent = []

    class _Order:
        id = 'x'
        status = 'filled'
        filled_qty = 1.0
        filled_avg_price = bid

    class _Api:
        def submit_order(self, **kw):
            sent.append(kw)
            return _Order()

    monkeypatch.setattr(order_utils, 'manage_order_lifecycle',
                        lambda *a, **k: _Order())
    monkeypatch.setattr(order_utils, '_journal_entry_fills',
                        lambda *a, **k: None)
    mid = (bid + ask) / 2
    q = {'bid': bid, 'ask': ask, 'midpoint': mid,
         'spread_pct': (ask - bid) / mid * 100}
    order_utils.place_maker_buy(_Api(), 'X/USD', 1000.0, lambda: q)
    return sent[0]['limit_price']


def test_g2_2_fixture_shape():
    # The owner re-fetches with the same read-only call and diffs.
    for sym, meta in ALPACA_CRYPTO_ASSETS.items():
        assert meta['price_increment'] > 0
        assert meta['min_order_size'] > 0
        bid, ask = TONIGHT_QUOTES[sym]
        assert 0 < bid < ask


def test_g2_2_maker_rung_quantified_tonight(monkeypatch):
    """Characterization that survives the G2-2 fix: tonight's DOGE bid
    0.0959143 posts either at the bid (fixed) or at 4-dp 0.0959 — 1.49 bps
    BEHIND the touch it claims to join (today). BTC/ETH/SOL/LINK quotes
    tonight carry <= 4 dp, so they post at the bid either way."""
    bid, ask = TONIGHT_QUOTES['DOGE/USD']
    px = _maker_rung_limit(monkeypatch, bid, ask)
    assert px == bid or (px == 0.0959 and
                         (px - bid) / bid * 1e4 == pytest.approx(-1.49, abs=0.01))
    for sym in ('BTC/USD', 'ETH/USD', 'SOL/USD', 'LINK/USD'):
        b, a = TONIGHT_QUOTES[sym]
        assert _maker_rung_limit(monkeypatch, b, a) == b


G2_2 = pytest.mark.xfail(strict=False, reason='G2-2 owner item')


@G2_2
@pytest.mark.parametrize('sym', sorted(ALPACA_CRYPTO_ASSETS))
def test_g2_2_maker_rung_joins_bid_exactly(monkeypatch, sym):
    """A bid that is on the venue grid must be posted UNCHANGED."""
    inc = ALPACA_CRYPTO_ASSETS[sym]['price_increment']
    for bid in _price_samples(sym, 60):
        ask = bid * 1.003
        px = _maker_rung_limit(monkeypatch, bid, ask)
        assert abs(px - bid) <= inc / 2 + 1e-12, (sym, bid, px)


@G2_2
def test_g2_2_maker_rung_never_marketable(monkeypatch):
    """A 'join the bid' rung must never reach the ask (it would fill as
    TAKER: 25 bps instead of 15). DOGE, 1-tick-of-4dp-wide book."""
    bid, ask = 0.09577, 0.09578
    assert _maker_rung_limit(monkeypatch, bid, ask) < ask


@G2_2
@pytest.mark.parametrize('sym', sorted(ALPACA_CRYPTO_ASSETS))
def test_g2_2_rounding_never_coarser_than_price_increment(sym):
    """compute_limit_price and _round_price_band('crypto') must round at
    (or finer than) the asset's price_increment: the rounding error is at
    most half an increment for every sampled price."""
    inc = ALPACA_CRYPTO_ASSETS[sym]['price_increment']
    rng = random.Random(SEED)
    for px in _price_samples(sym):
        assert abs(order_utils._round_price_band(px, 'crypto') - px) \
            <= inc / 2 + 1e-12, (sym, px)
        spread_pct = rng.choice([0.02, 0.08, 0.3, 0.6])
        q = {'midpoint': px, 'spread_pct': spread_pct}
        off = (px * spread_pct / 100 * 0.1 if spread_pct > 0.1
               else px * 5 / 1e4)
        for side, sign in (('buy', 1), ('sell', -1)):
            got = order_utils.compute_limit_price(side, q)
            assert abs(got - (px + sign * off)) <= inc / 2 + 1e-12, \
                (sym, side, px, got)


@G2_2
def test_g2_2_ioc_cap_respected_for_crypto():
    """ioc_limit_price('crypto') must stay within [touch, touch*(1+cap)]
    (the cap the helper exists to enforce). XRP ask 1.51, cap 40 bps ->
    2-dp rounding posts 1.52 = +66 bps."""
    for sym in ALPACA_CRYPTO_ASSETS:
        bid, ask = TONIGHT_QUOTES[sym]
        for a in (ask, 1.51, 1.0149):
            if sym != 'XRP/USD' and a != ask:
                continue
            q = {'bid': a * 0.997, 'ask': a, 'midpoint': a * 0.9985}
            for cap in (20, 40):
                px = order_utils.ioc_limit_price('buy', q, cap, 'crypto')
                assert a <= px <= a * (1 + cap / 1e4) + 1e-9, (sym, a, cap, px)
