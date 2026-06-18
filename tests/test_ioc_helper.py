"""Wave-7 Finding 2: marketable-IOC slippage-cap helper (order_utils).

Pins the capped-limit math, side signs, per-price-band rounding, the
derive-from-midpoint path, and the IOC submit shape against a fake API."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import order_utils as ou


class _FakeAPI:
    def __init__(self, fail=False):
        self.fail = fail
        self.orders = []

    def submit_order(self, **kw):
        if self.fail:
            raise RuntimeError("rejected")
        self.orders.append(kw)
        return {'id': 'ok', **kw}


class TestIOCLimitPrice:
    def test_buy_lifts_ask_by_cap(self):
        q = {'bid': 99.0, 'ask': 100.0, 'midpoint': 99.5}
        # 20 bps over the ask: 100*(1.002) = 100.20, 2dp band
        assert ou.ioc_limit_price('buy', q, 20, 'stock') == pytest.approx(100.20)

    def test_sell_hits_bid_minus_cap(self):
        q = {'bid': 50.0, 'ask': 50.2, 'midpoint': 50.1}
        # 20 bps under the bid: 50*(0.998) = 49.90
        assert ou.ioc_limit_price('sell', q, 20, 'stock') == pytest.approx(49.90)

    def test_sub_dollar_uses_4dp(self):
        q = {'bid': 0.50, 'ask': 0.52, 'midpoint': 0.51}
        px = ou.ioc_limit_price('buy', q, 30, 'stock')
        # 0.52*1.003 = 0.521560 -> 4dp
        assert px == pytest.approx(0.5216, abs=1e-9)

    def test_dollar_plus_uses_2dp(self):
        q = {'bid': 12.0, 'ask': 12.04, 'midpoint': 12.02}
        px = ou.ioc_limit_price('buy', q, 10, 'stock')
        assert px == round(px, 2)  # never sub-penny on a >=$1 name

    def test_derives_bid_ask_from_midpoint(self):
        # no explicit bid/ask -> derive from midpoint + full spread_pct
        q = {'midpoint': 100.0, 'spread_pct': 0.20}  # half-spread 0.10 -> ask 100.10
        # buy IOC at 0 cap == the derived ask
        assert ou.ioc_limit_price('buy', q, 0, 'stock') == pytest.approx(100.10)
        assert ou.ioc_limit_price('sell', q, 0, 'stock') == pytest.approx(99.90)

    def test_wider_cap_pays_more(self):
        q = {'bid': 99, 'ask': 100, 'midpoint': 99.5}
        assert ou.ioc_limit_price('buy', q, 50, 'stock') > ou.ioc_limit_price('buy', q, 5, 'stock')


class TestPlaceMarketableIOC:
    def test_submits_ioc_limit(self):
        api = _FakeAPI()
        q = {'bid': 99.0, 'ask': 100.0, 'midpoint': 99.5}
        ou.place_marketable_ioc(api, 'NVDA', 'buy', 3, q, 20, 'stock')
        assert len(api.orders) == 1
        o = api.orders[0]
        assert o['type'] == 'limit' and o['time_in_force'] == 'ioc'
        assert o['side'] == 'buy' and o['qty'] == 3
        assert o['limit_price'] == pytest.approx(100.20)

    def test_submit_failure_returns_none(self):
        api = _FakeAPI(fail=True)
        q = {'bid': 99.0, 'ask': 100.0, 'midpoint': 99.5}
        assert ou.place_marketable_ioc(api, 'X', 'sell', 1, q, 20) is None


class TestCapTable:
    def test_class_caps_ordered(self):
        from strategy_config import IOC_CAP_BPS, IOC_EXIT_CAP_BPS
        assert IOC_CAP_BPS['mega'] < IOC_CAP_BPS['mid'] < IOC_CAP_BPS['spec']
        # exit caps are wider than entry caps (a stop must fill)
        for k in ('mega', 'mid', 'spec'):
            assert IOC_EXIT_CAP_BPS[k] >= IOC_CAP_BPS[k]
