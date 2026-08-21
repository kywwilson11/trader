"""Panel batch v3 — order_utils adjudicated fixes (keep-best maker evidence,
NaN/crossed quote guard, stream pacing, bidirectional symbol variants,
symbols-filtered listings, list_positions reconstruct, scoped-cancel wait,
two-phase flatten, stranded-fill instrumentation)."""

import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import order_stream
import order_utils as ou
from order_utils import (
    _symbol_variants,
    cancel_all_open_orders,
    cancel_orders_for_symbol,
    check_circuit_breaker,
    emergency_flatten,
    get_quote,
    manage_order_lifecycle,
    place_limit_order,
    place_maker_buy,
    place_stock_limit_order,
    reconstruct_positions,
    verify_position,
)


@pytest.fixture(autouse=True)
def fast_clock(monkeypatch):
    """No-op sleep everywhere EXCEPT the two pacing tests, which install
    their own recording lambda (a monkeypatch inside the test overrides
    this fixture's patch for that test only)."""
    monkeypatch.setattr(time, 'sleep', lambda s: None)


# =====================================================================
# T1 — _symbol_variants bidirectional
# =====================================================================

class TestSymbolVariants:
    def test_crypto_slashed_expands_to_slashless(self):
        assert _symbol_variants('BTC/USD') == {'BTC/USD', 'BTCUSD'}

    def test_crypto_slashless_expands_to_slashed(self):
        assert _symbol_variants('BTCUSD') == {'BTCUSD', 'BTC/USD'}

    def test_other_slashless_crypto_expands(self):
        assert _symbol_variants('DOGEUSD') == {'DOGEUSD', 'DOGE/USD'}

    def test_stock_ticker_stays_single_variant(self):
        assert _symbol_variants('NVDA') == {'NVDA'}

    def test_short_usd_suffix_guarded_by_length(self):
        # len-5 guard: 'ABUSD' is exactly 5 chars, not > 5 — not expanded
        assert _symbol_variants('ABUSD') == {'ABUSD'}


# =====================================================================
# T2 — place_maker_buy keeps the best acquisition evidence
# =====================================================================

class _FakeOrder:
    def __init__(self, id, status='new', filled_qty=0, filled_avg_price=None,
                symbol='BTC/USD', qty=1.0, side='buy'):
        self.id = id
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.symbol = symbol
        self.qty = qty
        self.side = side


class _FakeAPI:
    """Scripted API modeled on tests/test_order_utils.py::_FakeAPI: each
    submit_order returns the next scripted order, and get_order returns its
    already-terminal state (this fake never simulates a pending window).
    If `raise_for` names an order id, get_order raises for that id on
    every call — models a broker that never recovers for one order."""

    def __init__(self, script, raise_for=None):
        self.script = list(script)   # list of _FakeOrder terminal states
        self.submitted = []          # kwargs of every submit_order call
        self.canceled = []
        self._orders = {}
        self.raise_for = raise_for

    def submit_order(self, **kw):
        terminal = self.script.pop(0)
        self.submitted.append(kw)
        self._orders[terminal.id] = terminal
        return _FakeOrder(terminal.id, status='new',
                          symbol=kw.get('symbol', 'BTC/USD'),
                          qty=kw.get('qty', 0), side=kw.get('side', 'buy'))

    def get_order(self, order_id):
        if self.raise_for is not None and order_id == self.raise_for:
            raise RuntimeError('broker unreachable')
        return self._orders[order_id]

    def cancel_order(self, order_id):
        self.canceled.append(order_id)


class TestMakerBuyKeepsAcquiredEvidence:
    QUOTE = {'bid': 100.0, 'ask': 100.2, 'midpoint': 100.1,
             'spread': 0.2, 'spread_pct': 0.2}

    def test_zero_fill_rung_does_not_erase_partial(self):
        # Rung 1 buys 6.0 (partial, then times out); rung 2 fills 0. The
        # fallback quote feed then dies too, so 'unfilled' is the tactic —
        # but rung 1's proven 6.0 must survive as the returned evidence.
        api = _FakeAPI([
            _FakeOrder('o1', 'canceled', 6.0, 100.0),
            _FakeOrder('o2', 'canceled', 0.0, None),
        ])
        quotes = iter([dict(self.QUOTE), dict(self.QUOTE)])
        result, tactic = place_maker_buy(api, 'BTC/USD', 1000,
                                         lambda: next(quotes, None),
                                         stage_timeout=4)
        assert tactic == 'unfilled'
        assert float(result.filled_qty) == 6.0   # pre-fix: 0

    def test_fallback_lifecycle_none_returns_rung_evidence(self):
        # Both maker rungs end with rung 1's 6.0 as best evidence; the
        # marketable fallback's lifecycle can never confirm (broker dead
        # for that one order) and returns None. The 6.0 must survive.
        api = _FakeAPI([
            _FakeOrder('o1', 'canceled', 6.0, 100.0),
            _FakeOrder('o2', 'canceled', 0.0, None),
            _FakeOrder('o3', 'canceled', 0.0, None),
        ], raise_for='o3')
        result, tactic = place_maker_buy(api, 'BTC/USD', 1000,
                                         lambda: dict(self.QUOTE),
                                         stage_timeout=4)
        assert tactic == 'taker_fallback'
        assert float(result.filled_qty) == 6.0   # pre-fix: result is None

    def test_fallback_fill_still_wins(self):
        # Guards against over-correcting: when the fallback genuinely fills
        # MORE than the rungs did, its result must still win (mirrors the
        # pinned test_taker_fallback_after_rungs_fail).
        api = _FakeAPI([
            _FakeOrder('o1', 'canceled', 0, None),
            _FakeOrder('o2', 'canceled', 0, None),
            _FakeOrder('o3', 'filled', 9.98, 100.15),
        ])
        result, tactic = place_maker_buy(api, 'BTC/USD', 1000,
                                         lambda: dict(self.QUOTE),
                                         stage_timeout=4)
        assert tactic == 'taker_fallback'
        assert float(result.filled_qty) == 9.98


# =====================================================================
# T3 — get_quote: NaN/inf finiteness guard + crossed-quote warning
# =====================================================================

class _Q:
    """Fake Alpaca crypto-quote client for get_quote()."""

    def __init__(self, bp, ap):
        self.bp = bp
        self.ap = ap

    def get_latest_crypto_quotes(self, symbols):
        return {symbols[0]: SimpleNamespace(bp=self.bp, ap=self.ap, t=None)}


class TestGetQuoteGuards:
    def test_nan_and_inf_rejected(self):
        bad_pairs = [
            (float('nan'), 100.0),
            (100.0, float('nan')),
            (float('inf'), 100.0),
            (100.0, float('-inf')),
        ]
        for bp, ap in bad_pairs:
            assert get_quote(_Q(bp, ap), 'BTC/USD', 'crypto') is None, (bp, ap)

    def test_crossed_quote_accepted_but_warned(self, caplog):
        with caplog.at_level('WARNING', logger='order_utils'):
            out = get_quote(_Q(100.1, 100.0), 'BTC/USD', 'crypto')
        assert out is not None
        assert out['spread_pct'] < 0
        assert any('CROSSED' in r.message for r in caplog.records)

    def test_normal_quote_unchanged(self):
        out = get_quote(_Q(100.0, 100.1), 'BTC/USD', 'crypto')
        assert out is not None
        assert out['spread_pct'] == pytest.approx(0.1 / 100.05 * 100)


# =====================================================================
# T4 — manage_order_lifecycle: stream-cache skip-wait pacing
# =====================================================================

class _AlwaysNewAPI:
    """get_order always returns a live 'new' order — never terminal."""

    def __init__(self):
        self.canceled = []

    def get_order(self, oid):
        return SimpleNamespace(status='new', symbol='BTC/USD', qty=1.0,
                               side='buy', filled_qty=0, filled_avg_price=None)

    def cancel_order(self, oid):
        self.canceled.append(oid)


class _AlwaysRaiseGetOrderAPI:
    """get_order always raises — broker unreachable for this order."""

    def __init__(self):
        self.canceled = []

    def get_order(self, oid):
        raise RuntimeError('broker unreachable')

    def cancel_order(self, oid):
        self.canceled.append(oid)


class TestLifecycleStreamPacing:
    def test_stream_terminal_skips_wait_only_once(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, 'sleep', lambda s: sleeps.append(s))
        monkeypatch.setattr(order_stream, 'get_order_state',
                            lambda oid: {'status': 'filled'})
        api = _AlwaysNewAPI()
        manage_order_lifecycle(api, 'o1', timeout=6, poll_interval=2,
                               fallback_to_market=False)
        # 3 iterations total; only the skip is used on iteration 1 — 2 and
        # 3 must pace normally (pre-fix: every iteration skipped -> 0).
        assert sleeps.count(2) == 2

    def test_stream_terminal_error_retries_are_paced(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, 'sleep', lambda s: sleeps.append(s))
        monkeypatch.setattr(order_stream, 'get_order_state',
                            lambda oid: {'status': 'filled'})
        api = _AlwaysRaiseGetOrderAPI()
        result = manage_order_lifecycle(api, 'o1', timeout=30, poll_interval=2,
                                        fallback_to_market=False)
        assert result is None
        assert sleeps.count(2) >= 2   # pre-fix: 0 (never paced, ever)

    def test_no_stream_paces_every_tick(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, 'sleep', lambda s: sleeps.append(s))
        monkeypatch.setattr(order_stream, 'get_order_state', lambda oid: None)
        api = _AlwaysNewAPI()
        manage_order_lifecycle(api, 'o1', timeout=6, poll_interval=2,
                               fallback_to_market=False)
        assert sleeps.count(2) == 3


# =====================================================================
# T5 — symbols-filtered listings + bounded cancel waits
# =====================================================================

class TestListingsAndWaits:
    def test_cancel_for_symbol_passes_symbols_and_limit(self):
        class _RecordingAPI:
            def __init__(self):
                self.records = []

            def list_orders(self, status='open', limit=None, symbols=None):
                self.records.append({'limit': limit, 'symbols': symbols})
                return []

        api = _RecordingAPI()
        assert cancel_orders_for_symbol(api, 'BTC/USD') is True
        assert api.records[0]['limit'] == 500
        assert set(api.records[0]['symbols']) == {'BTC/USD', 'BTCUSD'}

    def test_shim_without_symbols_kwarg_falls_back(self):
        class _NoSymbolsAPI:
            def __init__(self):
                self.limits = []

            def list_orders(self, status='open', limit=None):
                self.limits.append(limit)
                return []

        api = _NoSymbolsAPI()
        assert cancel_orders_for_symbol(api, 'X') is True
        # the symbols-kwarg attempt TypeErrors at binding, before recording
        assert api.limits == [500]

    def test_filtered_listing_error_falls_back_unfiltered(self):
        # A server that REJECTS the symbols-filtered call (not a TypeError
        # shim — a real API failure) must still be listed unfiltered, so
        # narrowing can never hide orders (the docstring's safety claim).
        class _FilteredFailsAPI:
            def __init__(self):
                self.calls = []

            def list_orders(self, status='open', limit=None, symbols=None):
                self.calls.append(symbols)
                if symbols is not None:
                    raise RuntimeError('symbols filter unsupported')
                return []

        api = _FilteredFailsAPI()
        assert cancel_orders_for_symbol(api, 'BTC/USD') is True
        assert api.calls[0] is not None      # filtered attempt made
        assert api.calls[-1] is None         # then fell back unfiltered

    def test_page_cap_warns(self, caplog):
        class _FullPageAPI:
            def list_orders(self, status='open', limit=None):
                return [SimpleNamespace(symbol='OTHER', status='new', id=str(i))
                       for i in range(500)]

        api = _FullPageAPI()
        with caplog.at_level('WARNING', logger='order_utils'):
            result = cancel_orders_for_symbol(api, 'BTC/USD')
        assert result is True
        assert any('cap' in r.message for r in caplog.records)

    def test_cancel_all_scoped_waits_until_clear(self):
        class _ClearsAfterAPI:
            def __init__(self):
                self.order = SimpleNamespace(id='o1', symbol='BTC/USD', status='new')
                self.canceled = []
                self.list_calls = 0

            def list_orders(self, status='open', limit=None, symbols=None):
                self.list_calls += 1
                if self.list_calls < 3:
                    return [self.order]
                return []

            def cancel_order(self, oid):
                self.canceled.append(oid)

        api = _ClearsAfterAPI()
        cancel_all_open_orders(api, symbols=['BTC/USD'])
        assert api.canceled == ['o1']
        assert api.list_calls >= 3

    def test_cancel_all_scoped_warns_when_never_clears(self, caplog):
        class _NeverClearsAPI:
            def __init__(self):
                self.order = SimpleNamespace(id='o1', symbol='BTC/USD', status='new')
                self.canceled = []

            def list_orders(self, status='open', limit=None, symbols=None):
                return [self.order]

            def cancel_order(self, oid):
                self.canceled.append(oid)

        api = _NeverClearsAPI()
        with caplog.at_level('WARNING', logger='order_utils'):
            cancel_all_open_orders(api, symbols=['BTC/USD'])   # must not raise
        assert any('pending' in r.message for r in caplog.records)


# =====================================================================
# T6 — reconstruct_positions: list_positions primary path
# =====================================================================

class TestReconstructListPath:
    def test_primary_path_uses_one_listing(self):
        class _ListOnlyAPI:
            def list_positions(self):
                return [SimpleNamespace(symbol='BTCUSD', qty=2.0,
                                        avg_entry_price=100.0, current_price=110.0)]

            def get_position(self, sym):
                raise AssertionError('must not probe')

        out = reconstruct_positions(_ListOnlyAPI(), ['BTC/USD', 'ETH/USD'])
        assert out == {'BTC/USD': {'qty': 2.0, 'entry_price': 100.0,
                                   'high_water_mark': 110.0}}

    def test_primary_path_skips_shorts_and_none_price(self):
        class _ShortAPI:
            def list_positions(self):
                return [SimpleNamespace(symbol='BTCUSD', qty=-2.0,
                                        avg_entry_price=100.0, current_price=110.0)]

            def get_position(self, sym):
                raise AssertionError('must not probe')

        assert reconstruct_positions(_ShortAPI(), ['BTC/USD']) == {}

        class _NonePriceAPI:
            def list_positions(self):
                return [SimpleNamespace(symbol='BTCUSD', qty=2.0,
                                        avg_entry_price=100.0, current_price=None)]

            def get_position(self, sym):
                raise AssertionError('must not probe')

        out = reconstruct_positions(_NonePriceAPI(), ['BTC/USD'])
        assert out['BTC/USD']['high_water_mark'] == 100.0

    def test_listing_failure_falls_back_to_probes(self):
        class _ProbeOnlyAPI:
            def list_positions(self):
                raise RuntimeError('listing unavailable')

            def get_position(self, sym):
                if sym != 'BTCUSD':
                    raise RuntimeError('position does not exist')
                return SimpleNamespace(symbol=sym, qty=2.0,
                                       avg_entry_price=100.0, current_price=110.0)

        out = reconstruct_positions(_ProbeOnlyAPI(), ['BTC/USD', 'ETH/USD'])
        assert out == {'BTC/USD': {'qty': 2.0, 'entry_price': 100.0,
                                   'high_water_mark': 110.0}}


# =====================================================================
# T7 — manage_order_lifecycle: stranded-fill instrumentation
# =====================================================================

class TestLifecycleStrandedFillLogs:
    def test_giveup_with_partial_logs_error(self, caplog):
        class _PartialThenDeadAPI:
            def __init__(self):
                self.calls = 0

            def get_order(self, oid):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(status='partially_filled', filled_qty=4.0,
                                           symbol='BTC/USD', qty=10.0, side='buy',
                                           filled_avg_price=100.0)
                raise RuntimeError('broker unreachable')

            def cancel_order(self, oid):
                raise RuntimeError('cancel rejected')

        api = _PartialThenDeadAPI()
        with caplog.at_level('WARNING', logger='order_utils'):
            result = manage_order_lifecycle(api, 'o1', timeout=30, poll_interval=5,
                                            fallback_to_market=False)
        assert result is None
        assert any(r.levelname == 'ERROR' and 'filled_qty=4.0' in r.message
                  for r in caplog.records)
        assert any(r.levelname == 'WARNING' and 'cancel' in r.message
                  for r in caplog.records)

    def test_post_cancel_unknown_with_partial_logs_error(self, caplog):
        # Clone of tests/test_review_b02.py::_StuckPartialAPI.
        class _StuckPartialAPI:
            def __init__(self, qty=10.0, filled=6.0, px=100.0, fetch_fail_from=None):
                self.order = SimpleNamespace(id='o1', symbol='BTC/USD', qty=qty,
                                             side='buy', status='partially_filled',
                                             filled_qty=filled, filled_avg_price=px)
                self.canceled = []
                self.get_calls = 0
                self.fetch_fail_from = fetch_fail_from

            def get_order(self, oid):
                self.get_calls += 1
                if self.fetch_fail_from and self.get_calls >= self.fetch_fail_from:
                    raise RuntimeError('broker unreachable')
                return self.order

            def cancel_order(self, oid):
                self.canceled.append(oid)
                if self.order.status not in ('filled',):
                    self.order.status = 'canceled'

        api = _StuckPartialAPI(qty=10.0, filled=6.0, fetch_fail_from=3)
        with caplog.at_level('ERROR', logger='order_utils'):
            result = manage_order_lifecycle(api, 'o1', timeout=4, poll_interval=2,
                                            fallback_to_market=False)
        assert result is None   # pinned behavior UNCHANGED
        assert any('filled_qty=6.0' in r.message for r in caplog.records)


# =====================================================================
# T8 — manage_order_lifecycle: market-fallback remainder qty floor
# =====================================================================

class TestMarketFallbackQtyFloor:
    def test_remainder_floored_to_8dp(self):
        class _PrecisionAPI:
            def __init__(self):
                self.order = SimpleNamespace(id='o1', symbol='BTC/USD', qty='0.5',
                                             side='buy', status='partially_filled',
                                             filled_qty=0.30000000000000004,
                                             filled_avg_price=100.0)
                self.submitted = []
                self.market_order = None

            def get_order(self, oid):
                if oid == 'm1':
                    return self.market_order
                return self.order

            def cancel_order(self, oid):
                self.order.status = 'canceled'

            def submit_order(self, **kw):
                self.submitted.append(kw)
                self.market_order = SimpleNamespace(
                    id='m1', status='filled', filled_qty=kw['qty'],
                    filled_avg_price=100.0, symbol=kw['symbol'],
                    qty=kw['qty'], side=kw['side'])
                return self.market_order

        api = _PrecisionAPI()
        manage_order_lifecycle(api, 'o1', timeout=4, poll_interval=2,
                               fallback_to_market=True)
        expected = math.floor((0.5 - 0.30000000000000004) * 1e8) / 1e8
        assert expected == 0.19999999   # sanity: the pre-fix value was 0.19999999999999996
        assert len(api.submitted) == 1
        assert api.submitted[0]['qty'] == expected


# =====================================================================
# T9 — verify_position: deferred aggregate warning
# =====================================================================

class TestVerifyDeferredWarning:
    def test_no_warning_when_any_variant_succeeds(self, caplog):
        # One variant errors transiently, the other finds a live position.
        # No '[VERIFY]' warning should fire regardless of set-iteration
        # order — a variant that failed while another succeeded is not a
        # drop.
        class _MixedAPI:
            def get_position(self, sym):
                if sym == 'BTC/USD':
                    raise RuntimeError('timeout')
                return SimpleNamespace(symbol=sym, qty=3.0,
                                       avg_entry_price=100.0, current_price=100.0)

        with caplog.at_level('WARNING', logger='order_utils'):
            result = verify_position(_MixedAPI(), 'BTC/USD')
        assert result is not None
        assert not any('[VERIFY]' in r.message for r in caplog.records)

    def test_warning_fires_when_no_variant_succeeds(self, caplog):
        # The deferred aggregate warning must still fire (once, with every
        # variant's error) when NO variant finds the position — deferral
        # must not become suppression.
        class _AllFailAPI:
            def get_position(self, sym):
                raise RuntimeError('timeout')

        with caplog.at_level('WARNING', logger='order_utils'):
            assert verify_position(_AllFailAPI(), 'BTC/USD') is None
        verify_warns = [r for r in caplog.records if '[VERIFY]' in r.message]
        assert len(verify_warns) == 1
        assert 'BTCUSD' in verify_warns[0].message
        assert 'BTC/USD' in verify_warns[0].message


# =====================================================================
# T10 — emergency_flatten v3: two-phase submit/confirm + logging
# =====================================================================

class TestFlattenV3:
    def test_two_phase_submits_before_confirms(self):
        class _TwoPhaseAPI:
            def __init__(self, positions):
                self.positions = positions
                self.events = []
                self._orders = {}
                self._n = 0

            def list_positions(self):
                return self.positions

            def list_orders(self, status='open', limit=None, symbols=None):
                return []

            def submit_order(self, **kw):
                self._n += 1
                oid = f'o{self._n}'
                self.events.append(('submit', kw['symbol']))
                order = SimpleNamespace(id=oid, symbol=kw['symbol'], qty=kw['qty'],
                                        side=kw['side'], status='filled',
                                        filled_qty=kw['qty'], filled_avg_price=100.0)
                self._orders[oid] = order
                return order

            def get_order(self, oid):
                self.events.append(('poll', oid))
                return self._orders[oid]

            def cancel_order(self, oid):
                pass

        symbols = ['BTC/USD', 'ETH/USD', 'LINK/USD']
        positions = [SimpleNamespace(symbol=s, qty=2.0) for s in symbols]
        api = _TwoPhaseAPI(positions)
        failures = emergency_flatten(api, symbols=symbols)
        assert failures == []
        assert [e[0] for e in api.events][:3] == ['submit', 'submit', 'submit']

    def test_zero_qty_position_skipped(self):
        class _NoSubmitAPI:
            def list_positions(self):
                return [SimpleNamespace(symbol='BTC/USD', qty=0.0)]

            def list_orders(self, status='open', limit=None, symbols=None):
                return []

            def submit_order(self, **kw):
                raise AssertionError('must not submit for a zero-qty position')

        failures = emergency_flatten(_NoSubmitAPI(), symbols=['BTC/USD'])
        assert failures == []

    def test_pending_cancel_logged_but_sell_proceeds(self, caplog):
        class _PendingCancelAPI:
            def __init__(self):
                self.positions = [SimpleNamespace(symbol='BTC/USD', qty=2.0)]
                self.submitted = []
                self._orders = {}
                self._n = 0

            def list_positions(self):
                return self.positions

            def list_orders(self, status='open', limit=None, symbols=None):
                # A resting order that never clears, regardless of scope.
                return [SimpleNamespace(id='s1', symbol='BTC/USD', status='new')]

            def cancel_order(self, oid):
                pass   # no-op: the cancel never actually takes

            def submit_order(self, **kw):
                self._n += 1
                oid = f'm{self._n}'
                self.submitted.append(kw)
                order = SimpleNamespace(id=oid, symbol=kw['symbol'], qty=kw['qty'],
                                        side=kw['side'], status='filled',
                                        filled_qty=kw['qty'], filled_avg_price=100.0)
                self._orders[oid] = order
                return order

            def get_order(self, oid):
                return self._orders[oid]

        api = _PendingCancelAPI()
        with caplog.at_level('ERROR', logger='order_utils'):
            emergency_flatten(api, symbols=['BTC/USD'])
        assert any('pending cancel' in r.message for r in caplog.records)
        assert len(api.submitted) == 1   # the market sell still went out

    def test_not_confirmed_log_includes_filled_qty(self, caplog):
        class _StuckFlattenAPI:
            def __init__(self):
                self.positions = [SimpleNamespace(symbol='BTC/USD', qty=2.0)]
                self.order = None

            def list_positions(self):
                return self.positions

            def list_orders(self, status='open', limit=None, symbols=None):
                return []

            def submit_order(self, **kw):
                self.order = SimpleNamespace(id='f1', symbol=kw['symbol'],
                                             qty=kw['qty'], side=kw['side'],
                                             status='new', filled_qty=1.5,
                                             filled_avg_price=100.0)
                return self.order

            def get_order(self, oid):
                return self.order

            def cancel_order(self, oid):
                if self.order is not None:
                    self.order.status = 'canceled'

        api = _StuckFlattenAPI()
        with caplog.at_level('ERROR', logger='order_utils'):
            failures = emergency_flatten(api, symbols=['BTC/USD'])
        assert failures == ['BTC/USD']
        assert any('1.5' in r.message for r in caplog.records)

    def test_flatten_cancels_cross_spelling_stop(self):
        # The position reports the BROKER (slashless) spelling; the resting
        # protective stop was submitted with the universe (slashed)
        # spelling. Only the Item-1 bidirectional _symbol_variants fix lets
        # cancel_orders_for_symbol(api, 'BTCUSD', ...) find and cancel it.
        class _CrossSpellFlattenAPI:
            def __init__(self):
                self.positions = [SimpleNamespace(symbol='BTCUSD', qty=2.0)]
                self.stop_order = SimpleNamespace(id='s1', symbol='BTC/USD',
                                                  status='new')
                self.canceled = []
                self.events = []
                self._orders = {}
                self._n = 0

            def list_positions(self):
                return self.positions

            def list_orders(self, status='open', limit=None, symbols=None):
                if self.stop_order.status == 'new':
                    return [self.stop_order]
                return []

            def cancel_order(self, oid):
                self.canceled.append(oid)
                self.events.append(('cancel', oid))
                if oid == self.stop_order.id:
                    self.stop_order.status = 'canceled'

            def submit_order(self, **kw):
                self._n += 1
                mid = f'm{self._n}'
                self.events.append(('submit', kw['symbol']))
                order = SimpleNamespace(id=mid, symbol=kw['symbol'], qty=kw['qty'],
                                        side=kw['side'], status='filled',
                                        filled_qty=kw['qty'], filled_avg_price=100.0)
                self._orders[mid] = order
                return order

            def get_order(self, oid):
                return self._orders[oid]

        api = _CrossSpellFlattenAPI()
        failures = emergency_flatten(api, symbols=['BTC/USD'])
        assert failures == []
        assert 's1' in api.canceled
        kinds = [e[0] for e in api.events]
        assert kinds.index('cancel') < kinds.index('submit')


# =====================================================================
# T11 — check_circuit_breaker: degenerate-baseline warning (return pinned)
# =====================================================================

class TestCircuitBreakerDegenerateBaseline:
    def test_zero_last_equity_warns_but_returns_pinned_value(self, caplog):
        class _ZeroBaselineAPI:
            def get_account(self):
                return SimpleNamespace(equity='1000', last_equity='0')

        with caplog.at_level('WARNING', logger='order_utils'):
            result = check_circuit_breaker(_ZeroBaselineAPI())
        assert result == (False, 0.0)   # pinned, unchanged
        assert any('last_equity' in r.message for r in caplog.records)


# =====================================================================
# T12 — limit-order placers: post-submit log safety
# =====================================================================

class TestLimitOrderLogSafety:
    def test_missing_quote_keys_do_not_orphan_the_order(self):
        class _SubmitRecordingAPI:
            def __init__(self):
                self.submitted = []

            def submit_order(self, **kw):
                self.submitted.append(kw)
                return SimpleNamespace(id='o1', status='new')

        api = _SubmitRecordingAPI()
        result = place_limit_order(api, 'BTC/USD', 'buy', 1000.0,
                                   {'midpoint': 100.0})
        assert result is not None
        assert len(api.submitted) == 1

        api2 = _SubmitRecordingAPI()
        result2 = place_stock_limit_order(api2, 'AAPL', 'buy', 10,
                                          {'midpoint': 100.0})
        assert result2 is not None
        assert len(api2.submitted) == 1


# =====================================================================
# T13 — docstring-rot pins
# =====================================================================

class TestDocPinsV3:
    def test_make_client_order_id_admits_not_idempotent(self):
        assert 'NOT an idempotency' in ou.make_client_order_id.__doc__
        assert ou.make_client_order_id('x') != ou.make_client_order_id('x')

    def test_circuit_breaker_scope_and_baseline_documented(self):
        doc = ou.check_circuit_breaker.__doc__
        assert 'ACCOUNT-WIDE' in doc
        assert 'weekend' in doc

    def test_lifecycle_docstring_still_pinned(self):
        doc = ou.manage_order_lifecycle.__doc__
        assert 'canceled order' in doc and 'filled_qty' in doc

    def test_maker_buy_docstring_mentions_escalation(self):
        assert 'escalat' in ou.place_maker_buy.__doc__

    def test_emergency_flatten_documents_sentinel(self):
        assert '<list_positions failed>' in ou.emergency_flatten.__doc__
