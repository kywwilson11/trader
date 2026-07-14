"""Tests for the order-path group's 2026-07 review edits (order_utils.py):

  EDIT 1  manage_order_lifecycle keeps final_order when the fully-filled-
          during-cancel remaining<=0 re-fetch fails (was: silently None).
  EDIT 2  _is_not_found classifier helper (not exercised directly here beyond
          its effect on EDIT 3/4's warning gating).
  EDIT 3  verify_position warns on transient (non-not-found) errors; return
          contract (None) is unchanged.
  EDIT 4  reconstruct_positions tolerates current_price=None (alpaca-py shim)
          and warns on transient errors while staying silent on not-found.
"""

import time
from types import SimpleNamespace

import pytest

from order_utils import manage_order_lifecycle, reconstruct_positions, verify_position


@pytest.fixture(autouse=True)
def fast_clock(monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda s: None)


# --- EDIT 1: fully-filled-during-cancel re-fetch failure keeps final_order ---

class _StuckPartialAPI:
    """The realistic GTC bid-join sequence: the order sits partially_filled
    for the whole poll window (no spontaneous terminal state) and only goes
    canceled when WE cancel it — filled_qty survives the cancel."""

    def __init__(self, qty=10.0, filled=6.0, px=100.0, fetch_fail_from=None):
        self.order = SimpleNamespace(id='o1', symbol='BTC/USD', qty=qty,
                                     side='buy', status='partially_filled',
                                     filled_qty=filled, filled_avg_price=px)
        self.canceled = []
        self.submitted = []
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

    def submit_order(self, **kw):
        self.submitted.append(kw)
        raise AssertionError('no fallback expected')


class TestFullyFilledDuringCancelReFetch:
    def test_fully_filled_during_cancel_keeps_final_order(self):
        # qty=10, filled=10 for calls 1-3 (2 polls + final_order fetch); the
        # remaining<=0 re-fetch (call 4) fails -> EDIT 1 returns final_order
        # instead of discarding it.
        api = _StuckPartialAPI(qty=10.0, filled=10.0, fetch_fail_from=4)
        result = manage_order_lifecycle(api, 'o1', timeout=4, poll_interval=2,
                                        fallback_to_market=True)
        assert result is not None
        assert result.status == 'canceled'
        assert float(result.filled_qty) == 10.0
        assert api.submitted == []

    def test_fully_filled_during_cancel_all_post_fetches_fail_stays_none(self):
        # 2 polls succeed (saved_filled=10 cached in-loop), but both the
        # final_order fetch (call 3) AND the remaining<=0 re-fetch (call 4)
        # fail -> nothing was ever held -> result stays None.
        api = _StuckPartialAPI(qty=10.0, filled=10.0, fetch_fail_from=3)
        result = manage_order_lifecycle(api, 'o1', timeout=4, poll_interval=2,
                                        fallback_to_market=True)
        assert result is None
        assert api.submitted == []


# --- EDIT 3 / EDIT 4: transient-error visibility + None current_price guard ---

class _PosAPI:
    """Model on tests/test_review_b02.py::_PosAPI. `book` maps spelled symbol
    -> a dict of position attrs, or the fake can be built to always raise."""

    def __init__(self, book=None, raise_exc=None):
        self.book = book or {}
        self.raise_exc = raise_exc

    def get_position(self, sym):
        if self.raise_exc is not None:
            raise self.raise_exc
        if sym not in self.book:
            raise RuntimeError('position does not exist')
        info = self.book[sym]
        return SimpleNamespace(symbol=sym, qty=info['qty'],
                               avg_entry_price=info['avg_entry_price'],
                               current_price=info['current_price'])


class TestReconstructPositionsGuards:
    def test_reconstruct_tracks_position_with_none_current_price(self):
        api = _PosAPI({'BTCUSD': {'qty': 2.0, 'avg_entry_price': 100.0,
                                  'current_price': None}})
        out = reconstruct_positions(api, ['BTC/USD'])
        assert out == {'BTC/USD': {'qty': 2.0, 'entry_price': 100.0,
                                   'high_water_mark': 100.0}}

    def test_reconstruct_transient_error_warns_but_skips(self, caplog):
        api = _PosAPI(raise_exc=RuntimeError('rate limited 429'))
        with caplog.at_level('WARNING', logger='order_utils'):
            out = reconstruct_positions(api, ['BTC/USD'])
        assert out == {}
        assert any('[RECONSTRUCT]' in r.message for r in caplog.records)

    def test_reconstruct_not_found_stays_silent(self, caplog):
        api = _PosAPI(raise_exc=RuntimeError('position does not exist'))
        with caplog.at_level('WARNING', logger='order_utils'):
            out = reconstruct_positions(api, ['BTC/USD'])
        assert out == {}
        assert not any(r.levelname == 'WARNING' for r in caplog.records
                       if r.name == 'order_utils')


class TestVerifyPositionGuards:
    def test_verify_position_transient_error_warns_returns_none(self, caplog):
        api = _PosAPI(raise_exc=RuntimeError('timeout'))
        with caplog.at_level('WARNING', logger='order_utils'):
            result = verify_position(api, 'BTC/USD')
        assert result is None
        assert any('[VERIFY]' in r.message for r in caplog.records)

    def test_verify_position_not_found_stays_silent(self, caplog):
        api = _PosAPI(raise_exc=RuntimeError('position does not exist'))
        with caplog.at_level('WARNING', logger='order_utils'):
            result = verify_position(api, 'BTC/USD')
        assert result is None
        assert not any(r.levelname == 'WARNING' for r in caplog.records
                       if r.name == 'order_utils')
