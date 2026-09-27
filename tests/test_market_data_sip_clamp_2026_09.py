"""SIP recent-data end clamp for the stock historical harvest (2026-09).

Defect (C_data blocker 1): `fetch_historical_bars` sent end=now; Alpaca's
Basic plan rejects any stock request whose end is inside the 15-min SIP delay
window ("subscription does not permit querying recent SIP data") for the
WHOLE chunk, and the final chunk was swallowed without retry — a full stock
rebuild lost its newest ~6-month chunk and every weekly incremental returned
nothing. Fix: clamp the stock `end` to now - SIP_RECENT_DELAY_MIN (16 min).

REVIEW M1 (2026-09-26): inside the extended session (04:00-20:00 ET, Mon-Fri)
the clamp alone lets the still-FORMING hourly bar into the store, and its
partial close then trips the 1% overlap merge guard on the next incremental.
The end is additionally floored to floor_hour(limit) - 1s there, so every
returned bar has open + 1h <= limit. Outside the session nothing changes.

Mac-safe: the Alpaca API is a plain stub object; alpaca is never imported.
"""

import inspect
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import market_data
from market_data import SIP_RECENT_DELAY_MIN, _clamp_sip_end


NOW = datetime(2026, 9, 26, 20, 0, tzinfo=timezone.utc)   # a SATURDAY
CLAMP = NOW - timedelta(minutes=16)


def _last_bar_open_bound(now):
    """Newest bar open a stock fetch may return at wall-clock `now`: the bar
    must be complete (open + 1h <= now - 16 min) and not newer than the
    clamp. Used by the real-clock stub tests, which run at any hour."""
    end = _clamp_sip_end(now + timedelta(days=1), 'stock', now=now)
    return pd.Timestamp(end).floor('h')


# ---------------------------------------------------------------------------
# (a) clamp math
# ---------------------------------------------------------------------------

class TestClampMath:
    def test_constant(self):
        assert SIP_RECENT_DELAY_MIN == 16

    @pytest.mark.parametrize('end', [
        NOW + timedelta(hours=1),        # future
        NOW,                             # now
        NOW - timedelta(minutes=5),      # 5 min ago (inside delay window)
        NOW - timedelta(minutes=15),     # exactly the documented delay
    ])
    def test_recent_stock_end_is_clamped(self, end):
        assert _clamp_sip_end(end, 'stock', now=NOW) == CLAMP

    @pytest.mark.parametrize('end', [
        NOW - timedelta(days=1),
        NOW - timedelta(minutes=16),     # exactly at the clamp: unchanged
        datetime(2021, 4, 1, tzinfo=timezone.utc),
    ])
    def test_old_stock_end_untouched(self, end):
        assert _clamp_sip_end(end, 'stock', now=NOW) == end

    @pytest.mark.parametrize('end', [NOW + timedelta(hours=1), NOW,
                                     NOW - timedelta(minutes=5)])
    def test_crypto_never_clamped(self, end):
        assert _clamp_sip_end(end, 'crypto', now=NOW) == end

    def test_default_now_is_utc_wallclock(self):
        before = datetime.now(timezone.utc)
        far = before + timedelta(days=1)
        got = _clamp_sip_end(far, 'stock')
        after = datetime.now(timezone.utc)
        # Equal to the explicit-now result for a wall-clock `now` between
        # the two reads (16-min clamp, hour-floored in-session).
        lo = _clamp_sip_end(far, 'stock', now=before)
        hi = _clamp_sip_end(far, 'stock', now=after)
        assert lo <= got <= hi
        lag = (after - got).total_seconds()
        assert 16 * 60 - 5 <= lag <= 16 * 60 + 3600 + 60


# ---------------------------------------------------------------------------
# (a2) REVIEW M1: in-session hour floor (no forming bar in the store)
# ---------------------------------------------------------------------------

ET = ZoneInfo('America/New_York')


def _et(*a):
    return datetime(*a, tzinfo=ET)


class TestInSessionHourFloor:
    def test_rth_tuesday_floors_to_last_completed_bar(self):
        # 15:37 ET Tuesday -> clamp 15:21 ET; the 15:00 bar is still forming
        # -> end = 14:59:59 ET (last completed bar = 14:00-15:00).
        now = _et(2026, 9, 22, 15, 37).astimezone(timezone.utc)
        got = _clamp_sip_end(now + timedelta(hours=1), 'stock', now=now)
        assert got == _et(2026, 9, 22, 14, 59, 59)
        assert got.astimezone(timezone.utc) == datetime(
            2026, 9, 22, 18, 59, 59, tzinfo=timezone.utc)   # EDT
        # also when the caller's end already lies inside the forming bar
        mid = _et(2026, 9, 22, 15, 10).astimezone(timezone.utc)
        assert _clamp_sip_end(mid, 'stock', now=now) == got
        # a completed-bar end is untouched
        old = _et(2026, 9, 22, 14, 0).astimezone(timezone.utc)
        assert _clamp_sip_end(old, 'stock', now=now) == old

    @pytest.mark.parametrize('now_et, want_et', [
        ((2026, 1, 13, 10, 5), (2026, 1, 13, 8, 59, 59)),    # EST; clamp 9:49
        ((2026, 9, 22, 4, 20), (2026, 9, 22, 3, 59, 59)),    # clamp 4:04 (pre)
        ((2026, 9, 22, 20, 10), (2026, 9, 22, 18, 59, 59)),  # clamp 19:54
        ((2026, 9, 22, 15, 16), (2026, 9, 22, 14, 59, 59)),  # clamp 15:00:00
    ])
    def test_session_edges_and_est(self, now_et, want_et):
        now = _et(*now_et).astimezone(timezone.utc)
        got = _clamp_sip_end(now + timedelta(days=1), 'stock', now=now)
        assert got == _et(*want_et)
        # the invariant: the newest bar the request can return is complete
        newest_open = pd.Timestamp(got).floor('h')
        assert (newest_open + pd.Timedelta(hours=1)
                <= pd.Timestamp(now) - pd.Timedelta(minutes=16))

    @pytest.mark.parametrize('now_et', [
        (2026, 9, 22, 20, 16),   # clamp exactly 20:00 ET: session over
        (2026, 9, 22, 21, 37),   # after the extended close
        (2026, 9, 23, 3, 30),    # before the pre-market open
        (2026, 9, 26, 15, 37),   # Saturday
        (2026, 9, 27, 11, 0),    # Sunday
    ])
    def test_outside_session_plain_clamp_unchanged(self, now_et):
        now = _et(*now_et).astimezone(timezone.utc)
        want = now - timedelta(minutes=16)
        assert _clamp_sip_end(now + timedelta(hours=2), 'stock',
                              now=now) == want
        assert _clamp_sip_end(now, 'stock', now=now) == want
        older = now - timedelta(minutes=30)
        assert _clamp_sip_end(older, 'stock', now=now) == older

    def test_crypto_untouched_in_session(self):
        now = _et(2026, 9, 22, 15, 37).astimezone(timezone.utc)
        for end in (now, now + timedelta(hours=1), now - timedelta(minutes=5)):
            assert _clamp_sip_end(end, 'crypto', now=now) == end

    def test_fetch_during_rth_returns_no_forming_bar(self, monkeypatch):
        # End-to-end through fetch_historical_bars with a frozen RTH clock:
        # the stub serves every bar whose open <= end (Alpaca semantics),
        # incl. the forming 15:00 ET bar if asked for it.
        now = _et(2026, 9, 22, 15, 37).astimezone(timezone.utc)

        class _FrozenDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return now if tz is not None else now.replace(tzinfo=None)
        import datetime as _dtmod
        monkeypatch.setattr(_dtmod, 'datetime', _FrozenDT)
        monkeypatch.setattr(market_data.time, 'sleep', lambda s: None)
        calls = []

        class _Api:
            def get_bars(self, symbol, tf, start=None, end=None, **kw):
                calls.append(end)
                idx = pd.date_range(pd.Timestamp(start).ceil('h'),
                                    pd.Timestamp(end), freq='1h')
                return [SimpleNamespace(t=t, o=1.0, h=1.0, l=1.0, c=1.0,
                                        v=10.0) for t in idx]
        df = market_data.fetch_historical_bars(_Api(), 'AMD', '2026-09-20',
                                               asset_type='stock')
        assert pd.Timestamp(calls[-1]) == pd.Timestamp('2026-09-22 18:59:59',
                                                       tz='UTC')
        assert df.index.max() == pd.Timestamp('2026-09-22 18:00', tz='UTC')
        assert (df.index.max() + pd.Timedelta(hours=1)
                <= pd.Timestamp(now) - pd.Timedelta(minutes=16))


# ---------------------------------------------------------------------------
# (b) stubbed API that enforces the SIP delay
# ---------------------------------------------------------------------------

class _SipApi:
    """get_bars raises the Basic-plan error for any end within 15 min of the
    real wall clock; otherwise returns hourly bars covering [start, end)."""

    SIP_ERR = 'subscription does not permit querying recent SIP data'

    def __init__(self):
        self.stock_calls = []
        self.crypto_calls = []

    @staticmethod
    def _bars(start, end):
        s = pd.Timestamp(start).ceil('h')
        e = pd.Timestamp(end)
        idx = pd.date_range(s, e, freq='1h', inclusive='left')
        return [SimpleNamespace(t=t, o=1.0, h=1.0, l=1.0, c=1.0, v=10.0)
                for t in idx]

    def get_bars(self, symbol, timeframe, start=None, end=None,
                 adjustment=None, **kw):
        self.stock_calls.append((start, end))
        end_ts = pd.Timestamp(end)
        if end_ts > pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=15):
            raise Exception(self.SIP_ERR)
        return self._bars(start, end)

    def get_crypto_bars(self, symbol, timeframe, start=None, end=None, **kw):
        self.crypto_calls.append((start, end))
        return self._bars(start, end)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(market_data.time, 'sleep', lambda s: None)


class TestStubbedSipFetch:
    def test_single_chunk_incremental_returns_bars(self, no_sleep):
        # The weekly-incremental shape: one chunk ending at "now". Pre-fix
        # this chunk was denied and the function returned None.
        api = _SipApi()
        start = (datetime.now(timezone.utc) - timedelta(days=7)).date()
        df = market_data.fetch_historical_bars(api, 'AMD', str(start),
                                               asset_type='stock')
        assert df is not None and not df.empty
        # newest bar is the newest COMPLETED one the clamp allows (<= 1 bar
        # of slack for a wall-clock hour rollover during the test)
        bound = _last_bar_open_bound(datetime.now(timezone.utc))
        assert bound - pd.Timedelta(hours=1) <= df.index.max() <= bound
        lag = pd.Timestamp.now(tz='UTC') - df.index.max()
        assert lag < pd.Timedelta(hours=2, minutes=20)
        # every stock request honoured the delay window
        now = pd.Timestamp.now(tz='UTC')
        for _, end in api.stock_calls:
            assert pd.Timestamp(end) <= now - pd.Timedelta(minutes=15)

    def test_multi_chunk_rebuild_returns_full_range(self, no_sleep):
        # Full-rebuild shape: several 6-month chunks; pre-fix the final one
        # (the newest months) was dropped.
        api = _SipApi()
        start = (datetime.now(timezone.utc) - timedelta(days=400)).date()
        df = market_data.fetch_historical_bars(api, 'AMD', str(start),
                                               asset_type='stock')
        assert len(api.stock_calls) >= 3
        assert df.index.min() == pd.Timestamp(str(start), tz='UTC')
        bound = _last_bar_open_bound(datetime.now(timezone.utc))
        assert bound - pd.Timedelta(hours=1) <= df.index.max() <= bound
        assert (pd.Timestamp.now(tz='UTC') - df.index.max()
                < pd.Timedelta(hours=2, minutes=20))
        # contiguous hourly coverage: nothing dropped between chunks
        full = pd.date_range(df.index.min(), df.index.max(), freq='1h')
        assert len(full.difference(df.index)) == 0

    def test_crypto_end_not_clamped(self, no_sleep):
        api = _SipApi()
        start = (datetime.now(timezone.utc) - timedelta(days=3)).date()
        before = datetime.now(timezone.utc)
        market_data.fetch_historical_bars(api, 'BTC/USD', str(start),
                                          asset_type='crypto')
        last_end = datetime.fromisoformat(api.crypto_calls[-1][1])
        assert last_end >= before - timedelta(seconds=1)

    def test_old_end_date_bounds_unchanged(self, no_sleep):
        api = _SipApi()
        market_data.fetch_historical_bars(api, 'AMD', '2021-01-01',
                                          asset_type='stock',
                                          end_date='2021-04-01')
        assert datetime.fromisoformat(api.stock_calls[-1][1]) == \
            datetime(2021, 4, 1, tzinfo=timezone.utc)

    def test_denied_chunk_warns_with_bounds_and_keeps_earlier(
            self, no_sleep, monkeypatch, caplog):
        # If a denial still happens (e.g. a plan change), it must be loud and
        # must not take the earlier chunks with it.
        api = _SipApi()
        orig = api.get_bars
        deny_from = pd.Timestamp('2021-07-01', tz='UTC')

        def get_bars(symbol, timeframe, start=None, end=None, **kw):
            if pd.Timestamp(start) >= deny_from:
                api.stock_calls.append((start, end))
                raise Exception(_SipApi.SIP_ERR)
            return orig(symbol, timeframe, start=start, end=end, **kw)
        monkeypatch.setattr(api, 'get_bars', get_bars)
        with caplog.at_level(logging.WARNING, logger='market_data'):
            df = market_data.fetch_historical_bars(
                api, 'AMD', '2021-01-01', asset_type='stock',
                end_date='2022-01-01')
        assert df is not None
        assert df.index.min() == pd.Timestamp('2021-01-01', tz='UTC')
        assert df.index.max() < deny_from
        msgs = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING]
        assert any('subscription denied' in m and '2021-07-01' in m
                   and '2022-01-01' in m for m in msgs)
        assert any('TAIL' in m and 'MISSING' in m for m in msgs)


# ---------------------------------------------------------------------------
# (c) source-text pins
# ---------------------------------------------------------------------------

def mod_src():
    return inspect.getsource(market_data)


def test_source_pins_clamp_in_stock_path():
    src = inspect.getsource(market_data.fetch_historical_bars)
    assert '_clamp_sip_end(end_dt, asset_type' in src
    helper = inspect.getsource(market_data._clamp_sip_end)
    assert 'SIP_RECENT_DELAY_MIN' in helper
    assert "asset_type == 'crypto'" in helper
    # REVIEW M1: in-session hour floor to the last completed bar
    assert 'timedelta(seconds=1)' in helper
    assert "_SIP_SESSION_TZ = 'America/New_York'" in mod_src()
    mod = inspect.getsource(market_data)
    assert 'SIP_RECENT_DELAY_MIN = 16' in mod
