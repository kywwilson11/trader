"""P4 (D07/B10) — earnings trading-day windows, default-OFF flag.

Pins:
  - default flag state is False (current calendar-day behavior, byte-compat).
  - flag ON walks weekend/NYSE-holiday-aware TRADING days in
    earnings_within_days / blocks_overnight_hold / reported_recently.
  - trading-day windows are always >= calendar windows (only ever block
    MORE / tilt MORE, never fewer) — the safety invariant from the spec.
  - a missing/deleted flag attribute falls back to calendar behavior (no crash).

Pure stdlib + pytest. No network, no cache-file IO: refresh_if_stale and
_mem are monkeypatched per the tests/test_grp_sentiment.py T6 pattern.
"""

import datetime
import types

import pytest

import events_calendar as ec
import strategy_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freeze(monkeypatch, iso):
    """Freeze ec.datetime.date.today() to the given ISO date string.

    Keeps .timedelta and .datetime real so downstream arithmetic still works.
    """
    frozen = datetime.date.fromisoformat(iso)

    class _FrozenDate(datetime.date):
        @classmethod
        def today(cls):
            return frozen

    fake_datetime = types.SimpleNamespace(
        date=_FrozenDate, timedelta=datetime.timedelta, datetime=datetime.datetime)
    monkeypatch.setattr(ec, 'datetime', fake_datetime)
    return frozen


def _cal(monkeypatch, sym_dates):
    """Inject a fake earnings calendar: sym_dates = {'AAPL': [{'date': ..., 'hour': ...}]}."""
    monkeypatch.setattr(ec, 'refresh_if_stale', lambda: True)
    monkeypatch.setattr(ec, '_mem', {'by_symbol': sym_dates})


def _flag(monkeypatch, on):
    monkeypatch.setattr(strategy_config, 'EVENTS_TRADING_DAY_WINDOWS', on, raising=False)


@pytest.fixture(autouse=True)
def _reset_mem(monkeypatch):
    yield
    # Safety net: monkeypatch already reverts attribute changes, but make the
    # cache-hygiene contract explicit and never leak into other test modules.
    monkeypatch.setattr(ec, '_mem', None, raising=False)


# ---------------------------------------------------------------------------
# 1 — flag default
# ---------------------------------------------------------------------------

def test_flag_default_off():
    assert strategy_config.EVENTS_TRADING_DAY_WINDOWS is False


# ---------------------------------------------------------------------------
# 2 — OFF mode pins current calendar-day semantics (both polarities)
# ---------------------------------------------------------------------------

def test_off_mode_pins_calendar_semantics(monkeypatch):
    _flag(monkeypatch, False)

    # Friday, AAPL reports Monday: calendar-day windows do NOT reach Monday.
    _freeze(monkeypatch, '2026-08-21')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-24'}]})
    assert ec.earnings_within_days('AAPL', days=1) is False
    assert ec.blocks_overnight_hold('AAPL') is False

    # Monday, AAPL reported Friday AMC: calendar window does not reach back.
    _freeze(monkeypatch, '2026-08-24')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-21', 'hour': 'amc'}]})
    assert ec.reported_recently('AAPL') is False

    # Midweek True cases (both polarities covered).
    _freeze(monkeypatch, '2026-08-19')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-20'}]})
    assert ec.earnings_within_days('AAPL', days=1) is True

    _freeze(monkeypatch, '2026-08-19')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-21'}]})
    assert ec.blocks_overnight_hold('AAPL') is True

    _freeze(monkeypatch, '2026-08-19')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-18', 'hour': 'amc'}]})
    assert ec.reported_recently('AAPL') is True


# ---------------------------------------------------------------------------
# 3 — flag ON: Friday protected against Monday earnings
# ---------------------------------------------------------------------------

def test_friday_before_monday_earnings_on(monkeypatch):
    _flag(monkeypatch, True)

    _freeze(monkeypatch, '2026-08-21')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-24'}]})
    assert ec.earnings_within_days('AAPL', days=1) is True
    assert ec.blocks_overnight_hold('AAPL') is True

    _freeze(monkeypatch, '2026-08-24')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-21', 'hour': 'amc'}]})
    assert ec.reported_recently('AAPL') is True

    # Weekend reporter (Saturday) also picked up on Monday.
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-22'}]})
    assert ec.reported_recently('AAPL') is True


# ---------------------------------------------------------------------------
# 3b — flag ON is still BOUNDED: widens the window, never degenerates to
#      always-True (guards against a broken gate that short-circuits)
# ---------------------------------------------------------------------------

def test_on_mode_still_bounded(monkeypatch):
    _flag(monkeypatch, True)

    # Friday, report next Wednesday: beyond the 1-trading-day horizon
    # (Mon 08-24) and beyond the +2-trading-day overnight buffer (Tue 08-25).
    _freeze(monkeypatch, '2026-08-21')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-26'}]})
    assert ec.earnings_within_days('AAPL', days=1) is False
    assert ec.blocks_overnight_hold('AAPL') is False

    # Monday, report last Thursday: before the trading-day lookback start
    # (Fri 08-21) -> no post-print tilt.
    _freeze(monkeypatch, '2026-08-24')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-20', 'hour': 'amc'}]})
    assert ec.reported_recently('AAPL') is False

    # Same-day AMC report: the inner d<today-or-bmo condition is unchanged
    # in ON mode — today's amc print is blocks_overnight_hold's job, not
    # reported_recently's.
    _freeze(monkeypatch, '2026-08-24')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-24', 'hour': 'amc'}]})
    assert ec.reported_recently('AAPL') is False
    assert ec.blocks_overnight_hold('AAPL') is True


# ---------------------------------------------------------------------------
# 4 — NYSE holiday (Labor Day) widens the trading-day window further
# ---------------------------------------------------------------------------

def test_holiday_monday(monkeypatch):
    # flag ON: Friday 09-04 -> next trading day is Tue 09-08 (Labor Day skipped).
    _flag(monkeypatch, True)
    _freeze(monkeypatch, '2026-09-04')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-09-08'}]})
    assert ec.earnings_within_days('AAPL', days=1) is True
    assert ec.blocks_overnight_hold('AAPL') is True

    # flag OFF: same dates, calendar days -> too far out.
    _flag(monkeypatch, False)
    _freeze(monkeypatch, '2026-09-04')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-09-08'}]})
    assert ec.earnings_within_days('AAPL', days=1) is False
    assert ec.blocks_overnight_hold('AAPL') is False

    # reported_recently: Tue 09-08 looking back at Fri 09-04 AMC report.
    _flag(monkeypatch, True)
    _freeze(monkeypatch, '2026-09-08')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-09-04', 'hour': 'amc'}]})
    assert ec.reported_recently('AAPL') is True

    _flag(monkeypatch, False)
    _freeze(monkeypatch, '2026-09-08')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-09-04', 'hour': 'amc'}]})
    assert ec.reported_recently('AAPL') is False


# ---------------------------------------------------------------------------
# 5 — midweek: trading-day mode is a pure no-op (no weekend/holiday to skip)
# ---------------------------------------------------------------------------

def test_midweek_equivalence(monkeypatch):
    report_dates = [
        (datetime.date.fromisoformat('2026-08-14') + datetime.timedelta(days=i)).isoformat()
        for i in range(11)  # 2026-08-14 .. 2026-08-24
    ]
    for report_date in report_dates:
        for hour in ('amc', 'bmo'):
            results = {}
            for on in (False, True):
                _flag(monkeypatch, on)
                _freeze(monkeypatch, '2026-08-19')
                _cal(monkeypatch, {'AAPL': [{'date': report_date, 'hour': hour}]})
                results[('within', on)] = ec.earnings_within_days('AAPL', days=1)
                _freeze(monkeypatch, '2026-08-19')
                _cal(monkeypatch, {'AAPL': [{'date': report_date, 'hour': hour}]})
                results[('overnight', on)] = ec.blocks_overnight_hold('AAPL')
                _freeze(monkeypatch, '2026-08-19')
                _cal(monkeypatch, {'AAPL': [{'date': report_date, 'hour': hour}]})
                results[('recent', on)] = ec.reported_recently('AAPL')
            assert results[('within', False)] == results[('within', True)], \
                (report_date, hour, 'earnings_within_days')
            assert results[('overnight', False)] == results[('overnight', True)], \
                (report_date, hour, 'blocks_overnight_hold')
            assert results[('recent', False)] == results[('recent', True)], \
                (report_date, hour, 'reported_recently')


# ---------------------------------------------------------------------------
# 6 — trading-day helper unit checks
# ---------------------------------------------------------------------------

def test_trading_day_helpers():
    d = datetime.date.fromisoformat
    assert ec._add_trading_days(d('2026-08-21'), 1) == d('2026-08-24')
    assert ec._add_trading_days(d('2026-09-04'), 1) == d('2026-09-08')  # holiday skip
    assert ec._add_trading_days(d('2026-08-21'), 2) == d('2026-08-25')
    assert ec._add_trading_days(d('2026-08-19'), 2) == d('2026-08-21')
    assert ec._prev_trading_day(d('2026-08-24')) == d('2026-08-21')
    assert ec._prev_trading_day(d('2026-09-08')) == d('2026-09-04')
    assert ec._is_trading_day(d('2026-07-03')) is False  # July-4 observed
    assert ec._is_trading_day(d('2026-08-19')) is True
    assert ec._is_trading_day(d('2026-08-22')) is False  # Saturday


# ---------------------------------------------------------------------------
# 7 — property pin: trading-day windows are always >= calendar windows
# ---------------------------------------------------------------------------

def test_trading_horizon_dominates_calendar():
    start = datetime.date.fromisoformat('2026-08-17')
    end = datetime.date.fromisoformat('2026-09-13')
    d = start
    while d <= end:
        for n in (1, 2, 3):
            assert ec._add_trading_days(d, n) >= d + datetime.timedelta(days=n), d
        assert ec._prev_trading_day(d) <= d - datetime.timedelta(days=1), d
        d += datetime.timedelta(days=1)


# ---------------------------------------------------------------------------
# 8 — missing flag attribute falls back to calendar behavior, no crash
# ---------------------------------------------------------------------------

def test_missing_flag_falls_back_to_calendar(monkeypatch):
    monkeypatch.delattr(strategy_config, 'EVENTS_TRADING_DAY_WINDOWS', raising=False)
    _freeze(monkeypatch, '2026-08-21')
    _cal(monkeypatch, {'AAPL': [{'date': '2026-08-24'}]})
    assert ec.earnings_within_days('AAPL', days=1) is False
