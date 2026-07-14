"""Scheduled macro-event stand-down (FOMC statements, CPI prints).

Scheduled macro releases produce the sharpest intraday vol-and-reversal
episodes of the year (Lucca-Moench pre-FOMC drift collapses into the
announcement; CPI surprises gap both stocks AND crypto — BTC's largest
2022-2024 hourly moves cluster at 8:30 ET CPI prints). An hourly-bar
momentum model has no edge inside these windows; it just donates spread
to faster traders. New ENTRIES are blocked in a window around each
event — exits and protective stops keep running.

Dates are STATIC (verified against the Fed and BLS published 2026
schedules; BLS blocks bot fetches, so no live sync). Refresh annually
AND on Fed/BLS reschedule notices — a mid-year reschedule silently
invalidates a row (the Jan-2026-data CPI slipped Feb 11 -> Feb 13 after
the early-Feb-2026 shutdown, so the bot stood down on a non-event day
and traded into the real print). When refreshing, re-verify the
remaining year's dates too, not just the new year's. Sound the alarm in
logs when the table runs dry.

  - FOMC: statement 14:00 ET on day 2; presser to ~15:15 ET.
    Window: 12:00 -> 15:30 ET.
  - CPI: release 08:30 ET. Window: 06:30 -> 09:30 ET (covers the
    pre-positioning hour and the first post-print hour).
"""

import datetime as dt
import zoneinfo

_ET = zoneinfo.ZoneInfo('America/New_York')

# (year, month, day) of each event in 2026 — refresh every January.
FOMC_STATEMENT_DAYS = [
    (2026, 1, 28), (2026, 3, 18), (2026, 4, 29), (2026, 6, 17),
    (2026, 7, 29), (2026, 9, 16), (2026, 10, 28), (2026, 12, 9),
]
CPI_RELEASE_DAYS = [
    # Feb: BLS originally scheduled 02-11; the Jan-data print slipped to
    # Fri 02-13 after the early-Feb-2026 shutdown (cpi_02132026.htm).
    (2026, 1, 13), (2026, 2, 13), (2026, 3, 11), (2026, 4, 10),
    (2026, 5, 12), (2026, 6, 10), (2026, 7, 14), (2026, 8, 12),
    (2026, 9, 11), (2026, 10, 14), (2026, 11, 10), (2026, 12, 10),
]

# Stand-down windows in ET around each event type
_WINDOWS = [
    ('FOMC', FOMC_STATEMENT_DAYS, dt.time(12, 0), dt.time(15, 30)),
    ('CPI', CPI_RELEASE_DAYS, dt.time(6, 30), dt.time(9, 30)),
]


def macro_standdown(now: dt.datetime | None = None) -> tuple[bool, str | None]:
    """True (with a reason) when inside a macro-event entry stand-down."""
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    et = now.astimezone(_ET)
    today = (et.year, et.month, et.day)
    for label, days, start, end in _WINDOWS:
        if today in days and start <= et.time() < end:
            return True, (f"{label} stand-down "
                          f"({start.strftime('%H:%M')}-"
                          f"{end.strftime('%H:%M')} ET)")
    return False, None


def calendar_exhausted(now: dt.datetime | None = None) -> bool:
    """True when the static table has no future events (needs a refresh)."""
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    elif now.tzinfo is None:
        # Same guard as macro_standdown: a naive datetime would otherwise
        # be read as SYSTEM LOCAL time by astimezone (machine-dependent).
        now = now.replace(tzinfo=dt.timezone.utc)
    et = now.astimezone(_ET)
    last = max(max(FOMC_STATEMENT_DAYS), max(CPI_RELEASE_DAYS))
    return (et.year, et.month, et.day) > last
