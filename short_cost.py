"""Regime-dated short-side cost model (wave-7, Finding 5).

A short's all-in cost is the round-trip spread + fees (same as a long) PLUS a
borrow-fee DRAG that accrues per day held. The trap this module avoids is
back-charging today's economics onto historical sims: Alpaca moved to $0
borrow on its Easy-To-Borrow set on 2025-10-01, so giving a June-2024 short
$0 borrow would itself be a look-ahead. Borrow cost is therefore REGIME-DATED.

In the $0-ETB regime the dominant short cost is the SPREAD (the wave-6 per-name
EDGE), not borrow — so this refocuses short cost onto the spread + fees and
treats borrow as a small, regime/HTB-conditional add-on. HTB names are pushed
out entirely by borrow_proxy.likely_shortable BEFORE this is ever called, so
the HTB schedule here is a conservative backstop, not a green light.

Also exposes point-in-time FINRA bi-monthly Short-Interest metrics (SI/float,
days-to-cover) — mapped by PUBLICATION date with a shift(1), the same PIT
discipline as the daily SVR pipeline in short_flow.py.
"""

from datetime import date, datetime

import numpy as np

# Alpaca dropped to $0 borrow on its ETB set here. Before this, assume a
# conservative ETB schedule; do NOT retroactively grant $0 to older sims.
BORROW_REGIME_START = date(2025, 10, 1)

PRE_REGIME_ETB_BPS = 30.0     # conservative pre-$0-regime ETB annual borrow
HTB_BPS = 300.0               # conservative annual borrow for a HTB backstop
_DAYS_PER_YEAR = 365.0


def _to_date(asof):
    if isinstance(asof, datetime):       # also covers pd.Timestamp
        return asof.date()
    if isinstance(asof, date):
        return asof
    try:
        import pandas as pd
        return pd.Timestamp(asof).date()
    except Exception:
        return date.fromisoformat(str(asof)[:10])


def borrow_cost_bps_annual(asof, likely_etb=True, htb_score=None):
    """Annual borrow fee (bps). Regime-dated: $0 for ETB on/after the regime
    start, a conservative ETB schedule before, and an HTB backstop (scaled by
    htb_score in [0,1]) when the name is not ETB."""
    d = _to_date(asof)
    if likely_etb:
        return 0.0 if d >= BORROW_REGIME_START else PRE_REGIME_ETB_BPS
    return HTB_BPS * (float(htb_score) if htb_score is not None else 1.0)


def borrow_drag_pct(asof, hold_days, likely_etb=True, htb_score=None):
    """Borrow drag as a PERCENT of notional over `hold_days` (accrues daily)."""
    bps = borrow_cost_bps_annual(asof, likely_etb, htb_score)
    return bps / 100.0 * max(float(hold_days), 0.0) / _DAYS_PER_YEAR


def short_round_trip_cost_pct(asof, eff_spread_pct, hold_days=1.0,
                              likely_etb=True, htb_score=None):
    """All-in short round-trip cost (PERCENT): spread + fees + borrow drag.

    Reuses fees.round_trip_cost_pct('stock', spread) for the spread+fee base
    (identical to a long) and adds the regime-dated borrow drag for the hold.
    Mirrors the long cost convention so offline short P&L is comparable.
    """
    from fees import round_trip_cost_pct
    base = round_trip_cost_pct('stock', max(float(eff_spread_pct), 0.0))
    return base + borrow_drag_pct(asof, hold_days, likely_etb, htb_score)


# ---------------------------------------------------------------------------
# FINRA bi-monthly Short Interest — PIT metrics (publication-date mapped)
# ---------------------------------------------------------------------------

def short_interest_metrics(short_interest, float_shares=None, adv_20d=None):
    """Per-print SI metrics. days_to_cover = SI / 20d ADV; si_pct_float =
    SI / float (LOW-confidence: float from a current snapshot is not PIT, so
    callers should treat it as soft). Returns a dict; None where inputs missing.
    """
    si = float(short_interest) if short_interest is not None else None
    dtc = (si / float(adv_20d)) if (si is not None and adv_20d and adv_20d > 0) else None
    si_pct = (si / float(float_shares)) if (si is not None and float_shares
                                            and float_shares > 0) else None
    return {'short_interest': si, 'days_to_cover': dtc,
            'si_pct_float': si_pct}


def pit_publication_map(values_by_pub_date, index):
    """Map bi-monthly SI values onto an intraday bar index, point-in-time.

    Each bar sees the most recent print whose PUBLICATION date is STRICTLY
    BEFORE that bar (shift-1: a print published on day P is used from P+1
    onward, never intraday on P). Because SI is sparse (bi-monthly) this is an
    AS-OF join, not the exact-date map the daily SVR pipeline can use.

    Args:
        values_by_pub_date: pandas Series indexed by publication date.
        index: the intraday DatetimeIndex to align onto.
    Returns a numpy array aligned to `index` (NaN before the first usable print).
    """
    import pandas as pd
    s = pd.Series(values_by_pub_date).sort_index()
    pub = pd.DatetimeIndex(s.index).normalize()
    bars = pd.DatetimeIndex(index).normalize()
    # last publication strictly before each bar (side='left' -> a bar ON a
    # publication date sees the PRIOR print, i.e. the shift-1 guard)
    pos = pub.searchsorted(bars, side='left') - 1
    out = np.full(len(bars), np.nan, dtype=float)
    valid = pos >= 0
    out[valid] = np.asarray(s.values, dtype=float)[pos[valid]]
    return out
