"""Likely-shortable universe filter from FREE data (wave-7, Finding 5).

An offline short backtest is FICTION until it stops crediting un-shortable
names: the MPP-2025 trap is that anomaly-short alpha concentrates in exactly
the hard-to-borrow (HTB) names a retail broker like Alpaca cannot reliably
short. There is no free, point-in-time borrow-availability feed, so we PROXY
it conservatively from the supply side — Markit's decomposition finds borrow
availability is dominated by SUPPLY (~13%, ~ inverse of market cap / size)
far more than demand (~1%). The rule is exclude-when-uncertain: it can only
fail to PREVENT phantom alpha, never invent it.

`/float` and IPO-recency are deliberately treated as LOW-confidence here
(a current yfinance snapshot is not point-in-time), so the score leans on the
market-cap bucket + the in-repo name_class, both of which we already have.
The market-cap input carries the SAME caveat: cap_lookup PIT-ness is the
CALLER's responsibility — a current-snapshot cap fed into a historical sim can
misclassify upward re-raters (small-cap then, large-cap now) as historically
borrowable. Prefer as-of caps when available.
"""

import logging

# Speculative / pre-profit classes (stock_config.SECTOR_BUCKETS) that lean HTB
# regardless of a transiently large cap — these are the meme/moonshot names
# where borrow dries up exactly when you want to short them.
HTB_LEANING_CLASSES = frozenset({'spec_growth'})

# Market-cap buckets (USD). Supply proxy: bigger float -> easier to borrow.
LARGE_CAP = 10e9
MID_CAP = 2e9
SMALL_CAP = 5e8

# htb_risk_score >= this -> treat as NOT reliably shortable (exclude).
HTB_EXCLUDE_THRESHOLD = 0.6


def htb_risk_score(market_cap=None, name_class=None):
    """Hard-to-borrow risk in [0, 1]; 0 = trivially borrowable, 1 = no-go.

    Driven primarily by the market-cap (supply) bucket; a speculative
    name_class floors the score upward. A MISSING cap leans HTB (0.7) — the
    conservative default, since we'd rather drop a borrowable name than
    backtest a short we could never put on.
    """
    # `not (cap > 0)` rather than `cap <= 0`: NaN caps (what a pandas-sourced
    # lookup yields for missing data) must also route to the missing default.
    if market_cap is None or not (market_cap > 0):
        base = 0.7
    elif market_cap >= LARGE_CAP:
        base = 0.05
    elif market_cap >= MID_CAP:
        base = 0.25
    elif market_cap >= SMALL_CAP:
        base = 0.6
    else:
        base = 0.9
    if name_class in HTB_LEANING_CLASSES:
        base = max(base, 0.6)
    return float(base)


def likely_shortable(symbol, market_cap=None, name_class=None,
                     threshold=HTB_EXCLUDE_THRESHOLD):
    """Conservative bool: is `symbol` probably ETB enough to short offline?

    Returns True only when htb_risk_score is strictly BELOW the exclude
    threshold — uncertain / HTB / speculative names return False by design.
    `symbol` is currently unused (kept for call-site clarity and future
    per-symbol overrides); the proxy is deliberately time-invariant (no
    `asof`) because no free point-in-time borrow history exists.
    """
    return htb_risk_score(market_cap, name_class) < threshold


def restrict_short_universe(symbols, cap_lookup=None, class_lookup=None):
    """Filter a symbol list down to the probable-ETB short set.

    cap_lookup(symbol)->market_cap and class_lookup(symbol)->name_class are
    optional callables; when omitted, names fall to the conservative
    missing-cap default (likely excluded). cap_lookup PIT-ness is the
    caller's responsibility (see module docstring): pass as-of caps for
    historical sims where available. Returns the kept list.
    """
    kept = []
    for s in symbols:
        mc = cap_lookup(s) if cap_lookup else None
        nc = class_lookup(s) if class_lookup else None
        if likely_shortable(s, mc, nc):
            kept.append(s)
    return kept


def class_lookup_from_config():
    """Convenience: a class_lookup backed by stock_config.SECTOR_BUCKETS."""
    try:
        from stock_config import SECTOR_BUCKETS
        return lambda s: SECTOR_BUCKETS.get(s)
    except Exception as exc:
        # fail-OPEN degradation (spec_growth names lose their HTB floor and
        # can pass on cap alone) — must never happen silently.
        logging.getLogger(__name__).warning(
            'stock_config.SECTOR_BUCKETS unavailable (%r); HTB class floor '
            'disabled — proxy degrades to cap-only', exc)
        return lambda s: None
