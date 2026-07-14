"""Overnight forfeited-drift + GTC gap-through audit (wave-7, overlay decision).

stock_loop flattens 100% at 15:50 and keeps only a tiny GTC-stop sleeve. A GTC
stop fills at the gapped OPEN, so it gives ZERO protection against the
overnight tail. Nothing in the system measures (a) the overnight drift the
flatten forfeits, or (b) how often / how badly an overnight gap blows through a
stop. This module measures both from FREE yfinance daily OHLC, so the overnight
overlay (options_overlay.py) is decided on data, not priors.

Pairs with options_overlay: gap_audit sizes the PROBLEM (annual $ of forfeited
drift + gap-through loss); options_overlay prices the PROPOSED FIX's friction.
On plausible numbers the gap-through term (~$300-450/yr on the 2-name sleeve)
sits far below even the cheapest overlay's friction -> a clean NO-GO.

The analysis functions are pure (take return series) and unit-tested on
synthetic Student-t; only fetch_daily() touches the network (lazy yfinance).
PIT-honest: gap stats are computed on .shift(1)-lagged history and the
sleeve-eligible set should come from panel_ranks as-of membership, never the
survivorship-biased surviving-names list.
"""

import numpy as np

TRADING_DAYS = 252.0


def overnight_intraday_returns(daily_df):
    """Split daily bars into overnight and intraday return series.

    overnight[t] = open[t]/close[t-1] - 1   (the gap the flatten forfeits)
    intraday[t]  = close[t]/open[t]   - 1   (what the day session captures)
    Returns (overnight, intraday) as numpy arrays aligned to rows 1..n-1.
    """
    o = np.asarray(daily_df['Open'], dtype=float)
    c = np.asarray(daily_df['Close'], dtype=float)
    if len(c) < 2:
        return np.array([]), np.array([])
    overnight = o[1:] / c[:-1] - 1.0
    intraday = c[1:] / o[1:] - 1.0
    m = np.isfinite(overnight) & np.isfinite(intraday)
    return overnight[m], intraday[m]


def gap_stats(overnight):
    """Distribution of overnight gaps: std, skew, excess kurtosis, Student-t df.

    Heavy tails (low t df, high excess kurtosis) are exactly the overnight
    risk the flatten avoids and a GTC stop cannot. Returns a dict.
    """
    r = np.asarray(overnight, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20:
        return {'n': int(n), 'std': None, 'skew': None,
                'excess_kurtosis': None, 't_df': None}
    mu = r.mean()
    sd = r.std()
    x = (r - mu)
    skew = float((x ** 3).mean() / (sd ** 3 + 1e-18))
    kurt = float((x ** 4).mean() / (sd ** 4 + 1e-18)) - 3.0
    t_df = None
    try:
        from scipy.stats import t as student_t
    except ImportError:
        student_t = None    # scipy optional — t_df stays None, silently
    if student_t is not None:
        try:
            df, _, _ = student_t.fit(r, floc=mu)
            t_df = float(df)
        except Exception as e:
            # A genuine fit failure must be visible to the operator, not
            # indistinguishable from scipy-not-installed.
            print(f"gap_stats: Student-t fit failed "
                  f"({type(e).__name__}: {e}) — t_df=None")
    return {'n': int(n), 'std': float(sd), 'skew': skew,
            'excess_kurtosis': kurt, 't_df': t_df}


def forfeited_drift_annual(overnight, notional):
    """Annual $ drift forfeited by flattening (signed).

    Positive => flattening gives up real overnight drift (a cost of the
    flatten). Negative => the overnight session is net-negative and flattening
    AVOIDS loss (a benefit). = mean(overnight) * notional * trading_days.
    """
    r = np.asarray(overnight, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return None
    return float(r.mean() * notional * TRADING_DAYS)


def gap_through_cost_annual(overnight, stop_dist_frac, notional,
                            side='long'):
    """Annual $ EXCESS loss beyond a GTC stop from adverse overnight gaps.

    The stop distance is lost whether or not the gap blows through it, so the
    INCREMENTAL cost of holding-with-a-stop vs flattening is only the loss
    BEYOND the stop: E[max(0, |gap| - stop_dist)] over adverse gaps. A long is
    hurt by gap-DOWNs; a short by gap-UPs.
    """
    r = np.asarray(overnight, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20 or stop_dist_frac <= 0:
        return None
    adverse = -r if side == 'long' else r           # positive = adverse move
    excess = np.maximum(adverse - stop_dist_frac, 0.0)
    per_night = float(excess.mean())                 # avg excess loss fraction
    return per_night * notional * TRADING_DAYS


def audit_name(daily_df, notional, stop_dist_frac, side='long'):
    """Full per-name overnight audit. daily_df needs Open/Close (>=~60 rows)."""
    overnight, intraday = overnight_intraday_returns(daily_df)
    stats = gap_stats(overnight)
    return {
        'gap_stats': stats,
        'forfeited_drift_annual': forfeited_drift_annual(overnight, notional),
        'gap_through_cost_annual': gap_through_cost_annual(
            overnight, stop_dist_frac, notional, side),
        'overnight_mean_bps': (round(float(np.mean(overnight)) * 1e4, 2)
                               if len(overnight) >= 20 else None),
        'intraday_mean_bps': (round(float(np.mean(intraday)) * 1e4, 2)
                              if len(intraday) >= 20 else None),
        'notional': notional,
        'stop_dist_frac': stop_dist_frac,
    }


def fetch_daily(symbol, period='3y'):
    """Free daily OHLC via yfinance (NOT fetch_with_fallback — that is hardcoded
    interval='1h' with no daily path). Lazy import so the module + pure
    analysis functions load without yfinance/network. Single ticker only:
    a multi-symbol download would silently interleave names' gaps into one
    series, so it is rejected up front."""
    if (not isinstance(symbol, str) or not symbol.strip()
            or len(symbol.split()) != 1 or ',' in symbol):
        raise ValueError(f"fetch_daily takes a single ticker, got {symbol!r}")
    import pandas as pd
    import yfinance as yf
    df = yf.download(symbol, period=period, interval='1d', progress=False,
                     auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance >= 0.2.x returns (Price, Ticker) MultiIndex columns;
        # collapse to the Price level (single ticker guaranteed above —
        # and rename(str.title) must never see the ticker level).
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df[['Open', 'High', 'Low', 'Close']].dropna()


def main():
    """CLI: audit the sleeve-eligible names. Run on a machine with network."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='+', required=True)
    ap.add_argument('--notional', type=float, default=5000.0)
    ap.add_argument('--stop-frac', type=float, default=0.02,
                    help='hourly-ATR stop distance as a fraction of price '
                         '(STOCK_POLICY atr_stop_mult*ATR ~1-3%, NOT 2*daily ATR)')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    results = {}
    total_drift = total_gap = 0.0
    for sym in args.symbols:
        df = fetch_daily(sym)
        if df is None:
            print(f"{sym}: no data"); continue
        a = audit_name(df, args.notional, args.stop_frac)
        results[sym] = a
        fd = a['forfeited_drift_annual'] or 0.0
        gt = a['gap_through_cost_annual'] or 0.0
        total_drift += fd
        total_gap += gt
        st = a['gap_stats']
        print(f"{sym:6s} overnight_mean={a['overnight_mean_bps']}bps "
              f"t_df={st['t_df']} exk={st['excess_kurtosis']} "
              f"forfeited_drift=${fd:,.0f}/yr gap_through=${gt:,.0f}/yr")
    print(f"\nSLEEVE TOTAL: forfeited_drift=${total_drift:,.0f}/yr  "
          f"gap_through=${total_gap:,.0f}/yr")
    print("Compare gap_through (the overlay's would-be job) to options_overlay "
          "friction: if friction >> gap_through, the overlay is NO-GO.")
    if args.json:
        import json
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
