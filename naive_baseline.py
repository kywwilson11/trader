"""FR-04 Nagel naive baseline — strictly-trailing EWMA-momentum / vol kernel.

Measurement-only (R2C-06a, ships direct, no flags). Nagel (2025, NBER WP
34104) showed over-parameterized return forecasts collapse toward exactly
this predictor: a recency-weighted momentum signal scaled by trailing
realized vol, zero fitting. If the deployed LSTM+LGB blend cannot beat this
one-liner on identical stage-0 rows (purged IC AND DSR, with the blend
deflated at the campaign's cum_trials and the naive rule at n_trials=1),
the blend's claimed edge is suspect — a promotion-culture finding reported
to the owner, never an auto-action.

Causality contract: the signal at bar i uses closes[0..i] ONLY (returns
completed by bar i's close). The stage-0 label at row i spans
Close[i] -> Close[i+h], so a through-i signal never overlaps its own label
window. Pure numpy — no pandas, no repo trading-path imports. Consumed by
scripts/naive_vs_blend.py; synthetic hand-checked tests live in
tests/test_r2c_measurement_kernels.py.

UNITS: bar_returns/ewma_momentum are PERCENT (matching harvest
Target_Return_* and the stage0 dump); naive_signal is a unitless
momentum/vol ratio — fine for rank IC and for sign-based admission, and
deliberately NOT a calibrated return forecast.
"""
import numpy as np

# Default lookbacks: half-life inside the tuned 12-48h horizon band
# (FR-04 spec), vol window = 3 half-lives of hourly bars.
DEFAULT_HALF_LIFE = 24.0
DEFAULT_VOL_WINDOW = 72
_VOL_EPS = 1e-12


def bar_returns(closes):
    """1-bar close-to-close PERCENT returns aligned to closes.

    out[0] = NaN (no prior close); out[i] = (c[i]-c[i-1])/c[i-1]*100.
    Non-finite or zero previous closes yield NaN at that bar.
    """
    c = np.asarray(closes, dtype=np.float64)
    out = np.full(len(c), np.nan)
    if len(c) < 2:
        return out
    prev, cur = c[:-1], c[1:]
    ok = np.isfinite(prev) & np.isfinite(cur) & (prev != 0.0)
    out[1:][ok] = (cur[ok] - prev[ok]) / prev[ok] * 100.0
    return out


def ewma_momentum(returns, half_life=DEFAULT_HALF_LIFE):
    """Recursive EWMA of returns (pandas ewm(halflife=h, adjust=False)
    semantics): alpha = 1 - 0.5**(1/half_life);
    m[i] = (1-alpha)*m[i-1] + alpha*r[i], seeded at the first finite
    return. Strictly trailing: m[i] depends on returns[0..i] only. NaN
    before the seed; NaN returns after the seed are carried through as
    "no new information" (m[i] = m[i-1])."""
    r = np.asarray(returns, dtype=np.float64)
    hl = float(half_life)
    if hl <= 0:
        raise ValueError('half_life must be > 0')
    alpha = 1.0 - 0.5 ** (1.0 / hl)
    out = np.full(len(r), np.nan)
    m = np.nan
    for i in range(len(r)):
        ri = r[i]
        if np.isfinite(ri):
            m = ri if not np.isfinite(m) else (1.0 - alpha) * m + alpha * ri
        out[i] = m
    return out


def trailing_vol(returns, window=DEFAULT_VOL_WINDOW, min_periods=None):
    """Trailing realized vol: std (ddof=1) of the last `window` returns
    ENDING AT and including bar i. NaN until min_periods finite returns
    (default = window) are in the trailing window; NaN returns inside the
    window are ignored (finite-only std)."""
    r = np.asarray(returns, dtype=np.float64)
    w = int(window)
    if w < 2:
        raise ValueError('window must be >= 2')
    mp = w if min_periods is None else max(2, int(min_periods))
    out = np.full(len(r), np.nan)
    for i in range(len(r)):
        seg = r[max(0, i - w + 1):i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= mp:
            out[i] = np.std(seg, ddof=1)
    return out


def naive_signal(closes, half_life=DEFAULT_HALF_LIFE,
                 vol_window=DEFAULT_VOL_WINDOW, min_periods=None):
    """The Nagel one-liner per bar: EWMA momentum / trailing vol.

    signal[i] uses closes[0..i] only (see module causality contract).
    NaN wherever the vol is NaN or ~0 (degenerate flat stretch) or the
    momentum is NaN — callers treat NaN as "no signal, no entry"."""
    r = bar_returns(closes)
    mom = ewma_momentum(r, half_life=half_life)
    vol = trailing_vol(r, window=vol_window, min_periods=min_periods)
    with np.errstate(invalid='ignore', divide='ignore'):
        sig = mom / vol
    sig[~np.isfinite(sig)] = np.nan
    sig[np.isfinite(vol) & (vol < _VOL_EPS)] = np.nan
    return sig


def lookup_at_times(series_times_ns, values, query_times_ns):
    """values at exact-timestamp matches: out[j] = values[i] where
    series_times_ns[i] == query_times_ns[j], NaN when the query time is
    absent. series_times_ns must be sorted int64 ns (the stage0_preds
    clock convention)."""
    t = np.asarray(series_times_ns, dtype=np.int64)
    v = np.asarray(values, dtype=np.float64)
    q = np.asarray(query_times_ns, dtype=np.int64)
    out = np.full(len(q), np.nan)
    if len(t) == 0 or len(q) == 0:
        return out
    pos = np.searchsorted(t, q)
    ok = (pos < len(t)) & (t[np.minimum(pos, len(t) - 1)] == q)
    out[ok] = v[pos[ok]]
    return out


def long_only_returns(preds, fwd_returns, threshold=0.0):
    """Admitted long-only trade returns: fwd where pred > threshold
    (STRICT >, matching the hypersearch objective —
    objective_utils.simulate_trades_core — and predict_now's CLI
    recommendation. NOTE backtest.simulate_ticker and base_loop._execute_buys
    skip on `pred < threshold`, i.e. they ADMIT pred == threshold; the
    boundary is measure-zero for float preds). Rows with non-finite pred or
    fwd are dropped. The DSR input for both the blend and the naive rule."""
    p = np.asarray(preds, dtype=np.float64)
    f = np.asarray(fwd_returns, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(f) & (p > float(threshold))
    return f[m]
