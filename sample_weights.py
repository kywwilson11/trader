"""Average-uniqueness sample weights and effective sample size (AFML ch. 4-7).

When labels are built from a forward window (triple-barrier / fixed-horizon),
a label entered at bar t0 and resolved at bar t1 "occupies" every bar in
[t0, t1]. Two labels whose windows overlap share information — they are NOT
independent observations. López de Prado's *average uniqueness* quantifies
this: a label that overlaps k others contributes ~1/k of a fresh sample.

Two consumers in this repo:

  - **Training loss weighting** (LSTM / LightGBM mean+q10 / meta-learner):
    weight each row by its average uniqueness so heavily-overlapped rows stop
    being over-counted ~1/u-bar times — the mechanism that drives memorization
    on a panel where every hourly bar is a sample.

  - **The Deflated-Sharpe promotion gate** (validation.py): the DSR null
    assumes n INDEPENDENT trial-Sharpe draws scattering with width 1/sqrt(n).
    With overlapping labels the effective count is n_eff = sum(u-bar) << n, so
    the honest null is wider and the certified edge bar is higher. Feeding the
    measured n_eff in removes a one-sided high bias in the gate.

Concurrency is computed PER TICKER: a label's forward window only overlaps
other labels in the SAME price series. Cross-sectional (same-calendar-bar,
different-name) correlation is a separate effect, not captured here.

Everything is point-in-time clean: u-bar for row i depends only on label spans,
which are a deterministic function of the harvested TB_Bars (no future returns).
"""

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is a hard dep on the Jetson
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # no-op decorator fallback
        def wrap(fn):
            return fn
        return wrap if not args else args[0]


@njit(cache=True)
def _avg_uniqueness_block(hold_bars):
    """Per-row average uniqueness for one contiguous (single-ticker) block.

    hold_bars[i] = number of bars label i is held (its window is [i, i+h]).
    NaN / negative spans -> the row is not a label; its uniqueness is NaN and
    it contributes nothing to anyone else's concurrency.
    """
    n = len(hold_bars)
    out = np.empty(n, dtype=np.float64)
    # difference array -> concurrency c_t = number of active labels at bar t
    diff = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        h = hold_bars[i]
        if not (h == h) or h < 0.0:  # NaN or negative -> not a label
            continue
        end = i + int(h)
        if end > n - 1:
            end = n - 1
        diff[i] += 1.0
        diff[end + 1] -= 1.0
    conc = np.empty(n, dtype=np.float64)
    run = 0.0
    for t in range(n):
        run += diff[t]
        conc[t] = run
    # u_i = mean over its span of 1/c_t
    for i in range(n):
        h = hold_bars[i]
        if not (h == h) or h < 0.0:
            out[i] = np.nan
            continue
        end = i + int(h)
        if end > n - 1:
            end = n - 1
        s = 0.0
        cnt = 0
        for t in range(i, end + 1):
            c = conc[t]
            if c > 0.0:
                s += 1.0 / c
                cnt += 1
        out[i] = s / cnt if cnt > 0 else 1.0
    return out


def average_uniqueness(hold_bars, ticker_boundaries=None):
    """Per-row average uniqueness u-bar in (0, 1].

    Args:
        hold_bars: 1-D array of per-row label spans (TB_Bars_{fb}). NaN rows
            (truncated windows at a series end) come back NaN.
        ticker_boundaries: optional dict {ticker: (start, end)} of half-open
            row ranges into the concatenated panel, OR an iterable of
            (start, end) tuples. Concurrency is computed independently within
            each block so one ticker's window never overlaps another's. If
            None, the whole array is treated as one block.

    Returns: float64 array, same length as hold_bars.
    """
    hb = np.ascontiguousarray(np.asarray(hold_bars, dtype=np.float64))
    n = len(hb)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if ticker_boundaries is None:
        return _avg_uniqueness_block(hb)

    if isinstance(ticker_boundaries, dict):
        spans = sorted(ticker_boundaries.values())
    else:
        spans = sorted(ticker_boundaries)
    out = np.full(n, np.nan, dtype=np.float64)
    for start, end in spans:
        start = max(int(start), 0)
        end = min(int(end), n)
        if end > start:
            out[start:end] = _avg_uniqueness_block(
                np.ascontiguousarray(hb[start:end]))
    return out


def effective_n(u_bar, mask=None):
    """Effective independent sample count n_eff = sum of u-bar.

    Args:
        u_bar: per-row average uniqueness (NaNs ignored).
        mask: optional boolean array selecting the subset (e.g. the holdout
            rows that actually traded). When given, only masked rows count.

    Returns float n_eff. With fully-unique (non-overlapping) labels this equals
    the row count; with heavy overlap it shrinks toward 1.
    """
    u = np.asarray(u_bar, dtype=np.float64)
    if mask is not None:
        u = u[np.asarray(mask, dtype=bool)]
    u = u[np.isfinite(u)]
    return float(u.sum())


def uniqueness_weights(hold_bars, returns=None, ret_cap=50.0,
                       ticker_boundaries=None):
    """Training sample weights: average-uniqueness, optionally blended with a
    return-magnitude emphasis, normalized so the mean weight is 1.

    The LSTM already weights by clamp(|r|+1, max=ret_cap); this preserves that
    magnitude emphasis while folding in uniqueness, so overlapping labels stop
    being over-counted. Pass returns=None for pure uniqueness weighting (the
    LightGBM / meta legs, which were previously uniform).

    Normalization is done over the FINITE rows only; NaN-span rows get weight 0.
    Callers that train per-book should weight one book's rows at a time so the
    mean-1 scaling does not silently re-balance crypto against stock.
    """
    u = average_uniqueness(hold_bars, ticker_boundaries)
    w = np.where(np.isfinite(u), u, 0.0)
    if returns is not None:
        r = np.abs(np.asarray(returns, dtype=np.float64)) + 1.0
        np.clip(r, None, ret_cap, out=r)
        r = np.where(np.isfinite(r), r, 0.0)
        w = w * r
    finite = w[np.isfinite(w) & (w > 0)]
    if finite.size and finite.mean() > 0:
        w = w / finite.mean()
    return w


def fold_train_weights(hold_bars_panel, row_indices, ticker_boundaries=None,
                       returns=None, ret_cap=50.0):
    """Mean-1 training weights for ONE fold's rows, de-biased for label overlap.

    The wiring artery the LSTM / LightGBM-mean / q10 trainers were missing: the
    producer below is consumed only by the DSR gate today, so the loss still
    over-counts a row overlapping k others ~k times.

    Correctness hinges on two things:
      1. Average uniqueness is computed over the FULL panel (with per-ticker
         boundaries) — a fold's rows still overlap labels OUTSIDE the fold, so
         slicing first then computing concurrency would be wrong.
      2. row_indices index into the PANEL (not a pre-sliced copy). This is the
         #1 alignment hazard; mismatching it silently mis-weights every row.

    Args:
        hold_bars_panel: 1-D label spans (TB_Bars_{fb}) for the ENTIRE panel.
        row_indices: integer positions of this fold's rows into the panel.
        ticker_boundaries: per-ticker (start, end) blocks into the panel so
            concurrency never crosses tickers (see average_uniqueness).
        returns: None = PURE uniqueness (the safe default). Supplying returns
            blends a |return| magnitude emphasis — do NOT enable silently: the
            return-attribution variant collapsed out-of-sample F1 in the AFML
            replication; use it only as an explicit shadow challenger.

    Returns a float64 array aligned to row_indices, mean 1 over its finite>0
    entries (NaN-span rows -> weight 0). Train each book separately so the mean-1
    scaling never re-balances crypto against stock.
    """
    u = average_uniqueness(hold_bars_panel, ticker_boundaries)   # FULL panel
    idx = np.asarray(row_indices, dtype=np.int64)
    u_fold = u[idx]
    w = np.where(np.isfinite(u_fold), u_fold, 0.0)
    if returns is not None:
        r = np.abs(np.asarray(returns, dtype=np.float64)) + 1.0
        np.clip(r, None, ret_cap, out=r)
        r = np.where(np.isfinite(r), r, 0.0)
        w = w * r
    finite = w[np.isfinite(w) & (w > 0)]
    if finite.size and finite.mean() > 0:
        w = w / finite.mean()
    return w
