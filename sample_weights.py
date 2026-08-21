"""Average-uniqueness sample weights and effective sample size (AFML ch. 4-7).

When labels are built from a forward window (triple-barrier / fixed-horizon),
a label entered at bar t0 and resolved at bar t1 "occupies" every bar in
[t0, t1]. Two labels whose windows overlap share information — they are NOT
independent observations. López de Prado's *average uniqueness* quantifies
this: a label that overlaps k others contributes ~1/k of a fresh sample.

Two consumers in this repo:

  - **Training loss weighting** (LightGBM mean+q10 legs ONLY, via
    fold_train_weights -> scripts/hypersearch_v2.py -> model_lgb.train_lgb,
    behind strategy_config.UNIQUENESS_WEIGHTS_ENABLED, default False):
    weight each row by its average uniqueness so heavily-overlapped rows stop
    being over-counted ~1/u-bar times — the mechanism that drives memorization
    on a panel where every hourly bar is a sample. The LSTM and the
    meta-learner are NOT wired (open wave-8 item).

  - **The Deflated-Sharpe promotion gate** (validation.py): the DSR null
    assumes n INDEPENDENT trial-Sharpe draws scattering with width 1/sqrt(n).
    With overlapping labels the effective count is n_eff = sum(u-bar) << n, so
    the honest null is wider and the certified edge bar is higher. Feeding the
    measured n_eff in removes a one-sided high bias in the gate.

Concurrency is computed PER TICKER: a label's forward window only overlaps
other labels in the SAME price series. Cross-sectional (same-calendar-bar,
different-name) correlation is a separate effect, not captured here.

u-bar is LABEL-SIDE, not point-in-time: TB_Bars is produced by
policy_exits.exit_walk scanning FORWARD price action until a barrier is
touched, and u-bar for row i also depends on labels that START after i. That
is legitimate for its two uses here (training weights, offline n_eff), but
u-bar must NEVER be used as a model feature or computed in a serving path
(predict_now / the live loops) — that would be a lookahead leak. Note also:
fold_train_weights measures concurrency over the FULL panel by design (AFML
ch. 4), so a train row within one span of a fold seam has its WEIGHT (not
its label) influenced by post-seam labels; purged walk-forward purges
labels, not weights.
"""

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is a hard dep on the Jetson
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # no-op decorator fallback (policy_exits form)
        def wrap(fn):
            return fn
        return wrap if not (len(args) == 1 and callable(args[0])) else args[0]


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
        if h > n:  # inf / absurd spans: pre-clip so int() cast is safe (numba int64 UB)
            h = float(n)
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
        if h > n:  # inf / absurd spans: pre-clip so int() cast is safe (numba int64 UB)
            h = float(n)
        end = i + int(h)
        if end > n - 1:
            end = n - 1
        # conc[t] >= 1 for every t in [i, end]: this row passed the same
        # NaN/negative filter and clip chain as pass 1, so it stamped +1
        # over exactly this window — the span is never empty and never has
        # zero concurrency (the old `else 1.0` fallback was unreachable and
        # pointed the gate-LOOSENING direction).
        s = 0.0
        for t in range(i, end + 1):
            s += 1.0 / conc[t]
        out[i] = s / (end - i + 1)
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
            None, the whole array is treated as one block. Blocks must be
            DISJOINT (ValueError otherwise); an empty container is an error
            (pass None for one block). Rows covered by NO block come back
            NaN (deliberate — they are non-labels downstream: weight 0 in
            fold_train_weights, excluded from effective_n).

    PRECONDITION (not checkable here): rows inside each block must be that
    ticker's bars in consecutive ascending-time order — hold_bars is a ROW
    offset, so an interleaved or re-sorted panel silently produces wrong
    u-bar. Note the first ~h rows of every block carry structurally higher
    u-bar (concurrency ramps from 1 at each block start — AFML convention).

    Returns: float64 array, same length as hold_bars. NaN for non-label
    rows (NaN/negative span) and for rows outside every block.
    """
    hb = np.ascontiguousarray(np.asarray(hold_bars, dtype=np.float64))
    if hb.ndim != 1:
        raise ValueError(f"hold_bars must be 1-D, got shape {hb.shape}")
    n = len(hb)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if ticker_boundaries is None:
        return _avg_uniqueness_block(hb)

    # Normalize spans to tuples BEFORE sorting: raw sorted() on a numpy
    # (k, 2) array compares rows elementwise ("truth value ... ambiguous")
    # and mixed list/tuple spans are not mutually orderable at all.
    if isinstance(ticker_boundaries, dict):
        spans = sorted(map(tuple, ticker_boundaries.values()))
    else:
        spans = sorted(map(tuple, ticker_boundaries))
    if not spans:
        raise ValueError(
            "ticker_boundaries is empty — pass None to treat the whole "
            "array as one block")
    for k in range(1, len(spans)):
        if int(spans[k][0]) < int(spans[k - 1][1]):
            raise ValueError(
                f"ticker_boundaries overlap: {tuple(spans[k - 1])} and "
                f"{tuple(spans[k])} — blocks must be disjoint")
    out = np.full(n, np.nan, dtype=np.float64)
    for start, end in spans:
        start = max(int(start), 0)
        end = min(int(end), n)
        if end > start:
            out[start:end] = _avg_uniqueness_block(
                np.ascontiguousarray(hb[start:end]))
    return out


def effective_n(u_bar, mask=None):
    """Effective independent sample count n_eff = sum of finite u-bar.

    POPULATION RULE: u_bar must have been computed over the SAME population
    whose independence is being measured. To measure n_eff among a SUBSET
    (e.g. the holdout rows that actually traded), NaN out the non-subset
    spans BEFORE calling average_uniqueness (the hypersearch_v2 holdout
    gate does exactly this); computing u_bar panel-wide and then masking
    here measures each trade against every non-traded bar and yields a
    systematically smaller n_eff (up to ~forward_bars times too small).

    SENTINEL: returns 0.0 when NO finite u-bar row exists (unmeasurable).
    Callers feeding a DSR MUST map 0.0 to None first —
    validation.dsr_from_trade_returns clamps any finite n_eff UP to its
    10-sample floor, so a raw 0.0 yields the harshest possible null
    instead of the intended IID fallback.

    INERT REGIME: when the scorer spaces observations >= the max label span
    apart (hypersearch's simulate_trades steps i += forward_bars), labels
    cannot overlap and this returns ~n — it provides no correction there.

    CLAUDE.md gotcha #4: this n_eff and validation.serial_correlation_factor
    are mutually exclusive non-IID corrections — never stack both on one
    DSR call.

    Args:
        u_bar: per-row average uniqueness (NaNs ignored).
        mask: optional BOOLEAN array, same shape as u_bar, selecting the
            rows that count. Positional index arrays are rejected
            (TypeError) — an int array of matching length would silently be
            reinterpreted as 0/1 booleans otherwise.

    Returns float n_eff: sum of the finite (selected) u-bar values. With
    fully-unique labels this equals the row count; with heavy overlap it
    shrinks toward 1; 0.0 means nothing was measurable (see SENTINEL).
    """
    u = np.asarray(u_bar, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask)
        if m.dtype != np.bool_:
            raise TypeError(
                f"mask must be a boolean array, got dtype {m.dtype} — "
                "positional indices are not accepted")
        if m.shape != u.shape:
            raise ValueError(
                f"mask shape {m.shape} != u_bar shape {u.shape}")
        u = u[m]
    u = u[np.isfinite(u)]
    return float(u.sum())


def clustered_effective_n(entry_times, exit_times):
    """Cross-sectional effective trade count via calendar-interval clustering.

    average_uniqueness / effective_n correct for overlap WITHIN one price
    series; they still count same-hour trades on N correlated names as N
    independent draws. On a 6-coin book at pairwise rho 0.7-0.9 that
    overstates the DSR z-statistic's sqrt(n) breadth ~2-4x (2026-07 review:
    a zero-edge model's holdout false-pass rate rises from 0.2% to 5-9% at
    realistic correlations). This collapses trades whose [entry, exit]
    CALENDAR intervals overlap — across any names — into single clusters:
    the rho=1 lockstep worst case, the same convention as CROSS_BOOK_RHO.

    Args: parallel arrays of entry/exit times (datetime64, pandas Timestamps
    via .values, or numeric epochs). NaT/NaN pairs are dropped.

    Returns the cluster count (int, <= n trades). The honest n_eff for a DSR
    null is min(this, the within-ticker effective_n) — this function never
    loosens a gate, only tightens it.

    NOTE: clustering is TRANSITIVE (connected components of the interval-
    overlap graph): A-B overlapping and B-C overlapping merge A, B, C even
    when A and C are disjoint (pinned deliberately by
    tests/test_cs_neff.py::test_chain_overlap_is_one_cluster). On a book
    whose trades tile the calendar this collapses toward the number of idle
    gaps — far harsher than the pairwise rho=1 model suggests. Returns 0
    ONLY for genuinely empty input; malformed input (length mismatch,
    mixed datetime/numeric or object dtypes) raises.
    """
    et = np.asarray(entry_times)
    xt = np.asarray(exit_times)
    if et.size == 0 and xt.size == 0:
        return 0
    if xt.size != et.size:
        raise ValueError(
            f"entry_times ({et.size}) and exit_times ({xt.size}) must be "
            "parallel arrays")
    if et.dtype == object or xt.dtype == object:
        raise TypeError(
            "object-dtype times are not accepted (e.g. a plain list of "
            "pandas Timestamps) — pass datetime64 arrays (.values / "
            ".to_numpy()) or numeric epochs")
    et_is_dt = np.issubdtype(et.dtype, np.datetime64)
    xt_is_dt = np.issubdtype(xt.dtype, np.datetime64)
    if et_is_dt != xt_is_dt:
        raise TypeError(
            f"entry_times ({et.dtype}) and exit_times ({xt.dtype}) must "
            "use the same time representation — both datetime64 or both "
            "numeric")
    if et_is_dt:
        xt = np.asarray(xt, dtype=et.dtype)
        ok = ~(np.isnat(et) | np.isnat(xt))
    else:
        et = et.astype(np.float64)
        xt = xt.astype(np.float64)
        ok = np.isfinite(et) & np.isfinite(xt)
    et, xt = et[ok], xt[ok]
    if et.size == 0:
        return 0
    order = np.argsort(et, kind='stable')
    et, xt = et[order], xt[order]
    clusters = 0
    cluster_end = None
    for e, x in zip(et, xt):
        if x < e:
            x = e  # degenerate span: treat as a point interval
        if cluster_end is None or e > cluster_end:
            clusters += 1
            cluster_end = x
        elif x > cluster_end:
            cluster_end = x
    return int(clusters)


def _coerce_time_intervals(entry_times, exit_times):
    """Coerce parallel entry/exit time arrays to float64 HOUR intervals.

    Replicates clustered_effective_n's input contract exactly (kept separate
    so that pinned function stays untouched): empty -> (empty, empty); size
    mismatch -> ValueError; object dtype -> TypeError; mixed datetime64 /
    numeric -> TypeError. datetime64 pairs are cast to a common dtype,
    NaT pairs dropped, and converted to float hours relative to the minimum
    kept entry time. Numeric pairs are float64 HOURS by contract; non-finite
    pairs dropped. Degenerate exit < entry becomes a point interval
    (exit = entry). Returns (start_hours, end_hours).
    """
    et = np.asarray(entry_times)
    xt = np.asarray(exit_times)
    if et.size == 0 and xt.size == 0:
        return (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
    if xt.size != et.size:
        raise ValueError(
            f"entry_times ({et.size}) and exit_times ({xt.size}) must be "
            "parallel arrays")
    if et.dtype == object or xt.dtype == object:
        raise TypeError(
            "object-dtype times are not accepted (e.g. a plain list of "
            "pandas Timestamps) — pass datetime64 arrays (.values / "
            ".to_numpy()) or numeric epochs")
    et_is_dt = np.issubdtype(et.dtype, np.datetime64)
    xt_is_dt = np.issubdtype(xt.dtype, np.datetime64)
    if et_is_dt != xt_is_dt:
        raise TypeError(
            f"entry_times ({et.dtype}) and exit_times ({xt.dtype}) must "
            "use the same time representation — both datetime64 or both "
            "numeric")
    if et_is_dt:
        xt = np.asarray(xt, dtype=et.dtype)
        ok = ~(np.isnat(et) | np.isnat(xt))
        et, xt = et[ok], xt[ok]
        if et.size == 0:
            return (np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64))
        t_min = et.min()
        s = (et - t_min) / np.timedelta64(1, 'h')
        e = (xt - t_min) / np.timedelta64(1, 'h')
        s = s.astype(np.float64)
        e = e.astype(np.float64)
    else:
        et = et.astype(np.float64)
        xt = xt.astype(np.float64)
        ok = np.isfinite(et) & np.isfinite(xt)
        s, e = et[ok], xt[ok]
        if s.size == 0:
            return (np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64))
    e = np.maximum(e, s)  # degenerate exit < entry -> point interval
    return s, e


def calendar_effective_n(entry_times, exit_times, rho_bar=None):
    """Calendar-concurrency average-uniqueness effective trade count.

    Lopez de Prado AFML ch. 4 average uniqueness applied to the CALENDAR-hour
    concurrency of realized trades across ALL names: c_t = number of trades
    open at hour t; each trade's uniqueness u_i is the mean of 1/c_t over its
    open hours and n_eff = sum(u_i). SUPERSEDES both the per-ticker
    effective_n and clustered_effective_n — under
    strategy_config.PROMOTION_GATE_V2 exactly ONE non-IID correction applies,
    and it is never stacked with the Lo-2002 serial factor (CLAUDE.md
    gotcha #4). Reduces to the within-ticker average uniqueness for a single
    name; every trade contributes u_i > 0 so n_eff stays in
    [n/max_concurrency, n] and degrades smoothly under crowding instead of
    collapsing to the idle-gap count the way connected components do; a
    trade DISJOINT from all others always adds exactly 1.0. (Global
    monotonicity under arbitrary overlapping additions does NOT hold — one
    long trade blanketing many short ones redistributes uniqueness downward —
    but the [n/c_max, n] bounds always do.)

    rho_bar in (0, 1] applies the Kish design-effect per-hour weight
    1/(1 + (c_t - 1)*rho_bar): rho_bar=1.0 equals the plain 1/c_t default,
    None takes the 1/c_t path, and values outside (0, 1] raise ValueError.

    Returns {'n_eff', 'n_trades', 'u', 'u_bar_mean', 'max_concurrency',
    'rho_bar'} where u is the per-kept-trade uniqueness array.
    """
    if rho_bar is not None:
        rb = float(rho_bar)
        if not (0.0 < rb <= 1.0):
            raise ValueError(
                f"rho_bar must be in (0, 1] (1.0 == plain 1/c_t), "
                f"got {rho_bar}")
    s, e = _coerce_time_intervals(entry_times, exit_times)
    if s.size == 0:
        return {'n_eff': 0.0, 'n_trades': 0,
                'u': np.empty(0, dtype=np.float64), 'u_bar_mean': None,
                'max_concurrency': 0, 'rho_bar': rho_bar}
    # Shift so hour bins start at 0 (concurrency is shift-invariant; numeric
    # inputs may carry negative hours, which would break the diff array).
    s0 = s.min()
    s = s - s0
    e = e - s0
    lo = np.floor(s).astype(np.int64)
    hi = np.floor(e).astype(np.int64)  # inclusive hour bins, hi >= lo
    T = int(hi.max()) + 1
    diff = np.zeros(T + 1, dtype=np.float64)
    np.add.at(diff, lo, 1.0)
    np.add.at(diff, hi + 1, -1.0)
    c = np.cumsum(diff[:T])
    # Idle hours between disjoint trades have c=0; no trade's span reads
    # them, but an inf/nan there would poison the prefix sums — weight 0.
    c_safe = np.maximum(c, 1.0)
    if rho_bar is None:
        w = np.where(c > 0.0, 1.0 / c_safe, 0.0)
    else:
        w = np.where(c > 0.0,
                     1.0 / (1.0 + (c_safe - 1.0) * float(rho_bar)), 0.0)
    # Prefix sums -> vectorized per-trade means over [lo_i, hi_i].
    p = np.concatenate([[0.0], np.cumsum(w)])
    u = (p[hi + 1] - p[lo]) / (hi - lo + 1).astype(np.float64)
    return {'n_eff': float(u.sum()), 'n_trades': int(u.size), 'u': u,
            'u_bar_mean': float(u.mean()), 'max_concurrency': int(c.max()),
            'rho_bar': rho_bar}


def _blend_normalize(w, returns, ret_cap):
    """Blend optional |return| emphasis into w, normalize mean-1 over the
    strictly-POSITIVE rows. A NaN in `returns` zeroes that row's weight
    (same as a NaN span). Raises on a fully-degenerate (no positive row)
    non-empty vector — zero total weight mass is never a valid training
    input (fail-closed)."""
    if returns is not None:
        r = np.abs(np.asarray(returns, dtype=np.float64)) + 1.0
        if r.ndim != 1:
            raise ValueError(
                f"returns must be a 1-D array aligned 1:1 with the "
                f"weighted rows, got shape {r.shape}")
        if len(r) != len(w):
            raise ValueError(
                f"returns length {len(r)} != weights length {len(w)}; "
                "returns must align 1:1 with the weighted rows")
        if not (ret_cap > 1.0):
            raise ValueError(
                f"ret_cap caps |return|+1 and must be > 1.0, got {ret_cap}")
        np.clip(r, None, ret_cap, out=r)
        r = np.where(np.isfinite(r), r, 0.0)
        w = w * r
    finite = w[np.isfinite(w) & (w > 0)]
    if finite.size and finite.mean() > 0:
        w = w / finite.mean()
    elif len(w):
        raise ValueError(
            f"degenerate sample weights: no positive finite row among "
            f"{len(w)} — every label span is NaN/negative, "
            "ticker_boundaries does not cover these rows, or the returns "
            "blend zeroed everything")
    return w


def uniqueness_weights(hold_bars, returns=None, ret_cap=50.0,
                       ticker_boundaries=None):
    """Training sample weights: average-uniqueness, optionally blended with
    a return-magnitude emphasis, normalized so the mean weight is 1 over
    the strictly-POSITIVE rows.

    NaN-span rows are returned as exactly 0 and are EXCLUDED from the
    normalizer, so the mean over ALL rows equals the positive fraction and
    total mass equals the positive-row count, not len(w). LightGBM's
    min_sum_hessian_in_leaf / min_child_weight / L1-L2 are denominated in
    weight units, so a weighted run is regularized differently from an
    unweighted one (CLAUDE.md gotcha #2 applies on the first weighted
    retrain).

    No production caller today — this is the whole-array convenience form
    of fold_train_weights (identity pinned in tests/test_improve_sweights.py).

    SCALE WARNING for the open wave-8 LSTM swap: the LSTM's current
    in-loop weights (torch.clamp(|y|+1, max=50) in scripts/hypersearch_v2.py)
    are NOT mean-1 — their mean is ~1+E|y|. Substituting this mean-1 vector
    there changes the loss SCALE and its interaction with
    clip_grad_norm_(max_norm=1.0) as well as the weighting; the swap is
    model-facing and must re-normalize or re-tune (owner decision).

    `returns` must be in PERCENT units (same as Target_Return / TB_Ret) so
    ret_cap=50.0 means 'cap at a ~49% move'; a NaN return zeroes that
    row's weight. Callers that train per-book should weight one book's
    rows at a time so the mean-1 scaling does not silently re-balance
    crypto against stock.
    """
    u = average_uniqueness(hold_bars, ticker_boundaries)
    w = np.where(np.isfinite(u), u, 0.0)
    return _blend_normalize(w, returns, ret_cap)


def fold_train_weights(hold_bars_panel, row_indices, ticker_boundaries=None,
                       returns=None, ret_cap=50.0):
    """Mean-1 training weights for ONE fold's rows, de-biased for label overlap.

    Wired at scripts/hypersearch_v2.py (train_lgb_ensemble) into
    model_lgb.train_lgb(sample_weight=..., sample_weight_val=...), behind
    strategy_config.UNIQUENESS_WEIGHTS_ENABLED (default False). The LSTM
    and meta-learner remain unweighted (open wave-8 item).

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
            Aligned 1:1 with row_indices (FOLD-aligned), NOT with
            hold_bars_panel; PERCENT units; a NaN return zeroes that row.

    Returns a float64 array aligned to row_indices, mean 1 over its finite>0
    entries (NaN-span rows -> weight 0). Train each book separately so the mean-1
    scaling never re-balances crypto against stock.
    """
    u = average_uniqueness(hold_bars_panel, ticker_boundaries)   # FULL panel
    idx = np.asarray(row_indices)
    if idx.dtype == np.bool_:
        raise TypeError(
            "row_indices must be integer PANEL positions, not a boolean "
            "mask — pass np.flatnonzero(mask)")
    if idx.size and not np.issubdtype(idx.dtype, np.integer):
        raise TypeError(
            f"row_indices must be an integer array, got dtype {idx.dtype}")
    if idx.ndim != 1:
        raise ValueError(f"row_indices must be 1-D, got shape {idx.shape}")
    idx = idx.astype(np.int64, copy=False)
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= len(u)):
        raise IndexError(
            f"row_indices out of panel range [0, {len(u)}): "
            f"min {int(idx.min())}, max {int(idx.max())}")
    u_fold = u[idx]
    w = np.where(np.isfinite(u_fold), u_fold, 0.0)
    return _blend_normalize(w, returns, ret_cap)
