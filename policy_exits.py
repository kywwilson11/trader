"""The exit-stack walk — ONE implementation shared by three consumers.

    1. backtest.py        — policy replay for the promotion gate
    2. harvest scripts    — triple-barrier label generation (TB_Ret_{fb})
    3. meta_label.py      — meta-labeling dataset (replayed trades)

Lopez de Prado's core point about labels: a model trained on raw fb-bar
forward returns is answering a question the live system never asks — the
bot exits via ATR stops / trailing / take-profit / EOD flatten, not by
holding exactly fb bars. Triple-barrier labels ask the RIGHT question
("what does the exit stack realize from an entry here?"), and sharing one
kernel keeps label semantics == backtest semantics == live semantics up to
documented bar-level approximations and live-only overlays (stocks even
get the EOD flatten as their vertical barrier).

Live-only overlays the kernel does NOT model (see also backtest.py's
"Bar-level approximations" note):
  - macro-regime stop tightening: base_loop multiplies stop/trail
    distances by macro_regime.stop_mult (< 1) in risk-off regimes
  - two-consecutive-reading breach confirmation before a stop exit
    (~1 extra 30s cycle); the kernel exits on the first touching bar
  - live HWM tracks 30s quote MIDPOINTS; the kernel uses bar HIGHS
  - stock EOD flatten fills at the day-last bar's CLOSE here vs the
    live ~15:50 ET flatten order
  - no-ATR fallback TP asymmetry: live crypto entries get NO take-profit
    and live stock fallbacks use tp = TAKE_PROFIT_CEIL_PCT, while the
    kernel prices tp = stop_fallback_pct * tp_rr on NaN ATR
  - take-profit gap asymmetry: a bar gapping DOWN through the stop fills
    at the worse open (min(open, stop)), but a bar gapping UP through the
    TP still fills exactly at tp_price. Conservative for the stock
    bracket's resting limit leg (stock_loop._execute_buys); live crypto
    has NO resting TP and market-sells on a polled 30s quote, so crypto
    TP exits that gap through are understated here
  - trailing-stop enforcement lag: bar j's own high updates the HWM only
    AFTER bar j's stop/TP checks, so the trail level enforced at bar j
    derives from highs through bar j-1; live re-derives the trail from
    the same-cycle HWM and can exit on an intrabar spike-then-fade that
    the kernel holds through
  - stock rank-drop sell: stock_loop._execute_sells also flattens a held
    name that leaves the top-HOLD_RANK cross-sectional ranking while its
    pred is merely negative; the kernel's only discretionary exit is the
    signal flip. Boundary: the crypto/base sell path
    (base_loop._execute_sells) fires at pred <= -threshold, one boundary
    case wider than the strict p < -threshold that BOTH the kernel and
    stock_loop's override use
  - LLM-veto liquidation: base_loop._execute_llm_veto_sells force-flattens
    after two consecutive catastrophic LLM scores (exit_reason='llm_veto')
  - overnight sleeve: with OVERNIGHT_SLEEVE_ENABLED, up to
    OVERNIGHT_SLEEVE_MAX_POSITIONS stock positions are deliberately kept
    through the close, so the kernel's EOD flatten is an upper bound on
    live stock hold time, not an equality

Consumer parameterization differs BY DESIGN — labels and the backtest
answer different questions with the same kernel:
  - compute_tb_labels: max_hold=fb (12-48), use_signal_exit=False
  - backtest.py / meta_label.py: max_hold=0 (unlimited), use_signal_exit=True
  - decision_report.py: max_hold=24, use_signal_exit=False
so a TB_Ret_{fb} label and a backtest trade from the same entry bar can
exit at different bars for reasons beyond the overlays above.

The kernel computes, for EVERY bar i, the exit the live stack would
produce for an entry at bar i's close:
  - gap-aware hard/trailing stop (filled at min(stop, open) on gaps)
  - take-profit at the limit price
  - trailing activation at +trail_act over entry, tracked on bar highs
  - optional signal-flip exit (pred < -threshold after a cooldown)
  - EOD exit for stocks (is_eod mask)
  - vertical barrier at max_hold bars (labels) or unlimited (backtest)
Stop checked FIRST when stop and TP touch in the same bar (conservative).

Same-bar precedence and bar-boundary conventions (NORMATIVE — a
reimplementation must match these exactly):
  1. Per-bar check order: effective stop (hard OR trailing, whichever is
     higher) -> take-profit -> HWM update / trailing activation ->
     signal flip -> EOD flatten. The vertical barrier is the loop
     bound's fall-through and loses every same-bar tie (a stop, signal
     or EOD landing exactly on bar i+max_hold wins; EOD overwrites the
     vertical default at the same index and price). The folded stop
     check means the TRAILING stop outranks the take-profit within a
     bar; live's elif chain (base_loop._manage_stops: hard_stop ->
     take_profit -> trailing) resolves a single quote satisfying both
     the other way.
  2. The walk starts at j = i + 1: entry is bar i's CLOSE, and the
     entry bar's own high/low AND its own is_eod flag are never
     examined (an entry ON a stock's day-last bar runs to the NEXT
     session's EOD — see compute_tb_labels).
  3. hwm starts at entry (bar i's close) and updates AFTER the stop/TP
     checks with a STRICT high[j] > hwm, so the trail level enforced at
     bar j uses highs through bar j-1 only, and a bar that arms or
     raises the trail can never fire that same trail.
  4. Barrier comparisons are INCLUSIVE (low <= stop, high >= tp).
     Reason is 3 'trailing' only when the armed trail sits STRICTLY
     above the hard stop (eff_stop > stop_price), else 1 'hard_stop'.
  5. Signal exit: p < -threshold (long) / p > +threshold (short), with
     (j - i) >= cooldown_bars as a minimum hold; preds are ignored
     entirely when use_signal_exit=False.
  6. max_hold=0 means unlimited (backtest mode): the no-barrier terminal
     reason is 0 'end_of_data'. With max_hold > 0 it is 6 'vertical',
     stamped even when the window was truncated by the series end.
     Negative max_hold is NOT validated and falls through to unlimited.

Exit reason codes:
  0 end_of_data, 1 hard_stop, 2 take_profit, 3 trailing,
  4 signal_sell, 5 eod_flatten, 6 vertical
"""

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba on Jetson/CI; dev Mac always takes this fallback
    _HAS_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]

REASON_NAMES = {0: 'end_of_data', 1: 'hard_stop', 2: 'take_profit',
                3: 'trailing', 4: 'signal_sell', 5: 'eod_flatten',
                6: 'vertical'}


@njit(cache=True)
def _exit_walk_kernel(close, high, low, open_, atr, preds, is_eod,
                      threshold, atr_stop_mult, atr_trail_mult,
                      trail_act_pct, stop_floor, stop_ceil,
                      tp_rr, tp_ceil, stop_fallback, trail_fallback,
                      cooldown_bars, max_hold, use_signal_exit):
    n = len(close)
    exit_idx = np.empty(n, dtype=np.int64)
    exit_px = np.empty(n, dtype=np.float64)
    reason = np.zeros(n, dtype=np.int8)

    for i in range(n):
        entry = close[i]
        a = atr[i]
        if a == a and entry > 0:  # not NaN
            sd = (a * atr_stop_mult) / entry
            if sd < stop_floor:
                sd = stop_floor
            elif sd > stop_ceil:
                sd = stop_ceil
            td = (a * atr_trail_mult) / entry
            if td < stop_floor:
                td = stop_floor
            elif td > stop_ceil:
                td = stop_ceil
        else:
            sd = stop_fallback
            td = trail_fallback
        tp_d = sd * tp_rr
        if tp_d > tp_ceil:
            tp_d = tp_ceil
        stop_price = entry * (1.0 - sd)
        tp_price = entry * (1.0 + tp_d)
        hwm = entry
        trailing = False

        last_j = n - 1
        if max_hold > 0 and i + max_hold < last_j:
            last_j = i + max_hold

        e_idx = last_j
        e_px = close[last_j]
        e_reason = 6 if max_hold > 0 else 0

        j = i + 1
        while j <= last_j:
            eff_stop = stop_price
            if trailing:
                ts = hwm * (1.0 - td)
                if ts > eff_stop:
                    eff_stop = ts
            if low[j] <= eff_stop:
                o = open_[j]
                e_px = o if o < eff_stop else eff_stop
                e_reason = 3 if (trailing and eff_stop > stop_price) else 1
                e_idx = j
                break
            if high[j] >= tp_price:
                e_px = tp_price
                e_reason = 2
                e_idx = j
                break
            if high[j] > hwm:
                hwm = high[j]
            if (not trailing) and hwm >= entry * (1.0 + trail_act_pct):
                trailing = True
            if use_signal_exit:
                p = preds[j]
                if p == p and p < -threshold and (j - i) >= cooldown_bars:
                    e_px = close[j]
                    e_reason = 4
                    e_idx = j
                    break
            if is_eod[j]:
                e_px = close[j]
                e_reason = 5
                e_idx = j
                break
            j += 1

        exit_idx[i] = e_idx
        exit_px[i] = e_px
        reason[i] = e_reason

    return exit_idx, exit_px, reason


@njit(cache=True)
def _exit_walk_kernel_short(close, high, low, open_, atr, preds, is_eod,
                            threshold, atr_stop_mult, atr_trail_mult,
                            trail_act_pct, stop_floor, stop_ceil,
                            tp_rr, tp_ceil, stop_fallback, trail_fallback,
                            cooldown_bars, max_hold, use_signal_exit):
    """The exact MIRROR of _exit_walk_kernel for a SHORT entry.

    Every barrier flips side: the protective stop sits ABOVE entry and is
    hit on bar HIGHS (gap-filled at the worse-for-us max(open, stop)); the
    take-profit sits BELOW entry and is hit on bar LOWS; the trailing stop
    rides a LOW-water-mark and ratchets DOWN; the signal-flip cover fires
    when the prediction turns positive (p > +threshold). EOD and vertical
    barriers are side-agnostic. exit_px is still the realized PRICE (the
    short P&L of (entry - exit_px)/entry is applied by the caller), so the
    reason codes keep identical meaning to the long kernel.

    NOTE on symmetry: the hard-stop / TP / EOD / vertical / signal exits are
    affine-invariant, so short(path) == long(reflect-about-entry(path)) for
    them (the kernel's mirror-symmetry test). The PERCENTAGE trailing stop is
    NOT affine-invariant (ts = lwm*(1+td) vs the reflected long's (2E-lwm)*
    (1-td) differ by 2*td*(E-lwm)) — both are nonetheless the CORRECT
    percentage-trailing definitions for their side; the trailing exit is
    therefore covered by a direct unit test, not by the reflection property.
    """
    n = len(close)
    exit_idx = np.empty(n, dtype=np.int64)
    exit_px = np.empty(n, dtype=np.float64)
    reason = np.zeros(n, dtype=np.int8)

    for i in range(n):
        entry = close[i]
        a = atr[i]
        if a == a and entry > 0:  # not NaN
            sd = (a * atr_stop_mult) / entry
            if sd < stop_floor:
                sd = stop_floor
            elif sd > stop_ceil:
                sd = stop_ceil
            td = (a * atr_trail_mult) / entry
            if td < stop_floor:
                td = stop_floor
            elif td > stop_ceil:
                td = stop_ceil
        else:
            sd = stop_fallback
            td = trail_fallback
        tp_d = sd * tp_rr
        if tp_d > tp_ceil:
            tp_d = tp_ceil
        stop_price = entry * (1.0 + sd)       # stop ABOVE for a short
        tp_price = entry * (1.0 - tp_d)       # take-profit BELOW
        lwm = entry                            # low-water-mark
        trailing = False

        last_j = n - 1
        if max_hold > 0 and i + max_hold < last_j:
            last_j = i + max_hold

        e_idx = last_j
        e_px = close[last_j]
        e_reason = 6 if max_hold > 0 else 0

        j = i + 1
        while j <= last_j:
            eff_stop = stop_price
            if trailing:
                ts = lwm * (1.0 + td)          # trails DOWN with the lwm
                if ts < eff_stop:
                    eff_stop = ts
            if high[j] >= eff_stop:            # stop hit on a HIGH
                o = open_[j]
                e_px = o if o > eff_stop else eff_stop   # gap-up fills worse
                e_reason = 3 if (trailing and eff_stop < stop_price) else 1
                e_idx = j
                break
            if low[j] <= tp_price:             # take-profit hit on a LOW
                e_px = tp_price
                e_reason = 2
                e_idx = j
                break
            if low[j] < lwm:
                lwm = low[j]
            if (not trailing) and lwm <= entry * (1.0 - trail_act_pct):
                trailing = True
            if use_signal_exit:
                p = preds[j]
                if p == p and p > threshold and (j - i) >= cooldown_bars:
                    e_px = close[j]
                    e_reason = 4
                    e_idx = j
                    break
            if is_eod[j]:
                e_px = close[j]
                e_reason = 5
                e_idx = j
                break
            j += 1

        exit_idx[i] = e_idx
        exit_px[i] = e_px
        reason[i] = e_reason

    return exit_idx, exit_px, reason


def exit_walk(close, high, low, open_, atr, is_eod, policy,
              preds=None, threshold=0.0, cooldown_bars=1,
              max_hold=0, use_signal_exit=False, side=1):
    """Vectorized-input wrapper around the kernel.

    Args:
        close/high/low/open_/atr: float arrays (atr NaN -> pct fallbacks).
            NOTE: non-finite HIGH/LOW/OPEN values are NOT validated; every
            barrier comparison against NaN is False, so a NaN bar is
            silently walked past (a stop/TP touched only on that bar is
            missed) and a NaN close at the walk terminus yields a NaN
            exit price. Only ATR may legitimately be NaN.
        is_eod: bool array, True on each day's LAST bar (stocks; all-False
            for 24/7 crypto)
        policy: strategy_config policy dict for the asset class
        preds: model predictions per bar (read ONLY when
            use_signal_exit=True; otherwise ignored)
        threshold: signal-exit trigger — long exits on p < -threshold,
            short on p > +threshold (one scalar, sign-flipped per side)
        cooldown_bars: minimum hold in bars before a signal exit may fire
            ((j - i) >= cooldown_bars); NOT validated — negative values
            behave like 0. Distinct from backtest.py's re-ENTRY cooldown,
            which reuses the same number with a different meaning.
        max_hold: vertical barrier in bars; 0 = unlimited (backtest mode).
            NOT validated — negative values silently behave like 0.
        use_signal_exit: enable the signal-flip exit (requires preds)
        side: +1 long (default — the LIVE path, unchanged), -1 short
            (OFFLINE research only: the mirror kernel for short labels /
            short-edge studies; no live wiring).

    Returns:
        (exit_idx, exit_px, reason_code) arrays, one entry per bar.

    Raises:
        ValueError on array-length mismatch, on use_signal_exit without
        preds, or on a side other than +1/-1.
    """
    # Fail loud BEFORE the kernel: njit does not bounds-check, so a length
    # mismatch that raises IndexError under the pure-python fallback reads
    # out-of-range memory (garbage labels / segfault) on the Jetson.
    n = len(close)
    for name, arr in (('high', high), ('low', low), ('open_', open_),
                      ('atr', atr), ('is_eod', is_eod)):
        if len(arr) != n:
            raise ValueError(
                f"exit_walk: len({name})={len(arr)} != len(close)={n}")
    if preds is not None and len(preds) != n:
        raise ValueError(
            f"exit_walk: len(preds)={len(preds)} != len(close)={n}")
    if use_signal_exit and preds is None:
        raise ValueError("exit_walk: use_signal_exit=True requires preds")
    if side not in (1, -1):
        raise ValueError(f"exit_walk: side must be +1 or -1, got {side!r}")
    if preds is None:
        preds = np.full(n, np.nan)
    kernel = _exit_walk_kernel if side >= 0 else _exit_walk_kernel_short
    return kernel(
        np.ascontiguousarray(close, dtype=np.float64),
        np.ascontiguousarray(high, dtype=np.float64),
        np.ascontiguousarray(low, dtype=np.float64),
        np.ascontiguousarray(open_, dtype=np.float64),
        np.ascontiguousarray(atr, dtype=np.float64),
        np.ascontiguousarray(preds, dtype=np.float64),
        np.ascontiguousarray(is_eod, dtype=np.bool_),
        float(threshold),
        float(policy['atr_stop_mult']), float(policy['atr_trail_mult']),
        float(policy['trail_activate_pct']),
        float(policy['stop_floor_pct']), float(policy['stop_ceil_pct']),
        float(policy['tp_rr']), float(policy['tp_ceil_pct']),
        float(policy['stop_fallback_pct']), float(policy['trail_fallback_pct']),
        int(cooldown_bars), int(max_hold), bool(use_signal_exit),
    )


def eod_mask_from_index(index, asset_type: str) -> np.ndarray:
    """True on each day's last bar (stocks); all-False for crypto.

    'Day' = calendar date of the raw index timestamps (UTC in every real
    caller), NOT the exchange session. Correct while stock frames are
    RTH-only (the current data paths). If extended-hours bars ever enter
    the harvest/backtest frames, the flatten bar shifts after-hours and
    (winter, SIP feed) one session can straddle two UTC dates — verify
    the Jetson parquet's ET hour distribution before changing feeds;
    session-aware masking is a deferred, label-semantics change.

    Two load-bearing conventions callers MUST know:
      - The frame's FINAL bar is ALWAYS flagged True for stocks (mask
        starts all-ones and only [:-1] is overwritten), whatever its
        clock time. Deliberate — it forces a flatten at end-of-data —
        but it conflates 'session end' with 'ran out of data': a
        reason-5 exit at the frame's last bar is not proof a session
        boundary was reached, and a mid-session-truncated frame gets a
        synthetic EOD.
      - The mask is a FORWARD difference (bar i's flag depends on bar
        i+1's date): build it on the FULL frame and slice the MASK,
        never mask a slice — masking a slice manufactures an EOD at the
        slice boundary (see decision_report.replay_entry). Assumes a
        sorted, duplicate-free index; neither is validated.
    """
    n = len(index)
    if asset_type != 'stock':
        return np.zeros(n, dtype=np.bool_)
    dates = np.asarray([t.date() for t in index])
    mask = np.ones(n, dtype=np.bool_)
    if n > 1:
        mask[:-1] = dates[:-1] != dates[1:]
    return mask


def compute_tb_labels(df, forward_bars_list, asset_type: str, side: int = 1):
    """Triple-barrier labels matched to the LIVE exit stack.

    Adds, per fb in forward_bars_list:
      TB_Ret_{fb}   gross % return realized by the exit stack (vertical
                    barrier = fb bars; stocks ALSO exit at each day's last
                    bar — the live ~15:50 flatten, approximated at the
                    day-last bar's CLOSE, fixing the old mismatch where
                    labels spanned multiple days while live stock holds
                    were capped at ~6.5 hours; other live-only overlays
                    the labels do not model are listed in the module
                    docstring)
      TB_Bars_{fb}  bars held
      TB_Reason_{fb} exit reason code

    Bars whose vertical window crosses the end of the series get NaN in
    ALL THREE columns (matching Target_Return semantics) — including
    TB_Reason, since the kernel stamps 'vertical' on truncated windows
    whose barrier was never actually reached.

    side: +1 long (default — what the harvest stamps and the live system
        trades). side=-1 produces SHORT labels from the mirror kernel: the
        gross return is (entry - exit_px)/entry so a price fall is a profit.
        Short labels are OFFLINE research only (short-edge existence study /
        meta_short / q90) — they are NOT written by the live harvest.

    Caveats (verified 2026-07 panel batch B2; pinned by
    tests/test_policy_exits_v3.py):
      - EOD-BAR ENTRIES ARE LABELED: the walk starts at i+1 and never
        reads is_eod[i], so an entry ON a stock's day-last bar holds
        overnight to the NEXT session's EOD (~1 row per ticker per
        trading day). backtest.py and meta_label.py both SKIP those
        entries at their entry gates; only the labels include them.
      - STOCK HORIZON DEGENERACY: the EOD barrier fires within one
        session, so every fb >= bars-per-session yields identical stock
        labels and reason 6 'vertical' is unreachable for stocks; only
        the NaN tails differ across fb.
      - TB_Bars_{fb} is a POSITIONAL offset in the frame passed HERE. It
        is invalid as a row offset after any row filtering (dropna,
        as-of tradability/membership masks) — downstream row-span
        consumers (sample_weights uniqueness/effective-n) require
        consecutive rows.
      - SILENT COLUMN FALLBACKS: a missing High/Low/Open column falls
        back to Close (no intrabar barrier can ever fire); a missing ATR
        column routes EVERY entry to the fixed-percent fallback stops.
        Both change the label distribution materially with no warning.
        Only 'Close' is required (KeyError).
      - df must be ONE ticker's bars with a sorted, unique index; not
        checked — a concatenated multi-ticker frame yields silently
        wrong cross-ticker labels.
      - fb <= 0 is NOT validated: fb=0 maps to the kernel's UNLIMITED
        mode and emits TB_*_0 columns with no truncation NaNs at all.
      - side=-1 emits the SAME column names as side=+1, so stamping both
        onto one frame silently overwrites the long labels. Short TB_Ret
        is GROSS: charge holding-period-dependent short carry via
        short_cost (short_round_trip_cost_pct / borrow_drag_pct with
        hold_days from TB_Bars), NOT fees.round_trip_cost_pct.
    """
    from strategy_config import policy_for
    policy = policy_for(asset_type)

    close = df['Close'].values
    high = df['High'].values if 'High' in df.columns else close
    low = df['Low'].values if 'Low' in df.columns else close
    open_ = df['Open'].values if 'Open' in df.columns else close
    atr = df['ATR'].values if 'ATR' in df.columns else np.full(len(close), np.nan)
    is_eod = eod_mask_from_index(df.index, asset_type)
    n = len(close)
    sign = 1.0 if side >= 0 else -1.0

    out = {}
    for fb in forward_bars_list:
        exit_idx, exit_px, reason = exit_walk(
            close, high, low, open_, atr, is_eod, policy,
            max_hold=int(fb), use_signal_exit=False, side=side)
        ret = sign * (exit_px - close) / np.where(close > 0, close, np.nan) * 100.0
        bars = (exit_idx - np.arange(n)).astype(np.float64)
        reason_f = reason.astype(np.float64)
        # Truncated windows at the series end -> NaN
        invalid = np.arange(n) + fb >= n
        ret[invalid] = np.nan
        bars[invalid] = np.nan
        reason_f[invalid] = np.nan
        out[f'TB_Ret_{fb}'] = ret
        out[f'TB_Bars_{fb}'] = bars
        out[f'TB_Reason_{fb}'] = reason_f
    return out
