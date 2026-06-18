"""The exit-stack walk — ONE implementation shared by three consumers.

    1. backtest.py        — policy replay for the promotion gate
    2. harvest scripts    — triple-barrier label generation (TB_Ret_{fb})
    3. meta_label.py      — meta-labeling dataset (replayed trades)

Lopez de Prado's core point about labels: a model trained on raw fb-bar
forward returns is answering a question the live system never asks — the
bot exits via ATR stops / trailing / take-profit / EOD flatten, not by
holding exactly fb bars. Triple-barrier labels ask the RIGHT question
("what does the exit stack realize from an entry here?"), and sharing one
kernel guarantees label semantics == backtest semantics == live semantics
(stocks even get the EOD flatten as their vertical barrier).

The kernel computes, for EVERY bar i, the exit the live stack would
produce for an entry at bar i's close:
  - gap-aware hard/trailing stop (filled at min(stop, open) on gaps)
  - take-profit at the limit price
  - trailing activation at +trail_act over entry, tracked on bar highs
  - optional signal-flip exit (pred < -threshold after a cooldown)
  - EOD exit for stocks (is_eod mask)
  - vertical barrier at max_hold bars (labels) or unlimited (backtest)
Stop checked FIRST when stop and TP touch in the same bar (conservative).

Exit reason codes:
  0 end_of_data, 1 hard_stop, 2 take_profit, 3 trailing,
  4 signal_sell, 5 eod_flatten, 6 vertical
"""

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba present on all targets
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
        close/high/low/open_/atr: float arrays (atr NaN -> pct fallbacks)
        is_eod: bool array, True on each day's LAST bar (stocks; all-False
            for 24/7 crypto)
        policy: strategy_config policy dict for the asset class
        preds: model predictions per bar (only used with use_signal_exit)
        max_hold: vertical barrier in bars; 0 = unlimited (backtest mode)
        side: +1 long (default — the LIVE path, unchanged), -1 short
            (OFFLINE research only: the mirror kernel for short labels /
            short-edge studies; no live wiring).

    Returns:
        (exit_idx, exit_px, reason_code) arrays, one entry per bar.
    """
    n = len(close)
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
    """True on each day's last bar (stocks); all-False for crypto."""
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
                    bar — exactly the live 15:50 flatten, fixing the old
                    mismatch where labels spanned multiple days while live
                    stock holds were capped at ~6.5 hours)
      TB_Bars_{fb}  bars held
      TB_Reason_{fb} exit reason code

    Bars whose vertical window crosses the end of the series get NaN
    (matching Target_Return semantics).

    side: +1 long (default — what the harvest stamps and the live system
        trades). side=-1 produces SHORT labels from the mirror kernel: the
        gross return is (entry - exit_px)/entry so a price fall is a profit.
        Short labels are OFFLINE research only (short-edge existence study /
        meta_short / q90) — they are NOT written by the live harvest.
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
        # Truncated windows at the series end -> NaN
        invalid = np.arange(n) + fb >= n
        ret[invalid] = np.nan
        bars[invalid] = np.nan
        out[f'TB_Ret_{fb}'] = ret
        out[f'TB_Bars_{fb}'] = bars
        out[f'TB_Reason_{fb}'] = reason.astype(np.float64)
    return out
