"""Tests for policy_exits — panel batch B2 (SACRED KERNEL, docstring/comment
edits only; zero executable-code changes).

These tests PIN CURRENT KERNEL SEMANTICS bar-for-bar, INCLUDING several known
defects that are deferred to the owner (see the NORMATIVE precedence block
and the Caveats sections the batch B2 docstring edits added). A failure here
means the kernel's BEHAVIOR changed — for this shared exit-stack kernel
(backtest.py, the harvest triple-barrier labels, AND meta_label.py all derive
from it) that is never accidental-OK, even when the new behavior looks like
an improvement. Every assertion below pins an EXACT value (never an
`in (...)` membership set).

No pytest.importorskip: policy_exits has a pure-python fallback for numba
(see _HAS_NUMBA below) and every other dependency here (numpy, pandas,
strategy_config) is present on the dev Mac.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import policy_exits
from policy_exits import (exit_walk, compute_tb_labels, eod_mask_from_index,
                          REASON_NAMES, _exit_walk_kernel,
                          _exit_walk_kernel_short, _HAS_NUMBA)
from strategy_config import CRYPTO_POLICY, STOCK_POLICY


class FakeTS:
    def __init__(self, t):
        self.t = t
        self.hour = t.hour

    def date(self):
        return self.t.date()


def _flat(n, px=100.0):
    c = np.full(n, px)
    return c, c + 0.05, c - 0.05, c.copy()


def _walk(c, h, l, o, *, atr=None, is_eod=None, policy=STOCK_POLICY, **kw):
    n = len(c)
    atr = np.full(n, 1.0) if atr is None else atr
    is_eod = np.zeros(n, dtype=bool) if is_eod is None else is_eod
    c = np.asarray(c, dtype=float)
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)
    o = np.asarray(o, dtype=float)
    atr = np.asarray(atr, dtype=float)
    is_eod = np.asarray(is_eod, dtype=bool)
    return exit_walk(c, h, l, o, atr, is_eod, policy, **kw)


def _rth_df(days=3, bars_per_day=7):
    """Flat-100 stock frame, ATR=1.0, hourly bars in day-blocks starting
    2026-05-04 14:30 (+ d days) — the RTH-block shape TestEodEntryLabels and
    TestEodMaskEdges' slice trap depend on."""
    start = dt.datetime(2026, 5, 4, 14, 30)
    idx = []
    for d in range(days):
        day_start = start + dt.timedelta(days=d)
        idx.extend(day_start + dt.timedelta(hours=b) for b in range(bars_per_day))
    n = len(idx)
    close = np.full(n, 100.0)
    return pd.DataFrame(
        {'Close': close, 'High': close + 0.05, 'Low': close - 0.05,
         'Open': close.copy(), 'ATR': np.full(n, 1.0)},
        index=pd.DatetimeIndex(idx))


# --------------------------------------------------------------------------

class TestFormatContract:
    def test_reason_names_exact(self):
        # on-disk contract; harvested Jetson parquet persists these integers
        assert REASON_NAMES == {
            0: 'end_of_data', 1: 'hard_stop', 2: 'take_profit',
            3: 'trailing', 4: 'signal_sell', 5: 'eod_flatten', 6: 'vertical'}

    def test_return_dtypes(self):
        idx, px, reason = _walk(*_flat(10), max_hold=5)
        assert idx.dtype == np.int64
        assert px.dtype == np.float64
        assert reason.dtype == np.int8

    def test_has_numba_flag(self):
        assert isinstance(_HAS_NUMBA, bool)
        try:
            import numba  # noqa: F401
            expected = True
        except ImportError:
            expected = False
        assert _HAS_NUMBA == expected


class TestSameBarPrecedenceLong:
    """n=6, _flat(6), a collision engineered on bar 2; assertions read entry
    row 1 (index 0 is a bar-1 entry that runs into the bar-2 collision)."""

    def test_stop_and_tp_same_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        h[2] = 110.0
        l[2] = 90.0
        idx, px, reason = _walk(c, h, l, o, max_hold=0)
        assert reason[1] == 1
        assert px[1] == 98.0
        assert idx[1] == 2

    def test_stop_and_eod_same_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        l[2] = 90.0
        is_eod = np.zeros(n, dtype=bool)
        is_eod[2] = True
        idx, px, reason = _walk(c, h, l, o, is_eod=is_eod, max_hold=0)
        assert reason[1] == 1
        assert px[1] == 98.0

    def test_tp_and_eod_same_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        h[2] = 110.0
        is_eod = np.zeros(n, dtype=bool)
        is_eod[2] = True
        idx, px, reason = _walk(c, h, l, o, is_eod=is_eod, max_hold=0)
        assert reason[1] == 2
        assert px[1] == 104.0

    def test_signal_and_eod_same_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        preds = np.zeros(n)
        preds[2] = -5.0
        is_eod = np.zeros(n, dtype=bool)
        is_eod[2] = True
        idx, px, reason = _walk(c, h, l, o, is_eod=is_eod, preds=preds,
                                threshold=0.5, cooldown_bars=1,
                                use_signal_exit=True, max_hold=0)
        assert reason[1] == 4
        assert px[1] == 100.0   # the close — signal exits fill flat, no gap logic

    def test_stop_and_signal_same_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        l[2] = 90.0
        preds = np.zeros(n)
        preds[2] = -5.0
        idx, px, reason = _walk(c, h, l, o, preds=preds, threshold=0.5,
                                cooldown_bars=1, use_signal_exit=True, max_hold=0)
        assert reason[1] == 1
        assert px[1] == 98.0

    def test_tp_and_signal_same_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        h[2] = 110.0
        preds = np.zeros(n)
        preds[2] = -5.0
        idx, px, reason = _walk(c, h, l, o, preds=preds, threshold=0.5,
                                cooldown_bars=1, use_signal_exit=True, max_hold=0)
        assert reason[1] == 2
        assert px[1] == 104.0

    def test_eod_exactly_on_vertical_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        is_eod = np.zeros(n, dtype=bool)
        is_eod[2] = True
        idx, px, reason = _walk(c, h, l, o, is_eod=is_eod, max_hold=1)
        # EOD overwrites the vertical default at the same index and price
        assert reason[1] == 5
        assert px[1] == 100.0
        assert idx[1] == 2

    def test_signal_exactly_on_vertical_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        preds = np.zeros(n)
        preds[2] = -5.0
        idx, px, reason = _walk(c, h, l, o, preds=preds, threshold=0.5,
                                cooldown_bars=1, use_signal_exit=True, max_hold=1)
        assert reason[1] == 4

    def test_stop_exactly_on_vertical_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        l[2] = 90.0
        idx, px, reason = _walk(c, h, l, o, max_hold=1)
        assert reason[1] == 1
        assert px[1] == 98.0

    def test_no_barrier_hits_vertical(self):
        n = 6
        c, h, l, o = _flat(n)
        idx, px, reason = _walk(c, h, l, o, max_hold=2)
        assert reason[1] == 6
        assert idx[1] == 3

    def test_no_barrier_unlimited_hits_end_of_data(self):
        # first reason-0 pin in the repo
        n = 6
        c, h, l, o = _flat(n)
        idx, px, reason = _walk(c, h, l, o, max_hold=0)
        assert reason[1] == 0
        assert idx[1] == 5


class TestSameBarPrecedenceShort:
    """Same constructions as TestSameBarPrecedenceLong with side=-1 (stop 102
    above entry, tp 96 below, on STOCK_POLICY + ATR=1.0)."""

    def test_stop_and_tp_same_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        h[2] = 110.0
        l[2] = 90.0
        idx, px, reason = _walk(c, h, l, o, max_hold=0, side=-1)
        assert reason[1] == 1
        assert px[1] == 102.0
        assert idx[1] == 2

    def test_stop_and_eod_same_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        h[2] = 110.0
        is_eod = np.zeros(n, dtype=bool)
        is_eod[2] = True
        idx, px, reason = _walk(c, h, l, o, is_eod=is_eod, max_hold=0, side=-1)
        assert reason[1] == 1
        assert px[1] == 102.0

    def test_tp_and_eod_same_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        l[2] = 90.0
        is_eod = np.zeros(n, dtype=bool)
        is_eod[2] = True
        idx, px, reason = _walk(c, h, l, o, is_eod=is_eod, max_hold=0, side=-1)
        assert reason[1] == 2
        assert px[1] == 96.0

    def test_signal_and_eod_same_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        preds = np.zeros(n)
        preds[2] = 5.0
        is_eod = np.zeros(n, dtype=bool)
        is_eod[2] = True
        idx, px, reason = _walk(c, h, l, o, is_eod=is_eod, preds=preds,
                                threshold=0.5, cooldown_bars=1,
                                use_signal_exit=True, max_hold=0, side=-1)
        assert reason[1] == 4
        assert px[1] == 100.0

    def test_eod_exactly_on_vertical_bar(self):
        n = 6
        c, h, l, o = _flat(n)
        is_eod = np.zeros(n, dtype=bool)
        is_eod[2] = True
        idx, px, reason = _walk(c, h, l, o, is_eod=is_eod, max_hold=1, side=-1)
        assert reason[1] == 5
        assert px[1] == 100.0
        assert idx[1] == 2

    def test_no_barrier_unlimited_hits_end_of_data(self):
        n = 6
        c, h, l, o = _flat(n)
        idx, px, reason = _walk(c, h, l, o, max_hold=0, side=-1)
        assert reason[1] == 0
        assert idx[1] == 5

    def test_no_barrier_hits_vertical(self):
        n = 6
        c, h, l, o = _flat(n)
        idx, px, reason = _walk(c, h, l, o, max_hold=2, side=-1)
        assert reason[1] == 6
        assert idx[1] == 3


class TestTrailingConventions:
    def test_trailing_outranks_tp_same_bar(self):
        c = [100, 110, 110, 110]
        h = [100, 111, 113, 111]
        l = [100, 106, 105, 109]
        o = [100, 106, 110, 110]
        idx, px, reason = _walk(c, h, l, o, atr=np.full(4, np.nan),
                                policy=CRYPTO_POLICY, max_hold=0)
        # bar 2 breaches BOTH tp 112 (high 113) and trail 105.45 (low 105);
        # the kernel books the trail. Live base_loop._manage_stops' elif
        # order is hard_stop -> take_profit -> trailing (opposite precedence).
        assert reason[0] == 3
        assert px[0] == pytest.approx(105.45)
        assert idx[0] == 2

    def test_hwm_one_bar_lag(self):
        c = [100, 108, 103.5, 100]
        h = [100, 108, 104, 100]
        l = [100, 101, 103, 100]
        o = [100, 101, 104, 100]
        idx, px, reason = _walk(c, h, l, o, atr=np.full(4, np.nan),
                                policy=STOCK_POLICY, max_hold=0)
        # bar 1's 108 high arms the trail but cannot tighten bar 1's own stop
        assert idx[0] == 2
        assert reason[0] == 3
        assert px[0] == pytest.approx(103.68)

    def test_armed_trail_below_hard_stop_reports_hard_stop(self):
        # trailing armed but ts=92 < stop 98, so the strict eff_stop>stop_price
        # guard fails -> reason 1, not 3. Unreachable under shipped policies
        # (their trail/stop mults never diverge this far); pinned via a local
        # dict per CLAUDE.md's "never mutate the shared policy dicts" rule.
        LOOSE = dict(STOCK_POLICY)
        LOOSE.update(atr_trail_mult=8.0, trail_activate_pct=0.0)
        n = 5
        c, h, l, o = _flat(n)
        l[2] = 90.0
        idx, px, reason = _walk(c, h, l, o, atr=np.full(n, 1.0), policy=LOOSE,
                                max_hold=0)
        assert reason[0] == 1
        assert px[0] == 98.0

    def test_short_lwm_one_bar_lag(self):
        c = [100, 92, 96.5, 100]
        h = [100, 99, 96.5, 100]
        l = [100, 92, 96, 100]
        o = [100, 99, 96, 100]
        idx, px, reason = _walk(c, h, l, o, atr=np.full(4, np.nan),
                                policy=STOCK_POLICY, max_hold=0, side=-1)
        # gap-fill takes the worse 96 open over the 95.68 trail
        assert idx[0] == 2
        assert reason[0] == 3
        assert px[0] == 96.0


class TestGapConventions:
    def test_long_tp_gap_up_no_improvement(self):
        c = [100, 106, 106]
        h = [100, 107, 106.5]
        l = [100, 105.5, 105.5]
        o = [100, 106, 106]
        idx, px, reason = _walk(c, h, l, o, atr=np.full(3, 1.0),
                                policy=STOCK_POLICY, max_hold=0)
        assert reason[0] == 2
        assert px[0] == 104.0   # not 106 — fills at tp_price, gap ignored

    def test_long_open_through_tp_then_stop_same_bar(self):
        c = [100, 97, 97]
        h = [100, 107, 97.5]
        l = [100, 97, 96.5]
        o = [100, 106, 97]
        idx, px, reason = _walk(c, h, l, o, atr=np.full(3, 1.0),
                                policy=STOCK_POLICY, max_hold=0)
        # live's stock bracket limit would have filled the TP leg ~106 at the
        # open — a deferred model-facing finding; this pins today's kernel
        # semantics (stop checked first, no same-bar TP credit for the gap).
        assert reason[0] == 1
        assert px[0] == 98.0

    def test_long_stop_gap_down_fills_at_open(self):
        c = [100, 90, 90]
        h = [100, 90.5, 90.5]
        l = [100, 89.5, 89.5]
        o = [100, 90, 90]
        idx, px, reason = _walk(c, h, l, o, atr=np.full(3, 1.0),
                                policy=STOCK_POLICY, max_hold=0)
        assert reason[0] == 1
        assert px[0] == 90.0

    def test_short_tp_gap_down(self):
        c = [100, 93, 93]
        h = [100, 93.5, 93.5]
        l = [100, 92.5, 92.5]
        o = [100, 93, 93]
        idx, px, reason = _walk(c, h, l, o, atr=np.full(3, 1.0),
                                policy=STOCK_POLICY, max_hold=0, side=-1)
        assert reason[0] == 2
        assert px[0] == 96.0   # not the better 93 open

    def test_short_stop_gap_up_fills_at_worse_open(self):
        c = [100, 110, 110]
        h = [100, 110.5, 110.5]
        l = [100, 109.5, 109.5]
        o = [100, 110, 110]
        idx, px, reason = _walk(c, h, l, o, atr=np.full(3, 1.0),
                                policy=STOCK_POLICY, max_hold=0, side=-1)
        assert reason[0] == 1
        assert px[0] == 110.0


class TestEodMaskEdges:
    def test_stock_mask_flags_final_bar_mid_session(self):
        idx = pd.date_range('2026-05-01 13:30', periods=5, freq='h', tz='UTC')
        mask = eod_mask_from_index(idx, 'stock')
        assert mask.tolist() == [False, False, False, False, True]

    def test_crypto_mask_all_false(self):
        idx = pd.date_range('2026-05-01 13:30', periods=5, freq='h', tz='UTC')
        mask = eod_mask_from_index(idx, 'crypto')
        assert mask.tolist() == [False, False, False, False, False]

    def test_empty_index(self):
        mask = eod_mask_from_index(pd.DatetimeIndex([]), 'stock')
        assert len(mask) == 0

    def test_single_element_index(self):
        idx = pd.date_range('2026-05-01 13:30', periods=1, freq='h', tz='UTC')
        mask = eod_mask_from_index(idx, 'stock')
        assert mask.tolist() == [True]

    def test_slice_trap(self):
        # build the mask on the FULL frame and slice the MASK vs. masking an
        # already-sliced index — the forward-difference convention means
        # these disagree at the slice boundary (decision_report.replay_entry
        # is the load-bearing caller that must slice the mask, not the frame)
        idx = _rth_df(3, 7).index
        sliced_then_masked = eod_mask_from_index(idx[3:8], 'stock')
        masked_then_sliced = eod_mask_from_index(idx, 'stock')[3:8]
        assert sliced_then_masked.tolist() == [False, False, False, True, True]
        assert masked_then_sliced.tolist() == [False, False, False, True, False]
        assert sliced_then_masked.tolist() != masked_then_sliced.tolist()

    def test_fakets_and_datetimeindex_agree(self):
        # guards any future vectorization of eod_mask_from_index: the
        # duck-typed FakeTS path (real Alpaca bar timestamps in some
        # callers) is load-bearing, not just a pd.DatetimeIndex convenience
        instants = [dt.datetime(2026, 5, 1, 9) + dt.timedelta(hours=h)
                   for h in range(30)]
        a = pd.DatetimeIndex(instants)
        b = [FakeTS(t) for t in instants]
        mask_a = eod_mask_from_index(a, 'stock')
        mask_b = eod_mask_from_index(b, 'stock')
        assert np.array_equal(mask_a, mask_b)


class TestEodEntryLabels:
    """Pins the deferred EOD-BAR-ENTRIES-ARE-LABELED caveat."""

    def test_eod_bar_entry_holds_to_next_session(self):
        df = _rth_df(3, 7)
        eod = eod_mask_from_index(df.index, 'stock')
        assert np.where(eod)[0].tolist() == [6, 13, 20]
        out = compute_tb_labels(df, [12], 'stock')
        assert out['TB_Bars_12'][:9].tolist() == [6, 5, 4, 3, 2, 1, 7, 6, 5]
        # index 6 is an EOD-bar entry holding 7 bars overnight to the NEXT
        # day's EOD, while the backtest.py / meta_label.py replay entry
        # gates (`or is_eod[i]`) both SKIP entries at is_eod bars — only
        # the labels include them.
        assert out['TB_Reason_12'][6] == 5.0
        assert np.isnan(out['TB_Bars_12'][9:]).all()

    def test_truncation_discards_resolved_walk(self):
        df = _rth_df(3, 7)
        eod = eod_mask_from_index(df.index, 'stock')
        out = compute_tb_labels(df, [12], 'stock')
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        open_ = df['Open'].values
        atr = df['ATR'].values
        exit_idx, exit_px, reason = exit_walk(close, high, low, open_, atr,
                                              eod, STOCK_POLICY, max_hold=12)
        # the walk resolves INSIDE the data (a real EOD flatten at bar 13)...
        assert exit_idx[9] == 13
        assert reason[9] == 5
        # ...but compute_tb_labels' horizon-positional truncation mask
        # (arange(n)+fb >= n) NaNs it anyway — a deferred owner decision.
        assert np.isnan(out['TB_Bars_12'][9])


class TestStockHorizonDegeneracy:
    def test_stock_labels_degenerate_across_horizons(self):
        rng = np.random.default_rng(7)
        days, bars_per_day = 40, 7
        start = dt.datetime(2026, 1, 5, 14, 30)
        idx = []
        for d in range(days):
            day_start = start + dt.timedelta(days=d)
            idx.extend(day_start + dt.timedelta(hours=b)
                      for b in range(bars_per_day))
        n = len(idx)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.006, n))
        spread = np.abs(rng.normal(0, 0.004, n)) * close
        high = close + spread
        low = close - spread
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        atr = np.full(n, 1.2)

        df = pd.DataFrame({'Close': close, 'High': high, 'Low': low,
                           'Open': open_, 'ATR': atr},
                          index=pd.DatetimeIndex(idx))
        out = compute_tb_labels(df, [12, 48], 'stock')
        r12, r48 = out['TB_Ret_12'], out['TB_Ret_48']
        common = np.isfinite(r12) & np.isfinite(r48)
        assert common.sum() > 0
        # forward_bars is a no-op dimension for stock tb-targets: the EOD
        # barrier always fires first within the 7-bar session — deferred.
        assert np.array_equal(r12[common], r48[common])
        reason48 = out['TB_Reason_48']
        finite48 = reason48[np.isfinite(reason48)]
        assert not (finite48 == 6.0).any()   # 'vertical' is unreachable

        # Crypto control on the SAME price arrays over a contiguous hourly
        # index (no EOD barrier at all): the two horizons DO differ.
        idx_c = pd.date_range('2026-01-05 14:30', periods=n, freq='h', tz='UTC')
        df_c = pd.DataFrame({'Close': close, 'High': high, 'Low': low,
                            'Open': open_, 'ATR': atr}, index=idx_c)
        out_c = compute_tb_labels(df_c, [12, 48], 'crypto')
        r12c, r48c = out_c['TB_Ret_12'], out_c['TB_Ret_48']
        common_c = np.isfinite(r12c) & np.isfinite(r48c)
        assert not np.array_equal(r12c[common_c], r48c[common_c])


class TestHorizonDerivability:
    """Property test: exit_walk(max_hold=fb) must be exactly derivable from
    a single exit_walk(max_hold=0) walk — protects prefix-determinism and
    gates the deferred single-pass compute_tb_labels refactor."""

    @pytest.mark.parametrize('seed', range(6))
    def test_max_hold_derivable_from_unlimited_walk(self, seed):
        n = 400
        rng = np.random.default_rng(seed)
        rets = rng.normal(0, 0.008, n)
        close = 100 * np.cumprod(1 + rets)
        spread = 0.004 * close
        high = close + spread
        low = close - spread
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        atr = np.abs(rng.normal(1.0, 0.2, n))

        for policy, is_eod in ((CRYPTO_POLICY, np.zeros(n, dtype=bool)),
                               (STOCK_POLICY, (np.arange(n) % 7) == 6)):
            for side in (1, -1):
                base_idx, base_px, base_reason = exit_walk(
                    close, high, low, open_, atr, is_eod, policy,
                    max_hold=0, side=side)
                for fb in (6, 12, 24, 48):
                    cap = np.minimum(np.arange(n) + fb, n - 1)
                    touched = (base_reason != 0) & (base_idx <= cap)
                    idx = np.where(touched, base_idx, cap)
                    px = np.where(touched, base_px, close[cap])
                    reason = np.where(touched, base_reason, 6)

                    idx2, px2, reason2 = exit_walk(
                        close, high, low, open_, atr, is_eod, policy,
                        max_hold=fb, side=side)
                    assert np.array_equal(idx, idx2)
                    assert np.array_equal(reason, reason2)
                    assert np.allclose(px, px2, rtol=0, atol=0)


class TestDegenerateParams:
    """Pins today's UNVALIDATED behavior for degenerate parameters; comments
    name the deferred guards each scenario would need."""

    def test_negative_max_hold_same_as_zero(self):
        # max_hold<=0 is not validated: -1 silently falls through to the
        # same unlimited-mode branch as 0 (see the two `> 0` guards).
        n = 20
        c, h, l, o = _flat(n)
        r_neg = _walk(c, h, l, o, max_hold=-1)
        r_zero = _walk(c, h, l, o, max_hold=0)
        assert np.array_equal(r_neg[0], r_zero[0])
        assert np.array_equal(r_neg[1], r_zero[1])
        assert np.array_equal(r_neg[2], r_zero[2])

    def test_fb_zero_is_unlimited_mode_zero_truncation_nans(self):
        # fb<=0 is not validated: fb=0 routes into the kernel's unlimited
        # mode, and the truncation mask (arange(n)+0 >= n) is never true —
        # so unlike every fb>0, NO row is truncated to NaN.
        n = 20
        close = np.full(n, 100.0)
        idx = pd.DatetimeIndex([dt.datetime(2026, 5, 1) + dt.timedelta(hours=h)
                                for h in range(n)])
        df = pd.DataFrame({'Close': close, 'High': close + 0.05,
                           'Low': close - 0.05, 'Open': close.copy(),
                           'ATR': np.full(n, 1.0)}, index=idx)
        out = compute_tb_labels(df, [0], 'crypto')
        assert set(out) == {'TB_Ret_0', 'TB_Bars_0', 'TB_Reason_0'}
        assert np.isnan(out['TB_Ret_0']).sum() == 0
        assert (out['TB_Reason_0'] == 0.0).all()
        expect_bars = np.array([n - 1 - i for i in range(n)], dtype=float)
        assert np.array_equal(out['TB_Bars_0'], expect_bars)

    def test_cooldown_zero_same_as_one(self):
        # cooldown_bars is not validated either; 0 and 1 behave identically
        # here because the strict (j - i) >= cooldown_bars check only ever
        # sees j - i >= 1 (the walk starts at j = i + 1).
        n = 20
        c, h, l, o = _flat(n)
        preds = np.zeros(n)
        preds[3] = -5.0
        r0 = _walk(c, h, l, o, policy=CRYPTO_POLICY, preds=preds,
                  threshold=0.5, cooldown_bars=0, max_hold=0,
                  use_signal_exit=True)
        r1 = _walk(c, h, l, o, policy=CRYPTO_POLICY, preds=preds,
                  threshold=0.5, cooldown_bars=1, max_hold=0,
                  use_signal_exit=True)
        assert np.array_equal(r0[0], r1[0])
        assert np.array_equal(r0[1], r1[1])
        assert np.array_equal(r0[2], r1[2])

    def test_cooldown_boundary_inclusive(self):
        n = 20
        c, h, l, o = _flat(n)
        preds = np.zeros(n)
        preds[2] = -5.0
        preds[10] = -5.0
        idx, px, reason = _walk(c, h, l, o, policy=CRYPTO_POLICY, preds=preds,
                                threshold=0.5, cooldown_bars=2, max_hold=0,
                                use_signal_exit=True)
        assert idx[0] == 2 and reason[0] == 4     # j-i==2 satisfies >=2, fires
        assert idx[1] == 10 and reason[1] == 4    # bar 2 is inside entry-1's hold

    def test_preds_ignored_when_signal_exit_disabled(self):
        n = 20
        c, h, l, o = _flat(n)
        preds = np.full(n, -9.0)
        idx, px, reason = _walk(c, h, l, o, policy=CRYPTO_POLICY, preds=preds,
                                threshold=0.5, max_hold=0,
                                use_signal_exit=False)
        assert reason[0] == 0
        assert idx[0] == n - 1


class TestSilentDataHazards:
    """Pins current fail-open behavior on malformed input. These are
    deferred DEFECTS, not xfail markers — they must PASS today."""

    def test_control_stop_breach_detected(self):
        n = 5
        c, h, l, o = _flat(n)
        l[2] = 50.0
        idx, px, reason = _walk(c, h, l, o, max_hold=0)
        assert reason[0] == 1
        assert idx[0] == 2

    def test_nan_bar_silently_walked_past(self):
        # identical to the control frame except the breaching bar's own
        # high/low are NaN — every barrier comparison against NaN is False,
        # so the genuine stop breach silently vanishes and the walk runs to
        # end-of-data instead. Deferred: exit_walk does not validate
        # non-finite HIGH/LOW/OPEN.
        n = 5
        c, h, l, o = _flat(n)
        h[2] = np.nan
        l[2] = np.nan
        idx, px, reason = _walk(c, h, l, o, max_hold=0)
        assert reason[0] == 0
        assert idx[0] == 4

    def test_zero_close_fabricates_a_take_profit(self):
        close = np.array([100, 0, 100, 100, 100, 100, 100, 100], dtype=float)
        high = close + 0.05
        low = close - 0.05
        open_ = close.copy()
        idx = pd.DatetimeIndex([dt.datetime(2026, 5, 1) + dt.timedelta(hours=h)
                                for h in range(len(close))])
        df = pd.DataFrame({'Close': close, 'High': high, 'Low': low,
                           'Open': open_, 'ATR': np.full(len(close), 1.0)},
                          index=idx)
        out = compute_tb_labels(df, [3], 'crypto')
        # row 0: a real crash into the zero-close bar -> genuine hard_stop
        assert out['TB_Ret_3'][0] == pytest.approx(-100.0)
        assert out['TB_Reason_3'][0] == 1.0
        # row 1 (entry AT the 0-close bar): entry=0 skips the ATR/fallback
        # stop-distance branch (entry > 0 guard), so tp_price == 0 too, and
        # bar 2's high (>= 0) fires an immediate, fabricated take_profit at
        # price 0. TB_Ret is NaN (close[i]<=0 guard in compute_tb_labels)
        # but TB_Bars/TB_Reason are NOT — violating the "ALL THREE columns"
        # NaN-alignment the docstring otherwise promises. Deferred.
        assert np.isnan(out['TB_Ret_3'][1])
        assert out['TB_Bars_3'][1] == 1.0
        assert out['TB_Reason_3'][1] == 2.0


class TestShortLabelKeyCollision:
    def test_side_plus_and_minus_share_column_names(self):
        # format-contract collision: side=-1 stamps the SAME TB_* column
        # names as side=+1, so writing both onto one frame silently
        # overwrites the long labels. Deferred key_suffix decision.
        n = 20
        close = np.full(n, 100.0)
        idx = pd.DatetimeIndex([dt.datetime(2026, 5, 1) + dt.timedelta(hours=h)
                                for h in range(n)])
        df = pd.DataFrame({'Close': close, 'High': close + 0.05,
                           'Low': close - 0.05, 'Open': close.copy(),
                           'ATR': np.full(n, 1.0)}, index=idx)
        expect = {'TB_Ret_12', 'TB_Bars_12', 'TB_Reason_12'}
        assert set(compute_tb_labels(df, [12], 'crypto', side=1)) == expect
        assert set(compute_tb_labels(df, [12], 'crypto', side=-1)) == expect


class TestNumbaParity:
    """Compares the wrapper's dispatch against a direct py_func call built
    with the exact 20-argument construction tests/test_short_kernel.py's
    TestLongPathUnchanged uses. Tautological where numba is absent (this
    Mac); on CI/Jetson (numba installed — ci.yml:36) this is the real
    compiled-vs-reference A/B."""

    @pytest.mark.parametrize('seed', range(4))
    def test_py_func_matches_wrapper_dispatch(self, seed):
        py_long = getattr(_exit_walk_kernel, 'py_func', _exit_walk_kernel)
        py_short = getattr(_exit_walk_kernel_short, 'py_func',
                          _exit_walk_kernel_short)
        rng = np.random.default_rng(seed)
        n = 60
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
        spread = np.abs(rng.normal(0, 0.004, n)) * close
        high = close + spread
        low = close - spread
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        atr = np.abs(rng.normal(1.0, 0.2, n))
        is_eod = np.zeros(n, dtype=bool)
        preds = np.full(n, np.nan)

        for max_hold in (0, 24):
            for policy in (CRYPTO_POLICY, STOCK_POLICY):
                for side, py_kernel in ((1, py_long), (-1, py_short)):
                    a, b, c = exit_walk(close, high, low, open_, atr, is_eod,
                                        policy, max_hold=max_hold, side=side)
                    d, e, f = py_kernel(
                        np.ascontiguousarray(close),
                        np.ascontiguousarray(high),
                        np.ascontiguousarray(low),
                        np.ascontiguousarray(open_),
                        np.ascontiguousarray(atr),
                        np.ascontiguousarray(preds),
                        np.ascontiguousarray(is_eod), 0.0,
                        policy['atr_stop_mult'], policy['atr_trail_mult'],
                        policy['trail_activate_pct'], policy['stop_floor_pct'],
                        policy['stop_ceil_pct'], policy['tp_rr'],
                        policy['tp_ceil_pct'], policy['stop_fallback_pct'],
                        policy['trail_fallback_pct'], 1, int(max_hold), False)
                    np.testing.assert_array_equal(a, d)
                    np.testing.assert_array_equal(b, e)
                    np.testing.assert_array_equal(c, f)


class TestDocstringContracts:
    """Pins the exact tokens Part 1's docstring edits introduced (and the
    b16 token constraints they must never violate)."""

    def test_module_doc_tokens(self):
        doc = policy_exits.__doc__
        for token in ('NORMATIVE', 'HOLD_RANK', 'llm_veto',
                     'OVERNIGHT_SLEEVE', 'gap asymmetry', 'enforcement lag'):
            assert token in doc, f'missing overlay/precedence mention: {token}'

    def test_exit_walk_doc_tokens(self):
        doc = exit_walk.__doc__
        for token in ('threshold', 'cooldown_bars', 'use_signal_exit'):
            assert token in doc

    def test_eod_mask_doc_tokens(self):
        doc = eod_mask_from_index.__doc__
        assert 'FINAL bar' in doc
        assert 'FORWARD difference' in doc

    def test_compute_tb_labels_doc_tokens(self):
        doc = compute_tb_labels.__doc__
        for token in ('short_cost', 'POSITIONAL', 'DEGENERACY'):
            assert token in doc

    def test_b16_invariants_still_hold(self):
        # double-guard against tests/test_review_b16.py's own assertions
        assert 'guarantees label semantics' not in policy_exits.__doc__
        assert 'exactly the live 15:50' not in compute_tb_labels.__doc__
