"""Wave-5 T1-4: offline side-aware short exit kernel (policy_exits).

The short kernel must be the EXACT mirror of the long kernel. The decisive
test reflects a price path about the entry price — which turns a short into a
long — and asserts short(original) == long(reflected) bar-for-bar (same exit
bar, same reason, equal realized return). Plus one explicit scenario per exit
reason, and a guard that side=+1 leaves the live long path untouched."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy_exits import exit_walk, compute_tb_labels, _exit_walk_kernel

# Fixed synthetic policy so barriers are known regardless of config drift.
# atr NaN -> fallbacks: stop 3%, trail 3%; tp_rr 2 -> tp 6%; trail activ 2%.
POLICY = {
    'atr_stop_mult': 1.0, 'atr_trail_mult': 1.0, 'trail_activate_pct': 0.02,
    'stop_floor_pct': 0.01, 'stop_ceil_pct': 0.20, 'tp_rr': 2.0,
    'tp_ceil_pct': 0.30, 'stop_fallback_pct': 0.03, 'trail_fallback_pct': 0.03,
}
NA = np.nan


def _short(close, high, low, open_, atr=None, is_eod=None, **kw):
    n = len(close)
    atr = np.full(n, NA) if atr is None else np.asarray(atr, float)
    is_eod = np.zeros(n, bool) if is_eod is None else np.asarray(is_eod, bool)
    return exit_walk(np.asarray(close, float), np.asarray(high, float),
                     np.asarray(low, float), np.asarray(open_, float),
                     atr, is_eod, POLICY, side=-1, **kw)


class TestMirrorSymmetry:
    """short(path) == long(reflect(path)) for entry bar 0, over random tails."""

    def _random_path(self, seed, n=40, eod=False, with_atr=True):
        rng = np.random.RandomState(seed)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        close[0] = 100.0
        prev = np.concatenate([[100.0], close[:-1]])
        op = prev * (1 + rng.normal(0, 0.005, n))
        u = np.abs(rng.normal(0, 0.01, n)) * close
        high = np.maximum(op, close) + u
        low = np.minimum(op, close) - u
        atr = (np.abs(rng.normal(0, 1.5, n)) + 0.5) if with_atr else np.full(n, NA)
        is_eod = np.zeros(n, bool)
        if eod:
            is_eod[n // 2] = True
        return close, high, low, op, atr, is_eod

    @pytest.mark.parametrize('seed', range(12))
    def test_short_equals_long_on_reflected(self, seed):
        # CARVE-OUT (wave-7 integrity fix): reflection about E is affine, but a
        # PERCENTAGE trailing stop is NOT affine-invariant — the short trail
        # ts=lwm*(1+td) and the reflected-long trail (2E-lwm)*(1-td) differ by
        # 2*td*(E-lwm). So the mirror identity holds ONLY for non-trailing
        # exits (hard-stop / TP / EOD / vertical / signal). Trailing fills are
        # asserted DIRECTLY in test_trailing_definition_direct instead. We skip
        # any seed whose bar-0 exit is trailing (reason 3) on either side.
        eod = seed % 3 == 0
        close, high, low, op, atr, is_eod = self._random_path(seed, eod=eod)
        E = close[0]
        # short on the original
        s_idx, s_px, s_rsn = _short(close, high, low, op, atr, is_eod)
        # long on the reflection about E: highs<->lows swap
        rc = 2 * E - close
        ro = 2 * E - op
        rh = 2 * E - low
        rl = 2 * E - high
        l_idx, l_px, l_rsn = exit_walk(rc, rh, rl, ro, atr, is_eod, POLICY,
                                       side=1)
        if 3 in (int(s_rsn[0]), int(l_rsn[0])):
            pytest.skip("trailing exit — not affine-invariant; see direct test")
        # entry bar 0 must agree exactly for non-trailing exits
        assert s_idx[0] == l_idx[0]
        assert s_rsn[0] == l_rsn[0]
        short_ret = (E - s_px[0]) / E
        long_ret = (l_px[0] - E) / E
        assert short_ret == pytest.approx(long_ret, abs=1e-9)

    def test_trailing_definition_direct(self):
        # Directly assert the SHORT percentage-trailing definition (no
        # reflection): after the low-water-mark activates the trail, the exit
        # fires when a later HIGH reaches lwm*(1+td) and fills exactly there.
        # td fallback = 3% (atr NaN), trail activates at -2% from entry.
        E = 100.0
        # bar1: fall to low 96 (lwm=96, profit 4% >= 2% -> trail armed),
        #       trail stop = 96*1.03 = 98.88; bar1 high 96.5 < 98.88 (no hit)
        # bar2: rebound, high 99 >= 98.88 -> trailing cover at 98.88
        c = [100.0, 96.2, 98.5]
        h = [100.0, 96.5, 99.0]
        lo = [100.0, 96.0, 98.0]
        o = [100.0, 96.4, 98.2]
        idx, px, rsn = _short(c, h, lo, o)
        td = POLICY['trail_fallback_pct']          # 0.03
        lwm = 96.0
        assert rsn[0] == 3 and idx[0] == 2
        assert px[0] == pytest.approx(lwm * (1.0 + td), abs=1e-9)  # 98.88
        # and it is strictly tighter than the initial stop (103) -> a real
        # trail, not the hard stop
        assert px[0] < E * (1.0 + POLICY['stop_fallback_pct'])

    def test_signal_cover_mirrors_long_sell(self):
        # short covers on p > +thr; long sells on (-p) < -thr -> negate preds.
        close, high, low, op, atr, is_eod = self._random_path(5)
        E = close[0]
        preds = np.linspace(-1, 1, len(close))
        s_idx, s_px, _ = _short(close, high, low, op, atr, is_eod,
                                preds=preds, threshold=0.3, cooldown_bars=1,
                                use_signal_exit=True)
        rc, ro = 2 * E - close, 2 * E - op
        rh, rl = 2 * E - low, 2 * E - high
        l_idx, l_px, _ = exit_walk(rc, rh, rl, ro, atr, is_eod, POLICY,
                                   side=1, preds=-preds, threshold=0.3,
                                   cooldown_bars=1, use_signal_exit=True)
        assert s_idx[0] == l_idx[0]
        assert (E - s_px[0]) / E == pytest.approx((l_px[0] - E) / E, abs=1e-9)


class TestShortScenarios:
    """One hand-built path per exit reason, read at entry bar 0."""

    def test_hard_stop_above(self):
        # entry 100, stop +3% = 103; bar1 spikes through on the high
        c = [100, 102]; h = [100, 103.5]; lo = [100, 101]; o = [100, 101]
        idx, px, rsn = _short(c, h, lo, o)
        assert rsn[0] == 1 and idx[0] == 1
        assert px[0] == pytest.approx(103.0)  # open 101 < stop -> filled at stop

    def test_gap_up_through_stop_fills_at_open(self):
        c = [100, 102]; h = [100, 106]; lo = [100, 104]; o = [100, 104.5]
        idx, px, rsn = _short(c, h, lo, o)
        assert rsn[0] == 1
        assert px[0] == pytest.approx(104.5)  # gap-up open is worse than stop

    def test_take_profit_below(self):
        # tp -6% = 94; price falls to it, stop (103) untouched
        c = [100, 96]; h = [100, 100.5]; lo = [100, 93]; o = [100, 99]
        idx, px, rsn = _short(c, h, lo, o)
        assert rsn[0] == 2 and px[0] == pytest.approx(94.0)

    def test_eod_flatten(self):
        # stock-style: no barrier hit, day ends on bar1 -> exit at close
        c = [100, 99]; h = [100, 101]; lo = [100, 98]; o = [100, 99.5]
        idx, px, rsn = _short(c, h, lo, o, is_eod=[False, True])
        assert rsn[0] == 5 and px[0] == pytest.approx(99.0)

    def test_vertical_barrier(self):
        c = [100, 99.5, 99, 98.5]; h = [100, 100.2, 100, 99]
        lo = [100, 99, 98.6, 98]; o = [100, 99.8, 99.2, 98.8]
        idx, px, rsn = _short(c, h, lo, o, max_hold=2)
        assert rsn[0] == 6 and idx[0] == 2  # exits at i+max_hold

    def test_trailing_stop_ratchets_down(self):
        # fall to lwm 97 (activates trail at <=98), trail stop = 97*1.03=99.91,
        # then a rebound high hits it before initial stop (103) or tp (94).
        c = [100, 97.5, 99.8]
        h = [100, 98.0, 100.0]
        lo = [100, 97.0, 99.0]
        o = [100, 98.0, 99.5]
        idx, px, rsn = _short(c, h, lo, o)
        assert rsn[0] == 3 and idx[0] == 2
        assert px[0] == pytest.approx(99.91, abs=1e-6)


class TestComputeTBLabelsShort:
    def _frame(self, closes):
        n = len(closes)
        idx = pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC')
        c = np.array(closes, float)
        return pd.DataFrame({'Open': c, 'High': c * 1.001, 'Low': c * 0.999,
                             'Close': c, 'ATR': np.full(n, NA)}, index=idx)

    def test_short_label_profits_on_falling_price(self):
        # steady decline -> short TB_Ret should be POSITIVE (mirror of long<0)
        df = self._frame([100, 99, 98, 97, 96, 95, 94, 93])
        out_long = compute_tb_labels(df, [3], 'crypto', side=1)
        out_short = compute_tb_labels(df, [3], 'crypto', side=-1)
        # first bar's long label is negative, short label is its negation
        assert out_long['TB_Ret_3'][0] < 0
        assert out_short['TB_Ret_3'][0] == pytest.approx(-out_long['TB_Ret_3'][0],
                                                         abs=1e-9)

    def test_bars_and_reason_keys_present(self):
        df = self._frame([100, 100.5, 101, 101.5, 102])
        out = compute_tb_labels(df, [2], 'crypto', side=-1)
        assert {'TB_Ret_2', 'TB_Bars_2', 'TB_Reason_2'} <= set(out)


class TestLongPathUnchanged:
    """side=+1 must dispatch to the original kernel, byte-for-byte."""

    def test_dispatch_matches_direct_kernel_call(self):
        rng = np.random.RandomState(99)
        n = 30
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        high = close * 1.01
        low = close * 0.99
        op = close * 1.001
        atr = np.full(n, NA)
        is_eod = np.zeros(n, bool)
        preds = np.full(n, NA)
        a, b, c = exit_walk(close, high, low, op, atr, is_eod, POLICY, side=1)
        d, e, f = _exit_walk_kernel(
            np.ascontiguousarray(close), np.ascontiguousarray(high),
            np.ascontiguousarray(low), np.ascontiguousarray(op),
            np.ascontiguousarray(atr), np.ascontiguousarray(preds),
            np.ascontiguousarray(is_eod), 0.0,
            POLICY['atr_stop_mult'], POLICY['atr_trail_mult'],
            POLICY['trail_activate_pct'], POLICY['stop_floor_pct'],
            POLICY['stop_ceil_pct'], POLICY['tp_rr'], POLICY['tp_ceil_pct'],
            POLICY['stop_fallback_pct'], POLICY['trail_fallback_pct'], 1, 0,
            False)
        np.testing.assert_array_equal(a, d)
        np.testing.assert_array_equal(b, e)
        np.testing.assert_array_equal(c, f)
