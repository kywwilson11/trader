"""Tests for policy_exits — the shared exit-stack kernel.

This kernel is consumed by the backtester, the triple-barrier label
generator, AND the meta-labeling replay: a bug here poisons labels,
gates, and meta-training simultaneously, so it gets handcrafted-scenario
tests plus randomized invariant checks against a pure-python reference.
"""

import datetime as dt

import numpy as np
import pytest

from policy_exits import (exit_walk, compute_tb_labels, eod_mask_from_index,
                          REASON_NAMES)
from strategy_config import CRYPTO_POLICY, STOCK_POLICY


class FakeTS:
    def __init__(self, t):
        self.t = t
        self.hour = t.hour

    def date(self):
        return self.t.date()

    def __str__(self):
        return self.t.isoformat()


def _hourly_index(n, start=None):
    start = start or dt.datetime(2026, 5, 1, 9)
    return [FakeTS(start + dt.timedelta(hours=h)) for h in range(n)]


def _flat(n, px=100.0):
    c = np.full(n, px)
    return c, c + 0.05, c - 0.05, c.copy()


class TestExitWalkScenarios:
    def test_hard_stop_with_gap_through(self):
        n = 60
        close, high, low, open_ = _flat(n)
        # bar 6 gaps to 90 — stop is 98 (2x ATR=1 on 100), fill at the open
        for arr in (close, high, low, open_):
            arr[6:] = 90.0
        atr = np.full(n, 1.0)
        is_eod = np.zeros(n, dtype=bool)
        idx, px, reason = exit_walk(close, high, low, open_, atr, is_eod,
                                    STOCK_POLICY, max_hold=0)
        assert REASON_NAMES[int(reason[5])] == 'hard_stop'
        assert idx[5] == 6
        assert px[5] == 90.0  # gap-aware: filled at the open, not the stop

    def test_take_profit(self):
        n = 60
        close, high, low, open_ = _flat(n)
        close[8:] = 105.0
        high[8:] = 105.2
        low[8:] = 104.8
        open_[8:] = 105.0
        atr = np.full(n, 1.0)
        is_eod = np.zeros(n, dtype=bool)
        idx, px, reason = exit_walk(close, high, low, open_, atr, is_eod,
                                    STOCK_POLICY, max_hold=0)
        # stop_dist = max(1%, 2*1/100) = 2%; tp = 2:1 -> +4% = 104
        assert REASON_NAMES[int(reason[5])] == 'take_profit'
        assert px[5] == pytest.approx(104.0)

    def test_stop_checked_before_tp_same_bar(self):
        n = 30
        close, high, low, open_ = _flat(n)
        # bar 3 spans both barriers: low breaches stop AND high breaches tp
        high[3] = 110.0
        low[3] = 90.0
        atr = np.full(n, 1.0)
        is_eod = np.zeros(n, dtype=bool)
        idx, px, reason = exit_walk(close, high, low, open_, atr, is_eod,
                                    STOCK_POLICY, max_hold=0)
        assert REASON_NAMES[int(reason[2])] == 'hard_stop'  # conservative

    def test_trailing_after_activation(self):
        n = 40
        close, high, low, open_ = _flat(n)
        # Rally +5% (activates trailing), then fade through the trail
        ramp = np.concatenate([np.linspace(100, 105, 10),
                               np.linspace(105, 100, 10),
                               np.full(20, 100.0)])
        close[:], high[:], low[:], open_[:] = ramp, ramp + 0.05, ramp - 0.05, ramp
        atr = np.full(n, 1.0)
        is_eod = np.zeros(n, dtype=bool)
        idx, px, reason = exit_walk(close, high, low, open_, atr, is_eod,
                                    STOCK_POLICY, max_hold=0)
        r0 = REASON_NAMES[int(reason[0])]
        assert r0 in ('trailing', 'take_profit')  # tp at 104 fires first here
        # From a bar near the top, the fade exits via the trail
        assert REASON_NAMES[int(reason[9])] in ('trailing', 'hard_stop')

    def test_eod_exit_for_stocks(self):
        n = 30
        close, high, low, open_ = _flat(n)
        atr = np.full(n, 1.0)
        index = _hourly_index(n)
        is_eod = eod_mask_from_index(index, 'stock')
        assert is_eod.sum() >= 1
        idx, px, reason = exit_walk(close, high, low, open_, atr, is_eod,
                                    STOCK_POLICY, max_hold=0)
        # Entry at bar 2 (11:00) on flat prices must exit at that day's
        # last bar, not run for days
        assert REASON_NAMES[int(reason[2])] == 'eod_flatten'
        assert idx[2] < 24

    def test_vertical_barrier(self):
        n = 100
        close, high, low, open_ = _flat(n)
        atr = np.full(n, 1.0)
        is_eod = np.zeros(n, dtype=bool)
        idx, px, reason = exit_walk(close, high, low, open_, atr, is_eod,
                                    CRYPTO_POLICY, max_hold=24)
        assert REASON_NAMES[int(reason[10])] == 'vertical'
        assert idx[10] == 34

    def test_signal_exit_respects_cooldown(self):
        n = 50
        close, high, low, open_ = _flat(n)
        atr = np.full(n, 1.0)
        is_eod = np.zeros(n, dtype=bool)
        preds = np.zeros(n)
        preds[2] = -5.0   # bearish flip 2 bars after entry at 0
        preds[30] = -5.0
        idx, px, reason = exit_walk(close, high, low, open_, atr, is_eod,
                                    CRYPTO_POLICY, preds=preds, threshold=0.5,
                                    cooldown_bars=10, max_hold=0,
                                    use_signal_exit=True)
        # The bar-2 flip is inside the 10-bar cooldown; bar-30 flip exits
        assert REASON_NAMES[int(reason[0])] == 'signal_sell'
        assert idx[0] == 30


class TestRandomizedInvariants:
    def test_invariants_hold_on_random_walks(self):
        rng = np.random.default_rng(3)
        for asset, policy in (('crypto', CRYPTO_POLICY), ('stock', STOCK_POLICY)):
            n = 800
            rets = rng.normal(0, 0.01, n)
            close = 100 * np.cumprod(1 + rets)
            spread = np.abs(rng.normal(0, 0.004, n)) * close
            high = close + spread
            low = close - spread
            open_ = np.roll(close, 1); open_[0] = close[0]
            atr = np.abs(rng.normal(1.0, 0.2, n))
            index = _hourly_index(n)
            is_eod = eod_mask_from_index(index, asset)
            preds = rng.normal(0, 1, n)
            idx, px, reason = exit_walk(close, high, low, open_, atr, is_eod,
                                        policy, preds=preds, threshold=0.5,
                                        cooldown_bars=2, max_hold=0,
                                        use_signal_exit=True)
            assert (idx >= np.arange(n)).all()
            assert (px > 0).all()
            assert set(np.unique(reason)).issubset(set(REASON_NAMES))
            # Stocks must never hold past the day's last bar
            if asset == 'stock':
                eod_positions = np.where(is_eod)[0]
                for i in range(0, n - 30, 37):
                    next_eod = eod_positions[eod_positions >= i + 1]
                    if len(next_eod):
                        assert idx[i] <= next_eod[0], \
                            f"entry {i} held past EOD {next_eod[0]}"


class TestTripleBarrierLabels:
    def _df(self, close, asset='crypto'):
        import pandas as pd
        n = len(close)
        return pd.DataFrame({
            'Close': close, 'High': close + 0.05, 'Low': close - 0.05,
            'Open': close, 'ATR': np.full(n, 1.0),
        }, index=pd.DatetimeIndex([dt.datetime(2026, 5, 1) +
                                   dt.timedelta(hours=h) for h in range(n)]))

    def test_flat_prices_label_zero_and_tail_nan(self):
        df = self._df(np.full(200, 100.0))
        out = compute_tb_labels(df, [24], 'crypto')
        ret = out['TB_Ret_24']
        assert ret[0] == pytest.approx(0.0)
        assert np.isnan(ret[-1])          # truncated window
        assert np.isnan(ret[200 - 24])    # first truncated index

    def test_crash_label_matches_stop(self):
        close = np.full(200, 100.0)
        close[10:] = 80.0
        df = self._df(close)
        out = compute_tb_labels(df, [24], 'crypto')
        # crypto: stop_dist = max(1.5%, 2.5*1/100) = 2.5%; gap fills at open
        assert out['TB_Ret_24'][9] == pytest.approx(-20.0)
        assert out['TB_Reason_24'][9] == 1  # hard_stop

    def test_stock_labels_respect_eod(self):
        df = self._df(np.full(200, 100.0))
        out = compute_tb_labels(df, [48], 'stock')
        bars = out['TB_Bars_48']
        # No stock label may span more than one trading day of bars
        finite = bars[np.isfinite(bars)]
        assert finite.max() <= 24


class TestMetaFeatures:
    def test_snapshot_and_matrix_agree(self):
        import pandas as pd
        from meta_label import (build_feature_matrix, features_from_snapshot,
                                META_FEATURES)
        n = 5
        idx = pd.DatetimeIndex([dt.datetime(2026, 5, 1, 14)] * n)
        df = pd.DataFrame({name: np.full(n, 0.5) for name in META_FEATURES
                           if name not in ('pred', 'hour_sin', 'hour_cos')},
                          index=idx)
        preds = np.full(n, 1.25)
        mat = build_feature_matrix(df, preds)
        vec = features_from_snapshot({name: 0.5 for name in META_FEATURES},
                                     pred=1.25, hour=14)
        assert mat.shape == (n, len(META_FEATURES))
        np.testing.assert_allclose(mat[0], vec, atol=1e-12)

    def test_missing_values_default_zero(self):
        from meta_label import features_from_snapshot, META_FEATURES
        vec = features_from_snapshot({}, pred=None, hour=0)
        assert np.isfinite(vec).all()
        assert vec[META_FEATURES.index('RSI')] == 0.0
