"""Review batch b14 — meta_label.py fixes.

P0  train_meta -> _predict_ticker call-site arity: fedf570 (q10 tail veto)
    made backtest._predict_ticker ALWAYS return a (preds, q10) tuple;
    train_meta kept assigning it straight to `preds`, so the tuple flowed
    into exit_walk's np.ascontiguousarray and every meta training run since
    2026-06-10 crashed (weekly pipeline phase aborts; shadow promotion left
    the meta veto/sizing layer silently absent).
P2  _calibrated exception fallback: warn once per process and clip raw
    booster scores to [0, 1] instead of silently serving them unclipped.
P3  _gen_meta_rows docstring arity, dead `nets` accumulator in train_meta,
    per-bar-cost fallback observability, single hour pass in
    build_feature_matrix.

meta_label's module top level only needs numpy, and policy_exits has a
pure-python numba fallback, so the replay glue runs on synthetic data here.
train_meta itself needs torch/lightgbm, so the call-site arity is pinned by
source inspection (pattern from test_prediction_cache.py).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import liquidity
import meta_label
from meta_label import (META_FEATURES, _calibrated, _gen_meta_rows,
                        _hour_encode, build_feature_matrix)
from strategy_config import policy_for

REPO = Path(__file__).resolve().parent.parent
SRC = (REPO / 'meta_label.py').read_text()

THRESHOLD = 0.15  # config default in train_meta


def _synthetic_tdf(n=400, with_spread=False, seed=7):
    """Harvest-shaped hourly frame + primary preds with entries to replay."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC')
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    df = pd.DataFrame({
        'Close': close, 'High': high, 'Low': low, 'Open': open_,
        'ATR': close * 0.01,
        'RSI': rng.uniform(30, 70, n),
        'ATR_Pct': np.full(n, 1.0),
    }, index=idx)
    if with_spread:
        df['Eff_Spread_Pct'] = rng.uniform(0.02, 0.2, n)
    preds = rng.uniform(-0.1, 0.3, n)  # plenty above 0.5x threshold
    return df, preds


# ---------------------------------------------------------------------------
# P0: train_meta must unpack _predict_ticker's (preds, q10) tuple
# ---------------------------------------------------------------------------

class TestP0PredictTickerArity:
    def test_train_meta_call_site_unpacks_the_tuple(self):
        # mirrors backtest.py:371 — the only correct call shape since fedf570
        assert re.search(r'preds,\s*_\s*=\s*_predict_ticker\(', SRC)
        # the pre-fix single-assignment form must never come back
        assert not re.search(r'(?m)^\s*preds\s*=\s*_predict_ticker\(', SRC)

    def test_predict_ticker_source_returns_a_pair(self):
        # pin the producer side of the contract without importing torch
        bsrc = (REPO / 'backtest.py').read_text()
        seg = bsrc.split('def _predict_ticker')[1].split('\ndef ')[0]
        returns = re.findall(r'(?m)^\s*return\s+(.+?)\s*$', seg)
        assert returns, 'no return statements found in _predict_ticker'
        assert all(r == 'preds, q10' for r in returns), (
            f'_predict_ticker return arity changed ({returns}) — update '
            f'meta_label.train_meta and this pin together')


class TestP0ReplayGlue:
    def test_gen_meta_rows_replays_synthetic_trades(self):
        tdf, preds = _synthetic_tdf()
        rows, labels, nets, times, exit_times = _gen_meta_rows(
            tdf, preds, 'crypto', THRESHOLD, policy_for('crypto'))
        assert len(rows) >= 5
        assert (len(rows) == len(labels) == len(nets)
                == len(times) == len(exit_times))
        assert set(labels) <= {0, 1}
        # the meta contract: label IS the sign of the net-of-cost return
        assert all(bool(lb) == (nt > 0) for lb, nt in zip(labels, nets))
        assert all(ex >= en for en, ex in zip(times, exit_times))
        assert all(r.shape == (len(META_FEATURES),) for r in rows)

    def test_tuple_preds_crash_is_the_p0_bug(self):
        # Pre-fix, train_meta handed _predict_ticker's (preds, q10) tuple
        # straight to this glue; the kernel rejects it loudly, so the ONLY
        # safe call site is the unpacking one pinned above.
        tdf, preds = _synthetic_tdf()
        with pytest.raises((ValueError, TypeError)):
            _gen_meta_rows(tdf, (preds, None), 'crypto', THRESHOLD,
                           policy_for('crypto'))


# ---------------------------------------------------------------------------
# P2: _calibrated fallback must clip and warn once, never silently serve raw
# ---------------------------------------------------------------------------

class _BoomCalib:
    def predict(self, raw):
        raise RuntimeError('unpickled calibrator ABI mismatch')


class _DoubleCalib:
    def predict(self, raw):
        return np.asarray(raw) * 2.0


class TestP2CalibratedFallback:
    def test_fallback_clips_and_warns_once_per_process(self, capsys,
                                                       monkeypatch):
        monkeypatch.setattr(meta_label, '_calib_fallback_warned', False)
        raw = np.array([-0.25, 0.4, 1.7])
        out = _calibrated(_BoomCalib(), raw)
        np.testing.assert_allclose(out, [0.0, 0.4, 1.0])  # clipped, not raw
        first = capsys.readouterr().out
        assert 'calibrator predict failed' in first
        assert 'ABI mismatch' in first  # the exception reaches the operator
        # second failure in the same process is silent (module-level flag)
        out2 = _calibrated(_BoomCalib(), raw)
        np.testing.assert_allclose(out2, [0.0, 0.4, 1.0])
        assert 'calibrator predict failed' not in capsys.readouterr().out

    def test_success_path_unchanged_and_silent(self, capsys, monkeypatch):
        monkeypatch.setattr(meta_label, '_calib_fallback_warned', False)
        out = _calibrated(_DoubleCalib(), np.array([0.1, 0.6]))
        np.testing.assert_allclose(out, [0.2, 1.0])  # calibrated then clipped
        assert capsys.readouterr().out == ''


# ---------------------------------------------------------------------------
# P3: docstring arity, dead nets, cost-fallback observability, hour pass
# ---------------------------------------------------------------------------

class TestP3Docstring:
    def test_gen_meta_rows_documents_five_returns(self):
        assert ('(features, labels, net_returns, entry_times, exit_times)'
                in _gen_meta_rows.__doc__)


class TestP3DeadNets:
    def test_train_meta_has_no_nets_accumulator(self):
        seg = SRC.split('def train_meta')[1]
        assert 'nets' not in seg


class TestP3PerBarCostFallback:
    def test_liquidity_regression_prints_and_falls_back_flat(self,
                                                             monkeypatch,
                                                             capsys):
        tdf, preds = _synthetic_tdf(with_spread=True)

        def _boom(*a, **k):
            raise RuntimeError('per-bar cost regressed')

        monkeypatch.setattr(liquidity, 'per_bar_round_trip_cost', _boom)
        rows, labels, nets, times, exit_times = _gen_meta_rows(
            tdf, preds, 'crypto', THRESHOLD, policy_for('crypto'))
        out = capsys.readouterr().out
        assert '[META] per-bar spread cost failed' in out
        assert 'per-bar cost regressed' in out  # the cause is visible
        assert len(rows) >= 5  # flat-cost fallback still produces labels

    def test_healthy_per_bar_path_is_silent(self, capsys):
        tdf, preds = _synthetic_tdf(with_spread=True)
        _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, policy_for('crypto'))
        assert 'per-bar spread cost failed' not in capsys.readouterr().out


class TestP3HourEncoding:
    def test_matrix_hours_match_scalar_hour_encode(self):
        # training matrix and live scalar path must encode hours identically
        tdf, preds = _synthetic_tdf(n=48)
        mat = build_feature_matrix(tdf, preds)
        i_sin = META_FEATURES.index('hour_sin')
        i_cos = META_FEATURES.index('hour_cos')
        for row, t in zip(mat, tdf.index):
            hs, hc = _hour_encode(t.hour)
            assert row[i_sin] == pytest.approx(hs, abs=1e-12)
            assert row[i_cos] == pytest.approx(hc, abs=1e-12)

    def test_single_hour_pass_in_source(self):
        # one hours array feeds both trig columns (no duplicated encoding)
        assert SRC.count('t.hour for t in tdf.index') == 1
