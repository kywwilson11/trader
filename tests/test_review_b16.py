"""Review batch b16 — regression guards for policy_exits / panel_ranks / rank_gradient.

Covers the applied review fixes:
  policy_exits:  TB_Reason NaN'd on truncated end-of-series windows; fail-loud
                 exit_walk input validation; docstring parity-claim scoping.
  panel_ranks:   compute_live_panel_ranks honors its "{} on failure" contract
                 (and caches the {} so the full-panel refetch cannot repeat every
                 30s cycle); coverage-below-top_k warning; cs_size_tilt documented
                 semantics (centered map, fail-open None dispersion).
  rank_gradient: direction guard in the Stage-0 verdict (inverted / flat-negative
                 panels must NOT read as CONFIRMED); decision_report.json wrapper
                 accepted; misleading negative-denominator ratio suppressed;
                 missing 'signal' fails loudly; live/holdout bucket parity tripwire.
"""

import datetime as dt
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import panel_ranks
import policy_exits
import rank_gradient
from policy_exits import REASON_NAMES, compute_tb_labels, exit_walk
from rank_gradient import rank_gradient_from_panel, rank_gradient_verdict
from strategy_config import CRYPTO_POLICY

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- policy_exits

def _flat(n, px=100.0):
    c = np.full(n, px)
    return c, c + 0.05, c - 0.05, c.copy()


def _tb_df(close):
    n = len(close)
    idx = pd.DatetimeIndex([dt.datetime(2026, 5, 1) + dt.timedelta(hours=h)
                            for h in range(n)])
    return pd.DataFrame({'Close': close, 'High': close + 0.05,
                         'Low': close - 0.05, 'Open': close.copy(),
                         'ATR': np.full(n, 1.0)}, index=idx)


class TestTBReasonTruncatedWindows:
    def test_reason_nan_exactly_where_ret_is_nan(self):
        out = compute_tb_labels(_tb_df(np.full(60, 100.0)), [10], 'crypto')
        reason, ret = out['TB_Reason_10'], out['TB_Ret_10']
        # valid rows: flat prices -> vertical barrier, reason stays concrete
        assert np.isfinite(reason[:50]).all()
        assert (reason[:50] == 6).all()
        # truncated rows used to be stamped 'vertical' (6) although the
        # barrier was never reached — now NaN, aligned with TB_Ret/TB_Bars
        assert np.isnan(reason[50:]).all()
        np.testing.assert_array_equal(np.isnan(ret), np.isnan(reason))
        np.testing.assert_array_equal(np.isnan(out['TB_Bars_10']),
                                      np.isnan(reason))

    def test_untruncated_reasons_unchanged_on_a_crash(self):
        close = np.full(100, 100.0)
        close[10:] = 80.0
        out = compute_tb_labels(_tb_df(close), [24], 'crypto')
        assert out['TB_Reason_24'][9] == 1          # hard_stop, as before


class TestExitWalkValidation:
    def _args(self, n=30):
        close, high, low, open_ = _flat(n)
        return close, high, low, open_, np.full(n, 1.0), np.zeros(n, bool)

    def test_length_mismatch_raises(self):
        close, high, low, open_, atr, is_eod = self._args()
        with pytest.raises(ValueError, match='high'):
            exit_walk(close, high[:-1], low, open_, atr, is_eod, CRYPTO_POLICY)
        with pytest.raises(ValueError, match='atr'):
            exit_walk(close, high, low, open_, atr[:-2], is_eod, CRYPTO_POLICY)
        with pytest.raises(ValueError, match='preds'):
            exit_walk(close, high, low, open_, atr, is_eod, CRYPTO_POLICY,
                      preds=np.zeros(len(close) - 1))

    def test_signal_exit_without_preds_raises(self):
        close, high, low, open_, atr, is_eod = self._args()
        with pytest.raises(ValueError, match='requires preds'):
            exit_walk(close, high, low, open_, atr, is_eod, CRYPTO_POLICY,
                      use_signal_exit=True)

    def test_bad_side_raises(self):
        close, high, low, open_, atr, is_eod = self._args()
        for bad in (0, 2, -2):
            with pytest.raises(ValueError, match='side'):
                exit_walk(close, high, low, open_, atr, is_eod, CRYPTO_POLICY,
                          side=bad)

    def test_compliant_calls_still_work_both_sides(self):
        close, high, low, open_, atr, is_eod = self._args()
        n = len(close)
        for side in (1, -1):
            idx, px, reason = exit_walk(close, high, low, open_, atr, is_eod,
                                        CRYPTO_POLICY, max_hold=5, side=side)
            assert len(idx) == len(px) == len(reason) == n
            assert set(np.unique(reason)).issubset(set(REASON_NAMES))
        # signal-exit path with matching preds is untouched
        idx, _, _ = exit_walk(close, high, low, open_, atr, is_eod,
                              CRYPTO_POLICY, preds=np.zeros(n),
                              use_signal_exit=True)
        assert len(idx) == n


class TestDocstringParityClaim:
    def test_module_doc_enumerates_live_only_overlays(self):
        doc = policy_exits.__doc__
        assert 'guarantees label semantics' not in doc
        for token in ('stop_mult', 'confirmation', 'MIDPOINT',
                      'TAKE_PROFIT_CEIL_PCT', 'Bar-level approximations'):
            assert token in doc, f'missing overlay mention: {token}'

    def test_tb_doc_scopes_the_1550_claim_and_reason_nan(self):
        doc = compute_tb_labels.__doc__
        assert 'exactly the live 15:50' not in doc
        assert 'ALL THREE columns' in doc


# ----------------------------------------------------------------- panel_ranks

class _RecLogger:
    def __init__(self):
        self.msgs = []

    def _rec(self, level, msg, *args):
        self.msgs.append((level, (msg % args) if args else msg))

    def debug(self, msg, *a):
        self._rec('debug', msg, *a)

    def info(self, msg, *a):
        self._rec('info', msg, *a)

    def warning(self, msg, *a):
        self._rec('warning', msg, *a)

    def error(self, msg, *a):
        self._rec('error', msg, *a)

    def level(self, level):
        return [m for lv, m in self.msgs if lv == level]


def _bars(n=100):
    idx = pd.date_range('2026-06-01', periods=n, freq='h', tz='UTC')
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame({'Close': close, 'High': close + 0.5,
                         'Low': close - 0.5, 'Open': close,
                         'Volume': np.full(n, 1e6)}, index=idx)


def _fake_env(monkeypatch, n_syms=12, fail=(), top_k=60):
    """Fake stock_config/market_data/indicators so the live pre-pass runs
    on this Mac with no heavy deps and no network. Returns (syms, calls, rec)."""
    syms = [f'S{i:02d}' for i in range(n_syms)]
    calls = {'fetch': 0}
    bars = _bars()

    def fetch(api, sym):
        calls['fetch'] += 1
        if sym in fail:
            raise RuntimeError('fetch down')
        return bars

    def feats(bars_, spy_close=None, symbol=None):
        i = int(symbol[1:])
        return pd.DataFrame(
            {'Return_4h': np.full(len(bars_), 0.01 * (i + 1)),
             'Price_SMA20_Ratio': np.full(len(bars_), 0.9 + 0.02 * i),
             'Ret_21d': np.full(len(bars_), 0.001 * i)},
            index=bars_.index)

    md = types.ModuleType('market_data')
    md.fetch_stock_bars_alpaca = fetch
    ind = types.ModuleType('indicators')
    ind.compute_stock_features = feats
    sc = types.ModuleType('stock_config')
    sc.AS_OF_TOP_K = top_k
    monkeypatch.setitem(sys.modules, 'market_data', md)
    monkeypatch.setitem(sys.modules, 'indicators', ind)
    monkeypatch.setitem(sys.modules, 'stock_config', sc)
    monkeypatch.setattr(panel_ranks, '_panel_symbols', lambda: syms)
    monkeypatch.setattr(panel_ranks, '_live_cache', None)
    rec = _RecLogger()
    monkeypatch.setattr(panel_ranks, 'logger', rec)
    return syms, calls, rec


class TestLivePanelFailureContract:
    def test_ranking_failure_returns_empty_and_caches(self, monkeypatch):
        syms, calls, rec = _fake_env(monkeypatch)

        def boom(*a, **k):
            raise RuntimeError('rank exploded')

        monkeypatch.setattr(panel_ranks, '_signed_rank', boom)
        out = panel_ranks.compute_live_panel_ranks(api=None)
        assert out == {}                       # documented neutral fallback
        assert panel_ranks._live_cache is not None
        assert panel_ranks._live_cache[1] == {}
        assert any('ranking failed' in m for m in rec.level('warning'))
        # cached {} stops the full-panel refetch hammer within the TTL
        n_fetches = calls['fetch']
        assert n_fetches == len(syms)
        assert panel_ranks.compute_live_panel_ranks(api=None) == {}
        assert calls['fetch'] == n_fetches

    def test_happy_path_intact_after_try_wrap(self, monkeypatch):
        syms, calls, rec = _fake_env(monkeypatch)
        out = panel_ranks.compute_live_panel_ranks(api=None)
        assert set(out) == set(syms)
        top = max(syms, key=lambda s: int(s[1:]))   # highest Return_4h
        assert out[top]['CS_Rank_Return_4h'] == pytest.approx(1.0)
        for s in syms:
            assert np.isfinite(out[s]['CS_Dispersion'])
            assert np.isfinite(out[s]['CS_Breadth'])
            assert 'MS_Interact' in out[s]
        # second call inside the TTL is served from cache, no refetch
        n_fetches = calls['fetch']
        assert panel_ranks.compute_live_panel_ranks(api=None) is out
        assert calls['fetch'] == n_fetches

    def test_coverage_below_top_k_warns(self, monkeypatch):
        syms, calls, rec = _fake_env(
            monkeypatch, n_syms=30,
            fail=tuple(f'S{i:02d}' for i in range(12, 30)), top_k=60)
        out = panel_ranks.compute_live_panel_ranks(api=None)
        assert len(out) == 12                  # ranks the reduced population
        assert any('coverage 12/30' in m and 'top_k=60' in m
                   for m in rec.level('warning'))

    def test_below_ten_names_still_neutral(self, monkeypatch):
        syms, calls, rec = _fake_env(
            monkeypatch, n_syms=12,
            fail=tuple(f'S{i:02d}' for i in range(4, 12)))
        assert panel_ranks.compute_live_panel_ranks(api=None) == {}
        assert panel_ranks._live_cache[1] == {}


class TestCsSizeTiltSemantics:
    def test_symmetric_defaults_are_exact(self):
        assert panel_ranks.cs_size_tilt(1.0) == pytest.approx(1.10)
        assert panel_ranks.cs_size_tilt(-1.0) == pytest.approx(0.90)
        assert panel_ranks.cs_size_tilt(0.0) == pytest.approx(1.0)

    def test_asymmetric_bounds_clip_as_documented(self):
        # centered at 1.0: never reaches lo=0.8, clips at hi=1.1
        assert panel_ranks.cs_size_tilt(-1.0, lo=0.8, hi=1.1) == pytest.approx(0.85)
        assert panel_ranks.cs_size_tilt(1.0, lo=0.8, hi=1.1) == pytest.approx(1.10)

    def test_dispersion_gate_none_skips_finite_low_noops(self):
        # None dispersion SKIPS the gate — the tilt still applies (documented)
        assert panel_ranks.cs_size_tilt(
            0.5, dispersion=None, dispersion_floor=0.5) == pytest.approx(1.05)
        # finite dispersion below the floor forces the no-op
        assert panel_ranks.cs_size_tilt(
            0.5, dispersion=0.1, dispersion_floor=0.5) == 1.0
        assert panel_ranks.cs_size_tilt(None) == 1.0
        assert panel_ranks.cs_size_tilt(float('nan')) == 1.0


# --------------------------------------------------------------- rank_gradient

def _buckets(r13, r67):
    return {'rank_1_3': {'n': 50, 'mean_net_pct': r13, 'hit_rate': 0.5},
            'rank_6_7': {'n': 40, 'mean_net_pct': r67, 'hit_rate': 0.5}}


class TestVerdictDirectionGuard:
    def test_inverted_gradient_fails(self):
        # top bucket is the WORST bucket — the old r67<=0 branch called this
        # CONFIRMED and would have justified concentrating into the losers
        v = rank_gradient_verdict(_buckets(-0.50, -0.05))
        assert v['gradient_exists'] is False
        assert 'NEITHER' in v['verdict']
        assert v['ratio_6_7_over_1_3'] is None   # neg/neg ratio suppressed

    def test_flat_negative_fails(self):
        v = rank_gradient_verdict(_buckets(-0.20, -0.20))
        assert v['gradient_exists'] is False

    def test_positive_top_negative_bottom_still_passes(self):
        # the pre-existing pass case (mirrors test_verdict_passes_when_...)
        v = rank_gradient_verdict(_buckets(0.30, -0.05))
        assert v['gradient_exists'] is True

    def test_ratio_branch_pass_preserved(self):
        v = rank_gradient_verdict(_buckets(0.40, 0.05))
        assert v['gradient_exists'] is True
        assert v['ratio_6_7_over_1_3'] == pytest.approx(0.125)

    def test_zero_top_negative_bottom_keeps_old_semantics(self):
        # r13=0 > r67<0: still a pass under the docstring's r67<=0 rule;
        # ratio undefined (denominator not > 0)
        v = rank_gradient_verdict(_buckets(0.0, -0.05))
        assert v['gradient_exists'] is True
        assert v['ratio_6_7_over_1_3'] is None


class TestDecisionReportWrapper:
    def test_full_report_shape_is_unwrapped(self):
        report = {'generated': '2026-07-02T00:00:00', 'days': 30, 'gates': {},
                  'conviction': dict(_buckets(0.40, 0.05), n=90),
                  'admitted_k': {}, 'signal_exit': {}}
        v = rank_gradient_verdict(report)
        assert v['gradient_exists'] is True
        assert v['rank_1_3'] == 0.40

    def test_wrapper_without_rank_buckets_is_insufficient_not_a_crash(self):
        v = rank_gradient_verdict({'conviction': {'n': 3}, 'gates': {}})
        assert v['gradient_exists'] is None
        assert 'insufficient' in v['verdict']

    def test_bare_bucket_dict_unchanged(self):
        assert rank_gradient_verdict(_buckets(0.30, -0.05))['gradient_exists'] is True


class TestPanelStrictness:
    def test_missing_signal_key_raises_like_fwd_return(self):
        period = [{'symbol': 'A', 'fwd_return': 1.0},
                  {'symbol': 'B', 'signal': 0.5, 'fwd_return': 0.2}]
        with pytest.raises(KeyError):
            rank_gradient_from_panel([period])

    def test_well_formed_panel_unchanged(self):
        period = [{'symbol': f'S{i}', 'signal': float(-i),
                   'fwd_return': float(1.0 - 0.2 * i)} for i in range(7)]
        b = rank_gradient_from_panel([period] * 3)
        assert b['rank_1_3']['mean_net_pct'] > b['rank_6_7']['mean_net_pct']


class TestBucketParityTripwire:
    def test_default_buckets_match_decision_report_literals(self):
        # decision_report.py re-declares the bucket boundaries inline; the
        # Stage-0 gate compares live vs holdout, so drift = silent apples/oranges
        src = (REPO / 'decision_report.py').read_text()
        for label, lo, hi in rank_gradient.DEFAULT_BUCKETS:
            assert f"('{label}', {lo}, {hi})" in src, \
                f'decision_report.py no longer declares ({label}, {lo}, {hi})'
        assert "('rank_8_plus', 8, 9999)" in src
