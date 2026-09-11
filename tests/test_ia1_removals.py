"""IA-1 influence-implementation removals (2026-08-22, owner-ruled wave).

Binding spec: research/campaign_2026-08/07_decision_influences.md (the
decision-influence ledger); archive of every removed block:
research/campaign_2026-08/08_removed_code.md (sections IA-1.1..IA-1.3).

Three removals, each pinned here:
  1. Pseudo-CAPE (ledger §3.5, unanimous NO; KILL_LIST pending ask #3 RULED
     2026-08-22): fetch_cape + the cape_z>1.5 -> 0.7x stock haircut + the v2
     exclusion-announce machinery deleted from macro_indicators.py. The one
     direct-ship legacy-behavior change (strictly size-increasing: removes an
     un-founded haircut driven by fabricated data).
  2. Hurst<0.45 threshold shift (ledger §3.3, unanimous NO): PROVABLY-NO-OP —
     the proof tests below establish that live snapshot Hurst (computed on
     price LEVELS, indicator_config.HURST_ON_RETURNS=False default) cannot
     read below 0.45 on any realistic price construction, so the removed
     branch could never fire. HURST_ON_RETURNS machinery untouched.
  3. Sentiment gate<=0 veto branch (ledger §3.3: "delete the dead limb"):
     PROVABLY-NO-OP — sentiment_gate clamps to [0.15, 1.5], so the return is
     strictly positive for every input; the [0.15,1.5] multiplier path is
     byte-unchanged.

Mac-runnable: macro_indicators/sentiment/indicators import clean (lazy heavy
imports); base_loop is covered by source-text pinning (test_base_loop_v3
style) because it cannot be imported without torch/joblib.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import indicator_config
import indicators
import macro_indicators as mi
import sentiment
from types_mod import MacroRegime

BASE_LOOP_SRC = (REPO / "base_loop.py").read_text()


def _method(name: str, src: str = BASE_LOOP_SRC) -> str:
    start = src.index(f"def {name}")
    return src[start:src.index("\n    def ", start + 10)]


@pytest.fixture(autouse=True)
def _clean_macro_cache():
    mi._cache.clear()
    yield
    mi._cache.clear()


# ---------------------------------------------------------------------------
# 1. Pseudo-CAPE deletion (08_removed_code.md IA-1.1)
# ---------------------------------------------------------------------------

class TestPseudoCapeDeleted:
    def test_module_carries_no_cape_machinery(self):
        for name in ('fetch_cape', '_CAPE_MEAN', '_CAPE_STD',
                     '_CAPE_CACHE_TTL', '_cape_exclusion_logged'):
            assert not hasattr(mi, name), name
        src = (REPO / 'macro_indicators.py').read_text()
        assert 'def fetch_cape' not in src
        assert 'cape_z' not in src
        assert "labels.append('overvalued')" not in src

    def test_stock_regime_no_longer_haircut_by_fake_valuation(self, monkeypatch):
        # Pre-ruling behavior: SPY trailingPE >= ~23.5 -> z > 1.5 -> 0.7x on
        # ~every stock entry. New truth: calm inputs => 1.0x for BOTH books.
        monkeypatch.setattr(mi, 'fetch_vix', lambda: 10.0)
        monkeypatch.setattr(mi, 'fetch_financial_stress', lambda: 0.0)
        for asset_type in ('stock', 'crypto'):
            r = mi.get_macro_regime(api=None, asset_type=asset_type)
            assert r.sizing_mult == pytest.approx(1.0)
            assert 'overvalued' not in r.regime_label
            assert r.cape is None       # field retained, permanently None

    def test_surviving_macro_rules_untouched(self, monkeypatch):
        # Guard against over-deletion: VIX tiers and the STLFSI2 stress rule
        # (both ledger survivors) still compose exactly as before.
        monkeypatch.setattr(mi, 'fetch_vix', lambda: 26.0)
        monkeypatch.setattr(mi, 'fetch_financial_stress', lambda: 1.5)
        r = mi.get_macro_regime(api=None, asset_type='stock')
        assert r.sizing_mult == pytest.approx(0.5 * 0.5)  # defensive x stress
        assert r.stop_mult == pytest.approx(0.8)
        assert 'defensive' in r.regime_label
        assert 'high_stress' in r.regime_label

    def test_v2_family_has_no_cape_and_no_announce_param(self):
        sig = inspect.signature(mi.regime_family_mults_v2)
        assert list(sig.parameters) == ['regime', 'asset_type']
        regime = MacroRegime(stress_level=0.0, vix=18.0, cape=None,
                             regime_label='x')
        mi._reset_vix_tier_state()
        fam = mi.regime_family_mults_v2(regime, 'stock')
        assert fam == {'vix': 1.0, 'stress': 1.0}
        assert 'cape' not in fam

    def test_base_loop_call_site_matches_new_signature(self):
        body = _method('_compute_position_size')
        assert 'regime_family_mults_v2(self.macro_regime' in body
        assert 'announce=' not in body


# ---------------------------------------------------------------------------
# 2. Hurst<0.45 threshold shift — proof the branch could never fire
#    (08_removed_code.md IA-1.2)
# ---------------------------------------------------------------------------

def _levels_hurst(series):
    return indicators.compute_hurst(pd.Series(series, dtype=float),
                                    window=100).dropna()


class TestHurstBranchProvablyDead:
    def test_live_input_mode_is_levels(self):
        # The live snapshot's Hurst comes from compute_features with the
        # default flag OFF => computed on price LEVELS (the documented bug
        # that made the gate structurally unable to fire).
        assert indicator_config.HURST_ON_RETURNS is False
        assert indicators.HURST_ON_RETURNS is False

    @pytest.mark.parametrize('name,prices', [
        ('random_walk', np.cumsum(np.random.default_rng(1).normal(0, 1, 800)) + 500.0),
        ('uptrend', 100.0 + 0.3 * np.arange(800)
         + np.random.default_rng(2).normal(0, 2, 800)),
        ('downtrend', 800.0 - 0.3 * np.arange(800)
         + np.random.default_rng(3).normal(0, 2, 800)),
        ('gbm_high_vol', 100.0 * np.exp(np.cumsum(
            np.random.default_rng(4).normal(0, 0.03, 800)))),
        ('strong_ou_mean_reverting', 0.2),    # OU theta (level phi = 0.8)
        ('extreme_ou_mean_reverting', 0.35),  # phi = 0.65 — far past any market
    ])
    def test_levels_hurst_never_below_045(self, name, prices):
        # The removed branch required snapshot['Hurst'] < 0.45. On price
        # LEVELS, R/S partial sums integrate the series, so persistent /
        # near-integrated inputs read ~0.8 regardless of regime. Even an OU
        # price losing 35% of its deviation EVERY HOUR (level lag-1
        # autocorr 0.65 — no traded asset is remotely close; real hourly
        # price levels sit at ~0.99+) never reaches 0.45.
        if isinstance(prices, float):
            theta = prices
            rng = np.random.default_rng(5)
            x = np.empty(2000)
            x[0] = 100.0
            for i in range(1, 2000):   # OU: mean-reverting PRICES
                x[i] = x[i - 1] + theta * (100.0 - x[i - 1]) + rng.normal(0, 1.5)
            prices = x
        h = _levels_hurst(prices)
        assert len(h) > 500
        assert h.min() >= 0.45, (
            f"{name}: levels-mode Hurst reached {h.min():.3f} < 0.45 — "
            f"the removed branch would have been reachable")

    def test_unreachability_boundary_documented(self):
        # Honest scope of the proof: only when price-LEVEL lag-1
        # autocorrelation collapses below ~0.5 (a price series that loses
        # half its deviation from the mean every hour — white-noise-like
        # LEVELS, impossible for a traded asset whose levels are
        # near-integrated) can levels-mode Hurst dip under 0.45. Pinning
        # the boundary keeps this proof from overclaiming.
        rng = np.random.default_rng(11)
        x = np.empty(2000)
        x[0] = 100.0
        for i in range(1, 2000):       # theta=0.9 -> level phi ~= 0.1
            x[i] = x[i - 1] + 0.9 * (100.0 - x[i - 1]) + rng.normal(0, 1.5)
        assert np.corrcoef(x[:-1], x[1:])[0, 1] < 0.2
        assert _levels_hurst(x).min() < 0.45   # branch reachable ONLY here

    def test_end_to_end_snapshot_construction_reads_high(self):
        # Full live construction: compute_features on a synthetic OHLCV
        # frame (flag OFF) — the exact producer of snapshot['Hurst'].
        rng = np.random.default_rng(7)
        n = 800
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        idx = pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC')
        df = pd.DataFrame({
            'Open': close * (1 + rng.normal(0, 0.001, n)),
            'High': close * (1 + np.abs(rng.normal(0, 0.004, n))),
            'Low': close * (1 - np.abs(rng.normal(0, 0.004, n))),
            'Close': close,
            'Volume': rng.uniform(1e5, 1e6, n),
        }, index=idx)
        out = indicators.compute_features(df)
        h = out['Hurst'].dropna()
        assert len(h) > 500
        assert h.min() >= 0.45

    def test_branch_removed_and_threshold_direct(self):
        body = _method('_execute_buys')
        assert 'effective_threshold' not in BASE_LOOP_SRC
        assert 'hurst < 0.45' not in BASE_LOOP_SRC
        assert "snapshot.get('Hurst')" not in body
        assert 'if pred_return < self.trade_threshold:' in body
        # The below_threshold veto itself survives (only the shift died)
        assert "vc['below_threshold']" in body

    def test_hurst_on_returns_machinery_untouched(self):
        # The legitimate future form stays: flag present (default OFF) and
        # the returns-mode computation still wired in compute_features.
        assert hasattr(indicator_config, 'HURST_ON_RETURNS')
        src = (REPO / 'indicators.py').read_text()
        assert 'if HURST_ON_RETURNS:' in src


# ---------------------------------------------------------------------------
# 3. Sentiment gate<=0 veto branch — proof of unreachability
#    (08_removed_code.md IA-1.3)
# ---------------------------------------------------------------------------

class TestSentimentVetoUnreachable:
    def test_worst_case_inputs_still_clamped_strictly_positive(self, monkeypatch):
        # Most negative composition possible: extreme greed (0.7) x
        # catastrophic symbol news (0.15) x bearish market (0.85) =
        # 0.089 pre-clamp -> floor 0.15. gate <= 0 is unsatisfiable.
        monkeypatch.setattr(sentiment, 'get_fear_greed', lambda: {'value': 95})
        monkeypatch.setattr(sentiment, 'get_news_sentiment',
                            lambda s, a: {'sentiment_score': -0.9,
                                          'article_count': 10})
        monkeypatch.setattr(sentiment, 'get_market_sentiment',
                            lambda: {'sentiment_score': -0.9,
                                     'article_count': 30})
        gate, reasons = sentiment.sentiment_gate('BTC/USD', 'crypto')
        assert gate == pytest.approx(0.15)
        assert gate > 0
        assert any('catastrophic' in r for r in reasons)

    def test_gate_floor_holds_across_score_sweep(self, monkeypatch):
        monkeypatch.setattr(sentiment, 'get_fear_greed', lambda: {'value': 92})
        monkeypatch.setattr(sentiment, 'get_market_sentiment',
                            lambda: {'sentiment_score': -1.0,
                                     'article_count': 30})
        for score in np.linspace(-1.0, 1.0, 41):
            monkeypatch.setattr(sentiment, 'get_news_sentiment',
                                lambda s, a, _sc=score: {
                                    'sentiment_score': float(_sc),
                                    'article_count': 5})
            for asset_type in ('crypto', 'stock'):
                gate, _ = sentiment.sentiment_gate('X', asset_type)
                assert 0.15 <= gate <= 1.5

    def test_no_data_path_neutral(self, monkeypatch):
        monkeypatch.setattr(sentiment, 'get_fear_greed', lambda: None)
        monkeypatch.setattr(sentiment, 'get_news_sentiment', lambda s, a: None)
        monkeypatch.setattr(sentiment, 'get_market_sentiment', lambda: None)
        gate, reasons = sentiment.sentiment_gate('ETH/USD', 'crypto')
        assert gate == pytest.approx(1.0)
        assert reasons == []

    def test_dead_branch_removed_multiplier_path_survives(self):
        body = _method('_execute_buys')
        assert 'if gate <= 0:' not in body
        assert "'sentiment_block'" not in body
        # The multiplier path is intact: the gate is still fetched and fed
        # into sizing exactly as before.
        assert 'gate, gate_reasons = sentiment_gate(' in body
        assert 'sentiment_mult=gate' in body

    def test_stock_loop_duplicate_reported_or_archived(self):
        # Ownership discipline: stock_loop.py is outside IA-1's file set, so
        # its identical unreachable branch was REPORTED (08_removed_code.md
        # IA-1.3), not edited by IA-1. Self-resolving pin (hardened): while
        # the branch is still present this passes; once the owning packet
        # removes it, this ENFORCES the wave's archival discipline — the
        # removed stock_loop block must appear VERBATIM in 08_removed_code.md
        # (its journal line is unique to stock_loop, so the IA-1.3 prose
        # cannot satisfy it by accident).
        stock_src = (REPO / 'stock_loop.py').read_text()
        if 'if gate <= 0:' in stock_src:
            return  # still present, awaiting its owning packet
        archive = (REPO / 'research' / 'campaign_2026-08' /
                   '08_removed_code.md').read_text()
        assert '"skip_reason": "sentiment_block"' in archive, (
            "stock_loop's sentiment veto branch was removed without its "
            "verbatim block being archived in 08_removed_code.md")
