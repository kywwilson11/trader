"""Review batch b11 — regression pins for short_cost / borrow_proxy /
options_overlay fixes.

Pins: pit_publication_map tz-aware handling (P2), htb_score clamp, the
si_pct_float FRACTION contract, NaN-cap routing in htb_risk_score, the
class-lookup fail-open warning, the put-branch friction strike (P2), the
non-finite row mask in overlay_decision, the overnight_frac convention, the
MIN_EDGE_MULTIPLE decoupling note, and the vertical_debit/value dedupe.
"""

import logging
import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import borrow_proxy as bp
import options_overlay as ov
import short_cost as sc


# ---------------------------------------------------------------------------
# short_cost.pit_publication_map — tz-aware index support (P2)
# ---------------------------------------------------------------------------

class TestPitPublicationMapTz:
    PUB = {pd.Timestamp('2025-01-15'): 10.0, pd.Timestamp('2025-02-01'): 20.0}
    BAR_TIMES = ['2025-01-10 14:00', '2025-01-15 14:00', '2025-01-20 14:00',
                 '2025-02-01 14:00', '2025-02-05 14:00']

    def _check(self, out):
        assert np.isnan(out[0])   # before any publication
        assert np.isnan(out[1])   # ON 1st pub date -> not yet usable (shift-1)
        assert out[2] == 10.0     # after 1st pub
        assert out[3] == 10.0     # ON 2nd pub date -> still the 1st print
        assert out[4] == 20.0     # after 2nd pub

    def test_tz_aware_bar_index(self):
        # the harvest's real intraday indexes are tz-aware UTC; naive FINRA
        # pub dates vs aware bars used to raise TypeError in searchsorted.
        idx = pd.to_datetime(self.BAR_TIMES).tz_localize('UTC')
        self._check(sc.pit_publication_map(pd.Series(self.PUB), idx))

    def test_tz_aware_pub_index(self):
        pub = pd.Series({k.tz_localize('UTC'): v for k, v in self.PUB.items()})
        idx = pd.to_datetime(self.BAR_TIMES)
        self._check(sc.pit_publication_map(pub, idx))

    def test_both_tz_aware(self):
        pub = pd.Series({k.tz_localize('UTC'): v for k, v in self.PUB.items()})
        idx = pd.to_datetime(self.BAR_TIMES).tz_localize('UTC')
        self._check(sc.pit_publication_map(pub, idx))

    def test_naive_path_unchanged(self):
        idx = pd.to_datetime(self.BAR_TIMES)
        self._check(sc.pit_publication_map(pd.Series(self.PUB), idx))


# ---------------------------------------------------------------------------
# short_cost.borrow_cost_bps_annual — htb_score clamp
# ---------------------------------------------------------------------------

class TestHtbScoreClamp:
    D = date(2026, 1, 1)

    def test_negative_score_never_rebates(self):
        # unclamped, -0.5 produced -150 bps: a borrow REBATE flattering shorts
        out = sc.borrow_cost_bps_annual(self.D, likely_etb=False, htb_score=-0.5)
        assert out == 0.0

    def test_nan_score_full_backstop(self):
        out = sc.borrow_cost_bps_annual(self.D, likely_etb=False,
                                        htb_score=float('nan'))
        assert out == sc.HTB_BPS
        # ...and no NaN propagates into the all-in short cost
        rt = sc.short_round_trip_cost_pct(self.D, 0.10, 5, likely_etb=False,
                                          htb_score=float('nan'))
        assert np.isfinite(rt)

    def test_above_one_clamped(self):
        out = sc.borrow_cost_bps_annual(self.D, likely_etb=False, htb_score=2.0)
        assert out == sc.HTB_BPS

    def test_in_domain_unchanged(self):
        assert sc.borrow_cost_bps_annual(self.D, likely_etb=False,
                                         htb_score=None) == sc.HTB_BPS
        assert sc.borrow_cost_bps_annual(self.D, likely_etb=False,
                                         htb_score=0.0) == 0.0
        assert sc.borrow_cost_bps_annual(self.D, likely_etb=False,
                                         htb_score=0.5) == pytest.approx(sc.HTB_BPS * 0.5)
        assert sc.borrow_cost_bps_annual(self.D, likely_etb=False,
                                         htb_score=1.0) == sc.HTB_BPS


# ---------------------------------------------------------------------------
# short_cost.short_interest_metrics — si_pct_float FRACTION contract
# ---------------------------------------------------------------------------

class TestSiPctFloatContract:
    def test_value_is_fraction_not_percent(self):
        # 2M short / 50M float = 4% -> 0.04; a wave-7 '>10%' veto must use 0.10
        m = sc.short_interest_metrics(2_000_000, float_shares=50_000_000)
        assert m['si_pct_float'] == pytest.approx(0.04)
        assert m['si_pct_float'] < 1.0

    def test_docstring_states_fraction_and_threshold(self):
        doc = sc.short_interest_metrics.__doc__
        assert 'FRACTION' in doc and '0.10' in doc
        assert 'NaN' in doc  # NaN-propagation caveat documented

    def test_nan_inputs_propagate_nan_not_none(self):
        m = sc.short_interest_metrics(float('nan'), float_shares=50_000_000,
                                      adv_20d=500_000)
        assert m['days_to_cover'] is not None and np.isnan(m['days_to_cover'])
        assert m['si_pct_float'] is not None and np.isnan(m['si_pct_float'])


# ---------------------------------------------------------------------------
# borrow_proxy.htb_risk_score — NaN cap routes to the missing-data default
# ---------------------------------------------------------------------------

class TestNanCapRouting:
    def test_nan_cap_scores_missing_default(self):
        # pandas-sourced lookups yield NaN (not None) for missing caps;
        # NaN used to fall through every bucket to the micro-cap 0.9
        assert bp.htb_risk_score(float('nan')) == pytest.approx(0.7)

    def test_finite_buckets_unchanged(self):
        assert bp.htb_risk_score(2e12) == pytest.approx(0.05)
        assert bp.htb_risk_score(3e9) == pytest.approx(0.25)
        assert bp.htb_risk_score(1e9) == pytest.approx(0.6)
        assert bp.htb_risk_score(1e8) == pytest.approx(0.9)
        assert bp.htb_risk_score(None) == pytest.approx(0.7)
        assert bp.htb_risk_score(0) == pytest.approx(0.7)
        assert bp.htb_risk_score(-5e9) == pytest.approx(0.7)

    def test_shortable_decision_still_excludes(self):
        assert bp.likely_shortable('X', market_cap=float('nan')) is False


# ---------------------------------------------------------------------------
# borrow_proxy.class_lookup_from_config — fail-open degradation must warn
# ---------------------------------------------------------------------------

class TestClassLookupFallbackWarns:
    def test_broken_config_warns_and_degrades(self, monkeypatch, caplog):
        fake = types.ModuleType('stock_config')  # no SECTOR_BUCKETS attr
        monkeypatch.setitem(sys.modules, 'stock_config', fake)
        with caplog.at_level(logging.WARNING, logger='borrow_proxy'):
            lk = bp.class_lookup_from_config()
        assert lk('IONQ') is None
        assert any('SECTOR_BUCKETS unavailable' in r.getMessage()
                   for r in caplog.records)

    def test_intact_config_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger='borrow_proxy'):
            lk = bp.class_lookup_from_config()
        assert lk('IONQ') == 'spec_growth'
        assert not caplog.records


# ---------------------------------------------------------------------------
# borrow_proxy — documented API contracts (doc-only fixes)
# ---------------------------------------------------------------------------

class TestBorrowProxyDocs:
    def test_symbol_documented_unused_and_time_invariant(self):
        doc = bp.likely_shortable.__doc__
        assert 'unused' in doc and 'time-invariant' in doc

    def test_cap_pit_responsibility_documented(self):
        assert "CALLER's responsibility" in bp.__doc__
        assert 'as-of caps' in bp.__doc__
        assert "caller's responsibility" in bp.restrict_short_universe.__doc__


# ---------------------------------------------------------------------------
# options_overlay.overlay_decision — put-branch friction strike (P2)
# ---------------------------------------------------------------------------

class TestPutBranchFriction:
    def test_put_friction_priced_at_traded_strike(self):
        # constant prices -> friction is exactly computable at the strikes the
        # simulation trades: ATM put + 5% OTM put BELOW spot (not ITM above).
        S, r, w = 100.0, 0.04, 0.05
        close = np.full(20, S)
        opens = np.full(20, S)
        out = ov.overlay_decision(close, opens, 'C', rv_sigma_annual=0.6,
                                  call=False, width_frac=w, r=r)
        T = 1 / ov.TRADING_DAYS
        _, iv, _ = ov.iv_from_har(0.6)
        legs = [ov.bs_price(S, S, T, r, iv, False),
                ov.bs_price(S, S * (1 - w), T, r, iv, False)]
        debit = ov.vertical_debit(S, S * (1 - w), S, T, r, iv, False)
        expected = ov.option_round_trip_cost(legs, ov.SPREAD_TIERS['C']) / debit
        assert out['friction_frac_of_debit'] == pytest.approx(expected, abs=2e-4)

    def test_put_friction_same_ballpark_as_call(self):
        # symmetric data: before the fix the put path priced its short leg ITM
        # (~3x the call friction); now both paths sit in the same ballpark.
        rng = np.random.RandomState(0)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 60))
        opens = close * (1 + rng.normal(0, 0.005, 60))
        fc = ov.overlay_decision(close, opens, 'C', 0.6, call=True)
        fp = ov.overlay_decision(close, opens, 'C', 0.6, call=False)
        ratio = fp['friction_frac_of_debit'] / fc['friction_frac_of_debit']
        assert 0.5 < ratio < 2.0

    def test_call_path_unchanged(self):
        # pins that the strike fix did not perturb the call-side friction
        S, r, w = 100.0, 0.04, 0.05
        close = np.full(20, S)
        opens = np.full(20, S)
        out = ov.overlay_decision(close, opens, 'C', rv_sigma_annual=0.6,
                                  call=True, width_frac=w, r=r)
        T = 1 / ov.TRADING_DAYS
        _, iv, _ = ov.iv_from_har(0.6)
        legs = [ov.bs_price(S, S, T, r, iv, True),
                ov.bs_price(S, S * (1 + w), T, r, iv, True)]
        debit = ov.vertical_debit(S, S, S * (1 + w), T, r, iv, True)
        expected = ov.option_round_trip_cost(legs, ov.SPREAD_TIERS['C']) / debit
        assert out['friction_frac_of_debit'] == pytest.approx(expected, abs=2e-4)


# ---------------------------------------------------------------------------
# options_overlay.overlay_decision — non-finite rows masked up front
# ---------------------------------------------------------------------------

class TestNonFiniteMask:
    def test_nan_rows_dropped_result_stays_finite(self):
        rng = np.random.RandomState(3)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 60))
        opens = close * (1 + rng.normal(0, 0.005, 60))
        close[5] = np.nan
        opens[10] = np.nan
        close[20] = np.inf
        out = ov.overlay_decision(close, opens, 'C', 0.6)
        assert out['verdict'] in ('GO', 'NO_GO')
        assert out['n'] == 57
        for k in ('mean_edge_frac_of_debit', 'friction_frac_of_debit',
                  'required_edge_frac', 'mean_debit'):
            assert np.isfinite(out[k]), k

    def test_mostly_nan_is_a_data_verdict_not_economics(self):
        # 7 of 10 rows bad -> the verdict must say DATA, not a nan-fielded NO_GO
        close = np.full(10, 100.0)
        opens = np.full(10, 100.0)
        close[:7] = np.nan
        out = ov.overlay_decision(close, opens, 'A', 0.4)
        assert out['verdict'] == 'INSUFFICIENT_DATA'
        assert out['n'] == 3

    def test_all_finite_behavior_unchanged(self):
        rng = np.random.RandomState(0)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 60))
        opens = close * (1 + rng.normal(0, 0.005, 60))
        out = ov.overlay_decision(close, opens, 'C', rv_sigma_annual=0.6)
        assert out['verdict'] == 'NO_GO'
        assert out['n'] == 60


# ---------------------------------------------------------------------------
# options_overlay — vertical_debit delegates to vertical_value (dedupe)
# ---------------------------------------------------------------------------

class TestVerticalDedupe:
    def test_debit_equals_value_at_entry(self):
        args = (100.0, 95.0, 105.0, 0.5, 0.03, 0.25)
        for call in (True, False):
            assert ov.vertical_debit(*args, call) == pytest.approx(
                ov.vertical_value(*args, call))

    def test_debit_still_validates_strike_order(self):
        with pytest.raises(ValueError):
            ov.vertical_debit(100, 105, 100, 0.5, 0.03, 0.25, True)

    def test_hull_numbers_still_hold(self):
        # entry debit of the bull call spread must remain a plain BS difference
        d = ov.vertical_debit(100, 100, 105, 0.5, 0.03, 0.25, True)
        expect = (ov.bs_price(100, 100, 0.5, 0.03, 0.25, True)
                  - ov.bs_price(100, 105, 0.5, 0.03, 0.25, True))
        assert d == pytest.approx(expect, abs=1e-12)


# ---------------------------------------------------------------------------
# options_overlay — documented conventions (source-inspection pins)
# ---------------------------------------------------------------------------

class TestOverlayConventions:
    SRC = (ROOT / 'options_overlay.py').read_text()

    def test_min_edge_multiple_decoupling_documented(self):
        head = self.SRC[:self.SRC.index('MIN_EDGE_MULTIPLE = 2.0')]
        assert 'DECOUPLED from fees.MIN_EDGE_MULTIPLE' in head

    def test_overnight_frac_corrected_and_documented(self):
        assert '17.67 / 24.0' in self.SRC        # 15:50->09:30 is 17h40m
        assert '(17.0 / 24.0)' not in self.SRC
        assert 'decay-conservative' in self.SRC  # the convention is stated
