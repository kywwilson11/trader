"""Wave-7 Finding 5: regime-dated short cost + likely-shortable proxy.

Pins the integrity-critical behaviors: borrow is $0 only on/after the ETB
regime date (no retroactive economics), HTB/uncertain names are excluded
conservatively, the SI/DTC math, and the publication-date PIT shift-1."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import short_cost as sc
import borrow_proxy as bp


class TestRegimeDatedBorrow:
    def test_zero_only_after_regime_start(self):
        assert sc.borrow_cost_bps_annual(date(2025, 10, 1), likely_etb=True) == 0.0
        assert sc.borrow_cost_bps_annual(date(2026, 6, 1), likely_etb=True) == 0.0
        # BEFORE the regime, ETB still costs the conservative schedule (no
        # retroactive $0 — that would be look-ahead)
        assert sc.borrow_cost_bps_annual(date(2024, 6, 1), likely_etb=True) == sc.PRE_REGIME_ETB_BPS

    def test_htb_backstop_scales_with_score(self):
        full = sc.borrow_cost_bps_annual(date(2026, 1, 1), likely_etb=False, htb_score=1.0)
        half = sc.borrow_cost_bps_annual(date(2026, 1, 1), likely_etb=False, htb_score=0.5)
        assert full == sc.HTB_BPS and half == pytest.approx(sc.HTB_BPS * 0.5)

    def test_drag_accrues_with_hold(self):
        d1 = sc.borrow_drag_pct(date(2024, 1, 1), 1, likely_etb=True)
        d10 = sc.borrow_drag_pct(date(2024, 1, 1), 10, likely_etb=True)
        assert d10 == pytest.approx(d1 * 10)
        # in-regime ETB drag is exactly zero
        assert sc.borrow_drag_pct(date(2026, 1, 1), 30, likely_etb=True) == 0.0

    def test_accepts_string_and_timestamp_dates(self):
        assert sc.borrow_cost_bps_annual('2026-06-01', likely_etb=True) == 0.0
        assert sc.borrow_cost_bps_annual(pd.Timestamp('2024-01-01'), likely_etb=True) == sc.PRE_REGIME_ETB_BPS

    def test_short_round_trip_adds_borrow_to_long_base(self):
        from fees import round_trip_cost_pct
        base = round_trip_cost_pct('stock', 0.10)
        # in-regime ETB: short cost == long base (borrow 0)
        assert sc.short_round_trip_cost_pct(date(2026, 1, 1), 0.10, 5,
                                            likely_etb=True) == pytest.approx(base)
        # pre-regime: strictly more than the long base
        assert sc.short_round_trip_cost_pct(date(2024, 1, 1), 0.10, 30,
                                            likely_etb=True) > base


class TestShortInterestMetrics:
    def test_dtc_and_pct_float(self):
        m = sc.short_interest_metrics(2_000_000, float_shares=50_000_000,
                                      adv_20d=500_000)
        assert m['days_to_cover'] == pytest.approx(4.0)
        assert m['si_pct_float'] == pytest.approx(0.04)

    def test_missing_inputs_are_none(self):
        m = sc.short_interest_metrics(1_000_000)
        assert m['days_to_cover'] is None and m['si_pct_float'] is None

    def test_publication_pit_shift_one(self):
        # two bi-monthly prints, as-of strictly-before (shift-1: a print is
        # used from the bar AFTER its publication date, never on it).
        pub = pd.Series({pd.Timestamp('2025-01-15'): 10.0,
                         pd.Timestamp('2025-02-01'): 20.0})
        idx = pd.to_datetime(['2025-01-10', '2025-01-15', '2025-01-20',
                              '2025-02-01', '2025-02-05'])
        out = sc.pit_publication_map(pub, idx)
        assert np.isnan(out[0])        # before any publication
        assert np.isnan(out[1])        # ON the 1st pub date -> not yet usable
        assert out[2] == 10.0          # after 1st pub -> sees 10
        assert np.isnan(out[3]) or out[3] == 10.0  # ON 2nd pub date -> still 1st
        assert out[3] == 10.0          # strictly-before: 02-01 bar sees 01-15
        assert out[4] == 20.0          # after 2nd pub -> sees 20


class TestBorrowProxy:
    def test_large_cap_shortable(self):
        assert bp.likely_shortable('NVDA', market_cap=2e12) is True
        assert bp.htb_risk_score(2e12) < 0.1

    def test_spec_class_excluded_even_if_midcap(self):
        # a speculative name is floored HTB regardless of a transient cap
        assert bp.likely_shortable('IONQ', market_cap=3e9,
                                   name_class='spec_growth') is False

    def test_micro_cap_excluded(self):
        assert bp.likely_shortable('XYZ', market_cap=1e8) is False

    def test_missing_cap_is_conservatively_excluded(self):
        assert bp.likely_shortable('UNKNOWN', market_cap=None) is False

    def test_restrict_universe(self):
        caps = {'NVDA': 2e12, 'IONQ': 3e9, 'TINY': 1e8}
        classes = {'IONQ': 'spec_growth'}
        kept = bp.restrict_short_universe(
            ['NVDA', 'IONQ', 'TINY'],
            cap_lookup=lambda s: caps.get(s),
            class_lookup=lambda s: classes.get(s))
        assert kept == ['NVDA']

    def test_config_class_lookup(self):
        lk = bp.class_lookup_from_config()
        # IONQ is spec_growth in stock_config.SECTOR_BUCKETS
        assert lk('IONQ') == 'spec_growth'
