"""Review batch b10 — fees.py / liquidity.py / cost_regime.py fixes.

Pins:
  fees — the maker-share cache is keyed by its (days, min_entries) args (no
    cross-arg poisoning; TTL per key; reset-to-None tolerated), malformed
    journal symbol rows are skipped not fatal, journal-scan failures and
    unknown asset_type warn instead of staying silent, the stale SEC/TAF
    dollar figures are gone, and FLAT_SPREAD_PCT is the canonical flat spread
    whose out-of-batch copies in backtest.py / meta_label.py must still agree.
  liquidity — the bidask->AR estimator swap and the inactive-impact paths log,
    the AR-proxy docstring declares its upward bias, and the per-bar impact
    term fail-opens on non-finite notional/k instead of NaN-poisoning costs.
  cost_regime — amihud_illiq preserves the caller's index (the harvest wiring
    pattern) and masks zero-close inf, small pct_window no longer crashes the
    VIX percentile, degenerate FRED CSVs parse to None, stale VIX ffill warns,
    and urlopen is context-managed.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cost_regime
import fees
import liquidity

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# helpers / fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def journals(tmp_path, monkeypatch):
    """Point trade_journal.JOURNAL_DIR at a tmp dir; reset the cache."""
    import trade_journal
    monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', tmp_path)
    fees._maker_share_cache = None
    yield tmp_path
    fees._maker_share_cache = None


def _write_entries(path_dir, n_maker, n_taker=0):
    day = datetime.now().date().isoformat()
    rows = ([{'symbol': 'BTC/USD', 'action': 'buy', 'entry_tactic': 'maker'}]
            * n_maker
            + [{'symbol': 'ETH/USD', 'action': 'buy',
                'entry_tactic': 'taker_fallback'}] * n_taker)
    with open(path_dir / f'{day}.jsonl', 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def _ohlc(n=60, seed=0):
    rng = np.random.RandomState(seed)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    close = mid * (1 + rng.choice([-0.002, 0.002], n))
    high = np.maximum(mid, close) * 1.003
    low = np.minimum(mid, close) * 0.997
    idx = pd.date_range('2025-01-01', periods=n, freq='h')
    return pd.DataFrame({'Open': mid, 'High': high, 'Low': low,
                         'Close': close}, index=idx)


# --------------------------------------------------------------------------
# fees: maker-share cache keyed by arguments
# --------------------------------------------------------------------------

class TestMakerShareCacheKeying:
    def test_non_default_args_get_their_own_slot(self, journals):
        _write_entries(journals, n_maker=40)
        assert fees.realized_crypto_maker_share() == pytest.approx(1.0)
        # A stricter min_entries must be computed fresh (40 < 100 -> None),
        # not served from the default-args slot...
        assert fees.realized_crypto_maker_share(min_entries=100) is None
        # ...and must not poison the default-args production caller.
        assert fees.realized_crypto_maker_share() == pytest.approx(1.0)
        assert fees.crypto_entry_fee_bps(live=True) == pytest.approx(
            fees.CRYPTO_MAKER_BPS)

    def test_ttl_cache_still_hits_per_key(self, journals):
        _write_entries(journals, n_maker=40)
        assert fees.realized_crypto_maker_share() == pytest.approx(1.0)
        for f in journals.glob('*.jsonl'):
            f.unlink()
        # journals gone, but the same-key call inside the TTL is served cached
        assert fees.realized_crypto_maker_share() == pytest.approx(1.0)

    def test_reset_to_none_mid_flight_tolerated(self, journals):
        # tests/test_fees_feedback.py resets the cache with None — the dict
        # rework must keep honoring that contract.
        _write_entries(journals, n_maker=40)
        assert fees.realized_crypto_maker_share() == pytest.approx(1.0)
        fees._maker_share_cache = None
        assert fees.realized_crypto_maker_share() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# fees: journal-scan robustness + logging
# --------------------------------------------------------------------------

class TestMakerShareScanRobustness:
    def test_malformed_symbol_row_skipped_not_fatal(self, journals):
        _write_entries(journals, n_maker=30)
        day = datetime.now().date().isoformat()
        with open(journals / f'{day}.jsonl', 'a') as f:
            f.write(json.dumps({'symbol': None, 'action': 'buy',
                                'entry_tactic': 'maker'}) + '\n')
        # pre-fix: TypeError ("'/' in None") aborted the whole scan -> None
        assert fees.realized_crypto_maker_share() == pytest.approx(1.0)

    def test_scan_failure_logs_warning_and_fails_safe(self, journals,
                                                      monkeypatch, caplog):
        import trade_journal
        monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', None)  # -> TypeError
        fees._maker_share_cache = None
        with caplog.at_level(logging.WARNING, logger='fees'):
            assert fees.realized_crypto_maker_share() is None
        msgs = [r.getMessage() for r in caplog.records if r.name == 'fees']
        assert any('maker-share' in m for m in msgs)
        # fail direction unchanged: live gate prices full taker
        assert fees.crypto_entry_fee_bps(live=True) == fees.CRYPTO_TAKER_BPS


# --------------------------------------------------------------------------
# fees: unknown asset_type warns (still priced as stock)
# --------------------------------------------------------------------------

class TestUnknownAssetType:
    def test_unknown_type_warns_and_prices_as_stock(self, caplog):
        with caplog.at_level(logging.WARNING, logger='fees'):
            got = fees.round_trip_cost_pct('option', 0.1)
        assert got == pytest.approx(fees.round_trip_cost_pct('stock', 0.1))
        msgs = [r.getMessage() for r in caplog.records if r.name == 'fees']
        assert any('asset_type' in m for m in msgs)

    def test_known_types_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger='fees'):
            fees.round_trip_cost_pct('stock', 0.1)
            fees.round_trip_cost_pct('crypto', 0.1)
        assert not [r for r in caplog.records if r.name == 'fees']


# --------------------------------------------------------------------------
# fees: FLAT_SPREAD_PCT is canonical; out-of-batch copies must agree
# --------------------------------------------------------------------------

class TestFlatSpreadCanonical:
    def test_values(self):
        assert fees.FLAT_SPREAD_PCT == {'crypto': 0.10, 'stock': 0.05}

    def _source_flat(self, fname, pattern):
        import re
        src = (REPO / fname).read_text()
        m = re.search(pattern, src)
        assert m, (f"flat-spread literal not found in {fname} — if it now "
                   f"imports fees.FLAT_SPREAD_PCT, update this test")
        return {'crypto': float(m.group(1)), 'stock': float(m.group(2))}

    def test_backtest_copy_agrees(self):
        # backtest.py is outside batch b10's ownership: enforce agreement by
        # source until it imports fees.FLAT_SPREAD_PCT directly.
        got = self._source_flat(
            'backtest.py',
            r"SPREAD_PCT\s*=\s*\{'crypto':\s*([\d.]+),\s*'stock':\s*([\d.]+)\}")
        assert got == fees.FLAT_SPREAD_PCT

    def test_meta_label_copy_agrees(self):
        got = self._source_flat(
            'meta_label.py',
            r"spread\s*=\s*([\d.]+)\s+if\s+asset_type\s*==\s*'crypto'"
            r"\s+else\s+([\d.]+)")
        assert got == fees.FLAT_SPREAD_PCT

    def test_per_bar_flat_fallback_reads_fees_constant(self, monkeypatch):
        # liquidity's NaN-spread fallback must track fees.FLAT_SPREAD_PCT,
        # not a private copy of the numbers.
        monkeypatch.setattr(fees, 'FLAT_SPREAD_PCT',
                            {'crypto': 0.10, 'stock': 0.33})
        cost = liquidity.per_bar_round_trip_cost('stock', np.array([np.nan]))
        assert cost[0] == pytest.approx(fees.round_trip_cost_pct('stock', 0.33))


class TestFeesDocstring:
    def test_stale_regulatory_dollar_rates_removed(self):
        src = (REPO / 'fees.py').read_text()
        assert '$20.60' not in src and '0.000195' not in src
        assert 'allowance' in fees.__doc__


# --------------------------------------------------------------------------
# liquidity: per-bar impact fail-open parity with market_impact_pct
# --------------------------------------------------------------------------

class TestPerBarImpactFailOpen:
    def test_nonfinite_or_nonpositive_notional_skips_impact(self):
        sp = np.array([0.1, 0.2])
        adv = np.array([5e6, 5e6])
        base = liquidity.per_bar_round_trip_cost('stock', sp)
        for bad in (np.nan, np.inf, -np.inf, 0.0, -5.0):
            got = liquidity.per_bar_round_trip_cost(
                'stock', sp, adv_dollar=adv, notional=bad)
            # pre-fix: notional=nan poisoned the ENTIRE array to NaN
            assert np.isfinite(got).all()
            np.testing.assert_allclose(got, base)

    def test_nonfinite_impact_k_skips_impact(self):
        sp = np.array([0.1, 0.2])
        adv = np.array([5e6, 5e6])
        base = liquidity.per_bar_round_trip_cost('stock', sp)
        got = liquidity.per_bar_round_trip_cost(
            'stock', sp, adv_dollar=adv, notional=25_000.0, impact_k=np.nan)
        np.testing.assert_allclose(got, base)

    def test_finite_impact_path_matches_scalar(self):
        sp = np.array([0.10, 0.30])
        adv = np.array([5e6, 1e6])
        base = liquidity.per_bar_round_trip_cost('stock', sp)
        got = liquidity.per_bar_round_trip_cost(
            'stock', sp, adv_dollar=adv, notional=25_000.0, impact_k=1.3)
        for i in range(2):
            expected = liquidity.market_impact_pct(25_000.0, adv[i], sp[i],
                                                   k=1.3, sides=2)
            assert got[i] - base[i] == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------
# liquidity: logging on silent-failure paths
# --------------------------------------------------------------------------

class TestLiquidityLogging:
    def test_bidask_failure_logs_and_falls_back(self, monkeypatch, caplog):
        import bidask

        def boom(*a, **k):
            raise RuntimeError('bidask exploded')

        monkeypatch.setattr(bidask, 'edge_rolling', boom)
        df = _ohlc(n=60)
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            s = liquidity.edge_spread_series(df, window=20)
        assert len(s) == 60 and s.notna().all()
        assert (s >= liquidity.SPREAD_FLOOR_PCT - 1e-9).all()
        msgs = [r.getMessage() for r in caplog.records if r.name == 'liquidity']
        assert any('fallback' in m and 'bidask exploded' in m for m in msgs)

    def test_bidask_success_stays_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            liquidity.edge_spread_series(_ohlc(n=60), window=20)
        assert not [r for r in caplog.records if r.name == 'liquidity']

    def test_impact_enabled_without_dv30_warns(self, monkeypatch, caplog):
        import strategy_config
        monkeypatch.setattr(strategy_config, 'IMPACT_COST_ENABLED', True,
                            raising=True)
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            adv, notional, _k = liquidity.impact_inputs_from_df(
                pd.DataFrame({'Close': [1.0]}))
        assert adv is None and notional is None
        msgs = [r.getMessage() for r in caplog.records if r.name == 'liquidity']
        assert any('DV30' in m for m in msgs)

    def test_impact_disabled_stays_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger='liquidity'):
            liquidity.impact_inputs_from_df(
                pd.DataFrame({'Close': [1.0], 'DV30': [5e6]}))
        assert not [r for r in caplog.records if r.name == 'liquidity']

    def test_ar_fallback_docstring_declares_upward_bias(self):
        doc = liquidity._abdi_ranaldo_rolling.__doc__
        assert 'UPWARD' in doc and 'UPPER BOUND' in doc


# --------------------------------------------------------------------------
# cost_regime: amihud_illiq index preservation + zero-close masking
# --------------------------------------------------------------------------

class TestAmihudIndex:
    def test_preserves_datetimeindex_for_harvest_wiring(self):
        idx = pd.date_range('2025-01-01', periods=40, freq='h')
        rng = np.random.RandomState(3)
        df = pd.DataFrame({'Close': 100 + rng.normal(0, 1, 40).cumsum(),
                           'Volume': np.full(40, 1e6)}, index=idx)
        out = cost_regime.amihud_illiq(df['Close'], df['Volume'], window=10)
        assert out.index.equals(idx)
        df['Amihud'] = out                 # the tracked wave-8 wiring pattern
        # pre-fix: RangeIndex output aligned to all-NaN on a DatetimeIndex df
        assert df['Amihud'].notna().sum() > 0

    def test_array_input_keeps_rangeindex(self):
        out = cost_regime.amihud_illiq(np.linspace(100, 110, 30),
                                       np.full(30, 1e6), window=10)
        assert isinstance(out.index, pd.RangeIndex)
        assert out.notna().sum() > 0

    def test_zero_close_masked_not_inf(self):
        # a 0.0 prior close makes pct_change() emit inf — mirror of the
        # existing zero-VOLUME guard test
        close = pd.Series([100.0, 0.0, 100.0, 101.0, 102.0, 103.0, 104.0,
                           105.0])
        vol = pd.Series(np.full(8, 1e6))
        out = cost_regime.amihud_illiq(close, vol, window=3)
        arr = out.to_numpy()
        assert not np.isinf(arr).any()
        assert np.isfinite(arr[-1])        # recovers after the junk row


# --------------------------------------------------------------------------
# cost_regime: VIX percentile min_periods clamp
# --------------------------------------------------------------------------

class TestVixSmallWindow:
    def test_small_pct_window_no_longer_crashes(self):
        days = pd.date_range('2024-01-01', periods=40, freq='D')
        vix = pd.Series(np.linspace(10, 40, 40), index=days)
        idx = pd.date_range('2024-01-20 14:00', periods=3, freq='h', tz='UTC')
        # pre-fix: ValueError (min_periods 20 > window 10)
        out = cost_regime.vix_features_for_index(vix, idx, pct_window=10)
        assert out is not None
        p = out['VIX_Pctile'][0]
        assert np.isfinite(p) and 0.0 <= p <= 1.0

    def test_min_periods_20_semantics_unchanged_for_large_windows(self):
        # monotone series: pctile needs 20 obs, so the first defined (lagged)
        # value lands on day 21 exactly as before the clamp
        days = pd.date_range('2024-01-01', periods=30, freq='D')
        vix = pd.Series(np.linspace(10, 40, 30), index=days)
        out20 = cost_regime.vix_features_for_index(
            vix, pd.DatetimeIndex(['2024-01-20 10:00']), pct_window=252)
        out21 = cost_regime.vix_features_for_index(
            vix, pd.DatetimeIndex(['2024-01-21 10:00']), pct_window=252)
        assert np.isnan(out20['VIX_Pctile'][0])
        assert out21['VIX_Pctile'][0] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# cost_regime: degenerate FRED CSVs -> None (documented contract)
# --------------------------------------------------------------------------

class TestFredDegenerate:
    def test_header_only_returns_none(self):
        assert cost_regime.parse_fred_vixcls("DATE,VIXCLS\n") is None

    def test_all_missing_values_returns_none(self):
        csv = "DATE,VIXCLS\n2024-01-02,.\n2024-01-03,.\n"
        assert cost_regime.parse_fred_vixcls(csv) is None

    def test_valid_csv_still_parses(self):
        csv = "DATE,VIXCLS\n2024-01-02,13.2\n2024-01-03,14.0\n"
        s = cost_regime.parse_fred_vixcls(csv)
        assert len(s) == 2 and s.iloc[0] == pytest.approx(13.2)


# --------------------------------------------------------------------------
# cost_regime: stale-ffill visibility
# --------------------------------------------------------------------------

class TestVixStaleWarning:
    def test_stale_ffill_warns_but_still_fills(self, capsys):
        days = pd.date_range('2024-01-01', periods=40, freq='D')
        vix = pd.Series(np.linspace(10, 40, 40), index=days)
        # bars ~4 months past the end of the VIX history
        idx = pd.date_range('2024-06-01 14:00', periods=2, freq='h', tz='UTC')
        out = cost_regime.vix_features_for_index(vix, idx, pct_window=30)
        captured = capsys.readouterr().out
        assert '[COST-REGIME]' in captured and 'stale' in captured
        # behavior unchanged: the stale last lagged value is still stamped
        assert out['VIX_Level'][0] == pytest.approx(vix.iloc[-2])

    def test_fresh_bars_stay_silent(self, capsys):
        days = pd.date_range('2024-01-01', periods=40, freq='D')
        vix = pd.Series(np.linspace(10, 40, 40), index=days)
        idx = pd.date_range('2024-02-05 14:00', periods=2, freq='h', tz='UTC')
        cost_regime.vix_features_for_index(vix, idx, pct_window=30)
        assert '[COST-REGIME]' not in capsys.readouterr().out


class TestSourceHygiene:
    def test_urlopen_uses_context_manager(self):
        src = (REPO / 'cost_regime.py').read_text()
        assert 'with urllib.request.urlopen' in src
