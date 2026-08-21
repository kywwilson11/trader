"""Packet R2 (2026-08 campaign): OOF primary persistence (D12) + meta replay
parity (D05-meta).

Mac-runnable: numpy + pandas + stdlib only (policy_exits pure-python fallback;
fees and strategy_config are pure). Covers:
  G1  new flags exist and default OFF
  G2  pure OOF helpers (pack build / npz roundtrip / join / starvation tiers)
  G3  _gen_meta_rows flag-OFF byte-identical + diag always populated
  G4  parity ON admission conditions + first-fail drop counters
  G5  entry_preds (OOF) semantics: admission + 'pred' feature, exits unchanged
  G6  _meta_payload legacy compatibility + new-key passthrough
  G7  py_compile of the touched sources (hypersearch_v2 is compile-only here)
"""
import json
import os
import py_compile
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import strategy_config
from fees import required_edge_pct
from policy_exits import exit_walk, REASON_NAMES
from strategy_config import policy_for
from meta_label import (META_FEATURES, OOF_MIN_ROWS, OOF_FULL_PARAMS_MIN_ROWS,
                        OOF_SHRUNK_PARAMS, _gen_meta_rows, _meta_payload,
                        join_oof_to_index, load_oof_npz, oof_pack_from_folds,
                        oof_starvation_tier, write_oof_npz)

THRESHOLD = 0.15                      # train_meta config default
POLICY = policy_for('crypto')
ENTRY_THRESHOLD = THRESHOLD * 0.5     # META_THRESHOLD_FRACTION
CRYPTO_EDGE_FLOOR = required_edge_pct('crypto', 0.10)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _synthetic_tdf(n=400, seed=7):
    """Harvest-shaped hourly frame + primary preds with entries to replay.
    (test_meta_label_v3._synthetic_tdf pattern.)"""
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
    preds = rng.uniform(-0.1, 0.3, n)  # plenty above 0.5x threshold
    return df, preds


def _flat_tdf(n=60, price=100.0):
    """Flat prices: no stop/TP/signal exits — one end_of_data trade per walk."""
    idx = pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC')
    p = np.full(n, price)
    return pd.DataFrame({'Close': p, 'High': p.copy(), 'Low': p.copy(),
                         'Open': p.copy(), 'ATR': np.full(n, 1.0),
                         'RSI': np.full(n, 50.0),
                         'ATR_Pct': np.full(n, 1.0)}, index=idx)


def _plunge_tdf(n=60):
    """Entry at bar 0 hard-stops on the bar-1 plunge; flat at 80 after."""
    tdf = _flat_tdf(n)
    for col in ('Close', 'High', 'Low'):
        tdf.iloc[1:, tdf.columns.get_loc(col)] = 80.0
    tdf.iloc[1, tdf.columns.get_loc('Open')] = 100.0
    tdf.iloc[2:, tdf.columns.get_loc('Open')] = 80.0
    return tdf


def _entry_positions(tdf, times):
    return [tdf.index.get_loc(t) for t in times]


def _mk_pack():
    """2-ticker contiguous pack fixture for G2."""
    t0 = 1_700_000_000
    all_times_s = t0 + np.arange(20, dtype=np.int64) * 3600
    tickers = ['AAA', 'BBB']
    boundaries = {'AAA': (0, 10), 'BBB': (10, 20)}
    fold_rows = [np.array([2, 3, 12]), np.array([5, 15, 19])]
    fold_preds = [np.array([0.1, 0.2, 0.3], np.float32),
                  np.array([0.4, 0.5, 0.6], np.float32)]
    holdout_boundary_s = int(all_times_s[18])   # row 19 must be dropped
    return (fold_rows, fold_preds, all_times_s, tickers, boundaries,
            holdout_boundary_s)


# ---------------------------------------------------------------------------
# G1: flags exist, default OFF
# ---------------------------------------------------------------------------

def test_flags_exist_default_off():
    assert strategy_config.META_OOF_PRED is False
    assert strategy_config.META_REPLAY_POLICY_PARITY is False


# ---------------------------------------------------------------------------
# G2: pure OOF helpers
# ---------------------------------------------------------------------------

def test_oof_pack_from_folds_attribution_and_holdout_drop():
    (fold_rows, fold_preds, all_times_s, tickers, boundaries,
     holdout_boundary_s) = _mk_pack()
    pack = oof_pack_from_folds(fold_rows, fold_preds, None, all_times_s,
                               tickers, boundaries, holdout_boundary_s)
    # row 19 (past the holdout boundary) dropped even though a fold had it
    assert len(pack['ts_ns']) == 5
    kept_rows = [2, 3, 12, 5, 15]
    assert list(pack['ticker']) == ['AAA', 'AAA', 'BBB', 'AAA', 'BBB']
    np.testing.assert_array_equal(
        pack['ts_ns'], all_times_s[kept_rows] * 10 ** 9)
    assert pack['ts_ns'].dtype == np.int64
    assert pack['oof_pred'].dtype == np.float32
    np.testing.assert_allclose(pack['oof_pred'],
                               [0.1, 0.2, 0.3, 0.4, 0.5], rtol=1e-6)
    # fold_ids None -> enumerate
    np.testing.assert_array_equal(pack['fold_id'], [0, 0, 0, 1, 1])
    assert pack['fold_id'].dtype == np.int16


def test_oof_pack_from_folds_explicit_fold_ids():
    (fold_rows, fold_preds, all_times_s, tickers, boundaries,
     holdout_boundary_s) = _mk_pack()
    pack = oof_pack_from_folds(fold_rows, fold_preds, [3, 7], all_times_s,
                               tickers, boundaries, holdout_boundary_s)
    np.testing.assert_array_equal(pack['fold_id'], [3, 3, 3, 7, 7])


def test_oof_pack_from_folds_length_mismatch_raises():
    (fold_rows, fold_preds, all_times_s, tickers, boundaries,
     holdout_boundary_s) = _mk_pack()
    bad_preds = [fold_preds[0][:2], fold_preds[1]]
    with pytest.raises(ValueError):
        oof_pack_from_folds(fold_rows, bad_preds, None, all_times_s,
                            tickers, boundaries, holdout_boundary_s)


def test_write_load_npz_roundtrip_and_statuses(tmp_path):
    (fold_rows, fold_preds, all_times_s, tickers, boundaries,
     holdout_boundary_s) = _mk_pack()
    pack = oof_pack_from_folds(fold_rows, fold_preds, None, all_times_s,
                               tickers, boundaries, holdout_boundary_s)
    path = tmp_path / 'oof_preds.npz'
    write_oof_npz(str(path), pack, '2026-08-19T00:00:00', 1.2345)
    # no tmp residue (tmp + os.replace)
    assert [p.name for p in tmp_path.iterdir()] == ['oof_preds.npz']

    manifest = {'saved_at': '2026-08-19T00:00:00', 'score': 1.2345}
    loaded, status = load_oof_npz(str(path), manifest)
    assert status == 'ok'
    np.testing.assert_array_equal(loaded['ts_ns'], pack['ts_ns'])
    np.testing.assert_array_equal(loaded['oof_pred'], pack['oof_pred'])
    np.testing.assert_array_equal(loaded['fold_id'], pack['fold_id'])
    assert list(loaded['ticker']) == list(pack['ticker'])

    # stale: saved_at mismatch, score mismatch, manifest None
    assert load_oof_npz(str(path), {'saved_at': 'other',
                                    'score': 1.2345})[1] == 'stale'
    assert load_oof_npz(str(path), {'saved_at': '2026-08-19T00:00:00',
                                    'score': 1.2355})[1] == 'stale'
    assert load_oof_npz(str(path), None)[1] == 'stale'
    # missing
    assert load_oof_npz(str(tmp_path / 'nope.npz'), manifest)[1] == 'missing'
    # unreadable: garbage bytes
    bad = tmp_path / 'bad.npz'
    bad.write_bytes(b'not an npz at all')
    assert load_oof_npz(str(bad), manifest)[1] == 'unreadable'
    # unreadable: valid npz missing required keys
    partial = tmp_path / 'partial.npz'
    with open(partial, 'wb') as f:
        np.savez_compressed(f, ticker=pack['ticker'], ts_ns=pack['ts_ns'])
    assert load_oof_npz(str(partial), manifest)[1] == 'unreadable'


def test_write_oof_npz_failure_leaves_no_tmp_residue(tmp_path):
    # fail-soft contract: the caller catches the exception; the tmp file
    # must not linger on the Jetson either way (finally-unlink convention)
    path = tmp_path / 'oof_preds.npz'
    with pytest.raises(KeyError):
        write_oof_npz(str(path), {'ticker': np.array(['A'])},
                      '2026-08-19T00:00:00', 1.0)
    assert list(tmp_path.iterdir()) == []


def test_join_oof_to_index_matching_and_nan(tmp_path):
    (fold_rows, fold_preds, all_times_s, tickers, boundaries,
     holdout_boundary_s) = _mk_pack()
    pack = oof_pack_from_folds(fold_rows, fold_preds, None, all_times_s,
                               tickers, boundaries, holdout_boundary_s)
    # ticker AAA's bar index with a sub-second component — floored to seconds
    idx_ns = all_times_s[0:10] * 10 ** 9 + 123_456_789
    joined = join_oof_to_index(idx_ns, 'AAA', pack)
    assert joined.shape == (10,)
    np.testing.assert_allclose(joined[[2, 3, 5]], [0.1, 0.2, 0.4], rtol=1e-6)
    other = np.ones(10, bool)
    other[[2, 3, 5]] = False
    assert np.isnan(joined[other]).all()
    # BBB rows never leak into AAA's join and vice versa
    idx_b = all_times_s[10:20] * 10 ** 9
    joined_b = join_oof_to_index(idx_b, 'BBB', pack)
    np.testing.assert_allclose(joined_b[[2, 5]], [0.3, 0.5], rtol=1e-6)
    assert np.isnan(joined_b[[0, 1, 3, 4, 6, 7, 8, 9]]).all()
    # unknown ticker -> all NaN
    assert np.isnan(join_oof_to_index(idx_ns, 'ZZZ', pack)).all()


def test_oof_starvation_tiers():
    assert oof_starvation_tier(199) == ('starved', None)
    for n in (200, 999):
        name, params = oof_starvation_tier(n)
        assert name == 'shrunk'
        assert params['num_leaves'] == 8
        assert params['max_depth'] == 3
        assert params['feature_fraction'] == 0.6
        assert params['min_data_in_leaf'] == max(20, n // 20)
    assert oof_starvation_tier(1000) == ('full', None)
    assert OOF_MIN_ROWS == 200 and OOF_FULL_PARAMS_MIN_ROWS == 1000
    assert 'min_data_in_leaf' not in OOF_SHRUNK_PARAMS  # computed per call


# ---------------------------------------------------------------------------
# G3: flag-OFF byte-identical + diag always populated
# ---------------------------------------------------------------------------

def test_gen_meta_rows_off_byte_identical_and_diag():
    tdf, preds = _synthetic_tdf()
    legacy = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY)
    diag = {}
    off = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY,
                         parity=False, diag=diag)
    assert len(legacy[0]) > 0
    for a, b in zip(legacy, off):
        assert len(a) == len(b)
        for va, vb in zip(a, b):
            if isinstance(va, np.ndarray):
                np.testing.assert_array_equal(va, vb)
            else:
                assert va == vb
    # diag is populated even with the flag off (instrumentation is direct)
    assert diag['rows_legacy'] == len(legacy[0])
    assert set(diag['drops']) == {'lockout', 'edge_floor', 'entry_window',
                                  'q10'}
    assert diag['edge_floor_pct'] == pytest.approx(CRYPTO_EDGE_FLOOR)
    assert diag['n_bars'] == len(tdf)
    assert diag['n_covered'] == int(np.isfinite(preds).sum())
    # the parity walk MEASURES even while the population stays legacy:
    # every synthetic pred is below the 1.2 crypto edge floor
    assert preds.max() < CRYPTO_EDGE_FLOOR
    assert diag['rows_parity'] == 0
    assert diag['drops']['edge_floor'] > 0


def test_parity_args_ignored_when_parity_off():
    # train_meta's OFF-path wiring passes entry_ok (stock) and q10 args with
    # parity=False — the LEGACY population must be untouched by all of them
    tdf, preds = _synthetic_tdf()
    legacy = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY)
    off = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY,
                         parity=False,
                         entry_ok=np.zeros(len(tdf), dtype=bool),
                         q10_preds=np.full(len(tdf), -99.0), q10_floor=-1.0)
    assert len(legacy[0]) > 0
    for a, b in zip(legacy, off):
        assert len(a) == len(b)
        for va, vb in zip(a, b):
            if isinstance(va, np.ndarray):
                np.testing.assert_array_equal(va, vb)
            else:
                assert va == vb


def test_rows_match_when_no_parity_condition_binds():
    # flat prices (no hard stop), preds above the edge floor -> the parity
    # walk admits exactly the legacy population
    tdf = _flat_tdf()
    preds = np.full(len(tdf), CRYPTO_EDGE_FLOOR + 1.0)
    diag = {}
    _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY,
                   parity=False, diag=diag)
    assert diag['rows_legacy'] == diag['rows_parity'] > 0
    assert all(v == 0 for v in diag['drops'].values())


# ---------------------------------------------------------------------------
# G4: parity ON admission conditions + first-fail drop counters
# ---------------------------------------------------------------------------

def test_parity_edge_floor_drop():
    tdf = _flat_tdf()
    # in [entry_threshold, edge_floor): admitted legacy, dropped parity
    assert ENTRY_THRESHOLD < 0.5 < CRYPTO_EDGE_FLOOR
    preds = np.full(len(tdf), 0.5)
    diag = {}
    rows, *_ = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY,
                              parity=True, diag=diag)
    assert len(rows) == 0
    assert diag['rows_parity'] == 0
    assert diag['rows_legacy'] > 0
    assert diag['drops']['edge_floor'] > 0
    assert diag['drops']['lockout'] == diag['drops']['entry_window'] == \
        diag['drops']['q10'] == 0


def test_parity_entry_window_drop():
    tdf = _flat_tdf()
    preds = np.full(len(tdf), CRYPTO_EDGE_FLOOR + 1.0)  # passes the floor
    entry_ok = np.zeros(len(tdf), dtype=bool)
    diag = {}
    rows, *_ = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY,
                              parity=True, entry_ok=entry_ok, diag=diag)
    assert len(rows) == 0
    assert diag['drops']['entry_window'] > 0
    assert diag['drops']['edge_floor'] == 0


def test_parity_q10_veto_and_nan_admitted():
    tdf = _flat_tdf()
    n = len(tdf)
    preds = np.full(n, CRYPTO_EDGE_FLOOR + 1.0)
    q10 = np.full(n, -5.0)
    diag = {}
    rows, *_ = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY,
                              parity=True, q10_preds=q10, q10_floor=-1.0,
                              diag=diag)
    assert len(rows) == 0 and diag['drops']['q10'] > 0
    # NaN q10 fails open (matches simulate_ticker)
    diag2 = {}
    rows2, *_ = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY,
                               parity=True, q10_preds=np.full(n, np.nan),
                               q10_floor=-1.0, diag=diag2)
    assert len(rows2) > 0 and diag2['drops']['q10'] == 0


def test_parity_hard_stop_lockout():
    tdf = _plunge_tdf()
    n = len(tdf)
    preds = np.full(n, CRYPTO_EDGE_FLOOR + 1.0)
    # confirm the engineered scenario: entry 0 exits via hard_stop at bar 1
    is_eod = np.zeros(n, bool)
    cooldown_bars = 1   # crypto: ceil(60/60)
    ei, _, rc = exit_walk(
        tdf['Close'].values, tdf['High'].values, tdf['Low'].values,
        tdf['Open'].values, tdf['ATR'].values, is_eod, POLICY,
        preds=preds, threshold=THRESHOLD, cooldown_bars=cooldown_bars,
        max_hold=0, use_signal_exit=True)
    assert REASON_NAMES[int(rc[0])] == 'hard_stop' and int(ei[0]) == 1

    legacy = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY)
    diag = {}
    par = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY,
                         parity=True, diag=diag)
    legacy_entries = _entry_positions(tdf, legacy[3])
    parity_entries = _entry_positions(tdf, par[3])
    lockout_bars = int(POLICY['lockout_hours'])
    # legacy re-enters after the plain cooldown; parity waits out the lockout
    assert legacy_entries[:2] == [0, 1 + cooldown_bars]
    assert parity_entries[0] == 0
    assert parity_entries[1] >= 1 + max(cooldown_bars, lockout_bars)
    assert diag['drops']['lockout'] == \
        max(cooldown_bars, lockout_bars) - cooldown_bars


# ---------------------------------------------------------------------------
# G5: entry_preds (OOF) semantics
# ---------------------------------------------------------------------------

def test_entry_preds_nan_holes_never_selected():
    tdf = _flat_tdf()
    n = len(tdf)
    preds = np.full(n, 2.0)
    entry_preds = np.full(n, 2.0)
    entry_preds[:5] = np.nan          # outside OOF coverage -> DROPPED
    rows, labels, nets, times, exit_times = _gen_meta_rows(
        tdf, preds, 'crypto', THRESHOLD, POLICY, entry_preds=entry_preds)
    entries = _entry_positions(tdf, times)
    assert len(entries) > 0
    assert all(e >= 5 for e in entries)


def test_entry_preds_drive_pred_feature_not_preds():
    tdf, preds = _synthetic_tdf()
    rng = np.random.default_rng(11)
    entry_preds = preds + rng.uniform(0.001, 0.002, len(preds))
    rows, labels, nets, times, exit_times = _gen_meta_rows(
        tdf, preds, 'crypto', THRESHOLD, POLICY, entry_preds=entry_preds)
    assert META_FEATURES[0] == 'pred'
    entries = _entry_positions(tdf, times)
    assert len(entries) > 0
    for row, e in zip(rows, entries):
        assert row[0] == pytest.approx(entry_preds[e])
        assert row[0] != preds[e]


def test_entry_preds_equal_to_preds_is_identical_to_legacy():
    tdf, preds = _synthetic_tdf()
    legacy = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY)
    via_entry = _gen_meta_rows(tdf, preds, 'crypto', THRESHOLD, POLICY,
                               entry_preds=preds.copy())
    for a, b in zip(legacy, via_entry):
        assert len(a) == len(b)
        for va, vb in zip(a, b):
            if isinstance(va, np.ndarray):
                np.testing.assert_array_equal(va, vb)
            else:
                assert va == vb


# ---------------------------------------------------------------------------
# G6: _meta_payload compatibility
# ---------------------------------------------------------------------------

def _payload_common():
    return dict(
        n_trades=500, base_rate=0.55,
        holdout_cutoff_utc='2026-01-01T00:00:00+00:00',
        n_rows_total=10000, n_rows_pre_cutoff=8800, n_tickers_used=12,
        skipped_tickers=[('XYZ', 'too_short')], zero_filled_features=['Hurst'],
        net_summary={'mean_pct': 0.1, 'median_pct': 0.05,
                     'p10_pct': -1.0, 'p90_pct': 1.0},
        calibration={'used': 'legacy'}, primary=None,
        trained_at='2026-07-25T00:00:00+00:00',
    )


def test_meta_payload_legacy_call_defaults():
    payload = _meta_payload(val_auc=0.6, **_payload_common())
    assert payload['pred_source'] == 'in_sample'
    assert payload['oof'] is None
    assert payload['replay_parity'] is None
    json.dumps(payload)


def test_meta_payload_new_kwargs_pass_through_legacy_byte_equal():
    common = _payload_common()
    base = _meta_payload(val_auc=0.6, **common)
    oof = {'status': 'ok', 'tier': 'full'}
    rp = {'enabled': True, 'drops': {'lockout': 1}}
    ext = _meta_payload(val_auc=0.6, pred_source='oof', oof=oof,
                        replay_parity=rp, **common)
    assert ext['pred_source'] == 'oof'
    assert ext['oof'] is oof
    assert ext['replay_parity'] is rp
    for k in base:
        if k not in ('pred_source', 'oof', 'replay_parity'):
            assert base[k] == ext[k]
    json.dumps(ext)


# ---------------------------------------------------------------------------
# G7: touched sources compile (hypersearch_v2 is compile-only on this Mac)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('rel', ['scripts/hypersearch_v2.py', 'meta_label.py',
                                 'strategy_config.py'])
def test_touched_sources_compile(rel, tmp_path):
    py_compile.compile(str(REPO / rel),
                       cfile=str(tmp_path / (Path(rel).name + 'c')),
                       doraise=True)
