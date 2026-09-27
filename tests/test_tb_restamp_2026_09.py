"""R4 (2026-09-26) — TB labels stamped AFTER every row filter (L7 full fix).

The 2026-09-26 Jetson stock re-harvest fired the R2C-06 [TB-GUARD] on every
name: RS_vs_SPY is NaN wherever SPY's 12-bar ROC is exactly 0, so the
per-ticker dropna removed ~30 INTERIOR bars/name AFTER compute_tb_labels had
stamped positional TB_Bars_* spans on the unfiltered frame. The crypto twin
had the same ordering defect silently (Volume_Ratio NaN on zero-volume
stretches). R4 stamps on the stored rows (continued into the real bars
after the last stored row), re-stamps tickers the cross-sectional
membership mask touched, and turns any remaining post-stamp interior
removal into a fatal exit 3 with no store write.

Synthetic, Mac-safe (numpy/pandas + policy_exits' pure-python fallback when
numba is absent). Three tickers: AAA has an interior NaN feature row, BBB a
raw price/timestamp gap, CCC an interior membership-mask gap.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts'))
try:
    import dotenv  # noqa: F401
except ImportError:  # dev-Mac: stub load_dotenv so harvest modules import
    _m = types.ModuleType('dotenv')
    _m.load_dotenv = lambda *a, **k: None
    sys.modules['dotenv'] = _m

import harvest_stock_data as h          # noqa: E402
import harvest_crypto_data as hc        # noqa: E402
from policy_exits import (compute_tb_labels, exit_walk,  # noqa: E402
                          eod_mask_from_index)
from strategy_config import policy_for  # noqa: E402

FB = [3, 6]
PX = ['Open', 'High', 'Low', 'Close', 'ATR']
N_DAYS, BARS_PER_DAY = 12, 7
AAA_NAN_DAY, AAA_NAN_BAR = 4, 3       # interior feature-NaN bar
BBB_HOLE_DAY = 3                        # raw bars 2..4 missing + a jump
CCC_GAP_DAY = 6                         # membership-mask gap (whole day)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _days():
    return pd.bdate_range('2026-01-05', periods=N_DAYS, tz='UTC')


def _raw_bars(seed, hole_day=None, jump=False):
    rng = np.random.default_rng(seed)
    idx = []
    for d, day in enumerate(_days()):
        for b in range(BARS_PER_DAY):
            if hole_day is not None and d == hole_day and 2 <= b <= 4:
                continue
            idx.append(day + pd.Timedelta(hours=14 + b))
    idx = pd.DatetimeIndex(idx)
    ret = rng.normal(0.0, 0.012, len(idx))
    if jump:  # overnight price gap into the day after the hole
        first_next = int(np.argmax(idx.normalize() > _days()[hole_day]))
        ret[first_next] = -0.08
    close = 50.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.0, 0.006, len(idx))) * close
    return pd.DataFrame({'Open': open_,
                         'High': np.maximum(open_, close) + spread,
                         'Low': np.minimum(open_, close) - spread,
                         'Close': close, 'Volume': 5e6}, index=idx)


RAW = {'AAA': _raw_bars(1),
       'BBB': _raw_bars(2, hole_day=BBB_HOLE_DAY, jump=True),
       'CCC': _raw_bars(3)}


def _features(ohlcv, spy_close=None, symbol=None):
    """Stub compute_stock_features: ATR (NaN warmup -> prefix drop) + one
    feature column; AAA gets ONE interior NaN (the RS_vs_SPY pattern)."""
    df = ohlcv.copy()
    tr = (df['High'] - df['Low'])
    df['ATR'] = tr.rolling(5).mean()
    df['Feat'] = df['Close'].pct_change()
    if symbol == 'AAA':
        ts = _days()[AAA_NAN_DAY] + pd.Timedelta(hours=14 + AAA_NAN_BAR)
        df.loc[ts, 'Feat'] = np.nan
    return df


def _stub_prepare_env(monkeypatch):
    monkeypatch.setattr(h, 'FORWARD_BARS', list(FB))
    monkeypatch.setattr(h, 'compute_stock_features', _features)
    monkeypatch.setattr(h, 'fetch_with_fallback',
                        lambda t, *a, **k: RAW[t].copy())
    monkeypatch.setitem(sys.modules, 'liquidity',
                        types.ModuleType('liquidity'))  # EDGE stamp skipped
    sf = types.ModuleType('short_flow')
    sf.svr_features_for_index = lambda t, idx: None
    sf.sync = lambda: None
    monkeypatch.setitem(sys.modules, 'short_flow', sf)
    cr = types.ModuleType('cost_regime')
    cr.stamp_cost_regime_features = lambda df, at: df
    monkeypatch.setitem(sys.modules, 'cost_regime', cr)
    h._TB_WALK_BARS.clear()


def _replay(stored, fb, asset_type='stock'):
    """What backtest.simulate_ticker / meta_label do: the exit kernel over
    the STORED rows (eod mask built on the stored frame)."""
    c = stored['Close'].to_numpy(float)
    exit_idx, exit_px, reason = exit_walk(
        c, stored['High'].to_numpy(float), stored['Low'].to_numpy(float),
        stored['Open'].to_numpy(float), stored['ATR'].to_numpy(float),
        eod_mask_from_index(stored.index, asset_type), policy_for(asset_type),
        max_hold=int(fb), use_signal_exit=False, side=1)
    return exit_idx - np.arange(len(c)), (exit_px - c) / c * 100.0


def _comparable_rows(stored, fb):
    """Rows whose replay cannot touch the stored frame's forced end-of-data
    flatten: not on the last stored date and with the fb window inside."""
    n = len(stored)
    d = stored.index.normalize()
    return (np.arange(n) + fb < n) & (d < d[-1])


def _assert_label_equals_backtest(stored, asset_type='stock'):
    for fb in FB:
        bars, ret = _replay(stored, fb, asset_type)
        m = _comparable_rows(stored, fb)
        assert m.sum() > 20
        np.testing.assert_array_equal(stored[f'TB_Bars_{fb}'].to_numpy()[m],
                                      bars[m].astype(float))
        np.testing.assert_allclose(stored[f'TB_Ret_{fb}'].to_numpy()[m],
                                   ret[m], rtol=0, atol=1e-9)


def _assert_positional_spans(stored, bars_all):
    """TB_Bars is the stored-row offset of the kernel's exit: re-walking
    stored rows + continuation reproduces it exactly."""
    tail = bars_all.loc[bars_all.index > stored.index[-1], PX]
    frame = pd.concat([stored[PX], tail])
    ref = compute_tb_labels(frame, FB, 'stock')
    n = len(stored)
    for fb in FB:
        for k in ('Bars', 'Ret', 'Reason'):
            np.testing.assert_array_equal(
                stored[f'TB_{k}_{fb}'].to_numpy(), ref[f'TB_{k}_{fb}'][:n])


# ---------------------------------------------------------------------------
# (i) interior NaN feature row — prepare_stock_data
# ---------------------------------------------------------------------------

def test_interior_nan_row_prepare_restamps_and_guard_silent(monkeypatch,
                                                            capsys):
    _stub_prepare_env(monkeypatch)
    out = h.prepare_stock_data('AAA', None, api=None)
    printed = capsys.readouterr().out
    assert '[TB-GUARD]' not in printed
    nan_ts = _days()[AAA_NAN_DAY] + pd.Timedelta(hours=14 + AAA_NAN_BAR)
    assert nan_ts not in out.index                      # interior removal
    assert out.index[0] > RAW['AAA'].index[0]           # ATR-warmup prefix
    assert not out[[c for c in out if c.startswith('TB_')]].isna().any().any()
    full = _features(RAW['AAA'], symbol='AAA')
    _assert_positional_spans(out, full)
    _assert_label_equals_backtest(out)
    # the walk-continuation registry holds real bars from the 1st stored row
    wb = h._TB_WALK_BARS['AAA']
    assert wb.index[0] == out.index[0] and nan_ts in wb.index


def test_old_ordering_would_have_been_invalid_on_same_fixture(monkeypatch):
    """Documents the defect R4 fixes: stamping the UNFILTERED frame then
    dropping the interior NaN row leaves spans pointing at the wrong row."""
    _stub_prepare_env(monkeypatch)
    out = h.prepare_stock_data('AAA', None, api=None)
    full = _features(RAW['AAA'], symbol='AAA')
    old = compute_tb_labels(full, FB, 'stock')
    pos = full.index.get_indexer(out.index)
    wrong = 0
    for fb in FB:
        b = old[f'TB_Bars_{fb}'][pos]
        ok = np.isfinite(b)
        i = np.arange(len(out))[ok]
        j = i + b[ok].astype(int)
        inb = j < len(out)
        exit_true = full.index[pos[ok][inb] + b[ok][inb].astype(int)]
        wrong += int((out.index[j[inb]] != exit_true).sum())
    assert wrong > 0


def test_rows_not_crossing_a_removed_bar_keep_legacy_labels(monkeypatch):
    """Byte-identity where it should hold: a row whose walk never crossed
    the removed bar gets exactly the pre-R4 (full-frame) label."""
    _stub_prepare_env(monkeypatch)
    out = h.prepare_stock_data('AAA', None, api=None)
    full = _features(RAW['AAA'], symbol='AAA')
    old = compute_tb_labels(full, FB, 'stock')
    pos = full.index.get_indexer(out.index)
    nan_pos = int(full.index.get_loc(
        _days()[AAA_NAN_DAY] + pd.Timedelta(hours=14 + AAA_NAN_BAR)))
    for fb in FB:
        ob = old[f'TB_Bars_{fb}'][pos]
        clean = (pos + ob < nan_pos) | (pos > nan_pos)
        assert clean.sum() > len(out) // 2
        for k in ('Bars', 'Ret', 'Reason'):
            np.testing.assert_array_equal(
                out[f'TB_{k}_{fb}'].to_numpy()[clean],
                old[f'TB_{k}_{fb}'][pos][clean])


# ---------------------------------------------------------------------------
# (iii) raw price/timestamp gap — no filter removal, legacy labels exactly
# ---------------------------------------------------------------------------

def test_price_gap_unfiltered_name_is_byte_identical_to_legacy(monkeypatch,
                                                               capsys):
    _stub_prepare_env(monkeypatch)
    out = h.prepare_stock_data('BBB', None, api=None)
    assert '[TB-GUARD]' not in capsys.readouterr().out
    full = _features(RAW['BBB'], symbol='BBB')
    old = compute_tb_labels(full, FB, 'stock')
    pos = full.index.get_indexer(out.index)
    assert ((pos[1:] - pos[:-1]) == 1).all()           # contiguous run
    for col, vals in old.items():
        np.testing.assert_array_equal(out[col].to_numpy(), vals[pos])
    _assert_label_equals_backtest(out)
    # legacy column order kept: TB_* right after the last Target_Return_{fb}
    cols = list(out.columns)
    k = max(i for i, c in enumerate(cols) if c.startswith('Target_Return_'))
    assert cols[k + 1].startswith('TB_')


# ---------------------------------------------------------------------------
# (ii) membership gap + all three names through main()
# ---------------------------------------------------------------------------

def _stub_main(monkeypatch, saved, drop_after_restamp=False):
    _stub_prepare_env(monkeypatch)
    monkeypatch.delenv('TRADER_RAW_SIDECAR', raising=False)
    monkeypatch.setattr(h, 'STOCK_TICKERS', ['AAA', 'BBB', 'CCC'])
    monkeypatch.setattr(h, '_get_alpaca_api', lambda: None)
    monkeypatch.setattr(h, 'fetch_spy_close', lambda api=None: None)
    monkeypatch.setattr(h, 'load_training_data',
                        lambda prefix: pd.DataFrame())
    monkeypatch.setattr(h, 'validate_training_data', lambda df, at: {})

    def save(df, prefix):
        saved['df'] = df.copy()
        return True
    monkeypatch.setattr(h, 'save_training_data', save)
    sh = types.ModuleType('sentiment_history')
    sh.fetch_stock_sentiment_history = lambda *a, **k: {}
    sh.stock_sentiment_lookup_dates = lambda idx: [None] * len(idx)
    monkeypatch.setitem(sys.modules, 'sentiment_history', sh)
    gap_day = _days()[CCC_GAP_DAY]

    def membership(df, top_k=None):   # CCC out of the top-K for one day
        m = (df['Ticker'] == 'CCC').to_numpy() & (
            df.index.normalize() == gap_day)
        return df[~m]
    monkeypatch.setattr(h, '_asof_membership_mask', membership)
    if drop_after_restamp:
        import panel_ranks
        real = panel_ranks.neutral_fill_cs

        def lossy(df):   # a downstream step that eats an INTERIOR row
            df = real(df)
            rows = np.flatnonzero((df['Ticker'] == 'BBB').to_numpy())
            keep = np.ones(len(df), bool)
            keep[rows[len(rows) // 2]] = False
            return df[keep]
        monkeypatch.setattr(panel_ranks, 'neutral_fill_cs', lossy)


def test_main_three_tickers_spans_valid_and_guard_silent(monkeypatch,
                                                         capsys):
    saved = {}
    _stub_main(monkeypatch, saved)
    h.main()
    printed = capsys.readouterr().out
    assert '[TB-GUARD]' not in printed
    assert '[TB-RESTAMP] CCC' in printed
    final = saved['df']
    assert set(final['Ticker']) == {'AAA', 'BBB', 'CCC'}
    for t in ('AAA', 'BBB', 'CCC'):
        stored = final[final['Ticker'] == t].sort_index()
        assert not stored[[c for c in stored if c.startswith('TB_')]
                          ].isna().any().any()
        _assert_positional_spans(stored, _features(RAW[t], symbol=t))
        _assert_label_equals_backtest(stored)
    ccc = final[final['Ticker'] == 'CCC']
    assert not (ccc.index.normalize() == _days()[CCC_GAP_DAY]).any()
    assert h._TB_WALK_BARS == {}          # registry released after use


def test_membership_restamp_changes_only_rows_crossing_the_gap(monkeypatch):
    """Stock labels cap at the session EOD, and the membership mask is
    day-granular, so only the EOD-bar entry before the gap (which backtest
    and meta_label never enter on) can change."""
    saved = {}
    _stub_main(monkeypatch, saved)
    pre = h.prepare_stock_data('CCC', None, api=None)
    h.main()
    ccc = saved['df'][saved['df']['Ticker'] == 'CCC'].sort_index()
    pre = pre.loc[ccc.index]
    changed = np.zeros(len(ccc), bool)
    for fb in FB:
        changed |= ccc[f'TB_Bars_{fb}'].to_numpy() != \
            pre[f'TB_Bars_{fb}'].to_numpy()
    before_gap = _days()[CCC_GAP_DAY - 1]
    eod_before_gap = ccc.index[ccc.index.normalize() == before_gap][-1]
    assert set(ccc.index[changed]) <= {eod_before_gap}


# ---------------------------------------------------------------------------
# fail-loud guard
# ---------------------------------------------------------------------------

def test_prepare_raises_on_post_stamp_interior_removal(monkeypatch):
    _stub_prepare_env(monkeypatch)
    real = h._stamp_tb_labels

    def poisoned(stored, bars, asset_type='stock'):
        out = real(stored, bars, asset_type)
        out.iloc[len(out) // 2, out.columns.get_loc(f'TB_Ret_{FB[0]}')] = \
            np.nan     # -> the TB-NaN dropna removes an INTERIOR row
        return out
    monkeypatch.setattr(h, '_stamp_tb_labels', poisoned)
    with pytest.raises(h.TBSpanError):
        h.prepare_stock_data('BBB', None, api=None)


def test_main_exits_3_without_saving_when_prepare_guard_fires(monkeypatch):
    saved = {}
    _stub_main(monkeypatch, saved)

    def boom(*a, **k):
        raise h.TBSpanError('AAA: interior rows removed after the TB stamp')
    monkeypatch.setattr(h, 'prepare_stock_data', boom)
    with pytest.raises(SystemExit) as ei:
        h.main()
    assert ei.value.code == h.TB_SPAN_EXIT_CODE == 3
    assert 'df' not in saved


def test_main_exits_3_without_saving_on_downstream_interior_removal(
        monkeypatch, capsys):
    saved = {}
    _stub_main(monkeypatch, saved, drop_after_restamp=True)
    with pytest.raises(SystemExit) as ei:
        h.main()
    assert ei.value.code == 3
    assert 'df' not in saved
    out = capsys.readouterr().out
    assert '[TB-GUARD] BBB' in out and 'NO training store written' in out


def test_restamp_without_walk_bars_on_interior_gap_raises():
    idx = pd.date_range('2026-01-05 14:00', periods=10, freq='h', tz='UTC')
    pre = pd.DataFrame({'Ticker': 'ZZZ'}, index=idx)
    post = pd.DataFrame({'Ticker': 'ZZZ', 'Close': 1.0,
                         f'TB_Ret_{FB[0]}': 0.0}, index=idx[[0, 1, 5, 6]])
    with pytest.raises(h.TBSpanError):
        h._restamp_after_membership(post, pre, {})


def test_stamp_precondition_unsorted_or_duplicate_index_raises():
    df = _features(RAW['AAA'], symbol='AAA').dropna()
    with pytest.raises(h.TBSpanError):
        h._stamp_tb_labels(df.iloc[::-1], df)
    dup = pd.concat([df.iloc[:5], df.iloc[4:10]])
    with pytest.raises(h.TBSpanError):
        h._stamp_tb_labels(dup, df)


# ---------------------------------------------------------------------------
# crypto twin (Volume_Ratio NaN on a zero-volume stretch)
# ---------------------------------------------------------------------------

def _crypto_env(monkeypatch, poison=False):
    monkeypatch.setattr(hc, 'FORWARD_BARS', list(FB))
    raw = _raw_bars(7)
    raw.index = pd.date_range('2026-01-05', periods=len(raw), freq='h',
                              tz='UTC')

    def feats(ohlcv, btc_close=None):
        df = ohlcv.copy()
        df['ATR'] = (df['High'] - df['Low']).rolling(5).mean()
        vr = df['Close'].pct_change()
        vr.iloc[40:46] = np.nan        # zero-volume stretch -> NaN ratio
        df['Volume_Ratio'] = vr
        return df
    monkeypatch.setattr(hc, 'compute_features', feats)
    monkeypatch.setattr(hc, 'fetch_with_fallback', lambda *a, **k: raw.copy())
    fa = types.ModuleType('funding_archive')
    fa.funding_features_for_index = lambda s, idx: None
    monkeypatch.setitem(sys.modules, 'funding_archive', fa)
    oa = types.ModuleType('oi_archive')
    for f in ('oi_features_for_index', 'ls_features_for_index',
              'taker_features_for_index'):
        setattr(oa, f, lambda s, idx: None)
    monkeypatch.setitem(sys.modules, 'oi_archive', oa)
    monkeypatch.setitem(sys.modules, 'liquidity',
                        types.ModuleType('liquidity'))
    cr = types.ModuleType('cost_regime')
    cr.stamp_cost_regime_features = lambda df, at: df
    monkeypatch.setitem(sys.modules, 'cost_regime', cr)
    if poison:
        real = hc._stamp_tb_labels

        def poisoned(stored, bars, asset_type='crypto'):
            out = real(stored, bars, asset_type)
            out.iloc[len(out) // 2,
                     out.columns.get_loc(f'TB_Bars_{FB[0]}')] = np.nan
            return out
        monkeypatch.setattr(hc, '_stamp_tb_labels', poisoned)
    return raw, feats


def test_crypto_interior_nan_stamped_after_filter(monkeypatch):
    raw, feats = _crypto_env(monkeypatch)
    out = hc.prepare_data('BTC-USD', api=None)
    full = feats(raw)
    assert not full.index[40:46].isin(out.index).any()   # interior removal
    tail = full.loc[full.index > out.index[-1], PX]
    ref = compute_tb_labels(pd.concat([out[PX], tail]), FB, 'crypto')
    n = len(out)
    for col, vals in ref.items():
        np.testing.assert_array_equal(out[col].to_numpy(), vals[:n])
    for fb in FB:   # label == backtest replay on the stored rows
        bars, ret = _replay(out, fb, 'crypto')
        m = np.arange(n) + fb < n
        np.testing.assert_array_equal(out[f'TB_Bars_{fb}'].to_numpy()[m],
                                      bars[m].astype(float))
        np.testing.assert_allclose(out[f'TB_Ret_{fb}'].to_numpy()[m],
                                   ret[m], rtol=0, atol=1e-9)


def test_crypto_prepare_raises_on_post_stamp_interior_removal(monkeypatch):
    _crypto_env(monkeypatch, poison=True)
    with pytest.raises(hc.TBSpanError):
        hc.prepare_data('BTC-USD', api=None)


def test_crypto_prefix_suffix_helper():
    idx = pd.date_range('2026-01-05', periods=10, freq='h', tz='UTC')
    assert hc._removals_prefix_suffix_only(idx, idx[2:8])
    assert hc._removals_prefix_suffix_only(idx, idx[:0])
    assert not hc._removals_prefix_suffix_only(idx, idx[[0, 1, 5]])
