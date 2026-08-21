"""Review batch b15 regression tests.

validation.py — NaN n_eff sanitization, unparseable fold-score rows, CSCV
non-finite row drop, block-width divisibility, Lo-2002 docstring formula.
shadow.py — realized-return window aliasing (P1), challenger booster-cache
eviction/healing (P1), promotion failure-path hardening, shadow-log pruning,
report hygiene. ic_diagnostic.py — honest t-stat on finite pairs, consistency
over ALL sub-periods, no abs() promotion of negative IC.

Heavy-dep surfaces (predict_now/torch, joblib) are exercised through fake
modules in sys.modules or failure paths that behave identically on the
dev Mac and the Jetson.
"""
import datetime as dt
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shadow
import validation as V
from ic_diagnostic import ic_by_name, promote_set


# =====================================================================
# validation.py
# =====================================================================

def test_dsr_nonfinite_n_eff_falls_back_to_iid():
    rng = np.random.default_rng(1)
    r = rng.normal(0.05, 1.0, 300)
    iid = V.dsr_from_trade_returns(r, n_trials=50)
    for bad in (float('nan'), float('inf'), float('-inf')):
        out = V.dsr_from_trade_returns(r, n_trials=50, n_eff=bad)
        assert math.isfinite(out['dsr'])         # was nan before the fix
        assert out == iid


def test_deflated_sharpe_nan_n_eff_matches_iid():
    base = V.deflated_sharpe_ratio(0.12, 0.05, 400, n_eff=None)
    out = V.deflated_sharpe_ratio(0.12, 0.05, 400, n_eff=float('nan'))
    assert math.isfinite(out)
    assert out == base


def test_pbo_fold_scores_skips_unparseable_rows():
    rng = np.random.default_rng(3)
    good = [list(rng.normal(size=3)) for _ in range(10)]
    clean = V.pbo_from_fold_scores(good)
    assert clean is not None
    # A None-bearing row used to raise TypeError from np.isfinite on the
    # raw object; strings / scalars / short rows must be filtered too.
    dirty = good + [[1.2, None, 0.8], ['a', 'b', 'c'], 3.5, [0.4]]
    assert V.pbo_from_fold_scores(dirty) == clean


def test_pbo_cscv_drops_nonfinite_rows_instead_of_zero_filling():
    rng = np.random.RandomState(7)
    m = rng.normal(0, 1, (12, 16))
    base = V.pbo_cscv(m, n_groups=8)
    assert base is not None
    all_nan = np.vstack([m, np.full(16, np.nan)])
    part_nan = np.vstack([m, np.r_[rng.normal(0, 1, 8), np.full(8, np.nan)]])
    # Zero-filling mutated trial performance and shifted OOS ranks; a
    # dropped row must leave the result bit-identical to the clean matrix.
    assert V.pbo_cscv(all_nan, n_groups=8) == base
    assert V.pbo_cscv(part_nan, n_groups=8) == base


def test_pbo_cscv_none_when_nonfinite_leaves_too_few_trials():
    m = np.vstack([np.arange(16.0), np.full(16, np.nan)])
    assert V.pbo_cscv(m, n_groups=8) is None


def test_pbo_oos_blocks_rejects_non_divisible_width():
    rng = np.random.default_rng(9)
    # 12 % 8 != 0: pbo_cscv would silently drop the 4 MOST RECENT blocks
    # of every trial; the wrapper must fail open (None) instead.
    rows12 = [rng.normal(size=12) for _ in range(6)]
    assert V.pbo_from_oos_blocks(rows12, n_groups=8) is None
    rows16 = [rng.normal(size=16) for _ in range(6)]
    assert V.pbo_from_oos_blocks(rows16, n_groups=8) is not None


def test_lo_factor_matches_newey_west_docstring():
    rng = np.random.default_rng(11)
    n = 240
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + e[i]
    q = 6
    out = V.serial_correlation_factor(x, max_lag=q)
    xm = x - x.mean()
    denom = float(np.sum(xm * xm))
    expected = 1.0
    for k in range(1, q + 1):
        rho = float(np.sum(xm[k:] * xm[:-k]) / denom)
        expected += 2.0 * (1.0 - k / (q + 1.0)) * rho   # Bartlett/NW weights
    assert out['factor'] == pytest.approx(max(expected, 1e-6))
    # docstring formula must state the weight the code implements
    assert '(1 - k/(q+1))' in V.serial_correlation_factor.__doc__
    assert '(1 - k/n)' not in V.serial_correlation_factor.__doc__


def test_validation_docstrings_match_gates():
    # Both real gates (backtest --gate, hypersearch holdout) use >=
    assert 'DSR >= DSR_MIN' in V.__doc__
    # Acklam's 1.15e-9 bound is a RELATIVE error bound
    assert 'relative |err|' in V._norm_ppf.__doc__


# =====================================================================
# shadow.py
# =====================================================================

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, 'BASE_DIR', tmp_path)
    return tmp_path


def _mk_manifest(base, prefix=''):
    man = shadow.challenger_manifest(prefix)
    man.write_text(json.dumps({'saved_at': 'x', 'holdout': {}}))
    return int(man.stat().st_mtime)


def _append_rows(base, prefix, rows):
    with open(shadow.shadow_log_file(prefix), 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def _fake_closes(start, hours, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=hours, freq='h', tz='UTC')
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.002, hours))
    return pd.Series(prices, index=idx)


def _install_fake_predict_now(monkeypatch, lgb=None, q10=None, pred=0.5):
    fake = types.ModuleType('predict_now')
    fake._lgb_models = dict(lgb or {})
    fake._q10_models = dict(q10 or {})
    fake.load_model = lambda inference_device=None, prefix='': (
        'model', 'scaler', {'forward_bars': 12, 'prefix': prefix}, 40, ['f'])
    fake.get_live_prediction = lambda *a, **k: pred
    monkeypatch.setitem(sys.modules, 'predict_now', fake)
    return fake


class _LoopStub:
    MODEL_PREFIX = ''
    api = object()
    config = {'forward_bars': 24}

    @staticmethod
    def get_asset_type():
        return 'crypto'


# --- P1: realized-return aliasing ---

def test_realized_none_when_ts_predates_window():
    start = dt.datetime(2026, 5, 10, tzinfo=dt.timezone.utc)
    idx = pd.date_range(start, periods=60, freq='h', tz='UTC')
    closes = pd.Series(np.linspace(100.0, 159.0, 60), index=idx)
    # ts before the fetched window: must be unresolved, NOT aliased to bar 0
    assert shadow._realized(closes, start - dt.timedelta(hours=1), 24) is None
    assert shadow._realized(closes, start - dt.timedelta(days=20), 24) is None
    # ts exactly at the window start IS the true anchor
    r = shadow._realized(closes, start, 24)
    expected = (closes.iloc[24] - closes.iloc[0]) / closes.iloc[0] * 100
    assert r == pytest.approx(float(expected))
    # in-window ts anchors at the right bar
    r2 = shadow._realized(closes, start + dt.timedelta(hours=10), 5)
    expected2 = (closes.iloc[15] - closes.iloc[10]) / closes.iloc[10] * 100
    assert r2 == pytest.approx(float(expected2))


def test_evaluate_excludes_rows_predating_fetched_window(sandbox, monkeypatch):
    cm = _mk_manifest(sandbox)
    win_start = dt.datetime(2026, 5, 20, tzinfo=dt.timezone.utc)
    closes = _fake_closes(win_start, 200, seed=7)
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    old_ts = win_start - dt.timedelta(days=10)   # row older than the window
    _append_rows(sandbox, '', [{'ts': old_ts.isoformat(), 'sym': 'BTC/USD',
                                'champ': 0.1, 'chall': 0.2,
                                'fb_champ': 24, 'fb_chall': 24, 'cm': cm}])
    report = shadow.evaluate_shadow('', api=object())
    assert report is not None
    assert report['n'] == 0   # unresolved — no fabricated bar-0 return


def test_fetch_closes_requests_shadow_horizon_window(monkeypatch):
    import market_data
    seen = {}
    monkeypatch.setattr(
        market_data, 'fetch_bars_alpaca',
        lambda api, symbol, limit=250, **k: seen.__setitem__('crypto', limit))
    monkeypatch.setattr(
        market_data, 'fetch_stock_bars_alpaca',
        lambda api, symbol, limit=320, **k: seen.__setitem__('stock', limit))
    assert shadow._fetch_closes(object(), 'BTC/USD', 'crypto') is None
    assert shadow._fetch_closes(object(), 'AAPL', 'stock') is None
    # crypto: 24/7 hourly bars must span MAX_SHADOW_DAYS + the 24h horizon
    assert seen['crypto'] >= 24 * shadow.MAX_SHADOW_DAYS + 24
    # stocks: the 45d API start binds; the limit must not truncate below it
    assert seen['stock'] >= 500


# --- P1: challenger booster cache eviction / healing ---

def test_reload_evicts_challenger_booster_caches(sandbox, monkeypatch):
    fake = _install_fake_predict_now(
        monkeypatch,
        lgb={'challenger': 'stale-booster', '': 'champ-booster'},
        q10={'challenger': ('stale-q10', 0.1)})
    _mk_manifest(sandbox)
    loop = _LoopStub()
    shadow.maybe_log_shadow(loop, {'BTC/USD': 0.3}, benchmark=None)
    # Stale generation evicted so the blend reloads from disk
    assert 'challenger' not in fake._lgb_models
    assert 'challenger' not in fake._q10_models
    # Champion key untouched (base_loop owns it)
    assert fake._lgb_models[''] == 'champ-booster'
    assert shadow.shadow_log_file('').exists()   # row still logged


def test_tick_heals_cached_none_booster_after_file_lands(sandbox, monkeypatch):
    fake = _install_fake_predict_now(
        monkeypatch, lgb={'challenger': None}, q10={'challenger': None})
    cm = _mk_manifest(sandbox)
    # Booster landed AFTER the manifest (hypersearch writes manifest first)
    (sandbox / 'challenger_lgb_model.txt').write_text('booster-file')
    (sandbox / 'challenger_lgb_q10.txt').write_text('q10-file')  # no meta json
    loop = _LoopStub()
    # Same manifest mtime -> no reload; only the healing check runs
    loop._shadow_stack = (cm, 'model', 'scaler', {'forward_bars': 12}, ['f'])
    shadow.maybe_log_shadow(loop, {'BTC/USD': 0.3}, benchmark=None)
    # Cached None evicted now that the file exists (next call reloads it)
    assert 'challenger' not in fake._lgb_models
    # q10 needs BOTH txt and meta json; meta still missing -> stays None
    assert fake._q10_models.get('challenger', 'gone') is None


# --- shadow-log pruning ---

def test_prune_stale_rows_drops_replaced_generation(sandbox):
    path = shadow.shadow_log_file('')
    with open(path, 'w') as f:
        f.write(json.dumps({'sym': 'A', 'cm': 100}) + '\n')
        f.write(json.dumps({'sym': 'B', 'cm': 200}) + '\n')
        f.write('not-json\n')
        f.write(json.dumps({'sym': 'C', 'cm': 200}) + '\n')
    shadow._prune_stale_rows('', 200)
    rows = [json.loads(line) for line in open(path)]
    assert [r['sym'] for r in rows] == ['B', 'C']
    before = path.read_text()
    shadow._prune_stale_rows('', 200)     # idempotent: no rewrite churn
    assert path.read_text() == before


def test_reload_prunes_stale_rows_from_log(sandbox, monkeypatch):
    _install_fake_predict_now(monkeypatch)
    cm = _mk_manifest(sandbox)
    path = shadow.shadow_log_file('')
    with open(path, 'w') as f:
        f.write(json.dumps({'sym': 'OLD', 'cm': cm - 999}) + '\n')
    shadow.maybe_log_shadow(_LoopStub(), {'BTC/USD': 0.3}, benchmark=None)
    rows = [json.loads(line) for line in open(path)]
    assert all(r['cm'] == cm for r in rows)          # stale generation gone
    assert any(r['sym'] == 'BTC/USD' for r in rows)  # new row appended


# --- promotion failure-path hardening ---

def test_promote_aborts_before_mutation_on_missing_manifest(sandbox):
    for suffix in shadow._ARTIFACT_SUFFIXES:
        (sandbox / f'challenger_{suffix}').write_text(f'challenger-{suffix}')
    # no challenger manifest at all
    (sandbox / 'model_v2.pth').write_text('champion')
    (sandbox / 'model_v2.manifest.json').write_text(json.dumps({'old': True}))
    assert shadow.promote_challenger('') is False
    assert (sandbox / 'model_v2.pth').read_text() == 'champion'   # untouched
    assert json.loads(
        (sandbox / 'model_v2.manifest.json').read_text()) == {'old': True}
    assert not (sandbox / 'model_v2.pth.prev').exists()  # aborted pre-backup


def test_promote_aborts_before_mutation_on_corrupt_manifest(sandbox):
    for suffix in shadow._ARTIFACT_SUFFIXES:
        (sandbox / f'challenger_{suffix}').write_text(f'challenger-{suffix}')
    shadow.challenger_manifest('').write_text('{not-json')
    (sandbox / 'model_v2.pth').write_text('champion')
    assert shadow.promote_challenger('') is False
    assert (sandbox / 'model_v2.pth').read_text() == 'champion'
    assert not (sandbox / 'model_v2.pth.prev').exists()


def test_promote_config_rewrite_failure_keeps_challenger_for_retry(sandbox):
    for suffix in shadow._ARTIFACT_SUFFIXES:
        (sandbox / f'challenger_{suffix}').write_text(f'challenger-{suffix}')
    shadow.challenger_manifest('').write_text(json.dumps({'saved_at': 'x'}))
    for suffix in shadow._ARTIFACT_SUFFIXES:
        (sandbox / suffix).write_text(f'champion-{suffix}')
    (sandbox / 'model_v2.manifest.json').write_text(json.dumps({'old': True}))
    (sandbox / 'meta_model.txt').write_text('meta')
    # config_v2.pkl is garbage text: on the Jetson joblib.load raises, on
    # the Mac `import joblib` raises — both must hit the same abort path.
    assert shadow.promote_challenger('') is False
    # Champion manifest never written -> bots keep the old champion
    assert json.loads(
        (sandbox / 'model_v2.manifest.json').read_text()) == {'old': True}
    # Champion meta not deleted (deletion is after the rewrite step)
    assert (sandbox / 'meta_model.txt').read_text() == 'meta'
    # Challenger retained so the next daily eval retries and heals
    assert shadow.challenger_manifest('').exists()
    assert (sandbox / 'challenger_lgb_model.txt').exists()
    # .prev backups exist -> partially-copied state is recoverable
    assert (sandbox / 'model_v2.pth.prev').read_text() == 'champion-model_v2.pth'


# --- report hygiene ---

def test_early_report_fb_max_reflects_row_horizons(sandbox, monkeypatch):
    cm = _mk_manifest(sandbox)
    start = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    closes = _fake_closes(start, 3, seed=6)   # too short to resolve anything
    monkeypatch.setattr(shadow, '_fetch_closes', lambda api, s, a: closes)
    _append_rows(sandbox, '', [{'ts': start.isoformat(), 'sym': 'BTC/USD',
                                'champ': 0.1, 'chall': 0.1,
                                'fb_champ': 4, 'fb_chall': 6, 'cm': cm}])
    report = shadow.evaluate_shadow('', api=object())
    assert report['n'] == 0
    assert report['fb_max'] == 6              # from the rows, not a fixed 24


def test_shadow_logging_and_doc_hygiene():
    import inspect
    import re
    src_eval = inspect.getsource(shadow.evaluate_and_maybe_promote)
    # daily status routes via logger — no bare print() calls (identifier
    # tails like _manifest_fingerprint( must not trip this check)
    assert not re.search(r'(?<![\w.])print\(', src_eval)
    assert 'logger.info' in src_eval
    src_log = inspect.getsource(shadow.maybe_log_shadow)
    assert 'failed to append' in src_log      # append failures leave evidence
    assert 'MIN_OBS' in shadow.__doc__        # promote branches require n>=200


# =====================================================================
# ic_diagnostic.py
# =====================================================================

def test_promote_set_t_stat_uses_finite_pair_count():
    table = {
        'SPARSE': {'ic': 0.18, 'n': 400, 'n_finite': 40,
                   'positive_consistency': 1.0},
        'DENSE': {'ic': 0.18, 'n': 400, 'n_finite': 400,
                  'positive_consistency': 1.0},
    }
    # 0.18*sqrt(399)=3.6 would promote both; honest 0.18*sqrt(39)=1.12
    # must hold SPARSE below min_t=2.
    assert promote_set(table) == ['DENSE']


def test_promote_set_backcompat_tables_without_n_finite():
    table = {'OLD': {'ic': 0.18, 'n': 400, 'positive_consistency': 1.0}}
    assert promote_set(table) == ['OLD']      # falls back to n


def test_negative_ic_never_promoted():
    # abs() used to make a significantly NEGATIVE IC clear min_t when a
    # negative min_ic was passed.
    table = {'NEG': {'ic': -0.15, 'n': 1000, 'n_finite': 1000,
                     'positive_consistency': 1.0}}
    assert promote_set(table, min_ic=-0.2) == []


def test_ic_by_name_counts_finite_pairs():
    rng = np.random.default_rng(5)
    pred = rng.normal(size=30)
    fwd = 0.8 * pred + rng.normal(0, 0.3, 30)
    rows = []
    for i, (p, f) in enumerate(zip(pred, fwd)):
        rows.append({'symbol': 'X',
                     'pred': None if i % 3 == 0 else float(p),
                     'fwd_return': float('nan') if i % 5 == 0 else float(f)})
    table = ic_by_name(rows, n_subperiods=4)
    n_finite = sum(1 for i in range(30) if i % 3 != 0 and i % 5 != 0)
    assert table['X']['n'] == 30
    assert table['X']['n_finite'] == n_finite


def test_consistency_counts_uncomputable_subperiods():
    # Finite pairs exist ONLY in the first quarter: one good quarter of
    # evidence must not read as consistency 1.0.
    rng = np.random.default_rng(6)
    rows = []
    for i in range(160):
        if i < 40:
            p = float(rng.normal())
            rows.append({'symbol': 'Q1', 'pred': p,
                         'fwd_return': 0.9 * p + float(rng.normal(0, 0.2))})
        else:
            rows.append({'symbol': 'Q1', 'pred': None, 'fwd_return': None})
    table = ic_by_name(rows, n_subperiods=4)
    m = table['Q1']
    assert m['ic'] is not None and m['ic'] > 0
    assert len(m['subperiod_ics']) == 1                  # only Q1 computable
    assert m['positive_consistency'] == pytest.approx(0.25)   # 1 of 4
    assert promote_set(table) == []           # held: 0.25 < min_consistency


def test_promote_docstring_sigma_corrected():
    assert '~1.4 sigma' in promote_set.__doc__
    assert '~1.3 sigma' not in promote_set.__doc__
