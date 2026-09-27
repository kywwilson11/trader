"""scripts/lgb_lag_ab.py (SCOUT-2 D1 LGB lag-subset A/B, measurement-only).

Synthetic data only. Pins: the lag->column map against model_lgb.
flatten_sequence; arm-B width; lags=all reproduces the full flatten
byte-for-byte; the weekly-block bootstrap returns a finite CI; the verdict
rule on hand-built inputs; --dry-run exits 0 and writes nothing; a tiny
end-to-end run writes one JSON with a verdict line.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('lightgbm')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import lgb_lag_ab as lab  # noqa: E402
from model_lgb import flatten_sequence  # noqa: E402

FEATS = ['Return_4h', 'Return_12h', 'Volatility_12h', 'ROC']


def test_lag_column_index_matches_flatten_sequence():
    rng = np.random.default_rng(1)
    seq, F = 9, 4
    names = [f'f{j}' for j in range(F)]
    win = rng.normal(size=(seq, F)).astype(np.float32)
    flat, flat_names = flatten_sequence(win, names)
    lags = (0, 1, 3, 8)
    idx = lab.lag_column_index(seq, F, lags)
    assert len(idx) == len(lags) * F
    assert np.all(np.diff(idx) > 0)  # oldest-first, features inner
    got = [flat_names[i] for i in idx]
    want = []
    for lag in sorted(lags, reverse=True):
        want += [n if lag == 0 else f'{n}_lag{lag}' for n in names]
    assert got == want
    rows = [seq - 1 - l for l in sorted(lags, reverse=True)]
    assert np.array_equal(flat[idx], win[rows].reshape(-1))


def test_lag_bounds_asserted():
    with pytest.raises(AssertionError):
        lab.lag_column_index(8, 3, (0, 8))
    with pytest.raises(AssertionError):
        lab.lag_column_index(8, 3, (-1, 2))


def test_arm_b_width_and_full_flatten_parity():
    pytest.importorskip('torch')
    hs = lab._hs()
    rng = np.random.default_rng(2)
    seq, F = 7, 3
    all_scaled = rng.normal(size=(200, F)).astype(np.float32)
    idx = np.arange(seq, 200, 3)
    full = hs.gather_windows(all_scaled, idx, np.arange(-seq, 0)).reshape(len(idx), -1)
    lags = (0, 2, 5)
    xb = lab._build(all_scaled, idx, lab.lag_offsets(seq, lags), chunk=17)
    assert xb.shape == (len(idx), len(lags) * F)
    assert np.array_equal(xb, full[:, lab.lag_column_index(seq, F, lags)])
    xa = lab._build(all_scaled, idx, lab.lag_offsets(seq, range(seq)), chunk=17)
    assert xa.tobytes() == full.tobytes()
    assert np.array_equal(lab.lag_offsets(seq, range(seq)), np.arange(-seq, 0))


def test_legacy_row_cap():
    assert lab.legacy_row_cap(24, 30, 120_000, 600_000_000) == 120_000
    assert lab.legacy_row_cap(40, 65, 120_000, 600_000_000) == 600_000_000 // (40 * 65 * 4)
    assert lab.legacy_row_cap(400, 400, 120_000, 600_000_000) == 20_000


def test_bootstrap_finite_and_point_is_exact_spearman():
    rng = np.random.default_rng(3)
    n = 3000
    y = rng.normal(size=n)
    pa = 0.2 * y + rng.normal(size=n)
    pb = 0.1 * y + rng.normal(size=n)
    blocks = np.arange(n) // 100
    r = lab.paired_block_bootstrap(pa, pb, y, blocks, n_boot=300, seed=0)
    assert np.isfinite(r['lo_one_sided']) and all(np.isfinite(r['ci90']))
    assert r['ci90'][0] <= r['lo_one_sided'] <= r['ci90'][1]
    assert r['point'] == pytest.approx(lab.spearman(pa, y) - lab.spearman(pb, y))
    assert r['n_blocks'] == 30
    scipy_stats = pytest.importorskip('scipy.stats')
    assert lab.spearman(pa, y) == pytest.approx(scipy_stats.spearmanr(pa, y)[0], abs=1e-12)


def _arm(point, lo, rss_ok=True, wall_ok=True):
    return {'point': point, 'lo': lo, 'rss_ok': rss_ok, 'wall_ok': wall_ok}


def test_verdict_rule():
    v = lab.verdict
    assert v({'C': _arm(0.002, -0.005), 'B': _arm(0.0, -0.004)})[0] == 'ADOPT-C'
    assert v({'C': _arm(0.002, -0.005, wall_ok=False), 'B': _arm(0.001, -0.004)})[0] == 'ADOPT-B'
    tok, why = v({'C': _arm(0.003, -0.015), 'B': _arm(0.001, -0.015)})
    assert tok == 'HOLD' and 'underpowered' in why
    assert v({'C': _arm(-0.004, -0.015), 'B': _arm(-0.001, -0.012)})[0] == 'HOLD'
    assert v({'C': _arm(0.01, -0.025), 'B': _arm(0.0, -0.005)})[0] == 'KILL'
    assert v({'C': _arm(0.01, -0.005, rss_ok=False)})[0] == 'KILL'
    both = {'C': _arm(-0.004, -0.015), 'sd_seed': 0.001}
    assert v(both, {'C': _arm(-0.002, -0.012), 'sd_seed': 0.001})[0] == 'KILL'
    noisy = {'C': _arm(0.001, -0.015), 'sd_seed': 0.01}
    assert v(noisy, {'C': _arm(0.002, -0.015), 'sd_seed': 0.01})[0] == 'KILL'
    assert v({'B': _arm(0.0, 0.0)})[0] == 'HOLD'  # C missing -> incomplete


def _write_store(path, n_per=2600, tickers=('AAA', 'BBB')):
    pd = pytest.importorskip('pandas')
    pytest.importorskip('pyarrow')
    rng = np.random.default_rng(4)
    frames = []
    for k, t in enumerate(tickers):
        idx = pd.date_range('2024-01-01', periods=n_per, freq='h')
        x = rng.normal(size=(n_per, len(FEATS))).astype(np.float64)
        df = pd.DataFrame(x, columns=FEATS, index=idx)
        df['Ticker'] = t
        df['Target_Return_24'] = 0.3 * x[:, 0] + rng.normal(size=n_per)
        df['Target_Return'] = df['Target_Return_24']
        frames.append(df)
    df = pd.concat(frames).sort_index()
    df.index.name = 'Datetime'
    df.to_parquet(path)


def test_dry_run_exits_zero_and_writes_nothing(tmp_path):
    pytest.importorskip('torch')
    store = tmp_path / 'training_data.parquet'
    _write_store(store)
    before = sorted(p.name for p in tmp_path.rglob('*'))
    rc = lab.main(['--dry-run', '--data', str(store), '--seq-len', '24',
                   '--out', str(tmp_path / 'out')])
    assert rc == 0
    assert sorted(p.name for p in tmp_path.rglob('*')) == before
    info = lab.dry_run_plan(lab.parse_args(['--data', str(store), '--seq-len', '24']),
                            lab.DEFAULT_LAGS)
    assert info['n_features'] == len(FEATS)
    assert info['arms']['B']['cols'] == len(lab.DEFAULT_LAGS) * len(FEATS)
    assert info['arms']['A']['cols'] == 24 * len(FEATS)


def test_tiny_end_to_end_run(tmp_path):
    pytest.importorskip('torch')
    import json
    store = tmp_path / 'training_data.parquet'
    _write_store(store, n_per=8000)  # scored half spans >= 3 calendar weeks
    out = tmp_path / 'res.json'
    rc = lab.main(['--data', str(store), '--seq-len', '6', '--fb', '24',
                   '--lags', '0,1,3', '--folds', '1', '--B', '50',
                   '--extra-seeds', '1', '--json', str(out)])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res['result']['status'] == 'ok', res['result']
    assert res['verdict_line'].startswith('VERDICT[crypto]: ')
    assert res['result']['verdict'] in ('ADOPT-C', 'ADOPT-B', 'HOLD', 'KILL')
    fold = res['result']['per_fold'][0]
    assert fold['arms']['B']['cols'] == 3 * len(FEATS)
    assert fold['arms']['A']['cols'] == 6 * len(FEATS)
    assert fold['arms']['C']['rows'] >= fold['arms']['B']['rows']
    assert np.isfinite(res['result']['arms']['C']['lo'])
