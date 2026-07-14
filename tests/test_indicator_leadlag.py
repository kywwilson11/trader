"""indicator_leadlag tests — synthetic panels with planted lead/lag structure.

Also pins the structural finding that motivated the instrument: ROC (length
12) and Return_12h are the SAME formula in indicators.py, so every preset
ships one column twice — the diagnostic must catch it as an exact duplicate.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicator_leadlag import (
    spearman, bh_fdr, pooled_ic, redundancy_clusters, run_diagnostic,
    format_report,
)


def _panel(n=3000, tickers=('AAA', 'BBB'), phi=0.0, seed=7):
    """Synthetic panel: per-ticker AR(phi) hourly returns, concatenated."""
    rng = np.random.default_rng(seed)
    frames = []
    for k, t in enumerate(tickers):
        r = np.empty(n)
        r[0] = rng.normal(0, 0.01)
        eps = rng.normal(0, 0.01, n)
        for i in range(1, n):
            r[i] = phi * r[i - 1] + eps[i]
        close = 100.0 * (10 ** k) * np.exp(np.cumsum(r))  # level-shifted
        idx = pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC')
        frames.append(pd.DataFrame({'Ticker': t, 'Close': close}, index=idx))
    return pd.concat(frames), rng


def test_spearman_basics():
    x = np.arange(100, dtype=float)
    rho, n = spearman(x, x ** 3)          # monotone -> 1.0
    assert n == 100 and abs(rho - 1.0) < 1e-12
    rho, _ = spearman(x, -x)
    assert abs(rho + 1.0) < 1e-12
    rho, n = spearman(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    assert n < 8 and not np.isfinite(rho)


def test_bh_fdr_controls_noise():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200)
    p[:5] = 1e-8                           # five real signals
    flags = bh_fdr(p, q=0.10)
    assert flags[:5].all()
    assert flags.sum() <= 15               # few false rejections


def test_planted_leading_feature_classified():
    df, rng = _panel()
    # feature = (noisy) future 12-bar return, computed within ticker
    fwd12 = df.groupby('Ticker')['Close'].transform(
        lambda c: c.pct_change(12).shift(-12))
    df['lead_f'] = fwd12 + rng.normal(0, fwd12.std() * 1.5, len(df))
    df['noise_f'] = rng.normal(0, 1, len(df))
    rep = run_diagnostic(df, ['lead_f', 'noise_f'], horizons=(4, 12, 24))
    lf = rep['features']['lead_f']
    assert lf['class'] in ('leading', 'momentum-carrier')
    assert 12 in lf['predictive_significant_at']
    assert abs(lf['ic_by_horizon'][12]['ic']) > 0.15
    assert rep['features']['noise_f']['class'] == 'inert'


def test_planted_lagging_feature_classified():
    df, rng = _panel(phi=0.0)              # IID returns: past predicts nothing
    df['lag_f'] = df.groupby('Ticker')['Close'].transform(
        lambda c: (c.pct_change().rolling(24).mean()))
    rep = run_diagnostic(df, ['lag_f'], horizons=(4, 12, 24))
    d = rep['features']['lag_f']
    assert d['class'] == 'lagging-state'   # reacts to the past...
    assert d['reactive_max_abs'] > 0.3     # ...strongly
    assert not d['predictive_significant_at']   # ...but times nothing


def test_momentum_world_flags_carrier():
    df, _ = _panel(phi=0.25, seed=11)      # persistent returns
    df['mom_f'] = df.groupby('Ticker')['Close'].transform(
        lambda c: c.pct_change(12))
    rep = run_diagnostic(df, ['mom_f'], horizons=(1, 4, 12))
    d = rep['features']['mom_f']
    assert d['class'] == 'momentum-carrier'
    assert d['reactive_max_abs'] > 0.3
    assert d['predictive_significant_at']


def test_fdr_keeps_noise_panel_quiet():
    df, rng = _panel(seed=23)
    feats = []
    for i in range(25):
        df[f'n{i}'] = rng.normal(0, 1, len(df))
        feats.append(f'n{i}')
    rep = run_diagnostic(df, feats, horizons=(1, 12, 24))
    flagged = [f for f in feats
               if rep['features'][f]['predictive_significant_at']]
    assert len(flagged) <= 3


def test_redundancy_and_exact_duplicates():
    df, rng = _panel()
    base = rng.normal(0, 1, len(df))
    df['a'] = base
    df['b'] = base                                   # exact clone
    df['c'] = base + rng.normal(0, 0.3, len(df))     # near clone
    df['d'] = rng.normal(0, 1, len(df))              # independent
    red = redundancy_clusters(df, ['a', 'b', 'c', 'd'])
    assert any({'a', 'b'} <= set(cl) for cl in red['clusters'])
    assert any(sorted((x, y)) == ['a', 'b']
               for x, y, _ in red['exact_duplicates'])
    assert all('d' not in cl for cl in red['clusters'])


def test_pooled_ic_survives_level_shifted_tickers():
    # ticker B trades at 1000x ticker A's price: within-ticker ranking must
    # make the pooled IC identical in spirit to per-name ICs (no level bleed)
    df, rng = _panel()
    fwd = df.groupby('Ticker')['Close'].transform(
        lambda c: c.pct_change(4).shift(-4))
    df['f'] = fwd + rng.normal(0, fwd.std(), len(df))
    rep = run_diagnostic(df, ['f'], horizons=(4,))
    assert rep['features']['f']['ic_by_horizon'][4]['ic'] > 0.2


def test_roc_equals_return_12h_structural_dupe():
    # The finding that motivated the exact-duplicate detector: indicators.py
    # computes ROC(12) and Return_12h with the SAME formula.
    from indicators import compute_features
    rng = np.random.default_rng(3)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC')
    raw = pd.DataFrame({'Open': close, 'High': close * 1.001,
                        'Low': close * 0.999, 'Close': close,
                        'Volume': rng.uniform(1e5, 2e5, n)}, index=idx)
    df = compute_features(raw)
    both = df[['ROC', 'Return_12h']].dropna()
    assert np.allclose(both['ROC'], both['Return_12h'])
    red = redundancy_clusters(df.assign(Ticker='X'),
                              ['ROC', 'Return_12h'])
    assert red['exact_duplicates']


def test_format_report_readable():
    df, rng = _panel(seed=5)
    df['x'] = rng.normal(0, 1, len(df))
    rep = run_diagnostic(df, ['x'], horizons=(4, 12))
    text = format_report(rep)
    assert 'LEAD/LAG DIAGNOSTIC' in text and 'x' in text
