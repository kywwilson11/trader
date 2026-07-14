"""Leading/lagging indicator diagnostic + redundancy clusters (2026-07).

"Learn from lagging indicators, utilize leading ones" needs a measurement,
not a taxonomy argument. For every feature this module reports two numbers:

  * PREDICTIVE IC — Spearman rank-IC of feature_t against the FORWARD h-bar
    return, per horizon. Significant predictive IC = the feature carries
    timing information ("leading" in the only sense that pays).
  * REACTIVE coupling — Spearman of feature_t against the PAST h-bar return.
    High reactive + no predictive = the feature merely restates where price
    has been (a lagging state descriptor: it can condition a model, it cannot
    time an entry).

Classification per feature (FDR-controlled across features x horizons):
  leading           predictive significant, low reactive coupling
  momentum-carrier  predictive significant AND high reactive (it reacts to
                    the past and the past continues — e.g. genuine momentum)
  lagging-state     reactive only — keep at most as regime context
  inert             neither — a cut candidate

Statistical honesty (repo conventions): forward h-bar returns sampled every
bar overlap h-fold, so each per-ticker block contributes n/h effective
observations (the same convention as the DSR gates); tickers are computed
independently and pooled by inverse-variance (n_eff) weighting so one long
series cannot masquerade as panel-wide evidence; significance is
Benjamini-Hochberg FDR across all (feature, horizon) predictive tests.

Also reports redundancy: |Spearman| clusters (union-find at a threshold) and
EXACT duplicates. Known structural dupes this catches by construction:
ROC == Return_12h (same formula, indicators.py), STOCHd = SMA3(STOCHk),
MACDs = MACD - MACDh.

Pure numpy/pandas — unit-tested on synthetic panels on the dev Mac; run it
on the Jetson against the harvested training files:

    python indicator_leadlag.py --data crypto_training_data.parquet
    python indicator_leadlag.py --data stock_training_data.csv \
        --preset stationary --horizons 1,4,12,24,48 --json leadlag_stock.json

Measurement-only: reads a harvested panel, writes a report. Feature-set
changes it motivates ship via a preset + retrain (challenger -> shadow).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (1, 4, 12, 24, 48)
REACTIVE_STRONG = 0.10   # |Spearman| vs past returns above this = "reacts"
CLUSTER_THRESHOLD = 0.80
EXACT_DUP_THRESHOLD = 0.999


# --- core statistics ---

def spearman(x, y):
    """Spearman rank correlation on pairwise-complete observations.

    Returns (rho, n). NaN-safe; n < 8 returns (nan, n).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 8:
        return float('nan'), n
    xr = pd.Series(x[ok]).rank().values
    yr = pd.Series(y[ok]).rank().values
    xs = xr - xr.mean()
    ys = yr - yr.mean()
    denom = math.sqrt(float((xs * xs).sum() * (ys * ys).sum()))
    if denom <= 0:
        return float('nan'), n
    return float((xs * ys).sum() / denom), n


def _norm_p_two_sided(t):
    return math.erfc(abs(t) / math.sqrt(2.0))


def bh_fdr(pvals, q=0.10):
    """Benjamini-Hochberg: boolean rejections at FDR q. NaN p -> False."""
    p = np.asarray(pvals, dtype=float)
    out = np.zeros(len(p), dtype=bool)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if passed.any():
        k = int(np.max(np.where(passed)[0]))
        out[order[:k + 1]] = True
    return out


def pooled_ic(df, feature, target_by_ticker, horizon,
              ticker_col='Ticker'):
    """Inverse-variance pooled Spearman IC across per-ticker blocks.

    target_by_ticker: dict {ticker: aligned target ndarray}. Each block's
    IC gets weight n_eff = n/h (overlapping h-bar targets); pooled
    t = IC_pooled * sqrt(sum n_eff). Computing WITHIN ticker first means a
    level-shifted or higher-vol name cannot corrupt the panel rank geometry.
    """
    ics, weights = [], []
    if ticker_col in df.columns:
        groups = df.groupby(ticker_col, sort=False).indices.items()
    else:
        groups = [('_all', np.arange(len(df)))]
    fvals = df[feature].values
    for tkr, rows in groups:
        tgt = target_by_ticker.get(tkr)
        if tgt is None:
            continue
        rho, n = spearman(fvals[rows], tgt)
        if not math.isfinite(rho):
            continue
        n_eff = n / max(horizon, 1)
        if n_eff < 8:
            continue
        ics.append(rho)
        weights.append(n_eff)
    if not ics:
        return {'ic': float('nan'), 't': float('nan'), 'p': float('nan'),
                'n_eff': 0.0}
    w = np.asarray(weights, dtype=float)
    ic = float(np.average(np.asarray(ics), weights=w))
    n_eff_total = float(w.sum())
    t = ic * math.sqrt(n_eff_total)
    return {'ic': round(ic, 4), 't': round(t, 2),
            'p': _norm_p_two_sided(t), 'n_eff': round(n_eff_total, 1)}


# --- panel targets ---

def _targets_by_ticker(df, horizon, close_col='Close', ticker_col='Ticker',
                       side='forward'):
    """Per-ticker h-bar % returns aligned to each row.

    side='forward': return over (t, t+h]  — the PREDICTIVE target.
    side='past':    return over (t-h, t]  — the REACTIVE reference.
    Computed strictly within ticker (groupby), never across a concat seam.
    """
    out = {}
    if ticker_col in df.columns:
        grouped = df.groupby(ticker_col, sort=False)
        items = [(t, g[close_col]) for t, g in grouped]
    else:
        items = [('_all', df[close_col])]
    for tkr, close in items:
        past = close.pct_change(horizon) * 100.0
        if side == 'forward':
            out[tkr] = past.shift(-horizon).values
        else:
            out[tkr] = past.values
    return out


# --- redundancy ---

class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def redundancy_clusters(df, features, threshold=CLUSTER_THRESHOLD,
                        max_rows=50_000, seed=0):
    """|Spearman| clusters + exact duplicates over the feature block.

    Returns {'clusters': [sorted feature lists, size>1 only],
             'exact_duplicates': [(a, b, rho)], 'pairs': {(a,b): rho}}.
    Row-sampled for speed on big panels (rank geometry is stable in n).
    """
    feats = [f for f in features if f in df.columns]
    sub = df[feats]
    if len(sub) > max_rows:
        rng = np.random.default_rng(seed)
        sub = sub.iloc[np.sort(rng.choice(len(sub), max_rows, replace=False))]
    ranked = sub.rank()
    corr = ranked.corr().abs()
    uf = _UnionFind(feats)
    exact, pairs = [], {}
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            rho = float(corr.loc[a, b])
            if not math.isfinite(rho):
                continue
            if rho >= threshold:
                uf.union(a, b)
                pairs[(a, b)] = round(rho, 4)
            if rho >= EXACT_DUP_THRESHOLD:
                exact.append((a, b, round(rho, 5)))
    clusters = {}
    for f in feats:
        clusters.setdefault(uf.find(f), []).append(f)
    return {'clusters': sorted([sorted(v) for v in clusters.values()
                                if len(v) > 1]),
            'exact_duplicates': exact,
            'pairs': pairs}


# --- orchestration ---

def run_diagnostic(df, features, horizons=DEFAULT_HORIZONS, fdr_q=0.10,
                   close_col='Close', ticker_col='Ticker'):
    """Full lead/lag + redundancy report for one harvested panel."""
    feats = [f for f in features if f in df.columns]
    horizons = tuple(int(h) for h in horizons)
    fwd = {h: _targets_by_ticker(df, h, close_col, ticker_col, 'forward')
           for h in horizons}
    past = {h: _targets_by_ticker(df, h, close_col, ticker_col, 'past')
            for h in horizons}

    rows = []   # (feature, horizon, stats) for the FDR sweep
    per_feature = {}
    for f in feats:
        ic_by_h = {h: pooled_ic(df, f, fwd[h], h, ticker_col)
                   for h in horizons}
        rc_by_h = {h: pooled_ic(df, f, past[h], h, ticker_col)
                   for h in horizons}
        per_feature[f] = {'ic': ic_by_h, 'rc': rc_by_h}
        for h in horizons:
            rows.append((f, h, ic_by_h[h]))

    flags = bh_fdr([s['p'] for _, _, s in rows], q=fdr_q)
    sig = {}
    for (f, h, s), flag in zip(rows, flags):
        s['fdr_significant'] = bool(flag)
        if flag:
            sig.setdefault(f, []).append(h)

    report = {'horizons': list(horizons), 'fdr_q': fdr_q, 'features': {}}
    for f in feats:
        ic_by_h = per_feature[f]['ic']
        rc_by_h = per_feature[f]['rc']
        finite_ics = {h: s['ic'] for h, s in ic_by_h.items()
                      if math.isfinite(s['ic'])}
        best_h = (max(finite_ics, key=lambda h: abs(finite_ics[h]))
                  if finite_ics else None)
        react_max = max((abs(s['ic']) for s in rc_by_h.values()
                         if math.isfinite(s['ic'])), default=0.0)
        pred_sig = f in sig
        if pred_sig and react_max < REACTIVE_STRONG:
            cls = 'leading'
        elif pred_sig:
            cls = 'momentum-carrier'
        elif react_max >= REACTIVE_STRONG:
            cls = 'lagging-state'
        else:
            cls = 'inert'
        report['features'][f] = {
            'class': cls,
            'predictive_significant_at': sorted(sig.get(f, [])),
            'best_horizon': best_h,
            'best_ic': finite_ics.get(best_h) if best_h else None,
            'reactive_max_abs': round(react_max, 4),
            'ic_by_horizon': ic_by_h,
            'rc_by_horizon': rc_by_h,
        }

    red = redundancy_clusters(df, feats)
    report['clusters'] = red['clusters']
    report['exact_duplicates'] = red['exact_duplicates']
    counts = {}
    for f in feats:
        counts[report['features'][f]['class']] = \
            counts.get(report['features'][f]['class'], 0) + 1
    report['summary'] = counts
    return report


def format_report(report):
    lines = []
    order = {'leading': 0, 'momentum-carrier': 1, 'lagging-state': 2,
             'inert': 3}
    feats = sorted(report['features'].items(),
                   key=lambda kv: (order[kv[1]['class']],
                                   -abs(kv[1]['best_ic'] or 0.0)))
    lines.append(f"LEAD/LAG DIAGNOSTIC  horizons={report['horizons']}  "
                 f"FDR q={report['fdr_q']}   {report['summary']}")
    lines.append(f"{'feature':32s} {'class':16s} {'best IC':>8s} "
                 f"{'@h':>4s} {'react':>6s}  sig@")
    for f, d in feats:
        ic = f"{d['best_ic']:+.3f}" if d['best_ic'] is not None else '  --'
        lines.append(
            f"{f:32s} {d['class']:16s} {ic:>8s} "
            f"{str(d['best_horizon'] or '--'):>4s} "
            f"{d['reactive_max_abs']:6.3f}  "
            f"{d['predictive_significant_at'] or ''}")
    if report['exact_duplicates']:
        lines.append("EXACT DUPLICATES (carry one, cut the rest):")
        for a, b, rho in report['exact_duplicates']:
            lines.append(f"  {a} == {b}  (|rho|={rho})")
    if report['clusters']:
        lines.append(f"REDUNDANCY CLUSTERS (|Spearman| >= "
                     f"{CLUSTER_THRESHOLD}) — keep the best-IC member:")
        for c in report['clusters']:
            lines.append("  " + ", ".join(c))
    lines.append("Read: 'leading' = times entries; 'momentum-carrier' = "
                 "reacts AND predicts (usable); 'lagging-state' = regime "
                 "context only — never a timing trigger; 'inert' = cut "
                 "candidate at the next retrain.")
    return "\n".join(lines)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True,
                    help='harvested training csv/parquet (has Ticker, Close)')
    ap.add_argument('--preset', default='stationary',
                    help='indicator_config preset supplying the feature list')
    ap.add_argument('--features', default=None,
                    help='comma list overriding the preset')
    ap.add_argument('--horizons', default='1,4,12,24,48')
    ap.add_argument('--fdr-q', type=float, default=0.10)
    ap.add_argument('--json', help='write the full report dict here')
    args = ap.parse_args()

    if args.data.endswith('.parquet'):
        frame = pd.read_parquet(args.data)
    else:
        frame = pd.read_csv(args.data, index_col=0, parse_dates=True)

    if args.features:
        feature_list = [f.strip() for f in args.features.split(',')]
    else:
        from indicator_config import get_preset_features
        feature_list = get_preset_features(args.preset) or [
            c for c in frame.columns
            if c not in ('Ticker', 'Open', 'High', 'Low', 'Close', 'Volume')
            and not c.startswith(('Target_', 'TB_'))]

    hs = tuple(int(x) for x in args.horizons.split(','))
    rep = run_diagnostic(frame, feature_list, horizons=hs, fdr_q=args.fdr_q)
    print(format_report(rep))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rep, f, indent=2, default=str)
        print(f"[indicator_leadlag] wrote {args.json}")
