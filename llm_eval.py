"""Measure whether the LLM gate adds signal BEYOND the ML model.

The system spends API budget and applies 0.65-1.5x sizing (plus forced
veto-sells) based on LLM conviction scores `s`. The question that justifies the
spend is NOT "does s correlate with returns?" — the prompt SHOWS the LLM the ML
`pred`, so a high raw correlation can be the LLM merely ECHOING the model
(redundant), and a low one can hide real orthogonal alpha. The question is
whether s adds value INCREMENTAL to pred.

This evaluator answers that, three ways (wave-8 #3):

  1. PRIMARY — partial Spearman rho(s, realized | pred): rank-correlation of s
     with realized return after partialling OUT the ML pred. The gap between the
     RAW and PARTIAL rho is the echo diagnostic.
  2. The 2x2 {LLM bull/bear} x {ML bull/bear} agree/disagree grid: the
     off-diagonal (disagreement) cells are the only place s can add orthogonal
     value; reported with per-cell n.
  3. SECONDARY/verdict-driver — an encompassing regression
        realized = a + b1*pred + b2*z_s
     with, when timestamps are present, Driscoll-Kraay (1998) standard errors
     clustered by t0-hour (PRIMARY — robust to both within-cluster
     cross-sectional dependence and the serial overlap of h-step returns,
     Bartlett lag = forward_bars-1 in cluster steps) and a Student-t p-value
     at G-1 cluster dof; without timestamps, the legacy Newey-West rows-HAC
     at lag = forward_bars-1. The old rows-HAC is also computed alongside the
     DK primary as 'legacy_b2' for one release (c26 D09). b2 significantly
     > 0 is the honest "the LLM earns its place" signal, but only when the
     panel has power: hard gates abstain below MIN_POWER_T0 distinct t0-hour
     clusters or MIN_EFFECTIVE_N span/horizon effective observations.

Realization steps the horizon in BARS over each symbol's bar index (matching
the policy_exits vertical barrier) — identical to wall-clock hours for
contiguous 24/7 hourly crypto, and the correct 24-bar horizon (not ~7 RTH
bars) for stocks.

The keep/kill verdict gates on the SIGNIFICANCE of b2 (with a sample-size floor),
not a bright-line rho — at typical journal volume a rho of 0.05 is noise.

Usage:
    python llm_eval.py --days 14
    python llm_eval.py --days 30 --asset crypto

The statistics live in compute_incremental_report(), which is pure
(numpy + scipy) and unit-tested on synthetic data without Alpaca.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

JOURNAL_DIR = BASE_DIR / "journals"
# Single-sourced from trading_utils so this evaluator can never drift from
# the threshold the live loop actually vetoes at (they were two independent
# 0.15 literals before — 2026-07 review).
try:
    from trading_utils import LLM_VETO_THRESHOLD as VETO_THRESHOLD
except Exception:  # standalone use without the trading stack
    VETO_THRESHOLD = 0.15
# Below this many realized samples a partial-rho / coefficient is indistinguishable
# from zero (null std ~0.15-0.19 at n=30-50); abstain rather than flip.
MIN_POWER_N = 60
# B07 (campaign 2026-08) panel-power floors for the Driscoll-Kraay (1998)
# clustered-HAC keep/kill verdict — below either, the p-value is geometry noise.
MIN_POWER_T0 = 120      # distinct t0-hour clusters required for a keep/kill verdict (B07)
MIN_EFFECTIVE_N = 20    # span_hours/forward_bars floor (B07 hard gate)


def _load_entries_by_action(days: int, action: str) -> list[dict]:
    """Shared journal-scan helper for both the 'llm_analysis' and
    'llm_advisor_v2' action rows. Newest-day-first iteration order is
    preserved (day 0 = today first). The `needle not in line` prefilter is a
    strict superset of the action check — a matching row always contains
    the quoted action string, so it can only ever produce false positives
    (harmless — they fall through to the e.get("action") == action check),
    never false negatives."""
    entries = []
    needle = f'"{action}"'
    today = datetime.now().date()
    for d in range(days + 1):
        day = today - timedelta(days=d)
        path = JOURNAL_DIR / f"{day.isoformat()}.jsonl"
        if not path.exists():
            continue
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line or needle not in line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("action") == action:
                        entries.append(e)
        except OSError as err:
            print(f"  [JOURNAL] skipping {path}: {err}")
            continue
    return entries


def _load_entries(days: int) -> list[dict]:
    return _load_entries_by_action(days, "llm_analysis")


def _load_advisor_entries(days: int) -> list[dict]:
    """Mirror of _load_entries filtering the advisor-v2 shadow rows
    (action=='llm_advisor_v2') journaled by llm_analyst.py when
    advisor_v2_enabled. Does not touch the 'llm_analysis' rows _load_entries
    reads — the existing run_eval verdict path is unaffected."""
    return _load_entries_by_action(days, "llm_advisor_v2")


def _bars_lookup(api, symbol: str, asset_type: str, start, end):
    """Hourly closes for [start, end] as a sorted (timestamps, closes)."""
    try:
        if asset_type == 'crypto':
            sym = symbol.replace('-', '/') if '-' in symbol else symbol
            bars = api.get_crypto_bars(sym, '1Hour', start=start.isoformat(),
                                       end=end.isoformat())
        else:
            bars = api.get_bars(symbol, '1Hour', start=start.isoformat(),
                                end=end.isoformat(), adjustment='all')
        ts, closes = [], []
        for b in bars:
            t = b.t
            if hasattr(t, 'to_pydatetime'):
                t = t.to_pydatetime()
            ts.append(t.timestamp())
            closes.append(float(b.c))
        order = np.argsort(ts)
        return np.array(ts)[order], np.array(closes)[order]
    except Exception as e:
        print(f"  [BARS] {symbol}: {e}")
        return np.array([]), np.array([])


def _realized_forward_return(ts_arr, closes, t0: float, horizon_bars: int):
    """Return (%return|None, elapsed_hours, entry_lag_hours, bars_spanned)
    from the first bar at/after t0 to horizon_bars BAR STEPS later. The
    horizon is journaled as forward_bars (a BAR count) and stepped here over
    the symbol's bar index — matching the policy_exits vertical barrier.
    Identical to the old wall-clock formula for contiguous 24/7 hourly
    crypto (exact searchsorted hit at i0+h); corrects the ~3.4x RTH stock
    mismatch (24 wall-clock hours span only ~7 RTH bars). elapsed_hours
    still reports the wall-clock stretch (c26 D09.b)."""
    if len(ts_arr) == 0:
        return None, None, None, None
    i0 = int(np.searchsorted(ts_arr, t0))
    if i0 >= len(ts_arr):
        return None, None, None, None
    i1 = i0 + int(horizon_bars)
    if i1 >= len(ts_arr):
        return None, None, None, None  # horizon not yet realized
    if not (np.isfinite(closes[i0]) and np.isfinite(closes[i1])) or closes[i0] <= 0:
        return None, None, None, None
    ret = (closes[i1] - closes[i0]) / closes[i0] * 100.0
    elapsed = (ts_arr[i1] - ts_arr[i0]) / 3600.0
    entry_lag = (ts_arr[i0] - t0) / 3600.0
    return ret, round(elapsed, 2), round(entry_lag, 2), int(i1 - i0)


def realize_scored_rows(rows: list[dict], api=None,
                        diag_out: list | None = None) -> list[tuple]:
    """Realize forward returns for a list of scored rows.

    rows: each a dict with keys {symbol, asset_type, t0, horizon, s, pred}
    (extra keys are ignored). Returns one (s, realized, pred, t0) tuple per
    input row, in the SAME order (realized is None when the horizon hasn't
    elapsed yet or bars are unavailable — callers filter, same convention as
    compute_incremental_report's own None-filter). run_eval() and
    advisor_report() both CALL this helper (rather than duplicating the
    group/lookup/realize loop inline) so they can never drift from its
    notion of "realized return".

    Groups rows by (symbol, asset_type) so each group needs only ONE
    _bars_lookup call spanning its full [min t0, max t0+horizon] range —
    shares the exact realization machinery run_eval() uses (same
    _bars_lookup / _realized_forward_return) so the offline prompt_ab.py
    harness can never drift from the live scorecard's notion of "realized
    return".

    diag_out: optional list; when provided, extended (not overwritten) with
    one (elapsed_hours, entry_lag_hours, bars_spanned) tuple per input row
    (None where realized is None) — same order as the returned list.

    api=None -> trading_utils.get_api() (Jetson/Alpaca-gated; not needed by
    callers that pass a fake/mock api, e.g. tests).
    """
    if not rows:
        return []
    if api is None:
        from trading_utils import get_api
        api = get_api()

    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r['symbol'], r.get('asset_type', 'crypto'))].append((i, r))

    out: list[tuple | None] = [None] * len(rows)
    diag: list[tuple | None] = [None] * len(rows)
    for (sym, asset), items in groups.items():
        t0s = [r['t0'] for _, r in items]
        max_h = max(int(r.get('horizon', 24) or 24) for _, r in items)
        start = datetime.fromtimestamp(min(t0s), tz=timezone.utc) - timedelta(hours=2)
        end = datetime.fromtimestamp(max(t0s), tz=timezone.utc) + timedelta(hours=max_h + 6)
        ts_arr, closes = _bars_lookup(api, sym, asset, start, end)
        for i, r in items:
            horizon = int(r.get('horizon', 24) or 24)
            realized, elapsed, lag, bars = _realized_forward_return(
                ts_arr, closes, r['t0'], horizon)
            out[i] = (r.get('s'), realized, r.get('pred'), r['t0'])
            diag[i] = (elapsed, lag, bars) if realized is not None else None
    if diag_out is not None:
        diag_out.extend(diag)
    return out


# --------------------------------------------------------------------------- #
# Pure statistics (no I/O, no Alpaca) — unit-tested in tests/test_llm_eval.py
# --------------------------------------------------------------------------- #

def _avg_rank(x):
    """Average ranks with proper tie handling (scipy.stats.rankdata)."""
    from scipy.stats import rankdata
    return rankdata(np.asarray(x, dtype=float), method='average')


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _ols_beta_resid(X, y):
    """OLS via lstsq. Returns (beta, residuals)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def _newey_west_se(X, resid, lag):
    """HAC (Newey-West, Bartlett kernel) standard errors for OLS beta.

    Var(beta) = (X'X)^-1 S (X'X)^-1 with the lag-weighted score sandwich
    S = sum u_t u_t' + sum_l w_l (G_l + G_l'), u_t = x_t e_t, w_l = 1 - l/(L+1).
    Rows are assumed time-ordered (the caller sorts by entry time); this corrects
    the serial overlap of h-step returns. Contemporaneous cross-sectional
    dependence (same bar, different symbols) is NOT corrected — a known limit.
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    u = X * resid[:, None]                      # n x k score contributions
    S = u.T @ u                                 # lag 0
    lag = int(max(0, min(lag, n - 1)))
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        G = u[l:].T @ u[:-l]                    # sum_t u_t u_{t-l}'
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    var = np.clip(np.diag(cov), 0.0, None)
    return np.sqrt(var)


def _driscoll_kraay_se(X, resid, cluster_ids, lag):
    """Driscoll-Kraay (1998) standard errors: sum the OLS scores within each
    time cluster, then apply the Bartlett-HAC sandwich over the CLUSTER
    sequence — robust to arbitrary cross-sectional dependence within a
    cluster AND to serial overlap across clusters (B07 binding parameters).

    cluster_ids: one integer time-cluster id per row; the caller passes rows
    already time-sorted, and `lag` is in CLUSTER steps. Nests _newey_west_se
    as the one-row-per-cluster degenerate case (modulo the G/(G-1)
    small-sample factor). Returns (se_vector, G).
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    u = X * resid[:, None]                      # n x k score contributions
    uniq = np.unique(cluster_ids)               # sorted
    G = len(uniq)
    h = np.zeros((G, k))                        # per-cluster summed scores
    np.add.at(h, np.searchsorted(uniq, cluster_ids), u)
    S = h.T @ h                                 # lag 0
    lag = int(max(0, min(lag, G - 1)))
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        Gl = h[l:].T @ h[:-l]
        S += w * (Gl + Gl.T)
    S *= G / max(G - 1, 1)                      # small-sample factor
    cov = XtX_inv @ S @ XtX_inv
    var = np.clip(np.diag(cov), 0.0, None)
    return np.sqrt(var), G


def _im_block_pvalue(X, y, cluster_ids, K=8):
    """Ibragimov-Muller block cross-check on b2 (B07, confidence medium —
    REPORT-ONLY, never drives the verdict): split the sorted unique clusters
    into K contiguous chunks, run per-chunk OLS, and t-test the collected
    b2 estimates against 0 with m-1 dof.

    Returns {'b2_im_p', 'b2_im_mean', 'im_blocks_used'} — None/None/0 when
    fewer than 4 usable blocks. A chunk is usable iff it has at least
    X.shape[1]+2 rows and both the pred and z_s columns vary within it.
    """
    out = {'b2_im_p': None, 'b2_im_mean': None, 'im_blocks_used': 0}
    uniq = np.unique(cluster_ids)
    betas = []
    for chunk in np.array_split(uniq, K):
        if len(chunk) == 0:
            continue
        mask = np.isin(cluster_ids, chunk)
        if int(mask.sum()) < X.shape[1] + 2:
            continue
        Xb, yb = X[mask], y[mask]
        if Xb[:, 1].std() <= 1e-12 or Xb[:, 2].std() <= 1e-12:
            continue
        beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        betas.append(float(beta[2]))
    m = len(betas)
    if m < 4:
        return out
    arr = np.array(betas, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    out['b2_im_mean'] = round(mean, 5)
    out['im_blocks_used'] = m
    if sd <= 1e-12:
        return out                              # degenerate spread -> p None
    from scipy.stats import t as _t
    tstat = mean / (sd / np.sqrt(m))
    out['b2_im_p'] = round(float(2.0 * _t.sf(abs(tstat), m - 1)), 4)
    return out


def partial_spearman(s, realized, pred):
    """Spearman rho(s, realized) after partialling out pred (rank-residualized).

    Returns (partial_rho, pred_degenerate). If pred has no rank variation we
    cannot control for it, so partial == raw and the flag is True.
    """
    rs, rr, rp = _avg_rank(s), _avg_rank(realized), _avg_rank(pred)
    if rp.std() < 1e-12:
        return _pearson(rs, rr), True
    A = np.column_stack([np.ones_like(rp), rp])
    _, es = _ols_beta_resid(A, rs)
    _, er = _ols_beta_resid(A, rr)
    return _pearson(es, er), False


def two_by_two_grid(s, realized, pred, s_thresh=0.5):
    """{LLM bull/bear} x {ML bull/bear} cells with n and mean realized return.

    Off-diagonal (disagreement) cells are the only place the LLM can add value
    orthogonal to the model; they are also the thinnest, so n is always reported.
    """
    s = np.asarray(s, float); realized = np.asarray(realized, float)
    pred = np.asarray(pred, float)
    llm_bull = s >= s_thresh
    ml_bull = pred > 0.0
    grid = {}
    for name, mask in (
        ('agree_bull', llm_bull & ml_bull),
        ('agree_bear', ~llm_bull & ~ml_bull),
        ('llm_bull_ml_bear', llm_bull & ~ml_bull),
        ('llm_bear_ml_bull', ~llm_bull & ml_bull),
    ):
        n = int(mask.sum())
        grid[name] = {
            'n': n,
            'avg_fwd_ret_pct': round(float(realized[mask].mean()), 3) if n else None,
        }
    return grid


def compute_incremental_report(samples, forward_bars: int = 24,
                               min_n: int = MIN_POWER_N, extra=None) -> dict:
    """Does s add signal incremental to pred? Pure; samples = (s, realized, pred[, t0]).

    samples may be 3-tuples (s, realized, pred) or 4-tuples with a trailing
    timestamp t0; when t0 is present rows are sorted by it so the Newey-West HAC
    is computed in time order.

    extra: optional (len(samples),) or (len(samples), k) array-like of
    additional covariates (e.g. journaled fng_value) — z-scored per column
    (missing/non-finite values mean-imputed before z-scoring, i.e. a neutral
    0 contribution) and appended to the encompassing regression's X. Reused
    unchanged for the advisor-v2 echo-gap question (p_up in the s slot).
    extra=None (the default) is numerically IDENTICAL to before this kwarg
    existed — no new columns, no behavior change.
    """
    idx_rows = [(i, r) for i, r in enumerate(samples)
               if r[0] is not None and r[1] is not None and r[2] is not None
               and np.isfinite(r[0]) and np.isfinite(r[1]) and np.isfinite(r[2])]
    n = len(idx_rows)
    rep = {'n': n, 'min_n': min_n}
    rep['n_input'] = len(samples)
    kept_idx = {i for i, _ in idx_rows}
    n_dropped = {'pred_none': 0, 's_none': 0, 'realized_none': 0, 'nonfinite': 0}
    for i, r in enumerate(samples):
        if i in kept_idx:
            continue
        if r[2] is None:
            n_dropped['pred_none'] += 1
        elif r[0] is None:
            n_dropped['s_none'] += 1
        elif r[1] is None:
            n_dropped['realized_none'] += 1
        else:
            n_dropped['nonfinite'] += 1
    rep['n_dropped'] = n_dropped
    if n < 5:
        rep['verdict'] = 'insufficient_power'
        rep['insufficient_power'] = True
        return rep

    has_ts = all(len(ir[1]) >= 4 for ir in idx_rows)
    rep['time_ordered'] = has_ts
    if has_ts:
        idx_rows = sorted(idx_rows,
                          key=lambda ir: (ir[1][3] if ir[1][3] is not None else 0.0))
    orig_idx = [ir[0] for ir in idx_rows]
    rows = [ir[1] for ir in idx_rows]
    s = np.array([r[0] for r in rows], float)
    realized = np.array([r[1] for r in rows], float)
    pred = np.array([r[2] for r in rows], float)
    s_degenerate = bool(s.std() < 1e-12)
    rep['s_degenerate'] = s_degenerate

    extra_arr = None
    if extra is not None:
        try:
            extra_full = np.asarray(extra, dtype=float)
            if extra_full.ndim == 1:
                extra_full = extra_full.reshape(-1, 1)
            extra_arr = extra_full[orig_idx]
        except Exception:
            extra_arr = None

    raw = _pearson(_avg_rank(s), _avg_rank(realized))
    partial, pred_degenerate = partial_spearman(s, realized, pred)
    rep['raw_spearman_s_vs_return'] = round(raw, 4) if raw is not None else None
    rep['partial_spearman_s_given_pred'] = round(partial, 4) if partial is not None else None
    if raw is not None and partial is not None:
        rep['echo_gap'] = round(raw - partial, 4)  # raw >> partial => s echoes pred
    rep['echo_gap_abs'] = (round(abs(raw) - abs(partial), 4)
                           if raw is not None and partial is not None else None)
    rep['n_s_exactly_half'] = int((s == 0.5).sum())
    rep['pred_degenerate'] = pred_degenerate
    rep['grid'] = two_by_two_grid(s, realized, pred)

    # Panel diagnostics (uses t0 when available) — overlapping-return / pseudo-
    # replication is the #1 way an HAC p-value can look significant while
    # being unreliable at this sample's geometry.
    cluster_ids = None
    if has_ts:
        t0s = np.array([r[3] for r in rows], dtype=float)
        n_t0 = len(np.unique(t0s))
        span_h = float((t0s.max() - t0s.min()) / 3600.0)
        rep['n_distinct_t0'] = int(n_t0)
        rep['rows_per_t0'] = round(n / n_t0, 2) if n_t0 else None
        rep['span_hours'] = round(span_h, 1)
        rep['effective_n_hint'] = round(span_h / forward_bars, 1) if forward_bars else None
        cluster_ids = np.floor_divide(t0s, 3600.0).astype(np.int64)   # t0-hour clusters (B07)
        rep['n_clusters'] = int(len(np.unique(cluster_ids)))
        rep['hac_lag_hours'] = int(max(0, forward_bars - 1))          # DK lag in cluster steps
    rep['hac_lag_rows'] = int(max(0, forward_bars - 1))
    pseudo = bool((has_ts and (rep.get('rows_per_t0') or 1) > 1.5) or
                  (rep.get('effective_n_hint') is not None and rep['effective_n_hint'] < 20))
    rep['pseudo_replication'] = pseudo

    # Encompassing regression realized = a + b1*pred + b2*z_s [+ b_extra*z_extra]
    # (HAC SEs).
    enc = None
    extra_cols_used = []
    if not pred_degenerate and s.std() > 1e-12 and n >= 6:
        z_s = (s - s.mean()) / s.std()
        cols = [np.ones(n), pred, z_s]
        if extra_arr is not None and extra_arr.shape[0] == n:
            for k in range(extra_arr.shape[1]):
                col = extra_arr[:, k]
                finite = np.isfinite(col)
                if finite.sum() < 2:
                    continue
                mean = col[finite].mean()
                col_filled = np.where(finite, col, mean)
                std = col_filled.std()
                if std <= 1e-12:
                    continue
                cols.append((col_filled - mean) / std)
                extra_cols_used.append(k)
        X = np.column_stack(cols)
        beta, resid = _ols_beta_resid(X, realized)
        b2 = float(beta[2])
        hac_lag = int(max(0, forward_bars - 1))
        if has_ts:
            # PRIMARY: Driscoll-Kraay clustered by t0-hour (c26 D09/B07).
            se, G = _driscoll_kraay_se(X, resid, cluster_ids, lag=hac_lag)
            se2 = float(se[2])
            dof = max(1, G - 1)
        else:
            se = _newey_west_se(X, resid, lag=hac_lag)
            se2 = float(se[2])
            dof = max(1, n - X.shape[1])
        if se2 > 1e-12:
            from scipy.stats import t as _t
            tstat = b2 / se2
            pval = float(2.0 * _t.sf(abs(tstat), dof))
        else:
            tstat, pval = None, None
        enc = {'b2_s': round(b2, 5), 'se_hac': round(se2, 5),
               't': round(tstat, 3) if tstat is not None else None,
               'p_value': round(pval, 4) if pval is not None else None,
               'b1_pred': round(float(beta[1]), 5), 'dof': dof}
        if has_ts:
            enc['estimator'] = 'driscoll_kraay'
            enc['g_clusters'] = G
            enc.update(_im_block_pvalue(X, realized, cluster_ids))
            # Legacy rows-HAC on the SAME X/resid — printed alongside for one
            # release (c26 D09); b2 identical, only SE/p differ.
            se_l = _newey_west_se(X, resid, lag=hac_lag)
            se2_l = float(se_l[2])
            dof_l = max(1, n - X.shape[1])
            if se2_l > 1e-12:
                from scipy.stats import t as _t
                t_l = b2 / se2_l
                p_l = float(2.0 * _t.sf(abs(t_l), dof_l))
            else:
                t_l, p_l = None, None
            rep['legacy_b2'] = {
                'b2_s': round(b2, 5), 'se_hac': round(se2_l, 5),
                't': round(t_l, 3) if t_l is not None else None,
                'p_value': round(p_l, 4) if p_l is not None else None,
                'dof': dof_l, 'hac_lag_rows': hac_lag,
                'estimator': 'newey_west_rows',
                'note': 'deprecated rows-HAC — printed alongside for one '
                        'release (c26 D09)'}
        else:
            enc['estimator'] = 'newey_west_rows'
        if extra_cols_used:
            b_extra = []
            for j, k in enumerate(extra_cols_used):
                col_idx = 3 + j
                b_extra.append({'col': k, 'beta': round(float(beta[col_idx]), 5),
                               'se_hac': round(float(se[col_idx]), 5)})
            enc['b_extra'] = b_extra
    rep['encompassing'] = enc
    if extra is not None:
        rep['extra_used'] = bool(extra_cols_used)

    # Verdict — gate on b2 significance + a sample floor, never a bright-line rho.
    degenerate_or_insufficient = False
    if n < min_n:
        rep['verdict'] = (f'insufficient_power (n={n} < {min_n}); collect more '
                          'journals before trusting the incremental estimate')
        rep['insufficient_power'] = True
        degenerate_or_insufficient = True
    elif pred_degenerate:
        rep['verdict'] = 'pred_degenerate — ML pred has no variation; cannot assess increment'
        degenerate_or_insufficient = True
    elif s_degenerate:
        rep['verdict'] = ('llm_score_degenerate — s has no variation (check the '
                          'analyst parse: llm_analyst._parse_response defaults s '
                          'to 0.5); cannot assess increment')
        degenerate_or_insufficient = True
    elif has_ts and (rep.get('n_clusters', 0) < MIN_POWER_T0 or
                     (rep.get('effective_n_hint') is not None
                      and rep['effective_n_hint'] < MIN_EFFECTIVE_N)):
        # B07 hard power gate — replaces the old pseudo-replication warning
        # suffix: the report still shows the numbers; the verdict abstains.
        rep['verdict'] = (
            f'insufficient_power (n_clusters={rep.get("n_clusters")} < '
            f'{MIN_POWER_T0} or effective_n~{rep.get("effective_n_hint")} < '
            f'{MIN_EFFECTIVE_N}) — HAC/DK p-value unreliable at this panel '
            'geometry; collect more journals before trusting the spend verdict')
        rep['insufficient_power'] = True
        degenerate_or_insufficient = True
    elif enc and enc['p_value'] is not None and enc['p_value'] < 0.05 and enc['b2_s'] > 0:
        rep['verdict'] = ('LLM adds signal INCREMENTAL to the ML pred '
                          f'(b2={enc["b2_s"]}%/SD, p={enc["p_value"]}) — keep it')
    elif raw is not None and partial is not None and abs(raw) > 0.05 and abs(partial) < 0.5 * abs(raw):
        rep['verdict'] = ('LLM largely ECHOES the ML pred (raw rho '
                          f'{raw:.3f} -> partial {partial:.3f}) — little independent value')
    elif enc and enc['p_value'] is not None and enc['p_value'] < 0.05 and enc['b2_s'] < 0:
        rep['verdict'] = ('LLM is ANTI-predictive incremental to the ML pred '
                          f'(b2={enc["b2_s"]}%/SD, p={enc["p_value"]}) — the gate '
                          'may be hurting; investigate before trusting it')
    else:
        rep['verdict'] = ('no measurable incremental value beyond the ML pred at '
                          'this sample — candidate to disable and save the spend')
    return rep


def compute_calibration_report(p_up, realized, conviction=None, abstain=None,
                               n_bins: int = 5, min_n: int = MIN_POWER_N) -> dict:
    """Is the advisor's stated p_up actually calibrated? Pure numpy/scipy —
    Mac-testable on synthetic data, no Alpaca/journals.

    Reliability bins (fixed edges [0,.2,.4,.6,.8,1.001]) of stated p_up vs
    empirical up-frequency, Brier score + skill vs the base rate, a
    calibration slope/intercept (outcome ~ a + b*p_up; overconfidence shows
    as slope < 1 — no frontier model hits nominal coverage in the
    literature, so this is expected, not a bug), plus two OPTIONAL blocks:

      conviction: per-level {n, avg_fwd_ret_pct, hit_rate} + Spearman(
        conviction, realized) — monotonicity check.
      abstain: the abstention-artifact check — abstain is only meaningful
        evidence if active-row IC exceeds abstain-row IC AND abstain
        hit-rate is ~coin-flip (an abstain that is itself predictive would
        mean the model is hiding a real view behind the abstain flag).

    verdict is a CALIBRATION description (well_calibrated / overconfident /
    underconfident / mixed) — never a keep/kill trade verdict; that
    authority stays with compute_incremental_report's b2 gate.
    """
    p_up = np.asarray(p_up, dtype=float)
    realized = np.asarray(realized, dtype=float)
    mask = np.isfinite(p_up) & np.isfinite(realized)
    # realized-finite-only mask — the abstention-artifact check needs
    # realized return, not a stated p_up, so its count-based stats use a
    # wider (realized-finite) denominator than the p_up-calibratable `mask`.
    mask_r = np.isfinite(realized)
    out_of_range = np.isfinite(p_up) & ((p_up < 0.0) | (p_up > 1.0))
    p = p_up[mask]
    r = realized[mask]
    n = int(mask.sum())
    rep = {'n': n, 'min_n': min_n}
    rep['n_total'] = int(len(p_up))
    rep['n_calibratable'] = n
    rep['n_p_up_out_of_range'] = int(out_of_range.sum())
    if n < 5:
        rep['verdict'] = (f'insufficient_power (n={n} < {min_n}); collect more '
                          'journals before trusting calibration')
        rep['insufficient_power'] = True
        return rep

    outcome = (r > 0).astype(float)

    edges = ([0.0, 0.2, 0.4, 0.6, 0.8, 1.001] if n_bins == 5
             else [i / n_bins for i in range(n_bins)] + [1.001])
    bins = []
    gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bmask = (p >= lo) & (p < hi)
        bn = int(bmask.sum())
        if bn == 0:
            bins.append({'lo': lo, 'hi': min(hi, 1.0), 'n': 0, 'mean_p': None,
                        'emp_freq': None, 'gap': None})
            continue
        mean_p = float(p[bmask].mean())
        emp_freq = float(outcome[bmask].mean())
        gap = emp_freq - mean_p
        gaps.append(gap)
        bins.append({'lo': lo, 'hi': min(hi, 1.0), 'n': bn,
                    'mean_p': round(mean_p, 4), 'emp_freq': round(emp_freq, 4),
                    'gap': round(gap, 4)})
    rep['bins'] = bins

    base_rate = float(outcome.mean())
    brier = float(np.mean((p - outcome) ** 2))
    brier_base = float(np.mean((base_rate - outcome) ** 2))
    rep['base_rate'] = round(base_rate, 4)
    rep['brier'] = round(brier, 5)
    rep['brier_base'] = round(brier_base, 5)
    rep['brier_skill'] = (round(1.0 - brier / brier_base, 4)
                          if brier_base >= 1e-12 else None)

    slope = intercept = None
    if p.std() > 1e-12 and n >= 3:
        X = np.column_stack([np.ones(n), p])
        beta, _resid = _ols_beta_resid(X, outcome)
        intercept, slope = float(beta[0]), float(beta[1])
    rep['calibration_intercept'] = round(intercept, 4) if intercept is not None else None
    rep['calibration_slope'] = round(slope, 4) if slope is not None else None

    mean_abs_gap = float(np.mean(np.abs(gaps))) if gaps else None
    rep['mean_abs_gap'] = round(mean_abs_gap, 4) if mean_abs_gap is not None else None

    if mean_abs_gap is None or slope is None:
        rep['verdict'] = 'insufficient_power'
    elif mean_abs_gap < 0.06 and 0.85 <= slope <= 1.15:
        rep['verdict'] = 'well_calibrated'
    elif slope < 0.85:
        rep['verdict'] = 'overconfident'
    elif slope > 1.15:
        rep['verdict'] = 'underconfident'
    else:
        rep['verdict'] = 'mixed'

    # Below the power floor the descriptive stats above are still useful
    # (bins/conviction/abstain are computed regardless) but the CALIBRATION
    # verdict itself is overridden to abstain from a label.
    if n < min_n:
        rep['verdict'] = (f'insufficient_power (n={n} < {min_n}); collect more '
                          'journals before trusting calibration')
        rep['insufficient_power'] = True

    if conviction is not None:
        conv = np.asarray(conviction, dtype=float)
        conv_paired = r_paired = None
        if conv.shape[0] == mask.shape[0]:
            conv_paired, r_paired = conv[mask_r], realized[mask_r]
        elif conv.shape[0] == n:
            conv_paired, r_paired = conv, r
        else:
            rep['conviction'] = {
                'error': f'length mismatch: conviction {conv.shape[0]} vs p_up {mask.shape[0]}'}
        if conv_paired is not None:
            levels = {}
            for lvl in range(1, 6):
                lvl_mask = conv_paired == lvl
                ln = int(lvl_mask.sum())
                if ln == 0:
                    continue
                levels[str(lvl)] = {
                    'n': ln,
                    'avg_fwd_ret_pct': round(float(r_paired[lvl_mask].mean()), 3),
                    'hit_rate': round(float((r_paired[lvl_mask] > 0).mean()), 3),
                }
            finite = np.isfinite(conv_paired)
            spearman = (_pearson(_avg_rank(conv_paired[finite]), _avg_rank(r_paired[finite]))
                       if finite.sum() >= 3 else None)
            rep['conviction'] = {
                'levels': levels,
                'spearman_conviction_vs_realized':
                    round(spearman, 4) if spearman is not None else None,
            }

    if abstain is not None:
        ab = np.asarray(abstain, dtype=bool)
        if ab.shape[0] == mask.shape[0]:
            # Full-length path: the count-based stats use the WIDER
            # realized-finite denominator (abstain doesn't need p_up), the
            # spearman diagnostics stay on the p_up-calibratable subset.
            ab_r = ab[mask_r]
            r_all = realized[mask_r]
            n_abstain = int(ab_r.sum())
            n_active = int((~ab_r).sum())
            block = {'n_abstain': n_abstain,
                    'abstain_rate': round(n_abstain / len(r_all), 4) if len(r_all) else None}
            block['hit_rate_abstain'] = (round(float((r_all[ab_r] > 0).mean()), 4)
                                         if n_abstain else None)
            block['mean_abs_ret_abstain'] = (round(float(np.abs(r_all[ab_r]).mean()), 4)
                                             if n_abstain else None)
            block['hit_rate_active'] = (round(float((r_all[~ab_r] > 0).mean()), 4)
                                        if n_active else None)
            block['mean_abs_ret_active'] = (round(float(np.abs(r_all[~ab_r]).mean()), 4)
                                            if n_active else None)
            ab_masked = ab[mask]
            n_active_p = int((~ab_masked).sum())
            n_abstain_p = int(ab_masked.sum())
            spearman_active = (_pearson(_avg_rank(p[~ab_masked]), _avg_rank(r[~ab_masked]))
                               if n_active_p >= 3 else None)
            spearman_abstain = (_pearson(_avg_rank(p[ab_masked]), _avg_rank(r[ab_masked]))
                                if n_abstain_p >= 3 else None)
            block['spearman_active_only'] = (round(spearman_active, 4)
                                             if spearman_active is not None else None)
            block['spearman_abstain_only'] = (round(spearman_abstain, 4)
                                              if spearman_abstain is not None else None)
            block['abstain_denominator'] = 'realized_finite'
            rep['abstain'] = block
        elif ab.shape[0] == n:
            n_abstain = int(ab.sum())
            n_active = int((~ab).sum())
            block = {'n_abstain': n_abstain,
                    'abstain_rate': round(n_abstain / n, 4) if n else None}
            block['hit_rate_abstain'] = (round(float((r[ab] > 0).mean()), 4)
                                         if n_abstain else None)
            block['mean_abs_ret_abstain'] = (round(float(np.abs(r[ab]).mean()), 4)
                                             if n_abstain else None)
            block['hit_rate_active'] = (round(float((r[~ab] > 0).mean()), 4)
                                        if n_active else None)
            block['mean_abs_ret_active'] = (round(float(np.abs(r[~ab]).mean()), 4)
                                            if n_active else None)
            spearman_active = (_pearson(_avg_rank(p[~ab]), _avg_rank(r[~ab]))
                               if n_active >= 3 else None)
            spearman_abstain = (_pearson(_avg_rank(p[ab]), _avg_rank(r[ab]))
                                if n_abstain >= 3 else None)
            block['spearman_active_only'] = (round(spearman_active, 4)
                                             if spearman_active is not None else None)
            block['spearman_abstain_only'] = (round(spearman_abstain, 4)
                                              if spearman_abstain is not None else None)
            block['abstain_denominator'] = 'p_up_calibratable'
            rep['abstain'] = block
        else:
            rep['abstain'] = {
                'error': f'length mismatch: abstain {ab.shape[0]} vs p_up {mask.shape[0]}'}

    return rep


def _write_report(path: Path, report: dict) -> None:
    """Atomic write-then-rename (shared by run_eval and advisor_report) so a
    concurrent reader (e.g. the GUI) never sees a partially written file."""
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    os.replace(tmp, path)


def _meta_block(days: int, asset_filter: str | None, horizons=None) -> dict:
    """Provenance block shared by every report/stub this module writes.
    horizons=None omits the forward_bars keys (pre-parse stub paths)."""
    meta = {'generated_at': datetime.now().astimezone().isoformat(),
            'days': days, 'asset_filter': asset_filter}
    if horizons is not None:
        meta['forward_bars_used'] = max(horizons) if horizons else 24
        meta['forward_bars_seen'] = sorted(set(horizons))
    meta['veto_threshold'] = VETO_THRESHOLD
    meta['min_power_n'] = MIN_POWER_N
    meta['min_power_t0'] = MIN_POWER_T0
    meta['min_effective_n'] = MIN_EFFECTIVE_N
    return meta


def _write_stub(path: Path, days: int, asset_filter: str | None, reason: str,
                horizons=None) -> None:
    """no_data stub for the empty-return paths — the report file always
    reflects the LAST run, even one that produced nothing."""
    _write_report(path, {'meta': _meta_block(days, asset_filter, horizons),
                         'n': 0, 'verdict': 'no_data', 'reason': reason})


def _realization_block(sample_diag: list) -> dict:
    """Realization diagnostics (elapsed/entry-lag/bars-spanned percentiles)
    from realize_scored_rows' diag_out — surfaces the RTH-vs-24/7
    bars_spanned gap, not a rejection. Shared by run_eval/advisor_report."""
    diag_vals = [d for d in sample_diag if d is not None]
    if not diag_vals:
        return {'elapsed_hours_p50': None, 'elapsed_hours_p90': None,
                'elapsed_hours_max': None, 'n_entry_lag_gt_2h': 0,
                'bars_spanned_p50': None}
    elapsed_arr = np.array([d[0] for d in diag_vals], dtype=float)
    lag_arr = np.array([d[1] for d in diag_vals], dtype=float)
    bars_arr = np.array([d[2] for d in diag_vals], dtype=float)
    return {
        'elapsed_hours_p50': round(float(np.percentile(elapsed_arr, 50)), 2),
        'elapsed_hours_p90': round(float(np.percentile(elapsed_arr, 90)), 2),
        'elapsed_hours_max': round(float(elapsed_arr.max()), 2),
        'n_entry_lag_gt_2h': int((lag_arr > 2).sum()),
        'bars_spanned_p50': round(float(np.percentile(bars_arr, 50)), 2),
    }


def _veto_counterfactual_block(realized_arr, veto_mask) -> dict:
    """n/avg/hit-rate view of the veto bucket (caller guarantees
    veto_mask.any()) — scored candidates, NOT simulated fills. Shared by
    run_eval/advisor_report."""
    return {
        'n': int(veto_mask.sum()),
        'avg_fwd_ret_pct': round(float(realized_arr[veto_mask].mean()), 3),
        'hit_rate': round(float((realized_arr[veto_mask] > 0).mean()), 3),
        'nonveto_avg_fwd_ret_pct': (round(float(realized_arr[~veto_mask].mean()), 3)
                                   if (~veto_mask).any() else None),
        'sum_fwd_ret_pct': round(float(realized_arr[veto_mask].sum()), 2),
    }


def _read_daily_cost():
    """Fail-soft read of llm_client.get_daily_cost() — never raises.
    Returns (spent_usd, limit_usd) or None when the ledger is unreadable."""
    try:
        from llm_client import get_daily_cost
        spent, limit = get_daily_cost()
        return float(spent), float(limit)
    except Exception:
        return None


def _spend_ledger_block(s_vals, realized, veto_mask, entries, days, cost) -> dict:
    """Pure spend-vs-benefit ledger (B07, c26 S1). Spend side: journaled
    per-entry cost_usd over the window + the llm_client daily cost ledger
    (cost=None -> cost_read_ok False, never an error). Benefit side: the
    realized forward return per trade of the DEPLOYED sizing tilt
    (llm_mult = 0.5 + s vs neutral 1.0x), in bps, plus the veto bucket's
    avoided return sum. Scored candidates, NOT simulated fills."""
    s_arr = np.asarray(s_vals, dtype=float)
    realized_arr = np.asarray(realized, dtype=float)
    tilt = (0.5 + s_arr) - 1.0
    # realized is in %, so *100 -> bps of fwd return per trade at deployed
    # sizing vs neutral 1.0x.
    llm_tilt_bps = (round(float(np.mean(tilt * realized_arr)) * 100.0, 2)
                    if len(s_arr) else None)
    veto_mask = np.asarray(veto_mask, dtype=bool)
    veto_avoided = (round(-float(realized_arr[veto_mask].sum()), 2)
                    if veto_mask.any() else 0.0)
    costs = [float(c) for c in (e.get('cost_usd') for e in entries)
             if isinstance(c, (int, float)) and not isinstance(c, bool)
             and np.isfinite(c)]
    daily_cost, daily_limit = cost if cost is not None else (None, None)
    return {
        'daily_cost_usd': daily_cost,
        'daily_cost_limit_usd': daily_limit,
        'cost_read_ok': cost is not None,
        'window_journaled_cost_usd': round(float(sum(costs)), 4),
        'n_entries_with_cost': len(costs),
        'days': days,
        'n_realized_trades': int(len(s_arr)),
        'llm_tilt_bps_per_trade': llm_tilt_bps,
        'veto_avoided_ret_pct_sum': veto_avoided,
        'sizing_formula': 'llm_mult = 0.5 + s',
        'note': 'scored candidates, NOT simulated fills; benefit in bps of '
                'fwd return/trade at deployed tilt; costs from journal rows '
                '+ llm_client daily ledger',
    }


def _asset_breakdown(samples, sample_assets, forward_bars) -> dict:
    """asset_mix / pooled_books (+ incremental_by_asset when pooled) — a
    pooled crypto+stock sample can hide book-level confounds in the pooled
    b2. Shared by run_eval/advisor_report."""
    mix = defaultdict(int)
    for a in sample_assets:
        mix[a] += 1
    out = {'asset_mix': dict(mix), 'pooled_books': len(mix) > 1}
    if out['pooled_books']:
        out['incremental_by_asset'] = {
            a: compute_incremental_report(
                [smp for smp, aa in zip(samples, sample_assets) if aa == a],
                forward_bars=forward_bars)
            for a in sorted(mix)}
    return out


def run_eval(days: int = 14, asset_filter: str | None = None, api=None) -> dict:
    entries = _load_entries(days)
    if asset_filter:
        entries = [e for e in entries if e.get('asset_type') == asset_filter]
    if not entries:
        print("No llm_analysis journal entries found — the bots journal one "
              "per LLM cycle; run again after some trading.")
        _write_stub(BASE_DIR / 'llm_eval_report.json', days, asset_filter,
                    'no_journal_entries')
        return {}

    if api is None:
        from trading_utils import get_api
        api = get_api()

    # Group needed bar ranges by (symbol, asset_type)
    needed = defaultdict(list)  # (symbol, asset) -> [(t0, horizon, s, pred)]
    horizons = []
    # D33-consumer (c26 S1): count silently-skipped s-null rows, and collapse
    # dedup_hit re-serves of an already-kept (prompt_sha256, symbol) so a
    # cached answer is not counted as a fresh observation. Entries iterate
    # newest-day-first (deterministic); rows a dedup hit cannot be attributed
    # to (no sha) are KEPT and counted separately.
    n_s_null = 0
    n_dedup_collapsed = 0
    n_dedup_unattributable = 0
    seen_serves = set()
    for e in entries:
        try:
            t0 = datetime.fromisoformat(e['ts']).timestamp()
        except (KeyError, ValueError):
            continue
        horizon = int(e.get('forward_bars', 24) or 24)
        horizons.append(horizon)
        sha = e.get('prompt_sha256')
        dh = bool(e.get('dedup_hit'))
        for sym, v in (e.get('scores') or {}).items():
            s = v.get('s')
            if s is None:
                n_s_null += 1
                continue
            if dh:
                if sha:
                    if (sha, sym) in seen_serves:
                        n_dedup_collapsed += 1
                        continue
                else:
                    n_dedup_unattributable += 1  # kept — cannot identify the serve
            if sha:
                seen_serves.add((sha, sym))
            needed[(sym, e.get('asset_type', 'crypto'))].append(
                (t0, horizon, float(s), v.get('pred')))
    forward_bars = max(horizons) if horizons else 24
    if len(set(horizons)) > 1:
        print(f"NOTE: multiple forward_bars values seen {sorted(set(horizons))} "
              f"in this window — using max={forward_bars}h as the realization "
              f"horizon; pass --asset to split crypto/stock if they run "
              f"different horizons.")

    # Group-major row dicts so realize_scored_rows (which shares realization
    # machinery with the offline prompt_ab.py harness) drives the samples —
    # sample ORDER stays byte-identical to the prior inline loop.
    rows_rsr = []
    for (sym, asset), rows in needed.items():
        for t0, horizon, s, pred in rows:
            rows_rsr.append({'symbol': sym, 'asset_type': asset, 't0': t0,
                             'horizon': horizon, 's': s, 'pred': pred})
    diag = []
    realized_tuples = realize_scored_rows(rows_rsr, api=api, diag_out=diag)
    samples, sample_assets, sample_diag = [], [], []
    n_missing_pred = 0
    for rt, row_d, dg in zip(realized_tuples, rows_rsr, diag):
        if rt[1] is None:
            continue
        samples.append(rt)
        sample_assets.append(row_d['asset_type'])
        sample_diag.append(dg)
        if rt[2] is None:
            n_missing_pred += 1

    n_rows_scored = len(rows_rsr)
    n_realized = len(samples)
    realized_by_key = defaultdict(list)
    for rt, row_d in zip(realized_tuples, rows_rsr):
        realized_by_key[(row_d['symbol'], row_d['asset_type'])].append(rt[1] is not None)
    symbols_all_unrealized = [key for key, flags in realized_by_key.items()
                              if not any(flags)]
    coverage = {
        'n_rows_scored': n_rows_scored,
        'n_realized': n_realized,
        'n_unrealized': n_rows_scored - n_realized,
        'n_missing_pred': n_missing_pred,
        'symbols_all_unrealized': symbols_all_unrealized,
        'n_s_null': n_s_null,
        'n_dedup_collapsed': n_dedup_collapsed,
        'n_dedup_unattributable': n_dedup_unattributable,
    }

    if not samples:
        print(f"No realized samples. {len(symbols_all_unrealized)}/{len(needed)} "
              f"symbols returned no usable bars; the rest may have unelapsed "
              f"horizons.")
        _write_stub(BASE_DIR / 'llm_eval_report.json', days, asset_filter,
                    'no_realized_samples', horizons=horizons)
        return {}

    s_vals = np.array([x[0] for x in samples])
    realized = np.array([x[1] for x in samples])

    buckets = [(0.0, VETO_THRESHOLD, 'VETO  '), (VETO_THRESHOLD, 0.35, 'bear  '),
               (0.35, 0.50, 'lean- '), (0.50, 0.65, 'lean+ '),
               (0.65, 0.85, 'bull  '), (0.85, 1.01, 'strong')]
    print(f"\n=== LLM GATE EVALUATION ({len(samples)} scored samples, "
          f"last {days}d) ===")
    print(f"{'bucket':<8}{'n':>5}{'avg fwd ret %':>15}{'hit rate':>10}")
    report = {'n': len(samples), 'buckets': {}}
    for lo, hi, label in buckets:
        mask = (s_vals >= lo) & (s_vals < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg = float(realized[mask].mean())
        hit = float((realized[mask] > 0).mean())
        report['buckets'][label.strip()] = {'n': n, 'avg_fwd_ret_pct': round(avg, 3),
                                            'hit_rate': round(hit, 3)}
        print(f"{label:<8}{n:>5}{avg:>15.3f}{hit:>10.2f}")

    # Incremental-over-pred: the question that justifies the LLM spend.
    inc = compute_incremental_report(samples, forward_bars=forward_bars)
    report['incremental'] = inc
    raw = inc.get('raw_spearman_s_vs_return')
    par = inc.get('partial_spearman_s_given_pred')
    print(f"\nRaw Spearman (s vs realized):        "
          f"{raw:+.3f}" if raw is not None else "\nRaw Spearman: n/a")
    print(f"Partial Spearman (s | ML pred):      "
          f"{par:+.3f}   <- the honest signal" if par is not None else "Partial Spearman: n/a")
    enc = inc.get('encompassing')
    if enc and enc.get('p_value') is not None:
        print(f"Encompassing b2 (return %/SD of s):  {enc['b2_s']:+.4f} "
              f"(HAC p={enc['p_value']:.3f}, n={inc['n']})")
    leg = inc.get('legacy_b2')
    if leg and leg.get('p_value') is not None:
        print(f"legacy_b2 (rows-HAC, deprecated): b2={leg['b2_s']:+.4f} "
              f"p={leg['p_value']:.3f} — superseded by Driscoll-Kraay above "
              f"(c26 D09)")
    g = inc.get('grid', {})
    if g:
        print("LLM-vs-ML disagreement cells (where s can add orthogonal value):")
        for cell in ('llm_bull_ml_bear', 'llm_bear_ml_bull'):
            c = g.get(cell, {})
            print(f"  {cell:<18} n={c.get('n', 0):>4}  avg fwd ret "
                  f"{c.get('avg_fwd_ret_pct')}")

    # Per-asset breakdown — a pooled crypto+stock sample can hide book-level
    # confounds in the pooled b2.
    report.update(_asset_breakdown(samples, sample_assets, forward_bars))
    if report['pooled_books']:
        print("NOTE: pooled across books — read incremental_by_asset; pooled "
              "b2 can be confounded by book-level mean differences.")

    veto_mask = s_vals < VETO_THRESHOLD
    if veto_mask.any():
        report['veto_counterfactual_pct'] = round(-float(realized[veto_mask].sum()), 2)
        vc = report['veto_counterfactual'] = _veto_counterfactual_block(
            realized, veto_mask)
        print(f"Veto counterfactual: {vc['n']} scored rows below s<{VETO_THRESHOLD} "
              f"averaged {vc['avg_fwd_ret_pct']:+.3f}% fwd return "
              f"(sum {vc['sum_fwd_ret_pct']:+.2f}%) — "
              f"scored candidates, NOT simulated fills; overlapping rows; no costs")

    # Spend-vs-benefit ledger (B07, c26 S1) — measurement-only; the daily
    # cost read is fail-soft and can never raise.
    ledger = report['spend_ledger'] = _spend_ledger_block(
        s_vals, realized, veto_mask, entries, days, _read_daily_cost())
    print(f"Spend: daily ${ledger['daily_cost_usd']}/"
          f"${ledger['daily_cost_limit_usd']} "
          f"(read_ok={ledger['cost_read_ok']}); window journaled "
          f"${ledger['window_journaled_cost_usd']} across "
          f"{ledger['n_entries_with_cost']} entries over {days}d")
    print(f"Benefit: LLM sizing tilt {ledger['llm_tilt_bps_per_trade']} "
          f"bps/trade (llm_mult=0.5+s vs neutral 1.0x); veto avoided "
          f"{ledger['veto_avoided_ret_pct_sum']}% summed fwd return")

    report['verdict'] = inc.get('verdict')
    print(f"\nVerdict: {inc.get('verdict')}")

    report['realization'] = _realization_block(sample_diag)

    report['coverage'] = coverage
    report['n_with_pred'] = report['n'] - n_missing_pred
    if n_missing_pred > 0.5 * len(samples):
        print(f"WARNING: {n_missing_pred}/{len(samples)} realized rows had no "
              f"ML pred — the incremental estimate is not measuring your "
              f"live book.")

    report['meta'] = _meta_block(days, asset_filter, horizons=horizons)

    out = BASE_DIR / 'llm_eval_report.json'
    _write_report(out, report)
    print(f"Report: {out}")
    return report


def _advisor_event_flag_stats(flag_lists, realized_arr):
    """Per-flag {n, avg_fwd_ret_pct, hit_rate}, cells with n<20 omitted —
    thin conditional cells are worse than no cell (wave-6 lesson)."""
    agg = defaultdict(list)
    for i, flags in enumerate(flag_lists):
        for f in (flags or []):
            agg[f].append(i)
    out = {}
    for f, idxs in agg.items():
        if len(idxs) < 20:
            continue
        rets = realized_arr[idxs]
        out[f] = {'n': len(idxs), 'avg_fwd_ret_pct': round(float(rets.mean()), 3),
                  'hit_rate': round(float((rets > 0).mean()), 3)}
    return out


def advisor_report(days: int = 14, asset_filter: str | None = None, api=None) -> dict:
    """Advisor-v2 verdict: echo-gap of p_up vs the ML pred (incremental),
    calibration (reliability/Brier/slope + conviction/abstain checks),
    per-prompt-version drift (prompt-instability detector), and
    per-event-flag conditional stats. Jetson-gated (Alpaca bars) — reuses
    _bars_lookup/_realized_forward_return via realize_scored_rows exactly
    like run_eval, so this can never silently drift from the live
    scorecard's notion of "realized return". Writes llm_advisor_report.json.
    """
    entries = _load_advisor_entries(days)
    if asset_filter:
        entries = [e for e in entries if e.get('asset_type') == asset_filter]
    if not entries:
        print("No llm_advisor_v2 journal entries found — enable "
              "advisor_v2_enabled and run again after some trading.")
        _write_stub(BASE_DIR / 'llm_advisor_report.json', days, asset_filter,
                    'no_journal_entries')
        return {}

    rows = []       # for realize_scored_rows: {symbol, asset_type, t0, horizon, s, pred}
    meta = []        # parallel: (p_up, conviction, abstain, fng_value, prompt_version,
                     #            computed_events, llm_event_flags, dedup_hit, model,
                     #            prompt_sha256, n_headlines)
    horizons = []
    # D33-consumer parity with run_eval (c26 S1): count s-null skips and
    # collapse dedup_hit re-serves of an already-kept (prompt_sha256, symbol).
    n_s_null = 0
    n_dedup_collapsed = 0
    n_dedup_unattributable = 0
    seen_serves = set()
    for e in entries:
        try:
            t0 = datetime.fromisoformat(e['ts']).timestamp()
        except (KeyError, ValueError):
            continue
        horizon = int(e.get('forward_bars', 24) or 24)
        horizons.append(horizon)
        sha = e.get('prompt_sha256')
        dh = bool(e.get('dedup_hit'))
        for sym, v in (e.get('scores') or {}).items():
            s = v.get('s')
            if s is None:
                n_s_null += 1
                continue
            if dh:
                if sha:
                    if (sha, sym) in seen_serves:
                        n_dedup_collapsed += 1
                        continue
                else:
                    n_dedup_unattributable += 1  # kept — cannot identify the serve
            if sha:
                seen_serves.add((sha, sym))
            rows.append({'symbol': sym, 'asset_type': e.get('asset_type', 'crypto'),
                        't0': t0, 'horizon': horizon, 's': float(s),
                        'pred': v.get('pred')})
            meta.append((v.get('p_up'), v.get('conviction'), v.get('abstain'),
                        e.get('fng_value'), e.get('prompt_version'),
                        v.get('computed_events') or [], v.get('event_flags') or [],
                        bool(e.get('dedup_hit')), e.get('model'), e.get('prompt_sha256'),
                        v.get('n_headlines')))
    forward_bars = max(horizons) if horizons else 24
    if len(set(horizons)) > 1:
        print(f"NOTE: multiple forward_bars values seen {sorted(set(horizons))} "
              f"in this window — using max={forward_bars}h as the realization "
              f"horizon; pass --asset to split crypto/stock if they run "
              f"different horizons.")

    if not rows:
        print("No advisor rows had a scored symbol — nothing to realize.")
        _write_stub(BASE_DIR / 'llm_advisor_report.json', days, asset_filter,
                    'no_scored_rows', horizons=horizons)
        return {}

    diag = []
    realized_tuples = realize_scored_rows(rows, api=api, diag_out=diag)  # (s, realized, pred, t0) per row, aligned

    samples, sample_assets, sample_diag = [], [], []
    s_list, p_up_list, conv_list, abstain_list, abstain_is_none_list, fng_list = (
        [], [], [], [], [], [])
    prompt_versions, computed_events_list, llm_flags_list = [], [], []
    dedup_list, model_list, sha_list, nheads_list = [], [], [], []
    n_p_up_fallback = 0
    abstain_missing = 0
    for rt, row_d, dg, (p_up, conviction, abstain, fng_value, prompt_version,
                       computed_events, llm_flags, dedup_hit, model, prompt_sha256,
                       n_headlines) in zip(realized_tuples, rows, diag, meta):
        if rt is None or rt[1] is None:
            continue
        s, realized, pred, t0 = rt
        signal = p_up if p_up is not None else s
        if p_up is None:
            n_p_up_fallback += 1
        samples.append((signal, realized, pred, t0))
        sample_assets.append(row_d['asset_type'])
        sample_diag.append(dg)
        s_list.append(s)
        p_up_list.append(p_up if p_up is not None else np.nan)
        conv_list.append(conviction if conviction is not None else np.nan)
        abstain_list.append(bool(abstain) if abstain is not None else False)
        is_abstain_missing = abstain is None
        abstain_is_none_list.append(is_abstain_missing)
        if is_abstain_missing:
            abstain_missing += 1
        fng_list.append(fng_value if fng_value is not None else np.nan)
        prompt_versions.append(prompt_version)
        computed_events_list.append(computed_events)
        llm_flags_list.append(llm_flags)
        dedup_list.append(bool(dedup_hit))
        model_list.append(model)
        sha_list.append(prompt_sha256)
        nheads_list.append(n_headlines)

    if not samples:
        print("No realized advisor samples yet (horizons may not have elapsed).")
        _write_stub(BASE_DIR / 'llm_advisor_report.json', days, asset_filter,
                    'no_realized_samples', horizons=horizons)
        return {}

    n = len(samples)
    fng_arr = np.array(fng_list, dtype=float)
    fng_present_frac = float(np.isfinite(fng_arr).mean()) if n else 0.0
    extra = fng_arr if fng_present_frac >= 0.8 else None

    # Echo-gap of p_up (falls back to s when p_up wasn't emitted) vs the ML
    # pred — reuses the same encompassing-regression machinery as run_eval.
    incremental = compute_incremental_report(samples, forward_bars=forward_bars,
                                             extra=extra)

    realized_arr = np.array([x[1] for x in samples], dtype=float)
    p_up_arr = np.array(p_up_list, dtype=float)
    conv_arr = np.array(conv_list, dtype=float)
    abstain_arr = np.array(abstain_list, dtype=bool)
    abstain_is_none_arr = np.array(abstain_is_none_list, dtype=bool)
    calibration = compute_calibration_report(p_up_arr, realized_arr,
                                             conviction=conv_arr,
                                             abstain=abstain_arr)

    # samples[i][0] is the SIGNAL (p_up when present, else s) — the raw
    # conviction score for the prompt-version/veto diagnostics below is the
    # separately-tracked s_list.
    s_arr = np.array(s_list, dtype=float)

    by_version = {}
    pv_arr = np.array(prompt_versions, dtype=object)
    for version in sorted({v for v in prompt_versions if v}):
        vmask = (pv_arr == version)
        vn = int(vmask.sum())
        if vn == 0:
            continue
        p_up_v = p_up_arr[vmask]
        finite_p = p_up_v[np.isfinite(p_up_v)]
        version_abstain_all_missing = bool(abstain_is_none_arr[vmask].all())
        by_version[version] = {
            'n': vn,
            'mean_s': round(float(s_arr[vmask].mean()), 4),
            'std_s': round(float(s_arr[vmask].std()), 4),
            'veto_rate': round(float((s_arr[vmask] < VETO_THRESHOLD).mean()), 4),
            'abstain_rate': (None if version_abstain_all_missing
                             else round(float(abstain_arr[vmask].mean()), 4)),
            'mean_p_up': round(float(finite_p.mean()), 4) if len(finite_p) else None,
            'avg_fwd_ret_pct': round(float(realized_arr[vmask].mean()), 3),
            'hit_rate': round(float((realized_arr[vmask] > 0).mean()), 3),
            'n_distinct_prompt_sha': len({sha_list[i] for i in np.where(vmask)[0]
                                         if sha_list[i]}),
        }

    by_model = {}
    model_arr = np.array(model_list, dtype=object)
    for m in sorted({mm for mm in model_list if mm}):
        mmask = (model_arr == m)
        mn = int(mmask.sum())
        if mn == 0:
            continue
        by_model[m] = {
            'n': mn,
            'mean_s': round(float(s_arr[mmask].mean()), 4),
            'veto_rate': round(float((s_arr[mmask] < VETO_THRESHOLD).mean()), 4),
            'abstain_rate': (None if bool(abstain_is_none_arr[mmask].all())
                            else round(float(abstain_arr[mmask].mean()), 4)),
            'avg_fwd_ret_pct': round(float(realized_arr[mmask].mean()), 3),
        }

    event_conditional = {
        'computed_events': _advisor_event_flag_stats(computed_events_list, realized_arr),
        'llm_event_flags': _advisor_event_flag_stats(llm_flags_list, realized_arr),
    }

    veto_mask = s_arr < VETO_THRESHOLD
    veto_counterfactual = (round(-float(realized_arr[veto_mask].sum()), 2)
                           if veto_mask.any() else None)

    finite_p_mask = np.isfinite(p_up_arr)
    s_vs_p_up = (_pearson(_avg_rank(s_arr[finite_p_mask]),
                          _avg_rank(p_up_arr[finite_p_mask]))
                if finite_p_mask.sum() >= 5 else None)

    report = {
        'n': n,
        'incremental': incremental,
        'calibration': calibration,
        'by_prompt_version': by_version,
        'event_conditional': event_conditional,
        'veto_counterfactual_pct': veto_counterfactual,
        's_vs_p_up_spearman': round(s_vs_p_up, 4) if s_vs_p_up is not None else None,
    }

    # Signal-source disclosure — p_up is the intended advisor-v2 signal; s
    # is a fallback used only for rows where the model didn't emit p_up.
    report['p_up_present_frac'] = round(1.0 - n_p_up_fallback / n, 4)
    report['n_p_up_fallback_to_s'] = n_p_up_fallback
    report['signal_source'] = ('p_up' if n_p_up_fallback == 0
                               else ('s' if n_p_up_fallback == n else 'mixed'))
    if 0 < n_p_up_fallback < n:
        report['incremental_p_up_only'] = compute_incremental_report(
            [smp for smp, pu in zip(samples, p_up_list) if np.isfinite(pu)],
            forward_bars=forward_bars)

    report['fng_present_frac'] = round(fng_present_frac, 4)
    report['fng_conditioned'] = extra is not None

    report['n_dedup_hit'] = sum(dedup_list)
    report['dedup_hit_frac'] = round(sum(dedup_list) / n, 4)
    unique_shas = {sh for sh in sha_list if sh}
    report['n_unique_llm_calls'] = len(unique_shas) if unique_shas else None
    report['by_model'] = by_model
    report['n_abstain_missing'] = abstain_missing

    nheads_buckets = {'zero': [], 'nonzero': [], 'unknown': []}
    for i, nh in enumerate(nheads_list):
        if nh is None:
            nheads_buckets['unknown'].append(i)
        elif nh == 0:
            nheads_buckets['zero'].append(i)
        else:
            nheads_buckets['nonzero'].append(i)
    by_headline_bucket = {}
    for bucket_name in ('zero', 'nonzero'):
        idxs = nheads_buckets[bucket_name]
        if idxs:
            rets = realized_arr[idxs]
            by_headline_bucket[bucket_name] = {
                'n': len(idxs), 'avg_fwd_ret_pct': round(float(rets.mean()), 3)}
        else:
            by_headline_bucket[bucket_name] = {'n': 0, 'avg_fwd_ret_pct': None}
    if nheads_buckets['unknown']:
        by_headline_bucket['unknown'] = {'n': len(nheads_buckets['unknown'])}
    report['by_headline_bucket'] = by_headline_bucket

    # Per-asset breakdown (same rationale as run_eval — a pooled sample can
    # hide book-level confounds in the pooled b2).
    report.update(_asset_breakdown(samples, sample_assets, forward_bars))
    if report['pooled_books']:
        print("NOTE: pooled across books — read incremental_by_asset; pooled "
              "b2 can be confounded by book-level mean differences.")

    vc = None
    if veto_mask.any():
        vc = report['veto_counterfactual'] = _veto_counterfactual_block(
            realized_arr, veto_mask)

    report['realization'] = _realization_block(sample_diag)
    report['coverage'] = {'n_rows_scored': len(rows), 'n_realized': n,
                          'n_unrealized': len(rows) - n,
                          'n_s_null': n_s_null,
                          'n_dedup_collapsed': n_dedup_collapsed,
                          'n_dedup_unattributable': n_dedup_unattributable}

    report['meta'] = _meta_block(days, asset_filter, horizons=horizons)

    print(f"\n=== LLM ADVISOR v2 EVALUATION ({n} scored samples, last {days}d) ===")
    print(f"Incremental verdict (signal={report['signal_source']} vs pred): "
          f"{incremental.get('verdict')}")
    print(f"Calibration verdict: {calibration.get('verdict')}")
    if s_vs_p_up is not None:
        note = ' <- p_up looks redundant with s' if abs(s_vs_p_up) > 0.9 else ''
        print(f"s vs p_up Spearman: {s_vs_p_up:+.3f}{note}")
    for version, stats in by_version.items():
        print(f"  [{version}] n={stats['n']} mean_s={stats['mean_s']:.3f} "
              f"veto_rate={stats['veto_rate']:.3f} "
              f"abstain_rate={stats['abstain_rate']}")
    if vc is not None:
        print(f"Veto counterfactual: {vc['n']} scored rows below s<{VETO_THRESHOLD} "
              f"averaged {vc['avg_fwd_ret_pct']:+.3f}% fwd return "
              f"(sum {vc['sum_fwd_ret_pct']:+.2f}%) — "
              f"scored candidates, NOT simulated fills; overlapping rows; no costs")
    if report['dedup_hit_frac'] > 0:
        print(f"Dedup cache hit rate: {report['dedup_hit_frac']:.1%} "
              f"({report['n_dedup_hit']}/{n} rows) share an evidence-hash "
              f"cache hit with an earlier cycle — those are NOT independent "
              f"LLM calls.")

    out = BASE_DIR / 'llm_advisor_report.json'
    _write_report(out, report)
    print(f"Report: {out}")
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Evaluate LLM gate vs realized returns')
    ap.add_argument('--days', type=int, default=14)
    ap.add_argument('--asset', choices=['crypto', 'stock'], default=None)
    ap.add_argument('--advisor', action='store_true',
                    help='Run the advisor-v2 calibration/incremental report '
                         '(llm_advisor_v2 shadow rows) instead of the base '
                         'llm_analysis eval')
    args = ap.parse_args()
    if args.advisor:
        advisor_report(args.days, args.asset)
    else:
        run_eval(args.days, args.asset)
