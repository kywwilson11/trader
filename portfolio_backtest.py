"""Cross-sectional A/B policy engine (wave-5 Tier1-2).

The conviction flagship — "make fewer, higher-conviction trades" via a
conviction-gated dynamic top-K admission — cannot be promoted on priors. The
shadow.py Diebold-Mariano gate tests FORECAST errors, which is the wrong
instrument for an ADMISSION-POLICY change (a verified category error in the
wave-5 kill-list). This module is the right instrument: a cross-sectional
portfolio backtest that replays two admission policies over the SAME ranked
panel and scores the difference net of turnover cost.

A "panel" is a time-ordered list of periods; each period is a list of candidate
dicts {symbol, signal, fwd_return, ...optional conviction fields}. A "policy"
is a pure callable(candidates_sorted_by_signal_desc) -> admitted subset. The
engine forms an equal-weight portfolio of the admitted names each period,
charges a round-trip cost on every newly-entered name (amortized once over its
hold), and reports return / Sharpe / turnover / admitted-k.

Everything is pure numpy — no model, no I/O, no look-ahead (a period only ever
sees its own candidates). Unit-tested on synthetic panels.
"""

import numpy as np

DEFAULT_PERIODS_PER_YEAR = 1638.0   # stock RTH hourly bars/yr (compute_sharpe)


# ---------------------------------------------------------------------------
# Admission policies (pure callables over one period's candidates)
# ---------------------------------------------------------------------------

def top_k(k):
    """Fixed top-K by signal — the incumbent admission policy."""
    def policy(cands):
        return cands[:max(int(k), 0)]
    return policy


def conviction_gated(k_max, signal_floor=None, meta_floor=None,
                     ratio_floor=None):
    """Dynamic K in [0, k_max]: admit a top-K name ONLY if it clears the
    conviction floors. K floats down when few names are convincing (the
    "fewer, higher-conviction" thesis) — the policy under test.

    signal_floor: minimum signal; meta_floor: minimum meta_p; ratio_floor:
    minimum pred/threshold ratio. Any floor left None is not applied.
    A non-finite (NaN) field fails any floor that is set (fail-closed).
    """
    def policy(cands):
        out = []
        for c in cands[:max(int(k_max), 0)]:
            if signal_floor is not None and not (c.get('signal', 0) >= signal_floor):
                continue
            if meta_floor is not None and not (c.get('meta_p', 1.0) >= meta_floor):
                continue
            if ratio_floor is not None and not (c.get('pred_thresh_ratio', 1.0) >= ratio_floor):
                continue
            out.append(c)
        return out
    return policy


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _sorted_desc(period):
    return sorted(period, key=lambda c: c.get('signal', 0.0), reverse=True)


def run_policy(panel, policy, cost_pct=0.0,
               periods_per_year=DEFAULT_PERIODS_PER_YEAR, with_series=False):
    """Replay one admission policy over the panel. Returns a metrics dict.

    Per period: admit = policy(sorted candidates); gross = equal-weight mean of
    admitted fwd_return; cost = cost_pct * (fraction of the book newly entered)
    — a round-trip charged once at entry, so a name held across periods is not
    re-charged. net = gross - cost. Cash (empty admit) earns 0.
    """
    prev = set()
    nets, grosses, ks, entries_frac = [], [], [], []
    n_entries_total = n_exits_total = 0
    for period in panel:
        admitted = policy(_sorted_desc(period))
        syms = [c['symbol'] for c in admitted]
        k = len(syms)
        ks.append(k)
        cur = set(syms)
        new = cur - prev
        gone = prev - cur
        n_entries_total += len(new)
        n_exits_total += len(gone)
        if k > 0:
            gross = float(np.mean([c['fwd_return'] for c in admitted]))
            ef = len(new) / k                       # fraction of book entered
        else:
            gross, ef = 0.0, 0.0                    # all cash
        entries_frac.append(ef)
        grosses.append(gross)
        nets.append(gross - cost_pct * ef)
        prev = cur

    nets = np.asarray(nets, dtype=float)
    n = len(nets)
    sharpe = 0.0
    if n > 1 and nets.std() > 1e-12:
        sharpe = float(nets.mean() / nets.std() * np.sqrt(periods_per_year))
    result = {
        'n_periods': n,
        'mean_admitted_k': round(float(np.mean(ks)), 3) if n else 0.0,
        'pct_periods_cash': round(float(np.mean(np.asarray(ks) == 0)), 4) if n else 0.0,
        'gross_total': round(float(np.sum(grosses)), 4),
        'net_total': round(float(nets.sum()), 4),
        'sharpe': round(sharpe, 4),
        'avg_entry_fraction': round(float(np.mean(entries_frac)), 4) if n else 0.0,
        'entries': n_entries_total,
        'exits': n_exits_total,
        'hit_rate': round(float(np.mean(nets > 0)), 4) if n else 0.0,
    }
    if with_series:
        result['_nets'] = nets
    return result


def compare(panel, policies, cost_pct=0.0,
            periods_per_year=DEFAULT_PERIODS_PER_YEAR, baseline=None):
    """Run several named policies over the same panel and report deltas.

    policies: {name: policy_callable}. baseline: the name to difference others
    against (default: the first key). Returns {'results': {name: metrics},
    'deltas': {name: {sharpe_delta, net_delta, k_delta}}, 'baseline': name}.
    """
    names = list(policies)
    base = baseline if baseline in policies else names[0]
    results = {name: run_policy(panel, policies[name], cost_pct, periods_per_year)
               for name in names}
    b = results[base]
    deltas = {}
    for name in names:
        r = results[name]
        deltas[name] = {
            'sharpe_delta': round(r['sharpe'] - b['sharpe'], 4),
            'net_delta': round(r['net_total'] - b['net_total'], 4),
            'k_delta': round(r['mean_admitted_k'] - b['mean_admitted_k'], 3),
        }
    return {'results': results, 'deltas': deltas, 'baseline': base}


def compare_deflated(panel, policies, cost_pct=0.0,
                     periods_per_year=DEFAULT_PERIODS_PER_YEAR, baseline=None,
                     fwd_bars=1):
    """compare() + a DEFLATED Sharpe and turnover per policy (wave-9 #4).

    The plain compare() ranks policies by RAW Sharpe with NO correction for how
    many policies were tried — a multiple-testing leak: the best of N noise
    policies looks good. dsr_from_trade_returns deflates by n_trials=len(policies),
    so the winner must beat the expected MAX Sharpe of that many tries. Turnover
    is promoted to first-class (a higher-Sharpe-but-higher-turnover policy is not
    free on a cost-bound book). THIS is the difference between certifying a real
    conviction edge and shipping a mined one.

    fwd_bars — REQUIRED to match the panel's forward-return horizon whenever
    the panel is sampled every bar (panel_from_frame on an hourly frame with a
    fb-bar fwd_return_col). Adjacent per-period nets then share fb-1 bars of
    the SAME move: they are not independent draws, and with fwd_bars=1 a
    zero-edge 24h-overlap panel scores DSR ~0.99 (2026-07 review, verified).
    The DSR null is widened by n_eff = n_periods / fwd_bars. Leave at 1 only
    for genuinely non-overlapping period returns.
    """
    from validation import dsr_from_trade_returns
    names = list(policies)
    base = baseline if baseline in policies else names[0]
    results = {}
    for name in names:
        m = run_policy(panel, policies[name], cost_pct, periods_per_year,
                       with_series=True)
        nets = m.pop('_nets')
        n_eff = (len(nets) / max(int(fwd_bars), 1)) if len(nets) else None
        d = dsr_from_trade_returns(nets, n_trials=len(names), n_eff=n_eff)
        m['dsr'] = round(d['dsr'], 4)
        m['n_eff'] = round(n_eff, 1) if n_eff else 0.0
        m['fwd_bars'] = int(fwd_bars)
        m['expected_max_sr'] = round(d['expected_max_sr'], 4)
        m['turnover'] = round((m['entries'] + m['exits']) / max(m['n_periods'], 1), 4)
        results[name] = m
    b = results[base]
    deltas = {name: {
        'sharpe_delta': round(results[name]['sharpe'] - b['sharpe'], 4),
        'net_delta': round(results[name]['net_total'] - b['net_total'], 4),
        'dsr': results[name]['dsr'],
        'turnover_delta': round(results[name]['turnover'] - b['turnover'], 4),
    } for name in names}
    return {'results': results, 'deltas': deltas, 'baseline': base}


def panel_from_frame(df, signal_col, fwd_return_col, ticker_col='Ticker',
                     extra_cols=None, signal_lag=0):
    """Build an A/B panel from a tidy frame (index=timestamp, one row per
    symbol-bar). Each timestamp becomes one period of candidate dicts.

    signal_lag>0 takes the signal from `signal_lag` bars earlier PER TICKER
    (strict PIT: the admission decides on info known before fwd_return realizes).
    Rows with a missing signal/fwd_return after lagging are dropped.
    Rows are stably sorted by timestamp before lagging, so an unsorted frame
    cannot leak a future bar's signal.
    """
    extra_cols = list(extra_cols or [])
    cols = [ticker_col, signal_col, fwd_return_col] + extra_cols
    work = df[cols].copy()
    if signal_lag:
        work = work.sort_index(kind='stable')
        work[signal_col] = work.groupby(ticker_col)[signal_col].shift(signal_lag)
    work = work.dropna(subset=[signal_col, fwd_return_col])
    panel = []
    for _ts, g in work.groupby(level=0):
        period = []
        for _, row in g.iterrows():
            c = {'symbol': row[ticker_col], 'signal': float(row[signal_col]),
                 'fwd_return': float(row[fwd_return_col])}
            for col in extra_cols:
                c[col] = row[col]
            period.append(c)
        if period:
            panel.append(period)
    return panel


# ---------------------------------------------------------------------------
# Tier sizing (edge-proportional vs equal-weight)
# ---------------------------------------------------------------------------

def edge_proportional_weights(signals, floor=0.0):
    """Weights proportional to (signal - floor)+, summing to 1. Falls back to
    equal-weight when no admitted name clears the floor. The growth-optimal book
    tracks edge (Kelly) rather than equal-weighting every admitted name."""
    s = np.asarray(signals, dtype=float)
    s = np.clip(np.where(np.isfinite(s), s, 0.0) - floor, 0.0, None)
    if len(s) == 0:
        return s
    if s.sum() <= 1e-12:
        return np.full(len(s), 1.0 / len(s))
    return s / s.sum()


def run_policy_weighted(panel, policy, weight_fn, cost_pct=0.0,
                        periods_per_year=DEFAULT_PERIODS_PER_YEAR):
    """run_policy but the admitted book is weighted by weight_fn([signals])
    instead of equal-weight — to test edge-proportional concentration. Cost is
    charged on the turnover of the WEIGHT vector (sum of positive weight changes)."""
    prev_w = {}
    nets = []
    for period in panel:
        admitted = policy(_sorted_desc(period))
        if not admitted:
            nets.append(0.0)
            prev_w = {}
            continue
        w = np.asarray(weight_fn([c.get('signal', 0.0) for c in admitted]), float)
        gross = float(np.sum(w * np.array([c['fwd_return'] for c in admitted])))
        cur_w = {c['symbol']: float(wi) for c, wi in zip(admitted, w)}
        syms = set(cur_w) | set(prev_w)
        turnover = sum(max(cur_w.get(s, 0.0) - prev_w.get(s, 0.0), 0.0) for s in syms)
        nets.append(gross - cost_pct * turnover)
        prev_w = cur_w
    nets = np.asarray(nets, float)
    sharpe = (float(nets.mean() / nets.std() * np.sqrt(periods_per_year))
              if len(nets) > 1 and nets.std() > 1e-12 else 0.0)
    return {'n_periods': len(nets), 'net_total': round(float(nets.sum()), 4),
            'sharpe': round(sharpe, 4)}
