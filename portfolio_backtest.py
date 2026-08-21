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

Panel contract: each period lists each symbol AT MOST ONCE, and every
candidate carries a FINITE 'signal' — a violation raises rather than silently
mis-ranking or double-counting a name (cf. rank_gradient.py's identical
fail-loud choice on the same panel objects). Units: cost_pct must be in the
SAME units as fwd_return — the repo convention is PERCENT (cf.
fees.round_trip_cost_pct and scripts/rank_gradient_report.py's --cost-pct).
Look-ahead: there is no CROSS-PERIOD look-ahead (a policy only ever sees its
own period's candidates), but candidate dicts handed to a policy DO carry
fwd_return alongside signal — a policy callable must never read fwd_return.
The Sharpe default annualization is the STOCK RTH calendar; pass
periods_per_year=BARS_PER_YEAR['crypto'] for a crypto panel. The value
actually used is echoed back in every result dict as 'periods_per_year'.
"""

import numpy as np

# Keep in sync with the identical dicts in backtest.py, volatility.py and
# scripts/hypersearch_v2.py (pinned by tests/test_portfolio_backtest_v3.py).
BARS_PER_YEAR = {'crypto': 8760.0, 'stock': 1638.0}
DEFAULT_PERIODS_PER_YEAR = BARS_PER_YEAR['stock']   # stock RTH hourly bars/yr


# ---------------------------------------------------------------------------
# Admission policies (pure callables over one period's candidates)
# ---------------------------------------------------------------------------

def top_k(k):
    """Fixed top-K by signal — the incumbent admission policy."""
    def policy(cands):
        return cands[:max(int(k), 0)]
    return policy


def conviction_gated(k_max, signal_floor=None, meta_floor=None,
                     ratio_floor=None, strict=False):
    """Dynamic K in [0, k_max]: admit a top-K name ONLY if it clears the
    conviction floors. K floats down when few names are convincing (the
    "fewer, higher-conviction" thesis) — the policy under test.

    signal_floor: minimum signal; meta_floor: minimum meta_p; ratio_floor:
    minimum pred/threshold ratio. Any floor left None is not applied.
    A non-finite (NaN) field fails any floor that is set (fail-closed).
    NOTE: by default a floored field ABSENT from the candidate dict PASSES
    that floor (missing != NaN — pinned open owner decision,
    tests/test_portfolio_backtest_v3.py). strict=True instead raises
    ValueError on the first candidate missing a floored field — use it for
    any REAL conviction A/B: the production predictions dump carries no
    meta_p/pred_thresh_ratio unless the panel is built with
    panel_from_frame(..., extra_cols=['meta_p', 'pred_thresh_ratio']), and
    the fail-open default would score floors that never executed as
    "floors change nothing".
    """
    def _field(c, key, floor_name):
        if key in c:
            return c[key]
        if strict:
            raise ValueError(
                f"conviction_gated(strict=True): {floor_name} is set but "
                f"candidate {c.get('symbol', '?')!r} carries no {key!r} — "
                "build the panel with panel_from_frame(..., "
                "extra_cols=['meta_p', 'pred_thresh_ratio'])")
        return 1.0   # fail-open default (pinned owner decision)

    def policy(cands):
        out = []
        for c in cands[:max(int(k_max), 0)]:
            if signal_floor is not None and not (c['signal'] >= signal_floor):
                continue
            if meta_floor is not None and not (
                    _field(c, 'meta_p', 'meta_floor') >= meta_floor):
                continue
            if ratio_floor is not None and not (
                    _field(c, 'pred_thresh_ratio', 'ratio_floor')
                    >= ratio_floor):
                continue
            out.append(c)
        return out
    return policy


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _sorted_desc(period):
    for c in period:
        s = c['signal']   # KeyError = malformed candidate: fail loud, cf. rank_gradient.py:30-33
        if not np.isfinite(s):
            raise ValueError(
                f"non-finite signal for {c.get('symbol', '?')!r}: a NaN signal makes "
                "the ranking row-order-dependent; drop or repair the row upstream "
                "(panel_from_frame already dropna's it)")
    return sorted(period, key=lambda c: c['signal'], reverse=True)


def _admitted_symbols(admitted):
    """Symbol list + set of the admitted book; raises on a duplicate symbol
    within one period (a double-counted book slot)."""
    syms = [c['symbol'] for c in admitted]
    cur = set(syms)
    if len(cur) != len(syms):
        dupes = sorted({s for s in syms if syms.count(s) > 1})
        raise ValueError(f"duplicate symbol(s) within one period: {dupes} — "
                         "each period must list each symbol at most once")
    return syms, cur


def _metrics_summary(nets, grosses, ks, entries_frac, n_entries, n_exits,
                     weight_turnover, periods_per_year, with_series):
    """Shared result-dict builder for run_policy / run_policy_weighted — ONE
    implementation so the two engines' metric surfaces cannot drift (pinned
    by tests/test_portfolio_backtest_v3.py's surface-parity test)."""
    nets = np.asarray(nets, dtype=float)
    n = len(nets)
    n_nonfinite = int((~np.isfinite(nets)).sum())
    if n_nonfinite:
        sharpe = float('nan')   # undefined series must not report a confident 0.0
    elif n > 1 and nets.std() > 1e-12:
        sharpe = float(nets.mean() / nets.std() * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0
    ks_arr = np.asarray(ks)
    inv = ks_arr > 0
    n_invested = int(inv.sum())
    hist = {str(v): int((ks_arr == v).sum()) for v in range(0, 8)} if n else {}
    if n:
        hist['8+'] = int((ks_arr >= 8).sum())
    result = {
        'n_periods': n,
        'mean_admitted_k': round(float(ks_arr.mean()), 3) if n else 0.0,
        'pct_periods_cash': round(float((ks_arr == 0).mean()), 4) if n else 0.0,
        'gross_total': round(float(np.sum(grosses)), 4),
        'net_total': round(float(nets.sum()), 4),
        'sharpe': round(sharpe, 4),
        'avg_entry_fraction': round(float(np.mean(entries_frac)), 4) if n else 0.0,
        'entries': n_entries,
        'exits': n_exits,
        'hit_rate': round(float(np.mean(nets > 0)), 4) if n else 0.0,
        'hit_rate_invested': (round(float(np.mean(nets[inv] > 0)), 4) if n_invested else None),
        'n_invested_periods': n_invested,
        'n_nonfinite_periods': n_nonfinite,
        'weight_turnover': round(float(weight_turnover), 4),
        'admitted_k_hist': hist,
        'pct_periods_k_ge_6': (round(float((ks_arr >= 6).mean()), 4) if n else 0.0),
        'periods_per_year': float(periods_per_year),
    }
    if with_series:
        result['_nets'] = nets
    return result


def run_policy(panel, policy, cost_pct=0.0,
               periods_per_year=DEFAULT_PERIODS_PER_YEAR, with_series=False):
    """Replay one admission policy over the panel. Returns a metrics dict.

    Per period: admit = policy(sorted candidates); gross = equal-weight mean of
    admitted fwd_return; cost = cost_pct * (fraction of the book newly entered)
    — a round-trip charged once at entry, so a name held across periods is not
    re-charged. net = gross - cost. Cash (empty admit) earns 0.

    Cost basis: cost is charged ONLY on the fraction of the book newly
    entered — a held name whose weight changes when K floats is NOT charged;
    weight_turnover (reported below) is the full buy-side weight-basis
    turnover (sum_t sum_s max(w_t - w_{t-1}, 0), w = 1/k), which equals the
    charged fraction only at constant K — charging the difference is an open
    owner decision. hit_rate uses a CALENDAR denominator (an all-cash period
    counts as a non-hit); hit_rate_invested conditions on k > 0 (None when
    never invested). entries/exits count only within-panel transitions — the
    terminal book is never flushed, so entries >= exits by the final book
    size. with_series=True adds a private '_nets' numpy array that makes the
    dict non-JSON-serializable until popped (compare_deflated pops it).
    cost_pct must be in the same units as fwd_return (PERCENT by repo
    convention).
    """
    prev = set()
    prev_k = 0
    weight_turnover = 0.0
    nets, grosses, ks, entries_frac = [], [], [], []
    n_entries_total = n_exits_total = 0
    for period in panel:
        admitted = policy(_sorted_desc(period))
        syms, cur = _admitted_symbols(admitted)
        k = len(syms)
        ks.append(k)
        new = cur - prev
        gone = prev - cur
        n_entries_total += len(new)
        n_exits_total += len(gone)
        w_cur = (1.0 / k) if k else 0.0
        w_prev = (1.0 / prev_k) if prev_k else 0.0
        weight_turnover += sum(
            max((w_cur if s in cur else 0.0) - (w_prev if s in prev else 0.0), 0.0)
            for s in (cur | prev))
        if k > 0:
            gross = float(np.mean([c['fwd_return'] for c in admitted]))
            ef = len(new) / k                       # fraction of book entered
        else:
            gross, ef = 0.0, 0.0                    # all cash
        entries_frac.append(ef)
        grosses.append(gross)
        nets.append(gross - cost_pct * ef)
        prev = cur
        prev_k = k

    return _metrics_summary(nets, grosses, ks, entries_frac, n_entries_total,
                            n_exits_total, weight_turnover, periods_per_year,
                            with_series)


def _resolve_baseline(policies, baseline):
    """Shared baseline/name resolution for compare() and compare_deflated():
    raise on an empty policies dict or an unknown explicit baseline, else
    default to the first key — used by both comparators so they agree."""
    if not policies:
        raise ValueError('compare() needs at least one named policy (got an empty policies dict)')
    names = list(policies)
    if baseline is not None and baseline not in policies:
        raise ValueError(f'baseline {baseline!r} not in policies {names}')
    return names, (baseline if baseline is not None else names[0])


def compare(panel, policies, cost_pct=0.0,
            periods_per_year=DEFAULT_PERIODS_PER_YEAR, baseline=None):
    """Run several named policies over the same panel and report deltas.

    policies: {name: policy_callable}. baseline: the name to difference others
    against (default: the first key). Returns {'results': {name: metrics},
    'deltas': {name: {sharpe_delta, net_delta, k_delta}}, 'baseline': name}.

    EXPLORATORY ONLY: ranks by RAW Sharpe with no correction for how many
    policies were tried. Use compare_deflated() for any ship/no-ship decision
    (strategy_config CONCENTRATION_ENABLED / EDGE_KELLY_ENABLED).
    """
    panel = list(panel)
    names, base = _resolve_baseline(policies, baseline)
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
                     fwd_bars=None, n_trials=None, weight_fns=None):
    """compare() + a DEFLATED Sharpe and turnover per policy (wave-9 #4).

    The plain compare() ranks policies by RAW Sharpe with NO correction for how
    many policies were tried — a multiple-testing leak: the best of N noise
    policies looks good. dsr_from_trade_returns deflates by n_trials (default
    len(policies)), so the winner must beat the expected MAX Sharpe of that
    many tries. Turnover is promoted to first-class (a higher-Sharpe-but-
    higher-turnover policy is not free on a cost-bound book). THIS is the
    difference between certifying a real conviction edge and shipping a mined
    one — PROVIDED fwd_bars and n_trials are supplied honestly for this
    panel's true horizon and cumulative selection pool (see below).

    fwd_bars — should match the panel's forward-return horizon whenever the
    panel is sampled every bar (panel_from_frame on an hourly frame with a
    fb-bar fwd_return_col). Adjacent per-period nets then share fb-1 bars of
    the SAME move: they are not independent draws, and with fwd_bars=1 a
    zero-edge 24h-overlap panel scores DSR ~0.99 (2026-07 review, verified).
    The DSR null is widened by n_eff = n_finite / fwd_bars. Leave unset (it
    defaults to 1) only for genuinely non-overlapping period returns —
    fwd_bars_defaulted=True in the result marks an artifact whose horizon was
    ASSUMED rather than declared.

    n_trials — must cover the TRUE cumulative selection pool across every
    compare_deflated call scored on this panel, not just this call's policies
    dict; defaults to len(policies), which understates the pool the instant
    this panel has screened more than one batch.

    weight_fns — optional {name: weight_fn}; routes that arm through
    run_policy_weighted (sizing/concentration) instead of run_policy
    (admission-only) so a sizing scheme is deflated in the same pool as the
    admission arms. A name in weight_fns not present in policies raises.

    Result-surface notes: (i) 'sharpe' is ANNUALIZED by periods_per_year while
    sr_per_period / expected_max_sr / dsr are PER-PERIOD — compare
    sr_per_period (not sharpe) against expected_max_sr; (ii) n_eff is the
    requested overlap-derived value and n_eff_used is what validation
    consumed after its 10-sample floor — n_eff_floored=True means the
    deflation was weaker than the panel's true breadth warrants; (iii) 'dsr'
    inside deltas is the policy's ABSOLUTE deflated-Sharpe probability, not a
    delta; (iv) 'turnover' is a per-NAME count (entries+exits / n_periods),
    not book-normalized — avg_entry_fraction(_delta) is the book-fraction
    measure.
    """
    from validation import dsr_from_trade_returns
    panel = list(panel)
    names, base = _resolve_baseline(policies, baseline)
    fb = 1 if fwd_bars is None else max(int(fwd_bars), 1)
    nt = len(names) if n_trials is None else int(n_trials)
    weight_fns = weight_fns or {}
    unknown = set(weight_fns) - set(names)
    if unknown:
        raise ValueError(f'weight_fns names not in policies: {sorted(unknown)}')
    results = {}
    for name in names:
        wf = weight_fns.get(name)
        if wf is None:
            m = run_policy(panel, policies[name], cost_pct, periods_per_year,
                           with_series=True)
        else:
            m = run_policy_weighted(panel, policies[name], wf, cost_pct,
                                    periods_per_year, with_series=True)
        nets = m.pop('_nets')
        n_finite = int(np.isfinite(nets).sum())
        n_eff = (n_finite / fb) if n_finite else None
        d = dsr_from_trade_returns(nets, n_trials=nt, n_eff=n_eff,
                                   n_eff_source='panel_overlap_fwd_bars')
        m['dsr'] = round(d['dsr'], 4)
        m['n_eff'] = round(n_eff, 1) if n_eff else 0.0
        m['n_eff_used'] = d['n_eff']
        # Floored ⇔ validation's 10-sample floor actually raised the request.
        # Comparing the request against the echoed d['n_eff'] would false-
        # positive both on the echo's round-to-2 (16.67 > 16.6667) and on the
        # fail-closed n<10 path (which echoes the raw count, floor never applied).
        m['n_eff_floored'] = bool(n_eff is not None and n_eff < 10.0
                                  and d['n_eff'] >= 10.0)
        m['n_dropped'] = d['n_dropped']
        m['sr_per_period'] = round(d['sr'], 6)
        m['n_trials'] = d['n_trials']
        m['fwd_bars'] = fb
        m['fwd_bars_defaulted'] = fwd_bars is None
        m['expected_max_sr'] = round(d['expected_max_sr'], 4)
        m['turnover'] = round((m['entries'] + m['exits']) / max(m['n_periods'], 1), 4)
        results[name] = m
    b = results[base]
    deltas = {name: {
        'sharpe_delta': round(results[name]['sharpe'] - b['sharpe'], 4),
        'net_delta': round(results[name]['net_total'] - b['net_total'], 4),
        'dsr': results[name]['dsr'],
        'turnover_delta': round(results[name]['turnover'] - b['turnover'], 4),
        'k_delta': round(results[name]['mean_admitted_k'] - b['mean_admitted_k'], 3),
        'avg_entry_fraction_delta': round(results[name]['avg_entry_fraction'] - b['avg_entry_fraction'], 4),
    } for name in names}
    return {'results': results, 'deltas': deltas, 'baseline': base}


def panel_from_frame(df, signal_col, fwd_return_col, ticker_col='Ticker',
                     extra_cols=None, signal_lag=0, stats_out=None):
    """Build an A/B panel from a tidy frame (index=timestamp, one row per
    symbol-bar). Each timestamp becomes one period of candidate dicts.

    signal_lag>0 takes the signal from `signal_lag` bars earlier PER TICKER
    (strict PIT: the admission decides on info known before fwd_return
    realizes). The lag is a ROW shift, not a time shift — on a ticker with a
    gap (a missing bar), it reaches back MORE than signal_lag bars; always
    backward (PIT holds), but staleness is unbounded. Rows with a missing
    signal/fwd_return after lagging are dropped. Rows are stably sorted by
    timestamp before lagging, so an unsorted frame cannot leak a future bar's
    signal.

    extra_cols are NOT lagged — they come from the fwd_return bar, so gating
    on an extra_cols floor together with a lagged signal would gate on one
    bar of look-ahead (owner decision pending; see conviction_gated's
    docstring note).

    Raises ValueError on a duplicate column request (a name given both
    positionally and in extra_cols) or on duplicate (timestamp, ticker) rows
    — a predictions dump must carry each symbol at most once per bar.

    stats_out: an optional caller-supplied dict, updated in place with
    {rows_in, rows_dropped, n_periods, mean_candidates_per_period} — coverage
    counters so a silently-shrunken panel (e.g. an unexpected dropna
    wipeout) is detectable (cf. the exit-2 contract in
    scripts/rank_gradient_report.py).
    """
    extra_cols = list(extra_cols or [])
    cols = [ticker_col, signal_col, fwd_return_col] + extra_cols
    if len(set(cols)) != len(cols):
        raise ValueError(f'duplicate column request in panel_from_frame: {cols}')
    work = df[cols].copy()
    if signal_lag:
        work = work.sort_index(kind='stable')
        work[signal_col] = work.groupby(ticker_col)[signal_col].shift(signal_lag)
    work = work.dropna(subset=[signal_col, fwd_return_col])
    # NaT drop matches groupby(level=0)'s default dropna=True key behavior
    # (the vectorized build below no longer uses groupby, so this is now
    # explicit rather than implicit) — a no-op vs today.
    work = work[work.index.notna()]
    # groupby(level=0) iterated in sorted key order with within-group original
    # order; a stable sort reproduces both, so this vectorized boundary-scan
    # build is value-identical to the old per-group iterrows loop. The
    # duplicate-(timestamp, ticker) guard rides the same pass: after the sort
    # a period is one contiguous slice, so a per-period seen-set is a full
    # duplicate scan with no extra groupby materialization.
    work = work.sort_index(kind='stable')
    syms = work[ticker_col].tolist()
    sig = work[signal_col].astype(float).tolist()
    fwd = work[fwd_return_col].astype(float).tolist()
    extras = {col: work[col].tolist() for col in extra_cols}
    idx_vals = work.index.values
    panel = []
    if len(work):
        bounds = np.flatnonzero(np.r_[True, idx_vals[1:] != idx_vals[:-1]]).tolist() + [len(work)]
        for a, b2 in zip(bounds[:-1], bounds[1:]):
            period = []
            seen = set()
            for i in range(a, b2):
                sym = syms[i]
                if sym in seen:
                    raise ValueError(
                        f'duplicate (timestamp, {ticker_col}) rows in panel_from_frame '
                        f'(first: {(work.index[a], sym)!r}) — a predictions dump must '
                        'carry each symbol at most once per bar')
                seen.add(sym)
                c = {'symbol': sym, 'signal': sig[i], 'fwd_return': fwd[i]}
                for col in extra_cols:
                    c[col] = extras[col][i]
                period.append(c)
            panel.append(period)
    if stats_out is not None:
        stats_out.update({
            'rows_in': int(len(df)),
            'rows_dropped': int(len(df) - len(work)),
            'n_periods': len(panel),
            'mean_candidates_per_period': (round(float(np.mean([len(p) for p in panel])), 2)
                                           if panel else 0.0),
        })
    return panel


# ---------------------------------------------------------------------------
# Tier sizing (edge-proportional vs equal-weight)
# ---------------------------------------------------------------------------

def edge_proportional_weights(signals, floor=0.0):
    """Weights proportional to (signal - floor)+, summing to 1. Falls back to
    equal-weight when no admitted name clears the floor. The growth-optimal book
    tracks edge (Kelly) rather than equal-weighting every admitted name.

    (i) A non-finite signal is zeroed BEFORE the floor subtraction, so with a
    NEGATIVE floor a NaN signal receives positive weight (pinned by test;
    changing it is an owner decision). (ii) Any admitted name with
    signal <= floor receives ZERO weight — this weighting also acts as an
    implicit admission filter, so an edge-vs-equal A/B conflates exclusion
    with sizing unless a third equal-weight-over-positive-edge arm is run.
    (iii) The output always sums to 1 (a fully-invested book), including the
    all-below-floor equal-weight fallback — the engine cannot express partial
    cash (owner decision pending).
    """
    s = np.asarray(signals, dtype=float)
    s = np.clip(np.where(np.isfinite(s), s, 0.0) - floor, 0.0, None)
    if len(s) == 0:
        return s
    if s.sum() <= 1e-12:
        return np.full(len(s), 1.0 / len(s))
    return s / s.sum()


def equal_weights(signals):
    """Equal weight over the admitted names — run_policy's book expressed as a
    weight_fn, so run_policy_weighted can A/B a sizing scheme against equal
    weight on the SAME (weight-turnover) cost basis."""
    n = len(signals)
    return np.full(n, 1.0 / n) if n else np.asarray([], dtype=float)


def run_policy_weighted(panel, policy, weight_fn, cost_pct=0.0,
                        periods_per_year=DEFAULT_PERIODS_PER_YEAR, with_series=False):
    """run_policy but the admitted book is weighted by weight_fn([signals])
    instead of equal-weight — to test edge-proportional concentration.

    Cost basis here is WEIGHT turnover (sum of positive weight changes) — a
    DIFFERENT basis from run_policy's entry-set charge; the two agree only at
    constant K under equal weights, so an apples-to-apples sizing A/B is
    run_policy_weighted(..., equal_weights) vs
    run_policy_weighted(..., edge_proportional_weights). Weights are expected
    to sum to 1 (avg_gross_exposure surfaces leverage; enforcing it is
    deliberately not done pending the partial-cash owner decision).
    """
    prev_w = {}
    nets, grosses, ks, entries_frac, gross_exposures = [], [], [], [], []
    n_entries_total = n_exits_total = 0
    weight_turnover = 0.0
    for period in panel:
        admitted = policy(_sorted_desc(period))
        syms, cur = _admitted_symbols(admitted)
        k = len(syms)
        ks.append(k)
        prev_syms = set(prev_w)
        new = cur - prev_syms
        gone = prev_syms - cur
        n_entries_total += len(new)
        n_exits_total += len(gone)
        if not admitted:
            nets.append(0.0)
            grosses.append(0.0)
            entries_frac.append(0.0)
            prev_w = {}
            continue
        w = np.asarray(weight_fn([c['signal'] for c in admitted]), float)
        if w.shape != (len(admitted),):
            raise ValueError(f'weight_fn returned shape {w.shape} for {len(admitted)} admitted names')
        if not np.isfinite(w).all():
            raise ValueError('weight_fn returned non-finite weights')
        gross = float(np.sum(w * np.array([c['fwd_return'] for c in admitted])))
        cur_w = {c['symbol']: float(wi) for c, wi in zip(admitted, w)}
        turnover = sum(max(cur_w.get(s, 0.0) - prev_w.get(s, 0.0), 0.0)
                       for s in cur | prev_syms)
        weight_turnover += turnover
        entries_frac.append(len(new) / k)
        grosses.append(gross)
        nets.append(gross - cost_pct * turnover)
        gross_exposures.append(float(w.sum()))
        prev_w = cur_w

    result = _metrics_summary(nets, grosses, ks, entries_frac, n_entries_total,
                              n_exits_total, weight_turnover, periods_per_year,
                              with_series)
    # avg_gross_exposure averages INVESTED periods only (leverage of the book
    # actually held); all-cash periods are excluded from the mean.
    result['avg_gross_exposure'] = (round(float(np.mean(gross_exposures)), 4)
                                    if gross_exposures else 0.0)
    return result
