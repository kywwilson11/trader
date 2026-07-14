"""Gate attribution + conviction calibration from the decision journals.

Every veto the system logs is a COUNTERFACTUAL trade: replaying it
through the SAME policy_exits kernel that prices live exits tells us
what that veto actually saved (or cost). A gate whose vetoed entries
would have averaged a positive net return is charging admission, not
providing protection — and the only way to know is to measure.

Two products, both feeding the high-conviction program:

  1. GATE ATTRIBUTION — per skip_reason: veto count, counterfactual
     mean/total net return (after the same round-trip costs live pays).
     "Saved" is the NEGATIVE of counterfactual P&L: a gate that vetoed
     trades averaging -0.8% net saved +0.8% per veto.

  2. CONVICTION CALIBRATION — for TAKEN entries: realized outcome by
     predicted-return decile and by meta-probability bucket. This is
     the red team's "does high conviction select signal or just tail
     noise" experiment, answered from our own logs.

2026-07 review methodology (see the printed banner + the module for
details, and treat pre-2026-07 report numbers as NOT comparable):
  - FETCH-FAILURE (bars API raised/None/too-short) is now counted
    separately from HORIZON-PENDING (replay hasn't resolved yet) —
    folding both into one "unresolved" bucket hid API outages.
  - Rows are deduped to one PER (symbol, reason, calendar-day) EPISODE
    before replay — undeduped, a symbol skipped/exited every cycle of a
    day emits ~24 overlapping replays/day that double-count the same P&L.
  - Every counterfactual mean carries a 90% bootstrap CI; verdicts only
    fire when the CI excludes zero, else the honest answer is "cannot
    conclude" rather than reading the raw sign of a noisy mean.

Usage:
    python decision_report.py --days 30
Writes decision_report.json beside the journals.
"""

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

JOURNAL_DIR = BASE_DIR / 'journals'

# Gates whose vetoes carry a per-symbol counterfactual row worth pricing
# (conviction/risk gates on real candidates). Mechanical skips
# (already-held, cooldown, budget) are summarized in entry_window rows
# but never priced per-symbol. Kept in sync with the loops' _journal_skip
# call sites (base_loop.py / stock_loop.py, wave-5 Tier1-1).
GATE_REASONS = ['sentiment_block', 'llm_veto', 'meta_veto', 'q10_tail_veto',
                'edgar_event', 'below_threshold', 'cost_floor',
                'winners_curse', 'correlation', 'bucket_cap', 'trend_filter',
                'sizing_zero', 'earnings', 'qty_zero']
MAX_HOLD_BARS = {'crypto': 24, 'stock': 24}   # vertical barrier for replays

# Skips that are mechanical/budget-driven and never priced per-symbol here —
# they only show up rolled up in entry_window rows (see admitted_k_distribution).
UNPRICED_GATES = ('already_held', 'cooldown', 'budget')


def _is_crypto(sym: str) -> bool:
    return '/' in sym


def load_journal(days: int) -> list[dict]:
    rows = []
    today = dt.date.today()
    for d in range(days + 1):
        p = JOURNAL_DIR / f"{(today - dt.timedelta(days=d)).isoformat()}.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _atr14(df) -> np.ndarray:
    """Lightweight ATR(14) — enough for the exit kernel's stop math."""
    import pandas as pd
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=5).mean().values


def _eod_mask(index, asset_type: str) -> np.ndarray:
    if asset_type == 'crypto':
        return np.zeros(len(index), dtype=bool)
    from policy_exits import eod_mask_from_index
    return eod_mask_from_index(index, asset_type)


def replay_entry(bars, ts, asset_type: str,
                 spread_pct: float | None = None) -> float | None:
    """Counterfactual NET % return of an entry at the first bar at/after
    ts, exited by the live policy stack (stops/trail/TP/EOD/vertical —
    no signal exits, since the counterfactual has no later model preds).
    None when the horizon isn't resolvable yet (HORIZON-PENDING — the
    caller's grouped replay, `_replay_grouped`, counts this separately
    from a FETCH-FAILURE upstream).

    spread_pct: per-row spread (percent of price) if the journal carries
    one ('spread_pct' — not journaled by the loops today, wave-9+ future
    work); falls back to fees.FLAT_SPREAD_PCT[asset_type] otherwise. See
    the cost_floor caveat run_report prints when this flat fallback is in
    play for a spread-sensitive gate.
    """
    from policy_exits import exit_walk
    from strategy_config import policy_for
    from fees import round_trip_cost_pct, FLAT_SPREAD_PCT

    idx = bars.index
    i = int(idx.searchsorted(ts))
    if i >= len(bars) - 2:
        return None
    sub = bars.iloc[i:]
    n = len(sub)
    max_hold = MAX_HOLD_BARS.get(asset_type, 24)
    if n < 3:
        return None
    closes = sub['Close'].values.astype(np.float64)
    exit_idx, exit_px, _reason = exit_walk(
        closes,
        sub['High'].values.astype(np.float64),
        sub['Low'].values.astype(np.float64),
        sub['Open'].values.astype(np.float64),
        _atr14(sub),
        _eod_mask(sub.index, asset_type),
        policy_for(asset_type),
        max_hold=max_hold, use_signal_exit=False)
    j = int(exit_idx[0])
    if j <= 0 or j >= n:
        return None
    # Unresolved horizon: vertical barrier coincides with the last bar we
    # have. 2026-07 fix: this guard used to apply to crypto only, so a
    # truncated STOCK replay (fewer bars available than max_hold) was
    # silently priced as if the vertical exit were real, instead of being
    # counted horizon-pending like its crypto counterpart.
    if j == n - 1 and (n - 1) < max_hold:
        return None
    gross = (float(exit_px[0]) - closes[0]) / closes[0] * 100.0
    spread = spread_pct if spread_pct is not None else FLAT_SPREAD_PCT[asset_type]
    return gross - round_trip_cost_pct(asset_type, spread)


def _dedup_first_per_day(rows: list[dict], key_fields: list[str],
                         ts_key: str = 'ts') -> list[dict]:
    """Keep only the FIRST row per (key_fields..., calendar-day) — a symbol
    skipped for the same reason (or exited on a signal_sell) every cycle of
    a day otherwise emits ~24 overlapping 24-bar replays/day, double-
    counting the same underlying P&L into saved_total_pct /
    given_up_total_pct. Rows are sorted by ts first so "first" means
    earliest-in-day, matching what the live system actually saw first.
    Rows with an unparseable ts sort last and get their own (None) day
    bucket, so they are never silently merged with a well-formed row.
    """
    import pandas as pd

    def _ts_key(r):
        try:
            return pd.Timestamp(r[ts_key])
        except (ValueError, KeyError, TypeError):
            return pd.Timestamp.max

    ordered = sorted(rows, key=_ts_key)
    seen = set()
    out = []
    for r in ordered:
        try:
            day = pd.Timestamp(r[ts_key]).date().isoformat()
        except (ValueError, KeyError, TypeError):
            day = None
        key = tuple(r.get(f) for f in key_fields) + (day,)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _replay_grouped(rows: list[dict], api, ts_key: str = 'ts'):
    """Shared fetch-once-per-symbol replay used by gate_attribution,
    signal_exit_audit and conviction_calibration.

    Distinguishes the two ways a row fails to price: FETCH-FAILURE (the
    bars call raised, or came back None/too-short — an infra problem that
    tells us nothing about the gate/exit being priced) vs HORIZON-PENDING
    (replay_entry returned None because the exit horizon hasn't resolved
    yet against the bars we DID get — expected, and self-heals as more
    bars land). Pre-2026-07, both were folded into one ambiguous
    "unresolved" bucket (or, in conviction_calibration, silently dropped
    with no counter at all).

    Returns (samples, n_fetch_failed, n_horizon_pending) where samples is
    a list of (row, net) for every row that priced successfully. ts
    parsing preserves the original offset-aware-ISO /
    tz-localize-only-if-naive logic; a row with an unparseable ts can't be
    replayed at all (no fetch problem, no horizon to wait on) and is
    counted as a fetch failure since there's nothing to price.
    """
    from market_data import fetch_bars_alpaca, fetch_stock_bars_alpaca
    import pandas as pd

    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get('symbol'):
            by_symbol[r['symbol']].append(r)

    samples = []
    n_fetch_failed = 0
    n_horizon_pending = 0
    for sym, srows in by_symbol.items():
        asset = 'crypto' if _is_crypto(sym) else 'stock'
        try:
            bars = (fetch_bars_alpaca(api, sym) if asset == 'crypto'
                    else fetch_stock_bars_alpaca(api, sym))
        except Exception:
            bars = None
        if bars is None or len(bars) < 30:
            n_fetch_failed += len(srows)
            continue
        if bars.index.tz is None:
            bars = bars.tz_localize('UTC')
        for r in srows:
            try:
                ts = pd.Timestamp(r[ts_key])
                if ts.tz is None:
                    ts = ts.tz_localize('UTC')
            except (ValueError, KeyError, TypeError):
                n_fetch_failed += 1
                continue
            net = replay_entry(bars, ts, asset, spread_pct=r.get('spread_pct'))
            if net is None:
                n_horizon_pending += 1
                continue
            samples.append((r, net))
    return samples, n_fetch_failed, n_horizon_pending


def _bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.10,
                  seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`.

    Resamples BY VALUE (each element treated as one independent draw) —
    valid here because gate_attribution/signal_exit_audit dedupe to one
    row per (symbol, reason-or-exit, calendar-day) EPISODE before replay,
    so consecutive elements are no longer ~24 overlapping intraday
    replays of the same underlying move. Deterministic seed (default 0)
    so report numbers reproduce run-to-run.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return (float('nan'), float('nan'))
    if vals.size == 1:
        v = round(float(vals[0]), 3)
        return (v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot_means = vals[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return (round(float(lo), 3), round(float(hi), 3))


def _bucket_stats(vals) -> dict:
    arr = np.asarray(vals, dtype=float)
    ci = _bootstrap_ci(arr)
    return {'n': int(arr.size),
            'mean_net_pct': round(float(arr.mean()), 3),
            'hit_rate': round(float((arr > 0).mean()), 3),
            'ci90': [ci[0], ci[1]]}


def _gate_verdict(mean: float, ci) -> str:
    """A gate reads REVIEW (charging admission) only when its bootstrap CI
    excludes zero on the positive side; OK only when it excludes zero on
    the negative side. Anything else is an honest 'cannot conclude' rather
    than reading the raw sign of a noisy mean."""
    lo, hi = ci
    if lo > 0:
        return 'REVIEW (charging admission — CI excludes zero)'
    if hi < 0:
        return 'OK (earning its keep — CI excludes zero)'
    return 'cannot conclude (CI spans zero)'


def _signal_exit_verdict(mean: float, ci) -> str:
    lo, hi = ci
    if lo > 0:
        return 'CHANGE — apply 2-reading confirmation to signal exits (CI excludes zero)'
    if hi < 0:
        return 'NO CHANGE — the flip is saving money (CI excludes zero)'
    return 'cannot conclude (CI spans zero)'


def gate_attribution(rows: list[dict], api) -> dict:
    """Counterfactual P&L per gate from journaled skips.

    2026-07: rows are deduped to one PER (symbol, skip_reason,
    calendar-day) EPISODE before replay — a symbol skipped for the same
    reason every cycle of a day otherwise emits ~24 overlapping 24-bar
    replays/day, double-counting the same underlying P&L into
    saved_total_pct. 'vetoes_priced' counts episodes (what's replayed and
    summed); 'vetoes_raw' is the pre-dedup journal row count, reported
    alongside for transparency.
    """
    skips_raw = [r for r in rows if r.get('action') == 'skip'
                 and r.get('skip_reason') in GATE_REASONS and r.get('symbol')]
    raw_by_gate: dict[str, int] = defaultdict(int)
    for r in skips_raw:
        raw_by_gate[r['skip_reason']] += 1
    skips = _dedup_first_per_day(skips_raw, ['symbol', 'skip_reason'])

    samples, n_fetch_failed, n_horizon_pending = _replay_grouped(skips, api)
    per_gate: dict[str, list[float]] = defaultdict(list)
    cost_floor_has_spread = False
    for r, net in samples:
        per_gate[r['skip_reason']].append(net)
        if r['skip_reason'] == 'cost_floor' and r.get('spread_pct') is not None:
            cost_floor_has_spread = True

    out = {}
    for gate in GATE_REASONS:
        vals = np.asarray(per_gate.get(gate, []), dtype=float)
        if vals.size == 0:
            continue
        ci = _bootstrap_ci(vals)
        out[gate] = {
            'vetoes_priced': int(vals.size),
            'vetoes_raw': int(raw_by_gate.get(gate, 0)),
            'counterfactual_mean_net_pct': round(float(vals.mean()), 3),
            'counterfactual_hit_rate': round(float((vals > 0).mean()), 3),
            # positive saved = the gate is earning its keep
            'saved_total_pct': round(float(-vals.sum()), 2),
            'ci90': [ci[0], ci[1]],
        }
    out['_fetch_failed'] = n_fetch_failed
    out['_horizon_pending'] = n_horizon_pending
    out['_unresolved'] = n_fetch_failed + n_horizon_pending   # backward-compat sum
    # cost_floor fires on WIDE-spread names but every row above priced at a
    # FLAT spread (no per-row spread_pct was journaled) — flag it so the
    # printed report can caveat that gate's verdict specifically.
    out['_cost_floor_flat_spread'] = ('cost_floor' in out) and not cost_floor_has_spread
    return out


def signal_exit_audit(rows: list[dict], api) -> dict:
    """Counterfactual for journaled `signal_sell` exits (2026-07 review).

    The signal-flip exit fires on a SINGLE closed-bar blend reading while
    stops/TP/trailing need 2 consecutive readings — so the model can dump a
    winner the trailing stop would have kept riding, and the cooldown then
    blocks re-entry. This prices that choice: replay each signal_sell moment
    forward through the stop/trail/TP/EOD stack WITHOUT signal exits
    (replay_entry) — "what would the position have earned from here had the
    flip not fired and the stops managed it instead."

    Positive counterfactual mean => signal exits are surrendering edge and
    deserve the same 2-reading confirmation the stops already have; negative
    => the flip is saving money and stays as-is. Two deliberate conservative
    biases (both make HOLDING look worse, so only a robustly positive result
    argues for change): the replay re-enters at the exit bar, forgetting the
    original entry's ratcheted trailing HWM (fresh, looser stop), and
    replay_entry charges a full fresh round-trip cost. A THIRD, non-
    conservative bias cuts the other way and is disclosed at print time:
    this audit models signal exits fully DISABLED, a stronger intervention
    than the 2-reading confirmation actually on the table, so a positive
    result is an upper bound on the case for change.

    2026-07: rows are deduped to one PER (symbol, calendar-day) EPISODE
    before replay (same rationale as gate_attribution) — 'episodes' is the
    deduped count that feeds the priced/mean/given-up numbers;
    'n_signal_sells' stays the raw pre-dedup row count.
    """
    sells_raw = [r for r in rows if r.get('action') == 'sell'
                 and r.get('exit_reason') == 'signal_sell' and r.get('symbol')]
    if not sells_raw:
        return {'n_signal_sells': 0}

    sells = _dedup_first_per_day(sells_raw, ['symbol'])

    priced, n_fetch_failed, n_horizon_pending = _replay_grouped(sells, api)
    nets, realized = [], []
    for r, net in priced:
        nets.append(net)
        if isinstance(r.get('pnl_pct'), (int, float)):
            realized.append(float(r['pnl_pct']))

    vals = np.asarray(nets, dtype=float)
    out = {'n_signal_sells': len(sells_raw), 'episodes': len(sells),
           'priced': int(vals.size),
           '_fetch_failed': n_fetch_failed, '_horizon_pending': n_horizon_pending,
           '_unresolved': n_fetch_failed + n_horizon_pending}
    if vals.size:
        ci = _bootstrap_ci(vals)
        out.update({
            'counterfactual_mean_net_pct': round(float(vals.mean()), 3),
            'counterfactual_hit_rate': round(float((vals > 0).mean()), 3),
            # positive = money the flip left on the table, summed
            'given_up_total_pct': round(float(vals.sum()), 2),
            'ci90': [ci[0], ci[1]],
            'realized_mean_pnl_pct': (round(float(np.mean(realized)), 3)
                                      if realized else None),
        })
    return out


def conviction_calibration(rows: list[dict], api) -> dict:
    """Realized counterfactual outcome of TAKEN entries by conviction
    bucket (prediction-magnitude tercile x meta-probability band).

    Uses the same kernel replay as the vetoes so taken and vetoed trades
    are priced identically (journaled realized exits depend on signal
    exits the replay can't see; the kernel replay is the comparable
    yardstick). 2026-07: fetch-failures and horizon-pending rows are now
    counted (_fetch_failed/_horizon_pending) instead of silently shrinking
    n with no trace; buys are discrete fills already, so unlike the two
    gate-side analyses these rows are NOT deduped."""
    buys = [r for r in rows if r.get('action') == 'buy'
            and r.get('pred_return') is not None and r.get('symbol')]
    if not buys:
        return {'_fetch_failed': 0, '_horizon_pending': 0, '_unresolved': 0}

    priced, n_fetch_failed, n_horizon_pending = _replay_grouped(buys, api)
    samples = []   # (pred, meta_prob_or_None, net, rank_or_None)
    for r, net in priced:
        # meta_p is the wave-5 field name; meta_prob the legacy one
        mp = r.get('meta_p', r.get('meta_prob'))
        samples.append((float(r['pred_return']), mp, net, r.get('entry_rank')))

    if len(samples) < 9:
        return {'n': len(samples),
                'note': 'too few resolvable taken entries to bucket',
                '_fetch_failed': n_fetch_failed,
                '_horizon_pending': n_horizon_pending,
                '_unresolved': n_fetch_failed + n_horizon_pending}

    preds = np.array([s[0] for s in samples])
    nets = np.array([s[2] for s in samples])
    out = {'n': len(samples), '_fetch_failed': n_fetch_failed,
           '_horizon_pending': n_horizon_pending,
           '_unresolved': n_fetch_failed + n_horizon_pending}

    # Prediction-magnitude terciles: does the top third out-earn?
    qs = np.quantile(preds, [1 / 3, 2 / 3])
    buckets = {'pred_low': preds <= qs[0],
               'pred_mid': (preds > qs[0]) & (preds <= qs[1]),
               'pred_high': preds > qs[1]}
    for name, mask in buckets.items():
        if mask.sum():
            out[name] = _bucket_stats(nets[mask])

    # Entry-rank buckets — Stage-0 experiment 1: does rank 6-7 carry
    # materially less edge than rank 1-3? (gates the concentration cap)
    ranked = [(s[3], s[2]) for s in samples if s[3] is not None]
    if len(ranked) >= 9:
        rk = np.array([r[0] for r in ranked], dtype=float)
        rn = np.array([r[1] for r in ranked], dtype=float)
        for name, lo, hi in (('rank_1_3', 1, 3), ('rank_4_5', 4, 5),
                             ('rank_6_7', 6, 7), ('rank_8_plus', 8, 9999)):
            mask = (rk >= lo) & (rk <= hi)
            if mask.sum():
                out[name] = _bucket_stats(rn[mask])

    metas = [(s[1], s[2]) for s in samples if s[1] is not None]
    if len(metas) >= 9:
        mp = np.array([m[0] for m in metas])
        mn = np.array([m[1] for m in metas])
        for name, lo, hi in (('meta_0.30_0.45', 0.30, 0.45),
                             ('meta_0.45_0.60', 0.45, 0.60),
                             ('meta_0.60_1.00', 0.60, 1.01)):
            mask = (mp >= lo) & (mp < hi)
            if mask.sum():
                out[name] = _bucket_stats(mn[mask])
    return out


def admitted_k_distribution(rows: list[dict]) -> dict:
    """Admitted-k distribution from entry_window summary rows — Stage-0
    experiment 2 (does the gate stack already enforce concentration?).
    No API/replay needed; reads the journals directly."""
    out = {}
    for asset in ('stock', 'crypto'):
        wins = [r for r in rows if r.get('action') == 'entry_window'
                and r.get('asset_type') == asset]
        if not wins:
            continue
        ks = np.array([int(r.get('admitted_k', 0)) for r in wins])
        veto_tot = defaultdict(int)
        for r in wins:
            for reason, c in (r.get('veto_counts') or {}).items():
                veto_tot[reason] += int(c)
        hist = {str(k): int((ks == k).sum()) for k in range(0, 8)}
        hist['8+'] = int((ks >= 8).sum())
        out[asset] = {
            'windows': len(wins),
            'mean_admitted_k': round(float(ks.mean()), 2),
            'pct_windows_k_ge_6': round(float((ks >= 6).mean()), 3),
            'pct_windows_zero': round(float((ks == 0).mean()), 3),
            'admitted_k_hist': hist,
            'total_vetoes_by_reason': dict(
                sorted(veto_tot.items(), key=lambda kv: -kv[1])),
        }
    return out


def _write_stale_report(days: int, api_available) -> dict:
    """run_report ALWAYS writes decision_report.json, even when nothing
    could be priced — 2026-07 fix: previously a get_api() failure returned
    {} without touching the file, so scripts/rank_gradient_report.py could
    silently read a STALE prior report as if it were current. Marking
    'stale': True lets downstream consumers refuse to trust it."""
    stale = {'generated': dt.datetime.now().isoformat(), 'days': days,
             'api_available': api_available, 'stale': True}
    out = BASE_DIR / 'decision_report.json'
    with open(out, 'w') as f:
        json.dump(stale, f, indent=2)
    return stale


def run_report(days: int = 30) -> dict:
    rows = load_journal(days)
    if not rows:
        print("No journal entries found.")
        return _write_stale_report(days, api_available=None)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from trading_utils import get_api
        api = get_api()
    except Exception as e:
        print(f"No API available ({e}) — cannot price counterfactuals. "
              f"Writing a STALE decision_report.json (api_available=False) "
              f"so downstream consumers (scripts/rank_gradient_report.py) "
              f"don't silently read an empty {{}} as a live report.")
        return _write_stale_report(days, api_available=False)

    gates = gate_attribution(rows, api)
    conviction = conviction_calibration(rows, api)
    admitted_k = admitted_k_distribution(rows)
    sig_exit = signal_exit_audit(rows, api)

    # --- headline banner (2026-07 review): fetch failures and pending
    # horizons must be visible BEFORE anyone reads a verdict below ---
    gate_priced = sum(g['vetoes_priced'] for k, g in gates.items()
                      if not k.startswith('_'))
    gate_unpriced = gates.get('_fetch_failed', 0) + gates.get('_horizon_pending', 0)
    sig_priced = sig_exit.get('priced', 0)
    sig_unpriced = sig_exit.get('_fetch_failed', 0) + sig_exit.get('_horizon_pending', 0)
    conv_priced = conviction.get('n', 0)
    conv_unpriced = conviction.get('_fetch_failed', 0) + conviction.get('_horizon_pending', 0)
    total_fetch_failed = (gates.get('_fetch_failed', 0)
                          + sig_exit.get('_fetch_failed', 0)
                          + conviction.get('_fetch_failed', 0))
    total_priced = gate_priced + sig_priced + conv_priced
    total_unpriced = gate_unpriced + sig_unpriced + conv_unpriced
    total_considered = total_priced + total_unpriced
    unpriced_rate = (total_unpriced / total_considered) if total_considered else 0.0

    if total_fetch_failed > 0 or unpriced_rate > 0.30:
        print(f"\nWARNING: {unpriced_rate:.0%} of rows unpriced "
              f"({total_fetch_failed} fetch failures) — sections below are "
              f"NOT representative")
    print("methodology 2026-07: per-episode dedup + bootstrap CIs — "
          "numbers not comparable to earlier reports")

    if admitted_k:
        print(f"\n=== ADMITTED-K DISTRIBUTION (last {days}d) ===")
        for asset, a in admitted_k.items():
            print(f"{asset}: {a['windows']} windows, "
                  f"mean k={a['mean_admitted_k']}, "
                  f"P(k>=6)={a['pct_windows_k_ge_6']:.0%}, "
                  f"P(k=0)={a['pct_windows_zero']:.0%}")
        print("If P(k>=6) is tiny, the gate stack already concentrates "
              "and a top-K cap adds little.")

    print(f"\n=== GATE ATTRIBUTION (last {days}d) ===")
    print(f"Unpriced gates (counted only in entry_window rollups): "
          f"{', '.join(UNPRICED_GATES)}")
    for gate, g in gates.items():
        if gate.startswith('_'):
            continue
        lo, hi = g['ci90']
        verdict = _gate_verdict(g['counterfactual_mean_net_pct'], g['ci90'])
        print(f"{gate:<18} n={g['vetoes_priced']:<4} (raw={g['vetoes_raw']:<4}) "
              f"mean={g['counterfactual_mean_net_pct']:+.2f}% "
              f"[{lo:+.2f}, {hi:+.2f}] hit={g['counterfactual_hit_rate']:.0%} "
              f"saved={g['saved_total_pct']:+.1f}%  -> {verdict}")
    print(f"(fetch failures: {gates.get('_fetch_failed', 0)}, "
          f"horizon-pending: {gates.get('_horizon_pending', 0)}, "
          f"unresolved total: {gates.get('_unresolved', 0)})")
    if gates.get('_cost_floor_flat_spread'):
        print("NOTE: cost_floor priced at FLAT spread but it fires on "
              "WIDE-spread names — its 'charging admission' reading is "
              "structurally unreliable; ignore until spread_pct is journaled.")
    print("Verdict key: REVIEW = 90% bootstrap CI excludes zero on the "
          "positive side (charging admission); OK = CI excludes zero on the "
          "negative side (earning its keep); cannot conclude = CI spans zero.")

    print(f"\n=== CONVICTION CALIBRATION (taken entries, kernel-replayed) ===")
    for k, v in conviction.items():
        if isinstance(v, dict) and 'mean_net_pct' in v:
            lo, hi = v.get('ci90', (float('nan'), float('nan')))
            print(f"{k:<18} n={v['n']:<5} mean={v['mean_net_pct']:+.2f}% "
                  f"[{lo:+.2f}, {hi:+.2f}]  hit={v['hit_rate']:.0%}")
    if conviction.get('n'):
        print(f"(n={conviction['n']} priced entries; fetch failures="
              f"{conviction.get('_fetch_failed', 0)}, horizon-pending="
              f"{conviction.get('_horizon_pending', 0)})")

    if sig_exit.get('priced'):
        print(f"\n=== SIGNAL-EXIT AUDIT (last {days}d) ===")
        lo, hi = sig_exit.get('ci90', (float('nan'), float('nan')))
        print(f"signal_sells={sig_exit['n_signal_sells']} "
              f"episodes={sig_exit.get('episodes', sig_exit['n_signal_sells'])} "
              f"priced={sig_exit['priced']} "
              f"cf mean={sig_exit['counterfactual_mean_net_pct']:+.2f}% "
              f"[{lo:+.2f}, {hi:+.2f}] "
              f"cf hit={sig_exit['counterfactual_hit_rate']:.0%} "
              f"given up={sig_exit['given_up_total_pct']:+.1f}%")
        print(f"Verdict: {_signal_exit_verdict(sig_exit['counterfactual_mean_net_pct'], (lo, hi))}")
        print("Audit bias disclosure: this counterfactual models signal "
              "exits fully DISABLED (stops/trail/TP/EOD manage the whole "
              "hold) — a STRONGER intervention than the 2-reading "
              "confirmation actually under decision, so a CHANGE verdict "
              "here is an upper bound on the case for change, not proof "
              "the milder fix earns as much.")

    report = {'generated': dt.datetime.now().isoformat(), 'days': days,
              'gates': gates, 'conviction': conviction,
              'admitted_k': admitted_k, 'signal_exit': sig_exit}
    out = BASE_DIR / 'decision_report.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {out}")
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Gate attribution report')
    ap.add_argument('--days', type=int, default=30)
    args = ap.parse_args()
    run_report(args.days)
