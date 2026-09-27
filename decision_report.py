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
  - 2026-07b: counterfactuals use FULL-FRAME ATR stops (the entry-bar-slice
    ATR head was always NaN -> fixed-pct fallback stops), and rows whose ts
    predates the fetched bar frame are counted OUT-OF-WINDOW instead of
    being silently replayed from the frame's first bar.

Usage:
    python decision_report.py --days 30 [--out DIR]
Writes decision_report.json to the repo root (BASE_DIR) — the path gui.py
and scripts/rank_gradient_report.py read; the journals live in BASE_DIR/journals.
`--out DIR` writes it under DIR instead (stale and normal paths alike).
"""

import argparse
import datetime as dt
import json
import os
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
                'sizing_zero', 'earnings', 'qty_zero',
                # 2026-08 influence audit (open question #4): the formerly
                # UNPRICED book-wide gates now journal per-symbol skip rows
                # (IA-3). NOTE: the loops' vc[...] counter keys for the two
                # VIX gates stay 'macro_halt'/'vix_block' (entry_window
                # back-compat) — only the skip_reason names are new.
                'vix_halt', 'vix25_block', 'macro_standdown',
                'rr5_demotion', 'rank_near_miss']
MAX_HOLD_BARS = {'crypto': 24, 'stock': 24}   # vertical barrier for replays

MIN_VERDICT_N = 9    # no REVIEW/OK/CHANGE verdict below this n (matches the >=9 bucketing floor)
MIN_BUCKET_N = 10    # rank_* buckets suppressed below this n (they feed rank_gradient_verdict, which reads only mean_net_pct)
_CRYPTO_BAR_CAP = 5000
# Stock replay frame = the last 320 hourly bars (market_data.py
# fetch_stock_bars_alpaca default limit=320 -> .tail(320)); at ~16 bars per
# trading day (extended hours included) that is ~20 trading days, i.e. a
# floor of ~26 calendar days (~30 observed). Report windows longer than this
# have stock rows that land _out_of_window. (The 45-day fetch start is NOT
# the binding constraint — the .tail(320) is.)
STOCK_FRAME_MIN_DAYS = 26

# Mechanical/budget/fail-closed vetoes never priced per-symbol — they only
# appear rolled up in entry_window veto_counts. Producer keys verified
# against base_loop.py / stock_loop.py vc[...] literals ('budget' was a
# phantom; the real key is 'trade_budget').
UNPRICED_GATES = ('already_held', 'bad_price', 'cooldown',
                  'hard_stop_lockout', 'macro_halt', 'max_exposure',
                  'no_pred', 'no_quote', 'position_cap', 'trade_budget',
                  'vix_block')

_KEPT_ACTIONS = frozenset({'skip', 'buy', 'sell', 'entry_window'})
_KEPT_FIELDS = ('action', 'skip_reason', 'symbol', 'ts', 'spread_pct',
                'exit_reason', 'pnl_pct', 'pred_return', 'meta_p',
                'meta_prob', 'entry_rank', 'asset_type', 'admitted_k',
                'veto_counts', 'n_candidates')


def _is_crypto(sym: str) -> bool:
    return '/' in sym


def load_journal(days: int) -> list[dict]:
    """Read the last `days` journal files and return only the rows this
    module ever consumes: dict rows whose 'action' is one of
    _KEPT_ACTIONS, projected down to _KEPT_FIELDS. The raw 30-day journal
    can be ~10^6 dicts (llm_analysis/account_risk rows, multi-KB
    llm_reasoning strings, ...) on the 8GB Jetson; this module uses a tiny
    fraction of that, so filter+project at read time rather than
    materializing everything."""
    rows = []
    today = dt.date.today()
    for d in range(days + 1):
        p = JOURNAL_DIR / f"{(today - dt.timedelta(days=d)).isoformat()}.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get('action') in _KEPT_ACTIONS:
                        rows.append({k: obj[k] for k in _KEPT_FIELDS if k in obj})
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
                 spread_pct: float | None = None,
                 atr_full=None, eod_full=None) -> float | None:
    """Counterfactual NET % return of an entry at the first bar at/after
    ts, exited by the live policy stack (stops/trail/TP/EOD/vertical —
    no signal exits, since the counterfactual has no later model preds).
    Returns None for ts before the frame (OUT-OF-WINDOW — the caller's
    grouped replay, `_replay_grouped`, counts this separately from a
    FETCH-FAILURE upstream), and None when the horizon isn't resolvable
    yet (HORIZON-PENDING, counted separately again).

    spread_pct: per-row spread (percent of price) if the journal carries
    one ('spread_pct' — both loops journal this on cost_floor skips since
    2026-07-13, base_loop._execute_buys / stock_loop._execute_buys); falls back to
    fees.FLAT_SPREAD_PCT[asset_type] for rows predating that (or other
    gates that don't journal it). See the cost_floor caveat run_report
    prints when this flat fallback is in play for a spread-sensitive gate.

    atr_full/eod_full: optional full-frame hoists (_atr14(bars) /
    _eod_mask(bars.index, asset_type)) so a caller pricing many rows
    against the same `bars` frame (see `_replay_grouped`) computes them
    ONCE per symbol instead of once per row. When None they are computed
    here on the FULL frame — never on the entry-bar slice, whose ATR head
    is always NaN (rolling min_periods=5 needs bars BEFORE the slice
    start), which previously forced every counterfactual onto the
    exit_walk kernel's fixed-pct fallback stops (crypto 6%/5%, stock
    5%/4%) instead of the real ATR-based stop distance.
    """
    from policy_exits import exit_walk
    from strategy_config import policy_for
    from fees import round_trip_cost_pct, FLAT_SPREAD_PCT

    idx = bars.index
    if ts < idx[0]:
        return None
    i = int(idx.searchsorted(ts))
    if i >= len(bars) - 2:
        return None
    max_hold = MAX_HOLD_BARS.get(asset_type, 24)
    if atr_full is None:
        atr_full = _atr14(bars)
    if eod_full is None:
        eod_full = _eod_mask(bars.index, asset_type)
    sub = bars.iloc[i:i + max_hold + 1]
    n = len(sub)
    closes = sub['Close'].values.astype(np.float64)
    exit_idx, exit_px, _reason = exit_walk(
        closes,
        sub['High'].values.astype(np.float64),
        sub['Low'].values.astype(np.float64),
        sub['Open'].values.astype(np.float64),
        np.asarray(atr_full[i:i + max_hold + 1], dtype=np.float64),
        np.asarray(eod_full[i:i + max_hold + 1]),
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



# Journal `ts` basis. Rows since 2026-08-20 (20a41db) carry an offset-aware
# ISO ts; OLDER rows were stamped by `datetime.datetime.now().isoformat()` —
# naive LOCAL wall clock of the writing box (trade_journal.py documents the
# change). A naive ts is therefore read as local time, exactly like llm_eval,
# journal_stats and chart_core do (G6-2, 2026-09-26: reading it as UTC
# replayed every legacy row one UTC offset early). None = the system zone
# (`datetime.astimezone()`, DST-aware); tests pin a zone via this hook.
_LOCAL_TZ = None


def _naive_local_to_utc(ts):
    """pd.Timestamp -> tz-aware UTC Timestamp. Offset-aware input is only
    converted; naive input is interpreted in the local zone (see _LOCAL_TZ)."""
    import pandas as pd
    if ts.tz is not None:
        return ts.tz_convert('UTC')
    dt = ts.to_pydatetime(warn=False)
    aware = dt.replace(tzinfo=_LOCAL_TZ) if _LOCAL_TZ is not None else dt.astimezone()
    return pd.Timestamp(aware).tz_convert('UTC')


def _dedup_first_per_day(rows: list[dict], key_fields: list[str],
                         ts_key: str = 'ts') -> list[dict]:
    """Keep only the FIRST row per (key_fields..., calendar-day) — a symbol
    skipped for the same reason (or exited on a signal_sell) every cycle of
    a day otherwise emits ~24 overlapping 24-bar replays/day, double-
    counting the same underlying P&L into saved_total_pct /
    given_up_total_pct. Rows are sorted by ts first so "first" means
    earliest-in-day, matching what the live system actually saw first.

    2026-07b fix: the old pd.Timestamp.max sentinel is tz-NAIVE, so
    sorted() raised `TypeError: Cannot compare tz-naive and tz-aware
    timestamps` the instant the row list mixed naive and aware ts (any
    --days 30 window straddling the 2026-07-13 offset-aware ts change) or
    contained any missing/garbage-ts row. This rewrites the sort key as a
    single int64 ns-since-epoch value (naive read as the writer's LOCAL
    wall clock via `_naive_local_to_utc` — the SAME convention
    `_replay_grouped` applies; G6-2), which totally orders regardless
    of tz-awareness; unparseable/NaT rows get the max int64 sentinel so
    they sort last and bucket under day=None, never silently merged with a
    well-formed row. Sort stability preserves first-wins for equal ts, and
    for uniform-tz inputs the ordering is identical to before (the int64
    key is a monotonic transform of the timestamp).
    """
    import pandas as pd
    _SENTINEL = 2 ** 63 - 1   # sorts last; int key is tz-total

    def _parse(r):
        try:
            t = pd.Timestamp(r[ts_key])
        except (ValueError, KeyError, TypeError):
            return (_SENTINEL, None)
        if t is pd.NaT:
            return (_SENTINEL, None)
        # int64 ns sort key: aware -> UTC epoch; naive -> LOCAL wall clock
        # (legacy writer), the SAME convention _replay_grouped applies. Day
        # bucket stays the row's OWN wall-clock date (matches the journal
        # file naming).
        try:
            key = int(_naive_local_to_utc(t).value)
        except (ValueError, OverflowError, OSError):
            return (_SENTINEL, None)
        return (key, t.date().isoformat())

    decorated = sorted(((_parse(r), r) for r in rows), key=lambda p: p[0][0])
    seen = set()
    out = []
    for (_ns, day), r in decorated:
        key = tuple(r.get(f) for f in key_fields) + (day,)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _replay_grouped(rows: list[dict], api, ts_key: str = 'ts',
                    bars_cache: dict | None = None,
                    crypto_limit: int | None = None):
    """Shared fetch-once-per-symbol replay used by gate_attribution,
    signal_exit_audit and conviction_calibration.

    Distinguishes THREE ways a row fails to price: FETCH-FAILURE (the bars
    call raised, or came back None/too-short — an infra problem that tells
    us nothing about the gate/exit being priced), OUT-OF-WINDOW (ts
    predates the fetched frame — previously silently priced from bar 0,
    identical to an entry at the frame's first bar) and HORIZON-PENDING
    (replay_entry returned None because the exit horizon hasn't resolved
    yet against the bars we DID get — expected, and self-heals as more
    bars land). Pre-2026-07, all of these were folded into one ambiguous
    "unresolved" bucket (or, in conviction_calibration, silently dropped
    with no counter at all).

    bars_cache: optional {symbol: (bars, atr_full, eod_full) | None} dict,
    shared by run_report across the three analyses (gate_attribution,
    conviction_calibration, signal_exit_audit) so a symbol touched by more
    than one of them is fetched only ONCE per report — market_data's 20s
    TTL cannot span a whole report. A cached None records a fetch failure
    so every section stays consistent about which symbols are unavailable.

    crypto_limit: bars to request for crypto symbols (run_report sizes
    this to the report window); None keeps market_data's own default
    (250). The bars_cache key is per-call (this dict is fresh per
    run_report invocation), so a wider report-driven fetch can never
    poison a live-loop's own cache entries — those live in market_data's
    separate module-level cache, untouched here.

    Returns (samples, n_fetch_failed, n_horizon_pending, n_out_of_window)
    where samples is a list of (row, net) for every row that priced
    successfully. ts parsing: offset-aware ISO is exact; a naive (legacy,
    pre-2026-08-20) ts is the writer's LOCAL wall clock (G6-2); a row with an unparseable ts can't be
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
    n_out_of_window = 0
    for sym, srows in by_symbol.items():
        asset = 'crypto' if _is_crypto(sym) else 'stock'
        if bars_cache is not None and sym in bars_cache:
            prepared = bars_cache[sym]
        else:
            try:
                # closed_only: Stage-0 replays must price against completed
                # bars only, or exits land on a partial bar and the report is
                # irreproducible (c26 D38 handoff; measurement-only).
                if asset == 'crypto':
                    bars = (fetch_bars_alpaca(api, sym, limit=crypto_limit,
                                              closed_only=True)
                            if crypto_limit
                            else fetch_bars_alpaca(api, sym, closed_only=True))
                else:
                    bars = fetch_stock_bars_alpaca(api, sym, closed_only=True)
            except Exception:
                bars = None
            if bars is None or len(bars) < 30:
                prepared = None
            else:
                if bars.index.tz is None:
                    bars = bars.tz_localize('UTC')
                prepared = (bars, _atr14(bars), _eod_mask(bars.index, asset))
            if bars_cache is not None:
                bars_cache[sym] = prepared
        if prepared is None:
            n_fetch_failed += len(srows)
            continue
        bars, atr_full, eod_full = prepared
        frame_start = bars.index[0]
        for r in srows:
            try:
                ts = pd.Timestamp(r[ts_key])
                if ts is pd.NaT:
                    raise ValueError('NaT ts')
                ts = _naive_local_to_utc(ts)   # naive = legacy LOCAL ts (G6-2)
            except (ValueError, KeyError, TypeError, OverflowError, OSError):
                n_fetch_failed += 1
                continue
            if ts < frame_start:
                n_out_of_window += 1
                continue
            net = replay_entry(bars, ts, asset, spread_pct=r.get('spread_pct'),
                               atr_full=atr_full, eod_full=eod_full)
            if net is None:
                n_horizon_pending += 1
                continue
            samples.append((r, net))
    return samples, n_fetch_failed, n_horizon_pending, n_out_of_window


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


# ---------------------------------------------------------------------------
# Day-cluster bootstrap (INTEL W17 / Scout B E3, 2026-09 — REPORT-ONLY).
# Episode dedup removes intraday overlap but NOT same-day cross-name market
# moves: every name vetoed on a risk-off day shares that day's shock, so the
# iid percentile CI above treats correlated episodes as independent draws and
# can over-issue REVIEW/OK/CHANGE verdicts (MacKinnon-Nielsen-Webb 2023: what
# counts is the number of effectively independent clusters). Beside every
# `ci90` the report now carries `ci90_dayclust` (resample CALENDAR DAYS with
# replacement), `n_days`, and — where a verdict exists — `verdict_dayclust`
# from the SAME verdict function. Nothing reads these to decide anything;
# the existing ci90/verdict/insufficient_n keys and every pre-existing printed
# line are unchanged. Pre-registered rule: E3_RULE below (owner decision).
# ---------------------------------------------------------------------------
E3_RULE = ('if ≥20% on the first post-retrain 30-day report → owner: switch '
           'the GUI verdict source to the day-cluster CI and raise '
           'MIN_VERDICT_N to 20')
DAYCLUST_FEW_DAYS = 3   # n_days below this => 'dayclust_few_days' flag


def _row_day(r: dict, ts_key: str = 'ts'):
    """Calendar-day cluster key of a priced row: the row's OWN wall-clock
    date as journaled — the SAME day convention as `_dedup_first_per_day`'s
    bucket (`t.date().isoformat()` on the unconverted pd.Timestamp; naive
    legacy ts = the writer's local wall clock, aware ts = the date in its own
    offset, which matches the journal file naming). The UTC conversion via
    `_naive_local_to_utc`/`_LOCAL_TZ` is used there only for the sort key,
    never for the day bucket, so it is not applied here either. None when the
    ts is missing/unparseable (such rows cannot be priced anyway —
    `_replay_grouped` counts them as fetch failures)."""
    import pandas as pd
    try:
        t = pd.Timestamp(r[ts_key])
        if t is pd.NaT:
            return None
        return t.date().isoformat()
    except (ValueError, KeyError, TypeError, OverflowError, OSError):
        return None


def _day_cluster_ci(values, days, n_boot: int = 2000, alpha: float = 0.10,
                    seed: int = 0):
    """Percentile CI for the mean from a CLUSTER bootstrap that resamples
    calendar days with replacement (n_boot, alpha and the fixed seed match
    `_bootstrap_ci`, so numbers reproduce run-to-run).

    Each bootstrap replicate draws n_days days with replacement and takes
    the pooled mean over every row of the drawn days (sum of day-sums / sum
    of day-counts — the usual cluster-bootstrap ratio estimator). Clusters
    keep first-appearance order, so when every row sits on its own day the
    draws are exactly `_bootstrap_ci`'s. NaN values are dropped (with their
    day) exactly like `_bootstrap_ci`; a row with day None is its own
    singleton cluster (no evidence it shares a day with anything).

    Returns (ci_or_None, n_days, reason_or_None). ci is None when fewer than
    two distinct days remain — resampling one cluster has zero spread, so a
    "CI" would be a point masquerading as an interval.
    """
    vals = np.asarray(values, dtype=float)
    days = list(days)
    if len(days) != vals.size:
        raise ValueError(f'days ({len(days)}) and values ({vals.size}) differ in length')
    index: dict = {}
    sums: list = []
    counts: list = []
    for i, (v, d) in enumerate(zip(vals.tolist(), days)):
        if v != v:          # NaN
            continue
        key = ('day', d) if d is not None else ('row', i)
        j = index.get(key)
        if j is None:
            index[key] = len(sums)
            sums.append(v)
            counts.append(1)
        else:
            sums[j] += v
            counts[j] += 1
    n_days = len(sums)
    if n_days == 0:
        return None, 0, 'no priced rows'
    if n_days == 1:
        return None, 1, ('single distinct day — day-cluster bootstrap '
                         'undefined (one cluster has zero resampling spread)')
    s_arr = np.asarray(sums, dtype=float)
    c_arr = np.asarray(counts, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_days, size=(n_boot, n_days))
    boot_means = s_arr[idx].sum(axis=1) / c_arr[idx].sum(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return [round(float(lo), 3), round(float(hi), 3)], n_days, None


def _dayclust_fields(vals, days, verdict_fn=None) -> dict:
    """The additive day-cluster keys for one ci90-bearing dict:
    ci90_dayclust / n_days / dayclust_few_days (+ ci90_dayclust_reason when
    the CI is null, + verdict_dayclust when `verdict_fn` — the SAME
    _gate_verdict/_signal_exit_verdict that produced 'verdict' — is given).
    With a null day-cluster CI at n >= MIN_VERDICT_N the honest day-cluster
    verdict is 'cannot conclude'; below MIN_VERDICT_N the verdict function's
    own insufficient-n refusal is used unchanged."""
    arr = np.asarray(vals, dtype=float)
    n = int(arr.size)
    ci_dc, n_days, reason = _day_cluster_ci(arr, days)
    out = {'ci90_dayclust': ci_dc, 'n_days': int(n_days),
           'dayclust_few_days': bool(n_days < DAYCLUST_FEW_DAYS)}
    if reason is not None:
        out['ci90_dayclust_reason'] = reason
    if verdict_fn is not None:
        if ci_dc is None and n >= MIN_VERDICT_N:
            out['verdict_dayclust'] = f'cannot conclude ({reason}) — no day-cluster verdict'
        else:
            out['verdict_dayclust'] = verdict_fn(
                n, ci_dc if ci_dc is not None else (float('nan'), float('nan')))
    return out


def _verdict_key(v) -> str:
    """Verdict class for disagreement counting: the leading words before
    any ' (' / ' —' qualifier ('REVIEW', 'OK', 'CHANGE', 'NO CHANGE',
    'cannot conclude', 'insufficient n'), so explanatory wording can never
    count as a disagreement — only a different call can."""
    return str(v).split(' (')[0].split(' —')[0].strip()


def verdict_disagreement(gates: dict, sig_exit: dict) -> dict:
    """Report-level iid-vs-day-cluster verdict disagreement (E3).

    Compared set: every gate_attribution gate plus the signal-exit audit
    whose priced n >= MIN_VERDICT_N and which carries both 'verdict' and
    'verdict_dayclust'. rate = n_disagree / n_compared (None when nothing
    qualifies). Report-only: nothing reads it to gate anything."""
    items = []
    for name, g in (gates or {}).items():
        if str(name).startswith('_') or not isinstance(g, dict):
            continue
        items.append((str(name), g, g.get('vetoes_priced')))
    if isinstance(sig_exit, dict) and 'verdict' in sig_exit:
        items.append(('signal_exit', sig_exit, sig_exit.get('priced')))
    compared, disagreeing = [], []
    for name, g, n in items:
        if not isinstance(n, int) or n < MIN_VERDICT_N:
            continue
        if 'verdict' not in g or 'verdict_dayclust' not in g:
            continue
        compared.append(name)
        if _verdict_key(g['verdict']) != _verdict_key(g['verdict_dayclust']):
            disagreeing.append(name)
    rate = round(len(disagreeing) / len(compared), 3) if compared else None
    return {'rate': rate, 'n_compared': len(compared),
            'n_disagree': len(disagreeing), 'disagreeing': disagreeing,
            'min_verdict_n': MIN_VERDICT_N, 'rule': E3_RULE}


def _disagreement_line(d: dict) -> str:
    """The ONE printed line the day-cluster addition contributes."""
    if d.get('rate') is None:
        head = (f"verdict_disagreement_rate (iid vs day-cluster CI): n/a "
                f"(0 verdicts with n>={d.get('min_verdict_n', MIN_VERDICT_N)})")
    else:
        head = (f"verdict_disagreement_rate (iid vs day-cluster CI): "
                f"{d['rate']:.0%} ({d['n_disagree']}/{d['n_compared']} verdicts "
                f"with n>={d.get('min_verdict_n', MIN_VERDICT_N)})")
    return f"{head} — pre-registered E3 rule: {d.get('rule', E3_RULE)}"


def _bucket_stats(vals, days=None) -> dict:
    arr = np.asarray(vals, dtype=float)
    ci = _bootstrap_ci(arr)
    out = {'n': int(arr.size),
           'mean_net_pct': round(float(arr.mean()), 3),
           'hit_rate': round(float((arr > 0).mean()), 3),
           'ci90': [ci[0], ci[1]],
           'insufficient_n': bool(arr.size < MIN_VERDICT_N)}
    if days is not None:   # additive day-cluster keys (report-only, E3)
        out.update(_dayclust_fields(arr, days))
    return out


def _gate_verdict(n: int, ci) -> str:
    """A gate reads REVIEW (charging admission) only when its bootstrap CI
    excludes zero on the positive side; OK only when it excludes zero on
    the negative side. Anything else is an honest 'cannot conclude' rather
    than reading the raw sign of a noisy mean. Below MIN_VERDICT_N there
    are too few priced episodes for the CI to mean anything — refuse to
    verdict at all rather than let a single noisy row read as REVIEW/OK."""
    if n < MIN_VERDICT_N:
        return f'insufficient n (n={n} < {MIN_VERDICT_N}) — no verdict'
    lo, hi = ci
    if lo > 0:
        return 'REVIEW (charging admission — CI excludes zero)'
    if hi < 0:
        return 'OK (earning its keep — CI excludes zero)'
    return 'cannot conclude (CI spans zero)'


def _signal_exit_verdict(n: int, ci) -> str:
    if n < MIN_VERDICT_N:
        return f'insufficient n (n={n} < {MIN_VERDICT_N}) — no verdict'
    lo, hi = ci
    if lo > 0:
        return 'CHANGE — apply 2-reading confirmation to signal exits (CI excludes zero)'
    if hi < 0:
        return 'NO CHANGE — the flip is saving money (CI excludes zero)'
    return 'cannot conclude (CI spans zero)'


def gate_attribution(rows: list[dict], api, bars_cache: dict | None = None,
                     crypto_limit: int | None = None) -> dict:
    """Counterfactual P&L per gate from journaled skips.

    2026-07: rows are deduped to one PER (symbol, skip_reason,
    calendar-day) EPISODE before replay — a symbol skipped for the same
    reason every cycle of a day otherwise emits ~24 overlapping 24-bar
    replays/day, double-counting the same underlying P&L into
    saved_total_pct. 'vetoes_priced' counts episodes (what's replayed and
    summed); 'vetoes_raw' is the pre-dedup journal row count, reported
    alongside for transparency.

    bars_cache/crypto_limit are forwarded to `_replay_grouped` (see its
    docstring) so run_report's three analyses share one fetch per symbol.
    """
    skips_raw = [r for r in rows if r.get('action') == 'skip'
                 and r.get('skip_reason') in GATE_REASONS and r.get('symbol')]
    raw_by_gate: dict[str, int] = defaultdict(int)
    for r in skips_raw:
        raw_by_gate[r['skip_reason']] += 1
    skips = _dedup_first_per_day(skips_raw, ['symbol', 'skip_reason'])

    samples, n_fetch_failed, n_horizon_pending, n_out_of_window = _replay_grouped(
        skips, api, bars_cache=bars_cache, crypto_limit=crypto_limit)
    per_gate: dict[str, list[float]] = defaultdict(list)
    per_gate_days: dict[str, list] = defaultdict(list)
    cf_priced = 0
    cf_with_spread = 0
    for r, net in samples:
        per_gate[r['skip_reason']].append(net)
        per_gate_days[r['skip_reason']].append(_row_day(r))
        if r['skip_reason'] == 'cost_floor':
            cf_priced += 1
            if r.get('spread_pct') is not None:
                cf_with_spread += 1

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
            'verdict': _gate_verdict(int(vals.size), ci),
            'insufficient_n': bool(vals.size < MIN_VERDICT_N),
        }
        out[gate].update(_dayclust_fields(
            vals, per_gate_days.get(gate, []), verdict_fn=_gate_verdict))

    # Gates that fired (raw journal rows exist) but priced ZERO episodes —
    # e.g. every fetch for that symbol failed. Distinct from UNPRICED_GATES
    # (never priced per-symbol AT ALL, by design).
    seen_unpriced = {g: int(raw_by_gate[g]) for g in GATE_REASONS
                     if raw_by_gate.get(g) and g not in out}
    # skip_reasons the journal carries that this module doesn't recognize —
    # either a brand-new gate the producers added (drift) or a typo.
    unclassified = defaultdict(int)
    for r in rows:
        if r.get('action') == 'skip':
            reason = r.get('skip_reason')
            if reason and reason not in GATE_REASONS and reason not in UNPRICED_GATES:
                unclassified[reason] += 1

    out['_fetch_failed'] = n_fetch_failed
    out['_horizon_pending'] = n_horizon_pending
    out['_out_of_window'] = n_out_of_window
    out['_unresolved'] = n_fetch_failed + n_horizon_pending + n_out_of_window
    out['_cost_floor_spread_coverage'] = (round(cf_with_spread / cf_priced, 3)
                                          if cf_priced else None)
    # True when ANY priced cost_floor episode lacked a journaled spread
    # (was: when NONE had one — a single modern row silenced the caveat
    # for a whole mixed-vintage window)
    out['_cost_floor_flat_spread'] = ('cost_floor' in out) and cf_with_spread < cf_priced
    if seen_unpriced:
        out['_gates_seen_unpriced'] = seen_unpriced
    if unclassified:
        out['_unclassified_skip_reasons'] = dict(
            sorted(unclassified.items(), key=lambda kv: -kv[1]))
    return out


def signal_exit_audit(rows: list[dict], api, bars_cache: dict | None = None,
                      crypto_limit: int | None = None) -> dict:
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

    priced, n_fetch_failed, n_horizon_pending, n_out_of_window = _replay_grouped(
        sells, api, bars_cache=bars_cache, crypto_limit=crypto_limit)
    nets, realized, net_days = [], [], []
    for r, net in priced:
        nets.append(net)
        net_days.append(_row_day(r))
        if isinstance(r.get('pnl_pct'), (int, float)):
            realized.append(float(r['pnl_pct']))

    vals = np.asarray(nets, dtype=float)
    out = {'n_signal_sells': len(sells_raw), 'episodes': len(sells),
           'priced': int(vals.size),
           '_fetch_failed': n_fetch_failed, '_horizon_pending': n_horizon_pending,
           '_out_of_window': n_out_of_window,
           '_unresolved': n_fetch_failed + n_horizon_pending + n_out_of_window}
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
            'verdict': _signal_exit_verdict(int(vals.size), ci),
            'insufficient_n': bool(vals.size < MIN_VERDICT_N),
        })
        out.update(_dayclust_fields(vals, net_days,
                                    verdict_fn=_signal_exit_verdict))
    return out


def conviction_calibration(rows: list[dict], api, bars_cache: dict | None = None,
                           crypto_limit: int | None = None) -> dict:
    """Realized counterfactual outcome of TAKEN entries by conviction
    bucket (prediction-magnitude tercile x meta-probability band).

    Uses the same kernel replay as the vetoes so taken and vetoed trades
    are priced identically (journaled realized exits depend on signal
    exits the replay can't see; the kernel replay is the comparable
    yardstick). 2026-07: fetch-failures, horizon-pending and out-of-window
    rows are all counted (_fetch_failed/_horizon_pending/_out_of_window)
    instead of silently shrinking n with no trace; buys are discrete fills
    already, so unlike the two gate-side analyses these rows are NOT
    deduped. Rank buckets are suppressed below MIN_BUCKET_N (see
    rank_coverage / _rank_buckets_suppressed) so a single stray rank-6-7
    trade can't drive a rank_gradient_verdict on its own.

    2026-09: buys journaled with ``pred_return: null`` (every legacy crypto
    buy) cannot be bucketed by prediction magnitude; they are still excluded
    from the replay but now COUNTED in ``_dropped_null_pred`` (and surfaced
    by run_report's banner + quality block) instead of silently shrinking
    the denominator."""
    all_buys = [r for r in rows if r.get('action') == 'buy' and r.get('symbol')]
    buys = [r for r in all_buys if r.get('pred_return') is not None]
    n_null_pred = len(all_buys) - len(buys)
    if not buys:
        return {'_fetch_failed': 0, '_horizon_pending': 0,
                '_out_of_window': 0, '_unresolved': 0,
                '_dropped_null_pred': n_null_pred}

    priced, n_fetch_failed, n_horizon_pending, n_out_of_window = _replay_grouped(
        buys, api, bars_cache=bars_cache, crypto_limit=crypto_limit)
    samples = []   # (pred, meta_prob_or_None, net, rank_or_None, day_or_None)
    n_bad_pred = 0
    n_stock_rank = n_crypto_rank = 0
    for r, net in priced:
        # meta_p is the wave-5 field name; meta_prob the legacy one — a
        # PRESENT-but-null meta_p must still fall back to meta_prob (the
        # dict.get default arg is defeated by an explicit None value).
        mp = r.get('meta_p')
        if mp is None:
            mp = r.get('meta_prob')
        try:
            pv = float(r['pred_return'])
        except (TypeError, ValueError):
            n_bad_pred += 1
            continue
        samples.append((pv, mp, net, r.get('entry_rank'), _row_day(r)))
        # per-asset rank coverage, counted HERE (post pred-validation) so
        # stock_with_rank + crypto_with_rank == n_with_rank always holds
        if r.get('entry_rank') is not None:
            if _is_crypto(r['symbol']):
                n_crypto_rank += 1
            else:
                n_stock_rank += 1

    if len(samples) < 9:
        return {'n': len(samples),
                'note': 'too few resolvable taken entries to bucket',
                '_fetch_failed': n_fetch_failed,
                '_horizon_pending': n_horizon_pending,
                '_out_of_window': n_out_of_window,
                '_unresolved': n_fetch_failed + n_horizon_pending + n_out_of_window,
                '_malformed_pred_return': n_bad_pred,
                '_dropped_null_pred': n_null_pred}

    preds = np.array([s[0] for s in samples])
    nets = np.array([s[2] for s in samples])
    sdays = np.array([s[4] for s in samples], dtype=object)
    out = {'n': len(samples), '_fetch_failed': n_fetch_failed,
           '_horizon_pending': n_horizon_pending,
           '_out_of_window': n_out_of_window,
           '_unresolved': n_fetch_failed + n_horizon_pending + n_out_of_window,
           '_malformed_pred_return': n_bad_pred,
           '_dropped_null_pred': n_null_pred}

    # Prediction-magnitude terciles: does the top third out-earn?
    qs = np.quantile(preds, [1 / 3, 2 / 3])
    buckets = {'pred_low': preds <= qs[0],
               'pred_mid': (preds > qs[0]) & (preds <= qs[1]),
               'pred_high': preds > qs[1]}
    for name, mask in buckets.items():
        if mask.sum():
            out[name] = _bucket_stats(nets[mask], days=sdays[mask])

    # Entry-rank buckets — Stage-0 experiment 1: does rank 6-7 carry
    # materially less edge than rank 1-3? (gates the concentration cap)
    # rank_coverage discloses whether ranks are even present across both
    # asset types — crypto rows historically carry no entry_rank at all.
    ranked = [(s[3], s[2]) for s in samples if s[3] is not None]
    ranked_days = np.array([s[4] for s in samples if s[3] is not None],
                           dtype=object)
    out['rank_coverage'] = {
        'n_total': len(samples), 'n_with_rank': len(ranked),
        'stock_with_rank': n_stock_rank,
        'crypto_with_rank': n_crypto_rank,
    }
    if len(ranked) >= 9:
        rk = np.array([r[0] for r in ranked], dtype=float)
        rn = np.array([r[1] for r in ranked], dtype=float)
        suppressed = {}
        for name, lo, hi in (('rank_1_3', 1, 3), ('rank_4_5', 4, 5),
                             ('rank_6_7', 6, 7), ('rank_8_plus', 8, 9999)):
            mask = (rk >= lo) & (rk <= hi)
            cnt = int(mask.sum())
            if cnt >= MIN_BUCKET_N:
                out[name] = _bucket_stats(rn[mask], days=ranked_days[mask])
            elif cnt:
                suppressed[name] = cnt
        if suppressed:
            out['_rank_buckets_suppressed'] = suppressed

    metas = [(s[1], s[2]) for s in samples if s[1] is not None]
    if len(metas) >= 9:
        mp = np.array([m[0] for m in metas])
        mn = np.array([m[1] for m in metas])
        md = np.array([s[4] for s in samples if s[1] is not None], dtype=object)
        for name, lo, hi in (('meta_0.30_0.45', 0.30, 0.45),
                             ('meta_0.45_0.60', 0.45, 0.60),
                             ('meta_0.60_1.00', 0.60, 1.01)):
            mask = (mp >= lo) & (mp < hi)
            if mask.sum():
                out[name] = _bucket_stats(mn[mask], days=md[mask])
    return out


def admitted_k_distribution(rows: list[dict]) -> dict:
    """Admitted-k distribution from entry_window summary rows — Stage-0
    experiment 2 (does the gate stack already enforce concentration?).
    No API/replay needed; reads the journals directly. Tolerates malformed
    admitted_k/veto_counts values (never lets one corrupt row abort the
    whole report) and surfaces fail-closed no_pred/no_quote vetoes and
    mean candidate count directly, rather than leaving them buried inside
    total_vetoes_by_reason."""
    out = {}
    for asset in ('stock', 'crypto'):
        wins = [r for r in rows if r.get('action') == 'entry_window'
                and r.get('asset_type') == asset]
        if not wins:
            continue
        ks_list, malformed = [], 0
        ncs = []
        for r in wins:
            try:
                ks_list.append(int(r.get('admitted_k', 0)))
            except (TypeError, ValueError):
                malformed += 1
            n_cand = r.get('n_candidates')
            if n_cand is not None:
                try:
                    ncs.append(int(n_cand))
                except (TypeError, ValueError):
                    pass
        if not ks_list:
            continue
        ks = np.array(ks_list)
        veto_tot = defaultdict(int)
        for r in wins:
            for reason, c in (r.get('veto_counts') or {}).items():
                try:
                    veto_tot[reason] += int(c)
                except (TypeError, ValueError):
                    continue
        hist = {str(k): int((ks == k).sum()) for k in range(0, 8)}
        hist['8+'] = int((ks >= 8).sum())
        out[asset] = {
            'windows': len(wins),
            'mean_admitted_k': round(float(ks.mean()), 2),
            'pct_windows_k_ge_6': round(float((ks >= 6).mean()), 3),
            'pct_windows_zero': round(float((ks == 0).mean()), 3),
            'pct_windows_zero_note': ('conditional on >=1 evaluatable candidate; '
                                      'cycles where every symbol dropped before '
                                      'evaluation journal NO entry_window row'),
            'mean_n_candidates': (round(float(np.mean(ncs)), 2) if ncs else None),
            'fail_closed': {'no_pred': int(veto_tot.get('no_pred', 0)),
                            'no_quote': int(veto_tot.get('no_quote', 0))},
            '_malformed_admitted_k': malformed,
            'admitted_k_hist': hist,
            'total_vetoes_by_reason': dict(
                sorted(veto_tot.items(), key=lambda kv: -kv[1])),
        }
    return out


def _write_json(path: Path, obj) -> None:
    """Atomic write: readers (gui.py, scripts/rank_gradient_report.py) must
    never see a truncated file; same-directory os.replace is atomic on
    POSIX, and a json.dumps failure leaves the previous report intact.
    pid-unique tmp (same pattern as events_calendar): the GUI's Decision
    Report button and a manual run can overlap — with a FIXED tmp name one
    process can os.replace the other's half-written tmp into the live path,
    defeating the atomicity this exists for."""
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _report_path(out_dir=None) -> Path:
    """Where decision_report.json goes. Default (out_dir None) is the repo
    root literal BASE_DIR / 'decision_report.json' — the path gui.py and
    scripts/rank_gradient_report.py read, so the default is unchanged.
    `--out DIR` (2026-09) redirects BOTH the stale and the normal write
    under DIR (created), so an evidence run does not overwrite the
    operator's root report."""
    if out_dir is None:
        return BASE_DIR / 'decision_report.json'
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / 'decision_report.json'


def _write_stale_report(days: int, api_available, out_dir=None) -> dict:
    """run_report ALWAYS writes decision_report.json, even when nothing
    could be priced — 2026-07 fix: previously a get_api() failure returned
    {} without touching the file, so scripts/rank_gradient_report.py could
    silently read a STALE prior report as if it were current. Marking
    'stale': True lets downstream consumers refuse to trust it."""
    stale = {'generated': dt.datetime.now().astimezone().isoformat(),
             'days': days, 'api_available': api_available, 'stale': True}
    out = _report_path(out_dir)
    _write_json(out, stale)
    return stale


def run_report(days: int = 30, out_dir=None) -> dict:
    days = max(0, int(days))
    rows = load_journal(days)
    if not rows:
        print("No journal entries found.")
        return _write_stale_report(days, api_available=None, out_dir=out_dir)
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
        return _write_stale_report(days, api_available=False, out_dir=out_dir)

    # One bars/atr/eod fetch per symbol for the WHOLE report — a symbol
    # touched by gates + conviction + signal-exit (common: a name that was
    # both skipped and bought/sold across the window) used to be fetched
    # up to 3x; market_data's own TTL cache (20s) can't span a report.
    bars_cache: dict = {}
    crypto_limit = min(max(250, days * 24 + 48), _CRYPTO_BAR_CAP)
    errors: dict = {}

    def _section(name, fn):
        try:
            return fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors[name] = repr(e)
            return {}

    gates = _section('gates', lambda: gate_attribution(
        rows, api, bars_cache=bars_cache, crypto_limit=crypto_limit))
    conviction = _section('conviction', lambda: conviction_calibration(
        rows, api, bars_cache=bars_cache, crypto_limit=crypto_limit))
    admitted_k = _section('admitted_k', lambda: admitted_k_distribution(rows))
    sig_exit = _section('signal_exit', lambda: signal_exit_audit(
        rows, api, bars_cache=bars_cache, crypto_limit=crypto_limit))

    # --- headline banner (2026-07 review): fetch failures, out-of-window
    # rows and pending horizons must be visible BEFORE anyone reads a
    # verdict below ---
    gate_priced = sum(g['vetoes_priced'] for k, g in gates.items()
                      if not k.startswith('_'))
    gate_unpriced = (gates.get('_fetch_failed', 0) + gates.get('_horizon_pending', 0)
                     + gates.get('_out_of_window', 0))
    sig_priced = sig_exit.get('priced', 0)
    sig_unpriced = (sig_exit.get('_fetch_failed', 0) + sig_exit.get('_horizon_pending', 0)
                    + sig_exit.get('_out_of_window', 0))
    conv_priced = conviction.get('n', 0)
    conv_unpriced = (conviction.get('_fetch_failed', 0) + conviction.get('_horizon_pending', 0)
                     + conviction.get('_out_of_window', 0))
    total_fetch_failed = (gates.get('_fetch_failed', 0)
                          + sig_exit.get('_fetch_failed', 0)
                          + conviction.get('_fetch_failed', 0))
    total_out_of_window = (gates.get('_out_of_window', 0)
                           + sig_exit.get('_out_of_window', 0)
                           + conviction.get('_out_of_window', 0))
    # 2026-09: buys dropped for pred_return=null are unpriced taken entries
    # too — fold them into the denominator so the unpriced rate is honest.
    dropped_null_pred = int(conviction.get('_dropped_null_pred', 0) or 0)
    total_priced = gate_priced + sig_priced + conv_priced
    total_unpriced = gate_unpriced + sig_unpriced + conv_unpriced + dropped_null_pred
    total_considered = total_priced + total_unpriced
    unpriced_rate = (total_unpriced / total_considered) if total_considered else 0.0

    if total_fetch_failed > 0 or unpriced_rate > 0.30:
        print(f"\nWARNING: {unpriced_rate:.0%} of rows unpriced "
              f"({total_fetch_failed} fetch failures, {total_out_of_window} "
              f"out-of-window, {dropped_null_pred} null-pred buys) — sections "
              f"below are NOT representative")
    if dropped_null_pred:
        print(f"WARNING: {dropped_null_pred} buy row(s) dropped from conviction "
              f"calibration for pred_return=null (cannot be bucketed by "
              f"prediction; counted as unpriced above)")
    print("methodology 2026-07b: per-episode dedup + bootstrap CIs + "
          "full-frame ATR stops + out-of-window exclusion — numbers not "
          "comparable to earlier reports")
    if days > STOCK_FRAME_MIN_DAYS:
        print(f"NOTE: stock replay frames hold only the last 320 hourly bars "
              f"(market_data.fetch_stock_bars_alpaca .tail(320), extended "
              f"hours included) ≈ {STOCK_FRAME_MIN_DAYS}-30 calendar days — "
              f"older stock rows are counted _out_of_window, not priced.")

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
    observed_unpriced: dict = {}
    for a in admitted_k.values():
        if isinstance(a, dict):
            for reason, c in (a.get('total_vetoes_by_reason') or {}).items():
                if reason not in GATE_REASONS:
                    observed_unpriced[reason] = observed_unpriced.get(reason, 0) + int(c)
    if observed_unpriced:
        top = sorted(observed_unpriced.items(), key=lambda kv: -kv[1])
        print("Unpriced gates observed this window (entry_window rollups): "
              + ", ".join(f"{k}={v}" for k, v in top))
        print(f"fail-closed vetoes: no_pred={observed_unpriced.get('no_pred', 0)} "
              f"no_quote={observed_unpriced.get('no_quote', 0)} "
              f"(nonzero no_pred = the model returned None for that many candidate-cycles)")
    else:
        print(f"Unpriced gates (never priced per-symbol): {', '.join(UNPRICED_GATES)}")
    for gate, g in gates.items():
        if gate.startswith('_'):
            continue
        lo, hi = g['ci90']
        verdict = g['verdict']
        print(f"{gate:<18} n={g['vetoes_priced']:<4} (raw={g['vetoes_raw']:<4}) "
              f"mean={g['counterfactual_mean_net_pct']:+.2f}% "
              f"[{lo:+.2f}, {hi:+.2f}] hit={g['counterfactual_hit_rate']:.0%} "
              f"saved={g['saved_total_pct']:+.1f}%  -> {verdict}")
    print(f"(fetch failures: {gates.get('_fetch_failed', 0)}, "
          f"horizon-pending: {gates.get('_horizon_pending', 0)}, "
          f"out-of-window: {gates.get('_out_of_window', 0)}, "
          f"unresolved total: {gates.get('_unresolved', 0)})")
    if gates.get('_gates_seen_unpriced'):
        print(f"NOTE: gates that fired but priced ZERO rows: {gates['_gates_seen_unpriced']}")
    if gates.get('_unclassified_skip_reasons'):
        print(f"WARNING: journaled skip_reasons unknown to GATE_REASONS "
              f"(producer drift?): {gates['_unclassified_skip_reasons']}")
    if gates.get('_cost_floor_flat_spread'):
        cov = gates.get('_cost_floor_spread_coverage')
        flat_pct = f"{(1 - cov):.0%}" if isinstance(cov, (int, float)) else "some"
        print(f"NOTE: cost_floor priced at FLAT spread for {flat_pct} of its "
              f"episodes but it fires on WIDE-spread names — its 'charging "
              f"admission' reading is structurally unreliable for those rows. "
              f"Both loops journal spread_pct on cost_floor skips since "
              f"2026-07-13; flat-priced rows predate that (or journaling was off).")
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
              f"{conviction.get('_horizon_pending', 0)}, out-of-window="
              f"{conviction.get('_out_of_window', 0)})")
    rc = conviction.get('rank_coverage') or {}
    if rc.get('n_with_rank') and rc.get('crypto_with_rank') == 0:
        print("NOTE: rank buckets are STOCK-ONLY this window — crypto buy "
              "rows carry no entry_rank (base_loop._conv_fields is called "
              "without rank).")

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
        print(f"Verdict: {sig_exit['verdict']}")
        print("Audit bias disclosure: this counterfactual models signal "
              "exits fully DISABLED (stops/trail/TP/EOD manage the whole "
              "hold) — a STRONGER intervention than the 2-reading "
              "confirmation actually under decision, so a CHANGE verdict "
              "here is an upper bound on the case for change, not proof "
              "the milder fix earns as much.")

    # E3 (report-only): iid-vs-day-cluster verdict disagreement — the ONE
    # new printed line; every line above/below is unchanged.
    disagreement = verdict_disagreement(gates, sig_exit)
    print(_disagreement_line(disagreement))

    report = {'generated': dt.datetime.now().astimezone().isoformat(),
              'days': days, 'gates': gates, 'conviction': conviction,
              'admitted_k': admitted_k, 'signal_exit': sig_exit}
    report['verdict_disagreement_rate'] = disagreement['rate']
    report['verdict_disagreement'] = disagreement
    report['quality'] = {
        'rows_loaded': len(rows), 'priced': total_priced,
        'unpriced': total_unpriced, 'fetch_failed': total_fetch_failed,
        'horizon_pending': (total_unpriced - total_fetch_failed
                            - total_out_of_window - dropped_null_pred),
        'out_of_window': total_out_of_window,
        'dropped_null_pred': dropped_null_pred,
        'unpriced_rate': round(unpriced_rate, 3),
        'representative': bool(total_priced > 0 and total_fetch_failed == 0
                               and unpriced_rate <= 0.30),
    }
    flags = {}
    try:
        import strategy_config
        flags['conviction_journal_enabled'] = bool(strategy_config.CONVICTION_JOURNAL_ENABLED)
    except Exception:
        flags['conviction_journal_enabled'] = None
    try:
        _cfg = BASE_DIR / 'llm_config.json'
        flags['llm_journal_enabled'] = (bool(json.loads(_cfg.read_text()).get('journal_enabled', True))
                                        if _cfg.exists() else True)
    except Exception:
        flags['llm_journal_enabled'] = None
    report['journal_flags'] = flags
    if flags.get('conviction_journal_enabled') is False or flags.get('llm_journal_enabled') is False:
        print("WARNING: journaling is DISABLED (see journal_flags) — absent "
              "gates below mean NOT MEASURED, not 'never fired'.")

    if errors:
        report['errors'] = errors
        report['stale'] = True
        report['stale_reason'] = 'analysis error: ' + ', '.join(sorted(errors))
        report['api_available'] = True
    elif total_considered > 0 and total_priced == 0:
        report['stale'] = True
        report['stale_reason'] = 'no rows priced (fetch failures / out-of-window / pending)'
        report['api_available'] = True

    out = _report_path(out_dir)
    _write_json(out, report)
    print(f"\nReport: {out}")
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Gate attribution report')
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--out', metavar='DIR', default=None,
                    help='write decision_report.json under DIR (created) '
                         'instead of the repo root (default: repo root, '
                         'the path gui.py reads)')
    args = ap.parse_args()
    run_report(args.days, out_dir=args.out)
