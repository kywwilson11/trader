"""Challenger shadow mode — promote retrained models on LIVE evidence.

Static gates (purged-CV holdout, DSR) catch overfit models, but they
score the challenger on the same historical data the search saw the tail
of. The decisive test is forward: run the challenger SILENTLY alongside
the champion on live data for 14-28 days, then promote only if its
prediction errors are significantly smaller by a Diebold-Mariano test
with the Harvey-Leybourne-Newbold small-sample correction.

Mechanics:
  - hypersearch --shadow saves a gated new model under the challenger
    prefix ('challenger' / 'stock_challenger') when a champion exists;
    a newer retrain simply replaces the challenger (clock restarts).
  - The bots log hourly side-by-side predictions (maybe_log_shadow) —
    one extra prediction round per hour, no trading impact.
  - The pipeline's daily check calls evaluate_and_maybe_promote():
    errors are variance-normalized squared errors on each model's OWN
    forward-bars horizon (skill score, comparable across horizons);
    DM uses Newey-West LRV with lag = horizon-1 (h-step forecasts are
    MA(h-1)).
  - Every eval cycle (and the terminal promote/discard outcome) is
    persisted to {prefix}shadow_status.json for the GUI's promotion-story
    panel — instrumentation only, best-effort, cannot affect the
    decision below (see _write_shadow_status).
  - Decisions (both promote branches also require n >= MIN_OBS=200):
    promote early after >=14d at p<0.05; at >=28d promote at p<0.10 if
    the mean loss diff favors the challenger, else discard.
    Status-quo bias is intentional — promotion churn has real costs.
  - KNOWN anti-conservative approximations (documented, not corrected —
    correcting them changes promotion behavior, owner decision): (1) the
    records pool K symbols per hour (crypto ~6, stocks ~56) but dm_hln
    truncates the Newey-West LRV at h-1 in RECORD units, so same-hour
    cross-sectional correlation and the ~K*(h-1)-record overlap
    dependence are under-covered (for stocks lag 23 spans <1 hour of
    records) — the LRV is underestimated and the DM stat inflated;
    (2) the daily eval re-tests an unadjusted p<0.05 from day 14
    (~14 sequential looks); (3) MIN_OBS counts pooled records, not
    independent hours. Fix sketch on file: collapse d to per-timestamp
    means, lag h-1 in hours, per-book MIN_OBS, spaced peeks.
  - That fix now exists as DM v2 behind TRADER_SHADOW_DM_V2 (default OFF):
    per-hour cross-sectional collapse, Ibragimov-Muller cluster t
    (block=2*h_max, q>=6) against Student t_{q-1}, two scheduled looks
    (day 21 alpha=0.025, final alpha=0.10 + mean_d>0), MIN collapsed
    timestamps 8*h_max, skill-score variance frozen from a >=90d trailing
    window with an (n_eff-2)/n_eff small-sample patch fallback, and a
    stock final look at 56d via Kiefer-Vogelsang fixed-b DM (M=2*h_max).
    OFF = legacy decides while v2 is computed + logged side-by-side into
    shadow_status.json (dm_v2_* additive fields). ON = v2 decides. See
    research/campaign_2026-08/02_research.md B03.3.
  - Promotion copies the full artifact stack (LSTM, scaler, config,
    feature cols, LGB, q10) with .prev backups, manifest LAST; the
    champion's meta-label artifacts are moved aside to .stale (stale
    pairing — the gate fails open to neutral) and a guarded staged meta
    retrain is kicked off in the background.
"""

import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from log_config import get_logger

logger = get_logger(__name__)

MIN_SHADOW_DAYS = 14
MAX_SHADOW_DAYS = 28
MIN_OBS = 200
EARLY_PROMOTE_P = 0.05
FINAL_PROMOTE_P = 0.10
SHADOW_LOG_INTERVAL_SEC = 3590   # ~hourly (the model's native bar rate)

# --- DM v2 (D34 rebuild, 02_research.md B03.3) — default OFF; pattern per
# funding.py's TRADER_FUNDING_Z_TIME_THINNING. OFF: legacy decides, v2 is
# computed and logged side-by-side. ON: v2 decides.
DM_V2_ENABLED = os.environ.get(
    'TRADER_SHADOW_DM_V2', '0').strip().lower() in ('1', 'true', 'yes')
V2_LOOK1_DAYS = 21           # interim look (crypto only)
V2_LOOK1_ALPHA = 0.025       # Lan-DeMets-calibrated two-look budget
V2_FINAL_ALPHA = 0.10        # final look, additionally requires mean_d > 0
V2_STOCK_MAX_SHADOW_DAYS = 56  # T~130 RTH bars at 28d is unsound (B03.3)
V2_MIN_TS_MULT = 8           # min collapsed timestamps = 8 * h_max
V2_BLOCK_MULT = 2            # IM block length = 2 * h_max (never 24h/h_max)
V2_MIN_BLOCKS = 6            # IM refuses below q = 6
V2_FROZEN_MIN_BARS = {'crypto': 2160, 'stock': 630}  # ~90d per book

_ARTIFACT_SUFFIXES = [
    'model_v2.pth', 'config_v2.pkl', 'scaler_v2.pkl', 'feature_cols_v2.pkl',
    'lgb_model.txt', 'lgb_q10.txt', 'lgb_q10_meta.json',
    # OOF predictions ride the stack (D12): fingerprinted to the manifest's
    # saved_at+score, so the copied npz stays valid for the promoted champion.
    'oof_preds.npz',
]
_STALE_META_SUFFIXES = ['meta_model.txt', 'meta_calib.pkl', 'meta_meta.json']


def challenger_prefix(prefix: str) -> str:
    """'': 'challenger'; 'stock': 'stock_challenger'."""
    return f'{prefix}_challenger' if prefix else 'challenger'


def _p(prefix: str) -> str:
    return f'{prefix}_' if prefix else ''


def champion_exists(prefix: str) -> bool:
    return (BASE_DIR / f'{_p(prefix)}model_v2.manifest.json').exists()


def challenger_manifest(prefix: str) -> Path:
    return BASE_DIR / f'{challenger_prefix(prefix)}_model_v2.manifest.json'


def shadow_log_file(prefix: str) -> Path:
    return BASE_DIR / f'{_p(prefix)}shadow_preds.jsonl'


def shadow_status_file(prefix: str) -> Path:
    """Latest shadow-evaluation snapshot for the GUI (gui_review_2026-07
    §7 challenger cell — Phase 2.2 producer side). Written by
    evaluate_and_maybe_promote on every daily eval cycle, never by the
    bots; same {prefix}-naming convention as shadow_log_file."""
    return BASE_DIR / f'{_p(prefix)}shadow_status.json'


def promotion_ledger_file(prefix: str) -> Path:
    """Append-only JSONL: one row per terminal/held promotion decision
    (c26 T7 / B19) — the substrate for post-promotion P&L attribution.
    {prefix}-named like shadow_log_file: promotion_ledger.jsonl /
    stock_promotion_ledger.jsonl."""
    return BASE_DIR / f'{_p(prefix)}promotion_ledger.jsonl'


def _v2_state_file(prefix: str) -> Path:
    """DM v2 look-ledger ({'cm', 'look1_done', 'ts'}) — written ONLY under
    TRADER_SHADOW_DM_V2 (the interim look is one-shot per challenger)."""
    return BASE_DIR / f'{_p(prefix)}shadow_v2_state.json'


def _v2_look1_done(prefix: str) -> bool:
    """True iff the CURRENT challenger already consumed its interim look.

    Keyed on the challenger manifest mtime (same identity token as the
    shadow rows' 'cm'). Any parse/stat error -> False: a lost state file
    re-tests look 1 once — bounded, and only ever reachable under flag ON.
    """
    try:
        with open(_v2_state_file(prefix)) as f:
            state = json.load(f)
        cm = int(challenger_manifest(prefix).stat().st_mtime)
        return bool(state.get('cm') == cm and state.get('look1_done'))
    except Exception:
        return False


def _v2_mark_look1_done(prefix: str) -> None:
    """Record the consumed interim look (tmp + os.replace, best-effort).
    NEVER called when DM_V2_ENABLED is False."""
    try:
        cm = int(challenger_manifest(prefix).stat().st_mtime)
        path = _v2_state_file(prefix)
        tmp = f'{path}.tmp'
        with open(tmp, 'w') as f:
            json.dump({'cm': cm, 'look1_done': True, 'ts': time.time()}, f)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("[SHADOW] v2 look-state write failed: %s", e)


# --- Bot-side: hourly side-by-side prediction logging ---

def maybe_log_shadow(loop, champ_preds: dict, benchmark) -> None:
    """Hourly challenger predictions for every symbol the champion just
    scored. Never raises; never trades. Called from _run_one_cycle."""
    if not champ_preds:
        return
    now = time.time()
    if now - getattr(loop, '_shadow_last_ts', 0) < SHADOW_LOG_INTERVAL_SEC:
        return
    prefix = loop.MODEL_PREFIX
    man = challenger_manifest(prefix)
    if not man.exists():
        return
    loop._shadow_last_ts = now  # set first: a broken challenger must not
    #                             retry every 30s cycle

    cp = challenger_prefix(prefix)
    try:
        mtime = int(man.stat().st_mtime)
        stack = getattr(loop, '_shadow_stack', None)
        if stack is None or stack[0] != mtime:
            import predict_now
            # New challenger artifacts: predict_now's per-prefix LGB/q10
            # booster caches pair with the OLD generation — drop them so the
            # blend reloads from disk (mirrors base_loop._hot_reload_check,
            # which only pops the champion prefix). Keys are challenger-only;
            # champion entries and combined-bot threads are untouched.
            predict_now._lgb_models.pop(cp, None)
            predict_now._q10_models.pop(cp, None)
            _prune_stale_rows(prefix, mtime)
            model, scaler, config, _seq, fcols = predict_now.load_model(
                inference_device='cpu', prefix=cp)
            stack = (mtime, model, scaler, config, fcols)
            loop._shadow_stack = stack
            logger.info("[SHADOW] challenger loaded (%s, fb=%s)",
                        cp, config.get('forward_bars'))
    except Exception as e:
        logger.warning("[SHADOW] challenger load failed: %s", e)
        return

    # Hypersearch writes the challenger manifest BEFORE train_lgb_ensemble
    # finishes, so the first tick can permanently cache a None booster for a
    # file that lands minutes later. Evict the cached None once the file
    # exists so the shadow record tests the same LSTM+LGB blend promotion
    # would deploy — not an LSTM-only chimera.
    try:
        import predict_now
        if (predict_now._lgb_models.get(cp) is None
                and (BASE_DIR / f'{cp}_lgb_model.txt').exists()):
            predict_now._lgb_models.pop(cp, None)
        if (predict_now._q10_models.get(cp) is None
                and (BASE_DIR / f'{cp}_lgb_q10.txt').exists()
                and (BASE_DIR / f'{cp}_lgb_q10_meta.json').exists()):
            predict_now._q10_models.pop(cp, None)
    except Exception:
        pass

    _, model, scaler, config, fcols = stack
    asset_type = loop.get_asset_type()
    kw = ({'btc_close': benchmark} if asset_type == 'crypto'
          else {'spy_close': benchmark})
    fb_champ = loop.config.get('forward_bars') if getattr(loop, 'config', None) else None
    rows = []
    from predict_now import get_live_prediction
    for symbol, champ in champ_preds.items():
        try:
            chall = get_live_prediction(
                symbol, model, scaler, config, fcols, api=loop.api,
                inference_device='cpu', asset_type=asset_type, **kw)
        except Exception:
            chall = None
        if chall is None or champ is None:
            continue
        rows.append(json.dumps({
            'ts': dt.datetime.now(dt.timezone.utc).isoformat(),
            'sym': symbol,
            'champ': round(float(champ), 6),
            'chall': round(float(chall), 6),
            'fb_champ': fb_champ,
            'fb_chall': config.get('forward_bars'),
            'cm': mtime,
        }))
    if rows:
        try:
            with open(shadow_log_file(prefix), 'a') as f:
                f.write('\n'.join(rows) + '\n')
            logger.info("[SHADOW] logged %d side-by-side predictions", len(rows))
        except OSError as e:
            # A silently-empty shadow record wastes the whole 4-week
            # experiment (discarded for lack of data) — leave evidence.
            logger.warning("[SHADOW] failed to append %d rows to %s: %s",
                           len(rows), shadow_log_file(prefix), e)


def _prune_stale_rows(prefix: str, current_mtime: int) -> None:
    """Drop rows logged against a REPLACED challenger (stale cm).

    The log is append-only and every challenger replacement restarts the
    clock, so without pruning it grows without bound across weekly retrains
    and every daily eval re-parses dead rows. Behavior-neutral: stale-cm
    rows are already excluded from every computation (_load_rows filters).
    Rewrite is tmp+os.replace so a concurrent reader sees a consistent file.
    """
    path = shadow_log_file(prefix)
    if not path.exists():
        return
    try:
        with open(path) as f:
            lines = f.readlines()
        keep = []
        for line in lines:
            try:
                if json.loads(line).get('cm') == current_mtime:
                    keep.append(line if line.endswith('\n') else line + '\n')
            except json.JSONDecodeError:
                continue
        if len(keep) == len(lines):
            return
        tmp = f'{path}.tmp'
        with open(tmp, 'w') as f:
            f.writelines(keep)
        os.replace(tmp, path)
        logger.info("[SHADOW] pruned %d stale rows from %s",
                    len(lines) - len(keep), path.name)
    except OSError as e:
        logger.warning("[SHADOW] shadow-log prune failed: %s", e)


# --- Statistics: Diebold-Mariano with HLN correction ---

def dm_hln(d: np.ndarray, h: int) -> tuple[float, float]:
    """One-sided DM test on loss differentials d_t (>0 = challenger better).

    Newey-West LRV truncated at h-1 (h-step-ahead forecast errors are
    MA(h-1)); HLN small-sample correction; p-value against t_{n-1}
    (normal fallback when scipy is unavailable).
    Returns (dm_stat, p_value_challenger_better).

    NOTE: the MA(h-1) truncation assumes ONE loss-diff per time step;
    evaluate_shadow feeds a POOLED cross-section (K records per hour),
    which under-covers the true dependence — see the module docstring.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        return 0.0, 1.0
    dbar = d.mean()
    dc = d - dbar
    h = max(1, min(int(h), n - 1))
    gamma0 = float(np.mean(dc * dc))
    lrv = gamma0
    for k in range(1, h):
        lrv += 2.0 * float(np.mean(dc[k:] * dc[:-k]))
    if lrv <= 0:
        lrv = gamma0
    if lrv <= 0:
        return 0.0, 1.0
    dm = dbar / math.sqrt(lrv / n)
    hln = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm_star = dm * hln
    try:
        from scipy import stats
        p = float(1.0 - stats.t.cdf(dm_star, df=n - 1))
    except Exception:
        p = 0.5 * math.erfc(dm_star / math.sqrt(2.0))
    return dm_star, p


# --- DM v2 statistics (pure numpy/stdlib — B03.3) ---

# One-sided Student-t critical values, alpha -> {df: crit}. Sanity anchors
# from the research: 1.943 = alpha .05 df 6 (q=7 blocks); 1.771 = df 13.
_T_ONE_SIDED = {
    0.05: {
        5: 2.0150, 6: 1.9432, 7: 1.8946, 8: 1.8595, 9: 1.8331, 10: 1.8125,
        11: 1.7959, 12: 1.7823, 13: 1.7709, 14: 1.7613, 15: 1.7531,
        16: 1.7459, 17: 1.7396, 18: 1.7341, 19: 1.7291, 20: 1.7247,
        21: 1.7207, 22: 1.7171, 23: 1.7139, 24: 1.7109, 25: 1.7081,
        26: 1.7056, 27: 1.7033, 28: 1.7011, 29: 1.6991, 30: 1.6973,
    },
    0.025: {
        5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281,
        11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
        16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860,
        21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639, 25: 2.0595,
        26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423,
    },
    0.10: {
        5: 1.4759, 6: 1.4398, 7: 1.4149, 8: 1.3968, 9: 1.3830, 10: 1.3722,
        11: 1.3634, 12: 1.3562, 13: 1.3502, 14: 1.3450, 15: 1.3406,
        16: 1.3368, 17: 1.3334, 18: 1.3304, 19: 1.3277, 20: 1.3253,
        21: 1.3232, 22: 1.3212, 23: 1.3195, 24: 1.3178, 25: 1.3163,
        26: 1.3150, 27: 1.3137, 28: 1.3125, 29: 1.3114, 30: 1.3104,
    },
}


def t_crit_one_sided(df: int, alpha: float) -> float:
    """One-sided Student-t critical value from the table above.

    df < 5 raises (callers guard via q >= V2_MIN_BLOCKS -> df >= 5);
    df > 30 uses the df=30 entry (marginally conservative); alpha must be
    one of the tabled levels.
    """
    df = int(df)
    if df < 5:
        raise ValueError(f'df={df} < 5 unsupported (q >= 6 guards this)')
    table = _T_ONE_SIDED[alpha]
    return table[min(df, 30)]


def collapse_by_hour(ts_list, d_arr) -> np.ndarray:
    """Per-hour cross-sectional means of loss diffs (QTZ-2023 collapse).

    Floors each tz-aware timestamp to the hour (shadow rows carry
    per-symbol wall-clock ts — exact-ts grouping would yield singleton
    buckets), bucket-means d, returns the means in ascending hour order.
    """
    buckets: dict = {}
    for ts, dv in zip(ts_list, np.asarray(d_arr, dtype=float)):
        key = ts.replace(minute=0, second=0, microsecond=0)
        s, c = buckets.get(key, (0.0, 0))
        buckets[key] = (s + float(dv), c + 1)
    return np.array([buckets[k][0] / buckets[k][1] for k in sorted(buckets)],
                    dtype=float)


def im_cluster_t(dbar: np.ndarray, block: int) -> tuple[float, int]:
    """Ibragimov-Muller cluster t on non-overlapping block means.

    Uses the MOST RECENT q*block observations (drops the oldest remainder
    — deterministic). Refuses (nan, q) when q < V2_MIN_BLOCKS or the
    block-mean sd is degenerate. Compare against t_{q-1}.
    """
    dbar = np.asarray(dbar, dtype=float)
    T = dbar.size
    block = int(block)
    q = T // block if block > 0 else 0
    if q < V2_MIN_BLOCKS:
        return float('nan'), int(q)
    used = dbar[T - q * block:]
    bm = used.reshape(q, block).mean(axis=1)
    sd = float(np.std(bm, ddof=1))
    if sd <= 0:
        return float('nan'), int(q)
    return float(math.sqrt(q) * float(np.mean(bm)) / sd), int(q)


def kv_fixed_b_crit(b: float) -> float:
    """Kiefer-Vogelsang fixed-b critical value, 5% ONE-SIDED (Bartlett),
    q(b) = 1.6449 + 2.1859 b + 0.3142 b^2 - 0.3427 b^3. The 5% level is
    the only calibrated level we have — the fixed-b fallback is therefore
    always an alpha=0.05 test."""
    b = float(b)
    return 1.6449 + 2.1859 * b + 0.3142 * b * b - 0.3427 * b ** 3


def dm_fixed_b(dbar: np.ndarray, h_max: int) -> tuple[float, float]:
    """Collapsed-series DM with a PROPER Bartlett kernel at bandwidth
    M = min(2*h_max, T-1), judged against Kiefer-Vogelsang fixed-b
    critical values (b = M/T; no HLN factor — fixed-b already prices the
    bandwidth). Refusal on non-positive LRV returns (0.0, inf) so the
    caller can never promote on it. NOTE: legacy dm_hln's flat truncated
    kernel is deliberately untouched (it stays the logged diagnostic)."""
    dbar = np.asarray(dbar, dtype=float)
    T = dbar.size
    if T < 2:
        return 0.0, float('inf')
    M = min(2 * int(h_max), T - 1)
    m = float(np.mean(dbar))
    dc = dbar - m
    gamma0 = float(np.mean(dc * dc))
    lrv = gamma0
    for k in range(1, M):
        lrv += 2.0 * (1.0 - k / M) * float(np.mean(dc[k:] * dc[:-k]))
    if lrv <= 0:
        return 0.0, float('inf')
    stat = m / math.sqrt(lrv / T)
    return float(stat), kv_fixed_b_crit(M / T)


def dm_v2_evaluate(dbar, h_max: int, age_days: float, book: str,
                   look1_done: bool) -> dict:
    """DM v2 decision core (pure). Two scheduled looks replace the ~14
    unadjusted daily peeks: interim (crypto only) at day 21 alpha=0.025,
    final at max duration alpha=0.10 + mean_d>0 (IM cluster t when q>=6,
    else the fixed-b DM at its calibrated 5%). A look is consumed ONLY
    when a statistic was actually compared to a critical value.

    Returns a JSON-safe dict {'t','q','T','block','look','alpha','crit',
    'stat','decision','mean_d','look_consumed','reason'} (unused fields
    None). decision in {'promote','discard','continue'}.
    """
    dbar = np.asarray(dbar, dtype=float)
    T = int(dbar.size)
    h_max = int(h_max)
    block = V2_BLOCK_MULT * h_max
    min_T = V2_MIN_TS_MULT * h_max
    max_days = (V2_STOCK_MAX_SHADOW_DAYS if book == 'stock'
                else MAX_SHADOW_DAYS)
    mean_d = float(np.mean(dbar)) if T else 0.0
    if age_days >= max_days:
        look = 2
    elif book != 'stock' and age_days >= V2_LOOK1_DAYS and not look1_done:
        look = 1
    else:
        look = 0
    base = {'t': None, 'q': None, 'T': T, 'block': int(block),
            'look': int(look), 'alpha': None, 'crit': None, 'stat': None,
            'decision': None, 'mean_d': mean_d, 'look_consumed': False,
            'reason': None}
    if look == 0:
        return {**base, 'decision': 'continue',
                'reason': 'no scheduled look'}
    if look == 1:
        if T < min_T:
            return {**base, 'decision': 'continue', 'reason': 'T<8*h_max'}
        t, q = im_cluster_t(dbar, block)
        base['q'] = int(q)
        if q < V2_MIN_BLOCKS or not math.isfinite(t):
            return {**base, 'decision': 'continue',
                    'reason': 'q<6 — IM infeasible'}
        crit = t_crit_one_sided(q - 1, V2_LOOK1_ALPHA)
        return {**base, 't': float(t), 'stat': 'im',
                'alpha': V2_LOOK1_ALPHA, 'crit': float(crit),
                'look_consumed': True,
                'decision': 'promote' if t > crit else 'continue',
                'reason': 'interim IM look'}
    # look == 2: final
    if T < min_T:
        return {**base, 'decision': 'discard',
                'reason': 'insufficient collapsed timestamps at max '
                          'duration'}
    t, q = im_cluster_t(dbar, block)
    base['q'] = int(q)
    if q >= V2_MIN_BLOCKS and math.isfinite(t):
        crit = t_crit_one_sided(q - 1, V2_FINAL_ALPHA)
        return {**base, 't': float(t), 'stat': 'im',
                'alpha': V2_FINAL_ALPHA, 'crit': float(crit),
                'look_consumed': True,
                'decision': ('promote' if (t > crit and mean_d > 0)
                             else 'discard'),
                'reason': 'final IM look'}
    t, crit = dm_fixed_b(dbar, h_max)
    return {**base, 't': float(t), 'stat': 'fixed_b', 'alpha': 0.05,
            'crit': float(crit), 'look_consumed': True,
            'decision': ('promote' if (t > crit and mean_d > 0)
                         else 'discard'),
            'reason': 'fixed-b fallback (q<6), documented ~7-10% real '
                      'level'}


def _prewindow_forward_returns(closes, t_min, fb: int) -> np.ndarray:
    """Forward fb-bar % returns fully resolved strictly BEFORE t_min (the
    first shadow row) — the frozen-variance sample, independent of the
    shadow window. Empty array when no anchor resolves pre-window."""
    vals = np.asarray(closes.values, dtype=float)
    n = vals.size
    fb = int(fb)
    if n == 0 or fb <= 0:
        return np.array([], dtype=float)
    # bars strictly before t_min; anchor i qualifies iff index[i+fb] < t_min
    k = int(closes.index.searchsorted(t_min))
    m = min(k - fb, n - fb)
    if m <= 0:
        return np.array([], dtype=float)
    c0 = vals[:m]
    c1 = vals[fb:fb + m]
    mask = c0 > 0
    r = (c1[mask] - c0[mask]) / c0[mask] * 100.0
    return r[np.isfinite(r)]


# --- Evaluation + promotion ---

def _load_rows(prefix: str) -> list[dict]:
    path = shadow_log_file(prefix)
    if not path.exists():
        return []
    rows = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    # Only the CURRENT challenger's rows count (a replaced challenger's
    # history must not contaminate the new one's test)
    man = challenger_manifest(prefix)
    if not man.exists():
        return []
    cur = int(man.stat().st_mtime)
    return [r for r in rows if r.get('cm') == cur]


def _fetch_closes(api, symbol: str, asset_type: str):
    from market_data import fetch_bars_alpaca, fetch_stock_bars_alpaca
    # The window must reach back past the OLDEST shadow row (up to
    # MAX_SHADOW_DAYS old) plus the forward horizon — the default live-loop
    # limits (250 crypto bars ~ 10d) cover only a third of it, and any older
    # row would silently anchor at the window's first bar (fabricated
    # returns). Distinct (asset, symbol, limit) cache keys keep this separate
    # from the live loops' bar cache entries.
    c_limit = 24 * (MAX_SHADOW_DAYS + 4)
    s_limit = 600
    if DM_V2_ENABLED:
        # frozen-variance window (B03.3): reach ~90d past the shadow window.
        # Extending BACKWARD cannot change any _realized value — the same
        # anchor bars are found by searchsorted, and the ts < closes.index[0]
        # guard only relaxes.
        c_limit = 24 * (MAX_SHADOW_DAYS + 4) + V2_FROZEN_MIN_BARS['crypto'] + 48
        s_limit = 1100   # ~630 RTH bars trailing + 56d window + horizon
    # closed_only: the DM evaluator must price against completed bars only —
    # a forming bar makes the promotion statistic irreproducible run-to-run
    # (c26 D38 handoff; measurement-only, always on here).
    df = (fetch_bars_alpaca(api, symbol, limit=c_limit, closed_only=True)
          if asset_type == 'crypto'
          else fetch_stock_bars_alpaca(api, symbol, limit=s_limit,
                                       closed_only=True))
    if df is None or df.empty:
        return None
    closes = df['Close']
    if closes.index.tz is None:
        closes = closes.tz_localize('UTC')
    return closes


def _realized(closes, ts: dt.datetime, fb: int) -> float | None:
    """Forward fb-bar % return from the first bar at/after ts."""
    if len(closes) == 0 or ts < closes.index[0]:
        # The true anchor bar predates the fetched window: searchsorted
        # would alias to the window's FIRST bar and fabricate a return
        # anchored days after the prediction. Stay unresolved instead.
        return None
    i = int(closes.index.searchsorted(ts))
    j = i + int(fb)
    if i >= len(closes) or j >= len(closes):
        return None
    c0, c1 = float(closes.iloc[i]), float(closes.iloc[j])
    if c0 <= 0:
        return None
    return (c1 - c0) / c0 * 100.0


def evaluate_shadow(prefix: str, api=None) -> dict | None:
    """Resolve realized returns and run the DM-HLN comparison.

    Returns {'n', 'age_days', 'dm', 'p', 'mean_d', 'hit_champ',
    'hit_chall', 'fb_max'} or None when there is nothing to evaluate.
    """
    rows = _load_rows(prefix)
    if not rows:
        return None
    if api is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from trading_utils import get_api
            api = get_api()
        except Exception as e:
            logger.warning("[SHADOW] no API for evaluation (%s)", e)
            return None
    asset_type = 'stock' if prefix == 'stock' else 'crypto'

    by_sym: dict[str, list[dict]] = {}
    for r in rows:
        by_sym.setdefault(r['sym'], []).append(r)

    recs = []  # (ts, e2_champ_raw, e2_chall_raw, r_champ, r_chall,
    #            champ, chall)
    closes_by_sym: dict = {}   # retained for DM v2's frozen-variance attempt
    for sym, srows in by_sym.items():
        closes = _fetch_closes(api, sym, asset_type)
        if closes is None:
            continue
        n_before = len(recs)
        for r in srows:
            try:
                ts = dt.datetime.fromisoformat(r['ts'])
                fb_c = int(r.get('fb_champ') or 24)
                fb_x = int(r.get('fb_chall') or 24)
            except (TypeError, ValueError, KeyError):
                continue
            rc = _realized(closes, ts, fb_c)
            rx = _realized(closes, ts, fb_x)
            if rc is None or rx is None:
                continue
            recs.append((ts, (r['champ'] - rc) ** 2, (r['chall'] - rx) ** 2,
                         rc, rx, r['champ'], r['chall']))
        if len(recs) > n_before:
            closes_by_sym[sym] = closes
    if len(recs) < 10:
        r0 = rows[0]
        fb_max = max(int(r0.get('fb_champ') or 24), int(r0.get('fb_chall') or 24))
        return {'n': len(recs), 'age_days': _age_days(rows), 'p': 1.0,
                'dm': 0.0, 'mean_d': 0.0, 'hit_champ': None,
                'hit_chall': None, 'fb_max': fb_max}

    recs.sort(key=lambda t: t[0])
    e2c = np.array([t[1] for t in recs])
    e2x = np.array([t[2] for t in recs])
    rc = np.array([t[3] for t in recs])
    rx = np.array([t[4] for t in recs])
    pc = np.array([t[5] for t in recs])
    px = np.array([t[6] for t in recs])
    # Variance-normalize each model's squared errors on ITS OWN horizon
    # so a 4h-horizon challenger and 24h champion are comparable
    # (skill-score view: error relative to the target's natural scale)
    vc = max(float(np.var(rc)), 1e-12)
    vx = max(float(np.var(rx)), 1e-12)
    d = e2c / vc - e2x / vx     # >0 -> challenger better
    r0 = rows[0]
    h = max(int(r0.get('fb_champ') or 24), int(r0.get('fb_chall') or 24))
    dm, p = dm_hln(d, h=h)
    age = _age_days(rows)
    v2 = None
    try:
        v2 = _compute_dm_v2(prefix, recs, closes_by_sym, vc, vx,
                            int(r0.get('fb_champ') or 24),
                            int(r0.get('fb_chall') or 24),
                            h, age, asset_type)
    except Exception as e:   # a v2 bug must never break the legacy evaluation
        logger.warning("[SHADOW] dm_v2 computation failed: %s", e)
    report = {
        'n': int(len(recs)),
        'age_days': age,
        'dm': round(float(dm), 3),
        'p': round(float(p), 4),
        'mean_d': round(float(np.mean(d)), 4),
        'hit_champ': round(float(np.mean(np.sign(pc) == np.sign(rc))), 4),
        'hit_chall': round(float(np.mean(np.sign(px) == np.sign(rx))), 4),
        'fb_max': h,
    }
    if v2 is not None:
        report['dm_v2'] = v2
    return report


def _compute_dm_v2(prefix: str, recs: list, closes_by_sym: dict,
                   vc: float, vx: float, fb_c: int, fb_x: int,
                   h_max: int, age: float, book: str) -> dict | None:
    """Side-by-side DM v2 statistic on the resolved records (B03.3).

    FROZEN variance mode when every contributing symbol has >= ~90d of
    bars strictly before the first shadow row (only reachable when the
    flag-ON extended fetch is active); otherwise the PATCHED mode deflates
    each side's in-window normalized loss by (n_eff-2)/n_eff with
    n_eff = collapsed_hours/fb_side (offsets the inverse-chi-square Jensen
    bias favoring the shorter horizon). Returns a JSON-safe dict (rides
    into the promoted manifest's shadow_report) or None when degenerate.
    """
    ts_list = [t[0] for t in recs]
    e2c = np.array([t[1] for t in recs], dtype=float)
    e2x = np.array([t[2] for t in recs], dtype=float)
    t_min = min(ts_list)
    d_arr = None
    var_mode = None
    if closes_by_sym and all(
            int(c.index.searchsorted(t_min)) >= V2_FROZEN_MIN_BARS[book]
            for c in closes_by_sym.values()):
        pooled_c = [_prewindow_forward_returns(c, t_min, fb_c)
                    for c in closes_by_sym.values()]
        pooled_x = [_prewindow_forward_returns(c, t_min, fb_x)
                    for c in closes_by_sym.values()]
        pooled_c = np.concatenate(pooled_c) if pooled_c else np.array([])
        pooled_x = np.concatenate(pooled_x) if pooled_x else np.array([])
        if pooled_c.size and pooled_x.size:
            vf_c = max(float(np.var(pooled_c)), 1e-12)
            vf_x = max(float(np.var(pooled_x)), 1e-12)
            d_arr = e2c / vf_c - e2x / vf_x
            var_mode = 'frozen'
    if d_arr is None:
        # PATCHED mode — the effective mode under flag OFF (the un-extended
        # fetch reaches only ~4d past the window, never 90d)
        T0 = len({ts.replace(minute=0, second=0, microsecond=0)
                  for ts in ts_list})
        n_eff_c = T0 / fb_c
        n_eff_x = T0 / fb_x
        if n_eff_c <= 2 or n_eff_x <= 2:
            return None   # degenerate (cannot happen once T0 >= 8*h_max)
        w_c = (n_eff_c - 2) / n_eff_c
        w_x = (n_eff_x - 2) / n_eff_x
        d_arr = w_c * e2c / vc - w_x * e2x / vx
        var_mode = 'patched'
    dbar = collapse_by_hour(ts_list, d_arr)
    verdict = dm_v2_evaluate(dbar, h_max, age, book, _v2_look1_done(prefix))
    out = {**verdict, 'var_mode': var_mode}
    for k, v in out.items():   # strictly JSON-safe (manifest embed)
        if isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
    return out


def _age_days(rows: list[dict]) -> float:
    try:
        oldest = min(dt.datetime.fromisoformat(r['ts']) for r in rows)
        return (dt.datetime.now(dt.timezone.utc) - oldest).total_seconds() / 86400
    except (ValueError, TypeError, KeyError):
        return 0.0


def _stash_stale_meta(prefix: str):
    """D13b: champion meta pairs with the OLD primary — move it ASIDE (never
    delete: if the guarded background retrain crashes or refuses, the operator
    can restore). The live gate fails open to neutral until meta_label's
    guarded staged promote lands a replacement, which also clears these."""
    p = _p(prefix)
    for suffix in _STALE_META_SUFFIXES:
        live = BASE_DIR / f'{p}{suffix}'
        try:
            if live.exists():
                os.replace(live, f'{live}.stale')
        except OSError:
            pass


def promote_challenger(prefix: str, report: dict | None = None) -> bool:
    """Copy the challenger stack over the champion (manifest LAST)."""
    import shutil
    cp = challenger_prefix(prefix)
    p = _p(prefix)
    # All artifacts must exist before touching the champion
    for suffix in _ARTIFACT_SUFFIXES[:4]:  # core four are mandatory
        if not (BASE_DIR / f'{cp}_{suffix}').exists():
            logger.error("[SHADOW] promote aborted: missing %s_%s", cp, suffix)
            return False
    # The challenger manifest is mandatory too — parse it in pre-flight
    # (reused for the final write). Discovering it missing/corrupt only at
    # the manifest-write step would leave a half-promoted champion, and a
    # missing manifest would be a silent zombie (evaluate_and_maybe_promote
    # returns None forever, so no retry ever heals the mixed state).
    try:
        with open(challenger_manifest(prefix)) as f:
            man = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("[SHADOW] promote aborted: challenger manifest "
                     "unreadable: %s", e)
        return False
    for suffix in _ARTIFACT_SUFFIXES + ['model_v2.manifest.json']:
        champ = BASE_DIR / f'{p}{suffix}'
        if champ.exists():
            try:
                shutil.copy2(champ, f'{champ}.prev')
            except OSError:
                pass
    for suffix in _ARTIFACT_SUFFIXES:
        src = BASE_DIR / f'{cp}_{suffix}'
        if src.exists():
            shutil.copy2(src, BASE_DIR / f'{p}{suffix}')
    # The config carries its load prefix — rewrite challenger -> champion
    try:
        import joblib
        cfg_path = BASE_DIR / f'{p}config_v2.pkl'
        cfg = joblib.load(cfg_path)
        cfg['prefix'] = prefix
        joblib.dump(cfg, cfg_path)
    except Exception as e:
        # Abort BEFORE the manifest write: continuing would deploy a config
        # still pointing at the challenger prefix, whose LGB/q10 files
        # _discard_challenger then deletes — the new champion would silently
        # lose its ensemble legs. The challenger stays intact, so the next
        # daily eval retries and heals the partially-copied champion.
        logger.error("[SHADOW] config prefix rewrite failed: %s — promotion "
                     "aborted (champion manifest not written; will retry)", e)
        return False
    # Champion meta artifacts pair with the OLD model — move aside (the meta
    # gate fails open to neutral until the background retrain finishes)
    _stash_stale_meta(prefix)
    # Manifest last: bots hot-reload on it (content parsed in pre-flight)
    try:
        man['promoted_from_shadow'] = dt.datetime.now(
            dt.timezone.utc).isoformat()
        if report:
            man['shadow_report'] = report
        mpath = BASE_DIR / f'{p}model_v2.manifest.json'
        tmp = f'{mpath}.tmp'
        with open(tmp, 'w') as f:
            json.dump(man, f, indent=2)
        os.replace(tmp, mpath)
    except Exception as e:
        logger.error("[SHADOW] manifest write failed: %s", e)
        return False
    # Background meta retrain for the new champion (fail-open neutral
    # meanwhile). meta_label's default flow now stages + guard-checks and
    # only then atomically promotes (D13b) — a crashed or refused retrain
    # leaves the neutral fail-open state, never a garbage calibrator.
    try:
        cmd = [sys.executable, '-u', str(BASE_DIR / 'meta_label.py')]
        if prefix:
            cmd += ['--prefix', prefix]
        with open(BASE_DIR / 'meta_retrain.log', 'a') as out:
            subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=out,
                             stderr=subprocess.STDOUT)
    except Exception as e:
        logger.warning("[SHADOW] meta retrain launch failed: %s", e)
    _discard_challenger(prefix)
    return True


def _gate_preflight(prefix: str) -> tuple[bool, str]:
    """(ok, detail) from {challenger}_policy_gate.json (written by backtest.py
    on challenger-targeted --gate runs). Consulted ONLY under
    strategy_config.GATE_TARGETS_CHALLENGER. Never raises; any failure
    (missing/stale/corrupt/gate-fail) is a HOLD, never a crash and never a
    discard — the challenger keeps shadowing until a fresh gate verdict lands."""
    try:
        path = BASE_DIR / f'{challenger_prefix(prefix)}_policy_gate.json'
        if not path.exists():
            return False, (f'{path.name} missing — challenger-targeted policy '
                           f'gate has not run for this challenger')
        with open(path) as f:
            payload = json.load(f)
        cm = int(challenger_manifest(prefix).stat().st_mtime)
        side = payload.get('challenger_manifest_mtime')
        if side != cm:
            return False, f'stale gate verdict (sidecar mtime {side} != manifest {cm})'
        if payload.get('passed') is not True:
            return False, (f"policy gate FAILED (sharpe={payload.get('sharpe')}, "
                           f"dsr={payload.get('dsr')}, n={payload.get('n_trades')})")
        return True, 'policy gate passed'
    except Exception as e:
        return False, f'pre-flight unreadable ({type(e).__name__}: {e})'


def _discard_challenger(prefix: str):
    cp = challenger_prefix(prefix)
    for suffix in _ARTIFACT_SUFFIXES + ['model_v2.manifest.json']:
        try:
            (BASE_DIR / f'{cp}_{suffix}').unlink(missing_ok=True)
        except OSError:
            pass
    try:
        shadow_log_file(prefix).unlink(missing_ok=True)
    except OSError:
        pass
    try:
        # v2 look ledger dies with the challenger (never exists under OFF)
        _v2_state_file(prefix).unlink(missing_ok=True)
    except OSError:
        pass


def _write_shadow_status(prefix: str, *, n: int, age_days: float, p, mean_d,
                          dm, hit_champ, hit_chall, decision: str,
                          detail: str, v2: dict | None = None) -> None:
    """Best-effort atomic snapshot of the latest shadow evaluation, for the
    GUI's promotion-story panel (gui_review_2026-07 §7). Pure
    instrumentation: wrapped so a write failure can never surface into
    evaluation or promotion, which have already completed by the time
    this runs.

    decision is a GUI-facing label — 'promote'/'discard'/'continue'/
    'insufficient_n' — distinct from report['decision'] ('promoted'/
    'promote_failed'/'discarded'/'continue') used for logging/notify/the
    promoted manifest's shadow_report, which this leaves untouched.
    p/mean_d/dm/hit_champ/hit_chall are None whenever evaluate_shadow did
    not actually compute them (no rows yet, or fewer than 10 resolved
    records — its <10 branch returns uninformative placeholders instead
    of real statistics, which are nulled here rather than passed through).
    """
    try:
        window_days = (MAX_SHADOW_DAYS if age_days >= MIN_SHADOW_DAYS
                       else MIN_SHADOW_DAYS)
        status = {
            'ts': time.time(),
            'n': int(n),
            'min_obs': MIN_OBS,
            'age_days': float(age_days),
            'window_days': window_days,
            'p_value': p,
            'mean_d': mean_d,
            'dm_stat': dm,
            'champ_hit_rate': hit_champ,
            'chall_hit_rate': hit_chall,
            'decision': decision,
            'detail': detail,
        }
        # dm_v2_* additive fields ONLY when a v2 evaluation actually ran —
        # the no-rows path must keep the exact legacy key set (pinned by
        # tests/test_shadow_status_persist.py on CI/Jetson).
        if v2 is not None:
            status.update({
                'dm_v2_enabled': DM_V2_ENABLED,
                'dm_v2_t': v2.get('t'), 'dm_v2_q': v2.get('q'),
                'dm_v2_T': v2.get('T'), 'dm_v2_look': v2.get('look'),
                'dm_v2_stat': v2.get('stat'), 'dm_v2_alpha': v2.get('alpha'),
                'dm_v2_crit': v2.get('crit'),
                'dm_v2_decision': v2.get('decision'),
                'dm_v2_mean_d': v2.get('mean_d'),
                'dm_v2_var_mode': v2.get('var_mode'),
            })
        path = shadow_status_file(prefix)
        tmp = f'{path}.tmp'
        with open(tmp, 'w') as f:
            json.dump(status, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("[SHADOW] status write failed: %s", e)


def _manifest_fingerprint(path) -> dict | None:
    """{'mtime', 'saved_at', 'score', 'promoted_from_shadow'} best-effort;
    None when the manifest is missing/unreadable. Never raises."""
    try:
        fp = {'mtime': int(Path(path).stat().st_mtime)}
        with open(path) as f:
            man = json.load(f)
        for k in ('saved_at', 'score', 'promoted_from_shadow'):
            if man.get(k) is not None:
                fp[k] = man[k]
        return fp
    except Exception:
        return None


def _append_promotion_ledger(prefix, label, decision, report, *,
                             gate_hold=None, fingerprints=None,
                             policy_gate=None) -> None:
    """Append one decision row to promotion_ledger_file. Pure
    instrumentation: never raises, never reads back, decision logic already
    completed by the time this runs."""
    try:
        # <10-resolved-records reports carry placeholder stats (p=1.0/dm=0.0/
        # mean_d=0.0, hit_champ None) — null them here exactly as
        # _write_shadow_status does, so the durable ledger never shows
        # fabricated-looking statistics.
        _stats = report.get('hit_champ') is not None
        row = {
            'ts': dt.datetime.now(dt.timezone.utc).isoformat(),
            'prefix': prefix, 'label': label, 'decision': decision,
            'n': report.get('n'), 'age_days': report.get('age_days'),
            'dm': (report.get('dm') if _stats else None),
            'p': (report.get('p') if _stats else None),
            'mean_d': (report.get('mean_d') if _stats else None),
            'stats_computed': _stats,
            'hit_champ': report.get('hit_champ'),
            'hit_chall': report.get('hit_chall'),
            'fb_max': report.get('fb_max'),
            'dm_v2_enabled': DM_V2_ENABLED,
            'dm_v2': report.get('dm_v2'),
            'gate_hold': gate_hold,
            'champion_manifest': (fingerprints or {}).get('champion'),
            'challenger_manifest': (fingerprints or {}).get('challenger'),
            'policy_gate': policy_gate,
        }
        with open(promotion_ledger_file(prefix), 'a') as f:
            f.write(json.dumps(row, default=str) + '\n')
            f.flush()
    except Exception as e:
        logger.warning("[SHADOW] promotion ledger append failed: %s", e)


def evaluate_and_maybe_promote(prefix: str, label: str, api=None) -> dict | None:
    """Daily entry point (called from the pipeline's drift check)."""
    if not challenger_manifest(prefix).exists():
        return None
    report = evaluate_shadow(prefix, api=api)
    if report is None:
        _write_shadow_status(
            prefix, n=0, age_days=0.0, p=None, mean_d=None, dm=None,
            hit_champ=None, hit_chall=None, decision='insufficient_n',
            detail=f"SHADOW {label}: no resolvable shadow predictions "
                   f"logged yet")
        return None
    age, n, p, mean_d = (report['age_days'], report['n'],
                         report['p'], report['mean_d'])
    logger.info("[SHADOW] %s: n=%d age=%.1fd DM=%s p=%s mean_d=%+.4f "
                "hit champ/chall=%s/%s", label, n, age, report['dm'], p,
                mean_d, report['hit_champ'], report['hit_chall'])
    v2 = report.get('dm_v2')
    if v2 is not None:
        logger.info("[SHADOW] %s dm_v2: %s", label, v2)

    decision = None
    if DM_V2_ENABLED:
        v2_max = (V2_STOCK_MAX_SHADOW_DAYS if prefix == 'stock'
                  else MAX_SHADOW_DAYS)
        if v2 is not None:
            if v2['decision'] in ('promote', 'discard'):
                decision = v2['decision']
            if v2.get('look_consumed') and v2['decision'] != 'promote':
                _v2_mark_look1_done(prefix)
        elif age >= v2_max:
            # v2 never became computable in the full window — challenger
            # produced nothing decidable; same terminal rule as legacy
            decision = 'discard'
    else:
        if n >= MIN_OBS and age >= MIN_SHADOW_DAYS and p < EARLY_PROMOTE_P:
            decision = 'promote'
        elif age >= MAX_SHADOW_DAYS:
            if n >= MIN_OBS and mean_d > 0 and p < FINAL_PROMOTE_P:
                decision = 'promote'
            else:
                decision = 'discard'

    # Q2 pre-flight (2026-08 R1): under GATE_TARGETS_CHALLENGER, a promote
    # additionally requires a fresh passed=True {challenger}_policy_gate.json
    # (backtest.py --gate on the challenger slot). A HOLD keeps the
    # challenger shadowing and deliberately does NOT consume a DM-v2 look
    # (v2['decision']=='promote' never marks look1 done), so the promote
    # re-fires daily until a fresh sidecar passes — mild alpha re-testing,
    # flag-ON only.
    gate_hold = None
    if decision == 'promote':
        try:
            import strategy_config as _sc
            _gc = bool(getattr(_sc, 'GATE_TARGETS_CHALLENGER', False))
        except Exception:
            _gc = False
        if _gc:   # flag OFF: this whole block is a no-op (byte-identical path)
            ok_gate, why = _gate_preflight(prefix)
            if not ok_gate:
                decision = None
                gate_hold = why
                logger.warning("[SHADOW] %s: promote HELD by policy-gate "
                               "pre-flight — %s", label, why)
                _notify(f"SHADOW {label}: promote HELD by policy-gate "
                        f"pre-flight — {why}")

    # Ledger capture must precede the branches: promote_challenger discards
    # the challenger manifest and rewrites the champion's; _discard_challenger
    # deletes the challenger's (c26 T7).
    _ledger_fp = _ledger_pg = None
    if decision in ('promote', 'discard') or gate_hold is not None:
        _ledger_fp = {
            'champion': _manifest_fingerprint(
                BASE_DIR / f'{_p(prefix)}model_v2.manifest.json'),
            'challenger': _manifest_fingerprint(challenger_manifest(prefix)),
        }
        try:
            _sp = BASE_DIR / f'{challenger_prefix(prefix)}_policy_gate.json'
            if _sp.exists():
                with open(_sp) as f:
                    _pg = json.load(f)
                _ledger_pg = {k: _pg.get(k) for k in
                              ('passed', 'sharpe', 'dsr', 'n_trades',
                               'challenger_manifest_mtime')}
        except Exception:
            _ledger_pg = None

    # The <10-resolved-records branch of evaluate_shadow sets hit_champ/
    # hit_chall to None and p/dm/mean_d to placeholders (1.0/0.0/0.0), not
    # real statistics — used below to null those fields in the persisted
    # status rather than pass placeholders through as if they were real.
    stats_computed = report.get('hit_champ') is not None

    if decision == 'promote':
        ok = promote_challenger(prefix, report)
        msg = (f"SHADOW {label}: challenger PROMOTED after {age:.0f}d "
               f"(n={n}, DM p={p}, hit {report['hit_champ']}->"
               f"{report['hit_chall']})" if ok else
               f"SHADOW {label}: promotion FAILED — check artifacts")
        if DM_V2_ENABLED and v2 is not None:
            msg += (f" [v2 {v2['stat']} t={v2['t']} q={v2['q']} "
                    f"look={v2['look']}]")
        logger.warning("[SHADOW] %s", msg)
        _notify(msg)
        report['decision'] = 'promoted' if ok else 'promote_failed'
        status_decision, detail = 'promote', msg
    elif decision == 'discard':
        _discard_challenger(prefix)
        msg = (f"SHADOW {label}: challenger discarded after {age:.0f}d "
               f"(n={n}, p={p}, mean_d={mean_d:+.4f}) — champion retained")
        if DM_V2_ENABLED and v2 is not None:
            msg += (f" [v2 {v2['stat']} t={v2['t']} q={v2['q']} "
                    f"look={v2['look']}]")
        logger.info("[SHADOW] %s", msg)
        _notify(msg)
        report['decision'] = 'discarded'
        status_decision, detail = 'discard', msg
    elif stats_computed:
        report['decision'] = 'continue'
        status_decision = 'continue'
        detail = (f"SHADOW {label}: promote HELD by policy-gate pre-flight "
                  f"({gate_hold}) — n={n}, age={age:.1f}d, p={p}"
                  if gate_hold else
                  f"SHADOW {label}: evaluating — n={n}/{MIN_OBS}, "
                  f"age={age:.1f}d, p={p}")
    else:
        report['decision'] = 'continue'
        status_decision = 'insufficient_n'
        detail = (f"SHADOW {label}: only {n} resolved shadow record(s) "
                  f"(<10 needed for DM test), age={age:.1f}d")

    # Durable promotion ledger (c26 T7): terminal outcomes + gate-holds only;
    # routine continue / insufficient_n cycles write nothing.
    if report['decision'] in ('promoted', 'promote_failed', 'discarded') \
            or gate_hold is not None:
        _append_promotion_ledger(
            prefix, label,
            ('held' if gate_hold is not None else report['decision']),
            report, gate_hold=gate_hold, fingerprints=_ledger_fp,
            policy_gate=_ledger_pg)

    _write_shadow_status(
        prefix, n=n, age_days=age,
        p=(p if stats_computed else None),
        mean_d=(mean_d if stats_computed else None),
        dm=(report['dm'] if stats_computed else None),
        hit_champ=report['hit_champ'], hit_chall=report['hit_chall'],
        decision=status_decision, detail=detail, v2=report.get('dm_v2'))
    return report


def _notify(msg: str):
    try:
        from notify import notify
        notify(msg, level='warning', dedupe_key=f'shadow-{msg[:40]}')
    except Exception:
        pass
