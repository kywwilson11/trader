"""PSI drift monitor — detect when live predictions leave the training
distribution, and trigger a retrain after sustained drift.

A model whose live prediction distribution no longer matches its holdout
distribution is extrapolating: the market moved, the model didn't. The
Population Stability Index over the holdout's prediction deciles is the
standard early-warning metric (credit-risk practice):

    PSI = sum_b (live_b - ref_b) * ln(live_b / ref_b)

with ref_b = 0.1 per decile bin. Conventional levels: < 0.10 stable,
0.10-0.25 moderate shift (warn), > 0.25 major shift (action). Two
CONSECUTIVE action days (one bad day is often just a news regime) write
{prefix}retrain_requested.flag and send a notification.

Wiring:
  - hypersearch saves holdout pred deciles in the model manifest
  - base_loop appends each cycle's predictions via log_predictions()
  - run_pipeline runs run_check() in-process once daily
    (_maybe_run_drift_check) and its Phase-C wait loop consumes
    {prefix}retrain_requested.flag (_check_drift_trigger); a standalone
    `python monitor_drift.py` invocation remains available for
    manual/cron use
"""

import argparse
import datetime as dt
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

try:
    import fcntl
except ImportError:  # non-POSIX — cross-process locking degrades to no-op
    fcntl = None

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

PSI_WARN = 0.10
PSI_ACTION = 0.25
CONSECUTIVE_ACTION_DAYS = 2     # days of PSI > action before retrain flag
WINDOW_HOURS = 24               # live window compared against holdout
HISTORY_KEEP_DAYS = 7
MIN_LIVE_SAMPLES = 50

_STATE_FILE = BASE_DIR / 'drift_state.json'


def _p(prefix: str) -> str:
    return f'{prefix}_' if prefix else ''


def history_file(prefix: str) -> Path:
    return BASE_DIR / f'{_p(prefix)}pred_history.jsonl'


def retrain_flag_file(prefix: str) -> Path:
    return BASE_DIR / f'{_p(prefix)}retrain_requested.flag'


@contextmanager
def _history_lock(prefix: str):
    """Advisory flock on a sidecar file — serializes the bot's append
    (log_predictions) against the daily read-rewrite-replace
    (prune_history), which run in different processes. Sidecar because
    os.replace swaps the history file's inode. Best-effort: on any lock
    failure the write proceeds unlocked rather than being dropped
    (mirrors trade_memory._cross_process_lock)."""
    if fcntl is None:
        yield
        return
    try:
        fd = open(str(history_file(prefix)) + '.lock', 'w')
    except OSError:
        yield
        return
    locked = False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
    except OSError:
        pass
    try:
        yield
    finally:
        if locked:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        fd.close()


# --- Live-side logging (called by the bots each prediction cycle) ---

def log_predictions(prefix: str, preds: dict) -> None:
    """Append one line of {ts, preds} to the rolling history (~1/cycle)."""
    if not preds:
        return
    try:
        line = json.dumps({
            'ts': dt.datetime.now(dt.timezone.utc).isoformat(),
            'preds': {s: round(float(v), 6) for s, v in preds.items()
                      if v is not None},
        })
        with _history_lock(prefix), open(history_file(prefix), 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass  # diagnostics must never break the trading cycle


def load_recent_predictions(prefix: str,
                            window_hours: float = WINDOW_HOURS) -> np.ndarray:
    """Pooled prediction values from the last window_hours."""
    path = history_file(prefix)
    if not path.exists():
        return np.array([])
    cutoff = (dt.datetime.now(dt.timezone.utc)
              - dt.timedelta(hours=window_hours))
    vals = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ts = dt.datetime.fromisoformat(rec['ts'])
                    if ts >= cutoff:
                        # float() per value inside the per-line try: a
                        # poison line (naive ts -> TypeError on >=, non-
                        # numeric pred, non-dict preds) drops itself
                        # instead of crashing the whole check at asarray
                        vals.extend([float(v)
                                     for v in rec['preds'].values()])
                except (json.JSONDecodeError, KeyError, ValueError,
                        TypeError, AttributeError):
                    continue
    except OSError:
        return np.array([])
    return np.asarray(vals, dtype=float)


def prune_history(prefix: str, keep_days: int = HISTORY_KEEP_DAYS) -> None:
    """Rewrite the history keeping only the last keep_days (bounded disk)."""
    path = history_file(prefix)
    if not path.exists():
        return
    cutoff = (dt.datetime.now(dt.timezone.utc)
              - dt.timedelta(days=keep_days))
    try:
        # Lock held across read + replace: an append by the bot process
        # between our read and the os.replace would otherwise be lost.
        with _history_lock(prefix):
            kept = []
            with open(path) as f:
                for line in f:
                    try:
                        if dt.datetime.fromisoformat(
                                json.loads(line)['ts']) >= cutoff:
                            kept.append(line)
                    except (json.JSONDecodeError, KeyError, ValueError,
                            TypeError, AttributeError):
                        continue
            tmp = str(path) + '.tmp'
            with open(tmp, 'w') as f:
                f.writelines(kept)
            os.replace(tmp, path)
    except OSError:
        pass


# --- PSI ---

def compute_psi(ref_deciles, live_values, eps: float = 1e-4) -> float | None:
    """PSI of live_values against decile edges saved at train time.

    ref_deciles: 11 edges (0th..100th percentile of holdout preds). The
    outer edges are widened to +-inf so live outliers land in the end
    bins instead of vanishing.
    """
    edges = np.asarray(ref_deciles, dtype=float)
    live = np.asarray(live_values, dtype=float)
    live = live[np.isfinite(live)]
    if edges.size != 11 or live.size < MIN_LIVE_SAMPLES:
        return None
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(live, bins=edges)
    live_frac = counts / live.size
    ref_frac = np.full(10, 0.1)
    lf = np.clip(live_frac, eps, None)
    rf = np.clip(ref_frac, eps, None)
    return float(np.sum((lf - rf) * np.log(lf / rf)))


def load_ref_deciles(prefix: str) -> list | None:
    """Holdout prediction deciles from the model manifest."""
    path = BASE_DIR / f'{_p(prefix)}model_v2.manifest.json'
    try:
        with open(path) as f:
            manifest = json.load(f)
        deciles = (manifest.get('holdout') or {}).get('pred_deciles')
        if deciles and len(deciles) == 11:
            return deciles
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        pass  # non-dict manifest/holdout or unsized deciles -> not checkable
    return None


# --- Daily check + retrain trigger ---

def _load_state() -> dict:
    try:
        with open(_STATE_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    try:
        tmp = str(_STATE_FILE) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, _STATE_FILE)
    except OSError:
        pass


def check_drift(prefix: str) -> dict | None:
    """One drift check. Returns {psi, n, level} or None if not checkable."""
    ref = load_ref_deciles(prefix)
    if ref is None:
        return None
    live = load_recent_predictions(prefix)
    psi = compute_psi(ref, live)
    if psi is None:
        return None
    level = ('action' if psi > PSI_ACTION
             else 'warn' if psi > PSI_WARN else 'ok')
    return {'psi': round(psi, 4), 'n': int(live.size), 'level': level}


# --- CUSUM on live hit-rate (outcome-side complement to input-side PSI) ---

CUSUM_K = 0.05      # detect ~10pp hit-rate degradation (k = drift/2)
CUSUM_H = 4.0       # decision interval (ARL trade-off, standard h=4-5)
HIT_RATE_DEFLATION = 0.90   # live underperforms holdout (McLean-Pontiff)
MIN_BASELINE = 0.30


def load_holdout_hit_rate(prefix: str) -> float | None:
    path = BASE_DIR / f'{_p(prefix)}model_v2.manifest.json'
    try:
        with open(path) as f:
            hr = (json.load(f).get('holdout') or {}).get('hit_rate')
        return float(hr) if hr is not None else None
    except (OSError, json.JSONDecodeError, AttributeError, TypeError,
            ValueError):
        return None


def _live_outcomes(asset: str, since_iso: str | None) -> list[tuple[str, int]]:
    """(ts, win) for confirmed live exits after since_iso, oldest first."""
    try:
        from trade_memory import _load as load_trades
        data = load_trades()
    except Exception as e:
        # _load never raises (it quarantines corruption internally), so
        # this is an import failure — loud, or CUSUM freezes silently
        # looking like "no trades"
        print(f"[CUSUM] trade_memory unavailable ({e}) — outcomes skipped")
        return []
    out = []
    for symbol, trades in data.items():
        is_crypto = '/' in symbol
        if (asset == 'crypto') != is_crypto:
            continue
        for t in trades:
            if t.get('action') != 'sell' and t.get('exit') is None:
                continue
            if t.get('estimated'):
                continue  # pre-slippage estimates flatter the hit rate
            ts = t.get('ts')
            if not ts or (since_iso and ts <= since_iso):
                continue
            out.append((ts, 1 if float(t.get('pnl_pct', 0)) > 0 else 0))
    out.sort()
    return out


def run_cusum(prefix: str, label: str) -> dict | None:
    """Update the one-sided (downward) CUSUM over new live outcomes.

    S <- max(0, S + (mu0 - k) - x) per trade; alarm at S > h means the
    live hit rate has run ~2k below the deflated holdout baseline for
    long enough to be signal, not noise. Alarm notifies and resets.
    """
    baseline = load_holdout_hit_rate(prefix)
    if baseline is None:
        return None
    mu0 = max(MIN_BASELINE, baseline * HIT_RATE_DEFLATION)
    asset = 'stock' if prefix == 'stock' else 'crypto'

    state = _load_state()
    st = state.get(label, {})
    s = float(st.get('cusum', 0.0))
    n_new = 0
    last_ts = st.get('cusum_last_ts')
    alarmed = False
    for ts, win in _live_outcomes(asset, last_ts):
        s = max(0.0, s + (mu0 - CUSUM_K) - win)
        last_ts = ts
        n_new += 1
        if s > CUSUM_H:
            alarmed = True
            s = 0.0  # restart surveillance after the alarm

    st['cusum'] = round(s, 4)
    if last_ts:
        st['cusum_last_ts'] = last_ts
    state[label] = st
    _save_state(state)

    if n_new:
        print(f"[CUSUM] {label}: {n_new} new outcomes, S={st['cusum']:.2f} "
              f"(h={CUSUM_H}, baseline mu0={mu0:.2f})")
    if alarmed:
        print(f"[CUSUM] {label}: ALARM — live hit rate persistently below "
              f"{mu0:.0%}")
        try:
            from notify import notify
            notify(f"CUSUM {label}: live hit rate persistently below the "
                   f"deflated holdout baseline ({mu0:.0%}) — review trades "
                   f"and consider retraining/halting",
                   level='warning',
                   dedupe_key=f'cusum-{label}-{dt.date.today().isoformat()}')
        except Exception:
            pass
    return {'cusum': st['cusum'], 'alarmed': alarmed, 'n_new': n_new,
            'baseline': round(mu0, 4)}


def run_check(prefix: str, label: str) -> dict | None:
    """Daily entry: check, update consecutive-day state, fire trigger."""
    result = check_drift(prefix)

    # CUSUM + pruning run even when PSI is not checkable (legacy manifest
    # without pred_deciles, weekend windows with <MIN_LIVE_SAMPLES stock
    # preds): CUSUM needs only manifest hit_rate + trade_memory — exactly
    # the outcome-side monitor wanted when predictions are missing — and
    # prune bounds pred_history.jsonl on disk. Both print/notify only,
    # never the retrain flag. run_cusum persists its own state keys
    # before we load ours below, so nothing is clobbered.
    try:
        run_cusum(prefix, label)
    except Exception as e:
        print(f"[CUSUM] {label}: check failed ({e})")
    prune_history(prefix)

    today = dt.date.today().isoformat()
    state = _load_state()
    st = state.get(label, {})

    if result is None:
        print(f"[DRIFT] {label}: not checkable (no manifest deciles or "
              f"<{MIN_LIVE_SAMPLES} live preds)")
        return None

    print(f"[DRIFT] {label}: PSI={result['psi']} over {result['n']} preds "
          f"-> {result['level'].upper()}")

    if result['level'] == 'action':
        # Count distinct consecutive calendar days, not repeated runs
        prev_day = st.get('last_action_date')
        yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
        if prev_day == today:
            pass  # already counted today
        elif prev_day == yesterday:
            st['action_days'] = st.get('action_days', 0) + 1
        else:
            st['action_days'] = 1
        st['last_action_date'] = today
    elif result['level'] == 'ok':
        st['action_days'] = 0
        st.pop('last_action_date', None)
    # 'warn' neither extends nor resets the streak

    st['last_psi'] = result['psi']
    st['last_check'] = today
    state[label] = st
    _save_state(state)

    if st.get('action_days', 0) >= CONSECUTIVE_ACTION_DAYS:
        flag = retrain_flag_file(prefix)
        if not flag.exists():
            flag.write_text(json.dumps({
                'reason': f"PSI {result['psi']} > {PSI_ACTION} for "
                          f"{st['action_days']} consecutive days",
                'requested': dt.datetime.now(dt.timezone.utc).isoformat(),
            }))
            print(f"[DRIFT] {label}: retrain flag written -> {flag.name}")
        try:
            from notify import notify
            notify(f"DRIFT {label}: prediction PSI {result['psi']} exceeded "
                   f"{PSI_ACTION} for {st['action_days']} consecutive days — "
                   f"retrain requested ({flag.name})",
                   level='warning', dedupe_key=f'drift-{label}-{today}')
        except Exception:
            pass

    return result


def main():
    ap = argparse.ArgumentParser(description='PSI prediction-drift monitor')
    ap.add_argument('--prefix', default=None,
                    help="model prefix ('' crypto, 'stock'); default: both")
    args = ap.parse_args()
    targets = ([(args.prefix, args.prefix or 'crypto')]
               if args.prefix is not None
               else [('', 'crypto'), ('stock', 'stock')])
    worst = 'ok'
    for prefix, label in targets:
        r = run_check(prefix, label)
        if r and r['level'] == 'action':
            worst = 'action'
        elif r and r['level'] == 'warn' and worst == 'ok':
            worst = 'warn'
    sys.exit(2 if worst == 'action' else 0)


if __name__ == '__main__':
    main()
