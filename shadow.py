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
  - Decisions: promote early after >=14d at p<0.05; at >=28d promote at
    p<0.10 if the mean loss diff favors the challenger, else discard.
    Status-quo bias is intentional — promotion churn has real costs.
  - Promotion copies the full artifact stack (LSTM, scaler, config,
    feature cols, LGB, q10) with .prev backups, manifest LAST; the
    champion's meta-label artifacts are deleted (stale pairing — the
    gate fails open to neutral) and a meta retrain is kicked off in the
    background.
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

_ARTIFACT_SUFFIXES = [
    'model_v2.pth', 'config_v2.pkl', 'scaler_v2.pkl', 'feature_cols_v2.pkl',
    'lgb_model.txt', 'lgb_q10.txt', 'lgb_q10_meta.json',
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
            from predict_now import load_model
            model, scaler, config, _seq, fcols = load_model(
                inference_device='cpu', prefix=cp)
            stack = (mtime, model, scaler, config, fcols)
            loop._shadow_stack = stack
            logger.info("[SHADOW] challenger loaded (%s, fb=%s)",
                        cp, config.get('forward_bars'))
    except Exception as e:
        logger.warning("[SHADOW] challenger load failed: %s", e)
        return

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
        except OSError:
            pass


# --- Statistics: Diebold-Mariano with HLN correction ---

def dm_hln(d: np.ndarray, h: int) -> tuple[float, float]:
    """One-sided DM test on loss differentials d_t (>0 = challenger better).

    Newey-West LRV truncated at h-1 (h-step-ahead forecast errors are
    MA(h-1)); HLN small-sample correction; p-value against t_{n-1}
    (normal fallback when scipy is unavailable).
    Returns (dm_stat, p_value_challenger_better).
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
    df = (fetch_bars_alpaca(api, symbol) if asset_type == 'crypto'
          else fetch_stock_bars_alpaca(api, symbol))
    if df is None or df.empty:
        return None
    closes = df['Close']
    if closes.index.tz is None:
        closes = closes.tz_localize('UTC')
    return closes


def _realized(closes, ts: dt.datetime, fb: int) -> float | None:
    """Forward fb-bar % return from the first bar at/after ts."""
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
    for sym, srows in by_sym.items():
        closes = _fetch_closes(api, sym, asset_type)
        if closes is None:
            continue
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
    if len(recs) < 10:
        return {'n': len(recs), 'age_days': _age_days(rows), 'p': 1.0,
                'dm': 0.0, 'mean_d': 0.0, 'hit_champ': None,
                'hit_chall': None, 'fb_max': 24}

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
    return {
        'n': int(len(recs)),
        'age_days': _age_days(rows),
        'dm': round(float(dm), 3),
        'p': round(float(p), 4),
        'mean_d': round(float(np.mean(d)), 4),
        'hit_champ': round(float(np.mean(np.sign(pc) == np.sign(rc))), 4),
        'hit_chall': round(float(np.mean(np.sign(px) == np.sign(rx))), 4),
        'fb_max': h,
    }


def _age_days(rows: list[dict]) -> float:
    try:
        oldest = min(dt.datetime.fromisoformat(r['ts']) for r in rows)
        return (dt.datetime.now(dt.timezone.utc) - oldest).total_seconds() / 86400
    except (ValueError, TypeError, KeyError):
        return 0.0


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
        logger.error("[SHADOW] config prefix rewrite failed: %s", e)
    # Champion meta artifacts pair with the OLD model — remove (the meta
    # gate fails open to neutral until the background retrain finishes)
    for suffix in _STALE_META_SUFFIXES:
        try:
            (BASE_DIR / f'{p}{suffix}').unlink(missing_ok=True)
        except OSError:
            pass
    # Manifest last: bots hot-reload on it
    try:
        with open(challenger_manifest(prefix)) as f:
            man = json.load(f)
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
    # Background meta retrain for the new champion (fail-open meanwhile)
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


def evaluate_and_maybe_promote(prefix: str, label: str, api=None) -> dict | None:
    """Daily entry point (called from the pipeline's drift check)."""
    if not challenger_manifest(prefix).exists():
        return None
    report = evaluate_shadow(prefix, api=api)
    if report is None:
        return None
    age, n, p, mean_d = (report['age_days'], report['n'],
                         report['p'], report['mean_d'])
    print(f"[SHADOW] {label}: n={n} age={age:.1f}d DM={report['dm']} "
          f"p={p} mean_d={mean_d:+.4f} "
          f"hit champ/chall={report['hit_champ']}/{report['hit_chall']}")

    decision = None
    if n >= MIN_OBS and age >= MIN_SHADOW_DAYS and p < EARLY_PROMOTE_P:
        decision = 'promote'
    elif age >= MAX_SHADOW_DAYS:
        if n >= MIN_OBS and mean_d > 0 and p < FINAL_PROMOTE_P:
            decision = 'promote'
        else:
            decision = 'discard'

    if decision == 'promote':
        ok = promote_challenger(prefix, report)
        msg = (f"SHADOW {label}: challenger PROMOTED after {age:.0f}d "
               f"(n={n}, DM p={p}, hit {report['hit_champ']}->"
               f"{report['hit_chall']})" if ok else
               f"SHADOW {label}: promotion FAILED — check artifacts")
        logger.warning("[SHADOW] %s", msg)
        _notify(msg)
        report['decision'] = 'promoted' if ok else 'promote_failed'
    elif decision == 'discard':
        _discard_challenger(prefix)
        msg = (f"SHADOW {label}: challenger discarded after {age:.0f}d "
               f"(n={n}, p={p}, mean_d={mean_d:+.4f}) — champion retained")
        logger.info("[SHADOW] %s", msg)
        _notify(msg)
        report['decision'] = 'discarded'
    else:
        report['decision'] = 'continue'
    return report


def _notify(msg: str):
    try:
        from notify import notify
        notify(msg, level='warning', dedupe_key=f'shadow-{msg[:40]}')
    except Exception:
        pass
