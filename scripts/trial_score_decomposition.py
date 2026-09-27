"""Trial-score decomposition — WHY does a hypersearch_v2 trial score what it scores?

Measurement-only (reads a study DB snapshot, the trainer log and the training
store; writes only under --out). Answers, per COMPLETE trial and per book:

  (a) NO EDGE            gross per-trade return ~ 0 (or below 0)
  (b) OVER-TRADING UNDER COST  gross > 0 but the round-trip cost eats it
  (c) SCORING ARTEFACT   score sign/rank set by the regime penalty, the
                         mean-0.5*std fold aggregation or a degenerate
                         annualisation, not by trade P&L
  (d) THRESHOLD / TRADE-FREQUENCY MISMATCH  the searched threshold admits a
                         policy the deployable book would not run (below the
                         break-even cost, <10 trades => the 0.0 floor, or a
                         pass rate outside [0.5 %, 20 %])

What the trainer computes (scripts/hypersearch_v2.py, cited by function so the
numbers survive line drift):
  * compute_sharpe: simulate_trades (objective_utils.simulate_trades_core —
    non-overlapping hold: entry on p > thr (and p < -thr unless long-only),
    trade return = r - TXN_COST_PCT[book], scan jumps fb bars), 0.0 when
    < 10 trades or std < 1e-8, else mean/std * sqrt(max(tpy, 1)) with
    tpy = min(n*fb/len(preds), 1) * bars_per_year / fb.
  * _train_walk_forward: score = mean(fold_sharpes) - 0.5*std(fold_sharpes)
    (np.std, ddof 0), then the regime penalty when regime_sharpes['min'] <
    -0.5: legacy `score *= 0.7`, TRAINING_REPAIRS_V1 `score -= 0.3*|score|`.
  * Recorded user_attrs (legacy): cfg, regime_sharpes, fold_sharpes,
    avg_sharpe, std_sharpe. The SIG-R3-DECOMP staged patch adds per-fold lists
    gross_ret_mean, gross_ret_std, net_ret_mean, cost_drag, n_trades,
    hit_rate, mean_hold_bars, threshold_pass_rate, n_rows (LAYER3_KEYS).

Layers:
  L1  (always) DB + log: score, cfg, folds, penalty flag + form, the
      pre-penalty score (definitive: mean - 0.5*std recomputed from the
      recorded folds), rank flips, Spearman of score vs thr / fb / seq_len.
  L2  (needs --data; PROXY) rebuilds each trial's walk-forward val slices
      from the store exactly as get_walk_forward_folds does (legacy calendar
      embargo, or bar embargo under TRAINING_REPAIRS_V1; V3 label purge + block
      ids under OBJECTIVE_V3) on the SAME target column the trainer used
      (Target_Return_{fb} for target_kind raw, TB_Ret_{fb} for tb). Per fold:
      admit rate of the TARGET at the trial threshold, an ORACLE ranker
      (predictions = the target itself) and a RANDOM ranker (predictions = a
      permutation of the target: same marginal => exactly the same number of
      rows above the threshold, i.e. the same admit rate; K seeded perms),
      both walked by the trainer's own simulate_trades_core at the trial
      threshold with cost 0 => gross per trade; the max feasible trade count
      (every row admitted); and the band of per-trade GROSS returns the model
      could have had that is consistent with its OBSERVED fold Sharpe:
      g(n) = cost + S * sd / sqrt(max(tpy(n), 1)), n in [10, n_possible],
      sd = the target's sd on the slice (identifying assumption: the model's
      trade-return sd ~ the target's). The zero-skill-consistent trade count
      n* solves the same equation with the random ranker's gross mean.
  L3  (definitive, when the LAYER3_KEYS attrs exist) classifies directly.

Exit status: 0 always (it is a report; missing inputs degrade the layer).

Usage:
  python scripts/trial_score_decomposition.py --db v2_study.db --prefix '' \
      --log <trainer stdout log> --out logs/trial_decomposition
The DB is SNAPSHOTTED into --out before it is opened (read-only sqlite URI on
the snapshot; optuna is not imported), so a live trainer's DB is never opened
for writing. Default --out is logs/trial_decomposition/ (gitignored logs/);
nothing is ever written to the repo root.
"""
import argparse
import ast
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from objective_utils import (embargo_end_time, holdout_boundary,  # noqa: E402
                             simulate_trades_core, ticker_block_ids)

TRAINER_PATH = BASE_DIR / 'scripts' / 'hypersearch_v2.py'
DEFAULT_OUT = BASE_DIR / 'logs' / 'trial_decomposition'

LAYER3_KEYS = ('gross_ret_mean', 'gross_ret_std', 'net_ret_mean', 'cost_drag',
               'n_trades', 'hit_rate', 'mean_hold_bars',
               'threshold_pass_rate', 'n_rows')

# --- pre-registered constants (see RULES_TEXT) ---
MIN_TRADES = 10            # compute_sharpe's floor: < 10 trades => 0.0
REGIME_PENALTY_MIN = -0.5  # regime_sharpes['min'] < this => penalty
PASS_RATE_LO = 0.005       # < ~1 signal-hour per 8 days per name
PASS_RATE_HI = 0.20        # ~ live cap MAX_TRADES_PER_SYMBOL_PER_DAY crypto 4/24h
SE_MULT = 1.0              # |gross| <= 1 SE => indistinguishable from 0

# Fallbacks only if the trainer source cannot be parsed.
_TRAINER_DEFAULTS = {
    'NUM_FOLDS': 3, 'EMBARGO_MULTIPLIER': 1, 'HOLDOUT_FRACTION': 0.12,
    'TXN_COST_PCT': {'crypto': 0.60, 'stock': 0.11},
    'BARS_PER_YEAR': {'crypto': 8760, 'stock': 1638},
    'FORWARD_BARS': [12, 18, 24, 32, 48],
}

RULES_TEXT = """PRE-REGISTERED CLASSIFICATION (applied mechanically; PRIMARY = first match in order d > c > economic)
 (d) THRESHOLD/FREQUENCY MISMATCH if ANY of:
     d_floor  a fold has n_trades < 10 [L3]  |  a fold Sharpe is exactly 0.0 (compute_sharpe's floor) [L1]
     d_rate   a fold's threshold_pass_rate (share of val rows with pred > thr) < 0.5 % or > 20 % [L3 only]
              (20 % ~ the live cap MAX_TRADES_PER_SYMBOL_PER_DAY crypto 4/24h = 16.7 % of hourly bars;
               0.5 % = < 1 signal-hour per 8 days per name => ~<10 non-overlapping trades per fold)
     d_cost   trade_threshold < TXN_COST_PCT[book] (the searched rule admits trades its own forecast
              says lose money after the round trip) [L1, definitive]
     (flag only, not primary: d_admit = thr < fees.required_edge_pct — the LIVE admission floor)
 (c) SCORING ARTEFACT if ANY of:
     c_sign   sign(score) != sign(mean fold Sharpe) (the -0.5*std term sets the sign) [L1]
     c_rank   the regime penalty changes THIS penalised trial's rank among COMPLETE trials [L1]
     c_ann    trades_per_year < 1 in a fold (sqrt(max(tpy,1)) floor degenerate) [L3]
 ECONOMIC class (reported for EVERY trial, also as its own histogram — the 'why negative' answer):
   L3: pooled over folds, gross = trade-weighted mean, SE = pooled sd / sqrt(sum n_trades)
     a   no edge        gross <= +1 SE   (sub-tag a0 |gross| <= 1 SE, a- gross < -1 SE)
     b   over-trading   gross > +1 SE and cost_drag >= gross (net <= 0)
     +   net edge       gross > +1 SE and net > 0
     ((a) is tested before (b) so a gross indistinguishable from 0 is NO EDGE, never 'over-trading')
   L2 PROXY (no L3 attrs): per fold, the gross band [g(10), g(n_possible)] consistent with the observed
     fold Sharpe S (g increasing in n when S < 0):
     a   g(n_possible) <= +1 SE(n_possible)          (even the most favourable trade count shows no edge)
     b   g(10) > +1 SE(10) and S <= 0                 (every feasible trade count has gross > 0, net <= 0)
     +   S > 0                                        (net-positive fold)
     a~  band straddles 0 and the zero-skill trade count n* is feasible (10 <= n* <= n_possible):
         the observed Sharpe is exactly what a RANDOM long ranker paying the cost earns at n* trades
     ?   otherwise UNRESOLVED
     trial = majority of its folds (tie => '?'); verdict tagged PROXY."""


# --------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------

def trainer_constants(path=TRAINER_PATH):
    """Module-level literals of the trainer, read by AST (no torch import)."""
    out = dict(_TRAINER_DEFAULTS)
    try:
        tree = ast.parse(Path(path).read_text())
    except Exception:
        return out, False
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in out:
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out, True


def trainer_flags(long_only='auto', repairs='auto', v3='auto'):
    """The objective-shaping flags, read from strategy_config (current file —
    the running trainer imported its value at start; override if it moved)."""
    try:
        import strategy_config as sc
    except Exception:
        sc = None

    def _get(name, default=False):
        return bool(getattr(sc, name, default)) if sc is not None else default

    def _pick(cli, name):
        if cli == 'on':
            return True
        if cli == 'off':
            return False
        return _get(name)

    fixed = os.environ.get('TRADER_FIXED_HOLDOUT_DAYS')
    try:
        fixed = float(fixed) if fixed not in (None, '') else (
            getattr(sc, 'FIXED_HOLDOUT_DAYS', None) if sc is not None else None)
    except ValueError:
        fixed = None
    return {
        'OBJECTIVE_LONG_ONLY': _pick(long_only, 'OBJECTIVE_LONG_ONLY'),
        'TRAINING_REPAIRS_V1': _pick(repairs, 'TRAINING_REPAIRS_V1'),
        'OBJECTIVE_V3': _pick(v3, 'OBJECTIVE_V3'),
        'FIXED_HOLDOUT_DAYS': (float(fixed) if fixed is not None else None),
    }


def bars_per_year(asset, table):
    try:
        from bars_calendar import bars_per_year as _bpy
        return float(_bpy(asset, table, 8760))
    except Exception:
        return float(table.get(asset, 8760))


def admission_floor(asset):
    """Live admission floor (fees.required_edge_pct at the flat spread)."""
    try:
        from fees import FLAT_SPREAD_PCT, required_edge_pct
        return float(required_edge_pct(asset, spread_pct=FLAT_SPREAD_PCT[asset]))
    except Exception:
        return None


def sharpe_from_trades(trade_returns, n_rows, forward_bars, bpy):
    """compute_sharpe's arithmetic on an already-simulated net trade array."""
    tr = np.asarray(trade_returns, dtype=np.float64)
    if len(tr) < MIN_TRADES:
        return 0.0
    std = tr.std()
    if std < 1e-8:
        return 0.0
    slots = bpy / forward_bars
    occ = min(len(tr) * forward_bars / max(n_rows, 1), 1.0)
    return float(tr.mean() / std * np.sqrt(max(occ * slots, 1.0)))


def trades_per_year(n_trades, n_rows, forward_bars, bpy):
    occ = min(n_trades * forward_bars / max(n_rows, 1), 1.0)
    return occ * bpy / forward_bars


def fold_trade_stats(preds, y, threshold, forward_bars, cost, long_only,
                     block_ids=None):
    """Per-fold trade decomposition of ONE prediction vector (the L3 attrs'
    definitions; the staged SIG-R3-DECOMP objective records the same)."""
    net, entries = simulate_trades_core(preds, y, threshold, forward_bars,
                                        cost, long_only=long_only,
                                        block_ids=block_ids)
    net = np.asarray(net, dtype=np.float64)
    gross = net + cost
    n = len(net)
    p = np.asarray(preds)
    hold = hold_bars(entries, forward_bars, len(p), block_ids)
    return {
        'n_trades': int(n),
        'gross_ret_mean': float(gross.mean()) if n else float('nan'),
        'gross_ret_std': float(gross.std()) if n else float('nan'),
        'net_ret_mean': float(net.mean()) if n else float('nan'),
        'cost_drag': float(gross.mean() - net.mean()) if n else float('nan'),
        'hit_rate': float((gross > 0).mean()) if n else float('nan'),
        'mean_hold_bars': float(hold.mean()) if n else float('nan'),
        'threshold_pass_rate': float((p > threshold).mean()) if len(p) else float('nan'),
        'n_rows': int(len(p)),
    }


def hold_bars(entries, forward_bars, n, block_ids=None):
    """Bars each walk entry blocks: min(fb, next block start, n) - entry."""
    e = np.asarray(entries, dtype=np.int64)
    if e.size == 0:
        return np.zeros(0)
    end = np.minimum(e + int(forward_bars), n)
    if block_ids is not None:
        b = np.asarray(block_ids)
        change = np.flatnonzero(np.diff(b) != 0) + 1
        ext = np.append(change, n)
        nbs = ext[np.searchsorted(change, e, side='right')]
        end = np.minimum(end, nbs)
    return (end - e).astype(np.float64)


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {'rho': None, 'p': None, 'n': int(len(x))}
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y)
        return {'rho': float(r.statistic if hasattr(r, 'statistic') else r[0]),
                'p': float(r.pvalue if hasattr(r, 'pvalue') else r[1]),
                'n': int(len(x))}
    except Exception:
        def _rank(a):
            order = a.argsort(kind='mergesort')
            ranks = np.empty(len(a))
            ranks[order] = np.arange(len(a))
            for v in np.unique(a):  # average ties
                m = a == v
                ranks[m] = ranks[m].mean()
            return ranks
        rx, ry = _rank(x), _rank(y)
        return {'rho': float(np.corrcoef(rx, ry)[0, 1]), 'p': None,
                'n': int(len(x))}


def _sign(v, eps=1e-12):
    return 0 if abs(v) <= eps else (1 if v > 0 else -1)


# --------------------------------------------------------------------------
# Layer 1 — study DB snapshot + trainer log
# --------------------------------------------------------------------------

def snapshot_db(db, out_dir):
    """Copy the study DB into out_dir (never open the original)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (Path(db).stem + '_snapshot.db')
    shutil.copy2(db, dst)
    return dst


def load_trials(db_path, study='auto'):
    """Trials of one study from an Optuna RDB sqlite file, READ-ONLY.

    Returns (study_name, [trial dict]) — trial dict: number, state, value,
    params, attrs (decoded user_attrs), start, complete."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        studies = con.execute(
            'SELECT study_id, study_name FROM studies').fetchall()
        if not studies:
            return None, []
        if study in (None, '', 'auto'):
            counts = []
            for sid, name in studies:
                c = con.execute("SELECT COUNT(*) FROM trials WHERE study_id=? "
                                "AND state='COMPLETE'", (sid,)).fetchone()[0]
                counts.append((c, sid, name))
            counts.sort(reverse=True)
            _, sid, name = counts[0]
        else:
            m = [s for s in studies if s[1] == study]
            if not m:
                return None, []
            sid, name = m[0]
        rows = con.execute(
            'SELECT trial_id, number, state, datetime_start, datetime_complete '
            'FROM trials WHERE study_id=? ORDER BY number', (sid,)).fetchall()
        vcols = [r[1] for r in con.execute('PRAGMA table_info(trial_values)')]
        trials = []
        for tid, number, state, ds, dc in rows:
            value = None
            if 'value_type' in vcols:
                v = con.execute('SELECT value, value_type FROM trial_values '
                                'WHERE trial_id=? AND objective=0',
                                (tid,)).fetchone()
                if v is not None:
                    value = {'INF_POS': math.inf, 'INF_NEG': -math.inf}.get(
                        v[1], v[0])
            else:
                v = con.execute('SELECT value FROM trial_values WHERE '
                                'trial_id=? AND objective=0', (tid,)).fetchone()
                value = v[0] if v is not None else None
            attrs = {}
            for k, vj in con.execute('SELECT key, value_json FROM '
                                     'trial_user_attributes WHERE trial_id=?',
                                     (tid,)):
                try:
                    attrs[k] = json.loads(vj)
                except Exception:
                    attrs[k] = vj
            params = {}
            for pn, pv, dj in con.execute(
                    'SELECT param_name, param_value, distribution_json FROM '
                    'trial_params WHERE trial_id=?', (tid,)):
                try:
                    d = json.loads(dj)
                    if d.get('name') == 'CategoricalDistribution':
                        pv = d['attributes']['choices'][int(pv)]
                    elif d.get('name') == 'IntDistribution':
                        pv = int(pv)
                except Exception:
                    pass
                params[pn] = pv
            trials.append({'number': int(number), 'state': state,
                           'value': value, 'params': params, 'attrs': attrs,
                           'start': ds, 'complete': dc})
        return name, trials
    finally:
        con.close()


_LOG_FIN = re.compile(r'Trial (\d+) finished with value: (\S+)')
_LOG_BRK = re.compile(r'^\[\s*(\d+)\] score=(\S+) \(mean=(\S+) std=(\S+)\) '
                      r'folds=\[([^\]]*)\]')
_LOG_TIMEOUT = re.compile(r'\[TIMEOUT\] Trial (\d+) at epoch (\d+) fold (\d+)')
_LOG_REGFAIL = re.compile(r'\[REGIME\] Penalty eval failed')
_LOG_FB = re.compile(r'forward_bars=\[([\d, ]+)\]')


def parse_log(path):
    """Trial lines of a hypersearch_v2 stdout log. The bracket line prints
    trial.number + 1 (trial_callback: n = trial.number + 1)."""
    info = {'trials': {}, 'forward_bars': None, 'path': str(path)}
    if not path or not Path(path).exists():
        info['missing'] = True
        return info
    last_regfail = False
    for line in Path(path).read_text(errors='replace').splitlines():
        m = _LOG_FB.search(line)
        if m and info['forward_bars'] is None and '[ADAPTIVE]' in line:
            info['forward_bars'] = [int(x) for x in m.group(1).split(',')]
        if _LOG_REGFAIL.search(line):
            last_regfail = True
        m = _LOG_TIMEOUT.search(line)
        if m:
            t = info['trials'].setdefault(int(m.group(1)), {})
            t['timeout'] = {'epoch': int(m.group(2)), 'fold': int(m.group(3))}
        m = _LOG_FIN.search(line)
        if m:
            t = info['trials'].setdefault(int(m.group(1)), {})
            t['value'] = float(m.group(2).rstrip('.'))
            if last_regfail:
                t['regime_eval_failed'] = True
            last_regfail = False
        m = _LOG_BRK.search(line)
        if m:
            num = int(m.group(1)) - 1
            t = info['trials'].setdefault(num, {})
            t['score_print'] = float(m.group(2))
            t['mean_print'] = float(m.group(3))
            t['std_print'] = float(m.group(4))
            t['folds_print'] = [float(x) for x in m.group(5).split('/') if x]
    return info


def trial_cfg(tr):
    cfg = dict(tr['params'])
    cfg.update(tr['attrs'].get('cfg') or {})
    return cfg


def layer1(trials, cost, admit_floor, log_info=None):
    """Per COMPLETE trial: the definitive score arithmetic."""
    rows = []
    for tr in trials:
        if tr['state'] != 'COMPLETE':
            continue
        a = tr['attrs']
        cfg = trial_cfg(tr)
        folds = [float(x) for x in (a.get('fold_sharpes') or [])]
        if not folds and log_info and tr['number'] in log_info['trials']:
            folds = log_info['trials'][tr['number']].get('folds_print', [])
        avg = float(np.mean(folds)) if folds else 0.0
        std = float(np.std(folds)) if len(folds) > 1 else 0.0
        pre = avg - 0.5 * std
        rs = a.get('regime_sharpes') or {}
        rmin = rs.get('min')
        rwhich = None
        if rs:
            cands = {k: v for k, v in rs.items() if k != 'min'}
            if cands:
                rwhich = min(cands, key=lambda k: cands[k])
        flagged = rmin is not None and rmin < REGIME_PENALTY_MIN
        score = tr['value'] if tr['value'] is not None else float('nan')
        legacy = pre * 0.7
        repaired = pre - 0.3 * abs(pre)
        if not flagged:
            form = 'none' if math.isclose(score, pre, rel_tol=1e-9,
                                          abs_tol=1e-12) else 'mismatch'
        elif pre >= 0 and math.isclose(score, legacy, rel_tol=1e-9, abs_tol=1e-12):
            form = 'either(pre>=0)'
        elif math.isclose(score, legacy, rel_tol=1e-9, abs_tol=1e-12):
            form = 'legacy*0.7'
        elif math.isclose(score, repaired, rel_tol=1e-9, abs_tol=1e-12):
            form = 'repaired-0.3|s|'
        else:
            form = 'mismatch'
        thr = cfg.get('trade_threshold')
        row = {
            'number': tr['number'], 'score': score, 'pre_penalty': pre,
            'backsolved_legacy': (score / 0.7 if flagged else score),
            'avg_sharpe': avg, 'std_sharpe': std,
            'avg_recorded': a.get('avg_sharpe'), 'std_recorded': a.get('std_sharpe'),
            'fold_sharpes': folds, 'regime_sharpes': rs, 'regime_min': rmin,
            'regime_min_which': rwhich, 'penalty_flag': bool(flagged),
            'penalty_form': form,
            'forward_bars': cfg.get('forward_bars'), 'seq_len': cfg.get('seq_len'),
            'trade_threshold': thr, 'target_kind': cfg.get('target_kind', 'raw'),
            'cfg': cfg,
            'flags': {
                'd_floor': any(f == 0.0 for f in folds),
                'd_cost': (thr is not None and thr < cost),
                'd_admit': (thr is not None and admit_floor is not None
                            and thr < admit_floor),
                'c_sign': (_sign(score) != 0 and _sign(avg) != 0
                           and _sign(score) != _sign(avg)),
                'c_rank': False, 'c_ann': False, 'd_rate': False,
            },
        }
        if log_info and tr['number'] in log_info['trials']:
            lt = log_info['trials'][tr['number']]
            row['log'] = lt
            if 'score_print' in lt:
                row['log_agrees'] = abs(lt['score_print'] - score) < 5e-4
        l3 = {k: a[k] for k in LAYER3_KEYS if k in a}
        if len(l3) == len(LAYER3_KEYS):
            row['layer3'] = l3
        rows.append(row)
    # c_rank: the penalty moves a penalised trial's rank among COMPLETE trials
    if rows:
        post = np.array([r['score'] for r in rows])
        pre = np.array([r['pre_penalty'] for r in rows])
        rk_post = (-post).argsort(kind='mergesort').argsort()
        rk_pre = (-pre).argsort(kind='mergesort').argsort()
        for i, r in enumerate(rows):
            r['rank_post'] = int(rk_post[i]) + 1
            r['rank_pre'] = int(rk_pre[i]) + 1
            if r['penalty_flag'] and rk_post[i] != rk_pre[i]:
                r['flags']['c_rank'] = True
    return rows


# --------------------------------------------------------------------------
# Layer 2 — reconstruction on the training store (PROXY)
# --------------------------------------------------------------------------

def load_store(prefix, data_path, forward_bars_set, max_rows=500_000):
    """The trainer's per-ticker contiguous arrays, column-projected.

    Mirrors hypersearch_v2.load_data: ticker order = first appearance in the
    loaded frame, per-ticker sort_index, label time = bar max_fb ahead
    (clamped), Target_Return_{fb} (fallback Target_Return), TB_Ret_{fb},
    --max-rows per-ticker tail cap."""
    import pandas as pd
    fbs = sorted(forward_bars_set)
    want = (['Ticker', 'Target_Return']
            + [f'Target_Return_{fb}' for fb in fbs]
            + [f'TB_Ret_{fb}' for fb in fbs])
    stem = 'stock_training_data' if prefix == 'stock' else 'training_data'
    if data_path:
        path = Path(data_path)
    else:
        pq, csv = BASE_DIR / f'{stem}.parquet', BASE_DIR / f'{stem}.csv'
        path = pq if pq.exists() else csv
        try:
            from data_utils import _csv_is_fresher
            if pq.exists() and _csv_is_fresher(pq, csv):
                path = csv
        except Exception:
            pass
    if not path.exists():
        return None
    if path.suffix == '.parquet':
        import pyarrow.parquet as papq
        names = set(papq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in want if c in names]
        df = pd.read_parquet(path, columns=cols)
        if not isinstance(df.index, pd.DatetimeIndex) and 'Datetime' in df.columns:
            df = df.set_index('Datetime')
            df.index = pd.to_datetime(df.index)
    else:
        head = pd.read_csv(path, nrows=0)
        cols = [head.columns[0]] + [c for c in want if c in head.columns]
        df = pd.read_csv(path, usecols=cols, index_col=0, parse_dates=True)
    tickers = df['Ticker'].unique()
    if len(df) > max_rows and len(tickers):
        per = max_rows // len(tickers)
        df = df.sort_index()
        df = pd.concat([df[df['Ticker'] == t].tail(per) for t in tickers]
                       ).sort_index()
    max_fb = max(fbs)
    times, ltimes, bounds = [], [], {}
    rets = {}
    off = 0
    for t in tickers:
        tdf = df[df['Ticker'] == t].sort_index()
        ts = (tdf.index.view('int64') // 10**9).astype(np.int64)
        n = len(ts)
        lab = np.minimum(np.arange(n) + max_fb, n - 1)
        times.append(ts)
        ltimes.append(ts[lab])
        for fb in fbs:
            col = f'Target_Return_{fb}'
            src = tdf[col] if col in tdf.columns else tdf.get('Target_Return')
            if src is not None:
                rets.setdefault(fb, []).append(src.values.astype(np.float32))
            tb = f'TB_Ret_{fb}'
            if tb in tdf.columns:
                rets.setdefault(('tb', fb), []).append(
                    tdf[tb].values.astype(np.float32))
        bounds[t] = (off, off + n)
        off += n
    returns = {k: np.concatenate(v) for k, v in rets.items()
               if len(v) == len(tickers)}
    return {'times': np.concatenate(times), 'label_times': np.concatenate(ltimes),
            'tickers': list(tickers), 'bounds': bounds, 'returns': returns,
            'path': str(path), 'n_rows': int(off)}


def walk_forward_folds(store, seq_len, consts, flags):
    """Replica of hypersearch_v2.get_walk_forward_folds (legacy + the
    TRAINING_REPAIRS_V1 bar embargo + the OBJECTIVE_V3 val-label purge)."""
    valid = []
    for t in store['tickers']:
        s, e = store['bounds'][t]
        if e - s > seq_len:
            valid.append(np.arange(s + seq_len, e))
    if not valid:
        return []
    valid = np.concatenate(valid)
    all_times = store['times']
    boundary = holdout_boundary(all_times, fixed_days=flags['FIXED_HOLDOUT_DAYS'],
                                holdout_fraction=consts['HOLDOUT_FRACTION'])
    t = all_times[valid]
    search_mask = t <= boundary
    search_times = t[search_mask]
    if len(search_times) < 1000:
        return []
    n_folds = int(consts['NUM_FOLDS'])
    emb_mult = consts['EMBARGO_MULTIPLIER']
    embargo_seconds = seq_len * emb_mult * 3600
    lt = store['label_times'][valid]
    folds = []
    for k in range(n_folds):
        tr_end = 0.55 + k * (0.45 / n_folds)
        va_end = tr_end + (0.45 / n_folds)
        t_tr = int(np.quantile(search_times, min(tr_end, 1.0)))
        t_va = int(np.quantile(search_times, min(va_end, 1.0)))
        train_mask = search_mask & (lt <= t_tr)
        if flags['TRAINING_REPAIRS_V1']:
            vs = embargo_end_time(search_times, t_tr, seq_len * emb_mult)
            val_mask = search_mask & (t >= vs) & (t < t_va)
        else:
            val_mask = search_mask & (t >= t_tr + embargo_seconds) & (t < t_va)
        if flags['OBJECTIVE_V3']:
            val_mask = val_mask & (lt <= boundary)
        tri, vai = valid[train_mask], valid[val_mask]
        if len(tri) < 500 or len(vai) < 200:
            continue
        folds.append((tri, vai))
    return folds


def _gband(S, sd, n, n_rows, fb, bpy, cost):
    """Per-trade gross consistent with fold Sharpe S at n trades."""
    tpy = trades_per_year(n, n_rows, fb, bpy)
    return cost + S * sd / math.sqrt(max(tpy, 1.0))


def zero_skill_trades(S, mu_rand, sd, n_rows, fb, bpy, cost):
    """n* such that a zero-skill ranker (gross mu_rand) scores S; None when
    no n reproduces S (sign mismatch)."""
    m = mu_rand - cost
    if S == 0 or m == 0 or _sign(S) != _sign(m):
        return None
    tpy = (S * sd / m) ** 2          # tpy = n * bpy / n_rows below full occupancy
    if tpy > bpy / fb:
        return math.inf               # beyond full occupancy: no n reproduces S
    return tpy * n_rows / bpy


def layer2_fold(y, S_obs, thr, fb, cost, long_only, bpy, n_perm, rng,
                block_ids=None):
    """Proxy stats for one fold's val slice (y = the trial's target)."""
    y = np.asarray(y, dtype=np.float64)
    n_rows = len(y)
    sd = float(y.std())
    out = {'n_rows': n_rows, 'target_mean': float(y.mean()), 'target_sd': sd,
           'pass_rate_long': float((y > thr).mean()),
           'pass_rate_abs': float((np.abs(y) > thr).mean())}
    all_in, _ = simulate_trades_core(np.full(n_rows, np.inf), y, thr, fb, 0.0,
                                     long_only=True, block_ids=block_ids)
    out['n_possible'] = int(len(all_in))
    # ORACLE: predictions = the target
    og, _ = simulate_trades_core(y, y, thr, fb, 0.0, long_only=long_only,
                                 block_ids=block_ids)
    out['oracle'] = {'n_trades': int(len(og)),
                     'gross_mean': float(og.mean()) if len(og) else float('nan'),
                     'sharpe': sharpe_from_trades(og - cost, n_rows, fb, bpy)}
    # RANDOM: predictions = a permutation of the target (same admit rate)
    ns, gm, gs, sh = [], [], [], []
    for _ in range(max(int(n_perm), 1)):
        pr = rng.permutation(y)
        rg, _ = simulate_trades_core(pr, y, thr, fb, 0.0, long_only=long_only,
                                     block_ids=block_ids)
        ns.append(len(rg))
        gm.append(rg.mean() if len(rg) else np.nan)
        gs.append(rg.std() if len(rg) else np.nan)
        sh.append(sharpe_from_trades(rg - cost, n_rows, fb, bpy))
    mu_r = float(np.nanmean(gm))
    out['random'] = {'n_trades': float(np.mean(ns)), 'gross_mean': mu_r,
                     'gross_sd': float(np.nanmean(gs)),
                     'sharpe_mean': float(np.mean(sh)),
                     'sharpe_sd': float(np.std(sh)), 'n_perm': int(n_perm)}
    out['S_obs'] = S_obs
    if S_obs is None:
        return out
    npos = max(out['n_possible'], MIN_TRADES)
    g_lo = _gband(S_obs, sd, MIN_TRADES, n_rows, fb, bpy, cost)
    g_hi = _gband(S_obs, sd, npos, n_rows, fb, bpy, cost)
    se_lo = sd / math.sqrt(MIN_TRADES)
    se_hi = sd / math.sqrt(npos)
    nstar = zero_skill_trades(S_obs, mu_r, sd, n_rows, fb, bpy, cost)
    out.update({'gross_band': [g_lo, g_hi], 'se_band': [se_lo, se_hi],
                'n_star_zero_skill': nstar,
                'z_vs_random': ((S_obs - out['random']['sharpe_mean'])
                                / out['random']['sharpe_sd']
                                if out['random']['sharpe_sd'] > 0 else None)})
    if S_obs > 0:
        cls = '+'
    elif g_hi <= SE_MULT * se_hi:
        cls = 'a'
    elif g_lo > SE_MULT * se_lo:
        cls = 'b'
    elif nstar is not None and MIN_TRADES <= nstar <= out['n_possible']:
        cls = 'a~'
    else:
        cls = '?'
    out['proxy_class'] = cls
    return out


def layer2(rows, store, consts, flags, asset, cost, bpy, n_perm, seed=0,
           log=print):
    fold_cache = {}
    rng = np.random.default_rng(seed)
    for r in rows:
        fb = int(r['forward_bars'] or 24)
        sl = int(r['seq_len'] or 24)
        thr = float(r['trade_threshold'] if r['trade_threshold'] is not None else 0.0)
        key = ('tb', fb) if r['target_kind'] == 'tb' else fb
        rets = store['returns'].get(key)
        if rets is None:
            rets = store['returns'].get(fb)
        if rets is None:
            r['layer2'] = {'error': f'no target column for {key}'}
            continue
        if sl not in fold_cache:
            fold_cache[sl] = walk_forward_folds(store, sl, consts, flags)
        folds = fold_cache[sl]
        per = []
        k_obs = 0
        for tri, vai in folds:
            tri = tri[~np.isnan(rets[tri])]
            vai = vai[~np.isnan(rets[vai])]
            if len(tri) < 500 or len(vai) < 100:  # mirrors the objective's skip
                continue
            S = (r['fold_sharpes'][k_obs] if k_obs < len(r['fold_sharpes'])
                 else None)
            k_obs += 1
            vb = (ticker_block_ids(vai, store['bounds'])
                  if flags['OBJECTIVE_V3'] else None)
            fs = layer2_fold(rets[vai], S, thr, fb, cost,
                             flags['OBJECTIVE_LONG_ONLY'], bpy, n_perm, rng,
                             block_ids=vb)
            fs['n_train'] = int(len(tri))  # cf. the log's [CACHE] train rows
            per.append(fs)
        classes = [f.get('proxy_class') for f in per if f.get('proxy_class')]
        cls = '?'
        if classes:
            vals, cnt = np.unique(classes, return_counts=True)
            top = cnt.max()
            winners = vals[cnt == top]
            cls = str(winners[0]) if len(winners) == 1 else '?'
        r['layer2'] = {'folds': per, 'n_folds_rebuilt': len(per),
                       'fold_count_matches': len(per) == len(r['fold_sharpes']),
                       'proxy_class': cls}
    return rows


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def pooled(l3):
    n = np.asarray(l3['n_trades'], dtype=float)
    g = np.asarray(l3['gross_ret_mean'], dtype=float)
    s = np.asarray(l3['gross_ret_std'], dtype=float)
    net = np.asarray(l3['net_ret_mean'], dtype=float)
    ok = n > 0
    N = n[ok].sum()
    if N <= 0:
        return None
    gbar = float((n[ok] * g[ok]).sum() / N)
    nbar = float((n[ok] * net[ok]).sum() / N)
    ss = ((n[ok] * s[ok] ** 2) + n[ok] * (g[ok] - gbar) ** 2).sum()
    sd = math.sqrt(ss / N)
    return {'gross': gbar, 'net': nbar, 'cost_drag': gbar - nbar,
            'se': sd / math.sqrt(N), 'n': int(N)}


def economic_class_l3(l3):
    p = pooled(l3)
    if p is None:
        return 'a0', p   # no trades at all: no edge by construction
    if p['gross'] <= SE_MULT * p['se']:
        return ('a-' if p['gross'] < -SE_MULT * p['se'] else 'a0'), p
    if p['cost_drag'] >= p['gross']:
        return 'b', p
    return '+', p


def classify(row, bpy, cost):
    """Mechanical classification (RULES_TEXT) of one layer-1 row."""
    fl = row['flags']
    l3 = row.get('layer3')
    source = 'L1'
    econ = None
    if l3:
        source = 'L3'
        fb = int(row['forward_bars'] or 24)
        for n, nr, pr in zip(l3['n_trades'], l3['n_rows'],
                             l3['threshold_pass_rate']):
            if n < MIN_TRADES:
                fl['d_floor'] = True
            if pr < PASS_RATE_LO or pr > PASS_RATE_HI:
                fl['d_rate'] = True
            if n >= MIN_TRADES and trades_per_year(n, nr, fb, bpy) < 1.0:
                fl['c_ann'] = True
        e, p = economic_class_l3(l3)
        econ = e
        row['pooled'] = p
    elif row.get('layer2') and 'proxy_class' in row['layer2']:
        source = 'L2-PROXY'
        econ = row['layer2']['proxy_class']
    if fl['d_floor'] or fl['d_rate'] or fl['d_cost']:
        primary = 'd'
    elif fl['c_sign'] or fl['c_rank'] or fl['c_ann']:
        primary = 'c'
    elif econ is None:
        primary = '?'
    else:
        primary = {'a0': 'a', 'a-': 'a', 'a': 'a', 'a~': 'a', 'b': 'b',
                   '+': '+', '?': '?'}.get(econ, '?')
    row['economic'] = econ
    row['primary'] = primary
    row['source'] = source
    return row


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

IMPLICATIONS = {
    'd': "(d) -> threshold range / cost floor: under OBJECTIVE_V3 the range is floor-anchored "
         "(objective_utils.v3_trade_threshold_range); a session/entry mask changes the admit rate",
    'c': "(c) -> scoring arithmetic: the regime-penalty form (TRAINING_REPAIRS_V1 sign-correct) and "
         "the mean-0.5*std aggregation decide rank, not P&L — fix the objective before adding trials",
    'a': "(a) -> no selection edge at these horizons/features: more trials cannot help; the lever is "
         "signal (features/target/horizon) or a session mask that concentrates on where edge exists",
    'b': "(b) -> edge exists but cost eats it: the lever is the cost model (maker share / true "
         "round-trip) and a threshold range that admits only trades clearing the cost",
    '+': "(+) -> net-positive trials exist: the lever is more trials / the holdout gate, not design",
    '?': "(?) -> unresolved on the proxy: land SIG-R3-DECOMP (per-fold gross/net attrs) and re-run",
}


def _fmt(v, nd=2):
    if v is None:
        return '-'
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def render(result):
    L = []
    h = result['header']
    L.append(f"TRIAL-SCORE DECOMPOSITION — book={h['book']} study={h['study']} "
             f"db={h['db_snapshot']}")
    L.append(f"  trainer flags assumed: {h['flags']}  cost={h['txn_cost_pct']}% "
             f"bpy={h['bars_per_year']} live admission floor={h['admission_floor']}%")
    L.append(f"  log: {h['log']}   store: {h.get('store')}")
    L.append(f"  trials: {h['n_trials']} total, {h['n_complete']} COMPLETE, "
             f"{h['n_layer3']} with L3 attrs")
    L.append(RULES_TEXT)
    L.append('')
    hdr = (f"{'#':>3} {'score':>7} {'pre':>7} {'pen':>4} {'form':>10} {'mean':>6} "
           f"{'std':>5} {'folds':>20} {'rmin':>6}{'(reg)':>7} {'fb':>3} {'sl':>3} "
           f"{'thr':>5} {'tk':>3} {'pass%':>6} {'randS':>6} {'orclS':>6} "
           f"{'n*':>7} {'econ':>4} {'P':>2} flags")
    L.append(hdr)
    rows = sorted(result['trials'], key=lambda r: -r['score'])
    for r in rows:
        l2 = r.get('layer2') or {}
        f = l2.get('folds') or []
        pr = np.mean([x['pass_rate_long'] for x in f]) * 100 if f else None
        rs = np.mean([x['random']['sharpe_mean'] for x in f]) if f else None
        os_ = np.mean([x['oracle']['sharpe'] for x in f]) if f else None
        ns = [x.get('n_star_zero_skill') for x in f]
        ns = [x for x in ns if x is not None and math.isfinite(x)]
        nstar = float(np.mean(ns)) if ns else None
        flags = ','.join(k for k, v in r['flags'].items() if v)
        L.append(f"{r['number']:>3} {r['score']:>7.3f} {r['pre_penalty']:>7.3f} "
                 f"{'Y' if r['penalty_flag'] else '-':>4} {r['penalty_form'][:10]:>10} "
                 f"{r['avg_sharpe']:>6.2f} {r['std_sharpe']:>5.2f} "
                 f"{'/'.join(f'{x:.2f}' for x in r['fold_sharpes']):>20} "
                 f"{_fmt(r['regime_min']):>6}{('(' + str(r['regime_min_which'])[:4] + ')'):>7} "
                 f"{_fmt(r['forward_bars']):>3} {_fmt(r['seq_len']):>3} "
                 f"{_fmt(r['trade_threshold']):>5} {str(r['target_kind'])[:3]:>3} "
                 f"{_fmt(pr, 1):>6} {_fmt(rs):>6} {_fmt(os_):>6} {_fmt(nstar, 0):>7} "
                 f"{str(r['economic']):>4} {r['primary']:>2} {flags}")
    L.append('')
    L.append(f"PRIMARY histogram ({h['n_complete']} COMPLETE): {result['hist_primary']}")
    L.append(f"ECONOMIC histogram [{result['econ_source']}]: {result['hist_economic']}")
    L.append(f"flag counts: {result['flag_counts']}")
    L.append(f"Spearman(score, x): {result['spearman']}")
    L.append(f"Spearman(pre-penalty score, x): {result['spearman_pre']}")
    L.append(f"Spearman(random-ranker Sharpe at the trial's fb/thr/target, observed mean fold "
             f"Sharpe): {result.get('zero_skill_baseline_vs_mean_fold_sharpe')}")
    L.append('')
    L.append('WHAT THIS IMPLIES FOR THE NEXT RETRAIN (levers, not recommendations — flips are the owner\'s):')
    for line in result['implications']:
        L.append('  ' + line)
    return '\n'.join(L)


def analyse(db, study='auto', log=None, data=None, prefix='', out=None,
            n_perm=20, seed=0, long_only='auto', repairs='auto', v3='auto',
            max_rows=500_000, layer2_enabled=True, snapshot=True):
    asset = 'stock' if prefix == 'stock' else 'crypto'
    consts, parsed = trainer_constants()
    flags = trainer_flags(long_only, repairs, v3)
    cost = float(consts['TXN_COST_PCT'].get(asset, 0.6))
    bpy = bars_per_year(asset, consts['BARS_PER_YEAR'])
    adm = admission_floor(asset)
    out_dir = Path(out) if out else DEFAULT_OUT
    db_used = snapshot_db(db, out_dir) if snapshot else Path(db)
    name, trials = load_trials(db_used, study)
    log_info = parse_log(log) if log else None
    rows = layer1(trials, cost, adm, log_info)
    header = {
        'book': asset, 'study': name, 'db': str(db), 'db_snapshot': str(db_used),
        'log': (str(log) if log else None), 'flags': flags,
        'trainer_constants_parsed': parsed, 'txn_cost_pct': cost,
        'bars_per_year': bpy, 'admission_floor': adm,
        'n_trials': len(trials),
        'n_complete': len(rows),
        'n_layer3': sum(1 for r in rows if 'layer3' in r),
        'states': {s: sum(1 for t in trials if t['state'] == s)
                   for s in sorted({t['state'] for t in trials})},
        'rules': RULES_TEXT,
    }
    fbs = consts['FORWARD_BARS']
    if log_info and log_info.get('forward_bars'):
        fbs = log_info['forward_bars']
    header['forward_bars_set'] = fbs
    if layer2_enabled and rows and any('layer3' not in r for r in rows):
        t0 = time.time()
        store = load_store(prefix, data, fbs, max_rows=max_rows)
        if store is None:
            header['store'] = 'MISSING — layer 2 skipped'
        else:
            header['store'] = f"{store['path']} ({store['n_rows']} rows, {len(store['tickers'])} names)"
            layer2(rows, store, consts, flags, asset, cost, bpy, n_perm, seed)
            header['layer2_seconds'] = round(time.time() - t0, 1)
            del store
    for r in rows:
        classify(r, bpy, cost)
    hist_p, hist_e, fc = {}, {}, {}
    for r in rows:
        hist_p[r['primary']] = hist_p.get(r['primary'], 0) + 1
        e = str(r['economic'])
        hist_e[e] = hist_e.get(e, 0) + 1
        for k, v in r['flags'].items():
            fc[k] = fc.get(k, 0) + int(bool(v))
    sources = sorted({r['source'] for r in rows})
    x = {'trade_threshold': [r['trade_threshold'] for r in rows],
         'forward_bars': [r['forward_bars'] for r in rows],
         'seq_len': [r['seq_len'] for r in rows]}
    sc = [r['score'] for r in rows]
    sp = [r['pre_penalty'] for r in rows]
    impl = []
    econ_map = {'a0': 'a', 'a-': 'a', 'a': 'a', 'a~': 'a', 'b': 'b', '+': '+',
                '?': '?', 'None': '?'}
    econ_top = {}
    for k, v in hist_e.items():
        econ_top[econ_map.get(k, '?')] = econ_top.get(econ_map.get(k, '?'), 0) + v
    for k in ('d', 'c', 'a', 'b', '+', '?'):
        if hist_p.get(k, 0) or econ_top.get(k, 0):
            impl.append(f"[primary {hist_p.get(k, 0)} / economic {econ_top.get(k, 0)}] "
                        + IMPLICATIONS[k])
    # How much of the cross-trial score ordering does the ZERO-SKILL baseline
    # (random ranker at the trial's own fb / threshold / target / cost)
    # already reproduce? High rho => the horizon/threshold cost arithmetic,
    # not model skill, orders the study.
    base = [(r['avg_sharpe'],
             float(np.mean([f['random']['sharpe_mean']
                            for f in r['layer2']['folds']])))
            for r in rows if (r.get('layer2') or {}).get('folds')]
    zsb = None
    if len(base) >= 3:
        obs, rnd = zip(*base)
        zsb = spearman(rnd, obs)
        zsb['obs_minus_random_mean'] = float(np.mean(np.subtract(obs, rnd)))
    result = {
        'zero_skill_baseline_vs_mean_fold_sharpe': zsb,
        'header': header, 'trials': rows, 'hist_primary': hist_p,
        'hist_economic': hist_e, 'econ_source': '+'.join(sources),
        'flag_counts': fc,
        'spearman': {k: spearman(sc, v) for k, v in x.items()},
        'spearman_pre': {k: spearman(sp, v) for k, v in x.items()},
        'implications': impl[:10],
    }
    return result


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default=None, help="study DB (default {prefix}v2_study.db); "
                    "snapshotted into --out before opening")
    ap.add_argument('--study', default='auto', help="study name or 'auto' "
                    "(the study with the most COMPLETE trials)")
    ap.add_argument('--log', default=None, help='hypersearch_v2 stdout log')
    ap.add_argument('--data', default=None, help='training store (default: the '
                    "trainer's own parquet/CSV choice for --prefix)")
    ap.add_argument('--prefix', default='', choices=['', 'stock'])
    ap.add_argument('--json', default=None, help='JSON out (default <out>/'
                    'trial_decomposition_<book>.json)')
    ap.add_argument('--out', default=None, help=f'output dir (default {DEFAULT_OUT})')
    ap.add_argument('--n-perm', type=int, default=20, help='random-ranker permutations')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-rows', type=int, default=500_000,
                    help="mirror the trainer's --max-rows (stock chain uses 200000)")
    ap.add_argument('--long-only', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--repairs', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--v3', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--no-layer2', action='store_true')
    ap.add_argument('--no-snapshot', action='store_true',
                    help='open --db directly (read-only URI) — only for a DB '
                    'that is already a copy')
    a = ap.parse_args(argv)
    try:
        db = a.db or str(BASE_DIR / (('stock_' if a.prefix == 'stock' else '')
                                     + 'v2_study.db'))
        if not Path(db).exists():
            print(f'[decomp] no study DB at {db} — nothing to decompose')
            return 0
        out_dir = Path(a.out) if a.out else DEFAULT_OUT
        res = analyse(db, a.study, a.log, a.data, a.prefix, out_dir, a.n_perm,
                      a.seed, a.long_only, a.repairs, a.v3, a.max_rows,
                      not a.no_layer2, not a.no_snapshot)
        text = render(res)
        print(text)
        book = res['header']['book']
        jpath = Path(a.json) if a.json else out_dir / f'trial_decomposition_{book}.json'
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(_jsonable(res), indent=1))
        (out_dir / f'trial_decomposition_{book}.txt').write_text(text + '\n')
        print(f'\n[decomp] wrote {jpath} and {out_dir / f"trial_decomposition_{book}.txt"}')
    except Exception as e:  # a report never fails the caller
        print(f'[decomp] error: {type(e).__name__}: {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
