"""Optuna orphan-trial helper — list RUNNING trials; mark them FAIL on --apply.

Why: a hypersearch_v2 process that dies mid-trial (kill, OOM-kill, reboot)
leaves its current trial in state RUNNING forever in `{prefix}v2_study.db`
(e.g. v2_study.db trial #4, started 2026-09-27 01:01:58, whose process was
stopped for the landing window). TPE ignores RUNNING trials unless
`constant_liar` is set (optuna 4.7.0 samplers/_tpe/sampler.py:528-531;
hypersearch_v2's TPESampler does not set it), but study.trials, the
trial counters and the GUI show a never-ending trial, and nothing ever
finishes it. Marking it FAIL is the
honest terminal state: FAIL is excluded from best_trial, from every
COMPLETE-only pool counter (hypersearch_v2 n_trials_pool / cum_trials) and
from TPE.

Safety contract:
  * DEFAULT = dry run. Listing opens the DB READ-ONLY through sqlite3
    (`file:<db>?mode=ro`, uri=True): the file is never written, never
    created, its mtime never changes. Optuna is not even imported.
  * --apply refuses (exit 3) while ANY hypersearch_v2 process is alive
    (pgrep), because the newest RUNNING trial then belongs to a live
    trainer. With no trainer alive every RUNNING trial is an orphan.
  * --apply sets state through the public storage API of optuna 4.7.0:
      optuna.storages.RDBStorage.set_trial_state_values(
          trial_id, state=TrialState.FAIL, values=None)
    (storages/_base.py:346-375 contract; RDB impl storages/_rdb/storage.py:
    653-677 — check_trial_is_updatable, trial.state = state, and
    datetime_complete = now() because FAIL.is_finished()). It never touches
    COMPLETE/PRUNED/FAIL trials (a finished trial raises
    UpdateFinishedTrialError there) and never writes trial values.
  * --backup PATH copies the DB (sqlite online backup) before the write.

Usage:
  python scripts/optuna_fail_orphans.py --db v2_study.db            # list
  python scripts/optuna_fail_orphans.py --db v2_study.db --apply    # FAIL them
  --study NAME|auto (auto = every study in the DB, discovered with
  optuna.get_all_study_summaries on --apply / the `studies` table on a dry
  run), --trial N [N ...] (restrict to these trial numbers),
  --min-age-min M (only RUNNING trials started >= M minutes ago).

Exit: 0 ok (incl. nothing to do), 2 bad input (missing DB, unknown study),
3 refused (--apply while a trainer is alive).
"""
import argparse
import datetime as _dt
import os
import sqlite3
import subprocess
import sys

TRAINER_PATTERN = r'python[0-9.]*( .*)? ([^ ]*/)?hypersearch_v2\.py'


def live_trainers():
    """PIDs of running hypersearch_v2 processes (never this process or its
    parent). Uses `pgrep -f`; if pgrep itself is unavailable the answer is
    UNKNOWN and we report a sentinel pid -1 so --apply refuses (fail closed)."""
    try:
        out = subprocess.run(['pgrep', '-f', TRAINER_PATTERN],
                             capture_output=True, text=True, timeout=10)
    except Exception:
        return [-1]
    if out.returncode not in (0, 1):
        return [-1]
    me = {os.getpid(), os.getppid()}
    pids = []
    for p in out.stdout.split():
        if not p.strip().isdigit() or int(p) in me:
            continue
        try:  # a shell whose command TEXT mentions the name is not a trainer
            with open(f'/proc/{p}/comm') as f:
                if not f.read().strip().startswith('python'):
                    continue
        except OSError:
            pass  # unknown -> keep it (fail closed)
        pids.append(int(p))
    return pids


def _parse_dt(s):
    if s is None:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return _dt.datetime.strptime(str(s), fmt)
        except ValueError:
            pass
    return None


def list_running_readonly(db):
    """{study_name: [(trial_id, number, datetime_start), ...]} of RUNNING
    trials, read through a read-only sqlite connection (no optuna)."""
    con = sqlite3.connect(f'file:{os.path.abspath(db)}?mode=ro', uri=True)
    try:
        studies = {sid: name for sid, name in
                   con.execute('SELECT study_id, study_name FROM studies')}
        out = {name: [] for name in studies.values()}
        rows = con.execute(
            "SELECT trial_id, number, study_id, datetime_start FROM trials "
            "WHERE state = 'RUNNING' ORDER BY study_id, number")
        for tid, num, sid, start in rows:
            out[studies[sid]].append((int(tid), int(num), _parse_dt(start)))
        return out
    finally:
        con.close()


def select(running, study, trials=None, min_age_min=None, now=None):
    """Filter the RUNNING map by study name / trial numbers / minimum age.
    Returns [(study_name, trial_id, number, start, age_min)]."""
    now = now or _dt.datetime.now()
    names = sorted(running) if study in (None, 'auto') else [study]
    sel = []
    for name in names:
        for tid, num, start in running.get(name, []):
            age = (now - start).total_seconds() / 60.0 if start else None
            if trials and num not in trials:
                continue
            if min_age_min is not None and (age is None or age < min_age_min):
                continue
            sel.append((name, tid, num, start, age))
    return sel


def _fmt(sel, newest):
    lines = []
    for name, tid, num, start, age in sel:
        tag = '  <- newest RUNNING (live trial if a trainer runs)' \
            if newest.get(name) == tid else ''
        age_s = f'{age:8.1f} min' if age is not None else '       ? min'
        lines.append(f'  study={name} trial #{num} (trial_id={tid}) '
                     f'start={start} age={age_s}{tag}')
    return lines


def apply_fail(db, sel, backup=None):
    """Set every selected trial to FAIL via the optuna storage API.
    Returns [(number, changed)]."""
    import optuna
    from optuna.trial import TrialState
    storage = optuna.storages.RDBStorage(f'sqlite:///{os.path.abspath(db)}')
    summaries = optuna.get_all_study_summaries(storage,
                                               include_best_trial=False)
    known = {s.study_name for s in summaries}
    if backup:
        src = sqlite3.connect(os.path.abspath(db))
        dst = sqlite3.connect(os.path.abspath(backup))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        print(f'[backup] {db} -> {backup}')
    done = []
    for name, tid, num, _start, _age in sel:
        if name not in known:
            raise SystemExit(f'study {name!r} vanished from {db}')
        frozen = storage.get_trial(tid)
        if frozen.state != TrialState.RUNNING or frozen.number != num:
            print(f'  skip trial #{num}: state now {frozen.state.name}')
            done.append((num, False))
            continue
        ok = storage.set_trial_state_values(tid, state=TrialState.FAIL)
        done.append((num, bool(ok)))
        print(f'  trial #{num} (trial_id={tid}, study={name}): RUNNING -> '
              f'{"FAIL" if ok else "UNCHANGED"}')
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--db', default='v2_study.db',
                    help='Optuna sqlite DB (default v2_study.db; the stock '
                         'book uses stock_v2_study.db)')
    ap.add_argument('--study', default='auto',
                    help="study name, or 'auto' = every study in the DB "
                         "(hypersearch_v2 names it v2_search)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--apply', action='store_true',
                   help='mark the listed RUNNING trials FAIL (refused while '
                        'a hypersearch_v2 process is alive)')
    g.add_argument('--dry-run', action='store_true',
                   help='list only (the default)')
    ap.add_argument('--trial', type=int, nargs='+', default=None,
                    help='restrict to these trial NUMBERS')
    ap.add_argument('--min-age-min', type=float, default=None,
                    help='only RUNNING trials started at least this many '
                         'minutes ago')
    ap.add_argument('--backup', default=None,
                    help='with --apply: sqlite-backup the DB here first')
    a = ap.parse_args(argv)

    if not os.path.isfile(a.db):
        print(f'ERROR: no such DB file: {a.db} (nothing created)')
        return 2
    running = list_running_readonly(a.db)
    if a.study not in ('auto',) and a.study not in running:
        print(f'ERROR: study {a.study!r} not in {a.db}; studies: '
              f'{sorted(running)}')
        return 2
    newest = {}
    for name, trs in running.items():
        if trs:
            newest[name] = max(trs, key=lambda t: (t[2] or _dt.datetime.min,
                                                   t[1]))[0]
    sel = select(running, a.study, set(a.trial) if a.trial else None,
                 a.min_age_min)
    print(f'[orphans] db={a.db} studies={sorted(running)} '
          f'RUNNING selected={len(sel)} mode={"APPLY" if a.apply else "dry-run"}')
    for ln in _fmt(sel, newest):
        print(ln)
    if not a.apply:
        if sel:
            print('[orphans] dry run — nothing written. Re-run with --apply '
                  'once no hypersearch_v2 process is alive.')
        return 0
    pids = live_trainers()
    if pids:
        print(f'REFUSED: hypersearch_v2 process(es) alive {pids} — the newest '
              f'RUNNING trial may be live. Stop the trainer first. '
              f'DB untouched.')
        return 3
    if not sel:
        print('[orphans] nothing to do — DB untouched.')
        return 0
    apply_fail(a.db, sel, backup=a.backup)
    return 0


if __name__ == '__main__':
    sys.exit(main())
