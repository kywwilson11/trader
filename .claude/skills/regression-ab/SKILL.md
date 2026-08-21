---
name: regression-ab
description: Canonical zero-regression check for this repo. Primary method — `bash scripts/ab_check.sh`, which diffs FAILED/ERROR test NAMES against the committed tests/baseline_failures.txt (no stash, no counts). Fallback — full git-stash A/B when the baseline might be stale. Use before declaring any change regression-free, or whenever asked to verify the suite.
---

# /regression-ab — zero-regression check

Proves your uncommitted changes introduce **zero new test failures** on the dev Mac, where a
fixed set of tests always fails from missing heavy deps (torch / lightgbm / optuna / joblib /
numba / sklearn / dotenv). **Counts are advisory; the failure-NAME diff is the truth.**

Baseline for orientation (2026-07-15): `1887 passed / 21 failed / 15 skipped / 7 errors` —
all failures+errors are pre-existing missing-dep noise. Counts drift as tests are added; never
alarm (or all-clear) on counts alone.

## Primary method — `scripts/ab_check.sh` (one command, no stash)

```bash
bash scripts/ab_check.sh
```

Runs the suite once, diffs the current FAILED/ERROR test names against the committed snapshot
`tests/baseline_failures.txt`, prints NEW vs DISAPPEARED names separately, and exits 0 iff no
NEW names appear (1 otherwise). It never touches git state — no stash, no pop — so it is safe to
run even while sibling agents are actively editing elsewhere in the tree.

Use this by default. Drop to the fallback ritual below when:
- `ab_check.sh` reports NEW names you can't explain from your own diff, or you otherwise suspect
  `tests/baseline_failures.txt` is stale (e.g. it predates several unrelated commits);
- you need to regenerate the baseline itself — see the regen command in the file's own 3-line
  header comment, then re-add that header (it is MACHINE-SPECIFIC to this no-heavy-deps dev Mac);
- `scripts/ab_check.sh` is missing or broken.

## Fallback method — git-stash A/B (from scratch, no baseline file needed)

### Preconditions
- Dev Mac only (no heavy deps installed). Missing-dep failures are normal — do not "fix" them.
- **No sibling agent is actively editing.** `git stash` sweeps the WHOLE tree; a concurrent
  writer mid-edit will be disrupted and `git stash pop` may conflict. If other agents are
  active, wait or get the user's go-ahead first.
- The suite runs twice (several minutes total). Never interrupt between stash and pop.

### Ritual (run exactly this)
```bash
# 1. WITH your changes
python3 -m pytest tests/ --continue-on-collection-errors -q 2>&1 \
  | grep -oE '^(FAILED|ERROR) [^[:space:]]+' | sort > /tmp/ab_with.txt

# 2. Baseline (changes stashed)
git stash push -m "regression-ab"
python3 -m pytest tests/ --continue-on-collection-errors -q 2>&1 \
  | grep -oE '^(FAILED|ERROR) [^[:space:]]+' | sort > /tmp/ab_base.txt
git stash pop

# 3. Compare NAMES
diff /tmp/ab_base.txt /tmp/ab_with.txt && echo "ZERO REGRESSIONS"
```

### Interpreting the diff
- **Empty diff → zero regressions.** Report the counts plus "failure set identical".
- Lines starting `>` (present only WITH your changes) = **new failures — regressions.**
  Investigate every one in a file you touched; fix YOUR edit, never weaken the test.
- Lines starting `<` (present only in baseline) = failures your changes **fixed** — report as
  wins. (New untracked test files also land here: plain `git stash` leaves untracked files in
  place, so brand-new tests run against the stashed old source and fail in baseline. Expected.)
- A `>` failure in a file you did NOT touch: check `git status` — it is likely a concurrent
  sibling's edit. Say so explicitly instead of claiming or fixing it.

### Recovery
- If `git stash pop` conflicts: STOP. Do not resolve destructively; show the user `git status`
  and `git stash list` and ask.
- Never finish with a leftover `regression-ab` stash entry — confirm `git stash list` is clean
  at the end.

(If the baseline numbers above have drifted, trust the diff, mention the drift, and only update
CLAUDE.md's baseline with the user's confirmation. Same rule for `tests/baseline_failures.txt`:
regenerate it the same way, subject to the same confirmation.)
