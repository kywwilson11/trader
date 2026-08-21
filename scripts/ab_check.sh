#!/bin/sh
# ab_check.sh — one-command regression check against tests/baseline_failures.txt.
#
# Runs the dev-Mac pytest suite, extracts the FAILED/ERROR test IDs the same way
# tests/baseline_failures.txt was built, and diffs the NAME sets (never counts —
# counts drift as tests are added/removed and are not a signal of regression).
#
# Exit 0 = no NEW failing names (safe to call regression-free; some baseline
#          failures may have disappeared too, which is a bonus, not a problem).
# Exit 1 = at least one NEW failing name showed up — investigate before shipping.
#
# This replaces the from-scratch git-stash A/B ritual for the common case where
# tests/baseline_failures.txt is already current. Regenerate that file after an
# intentional change (see the command in its own header comment).
#
# Usage: bash scripts/ab_check.sh   (or ./scripts/ab_check.sh, or sh scripts/ab_check.sh)

set -eu

TIMEOUT_S="${AB_CHECK_TIMEOUT_S:-900}"          # hard wall for the full run (~15 min)
RERUN_TIMEOUT_S="${AB_CHECK_RERUN_TIMEOUT_S:-300}"
MIN_PASSED="${AB_CHECK_MIN_PASSED:-1500}"       # sanity floor: fewer passed => suite did not really run
PYTEST_CMD="${AB_CHECK_PYTEST:-python3 -m pytest}"  # overridable ONLY so the gate itself is testable (tests/test_imports_v3.py)

# Color must be OFF: some harnesses export FORCE_COLOR, and ANSI-coded output
# blinds every grep below (name extraction, summary parse) — worst case the
# empty CUR_NAMES reads as a false clean. PY_COLORS takes precedence over
# FORCE_COLOR in pytest and changes only markup, never test results.
export PY_COLORS=0

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
BASELINE="$REPO_ROOT/tests/baseline_failures.txt"

if [ ! -f "$BASELINE" ]; then
    echo "ab_check: baseline file not found: $BASELINE" >&2
    echo "ab_check: generate it first — see CLAUDE.md 'Running tests'." >&2
    exit 1
fi

RAW_OUT=$(mktemp "${TMPDIR:-/tmp}/ab_check_raw.XXXXXX")
CUR_NAMES=$(mktemp "${TMPDIR:-/tmp}/ab_check_cur.XXXXXX")
BASE_NAMES=$(mktemp "${TMPDIR:-/tmp}/ab_check_base.XXXXXX")
RERUN_OUT=$(mktemp "${TMPDIR:-/tmp}/ab_check_rerun.XXXXXX")
# Marker files signal "the watchdog killed the run": create the paths via mktemp
# then remove them, so only the watchdog ever (re)creates them.
TIMEOUT_MARKER=$(mktemp "${TMPDIR:-/tmp}/ab_check_tmark.XXXXXX"); rm -f "$TIMEOUT_MARKER"
RERUN_MARKER=$(mktemp "${TMPDIR:-/tmp}/ab_check_rmark.XXXXXX"); rm -f "$RERUN_MARKER"
trap 'rm -f "$RAW_OUT" "$CUR_NAMES" "$BASE_NAMES" "$RERUN_OUT" "$TIMEOUT_MARKER" "$RERUN_MARKER"' EXIT INT TERM HUP

# run_with_timeout <secs> <outfile> <markerfile> <cmd...>
# POSIX-sh watchdog: runs cmd with stdout+stderr captured into outfile; if it is
# still alive after <secs>, touches <markerfile> and kills it (TERM, then KILL).
# The watchdog polls in 1s ticks and detaches from our stdout/stderr — a
# long-lived orphan inheriting the caller's pipes would hold them open past our
# own exit and hang whatever spawned us (observed with subprocess pipes).
run_with_timeout() {
    _t=$1; _out=$2; _marker=$3; shift 3
    "$@" >"$_out" 2>&1 &
    _pid=$!
    ( _i=0
      while [ "$_i" -lt "$_t" ]; do
          kill -0 "$_pid" 2>/dev/null || exit 0
          sleep 1
          _i=$((_i + 1))
      done
      if kill -0 "$_pid" 2>/dev/null; then
          : >"$_marker"
          kill "$_pid" 2>/dev/null; sleep 5; kill -9 "$_pid" 2>/dev/null
      fi ) >/dev/null 2>&1 &
    _wd=$!
    _rc=0; wait "$_pid" || _rc=$?
    kill "$_wd" 2>/dev/null || true
    pkill -P "$_wd" 2>/dev/null || true   # reap the watchdog's sleep
    return "$_rc"
}

cd "$REPO_ROOT"
echo "ab_check: running $PYTEST_CMD tests/ --continue-on-collection-errors -q (timeout ${TIMEOUT_S}s) ..."
# $PYTEST_CMD intentionally unquoted so it word-splits (default: python3 -m pytest).
run_with_timeout "$TIMEOUT_S" "$RAW_OUT" "$TIMEOUT_MARKER" $PYTEST_CMD tests/ --continue-on-collection-errors -q || true
if [ -f "$TIMEOUT_MARKER" ]; then
    echo "ab_check: FAIL — pytest run TIMED OUT after ${TIMEOUT_S}s; no verdict is possible (this is NOT a pass)." >&2
    echo "ab_check: last output lines:" >&2
    tail -n 20 "$RAW_OUT" >&2
    exit 1
fi

grep -E '^(FAILED|ERROR)' "$RAW_OUT" | sed 's/ - .*//' | sort -u >"$CUR_NAMES"
grep -v '^#' "$BASELINE" | sed '/^[[:space:]]*$/d' | sort -u >"$BASE_NAMES"

NEW_NAMES=$(comm -23 "$CUR_NAMES" "$BASE_NAMES")
GONE_NAMES=$(comm -13 "$CUR_NAMES" "$BASE_NAMES")

SUMMARY_LINE=$(grep -E '^=+ .* =+$' "$RAW_OUT" | tail -n 1)
if [ -z "$SUMMARY_LINE" ]; then
    SUMMARY_LINE=$(tail -n 1 "$RAW_OUT")
fi

# Launch-sanity gate: an empty/garbage RAW_OUT (pytest missing, launch failure,
# partial run) must never turn into a silent PASS via an empty CUR_NAMES set.
PASSED_COUNT=$(printf '%s\n' "$SUMMARY_LINE" | grep -Eo '[0-9]+ passed' | grep -Eo '[0-9]+' || true)
if [ -z "$PASSED_COUNT" ] || [ "$PASSED_COUNT" -lt "$MIN_PASSED" ]; then
    echo "ab_check: FAIL — pytest summary line missing or passed count (${PASSED_COUNT:-0}) below sanity floor ${MIN_PASSED}." >&2
    echo "ab_check: the suite did not actually run to completion; refusing to report PASS. Last output lines:" >&2
    tail -n 20 "$RAW_OUT" >&2
    exit 1
fi

echo ""
echo "ab_check: suite summary: $SUMMARY_LINE"
echo ""

STATUS=0
if [ -n "$NEW_NAMES" ]; then
    STATUS=1
    echo "NEW failures (regressions — not in baseline):"
    echo "$NEW_NAMES" | sed 's/^/  /'

    # Flaky/persistent triage: rerun ONLY the new node ids once, in isolation.
    # Labels are informational — any NEW name still exits 1, flaky or not.
    NEW_IDS=$(printf '%s\n' "$NEW_NAMES" | sed 's/^[A-Z]* //')
    # Node ids contain glob chars ([param]) — disable globbing and split the
    # command line in two IFS stages: PYTEST_CMD on spaces, NEW_IDS on newlines.
    set -f; OLD_IFS=$IFS
    IFS=' '; set -- $PYTEST_CMD
    IFS='
'
    set -- "$@" $NEW_IDS --continue-on-collection-errors -q
    IFS=$OLD_IFS; set +f
    run_with_timeout "$RERUN_TIMEOUT_S" "$RERUN_OUT" "$RERUN_MARKER" "$@" || true

    echo ""
    if [ -f "$RERUN_MARKER" ]; then
        echo "ab_check: targeted rerun TIMED OUT — treating all NEW names as persistent"
        printf '%s\n' "$NEW_NAMES" | while read -r name; do
            [ -n "$name" ] || continue
            echo "  $name  [persistent]"
        done
    else
        RERUN_FAILED=$(grep -E '^(FAILED|ERROR)' "$RERUN_OUT" | sed 's/ - .*//' | sort -u)
        printf '%s\n' "$NEW_NAMES" | while read -r name; do
            [ -n "$name" ] || continue
            if printf '%s\n' "$RERUN_FAILED" | grep -Fxq "$name"; then
                echo "  $name  [persistent]"
            else
                echo "  $name  [flaky — failed in full run, passed in isolation; STILL A FAIL, investigate ordering/state]"
            fi
        done
    fi
else
    echo "NEW failures: none"
fi

echo ""
if [ -n "$GONE_NAMES" ]; then
    echo "DISAPPEARED failures (in baseline, passing now):"
    echo "$GONE_NAMES" | sed 's/^/  /'
else
    echo "DISAPPEARED failures: none"
fi

echo ""
if [ "$STATUS" -eq 0 ]; then
    echo "ab_check: PASS — zero regressions vs tests/baseline_failures.txt"
else
    echo "ab_check: FAIL — new failing test names vs tests/baseline_failures.txt"
fi

exit "$STATUS"
