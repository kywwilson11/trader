#!/bin/bash
# Serialized full-suite regression gate. Tests a SNAPSHOT of the tree (rsync copy, data excluded) so a gate
# certifies exactly the tree it started from, even while other departments keep landing files.
set -u
LABEL=${1:-gate}
S=/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad
. $S/jenv.sh; unset PYTHONPATH; export CUDA_VISIBLE_DEVICES=''
OUT=$S/gates/${LABEL}_$(date +%Y%m%d_%H%M%S).log; mkdir -p $S/gates
# memory admission FIRST (never park while holding the lock — that starved every other department on 2026-09-27)
mem() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
while [ "$(mem)" -lt 2200 ]; do sleep 15; done
exec 9>$S/gates/.suite.lock
echo "[gate] waiting for the suite lock ($LABEL)…"; flock 9; echo "[gate] lock acquired $(date)"
# bounded re-check inside the lock: if memory vanished while we queued, release and re-queue instead of parking
n=0; while [ "$(mem)" -lt 2200 ]; do n=$((n+1)); if [ $n -gt 8 ]; then flock -u 9; echo "[gate] memory dropped while queued — released the lock, re-queueing"; exec bash "$0" "$LABEL"; fi; sleep 15; done
SNAP=$S/gates/snapshot
rsync -a --delete --exclude-from=$S/gate_snapshot_excludes.txt /home/kyle/trader/ $SNAP/ || { echo "[gate] snapshot failed"; exit 2; }
SNAP_TS=$(date +%H:%M:%S); echo "[gate] snapshot taken $SNAP_TS ($(du -sh $SNAP | cut -f1)) — edits after this instant are NOT in this gate"
cd $SNAP
timeout 1800 nice -n 5 $JPY -m pytest tests/ --continue-on-collection-errors -q -p no:cacheprovider > "$OUT" 2>&1; RC=$?
tail -1 "$OUT"; echo "exit=$RC log=$OUT snapshot_at=$SNAP_TS"
grep -E '^(FAILED|ERROR)' "$OUT" | sed 's/ - .*//' | sort -u
[ $RC -eq 0 ] && echo "[gate] GREEN (tree as of $SNAP_TS)" || echo "[gate] RED — fix or revert before handing back"
exit $RC
