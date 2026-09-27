#!/usr/bin/env bash
# Back up the trader's mutable state (everything that can't be re-derived
# from git or re-harvested). SQLite files are snapshotted with the online
# .backup command — a raw cp of a live Optuna DB under WAL can be corrupt.
#
# Destination:
#   - If RESTIC_REPOSITORY (+RESTIC_PASSWORD) is set: restic snapshot with
#     a 14-snapshot retention policy.
#   - Otherwise: timestamped tar.gz in ~/trader_backups (keep last 14).
#
# Cron (02:30 daily, before the 02:00 Saturday retrain finishes):
#   30 2 * * * /bin/bash /path/to/trader/scripts/backup_state.sh >> ~/trader_backups/backup.log 2>&1
set -euo pipefail

TRADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d /tmp/trader_backup.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

cd "$TRADER_DIR"

# 1. SQLite: online-consistent snapshots. The sqlite3 CLI is NOT installed
#    on the prod Jetson (2026-09-26 Jetson campaign G3-7) — fall back to the
#    stdlib online-backup API (sqlite3.Connection.backup: same page-level
#    .backup semantics, WAL-safe) in the jetson python. A bare `import
#    sqlite3` needs no LD_PRELOAD (the CXXABI clash needs torch/pandas first).
PYBIN="${TRADER_PYBIN:-/home/kyle/miniforge3/envs/jetson/bin/python}"
sqlite_backup() {  # $1 = source db, $2 = destination file
  if command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 "$1" ".backup '$2'"
  else
    "$PYBIN" -c 'import sqlite3, sys
s = sqlite3.connect(sys.argv[1]); d = sqlite3.connect(sys.argv[2])
s.backup(d); d.close(); s.close()' "$1" "$2"
  fi
}
for db in v2_study.db stock_v2_study.db; do
  if [[ -f "$db" ]]; then
    sqlite_backup "$db" "$STAGE/$db" \
      || echo "WARN: sqlite backup failed for $db"
  fi
done

# 2. JSON/flat state: positions, lockouts, adaptive state, drift state,
#    trade memory, prediction caches, model manifests, funding history
shopt -s nullglob
cp -p \
  *position_state*.json *lockout*.json adaptive_state*.json \
  drift_state.json trade_memory.json funding_history.json \
  *predictions.json pipeline_status.json indicator_config.json \
  stock_universe.json llm_config.json \
  *model_v2.manifest.json *lgb_q10_meta.json \
  "$STAGE/" 2>/dev/null || true

# 3. Model artifacts (small: a few MB) + journals
cp -p *.pth *.pkl *lgb_model.txt *lgb_q10.txt meta_model.txt \
  stock_meta_model.txt *meta_calib.pkl *meta_meta.json \
  "$STAGE/" 2>/dev/null || true
[[ -d journals ]] && cp -rp journals "$STAGE/journals"

if [[ -n "${RESTIC_REPOSITORY:-}" ]]; then
  restic backup "$STAGE" --tag trader-state
  restic forget --tag trader-state --keep-last 14 --prune
  echo "[backup] restic snapshot done -> $RESTIC_REPOSITORY"
else
  DEST="$HOME/trader_backups"
  mkdir -p "$DEST"
  OUT="$DEST/trader_state_$(date +%Y%m%d_%H%M%S).tar.gz"
  tar -czf "$OUT" -C "$STAGE" .
  ls -1t "$DEST"/trader_state_*.tar.gz | tail -n +15 | xargs -r rm -f
  echo "[backup] $OUT ($(du -h "$OUT" | cut -f1))"
fi
