#!/usr/bin/env bash
# start_bots.sh — launch run_bots.py (combined mode: both loops as threads in ONE
# process) exactly as run_pipeline._launch_bots would under BOT_ENV
# (run_pipeline.py:276-311, :677-697), but detached (setsid + nohup) and with
# stdout/stderr going to the phase5 scratch log instead of crypto_bot_output.log.
#
# Usage:  start_bots.sh [--crypto-only | --stock-only] [--dry-run]
#   --crypto-only / --stock-only   passed straight through to run_bots.py
#   --dry-run                      run every safety check, print the command +
#                                  env, and exit WITHOUT launching anything
#
# Refuses to start if phase5/bots.pid points at a live process, or if ANY
# run_bots.py / crypto_loop.py / stock_loop.py / run_pipeline.py process is
# already running (two bot processes on one account would fight over orders).
set -u

REPO="${HARNESS_REPO:-/home/kyle/trader}"          # overrides exist ONLY for the fake-bot self-test
PHASE5="${HARNESS_PHASE5:-/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/phase5}"
PIDFILE="$PHASE5/bots.pid"
OUTLOG="$PHASE5/bots_stdout.log"
STARTFILE="$PHASE5/bots_started.json"
JPY=/home/kyle/miniforge3/envs/jetson/bin/python

BOOK_ARG=""
DRY=0
for a in "$@"; do
  case "$a" in
    --crypto-only|--stock-only)
      if [ -n "$BOOK_ARG" ]; then echo "ERROR: pass at most one of --crypto-only/--stock-only" >&2; exit 2; fi
      BOOK_ARG="$a" ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument: $a" >&2; exit 2 ;;
  esac
done

mkdir -p "$PHASE5"

# --- safety check 1: pid file pointing at a live process -------------------
if [ -f "$PIDFILE" ]; then
  OLDPID="$(tr -dc '0-9' < "$PIDFILE")"
  if [ -n "$OLDPID" ] && kill -0 "$OLDPID" 2>/dev/null; then
    echo "REFUSING: $PIDFILE -> PID $OLDPID is alive: $(tr '\0' ' ' < /proc/$OLDPID/cmdline 2>/dev/null)" >&2
    echo "          stop it first with stop_bots.sh (or remove a stale pid file by hand)." >&2
    exit 1
  fi
  echo "note: stale pid file ($OLDPID not running) — will be overwritten"
fi

# --- safety check 2: any other bot / pipeline process ------------------------
OTHERS="$(pgrep -af 'run_bots\.py|crypto_loop\.py|stock_loop\.py|run_pipeline\.py' | grep -v -e pgrep -e start_bots.sh || true)"
if [ -n "$OTHERS" ]; then
  echo "REFUSING: bot/pipeline process(es) already running:" >&2
  echo "$OTHERS" >&2
  exit 1
fi

# --- warning only: heavy jobs sharing the 8 GB unified memory -----------------
HEAVY="$(pgrep -af 'hypersearch_v2\.py|harvest_(crypto|stock)_data\.py|meta_label\.py|backtest\.py' | grep -v pgrep || true)"
if [ -n "$HEAVY" ]; then
  echo "WARNING: training/harvest/backtest process(es) running alongside (bots add ~0.65-1.0 GB RSS):"
  echo "$HEAVY" | cut -c1-160
fi

# --- environment: run_pipeline.ENV + BOT_ENV, verbatim values ----------------
export LD_LIBRARY_PATH="/home/kyle/miniforge3/envs/jetson/lib:/home/kyle/miniforge3/envs/jetson/lib/python3.10/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD=/home/kyle/miniforge3/envs/jetson/lib/libstdc++.so.6
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export TORCH_NUM_THREADS=2

CMD=("$JPY" -u run_bots.py)
[ -n "$BOOK_ARG" ] && CMD+=("$BOOK_ARG")

echo "cwd : $REPO"
echo "cmd : ${CMD[*]}"
echo "env : CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' OMP_NUM_THREADS=$OMP_NUM_THREADS TORCH_NUM_THREADS=$TORCH_NUM_THREADS PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
echo "      LD_PRELOAD=$LD_PRELOAD"
echo "      LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "log : $OUTLOG"
echo "pid : $PIDFILE"
[ -f "$REPO/trading_halt.flag" ] && echo "WARNING: $REPO/trading_halt.flag exists — bots will start HALTED (no new entries)."

if [ "$DRY" = 1 ]; then
  echo "DRY RUN — all checks passed, nothing launched."
  exit 0
fi

# Snapshot the LLM daily-cost file so summarize.py can compute the spend delta.
cp -f "$REPO/llm_cost.json" "$PHASE5/llm_cost_start.json" 2>/dev/null || true

cd "$REPO" || { echo "ERROR: cannot cd $REPO" >&2; exit 1; }
{ echo; echo "===== start_bots.sh $(date -Is) : ${CMD[*]} ====="; } >> "$OUTLOG"

# setsid -> new session (survives the launching shell); the inner bash writes
# ITS OWN pid and then execs python, so the pid file is the python pid exactly
# (no setsid fork ambiguity).
nohup setsid bash -c 'echo $$ > "$0"; exec "$@"' "$PIDFILE" "${CMD[@]}" \
  >> "$OUTLOG" 2>&1 < /dev/null &

# wait for the pid file (inner bash writes it immediately)
for _ in $(seq 1 50); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
NEWPID="$(tr -dc '0-9' < "$PIDFILE" 2>/dev/null)"
if [ -z "$NEWPID" ]; then echo "ERROR: pid file not written" >&2; exit 1; fi
sleep 3
if kill -0 "$NEWPID" 2>/dev/null; then
  printf '{"pid": %s, "started_at": "%s", "started_epoch": %s, "args": "%s"}\n' \
    "$NEWPID" "$(date -Is)" "$(date +%s)" "$BOOK_ARG" > "$STARTFILE"
  echo "STARTED run_bots.py PID $NEWPID ($(tr '\0' ' ' < /proc/$NEWPID/cmdline))"
  echo "tail -f $OUTLOG"
else
  echo "ERROR: PID $NEWPID died within 3 s — last log lines:" >&2
  tail -n 30 "$OUTLOG" >&2
  exit 1
fi
