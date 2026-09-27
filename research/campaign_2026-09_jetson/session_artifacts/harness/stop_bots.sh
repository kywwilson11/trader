#!/usr/bin/env bash
# stop_bots.sh — stop / halt the harness-launched run_bots.py.
#
#   stop_bots.sh            (= stop) SIGTERM the pid in phase5/bots.pid, wait up to
#                           15 s, then SIGKILL (+ up to 5 s). Mirrors
#                           run_pipeline._stop_bots (terminate -> wait -> kill), with a
#                           longer grace than its 10 s. run_bots' SIGTERM handler sets
#                           _shutdown and main() returns within ~5 s; the loop threads
#                           are daemons and die with it. Resting broker-side stops /
#                           GTC orders are NOT cancelled on shutdown.
#   stop_bots.sh halt       touch-equivalent of notify.set_halt(): writes
#                           <repo>/trading_halt.flag. The process KEEPS RUNNING: no new
#                           entries (base_loop._entries_allowed checks the flag every
#                           cycle), but stop management, signal/LLM-veto sells and the
#                           circuit breaker keep running. Takes effect next cycle (<=~35 s).
#   stop_bots.sh resume     remove <repo>/trading_halt.flag (= notify.clear_halt / /resume).
#   stop_bots.sh status     show pid liveness + which control flag files exist.
#
# WHEN TO USE WHICH
#   halt  : you want to stop NEW risk but keep observing / keep protecting open
#           positions (exits + stops still run). First response to anything odd.
#   stop  : end of the observation window, memory/thermal emergency, crash loop,
#           or anything where the process itself is the problem. Open positions are
#           then unmanaged by the bot (only resting broker orders remain).
#   (flatten is deliberately NOT a sub-command: it SELLS the book. If ever needed:
#    `touch /home/kyle/trader/flatten_request.flag` — each loop fans it out to
#    flatten_crypto.flag / flatten_stock.flag, liquidates its own book within one
#    cycle, and sets trading_halt.flag. Per-book flags older than 1 h are discarded.)
set -u

REPO="${HARNESS_REPO:-/home/kyle/trader}"   # override only for testing
PHASE5="${HARNESS_PHASE5:-/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/phase5}"
PIDFILE="$PHASE5/bots.pid"
HALT_FLAG="$REPO/trading_halt.flag"          # notify.py:131 _HALT_FLAG
FLAGS=("$REPO/trading_halt.flag" "$REPO/flatten_request.flag" "$REPO/flatten_crypto.flag" "$REPO/flatten_stock.flag")

cmd="${1:-stop}"

readpid() { [ -f "$PIDFILE" ] && tr -dc '0-9' < "$PIDFILE"; }

case "$cmd" in
  halt)
    # same payload shape notify.set_halt writes
    printf '{"reason": "phase5 harness halt", "ts": "%s"}' "$(date +%Y-%m-%dT%H:%M:%S)" > "$HALT_FLAG" \
      && echo "HALT set: $HALT_FLAG (entries blocked from next cycle; exits/stops continue; process untouched)"
    exit $?
    ;;
  resume)
    if [ -f "$HALT_FLAG" ]; then rm -f "$HALT_FLAG" && echo "HALT cleared: removed $HALT_FLAG"; else echo "no halt flag present"; fi
    exit 0
    ;;
  status)
    PID="$(readpid)"
    if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
      echo "bots: PID $PID ALIVE  $(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)"
      grep -E '^(VmRSS|VmHWM|Threads)' /proc/$PID/status 2>/dev/null | tr -s ' \t' ' ' | paste -sd' '
    else
      echo "bots: not running (pid file: ${PID:-none})"
    fi
    for f in "${FLAGS[@]}"; do [ -e "$f" ] && echo "flag present: $f  ($(cat "$f" 2>/dev/null | head -c 120))"; done
    exit 0
    ;;
  stop) ;;
  -h|--help) sed -n 2,28p "$0"; exit 0 ;;
  *) echo "usage: $0 [stop|halt|resume|status]" >&2; exit 2 ;;
esac

PID="$(readpid)"
if [ -z "${PID:-}" ]; then echo "no pid in $PIDFILE — nothing to stop"; exit 0; fi
if ! kill -0 "$PID" 2>/dev/null; then echo "PID $PID not running (stale pid file)"; exit 0; fi
CMDLINE="$(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)"
case "$CMDLINE" in
  *run_bots.py*) ;;
  *) echo "REFUSING: PID $PID is not run_bots.py (cmdline: $CMDLINE)" >&2; exit 1 ;;
esac

HWM="$(grep -E '^VmHWM' /proc/$PID/status 2>/dev/null | tr -s ' \t' ' ')"
echo "SIGTERM -> PID $PID ($CMDLINE) [$HWM]"
kill -TERM "$PID"
T0=$(date +%s)
for _ in $(seq 1 150); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 0.1
done
if ! kill -0 "$PID" 2>/dev/null; then
  echo "EXITED CLEANLY after SIGTERM in $(( $(date +%s) - T0 )) s"
  RC=0
else
  echo "still alive after 15 s — SIGKILL"
  kill -KILL "$PID" 2>/dev/null
  for _ in $(seq 1 50); do kill -0 "$PID" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$PID" 2>/dev/null; then echo "ERROR: PID $PID survived SIGKILL" >&2; exit 1; fi
  echo "KILLED (SIGKILL) — NOT a clean exit"
  RC=3
fi
# any orphans from the same session (there should be none: run_bots spawns no children)
LEFT="$(pgrep -s "$PID" -a 2>/dev/null || true)"
[ -n "$LEFT" ] && echo "WARNING: processes left in session $PID:" && echo "$LEFT"
mv -f "$PIDFILE" "$PIDFILE.last" 2>/dev/null
echo "$(date -Is) stopped PID $PID rc=$RC" >> "$PHASE5/bots_stop.log"
exit $RC
