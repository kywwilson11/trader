#!/bin/bash
# Hardware coordination for the Jetson (6 cores / 7.4 GB shared by the CEO's pipeline + 3 departments).
#   bash hwlock.sh heavy <label> -- <command...>   # run a HEAVY step (pytest file, python analysis >300 MB,
#                                                   # headless GUI, parquet load) in one of the shared slots
#   bash hwlock.sh suite <label>                    # = gate.sh (exclusive full suite)
#   bash hwlock.sh status                           # who holds what, free memory, pipeline state
# Slots: 2 heavy slots normally; 1 while a pipeline process (hypersearch/meta_label/backtest/harvest/
# run_bots) is running. Admission also waits for MemAvailable >= 1800 MB. GPU is exclusive to the CEO.
S=/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad
HW=$S/hw; mkdir -p $HW
pipeline_busy() { pgrep -f "hypersearch_v2|meta_label.py|backtest.py|harvest_(crypto|stock)_data|run_pipeline.py" >/dev/null; }  # run_bots is light (~0.5 GB, <1% CPU) and is NOT counted
mem_avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
case "${1:-}" in
  status)
    echo "MemAvailable: $(mem_avail_mb) MB   load: $(cut -d' ' -f1-3 /proc/loadavg)   pipeline_busy: $(pipeline_busy && echo yes || echo no)"
    for f in $HW/slot0 $HW/slot1 $S/gates/.suite.lock; do
      [ -e "$f" ] || continue
      if flock -n "$f" true 2>/dev/null; then echo "  $(basename $f): free"; else echo "  $(basename $f): HELD by $(cat $f 2>/dev/null | head -1)"; fi
    done
    pgrep -af "pytest|hypersearch_v2|meta_label|backtest.py|harvest_|run_bots" | grep -v pgrep | cut -c1-120 | sed 's/^/  proc: /'
    exit 0;;
  suite) shift; exec bash $S/gate.sh "${1:-gate}";;
  heavy) shift; LABEL=$1; shift; [ "$1" = "--" ] && shift;;
  *) echo "usage: hwlock.sh heavy <label> -- <cmd> | suite <label> | status"; exit 2;;
esac
touch $HW/slot0 $HW/slot1
while :; do
  NSLOTS=2; pipeline_busy && NSLOTS=1
  for i in $(seq 0 $((NSLOTS-1))); do
    exec {fd}>$HW/slot$i
    if flock -n $fd; then
      while [ "$(mem_avail_mb)" -lt 1800 ]; do sleep 10; done
      echo "$LABEL pid=$$ $(date +%H:%M:%S)" > $HW/slot$i
      export CUDA_VISIBLE_DEVICES=''
      nice -n 10 "$@"; RC=$?
      : > $HW/slot$i; exit $RC
    fi
    exec {fd}>&-
  done
  sleep 7
done
