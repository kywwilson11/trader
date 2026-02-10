#!/bin/bash
# Monitor the alternating search every 5 minutes
# Checks: process alive, trials progressing, no stalls

LOG="/home/kyle/trader/alternating_search.log"
MONITOR_LOG="/home/kyle/trader/monitor.log"
LAST_COUNT=0

echo "[MONITOR] Started $(date)" > "$MONITOR_LOG"

while true; do
    # Check if hypersearch process exists
    PID=$(pgrep -f "hypersearch_dual" | head -1)
    
    if [ -z "$PID" ]; then
        # Check if run_alternating.sh is still running
        ALT_PID=$(pgrep -f "run_alternating" | head -1)
        if [ -z "$ALT_PID" ]; then
            echo "[MONITOR] $(date) — SEARCH COMPLETE (no processes found)" | tee -a "$MONITOR_LOG"
            # Print final summary
            echo "--- FINAL RESULTS ---" | tee -a "$MONITOR_LOG"
            grep -E 'BEST|saved|DONE|ALL DONE' "$LOG" | tail -20 | tee -a "$MONITOR_LOG"
            exit 0
        fi
        echo "[MONITOR] $(date) — Between batches (alternating script alive, no hypersearch)" | tee -a "$MONITOR_LOG"
        sleep 300
        continue
    fi

    # Count completed trials
    TRIAL_COUNT=$(grep -c '^\[' "$LOG" 2>/dev/null || echo 0)
    
    # Get CPU usage
    CPU=$(ps -p "$PID" -o %cpu= 2>/dev/null | tr -d ' ')
    
    # Get last trial timestamp
    LAST_TIME=$(grep '^\[I ' "$LOG" | tail -1 | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || echo "unknown")
    
    # Get latest BEST
    BEST=$(grep 'BEST' "$LOG" | tail -1 | grep -oP '(bear|bull)=\S+' || echo "none yet")
    
    # Check for stall: same trial count as 5 min ago
    if [ "$TRIAL_COUNT" -eq "$LAST_COUNT" ] && [ "$TRIAL_COUNT" -gt 0 ]; then
        # How long since last trial?
        if [ "$LAST_TIME" != "unknown" ]; then
            LAST_EPOCH=$(date -d "$LAST_TIME" +%s 2>/dev/null || echo 0)
            NOW_EPOCH=$(date +%s)
            STALL_MIN=$(( (NOW_EPOCH - LAST_EPOCH) / 60 ))
            if [ "$STALL_MIN" -gt 10 ]; then
                echo "[MONITOR] $(date) — WARNING: STALLED! No new trial in ${STALL_MIN}min. Trials=$TRIAL_COUNT CPU=${CPU}% Last=$LAST_TIME" | tee -a "$MONITOR_LOG"
                sleep 300
                LAST_COUNT=$TRIAL_COUNT
                continue
            fi
        fi
    fi

    ROUND=$(grep -c 'ROUND.*BEAR\|ROUND.*BULL' "$LOG" 2>/dev/null || echo 0)
    echo "[MONITOR] $(date) — OK. Trials=$TRIAL_COUNT CPU=${CPU}% Best=$BEST Batches=$ROUND/6 Last=$LAST_TIME" | tee -a "$MONITOR_LOG"
    
    LAST_COUNT=$TRIAL_COUNT
    sleep 300
done
