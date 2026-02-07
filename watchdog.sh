#!/bin/bash
# Watchdog: monitor hypersearch, then launch predict + crypto_loop
set -e

source activate jetson
export LD_LIBRARY_PATH="/home/kyle/miniforge3/lib/python3.12/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"

HSEARCH_PID=$1
LOG="/home/kyle/trader/pipeline_output.log"

echo "[watchdog] Monitoring hypersearch PID $HSEARCH_PID"

# Wait for hypersearch to finish
while kill -0 "$HSEARCH_PID" 2>/dev/null; do
    sleep 30
done

echo "" >> "$LOG"
echo "=== HYPERSEARCH COMPLETE ===" >> "$LOG"
echo "" >> "$LOG"

# Phase 2: test predictions
echo "=== PHASE 2: Test Predictions ===" >> "$LOG"
python -u /home/kyle/trader/predict_now.py >> "$LOG" 2>&1

echo "" >> "$LOG"

# Phase 3: launch live trading
echo "=== PHASE 3: Launch Live Trading ===" >> "$LOG"
python -u /home/kyle/trader/crypto_loop.py >> "$LOG" 2>&1
