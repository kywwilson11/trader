#!/bin/bash
# Alternating bear/bull hypersearch: 50 trials per batch, 150 per model total
# Uses persistent SQLite studies — Bayesian memory preserved across batches
set -e

source activate jetson
export LD_LIBRARY_PATH="/home/kyle/miniforge3/lib/python3.12/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"

BATCH=50
TOTAL_PER_MODEL=150
ROUNDS=$((TOTAL_PER_MODEL / BATCH))  # 3 rounds
LOG="/home/kyle/trader/alternating_search.log"

echo "=== ALTERNATING BEAR/BULL SEARCH ===" | tee "$LOG"
echo "  ${TOTAL_PER_MODEL} trials/model, ${ROUNDS} rounds x ${BATCH} batch" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "  Persistent studies: bear_study.db / bull_study.db" | tee -a "$LOG"

for i in $(seq 1 $ROUNDS); do
    echo "" | tee -a "$LOG"
    echo "=== ROUND $i/$ROUNDS: BEAR ($BATCH trials) ===" | tee -a "$LOG"
    python -u /home/kyle/trader/hypersearch_dual.py --target bear --trials $BATCH 2>&1 | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "=== ROUND $i/$ROUNDS: BULL ($BATCH trials) ===" | tee -a "$LOG"
    python -u /home/kyle/trader/hypersearch_dual.py --target bull --trials $BATCH 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== ALL DONE: $(date) ===" | tee -a "$LOG"
