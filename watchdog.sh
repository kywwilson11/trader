#!/bin/bash
# Watchdog: launch evolve.py (continuous improvement) alongside crypto_loop.py and stock_loop.py
set -e

source activate jetson
export LD_LIBRARY_PATH="/home/kyle/miniforge3/lib/python3.12/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"

LOG="/home/kyle/trader/pipeline_output.log"
EVOLVE_LOG="/home/kyle/trader/evolve_output.log"
STOCK_LOG="/home/kyle/trader/stock_loop_output.log"

echo "[watchdog] Starting continuous improvement pipeline"
echo "[watchdog] $(date)" >> "$LOG"

# Launch evolve.py in background (3-day retrain cycle for crypto + stock models)
echo "[watchdog] Starting evolve.py..." >> "$LOG"
nohup python -u /home/kyle/trader/evolve.py >> "$EVOLVE_LOG" 2>&1 &
EVOLVE_PID=$!
echo "[watchdog] evolve.py PID: $EVOLVE_PID" >> "$LOG"

# Launch stock_loop.py in background (market-hours stock trading)
echo "[watchdog] Starting stock_loop.py..." >> "$LOG"
nohup python -u /home/kyle/trader/stock_loop.py >> "$STOCK_LOG" 2>&1 &
STOCK_PID=$!
echo "[watchdog] stock_loop.py PID: $STOCK_PID" >> "$LOG"

# Launch crypto_loop.py in foreground (24/7 crypto trading with hot-reload)
echo "[watchdog] Starting crypto_loop.py..." >> "$LOG"
python -u /home/kyle/trader/crypto_loop.py >> "$LOG" 2>&1
