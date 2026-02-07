#!/bin/bash
# Full pipeline: hyperparameter search -> deploy best model -> run live trading
set -e

source activate jetson
export LD_LIBRARY_PATH="/home/kyle/miniforge3/lib/python3.12/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"

echo "=== PHASE 1a: Harvest Fresh Data ==="
python -u harvest_data.py

echo ""
echo "=== PHASE 1b: Hyperparameter Search (500 iterations) ==="
python -u hypersearch.py

echo ""
echo "=== PHASE 2: Test Predictions ==="
python -u predict_now.py

echo ""
echo "=== PHASE 3: Launch Live Trading ==="
python -u crypto_loop.py
