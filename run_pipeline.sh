#!/bin/bash
# Full pipeline: harvest data -> dual hypersearch (bear+bull) -> test predictions -> live trading
set -e

source activate jetson
export LD_LIBRARY_PATH="/home/kyle/miniforge3/lib/python3.12/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"

echo "=== PHASE 1a: Harvest Fresh Data ==="
python -u harvest_data.py

echo ""
echo "=== PHASE 1b: Bear Hyperparameter Search (250 trials) ==="
python -u hypersearch_dual.py --target bear

echo ""
echo "=== PHASE 1c: Bull Hyperparameter Search (250 trials) ==="
python -u hypersearch_dual.py --target bull

echo ""
echo "=== PHASE 2: Test Predictions ==="
python -u predict_now.py

echo ""
echo "=== PHASE 3: Launch Live Trading ==="
python -u crypto_loop.py
