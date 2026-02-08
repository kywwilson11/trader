#!/bin/bash
# Trader setup script for Ubuntu / Jetson Orin Nano
#
# Usage:
#   ./setup.sh              # Desktop Ubuntu (pip PyTorch)
#   ./setup.sh --jetson     # Jetson Orin Nano (JetPack 6.x)
#
# Prerequisites:
#   - Python 3.10+ (miniforge/conda recommended)
#   - CUDA toolkit (Jetson: included with JetPack)

set -euo pipefail

JETSON=false
if [[ "${1:-}" == "--jetson" ]]; then
    JETSON=true
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "Trader Setup"
echo "========================================"

# --- Check Python ---
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.10+ first."
    exit 1
fi

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYTHON ($PY_VER)"

# --- Install PyTorch ---
if $JETSON; then
    echo ""
    echo "Installing PyTorch for Jetson (JetPack 6.x, CUDA 12.6)..."
    "$PYTHON" -m pip install torch==2.8.0 torchvision==0.23.0 \
        --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

    echo ""
    echo "Installing Jetson dependencies..."
    "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements-jetson.txt"
else
    echo ""
    echo "Installing PyTorch (pip, CUDA auto-detect)..."
    "$PYTHON" -m pip install torch torchvision

    echo ""
    echo "Installing dependencies..."
    "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# --- Create .env if missing ---
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    echo ""
    echo "Creating .env template..."
    cat > "$SCRIPT_DIR/.env" <<'ENV'
# Alpaca Paper Trading API (https://alpaca.markets)
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Finnhub API (https://finnhub.io — free tier, optional)
FINNHUB_API_KEY=your_finnhub_key_here
ENV
    echo "  Created .env — edit it with your API keys"
else
    echo ""
    echo ".env already exists, skipping"
fi

# --- Verify ---
echo ""
echo "Verifying installation..."
"$PYTHON" -c "
import torch, alpaca_trade_api, pandas, numpy, numba, optuna
gpu = torch.cuda.is_available()
print(f'  PyTorch {torch.__version__} (GPU: {gpu})')
print(f'  NumPy {numpy.__version__}, Pandas {pandas.__version__}')
print(f'  Numba {numba.__version__}, Optuna {optuna.__version__}')
if gpu:
    print(f'  CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your Alpaca + Finnhub API keys"
echo "  2. python connection_test.py        # Verify API access"
echo "  3. python run_pipeline.py           # Start the full system"
echo "  4. python gui.py                    # Launch dashboard (separate terminal)"
echo "========================================"
