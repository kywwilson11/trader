#!/usr/bin/env bash
# Jetson Orin Nano 8GB system setup for the trader stack.
#
# Applies the system-level memory/performance configuration the app cannot
# do for itself. Each step is idempotent and individually skippable.
#
# What it does (and why):
#   1. headless    — disable the Ubuntu desktop (~800MB RAM freed). The GUI
#                    runs fine on another machine pointed at the same keys.
#   2. swap        — replace default zram with a 12GB NVMe swapfile,
#                    swappiness=15. zram steals CPU and ~50% of RAM as
#                    compressed swap; an NVMe swapfile is a crash-net that
#                    keeps the Saturday retrain from OOM-killing the bots.
#   3. cuda-libs   — install cuDSS + cuSPARSELt system-wide (ldconfig).
#                    PyTorch 2.8.0 Jetson wheels need libcudss.so.0 /
#                    libcusparseLt.so; per-shell LD_LIBRARY_PATH exports do
#                    NOT survive systemd/cron, which is how the import error
#                    resurfaces in production.
#   4. monitoring  — install jetson-stats (jtop) for RAM/temp telemetry.
#   5. power       — print (not set) the recommended nvpmodel usage.
#
# Usage:  sudo bash scripts/setup_jetson_system.sh [--skip-headless] [--skip-swap]
set -euo pipefail

SKIP_HEADLESS=0
SKIP_SWAP=0
for arg in "$@"; do
  case "$arg" in
    --skip-headless) SKIP_HEADLESS=1 ;;
    --skip-swap)     SKIP_SWAP=1 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash $0"
  exit 1
fi

echo "=== Trader Jetson system setup ==="

# --- 1. Headless ----------------------------------------------------------
if [[ $SKIP_HEADLESS -eq 0 ]]; then
  current=$(systemctl get-default || true)
  if [[ "$current" != "multi-user.target" ]]; then
    systemctl set-default multi-user.target
    echo "[headless] Default target -> multi-user.target (frees ~800MB)."
    echo "[headless] Takes effect on reboot. Revert: sudo systemctl set-default graphical.target"
  else
    echo "[headless] Already headless."
  fi
else
  echo "[headless] Skipped."
fi

# --- 2. Swap: disable zram, add NVMe swapfile ------------------------------
if [[ $SKIP_SWAP -eq 0 ]]; then
  if systemctl list-unit-files | grep -q nvzramconfig; then
    systemctl disable nvzramconfig 2>/dev/null || true
    echo "[swap] nvzramconfig disabled (takes effect on reboot)."
  fi
  SWAPFILE=/swapfile
  if [[ ! -f $SWAPFILE ]]; then
    echo "[swap] Creating 12GB swapfile at $SWAPFILE ..."
    fallocate -l 12G $SWAPFILE
    chmod 600 $SWAPFILE
    mkswap $SWAPFILE
    swapon $SWAPFILE
    if ! grep -q "^$SWAPFILE" /etc/fstab; then
      echo "$SWAPFILE none swap sw 0 0" >> /etc/fstab
    fi
    echo "[swap] 12GB NVMe swap active + persisted in fstab."
  else
    echo "[swap] $SWAPFILE already exists."
  fi
  # Swap as crash-net, not working set
  sysctl -w vm.swappiness=15 >/dev/null
  if ! grep -q "^vm.swappiness" /etc/sysctl.conf; then
    echo "vm.swappiness=15" >> /etc/sysctl.conf
  fi
  echo "[swap] vm.swappiness=15."
else
  echo "[swap] Skipped."
fi

# --- 3. CUDA companion libraries (cuDSS, cuSPARSELt) -----------------------
# PyTorch 2.8.0 from pypi.jetson-ai-lab.io needs these at runtime.
echo "[cuda-libs] Checking for libcudss / libcusparseLt ..."
NEED_LDCONFIG=0
if ! ldconfig -p | grep -q libcusparseLt; then
  CSLT_SRC=$(find / -name "libcusparseLt.so*" -not -path "/proc/*" 2>/dev/null | head -1 || true)
  if [[ -n "${CSLT_SRC}" ]]; then
    cp -a "$(dirname "$CSLT_SRC")"/libcusparseLt* /usr/local/cuda/lib64/ 2>/dev/null || true
    NEED_LDCONFIG=1
    echo "[cuda-libs] Copied cuSPARSELt from $CSLT_SRC to /usr/local/cuda/lib64."
  else
    echo "[cuda-libs] cuSPARSELt NOT found. Install per https://developer.nvidia.com/cusparselt-downloads"
    echo "            (aarch64-jetson, Ubuntu 22.04), then re-run this script."
  fi
else
  echo "[cuda-libs] cuSPARSELt OK."
fi
if ! ldconfig -p | grep -q libcudss; then
  CUDSS_SRC=$(find / -name "libcudss.so*" -not -path "/proc/*" 2>/dev/null | head -1 || true)
  if [[ -n "${CUDSS_SRC}" ]]; then
    cp -a "$(dirname "$CUDSS_SRC")"/libcudss* /usr/local/cuda/lib64/ 2>/dev/null || true
    NEED_LDCONFIG=1
    echo "[cuda-libs] Copied cuDSS from $CUDSS_SRC to /usr/local/cuda/lib64."
  else
    echo "[cuda-libs] cuDSS NOT found. Install cuDSS 0.7.x (aarch64-jetson) from"
    echo "            https://developer.nvidia.com/cudss-downloads, then re-run."
  fi
else
  echo "[cuda-libs] cuDSS OK."
fi
if [[ $NEED_LDCONFIG -eq 1 ]]; then
  ldconfig
  echo "[cuda-libs] ldconfig refreshed — LD_LIBRARY_PATH exports no longer needed."
fi

# --- 4. Monitoring ----------------------------------------------------------
if ! command -v jtop >/dev/null 2>&1; then
  pip3 install -U jetson-stats >/dev/null 2>&1 \
    && echo "[monitoring] jetson-stats installed (run: jtop)." \
    || echo "[monitoring] jetson-stats install failed — try: sudo pip3 install jetson-stats"
else
  echo "[monitoring] jtop already installed."
fi

# --- 5. Time sync (chrony) ---------------------------------------------------
# The Orin Nano dev kit has NO RTC battery: every cold boot starts with a
# bogus clock until NTP syncs. The bots compare bar/quote timestamps for
# staleness rejection and GTC order bookkeeping — a wrong clock silently
# breaks both. chrony with makestep corrects large offsets immediately
# instead of slewing for hours like systemd-timesyncd.
if ! command -v chronyd >/dev/null 2>&1; then
  apt-get install -y chrony >/dev/null 2>&1 \
    && echo "[chrony] installed." \
    || echo "[chrony] install failed — apt-get install chrony manually."
fi
if command -v chronyd >/dev/null 2>&1; then
  if ! grep -q '^makestep' /etc/chrony/chrony.conf 2>/dev/null; then
    echo 'makestep 1.0 -1' >> /etc/chrony/chrony.conf
    echo "[chrony] makestep enabled (always step large offsets — no RTC battery)."
  fi
  systemctl enable --now chrony >/dev/null 2>&1 || true
  systemctl disable systemd-timesyncd >/dev/null 2>&1 || true
  echo "[chrony] active. Verify: chronyc tracking"
fi

# --- 6. systemd service with watchdog --------------------------------------
# Type=notify + WatchdogSec: run_pipeline sends READY=1 at startup and
# WATCHDOG=1 every 30s from its heartbeat thread. A hung pipeline (not
# just a dead one) gets killed and restarted automatically.
TRADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRADER_USER="${SUDO_USER:-$(whoami)}"
PYBIN="$(command -v python3)"
if [[ ! -f /etc/systemd/system/trader.service ]]; then
  cat > /etc/systemd/system/trader.service <<UNIT
[Unit]
Description=Trader pipeline (bots + weekly retrain)
After=network-online.target chrony.service
Wants=network-online.target

[Service]
Type=notify
NotifyAccess=all
User=${TRADER_USER}
WorkingDirectory=${TRADER_DIR}
Environment=CUDA_VISIBLE_DEVICES=
ExecStart=${PYBIN} -u run_pipeline.py --combined-bots --bot-only
Restart=on-failure
RestartSec=30
WatchdogSec=900
# OOM: kill the pipeline before the kernel picks a victim at random
OOMScoreAdjust=200
MemoryMax=6G

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  echo "[systemd] trader.service installed (NOT enabled — review ExecStart"
  echo "          flags first, e.g. drop --bot-only to retrain on boot)."
  echo "          Enable with: sudo systemctl enable --now trader.service"
else
  echo "[systemd] trader.service already exists — left untouched."
fi

# --- 7. State backups -------------------------------------------------------
echo "[backup] Daily state backup: add to ${TRADER_USER}'s crontab:"
echo "  30 2 * * * /bin/bash ${TRADER_DIR}/scripts/backup_state.sh >> \$HOME/trader_backups/backup.log 2>&1"
echo "  (restic mode: export RESTIC_REPOSITORY + RESTIC_PASSWORD first)"

# --- 8. Power-mode guidance (printed, not applied) --------------------------
cat <<'EOF'

[power] Recommended usage (JetPack >= 6.2 "Super" modes):
  - Trading (24/7):       sudo nvpmodel -m 1     # 15W — bots are I/O-bound
  - Saturday retrain:     sudo nvpmodel -m 2     # 25W (best perf/W) or MAXN SUPER
                          sudo jetson_clocks      # pin clocks during training
  - Check current mode:   sudo nvpmodel -q
  run_pipeline's wait_for_cool_gpu already throttles on temperature.

[kill switch] With TRADER_TELEGRAM_BOT_TOKEN/CHAT_ID set, the pipeline
  accepts /halt /resume /flatten /status from the configured chat.
  Manual equivalent: touch trading_halt.flag in the trader directory.

Done. Reboot to apply headless/zram changes:  sudo reboot
EOF
