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
#                    swappiness=15. An EXISTING /swapfile is kept as-is (never
#                    resized) — the prod box's is a pre-existing 8GB file
#                    (created 2026-01-13, before this script existed). zram steals CPU and ~50% of RAM as
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
#         sudo TRADER_PYBIN=/path/to/python bash scripts/setup_jetson_system.sh
#         (sudo strips the caller's env, so pass TRADER_PYBIN after `sudo`.)
#         bash scripts/setup_jetson_system.sh --user        # NO sudo: user unit only
#         bash scripts/setup_jetson_system.sh [--user] --print-unit   # print, touch nothing
#         (--user / --print-unit run ONLY the unit step; see "U." below.)
set -euo pipefail

SKIP_HEADLESS=0
SKIP_SWAP=0
USER_MODE=0
PRINT_UNIT=0
for arg in "$@"; do
  case "$arg" in
    --skip-headless) SKIP_HEADLESS=1 ;;
    --skip-swap)     SKIP_SWAP=1 ;;
    --user)          USER_MODE=1 ;;
    --print-unit)    PRINT_UNIT=1 ;;
  esac
done

# --- U. User-level unit / render-only modes (no root, unit step only) --------
# The prod box has no sudo, so the system unit of step 6 cannot be installed
# there (systemd 249; the memory+pids cgroup controllers are delegated to the
# user manager, so MemoryMax= is enforced in a --user unit too).
#   --print-unit          print the step-6 system unit to stdout, touch nothing
#   --user --print-unit   print the user unit to stdout, touch nothing
#   --user                install ~/.config/systemd/user/trader.service,
#                         daemon-reload + enable it (NOT start), print linger
# Single source: the values (step 0 + step 6 assignments) and the unit body
# (the step-6 heredoc) are read back from THIS file, so the system unit, the
# printed unit and the user unit cannot drift apart. The user unit differs
# only in: no User=, no ordering on system-only targets, WantedBy=default.target.
# Pinned by tests/test_setup_jetson_user_unit.py.
if [[ $USER_MODE -eq 1 || $PRINT_UNIT -eq 1 ]]; then
  SELF="${BASH_SOURCE[0]}"
  _vars="$(grep -E '^(PYBIN|UNIT_LD_PRELOAD|UNIT_LD_LIBRARY_PATH|TRADER_DIR|TRADER_USER)=' "$SELF")"
  if [[ $(printf '%s\n' "$_vars" | wc -l) -ne 5 ]]; then
    echo "[unit] FATAL: expected 5 unit variable assignments in $SELF" >&2
    exit 3
  fi
  eval "$_vars"

  _render_system_unit() {  # step-6 heredoc body, expanded exactly as step 6 does
    local tpl
    tpl="$(awk '/^UNIT$/{f=0} f{print} /^  cat > \/etc\/systemd\/system\/trader\.service <<UNIT$/{f=1}' "$SELF")"
    if [[ -z "$tpl" ]]; then
      echo "[unit] FATAL: step-6 trader.service heredoc not found in $SELF" >&2
      return 3
    fi
    eval "cat <<UNIT
${tpl}
UNIT"
  }

  _render_user_unit() {
    local txt
    txt="$(_render_system_unit)"
    printf '%s\n' "$txt" | awk '
      /^User=/ {next}
      /^After=network-online\.target/ {
        print "# User manager: it cannot order on system units (network-online.target,"
        print "# chrony.service) - those lines of the system unit are dropped here."
        next }
      /^Wants=network-online\.target$/ {next}
      /^WantedBy=multi-user\.target$/ {print "WantedBy=default.target"; next}
      {print}'
  }

  if [[ $PRINT_UNIT -eq 1 ]]; then
    if [[ $USER_MODE -eq 1 ]]; then _render_user_unit; else _render_system_unit; fi
    exit 0
  fi

  # ---- --user install ----
  if [[ $EUID -eq 0 ]]; then
    echo "[user-unit] FATAL: --user installs into the CALLING user's systemd manager." >&2
    echo "            Run it WITHOUT sudo, as the trading user." >&2
    exit 1
  fi
  if [[ -z "${XDG_RUNTIME_DIR:-}" ]] || ! systemctl --user show-environment >/dev/null 2>&1; then
    echo "[user-unit] FATAL: no systemd user manager reachable ('systemctl --user' failed;" >&2
    echo "            XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-<unset>}). Nothing was written." >&2
    echo "            Fix: run from a real login session (ssh/console - not su, sudo or cron)," >&2
    echo "            or export XDG_RUNTIME_DIR=/run/user/$(id -u) if that directory exists;" >&2
    echo "            if it does not, enable linger first: loginctl enable-linger $(id -un)" >&2
    exit 4
  fi
  # Same interpreter gate as step 0 (run verbatim from this file).
  eval "$(awk '/^# --- 0\. Interpreter/{f=1} /^# --- 1\. Headless/{f=0} f' "$SELF")"

  _txt="$(_render_user_unit)"
  if ! grep -qx 'WantedBy=default.target' <<<"$_txt" || grep -q '^User=' <<<"$_txt"; then
    echo "[user-unit] FATAL: rendered user unit failed its self-check; nothing written." >&2
    exit 3
  fi
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  UNIT_PATH="$UNIT_DIR/trader.service"
  if [[ -e "$UNIT_PATH" ]]; then
    echo "[user-unit] $UNIT_PATH already exists - left untouched (move it aside and re-run to regenerate)."
    echo "            Check it with: systemctl --user status trader"
    exit 0
  fi
  mkdir -p "$UNIT_DIR"
  _tmp="$(mktemp "$UNIT_DIR/.trader.service.XXXXXX")"
  trap 'rm -f "$_tmp"' EXIT
  printf '%s\n' "$_txt" > "$_tmp"
  chmod 644 "$_tmp"
  mv -f "$_tmp" "$UNIT_PATH"        # atomic rename: no half-written unit
  trap - EXIT
  if ! systemctl --user daemon-reload || ! systemctl --user enable trader.service; then
    systemctl --user disable trader.service >/dev/null 2>&1 || true
    rm -f "$UNIT_PATH" "$UNIT_DIR/default.target.wants/trader.service"
    systemctl --user daemon-reload >/dev/null 2>&1 || true
    echo "[user-unit] FATAL: 'systemctl --user daemon-reload/enable' failed;" >&2
    echo "            $UNIT_PATH removed again (nothing left half-installed)." >&2
    exit 5
  fi
  echo "[user-unit] Installed + ENABLED: $UNIT_PATH (NOT started)."
  echo "            Enabled = the bots START whenever your user manager starts"
  echo "            (next login after all sessions closed, or at boot with linger)."
  echo "            Linger is an owner decision (keeps your user manager alive across"
  echo "            logouts and starts it at boot, so trader runs with nobody logged in):"
  echo "              loginctl enable-linger $(id -un)"
  echo "            Start:   systemctl --user start trader"
  echo "            Stop:    systemctl --user stop trader"
  echo "            Status:  systemctl --user status trader"
  echo "            Journal: journalctl --user -u trader -f"
  echo "                     (volatile journal, e.g. this Jetson: journalctl --user-unit=trader -f)"
  echo "            Do NOT also launch run_pipeline/run_bots by hand or from the GUI."
  exit 0
fi

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash $0"
  exit 1
fi

echo "=== Trader Jetson system setup ==="

# --- 0. Interpreter for the systemd unit (checked FIRST, before any change) --
# MUST be the jetson conda env (= run_pipeline.PYTHON). `command -v python3`
# under sudo resolves to /usr/bin/python3 (secure_path has no conda), which
# lacks pyarrow/dotenv/torch: the weekly retrain then dies in
# _needs_force_harvest before the bots stop, and systemd restarts --bot-only
# forever (ops audit 2026-09-26, P0-1). Fail loudly rather than install that.
PYBIN="${TRADER_PYBIN:-/home/kyle/miniforge3/envs/jetson/bin/python}"
# Same values as run_pipeline.ENV (LD_PRELOAD: conda libstdc++ — without it
# `import torch`/pandas then `import sqlite3` dies with CXXABI_1.3.15).
UNIT_LD_PRELOAD=/home/kyle/miniforge3/envs/jetson/lib/libstdc++.so.6
UNIT_LD_LIBRARY_PATH=/home/kyle/miniforge3/envs/jetson/lib:/home/kyle/miniforge3/envs/jetson/lib/python3.10/site-packages/nvidia/cusparselt/lib
if [[ ! -x "$PYBIN" ]]; then
  echo "[python] FATAL: interpreter not found/executable: $PYBIN" >&2
  echo "         Set TRADER_PYBIN to the jetson env python and re-run." >&2
  exit 2
fi
if ! env LD_PRELOAD="$UNIT_LD_PRELOAD" LD_LIBRARY_PATH="$UNIT_LD_LIBRARY_PATH" \
     CUDA_VISIBLE_DEVICES= "$PYBIN" -c 'import pyarrow, dotenv, torch' ; then
  echo "[python] FATAL: $PYBIN cannot 'import pyarrow, dotenv, torch'." >&2
  echo "         This is not the trader (jetson) env. Set TRADER_PYBIN." >&2
  exit 2
fi
echo "[python] Unit interpreter OK: $PYBIN"

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
    _swap_gb=$(( $(stat -c %s "$SWAPFILE") / 1024 / 1024 / 1024 ))
    echo "[swap] $SWAPFILE already exists (${_swap_gb}GB) — kept as-is, NOT resized to 12GB."
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
ExecStart=${PYBIN} -u run_pipeline.py --combined-bots --bot-only
Environment=PYTHONUNBUFFERED=1
Environment=LD_PRELOAD=${UNIT_LD_PRELOAD}
Environment=LD_LIBRARY_PATH=${UNIT_LD_LIBRARY_PATH}
# NO CUDA_VISIBLE_DEVICES here: training children MUST see the GPU. The bots
# are already hidden from it by run_pipeline.BOT_ENV (run_pipeline.py:310,
# CUDA_VISIBLE_DEVICES='') and run_bots.py:45 (setdefault ''); run_pipeline
# also drops an inherited empty value for training (_training_env).
# Leading '-': a missing .env is not fatal. Makes TRADER_TELEGRAM_* /
# TRADER_HEALTHCHECK_URL visible to the PARENT (kill switch, crash alerts).
# systemd syntax: KEY=VALUE lines, no 'export' prefix.
EnvironmentFile=-${TRADER_DIR}/.env
Restart=on-failure
RestartSec=30
WatchdogSec=900
# An OOM-killed child must not stop the whole unit (systemd default
# DefaultOOMPolicy=stop): let run_pipeline's phase retry / bot restart act.
OOMPolicy=continue
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
  echo "[systemd] trader.service already exists — left untouched (move it aside and re-run to regenerate)."
fi

# --- 7. State backups -------------------------------------------------------
echo "[backup] Daily state backup: add to ${TRADER_USER}'s crontab:"
echo "  30 2 * * * /bin/bash ${TRADER_DIR}/scripts/backup_state.sh >> \$HOME/trader_backups/backup.log 2>&1"
echo "  (restic mode: export RESTIC_REPOSITORY + RESTIC_PASSWORD first)"

# --- 8. Power-mode guidance (printed, not applied) --------------------------
cat <<'EOF'

[power] Recommended usage (JetPack >= 6.2 "Super" modes). Mode IDs verified
  2026-09-26 in this Orin Nano Super's /etc/nvpmodel.conf:
  0 = 15W, 1 = 25W, 2 = MAXN_SUPER, 3 = 7W  (confirm: sudo nvpmodel -q --verbose)
  - Trading (24/7):       sudo nvpmodel -m 0     # 15W — bots are I/O-bound
  - Saturday retrain:     sudo nvpmodel -m 1     # 25W (best perf/W)
                     or:  sudo nvpmodel -m 2     # MAXN_SUPER
                          sudo jetson_clocks      # pin clocks during training
  - Check current mode:   sudo nvpmodel -q
  run_pipeline._bounded_thermal_wait already gates each search phase on
  GPU temperature (<= 70C).

[kill switch] With TRADER_TELEGRAM_BOT_TOKEN/CHAT_ID set, the pipeline
  accepts /halt /resume /flatten /status from the configured chat.
  Manual equivalent: touch trading_halt.flag in the trader directory.

Done. Reboot to apply headless/zram changes:  sudo reboot
EOF
