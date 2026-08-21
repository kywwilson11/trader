#!/usr/bin/env python3
"""PySide6 Trading Dashboard for Alpaca paper trading system.

Monitors positions, P&L, trade history, account balance, tax estimation,
model status, pipeline progress, and hardware health — all in one window.

Themes: Batman, Joker, Harley Quinn, Two-Face, Bubblegum Goth, Dark, Space, Money
All timestamps displayed in US/Central time.
"""

import os
import sys
import re
import html
import json
import math
import time
import shutil
import pickle
import datetime as dt
from collections import deque
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from trading_utils import get_api
from tax_lots import estimate_taxes
import notify
from notify import halt_active, set_halt, clear_halt, request_flatten
from strategy_config import (CRYPTO_POLICY, STOCK_POLICY,
                             MIN_ORDER_NOTIONAL, RISK_PCT_PER_TRADE,
                             MAX_BOOK_RISK_PCT)
from hw_monitor import get_gpu_temp as hw_get_gpu_temp

from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, Slot, QObject, QRectF, QPointF,
)
from PySide6.QtGui import QColor, QPalette, QFont, QAction, QPainter, QPixmap, QDesktopServices, QIcon, QPicture, QFontDatabase, QShortcut, QKeySequence
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPlainTextEdit, QComboBox, QCheckBox, QFrame,
    QSplitter, QGroupBox, QProgressBar, QToolBar,
    QSizePolicy, QLineEdit, QPushButton, QSpinBox,
    QScrollArea, QMessageBox, QDialog,
    QListWidget, QListWidgetItem,
)
import pyqtgraph as pg
import numpy as np
import chart_core
import design_tokens
import journal_stats

pg.setConfigOptions(antialias=True)

class NumericTableItem(QTableWidgetItem):
    """QTableWidgetItem that sorts by UserRole (numeric) when available, and
    renders in the tabular-numerals font (IBM Plex Mono) so columns of digits
    line up and don't jitter as values tick — every numeric cell, automatically."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont(QFont(design_tokens.NUMERIC_FAMILY))
    def __lt__(self, other):
        v1 = self.data(Qt.UserRole)
        v2 = other.data(Qt.UserRole) if other else None
        if v1 is not None and v2 is not None:
            return float(v1) < float(v2)
        return super().__lt__(other)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TZ_CENTRAL = ZoneInfo("America/Chicago")
# Used only for the cockpit's cheap in-RTH heartbeat check (no API call).
TZ_EASTERN = ZoneInfo("America/New_York")
HEARTBEAT_FILES = {
    "crypto": BASE_DIR / "crypto_heartbeat",
    "stock": BASE_DIR / "stock_heartbeat",
}
JOURNAL_DIR = BASE_DIR / "journals"
ACCOUNT_RISK_REGISTRY_FILE = BASE_DIR / "account_risk_registry.json"
# Custom item-data role: the article URL stashed on the news table's Time cell so
# a row click resolves the right link even after the user sorts the table.
NEWS_URL_ROLE = Qt.UserRole + 1

_JETSON_PREFIX = "/home/kyle/miniforge3/envs/jetson"
_JETSON_PY = _JETSON_PREFIX + "/bin/python"


def _engine_python():
    """Python interpreter for engine subprocesses (Jetson env if present)."""
    return _JETSON_PY if os.path.exists(_JETSON_PY) else sys.executable


def _engine_env(cusparselt=False):
    """Environment for engine subprocesses (Jetson lib paths when present)."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if os.path.exists(_JETSON_PY):
        paths = [_JETSON_PREFIX + "/lib"]
        if cusparselt:
            paths.append(
                _JETSON_PREFIX
                + "/lib/python3.10/site-packages/nvidia/cusparselt/lib")
        env["LD_LIBRARY_PATH"] = (
            ":".join(paths) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    return env

LOG_FILES = {
    "Pipeline": BASE_DIR / "pipeline_output.log",
    "Crypto Bot": BASE_DIR / "crypto_bot_output.log",
    "Stock Bot": BASE_DIR / "stock_bot_output.log",
}

# Per-file in-memory log buffer cap (chars). Trimming is newline-aligned so a
# line is never cut in half (see _trim_to_newline).
LOG_BUFFER_MAXLEN = 200_000


def _trim_to_newline(text, maxlen=LOG_BUFFER_MAXLEN):
    """Cap `text` to ~maxlen trailing chars WITHOUT cutting a line in half:
    keep the last maxlen chars, then drop the (partial) first line up to and
    including its newline so the buffer starts at a clean line boundary. A tail
    with no newline (one giant line) is returned unchanged. Fixes the mid-line
    byte-slice trim the log buffer used to do."""
    if len(text) <= maxlen:
        return text
    tail = text[-maxlen:]
    nl = tail.find("\n")
    return tail[nl + 1:] if nl != -1 else tail

CONFIG_FILES = {
    "Crypto": BASE_DIR / "config_v2.pkl",
    "Stock": BASE_DIR / "stock_config_v2.pkl",
}

MODEL_FILES = {
    "Crypto": BASE_DIR / "model_v2.pth",
    "Stock": BASE_DIR / "stock_model_v2.pth",
}

MANIFEST_FILES = {
    "Crypto": BASE_DIR / "model_v2.manifest.json",
    "Stock": BASE_DIR / "stock_model_v2.manifest.json",
}

# Shadow-mode challenger manifests (names per shadow.py challenger_manifest)
CHALLENGER_MANIFESTS = {
    "Crypto": BASE_DIR / "challenger_model_v2.manifest.json",
    "Stock": BASE_DIR / "stock_challenger_model_v2.manifest.json",
}

# Shadow-evaluation snapshots (shadow.py shadow_status_file — {prefix}naming).
SHADOW_STATUS_FILES = {
    "Crypto": BASE_DIR / "shadow_status.json",
    "Stock": BASE_DIR / "stock_shadow_status.json",
}

# PSI drift state (single file, keyed by label 'crypto'/'stock'; monitor_drift.py).
DRIFT_STATE_FILE = BASE_DIR / "drift_state.json"
# Mirror of monitor_drift.py's PSI bands (kept in sync with a comment rather
# than imported — monitor_drift pulls numpy, unavailable on the dev Mac).
DRIFT_PSI_WARN = 0.10      # monitor_drift.PSI_WARN — moderate shift (yellow)
DRIFT_PSI_ACTION = 0.25    # monitor_drift.PSI_ACTION — major shift (red)
DRIFT_CONSECUTIVE_ACTION_DAYS = 2   # monitor_drift.CONSECUTIVE_ACTION_DAYS

# Persistent beta_ledger --json artifact (written by the Beta Ledger button;
# feeds the reports-freshness strip).
BETA_REPORT_FILE = BASE_DIR / "beta_report.json"
# Meta-gate sidecars (meta_label._meta_payload / _write_refusal, {prefix} naming).
META_META_FILES = {"Crypto": BASE_DIR / "meta_meta.json",
                   "Stock": BASE_DIR / "stock_meta_meta.json"}
META_REFUSED_FILES = {"Crypto": BASE_DIR / "meta_refused.json",
                      "Stock": BASE_DIR / "stock_meta_refused.json"}
# Challenger policy-gate sidecars (backtest._write_policy_gate_sidecar,
# slot = shadow.challenger_prefix).
POLICY_GATE_FILES = {"Crypto": BASE_DIR / "challenger_policy_gate.json",
                     "Stock": BASE_DIR / "stock_challenger_policy_gate.json"}
# Reports-freshness strip: (label, path, stale_after_s); None = never age-stale.
REPORT_FRESHNESS_ITEMS = [
    ("decision_report", BASE_DIR / "decision_report.json", 7 * 86400),
    ("llm_eval", BASE_DIR / "llm_eval_report.json", 7 * 86400),
    ("llm_advisor", BASE_DIR / "llm_advisor_report.json", 7 * 86400),
    ("execution", BASE_DIR / "execution_report.json", 7 * 86400),
    ("beta", BETA_REPORT_FILE, 7 * 86400),
    ("shadow crypto", BASE_DIR / "shadow_status.json", 2 * 86400),
    ("shadow stock", BASE_DIR / "stock_shadow_status.json", 2 * 86400),
    ("policy-gate crypto", POLICY_GATE_FILES["Crypto"], 8 * 86400),
    ("policy-gate stock", POLICY_GATE_FILES["Stock"], 8 * 86400),
    ("promotion_ledger", BASE_DIR / "promotion_ledger.jsonl", None),
    ("drift_state", DRIFT_STATE_FILE, 2 * 86400),
]

STUDY_DBS = {
    "Crypto": ("v2_study.db", "v2_search"),
    "Stock": ("stock_v2_study.db", "stock_v2_search"),
}

PIPELINE_COMMAND = BASE_DIR / "pipeline_command.json"

# Unified pipeline_status.json staleness threshold (seconds). Trials can take
# up to ~10 minutes, so this must be generous enough that mid-trial gaps
# don't look dead. Shared by every staleness check (Models tab render, the
# retrain click handler, and the running/not-running probe) so they never
# disagree with each other (previously they used 600 vs 120, which could
# make the tab render "running" while the click handler said "not running").
PIPELINE_STALE_SEC = 600


def _build_crypto_symbol_set():
    """Crypto-symbol set for the trade filter, sourced from stock_config (the
    live universe) instead of a hand-maintained literal that silently drifts.
    Both spellings Alpaca may report are included (slash + slash-stripped
    uppercase). Falls back to the historical literal if the import fails so a
    broken config can never crash GUI import."""
    try:
        from stock_config import CRYPTO_SYMBOLS, CRYPTO_POOL
        out = set()
        for s in set(CRYPTO_SYMBOLS) | set(CRYPTO_POOL):
            u = str(s).upper()
            out.add(u)
            out.add(u.replace('/', ''))
        if out:
            return out
    except Exception:
        pass
    return {
        "BTCUSD", "ETHUSD", "XRPUSD", "SOLUSD", "DOGEUSD",
        "LINKUSD", "AVAXUSD", "DOTUSD", "LTCUSD", "BCHUSD",
        "BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "DOGE/USD",
        "LINK/USD", "AVAX/USD", "DOT/USD", "LTC/USD", "BCH/USD",
    }


CRYPTO_SYMBOL_SET = _build_crypto_symbol_set()


def _model_deployed_ts(name):
    """Champion deployment timestamp (epoch seconds) or None.

    Prefers the manifest's promoted_from_shadow (aware UTC) or saved_at
    (naive local) over the .pth mtime: shadow promotion copies the
    challenger artifacts with copy2 (mtime preserved), which makes a
    freshly promoted champion's .pth look weeks old.
    """
    manifest = MANIFEST_FILES.get(name)
    if manifest and manifest.exists():
        try:
            with open(manifest) as f:
                man = json.load(f)
            for key in ("promoted_from_shadow", "saved_at"):
                val = man.get(key)
                if val:
                    # naive = local time (saved_at); aware = UTC (promotion)
                    return dt.datetime.fromisoformat(val).timestamp()
        except Exception:
            pass
        try:
            return manifest.stat().st_mtime
        except OSError:
            pass
    model_path = MODEL_FILES.get(name)
    if model_path and model_path.exists():
        try:
            return model_path.stat().st_mtime
        except OSError:
            pass
    return None


# (path, mtime, size) -> best_value cache. optuna.load_study reopens the
# sqlite study and unpickles it, which the Models-tab refresh was doing for
# BOTH books every 60s on the shared 8 GB box (gui_review_2026-07 §7, 2.9).
# The study db only changes when a trial completes, so key the cached score by
# the file's stat signature and reload only when it moves.
_BEST_SCORE_CACHE = {}


def _get_best_score(name):
    """Read best Optuna score for a model (mtime-cached). Returns None on
    failure."""
    db_file, study_name = STUDY_DBS.get(name, (None, None))
    if not db_file:
        return None
    db_path = BASE_DIR / db_file
    try:
        st = db_path.stat()
    except OSError:
        return None
    sig = (str(db_path), st.st_mtime, st.st_size)
    cached = _BEST_SCORE_CACHE.get(name)
    if cached is not None and cached[0] == sig:
        return cached[1]
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.load_study(study_name=study_name,
                                  storage=f"sqlite:///{db_path}")
        value = study.best_value
    except Exception:
        return None
    _BEST_SCORE_CACHE[name] = (sig, value)
    return value

# Persistence files
NEWS_CACHE_FILE = BASE_DIR / "news_cache.json"
NEWS_CACHE_MAX_AGE_DAYS = 7
GUI_SETTINGS_FILE = BASE_DIR / "gui_settings.json"


def _load_gui_settings():
    """Load persisted GUI settings (theme, etc.)."""
    try:
        if GUI_SETTINGS_FILE.exists():
            with open(GUI_SETTINGS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_gui_settings(settings):
    """Save GUI settings to disk."""
    try:
        with open(GUI_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


# Refresh-cadence spec (Settings ops page, gui_review §8 / Phase 5.5). Values
# are stored in gui_settings under "cadences" as SECONDS; the Settings spinboxes
# clamp to these bounds; start_timers + the model timer read the effective value
# at build, and set_interval applies changes live. 'news' is displayed in
# minutes but stored in seconds like the rest.
DEFAULT_CADENCES = {  # stream -> seconds
    'account': 10, 'positions': 5, 'orders': 30, 'hw': 5,
    'news': 300, 'stocks': 30, 'models': 60,
}
CADENCE_BOUNDS = {  # stream -> (min_seconds, max_seconds)
    'account': (5, 60), 'positions': (2, 30), 'orders': (10, 120),
    'hw': (2, 30), 'news': (60, 3600), 'stocks': (10, 300), 'models': (30, 600),
}


def _cadence_seconds(stream):
    """Effective (clamped) refresh cadence in seconds for `stream`, read fresh
    from gui_settings so a fetcher thread can consult it without shared state."""
    lo, hi = CADENCE_BOUNDS[stream]
    try:
        val = int((_load_gui_settings().get('cadences', {}) or {}).get(
            stream, DEFAULT_CADENCES[stream]))
    except (TypeError, ValueError):
        return DEFAULT_CADENCES[stream]
    return max(lo, min(hi, val))


def _load_news_cache():
    """Load cached news articles (up to NEWS_CACHE_MAX_AGE_DAYS old)."""
    try:
        if NEWS_CACHE_FILE.exists():
            with open(NEWS_CACHE_FILE) as f:
                cache = json.load(f)
            cutoff = dt.datetime.now().timestamp() - NEWS_CACHE_MAX_AGE_DAYS * 86400
            articles = [a for a in cache.get('articles', [])
                        if a.get('datetime', 0) > cutoff]
            return {
                'articles': articles,
                'fng': cache.get('fng'),
                'cached_at': cache.get('cached_at', 0),
            }
    except Exception:
        pass
    return None


def _save_news_cache(articles, fng):
    """Save news articles + sentiment to disk cache."""
    try:
        # Only keep articles from last 7 days
        cutoff = dt.datetime.now().timestamp() - NEWS_CACHE_MAX_AGE_DAYS * 86400
        recent = [a for a in articles if a.get('datetime', 0) > cutoff]
        # Strip non-serializable fields, keep only what we need
        clean = []
        for a in recent:
            clean.append({
                'headline': a.get('headline', ''),
                'summary': a.get('summary', ''),
                'source': a.get('source', ''),
                'url': a.get('url', ''),
                'datetime': a.get('datetime', 0),
                '_category': a.get('_category', ''),
                '_symbol': a.get('_symbol', ''),
                '_sentiment': a.get('_sentiment', 0.0),
                '_sent_method': a.get('_sent_method', ''),
                '_scored_by_model': a.get('_scored_by_model', ''),
            })
        with open(NEWS_CACHE_FILE, 'w') as f:
            json.dump({
                'articles': clean,
                'fng': fng,
                'cached_at': dt.datetime.now().timestamp(),
            }, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Theme System
# ---------------------------------------------------------------------------
THEMES = {
    # --- Batman characters ---
    "Batman": {
        "green":     QColor(76, 175, 80),
        "red":       QColor(244, 67, 54),
        "yellow":    QColor(255, 215, 0),
        "white":     QColor(224, 224, 224),
        "muted":     QColor(136, 136, 136),
        "bg_dark":   QColor(10, 10, 10),
        "bg_card":   QColor(26, 26, 26),
        "bg_table":  QColor(20, 20, 20),
        "accent":    QColor(255, 215, 0),
        "bg_header": QColor(34, 34, 34),
        "bg_border": QColor(58, 58, 58),
        "bg_hover":  QColor(42, 42, 42),
        "bg_log":    QColor(5, 5, 5),
    },
    "Joker": {
        "green":     QColor(0, 255, 102),
        "red":       QColor(255, 34, 68),
        "yellow":    QColor(170, 255, 170),
        "white":     QColor(220, 210, 240),
        "muted":     QColor(140, 120, 180),
        "bg_dark":   QColor(13, 10, 24),
        "bg_card":   QColor(31, 16, 53),
        "bg_table":  QColor(22, 12, 38),
        "accent":    QColor(0, 255, 102),
        "bg_header": QColor(40, 20, 66),
        "bg_border": QColor(70, 40, 110),
        "bg_hover":  QColor(50, 28, 82),
        "bg_log":    QColor(8, 5, 16),
    },
    "Harley Quinn": {
        "green":     QColor(0, 200, 150),
        "red":       QColor(255, 23, 68),
        "yellow":    QColor(255, 224, 232),
        "white":     QColor(240, 230, 235),
        "muted":     QColor(160, 120, 140),
        "bg_dark":   QColor(10, 10, 10),
        "bg_card":   QColor(42, 16, 24),
        "bg_table":  QColor(30, 12, 18),
        "accent":    QColor(255, 23, 68),
        "bg_header": QColor(50, 20, 30),
        "bg_border": QColor(90, 40, 55),
        "bg_hover":  QColor(65, 28, 40),
        "bg_log":    QColor(5, 5, 5),
    },
    "Two-Face": {
        "green":     QColor(0, 200, 160),
        "red":       QColor(220, 50, 50),
        "yellow":    QColor(100, 220, 200),
        "white":     QColor(200, 220, 220),
        "muted":     QColor(80, 130, 130),
        "bg_dark":   QColor(8, 14, 16),
        "bg_card":   QColor(16, 30, 34),
        "bg_table":  QColor(12, 24, 28),
        "accent":    QColor(0, 190, 180),
        "bg_header": QColor(20, 38, 42),
        "bg_border": QColor(40, 75, 80),
        "bg_hover":  QColor(26, 48, 54),
        "bg_log":    QColor(4, 10, 12),
    },
    # --- Hacker noir ---
    "Salander": {
        # Lisbeth Salander — matte cold-black, toxic terminal-green signature,
        # blood red for danger, steel-cyan secondary. The Dragon Tattoo aesthetic.
        "green":     QColor(0, 230, 118),   # profit
        "red":       QColor(255, 40, 60),   # blood red / danger
        "yellow":    QColor(92, 214, 205),  # cold steel-cyan highlight
        "white":     QColor(222, 230, 224), # cold green-tinted off-white
        "muted":     QColor(104, 126, 114), # green-grey
        "bg_dark":   QColor(7, 9, 8),
        "bg_card":   QColor(15, 19, 17),
        "bg_table":  QColor(11, 14, 12),
        "accent":    QColor(0, 245, 130),   # toxic terminal green
        "bg_header": QColor(19, 25, 21),
        "bg_border": QColor(36, 52, 43),
        "bg_hover":  QColor(24, 34, 28),
        "bg_log":    QColor(3, 5, 4),
    },
    # --- Other themes ---
    "Black Metal": {
        "green":     QColor(180, 180, 180),
        "red":       QColor(180, 30, 30),
        "yellow":    QColor(160, 160, 160),
        "white":     QColor(190, 190, 190),
        "muted":     QColor(90, 90, 90),
        "bg_dark":   QColor(5, 5, 5),
        "bg_card":   QColor(14, 14, 14),
        "bg_table":  QColor(10, 10, 10),
        "accent":    QColor(160, 160, 160),
        "bg_header": QColor(18, 18, 18),
        "bg_border": QColor(40, 40, 40),
        "bg_hover":  QColor(24, 24, 24),
        "bg_log":    QColor(2, 2, 2),
    },
    "Bubblegum Goth": {
        "green":     QColor(0, 230, 118),
        "red":       QColor(255, 56, 96),
        "yellow":    QColor(255, 170, 230),
        "white":     QColor(240, 210, 245),
        "muted":     QColor(170, 130, 190),
        "bg_dark":   QColor(18, 10, 26),
        "bg_card":   QColor(35, 20, 50),
        "bg_table":  QColor(28, 16, 40),
        "accent":    QColor(255, 105, 180),
        "bg_header": QColor(45, 25, 60),
        "bg_border": QColor(80, 40, 100),
        "bg_hover":  QColor(55, 30, 75),
        "bg_log":    QColor(12, 6, 18),
    },
    "Dark": {
        "green":     QColor(0, 200, 83),
        "red":       QColor(255, 68, 68),
        "yellow":    QColor(255, 193, 7),
        "white":     QColor(220, 220, 220),
        "muted":     QColor(160, 160, 160),
        "bg_dark":   QColor(43, 43, 43),
        "bg_card":   QColor(55, 55, 55),
        "bg_table":  QColor(50, 50, 50),
        "accent":    QColor(100, 181, 246),
        "bg_header": QColor(58, 58, 58),
        "bg_border": QColor(85, 85, 85),
        "bg_hover":  QColor(69, 69, 69),
        "bg_log":    QColor(30, 30, 30),
    },
    "Space": {
        "green":     QColor(0, 230, 118),
        "red":       QColor(255, 82, 82),
        "yellow":    QColor(255, 171, 64),
        "white":     QColor(210, 225, 255),
        "muted":     QColor(110, 130, 170),
        "bg_dark":   QColor(8, 12, 21),
        "bg_card":   QColor(16, 24, 42),
        "bg_table":  QColor(12, 18, 32),
        "accent":    QColor(0, 229, 255),
        "bg_header": QColor(22, 32, 56),
        "bg_border": QColor(36, 52, 86),
        "bg_hover":  QColor(26, 40, 68),
        "bg_log":    QColor(4, 8, 16),
    },
    "Money": {
        "green":     QColor(0, 230, 118),
        "red":       QColor(255, 107, 107),
        "yellow":    QColor(255, 215, 0),
        "white":     QColor(212, 232, 212),
        "muted":     QColor(122, 154, 122),
        "bg_dark":   QColor(10, 18, 10),
        "bg_card":   QColor(20, 32, 20),
        "bg_table":  QColor(15, 26, 15),
        "accent":    QColor(255, 215, 0),
        "bg_header": QColor(26, 46, 26),
        "bg_border": QColor(42, 74, 42),
        "bg_hover":  QColor(34, 54, 34),
        "bg_log":    QColor(6, 12, 6),
    },
    # --- Restrained professional pair (Phase 4.6) ---
    "Terminal": {
        # Koyfin-like discipline: elevated charcoal grays, one desaturated
        # steel-cyan accent, muted low-chroma up/down. No theatrics.
        "green":     QColor(74, 158, 109),   # muted profit green
        "red":       QColor(192, 91, 91),    # muted loss red
        "yellow":    QColor(217, 164, 65),   # muted amber (warn)
        "white":     QColor(212, 218, 224),  # cool light gray text
        "muted":     QColor(122, 132, 143),  # steel gray secondary text
        "bg_dark":   QColor(15, 18, 22),     # charcoal base
        "bg_card":   QColor(23, 27, 33),     # raised surface
        "bg_table":  QColor(18, 22, 27),     # recessed well
        "accent":    QColor(86, 163, 192),   # desaturated steel-cyan
        "bg_header": QColor(28, 33, 41),
        "bg_border": QColor(42, 49, 59),
        "bg_hover":  QColor(34, 40, 49),
        "bg_log":    QColor(11, 14, 17),
    },
    "Paper": {
        # Light theme: warm off-white ladder, near-black text, one ink-blue
        # accent, darker up/down shades that read on a white background.
        "green":     QColor(31, 138, 76),    # dark green for white-bg contrast
        "red":       QColor(192, 57, 43),    # dark red for white-bg contrast
        "yellow":    QColor(184, 134, 11),   # dark goldenrod (warn)
        "white":     QColor(26, 28, 32),     # near-black primary text
        "muted":     QColor(95, 102, 112),   # mid-gray secondary text
        "bg_dark":   QColor(236, 234, 227),  # warm off-white base
        "bg_card":   QColor(249, 247, 241),  # near-white raised surface
        "bg_table":  QColor(228, 225, 216),  # recessed well
        "accent":    QColor(47, 92, 158),    # ink blue
        "bg_header": QColor(221, 217, 206),
        "bg_border": QColor(201, 196, 182),
        "bg_hover":  QColor(232, 229, 219),
        "bg_log":    QColor(240, 238, 231),
    },
}

# Active theme — module-level so helpers can reference it
T = THEMES["Batman"]


def set_theme(name):
    """Switch the active theme colors + refresh the shared pnl palette."""
    global T, PAL
    T = THEMES[name]
    PAL = _chart_palette()


def _chart_palette():
    """Theme -> chart-color derivation. The ONLY place this happens."""
    return chart_core.derive_chart_palette({k: v.name() for k, v in T.items()})


# ONE contrast-adjusted up/down palette shared by charts AND widgets (pnl_color,
# BUY/SELL buttons) so a P&L cell and a candle render the identical green/red.
# Seeded here (now that _chart_palette exists); set_theme refreshes it on switch.
PAL = _chart_palette()


def _on_color(bg_hex):
    """Black or white, whichever reads better on `bg_hex` — the same contrast
    -pole idiom chart_core.heatmap_style uses for tile text. For BUY/SELL/Close
    button labels sitting on a pnl-colored fill."""
    return ('#000000'
            if chart_core.contrast_ratio('#000000', bg_hex)
            >= chart_core.contrast_ratio('#ffffff', bg_hex) else '#ffffff')


class ChartCrosshair:
    """Reusable time/value crosshair for a pyqtgraph PlotItem — display only,
    never touches files or triggers a fetch. Snaps to the nearest data point
    via chart_core.nearest_index so the readout always matches a real bar."""

    def __init__(self, plot_item, pal, y_fmt=lambda v: f'${v:,.2f}'):
        self._plot_item = plot_item
        self._y_fmt = y_fmt
        self._t = np.array([], dtype=float)
        self._y = np.array([], dtype=float)
        # Rich OHLC readout (price chart only; enabled via set_ohlc). Perf/line
        # crosshairs never call set_ohlc, so _rich stays False and they keep the
        # existing single date+value readout.
        self._rich = False
        self._o = self._h = self._l = self._c = self._v = None
        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen(pal['crosshair'], style=Qt.DotLine))
        self._hline = pg.InfiniteLine(angle=0, movable=False,
                                       pen=pg.mkPen(pal['crosshair'], style=Qt.DotLine))
        plot_item.addItem(self._vline, ignoreBounds=True)
        plot_item.addItem(self._hline, ignoreBounds=True)
        self._text = pg.TextItem(anchor=(0, 1), color=pal['fg'])
        plot_item.addItem(self._text, ignoreBounds=True)
        # Top-left OHLC legend + a bg-filled price tag pinned to the right edge
        # at the cursor's y. Both live in data coords but are repositioned to the
        # viewbox corners on every move so they stay put under pan/zoom.
        self._legend = pg.TextItem(anchor=(0, 0), color=pal['fg'],
                                    fill=pg.mkBrush(0, 0, 0, 150))
        self._legend.setZValue(100)
        plot_item.addItem(self._legend, ignoreBounds=True)
        self._price_tag = pg.TextItem(anchor=(1, 0.5), color=pal['bg'],
                                      fill=pg.mkBrush(pal['crosshair']))
        self._price_tag.setZValue(100)
        plot_item.addItem(self._price_tag, ignoreBounds=True)
        self._vline.hide()
        self._hline.hide()
        self._text.hide()
        self._legend.hide()
        self._price_tag.hide()
        self._proxy = pg.SignalProxy(plot_item.scene().sigMouseMoved, rateLimit=30,
                                      slot=self._moved)

    def set_series(self, t, y):
        self._t = np.asarray(t, dtype=float)
        self._y = np.asarray(y, dtype=float)

    def set_ohlc(self, o, h, l, c, v):
        """Enable the rich O/H/L/C/%chg/volume readout. Any of o/h/l/v may be
        None (line mode shows just C + %chg). Arrays must be index-aligned with
        the series passed to set_series."""
        def _arr(a):
            if a is None:
                return None
            arr = np.asarray(a, dtype=float)
            return arr if len(arr) else None
        self._o, self._h, self._l = _arr(o), _arr(h), _arr(l)
        self._c, self._v = _arr(c), _arr(v)
        self._rich = self._c is not None

    def clear_ohlc(self):
        self._rich = False
        self._o = self._h = self._l = self._c = self._v = None
        self._legend.hide()
        self._price_tag.hide()

    def set_palette(self, pal):
        self._vline.setPen(pg.mkPen(pal['crosshair'], style=Qt.DotLine))
        self._hline.setPen(pg.mkPen(pal['crosshair'], style=Qt.DotLine))
        self._text.setColor(pal['fg'])
        self._legend.setColor(pal['fg'])
        self._legend.fill = pg.mkBrush(0, 0, 0, 150)
        self._legend.update()
        self._price_tag.setColor(pal['bg'])
        self._price_tag.fill = pg.mkBrush(pal['crosshair'])
        self._price_tag.update()

    def _hide_all(self):
        self._vline.hide()
        self._hline.hide()
        self._text.hide()
        self._legend.hide()
        self._price_tag.hide()

    def _moved(self, evt):
        pos = evt[0]
        if not self._plot_item.sceneBoundingRect().contains(pos) or len(self._t) == 0:
            self._hide_all()
            return
        vb = self._plot_item.getViewBox()
        p = vb.mapSceneToView(pos)
        i = chart_core.nearest_index(self._t, p.x())
        t_val, y_val = self._t[i], self._y[i]
        self._vline.setPos(t_val)
        self._hline.setPos(y_val)
        ts_str = dt.datetime.fromtimestamp(t_val).strftime('%m-%d %H:%M')
        if self._rich:
            # Top-left legend: O H L C (+chg%) V, whichever series are present.
            def _fnum(a):
                return self._y_fmt(a[i]) if (a is not None and i < len(a)) else None
            parts = [ts_str]
            o_, h_, l_, c_ = _fnum(self._o), _fnum(self._h), _fnum(self._l), _fnum(self._c)
            if o_ and h_ and l_:
                parts += [f"O {o_}", f"H {h_}", f"L {l_}"]
            if c_:
                chg = ''
                if self._c is not None and i > 0 and self._c[i - 1]:
                    pc = (self._c[i] - self._c[i - 1]) / self._c[i - 1] * 100.0
                    chg = f" ({pc:+.2f}%)"
                parts.append(f"C {c_}{chg}")
            if self._v is not None and i < len(self._v):
                parts.append(f"V {chart_core.format_si(self._v[i])}")
            self._legend.setText('  '.join(parts))
            (xmin, xmax), (ymin, ymax) = vb.viewRange()
            self._legend.setPos(xmin, ymax)
            self._legend.show()
            # Price tag hugging the right axis at cursor height.
            self._price_tag.setText(self._y_fmt(y_val))
            self._price_tag.setPos(xmax, y_val)
            self._price_tag.show()
            self._text.hide()
        else:
            self._text.setText(ts_str + '  ' + self._y_fmt(y_val))
            self._text.setPos(t_val, y_val)
            self._text.show()
            self._legend.hide()
            self._price_tag.hide()
        self._vline.show()
        self._hline.show()


class CandlestickItem(pg.GraphicsObject):
    """OHLC candlestick series drawn into a cached QPicture — repaint cost is
    zero on pan/theme-repaint (Jetson-friendly). up/down colors passed in by
    the caller so it never touches the theme system directly."""

    def __init__(self):
        super().__init__()
        self._picture = QPicture()
        self._rect = QRectF()

    def set_data(self, t, o, h, l, c, w, up_mask, up_color, down_color,
                 bg_color=None):
        t = np.asarray(t, dtype=float)
        if len(t) == 0:
            self._picture = QPicture()
            self._rect = QRectF()
            self.prepareGeometryChange()
            self.update()
            return
        o = np.asarray(o, dtype=float)
        h = np.asarray(h, dtype=float)
        l = np.asarray(l, dtype=float)
        c = np.asarray(c, dtype=float)
        w = np.asarray(w, dtype=float)
        up_mask = np.asarray(up_mask, dtype=bool)

        self._picture = QPicture()
        painter = QPainter(self._picture)
        try:
            # Direction is never encoded by hue alone (palette validator:
            # green/red collapses under deuteranopia in some themes):
            # up-candles are HOLLOW (surface fill), down-candles FILLED.
            up_fill = bg_color if bg_color is not None else up_color
            for mask, color, fill in ((up_mask, up_color, up_fill),
                                      (~up_mask, down_color, down_color)):
                pen = pg.mkPen(color, width=1)
                brush = pg.mkBrush(fill)
                painter.setPen(pen)
                painter.setBrush(brush)
                for i in np.nonzero(mask)[0]:
                    painter.drawLine(QPointF(t[i], l[i]), QPointF(t[i], h[i]))
                    painter.drawRect(QRectF(t[i] - w[i] / 2, o[i], w[i], c[i] - o[i]))
        finally:
            painter.end()

        wmax = float(np.max(w)) if len(w) else 0.0
        self._rect = QRectF(
            float(t.min()) - wmax, float(l.min()),
            float(t.max() - t.min()) + 2 * wmax,
            float(h.max() - l.min()) or 1e-9,
        )
        self.prepareGeometryChange()
        self.update()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self._picture)

    def boundingRect(self):
        return self._rect


class SIAxisItem(pg.AxisItem):
    """AxisItem whose tick labels are SI-abbreviated (1.2K / 3.4M) via
    chart_core.format_si — used for the volume pane's left axis, where raw
    share/coin counts are unreadable."""

    def tickStrings(self, values, scale, spacing):
        return [chart_core.format_si(v) for v in values]


_THEME_SVGS = {
    "Batman": """\
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#141420"/>
      <stop offset="100%" stop-color="#08080c"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="97" fill="url(#bg)" stroke="#ffd700" stroke-width="3"/>
  <ellipse cx="100" cy="102" rx="68" ry="44" fill="#ffd700"/>
  <path d="
    M 100,66
    L 94,54  L 97,68
    L 38,62
    L 46,84  L 54,74
    L 60,90  L 70,78
    L 78,96
    L 100,126
    L 122,96
    L 130,78  L 140,90
    L 146,74  L 154,84
    L 162,62
    L 103,68  L 106,54
    Z" fill="#0a0a0a"/>
</svg>""",

    "Joker": """\
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#301055"/>
      <stop offset="100%" stop-color="#180830"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="97" fill="url(#bg)" stroke="#00ff66" stroke-width="3"/>
  <path d="
    M 45,105 Q 42,50 70,35 Q 85,28 100,32 Q 115,28 130,35 Q 158,50 155,105
    Q 140,70 120,62 Q 108,58 100,62 Q 92,58 80,62 Q 60,70 45,105 Z" fill="#00bb44"/>
  <ellipse cx="100" cy="118" rx="42" ry="52" fill="#e8e4e0"/>
  <circle cx="82" cy="105" r="6" fill="#1a1a1a"/>
  <circle cx="118" cy="105" r="6" fill="#1a1a1a"/>
  <path d="M 68,135 Q 84,158 100,155 Q 116,158 132,135"
        fill="none" stroke="#dd1133" stroke-width="5" stroke-linecap="round"/>
</svg>""",

    "Harley Quinn": """\
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#1a0a10"/>
      <stop offset="100%" stop-color="#0a0408"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="97" fill="url(#bg)" stroke="#ff1744" stroke-width="3"/>
  <path d="M 50,100 L 78,58 L 106,100 L 78,142 Z" fill="#dd1133"/>
  <path d="M 94,100 L 122,58 L 150,100 L 122,142 Z" fill="#151515" stroke="#333"
        stroke-width="1.5"/>
</svg>""",

    "Bubblegum Goth": """\
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#1a0c22"/>
      <stop offset="100%" stop-color="#0a0510"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="97" fill="url(#bg)" stroke="#ff69b4" stroke-width="3"/>
  <ellipse cx="100" cy="92" rx="44" ry="40" fill="#ff7eb3"/>
  <rect x="68" y="122" width="64" height="28" rx="14" fill="#ff7eb3"/>
  <ellipse cx="82" cy="90" rx="14" ry="15" fill="#1a0c22"/>
  <ellipse cx="118" cy="90" rx="14" ry="15" fill="#1a0c22"/>
  <ellipse cx="100" cy="112" rx="5" ry="6" fill="#1a0c22"/>
  <path d="M 80,142 Q 90,138 100,140 Q 110,138 120,142"
        fill="none" stroke="#1a0c22" stroke-width="3" stroke-linecap="round"/>
  <path d="M 78,58 Q 62,40 68,52 Q 72,60 78,58 Z" fill="#ff1493"/>
  <path d="M 78,58 Q 92,42 88,54 Q 84,60 78,58 Z" fill="#ff1493"/>
  <circle cx="78" cy="58" r="3.5" fill="#ff69b4"/>
</svg>""",

    "Dark": """\
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#2a2a35"/>
      <stop offset="100%" stop-color="#141418"/>
    </radialGradient>
    <radialGradient id="moon" cx="35%" cy="35%" r="50%">
      <stop offset="0%" stop-color="#fffff0"/>
      <stop offset="60%" stop-color="#f0e8d0"/>
      <stop offset="100%" stop-color="#ddd0b0"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="97" fill="url(#bg)" stroke="#64b5f6" stroke-width="3"/>
  <circle cx="35" cy="42" r="2" fill="white" opacity="0.9"/>
  <circle cx="155" cy="35" r="1.5" fill="white" opacity="0.7"/>
  <circle cx="165" cy="85" r="2" fill="white" opacity="0.8"/>
  <circle cx="40" cy="145" r="1.5" fill="white" opacity="0.6"/>
  <circle cx="160" cy="150" r="1.8" fill="white" opacity="0.7"/>
  <circle cx="50" cy="70" r="1" fill="white" opacity="0.5"/>
  <circle cx="145" cy="170" r="1.2" fill="white" opacity="0.5"/>
  <circle cx="88" cy="95" r="50" fill="url(#moon)"/>
  <circle cx="112" cy="82" r="42" fill="url(#bg)"/>
  <circle cx="62" cy="85" r="7" fill="#d8ccaa" opacity="0.4"/>
  <circle cx="72" cy="115" r="5" fill="#d8ccaa" opacity="0.35"/>
  <circle cx="55" cy="105" r="4" fill="#d8ccaa" opacity="0.3"/>
  <circle cx="80" cy="130" r="3.5" fill="#d8ccaa" opacity="0.3"/>
  <circle cx="88" cy="95" r="55" fill="none" stroke="#fffff0" stroke-width="2"
          opacity="0.1"/>
</svg>""",

    "Space": """\
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#0c1428"/>
      <stop offset="100%" stop-color="#040810"/>
    </radialGradient>
    <radialGradient id="planet" cx="40%" cy="35%" r="55%">
      <stop offset="0%" stop-color="#e8c870"/>
      <stop offset="40%" stop-color="#c8a050"/>
      <stop offset="100%" stop-color="#886830"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="97" fill="url(#bg)" stroke="#00e5ff" stroke-width="3"/>
  <circle cx="25" cy="35" r="2" fill="white" opacity="0.9"/>
  <circle cx="170" cy="28" r="1.5" fill="white" opacity="0.7"/>
  <circle cx="172" cy="78" r="2" fill="white" opacity="0.8"/>
  <circle cx="30" cy="155" r="1.5" fill="white" opacity="0.6"/>
  <circle cx="55" cy="55" r="1" fill="white" opacity="0.5"/>
  <circle cx="160" cy="158" r="1.5" fill="white" opacity="0.5"/>
  <ellipse cx="100" cy="105" rx="75" ry="18" fill="none" stroke="#c8a050"
           stroke-width="10" opacity="0.4"/>
  <circle cx="100" cy="98" r="38" fill="url(#planet)"/>
  <path d="M 63,90 Q 100,86 137,90" fill="none" stroke="#d4b060" stroke-width="2"
        opacity="0.4"/>
  <path d="M 62,100 Q 100,96 138,100" fill="none" stroke="#b89040" stroke-width="3"
        opacity="0.3"/>
  <path d="M 65,108 Q 100,112 135,108" fill="none" stroke="#d4b060" stroke-width="2"
        opacity="0.3"/>
  <path d="M 25,105 Q 55,126 100,128 Q 145,126 175,105"
        fill="none" stroke="#c8a050" stroke-width="9" stroke-linecap="round" opacity="0.8"/>
  <path d="M 28,105 Q 58,122 100,124 Q 142,122 172,105"
        fill="none" stroke="#040810" stroke-width="2" opacity="0.5"/>
</svg>""",

    "Money": """\
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#0f1e0f"/>
      <stop offset="100%" stop-color="#060c06"/>
    </radialGradient>
    <radialGradient id="coin" cx="42%" cy="38%" r="55%">
      <stop offset="0%" stop-color="#ffe680"/>
      <stop offset="40%" stop-color="#ffd700"/>
      <stop offset="100%" stop-color="#aa8800"/>
    </radialGradient>
    <radialGradient id="shine" cx="35%" cy="30%" r="40%">
      <stop offset="0%" stop-color="white" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="white" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="97" fill="url(#bg)" stroke="#ffd700" stroke-width="3"/>
  <circle cx="100" cy="100" r="58" fill="url(#coin)" stroke="#c8a800" stroke-width="3"/>
  <circle cx="100" cy="100" r="50" fill="none" stroke="#b89900" stroke-width="2"/>
  <circle cx="100" cy="100" r="56" fill="none" stroke="#ddbb22" stroke-width="1"
          stroke-dasharray="5,3"/>
  <text x="100" y="125" text-anchor="middle" font-family="serif" font-size="80"
        font-weight="bold" fill="#8a6e00">$</text>
  <circle cx="100" cy="100" r="55" fill="url(#shine)"/>
</svg>""",

    "Salander": """\
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="50%" cy="44%" r="58%">
      <stop offset="0%" stop-color="#0c130f"/>
      <stop offset="100%" stop-color="#020403"/>
    </radialGradient>
    <linearGradient id="scale" x1="20%" y1="0%" x2="80%" y2="100%">
      <stop offset="0%" stop-color="#5cffa8"/>
      <stop offset="55%" stop-color="#00f582"/>
      <stop offset="100%" stop-color="#00945a"/>
    </linearGradient>
  </defs>
  <circle cx="100" cy="100" r="97" fill="url(#bg)" stroke="#00f582" stroke-width="3"/>
  <!-- coiled tribal dragon body (S-curve) -->
  <path d="M 70,158
           C 56,128 86,120 86,98
           C 86,78 58,76 64,54
           C 69,36 98,34 112,46
           C 100,42 82,48 84,64
           C 86,82 114,84 110,108
           C 107,132 80,132 82,156 Z"
        fill="url(#scale)"/>
  <!-- horned head -->
  <path d="M 112,46
           C 128,37 150,45 150,61
           C 150,70 142,75 133,72
           L 141,82 128,77 131,90 119,77
           C 110,73 104,65 108,55 Z"
        fill="#00f582"/>
  <!-- swept-back horn -->
  <path d="M 138,50 C 150,40 162,42 160,52 C 154,49 146,50 142,57 Z" fill="#00945a"/>
  <!-- eye -->
  <circle cx="130" cy="59" r="3.2" fill="#020403"/>
  <!-- dorsal spikes -->
  <path d="M 74,98 l -12,-3 9,11 z" fill="#00b86a"/>
  <path d="M 96,72 l 11,-7 -3,13 z" fill="#00b86a"/>
  <path d="M 96,128 l -12,4 11,8 z" fill="#00b86a"/>
</svg>""",

    "Terminal": """\
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#171b21"/>
      <stop offset="100%" stop-color="#0b0e11"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="97" fill="url(#bg)" stroke="#56a3c0" stroke-width="3"/>
  <rect x="48" y="52" width="104" height="96" rx="10" fill="none" stroke="#2a313b" stroke-width="3"/>
  <path d="M 66,88 L 86,104 L 66,120" fill="none" stroke="#56a3c0" stroke-width="7"
        stroke-linecap="round" stroke-linejoin="round"/>
  <line x1="96" y1="122" x2="130" y2="122" stroke="#4a9e6d" stroke-width="7" stroke-linecap="round"/>
</svg>""",

    "Paper": """\
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#f9f7f1"/>
      <stop offset="100%" stop-color="#e4e1d8"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="97" fill="url(#bg)" stroke="#2f5c9e" stroke-width="3"/>
  <polyline points="52,132 84,104 108,120 148,68" fill="none" stroke="#2f5c9e"
            stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="148" cy="68" r="7" fill="#1f8a4c"/>
</svg>""",
}


_THEME_IMAGES = {
    "Salander": BASE_DIR / "logos" / "salander.png",
    "Batman": BASE_DIR / "logos" / "batman.png",
    "Joker": BASE_DIR / "logos" / "joker.png",
    "Harley Quinn": BASE_DIR / "logos" / "harley_quinn.png",
    "Two-Face": BASE_DIR / "logos" / "two_face.png",
    "Black Metal": BASE_DIR / "logos" / "black_metal.png",
    "Bubblegum Goth": BASE_DIR / "logos" / "bubblegum_goth.png",
    "Dark": BASE_DIR / "logos" / "night.png",
    "Space": BASE_DIR / "logos" / "space.png",
    "Money": BASE_DIR / "logos" / "money.png",
}


# (theme_name, size) -> QPixmap. 10 themes x one size = trivial; never cleared.
_LOGO_CACHE = {}


def generate_theme_logo(theme_name, size=80):
    """Logo icon (QPixmap) for a theme, memoized by (theme_name, size).

    Prefers the pre-scaled logos/96/<file> asset (~20 KB) over the full-res
    logos/<file> (up to ~9 MB) so the 8 GB Jetson never decodes a multi-megabyte
    PNG just to draw an 80 px icon; falls back to the full-res original, then to
    SVG rendering, then to a transparent placeholder. Themes with no art at all
    (Terminal/Paper fall through to their SVGs; anything unknown to a blank
    pixmap) never raise KeyError.
    """
    cache_key = (theme_name, size)
    cached = _LOGO_CACHE.get(cache_key)
    if cached is not None:
        return cached

    result = None
    img_path = _THEME_IMAGES.get(theme_name)
    if img_path:
        # @96 pre-scaled first, then the full-res original as a fallback.
        for path in (img_path.parent / "96" / img_path.name, img_path):
            if path.exists():
                pix = QPixmap(str(path))
                if not pix.isNull():
                    result = pix.scaled(size, size, Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation)
                    break

    if result is None:
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtCore import QByteArray
        result = QPixmap(size, size)
        result.fill(QColor(0, 0, 0, 0))
        svg_data = _THEME_SVGS.get(theme_name)
        if svg_data:
            renderer = QSvgRenderer(QByteArray(svg_data.encode("utf-8")))
            painter = QPainter(result)
            renderer.render(painter)
            painter.end()

    _LOGO_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_money(val):
    """Format a numeric value as $X,XXX.XX."""
    try:
        v = float(val)
        sign = "-" if v < 0 else ""
        return f"{sign}${abs(v):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def fmt_pct(val):
    """Format as percentage."""
    try:
        return f"{float(val):+.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


def fmt_qty(val):
    """Format a share/coin quantity without scientific notation or trailing
    zeros (crypto qtys can be tiny fractions; stocks are whole shares)."""
    try:
        s = f"{float(val):,.8f}".rstrip("0").rstrip(".")
        return s or "0"
    except (TypeError, ValueError):
        return str(val)


def fmt_time(ts_str):
    """Convert an ISO timestamp string to Central Time display."""
    if not ts_str:
        return ""
    try:
        parsed = dt.datetime.fromisoformat(str(ts_str))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        central = parsed.astimezone(TZ_CENTRAL)
        return central.strftime("%m/%d %I:%M:%S %p")
    except Exception:
        return str(ts_str)[:19] if len(str(ts_str)) >= 19 else str(ts_str)


def fmt_time_short(ts_str):
    """Convert an ISO timestamp to short Central Time (no seconds)."""
    if not ts_str:
        return ""
    try:
        parsed = dt.datetime.fromisoformat(str(ts_str))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        central = parsed.astimezone(TZ_CENTRAL)
        return central.strftime("%m/%d %I:%M %p")
    except Exception:
        return str(ts_str)[:16] if len(str(ts_str)) >= 16 else str(ts_str)


def pnl_color(val):
    """Return the sign-colored QColor for a P&L value. Uses the SAME
    contrast-adjusted up/down as the charts (PAL, via derive_chart_palette),
    so tables and candles never show two different greens for one semantic."""
    try:
        v = float(val)
        if v > 0:
            return QColor(PAL["up"])
        elif v < 0:
            return QColor(PAL["down"])
    except (TypeError, ValueError):
        pass
    return T["white"]


def make_card(title, value="\u2014", parent=None):
    """Create a styled info card widget. The frame, title and value are styled
    entirely by the global QSS attribute selectors (QFrame[card="true"],
    QLabel[card_title="true"], QLabel#card_value) set in apply_theme \u2014 so a new
    card is themed automatically, no per-widget restyle loop needed. The value
    carries numeric="true" so its digits render in the tabular numerals font."""
    frame = QFrame(parent)
    frame.setProperty("card", True)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(design_tokens.SPACE["s3"], design_tokens.SPACE["s2"],
                              design_tokens.SPACE["s3"], design_tokens.SPACE["s2"])

    lbl_title = QLabel(title.upper())
    lbl_title.setProperty("card_title", True)
    lbl_title.setAlignment(Qt.AlignLeft)

    lbl_value = QLabel(str(value))
    lbl_value.setAlignment(Qt.AlignLeft)
    lbl_value.setObjectName("card_value")
    lbl_value.setProperty("numeric", True)

    layout.addWidget(lbl_title)
    layout.addWidget(lbl_value)
    return frame


# (path, mtime, size) -> unpickled config cache — same mtime-keyed treatment as
# _get_best_score (gui_review_2026-07 §7 2.9): the Models-tab refresh unpickled
# both config_v2.pkl files every 60s though they only change on a retrain.
_CONFIG_CACHE = {}


def read_config(path):
    """Safely read a pickle config file (mtime-cached)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    sig = (str(path), st.st_mtime, st.st_size)
    cached = _CONFIG_CACHE.get(str(path))
    if cached is not None and cached[0] == sig:
        return cached[1]
    try:
        with open(path, "rb") as f:
            cfg = pickle.load(f)
    except Exception:
        return None
    _CONFIG_CACHE[str(path)] = (sig, cfg)
    return cfg


def _ago(ts_epoch):
    """Compact 'Ns/Nm/Nh/Nd ago' for an epoch-second timestamp (best-effort)."""
    try:
        age = dt.datetime.now().timestamp() - float(ts_epoch)
    except (TypeError, ValueError):
        return "?"
    if age < 0:
        age = 0
    if age < 90:
        return f"{age:.0f}s ago"
    if age < 5400:
        return f"{age / 60:.0f}m ago"
    if age < 172800:
        return f"{age / 3600:.0f}h ago"
    return f"{age / 86400:.0f}d ago"


def _read_pipeline_status():
    """Read pipeline_status.json, returning empty dict on failure."""
    try:
        with open(BASE_DIR / "pipeline_status.json") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_pipeline_command(command, crypto=False, stock=False):
    """Write a pipeline command file atomically for the pipeline to consume.

    Returns True on success, error string on failure.
    """
    try:
        payload = {"command": command, "crypto": crypto, "stock": stock}
        tmp = str(PIPELINE_COMMAND) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, str(PIPELINE_COMMAND))
        return True
    except Exception as e:
        return str(e)


# ---------------------------------------------------------------------------
# Data Fetcher Thread
# ---------------------------------------------------------------------------
class DataFetcher(QObject):
    """Fetches data from Alpaca API on background timers."""

    account_updated = Signal(dict)
    positions_updated = Signal(list)
    orders_updated = Signal(list, bool)  # (orders, truncated)
    history_updated = Signal(dict)
    hw_updated = Signal(dict)
    news_updated = Signal(dict)
    stocks_updated = Signal(dict)
    chart_updated = Signal(dict)
    error_occurred = Signal(str, str)  # (stream, message) — per-stream health

    def __init__(self, api, role="hot"):
        super().__init__()
        self.api = api
        # role decides which timers start_timers starts: 'hot' = account/
        # positions/orders/hw (must never freeze); 'slow' = news/stocks + the
        # on-demand history/chart slots (the multi-minute news/LLM path lives
        # here so it can't stall live P&L).
        self.role = role
        # Main thread flips this on Markets-tab visibility (fetch_stocks gate).
        # Plain bool, GIL-atomic read/write across threads — no lock needed.
        self.markets_visible = True
        # Per-stream consecutive-fail counters + base intervals for timer
        # backoff (adjusted only from fetch methods, i.e. this thread).
        self._stream_fails = {}
        self._stream_base_ms = {}
        self._stocks_last_run = 0.0  # monotonic — throttled background refresh
        self._news_rotation_idx = 0  # rotating company-news cursor (fetch_news)
        # sym -> {'daily': {closes,timestamps,cached_at}, 'hourly': ..., '15min': ...}
        self._chart_cache = {}
        self._chart_cache_ttl = 300  # 5 min before re-fetching same resolution

    @Slot()
    def start_timers(self):
        """Create and start this role's polling timers (called after moveToThread).

        HOT role: account (10s) / positions (5s) / orders (30s) / hw (5s).
        SLOW role: news (5min) / stocks (30s) — plus the on-demand history and
        chart slots, which have no timer and are driven by GUI invokeMethod.
        """
        # Initial intervals come from the persisted refresh cadences (Settings
        # ops page) so a user-tuned Jetson load survives a restart; set_interval
        # applies later changes live.
        def _cad_ms(stream):
            return _cadence_seconds(stream) * 1000
        if self.role == "hot":
            self._timer_account = QTimer(self)
            self._timer_account.timeout.connect(self.fetch_account)
            self._timer_account.start(_cad_ms("account"))

            self._timer_positions = QTimer(self)
            self._timer_positions.timeout.connect(self.fetch_positions)
            self._timer_positions.start(_cad_ms("positions"))

            self._timer_orders = QTimer(self)
            self._timer_orders.timeout.connect(self.fetch_orders)
            self._timer_orders.start(_cad_ms("orders"))

            self._timer_hw = QTimer(self)
            self._timer_hw.timeout.connect(self.fetch_hw)
            self._timer_hw.start(_cad_ms("hw"))

            self._stream_base_ms = {
                "account": _cad_ms("account"), "positions": _cad_ms("positions"),
                "orders": _cad_ms("orders"), "hw": _cad_ms("hw"),
            }
            burst = (self.fetch_account, self.fetch_positions,
                     self.fetch_orders, self.fetch_hw)
        else:  # slow
            self._timer_news = QTimer(self)
            self._timer_news.timeout.connect(self.fetch_news)
            self._timer_news.start(_cad_ms("news"))

            self._timer_stocks = QTimer(self)
            self._timer_stocks.timeout.connect(self.fetch_stocks)
            self._timer_stocks.start(_cad_ms("stocks"))

            # (history refresh is driven by the GUI's _perf_timer so it targets
            # the active zoom, not always the default 1M — no timer here.)

            self._stream_base_ms = {
                "news": _cad_ms("news"), "stocks": _cad_ms("stocks")}

            # Load cached news immediately (instant startup)
            cached = _load_news_cache()
            if cached and cached.get('articles'):
                self.news_updated.emit({
                    'articles': cached['articles'],
                    'fng': cached.get('fng'),
                })

            # history + stocks first so they populate fast; news last because it
            # can take minutes (LLM upgrades) and must not delay the others.
            burst = (self.fetch_history, self.fetch_stocks, self.fetch_news)

        # Immediate first fetch (news will merge with cache).
        # Check interruption between calls so closeEvent can break the burst.
        for fn in burst:
            if QThread.currentThread().isInterruptionRequested():
                return
            fn()

    @Slot()
    def stop_timers(self):
        """Stop all timers (must be called from this object's thread)."""
        for attr in ("_timer_account", "_timer_positions", "_timer_orders",
                      "_timer_hw", "_timer_news",
                      "_timer_stocks"):
            timer = getattr(self, attr, None)
            if timer:
                timer.stop()

    @Slot(str, int)
    def set_interval(self, stream, ms):
        """Retune one of THIS thread's polling timers live (Settings ops page).
        Invoked from the GUI thread via QMetaObject.invokeMethod so the QTimer is
        touched only from the thread that owns it. Also resets the backoff base so
        a subsequent failure/success re-derives its multiplier from the new value;
        no-op for a stream this role doesn't own."""
        timer = getattr(self, f"_timer_{stream}", None)
        if timer is None or stream not in self._stream_base_ms:
            return
        ms = int(ms)
        if ms <= 0:
            return
        self._stream_base_ms[stream] = ms
        self._stream_fails[stream] = 0
        timer.start(ms)

    def _stream_result(self, stream, ok):
        """Per-stream timer backoff (this thread only — timers live here).

        On >=3 consecutive failures, double the stream's poll interval, capped
        at 4x its base; restore the base interval on the first success. No-op
        for on-demand streams (history/chart) which have no timer.
        """
        timer = getattr(self, f"_timer_{stream}", None)
        base = self._stream_base_ms.get(stream)
        if timer is None or base is None:
            return
        if ok:
            if self._stream_fails.get(stream, 0):
                self._stream_fails[stream] = 0
                if timer.interval() != base:
                    timer.start(base)
        else:
            n = self._stream_fails.get(stream, 0) + 1
            self._stream_fails[stream] = n
            if n >= 3:
                mult = min(4, 2 ** (n - 2))  # 3->x2, 4+->x4 (capped)
                target = base * mult
                if timer.interval() != target:
                    timer.start(target)

    @Slot()
    def fetch_account(self):
        try:
            acct = self.api.get_account()
            self.account_updated.emit({
                "equity": acct.equity,
                "cash": acct.cash,
                "buying_power": acct.buying_power,
                "last_equity": acct.last_equity,
            })
            self._stream_result("account", True)
        except Exception as e:
            self.error_occurred.emit("account", f"Account fetch: {e}")
            self._stream_result("account", False)

    @Slot()
    def fetch_positions(self):
        try:
            positions = self.api.list_positions()
            data = []
            for p in positions:
                data.append({
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "side": p.side,
                    "avg_entry_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "unrealized_pl": p.unrealized_pl,
                    "unrealized_plpc": p.unrealized_plpc,
                    "market_value": p.market_value,
                })
            self.positions_updated.emit(data)
            self._stream_result("positions", True)
        except Exception as e:
            self.error_occurred.emit("positions", f"Positions fetch: {e}")
            self._stream_result("positions", False)

    @Slot()
    def fetch_orders(self):
        # Paginate backward (newest-first) so the tax cost-basis sees older buy
        # lots too: a single limit=100 window silently truncated basis beyond
        # ~100 orders. Walk `until` back one page at a time, accumulating until
        # a short page (history exhausted), the clean-slate cutoff, or a hard
        # cap (1000 orders / 365 days lookback) — emitting `truncated` True only
        # when a cap cut history short so the tax card can flag incomplete basis.
        PAGE = 100
        HARD_ORDER_CAP = 1000
        LOOKBACK_DAYS = 365
        try:
            # Only count orders after the clean-slate cutoff (if set)
            after = None
            slate = BASE_DIR / ".clean_slate"
            if slate.exists():
                after = slate.read_text().strip() or None

            oldest_allowed = (dt.datetime.now(dt.timezone.utc)
                              - dt.timedelta(days=LOOKBACK_DAYS))
            data = []
            seen_ids = set()
            until = None
            prev_until = object()  # sentinel so first compare never matches
            truncated = False
            while True:
                batch = list(self.api.list_orders(
                    status="all", limit=PAGE, after=after, until=until,
                    direction="desc"))
                if not batch:
                    break
                for o in batch:
                    oid = getattr(o, "id", None)
                    if oid is not None:
                        if oid in seen_ids:
                            continue  # boundary-overlap dupe
                        seen_ids.add(oid)
                    data.append({
                        "id": str(oid) if oid is not None else "",
                        "symbol": o.symbol,
                        "side": o.side,
                        "qty": o.qty,
                        "type": o.type,
                        "status": o.status,
                        "submitted_at": str(o.submitted_at) if o.submitted_at else "",
                        "filled_at": str(o.filled_at) if o.filled_at else "",
                        "filled_avg_price": o.filled_avg_price,
                        "notional": getattr(o, "notional", None),
                        "filled_qty": o.filled_qty,
                    })
                if len(batch) < PAGE:
                    break  # history exhausted since the cutoff
                if len(data) >= HARD_ORDER_CAP:
                    truncated = True
                    break
                oldest = batch[-1]
                new_until = str(oldest.submitted_at) if oldest.submitted_at else None
                # 365-day lookback backstop on a still-full page
                if oldest.submitted_at is not None:
                    try:
                        ots = dt.datetime.fromisoformat(
                            str(oldest.submitted_at).replace("Z", "+00:00"))
                        if ots < oldest_allowed:
                            truncated = True
                            break
                    except (TypeError, ValueError):
                        pass
                # Can't advance the cursor (no ts, or a page of identical
                # timestamps) — stop rather than loop forever
                if new_until is None or new_until == prev_until:
                    break
                prev_until = until = new_until
            self.orders_updated.emit(data, truncated)
            self._stream_result("orders", True)
        except Exception as e:
            self.error_occurred.emit("orders", f"Orders fetch: {e}")
            self._stream_result("orders", False)

    @Slot(str, str)
    def fetch_history(self, period="1M", timeframe="1D"):
        try:
            hist = self.api.get_portfolio_history(period=period, timeframe=timeframe)
            self.history_updated.emit({
                "equity": list(hist.equity),
                "timestamp": list(hist.timestamp),
                "profit_loss": list(hist.profit_loss) if hist.profit_loss else [],
                "profit_loss_pct": list(hist.profit_loss_pct) if hist.profit_loss_pct else [],
                "period": period,
            })
        except Exception as e:
            self.error_occurred.emit("history", f"History fetch: {e}")

    @Slot()
    def fetch_hw(self):
        """Read hardware stats from sysfs (no torch, no sudo)."""
        try:
            gpu_temp = self._read_gpu_temp()
            cpu_temp = self._read_cpu_temp()
            ram_used, ram_total = self._read_ram()
            gpu_load = self._read_gpu_load()
            gpu_freq, gpu_max_freq = self._read_gpu_freq()
            cpu_usage = self._read_cpu_usage()
            cpu_freq, cpu_max_freq = self._read_cpu_freq()
            # Disk headroom on the repo/base volume: a full SD card silently
            # breaks status writes (write_status swallows OSError), so surface
            # used% + free bytes for the Models-tab Disk gauge.
            try:
                du = shutil.disk_usage(str(BASE_DIR))
                disk_total, disk_used, disk_free = du.total, du.used, du.free
            except OSError:
                disk_total = disk_used = disk_free = None
            self.hw_updated.emit({
                "gpu_temp": gpu_temp,
                "cpu_temp": cpu_temp,
                "ram_used": ram_used,
                "ram_total": ram_total,
                "gpu_load": gpu_load,
                "gpu_freq_mhz": gpu_freq,
                "gpu_max_freq_mhz": gpu_max_freq,
                "cpu_usage": cpu_usage,
                "cpu_freq_mhz": cpu_freq,
                "cpu_max_freq_mhz": cpu_max_freq,
                "disk_total": disk_total,
                "disk_used": disk_used,
                "disk_free": disk_free,
            })
            self._stream_result("hw", True)
        except Exception as e:
            self.error_occurred.emit("hw", f"HW fetch: {e}")
            self._stream_result("hw", False)

    @Slot()
    def fetch_news(self):
        """Fetch news headlines from Finnhub and Fear & Greed Index."""
        try:
            from sentiment import get_fear_greed, get_cnn_fear_greed, _get_finnhub, score_article_batch, try_llm_upgrade, _MODEL_RANK
            from stock_config import CRYPTO_SYMBOLS
            import datetime as _dt

            # Build base-symbol set for crypto headline matching
            crypto_bases = {s.split('/')[0] for s in CRYPTO_SYMBOLS}

            articles = []
            fng = get_fear_greed()
            cnn_fng = get_cnn_fear_greed()

            client = _get_finnhub()
            if client is not None:
                # Crypto general news
                try:
                    crypto_news = client.general_news('crypto', min_id=0)
                    for a in crypto_news[:15]:
                        a['_category'] = 'Crypto'
                        # Tag _symbol by scanning headline for known crypto bases
                        headline_upper = (a.get('headline', '') + ' ' + a.get('summary', '')).upper()
                        for sym in crypto_bases:
                            if sym in headline_upper:
                                a['_symbol'] = sym
                                break
                    articles.extend(crypto_news[:15])
                except Exception:
                    pass

                # General / market news
                try:
                    general_news = client.general_news('general', min_id=0)
                    for a in general_news[:15]:
                        a['_category'] = 'Market'
                    articles.extend(general_news[:15])
                except Exception:
                    pass

                # Company news — universe-driven and rotating. Replaces the old
                # 10 hardcoded high-beta tickers, which decoupled the feed from
                # load_stock_universe (hold AAPL, never see AAPL news; combined
                # sentiment dominated by those names). Cover up to 10 stock names
                # per cycle and ROTATE a persisted index through the whole
                # universe so every name gets coverage over time without
                # exploding Finnhub calls. (Held-symbol priority isn't used: the
                # positions cache is main-thread — this runs in the fetcher
                # thread — so the universe stands in for it.)
                _today = _dt.date.today()
                _from = (_today - _dt.timedelta(days=2)).isoformat()
                _to = _today.isoformat()
                try:
                    from stock_config import load_stock_universe
                    _uni = [s for s in load_stock_universe() if '/' not in s]
                except Exception:
                    _uni = []
                _BATCH = 10
                if _uni:
                    _start = getattr(self, '_news_rotation_idx', 0) % len(_uni)
                    _batch = [_uni[(_start + i) % len(_uni)]
                              for i in range(min(_BATCH, len(_uni)))]
                    self._news_rotation_idx = (_start + len(_batch)) % len(_uni)
                else:
                    _batch = []
                for stock in _batch:
                    try:
                        co_news = client.company_news(stock, _from=_from, to=_to)
                        for a in co_news[:5]:
                            a['_category'] = 'Stock'
                            a['_symbol'] = stock
                        articles.extend(co_news[:5])
                    except Exception:
                        pass

            # Deduplicate by headline before scoring
            seen_headlines = set()
            unique = []
            for a in articles:
                h = a.get('headline', '').strip().lower()
                if h and h not in seen_headlines:
                    seen_headlines.add(h)
                    unique.append(a)
            articles = unique

            # Build cache of already-scored headlines to avoid re-scoring
            cache = _load_news_cache()
            cached_scores = {}
            if cache and cache.get('articles'):
                for ca in cache['articles']:
                    key = ca.get('headline', '').strip().lower()
                    if key and '_sentiment' in ca:
                        cached_scores[key] = (ca['_sentiment'], ca.get('_sent_method', ''), ca.get('_scored_by_model', ''))

            # Split articles into already-scored (from cache) and new
            need_scoring = []
            for a in articles:
                key = a.get('headline', '').strip().lower()
                if key in cached_scores:
                    a['_sentiment'], a['_sent_method'], a['_scored_by_model'] = cached_scores[key]
                else:
                    need_scoring.append(a)

            # Only score genuinely new articles
            if need_scoring:
                scores, method = score_article_batch(need_scoring)
                for a, score in zip(need_scoring, scores):
                    a['_sentiment'] = score
                    a['_sent_method'] = method

            # Sort by datetime descending
            articles.sort(key=lambda a: a.get('datetime', 0), reverse=True)

            # Merge with cache: keep new articles + older cached ones not in this fetch
            if cache and cache.get('articles'):
                # Deduplicate by normalized headline — new articles take priority
                seen = {a.get('headline', '').strip().lower() for a in articles}
                for cached_a in cache['articles']:
                    key = cached_a.get('headline', '').strip().lower()
                    if key and key not in seen:
                        articles.append(cached_a)
                        seen.add(key)
                articles.sort(key=lambda a: a.get('datetime', 0), reverse=True)
            # Cap total articles to prevent unbounded growth (was 1000+ after days)
            articles = articles[:200]

            # Try to upgrade KW-scored or lower-tier articles to better models
            # Cap at 10 newest to avoid blocking the fetcher thread for minutes
            _top_rank = max(_MODEL_RANK.values())
            upgradeable = [a for a in articles
                           if _MODEL_RANK.get(
                               a.get('_scored_by_model', ''), 0) < _top_rank]
            if upgradeable:
                # Lowest-rank first so KW articles claim slots before
                # already-LLM-scored ones (stable: newest-first within rank)
                upgradeable.sort(key=lambda a: _MODEL_RANK.get(
                    a.get('_scored_by_model', ''), 0))
                upgradeable = upgradeable[:10]
                upgrade_scores = try_llm_upgrade(upgradeable)
                if upgrade_scores is not None:
                    for a, score in zip(upgradeable, upgrade_scores):
                        if score is not None:
                            a['_sentiment'] = score
                            a['_sent_method'] = 'LLM'

            # Save merged articles to cache
            _save_news_cache(articles, fng)

            # Compute 24h / 7d aggregate sentiment — combined + per asset class
            now_ts = _dt.datetime.now().timestamp()

            def _avg(vals):
                return sum(vals) / len(vals) if vals else None

            def _sent_by_window(secs):
                all_s, crypto_s, stock_s = [], [], []
                for a in articles:
                    if '_sentiment' not in a:
                        continue
                    if now_ts - a.get('datetime', 0) > secs:
                        continue
                    s = a['_sentiment']
                    if s is None:
                        continue
                    cat = a.get('_category', '')
                    all_s.append(s)
                    if cat in ('Crypto', 'Market'):
                        crypto_s.append(s)
                    if cat in ('Stock', 'Market'):
                        stock_s.append(s)
                return _avg(all_s), _avg(crypto_s), _avg(stock_s)

            avg_24h, crypto_24h, stock_24h = _sent_by_window(86400)
            avg_7d, crypto_7d, stock_7d = _sent_by_window(7 * 86400)

            self.news_updated.emit({
                'articles': articles,
                'fng': fng,
                'cnn_fng': cnn_fng,
                'sent_24h': avg_24h,
                'sent_7d': avg_7d,
                'crypto_24h': crypto_24h,
                'crypto_7d': crypto_7d,
                'stock_24h': stock_24h,
                'stock_7d': stock_7d,
            })
            self._stream_result("news", True)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error_occurred.emit("news", f"News fetch: {e}")
            self._stream_result("news", False)

    @Slot(str, str)
    def fetch_chart(self, symbol, resolution):
        """Fetch bars for a symbol at a given resolution (background thread).

        resolution: 'daily' | 'hourly' | '15min' | '5min'
        Uses get_crypto_bars() for crypto symbols (containing '/').
        Caches per (symbol, resolution) so zoom switches within the same
        resolution are instant. This is the ONLY place chart trade-marker
        file IO (chart_core.load_trade_markers) may happen — never from a
        paint/update path.
        """
        import datetime as _dt
        now_ts = _dt.datetime.now().timestamp()
        try:
            ttl_by_resolution = {'daily': 300, 'hourly': 300, '15min': 120, '5min': 60}
            ttl = ttl_by_resolution.get(resolution, 300)

            # Check cache
            sym_cache = self._chart_cache.get(symbol, {})
            cached = sym_cache.get(resolution)
            if cached and (now_ts - cached['cached_at']) < ttl:
                payload = dict(cached)
                payload['symbol'] = symbol
                payload['resolution'] = resolution
                try:
                    payload['markers'] = chart_core.load_trade_markers(
                        symbol, str(BASE_DIR / 'journals'), since_ts=now_ts - 366 * 86400)
                except Exception:
                    payload['markers'] = None
                self.chart_updated.emit(payload)
                return

            # Resolution → API params. Limits must cover the whole lookback
            # window: bars come back ascending, so a too-small limit keeps
            # the OLDEST bars and drops the most recent ones.
            res_config = {
                'daily':  ('1Day',  365, 370),
                'hourly': ('1Hour',  10, 250),   # 10d * 24 = 240 crypto bars
                '15min':  ('15Min',   5, 500),   # 5d * 96 = 480 crypto bars
                '5min':   ('5Min',    3, 900),   # 3d * 288 = 864 crypto bars
            }
            tf, lookback_days, limit = res_config.get(resolution, ('1Day', 365, 370))

            start = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=lookback_days)
            if '/' in symbol:
                bars = self.api.get_crypto_bars(
                    symbol, tf, start=start.isoformat(), limit=limit,
                )
            else:
                bars = self.api.get_bars(
                    symbol, tf, start=start.isoformat(), limit=limit,
                )

            closes = []
            timestamps = []
            opens, highs, lows, volumes = [], [], [], []
            ohlc_complete = True
            for b in bars:
                closes.append(float(b.c))
                o_val = getattr(b, 'o', None)
                h_val = getattr(b, 'h', None)
                l_val = getattr(b, 'l', None)
                v_val = getattr(b, 'v', None)
                if o_val is None or h_val is None or l_val is None:
                    ohlc_complete = False
                else:
                    opens.append(float(o_val))
                    highs.append(float(h_val))
                    lows.append(float(l_val))
                volumes.append(float(v_val) if v_val is not None else 0.0)
                try:
                    t = b.t
                    if hasattr(t, 'timestamp'):
                        timestamps.append(t.timestamp())
                    else:
                        ts_parsed = _dt.datetime.fromisoformat(
                            str(t).replace('Z', '+00:00'))
                        timestamps.append(ts_parsed.timestamp())
                except Exception:
                    timestamps.append(start.timestamp() + len(timestamps) * 3600)

            entry = {
                'closes': closes, 'timestamps': timestamps, 'cached_at': now_ts,
                'volumes': volumes,
            }
            if ohlc_complete and len(opens) == len(closes):
                entry['opens'] = opens
                entry['highs'] = highs
                entry['lows'] = lows

            # Store in nested cache: sym -> {resolution -> data}
            if symbol not in self._chart_cache:
                # Evict oldest symbol if cache is full (max 6 symbols)
                if len(self._chart_cache) >= 6:
                    oldest_sym = min(
                        self._chart_cache,
                        key=lambda s: max(
                            (v.get('cached_at', 0)
                             for v in self._chart_cache[s].values()), default=0))
                    del self._chart_cache[oldest_sym]
                self._chart_cache[symbol] = {}

            self._chart_cache[symbol][resolution] = entry

            payload = dict(entry)
            payload['symbol'] = symbol
            payload['resolution'] = resolution
            try:
                payload['markers'] = chart_core.load_trade_markers(
                    symbol, str(BASE_DIR / 'journals'), since_ts=now_ts - 366 * 86400)
            except Exception:
                payload['markers'] = None
            self.chart_updated.emit(payload)
        except Exception as e:
            # Emit the error payload (drives the on-chart status) AND tag the
            # 'chart' stream so a persistently broken chart fetch shows in the
            # per-stream API health, not just on the chart itself.
            self.chart_updated.emit({
                'symbol': symbol, 'resolution': resolution,
                'closes': [], 'timestamps': [], 'error': str(e), 'cached_at': now_ts,
            })
            self.error_occurred.emit("chart", f"Chart fetch: {e}")

    @Slot()
    def fetch_stocks(self):
        """Fetch stock + crypto snapshots + prediction cache for Markets tab."""
        # Visibility gate: when the Markets tab isn't showing, skip the snapshot/
        # prediction work — but still refresh at 4x the base interval so
        # background data isn't infinitely stale. `markets_visible` is a plain
        # bool set from the main thread on tab change; a GIL-atomic read here is
        # acceptable (no lock needed for a single bool).
        if not self.markets_visible:
            now = time.monotonic()
            if now - self._stocks_last_run < 120.0:  # 4x the 30s base interval
                return
        self._stocks_last_run = time.monotonic()
        try:
            from stock_config import load_stock_universe
            symbols = load_stock_universe()
            if not symbols:
                return

            stock_syms = [s for s in symbols if '/' not in s]
            crypto_syms = [s for s in symbols if '/' in s]

            # Batch snapshot from Alpaca
            snapshots = {}
            if stock_syms:
                try:
                    snaps = self.api.get_snapshots(stock_syms)
                    for sym, snap in snaps.items():
                        try:
                            latest = snap.latest_trade
                            price = float(latest.p) if latest else 0
                            bar = snap.daily_bar
                            prev_close = float(snap.prev_daily_bar.c) if snap.prev_daily_bar else price
                            day_open = float(bar.o) if bar else price
                            day_high = float(bar.h) if bar else price
                            day_low = float(bar.l) if bar else price
                            volume = int(bar.v) if bar else 0
                            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
                            snapshots[sym] = {
                                'price': price,
                                'prev_close': prev_close,
                                'open': day_open,
                                'high': day_high,
                                'low': day_low,
                                'volume': volume,
                                'change_pct': change_pct,
                            }
                        except Exception:
                            pass
                except Exception as e:
                    self.error_occurred.emit("stocks", f"Stock snapshots: {e}")

            if crypto_syms:
                try:
                    csnaps = self.api.get_crypto_snapshots(crypto_syms)
                    for sym, snap in csnaps.items():
                        try:
                            bar = snap.daily_bar
                            prev_bar = snap.prev_daily_bar
                            price = float(bar.c) if bar else 0
                            prev_close = float(prev_bar.c) if prev_bar else price
                            day_open = float(bar.o) if bar else price
                            day_high = float(bar.h) if bar else price
                            day_low = float(bar.l) if bar else price
                            volume = int(bar.v) if bar else 0
                            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
                            snapshots[sym] = {
                                'price': price,
                                'prev_close': prev_close,
                                'open': day_open,
                                'high': day_high,
                                'low': day_low,
                                'volume': volume,
                                'change_pct': change_pct,
                            }
                        except Exception:
                            pass
                except Exception as e:
                    self.error_occurred.emit("stocks", f"Crypto snapshots: {e}")

            # Read prediction caches (written by stock_loop / crypto_loop in
            # jetson env). Pass the WHOLE per-symbol dict through — do NOT
            # cherry-pick keys here: on_stocks needs every optional stance key
            # (meta_p, conviction, rank, llm_gate, regime) the loops may add.
            predictions = {}
            for pred_name in ("stock_predictions.json", "crypto_predictions.json"):
                pred_file = BASE_DIR / pred_name
                try:
                    if pred_file.exists():
                        with open(pred_file) as f:
                            predictions.update(json.load(f))
                except (OSError, json.JSONDecodeError):
                    pass

            # Read LLM analysis (written by llm_analyst.py). Same rule: fold each
            # book section in whole so advisor-v2 keys (p_up, conviction,
            # abstain, key_risks, event_flags) reach the dossier panel intact.
            llm_analysis = {}
            analysis_file = BASE_DIR / "llm_analysis.json"
            try:
                if analysis_file.exists():
                    with open(analysis_file) as f:
                        raw = json.load(f)
                    # Merge crypto + stock sections into flat dict
                    for section in raw.values():
                        if isinstance(section, dict):
                            llm_analysis.update(section)
            except (OSError, json.JSONDecodeError):
                pass

            self.stocks_updated.emit({
                'symbols': symbols,
                'snapshots': snapshots,
                'predictions': predictions,
                'llm_analysis': llm_analysis,
            })
            self._stream_result("stocks", True)
        except Exception as e:
            self.error_occurred.emit("stocks", f"Stocks fetch: {e}")
            self._stream_result("stocks", False)

    @staticmethod
    def _read_gpu_temp():
        """GPU temp via hw_monitor: zone-type scan, 20s cache, no subprocess.

        (The old tegrastats fallback never exits, so subprocess.run always
        burned its full 2s timeout before raising — fixed engine-side.)
        """
        return hw_get_gpu_temp()

    @staticmethod
    def _read_ram():
        """Read RAM from /proc/meminfo (no torch)."""
        try:
            with open("/proc/meminfo") as f:
                info = f.read()
            total = int(re.search(r"MemTotal:\s+(\d+)", info).group(1)) / 1024.0
            avail = int(re.search(r"MemAvailable:\s+(\d+)", info).group(1)) / 1024.0
            return round(total - avail, 1), round(total, 1)
        except Exception:
            return None, None

    @staticmethod
    def _read_gpu_load():
        """Read GPU load % from Jetson sysfs (0-1000 scale -> 0-100%)."""
        try:
            with open("/sys/devices/platform/bus@0/17000000.gpu/load") as f:
                return int(f.read().strip()) / 10.0
        except (FileNotFoundError, ValueError, OSError):
            return None

    @staticmethod
    def _read_gpu_freq():
        """Read GPU current/max frequency in MHz from devfreq."""
        try:
            with open("/sys/class/devfreq/17000000.gpu/cur_freq") as f:
                cur = int(f.read().strip()) / 1e6
            with open("/sys/class/devfreq/17000000.gpu/max_freq") as f:
                mx = int(f.read().strip()) / 1e6
            return cur, mx
        except (FileNotFoundError, ValueError, OSError):
            return None, None

    def _read_cpu_usage(self):
        """Read CPU usage % from /proc/stat (diff of two snapshots)."""
        try:
            with open("/proc/stat") as f:
                line = f.readline()  # first line: cpu  user nice system idle ...
            parts = list(map(int, line.split()[1:]))
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0)  # idle + iowait
            total = sum(parts)
            prev = getattr(self, '_cpu_prev', None)
            self._cpu_prev = (idle, total)
            if prev is None:
                return None
            d_idle = idle - prev[0]
            d_total = total - prev[1]
            if d_total == 0:
                return 0.0
            return (1.0 - d_idle / d_total) * 100.0
        except (FileNotFoundError, ValueError, OSError):
            return None

    @staticmethod
    def _read_cpu_freq():
        """Read CPU current/max frequency in MHz from cpu0 cpufreq."""
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
                cur = int(f.read().strip()) / 1000.0
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq") as f:
                mx = int(f.read().strip()) / 1000.0
            return cur, mx
        except (FileNotFoundError, ValueError, OSError):
            return None, None

    @staticmethod
    def _read_cpu_temp():
        """Read CPU temp from thermal_zone0."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError, OSError):
            return None


# ---------------------------------------------------------------------------
# Log Tailer Thread
# ---------------------------------------------------------------------------
class LogTailer(QObject):
    """Tails log files and emits new lines."""

    new_lines = Signal(str, str)  # (log_name, text)

    def __init__(self):
        super().__init__()
        self._positions = {}

    @Slot()
    def start_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.check_logs)
        self._timer.start(2_000)
        for name, path in LOG_FILES.items():
            try:
                self._positions[name] = path.stat().st_size
            except OSError:
                self._positions[name] = 0

    @Slot()
    def stop_timer(self):
        if hasattr(self, "_timer"):
            self._timer.stop()

    @Slot()
    def check_logs(self):
        for name, path in LOG_FILES.items():
            try:
                size = path.stat().st_size
            except OSError:
                continue
            last_pos = self._positions.get(name, 0)
            if size < last_pos:
                last_pos = 0
            if size > last_pos:
                with open(path, "r", errors="replace") as f:
                    f.seek(last_pos)
                    text = f.read(size - last_pos)
                self._positions[name] = size
                if text.strip():
                    self.new_lines.emit(name, text)


# ---------------------------------------------------------------------------
# Theme Application
# ---------------------------------------------------------------------------
def apply_theme(app):
    """Apply the current theme's QPalette and global stylesheet."""
    t = T
    palette = QPalette()
    palette.setColor(QPalette.Window, t["bg_dark"])
    palette.setColor(QPalette.WindowText, t["white"])
    palette.setColor(QPalette.Base, t["bg_log"])
    palette.setColor(QPalette.AlternateBase, t["bg_card"])
    palette.setColor(QPalette.ToolTipBase, t["bg_card"])
    palette.setColor(QPalette.ToolTipText, t["white"])
    palette.setColor(QPalette.Text, t["white"])
    palette.setColor(QPalette.Button, t["bg_header"])
    palette.setColor(QPalette.ButtonText, t["white"])
    palette.setColor(QPalette.BrightText, t["accent"])
    palette.setColor(QPalette.Link, t["accent"])
    palette.setColor(QPalette.Highlight, t["accent"])
    palette.setColor(QPalette.HighlightedText, t["bg_dark"])
    app.setPalette(palette)

    # Semantic color tokens (Phase 4.2). resolve_colors returns the SAME QColor
    # objects the theme carries, so these .name() strings are byte-identical to
    # the old t["..."] lookups — a pure rename. The QSS below references these.
    TOK = design_tokens.resolve_colors(t)
    bg_base   = TOK["bg"]["base"].name()        # was t["bg_dark"]
    bg_raised = TOK["bg"]["raised"].name()       # was t["bg_card"]
    bg_inset  = TOK["bg"]["inset"].name()        # was t["bg_table"]
    text_hi   = TOK["text"]["hi"].name()         # was t["white"]
    text_mid  = TOK["text"]["mid"].name()        # was t["muted"]
    accent_c  = TOK["accent"].name()             # was t["accent"]
    bg_header = TOK["_raw"]["bg_header"].name()   # ungrouped control-chrome
    bg_border = TOK["_raw"]["bg_border"].name()   # ungrouped stroke color
    bg_hover  = TOK["_raw"]["bg_hover"].name()    # ungrouped hover state

    # A subtly translucent accent for selections/glows (works on any theme).
    acc = TOK["accent"]
    acc_soft = QColor(acc.red(), acc.green(), acc.blue(), 46).name(QColor.HexArgb)
    acc_glow = QColor(acc.red(), acc.green(), acc.blue(), 90).name(QColor.HexArgb)

    # Design-system typography/space/radius tokens (Phase 4.1/4.2).
    ui_font = design_tokens.font_qss()
    numeric_font = design_tokens.numeric_qss()
    card_title_px, card_title_w = design_tokens.TYPE["small"]
    card_value_px, card_value_w = design_tokens.TYPE["display"]
    card_radius = design_tokens.RADIUS["panel"]  # 10px — current card radius

    app.setStyleSheet(f"""
        QWidget {{
            {ui_font}
            font-size: 13px;
        }}
        QMainWindow, QDialog {{ background-color: {bg_base}; }}
        QToolTip {{ color: {text_hi}; background-color: {bg_raised};
                    border: 1px solid {accent_c}; padding: 5px 8px;
                    border-radius: 4px; }}

        /* ---- Design-system cards + numeric role (Phase 4.2) — global
               attribute selectors replace the per-widget _restyle card loop ---- */
        QFrame[card="true"] {{
            background-color: {bg_raised}; border: 1px solid {bg_border};
            border-radius: {card_radius}px; padding: 14px;
        }}
        QLabel[card_title="true"] {{
            color: {text_mid}; font-size: {card_title_px}px;
            font-weight: {card_title_w}; letter-spacing: 1px;
        }}
        QLabel#card_value {{
            color: {text_hi}; font-size: {card_value_px}px; font-weight: {card_value_w};
        }}
        QLabel[numeric="true"] {{ {numeric_font} }}

        /* ---- Tabs: premium top nav ---- */
        QTabWidget::pane {{
            border: 1px solid {bg_border}; border-radius: 8px;
            top: -1px; background: {bg_base};
        }}
        QTabBar {{ qproperty-drawBase: 0; }}
        QTabBar::tab {{
            background: transparent; color: {text_mid};
            padding: 9px 22px; border: none; margin-right: 2px;
            border-top-left-radius: 7px; border-top-right-radius: 7px;
            font-size: 12px; font-weight: 600; letter-spacing: 0.4px;
        }}
        QTabBar::tab:selected {{
            background: {bg_raised}; color: {accent_c};
            border-bottom: 2px solid {accent_c};
        }}
        QTabBar::tab:hover:!selected {{
            background: {bg_hover}; color: {text_hi};
        }}

        /* ---- Default buttons (inline-styled buttons still override) ---- */
        QPushButton {{
            background-color: {bg_header}; color: {text_hi};
            border: 1px solid {bg_border}; border-radius: 6px;
            padding: 6px 14px; font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: {bg_hover}; border-color: {accent_c};
            color: {accent_c};
        }}
        QPushButton:pressed {{ background-color: {acc_glow}; color: {bg_base}; }}
        QPushButton:disabled {{
            color: {text_mid}; background-color: {bg_base};
            border-color: {bg_border};
        }}

        /* ---- Inputs ---- */
        QComboBox, QLineEdit, QSpinBox {{
            background: {bg_inset}; color: {text_hi};
            border: 1px solid {bg_border}; padding: 5px 8px;
            border-radius: 6px; selection-background-color: {accent_c};
            selection-color: {bg_base};
        }}
        QComboBox:hover, QLineEdit:hover, QSpinBox:hover {{ border-color: {text_mid}; }}
        QComboBox:focus, QLineEdit:focus, QSpinBox:focus {{ border: 1px solid {accent_c}; }}
        QComboBox::drop-down {{ border: none; width: 20px; }}
        QComboBox QAbstractItemView {{
            background: {bg_raised}; color: {text_hi};
            border: 1px solid {bg_border}; border-radius: 6px;
            padding: 4px; outline: none;
            selection-background-color: {accent_c};
            selection-color: {bg_base};
        }}
        QSpinBox::up-button, QSpinBox::down-button {{
            background: {bg_header}; border: none; width: 16px;
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {bg_hover}; }}

        /* ---- Checkboxes ---- */
        QCheckBox {{ spacing: 7px; color: {text_hi}; }}
        QCheckBox::indicator {{ width: 16px; height: 16px; }}
        QCheckBox::indicator:checked {{
            background-color: {accent_c};
            border: 1px solid {accent_c}; border-radius: 4px;
        }}
        QCheckBox::indicator:unchecked {{
            background-color: {bg_header};
            border: 1px solid {bg_border}; border-radius: 4px;
        }}
        QCheckBox::indicator:unchecked:hover {{ border: 1px solid {accent_c}; }}

        /* ---- Tables: selection accent + header hover ---- */
        QTableWidget {{
            selection-background-color: {acc_soft}; selection-color: {text_hi};
            alternate-background-color: {bg_raised};
        }}
        QTableWidget::item:selected {{ background: {acc_soft}; color: {text_hi}; }}
        QHeaderView::section:hover {{ background-color: {bg_hover}; }}

        /* ---- Progress ---- */
        QProgressBar {{
            background: {bg_header}; text-align: center;
            color: {text_hi};
            border: 1px solid {bg_border}; border-radius: 6px;
        }}
        QProgressBar::chunk {{ background: {accent_c}; border-radius: 5px; }}

        /* ---- Status / toolbar / groups ---- */
        QStatusBar {{
            background: {bg_base}; color: {text_mid};
            border-top: 1px solid {bg_border};
        }}
        QStatusBar QLabel {{ padding: 0 8px; }}
        QStatusBar::item {{ border: none; }}
        QGroupBox {{ color: {accent_c}; }}

        /* ---- Scrollbars (vertical + horizontal) ---- */
        QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
        QScrollBar::handle:vertical {{
            background: {bg_border}; border-radius: 5px; min-height: 28px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {accent_c}; }}
        QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
        QScrollBar::handle:horizontal {{
            background: {bg_border}; border-radius: 5px; min-width: 28px;
        }}
        QScrollBar::handle:horizontal:hover {{ background: {accent_c}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

        /* ---- Splitter ---- */
        QSplitter::handle {{ background: {bg_border}; }}
        QSplitter::handle:hover {{ background: {accent_c}; }}
        QSplitter::handle:horizontal {{ width: 3px; }}
        QSplitter::handle:vertical {{ height: 3px; }}

        QToolBar {{
            background: {bg_base}; border-bottom: 1px solid {bg_border};
            spacing: 8px; padding: 4px 8px;
        }}
        QToolBar::separator {{ background: {bg_border}; width: 1px; margin: 6px 6px; }}
        QLabel#toolbar_label {{ color: {text_mid}; font-size: 11px;
                                font-weight: 600; letter-spacing: 0.5px; }}

        QMessageBox {{ background-color: {bg_raised}; color: {text_hi}; }}
        QMessageBox QLabel {{ color: {text_hi}; }}
        QMessageBox QPushButton {{
            background-color: {bg_header}; color: {text_hi};
            border: 1px solid {bg_border}; border-radius: 6px;
            padding: 6px 16px; min-width: 64px;
        }}
        QMessageBox QPushButton:hover {{
            background-color: {accent_c}; color: {bg_base};
        }}
    """)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class TradingDashboard(QMainWindow):
    _llm_test_done = Signal(bool, float, str, str)  # ok, elapsed_ms, error, model
    # Journal-analytics worker result (compute_stats dict, or None on failure)
    # delivered back to the UI thread from the off-thread load_trades worker.
    _journal_stats_ready = Signal(object)
    # Notification self-test result (ok, detail) from the off-thread notify send.
    _notify_test_done = Signal(bool, str)

    def __init__(self, api, app):
        super().__init__()
        self.api = api
        self._app = app
        self.setWindowTitle("Trader Dashboard")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        self._orders_cache = []
        self._hw_cache = {}
        # HW-history sparklines (Models tab): GPU-temp + RAM% ring buffers, fed
        # by on_hw; a fingerprint memo skips redundant redraws.
        self._hw_gpu_temp_hist = deque(maxlen=60)
        self._hw_ram_hist = deque(maxlen=60)
        self._spark_fp = {}
        self._positions_cache = []   # last on_positions payload (manual-trade context)
        self._account_cache = {}     # last on_account payload (buying power / equity)
        self._flatten_was_pending = False  # flatten pending->complete transition
        # Unified linked-selection (5.1): one router drives combo/table/chart/
        # detail. _syncing_symbol guards against widget-signal feedback loops;
        # _active_symbol is the last symbol the router committed to (chart TTL).
        self._syncing_symbol = False
        self._active_symbol = None
        # Positions-table diff-update (5.3): stable sym->row map + persistent
        # per-symbol Close buttons + the row->symbol order the button column
        # currently reflects (buttons only rebuilt when this order changes).
        self._pos_row_by_sym = {}
        self._pos_close_btn = {}
        self._pos_btn_order = None
        self._last_fng = None  # Fear & Greed, fetched off-thread by fetch_news
        self._chart_fp = {}  # chart fingerprint memo — skips redundant repaints
        self._chart_last_ok = {'perf': None, 'price': None}  # staleness ticker
        # Manual pan/zoom flags: while true, a data refresh preserves the user's
        # current x-range instead of snapping back to the zoom window (separate
        # flag per plot). Cleared by the zoom presets + the Reset-view buttons.
        self._chart_user_viewport = False
        self._perf_user_viewport = False
        # Equity benchmark overlay (SPY/BTC): mapped symbol + last daily payload
        # + the last equity view it was aligned against + a legend-label memo.
        self._bench_symbol = None
        self._bench_chart_data = None
        self._last_equity_view = None
        self._bench_legend_label = None
        self._bench_req_symbol = None
        self._bench_req_ts = 0.0

        # Cockpit alert-feed state: edge-triggered flags so a condition that
        # stays true for many ticks pushes ONE alert, not one per refresh.
        self._alert_halt_state = False
        self._alert_flatten_pending = False
        self._alert_hb_stale = {'crypto': False, 'stock': False}
        self._alert_pipe_stale = False
        self._alert_last_cmd_ts = 0.0
        self._today_spark_cache = None   # last "1D" history dict (sparkline)

        # Per-stream health (replaces the single _error_count that any account
        # success reset — which masked a permanently-dead stream). Each on_*
        # success stamps its own last_ok (monotonic) and zeroes its own fails;
        # on_error bumps only the failing stream. The "API:" label is a
        # worst-case summary refreshed from on_error + the 10s on_account tick.
        self._boot_monotonic = time.monotonic()
        _streams = ('account', 'positions', 'orders', 'hw',
                    'news', 'stocks', 'history', 'chart')
        self._stream_health = {s: {'last_ok': None, 'fails': 0} for s in _streams}
        # Nominal poll interval (s) per stream for the staleness check + tooltip.
        self._stream_intervals = {
            'account': 10, 'positions': 5, 'orders': 30, 'hw': 5,
            'news': 300, 'stocks': 30, 'history': 300, 'chart': 120,
        }

        # Restore last used theme
        settings = _load_gui_settings()
        saved_theme = settings.get('theme', 'Batman')
        self._current_theme = saved_theme if saved_theme in THEMES else "Batman"
        set_theme(self._current_theme)

        # Toolbar with theme selector and clock
        self._build_toolbar()

        # Central tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Build tabs
        self._build_dashboard_tab()
        self._build_trading_tab()
        self._build_performance_tab()
        self._build_news_tab()
        self._build_stocks_tab()
        self._build_models_tab()
        self._build_logs_tab()
        self._build_settings_tab()

        # Status bar
        self._status_conn = QLabel("API: \u2014")
        self._status_positions = QLabel("Pos: 0")
        self._status_sentiment = QLabel("FnG: \u2014")
        self._status_gpu = QLabel("GPU: \u2014")
        self._status_ram = QLabel("RAM: \u2014")
        self._status_gpu_info = QLabel("GPU: \u2014")
        self._status_updated = QLabel("")
        # Ticking-digit status readouts render in the tabular numerals font
        # (numeric="true" -> global QLabel[numeric="true"] QSS) so the numbers
        # don't jitter as they update. Text-only labels (API/updated) stay body.
        for _num_lbl in (self._status_positions, self._status_sentiment,
                         self._status_gpu, self._status_ram, self._status_gpu_info):
            _num_lbl.setProperty("numeric", True)
        # Persistent flatten-pending banner (empty unless a flatten is in
        # flight). Permanent so a transient showMessage never hides it.
        self._flatten_banner = QLabel("")
        self.statusBar().addWidget(self._status_updated)
        self.statusBar().addWidget(self._status_positions)
        self.statusBar().addWidget(self._status_sentiment)
        self.statusBar().addPermanentWidget(self._flatten_banner)
        self.statusBar().addPermanentWidget(self._status_conn)
        self.statusBar().addPermanentWidget(self._status_gpu)
        self.statusBar().addPermanentWidget(self._status_ram)
        self.statusBar().addPermanentWidget(self._status_gpu_info)
        # Keyboard-shortcuts hint lives on the status-bar tooltip (5.3).
        self.statusBar().setToolTip(
            "Shortcuts: Ctrl+K palette · Ctrl+1-8 tabs · Ctrl+L logs · "
            "Ctrl+F filter · F5 refresh")

        # Apply initial styling
        self._restyle()

        # Data fetchers on TWO background threads (hot/slow split): the
        # multi-minute news/LLM path must never freeze live P&L, so it runs on
        # its own thread apart from account/positions/orders/hw.
        #   HOT  = account (10s) / positions (5s) / orders (30s) / hw (5s)
        #   SLOW = news (5min) / stocks (30s) + on-demand history/chart slots
        self._fetcher_hot_thread = QThread()
        self._fetcher_hot = DataFetcher(api, role="hot")
        self._fetcher_hot.moveToThread(self._fetcher_hot_thread)
        self._fetcher_hot.account_updated.connect(self.on_account)
        self._fetcher_hot.positions_updated.connect(self.on_positions)
        self._fetcher_hot.orders_updated.connect(self.on_orders)
        self._fetcher_hot.hw_updated.connect(self.on_hw)
        self._fetcher_hot.error_occurred.connect(self.on_error)
        self._fetcher_hot_thread.started.connect(self._fetcher_hot.start_timers)

        self._fetcher_slow_thread = QThread()
        self._fetcher_slow = DataFetcher(api, role="slow")
        self._fetcher_slow.moveToThread(self._fetcher_slow_thread)
        self._fetcher_slow.news_updated.connect(self.on_news)
        self._fetcher_slow.stocks_updated.connect(self.on_stocks)
        self._fetcher_slow.history_updated.connect(self.on_history)
        self._fetcher_slow.chart_updated.connect(self.on_chart)
        self._fetcher_slow.error_occurred.connect(self.on_error)
        self._fetcher_slow_thread.started.connect(self._fetcher_slow.start_timers)

        # fetch_stocks skips work when the Markets tab isn't visible: seed the
        # flag to the current tab and keep it in sync on every tab change.
        self._fetcher_slow.markets_visible = (
            self.tabs.currentIndex() == self._markets_tab_index)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._fetcher_hot_thread.start()
        self._fetcher_slow_thread.start()

        # One boot-time full-range history fetch so account_baseline.json is
        # written (via on_history's period=="1A" path) without the user having
        # to click the 1A zoom. Delayed so the fetcher event loop is up first.
        QTimer.singleShot(8000, self._boot_baseline_fetch)

        # Log tailer on background thread
        self._tailer_thread = QThread()
        self._tailer = LogTailer()
        self._tailer.moveToThread(self._tailer_thread)
        self._tailer.new_lines.connect(self.on_log_lines)
        self._tailer_thread.started.connect(self._tailer.start_timer)
        self._tailer_thread.start()

        # Model refresh timer (cadence from Settings ops page; default 60s)
        self._model_timer = QTimer(self)
        self._model_timer.timeout.connect(self._refresh_models_tab)
        self._model_timer.start(_cadence_seconds('models') * 1000)
        self._refresh_models_tab()

        # Performance-history refresh (every 5 min, targets active zoom)
        self._perf_timer = QTimer(self)
        self._perf_timer.timeout.connect(self._request_perf_history)
        self._perf_timer.start(300_000)

        # Market-session clock (every 30s — no seconds shown, so 1s is wasteful)
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(30_000)
        self._update_clock()

        # Markets-tab chart auto-refresh (every 2 min, only while tab visible)
        self._chart_timer = QTimer(self)
        self._chart_timer.timeout.connect(self._auto_refresh_chart)
        self._chart_timer.start(120_000)

        # Chart staleness ticker (every 30s — re-titles without re-fetching)
        self._chart_stale_timer = QTimer(self)
        self._chart_stale_timer.timeout.connect(self._refresh_chart_staleness)
        self._chart_stale_timer.start(30_000)

        # Global keyboard shortcuts + Ctrl-K command palette (5.2 / 5.3).
        self._install_shortcuts()

    # ---- Toolbar ---------------------------------------------------------
    def _build_toolbar(self):
        toolbar = QToolBar("Settings")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._logo_label = QLabel()
        self._logo_label.setFixedSize(80, 80)
        self._logo_label.setPixmap(generate_theme_logo(self._current_theme))
        toolbar.addWidget(self._logo_label)

        theme_label = QLabel(" Theme: ")
        theme_label.setObjectName("toolbar_label")
        toolbar.addWidget(theme_label)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(list(THEMES.keys()))
        self._theme_combo.setCurrentText(self._current_theme)
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)
        toolbar.addWidget(self._theme_combo)

        toolbar.addSeparator()

        self._clock_label = QLabel("")
        self._clock_label.setStyleSheet("font-size: 12px; padding: 0 12px;")
        toolbar.addWidget(self._clock_label)

        # Spacer to push clock right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        self._clock_label_right = QLabel("")
        self._clock_label_right.setStyleSheet("font-size: 12px; font-weight: bold; padding: 0 8px;")
        toolbar.addWidget(self._clock_label_right)

    def _market_session(self):
        """(is_open, timedelta_to_next_boundary) for US equity RTH. Extends the
        cheap _in_rth zoneinfo check with the next open/close boundary, rolling
        over weekends. Market holidays are IGNORED (comment in _in_rth) — the
        countdown may read early on a holiday, never wrong-direction."""
        now_et = dt.datetime.now(TZ_EASTERN)
        open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        if now_et.weekday() < 5 and open_t <= now_et < close_t:
            return True, close_t - now_et
        # Closed: next open is today's open (if still ahead) else the next
        # weekday's open (skip Sat/Sun).
        cand = open_t
        if now_et >= open_t:
            cand = cand + dt.timedelta(days=1)
        while cand.weekday() >= 5:
            cand = cand + dt.timedelta(days=1)
        return False, cand - now_et

    @staticmethod
    def _fmt_countdown(td):
        """'2h17m' / '16h02m' from a timedelta (floored, minutes zero-padded)."""
        total = int(max(0, td.total_seconds()))
        return f"{total // 3600}h{(total % 3600) // 60:02d}m"

    def _update_clock(self):
        """Market-session clock: local CT time + NYSE open/closed with a
        countdown to the next boundary, plus a Crypto 24/7 reminder (that book
        never closes). Replaces the seconds wall-clock."""
        now_ct = dt.datetime.now(TZ_CENTRAL)
        tstr = now_ct.strftime("%-I:%M %p CT")
        is_open, delta = self._market_session()
        core = (f"NYSE OPEN (closes {self._fmt_countdown(delta)})" if is_open
                else f"NYSE CLOSED (opens {self._fmt_countdown(delta)})")
        self._clock_label_right.setText(f"{tstr} · {core} · Crypto 24/7")

    # ---- Global shortcuts + Ctrl-K command palette (gui_review §10/§11) ---
    def _install_shortcuts(self):
        """Window-parented QShortcuts (fire from any tab). Ctrl+K opens the
        command palette; Ctrl+1..8 switch tabs (tab-widget order); Ctrl+L jumps
        to the latest logs; Ctrl+F focuses the active tab's filter; F5 nudges the
        hot fetchers + refreshes the Models tab. Cmd+K maps automatically from
        QKeySequence('Ctrl+K') on macOS; on the Jetson it stays Ctrl+K."""
        def sc(seq, slot):
            s = QShortcut(QKeySequence(seq), self)
            s.activated.connect(slot)
            return s
        sc("Ctrl+K", self._open_command_palette)
        for i in range(1, 9):  # Ctrl+1..8 -> tabs 0..7
            sc(f"Ctrl+{i}", lambda idx=i - 1: self._shortcut_switch_tab(idx))
        sc("Ctrl+L", self._shortcut_logs)
        sc("Ctrl+F", self._shortcut_focus_filter)
        sc("F5", self._shortcut_refresh)

    def _shortcut_switch_tab(self, idx):
        if 0 <= idx < self.tabs.count():
            self.tabs.setCurrentIndex(idx)

    def _shortcut_logs(self):
        """Ctrl+L -> Logs tab + jump to the newest line."""
        idx = getattr(self, '_logs_tab_index', -1)
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)
        try:
            self._log_jump_to_latest()
        except Exception:
            pass

    def _shortcut_focus_filter(self):
        """Ctrl+F focuses the active tab's filter field: the logs regex box on
        Logs, the news filter combo on News; no-op on every other tab."""
        idx = self.tabs.currentIndex()
        if idx == getattr(self, '_logs_tab_index', -1) and hasattr(self, '_log_filter'):
            self._log_filter.setFocus()
            self._log_filter.selectAll()
        elif idx == getattr(self, '_news_tab_index', -1) \
                and hasattr(self, '_news_filter_combo'):
            self._news_filter_combo.setFocus()

    def _shortcut_refresh(self):
        """F5 nudges the hot fetchers (account/positions/orders) + refreshes the
        Models tab — no new timers, just queued invocations of existing slots."""
        from PySide6.QtCore import QMetaObject
        try:
            for slot in ("fetch_account", "fetch_positions", "fetch_orders"):
                QMetaObject.invokeMethod(self._fetcher_hot, slot, Qt.QueuedConnection)
        except Exception:
            pass
        try:
            self._refresh_models_tab()
        except Exception:
            pass

    def _palette_jump_symbol(self, sym):
        """Palette symbol entry -> Markets tab + unified linked selection."""
        idx = getattr(self, '_markets_tab_index', -1)
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)
        self._set_active_symbol(sym, 'palette')

    def _palette_jump_logs(self):
        self._shortcut_logs()

    def _command_palette_items(self):
        """Build the (label, callable) command list for the Ctrl-K palette:
        every tab jump, every universe symbol + crypto pair, and the action
        commands. Every action routes through its EXISTING handler so the
        confirmation dialogs (halt-resume, flatten, restart, retrain, LLM
        re-bill) still fire — the palette never bypasses a guard."""
        items = []
        # Tab jumps.
        for i in range(self.tabs.count()):
            items.append((f"Go to {self.tabs.tabText(i)}",
                          lambda idx=i: self.tabs.setCurrentIndex(idx)))
        # Symbol jumps (Markets tab + linked selection).
        try:
            from stock_config import load_stock_universe, CRYPTO_SYMBOLS
            syms = list(load_stock_universe()) + list(CRYPTO_SYMBOLS)
        except Exception:
            syms = []
        seen = set()
        for s in syms:
            if not s or s in seen:
                continue
            seen.add(s)
            items.append((f"Symbol: {s}",
                          lambda sym=s: self._palette_jump_symbol(sym)))
        # Actions — each invokes the existing guarded handler.
        try:
            resume = halt_active()
        except Exception:
            resume = False
        items.append(("Resume entries" if resume else "Halt entries",
                      self._toggle_halt_clicked))
        items.append(("Flatten all", self._flatten_all_clicked))
        items.append(("Restart pipeline", self._restart_pipeline_clicked))
        items.append(("Retrain crypto", lambda: self._trigger_retrain(crypto=True)))
        items.append(("Retrain stock", lambda: self._trigger_retrain(stock=True)))
        items.append(("Retrain both",
                      lambda: self._trigger_retrain(crypto=True, stock=True)))
        # No stale-only refresh path exists (refresh_all is whole-universe); the
        # handler's confirm dialog surfaces the still-fresh count before re-bill.
        items.append(("Refresh stale LLM", self._refresh_all_llm_clicked))
        items.append(("Jump to latest logs", self._palette_jump_logs))
        items.append(("Reset chart view", self._reset_chart_view))
        return items

    def _open_command_palette(self):
        """Ctrl-K modal palette: QLineEdit + QListWidget over the command list.
        Case-insensitive substring match, prefix hits ranked first; Enter runs
        the top/selected item, Esc closes, Up/Down navigate. The chosen action
        runs AFTER the palette closes (QTimer(0)) so its own confirm dialog owns
        the screen cleanly."""
        existing = getattr(self, '_cmd_palette', None)
        if existing is not None:
            try:
                if existing.isVisible():
                    existing.raise_()
                    existing._input.setFocus()
                    return
            except RuntimeError:
                pass  # C++ dialog already deleted — fall through and rebuild

        items = self._command_palette_items()
        dlg = QDialog(self)
        dlg.setWindowTitle("Command Palette")
        dlg.setModal(True)
        dlg.setMinimumWidth(460)
        # Delete on close so repeated opens don't accumulate hidden dialogs
        # (matters on the 8 GB Jetson); the isVisible() guard above catches the
        # resulting deleted-wrapper RuntimeError and rebuilds cleanly.
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(8, 8, 8, 8)
        inp = QLineEdit()
        inp.setPlaceholderText("Jump to symbol / tab, or run an action…")
        v.addWidget(inp)
        lst = QListWidget()
        v.addWidget(lst)
        dlg._input = inp

        t = T
        dlg.setStyleSheet(
            f"QDialog {{ background-color: {t['bg_dark'].name()}; }}"
            f" QLineEdit {{ background-color: {t['bg_table'].name()};"
            f" color: {t['white'].name()}; border: 1px solid {t['bg_border'].name()};"
            f" border-radius: 6px; padding: 6px 8px; font-size: 14px; }}"
            f" QLineEdit:focus {{ border: 1px solid {t['accent'].name()}; }}"
            f" QListWidget {{ background-color: {t['bg_table'].name()};"
            f" color: {t['white'].name()}; border: 1px solid {t['bg_border'].name()};"
            f" border-radius: 6px; }}"
            f" QListWidget::item {{ padding: 4px 6px; }}"
            f" QListWidget::item:selected {{ background-color: {t['accent'].name()};"
            f" color: {t['bg_dark'].name()}; }}")

        def repopulate():
            q = inp.text().strip().lower()
            lst.clear()
            if not q:
                matches = items
            else:
                prefix, sub = [], []
                for label, cb in items:
                    pos = label.lower().find(q)
                    if pos == 0:
                        prefix.append((label, cb))
                    elif pos > 0:
                        sub.append((label, cb))
                matches = prefix + sub
            for label, cb in matches[:200]:
                it = QListWidgetItem(label)
                it.setData(Qt.UserRole, cb)
                lst.addItem(it)
            if lst.count():
                lst.setCurrentRow(0)

        def run_current():
            it = lst.currentItem() or (lst.item(0) if lst.count() else None)
            if it is None:
                return
            cb = it.data(Qt.UserRole)
            dlg.accept()
            if callable(cb):
                QTimer.singleShot(0, cb)  # run after the palette closes

        def on_key(event):
            key = event.key()
            if key in (Qt.Key_Down, Qt.Key_Up) and lst.count():
                row = lst.currentRow()
                lst.setCurrentRow(min(row + 1, lst.count() - 1) if key == Qt.Key_Down
                                  else max(row - 1, 0))
                event.accept()
                return
            if key in (Qt.Key_Return, Qt.Key_Enter):
                run_current()
                event.accept()
                return
            if key == Qt.Key_Escape:
                dlg.reject()
                event.accept()
                return
            QLineEdit.keyPressEvent(inp, event)

        inp.textChanged.connect(repopulate)
        inp.keyPressEvent = on_key
        lst.itemActivated.connect(lambda _it: run_current())  # double-click

        repopulate()
        self._cmd_palette = dlg
        inp.setFocus()
        dlg.exec()

    def _on_theme_changed(self, name):
        if name in THEMES:
            self._current_theme = name
            set_theme(name)
            apply_theme(self._app)
            self._logo_label.setPixmap(generate_theme_logo(name))
            self.setWindowTitle(f"Trader Dashboard \u2014 {name}")
            self._restyle()
            # Persist theme choice
            settings = _load_gui_settings()
            settings['theme'] = name
            _save_gui_settings(settings)

    # ---- Restyle (called on theme change) --------------------------------
    def _restyle(self):
        """Re-apply all inline widget styles from current theme."""
        t = T

        # Table styling helper
        table_style = (
            f"QTableWidget {{ background-color: {t['bg_table'].name()};"
            f" gridline-color: {t['bg_border'].name()};"
            f" border: 1px solid {t['bg_border'].name()}; border-radius: 8px; }}"
            f" QTableWidget::item {{ padding: 6px 8px; }}"
            f" QHeaderView {{ background-color: {t['bg_header'].name()}; }}"
            f" QHeaderView::section {{ background-color: {t['bg_header'].name()};"
            f" color: {t['muted'].name()}; padding: 8px 8px; border: none;"
            f" border-bottom: 2px solid {t['bg_border'].name()};"
            f" font-weight: 600; }}"
        )
        group_style = (
            f"QGroupBox {{ font-weight: bold; border: 1px solid {t['bg_border'].name()};"
            f" border-radius: 6px; margin-top: 8px; padding-top: 16px; }}"
            f" QGroupBox::title {{ subcontrol-position: top left; padding: 0 6px; }}"
        )

        # Cards are styled entirely by the global QSS attribute selectors
        # (QFrame[card="true"] / QLabel[card_title="true"] / QLabel#card_value in
        # apply_theme) — no per-widget restyle loop. This also picks up cards the
        # old hand-maintained list silently missed (e.g. the _jstat_* journal
        # cards). Dynamic P&L colors are re-applied by _set_card on data updates.

        # Tables
        for table in [self._positions_table, self._open_orders_table,
                       self._fills_table, self._model_table, self._news_table,
                       self._stock_table]:
            table.setStyleSheet(table_style)

        # Group boxes
        for group in self.findChildren(QGroupBox):
            group.setStyleSheet(group_style)

        # Cockpit feed lists (alerts + recent trades) — same surface as logs
        feed_style = (
            f"QListWidget {{ background-color: {t['bg_log'].name()};"
            f" color: {t['white'].name()};"
            f" border: 1px solid {t['bg_border'].name()};"
            f" border-radius: 6px; }}"
            f" QListWidget::item {{ padding: 2px 4px; }}"
        )
        for lst in [getattr(self, '_alerts_list', None),
                    getattr(self, '_last_actions_list', None)]:
            if lst is not None:
                lst.setStyleSheet(feed_style)

        # Markets book-wide regime chips (pill labels above the stance table)
        chip_style = (
            f"QLabel {{ background-color: {t['bg_header'].name()};"
            f" color: {t['accent'].name()}; font-size: 11px; font-weight: 600;"
            f" border: 1px solid {t['bg_border'].name()};"
            f" border-radius: 4px; padding: 2px 8px; }}"
        )
        for chip in [getattr(self, '_crypto_regime_chip', None),
                     getattr(self, '_stock_regime_chip', None)]:
            if chip is not None:
                chip.setStyleSheet(chip_style)

        # Plots — theme -> chart palette flows through chart_core exclusively
        pal = _chart_palette()
        self._chart_pal = pal
        self._equity_plot.setBackground(pal['bg'])
        self._pnl_plot.setBackground(pal['bg'])
        self._stock_chart_widget.setBackground(pal['bg'])
        if hasattr(self, '_today_spark'):
            self._today_spark.setBackground(pal['bg'])
        for _spark_pw_attr in ('_gpu_temp_spark_pw', '_ram_spark_pw'):
            _sp = getattr(self, _spark_pw_attr, None)
            if _sp is not None:
                _sp.setBackground(pal['bg'])
        for plot in [self._equity_plot, self._pnl_plot, self._stock_chart, self._stock_vol_plot]:
            self._style_plot(plot, pal)

        # Plot pen colors
        self._equity_curve.setPen(pg.mkPen(pal['equity'], width=2))
        self._equity_hwm.setPen(pg.mkPen(pal['hwm'], width=1, style=Qt.DashLine))
        self._equity_dd_fill.setBrush(pg.mkBrush(pal['dd_fill']))
        self._stock_chart_line.setPen(pg.mkPen(pal['equity'], width=2))
        self._equity_xhair.set_palette(pal)
        self._stock_xhair.set_palette(pal)
        self._entry_scatter.setPen(pg.mkPen(pal['bg'], width=2))
        self._exit_scatter.setPen(pg.mkPen(pal['bg'], width=2))

        # Price-chart indicator overlays + open-position / last-price guide lines
        if hasattr(self, '_sma20_line'):
            self._sma20_line.setPen(pg.mkPen(pal['crosshair'], width=1))
            self._sma50_line.setPen(pg.mkPen(pal['equity'], width=1))
            for ln in (self._atr_upper_line, self._atr_lower_line):
                ln.setPen(pg.mkPen(pal['hwm'], width=1, style=Qt.DashLine))
            _afc = QColor(pal['hwm']); _afc.setAlpha(30)
            self._atr_fill.setBrush(pg.mkBrush(_afc))
        if hasattr(self, '_pos_entry_line'):
            for ln, col, dashed in (
                    (self._pos_entry_line, pal['hwm'], False),
                    (self._pos_stop_line, pal['down'], True),
                    (self._pos_tp_line, pal['up'], True)):
                ln.setPen(pg.mkPen(col, width=1,
                                   style=Qt.DashLine if dashed else Qt.SolidLine))
                try:
                    ln.label.setColor(col)
                except Exception:
                    pass
        if hasattr(self, '_last_price_line'):
            self._last_price_line.setPen(pg.mkPen(pal['fg'], width=1, style=Qt.DashLine))
            try:
                self._last_price_line.label.setColor(pal['fg'])
            except Exception:
                pass
        if hasattr(self, '_bench_curve'):
            self._bench_curve.setPen(pg.mkPen(pal['hwm'], width=1, style=Qt.DashLine))

        # Zoom buttons
        for buttons in [self._stock_zoom_buttons, self._perf_zoom_buttons]:
            for z, btn in buttons.items():
                if btn.isChecked():
                    btn.setStyleSheet(
                        f"background-color: {t['accent'].name()}; color: {t['bg_dark'].name()};"
                        f" font-weight: bold; border-radius: 4px;")
                else:
                    btn.setStyleSheet(
                        f"background-color: {t['bg_header'].name()}; color: {t['muted'].name()};"
                        f" border: 1px solid {t['bg_border'].name()}; border-radius: 4px;")

        # Log display
        self._log_display.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {t['bg_log'].name()};"
            f" color: {t['white'].name()};"
            f" border: 1px solid {t['bg_border'].name()}; }}"
        )

        # Log filter row (regex box + level combo + jump button)
        if hasattr(self, '_log_filter'):
            log_input_style = (
                f"QLineEdit, QComboBox {{ background-color: {t['bg_table'].name()};"
                f" color: {t['white'].name()}; border: 1px solid {t['bg_border'].name()};"
                f" border-radius: 6px; padding: 4px 8px; }}"
                f" QLineEdit:focus, QComboBox:focus {{"
                f" border: 1px solid {t['accent'].name()}; }}"
            )
            self._log_filter.setStyleSheet(log_input_style)
            self._log_level.setStyleSheet(log_input_style)
            self._log_jump_btn.setStyleSheet(
                f"QPushButton {{ background-color: {t['bg_header'].name()};"
                f" color: {t['white'].name()};"
                f" border: 1px solid {t['bg_border'].name()};"
                f" border-radius: 4px; padding: 4px 10px; }}"
                f" QPushButton:hover {{ background-color: {t['accent'].name()};"
                f" color: {t['bg_dark'].name()}; }}")

        # Manual trade inputs
        trade_input_style = (
            f"QLineEdit {{ background-color: {t['bg_table'].name()};"
            f" color: {t['white'].name()}; border: 1px solid {t['bg_border'].name()};"
            f" border-radius: 6px; padding: 5px 8px; }}"
            f" QLineEdit:hover {{ border: 1px solid {t['muted'].name()}; }}"
            f" QLineEdit:focus {{ border: 1px solid {t['accent'].name()}; }}"
        )
        for widget in [self._manual_symbol, self._manual_qty, self._manual_notional]:
            widget.setStyleSheet(trade_input_style)

        # Settings tab inputs
        input_style = (
            f"QLineEdit, QSpinBox, QComboBox {{ background-color: {t['bg_table'].name()};"
            f" color: {t['white'].name()}; border: 1px solid {t['bg_border'].name()};"
            f" border-radius: 6px; padding: 5px 8px; }}"
            f" QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{"
            f" border: 1px solid {t['accent'].name()}; }}"
        )
        btn_style = (
            f"QPushButton {{ background-color: {t['bg_header'].name()}; color: {t['white'].name()};"
            f" border: 1px solid {t['bg_border'].name()}; border-radius: 4px; padding: 4px 8px; }}"
            f" QPushButton:hover {{ background-color: {t['accent'].name()}; color: {t['bg_dark'].name()}; }}"
        )
        if hasattr(self, '_settings_test_btn'):
            self._settings_test_btn.setStyleSheet(btn_style)
            for key_edit in self._settings_api_keys.values():
                key_edit.setStyleSheet(input_style)
            self._settings_fmp_key.setStyleSheet(input_style)
            self._settings_latency.setStyleSheet(input_style)
            if hasattr(self, '_settings_tier_override'):
                self._settings_tier_override.setStyleSheet(input_style)
            if hasattr(self, '_settings_analyst_model'):
                self._settings_analyst_model.setStyleSheet(input_style)
            if hasattr(self, '_settings_sentiment_model'):
                self._settings_sentiment_model.setStyleSheet(input_style)
            for toggle in self._settings_key_toggles.values():
                toggle.setStyleSheet(btn_style)
        if hasattr(self, '_settings_indicator_preset'):
            self._settings_indicator_preset.setStyleSheet(input_style)
            self._indicator_feature_list.setStyleSheet(
                f"QPlainTextEdit {{ background-color: {t['bg_table'].name()};"
                f" color: {t['white'].name()}; border: 1px solid {t['bg_border'].name()};"
                f" border-radius: 4px; padding: 4px; }}"
            )

        # Retrain buttons
        if hasattr(self, '_retrain_crypto_btn'):
            retrain_btn_style = (
                f"QPushButton {{ background-color: {t['bg_header'].name()}; color: {t['white'].name()};"
                f" border: 1px solid {t['bg_border'].name()}; border-radius: 4px;"
                f" padding: 4px 12px; font-weight: bold; font-size: 11px; }}"
                f" QPushButton:hover {{ background-color: {t['accent'].name()}; color: {t['bg_dark'].name()}; }}"
                f" QPushButton:disabled {{ color: {t['muted'].name()}; background-color: {t['bg_dark'].name()}; }}"
            )
            for btn in [self._retrain_crypto_btn, self._retrain_stock_btn, self._retrain_both_btn]:
                btn.setStyleSheet(retrain_btn_style)
            cancel_btn_style = (
                f"QPushButton {{ background-color: {t['bg_header'].name()}; color: {t['red'].name()};"
                f" border: 1px solid {t['red'].name()}; border-radius: 4px;"
                f" padding: 4px 12px; font-weight: bold; font-size: 11px; }}"
                f" QPushButton:hover {{ background-color: {t['red'].name()}; color: white; }}"
            )
            self._retrain_cancel_btn.setStyleSheet(cancel_btn_style)

        # Restart pipeline button
        if hasattr(self, '_restart_pipeline_btn'):
            self._restart_pipeline_btn.setStyleSheet(retrain_btn_style)

        # LLM refresh buttons
        if hasattr(self, '_llm_refresh_btn'):
            self._llm_refresh_btn.setStyleSheet(retrain_btn_style)
        if hasattr(self, '_llm_refresh_one_btn'):
            self._llm_refresh_one_btn.setStyleSheet(retrain_btn_style)

        # Bot control buttons
        if hasattr(self, '_crypto_start_btn'):
            bot_btn_style = (
                f"QPushButton {{ background-color: {t['bg_header'].name()}; color: {t['white'].name()};"
                f" border: 1px solid {t['bg_border'].name()}; border-radius: 4px;"
                f" padding: 4px 12px; font-weight: bold; font-size: 11px; }}"
                f" QPushButton:hover {{ background-color: {t['accent'].name()}; color: {t['bg_dark'].name()}; }}"
                f" QPushButton:disabled {{ color: {t['muted'].name()}; background-color: {t['bg_dark'].name()}; }}"
            )
            for btn in [self._crypto_start_btn, self._crypto_stop_btn,
                        self._stock_start_btn, self._stock_stop_btn]:
                btn.setStyleSheet(bot_btn_style)

        # Manual BUY/SELL buttons — pnl up/down, follow the theme.
        if hasattr(self, '_manual_buy_btn'):
            self._style_trade_buttons()

        # Clock
        self._clock_label_right.setStyleSheet(
            f"font-size: 12px; font-weight: bold; padding: 0 8px; color: {t['accent'].name()};"
        )

        # HW sparklines (muted line follows the theme).
        for _spark in (getattr(self, '_gpu_temp_spark', None),
                       getattr(self, '_ram_spark', None)):
            if _spark is not None:
                _spark.setPen(pg.mkPen(t['muted'].name(), width=1))

        # Re-apply cached data so data-colored items (P&L bars, candles,
        # heatmap tiles) repaint under the new theme instead of staying
        # stuck with the previous theme's colors until the next fetch.
        self._chart_fp.clear()
        cached = self._perf_history_cache.get(self._perf_api_period()[0])
        if cached:
            self._apply_perf_data(cached)
        self._apply_chart_zoom()
        if getattr(self, '_stock_data_cache', None):
            self.on_stocks(self._stock_data_cache)

        # Theme-switch transient: the colored P&L cards keep the previous
        # theme's baked-in hex until the next 10s account tick — re-tint them
        # now from the cached account payload (best-effort; skip if absent).
        # Mirrors on_account's day/total-P&L math so the strings stay identical.
        acct = getattr(self, '_account_cache', None)
        if acct:
            try:
                eq, le = float(acct["equity"]), float(acct["last_equity"])
                base, approx = self._account_baseline()
                dpl, tpl = eq - le, eq - base
                self._set_card(
                    self._card_day_pl,
                    f"{fmt_money(dpl)} ({fmt_pct(dpl / le * 100.0 if le else 0.0)})",
                    pnl_color(dpl))
                self._set_card(
                    self._card_total_pl,
                    f"{'~' if approx else ''}{fmt_money(tpl)} "
                    f"({fmt_pct(tpl / base * 100.0 if base else 0.0)})",
                    pnl_color(tpl))
            except (KeyError, ValueError, TypeError):
                pass

        # Cockpit data-colored widgets (chips, risk bars, DD badge, banner) and
        # the today sparkline repaint under the new theme (fingerprint cleared
        # above, so the sparkline redraws).
        if getattr(self, '_today_spark_cache', None):
            self._repaint_today_sparkline(self._today_spark_cache)
        try:
            self._refresh_cockpit()
        except Exception:
            pass

        # Log severity colors are baked in at append time — re-render the log
        # view so error/warning coloring follows the new theme.
        if hasattr(self, '_log_display'):
            self._rerender_log_view()

    def _style_trade_buttons(self):
        """Style the manual BUY/SELL buttons from the shared pnl palette
        (BUY=up, SELL=down) with a contrast-picked label, so they follow the
        theme and match the charts' up/down instead of hardcoded Material hex."""
        for btn, key in ((self._manual_buy_btn, 'up'), (self._manual_sell_btn, 'down')):
            bg = PAL[key]
            hover = chart_core.mix(bg, '#ffffff', 0.15)
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {bg}; color: {_on_color(bg)};"
                f" font-weight: bold; border-radius: 4px; }}"
                f" QPushButton:hover {{ background-color: {hover}; }}")

    def _style_plot(self, plot, pal):
        """Themed axes for a PlotWidget OR PlotItem — fixes the grey-tick-on
        -every-theme DateAxisItem defect. Backgrounds stay on the top-level
        widget (setBackground); this only touches axis pens/text/grid."""
        pi = plot.getPlotItem() if hasattr(plot, 'getPlotItem') else plot
        for side in ('left', 'bottom'):
            ax = pi.getAxis(side)
            ax.setPen(pg.mkPen(pal['grid']))
            ax.setTextPen(pg.mkPen(pal['fg']))
        pi.showGrid(x=True, y=True, alpha=0.25)

    def _set_chart_status(self, plot, base_title, status):
        """Render a ChartStatus as a colored plot title — every empty/stale
        /error chart state is visible, never a silently blank chart."""
        pal = getattr(self, '_chart_pal', None) or _chart_palette()
        suffix = status.title_suffix(time.time())
        if status.status == 'ok' and not status.note:
            color = pal['fg']
        elif status.status == 'error':
            color = pal['title_err']
        else:
            color = pal['title_warn']
        pi = plot.getPlotItem() if hasattr(plot, 'getPlotItem') else plot
        pi.setTitle(base_title + suffix, color=color, size='10pt')

    def _refresh_chart_staleness(self):
        """Re-title charts whose data hasn't refreshed recently — data stays
        visible, the title just warns. Does not re-fetch or re-plot data."""
        now = time.time()
        perf_last = self._chart_last_ok.get('perf')
        if perf_last is not None and (now - perf_last) > 900:
            status = chart_core.ChartStatus(status=chart_core.PARTIAL,
                                             message='stale', updated_at=perf_last)
            self._set_chart_status(self._equity_plot, 'Equity Curve', status)
        price_last = self._chart_last_ok.get('price')
        if price_last is not None and (now - price_last) > 600:
            sym = self._stock_symbol_combo.currentText()
            status = chart_core.ChartStatus(status=chart_core.PARTIAL,
                                             message='stale', updated_at=price_last)
            self._set_chart_status(self._stock_chart, f'{sym} ({self._stock_zoom})', status)

    # ---- Tab 1: Cockpit (was Dashboard) ----------------------------------
    def _build_dashboard_tab(self):
        """Cockpit landing: mode/halt banner, per-book heartbeat strip, account
        cards + today sparkline + live-DD badge, open-risk gauge, and a
        two-column positions | (alerts + recent-trades) split. See gui_review
        2026-07 §10 for the target information architecture."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        # 1) Mode / halt / flatten banner row (always visible) --------------
        banner_row = QHBoxLayout()
        self._mode_chip = QLabel("PAPER")
        self._mode_chip.setObjectName("mode_chip")
        self._mode_chip.setAlignment(Qt.AlignCenter)
        banner_row.addWidget(self._mode_chip)
        self._halt_banner = QLabel("")   # populated red when notify.halt_active()
        self._halt_banner.setObjectName("halt_banner")
        banner_row.addWidget(self._halt_banner)
        banner_row.addStretch()
        self._flatten_echo = QLabel("")  # larger cockpit echo of the flatten flag
        self._flatten_echo.setObjectName("flatten_echo")
        banner_row.addWidget(self._flatten_echo)
        layout.addLayout(banner_row)

        # 2) Heartbeat strip: Crypto / Stock chips + newest journal-write age
        hb_row = QHBoxLayout()
        hb_lead = QLabel("Bots:")
        hb_lead.setStyleSheet("font-weight: bold;")
        hb_row.addWidget(hb_lead)
        self._hb_crypto = QLabel("Crypto —")
        self._hb_stock = QLabel("Stock —")
        for chip in (self._hb_crypto, self._hb_stock):
            chip.setObjectName("hb_chip")
            hb_row.addWidget(chip)
        self._hb_journal = QLabel("journal: —")
        self._hb_journal.setObjectName("hb_sub")
        hb_row.addWidget(self._hb_journal)
        hb_row.addStretch()
        layout.addLayout(hb_row)

        # 3) Account cards ---------------------------------------------------
        cards_layout = QHBoxLayout()
        self._card_equity = make_card("Equity")
        self._card_cash = make_card("Cash")
        self._card_buying_power = make_card("Buying Power")
        self._card_day_pl = make_card("P&L (since prior close)")
        self._card_total_pl = make_card("Total P&L")
        for c in [self._card_equity, self._card_cash, self._card_buying_power,
                   self._card_day_pl, self._card_total_pl]:
            cards_layout.addWidget(c)
        layout.addLayout(cards_layout)

        # ... today sparkline + live-DD badge (grouped with equity per §10 IA)
        spark_row = QHBoxLayout()
        today_lbl = QLabel("Today")
        today_lbl.setStyleSheet("font-size: 11px; font-weight: bold;")
        spark_row.addWidget(today_lbl)
        self._today_spark = pg.PlotWidget()
        self._today_spark.setFixedHeight(60)
        self._today_spark.setMenuEnabled(False)
        self._today_spark.setMouseEnabled(x=False, y=False)
        self._today_spark.hideAxis('left')
        self._today_spark.hideAxis('bottom')
        self._today_spark.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self._today_spark_curve = self._today_spark.plot(pen=pg.mkPen(width=2))
        spark_row.addWidget(self._today_spark, stretch=1)
        self._dd_badge = QLabel("DD: —")
        self._dd_badge.setObjectName("dd_badge")
        spark_row.addWidget(self._dd_badge)
        layout.addLayout(spark_row)

        # 4) Open-risk gauge: per-book stop-risk vs MAX_BOOK_RISK_PCT budget --
        risk_group = QGroupBox("Open Risk — stop-risk vs per-book budget")
        risk_v = QVBoxLayout(risk_group)
        self._risk_rows = {}
        for book in ('crypto', 'stock'):
            r = QHBoxLayout()
            name_lbl = QLabel(f"{book.capitalize()} —")
            name_lbl.setMinimumWidth(170)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(14)
            bar.setTextVisible(False)
            detail = QLabel("—")
            r.addWidget(name_lbl)
            r.addWidget(bar, stretch=1)
            r.addWidget(detail)
            risk_v.addLayout(r)
            self._risk_rows[book] = (name_lbl, bar, detail)
        self._risk_largest = QLabel("Largest position: —")
        self._risk_largest.setObjectName("hb_sub")
        risk_v.addWidget(self._risk_largest)
        layout.addWidget(risk_group)

        # 5) Two-column: positions table | (alerts feed + recent-trades feed)
        two_col = QHBoxLayout()

        left_col = QVBoxLayout()
        pos_label = QLabel("Open Positions")
        pos_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 4px;")
        left_col.addWidget(pos_label)
        # ---- positions table: unchanged setup, only re-parented into left col
        # Cols 8-10 (Stop/TP/%→Stop) are exit-distance estimates from policy +
        # position_state.json (gui_review_2026-07 §4, Phase 2.4); the Close
        # button stays last (now col 11).
        self._positions_table = QTableWidget(0, 12)
        self._positions_table.setHorizontalHeaderLabels(
            ["Symbol", "Qty", "Side", "Avg Entry", "Current Price",
             "Mkt Value", "Unrealized P&L", "P&L %",
             "Stop", "TP", "%→Stop", ""]
        )
        header = self._positions_table.horizontalHeader()
        for col in range(11):
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        header.setSectionResizeMode(11, QHeaderView.Fixed)
        self._positions_table.setColumnWidth(11, 60)
        self._positions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._positions_table.setAlternatingRowColors(True)
        self._positions_table.setSortingEnabled(True)
        left_col.addWidget(self._positions_table)
        two_col.addLayout(left_col, stretch=3)

        right_col = QVBoxLayout()
        alerts_group = QGroupBox("Alerts")
        ag_v = QVBoxLayout(alerts_group)
        self._alerts_list = QListWidget()
        self._alerts_list.setObjectName("feed_list")
        ag_v.addWidget(self._alerts_list)
        right_col.addWidget(alerts_group, stretch=1)

        actions_group = QGroupBox("Recent Trades (7d)")
        act_v = QVBoxLayout(actions_group)
        self._last_actions_list = QListWidget()
        self._last_actions_list.setObjectName("feed_list")
        act_v.addWidget(self._last_actions_list)
        right_col.addWidget(actions_group, stretch=1)
        two_col.addLayout(right_col, stretch=2)

        layout.addLayout(two_col, stretch=1)

        self.tabs.addTab(tab, "Cockpit")

    # ---- Cockpit refresh helpers -----------------------------------------
    def _chip_style(self, color):
        """Pill stylesheet for a heartbeat/status chip (colored text on a card
        background). color is a QColor from T — no raw hex."""
        return (f"QLabel {{ color: {color.name()};"
                f" background-color: {T['bg_card'].name()};"
                f" border: 1px solid {T['bg_border'].name()};"
                f" border-radius: 8px; padding: 3px 10px; font-weight: 600; }}")

    def _alert_color(self, kind):
        """Map an alert kind to a theme color (no raw hex)."""
        if kind in ('halt', 'flatten', 'order-error', 'rejected'):
            return T['red']
        if kind in ('stale', 'stream', 'heartbeat'):
            return T.get('yellow', T['white'])
        if kind in ('resume', 'flatten-complete'):
            return T['green']
        return T['white']

    def _push_alert(self, kind, text):
        """Prepend a timestamped alert to the cockpit feed (newest-first, capped
        100). Deduped: skipped if an identical kind+text is already the newest
        entry, so an edge condition that stays true across ticks pushes once."""
        lst = getattr(self, '_alerts_list', None)
        if lst is None:
            return
        top = lst.item(0)
        if top is not None and top.data(Qt.UserRole) == (kind, text):
            return
        ts = dt.datetime.now(TZ_CENTRAL).strftime("%H:%M:%S")
        item = QListWidgetItem(f"{ts}  {text}")
        item.setData(Qt.UserRole, (kind, text))
        item.setForeground(self._alert_color(kind))
        lst.insertItem(0, item)
        while lst.count() > 100:
            lst.takeItem(lst.count() - 1)

    def _refresh_last_actions(self):
        """Render the 10 most recent closed round-trips (last 7d) in the cockpit
        recent-trades feed via journal_stats.load_trades. Journals may not exist
        on this dev Mac — every disk touch is wrapped; a failure leaves the feed
        untouched. 'side' slot shows exit_reason: every trade is a long-book
        round-trip, so the exit cause is the informative token."""
        lst = getattr(self, '_last_actions_list', None)
        if lst is None:
            return
        try:
            since = time.time() - 7 * 86400
            trades = journal_stats.load_trades(str(JOURNAL_DIR), since_ts=since)
        except Exception:
            return
        lst.clear()
        if not trades:
            item = QListWidgetItem("no closed trades in 7d")
            item.setForeground(T.get('muted', T['white']))
            lst.addItem(item)
            return
        for t in reversed(trades[-10:]):   # load_trades is ascending; newest-first
            try:
                hhmm = dt.datetime.fromtimestamp(
                    t['exit_ts'], tz=TZ_CENTRAL).strftime("%H:%M")
            except Exception:
                hhmm = "--:--"
            pnl = t.get('pnl_pct')
            pnl_str = f"{pnl:+.2f}%" if pnl is not None else "?"
            reason = t.get('exit_reason') or 'exit'
            item = QListWidgetItem(
                f"{hhmm}  {t.get('symbol', '?')}  {reason}  {pnl_str}")
            item.setForeground(pnl_color(pnl))
            lst.addItem(item)

    def _in_rth(self):
        """True during US equity regular hours (Mon–Fri 09:30–16:00 ET). Cheap
        zoneinfo wall-clock check — no Alpaca get_clock() call. Ignores market
        holidays: acceptable, since a stale stock heartbeat on a holiday then
        reads as 'off-hours' gray rather than a false red alert."""
        now = dt.datetime.now(TZ_EASTERN)
        if now.weekday() >= 5:
            return False
        mins = now.hour * 60 + now.minute
        return 9 * 60 + 30 <= mins < 16 * 60

    def _newest_journal_age(self):
        """Age (s) of the newest journals/*.jsonl mtime, or None if the dir is
        missing/empty/unreadable."""
        try:
            newest = max(
                (p.stat().st_mtime for p in JOURNAL_DIR.glob('*.jsonl')),
                default=None)
        except OSError:
            return None
        if newest is None:
            return None
        return max(0.0, time.time() - newest)

    def _refresh_heartbeats(self):
        """Per-book heartbeat chips + newest-journal age. Healthy = heartbeat
        mtime age < 180s (notify.ping_heartbeat rate-limits to 1/min). The stock
        bot only pings in RTH, so a stale stock heartbeat outside RTH is normal
        and renders gray 'off-hours', never red."""
        now = time.time()
        in_rth = self._in_rth()
        for book, chip in (('crypto', self._hb_crypto), ('stock', self._hb_stock)):
            try:
                age = now - HEARTBEAT_FILES[book].stat().st_mtime
            except OSError:
                age = None
            off_hours = (book == 'stock' and not in_rth)
            if age is not None and age < 180:
                state = 'ok'
                txt = f"{book.capitalize()} alive {age:.0f}s ago"
            elif off_hours:
                state = 'off'
                txt = f"{book.capitalize()} off-hours"
            elif age is None:
                state = 'stale'
                txt = f"{book.capitalize()} STALE (no ping)"
            else:
                state = 'stale'
                txt = f"{book.capitalize()} STALE {age / 60:.0f}m"
            color = {'ok': T['green'], 'stale': T['red']}.get(
                state, T.get('muted', T['white']))
            chip.setText(txt)
            chip.setStyleSheet(self._chip_style(color))
            # Edge-triggered stale alert (crypto always; stock only in-hours).
            was = self._alert_hb_stale.get(book, False)
            is_stale = (state == 'stale')
            if is_stale and not was:
                self._push_alert('heartbeat', f"{book} heartbeat stale")
            self._alert_hb_stale[book] = is_stale
        age_j = self._newest_journal_age()
        self._hb_journal.setText(
            "journal: —" if age_j is None
            else f"journal: {chart_core.format_age(age_j)} ago")

    def _refresh_risk_gauge(self):
        """Per-book open-risk gauge vs MAX_BOOK_RISK_PCT budget, + positions/book
        and largest single-name exposure. Reads account_risk_registry.json
        ({book: {risk, rho, ts}} per risk_budget.write_book_risk); a missing file
        or an entry older than 10 min renders gray 'no risk data'."""
        now = time.time()
        reg = {}
        try:
            with open(ACCOUNT_RISK_REGISTRY_FILE) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                reg = loaded
        except (OSError, json.JSONDecodeError, ValueError):
            reg = {}

        try:
            equity = float(self._account_cache.get('equity') or 0.0)
        except (TypeError, ValueError, AttributeError):
            equity = 0.0
        counts = {'crypto': 0, 'stock': 0}
        largest_sym, largest_frac = None, 0.0
        for p in (self._positions_cache or []):
            sym = p.get('symbol', '')
            book = 'crypto' if sym in CRYPTO_SYMBOL_SET else 'stock'
            counts[book] = counts.get(book, 0) + 1
            try:
                mv = abs(float(p.get('market_value')))
            except (TypeError, ValueError):
                mv = 0.0
            frac = (mv / equity) if equity > 0 else 0.0
            if frac > largest_frac:
                largest_sym, largest_frac = sym, frac

        for book, (name_lbl, bar, detail) in self._risk_rows.items():
            entry = reg.get(book)
            entry = entry if isinstance(entry, dict) else None
            ts = entry.get('ts') if entry else None
            try:
                stale = (ts is None) or (now - float(ts) > 600)
            except (TypeError, ValueError):
                stale = True
            detail.setText(f"{counts.get(book, 0)} pos")
            detail.setStyleSheet(f"color: {T.get('muted', T['white']).name()};")
            if entry is None or stale:
                name_lbl.setText(f"{book.capitalize()}: no risk data")
                name_lbl.setStyleSheet(
                    f"color: {T.get('muted', T['white']).name()};")
                bar.setValue(0)
                bar.setStyleSheet(
                    "QProgressBar::chunk { background-color: "
                    f"{T.get('muted', T['white']).name()}; }}")
                continue
            try:
                risk = float(entry.get('risk') or 0.0)
            except (TypeError, ValueError):
                risk = 0.0
            pct = (risk / MAX_BOOK_RISK_PCT * 100.0) if MAX_BOOK_RISK_PCT else 0.0
            if pct < 70:
                col = T['green']
            elif pct < 100:
                col = T.get('yellow', T['white'])
            else:
                col = T['red']
            name_lbl.setText(f"{book.capitalize()} risk: {pct:.0f}% of budget")
            name_lbl.setStyleSheet(
                f"color: {T['white'].name()}; font-weight: 600;")
            bar.setValue(int(min(pct, 100)))
            bar.setStyleSheet(
                "QProgressBar::chunk { background-color: " f"{col.name()}; }}")

        if largest_sym is not None and largest_frac > 0:
            self._risk_largest.setText(
                f"Largest position: {largest_sym} "
                f"{largest_frac * 100:.1f}% of equity")
        else:
            self._risk_largest.setText("Largest position: —")

    def _refresh_cockpit_banner(self):
        """Mode chip (static PAPER) + halt banner + flatten echo. Edge-triggered
        halt/flatten alerts. Reason read the same way _refresh_models_tab does."""
        self._mode_chip.setStyleSheet(
            f"QLabel {{ color: {T['bg_dark'].name()};"
            f" background-color: {T['accent'].name()};"
            f" border-radius: 8px; padding: 3px 12px; font-weight: 700; }}")

        halted = halt_active()
        if halted:
            reason = ""
            try:
                reason = json.loads(
                    (BASE_DIR / "trading_halt.flag").read_text() or "{}"
                ).get("reason", "")
            except Exception:
                pass  # `touch trading_halt.flag` leaves a non-JSON file
            self._halt_banner.setText(
                "TRADING HALTED" + (f": {reason}" if reason else ""))
            self._halt_banner.setStyleSheet(
                f"QLabel {{ color: {T['red'].name()}; font-weight: 700;"
                f" font-size: 14px; }}")
        else:
            self._halt_banner.setText("")
        if halted and not self._alert_halt_state:
            self._push_alert('halt', "Trading halted — entries blocked")
        elif not halted and self._alert_halt_state:
            self._push_alert('resume', "Entries resumed")
        self._alert_halt_state = halted

        pending = notify.flatten_requested()
        if pending:
            n = len(self._positions_cache or [])
            self._flatten_echo.setText(f"FLATTEN PENDING ({n} pos)")
            self._flatten_echo.setStyleSheet(
                f"QLabel {{ color: {T['red'].name()}; font-weight: 700;"
                f" font-size: 14px; }}")
        else:
            self._flatten_echo.setText("")
        if pending and not self._alert_flatten_pending:
            self._push_alert('flatten', "Flatten requested")
        self._alert_flatten_pending = pending

    def _refresh_dd_badge(self):
        """Live drawdown badge from the longest cached perf series (1A if present,
        else the longest available). 'DD: -X.X% (peak N d ago)'; gray '—' with no
        cache."""
        cache = getattr(self, '_perf_history_cache', {}) or {}
        data = cache.get('1A')
        if data is None and cache:
            data = max(cache.values(),
                       key=lambda d: len(d.get('equity') or []), default=None)
        eq, ts = [], []
        if data:
            eq_raw = data.get('equity') or []
            ts_raw = data.get('timestamp') or []
            for i, v in enumerate(eq_raw):
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv) and fv > 0:
                    eq.append(fv)
                    ts.append(ts_raw[i] if i < len(ts_raw) else None)
        if len(eq) < 2:
            self._dd_badge.setText("DD: —")
            self._dd_badge.setStyleSheet(
                f"color: {T.get('muted', T['white']).name()};")
            return
        arr = np.asarray(eq, dtype=float)
        hwm = chart_core.compute_hwm(arr)
        peak = float(hwm[-1])
        cur_dd = (arr[-1] / peak - 1.0) * 100.0 if peak > 0 else 0.0
        at_peak = np.nonzero(arr >= peak - 1e-9)[0]
        peak_idx = int(at_peak.max()) if at_peak.size else len(arr) - 1
        age_str = ""
        try:
            if ts[peak_idx] is not None:
                days = (time.time() - float(ts[peak_idx])) / 86400.0
                age_str = f" (peak {days:.0f}d ago)"
        except (TypeError, ValueError, IndexError):
            age_str = ""
        if cur_dd < -0.05:
            self._dd_badge.setText(f"DD: {cur_dd:.1f}%{age_str}")
            self._dd_badge.setStyleSheet(
                f"color: {T['red'].name()}; font-weight: 600;")
        else:
            self._dd_badge.setText("DD: 0.0% (at peak)")
            self._dd_badge.setStyleSheet(
                f"color: {T['green'].name()}; font-weight: 600;")

    def _repaint_today_sparkline(self, data):
        """Repaint the ~60px today sparkline from a '1D' history dict. Fingerprint
        -guarded like _apply_perf_data so an unchanged series is a no-op."""
        self._today_spark_cache = data
        eq, ts = [], []
        if data:
            eq_raw = data.get('equity') or []
            ts_raw = data.get('timestamp') or []
            for i, v in enumerate(eq_raw):
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv) and fv > 0:
                    eq.append(fv)
                    ts.append(float(ts_raw[i]) if i < len(ts_raw) else float(i))
        if len(eq) < 2:
            if self._chart_fp.get('today') != 'empty':
                self._chart_fp['today'] = 'empty'
                self._today_spark_curve.clear()
            return
        fp = f"{len(eq)}:{eq[0]:.2f}:{eq[-1]:.2f}"
        if self._chart_fp.get('today') == fp:
            return
        self._chart_fp['today'] = fp
        pal = getattr(self, '_chart_pal', None) or _chart_palette()
        up = eq[-1] >= eq[0]
        self._today_spark_curve.setData(
            ts, eq, pen=pg.mkPen(pal['up'] if up else pal['down'], width=2))

    def _refresh_cockpit(self):
        """One cockpit refresh (banner + heartbeats + risk gauge + DD badge),
        driven by existing ticks (on_account 10s, on_positions 5s, models 60s) —
        no new timer. Each section is independently guarded so one bad read never
        blanks the rest."""
        for fn in (self._refresh_cockpit_banner, self._refresh_heartbeats,
                   self._refresh_risk_gauge, self._refresh_dd_badge):
            try:
                fn()
            except Exception:
                pass

    # ---- Tab 2: Trading --------------------------------------------------
    def _build_trading_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- Manual Trade controls ---
        trade_group = QGroupBox("Manual Trade")
        trade_layout = QHBoxLayout(trade_group)

        trade_layout.addWidget(QLabel("Symbol:"))
        self._manual_symbol = QLineEdit()
        self._manual_symbol.setPlaceholderText("TSLA or BTC/USD")
        self._manual_symbol.setFixedWidth(120)
        trade_layout.addWidget(self._manual_symbol)

        trade_layout.addWidget(QLabel("Qty:"))
        self._manual_qty = QLineEdit()
        self._manual_qty.setPlaceholderText("1")
        self._manual_qty.setFixedWidth(80)
        trade_layout.addWidget(self._manual_qty)

        trade_layout.addWidget(QLabel("Notional $:"))
        self._manual_notional = QLineEdit()
        self._manual_notional.setPlaceholderText("250")
        self._manual_notional.setFixedWidth(80)
        trade_layout.addWidget(self._manual_notional)

        self._manual_size_btn = QPushButton("Size by policy")
        self._manual_size_btn.setFixedWidth(110)
        self._manual_size_btn.setToolTip(
            "Policy-floor sizing: equity × RISK_PCT / stop-floor")
        self._manual_size_btn.clicked.connect(self._size_by_policy)
        trade_layout.addWidget(self._manual_size_btn)

        self._manual_buy_btn = QPushButton("BUY")
        self._manual_buy_btn.setFixedWidth(60)
        self._manual_buy_btn.clicked.connect(lambda: self._manual_trade("buy"))
        trade_layout.addWidget(self._manual_buy_btn)

        self._manual_sell_btn = QPushButton("SELL")
        self._manual_sell_btn.setFixedWidth(60)
        self._manual_sell_btn.clicked.connect(lambda: self._manual_trade("sell"))
        trade_layout.addWidget(self._manual_sell_btn)
        # BUY=pnl up, SELL=pnl down — one semantic, re-themed in _restyle.
        self._style_trade_buttons()

        self._manual_status = QLabel("")
        self._manual_status.setStyleSheet("font-size: 11px;")
        trade_layout.addWidget(self._manual_status)
        trade_layout.addStretch()
        layout.addWidget(trade_group)

        # --- Filter + orders ---
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self._trade_filter = QComboBox()
        self._trade_filter.addItems(["All", "Crypto", "Stock"])
        self._trade_filter.currentTextChanged.connect(self._apply_trade_filter)
        filter_layout.addWidget(self._trade_filter)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        open_label = QLabel("Open Orders")
        open_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(open_label)

        self._open_orders_table = QTableWidget(0, 7)
        self._open_orders_table.setHorizontalHeaderLabels(
            ["Symbol", "Side", "Qty", "Type", "Status", "Submitted (CT)", ""]
        )
        _oo_hdr = self._open_orders_table.horizontalHeader()
        _oo_hdr.setSectionResizeMode(QHeaderView.Stretch)
        _oo_hdr.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Cancel col
        self._open_orders_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._open_orders_table.setAlternatingRowColors(True)
        # Persistent per-order Cancel buttons keyed by order id (5-A button
        # lifecycle): rebuilt only when the visible row->id order changes.
        self._open_order_cancel_btn = {}
        self._open_order_btn_order = None
        layout.addWidget(self._open_orders_table)

        fills_label = QLabel("Recent Fills")
        fills_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 8px;")
        layout.addWidget(fills_label)

        self._fills_table = QTableWidget(0, 6)
        self._fills_table.setHorizontalHeaderLabels(
            ["Symbol", "Side", "Qty", "Filled Price", "Notional", "Filled At (CT)"]
        )
        self._fills_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._fills_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._fills_table.setAlternatingRowColors(True)
        layout.addWidget(self._fills_table)

        # --- Gate attribution + journal analytics (gui_review_2026-07 §4,
        # Phase 2.5/2.6). A vertical splitter so the user can trade off space
        # between the counterfactual-gate story and the closed-trade stats. ---
        analytics_split = QSplitter(Qt.Vertical)

        gate_group = QGroupBox("Decision gates (last run)")
        gate_v = QVBoxLayout(gate_group)
        self._gate_attr_label = QLabel("—")
        self._gate_attr_label.setWordWrap(True)
        self._gate_attr_label.setTextFormat(Qt.RichText)
        self._gate_attr_label.setAlignment(Qt.AlignTop)
        gate_v.addWidget(self._gate_attr_label)
        analytics_split.addWidget(gate_group)

        journal_group = QGroupBox("Journal analytics (7d / 30d)")
        journal_v = QVBoxLayout(journal_group)
        cards_row = QHBoxLayout()
        self._jstat_winrate = make_card("Win rate")
        self._jstat_expectancy = make_card("Expectancy")
        self._jstat_pf = make_card("Profit factor")
        for c in (self._jstat_winrate, self._jstat_expectancy, self._jstat_pf):
            cards_row.addWidget(c)
        cards_row.addStretch()
        journal_v.addLayout(cards_row)

        jr_ctrl = QHBoxLayout()
        self._journal_refresh_btn = QPushButton("Refresh")
        self._journal_refresh_btn.setFixedHeight(26)
        self._journal_refresh_btn.setCursor(Qt.PointingHandCursor)
        self._journal_refresh_btn.clicked.connect(
            lambda: self._refresh_journal_analytics(force=True))
        jr_ctrl.addWidget(self._journal_refresh_btn)
        self._journal_status = QLabel("")
        self._journal_status.setStyleSheet("font-size: 11px;")
        jr_ctrl.addWidget(self._journal_status)
        jr_ctrl.addStretch()
        journal_v.addLayout(jr_ctrl)

        self._journal_view = QPlainTextEdit()
        self._journal_view.setReadOnly(True)
        self._journal_view.setFont(QFont("Monospace", 10))
        journal_v.addWidget(self._journal_view)
        analytics_split.addWidget(journal_group)

        layout.addWidget(analytics_split)

        # Worker result -> UI render; first paint deferred to the first
        # Trading-tab show (journals can be large — never on the build path).
        self._journal_stats_ready.connect(self._on_journal_stats_ready)
        self._journal_loaded = False
        self._journal_busy = False

        self.tabs.addTab(tab, "Trading")
        self._trading_tab_index = self.tabs.indexOf(tab)

    def _refresh_gate_attribution(self):
        """Render decision_report.json (gates + conviction + admitted_k) into
        the Trading-tab gate box. Producer: decision_report.run_report — the
        'Decision Report' button (Models tab) regenerates it; this only reads.
        Verdict-first (c26 U1): color encodes the writer's CI-based `verdict`
        (REVIEW=red, OK=green, insufficient/cannot-conclude=muted), never the
        raw counterfactual-mean sign; also adopts `insufficient_n`,
        `quality.representative`, and `stale_reason`. Render model:
        chart_core.gate_panel_model. Schema (decision_report.py run_report /
        gate_attribution): gates[gate] = {vetoes_priced, vetoes_raw,
        counterfactual_mean_net_pct, counterfactual_hit_rate, saved_total_pct,
        ci90, verdict, insufficient_n}; signal_exit; quality; conviction
        buckets carry mean_net_pct/n; admitted_k[asset] = {mean_admitted_k,
        pct_windows_k_ge_6, pct_windows_zero, windows}."""
        lbl = getattr(self, '_gate_attr_label', None)
        if lbl is None:
            return
        muted = T.get('muted', T['white']).name()
        try:
            with open(BASE_DIR / 'decision_report.json') as f:
                rep = json.load(f)
        except (OSError, json.JSONDecodeError):
            lbl.setText(f"<span style='color:{muted}'>no decision_report.json — "
                        f"run Decision Report (Models tab) to generate</span>")
            return
        if not isinstance(rep, dict):
            lbl.setText(f"<span style='color:{muted}'>decision_report.json "
                        f"unreadable</span>")
            return
        parts = []
        m = chart_core.gate_panel_model(rep)
        if m['stale']:
            parts.append(
                f"<span style='color:{T['yellow'].name()}'>⚠ STALE report — "
                f"{m['stale_reason']}</span>")
        if m['quality_line']:
            rep_ok = m['representative']
            qcol = muted if rep_ok in (True, None) else T['yellow'].name()
            tag = '' if rep_ok in (True, None) else ' · NOT representative'
            parts.append(
                f"<span style='color:{qcol}'>{m['quality_line']}{tag}</span>")
        # Color encodes the CI-based verdict, never the raw mean sign:
        # REVIEW/CHANGE (charging admission) = red, OK/NO CHANGE = green,
        # insufficient-n / cannot-conclude = muted.
        _VCLASS_COLOR = {'review': T['red'], 'ok': T['green'],
                         'change': T['red'], 'no_change': T['green']}
        gate_lines = []
        for g in m['gates'][:12]:
            col = _VCLASS_COLOR.get(g['verdict_class'],
                                    T.get('muted', T['white'])).name()
            seg = (f"<b>{g['name']}</b>: "
                   f"<span style='color:{col}'>{g['verdict']}</span>"
                   f" · n {g['n']}")
            if isinstance(g['mean'], (int, float)):
                lo, hi = (g['ci90'] or [None, None])[:2]
                seg += f" · cf {g['mean']:+.2f}%"
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    seg += f" [{lo:+.2f},{hi:+.2f}]"
            if isinstance(g['saved'], (int, float)):
                seg += f" · saved {g['saved']:+.1f}%"
            gate_lines.append(seg)
        if gate_lines:
            parts.append("Gate attribution (counterfactual net / veto):")
            parts.extend("&nbsp;&nbsp;" + s for s in gate_lines)
        else:
            parts.append(f"<span style='color:{muted}'>no priced gate "
                         f"vetoes in window</span>")
        if m['signal_exit']:
            s_col = _VCLASS_COLOR.get(m['signal_exit']['verdict_class'],
                                      T.get('muted', T['white'])).name()
            parts.append(
                f"signal-exit: <span style='color:{s_col}'>"
                f"{m['signal_exit']['verdict']}</span> "
                f"(priced {m['signal_exit']['priced']})")
        conv = rep.get('conviction') or {}
        if isinstance(conv, dict):
            terc = [f"{k} {v['mean_net_pct']:+.2f}% (n{v.get('n', '?')})"
                    for k, v in conv.items()
                    if isinstance(v, dict) and 'mean_net_pct' in v
                    and str(k).startswith('pred_')]
            if terc:
                parts.append("Conviction (pred terciles): " + " · ".join(terc))
            elif conv.get('note'):
                parts.append(f"Conviction: {conv['note']}")
        ak = rep.get('admitted_k') or {}
        if isinstance(ak, dict):
            for asset in ('crypto', 'stock'):
                a = ak.get(asset)
                if isinstance(a, dict):
                    parts.append(
                        f"admitted-k [{asset}]: mean {a.get('mean_admitted_k', '?')} · "
                        f"P(k≥6) {(a.get('pct_windows_k_ge_6') or 0) * 100:.0f}% · "
                        f"P(k=0) {(a.get('pct_windows_zero') or 0) * 100:.0f}% · "
                        f"{a.get('windows', '?')} windows")
        gen = rep.get('generated')
        try:
            ago = _ago(dt.datetime.fromisoformat(gen).timestamp())
        except (TypeError, ValueError):
            ago = str(gen) if gen else '?'
        parts.append(f"<span style='color:{muted}'>generated {ago} — run "
                     f"Decision Report to refresh</span>")
        lbl.setText("<br>".join(parts))

    def _refresh_journal_analytics(self, force=False):
        """Load closed-trade stats off the UI thread (journals can be large)
        and render via _on_journal_stats_ready. First paint is lazy (first
        Trading-tab show); Refresh forces a reload. Matches the file's
        threading.Thread+Signal idiom (_on_test_llm) — journal_stats is pure
        stdlib, so a daemon worker emitting a queued signal is safe."""
        if getattr(self, '_journal_busy', False):
            return
        if getattr(self, '_journal_loaded', False) and not force:
            return
        self._journal_busy = True
        self._journal_status.setText("Loading…")
        self._journal_status.setStyleSheet(
            f"color: {T['accent'].name()}; font-size: 11px;")

        import threading

        def worker():
            stats = None
            sizing = None
            try:
                since = time.time() - 30 * 86400
                trades = journal_stats.load_trades(
                    str(JOURNAL_DIR), since_ts=since)
                stats = journal_stats.compute_stats(trades)
                try:
                    sizing = chart_core.sizing_stack_summary(
                        str(JOURNAL_DIR), since)
                except Exception:
                    sizing = None
            except Exception:
                stats = None
            try:
                self._journal_stats_ready.emit({'stats': stats,
                                                'sizing': sizing})
            except RuntimeError:
                pass  # window closed mid-load

        threading.Thread(target=worker, daemon=True,
                         name="journal-stats").start()

    @Slot(object)
    def _on_journal_stats_ready(self, stats):
        """Render compute_stats() output: three overall cards + the full
        format_summary() text. Runs on the UI thread (queued signal)."""
        self._journal_busy = False
        self._journal_loaded = True
        # Payload is {'stats':…, 'sizing':…} since c26 U1; stay tolerant of a
        # bare stats dict (older emitters).
        payload = (stats if isinstance(stats, dict) and 'stats' in stats
                   and 'sizing' in stats else {'stats': stats, 'sizing': None})
        stats = payload['stats']
        muted = T.get('muted', T['white'])
        cards = (self._jstat_winrate, self._jstat_expectancy, self._jstat_pf)
        overall = (stats or {}).get('overall') or {}
        n = overall.get('n_trades', 0) if isinstance(stats, dict) else 0
        if not stats or not n:
            for c in cards:
                self._set_card(c, "—")
            self._journal_view.setPlainText("no closed trades")
            self._journal_status.setText("no closed trades")
            self._journal_status.setStyleSheet(
                f"color: {muted.name()}; font-size: 11px;")
            return
        try:
            wr = overall.get('win_rate')
            exp = overall.get('expectancy_pct')
            pf = overall.get('profit_factor')
            self._set_card(
                self._jstat_winrate,
                f"{wr * 100:.1f}%" if wr is not None else "—",
                T['green'] if (wr is not None and wr >= 0.5) else muted)
            self._set_card(
                self._jstat_expectancy,
                f"{exp:+.2f}%" if exp is not None else "—",
                pnl_color(exp) if exp is not None else muted)
            self._set_card(
                self._jstat_pf,
                f"{pf:.2f}" if pf is not None else "—",
                (T['green'] if pf >= 1.0 else T['red']) if pf is not None
                else muted)
            text = journal_stats.format_summary(stats)
            if payload['sizing'] is not None:
                text += "\n\n" + chart_core.format_sizing_stack(payload['sizing'])
            self._journal_view.setPlainText(text)
            self._journal_status.setText(f"{n} closed trades (30d)")
            self._journal_status.setStyleSheet(
                f"color: {muted.name()}; font-size: 11px;")
        except Exception:
            self._journal_view.setPlainText("failed to render journal stats")
            self._journal_status.setText("render error")
            self._journal_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")

    def _manual_trade_error(self, msg):
        """Show a red error in the manual-trade status label and abort."""
        self._manual_status.setText(msg)
        self._manual_status.setStyleSheet(
            f"color: {T['red'].name()}; font-size: 11px;")

    def _position_row(self, symbol):
        """The cached position dict for `symbol` (slash-insensitive match), or
        None. Reads self._positions_cache (populated by on_positions)."""
        target = str(symbol).replace('/', '').upper()
        try:
            for p in (self._positions_cache or []):
                if str(p.get('symbol', '')).replace('/', '').upper() == target:
                    return p
        except (AttributeError, TypeError):
            pass
        return None

    def _position_qty(self, symbol):
        """Signed qty of the current cached position in `symbol` (0.0 if none)."""
        p = self._position_row(symbol)
        try:
            return float(p.get('qty')) if p is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _latest_price(self, symbol):
        """Best-effort latest price for `symbol` from the stocks-snapshot cache
        or the positions cache. Returns a positive float or None."""
        target = str(symbol).replace('/', '').upper()
        try:
            snaps = (getattr(self, '_stock_data_cache', {}) or {}).get(
                'snapshots', {}) or {}
            for k, snap in snaps.items():
                if str(k).replace('/', '').upper() == target:
                    px = float((snap or {}).get('price'))
                    if math.isfinite(px) and px > 0:
                        return px
        except (TypeError, ValueError, AttributeError):
            pass
        p = self._position_row(symbol)
        if p is not None:
            try:
                px = float(p.get('current_price'))
                if math.isfinite(px) and px > 0:
                    return px
            except (TypeError, ValueError):
                pass
        return None

    def _size_by_policy(self):
        """Fill the notional field with policy-floor risk sizing:
        equity × RISK_PCT_PER_TRADE / stop_floor_pct (crypto vs stock chosen by
        the symbol's asset class). The stop FLOOR is the tightest stop the
        policy allows, so this is the LARGEST size the risk budget permits."""
        symbol = self._manual_symbol.text().strip().upper()
        try:
            equity = float((self._account_cache or {}).get('equity'))
        except (TypeError, ValueError):
            equity = None
        if not equity or not math.isfinite(equity) or equity <= 0:
            self._manual_trade_error("No equity data yet — cannot size")
            return
        asset_type = 'crypto' if '/' in symbol else 'stock'
        policy = CRYPTO_POLICY if asset_type == 'crypto' else STOCK_POLICY
        stop_floor = policy.get('stop_floor_pct')
        if not stop_floor or stop_floor <= 0:
            self._manual_trade_error("No stop-floor in policy — cannot size")
            return
        notional = equity * RISK_PCT_PER_TRADE / stop_floor
        self._manual_notional.setText(f"{notional:.2f}")
        self._manual_qty.clear()
        self._manual_status.setText(
            f"Policy size: {fmt_money(notional)} ({asset_type})")
        self._manual_status.setStyleSheet(
            f"color: {T['muted'].name()}; font-size: 11px;")

    def _manual_trade(self, side):
        """Execute a manual buy/sell order via the Alpaca API — validated,
        policy-sized, and behind a confirmation dialog (the GUI review flagged
        the old one-click market order as a P0: it bypassed every sizing/risk
        gate and could open a short on this long-only book)."""
        symbol = self._manual_symbol.text().strip().upper()
        if not symbol:
            self._manual_trade_error("Enter a symbol")
            return

        qty_text = self._manual_qty.text().strip()
        notional_text = self._manual_notional.text().strip()
        if not qty_text and not notional_text:
            self._manual_trade_error("Enter qty or notional")
            return

        # Parse + validate: each supplied field must be a finite float > 0.
        qty = notional = None
        if qty_text:
            try:
                qty = float(qty_text)
            except ValueError:
                qty = None
            if qty is None or not math.isfinite(qty) or qty <= 0:
                self._manual_trade_error("Qty must be a positive number")
                return
        if notional_text:
            try:
                notional = float(notional_text)
            except ValueError:
                notional = None
            if notional is None or not math.isfinite(notional) or notional <= 0:
                self._manual_trade_error("Notional must be a positive number")
                return

        # If BOTH are given, qty wins (Alpaca accepts one or the other).
        qty_wins = qty is not None and notional is not None
        use_qty = qty is not None

        # Reference price for cost/impact estimation (may be None).
        price = self._latest_price(symbol)
        if use_qty:
            est_cost = qty * price if price else None
        else:
            est_cost = notional

        # MIN_ORDER_NOTIONAL floor (skip dust orders that fees would eat).
        floor_ref = est_cost if est_cost is not None else (
            None if use_qty else notional)
        if floor_ref is not None and floor_ref < MIN_ORDER_NOTIONAL:
            self._manual_trade_error(
                f"Below MIN_ORDER_NOTIONAL (${MIN_ORDER_NOTIONAL:.0f})")
            return

        # Account + position context.
        acct = self._account_cache or {}
        try:
            equity = float(acct.get('equity'))
        except (TypeError, ValueError):
            equity = None
        try:
            buying_power = float(acct.get('buying_power'))
        except (TypeError, ValueError):
            buying_power = None
        cur_pos_qty = self._position_qty(symbol)

        # SELL guards (long-only book): a manual sell must never open OR deepen
        # a short. Block no-position, notional-sell (unbounded vs holdings), and
        # over-qty (the remainder would flip short). Close button = full exit.
        if side == 'sell':
            if cur_pos_qty <= 0:
                QMessageBox.warning(
                    self, "No position to sell",
                    f"No open long position in {symbol}.\n\nThis is a long-only "
                    "book — a market SELL here would open a short. Order blocked.")
                self._manual_trade_error("Sell blocked — no position (long-only)")
                return
            if not use_qty:
                # A notional sell can't be bounded against the held qty (price
                # may be stale/unknown) and could silently over-sell into a short.
                QMessageBox.warning(
                    self, "Notional sell not supported",
                    "Notional sells are not supported for manual orders — use "
                    "qty (≤ position) or the Close button. Order blocked.")
                self._manual_trade_error("Sell blocked — use qty or Close")
                return
            # Tolerate float dust (allow qty <= position * 1.0000001).
            if qty > cur_pos_qty * 1.0000001:
                QMessageBox.warning(
                    self, "Sell exceeds position",
                    f"Sell qty {fmt_qty(qty)} exceeds position "
                    f"{fmt_qty(cur_pos_qty)} — reduce qty or use the Close "
                    "button. Order blocked.")
                self._manual_trade_error("Sell blocked — qty exceeds position")
                return

        # Equity sanity cap: hard-block > 50%, extra warning >= 10%.
        warn_lines = []
        if est_cost is not None and equity and equity > 0:
            frac = est_cost / equity
            if frac > 0.50:
                QMessageBox.warning(
                    self, "Order too large",
                    f"Estimated cost {fmt_money(est_cost)} exceeds 50% of "
                    f"equity ({fmt_money(equity)}). Order blocked.")
                self._manual_trade_error("Blocked — cost > 50% of equity")
                return
            if frac >= 0.10:
                warn_lines.append(
                    f"WARNING: this order is {frac * 100:.0f}% of equity.")

        # --- Confirmation dialog (before any submit) ---
        lines = [f"Symbol: {symbol}", f"Side: {side.upper()}"]
        if qty_wins:
            lines.append(f"Qty: {fmt_qty(qty)}  "
                         "(both filled — qty wins, notional ignored)")
        elif use_qty:
            lines.append(f"Qty: {fmt_qty(qty)}")
        else:
            lines.append(f"Notional: {fmt_money(notional)}")
        if price and est_cost is not None:
            lines.append(f"Est. cost: {fmt_money(est_cost)}  (@ {fmt_money(price)})")
        elif not use_qty:
            lines.append(f"Est. cost: {fmt_money(notional)}  (notional)")
        else:
            lines.append("Est. cost: unknown (no recent price)")
        lines.append(f"Current position: {fmt_qty(cur_pos_qty)}")
        if buying_power is not None:
            if est_cost is None:
                after_str = "unknown"
            elif side == 'buy':
                after_str = fmt_money(buying_power - est_cost)
            else:
                after_str = fmt_money(buying_power + est_cost)
            lines.append(f"Buying power: {fmt_money(buying_power)} → {after_str}")
        lines.extend(warn_lines)
        lines.append("")
        lines.append("This is a MARKET order.")

        reply = QMessageBox.question(
            self, f"Confirm {side.upper()} {symbol}", "\n".join(lines),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            self._manual_status.setText("Cancelled")
            self._manual_status.setStyleSheet(
                f"color: {T['muted'].name()}; font-size: 11px;")
            return

        # Disable buttons to prevent double-submit
        self._manual_buy_btn.setEnabled(False)
        self._manual_sell_btn.setEnabled(False)
        self._manual_status.setText("Submitting...")
        self._manual_status.setStyleSheet(f"color: {T['muted'].name()}; font-size: 11px;")
        QApplication.processEvents()

        try:
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'market',
                'time_in_force': 'gtc' if '/' in symbol else 'day',
            }
            if use_qty:
                order_params['qty'] = qty
            else:
                order_params['notional'] = notional

            order = self.api.submit_order(**order_params)
            self._manual_status.setText(
                f"{side.upper()} {symbol} submitted (ID: {str(order.id)[:8]}...)")
            self._manual_status.setStyleSheet(
                f"color: {T['green'].name()}; font-size: 11px;")
            self.statusBar().showMessage(
                f"{side.upper()} {symbol} order submitted", 7000)
            # Trigger a refresh (positions/orders live on the hot fetcher)
            from PySide6.QtCore import QMetaObject
            QMetaObject.invokeMethod(
                self._fetcher_hot, "fetch_positions", Qt.QueuedConnection)
            QMetaObject.invokeMethod(
                self._fetcher_hot, "fetch_orders", Qt.QueuedConnection)
        except Exception as e:
            self._manual_status.setText(f"Error: {e}")
            self._manual_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")
            self.statusBar().showMessage(f"Order error ({symbol}): {e}", 7000)
            self._push_alert('order-error', f"{symbol} order error: {e}")
        finally:
            self._manual_buy_btn.setEnabled(True)
            self._manual_sell_btn.setEnabled(True)

    def _close_position(self, symbol):
        """Close a position at market — full or partial (25% / 50% / 100%).
        The confirm dialog shows the qty + unrealized P&L that will be realized;
        100% (the default) keeps the position-DELETE call so it closes to exactly
        flat, while 25%/50% submit a market SELL of a rounded fraction of the held
        qty through the same submit path."""
        pos = self._position_row(symbol)
        qty = self._position_qty(symbol)
        detail = ""
        if pos is not None:
            if qty:
                detail += f"\nQty: {fmt_qty(qty)}"
            try:
                detail += ("\nUnrealized P&L: "
                           f"{fmt_money(float(pos.get('unrealized_pl')))}")
            except (TypeError, ValueError):
                pass

        # Fraction picker: three Accept-role buttons + Cancel, 100% default.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Close Position")
        box.setText(f"Close {symbol} at market price?{detail}\n\n"
                    "Choose how much to close (MARKET order):")
        b25 = box.addButton("Close 25%", QMessageBox.AcceptRole)
        b50 = box.addButton("Close 50%", QMessageBox.AcceptRole)
        b100 = box.addButton("Close 100%", QMessageBox.AcceptRole)
        cancel = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(b100)
        box.exec()
        clicked = box.clickedButton()
        frac = {b25: 0.25, b50: 0.50, b100: 1.0}.get(clicked)
        if clicked is None or clicked is cancel or frac is None:
            return

        self._manual_status.setText(f"Closing {symbol}...")
        self._manual_status.setStyleSheet(f"color: {T['muted'].name()}; font-size: 11px;")
        QApplication.processEvents()

        try:
            if frac >= 1.0:
                # Full exit: DELETE /positions closes to exactly flat (no rounding
                # residue). Crypto symbols contain '/' (e.g. BTC/USD) which breaks
                # the URL path, so URL-encode the symbol.
                from urllib.parse import quote
                self.api.delete('/positions/{}'.format(quote(symbol, safe='')))
                self._manual_status.setText(f"Close order submitted for {symbol}")
            else:
                # Partial close: market-sell a rounded fraction of the held qty.
                # Asset-appropriate precision — crypto 8dp (tiny fractions),
                # stocks 3dp (Alpaca fractional day orders). frac<1 keeps the sell
                # strictly under the holding, so it can never flip short.
                precision = 8 if '/' in symbol else 3
                sell_qty = round(abs(qty) * frac, precision)
                if not sell_qty or sell_qty <= 0:
                    self._manual_status.setText(
                        f"{int(frac * 100)}% of {fmt_qty(qty)} rounds to 0 — "
                        "use a larger fraction")
                    self._manual_status.setStyleSheet(
                        f"color: {T['yellow'].name()}; font-size: 11px;")
                    return
                self.api.submit_order(
                    symbol=symbol, side='sell', type='market', qty=sell_qty,
                    time_in_force='gtc' if '/' in symbol else 'day')
                self._manual_status.setText(
                    f"Sell {fmt_qty(sell_qty)} {symbol} submitted "
                    f"({int(frac * 100)}%)")
            self._manual_status.setStyleSheet(
                f"color: {T['green'].name()}; font-size: 11px;")
            self.statusBar().showMessage(
                f"Close order submitted for {symbol}", 7000)
            from PySide6.QtCore import QMetaObject
            QMetaObject.invokeMethod(
                self._fetcher_hot, "fetch_positions", Qt.QueuedConnection)
            QMetaObject.invokeMethod(
                self._fetcher_hot, "fetch_orders", Qt.QueuedConnection)
        except Exception as e:
            self._manual_status.setText(f"Close error: {e}")
            self._manual_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")
            self.statusBar().showMessage(f"Close error ({symbol}): {e}", 7000)
            self._push_alert('order-error', f"{symbol} close error: {e}")

    # ---- Tab 3: Performance ----------------------------------------------
    def _build_performance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Zoom buttons row
        zoom_row = QHBoxLayout()
        eq_label = QLabel("Equity Curve")
        eq_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        zoom_row.addWidget(eq_label)
        zoom_row.addStretch()

        self._perf_zoom = "1M"
        self._perf_zoom_buttons = {}
        self._perf_history_cache = {}  # period -> data dict
        for label in ["1A", "6M", "3M", "1M", "1W"]:
            btn = QPushButton(label)
            btn.setFixedWidth(36)
            btn.setCheckable(True)
            btn.setChecked(label == self._perf_zoom)
            btn.clicked.connect(lambda checked, z=label: self._on_perf_zoom_clicked(z))
            zoom_row.addWidget(btn)
            self._perf_zoom_buttons[label] = btn
        layout.addLayout(zoom_row)

        # Benchmark (SPY/BTC) + log-scale + reset-view controls row.
        gset = _load_gui_settings()
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Benchmark:"))
        self._bench_combo = QComboBox()
        self._bench_combo.addItems(["None", "SPY", "BTC"])
        self._bench_combo.setFixedWidth(90)
        _bsel = gset.get('perf_benchmark', 'None')
        if _bsel in ("None", "SPY", "BTC"):
            self._bench_combo.setCurrentText(_bsel)
        self._bench_symbol = {'SPY': 'SPY', 'BTC': 'BTC/USD'}.get(
            self._bench_combo.currentText())
        self._bench_combo.currentIndexChanged.connect(self._on_benchmark_changed)
        ctrl_row.addWidget(self._bench_combo)
        self._perf_log_check = QCheckBox("Log scale")
        self._perf_log_check.setChecked(bool(gset.get('perf_logscale', False)))
        self._perf_log_check.toggled.connect(self._on_perf_logscale_toggled)
        ctrl_row.addWidget(self._perf_log_check)
        ctrl_row.addStretch()
        self._perf_reset_btn = QPushButton("Reset view")
        self._perf_reset_btn.setFixedWidth(90)
        self._perf_reset_btn.setToolTip("Clear manual pan/zoom and refit the curve")
        self._perf_reset_btn.clicked.connect(self._reset_perf_view)
        ctrl_row.addWidget(self._perf_reset_btn)
        layout.addLayout(ctrl_row)

        eq_axis = pg.DateAxisItem(orientation='bottom')
        self._equity_plot = pg.PlotWidget(axisItems={'bottom': eq_axis})
        self._equity_plot.showGrid(x=True, y=True, alpha=0.3)
        self._equity_plot.setLabel("left", "Equity ($)")
        self._equity_legend = self._equity_plot.addLegend(offset=(8, 8))
        self._equity_hwm = self._equity_plot.plot(pen=pg.mkPen(width=1), name='High-water')
        self._equity_curve = self._equity_plot.plot(pen=pg.mkPen(width=2), name='Equity')
        self._equity_dd_fill = pg.FillBetweenItem(
            self._equity_hwm, self._equity_curve, brush=pg.mkBrush(0, 0, 0, 0))
        self._equity_plot.addItem(self._equity_dd_fill)
        # Benchmark overlay curve (muted dashed; themed in _restyle). Added to the
        # legend on demand by _apply_benchmark_overlay ("SPY (norm.)").
        self._bench_curve = self._equity_plot.plot(
            [], [], pen=pg.mkPen(width=1, style=Qt.DashLine))
        self._equity_xhair = ChartCrosshair(self._equity_plot.getPlotItem(), _chart_palette())
        # Bounded x pan + cursor-anchored wheel zoom; y stays on auto-range.
        _eqvb = self._equity_plot.getViewBox()
        _eqvb.setMouseEnabled(x=True, y=False)
        _eqvb.setAutoVisible(y=True)
        _eqvb.enableAutoRange(x=False, y=True)
        _eqvb.sigRangeChangedManually.connect(self._on_perf_manual_range)
        layout.addWidget(self._equity_plot, stretch=1)

        pnl_axis = pg.DateAxisItem(orientation='bottom')
        self._pnl_plot = pg.PlotWidget(axisItems={'bottom': pnl_axis})
        self._pnl_plot.showGrid(x=True, y=True, alpha=0.3)
        self._pnl_plot.setLabel("left", "P&L ($)")
        self._pnl_bars_pos = pg.BarGraphItem(x=[], height=[], width=1)
        self._pnl_bars_neg = pg.BarGraphItem(x=[], height=[], width=1)
        self._pnl_plot.addItem(self._pnl_bars_pos)
        self._pnl_plot.addItem(self._pnl_bars_neg)
        self._pnl_plot.setXLink(self._equity_plot.getPlotItem())
        self._pnl_plot.setMouseEnabled(x=False, y=False)
        layout.addWidget(self._pnl_plot, stretch=1)

        stats_group = QGroupBox("Performance Stats")
        stats_layout = QHBoxLayout(stats_group)
        self._stat_return = make_card("Total Return")
        self._stat_best = make_card("Best Day")
        self._stat_worst = make_card("Worst Day")
        self._stat_drawdown = make_card("Max Drawdown")
        for c in [self._stat_return, self._stat_best, self._stat_worst, self._stat_drawdown]:
            stats_layout.addWidget(c)
        layout.addWidget(stats_group)

        # Risk-adjusted tiles (from the extended chart_core.perf_stats).
        stats2_group = QGroupBox("Risk-Adjusted")
        stats2_layout = QHBoxLayout(stats2_group)
        self._stat_sharpe = make_card("Sharpe")
        self._stat_sortino = make_card("Sortino")
        self._stat_winrate = make_card("Win Rate")
        self._stat_vol = make_card("Volatility")
        self._stat_cagr = make_card("CAGR")
        for c in [self._stat_sharpe, self._stat_sortino, self._stat_winrate,
                  self._stat_vol, self._stat_cagr]:
            stats2_layout.addWidget(c)
        layout.addWidget(stats2_group)

        # Tax estimation lives here (moved off the Cockpit landing per §10 —
        # tax on a paper account is not mission-control data). Same group box,
        # cards and _update_tax wiring as before, just re-parented.
        tax_group = QGroupBox("Est. Tax — MinTax, indicative (paper)")
        tax_layout = QHBoxLayout(tax_group)
        self._tax_realized = make_card("Realized Gains")
        self._tax_st = make_card("Short-Term Gains")
        self._tax_lt = make_card("Long-Term Gains")
        self._tax_owed = make_card("Est. Tax Owed")
        self._tax_net = make_card("Net After Tax")
        for c in [self._tax_realized, self._tax_st, self._tax_lt, self._tax_owed, self._tax_net]:
            tax_layout.addWidget(c)
        layout.addWidget(tax_group)

        # Apply the persisted log-scale state now that the plot + fill exist.
        self._apply_perf_logscale(self._perf_log_check.isChecked())

        self.tabs.addTab(tab, "Performance")

    def _perf_api_period(self, zoom=None):
        """Map zoom label to Alpaca API period and timeframe."""
        z = zoom or self._perf_zoom
        return {
            "1A": ("1A", "1D"),
            "6M": ("6M", "1D"),
            "3M": ("3M", "1D"),
            "1M": ("1M", "1D"),
            "1W": ("1W", "1D"),
        }.get(z, ("1M", "1D"))

    def _on_perf_zoom_clicked(self, zoom):
        self._perf_zoom = zoom
        # A preset re-selects the window: drop any manual pan/zoom.
        self._perf_user_viewport = False
        t = T
        for z, btn in self._perf_zoom_buttons.items():
            checked = (z == zoom)
            btn.setChecked(checked)
            if checked:
                btn.setStyleSheet(
                    f"background-color: {t['accent'].name()}; color: {t['bg_dark'].name()};"
                    f" font-weight: bold; border-radius: 4px;")
            else:
                btn.setStyleSheet(
                    f"background-color: {t['bg_header'].name()}; color: {t['muted'].name()};"
                    f" border: 1px solid {t['bg_border'].name()}; border-radius: 4px;")

        period, timeframe = self._perf_api_period(zoom)
        cached = self._perf_history_cache.get(period)
        if cached:
            self._apply_perf_data(cached)  # instant paint from snapshot
        if not cached or time.monotonic() - cached.get("_fetched_at", 0.0) > 300:
            self._request_perf_history()

    def _request_perf_history(self):
        """Ask DataFetcher to fetch portfolio history for the current zoom, plus
        a small intraday ("1D","15Min") pull that feeds the cockpit today
        sparkline (period "1D" is never a zoom target, so on_history routes it to
        the sparkline instead of the equity plot)."""
        period, timeframe = self._perf_api_period()
        from PySide6.QtCore import QMetaObject, Q_ARG
        QMetaObject.invokeMethod(
            self._fetcher_slow, "fetch_history", Qt.QueuedConnection,
            Q_ARG(str, period), Q_ARG(str, timeframe),
        )
        QMetaObject.invokeMethod(
            self._fetcher_slow, "fetch_history", Qt.QueuedConnection,
            Q_ARG(str, "1D"), Q_ARG(str, "15Min"),
        )

    # ---- Performance chart interactivity (benchmark / log / pan-zoom) -----
    def _on_perf_manual_range(self, *args):
        """A mouse pan/zoom on the equity plot — preserve the viewport across
        subsequent data refreshes until a preset / Reset-view clears it."""
        self._perf_user_viewport = True

    def _reset_perf_view(self):
        self._perf_user_viewport = False
        self._chart_fp.pop('perf', None)  # force a full repaint that re-snaps x
        cached = self._perf_history_cache.get(self._perf_api_period()[0])
        if cached:
            self._apply_perf_data(cached)

    def _apply_perf_logscale(self, checked):
        """Toggle log-y on the equity plot. The drawdown FillBetween fills in
        linear scene coordinates between two curves whose y is now log10, so it
        smears under a log axis — hide it while log is active (the hwm dashed
        line and the equity curve are strictly positive and transform fine)."""
        self._equity_plot.setLogMode(x=False, y=bool(checked))
        self._equity_dd_fill.setVisible(not checked)

    def _on_perf_logscale_toggled(self, checked):
        self._apply_perf_logscale(checked)
        settings = _load_gui_settings()
        settings['perf_logscale'] = bool(checked)
        _save_gui_settings(settings)

    def _on_benchmark_changed(self, _idx=None):
        sel = self._bench_combo.currentText()
        self._bench_symbol = {'SPY': 'SPY', 'BTC': 'BTC/USD'}.get(sel)
        settings = _load_gui_settings()
        settings['perf_benchmark'] = sel
        _save_gui_settings(settings)
        if self._bench_symbol is None:
            self._bench_chart_data = None
        else:
            self._request_benchmark()
        self._apply_benchmark_overlay()

    def _request_benchmark(self):
        """Pull the benchmark's DAILY bars through the same slow-thread
        fetch_chart path the price chart uses (journal-marker IO stays in the
        fetcher thread). No-op before the fetcher event loop is up."""
        sym = getattr(self, '_bench_symbol', None)
        if not sym:
            return
        fetcher = getattr(self, '_fetcher_slow', None)
        if fetcher is None:
            return
        from PySide6.QtCore import QMetaObject, Q_ARG
        QMetaObject.invokeMethod(
            fetcher, "fetch_chart", Qt.QueuedConnection,
            Q_ARG(str, sym), Q_ARG(str, "daily"))
        self._bench_req_symbol = sym
        self._bench_req_ts = time.monotonic()

    def _maybe_request_benchmark(self):
        """Throttled benchmark (re)fetch driven off equity repaints, so a
        selection restored at startup — before the fetcher thread exists —
        still loads once the perf tab has data."""
        sym = getattr(self, '_bench_symbol', None)
        if not sym:
            return
        if getattr(self, '_fetcher_slow', None) is None:
            return
        now = time.monotonic()
        if self._bench_req_symbol == sym:
            if now - self._bench_req_ts < 60:
                return
            if self._bench_chart_data is not None and now - self._bench_req_ts < 300:
                return
        self._request_benchmark()

    def _apply_benchmark_overlay(self):
        """Re-align the cached benchmark daily series onto the current equity
        view (chart_core.align_benchmark) and manage its legend entry. Cheap;
        called on equity repaint and when the benchmark payload arrives."""
        curve = getattr(self, '_bench_curve', None)
        if curve is None:
            return
        active = getattr(self, '_bench_symbol', None) is not None
        ev = getattr(self, '_last_equity_view', None)
        data = getattr(self, '_bench_chart_data', None)
        if active and ev is not None and data is not None and len(ev.ts):
            aligned = chart_core.align_benchmark(
                ev.ts, data.get('timestamps', []), data.get('closes', []), ev.equity)
            curve.setData(ev.ts, aligned, connect='finite')
        else:
            curve.setData([], [])
        want = f"{self._bench_combo.currentText()} (norm.)" if active else None
        if getattr(self, '_bench_legend_label', None) != want:
            if self._bench_legend_label is not None:
                try:
                    self._equity_legend.removeItem(self._bench_curve)
                except Exception:
                    pass
            if want is not None:
                try:
                    self._equity_legend.addItem(self._bench_curve, want)
                except Exception:
                    pass
            self._bench_legend_label = want

    def _boot_baseline_fetch(self):
        """One-shot full-range (1A/1D) history fetch at startup so on_history's
        period=="1A" branch writes account_baseline.json without the user
        having to open the Performance tab and click the 1A zoom."""
        try:
            from PySide6.QtCore import QMetaObject, Q_ARG
            QMetaObject.invokeMethod(
                self._fetcher_slow, "fetch_history", Qt.QueuedConnection,
                Q_ARG(str, "1A"), Q_ARG(str, "1D"),
            )
            # Also seed the cockpit today sparkline + DD badge at startup so
            # they aren't blank until the first 5-min perf timer fires.
            QMetaObject.invokeMethod(
                self._fetcher_slow, "fetch_history", Qt.QueuedConnection,
                Q_ARG(str, "1D"), Q_ARG(str, "15Min"),
            )
        except Exception:
            pass

    # ---- Tab 4: Models ---------------------------------------------------
    # ---- Tab 4: News ----------------------------------------------------
    def _build_news_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Sentiment indicator rows — grouped by asset class
        s = "font-size: 13px; font-weight: bold;"
        lbl_style = f"font-size: 13px; font-weight: bold; color: {T['muted'].name()};"
        sep = " | "

        # Row 1: Crypto
        crypto_row = QHBoxLayout()
        lbl = QLabel("CRYPTO")
        lbl.setStyleSheet(lbl_style)
        lbl.setFixedWidth(84)
        crypto_row.addWidget(lbl)
        self._news_crypto_fng = QLabel("FnG: —")
        self._news_crypto_fng.setStyleSheet(s)
        crypto_row.addWidget(self._news_crypto_fng)
        crypto_row.addWidget(QLabel(sep))
        self._news_crypto_24h = QLabel("24h: —")
        self._news_crypto_24h.setStyleSheet(s)
        crypto_row.addWidget(self._news_crypto_24h)
        crypto_row.addWidget(QLabel(sep))
        self._news_crypto_7d = QLabel("7d: —")
        self._news_crypto_7d.setStyleSheet(s)
        crypto_row.addWidget(self._news_crypto_7d)
        crypto_row.addStretch()
        layout.addLayout(crypto_row)

        # Row 2: Stocks
        stock_row = QHBoxLayout()
        lbl = QLabel("STOCKS")
        lbl.setStyleSheet(lbl_style)
        lbl.setFixedWidth(84)
        stock_row.addWidget(lbl)
        self._news_stock_fng = QLabel("FnG: —")
        self._news_stock_fng.setStyleSheet(s)
        stock_row.addWidget(self._news_stock_fng)
        stock_row.addWidget(QLabel(sep))
        self._news_vix = QLabel("VIX: —")
        self._news_vix.setStyleSheet(s)
        stock_row.addWidget(self._news_vix)
        stock_row.addWidget(QLabel(sep))
        self._news_stock_24h = QLabel("24h: —")
        self._news_stock_24h.setStyleSheet(s)
        stock_row.addWidget(self._news_stock_24h)
        stock_row.addWidget(QLabel(sep))
        self._news_stock_7d = QLabel("7d: —")
        self._news_stock_7d.setStyleSheet(s)
        stock_row.addWidget(self._news_stock_7d)
        stock_row.addStretch()
        layout.addLayout(stock_row)

        # Row 3: Combined + refresh timestamp. "Tracked avg" (not "ALL") because
        # the number is an unweighted average across every scored headline, not a
        # universe-weighted figure — the tooltip spells out the composition.
        combined_row = QHBoxLayout()
        lbl = QLabel("Tracked avg")
        lbl.setStyleSheet(lbl_style)
        lbl.setFixedWidth(84)
        lbl.setToolTip("Unweighted mean sentiment across all scored headlines "
                       "(company + crypto + macro), not universe-weighted.")
        combined_row.addWidget(lbl)
        self._news_sent_24h = QLabel("24h: —")
        self._news_sent_24h.setStyleSheet(s)
        combined_row.addWidget(self._news_sent_24h)
        combined_row.addWidget(QLabel(sep))
        self._news_sent_7d = QLabel("7d: —")
        self._news_sent_7d.setStyleSheet(s)
        combined_row.addWidget(self._news_sent_7d)
        combined_row.addStretch()
        self._news_refresh_label = QLabel("")
        self._news_refresh_label.setStyleSheet("font-size: 11px;")
        combined_row.addWidget(self._news_refresh_label)
        layout.addLayout(combined_row)

        # Filter combo
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        filter_layout.addWidget(filter_label)
        self._news_filter_combo = QComboBox()
        self._news_filter_combo.addItems(["My Universe", "All News", "Global / Macro", "Crypto", "Stocks"])
        self._news_filter_combo.setCurrentIndex(0)
        self._news_filter_combo.currentIndexChanged.connect(self._apply_news_filter)
        self._news_filter_combo.setFixedWidth(180)
        filter_layout.addWidget(self._news_filter_combo)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Initialize article cache
        self._news_articles = []
        self._news_fng = None

        # News table — Sym column (from the article's tracked _symbol, blank for
        # macro) + numerically-sortable Sentiment. Sorting is enabled, so the row
        # click can't index _news_filtered by position; the URL rides on col 0's
        # user-role instead (NEWS_URL_ROLE) and survives any re-sort.
        self._news_table = QTableWidget(0, 6)
        self._news_table.setHorizontalHeaderLabels(
            ["Time", "Source", "Category", "Sym", "Headline", "Sentiment"]
        )
        header = self._news_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self._news_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._news_table.setAlternatingRowColors(True)
        self._news_table.setWordWrap(True)
        self._news_table.setSortingEnabled(True)
        # Default view: newest first (Time desc). Time cells carry the article
        # epoch as their numeric sort key, so this stays chronological and each
        # rebuild re-applies whatever column the user last sorted by.
        self._news_table.horizontalHeader().setSortIndicator(0, Qt.DescendingOrder)
        self._news_table.verticalHeader().setDefaultSectionSize(32)
        self._news_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._news_table.setCursor(Qt.PointingHandCursor)
        self._news_table.cellClicked.connect(self._on_news_row_clicked)
        self._news_filtered = []  # track filtered articles for click lookup
        layout.addWidget(self._news_table)

        self.tabs.addTab(tab, "News")
        self._news_tab_index = self.tabs.indexOf(tab)

    # ---- Tab 5: Markets --------------------------------------------------
    def _build_stocks_tab(self):
        try:
            from stock_config import load_stock_universe, save_stock_universe
        except Exception:
            load_stock_universe = lambda: []
            save_stock_universe = lambda s: None
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        # --- Top bar: symbol/timeframe selectors + universe management ---
        top_layout = QHBoxLayout()

        top_layout.addWidget(QLabel("Symbol:"))
        self._stock_symbol_combo = QComboBox()
        self._stock_symbol_combo.setMinimumWidth(100)
        symbols = load_stock_universe()
        self._stock_symbol_combo.addItems(symbols)
        self._stock_symbol_combo.currentTextChanged.connect(self._on_stock_symbol_changed)
        top_layout.addWidget(self._stock_symbol_combo)

        # Zoom buttons — these just change the view range, no API call. The
        # default zoom + grid come from Settings "Chart defaults" (gui_settings),
        # applied here at build.
        _chart_defaults = _load_gui_settings()
        _dz = _chart_defaults.get('chart_default_zoom', '1M')
        self._stock_zoom = _dz if _dz in ("1Y", "3M", "1M", "1W", "1D") else "1M"
        self._stock_zoom_buttons = {}
        for label in ["1Y", "3M", "1M", "1W", "1D"]:
            btn = QPushButton(label)
            btn.setFixedWidth(36)
            btn.setCheckable(True)
            btn.setChecked(label == self._stock_zoom)
            btn.clicked.connect(lambda checked, z=label: self._on_zoom_clicked(z))
            top_layout.addWidget(btn)
            self._stock_zoom_buttons[label] = btn

        top_layout.addStretch()

        self._stock_universe_label = QLabel(f"Universe ({len(symbols)})")
        self._stock_universe_label.setStyleSheet("font-weight: bold;")
        top_layout.addWidget(self._stock_universe_label)

        self._stock_add_input = QLineEdit()
        self._stock_add_input.setPlaceholderText("AAPL or BTC/USD")
        self._stock_add_input.setFixedWidth(110)
        self._stock_add_input.returnPressed.connect(self._on_stock_add)
        top_layout.addWidget(self._stock_add_input)

        add_btn = QPushButton("+")
        add_btn.setFixedWidth(30)
        add_btn.clicked.connect(self._on_stock_add)
        top_layout.addWidget(add_btn)

        remove_btn = QPushButton("\u2212")  # minus sign
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove selected symbol from universe")
        remove_btn.clicked.connect(self._on_stock_remove)
        top_layout.addWidget(remove_btn)

        # Watchlist add/remove feedback (validation rejects, duplicates, adds).
        self._stock_add_status = QLabel("")
        self._stock_add_status.setStyleSheet("font-size: 11px;")
        top_layout.addWidget(self._stock_add_status)

        main_layout.addLayout(top_layout)

        # --- Overlay toggles + Reset-view row (above the price chart) ---
        gset = _load_gui_settings()
        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("Overlays:"))
        self._ov_sma20 = QCheckBox("SMA20")
        self._ov_sma20.setChecked(bool(gset.get('ov_sma20', False)))
        self._ov_sma50 = QCheckBox("SMA50")
        self._ov_sma50.setChecked(bool(gset.get('ov_sma50', False)))
        self._ov_atr = QCheckBox("ATR band")
        self._ov_atr.setChecked(bool(gset.get('ov_atr', False)))
        for cb in (self._ov_sma20, self._ov_sma50, self._ov_atr):
            cb.toggled.connect(self._on_overlay_toggled)
            overlay_row.addWidget(cb)
        overlay_row.addStretch()
        self._chart_reset_btn = QPushButton("Reset view")
        self._chart_reset_btn.setFixedWidth(90)
        self._chart_reset_btn.setToolTip("Clear manual pan/zoom and refit the window")
        self._chart_reset_btn.clicked.connect(self._reset_chart_view)
        overlay_row.addWidget(self._chart_reset_btn)
        main_layout.addLayout(overlay_row)

        # --- Middle: chart (left) + heatmap (right) via splitter ---
        splitter = QSplitter(Qt.Horizontal)

        # Price chart with date axis (GraphicsLayoutWidget: price + volume panes)
        self._stock_chart_widget = pg.GraphicsLayoutWidget()
        self._stock_chart = self._stock_chart_widget.addPlot(
            row=0, col=0, axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
        self._stock_chart.getAxis('bottom').setStyle(showValues=False)
        _grid_on = bool(_chart_defaults.get('chart_grid', True))
        self._stock_chart.showGrid(x=_grid_on, y=_grid_on, alpha=0.3)
        self._stock_chart.setLabel("left", "Price ($)")
        # Bounded x pan + cursor-anchored wheel zoom; y stays on auto-range.
        _pcvb = self._stock_chart.getViewBox()
        _pcvb.setMouseEnabled(x=True, y=False)
        _pcvb.setAutoVisible(y=True)
        _pcvb.enableAutoRange(x=False, y=True)
        _pcvb.sigRangeChangedManually.connect(self._on_chart_manual_range)
        self._stock_chart_line = self._stock_chart.plot(pen=pg.mkPen(width=2))
        self._candle_item = CandlestickItem()
        self._stock_chart.addItem(self._candle_item)
        chart_pal = _chart_palette()
        # Indicator overlays (thin, palette-derived, NaN warmup masked via
        # connect='finite'; data comes from build_price_view(overlays=...)).
        self._sma20_line = self._stock_chart.plot(
            [], [], pen=pg.mkPen(chart_pal['crosshair'], width=1))
        self._sma50_line = self._stock_chart.plot(
            [], [], pen=pg.mkPen(chart_pal['equity'], width=1))
        self._atr_upper_line = self._stock_chart.plot(
            [], [], pen=pg.mkPen(chart_pal['hwm'], width=1, style=Qt.DashLine))
        self._atr_lower_line = self._stock_chart.plot(
            [], [], pen=pg.mkPen(chart_pal['hwm'], width=1, style=Qt.DashLine))
        _atr_fc = QColor(chart_pal['hwm']); _atr_fc.setAlpha(30)
        self._atr_fill = pg.FillBetweenItem(
            self._atr_upper_line, self._atr_lower_line, brush=pg.mkBrush(_atr_fc))
        self._atr_fill.setZValue(-10)  # behind the candles
        self._stock_chart.addItem(self._atr_fill)
        # Surface-colored ring separates markers from candles beneath them;
        # re-themed in _restyle alongside every other chart pen. The exit
        # scatter is hoverable — per-point fill by win/loss, tooltip via tip().
        self._entry_scatter = pg.ScatterPlotItem(
            symbol='t1', size=11, pen=pg.mkPen(chart_pal['bg'], width=2))
        self._exit_scatter = pg.ScatterPlotItem(
            symbol='t', size=11, pen=pg.mkPen(chart_pal['bg'], width=2),
            hoverable=True, tip=self._exit_tip)
        self._stock_chart.addItem(self._entry_scatter)
        self._stock_chart.addItem(self._exit_scatter)
        # Open-position guide lines (entry / est-stop / est-TP) + last price;
        # InfiniteLine labels ride the right edge and follow pan. Themed/valued
        # in _restyle / _update_position_lines / _apply_chart_zoom.
        def _mk_pos_line(color, dashed):
            ln = pg.InfiniteLine(
                angle=0, movable=False,
                pen=pg.mkPen(color, width=1,
                             style=Qt.DashLine if dashed else Qt.SolidLine),
                label='', labelOpts={'position': 0.97, 'anchor': (1, 0.5),
                                     'color': color, 'fill': pg.mkBrush(0, 0, 0, 150)})
            ln.hide()
            self._stock_chart.addItem(ln)
            return ln
        self._pos_entry_line = _mk_pos_line(chart_pal['hwm'], False)
        self._pos_stop_line = _mk_pos_line(chart_pal['down'], True)
        self._pos_tp_line = _mk_pos_line(chart_pal['up'], True)
        self._last_price_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(chart_pal['fg'], width=1, style=Qt.DashLine),
            label='', labelOpts={'position': 0.98, 'anchor': (1, 0.5),
                                 'color': chart_pal['fg'], 'fill': pg.mkBrush(0, 0, 0, 150)})
        self._last_price_line.hide()
        self._stock_chart.addItem(self._last_price_line)
        leg = self._stock_chart.addLegend(offset=(8, 8))
        leg.addItem(self._entry_scatter, 'Entry')
        leg.addItem(self._exit_scatter, 'Exit')
        self._stock_xhair = ChartCrosshair(self._stock_chart, chart_pal)

        self._stock_vol_plot = self._stock_chart_widget.addPlot(
            row=1, col=0, axisItems={'left': SIAxisItem(orientation='left'),
                                     'bottom': pg.DateAxisItem(orientation='bottom')})
        self._stock_vol_plot.setXLink(self._stock_chart)
        self._stock_vol_plot.setLabel("left", "Vol")
        self._stock_vol_plot.setMouseEnabled(x=False, y=False)
        self._vol_bars = pg.BarGraphItem(x=[], height=[], width=1)
        self._stock_vol_plot.addItem(self._vol_bars)

        self._stock_chart_widget.ci.layout.setRowStretchFactor(0, 3)
        self._stock_chart_widget.ci.layout.setRowStretchFactor(1, 1)

        self._stock_chart_data = {}  # resolution -> fetch_chart payload
        splitter.addWidget(self._stock_chart_widget)

        # Heatmap panel — asset-class-grouped tiles + a metric combo. Sector
        # grouping isn't possible: sector data lives ONLY in fundamentals.py's
        # in-memory yfinance cache (a live per-symbol network call, unfit for the
        # render path) and is persisted to no prediction/LLM file the GUI reads —
        # so the honest grouping is by asset class (Crypto / Stocks).
        heatmap_container = QWidget()
        heatmap_outer = QVBoxLayout(heatmap_container)
        heatmap_outer.setContentsMargins(4, 0, 4, 0)
        hm_head = QHBoxLayout()
        hm_title = QLabel("Heatmap")
        hm_title.setStyleSheet("font-size: 13px; font-weight: bold;")
        hm_head.addWidget(hm_title)
        hm_head.addStretch()
        self._heatmap_metric_combo = QComboBox()
        for _hm_label, _hm_key in (("Day %", "day"), ("Pred", "pred"),
                                   ("Meta-p", "metap")):
            self._heatmap_metric_combo.addItem(_hm_label, _hm_key)
        self._heatmap_metric_combo.setFixedWidth(92)
        self._heatmap_metric_combo.setToolTip(
            "Tile color: Day % (change), model Pred, or Meta-p")
        self._heatmap_metric_combo.currentIndexChanged.connect(
            self._on_heatmap_metric_changed)
        hm_head.addWidget(self._heatmap_metric_combo)
        heatmap_outer.addLayout(hm_head)

        self._heatmap_widget = QWidget()
        self._heatmap_layout = QGridLayout(self._heatmap_widget)
        self._heatmap_layout.setSpacing(3)
        self._heatmap_layout.setContentsMargins(0, 0, 0, 0)
        self._heatmap_labels = {}    # sym -> tile QLabel (lazy create / prune)
        self._heatmap_headers = {}   # group name -> header QLabel (created once)
        for _hm_group in ("Crypto", "Stocks"):
            _h = QLabel(_hm_group)
            _h.setStyleSheet(
                f"color: {T['muted'].name()}; font-size: 10px; font-weight: bold;")
            self._heatmap_headers[_hm_group] = _h
        self._heatmap_layout_sig = None  # last group->symbols layout signature
        self._heatmap_cols = 7
        heatmap_outer.addWidget(self._heatmap_widget)
        heatmap_outer.addStretch()
        splitter.addWidget(heatmap_container)

        splitter.setSizes([600, 300])
        main_layout.addWidget(splitter, stretch=1)

        # --- Book-wide regime chips (above the stance table) ---
        # `regime` is a book-wide string the loops duplicate onto every symbol
        # of a book; render it once per book instead of wasting a table column.
        regime_row = QHBoxLayout()
        regime_row.setContentsMargins(2, 2, 2, 2)
        self._crypto_regime_chip = QLabel("Crypto regime: —")
        self._stock_regime_chip = QLabel("Stock regime: —")
        regime_row.addWidget(self._crypto_regime_chip)
        regime_row.addWidget(self._stock_regime_chip)
        regime_row.addStretch()
        main_layout.addLayout(regime_row)

        # --- Bottom: metrics table + detail panel ---
        bottom_splitter = QSplitter(Qt.Vertical)

        # 13-column "what does the system think about this name" stance table.
        # Col order is load-bearing (referenced by index in the row-render path):
        # 0 Symbol 1 Price 2 Day% 3 Volume 4 Pred 5 Meta-p 6 Conv 7 Rank
        # 8 Signal 9 Gate 10 Held 11 LLM 12 LLM Age
        self._stock_table = QTableWidget(0, 13)
        self._stock_table.setHorizontalHeaderLabels(
            ["Symbol", "Price", "Day %", "Volume", "Pred", "Meta-p",
             "Conv", "Rank", "Signal", "Gate", "Held", "LLM", "LLM Age"]
        )
        self._stock_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._stock_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._stock_table.setAlternatingRowColors(True)
        self._stock_table.setSortingEnabled(True)
        self._stock_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._stock_table.cellDoubleClicked.connect(self._on_stock_table_dblclick)
        self._stock_table.currentCellChanged.connect(self._on_stock_row_selected)
        bottom_splitter.addWidget(self._stock_table)

        # LLM detail panel — shows full reasoning for selected symbol
        detail_frame = QFrame()
        detail_layout = QVBoxLayout(detail_frame)
        detail_layout.setContentsMargins(4, 4, 4, 4)

        detail_header = QHBoxLayout()
        self._llm_detail_symbol = QLabel("Select a symbol to see LLM analysis")
        self._llm_detail_symbol.setStyleSheet(
            "font-size: 14px; font-weight: bold;")
        detail_header.addWidget(self._llm_detail_symbol)
        detail_header.addStretch()
        self._llm_refresh_one_btn = QPushButton("Refresh Selected")
        self._llm_refresh_one_btn.setFixedHeight(26)
        self._llm_refresh_one_btn.setCursor(Qt.PointingHandCursor)
        self._llm_refresh_one_btn.clicked.connect(self._refresh_one_llm_clicked)
        detail_header.addWidget(self._llm_refresh_one_btn)
        self._llm_refresh_btn = QPushButton("Refresh All")
        self._llm_refresh_btn.setFixedHeight(26)
        self._llm_refresh_btn.setCursor(Qt.PointingHandCursor)
        self._llm_refresh_btn.clicked.connect(self._refresh_all_llm_clicked)
        detail_header.addWidget(self._llm_refresh_btn)
        self._llm_refresh_status = QLabel("")
        self._llm_refresh_status.setStyleSheet("font-size: 11px;")
        detail_header.addWidget(self._llm_refresh_status)
        detail_layout.addLayout(detail_header)

        self._llm_detail_text = QLabel("")
        self._llm_detail_text.setWordWrap(True)
        self._llm_detail_text.setTextFormat(Qt.RichText)
        self._llm_detail_text.setStyleSheet(
            f"font-size: 12px; padding: 6px; border: 1px solid {T['bg_border'].name()};"
            " border-radius: 4px; background: rgba(0,0,0,0.2);")
        self._llm_detail_text.setMinimumHeight(60)
        detail_layout.addWidget(self._llm_detail_text)
        bottom_splitter.addWidget(detail_frame)

        bottom_splitter.setSizes([400, 150])
        main_layout.addWidget(bottom_splitter, stretch=1)

        self._stock_data_cache = {}  # latest data from fetch_stocks
        self._llm_analysis_cache = {}  # latest llm analysis (MERGED, never wiped)
        self._stocks_row_by_sym = {}  # stable sym -> row map for diff updates
        self.tabs.addTab(tab, "Markets")
        self._markets_tab_index = self.tabs.indexOf(tab)

    def _set_add_status(self, msg, color):
        """Colored one-line feedback next to the watchlist add box."""
        lbl = getattr(self, '_stock_add_status', None)
        if lbl is None:
            return
        lbl.setText(msg)
        lbl.setStyleSheet(f"color: {color.name()}; font-size: 11px;")

    def _on_stock_add(self):
        from stock_config import load_stock_universe, save_stock_universe
        text = self._stock_add_input.text().strip().upper()
        if not text:
            return
        # Reject garbage before it flows into an LLM subprocess / snapshot call:
        # a plain ticker (AAPL, BRK.B) or a crypto pair (BTC/USD).
        if not re.match(r'^[A-Z0-9.]{1,10}(/[A-Z]{3,4})?$', text):
            self._set_add_status(f"Invalid symbol: {text}", T['red'])
            return
        symbols = load_stock_universe()
        if text in symbols:
            self._set_add_status(f"{text} already in universe",
                                 T.get('yellow', T['white']))
            return
        symbols.append(text)
        save_stock_universe(symbols)
        symbols = load_stock_universe()  # re-read sorted
        self._stock_symbol_combo.blockSignals(True)
        self._stock_symbol_combo.clear()
        self._stock_symbol_combo.addItems(symbols)
        self._stock_symbol_combo.setCurrentText(text)
        self._stock_symbol_combo.blockSignals(False)
        self._on_stock_symbol_changed(self._stock_symbol_combo.currentText())
        self._stock_universe_label.setText(f"Universe ({len(symbols)})")
        self._set_add_status(f"Added {text}", T['green'])
        self._stock_add_input.clear()

    def _on_stock_remove(self):
        from stock_config import load_stock_universe, save_stock_universe
        current = self._stock_symbol_combo.currentText()
        if not current:
            return
        symbols = load_stock_universe()
        if current in symbols:
            symbols.remove(current)
            save_stock_universe(symbols)
            symbols = load_stock_universe()
            self._stock_symbol_combo.blockSignals(True)
            self._stock_symbol_combo.clear()
            self._stock_symbol_combo.addItems(symbols)
            self._stock_symbol_combo.blockSignals(False)
            self._on_stock_symbol_changed(self._stock_symbol_combo.currentText())
            self._stock_universe_label.setText(f"Universe ({len(symbols)})")

    def _set_active_symbol(self, sym, source):
        """Unified linked-selection router (5.1): one symbol drives the combo,
        the stance-table selection, the LLM detail panel, and the chart. Guarded
        by _syncing_symbol so the widget signals it sets don't recurse back in.
        `source` ('heatmap'|'table'|'dblclick'|'combo'|'palette') gates the
        source-specific extras only (dblclick also prefills the manual ticket).
        Routes through the existing request paths — adds no new fetch."""
        if not sym:
            return
        if getattr(self, '_syncing_symbol', False):
            return
        self._syncing_symbol = True
        try:
            changed = (sym != getattr(self, '_active_symbol', None))
            self._active_symbol = sym

            # 1) Combo shows sym (silent — don't re-enter via currentTextChanged).
            combo = self._stock_symbol_combo
            if combo.currentText() != sym:
                combo.blockSignals(True)
                combo.setCurrentText(sym)
                combo.blockSignals(False)

            # 2) Select + scroll the stance-table row for sym (silent — don't
            #    re-enter via currentCellChanged).
            tbl = getattr(self, '_stock_table', None)
            if tbl is not None:
                for r in range(tbl.rowCount()):
                    it = tbl.item(r, 0)
                    if it and it.text() == sym:
                        tbl.blockSignals(True)
                        tbl.setCurrentCell(r, 0)
                        tbl.blockSignals(False)
                        tbl.scrollToItem(it)
                        break

            # 3) Detail panel renders sym (works even before a table row exists).
            self._render_llm_detail(sym)

            # 4) Chart: only clear + refetch when the symbol actually changed;
            #    an unchanged re-select leaves the current chart/TTL alone.
            if changed:
                self._stock_chart_data = {}
                self._stock_chart_line.clear()
                self._clear_price_items()
                self._chart_fp.pop('price', None)
                self._request_chart()

            # Source-specific extras.
            if source == 'dblclick':
                self._manual_symbol.setText(sym)
        finally:
            self._syncing_symbol = False

    def _on_stock_symbol_changed(self, sym):
        """Combo change -> unified router (clears+refetches chart, selects the
        table row, renders the dossier). Kept as the combo's slot; also called
        directly by add/remove after they set the combo text."""
        self._set_active_symbol(sym, 'combo')

    def _clear_price_items(self):
        """Clear candle/volume/marker items (symbol switch or empty state)."""
        self._candle_item.set_data([], [], [], [], [], [], [], self._chart_pal['up'],
                                    self._chart_pal['down'])
        self._vol_bars.setOpts(x=[], height=[], width=1, brushes=None)
        self._entry_scatter.setData(x=[], y=[])
        self._exit_scatter.setData(x=[], y=[])
        # Overlays + guide lines follow the symbol switch too.
        for ln in (self._sma20_line, self._sma50_line,
                   self._atr_upper_line, self._atr_lower_line):
            ln.setData([], [])
        for il in (self._pos_entry_line, self._pos_stop_line,
                   self._pos_tp_line, self._last_price_line):
            il.hide()

    def _zoom_resolution(self, zoom=None):
        """Map zoom level to the API resolution needed."""
        z = zoom or self._stock_zoom
        if z in ('1Y', '3M', '1M'):
            return 'daily'
        elif z == '1W':
            return 'hourly'
        elif z == '1D':
            return '5min'
        else:
            return '15min'

    def _on_zoom_clicked(self, zoom):
        self._stock_zoom = zoom
        # A preset re-selects the window: drop any manual pan/zoom.
        self._chart_user_viewport = False
        t = T
        for z, btn in self._stock_zoom_buttons.items():
            checked = (z == zoom)
            btn.setChecked(checked)
            if checked:
                btn.setStyleSheet(
                    f"background-color: {t['accent'].name()}; color: {t['bg_dark'].name()};"
                    f" font-weight: bold; border-radius: 4px;")
            else:
                btn.setStyleSheet(
                    f"background-color: {t['bg_header'].name()}; color: {t['muted'].name()};"
                    f" border: 1px solid {t['bg_border'].name()}; border-radius: 4px;")

        # Check if we already have cached data for this resolution
        res = self._zoom_resolution(zoom)
        data = self._stock_chart_data.get(res)
        if data and data.get('closes'):
            self._apply_chart_zoom()
        else:
            self._request_chart()

    def _on_stock_table_dblclick(self, row, _col):
        item = self._stock_table.item(row, 0)
        if item:
            # Router drives combo/table/chart/detail; dblclick's source-specific
            # extra (prefill the manual-ticket symbol) is applied inside it.
            self._set_active_symbol(item.text(), 'dblclick')

    def _on_stock_row_selected(self, row, _col, _prev_row, _prev_col):
        """A user row selection drives the unified linked-selection router
        (combo/chart/detail all follow). Programmatic selection changes — the
        router itself and the 30s stance-table sync — set _syncing_symbol and
        are ignored here so they can't trigger a feedback loop or a spurious
        chart refetch."""
        if getattr(self, '_syncing_symbol', False):
            return
        if row < 0:
            return
        item = self._stock_table.item(row, 0)
        if not item:
            return
        self._set_active_symbol(item.text(), 'table')

    def _render_llm_detail(self, sym):
        """Render the LLM dossier for `sym` into the detail panel. v1 records (no
        advisor-v2 keys) render exactly as before; v2 keys add a stance/echo-gap/
        risks/events block above bull/bear/summary. Symbol-addressed so the
        linked-selection router can drive it from any source (row, combo,
        heatmap, palette)."""
        from html import escape as _esc  # module is shadowed by `html_out` below
        if not sym:
            return
        llm = self._llm_analysis_cache.get(sym, {})
        if not llm:
            self._llm_detail_symbol.setText(f"{sym} — No LLM analysis available")
            self._llm_detail_text.setText(
                "<i>No analysis found. Click 'Refresh All LLM Analysis' to generate.</i>")
            return

        score = llm.get('s', llm.get('m', '?'))
        model = llm.get('model', '?')
        ts = llm.get('timestamp', '')
        age_str = ""
        if ts:
            try:
                t = dt.datetime.fromisoformat(ts)
                age_h = (dt.datetime.now(tz=t.tzinfo or None) - t).total_seconds() / 3600
                if age_h < 1:
                    age_str = f"{age_h * 60:.0f}m ago"
                elif age_h < 24:
                    age_str = f"{age_h:.1f}h ago"
                else:
                    age_str = f"{age_h / 24:.1f}d ago"
            except (ValueError, TypeError):
                age_str = ts[:16]

        bull = llm.get('bull', '').strip()
        bear = llm.get('bear', '').strip()
        summary = llm.get('r', '').strip()

        green = T['green'].name()
        red = T['red'].name()
        accent = T['accent'].name()
        muted = T['muted'].name()
        yellow = T.get('yellow', T['white']).name()

        self._llm_detail_symbol.setText(
            f"{sym}  |  Score: {score}  |  {model}  |  {age_str}")

        # --- advisor-v2 dossier (each sub-block silently absent for v1) -------
        p_up = llm.get('p_up')
        conv = llm.get('conviction')
        abstain = llm.get('abstain')
        key_risks = llm.get('key_risks') or []
        event_flags = llm.get('event_flags') or []

        v2 = ""
        stance_bits = []
        if p_up is not None:
            try:
                stance_bits.append(f"LLM p_up: <b>{float(p_up):.2f}</b>")
            except (TypeError, ValueError):
                pass
        if conv is not None:
            stance_bits.append(f"conviction {conv}/5")
        if abstain:
            stance_bits.append(f"<b style='color:{red};'>ABSTAIN</b>")
        if stance_bits:
            v2 += (f"<span style='color:{accent};'>"
                   f"{' · '.join(stance_bits)}</span><br>")

        # Echo-gap: model pred sign vs LLM p_up (only when BOTH exist).
        preds = (self._stock_data_cache or {}).get('predictions', {})
        pred_val = (preds.get(sym, {}).get('pred')
                    if isinstance(preds, dict) else None)
        if pred_val is not None and p_up is not None:
            try:
                pv = float(pred_val)
                pu = float(p_up)
                model_bull = pv > 0
                llm_bull = pu >= 0.5
                if model_bull == llm_bull:
                    side = "bullish" if model_bull else "bearish"
                    v2 += (f"<span style='color:{green};'>LLM agrees with model "
                           f"(both {side})</span><br>")
                else:
                    v2 += (f"<span style='color:{yellow};'>LLM fights model "
                           f"(model {'bullish' if model_bull else 'bearish'}, "
                           f"LLM {pu:.2f})</span><br>")
            except (TypeError, ValueError):
                pass

        if key_risks:
            risk_lines = "".join(
                f"&nbsp;&nbsp;• {_esc(str(r))}<br>" for r in key_risks[:5])
            v2 += f"<span style='color:{muted};'>Risks:</span><br>{risk_lines}"

        if event_flags:
            chips = " ".join(f"[{_esc(str(e))}]" for e in event_flags[:5])
            v2 += f"<span style='color:{yellow};'>{chips}</span><br>"

        # --- bull/bear/summary (unchanged; v1 output is byte-identical) -------
        html_out = v2
        if bull:
            html_out += f"<b style='color:{green};'>BULL:</b> {bull}<br>"
        if bear:
            html_out += f"<b style='color:{red};'>BEAR:</b> {bear}<br>"
        if summary:
            html_out += f"<b style='color:{accent};'>Summary:</b> {summary}"
        if not html_out:
            html_out = f"<span style='color:{muted};'>No reasoning available</span>"
        self._llm_detail_text.setText(html_out)

    def _llm_refresh_busy(self):
        """Shared single-flight guard: True while EITHER the one-symbol or the
        whole-universe LLM refresh subprocess is in flight. Two heavy Jetson-
        env interpreters must never run beside the bots on 8 GB."""
        return (getattr(self, '_llm_refresh_one_proc', None) is not None
                or getattr(self, '_llm_refresh_proc', None) is not None)

    def _set_llm_refresh_enabled(self, enabled):
        """Enable/disable BOTH refresh buttons together (single-flight)."""
        for attr in ('_llm_refresh_one_btn', '_llm_refresh_btn'):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(enabled)

    def _refresh_one_llm_clicked(self):
        """Refresh LLM analysis for the currently selected symbol."""
        if self._llm_refresh_busy():
            self._llm_refresh_status.setText("LLM refresh already running...")
            self._llm_refresh_status.setStyleSheet(
                f"color: {T['yellow'].name()}; font-size: 11px;")
            return
        row = self._stock_table.currentRow()
        if row < 0:
            self._llm_refresh_status.setText("No symbol selected")
            self._llm_refresh_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")
            return
        item = self._stock_table.item(row, 0)
        if not item:
            return
        sym = item.text()
        asset_type = 'crypto' if '/' in sym else 'stock'

        self._set_llm_refresh_enabled(False)
        self._llm_refresh_status.setText(f"Analyzing {sym}...")
        self._llm_refresh_status.setStyleSheet(
            f"color: {T['accent'].name()}; font-size: 11px;")
        QApplication.processEvents()

        import subprocess
        python = _engine_python()
        env = _engine_env()
        try:
            # argv passing (NOT an f-string spliced into -c): the symbol is
            # user-typed, so interpolating it into source is a shell/eval
            # injection. sys.argv[1:] delivers it as opaque data instead.
            # Log file, not PIPE: nothing reads the pipe, so a chatty child
            # would fill it and hang forever.
            with open(BASE_DIR / "llm_refresh_one.log", "a") as lf:
                proc = subprocess.Popen(
                    [python, "-u", "-c",
                     "import sys, llm_analyst; "
                     "llm_analyst.refresh_one(sys.argv[1], sys.argv[2])",
                     sym, asset_type],
                    stdout=lf, stderr=subprocess.STDOUT,
                    env=env, cwd=str(BASE_DIR),
                )
            self._llm_refresh_one_proc = proc
            self._llm_refresh_one_sym = sym
            self._llm_refresh_one_start = time.monotonic()
            from PySide6.QtCore import QTimer
            self._llm_refresh_one_timer = QTimer()
            self._llm_refresh_one_timer.timeout.connect(self._check_one_llm_refresh)
            self._llm_refresh_one_timer.start(1000)
        except Exception as e:
            # A crash between Popen and here must not leave the buttons stuck
            self._llm_refresh_status.setText(f"Error: {e}")
            self._llm_refresh_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")
            self._set_llm_refresh_enabled(True)

    def _check_one_llm_refresh(self):
        """Poll single-symbol LLM refresh for completion."""
        if not hasattr(self, '_llm_refresh_one_proc'):
            return
        proc = self._llm_refresh_one_proc
        rc = proc.poll()
        sym = getattr(self, '_llm_refresh_one_sym', '?')
        if rc is None:
            el = int(time.monotonic()
                     - getattr(self, '_llm_refresh_one_start', time.monotonic()))
            self._llm_refresh_status.setText(f"Analyzing {sym}... ({el}s)")
            return
        self._llm_refresh_one_timer.stop()
        del self._llm_refresh_one_proc
        self._set_llm_refresh_enabled(True)
        if rc == 0:
            # Reload analysis from disk and update table + detail panel
            self._reload_llm_from_disk()
            self._update_llm_table_cells(sym)
            self._on_stock_row_selected(self._stock_table.currentRow(), 0, -1, 0)
            self._llm_refresh_status.setText(f"{sym} updated")
            self._llm_refresh_status.setStyleSheet(
                f"color: {T['green'].name()}; font-size: 11px;")
        else:
            self._llm_refresh_status.setText(f"{sym} failed (exit {rc})")
            self._llm_refresh_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")

    def _refresh_all_llm_clicked(self):
        """Trigger LLM analysis for all symbols in the universe.

        refresh_all() has no symbol-list parameter (whole-universe only), so a
        stale-only pass isn't possible here — instead confirm the re-bill first,
        surfacing how many symbols are still fresh (< the 12h the LLM Age column
        treats as fresh) and would be needlessly re-analyzed.
        """
        if self._llm_refresh_busy():
            self._llm_refresh_status.setText("LLM refresh already running...")
            self._llm_refresh_status.setStyleSheet(
                f"color: {T['yellow'].name()}; font-size: 11px;")
            return

        # Count universe + how many analyses are still fresh (< 12h, matching
        # the LLM Age column's green threshold) for the confirm dialog.
        try:
            from stock_config import load_stock_universe, CRYPTO_SYMBOLS
            syms = list(load_stock_universe()) + list(CRYPTO_SYMBOLS)
        except Exception:
            syms = []
        n_total = len(syms)
        fresh = 0
        for s in syms:
            ts = self._llm_analysis_cache.get(s, {}).get('timestamp', '')
            if not ts:
                continue
            try:
                t = dt.datetime.fromisoformat(ts)
                age_h = (dt.datetime.now(tz=t.tzinfo or None)
                         - t).total_seconds() / 3600
                if age_h < 12:
                    fresh += 1
            except (ValueError, TypeError):
                pass
        reply = QMessageBox.question(
            self, "Refresh All LLM Analysis",
            f"Re-analyze {n_total} symbols ({fresh} still fresh — will be "
            f"re-billed). Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        import subprocess
        self._set_llm_refresh_enabled(False)
        self._llm_refresh_status.setText("Analyzing universe...")
        self._llm_refresh_status.setStyleSheet(
            f"color: {T['accent'].name()}; font-size: 11px;")
        QApplication.processEvents()

        python = _engine_python()
        # refresh_all() takes no user input, so the module CLI entry point is
        # already injection-free argv (no f-string) — kept as-is.
        script = str(BASE_DIR / "llm_analyst.py")
        env = _engine_env()
        try:
            # Log file, not PIPE: nothing reads the pipe, so a chatty
            # child would fill it and hang forever
            with open(BASE_DIR / "llm_refresh.log", "a") as lf:
                proc = subprocess.Popen(
                    [python, "-u", script, "--refresh-all"],
                    stdout=lf, stderr=subprocess.STDOUT,
                    env=env, cwd=str(BASE_DIR),
                )
            # Don't block GUI — check completion on timer
            self._llm_refresh_proc = proc
            self._llm_refresh_start = time.monotonic()
            from PySide6.QtCore import QTimer
            self._llm_refresh_timer = QTimer()
            self._llm_refresh_timer.timeout.connect(self._check_llm_refresh)
            self._llm_refresh_timer.start(2000)
        except Exception as e:
            # Re-enable on spawn failure so the buttons never stick
            self._llm_refresh_status.setText(f"Error: {e}")
            self._llm_refresh_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")
            self._set_llm_refresh_enabled(True)

    def _check_llm_refresh(self):
        """Poll the LLM refresh subprocess for completion."""
        if not hasattr(self, '_llm_refresh_proc'):
            return
        proc = self._llm_refresh_proc
        rc = proc.poll()
        if rc is None:
            # Still running (output goes to llm_refresh.log) — show elapsed
            el = int(time.monotonic()
                     - getattr(self, '_llm_refresh_start', time.monotonic()))
            self._llm_refresh_status.setText(f"Analyzing universe... ({el}s)")
            return
        # Done
        self._llm_refresh_timer.stop()
        del self._llm_refresh_proc
        self._set_llm_refresh_enabled(True)
        if rc == 0:
            self._reload_llm_from_disk()
            self._update_llm_table_cells()  # all symbols
            self._on_stock_row_selected(self._stock_table.currentRow(), 0, -1, 0)
            self._llm_refresh_status.setText("Analysis complete")
            self._llm_refresh_status.setStyleSheet(
                f"color: {T['green'].name()}; font-size: 11px;")
        else:
            self._llm_refresh_status.setText(f"Failed (exit {rc})")
            self._llm_refresh_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")

    def _reload_llm_from_disk(self):
        """Reload llm_analysis.json into the cache."""
        try:
            analysis_file = BASE_DIR / "llm_analysis.json"
            if analysis_file.exists():
                with open(analysis_file) as f:
                    raw = json.load(f)
                for section in raw.values():
                    if isinstance(section, dict):
                        self._llm_analysis_cache.update(section)
        except (OSError, json.JSONDecodeError):
            pass

    def _update_llm_table_cells(self, only_sym=None):
        """Re-render rows after an LLM refresh, reusing the single row-render
        path so column formatting (LLM p_up/age + every stance column) never
        diverges. Scans col 0 so it is correct regardless of the user's sort.

        If only_sym is set, only update that row. Otherwise update all rows.
        """
        tbl = getattr(self, '_stock_table', None)
        if tbl is None:
            return
        cache = self._stock_data_cache or {}
        snaps = cache.get('snapshots', {}) if isinstance(cache, dict) else {}
        preds = cache.get('predictions', {}) if isinstance(cache, dict) else {}
        tbl.setSortingEnabled(False)
        for row in range(tbl.rowCount()):
            item = tbl.item(row, 0)
            if not item:
                continue
            sym = item.text()
            if only_sym and sym != only_sym:
                continue
            self._update_stock_row(
                row, sym, snaps.get(sym, {}), preds.get(sym, {}),
                self._llm_analysis_cache.get(sym, {}), flash=False)
        tbl.setSortingEnabled(True)

    def _on_heatmap_clicked(self, sym):
        # Heatmap click now drives everything (table/detail/chart), not just
        # the combo/chart — unified linked selection (5.1).
        self._set_active_symbol(sym, 'heatmap')

    def _on_heatmap_metric_changed(self, _idx=None):
        """Metric combo changed — recolor tiles (no relayout, same members)."""
        self._repaint_heatmap()

    def _relayout_heatmap(self, groups):
        """Reposition group headers + tiles in the heatmap grid. Called only when
        the group->symbol layout changed (membership shift), never on a value
        tick. Detaches every managed widget first (kept alive), then re-adds a
        header row spanning the grid per non-empty group with its tiles below."""
        lay = self._heatmap_layout
        while lay.count():
            w = lay.takeAt(0).widget()
            if w is not None:
                w.hide()
        cols = self._heatmap_cols
        r = 0
        for gname, syms in groups:
            present = [s for s in syms if s in self._heatmap_labels]
            if not present:
                continue
            hdr = self._heatmap_headers.get(gname)
            if hdr is not None:
                lay.addWidget(hdr, r, 0, 1, cols)
                hdr.show()
                r += 1
            c = 0
            for sym in present:
                tile = self._heatmap_labels[sym]
                lay.addWidget(tile, r, c)
                tile.show()
                c += 1
                if c >= cols:
                    c = 0
                    r += 1
            if c != 0:
                r += 1  # next group starts on a fresh row

    def _heatmap_diverging(self, value, scale, pal):
        """(bg_hex, text_hex) for a symmetric diverging tile: value>0 mixes toward
        the up color, <0 toward down, intensity = sqrt(|value|/scale). A missing/
        non-finite value or non-positive scale renders the neutral tile."""
        bg = pal['tile_neutral']
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = None
        if v is not None and math.isfinite(v) and scale > 0 and v != 0:
            intensity = math.sqrt(min(abs(v) / scale, 1.0))
            bg = chart_core.mix(pal['tile_neutral'],
                                pal['up'] if v > 0 else pal['down'], intensity)
        text = ('#000000'
                if chart_core.contrast_ratio('#000000', bg)
                >= chart_core.contrast_ratio('#ffffff', bg) else '#ffffff')
        return bg, text

    def _repaint_heatmap(self):
        """Recolor every heatmap tile for the selected metric. Day % uses
        chart_core.heatmap_style; Pred/Meta-p use a symmetric diverging map
        (palette up/down, intensity by |value|); missing values -> neutral. Also
        re-themes the group headers (theme flips route here via on_stocks)."""
        if not getattr(self, '_heatmap_labels', None):
            return
        data = getattr(self, '_stock_data_cache', None) or {}
        symbols = data.get('symbols', [])
        snapshots = data.get('snapshots', {})
        predictions = data.get('predictions', {})
        metric = (self._heatmap_metric_combo.currentData()
                  if hasattr(self, '_heatmap_metric_combo') else 'day')
        pal = self._chart_pal or _chart_palette()
        for hdr in getattr(self, '_heatmap_headers', {}).values():
            hdr.setStyleSheet(
                f"color: {T['muted'].name()}; font-size: 10px; font-weight: bold;")
        # Pred has no fixed scale — normalize intensity against the largest |pred|
        # currently displayed (floored so an all-zero batch stays neutral).
        pred_scale = 0.0
        if metric == 'pred':
            for s in symbols:
                p = predictions.get(s)
                v = p.get('pred') if isinstance(p, dict) else None
                try:
                    fv = abs(float(v))
                    if math.isfinite(fv):
                        pred_scale = max(pred_scale, fv)
                except (TypeError, ValueError):
                    pass
        for sym, lbl in self._heatmap_labels.items():
            snap = snapshots.get(sym, {})
            short = sym.split('/')[0] if '/' in sym else sym
            if metric == 'pred':
                p = predictions.get(sym)
                v = p.get('pred') if isinstance(p, dict) else None
                bg_hex, text_hex = self._heatmap_diverging(v, pred_scale, pal)
                try:
                    valtext = f"{float(v):+.3f}"
                except (TypeError, ValueError):
                    valtext = "—"
            elif metric == 'metap':
                p = predictions.get(sym)
                mp = p.get('meta_p') if isinstance(p, dict) else None
                try:
                    mpf = float(mp)
                    bg_hex, text_hex = self._heatmap_diverging(mpf - 0.5, 0.5, pal)
                    valtext = f"{mpf:.2f}"
                except (TypeError, ValueError):
                    bg_hex, text_hex = self._heatmap_diverging(None, 0.5, pal)
                    valtext = "—"
            else:  # day
                chg = snap.get('change_pct', 0) or 0
                bg_hex, text_hex = chart_core.heatmap_style(chg, pal)
                valtext = f"{chg:+.1f}%"
            lbl.setText(f"{short}\n{valtext}")
            lbl.setStyleSheet(
                f"background-color: {bg_hex}; color: {text_hex};"
                f" font-size: 10px; font-weight: bold;"
                f" border-radius: 4px; padding: 2px;")

    def _request_chart(self):
        """Ask DataFetcher to fetch bars at the resolution needed for current zoom."""
        sym = self._stock_symbol_combo.currentText()
        if not sym:
            return
        res = self._zoom_resolution()
        self._stock_chart.setTitle(f"{sym} — Loading...")
        from PySide6.QtCore import QMetaObject, Q_ARG
        QMetaObject.invokeMethod(
            self._fetcher_slow, "fetch_chart", Qt.QueuedConnection,
            Q_ARG(str, sym), Q_ARG(str, res),
        )

    def _on_tab_changed(self, index):
        """Keep the slow fetcher's Markets-visibility flag in sync so
        fetch_stocks can throttle itself off-tab (plain bool, GIL-atomic)."""
        try:
            self._fetcher_slow.markets_visible = (index == self._markets_tab_index)
        except AttributeError:
            pass
        # Trading tab: lazy first paint of the analytics blocks (journal load
        # is off-thread; gate attribution is a small JSON read).
        if index == getattr(self, '_trading_tab_index', -1):
            try:
                self._refresh_gate_attribution()
            except Exception:
                pass
            try:
                self._refresh_journal_analytics()
            except Exception:
                pass

    def _auto_refresh_chart(self):
        """120s TTL-bounded auto-refresh — only while Markets tab is visible."""
        if self.tabs.currentIndex() == self._markets_tab_index:
            self._request_chart()

    def _apply_chart_zoom(self):
        """Slice cached data to the zoom window (via chart_core) and paint."""
        res = self._zoom_resolution()
        data = self._stock_chart_data.get(res)
        sym = self._stock_symbol_combo.currentText()
        if not data:
            return  # _request_chart already set the "Loading..." title

        view = chart_core.build_price_view(data, self._stock_zoom,
                                           overlays=self._selected_overlays())
        if self._chart_fp.get('price') == view.fingerprint:
            if view.status.status == 'ok':
                self._chart_last_ok['price'] = view.status.updated_at
            self._set_chart_status(self._stock_chart, f'{sym} ({self._stock_zoom})', view.status)
            return
        self._chart_fp['price'] = view.fingerprint

        # Preserve a manual pan/zoom across this refresh; else snap to window.
        saved_x = None
        if getattr(self, '_chart_user_viewport', False):
            saved_x = self._stock_chart.getViewBox().viewRange()[0]

        pal = self._chart_pal
        if view.mode == 'candles':
            self._candle_item.set_data(view.t, view.o, view.h, view.l, view.c, view.w,
                                        view.up, pal['up'], pal['down'],
                                        bg_color=pal['bg'])
            self._stock_chart_line.setData([], [])
        elif view.mode == 'line':
            self._stock_chart_line.setData(view.line_t, view.line_c)
            self._candle_item.set_data([], [], [], [], [], [], [], pal['up'], pal['down'])
        else:
            self._stock_chart_line.setData([], [])
            self._candle_item.set_data([], [], [], [], [], [], [], pal['up'], pal['down'])

        # Indicator overlays (present only in candles mode with a checkbox on).
        ov = view.overlays
        self._render_overlay_line(self._sma20_line, view.t, ov.get('sma20'))
        self._render_overlay_line(self._sma50_line, view.t, ov.get('sma50'))
        band = ov.get('atr_band')
        if band is not None:
            self._render_overlay_line(self._atr_upper_line, view.t, band[0])
            self._render_overlay_line(self._atr_lower_line, view.t, band[1])
        else:
            self._atr_upper_line.setData([], [])
            self._atr_lower_line.setData([], [])

        if view.has_volume:
            # Volume is a magnitude, not a direction: one recessive hue.
            # Direction already lives on the candles — repeating it here
            # would double-encode (and reintroduce the green/red CVD trap).
            vol_color = QColor(pal['grid'])
            vol_color.setAlpha(110)
            self._vol_bars.setOpts(x=view.vol_t, height=view.vol_v,
                                    width=view.vol_w, brushes=None,
                                    brush=vol_color, pen=pg.mkPen(None))
        else:
            self._vol_bars.setOpts(x=[], height=[], width=1, brushes=None)
        if view.vol_y_range is not None:
            self._stock_vol_plot.setYRange(*view.vol_y_range, padding=0)

        mk = view.markers  # empty dict on mode-'none' views — .get, not []
        self._entry_scatter.setData(x=mk.get('entry_t', []), y=mk.get('entry_p', []),
                                     brush=pal['marker_entry'])
        # Exit markers: per-point fill by realized-P&L sign (win=up, loss=down,
        # missing=neutral) + per-point data for the hover tooltip.
        exit_t = mk.get('exit_t', [])
        exit_p = mk.get('exit_p', [])
        exit_pnl = mk.get('exit_pnl', [])
        ex_brushes = []
        for k in range(len(exit_t)):
            pv = exit_pnl[k] if k < len(exit_pnl) else float('nan')
            if pv == pv and pv > 0:
                ex_brushes.append(pg.mkBrush(pal['up']))
            elif pv == pv and pv < 0:
                ex_brushes.append(pg.mkBrush(pal['down']))
            else:
                ex_brushes.append(pg.mkBrush(pal['marker_exit']))
        self._exit_scatter.setData(x=exit_t, y=exit_p, brush=ex_brushes,
                                    data=list(exit_pnl))

        # Last-price dashed level + label (follows pan via the InfiniteLine).
        last_c = None
        if view.mode == 'candles' and len(view.c):
            last_c = float(view.c[-1])
        elif view.mode == 'line' and len(view.line_c):
            last_c = float(view.line_c[-1])
        if last_c is not None and last_c > 0:
            self._last_price_line.setPos(last_c)
            self._last_price_line.label.setFormat(fmt_money(last_c))
            self._last_price_line.show()
        else:
            self._last_price_line.hide()

        # x-pan limits: full loaded data range (+/-5%) so panning can't leave
        # the data. y stays on auto-range (set at build); no setYRange here.
        try:
            ts_all = data.get('timestamps') or []
            if len(ts_all) >= 2:
                x0f, x1f = float(ts_all[0]), float(ts_all[-1])
                m = (x1f - x0f) * 0.05 or 86400.0
                self._stock_chart.getViewBox().setLimits(xMin=x0f - m, xMax=x1f + m)
        except Exception:
            pass
        if saved_x is not None:
            self._stock_chart.setXRange(*saved_x, padding=0)
        elif view.x_range is not None:
            self._stock_chart.setXRange(*view.x_range, padding=0.02)

        if view.mode == 'candles':
            self._stock_xhair.set_series(view.t, view.c)
            self._stock_xhair.set_ohlc(view.o, view.h, view.l, view.c,
                                       view.vol_v if view.has_volume else None)
        elif view.mode == 'line':
            self._stock_xhair.set_series(view.line_t, view.line_c)
            self._stock_xhair.clear_ohlc()
        else:
            self._stock_xhair.set_series([], [])
            self._stock_xhair.clear_ohlc()

        # Open-position guide lines track the charted symbol's live position.
        self._update_position_lines()

        if view.status.status == 'ok':
            self._chart_last_ok['price'] = view.status.updated_at

        self._set_chart_status(self._stock_chart, f'{sym} ({self._stock_zoom})', view.status)

    # ---- Markets chart interactivity (overlays / lines / pan-zoom) --------
    def _selected_overlays(self):
        """Tuple of the checked build_price_view overlay names."""
        ov = []
        if getattr(self, '_ov_sma20', None) is not None and self._ov_sma20.isChecked():
            ov.append('sma20')
        if getattr(self, '_ov_sma50', None) is not None and self._ov_sma50.isChecked():
            ov.append('sma50')
        if getattr(self, '_ov_atr', None) is not None and self._ov_atr.isChecked():
            ov.append('atr_band')
        return tuple(ov)

    def _on_overlay_toggled(self, _checked=None):
        settings = _load_gui_settings()
        settings['ov_sma20'] = self._ov_sma20.isChecked()
        settings['ov_sma50'] = self._ov_sma50.isChecked()
        settings['ov_atr'] = self._ov_atr.isChecked()
        _save_gui_settings(settings)
        # Overlays don't alter the chart_core fingerprint — bust the memo so the
        # repaint actually re-runs build_price_view with the new overlays tuple.
        self._chart_fp.pop('price', None)
        self._apply_chart_zoom()

    @staticmethod
    def _render_overlay_line(line, t, arr):
        """Set an overlay PlotDataItem, masking NaN warmup (connect='finite')."""
        if arr is not None and len(arr):
            line.setData(t, np.asarray(arr, dtype=float), connect='finite')
        else:
            line.setData([], [])

    @staticmethod
    def _exit_tip(x, y, data):
        """Hover tooltip for an exit marker: 'SOLD <price> (+2.3%)'."""
        pct = ''
        try:
            pnl = float(data)
            if pnl == pnl:  # not NaN
                pct = f" ({pnl:+.1f}%)"
        except (TypeError, ValueError):
            pass
        return f"SOLD {y:,.2f}{pct}"

    def _reset_chart_view(self):
        self._chart_user_viewport = False
        self._chart_fp.pop('price', None)  # force a repaint that re-snaps x
        self._apply_chart_zoom()

    def _on_chart_manual_range(self, *args):
        """A mouse pan/zoom on the price plot — preserve the viewport across
        subsequent data refreshes until a preset / Reset-view clears it."""
        self._chart_user_viewport = True

    def _update_position_lines(self):
        """Draw entry / est-stop / est-TP horizontals for the charted symbol's
        open position; hide them when flat. Cheap (setPos/label/visibility on
        pre-created InfiniteLines) — called on chart repaint AND on_positions."""
        if getattr(self, '_pos_entry_line', None) is None:
            return
        sym = self._stock_symbol_combo.currentText()
        p = self._position_row(sym)
        if not p:
            for ln in (self._pos_entry_line, self._pos_stop_line, self._pos_tp_line):
                ln.hide()
            return
        pstates = self._load_position_states()
        entry, est_stop, est_tp = self._exit_levels_raw(
            sym, p.get('avg_entry_price'), p.get('current_price'), pstates)

        def _set(ln, val, prefix):
            if val is None or not (val > 0):
                ln.hide()
                return
            ln.setPos(val)
            ln.label.setFormat(f"{prefix} {fmt_money(val)}")
            ln.show()

        _set(self._pos_entry_line, entry, 'entry')
        _set(self._pos_stop_line, est_stop, '~stop')
        _set(self._pos_tp_line, est_tp, '~TP')

    @Slot(dict)
    def on_chart(self, data):
        """Handle chart_updated signal — store full payload, apply zoom.
        build_price_view (inside _apply_chart_zoom) turns error/empty
        payloads into a ChartStatus, so no branching happens here."""
        sym = data.get('symbol', '')
        res = data.get('resolution', 'daily')

        # A successful bar payload stamps the chart stream healthy; an error
        # payload leaves it to the fetcher's error_occurred('chart', ...) emit.
        if 'error' not in data:
            self._stream_ok("chart")

        # Benchmark payloads (SPY/BTC daily) feed the equity overlay, not the
        # price chart. Routed independently so a benchmark fetch never clobbers
        # the selected symbol's chart (and vice-versa when they coincide).
        if getattr(self, '_bench_symbol', None) == sym and 'error' not in data:
            self._bench_chart_data = data
            try:
                self._apply_benchmark_overlay()
            except Exception:
                pass

        # Only update if this symbol is still selected (mid-flight drop)
        if self._stock_symbol_combo.currentText() != sym:
            return

        self._stock_chart_data[res] = data
        if self._zoom_resolution() == res:
            self._apply_chart_zoom()

    @Slot(dict)
    def on_stocks(self, data):
        """Handle stocks_updated signal — update heatmap, table, chart."""
        self._stream_ok("stocks")
        self._stock_data_cache = data
        symbols = data.get('symbols', [])
        snapshots = data.get('snapshots', {})
        predictions = data.get('predictions', {})
        llm_raw = data.get('llm_analysis', {})
        # MERGE, never wholesale-replace: a torn/empty read of llm_analysis.json
        # (the file is written per-book, so a mid-write read can come back empty)
        # used to blank every LLM column for a cycle. Mirror _reload_llm_from_disk
        # and only fold in non-empty state, so the dossier columns never flicker.
        if isinstance(llm_raw, dict) and llm_raw:
            self._llm_analysis_cache.update(llm_raw)
        llm_analysis = self._llm_analysis_cache

        # Book-wide regime chips (pulled from any symbol carrying `regime`).
        self._update_regime_chips(predictions)

        # --- Heatmap (asset-class grouped; recolored by the metric combo) ---
        # Lazy-create a tile per new symbol; DON'T place it here — the grouped
        # relayout owns placement so tiles sit under their group header.
        for sym in symbols:
            if sym not in self._heatmap_labels:
                lbl = QLabel()
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setFixedSize(62, 38)
                lbl.setCursor(Qt.PointingHandCursor)
                lbl.mousePressEvent = lambda _, s=sym: self._on_heatmap_clicked(s)
                self._heatmap_labels[sym] = lbl
        # Prune tiles for symbols that left the universe.
        for sym in list(self._heatmap_labels):
            if sym not in symbols:
                lbl = self._heatmap_labels.pop(sym)
                self._heatmap_layout.removeWidget(lbl)
                lbl.deleteLater()
        # Group by asset class (sector data unavailable — see build comment),
        # preserving universe order within each group.
        crypto_syms = [s for s in symbols
                       if '/' in s or str(s).upper() in CRYPTO_SYMBOL_SET]
        _crypto_set = set(crypto_syms)
        stock_syms = [s for s in symbols if s not in _crypto_set]
        groups = [('Crypto', crypto_syms), ('Stocks', stock_syms)]
        sig = tuple((g, tuple(ss)) for g, ss in groups)
        if sig != getattr(self, '_heatmap_layout_sig', None):
            self._relayout_heatmap(groups)
            self._heatmap_layout_sig = sig
        # Recolor every tile for the current metric (also re-themes on flip).
        self._repaint_heatmap()

        # --- Metrics table (diff update, not full teardown) ---
        self._sync_stock_table(symbols, snapshots, predictions, llm_analysis)

        # Refresh chart if no data yet
        if not hasattr(self, '_stock_chart_loaded'):
            self._stock_chart_loaded = True
            self._request_chart()

    def _update_regime_chips(self, predictions):
        """Set the two book-wide regime chips from any symbol carrying `regime`.
        The loops write one book-wide regime string duplicated onto every symbol
        of that book, so the last one seen per book wins ('—' if none)."""
        crypto_regime = None
        stock_regime = None
        try:
            for sym, p in (predictions or {}).items():
                if not isinstance(p, dict):
                    continue
                reg = p.get('regime')
                if not reg:
                    continue
                if '/' in str(sym) or str(sym).upper() in CRYPTO_SYMBOL_SET:
                    crypto_regime = str(reg)
                else:
                    stock_regime = str(reg)
        except (AttributeError, TypeError):
            pass
        if getattr(self, '_crypto_regime_chip', None) is not None:
            self._crypto_regime_chip.setText(
                f"Crypto regime: {crypto_regime or '—'}")
        if getattr(self, '_stock_regime_chip', None) is not None:
            self._stock_regime_chip.setText(
                f"Stock regime: {stock_regime or '—'}")

    def _score01_color(self, v):
        """Palette color for a 0–1 bullishness proxy (LLM `s` or advisor
        p_up). Same bands the LLM column has always used: veto-dark<0.15,
        red<0.40, yellow<=0.60, green above."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return T['muted']
        if v < 0.15:
            return QColor(180, 0, 0)   # dark red for VETO
        if v < 0.40:
            return T['red']
        if v <= 0.60:
            return T.get('yellow', T['white'])
        return T['green']

    def _sync_stock_table(self, symbols, snapshots, predictions, llm_analysis):
        """Diff-update the stance table in place: create rows only for new
        symbols, drop rows only for removed ones, touch only changed cells, and
        preserve the user's selected SYMBOL + scroll across the refresh. Replaces
        the old full teardown that dropped selection/scroll 2x/min while you read
        a dossier."""
        tbl = self._stock_table
        markets_visible = (self.tabs.currentIndex() == self._markets_tab_index)

        # Capture selection (by symbol, survives re-sort) + scroll to restore.
        sel_sym = None
        cur = tbl.currentRow()
        if cur >= 0:
            it = tbl.item(cur, 0)
            if it:
                sel_sym = it.text()
        vbar = tbl.verticalScrollBar()
        scroll_val = vbar.value() if vbar is not None else 0

        # Sorting OFF during structural edits: index-addressed multi-cell writes
        # must not have rows shuffled out from under them mid-loop.
        tbl.setSortingEnabled(False)
        tbl.setUpdatesEnabled(False)
        # Suppress currentCellChanged during the structural edits + the re-sort
        # below: a mid-sync removal of the selected row would otherwise fire the
        # linked-selection router and spuriously refetch a chart. Unblocked
        # before the explicit selection restore (which does its own local block).
        # Self-heals each tick (re-blocked at the next sync) even on exception.
        tbl.blockSignals(True)

        # Authoritative current sym->row from the live table (survives user sorts
        # that reordered rows since we last built the map).
        existing = {}
        for r in range(tbl.rowCount()):
            it = tbl.item(r, 0)
            if it:
                existing[it.text()] = r

        want = list(symbols)
        want_set = set(want)

        # Drop rows whose symbol left the universe (descending, so lower indices
        # stay valid as we remove).
        stale_rows = sorted((r for s, r in existing.items() if s not in want_set),
                            reverse=True)
        for r in stale_rows:
            tbl.removeRow(r)
        if stale_rows:
            existing = {}
            for r in range(tbl.rowCount()):
                it = tbl.item(r, 0)
                if it:
                    existing[it.text()] = r

        # Append brand-new symbols (rows created once, then filled in place).
        for sym in want:
            if sym not in existing:
                r = tbl.rowCount()
                tbl.insertRow(r)
                self._make_stock_row(r, sym)
                existing[sym] = r

        # In-place cell updates for every wanted symbol.
        for sym in want:
            r = existing.get(sym)
            if r is None:
                continue
            self._update_stock_row(
                r, sym, snapshots.get(sym, {}), predictions.get(sym, {}),
                llm_analysis.get(sym, {}), flash=markets_visible)

        self._stocks_row_by_sym = existing

        tbl.setUpdatesEnabled(True)
        tbl.setSortingEnabled(True)  # re-sorts appended rows into user order
        tbl.blockSignals(False)      # end the currentCellChanged suppression

        # Restore selection by symbol (its row index may have moved on re-sort)
        # + scroll. Block signals so the silent re-select doesn't re-render the
        # detail panel (same symbol stays shown).
        if sel_sym is not None:
            for r in range(tbl.rowCount()):
                it = tbl.item(r, 0)
                if it and it.text() == sel_sym:
                    tbl.blockSignals(True)
                    tbl.setCurrentCell(r, 0)
                    tbl.blockSignals(False)
                    break
        if vbar is not None:
            vbar.setValue(scroll_val)

    def _make_stock_row(self, row, sym):
        """Create the 13 cells of a new stance row once. All cells are
        NumericTableItem (text columns simply carry no UserRole and fall back to
        text sort); values are filled by _update_stock_row."""
        tbl = self._stock_table
        for col in range(13):
            item = NumericTableItem("")
            item.setTextAlignment(Qt.AlignCenter)
            tbl.setItem(row, col, item)
        sym_item = tbl.item(row, 0)
        sym_item.setText(sym)          # Symbol cell is the stable row identity
        sym_item.setForeground(T['white'])

    def _set_cell(self, item, text, color, sort_key=None):
        """Update one existing cell: setText only when it actually changed (keeps
        churn/re-sort minimal), always refresh color (cheap; keeps theme flips
        correct), set the numeric sort key when provided."""
        if item is None:
            return
        if item.text() != text:
            item.setText(text)
        item.setForeground(color)
        if sort_key is not None:
            item.setData(Qt.UserRole, float(sort_key))

    def _set_held_item(self, sym, item):
        """Render the Held cell for `sym` into `item` (qty in the cell, unrealized
        P&L% in the tooltip; '—' when flat). Takes the item ref (not a row
        index) so the cheap 5s positions-tick refresh is safe even if a value
        change re-sorts rows mid-loop."""
        if item is None:
            return
        prow = self._position_row(sym)  # slash-insensitive match
        held_qty = None
        pnl_pct = None
        if prow is not None:
            try:
                held_qty = float(prow.get('qty'))
            except (TypeError, ValueError):
                held_qty = None
            try:
                pnl_pct = float(prow.get('unrealized_plpc')) * 100
            except (TypeError, ValueError):
                pnl_pct = None
        if held_qty is not None and held_qty != 0:
            color = pnl_color(pnl_pct) if pnl_pct is not None else T['white']
            self._set_cell(item, fmt_qty(held_qty), color, sort_key=held_qty)
            item.setToolTip(
                f"{sym}: {fmt_qty(held_qty)} · unrealized {pnl_pct:+.2f}%"
                if pnl_pct is not None else f"{sym}: {fmt_qty(held_qty)}")
        else:
            self._set_cell(item, "—", T['muted'], sort_key=float('-inf'))
            item.setToolTip("")

    def _update_stock_row(self, row, sym, snap, pred, llm, flash=False):
        """Update cols 1..12 of an existing stance row in place. `flash` (only
        true when the Markets tab is visible) briefly tints the Price cell on a
        real price change."""
        tbl = self._stock_table
        green, red = T['green'], T['red']
        yellow = T.get('yellow', T['white'])
        white, muted = T['white'], T['muted']

        price = snap.get('price', 0)
        chg = snap.get('change_pct', 0)
        vol = snap.get('volume', 0)
        pred_val = pred.get('pred')
        signal = pred.get('signal', '')
        meta_p = pred.get('meta_p')
        conv = pred.get('conviction')
        rank = pred.get('rank')
        gate = pred.get('llm_gate')

        # 1 Price (+ change flash)
        price_item = tbl.item(row, 1)
        old_price = price_item.data(Qt.UserRole) if price_item else None
        price_text = f"${price:.2f}" if price else "—"
        price_sort = float(price) if price else float('-inf')
        self._set_cell(price_item, price_text, white, sort_key=price_sort)
        if (flash and price and old_price is not None
                and math.isfinite(old_price) and old_price > 0
                and abs(float(old_price) - float(price)) > 1e-9):
            self._flash_price_cell(price_item)

        # 2 Day %
        self._set_cell(tbl.item(row, 2), f"{chg:+.2f}%",
                       green if chg > 0 else (red if chg < 0 else white),
                       sort_key=chg)

        # 3 Volume — SI units (1.2M) to match the chart's volume axis (SIAxisItem)
        self._set_cell(tbl.item(row, 3),
                       chart_core.format_si(vol) if vol else "—",
                       muted, sort_key=float(vol) if vol else 0.0)

        # 4 Pred
        if pred_val is not None:
            self._set_cell(
                tbl.item(row, 4), f"{pred_val:+.4f}",
                green if pred_val > 0 else (red if pred_val < 0 else muted),
                sort_key=pred_val)
        else:
            self._set_cell(tbl.item(row, 4), "—", muted,
                           sort_key=float('-inf'))

        # 5 Meta-p — display heuristic only (NOT the exact meta veto threshold
        # family): >=0.5 green, <0.3 red, else default — a legibility cue.
        if meta_p is not None:
            try:
                mp = float(meta_p)
                self._set_cell(
                    tbl.item(row, 5), f"{mp:.2f}",
                    green if mp >= 0.5 else (red if mp < 0.3 else white),
                    sort_key=mp)
            except (TypeError, ValueError):
                self._set_cell(tbl.item(row, 5), "—", muted,
                               sort_key=float('-inf'))
        else:
            self._set_cell(tbl.item(row, 5), "—", muted,
                           sort_key=float('-inf'))

        # 6 Conv (model conviction, as delivered)
        if conv is not None and conv != '':
            try:
                conv_sort = float(conv)
            except (TypeError, ValueError):
                conv_sort = float('-inf')
            self._set_cell(tbl.item(row, 6), str(conv), white,
                           sort_key=conv_sort)
        else:
            self._set_cell(tbl.item(row, 6), "—", muted,
                           sort_key=float('-inf'))

        # 7 Rank (stocks only; '#N', missing sorts last)
        if rank is not None:
            try:
                rk = int(rank)
                self._set_cell(tbl.item(row, 7), f"#{rk}", white,
                               sort_key=float(rk))
            except (TypeError, ValueError):
                self._set_cell(tbl.item(row, 7), "—", muted,
                               sort_key=float('inf'))
        else:
            self._set_cell(tbl.item(row, 7), "—", muted,
                           sort_key=float('inf'))

        # 8 Signal (text sort)
        self._set_cell(tbl.item(row, 8), signal,
                       green if signal == 'BULL' else
                       (red if signal == 'BEAR' else muted))

        # 9 Gate (LLM gate pass/veto)
        if gate == 'pass':
            self._set_cell(tbl.item(row, 9), "✓", green)
        elif gate == 'veto':
            self._set_cell(tbl.item(row, 9), "✗", red)
        else:
            self._set_cell(tbl.item(row, 9), "—", muted)

        # 10 Held (position join)
        self._set_held_item(sym, tbl.item(row, 10))

        # 11 LLM (advisor p_up preferred, else s, else legacy m)
        llm_s = llm.get('s')
        llm_m = llm.get('m')
        p_up = llm.get('p_up')
        conviction_llm = llm.get('conviction')
        llm_bull = llm.get('bull', '')
        llm_bear = llm.get('bear', '')
        llm_r = llm.get('r', '')
        if p_up is not None:
            try:
                pu = float(p_up)
                arrow = "↑" if pu >= 0.5 else "↓"
                self._set_cell(tbl.item(row, 11), f"{pu:.2f}{arrow}",
                               self._score01_color(pu), sort_key=pu)
            except (TypeError, ValueError):
                p_up = None
        if p_up is None:
            if llm_s is not None:
                self._set_cell(tbl.item(row, 11), f"{float(llm_s):.2f}",
                               self._score01_color(llm_s), sort_key=float(llm_s))
            elif llm_m is not None:
                self._set_cell(
                    tbl.item(row, 11), f"{float(llm_m):.1f}x",
                    green if llm_m >= 1.0 else (red if llm_m <= 0.5 else yellow),
                    sort_key=float(llm_m))
            else:
                self._set_cell(tbl.item(row, 11), "—", muted,
                               sort_key=float('-inf'))
        # LLM tooltip: bull/bear/summary + m/s/conviction numeric context.
        llm_item = tbl.item(row, 11)
        if llm_item is not None:
            tip_parts = []
            if llm_bull:
                tip_parts.append(f"BULL: {llm_bull}")
            if llm_bear:
                tip_parts.append(f"BEAR: {llm_bear}")
            if llm_r:
                tip_parts.append(f"Summary: {llm_r}")
            ms = []
            if llm_m is not None:
                ms.append(f"m={llm_m:.2f}")
            if llm_s is not None:
                ms.append(f"s={llm_s:.2f}")
            if conviction_llm is not None:
                ms.append(f"conv={conviction_llm}")
            if ms:
                tip_parts.append(" ".join(ms))
            ts = llm.get('timestamp', '')
            if ts:
                tip_parts.append(f"({ts})")
            llm_item.setToolTip("\n".join(tip_parts))

        # 12 LLM Age
        llm_ts = llm.get('timestamp', '')
        age_hours = float('inf')
        age_text = "—"
        age_color = muted
        if llm_ts:
            try:
                t = dt.datetime.fromisoformat(llm_ts)
                now = dt.datetime.now(tz=t.tzinfo or None)
                age_hours = (now - t).total_seconds() / 3600
                if age_hours < 1:
                    age_text = f"{age_hours * 60:.0f}m"
                elif age_hours < 24:
                    age_text = f"{age_hours:.0f}h"
                else:
                    age_text = f"{age_hours / 24:.0f}d"
                age_color = (green if age_hours < 12 else
                             (yellow if age_hours < 48 else red))
            except (ValueError, TypeError):
                pass
        self._set_cell(tbl.item(row, 12), age_text, age_color,
                       sort_key=age_hours)

    def _flash_price_cell(self, item):
        """Brief accent-tinted flash on a price change; reverts after 600ms.
        Item refs stay valid across re-sorts; guard the revert for the case the
        row was removed before the timer fired."""
        try:
            c = QColor(T['accent'])
            c.setAlpha(60)
            item.setBackground(c)
        except (RuntimeError, KeyError):
            return
        QTimer.singleShot(600, lambda: self._revert_flash(item))

    def _revert_flash(self, item):
        try:
            item.setData(Qt.BackgroundRole, None)  # clear -> default row bg
        except RuntimeError:
            pass

    def _refresh_held_cells(self):
        """Cheap targeted refresh of just the Held column from the 5s positions
        tick — no full table rebuild. Snapshots (symbol, held-item) pairs first
        so a value-change re-sort can't make us skip/double-visit a row."""
        tbl = getattr(self, '_stock_table', None)
        if tbl is None:
            return
        try:
            pairs = []
            for row in range(tbl.rowCount()):
                sym_item = tbl.item(row, 0)
                held_item = tbl.item(row, 10)
                if sym_item and held_item:
                    pairs.append((sym_item.text(), held_item))
            for sym, held_item in pairs:
                self._set_held_item(sym, held_item)
        except Exception:
            pass

    # ---- Tab 6: Models ---------------------------------------------------
    def _build_models_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        model_label = QLabel("Model Status")
        model_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(model_label)

        self._model_table = QTableWidget(0, 11)
        self._model_table.setHorizontalHeaderLabels(
            ["Model", "Status", "Score", "Last Trained", "Age",
             "Hidden Dim", "Layers", "Seq Len", "Threshold", "Preset",
             "Challenger"]
        )
        self._model_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._model_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._model_table.setAlternatingRowColors(True)
        layout.addWidget(self._model_table)

        # Shadow / DM-HLN promotion panel (gui_review_2026-07 §7 challenger
        # cell, Phase 2.2) — the full promotion story shadow.py persists to
        # {prefix}shadow_status.json, which the challenger cell only hinted at
        # via a manifest mtime. One rich label per book; filled in
        # _refresh_shadow_panel (60s, guarded).
        shadow_group = QGroupBox("Shadow / Promotion (DM-HLN)")
        shadow_v = QVBoxLayout(shadow_group)
        self._shadow_labels = {}
        for book in ("Crypto", "Stock"):
            lbl = QLabel(f"{book}: —")
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.RichText)
            shadow_v.addWidget(lbl)
            self._shadow_labels[book] = lbl
        layout.addWidget(shadow_group)

        # Meta-gate panel (c26 U1) — pred_source / AUC / refusal sidecar from
        # {prefix}meta_meta.json + {prefix}meta_refused.json; filled in
        # _refresh_meta_panel (60s, guarded).
        meta_group = QGroupBox("Meta gate (meta_meta.json)")
        meta_v = QVBoxLayout(meta_group)
        self._meta_gate_labels = {}
        for book in ("Crypto", "Stock"):
            lbl = QLabel(f"{book}: —")
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.RichText)
            meta_v.addWidget(lbl)
            self._meta_gate_labels[book] = lbl
        layout.addWidget(meta_group)

        # Drift panel (gui_review_2026-07 §7 missing item, Phase 2.7) — PSI per
        # label from monitor_drift.py's drift_state.json.
        drift_group = QGroupBox("Drift Monitor (PSI)")
        drift_v = QVBoxLayout(drift_group)
        self._drift_label = QLabel("—")
        self._drift_label.setWordWrap(True)
        self._drift_label.setTextFormat(Qt.RichText)
        drift_v.addWidget(self._drift_label)
        layout.addWidget(drift_group)

        pipeline_group = QGroupBox("Pipeline Status")
        pipeline_layout = QGridLayout(pipeline_group)
        self._pipeline_status = QLabel("Status: \u2014")
        self._pipeline_phase = QLabel("Phase: \u2014")
        self._pipeline_trial = QLabel("Trial: \u2014")
        self._pipeline_best = QLabel("Best Score: \u2014")
        self._pipeline_elapsed = QLabel("Elapsed: \u2014")
        self._pipeline_scores = QLabel("")

        self._pipeline_progress = QProgressBar()
        self._pipeline_progress.setRange(0, 100)
        self._pipeline_progress.setFixedHeight(16)
        self._pipeline_progress.setTextVisible(True)
        self._pipeline_progress.setFormat("%v / %m trials")

        pipeline_layout.addWidget(self._pipeline_status, 0, 0)
        pipeline_layout.addWidget(self._pipeline_phase, 0, 1)
        pipeline_layout.addWidget(self._pipeline_trial, 1, 0)
        pipeline_layout.addWidget(self._pipeline_best, 1, 1)
        pipeline_layout.addWidget(self._pipeline_progress, 2, 0, 1, 2)
        self._pipeline_retrain = QLabel("")
        pipeline_layout.addWidget(self._pipeline_elapsed, 3, 0)
        pipeline_layout.addWidget(self._pipeline_scores, 3, 1)
        pipeline_layout.addWidget(self._pipeline_retrain, 4, 0, 1, 2)

        # Per-phase outcome badges (ok / failed / gate-rolled-back), incl.
        # attempt counts — makes a gate rollback visible instead of silent.
        self._pipeline_phase_results = QLabel("")
        self._pipeline_phase_results.setWordWrap(True)
        pipeline_layout.addWidget(self._pipeline_phase_results, 5, 0, 1, 2)

        # Command acknowledgement: run_pipeline.py echoes start/stop verdicts
        # into command_result.json — otherwise a rejected command is invisible
        # while the GUI optimistically says "Starting…".
        self._pipeline_cmd_ack = QLabel("")
        self._pipeline_cmd_ack.setWordWrap(True)
        pipeline_layout.addWidget(self._pipeline_cmd_ack, 8, 0, 1, 2)

        # Pipeline restart control
        restart_row = QHBoxLayout()
        self._restart_pipeline_btn = QPushButton("Restart Pipeline")
        self._restart_pipeline_btn.setFixedHeight(28)
        self._restart_pipeline_btn.setCursor(Qt.PointingHandCursor)
        self._restart_pipeline_btn.clicked.connect(self._restart_pipeline_clicked)
        self._restart_pipeline_status = QLabel("")
        self._restart_pipeline_status.setStyleSheet("font-size: 11px;")
        restart_row.addWidget(self._restart_pipeline_btn)
        restart_row.addWidget(self._restart_pipeline_status)
        restart_row.addStretch()
        pipeline_layout.addLayout(restart_row, 6, 0, 1, 2)

        # Manual retrain controls
        retrain_row = QHBoxLayout()
        self._retrain_crypto_btn = QPushButton("Retrain Crypto")
        self._retrain_stock_btn = QPushButton("Retrain Stocks")
        self._retrain_both_btn = QPushButton("Retrain Both")
        self._retrain_cancel_btn = QPushButton("Cancel")
        self._retrain_cancel_btn.setVisible(False)
        self._retrain_status = QLabel("")
        self._retrain_status.setStyleSheet("font-size: 11px;")
        for btn in [self._retrain_crypto_btn, self._retrain_stock_btn, self._retrain_both_btn]:
            btn.setFixedHeight(28)
            btn.setCursor(Qt.PointingHandCursor)
        self._retrain_cancel_btn.setFixedHeight(28)
        self._retrain_cancel_btn.setCursor(Qt.PointingHandCursor)
        self._retrain_crypto_btn.clicked.connect(lambda: self._trigger_retrain(crypto=True, stock=False))
        self._retrain_stock_btn.clicked.connect(lambda: self._trigger_retrain(crypto=False, stock=True))
        self._retrain_both_btn.clicked.connect(lambda: self._trigger_retrain(crypto=True, stock=True))
        self._retrain_cancel_btn.clicked.connect(self._cancel_retrain)
        retrain_row.addWidget(self._retrain_crypto_btn)
        retrain_row.addWidget(self._retrain_stock_btn)
        retrain_row.addWidget(self._retrain_both_btn)
        retrain_row.addWidget(self._retrain_cancel_btn)
        retrain_row.addWidget(self._retrain_status)
        retrain_row.addStretch()
        pipeline_layout.addLayout(retrain_row, 7, 0, 1, 2)

        layout.addWidget(pipeline_group)

        # Bot Control group
        bot_group = QGroupBox("Bot Control")
        bot_layout = QHBoxLayout(bot_group)

        # Crypto bot controls
        self._crypto_bot_label = QLabel("Crypto Bot: --")
        self._crypto_bot_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        bot_layout.addWidget(self._crypto_bot_label)
        self._crypto_start_btn = QPushButton("Start")
        self._crypto_stop_btn = QPushButton("Stop")
        for btn in [self._crypto_start_btn, self._crypto_stop_btn]:
            btn.setFixedHeight(28)
            btn.setCursor(Qt.PointingHandCursor)
        self._crypto_start_btn.clicked.connect(lambda: self._start_bot_clicked("Crypto"))
        self._crypto_stop_btn.clicked.connect(lambda: self._stop_bot_clicked("Crypto"))
        bot_layout.addWidget(self._crypto_start_btn)
        bot_layout.addWidget(self._crypto_stop_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        bot_layout.addWidget(sep)

        # Stock bot controls
        self._stock_bot_label = QLabel("Stock Bot: --")
        self._stock_bot_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        bot_layout.addWidget(self._stock_bot_label)
        self._stock_start_btn = QPushButton("Start")
        self._stock_stop_btn = QPushButton("Stop")
        for btn in [self._stock_start_btn, self._stock_stop_btn]:
            btn.setFixedHeight(28)
            btn.setCursor(Qt.PointingHandCursor)
        self._stock_start_btn.clicked.connect(lambda: self._start_bot_clicked("Stock"))
        self._stock_stop_btn.clicked.connect(lambda: self._stop_bot_clicked("Stock"))
        bot_layout.addWidget(self._stock_start_btn)
        bot_layout.addWidget(self._stock_stop_btn)

        # Kill switch (shared trading_halt.flag / flatten_request.flag
        # contract with notify.py — Telegram /halt and ssh use the same file)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        bot_layout.addWidget(sep2)

        self._halt_btn = QPushButton(
            "Resume Entries" if halt_active() else "Halt Entries")
        self._flatten_btn = QPushButton("Flatten All")
        for btn in [self._halt_btn, self._flatten_btn]:
            btn.setFixedHeight(28)
            btn.setCursor(Qt.PointingHandCursor)
        self._halt_btn.clicked.connect(self._toggle_halt_clicked)
        self._flatten_btn.clicked.connect(self._flatten_all_clicked)
        bot_layout.addWidget(self._halt_btn)
        bot_layout.addWidget(self._flatten_btn)

        self._bot_cmd_status = QLabel("")
        self._bot_cmd_status.setStyleSheet("font-size: 11px;")
        bot_layout.addWidget(self._bot_cmd_status)
        bot_layout.addStretch()

        layout.addWidget(bot_group)

        hw_group = QGroupBox("Hardware")
        hw_grid = QGridLayout(hw_group)

        def _hw_gauge(label_text, row, col):
            frame = QFrame()
            inner = QVBoxLayout(frame)
            inner.setContentsMargins(4, 2, 4, 2)
            inner.addWidget(QLabel(label_text))
            val = QLabel("\u2014")
            val.setStyleSheet("font-size: 22px; font-weight: bold;")
            inner.addWidget(val)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            inner.addWidget(bar)
            hw_grid.addWidget(frame, row, col)
            return val, bar

        self._gpu_temp_label, self._gpu_temp_bar = _hw_gauge("GPU Temp", 0, 0)
        self._gpu_load_label, self._gpu_load_bar = _hw_gauge("GPU Load", 0, 1)
        self._gpu_clock_label, self._gpu_clock_bar = _hw_gauge("GPU Clock", 0, 2)
        self._cpu_temp_label, self._cpu_temp_bar = _hw_gauge("CPU Temp", 1, 0)
        self._cpu_load_label, self._cpu_load_bar = _hw_gauge("CPU Load", 1, 1)
        self._ram_label, self._ram_bar = _hw_gauge("Shared Memory", 1, 2)
        self._disk_label, self._disk_bar = _hw_gauge("Disk", 2, 0)

        # Two axis-less sparklines (GPU-temp + RAM%, 60-sample ring buffers) — the
        # trend at a glance beside the gauges. No interaction; muted line only.
        # Background + line follow the theme in _restyle (like _today_spark).
        _spark_pal = _chart_palette()

        def _hw_spark(label_text, row, col):
            frame = QFrame()
            inner = QVBoxLayout(frame)
            inner.setContentsMargins(4, 2, 4, 2)
            inner.addWidget(QLabel(label_text))
            pw = pg.PlotWidget()
            pw.setFixedHeight(40)
            pw.setMouseEnabled(x=False, y=False)
            pw.hideAxis('left')
            pw.hideAxis('bottom')
            pw.setMenuEnabled(False)
            pw.hideButtons()
            pw.setBackground(_spark_pal['bg'])
            curve = pw.plot([], [], pen=pg.mkPen(T['muted'].name(), width=1))
            inner.addWidget(pw)
            hw_grid.addWidget(frame, row, col)
            return pw, curve
        self._gpu_temp_spark_pw, self._gpu_temp_spark = _hw_spark("GPU Temp trend", 2, 1)
        self._ram_spark_pw, self._ram_spark = _hw_spark("RAM % trend", 2, 2)

        layout.addWidget(hw_group)

        # LLM Usage group
        llm_usage_group = QGroupBox("LLM Usage (Today)")
        llm_usage_layout = QGridLayout(llm_usage_group)
        self._llm_cost_label = QLabel("Cost: $0.000 / $0.65")
        self._llm_cost_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        llm_usage_layout.addWidget(self._llm_cost_label, 0, 0)

        self._llm_cost_bar = QProgressBar()
        self._llm_cost_bar.setRange(0, 100)
        self._llm_cost_bar.setFixedHeight(10)
        self._llm_cost_bar.setTextVisible(False)
        llm_usage_layout.addWidget(self._llm_cost_bar, 0, 1)

        self._llm_pro_label = QLabel("Pro: \u2014")
        self._llm_pro_label.setStyleSheet("font-size: 12px;")
        llm_usage_layout.addWidget(self._llm_pro_label, 1, 0)

        self._llm_flash_label = QLabel("Flash: \u2014")
        self._llm_flash_label.setStyleSheet("font-size: 12px;")
        llm_usage_layout.addWidget(self._llm_flash_label, 1, 1)

        self._llm_lite_label = QLabel("Lite: \u2014")
        self._llm_lite_label.setStyleSheet("font-size: 12px;")
        llm_usage_layout.addWidget(self._llm_lite_label, 1, 2)

        layout.addWidget(llm_usage_group)

        # Reports group (offline measurement-only scripts — safe to launch
        # directly, no promotion gate needed per repo conventions)
        reports_group = QGroupBox("Reports")
        reports_v = QVBoxLayout(reports_group)
        reports_layout = QHBoxLayout()
        self._decision_report_btn = QPushButton("Decision Report (30d)")
        self._beta_ledger_btn = QPushButton("Beta Ledger (90d)")
        self._leadlag_btn = QPushButton("Indicator Lead/Lag (Crypto)")
        self._leadlag_stock_btn = QPushButton("Indicator Lead/Lag (Stock)")
        self._gap_audit_btn = QPushButton("Gap Audit")
        self._llm_eval_btn = QPushButton("LLM Eval (14d)")
        self._llm_advisor_btn = QPushButton("LLM Advisor (14d)")
        self._exec_report_btn = QPushButton("Execution Report (14d)")
        # One list drives the enable/disable single-flight in both
        # _run_report_clicked and _check_report_run — new buttons stay in sync.
        self._report_btns = [
            self._decision_report_btn, self._beta_ledger_btn,
            self._leadlag_btn, self._leadlag_stock_btn, self._gap_audit_btn,
            self._llm_eval_btn, self._llm_advisor_btn, self._exec_report_btn]
        for btn in self._report_btns:
            btn.setFixedHeight(28)
            btn.setCursor(Qt.PointingHandCursor)
        self._decision_report_btn.clicked.connect(
            lambda: self._run_report_clicked(
                ["decision_report.py", "--days", "30"],
                "Decision Report (30d)"))
        self._beta_ledger_btn.clicked.connect(
            lambda: self._run_report_clicked(
                ["beta_ledger.py", "--days", "90"],
                "Beta Ledger (90d)"))
        self._leadlag_btn.clicked.connect(
            lambda: self._run_report_clicked(
                ["indicator_leadlag.py", "--data", "crypto_training_data.parquet"],
                "Indicator Lead/Lag (Crypto)"))
        self._leadlag_stock_btn.clicked.connect(
            lambda: self._run_report_clicked(
                ["indicator_leadlag.py", "--data", "stock_training_data.parquet"],
                "Indicator Lead/Lag (Stock)"))
        self._gap_audit_btn.clicked.connect(self._run_gap_audit_clicked)
        self._llm_eval_btn.clicked.connect(
            lambda: self._run_report_clicked(
                ["llm_eval.py", "--days", "14"], "LLM Eval (14d)"))
        self._llm_advisor_btn.clicked.connect(
            lambda: self._run_report_clicked(
                ["llm_eval.py", "--days", "14", "--advisor"],
                "LLM Advisor (14d)"))
        self._exec_report_btn.clicked.connect(
            lambda: self._run_report_clicked(
                ["execution_report.py", "--days", "14"],
                "Execution Report (14d)"))
        for btn in self._report_btns:
            reports_layout.addWidget(btn)
        self._reports_status = QLabel("")
        self._reports_status.setStyleSheet("font-size: 11px;")
        reports_layout.addWidget(self._reports_status)
        reports_layout.addStretch()
        reports_v.addLayout(reports_layout)
        # Freshness strip (c26 U1): one line of artifact ages, filled in
        # _refresh_reports_freshness (60s piggyback + after every report run).
        self._reports_fresh_label = QLabel("—")
        self._reports_fresh_label.setWordWrap(True)
        self._reports_fresh_label.setTextFormat(Qt.RichText)
        self._reports_fresh_label.setStyleSheet("font-size: 11px;")
        reports_v.addWidget(self._reports_fresh_label)

        layout.addWidget(reports_group)
        layout.addStretch()
        self.tabs.addTab(tab, "Models")

    # ---- Tab 7: Logs -----------------------------------------------------
    def _build_logs_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Row 1: file selector + auto-scroll (existing controls, unchanged)
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Log File:"))
        self._log_selector = QComboBox()
        self._log_selector.addItems(list(LOG_FILES.keys()))
        self._log_selector.currentTextChanged.connect(self._on_log_selected)
        ctrl_layout.addWidget(self._log_selector)

        self._auto_scroll = QCheckBox("Auto-scroll")
        self._auto_scroll.setChecked(True)
        ctrl_layout.addWidget(self._auto_scroll)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Row 2: regex filter + level combo + pause + jump-to-latest
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self._log_filter = QLineEdit()
        self._log_filter.setPlaceholderText(
            "regex (case-insensitive) — invalid pattern falls back to literal")
        self._log_filter.textChanged.connect(self._rerender_log_view)
        filter_layout.addWidget(self._log_filter, stretch=1)
        filter_layout.addWidget(QLabel("Level:"))
        self._log_level = QComboBox()
        self._log_level.addItems(["All", "Warning+", "Error"])
        self._log_level.currentTextChanged.connect(self._rerender_log_view)
        filter_layout.addWidget(self._log_level)
        self._log_paused = QCheckBox("Pause")
        filter_layout.addWidget(self._log_paused)
        self._log_jump_btn = QPushButton("Jump to latest")
        self._log_jump_btn.clicked.connect(self._log_jump_to_latest)
        filter_layout.addWidget(self._log_jump_btn)
        layout.addLayout(filter_layout)

        self._log_display = QPlainTextEdit()
        self._log_display.setReadOnly(True)
        self._log_display.setFont(QFont("Monospace", 10))
        self._log_display.setMaximumBlockCount(5000)
        layout.addWidget(self._log_display)

        self._log_buffers = {name: "" for name in LOG_FILES}
        self._on_log_selected(self._log_selector.currentText())

        self.tabs.addTab(tab, "Logs")
        self._logs_tab_index = self.tabs.indexOf(tab)

    # ---- Logs: severity coloring + filtering ----------------------------
    def _log_severity(self, line):
        """'error' | 'warning' | None for one raw log line."""
        if " ERROR" in line or "CRITICAL" in line or "Traceback" in line:
            return 'error'
        if " WARNING" in line:
            return 'warning'
        return None

    def _compile_log_filter(self):
        """Current filter as a compiled case-insensitive regex; a lowercased
        literal string when the pattern is invalid; or None when blank."""
        raw = self._log_filter.text().strip() if hasattr(self, '_log_filter') else ""
        if not raw:
            return None
        try:
            return re.compile(raw, re.IGNORECASE)
        except re.error:
            return raw.lower()  # invalid regex -> literal substring match

    def _log_line_passes(self, line, pattern, level):
        """Level (All / Warning+ / Error) + regex/literal filter for one line."""
        if level == "Error":
            if self._log_severity(line) != 'error':
                return False
        elif level == "Warning+":
            if self._log_severity(line) not in ('error', 'warning'):
                return False
        if pattern is None:
            return True
        if isinstance(pattern, str):
            return pattern in line.lower()
        return pattern.search(line) is not None

    def _append_log_line_html(self, line):
        """Append one line with severity coloring, monospace + spacing preserved
        (html.escape + a white-space:pre-wrap monospace span). Colors via T."""
        sev = self._log_severity(line)
        if sev == 'error':
            color = T['red'].name()
        elif sev == 'warning':
            color = T.get('yellow', T['white']).name()
        else:
            color = T['white'].name()
        safe = html.escape(line)
        self._log_display.appendHtml(
            f"<span style='color:{color}; white-space:pre-wrap;"
            f" font-family:monospace;'>{safe}</span>")

    def _log_scroll_to_bottom(self):
        sb = self._log_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _log_jump_to_latest(self):
        """Re-enable auto-scroll and scroll to the newest line."""
        self._auto_scroll.setChecked(True)
        self._log_scroll_to_bottom()

    def _rerender_log_view(self):
        """Re-render the visible view from the selected file's in-memory buffer
        (self._log_buffers — the getter), applying the regex + level filter.
        Called on file switch and whenever the filter or level changes."""
        if not hasattr(self, '_log_display'):
            return
        name = self._log_selector.currentText()
        buf = self._log_buffers.get(name, "")
        pattern = self._compile_log_filter()
        level = self._log_level.currentText() if hasattr(self, '_log_level') else "All"
        self._log_display.clear()
        for line in buf.split("\n"):
            if line == "":
                continue
            if self._log_line_passes(line, pattern, level):
                self._append_log_line_html(line)
        if self._auto_scroll.isChecked():
            self._log_scroll_to_bottom()

    def _build_settings_tab(self):
        from llm_config import load_llm_config, save_llm_config

        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area so the page resizes cleanly
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        config = load_llm_config()

        # --- LLM Configuration group ---
        llm_group = QGroupBox("LLM Configuration")
        llm_layout = QVBoxLayout(llm_group)
        llm_layout.setSpacing(6)

        # Enable + Provider row
        top_row = QHBoxLayout()
        self._settings_llm_enabled = QCheckBox("LLM Enabled")
        self._settings_llm_enabled.setChecked(config.get("enabled", True))
        self._settings_llm_enabled.toggled.connect(self._on_settings_changed)
        top_row.addWidget(self._settings_llm_enabled)
        top_row.addStretch()
        self._settings_journal = QCheckBox("Trade Journal")
        self._settings_journal.setChecked(config.get("journal_enabled", True))
        self._settings_journal.toggled.connect(self._on_settings_changed)
        top_row.addWidget(self._settings_journal)
        llm_layout.addLayout(top_row)

        # API Keys — compact grid with constrained widths
        keys_group = QGroupBox("API Keys")
        keys_layout = QGridLayout(keys_group)
        keys_layout.setColumnStretch(1, 1)
        keys_layout.setColumnMinimumWidth(0, 55)

        self._settings_api_keys = {}
        self._settings_key_toggles = {}
        # provider keys match config["models"] slots ("claude"/"openai") so the
        # existing _on_settings_changed save loop persists them unchanged.
        _key_env_hint = {
            "claude": "Falls back to the ANTHROPIC_API_KEY env var when empty.",
            "openai": "Falls back to the OPENAI_API_KEY env var when empty.",
        }
        for i, (provider, label) in enumerate([
            ("gemini", "Gemini"), ("claude", "Anthropic"),
            ("openai", "OpenAI"), ("fmp", "FMP"),
        ]):
            keys_layout.addWidget(QLabel(f"{label}:"), i, 0)
            key_edit = QLineEdit()
            key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            key_edit.setMaximumWidth(320)
            if provider == "fmp":
                key_edit.setPlaceholderText("Financial Modeling Prep key")
                key_edit.setText(config.get("fmp_api_key", ""))
            else:
                key_edit.setPlaceholderText(f"{label} API key")
                key_edit.setText(config.get("models", {}).get(provider, {}).get("api_key", ""))
                if provider in _key_env_hint:
                    key_edit.setToolTip(_key_env_hint[provider])
            key_edit.editingFinished.connect(self._on_settings_changed)
            keys_layout.addWidget(key_edit, i, 1)

            toggle_btn = QPushButton("Show")
            toggle_btn.setFixedWidth(54)
            toggle_btn.setCheckable(True)
            toggle_btn.toggled.connect(lambda checked, le=key_edit, btn=toggle_btn: (
                le.setEchoMode(QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password),
                btn.setText("Hide" if checked else "Show"),
            ))
            keys_layout.addWidget(toggle_btn, i, 2)

            if provider == "fmp":
                self._settings_fmp_key = key_edit
            else:
                self._settings_api_keys[provider] = key_edit
                self._settings_key_toggles[provider] = toggle_btn

        llm_layout.addWidget(keys_group)

        # --- Provider Selection (multi-provider engine, see llm_config.py) ---
        prov_group = QGroupBox("Provider Selection")
        prov_layout = QGridLayout(prov_group)
        prov_layout.setColumnStretch(1, 1)
        prov_layout.setColumnMinimumWidth(0, 70)

        # selection_mode — how llm_client.resolve_provider_chain builds its
        # candidate list (exact field values from llm_config.py's contract)
        prov_layout.addWidget(QLabel("Mode:"), 0, 0)
        self._settings_selection_mode = QComboBox()
        self._settings_selection_mode.setMaximumWidth(320)
        for mid, mlabel in [
            ("auto", "Auto (every keyed provider)"),
            ("single", "Single (primary only)"),
            ("free-only", "Free only"),
            ("best-free", "Best free"),
        ]:
            self._settings_selection_mode.addItem(mlabel, mid)
        cur_mode = config.get("selection_mode") or "auto"
        idx = self._settings_selection_mode.findData(cur_mode)
        if idx >= 0:
            self._settings_selection_mode.setCurrentIndex(idx)
        self._settings_selection_mode.currentIndexChanged.connect(self._on_settings_changed)
        prov_layout.addWidget(self._settings_selection_mode, 0, 1)

        # Primary provider — heads provider_preference (auto mode tries the
        # rest after it) and is the provider 'single' mode uses. Writes both
        # provider_preference (reordered) and the legacy 'provider' field.
        prov_layout.addWidget(QLabel("Primary:"), 1, 0)
        self._settings_primary_provider = QComboBox()
        self._settings_primary_provider.setMaximumWidth(320)
        for pid, plabel in [
            ("anthropic", "Anthropic (Claude)"),
            ("openai", "OpenAI (GPT)"),
            ("gemini", "Gemini"),
        ]:
            self._settings_primary_provider.addItem(plabel, pid)
        pref = config.get("provider_preference") or ["anthropic", "openai", "gemini"]
        cur_primary = (pref[0] if pref else "anthropic")
        if cur_primary == "claude":  # legacy alias for anthropic
            cur_primary = "anthropic"
        idx = self._settings_primary_provider.findData(cur_primary)
        if idx >= 0:
            self._settings_primary_provider.setCurrentIndex(idx)
        self._settings_primary_provider.currentIndexChanged.connect(self._on_settings_changed)
        prov_layout.addWidget(self._settings_primary_provider, 1, 1)

        prov_hint = QLabel("Auto tries every provider that has a key, primary "
                           "first. Backfill stays pinned to Gemini.")
        prov_hint.setStyleSheet("font-size: 10px;")
        prov_hint.setWordWrap(True)
        prov_layout.addWidget(prov_hint, 2, 0, 1, 2)

        llm_layout.addWidget(prov_group)

        # --- Tier Detection ---
        tier_group = QGroupBox("API Tier")
        tier_layout = QHBoxLayout(tier_group)

        tier_layout.addWidget(QLabel("Tier:"))
        self._settings_tier_override = QComboBox()
        self._settings_tier_override.setMaximumWidth(200)
        for tid, tlabel in [
            ("auto", "Auto-detect"),
            ("free", "Force Free"),
            ("paid", "Force Paid"),
        ]:
            self._settings_tier_override.addItem(tlabel, tid)
        cur_tier = config.get("tier_override") or "auto"
        idx = self._settings_tier_override.findData(cur_tier)
        if idx >= 0:
            self._settings_tier_override.setCurrentIndex(idx)
        self._settings_tier_override.currentIndexChanged.connect(self._on_settings_changed)
        tier_layout.addWidget(self._settings_tier_override)

        self._settings_tier_label = QLabel("")
        try:
            from llm_client import get_tier
            detected = get_tier()
            self._settings_tier_label.setText(f"(detected: {detected})")
        except Exception:
            pass
        tier_layout.addWidget(self._settings_tier_label)
        tier_layout.addStretch()

        llm_layout.addWidget(tier_group)

        # --- Smart Model Routing ---
        role_group = QGroupBox("Smart Model Routing")
        role_layout = QGridLayout(role_group)
        role_layout.setColumnStretch(1, 1)
        role_layout.setColumnMinimumWidth(0, 70)

        # Model combos offer only IDs the engine actually routes — the full
        # multi-provider universe (llm_client.get_recommended_model / the
        # *_model_override contract accept any id in KNOWN_MODELS; unknown ids
        # are silently ignored).
        from llm_client import GEMINI_MODELS, ANTHROPIC_MODELS, OPENAI_MODELS
        model_labels = {
            "gemini-2.5-pro": "Gemini 2.5 Pro (best reasoning)",
            "gemini-2.5-flash": "Gemini 2.5 Flash (fast, capable)",
            "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite (budget)",
            "claude-sonnet-5": "Claude Sonnet 5 (quality)",
            "claude-haiku-4-5": "Claude Haiku 4.5 (fast, cheap)",
            "gpt-5.4": "GPT-5.4 (quality)",
            "gpt-5.4-mini": "GPT-5.4 Mini",
            "gpt-5.4-nano": "GPT-5.4 Nano (budget)",
        }

        # Analyst model override (any provider's model, or auto smart routing)
        role_layout.addWidget(QLabel("Analyst:"), 0, 0)
        self._settings_analyst_model = QComboBox()
        self._settings_analyst_model.setMaximumWidth(320)
        for mid in ["auto"] + GEMINI_MODELS + ANTHROPIC_MODELS + OPENAI_MODELS:
            self._settings_analyst_model.addItem(
                "Auto (Smart Routing)" if mid == "auto"
                else model_labels.get(mid, mid), mid)
        cur_analyst = config.get("analyst_model_override") or "auto"
        idx = self._settings_analyst_model.findData(cur_analyst)
        if idx >= 0:
            self._settings_analyst_model.setCurrentIndex(idx)
        self._settings_analyst_model.currentIndexChanged.connect(self._on_settings_changed)
        role_layout.addWidget(self._settings_analyst_model, 0, 1)

        # Sentiment model override (Gemini Pro excluded — never routed for
        # scoring; Anthropic/OpenAI models available as explicit picks)
        role_layout.addWidget(QLabel("Sentiment:"), 1, 0)
        self._settings_sentiment_model = QComboBox()
        self._settings_sentiment_model.setMaximumWidth(320)
        for mid in (["auto"]
                    + [m for m in GEMINI_MODELS if m != "gemini-2.5-pro"]
                    + ANTHROPIC_MODELS + OPENAI_MODELS):
            self._settings_sentiment_model.addItem(
                "Auto (Smart Routing)" if mid == "auto"
                else model_labels.get(mid, mid), mid)
        cur_sentiment = config.get("sentiment_model_override") or "auto"
        idx = self._settings_sentiment_model.findData(cur_sentiment)
        if idx >= 0:
            self._settings_sentiment_model.setCurrentIndex(idx)
        self._settings_sentiment_model.currentIndexChanged.connect(self._on_settings_changed)
        role_layout.addWidget(self._settings_sentiment_model, 1, 1)

        # Routing status display
        self._settings_routing_label = QLabel("")
        try:
            from llm_client import get_routing_info
            info = get_routing_info()
            self._settings_routing_label.setText(
                f"Current: analyst={info['analyst_model'].split('-')[-1]}, "
                f"sentiment={info['sentiment_model'].split('-')[-1]} "
                f"(${info['daily_cost']:.3f}/${info['daily_limit']:.2f})")
        except Exception:
            pass
        role_layout.addWidget(self._settings_routing_label, 2, 0, 1, 2)

        llm_layout.addWidget(role_group)

        # Latency + Test row
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QLabel("Max Latency:"))
        self._settings_latency = QSpinBox()
        self._settings_latency.setRange(5, 60)
        self._settings_latency.setValue(config.get("max_llm_latency_sec", 15))
        self._settings_latency.setSuffix("s")
        self._settings_latency.setMaximumWidth(70)
        self._settings_latency.valueChanged.connect(self._on_settings_changed)
        bottom_row.addWidget(self._settings_latency)
        bottom_row.addSpacing(12)
        self._settings_test_btn = QPushButton("Test Connection")
        self._settings_test_btn.clicked.connect(self._on_test_llm)
        self._llm_test_done.connect(self._on_test_llm_done)
        bottom_row.addWidget(self._settings_test_btn)
        self._settings_test_status = QLabel("")
        bottom_row.addWidget(self._settings_test_status)
        bottom_row.addStretch()
        llm_layout.addLayout(bottom_row)

        layout.addWidget(llm_group)

        # --- Indicator Presets group ---
        from indicator_config import load_indicator_config, save_indicator_config, get_all_preset_info, PRESETS, CRYPTO_ONLY_COLS, STOCK_ONLY_COLS

        ind_group = QGroupBox("Indicator Presets")
        ind_layout = QVBoxLayout(ind_group)
        ind_layout.setSpacing(4)

        preset_info = get_all_preset_info()
        ind_config = load_indicator_config()
        current_preset = ind_config.get("preset", "standard")

        # Preset selector row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._settings_indicator_preset = QComboBox()
        self._settings_indicator_preset.setMaximumWidth(280)
        preset_labels = {
            "minimal": f"Minimal (~{preset_info['minimal']['count']} features)",
            "standard": f"Standard (~{preset_info['standard']['count']} features)",
            "stationary": f"Stationary (~{preset_info['stationary']['count']} features)",
            "full": "Full (all features)",
        }
        for name in ["minimal", "standard", "stationary", "full"]:
            self._settings_indicator_preset.addItem(preset_labels[name], name)
        idx = self._settings_indicator_preset.findData(current_preset)
        if idx >= 0:
            self._settings_indicator_preset.setCurrentIndex(idx)
        preset_row.addWidget(self._settings_indicator_preset)
        preset_row.addStretch()
        ind_layout.addLayout(preset_row)

        # Description
        self._indicator_desc_label = QLabel()
        self._indicator_desc_label.setWordWrap(True)
        self._indicator_desc_label.setStyleSheet("font-size: 11px;")
        ind_layout.addWidget(self._indicator_desc_label)

        # Feature list (compact)
        self._indicator_feature_list = QPlainTextEdit()
        self._indicator_feature_list.setReadOnly(True)
        self._indicator_feature_list.setMaximumHeight(72)
        self._indicator_feature_list.setFont(QFont("monospace", 8))
        ind_layout.addWidget(self._indicator_feature_list)

        # Cross-asset + warning on one line
        self._indicator_cross_note = QLabel()
        self._indicator_cross_note.setStyleSheet(
            f"color: {T['muted'].name()}; font-size: 10px;")
        self._indicator_cross_note.setWordWrap(True)
        ind_layout.addWidget(self._indicator_cross_note)

        warn_label = QLabel("Changing presets requires model retraining.")
        warn_label.setStyleSheet(
            f"color: {T['yellow'].name()}; font-weight: bold; font-size: 10px;")
        ind_layout.addWidget(warn_label)

        self._update_indicator_feature_display()
        self._settings_indicator_preset.currentIndexChanged.connect(
            self._on_indicator_preset_changed)

        layout.addWidget(ind_group)

        # --- Refresh cadences (Settings ops page, gui_review §8 / Phase 5.5) ---
        # Live Jetson-load levers: each spinbox retunes a poll timer. The main-
        # thread models timer is set directly; fetcher-thread timers via the
        # DataFetcher.set_interval slot invoked with invokeMethod. 'news' shows
        # minutes; everything else seconds.
        cad_group = QGroupBox("Refresh cadences")
        cad_grid = QGridLayout(cad_group)
        saved_cad = (_load_gui_settings().get('cadences', {}) or {})
        cad_specs = [  # (stream, label, is_minutes)
            ('positions', 'Positions', False), ('account', 'Account', False),
            ('orders', 'Orders', False), ('hw', 'Hardware', False),
            ('stocks', 'Stocks', False), ('news', 'News', True),
            ('models', 'Models', False),
        ]
        self._cadence_spins = {}
        for i, (stream, label, is_min) in enumerate(cad_specs):
            row, col = divmod(i, 2)
            lo, hi = CADENCE_BOUNDS[stream]
            spin = QSpinBox()
            cur = int(saved_cad.get(stream, DEFAULT_CADENCES[stream]))
            if is_min:  # 60-3600s displayed as 1-60 min
                spin.setRange(lo // 60, hi // 60)
                spin.setSuffix(" min")
                spin.setValue(max(lo // 60, min(hi // 60, cur // 60)))
            else:
                spin.setRange(lo, hi)
                spin.setSuffix(" s")
                spin.setValue(max(lo, min(hi, cur)))
            spin.setFixedWidth(92)
            spin.valueChanged.connect(
                lambda _v, s=stream, sp=spin, m=is_min:
                self._on_cadence_changed(s, sp, m))
            cad_grid.addWidget(QLabel(f"{label}:"), row, col * 2)
            cad_grid.addWidget(spin, row, col * 2 + 1)
            self._cadence_spins[stream] = (spin, is_min)
        cad_reset = QPushButton("Reset defaults")
        cad_reset.setFixedWidth(120)
        cad_reset.clicked.connect(self._reset_cadences)
        cad_grid.addWidget(cad_reset, (len(cad_specs) + 1) // 2, 0, 1, 4)
        layout.addWidget(cad_group)

        # --- Notifications (read-only env status + self-test) ---
        notif_group = QGroupBox("Notifications")
        notif_grid = QGridLayout(notif_group)
        notif_grid.setColumnStretch(1, 1)
        webhook_ok = bool(os.getenv('TRADER_WEBHOOK_URL'))
        tg_ok = bool(os.getenv('TRADER_TELEGRAM_BOT_TOKEN')
                     and os.getenv('TRADER_TELEGRAM_CHAT_ID'))
        hc_ok = bool(os.getenv('TRADER_HEALTHCHECK_URL')
                     or os.getenv('TRADER_HEALTHCHECK_URL_CRYPTO')
                     or os.getenv('TRADER_HEALTHCHECK_URL_STOCK'))
        for i, (name, ok) in enumerate([
                ("Webhook:", webhook_ok), ("Telegram:", tg_ok),
                ("Healthcheck:", hc_ok)]):
            notif_grid.addWidget(QLabel(name), i, 0)
            val = QLabel("configured" if ok else "—")
            val.setStyleSheet(
                f"color: {(T['green'] if ok else T['muted']).name()};"
                " font-weight: bold;")
            notif_grid.addWidget(val, i, 1)
        notif_test_row = QHBoxLayout()
        self._settings_notify_test_btn = QPushButton("Send test notification")
        self._settings_notify_test_btn.clicked.connect(self._on_test_notify)
        self._notify_test_done.connect(self._on_test_notify_done)
        notif_test_row.addWidget(self._settings_notify_test_btn)
        self._settings_notify_status = QLabel("")
        self._settings_notify_status.setStyleSheet("font-size: 11px;")
        notif_test_row.addWidget(self._settings_notify_status)
        notif_test_row.addStretch()
        notif_grid.addLayout(notif_test_row, 3, 0, 1, 2)
        layout.addWidget(notif_group)

        # --- Safe mode (shadow-mode state chip + halt-switch mirror) ---
        safe_group = QGroupBox("Safe mode")
        safe_row = QHBoxLayout(safe_group)
        shadow_on = bool(os.getenv('TRADER_SHADOW_MODE'))
        self._settings_shadow_chip = QLabel(
            "SHADOW MODE — orders suppressed" if shadow_on
            else "Live trading (shadow off)")
        self._settings_shadow_chip.setStyleSheet(
            f"color: {(T['yellow'] if shadow_on else T['muted']).name()};"
            " font-weight: bold; font-size: 12px;")
        safe_row.addWidget(self._settings_shadow_chip)
        safe_row.addStretch()
        self._settings_halt_btn = QPushButton(
            "Resume Entries" if halt_active() else "Halt Entries")
        self._settings_halt_btn.setToolTip(
            "Mirror of the Models-tab kill switch (entries only)")
        self._settings_halt_btn.clicked.connect(self._toggle_halt_clicked)
        safe_row.addWidget(self._settings_halt_btn)
        layout.addWidget(safe_group)

        # --- Chart defaults (persisted in gui_settings; applied at chart build,
        # and live where cheap) ---
        chart_group = QGroupBox("Chart defaults")
        chart_row = QHBoxLayout(chart_group)
        chart_row.addWidget(QLabel("Default zoom:"))
        self._settings_chart_zoom = QComboBox()
        for z in ("1D", "1W", "1M", "3M", "1Y"):
            self._settings_chart_zoom.addItem(z, z)
        _cz = _load_gui_settings().get('chart_default_zoom', '1M')
        _czi = self._settings_chart_zoom.findData(_cz)
        if _czi >= 0:
            self._settings_chart_zoom.setCurrentIndex(_czi)
        self._settings_chart_zoom.setFixedWidth(70)
        self._settings_chart_zoom.currentIndexChanged.connect(
            self._on_chart_zoom_default_changed)
        chart_row.addWidget(self._settings_chart_zoom)
        chart_row.addSpacing(16)
        self._settings_chart_grid = QCheckBox("Show grid")
        self._settings_chart_grid.setChecked(
            bool(_load_gui_settings().get('chart_grid', True)))
        self._settings_chart_grid.toggled.connect(self._on_chart_grid_toggled)
        chart_row.addWidget(self._settings_chart_grid)
        chart_row.addStretch()
        layout.addWidget(chart_group)

        layout.addStretch()

        scroll.setWidget(scroll_widget)
        tab_layout.addWidget(scroll)
        self.tabs.addTab(tab, "Settings")

    def _on_settings_changed(self, *_args):
        """Auto-save settings when any field changes."""
        from llm_config import load_llm_config, save_llm_config

        config = load_llm_config()
        config["enabled"] = self._settings_llm_enabled.isChecked()
        config["journal_enabled"] = self._settings_journal.isChecked()
        config["max_llm_latency_sec"] = self._settings_latency.value()
        config["fmp_api_key"] = self._settings_fmp_key.text().strip()

        for provider, key_edit in self._settings_api_keys.items():
            config.setdefault("models", {}).setdefault(provider, {})["api_key"] = key_edit.text().strip()

        # Provider selection engine (exact llm_config.py contract fields).
        # selection_mode governs how resolve_provider_chain builds candidates;
        # provider_preference is the 'auto'-mode order (primary first); the
        # legacy 'provider' field is what 'single' mode consults, so keep it
        # in sync with the chosen primary.
        config["selection_mode"] = self._settings_selection_mode.currentData()
        primary = self._settings_primary_provider.currentData()
        default_order = ["anthropic", "openai", "gemini"]
        config["provider_preference"] = (
            [primary] + [p for p in default_order if p != primary])
        config["provider"] = primary

        # Tier override
        tier = self._settings_tier_override.currentData()
        config["tier_override"] = None if tier == "auto" else tier

        # Model role overrides (None = smart routing); never persist an ID the
        # engine doesn't route — it would be silently ignored. KNOWN_MODELS
        # spans every provider (Gemini + Anthropic + OpenAI).
        from llm_client import KNOWN_MODELS
        analyst = self._settings_analyst_model.currentData()
        config["analyst_model_override"] = (
            analyst if analyst in KNOWN_MODELS else None)
        sentiment = self._settings_sentiment_model.currentData()
        config["sentiment_model_override"] = (
            sentiment if sentiment in KNOWN_MODELS else None)

        save_llm_config(config)

    def _on_test_llm(self):
        """Test LLM connection with a trivial prompt (off the UI thread)."""
        self._settings_test_btn.setEnabled(False)
        self._settings_test_status.setText("Testing...")
        self._settings_test_status.setStyleSheet("")

        # Force-save first so the client reads current keys
        # (touches widgets — must stay on the UI thread)
        self._on_settings_changed()

        import time
        import threading

        def probe():
            start = time.time()
            ok, error, model = False, "", ""
            try:
                from llm_client import call_llm, get_last_model_used
                result = call_llm("Respond with just the word OK.",
                                  max_tokens=16)
                ok = bool(result)
                if ok:
                    # Which provider/model actually answered (the chain may
                    # have fallen through several) — not a hardcoded "Gemini".
                    model = get_last_model_used() or ""
            except Exception as e:
                error = str(e)
            try:
                self._llm_test_done.emit(
                    ok, (time.time() - start) * 1000, error, model)
            except RuntimeError:
                pass  # window closed while the probe was in flight

        threading.Thread(target=probe, daemon=True, name="llm-test").start()

    def _llm_probe_identity(self, model):
        """(provider, model) label for the test result. Prefer the model that
        actually answered; fall back to the active config's resolved analyst
        candidate so we never claim a provider that didn't respond."""
        def _prov(m):
            try:
                from llm_client import _provider_for
                return _provider_for(m)
            except Exception:
                m = str(m)
                if m.startswith("claude"):
                    return "anthropic"
                if m.startswith("gpt") or m.startswith("o"):
                    return "openai"
                return "gemini"
        if model:
            return _prov(model), model
        try:
            from llm_config import load_llm_config
            from llm_client import resolve_provider_chain
            chain = resolve_provider_chain("analyst", load_llm_config())
            if chain:
                return chain[0][0], chain[0][1]
        except Exception:
            pass
        return "config", "unknown"

    @Slot(bool, float, str, str)
    def _on_test_llm_done(self, ok, elapsed_ms, error, model):
        if error:
            self._settings_test_status.setText(f"Error: {error}")
            self._settings_test_status.setStyleSheet(f"color: {T['red'].name()};")
        elif ok:
            prov, mdl = self._llm_probe_identity(model)
            self._settings_test_status.setText(
                f"Connected — {prov}/{mdl} ({elapsed_ms:.0f}ms)")
            self._settings_test_status.setStyleSheet(f"color: {T['green'].name()};")
        else:
            self._settings_test_status.setText("No response — check API key")
            self._settings_test_status.setStyleSheet(f"color: {T['red'].name()};")
        self._settings_test_btn.setEnabled(True)

    # ---- Settings ops page (refresh cadences / notifications / chart) --------
    def _on_cadence_changed(self, stream, spin, is_min):
        """Persist + live-apply one refresh cadence (Settings ops page)."""
        seconds = spin.value() * 60 if is_min else spin.value()
        settings = _load_gui_settings()
        settings.setdefault('cadences', {})[stream] = int(seconds)
        _save_gui_settings(settings)
        self._apply_cadence(stream, int(seconds))

    def _apply_cadence(self, stream, seconds):
        """Retune the live timer for `stream`: the main-thread models timer
        directly; fetcher-thread timers via DataFetcher.set_interval (invokeMethod
        so a QTimer is only ever touched from its owning thread)."""
        ms = max(1, int(seconds)) * 1000
        if stream == 'models':
            self._model_timer.start(ms)
            return
        from PySide6.QtCore import QMetaObject, Q_ARG
        target = (self._fetcher_hot
                  if stream in ('account', 'positions', 'orders', 'hw')
                  else self._fetcher_slow)
        QMetaObject.invokeMethod(
            target, "set_interval", Qt.QueuedConnection,
            Q_ARG(str, stream), Q_ARG(int, ms))

    def _reset_cadences(self):
        """Restore default refresh cadences: clear stored overrides, reset each
        spinbox (signals blocked), and apply the defaults live."""
        settings = _load_gui_settings()
        settings['cadences'] = {}
        _save_gui_settings(settings)
        for stream, (spin, is_min) in self._cadence_spins.items():
            default = DEFAULT_CADENCES[stream]
            spin.blockSignals(True)
            spin.setValue(default // 60 if is_min else default)
            spin.blockSignals(False)
            self._apply_cadence(stream, default)

    def _on_test_notify(self):
        """Fire a test notification via notify.notify() on a daemon thread; the
        result label reports which channels were targeted (or none configured)."""
        self._settings_notify_test_btn.setEnabled(False)
        self._settings_notify_status.setText("Sending...")
        self._settings_notify_status.setStyleSheet("")
        import threading
        import time as _t

        def worker():
            ok, detail = False, ""
            try:
                chans = []
                if os.getenv('TRADER_WEBHOOK_URL'):
                    chans.append('webhook')
                if (os.getenv('TRADER_TELEGRAM_BOT_TOKEN')
                        and os.getenv('TRADER_TELEGRAM_CHAT_ID')):
                    chans.append('telegram')
                if not chans:
                    detail = "No channels configured"
                else:
                    # Unique dedupe_key each click so notify()'s 10-min dedupe
                    # gate never silently swallows the test.
                    notify.notify(
                        f"GUI test notification {_t.strftime('%H:%M:%S')}",
                        level='info', dedupe_key=f'gui_test_{_t.time()}')
                    ok = True
                    detail = "Sent to: " + ", ".join(chans) + " (check device)"
            except Exception as e:
                detail = f"Error: {e}"
            try:
                self._notify_test_done.emit(ok, detail)
            except RuntimeError:
                pass  # window closed mid-send

        threading.Thread(target=worker, daemon=True, name="notify-test").start()

    @Slot(bool, str)
    def _on_test_notify_done(self, ok, detail):
        if str(detail).startswith("Error"):
            color = T['red']
        elif ok:
            color = T['green']
        else:
            color = T['muted']
        self._settings_notify_status.setText(detail)
        self._settings_notify_status.setStyleSheet(
            f"color: {color.name()}; font-size: 11px;")
        self._settings_notify_test_btn.setEnabled(True)

    def _on_chart_zoom_default_changed(self, _idx=None):
        """Persist the default chart zoom; live-apply it to the Markets chart."""
        zoom = self._settings_chart_zoom.currentData()
        settings = _load_gui_settings()
        settings['chart_default_zoom'] = zoom
        _save_gui_settings(settings)
        if getattr(self, '_stock_zoom_buttons', {}).get(zoom) is not None:
            self._on_zoom_clicked(zoom)

    def _on_chart_grid_toggled(self, checked):
        """Persist + live-toggle the Markets price-chart grid."""
        settings = _load_gui_settings()
        settings['chart_grid'] = bool(checked)
        _save_gui_settings(settings)
        if hasattr(self, '_stock_chart'):
            self._stock_chart.showGrid(x=checked, y=checked, alpha=0.3)

    def _on_indicator_preset_changed(self, _index):
        """Save new preset and update feature display."""
        from indicator_config import load_indicator_config, save_indicator_config
        preset_name = self._settings_indicator_preset.currentData()
        config = load_indicator_config()
        config["preset"] = preset_name
        save_indicator_config(config)
        self._update_indicator_feature_display()

    def _update_indicator_feature_display(self):
        """Update description, feature list, and cross-asset note for selected preset."""
        from indicator_config import get_all_preset_info, CRYPTO_ONLY_COLS, STOCK_ONLY_COLS
        preset_name = self._settings_indicator_preset.currentData()
        info = get_all_preset_info()
        p = info.get(preset_name, info["standard"])

        self._indicator_desc_label.setText(p["description"])

        if p["features"] is not None:
            self._indicator_feature_list.setPlainText(
                ", ".join(p["features"]))
        else:
            self._indicator_feature_list.setPlainText(
                "All available columns (varies by training data)")

        crypto_note = "+ Crypto adds: " + ", ".join(CRYPTO_ONLY_COLS)
        stock_note = "+ Stocks add: " + ", ".join(STOCK_ONLY_COLS)
        self._indicator_cross_note.setText(f"{crypto_note}\n{stock_note}")

    # ---- Signal Handlers -------------------------------------------------
    def _account_baseline(self):
        """(baseline_equity, is_approx). Reads account_baseline.json (written by
        on_history from the earliest equity point); falls back to 100_000 flagged
        approximate when the file is missing/corrupt — so Total P&L is honest
        after any reset/.clean_slate instead of hardcoding a $100k start."""
        try:
            with open(BASE_DIR / "account_baseline.json") as f:
                val = float(json.load(f).get("baseline_equity"))
            if math.isfinite(val) and val > 0:
                return val, False
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass
        return 100_000.0, True

    @Slot(dict)
    def on_account(self, data):
        self._account_cache = data
        equity = float(data["equity"])
        cash = float(data["cash"])
        buying_power = float(data["buying_power"])
        last_equity = float(data["last_equity"])
        day_pl = equity - last_equity
        baseline, approx = self._account_baseline()
        total_pl = equity - baseline

        day_pct = (day_pl / last_equity * 100.0) if last_equity else 0.0
        tot_pct = (total_pl / baseline * 100.0) if baseline else 0.0

        self._set_card(self._card_equity, fmt_money(equity))
        self._set_card(self._card_cash, fmt_money(cash))
        self._set_card(self._card_buying_power, fmt_money(buying_power))
        self._set_card(self._card_day_pl,
                       f"{fmt_money(day_pl)} ({fmt_pct(day_pct)})",
                       pnl_color(day_pl))
        self._set_card(self._card_total_pl,
                       f"{'~' if approx else ''}{fmt_money(total_pl)} "
                       f"({fmt_pct(tot_pct)})",
                       pnl_color(total_pl))

        now = dt.datetime.now(TZ_CENTRAL).strftime("%I:%M:%S %p")
        self._status_updated.setText(f"Last update: {now}")
        # Stamp this stream's health (zeroes only ITS OWN fails — no global
        # reset that would mask a dead news/stocks stream) and piggyback the
        # 10s API-label refresh here (no dedicated timer).
        self._stream_ok("account")
        self._refresh_api_health()

        # Cockpit landing refresh (banner + heartbeats + risk gauge + DD badge)
        # piggybacks the 10s account tick — no dedicated timer.
        self._refresh_cockpit()

        # Update sentiment indicator from the last off-thread fetch
        # (fetch_news refreshes it every 5 min — no HTTP on the UI thread)
        try:
            fng = self._last_fng
            if fng is not None:
                val = fng.get('value')
                label = fng.get('label', '')
                if val is None:
                    pass
                elif val <= 25:
                    color = T['red'].name()
                elif val <= 45:
                    color = T.get('yellow', T['white']).name()
                elif val >= 75:
                    color = T['green'].name()
                elif val >= 55:
                    color = T.get('yellow', T['white']).name()
                else:
                    color = T['white'].name()
                self._status_sentiment.setText(f"FnG: {val} ({label})")
                self._status_sentiment.setStyleSheet(f"color: {color};")
        except Exception:
            pass

    def _load_position_states(self):
        """Merged {symbol: {'hwm': price|None, 'trailing': bool|None}} from
        position_state.json (crypto) + stock_position_state.json (stock),
        mtime-guarded so on_positions reads each file at most once per change.

        base_loop._save_position_state stores 'hwm' = Position.high_water_mark
        (a PRICE — the highest midpoint seen since entry) and 'trailing' =
        Position.trailing_activated (a BOOLEAN flag, NOT a stop level; the
        trailing stop level itself is computed live in
        base_loop._desired_stop_for from the entry ATR and is never persisted).
        """
        files = (BASE_DIR / "position_state.json",
                 BASE_DIR / "stock_position_state.json")
        sig = []
        for p in files:
            try:
                sig.append(p.stat().st_mtime)
            except OSError:
                sig.append(None)
        sig = tuple(sig)
        if getattr(self, "_posstate_sig", None) == sig:
            return self._posstate_cache
        merged = {}
        for p in files:
            try:
                with open(p) as f:
                    data = json.load(f)
                hwm = data.get("hwm") or {}
                trailing = data.get("trailing") or {}
                for sym in set(hwm) | set(trailing):
                    merged[sym] = {"hwm": hwm.get(sym),
                                   "trailing": trailing.get(sym)}
            except (OSError, json.JSONDecodeError, AttributeError, TypeError):
                continue
        self._posstate_sig = sig
        self._posstate_cache = merged
        return merged

    def _exit_levels_raw(self, symbol, entry_price, current_price, pstates):
        """Raw (entry, est_stop, est_tp) price floats for a position — the
        single source of the estimate math shared by _compute_exit_levels (the
        positions-table display strings) and _update_position_lines (the price-
        chart guide lines). Any component that can't be computed comes back as
        None. Stop = entry×(1−stop_floor_pct), ratcheted toward hwm×(1−...) when
        trailing is active; TP = entry×(1+tp_rr×stop_floor_pct)."""
        try:
            entry = float(entry_price)
        except (TypeError, ValueError):
            return None, None, None
        if not (entry > 0):
            return None, None, None
        pol = CRYPTO_POLICY if "/" in str(symbol) else STOCK_POLICY
        floor = pol.get("stop_floor_pct")
        tp_rr = pol.get("tp_rr")
        if not floor or floor <= 0:
            return entry, None, None
        st = pstates.get(symbol, {})
        est_stop = entry * (1 - floor)
        if st.get("trailing"):
            try:
                h = float(st.get("hwm"))
                if h > 0:
                    est_stop = max(est_stop, h * (1 - floor))
            except (TypeError, ValueError):
                pass
        est_tp = entry * (1 + tp_rr * floor) if tp_rr else None
        return entry, est_stop, est_tp

    def _compute_exit_levels(self, symbol, entry_price, current_price, pstates):
        """(stop_disp, tp_disp, dist_disp, dist_close) exit-distance cells for
        one position. Every level is an ESTIMATE ('~'): the live stop is
        ATR-based and only its high-water mark + trailing-activated flag are
        persisted, never the stop level itself (see _load_position_states).
        Missing/invalid data -> '—'. Level math lives in _exit_levels_raw."""
        # Preserve original semantics: an unparseable current_price returns all
        # dashes (the old code parsed entry+cur together and bailed on either).
        try:
            cur = float(current_price)
        except (TypeError, ValueError):
            return "—", "—", "—", False
        entry, est_stop, est_tp = self._exit_levels_raw(
            symbol, entry_price, current_price, pstates)
        if entry is None or est_stop is None:
            return "—", "—", "—", False
        stop_disp = f"~{fmt_money(est_stop)}"
        tp_disp = f"~{fmt_money(est_tp)}" if est_tp is not None else "—"
        dist_disp, dist_close = "—", False
        if cur > 0:
            dist = (cur - est_stop) / cur * 100.0
            dist_disp = f"~{dist:+.1f}%"
            dist_close = dist < 1.0
        return stop_disp, tp_disp, dist_disp, dist_close

    # ---- Positions-table diff update (5.3) --------------------------------
    def _make_pos_row(self, row, sym):
        """Create the 11 display cells of a new positions row once. Numeric cols
        (Qty/Avg/Cur/MktVal/Unr/PnL%) are NumericTableItem (UserRole sort key set
        in _update_pos_row + tabular figures); text cols (Symbol/Side/Stop/TP/
        %→Stop) are plain items (text sort) — matching the old rebuild's types.
        The col-11 Close button is (re)built by _rebuild_pos_buttons."""
        tbl = self._positions_table
        numeric_cols = (1, 3, 4, 5, 6, 7)
        for col in range(11):
            item = NumericTableItem("") if col in numeric_cols else QTableWidgetItem("")
            item.setTextAlignment(Qt.AlignCenter)
            tbl.setItem(row, col, item)
        tbl.item(row, 0).setText(sym)  # Symbol is the stable row identity

    def _update_pos_row(self, row, p, pstates):
        """Update cols 0..10 of an existing positions row in place; returns the
        row's unrealized P&L (float) for the footer total. Mirrors the old full-
        rebuild's values / colors / sort-keys exactly (P&L cols pnl-colored;
        %→Stop red when <1% from the estimated stop)."""
        tbl = self._positions_table

        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        unr = _f(p.get("unrealized_pl")) or 0.0
        cur_px = _f(p.get("current_price"))
        qty = _f(p.get("qty"))
        mkt_val = (qty * cur_px) if (qty is not None and cur_px is not None) else 0.0
        plpc = _f(p.get("unrealized_plpc"))
        pnl_pct = plpc * 100 if plpc is not None else 0.0
        stop_disp, tp_disp, dist_disp, dist_close = self._compute_exit_levels(
            p["symbol"], p["avg_entry_price"], p["current_price"], pstates)
        color = pnl_color(p["unrealized_pl"])
        white, red = T["white"], T["red"]
        self._set_cell(tbl.item(row, 0), str(p["symbol"]), white)
        self._set_cell(tbl.item(row, 1), str(p["qty"]), white, sort_key=qty)
        self._set_cell(tbl.item(row, 2), str(p["side"]), white)
        self._set_cell(tbl.item(row, 3), fmt_money(p["avg_entry_price"]), white,
                       sort_key=_f(p.get("avg_entry_price")))
        self._set_cell(tbl.item(row, 4), fmt_money(p["current_price"]), white,
                       sort_key=cur_px)
        self._set_cell(tbl.item(row, 5), fmt_money(mkt_val), white, sort_key=mkt_val)
        self._set_cell(tbl.item(row, 6), fmt_money(p["unrealized_pl"]), color,
                       sort_key=unr)
        self._set_cell(tbl.item(row, 7), fmt_pct(pnl_pct), color, sort_key=pnl_pct)
        self._set_cell(tbl.item(row, 8), stop_disp, white)
        self._set_cell(tbl.item(row, 9), tp_disp, white)
        self._set_cell(tbl.item(row, 10), dist_disp, red if dist_close else white)
        return unr

    def _make_close_button(self, sym):
        """One Close button bound to `sym` — the lambda captures the SYMBOL (not
        a row), so it stays correct regardless of sort. Styled from the shared
        pnl-down palette (identical to the old inline button)."""
        btn = QPushButton("Close")
        btn.setFixedHeight(24)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {PAL['down']};"
            f" color: {_on_color(PAL['down'])}; font-size: 10px; font-weight: bold;"
            f" border-radius: 3px; padding: 2px 6px; }}"
            f" QPushButton:hover {{"
            f" background-color: {chart_core.mix(PAL['down'], '#ffffff', 0.15)}; }}")
        btn.clicked.connect(lambda _, s=sym: self._close_position(s))
        return btn

    def _rebuild_pos_buttons(self):
        """(Re)create the col-11 Close button for every positions row in current
        visual order. Qt cell widgets don't follow item re-sorts and can't be
        safely relocated (setCellWidget deletes whatever widget is already in the
        target cell), so recreate the whole column, reading each row's symbol
        from the already-sorted items. Called only when the row->symbol layout
        changed — never on a plain value tick, so the common case keeps its
        persistent buttons untouched."""
        tbl = self._positions_table
        self._pos_close_btn = {}
        order = []
        for r in range(tbl.rowCount()):
            it = tbl.item(r, 0)
            sym = it.text() if it else None
            order.append(sym)
            if not sym:
                continue
            btn = self._make_close_button(sym)
            tbl.setCellWidget(r, 11, btn)  # deletes any prior widget in this cell
            self._pos_close_btn[sym] = btn
        self._pos_btn_order = order

    @Slot(list)
    def on_positions(self, positions):
        self._stream_ok("positions")
        self._positions_cache = positions
        tbl = self._positions_table

        # Diff-update (5.3): mirror the stance-table pattern — insert/remove rows
        # only on membership change, update cells in place, keep ONE persistent
        # Close button per symbol, and preserve the user's selected SYMBOL +
        # scroll. Replaces the old full teardown that recreated every Close
        # button + dropped selection 12x/min.
        sel_sym = None
        cur = tbl.currentRow()
        if cur >= 0:
            it = tbl.item(cur, 0)
            if it:
                sel_sym = it.text()
        vbar = tbl.verticalScrollBar()
        scroll_val = vbar.value() if vbar is not None else 0

        # Sorting OFF during structural + index-addressed edits.
        tbl.setSortingEnabled(False)
        tbl.setUpdatesEnabled(False)

        # Exit-distance estimates need position_state.json — read once per tick
        # (mtime-guarded inside), not once per row.
        pstates = self._load_position_states()

        want = [p["symbol"] for p in positions]
        want_set = set(want)
        by_sym = {p["symbol"]: p for p in positions}

        # Current sym -> row from the live table (survives user re-sorts).
        existing = {}
        for r in range(tbl.rowCount()):
            it = tbl.item(r, 0)
            if it:
                existing[it.text()] = r

        # Drop rows whose symbol left (descending so lower indices stay valid);
        # removeRow deletes that row's Close-button cell widget with it.
        stale_rows = sorted((r for s, r in existing.items() if s not in want_set),
                            reverse=True)
        for r in stale_rows:
            tbl.removeRow(r)
        if stale_rows:
            existing = {}
            for r in range(tbl.rowCount()):
                it = tbl.item(r, 0)
                if it:
                    existing[it.text()] = r

        # Append brand-new symbols (cells created once; filled in place below).
        for sym in want:
            if sym not in existing:
                r = tbl.rowCount()
                tbl.insertRow(r)
                self._make_pos_row(r, sym)
                existing[sym] = r

        # In-place cell updates + running unrealized-P&L total.
        total_unr = 0.0
        for sym in want:
            r = existing.get(sym)
            if r is None:
                continue
            total_unr += self._update_pos_row(r, by_sym[sym], pstates)

        self._pos_row_by_sym = existing

        tbl.setUpdatesEnabled(True)
        tbl.setSortingEnabled(True)  # re-sorts appended rows into user order

        # Qt cell widgets don't follow item re-sorts, so (re)build the Close-
        # button column ONLY when the visual row->symbol order actually changed
        # (a membership change, or a user sort on a volatile column reordered
        # rows). A plain 5s value tick under the default symbol sort leaves the
        # persistent buttons untouched — the whole point of the diff update.
        new_order = [tbl.item(r, 0).text() if tbl.item(r, 0) else None
                     for r in range(tbl.rowCount())]
        if new_order != getattr(self, '_pos_btn_order', None):
            self._rebuild_pos_buttons()

        # Restore selection by symbol (row may have moved on re-sort) + scroll.
        if sel_sym is not None:
            for r in range(tbl.rowCount()):
                it = tbl.item(r, 0)
                if it and it.text() == sel_sym:
                    tbl.blockSignals(True)
                    tbl.setCurrentCell(r, 0)
                    tbl.blockSignals(False)
                    break
        if vbar is not None:
            vbar.setValue(scroll_val)

        self._status_positions.setText(
            f"Pos: {len(positions)} | Unr: {fmt_money(total_unr)}")

        # Flatten-pending indicator: a GUI flatten sets the halt flag AND the
        # flatten flag, but only a live bot liquidates — surface the in-flight
        # state, and announce completion when positions reach zero.
        try:
            n = len(positions)
            if notify.flatten_requested() and n > 0:
                self._flatten_was_pending = True
                self._flatten_banner.setText(f"FLATTEN PENDING ({n} positions)")
                self._flatten_banner.setStyleSheet(
                    f"color: {T['red'].name()}; font-weight: bold;")
            else:
                if getattr(self, '_flatten_was_pending', False) and n == 0:
                    self.statusBar().showMessage(
                        "Flatten complete — trading halted", 10000)
                    self._push_alert('flatten-complete',
                                     "Flatten complete — trading halted")
                self._flatten_was_pending = False
                self._flatten_banner.setText("")
        except Exception:
            pass

        # Positions drive the open-risk gauge (counts + largest name); refresh
        # it on the 5s positions tick so it doesn't lag the 10s account tick.
        try:
            self._refresh_risk_gauge()
        except Exception:
            pass

        # Markets-tab "Held" column tracks holdings on the same 5s tick — cheap
        # targeted cell update, not a stance-table rebuild.
        self._refresh_held_cells()

        # Refresh the price-chart position guide lines on the positions tick
        # (a fill/exit can appear between chart repaints).
        try:
            self._update_position_lines()
        except Exception:
            pass

    @Slot(list, bool)
    def on_orders(self, orders, truncated=False):
        self._stream_ok("orders")
        self._orders_cache = orders
        self._orders_truncated = truncated
        self._apply_trade_filter(self._trade_filter.currentText())
        self._update_tax(orders)

    def _maybe_write_baseline(self, data):
        """Write account_baseline.json from the earliest positive equity point of
        the full-range history — only if missing or off by >$1. Atomic tmp-rename
        (matches _write_pipeline_command); the on_history caller wraps errors."""
        eq = [float(v) for v in (data.get("equity") or []) if v is not None]
        base = next((v for v in eq if math.isfinite(v) and v > 0), None)
        if base is None:
            return
        path = BASE_DIR / "account_baseline.json"
        try:
            old = float(json.loads(path.read_text()).get("baseline_equity"))
            if abs(old - base) <= 1.0:
                return
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"baseline_equity": base, "ts": time.time()}, f)
        os.replace(tmp, str(path))

    @Slot(dict)
    def on_history(self, data):
        self._stream_ok("history")
        period = data.get("period", "1M")
        data["_fetched_at"] = time.monotonic()
        self._perf_history_cache[period] = data

        # Cockpit today-sparkline is fed by a dedicated ("1D","15Min") fetch
        # (period "1D" is never a zoom target, so it won't reach the equity plot
        # via _apply_perf_data below).
        if period == "1D":
            try:
                self._repaint_today_sparkline(data)
            except Exception:
                pass
        # Any perf arrival can move the live-DD badge (longest cached series).
        try:
            self._refresh_dd_badge()
        except Exception:
            pass

        # The longest zoom carries the whole account history — capture its start
        # equity as the honest Total-P&L baseline (consumed by _account_baseline).
        if period == "1A":
            try:
                self._maybe_write_baseline(data)
            except Exception:
                pass

        # Only apply if this period matches the current zoom
        api_period, _ = self._perf_api_period()
        if period == api_period:
            self._apply_perf_data(data)

    def _apply_perf_data(self, data):
        view = chart_core.build_equity_view(data)

        if view.status.status in ('ok', 'partial'):
            # Stash the view so the benchmark overlay can re-align even when the
            # fingerprint memo below short-circuits the repaint.
            self._last_equity_view = view
            if self._chart_fp.get('perf') == view.fingerprint:
                if view.status.status == 'ok':
                    self._chart_last_ok['perf'] = view.status.updated_at
                self._set_chart_status(self._equity_plot, 'Equity Curve', view.status)
                self._maybe_request_benchmark()
                self._apply_benchmark_overlay()
                return
            self._chart_fp['perf'] = view.fingerprint

            # Preserve a manual pan/zoom across the refresh; else snap to window.
            saved_x = None
            if getattr(self, '_perf_user_viewport', False):
                saved_x = self._equity_plot.getViewBox().viewRange()[0]

            self._equity_curve.setData(view.ts, view.equity)
            self._equity_hwm.setData(view.ts, view.hwm)
            # Bound x-pan to the loaded series (+/-5%); y stays on auto-range.
            if len(view.ts) >= 2:
                x0f, x1f = float(view.ts[0]), float(view.ts[-1])
                m = (x1f - x0f) * 0.05 or 86400.0
                self._equity_plot.getViewBox().setLimits(xMin=x0f - m, xMax=x1f + m)
            if saved_x is not None:
                self._equity_plot.setXRange(*saved_x, padding=0)
            elif view.x_range is not None:
                self._equity_plot.setXRange(*view.x_range, padding=0.02)

            pos = view.pnl >= 0
            neg = ~pos
            self._pnl_bars_pos.setOpts(
                x=view.pnl_ts[pos], height=view.pnl[pos],
                width=view.pnl_widths[pos], brush=self._chart_pal['up'])
            self._pnl_bars_neg.setOpts(
                x=view.pnl_ts[neg], height=view.pnl[neg],
                width=view.pnl_widths[neg], brush=self._chart_pal['down'])

            s = view.stats or {}
            self._set_card(self._stat_return, fmt_money(s.get('total_return', 0)),
                            pnl_color(s.get('total_return', 0)))
            self._set_card(self._stat_best, fmt_money(s.get('best_day', 0)), T["green"])
            self._set_card(self._stat_worst, fmt_money(s.get('worst_day', 0)), T["red"])
            max_dd = s.get('max_dd_pct', 0)
            self._set_card(self._stat_drawdown, f"-{max_dd:.2f}%",
                           T["red"] if max_dd > 0 else T["white"])

            # Risk-adjusted tiles ("—" when perf_stats couldn't compute them).
            def _ratio(v):
                return f"{v:.2f}" if v is not None else "—"

            def _pctv(v):
                return f"{v * 100:.1f}%" if v is not None else "—"

            wr = s.get('win_rate')
            cg = s.get('cagr')
            self._set_card(self._stat_sharpe, _ratio(s.get('sharpe')), T["white"])
            self._set_card(self._stat_sortino, _ratio(s.get('sortino')), T["white"])
            self._set_card(self._stat_winrate,
                           f"{wr * 100:.0f}%" if wr is not None else "—", T["white"])
            self._set_card(self._stat_vol, _pctv(s.get('volatility')), T["white"])
            self._set_card(self._stat_cagr, _pctv(cg),
                           pnl_color(cg) if cg is not None else T["white"])

            self._equity_xhair.set_series(view.ts, view.equity)
            self._chart_last_ok['perf'] = view.status.updated_at
            self._maybe_request_benchmark()
            self._apply_benchmark_overlay()
        else:
            if not self._chart_fp.get('perf'):
                self._equity_curve.clear()
                self._equity_hwm.clear()
                self._pnl_bars_pos.setOpts(x=[], height=[], width=1)
                self._pnl_bars_neg.setOpts(x=[], height=[], width=1)

        self._set_chart_status(self._equity_plot, 'Equity Curve', view.status)
        self._set_chart_status(self._pnl_plot, 'Daily P&L', view.status)

    @Slot(dict)
    def on_hw(self, data):
        self._stream_ok("hw")
        self._hw_cache = data

        def _temp_color(t):
            if t < 60: return T["green"].name()
            if t < 70: return T["yellow"].name()
            return T["red"].name()

        def _pct_color(p):
            if p < 80: return T["green"].name()
            if p < 90: return T["yellow"].name()
            return T["red"].name()

        def _set_gauge(label, bar, text, pct, color, font_size=22):
            label.setText(text)
            label.setStyleSheet(f"font-size: {font_size}px; font-weight: bold; color: {color};")
            bar.setValue(int(min(max(pct, 0), 100)))
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")

        # --- GPU Temp ---
        gpu_temp = data.get("gpu_temp")
        if gpu_temp is not None:
            _set_gauge(self._gpu_temp_label, self._gpu_temp_bar,
                       f"{gpu_temp:.0f}\u00b0C", gpu_temp, _temp_color(gpu_temp))

        # --- GPU Load ---
        gpu_load = data.get("gpu_load")
        if gpu_load is not None:
            _set_gauge(self._gpu_load_label, self._gpu_load_bar,
                       f"{gpu_load:.0f}%", gpu_load, _pct_color(gpu_load))

        # --- GPU Clock ---
        gpu_freq = data.get("gpu_freq_mhz")
        gpu_max = data.get("gpu_max_freq_mhz")
        if gpu_freq is not None and gpu_max:
            pct = gpu_freq / gpu_max * 100
            _set_gauge(self._gpu_clock_label, self._gpu_clock_bar,
                       f"{gpu_freq:.0f}/{gpu_max:.0f} MHz", pct, T["accent"].name())

        # --- CPU Temp ---
        cpu_temp = data.get("cpu_temp")
        if cpu_temp is not None:
            _set_gauge(self._cpu_temp_label, self._cpu_temp_bar,
                       f"{cpu_temp:.0f}\u00b0C", cpu_temp, _temp_color(cpu_temp))

        # --- CPU Load ---
        cpu_usage = data.get("cpu_usage")
        if cpu_usage is not None:
            _set_gauge(self._cpu_load_label, self._cpu_load_bar,
                       f"{cpu_usage:.0f}%", cpu_usage, _pct_color(cpu_usage))

        # --- Shared Memory ---
        used = data.get("ram_used")
        total = data.get("ram_total")
        if used is not None and total is not None:
            pct = int(used / total * 100) if total else 0
            _set_gauge(self._ram_label, self._ram_bar,
                       f"{used:.0f}/{total:.0f} MB", pct, _pct_color(pct))

        # --- Disk (repo/base volume) ---
        disk_total = data.get("disk_total")
        disk_used = data.get("disk_used")
        disk_free = data.get("disk_free")
        if disk_total and disk_used is not None:
            dpct = disk_used / disk_total * 100 if disk_total else 0
            free_gb = (disk_free or 0) / 1e9
            # Red when <1 GB free (a full SD silently breaks status writes),
            # otherwise the standard pct bands.
            dcolor = T["red"].name() if free_gb < 1.0 else _pct_color(dpct)
            _set_gauge(self._disk_label, self._disk_bar,
                       f"{dpct:.0f}% · {free_gb:.1f}GB free", dpct, dcolor,
                       font_size=15)

        # --- Sparkline history (fingerprint-skipped when unchanged) ---
        if gpu_temp is not None:
            self._hw_gpu_temp_hist.append(float(gpu_temp))
        if used is not None and total:
            self._hw_ram_hist.append(used / total * 100.0)
        self._update_spark('gpu_temp', getattr(self, '_gpu_temp_spark', None),
                           self._hw_gpu_temp_hist)
        self._update_spark('ram', getattr(self, '_ram_spark', None),
                           self._hw_ram_hist)

        # --- Status bar ---
        if gpu_temp is not None:
            self._status_gpu.setText(f"GPU: {gpu_temp:.0f}\u00b0C")
        else:
            self._status_gpu.setText("GPU: N/A")

        if used is not None and total is not None:
            self._status_ram.setText(f"RAM: {used:.0f}/{total:.0f} MB")
        else:
            self._status_ram.setText("RAM: N/A")

        # GPU load + clock in status bar
        if gpu_load is not None and gpu_freq is not None:
            self._status_gpu_info.setText(f"GPU: {gpu_load:.0f}% @ {gpu_freq:.0f}MHz")
        elif gpu_load is not None:
            self._status_gpu_info.setText(f"GPU: {gpu_load:.0f}%")
        else:
            self._status_gpu_info.setText("GPU: idle")

    def _update_spark(self, key, curve, hist):
        """Redraw a HW sparkline only when its rounded series changed (memo)."""
        if curve is None or not hist:
            return
        ys = [round(float(v), 1) for v in hist]
        if self._spark_fp.get(key) == ys:
            return
        self._spark_fp[key] = ys
        curve.setData(list(range(len(ys))), ys)

    @Slot(dict)
    def on_news(self, data):
        self._stream_ok("news")
        import datetime as _dt
        articles = data.get('articles', [])
        fng = data.get('fng')
        if fng is not None:
            # cache for UI-thread consumers (on_account renders from this;
            # a failed fetch must not wipe the last good value)
            self._last_fng = fng
        cnn_fng = data.get('cnn_fng')
        sent_24h = data.get('sent_24h')
        sent_7d = data.get('sent_7d')

        def _fng_color(val):
            if val <= 25:
                return T['red'].name()
            elif val >= 75:
                return T['green'].name()
            return T.get('yellow', T['white']).name()

        def _sent_color(val):
            if val > 0.05:
                return T['green'].name()
            elif val < -0.05:
                return T['red'].name()
            return T.get('yellow', T['white']).name()

        def _update_sent(label, prefix, val):
            if val is not None:
                c = _sent_color(val)
                label.setText(f"{prefix}: <span style='color:{c};'>{val:+.2f}</span>")

        # Crypto row: FnG + 24h + 7d
        if fng is not None:
            val = fng.get('value')
            label = fng.get('label', '')
            if val is not None:
                c = _fng_color(val)
                self._news_crypto_fng.setText(
                    f"FnG: <span style='color:{c};'>{val} ({label})</span>")
        _update_sent(self._news_crypto_24h, "24h", data.get('crypto_24h'))
        _update_sent(self._news_crypto_7d, "7d", data.get('crypto_7d'))

        # Stock row: FnG + VIX + 24h + 7d
        if cnn_fng is not None:
            sv = cnn_fng.get('score')
            rating = cnn_fng.get('rating', '')
            if sv is not None:
                sc = _fng_color(sv)
                self._news_stock_fng.setText(
                    f"FnG: <span style='color:{sc};'>{sv:.0f} ({rating})</span>")
            vix = cnn_fng.get('vix', 0)
            if vix > 25:
                vc = T['red'].name()
            elif vix < 15:
                vc = T['green'].name()
            else:
                vc = T.get('yellow', T['white']).name()
            self._news_vix.setText(
                f"VIX: <span style='color:{vc};'>{vix:.1f}</span>")
        _update_sent(self._news_stock_24h, "24h", data.get('stock_24h'))
        _update_sent(self._news_stock_7d, "7d", data.get('stock_7d'))

        # Combined row
        _update_sent(self._news_sent_24h, "24h", sent_24h)
        _update_sent(self._news_sent_7d, "7d", sent_7d)

        now = _dt.datetime.now(TZ_CENTRAL).strftime("%I:%M %p")
        self._news_refresh_label.setText(f"Updated {now}")

        # Cache articles and apply current filter
        self._news_articles = articles
        self._news_fng = fng
        self._apply_news_filter()

    def _apply_news_filter(self):
        """Filter cached news articles based on combo selection."""
        import datetime as _dt
        from stock_config import CRYPTO_SYMBOLS
        from stock_config import load_stock_universe

        idx = self._news_filter_combo.currentIndex()
        articles = getattr(self, '_news_articles', [])

        if idx == 0:  # My Universe
            # Cache universe set to avoid re-reading file on every filter call
            if not hasattr(self, '_news_universe_cache') or self._news_universe_age < _dt.datetime.now().timestamp() - 60:
                crypto_bases = {s.split('/')[0] for s in CRYPTO_SYMBOLS}
                stock_set = set(load_stock_universe())
                self._news_universe_cache = crypto_bases | stock_set
                self._news_universe_age = _dt.datetime.now().timestamp()
            universe = self._news_universe_cache
            filtered = []
            for a in articles:
                sym = a.get('_symbol', '')
                if sym and sym in universe:
                    filtered.append(a)
                    continue
                # Scan headline for any universe symbol (word boundary)
                headline = a.get('headline', '') + ' ' + a.get('summary', '')
                headline_upper = headline.upper()
                for s in universe:
                    # Only prose-scan tickers of length >=3: 1-2 char symbols
                    # (A, ON, IT, F) false-match ordinary words in a headline.
                    # The exact _symbol tag above still covers short tickers.
                    if len(s) < 3:
                        continue
                    if re.search(r'\b' + re.escape(s) + r'\b', headline_upper):
                        filtered.append(a)
                        break
            articles = filtered
        elif idx == 2:  # Global / Macro
            articles = [a for a in articles if a.get('_category') == 'Market']
        elif idx == 3:  # Crypto (crypto-specific + global/macro)
            articles = [a for a in articles
                        if a.get('_category') in ('Crypto', 'Market')]
        elif idx == 4:  # Stocks (stock-specific + global/macro)
            articles = [a for a in articles
                        if a.get('_category') in ('Stock', 'Market')]
        # idx == 1 (All News) — no filtering

        self._news_filtered = articles  # store for click lookup

        tbl = self._news_table
        tbl.setUpdatesEnabled(False)
        # Sorting OFF during the index-addressed rebuild, then restored so the
        # user's chosen sort column re-applies without misplacing items.
        tbl.setSortingEnabled(False)
        display_articles = articles[:150]  # cap table rows to avoid UI stall
        tbl.setRowCount(len(display_articles))
        for row, a in enumerate(display_articles):
            ts = a.get('datetime', 0)
            if ts:
                time_str = _dt.datetime.fromtimestamp(ts, tz=TZ_CENTRAL).strftime("%m/%d %I:%M %p")
            else:
                time_str = "—"

            source = a.get('source', '—')
            category = a.get('_category', '—')
            sym = a.get('_symbol', '') or ''   # blank for macro/general news
            headline = a.get('headline', '—')
            summary = a.get('summary', '')
            sentiment = a.get('_sentiment') or 0.0
            sent_method = a.get('_sent_method', '')

            if sentiment > 0.1:
                sent_color = T['green']
                sent_text = f"+{sentiment:.2f}"
            elif sentiment < -0.1:
                sent_color = T['red']
                sent_text = f"{sentiment:.2f}"
            else:
                sent_color = T.get('yellow', T['white'])
                sent_text = f"{sentiment:.2f}"
            if sent_method:
                sent_text = f"{sent_text} ({sent_method})"

            # Tooltip: show summary on hover for every cell in the row
            tooltip = summary if summary else ''

            # Time (col 0) + Sentiment (col 5) are NumericTableItems so they sort
            # by value (epoch / raw sentiment), not by the display string — Time
            # by timestamp keeps the default Time-desc order chronological, and
            # Sentiment sorts by the float behind "+0.42 (LLM)".
            items_data = [time_str, source, category, sym, headline, sent_text]
            for col, val in enumerate(items_data):
                item = (NumericTableItem(str(val)) if col in (0, 5)
                        else QTableWidgetItem(str(val)))
                if col == 5:  # Sentiment
                    item.setForeground(sent_color)
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setData(Qt.UserRole, float(sentiment))
                elif col <= 3:  # Time, Source, Category, Sym
                    item.setTextAlignment(Qt.AlignCenter)
                if tooltip:
                    item.setToolTip(tooltip)
                if col == 0:  # timestamp sort key + URL for a sort-safe row click
                    item.setData(Qt.UserRole, float(ts or 0))
                    item.setData(NEWS_URL_ROLE, a.get('url', '') or '')
                tbl.setItem(row, col, item)
        tbl.setSortingEnabled(True)
        tbl.setUpdatesEnabled(True)

    def _on_news_row_clicked(self, row, _col):
        """Open the article URL in the system browser. The URL rides on the
        clicked row's Time cell (NEWS_URL_ROLE), so it stays correct after any
        column sort — indexing _news_filtered by visual row would not."""
        it = self._news_table.item(row, 0)
        url = it.data(NEWS_URL_ROLE) if it is not None else ''
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def _stream_ok(self, stream):
        """Mark a stream healthy: stamp last-ok (monotonic) + zero ITS OWN fails."""
        h = self._stream_health.get(stream)
        if h is not None:
            h['last_ok'] = time.monotonic()
            h['fails'] = 0

    def _refresh_api_health(self):
        """Recompute the worst-case 'API:' label + tooltip. Cheap (8 streams);
        driven by on_error and the 10s on_account tick — no dedicated timer.

        Red when any stream has fails>=3 or a stale age (>3x its interval);
        the label names the worst stream. Age-staleness is only judged for the
        continuously-polled streams — on-demand (history/chart) and the
        visibility-throttled stocks stream can be legitimately old."""
        now = time.monotonic()
        AGE_CHECK = ('account', 'positions', 'orders', 'hw', 'news')
        worst = None  # (key=(fails, age), stream, red)
        for s, h in self._stream_health.items():
            fails = h['fails']
            last_ok = h['last_ok']
            interval = self._stream_intervals.get(s, 60)
            # Never-ok streams age from boot so startup doesn't flash "stale".
            age = now - (last_ok if last_ok is not None else self._boot_monotonic)
            stale = s in AGE_CHECK and age > 3 * interval
            if fails == 0 and not stale:
                continue  # healthy
            key = (fails, age)
            if worst is None or key > worst[0]:
                worst = (key, s, fails >= 3 or stale)
        if worst is None:
            self._status_conn.setText("API: OK")
            self._status_conn.setStyleSheet(f"color: {T['green'].name()};")
        else:
            _, s, red = worst
            fails = self._stream_health[s]['fails']
            text = f"API: {s} ERR×{fails}" if fails else f"API: {s} stale"
            color = T['red'].name() if red else T.get('yellow', T['white']).name()
            self._status_conn.setText(text)
            self._status_conn.setStyleSheet(f"color: {color};")
        self._status_conn.setToolTip(self._api_health_tooltip(now))

    def _api_health_tooltip(self, now):
        """Per-stream detail for the API label: name, last-ok age, fail count."""
        lines = []
        for s, h in self._stream_health.items():
            last_ok = h['last_ok']
            if last_ok is not None:
                state = f"ok {chart_core.format_age(now - last_ok)} ago"
            else:
                state = "no data yet"
            lines.append(f"{s}: {state}, fails {h['fails']}")
        return "\n".join(lines)

    @Slot(str, str)
    def on_error(self, stream, msg):
        h = self._stream_health.get(stream)
        if h is not None:
            h['fails'] += 1
            # Alert once when a stream crosses into sustained-failure territory
            # (dedupe backstops if it re-crosses on the next tick).
            if h['fails'] == 3:
                self._push_alert('stream', f"{stream} stream failing (x3)")
        self._refresh_api_health()
        self.statusBar().showMessage(f"{stream}: {msg}", 10_000)

    @Slot(str, str)
    def on_log_lines(self, name, text):
        # Always buffer (newline-aligned trim — no mid-line cut), even while
        # Paused, so a later filter/level change or un-pause re-renders in full.
        self._log_buffers[name] = _trim_to_newline(
            self._log_buffers.get(name, "") + text)

        if self._log_selector.currentText() != name:
            return
        if hasattr(self, '_log_paused') and self._log_paused.isChecked():
            return  # buffered above; the view just doesn't advance

        pattern = self._compile_log_filter()
        level = (self._log_level.currentText()
                 if hasattr(self, '_log_level') else "All")
        appended = False
        for line in text.split("\n"):
            if line == "":
                continue
            if self._log_line_passes(line, pattern, level):
                self._append_log_line_html(line)
                appended = True
        if appended and self._auto_scroll.isChecked():
            self._log_scroll_to_bottom()

    # ---- Private Helpers -------------------------------------------------
    def _set_card(self, card, value, color=None):
        lbl = card.findChild(QLabel, "card_value")
        if lbl:
            lbl.setText(str(value))
            if color:
                # Dynamic P&L color. Keep the tabular numeric font + display
                # size so a colored card matches the globally-styled neutral
                # cards exactly — only the color differs.
                px, weight = design_tokens.TYPE["display"]
                lbl.setStyleSheet(
                    design_tokens.numeric_qss(px=px, weight=weight)
                    + f" color: {color.name()};")

    def _apply_trade_filter(self, filter_text):
        orders = self._orders_cache
        # Sourced from stock_config (CRYPTO_SYMBOLS + CRYPTO_POOL) so this can't
        # drift from the live universe; built once at module load.
        crypto_symbols = CRYPTO_SYMBOL_SET
        if filter_text == "Crypto":
            orders = [o for o in orders if o["symbol"] in crypto_symbols]
        elif filter_text == "Stock":
            orders = [o for o in orders if o["symbol"] not in crypto_symbols]

        open_orders = [o for o in orders if o["status"] in
                       ("new", "accepted", "partially_filled", "pending_new")]
        filled_orders = [o for o in orders if o["status"] == "filled"]

        self._open_orders_view = open_orders  # row->order for the Cancel column
        self._open_orders_table.setUpdatesEnabled(False)
        self._open_orders_table.setRowCount(len(open_orders))
        for row, o in enumerate(open_orders):
            vals = [o["symbol"], o["side"], str(o["qty"]), o["type"],
                    o["status"], fmt_time_short(o["submitted_at"])]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(T["green"] if o["side"] == "buy" else T["red"])
                self._open_orders_table.setItem(row, col, item)
        self._open_orders_table.setUpdatesEnabled(True)
        # (Re)build the Cancel button column only when the row->id layout changed
        # — a plain status tick under the same order set leaves buttons untouched.
        new_order = [o.get("id", "") for o in open_orders]
        if new_order != getattr(self, "_open_order_btn_order", None):
            self._rebuild_cancel_buttons(open_orders)

        filled_orders = filled_orders[:50]
        self._fills_table.setUpdatesEnabled(False)
        self._fills_table.setRowCount(len(filled_orders))
        for row, o in enumerate(filled_orders):
            notional = o.get("notional") or ""
            if notional:
                notional = fmt_money(notional)
            vals = [
                o["symbol"], o["side"],
                str(o.get("filled_qty") or o["qty"]),
                fmt_money(o["filled_avg_price"]) if o["filled_avg_price"] else "\u2014",
                notional,
                fmt_time(o["filled_at"]),
            ]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(T["green"] if o["side"] == "buy" else T["red"])
                self._fills_table.setItem(row, col, item)
        self._fills_table.setUpdatesEnabled(True)

    def _make_cancel_button(self, order):
        """One Cancel button bound to an ORDER dict (captures id/symbol, not a
        row) so it stays correct across rebuilds. Styled from the pnl-down
        palette like the positions Close button."""
        btn = QPushButton("Cancel")
        btn.setFixedHeight(22)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {PAL['down']};"
            f" color: {_on_color(PAL['down'])}; font-size: 10px; font-weight: bold;"
            f" border-radius: 3px; padding: 1px 6px; }}"
            f" QPushButton:hover {{"
            f" background-color: {chart_core.mix(PAL['down'], '#ffffff', 0.15)}; }}")
        btn.clicked.connect(lambda _, o=order: self._cancel_order_clicked(o))
        return btn

    def _rebuild_cancel_buttons(self, open_orders):
        """(Re)create the Cancel button for every open-orders row in visual order
        (Qt cell widgets don't follow item edits/re-sorts). Mirrors
        _rebuild_pos_buttons; skips rows whose order carries no id."""
        tbl = self._open_orders_table
        self._open_order_cancel_btn = {}
        for r, o in enumerate(open_orders):
            oid = o.get("id", "")
            if not oid:
                tbl.removeCellWidget(r, 6)
                continue
            btn = self._make_cancel_button(o)
            tbl.setCellWidget(r, 6, btn)
            self._open_order_cancel_btn[oid] = btn
        self._open_order_btn_order = [o.get("id", "") for o in open_orders]

    def _cancel_order_clicked(self, order):
        """Confirm + cancel one open order, then nudge a fresh orders fetch."""
        oid = order.get("id", "")
        if not oid:
            return
        sym = order.get("symbol", "?")
        qty = order.get("qty")
        notional = order.get("notional")
        size = (f"qty {qty}" if qty not in (None, "", "0", 0)
                else (f"${notional}" if notional else "?"))
        reply = QMessageBox.question(
            self, "Cancel Order",
            f"Cancel this open order?\n\nSymbol: {sym}\nSize: {size}\n"
            f"Order id: {str(oid)[:8]}…",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            self.api.cancel_order(oid)
            self.statusBar().showMessage(f"Cancel requested for {sym} order", 7000)
            from PySide6.QtCore import QMetaObject
            QMetaObject.invokeMethod(
                self._fetcher_hot, "fetch_orders", Qt.QueuedConnection)
        except Exception as e:
            self.statusBar().showMessage(f"Cancel error ({sym}): {e}", 7000)
            self._push_alert('order-error', f"{sym} cancel error: {e}")

    def _update_tax(self, orders):
        # tax_lots.estimate_taxes is the shared pure-stdlib MinTax kernel;
        # crypto_symbols excludes coins from any wash-sale path, and
        # window_truncated propagates the paginated-order-window cap so the
        # basis_complete flag is honest when history was cut short.
        tax = estimate_taxes(orders, crypto_symbols=CRYPTO_SYMBOL_SET,
                             window_truncated=getattr(self, "_orders_truncated", False))
        self._set_card(self._tax_realized, fmt_money(tax["realized_gain"]),
                       pnl_color(tax["realized_gain"]))
        self._set_card(self._tax_st, fmt_money(tax["short_term_gain"]),
                       pnl_color(tax["short_term_gain"]))
        self._set_card(self._tax_lt, fmt_money(tax["long_term_gain"]),
                       pnl_color(tax["long_term_gain"]))
        # Flag incomplete cost basis on the Est. Tax card (muted, suffixed)
        # rather than presenting a truncated-basis number as authoritative.
        if tax.get("basis_complete", True):
            self._set_card(self._tax_owed, fmt_money(tax["estimated_tax"]),
                           T["red"])
        else:
            self._set_card(self._tax_owed,
                           fmt_money(tax["estimated_tax"]) + " (incomplete basis)",
                           T["muted"])
        self._set_card(self._tax_net, fmt_money(tax["net_after_tax"]),
                       pnl_color(tax["net_after_tax"]))

    def _on_log_selected(self, name):
        buf = self._log_buffers.get(name, "")
        if not buf:
            path = LOG_FILES.get(name)
            if path and path.exists():
                try:
                    # Newline-aligned trim (was a mid-line byte slice).
                    text = _trim_to_newline(path.read_text(errors="replace"))
                    self._log_buffers[name] = text
                except OSError:
                    pass
        # Render through the filter + severity coloring (handles empty buffer).
        self._rerender_log_view()

    def _trigger_retrain(self, crypto=False, stock=False):
        """Write a retrain trigger file for the pipeline to pick up."""
        trigger_path = BASE_DIR / "retrain_trigger.json"

        # Check if pipeline is running
        status_path = BASE_DIR / "pipeline_status.json"
        is_running = False
        is_training = False
        try:
            age = dt.datetime.now().timestamp() - status_path.stat().st_mtime
            is_running = age < PIPELINE_STALE_SEC
            if is_running:
                pinfo = _read_pipeline_status()
                phase = pinfo.get("phase", "")
                is_training = phase not in ("trading", "idle", "failed", "complete", "")
        except OSError:
            pass

        if not is_running:
            self._retrain_status.setText("Pipeline not running")
            self._retrain_status.setStyleSheet(f"color: {T['red'].name()}; font-size: 11px;")
            return

        # Build description for confirmation
        parts = []
        if crypto:
            parts.append("Crypto")
        if stock:
            parts.append("Stocks")
        target = " + ".join(parts)

        if is_training:
            msg = (f"Training is already in progress.\n"
                   f"Queue {target} retrain to start after current "
                   f"training completes?")
        else:
            msg = f"Start {target} retraining now?"

        reply = QMessageBox.question(
            self, "Retrain Models", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        # Write trigger file
        try:
            trigger = {"crypto": crypto, "stock": stock}
            tmp = str(trigger_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(trigger, f)
            os.replace(tmp, str(trigger_path))
            self._retrain_status.setText(
                f"{target} retrain queued (trains challenger; "
                f"promoted after shadow eval)")
            self._retrain_status.setStyleSheet(f"color: {T['green'].name()}; font-size: 11px;")
            self._schedule_models_refresh(5000)
        except Exception as e:
            self._retrain_status.setText(f"Error: {e}")
            self._retrain_status.setStyleSheet(f"color: {T['red'].name()}; font-size: 11px;")

    def _schedule_models_refresh(self, delay_ms=3000):
        """Schedule a quick Models tab refresh after a button action."""
        from PySide6.QtCore import QTimer
        QTimer.singleShot(delay_ms, self._refresh_models_tab)

    def _run_report_clicked(self, script_args, title):
        """Launch a measurement-only report script (U5).

        Mirrors _refresh_all_llm_clicked's subprocess pattern: stdout goes
        to a file (not PIPE — nothing reads the pipe, so a chatty child
        would fill it and hang forever), completion is polled on a QTimer,
        and the triggering buttons stay disabled while it runs. Unlike the
        LLM refresh, we capture to a per-run tempfile so the output can be
        shown back to the user in a dialog once the process exits.
        """
        import subprocess
        import tempfile

        if getattr(self, '_report_proc', None) is not None:
            self._reports_status.setText("A report is already running...")
            self._reports_status.setStyleSheet(
                f"color: {T['yellow'].name()}; font-size: 11px;")
            return

        report_btns = self._report_btns
        for btn in report_btns:
            btn.setEnabled(False)
        self._reports_status.setText(f"Running {title}...")
        self._reports_status.setStyleSheet(
            f"color: {T['accent'].name()}; font-size: 11px;")
        QApplication.processEvents()

        python = _engine_python()
        script_name = script_args[0]
        script = str(BASE_DIR / script_name)
        args = list(script_args[1:])
        env = _engine_env()
        # Structured reports (gui_review_2026-07 §7 2.8): scripts that support
        # --json get a temp path so we can parse the result dict and render a
        # summary block above the raw stdout instead of a 2005-era text dump.
        json_path = None
        json_persistent = False
        if script_name == "beta_ledger.py":
            json_path = str(BETA_REPORT_FILE)   # persistent: feeds the freshness strip
            json_persistent = True
            args += ["--json", json_path]
        elif script_name == "gap_audit.py":
            jf = tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", prefix="gui_report_json_",
                delete=False)
            jf.close()
            json_path = jf.name
            args += ["--json", json_path]
        tf = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".log", prefix="gui_report_", delete=False)
        try:
            proc = subprocess.Popen(
                [python, "-u", script, *args],
                stdout=tf, stderr=subprocess.STDOUT,
                env=env, cwd=str(BASE_DIR),
            )
            self._report_proc = proc
            self._report_tempfile = tf
            self._report_title = title
            self._report_script = script_name
            self._report_json_path = json_path
            self._report_json_persistent = json_persistent
            self._report_args = list(script_args)
            from PySide6.QtCore import QTimer
            self._report_timer = QTimer()
            self._report_timer.timeout.connect(self._check_report_run)
            self._report_timer.start(2000)
        except Exception as e:
            tf.close()
            if json_path and not json_persistent:
                try:
                    os.unlink(json_path)
                except OSError:
                    pass
            self._reports_status.setText(f"Error: {e}")
            self._reports_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")
            for btn in report_btns:
                btn.setEnabled(True)

    def _check_report_run(self):
        """Poll the report subprocess for completion; show output on finish."""
        if not hasattr(self, '_report_proc'):
            return
        proc = self._report_proc
        rc = proc.poll()
        if rc is None:
            # Still running (output accumulates in the tempfile)
            return
        self._report_timer.stop()
        for btn in self._report_btns:
            btn.setEnabled(True)

        tf = self._report_tempfile
        title = self._report_title
        script_name = getattr(self, '_report_script', '')
        json_path = getattr(self, '_report_json_path', None)
        output = ""
        try:
            tf.flush()
            tf.seek(0)
            output = tf.read()
        except OSError:
            pass
        finally:
            try:
                tf.close()
                os.unlink(tf.name)
            except OSError:
                pass

        # Parse the structured --json artifact into a summary block above the
        # raw stdout (gui_review_2026-07 §7 2.8). llm_eval/execution_report
        # write their own canonical report files (no --json needed).
        summary = ""
        if json_path:
            try:
                with open(json_path) as jf:
                    data = json.load(jf)
                if script_name == "beta_ledger.py":
                    summary = self._format_beta_summary(data)
                elif script_name == "gap_audit.py":
                    summary = self._format_gap_summary(data)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                summary = ""
            finally:
                if not getattr(self, '_report_json_persistent', False):
                    try:
                        os.unlink(json_path)
                    except OSError:
                        pass
        elif script_name == "llm_eval.py":
            art = (BASE_DIR / "llm_advisor_report.json"
                   if "--advisor" in (getattr(self, '_report_args', None) or [])
                   else BASE_DIR / "llm_eval_report.json")
            try:
                with open(art) as jf:
                    data = json.load(jf)
                summary = (chart_core.format_advisor_summary(data)
                           if art.name.startswith("llm_advisor")
                           else chart_core.format_llm_eval_summary(data))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                summary = ""
        elif script_name == "execution_report.py":
            try:
                with open(BASE_DIR / "execution_report.json") as jf:
                    summary = chart_core.format_execution_summary(json.load(jf))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                summary = ""
        if summary:
            output = summary + "\n" + "-" * 60 + "\n\n" + output

        if rc == 0:
            self._reports_status.setText(f"{title} complete")
            self._reports_status.setStyleSheet(
                f"color: {T['green'].name()}; font-size: 11px;")
        else:
            self._reports_status.setText(f"{title} failed (exit {rc})")
            self._reports_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")

        # A fresh decision_report.json means the Trading-tab gate box is stale.
        if script_name == "decision_report.py" and rc == 0:
            try:
                self._refresh_gate_attribution()
            except Exception:
                pass
        # Any finished report changes the freshness strip.
        try:
            self._refresh_reports_freshness()
        except Exception:
            pass

        self._show_report_dialog(title, output)

        del self._report_proc
        del self._report_tempfile
        del self._report_title
        self._report_script = None
        self._report_json_path = None
        self._report_json_persistent = False
        self._report_args = None

    def _show_report_dialog(self, title, text):
        """Modal dialog with the captured report output (read-only, monospace)."""
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(900, 600)
        vbox = QVBoxLayout(dlg)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setFont(QFont("Monospace", 10))
        view.setPlainText(text)
        vbox.addWidget(view)
        dlg.exec()

    def _run_gap_audit_clicked(self):
        """Launch gap_audit.py over the live stock universe. gap_audit needs
        --symbols and the overnight sleeve is chosen dynamically, so the
        universe is the honest candidate set. Resolved fresh at click time."""
        try:
            from stock_config import load_stock_universe
            symbols = [str(s) for s in load_stock_universe() if s]
        except Exception:
            symbols = []
        if not symbols:
            self._reports_status.setText("Gap Audit: no stock universe")
            self._reports_status.setStyleSheet(
                f"color: {T['yellow'].name()}; font-size: 11px;")
            return
        self._run_report_clicked(
            ["gap_audit.py", "--symbols", *symbols], "Gap Audit")

    def _format_beta_summary(self, rep):
        """Compact summary block from beta_ledger's --json report dict
        (beta_ledger.beta_report: period/strategy/joint{alpha_annual,alpha_t,
        r2,betas} + per-benchmark up_down/trend_conditional/rolling_beta_last).
        """
        import math
        if not isinstance(rep, dict):
            return ""

        def num(x, fmt):
            try:
                v = float(x)
                return format(v, fmt) if math.isfinite(v) else "n/a"
            except (TypeError, ValueError):
                return "n/a"

        lines = ["BETA LEDGER — summary"]
        p = rep.get('period') or {}
        if p:
            lines.append(f"  period: {p.get('start', '?')} .. "
                         f"{p.get('end', '?')} "
                         f"({p.get('n_days', '?')} trading days)")
            if p.get('obs_per_year_grid') is not None:
                lines.append(f"  grid: {num(p.get('obs_per_year_grid'), '.0f')} "
                             f"obs/yr observed vs 252 assumed")
        s = rep.get('strategy') or {}
        if s:
            strat_line = (
                f"  strategy: {num(s.get('ann_return'), '+.1%')}/yr @ "
                f"{num(s.get('ann_vol'), '.1%')} vol "
                f"(Sharpe {num(s.get('sharpe'), '.2f')})")
            if 'sharpe_clean' in s:
                strat_line += f" (Sharpe clean {num(s.get('sharpe_clean'), '.2f')})"
            lines.append(strat_line)
        j = rep.get('joint') or {}
        if j:
            lines.append(
                f"  alpha: {num(j.get('alpha_annual'), '+.1%')}/yr "
                f"(HAC t={num(j.get('alpha_t'), '+.2f')}) · "
                f"R²={num(j.get('r2'), '.2f')}  "
                f"[must clear before any 'alpha' claim]")
            if 'alpha_t_corrected' in j:
                lines.append(f"  alpha t (dof-corrected): "
                             f"{num(j.get('alpha_t_corrected'), '+.2f')}")
            if 'alpha_annual_clean' in j:
                lines.append(
                    f"  alpha CLEAN: {num(j.get('alpha_annual_clean'), '+.1%')}/yr "
                    f"(t={num(j.get('alpha_t_clean'), '+.2f')}) · "
                    f"contamination Δ "
                    f"{num(j.get('contamination_delta'), '+.1%')}/yr")
            for name, b in (j.get('betas') or {}).items():
                if not isinstance(b, dict):
                    continue
                diag = rep.get(name) or {}
                ud = diag.get('up_down') or {}
                tc = diag.get('trend_conditional') or {}
                above = (tc.get('above_200d_true') or {}).get('beta')
                below = (tc.get('above_200d_false') or {}).get('beta')
                line = (f"  {name} beta: contemp "
                        f"{num(b.get('contemporaneous'), '+.3f')} · "
                        f"summed(AKL) {num(b.get('summed'), '+.3f')}")
                if ud:
                    line += (f" · up/down {num(ud.get('beta_up'), '+.3f')}/"
                             f"{num(ud.get('beta_down'), '+.3f')}")
                if above is not None or below is not None:
                    line += (f" · above/below 200d {num(above, '+.3f')}/"
                             f"{num(below, '+.3f')}")
                tcl = diag.get('trend_conditional_lagged') or {}
                if tcl:
                    line += (f" · above/below 200d (PIT) "
                             f"{num((tcl.get('above_200d_true') or {}).get('beta'), '+.3f')}/"
                             f"{num((tcl.get('above_200d_false') or {}).get('beta'), '+.3f')}")
                if 'rolling_beta_last' in diag:
                    line += (f" · rolling30d "
                             f"{num(diag.get('rolling_beta_last'), '+.3f')}")
                lines.append(line)
        return "\n".join(lines)

    def _format_gap_summary(self, results):
        """Compact summary from gap_audit's --json (dict of {sym: audit_name})
        + a sleeve total (forfeited drift vs gap-through cost)."""
        import math
        if not isinstance(results, dict) or not results:
            return ""

        def num(x, fmt):
            try:
                v = float(x)
                return format(v, fmt) if math.isfinite(v) else "n/a"
            except (TypeError, ValueError):
                return "n/a"

        lines = ["GAP AUDIT — overnight sleeve summary"]
        tot_drift = tot_gap = 0.0
        for sym in sorted(results):
            a = results[sym]
            if not isinstance(a, dict):
                continue
            gs = a.get('gap_stats') or {}
            fd, gt = a.get('forfeited_drift_annual'), a.get('gap_through_cost_annual')
            try:
                tot_drift += float(fd or 0.0)
                tot_gap += float(gt or 0.0)
            except (TypeError, ValueError):
                pass
            lines.append(
                f"  {sym:<6} overnight {num(a.get('overnight_mean_bps'), '.1f')}bps · "
                f"t_df {num(gs.get('t_df'), '.1f')} · "
                f"exk {num(gs.get('excess_kurtosis'), '.2f')} · "
                f"forfeited ${num(fd, ',.0f')}/yr · "
                f"gap_through ${num(gt, ',.0f')}/yr")
        lines.append(f"  SLEEVE TOTAL: forfeited_drift ${tot_drift:,.0f}/yr · "
                     f"gap_through ${tot_gap:,.0f}/yr")
        lines.append("  (if overlay friction >> gap_through, the overnight "
                     "overlay is NO-GO)")
        return "\n".join(lines)

    def _is_pipeline_running(self):
        """Check if pipeline process is running (status file recently updated).

        Uses the same PIPELINE_STALE_SEC threshold as the Models tab render
        and the retrain click handler (U4) — previously this used a much
        tighter 120s cutoff, so the tab could show "running" while this
        method (and the retrain button) said "not running": an enabled but
        effectively dead button.
        """
        try:
            age = dt.datetime.now().timestamp() - (
                BASE_DIR / "pipeline_status.json").stat().st_mtime
            return age < PIPELINE_STALE_SEC
        except OSError:
            return False

    def _combined_bots_running(self):
        """Detect a live combined-mode run_bots.py process.

        run_pipeline.py --combined-bots runs both loops inside a single
        'Bots' process that its per-bot status flags don't track, so the
        status file reports both bots stopped while they are trading.
        """
        import subprocess
        try:
            result = subprocess.run(
                ["pgrep", "-f", "run_bots\\.py"],
                capture_output=True, text=True, timeout=5)
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _pipeline_pids(self):
        """Live run_pipeline.py PIDs via pgrep ([] on any failure)."""
        import subprocess
        try:
            result = subprocess.run(
                ["pgrep", "-f", "run_pipeline\\.py"],
                capture_output=True, text=True, timeout=5)
            return [int(p) for p in result.stdout.strip().split()
                    if p.strip()]
        except Exception:
            return []

    def _restart_pipeline_status_set(self, text, color):
        self._restart_pipeline_status.setText(text)
        self._restart_pipeline_status.setStyleSheet(
            f"color: {color.name()}; font-size: 11px;")

    def _restart_pipeline_clicked(self):
        """Restart the pipeline via a non-blocking stop -> confirm-dead ->
        launch -> confirm-alive handshake (QTimer state machine — mirrors the
        report-runner poll pattern).

        A mid-training shutdown legitimately takes >10s (Optuna child wait then
        bot stop); the old blocking 5s-then-relaunch could leave TWO live
        orchestrators fighting over the same book (duplicate order flow). This
        never launches while any old run_pipeline.py PID survives: SIGTERM,
        escalate SIGKILL at >10s, and ABORT (no launch) at >15s.
        """
        import signal
        import subprocess

        # Re-entry guard — a handshake is already in flight
        if getattr(self, '_restart_timer', None) is not None \
                and self._restart_timer.isActive():
            return

        was_running = self._is_pipeline_running()
        reply = QMessageBox.question(
            self, "Restart Pipeline",
            ("Stop the running pipeline and start a fresh bot-only "
             "orchestrator?" if was_running else
             "Start a fresh bot-only pipeline orchestrator?"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self._restart_pipeline_btn.setEnabled(False)

        # Capture the deployment's launch flags so the restart preserves
        # them (--combined-bots, --no-retrain, --retrain-day N, ...)
        prev_flags = _read_pipeline_status().get("launch_args")
        if not (isinstance(prev_flags, list)
                and all(isinstance(a, str) for a in prev_flags)):
            prev_flags = None
        pids = self._pipeline_pids()
        if prev_flags is None and pids:
            # No launch_args in status — read argv off the live process
            # before killing it
            try:
                ps = subprocess.run(
                    ["ps", "-o", "command=", "-p", str(pids[0])],
                    capture_output=True, text=True, timeout=5)
                argv = ps.stdout.strip().split()
                for i, a in enumerate(argv):
                    if a.endswith("run_pipeline.py"):
                        prev_flags = argv[i + 1:]
                        break
            except Exception:
                pass
        self._restart_flags = prev_flags or []

        # SIGTERM every old orchestrator, then poll for death off-thread
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                pass

        if not pids:
            # Nothing to stop — launch immediately
            self._restart_launch()
            return

        self._restart_stop_start = time.monotonic()
        self._restart_sigkilled = False
        self._restart_pipeline_status_set("Stopping pipeline... (0s)", T['accent'])
        from PySide6.QtCore import QTimer
        self._restart_timer = QTimer()
        self._restart_timer.timeout.connect(self._restart_poll_stop)
        self._restart_timer.start(1000)

    def _restart_poll_stop(self):
        """Poll for the old pipeline to die; SIGKILL >10s; ABORT (no launch) at
        >15s — never launch a second orchestrator beside a live one."""
        import signal
        elapsed = time.monotonic() - self._restart_stop_start
        pids = self._pipeline_pids()

        if not pids:
            self._restart_timer.stop()
            self._restart_launch()
            return

        if elapsed > 15:
            self._restart_timer.stop()
            self._restart_pipeline_status_set(
                f"Old pipeline won't die (pids "
                f"{', '.join(str(p) for p in pids)}) — NOT launching a "
                f"second orchestrator", T['red'])
            self._restart_pipeline_btn.setEnabled(True)
            return

        if elapsed > 10 and not self._restart_sigkilled:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    pass
            self._restart_sigkilled = True

        label = "Force-stopping" if self._restart_sigkilled else "Stopping"
        self._restart_pipeline_status_set(
            f"{label} pipeline... ({int(elapsed)}s)", T['accent'])

    def _restart_launch(self):
        """Launch the new bot-only orchestrator, then confirm-alive."""
        import subprocess
        flags = [a for a in (getattr(self, '_restart_flags', None) or [])
                 if a not in ("--skip-harvest", "--bot-only")]
        flags += ["--skip-harvest", "--bot-only"]
        pipeline_py = str(BASE_DIR / "run_pipeline.py")
        python = _engine_python()
        log_file = str(BASE_DIR / "pipeline_output.log")
        env = _engine_env(cusparselt=True)
        self._restart_launch_ts = time.time()  # wall clock, to compare mtime
        try:
            with open(log_file, "a") as lf:
                subprocess.Popen(
                    [python, "-u", pipeline_py, *flags],
                    stdout=lf, stderr=subprocess.STDOUT,
                    env=env, cwd=str(BASE_DIR),
                    start_new_session=True,
                )
        except Exception as e:
            self._restart_pipeline_status_set(
                f"Launch failed — check pipeline_output.log ({e})", T['red'])
            self._restart_pipeline_btn.setEnabled(True)
            return

        self._restart_alive_start = time.monotonic()
        self._restart_pipeline_status_set("Launching... (0s)", T['accent'])
        from PySide6.QtCore import QTimer
        self._restart_timer = QTimer()
        self._restart_timer.timeout.connect(self._restart_poll_alive)
        self._restart_timer.start(1000)

    def _restart_poll_alive(self):
        """Confirm the new orchestrator is up: a run_pipeline.py PID AND a
        pipeline_status.json mtime newer than launch. Up to 20s, then report."""
        elapsed = time.monotonic() - self._restart_alive_start
        pids = self._pipeline_pids()
        status_fresh = False
        try:
            mtime = (BASE_DIR / "pipeline_status.json").stat().st_mtime
            status_fresh = mtime > self._restart_launch_ts
        except OSError:
            status_fresh = False

        if pids and status_fresh:
            self._restart_timer.stop()
            self._restart_pipeline_status_set(
                f"Pipeline restarted (pid {pids[0]})", T['green'])
            self._restart_pipeline_btn.setEnabled(True)
            self._schedule_models_refresh(2000)
            return

        if elapsed > 20:
            self._restart_timer.stop()
            self._restart_pipeline_status_set(
                "Launch failed — check pipeline_output.log", T['red'])
            self._restart_pipeline_btn.setEnabled(True)
            return

        self._restart_pipeline_status_set(
            f"Launching... ({int(elapsed)}s)", T['accent'])

    def _cancel_retrain(self):
        """Remove a pending retrain trigger file."""
        trigger_path = BASE_DIR / "retrain_trigger.json"
        try:
            if trigger_path.exists():
                trigger_path.unlink()
            self._retrain_status.setText("Retrain cancelled")
            self._retrain_status.setStyleSheet(f"color: {T['muted'].name()}; font-size: 11px;")
        except Exception as e:
            self._retrain_status.setText(f"Cancel error: {e}")
            self._retrain_status.setStyleSheet(f"color: {T['red'].name()}; font-size: 11px;")

    def _debounce_button(self, btn):
        """Disable a control button for 5s to prevent double-submitting a
        bot/pipeline command on a fast double-click."""
        try:
            btn.setEnabled(False)
            QTimer.singleShot(5000, lambda: btn.setEnabled(True))
        except Exception:
            pass

    def _start_bot_clicked(self, bot_name):
        """Handle Start button click for a bot."""
        self._debounce_button(
            self._crypto_start_btn if bot_name == "Crypto"
            else self._stock_start_btn)
        if not self._is_pipeline_running():
            self._bot_cmd_status.setText("Pipeline not running")
            self._bot_cmd_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")
            return

        if self._combined_bots_running():
            # Combined run_bots.py process already runs both loops;
            # starting a per-bot loop would duplicate order flow.
            self._bot_cmd_status.setText(
                "Bots already running (combined mode)")
            self._bot_cmd_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")
            return

        pinfo = _read_pipeline_status()
        phase = pinfo.get("phase", "")
        is_training = phase not in (
            "trading", "idle", "failed", "complete", "suspended", "")

        crypto = (bot_name == "Crypto")
        stock = (bot_name == "Stock")

        if is_training:
            reply = QMessageBox.question(
                self, "Training In Progress",
                f"Training is in progress. Suspend training to start "
                f"{bot_name} bot?\n\n"
                "Completed trials are preserved and training can resume later.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            result = _write_pipeline_command(
                "suspend_and_start_bot", crypto=crypto, stock=stock)
        else:
            result = _write_pipeline_command(
                "start_bot", crypto=crypto, stock=stock)

        if result is True:
            msg = (f"Suspending training, starting {bot_name}..."
                   if is_training else f"Starting {bot_name} bot...")
            self._bot_cmd_status.setText(msg)
            self._bot_cmd_status.setStyleSheet(
                f"color: {T['accent'].name()}; font-size: 11px;")
            self._schedule_models_refresh(3000)
            self._schedule_models_refresh(8000)
        else:
            self._bot_cmd_status.setText(f"Error: {result}")
            self._bot_cmd_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")

    def _stop_bot_clicked(self, bot_name):
        """Handle Stop button click for a bot."""
        self._debounce_button(
            self._crypto_stop_btn if bot_name == "Crypto"
            else self._stock_stop_btn)
        crypto = (bot_name == "Crypto")
        stock = (bot_name == "Stock")
        # A stop written while the pipeline is dead just sits on disk and fires
        # on the NEXT startup (surprise stop). Confirm before writing one.
        if not self._is_pipeline_running():
            reply = QMessageBox.question(
                self, "Pipeline Not Running",
                "Pipeline not running — a stop command written now would fire "
                "on next pipeline startup. Write anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        result = _write_pipeline_command(
            "stop_bot", crypto=crypto, stock=stock)
        if result is True:
            self._bot_cmd_status.setText(f"Stopping {bot_name} bot...")
            self._bot_cmd_status.setStyleSheet(
                f"color: {T['accent'].name()}; font-size: 11px;")
            self._schedule_models_refresh(3000)
            self._schedule_models_refresh(8000)
        else:
            self._bot_cmd_status.setText(f"Error: {result}")
            self._bot_cmd_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")

    def _toggle_halt_clicked(self):
        """Toggle the trading_halt.flag kill switch (entries only). Halting is
        one-click (fast is correct); resuming is confirmed so entries never
        re-enable by an accidental click."""
        if halt_active():
            reply = QMessageBox.question(
                self, "Resume Entries",
                "Re-enable entries? The bots will resume taking new positions.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            clear_halt()
            self._bot_cmd_status.setText("Halt cleared — entries allowed")
            self._bot_cmd_status.setStyleSheet(
                f"color: {T['green'].name()}; font-size: 11px;")
        else:
            set_halt("GUI halt button")
            self._bot_cmd_status.setText("Trading halted — entries blocked")
            self._bot_cmd_status.setStyleSheet(
                f"color: {T['red'].name()}; font-size: 11px;")
        self._refresh_models_tab()

    def _flatten_all_clicked(self):
        """Request liquidation of all positions AND halt entries directly.

        The flatten flag is only consumed by a live bot — if the bots are
        wedged/dead it would silently do nothing. So set the halt FIRST
        (GUI-owned, blocks new entries regardless of bot health), THEN request
        the flatten; on_positions renders the pending state until positions
        reach zero.
        """
        reply = QMessageBox.question(
            self, "Flatten All Positions",
            "Liquidate ALL positions in both books and halt entries?\n\n"
            "Entries are halted immediately; a running bot liquidates within "
            "one cycle.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        set_halt("flatten requested via GUI")
        request_flatten("GUI flatten button")
        self._bot_cmd_status.setText("Flatten requested — entries halted")
        self._bot_cmd_status.setStyleSheet(
            f"color: {T['red'].name()}; font-size: 11px;")
        self._refresh_models_tab()

    def _shadow_decision_style(self, decision):
        """(color, label) for a shadow decision. Colors via T (no raw hex)."""
        d = str(decision or "").lower()
        if d == "promote":
            return T["green"], "PROMOTE"
        if d == "discard":
            return T["red"], "DISCARD"
        if d == "continue":
            return T["accent"], "continue"
        return T.get("muted", T["white"]), "insufficient n"

    def _shadow_cell_text(self, book):
        """Compact challenger-cell summary from shadow_status.json, or the
        legacy manifest-mtime fallback. Returns a plain string (the model
        table cells are plain QTableWidgetItems)."""
        path = SHADOW_STATUS_FILES.get(book)
        try:
            with open(path) as f:
                s = json.load(f)
            _, dlabel = self._shadow_decision_style(s.get("decision"))
            n, min_obs = s.get("n", 0), s.get("min_obs", 0)
            age_d, win_d = s.get("age_days", 0.0), s.get("window_days", 0)
            parts = [dlabel, f"{n}/{min_obs}", f"d{age_d:.0f}/{win_d}"]
            p = s.get("p_value")
            if p is not None:
                parts.append(f"p={p:.3f}")
            return " · ".join(parts)
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            pass
        chal_path = CHALLENGER_MANIFESTS.get(book)
        if chal_path and chal_path.exists():
            try:
                cts = chal_path.stat().st_mtime
                return ("in shadow since "
                        + dt.datetime.fromtimestamp(
                            cts, tz=TZ_CENTRAL).strftime("%Y-%m-%d"))
            except OSError:
                return "in shadow"
        return "—"

    def _refresh_shadow_panel(self):
        """Render the per-book shadow/DM-HLN promotion story from
        {prefix}shadow_status.json (schema: shadow.py _write_shadow_status).
        Missing file -> the manifest-mtime fallback so the panel degrades to
        today's behavior rather than going blank."""
        labels = getattr(self, "_shadow_labels", None)
        if not labels:
            return
        for book, lbl in labels.items():
            path = SHADOW_STATUS_FILES.get(book)
            s = None
            try:
                with open(path) as f:
                    s = json.load(f)
            except (OSError, json.JSONDecodeError):
                s = None
            if not isinstance(s, dict):
                # Fallback: challenger present but no eval snapshot yet.
                chal_path = CHALLENGER_MANIFESTS.get(book)
                if chal_path and chal_path.exists():
                    try:
                        since = dt.datetime.fromtimestamp(
                            chal_path.stat().st_mtime,
                            tz=TZ_CENTRAL).strftime("%Y-%m-%d")
                    except OSError:
                        since = "?"
                    lbl.setText(
                        f"<b>{book}</b>: challenger in shadow since {since} — "
                        f"<span style='color:{T.get('muted', T['white']).name()}'>"
                        f"no eval snapshot yet</span>")
                else:
                    lbl.setText(
                        f"<b>{book}</b>: <span style='color:"
                        f"{T.get('muted', T['white']).name()}'>"
                        f"no challenger in shadow</span>")
                continue
            color, dlabel = self._shadow_decision_style(s.get("decision"))
            n = s.get("n", 0)
            min_obs = s.get("min_obs", 0)
            age_d = s.get("age_days", 0.0) or 0.0
            win_d = s.get("window_days", 0)
            parts = [
                f"<b>{book}</b>: <span style='color:{color.name()}'>"
                f"{dlabel}</span>",
                f"n {n}/{min_obs}",
                f"day {age_d:.0f} of {win_d}",
            ]
            p = s.get("p_value")
            if p is not None:
                pcol = (T["green"] if p < 0.05
                        else T.get("muted", T["white"])).name()
                parts.append(f"p=<span style='color:{pcol}'>{p:.3f}</span>")
            hc, hx = s.get("champ_hit_rate"), s.get("chall_hit_rate")
            if hc is not None and hx is not None:
                parts.append(f"hit {hc * 100:.1f}% → {hx * 100:.1f}%")
            # c26 U1: DM-v2 decision (shadow or deciding) + policy-gate sidecar.
            v2d = s.get("dm_v2_decision")
            if v2d is not None:
                v2col, v2lab = self._shadow_decision_style(v2d)
                mode = "deciding" if s.get("dm_v2_enabled") else "shadow"
                seg = (f"v2[{mode}]: <span style='color:{v2col.name()}'>"
                       f"{v2lab}</span>")
                t_v2, T_v2 = s.get("dm_v2_t"), s.get("dm_v2_T")
                if t_v2 is not None and T_v2 is not None:
                    seg += f" t{t_v2:.0f}/T{T_v2:.0f}"
                parts.append(seg)
            try:
                with open(POLICY_GATE_FILES.get(book)) as f:
                    pg = json.load(f)
                ok = bool(pg.get("passed"))
                gcol = (T["green"] if ok else T["red"]).name()
                parts.append(f"gate: <span style='color:{gcol}'>"
                             f"{'PASS' if ok else 'FAIL'}</span>")
            except (OSError, json.JSONDecodeError, TypeError):
                pass
            ts = s.get("ts")
            if ts:
                parts.append(f"<span style='color:"
                             f"{T.get('muted', T['white']).name()}'>"
                             f"{_ago(ts)}</span>")
            lbl.setText("  ·  ".join(parts))
            detail = s.get("detail")
            lbl.setToolTip(str(detail) if detail else "")

    def _refresh_drift_panel(self):
        """Render PSI drift per label from drift_state.json (monitor_drift.py:
        {label: {last_psi, last_check, action_days, last_action_date, ...}}).
        PSI bands mirror monitor_drift's PSI_WARN/PSI_ACTION. Missing file or
        no PSI recorded yet -> 'no drift data'."""
        lbl = getattr(self, "_drift_label", None)
        if lbl is None:
            return
        muted = T.get("muted", T["white"]).name()
        try:
            with open(DRIFT_STATE_FILE) as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            state = None
        if not isinstance(state, dict):
            lbl.setText(f"<span style='color:{muted}'>no drift data</span>")
            return
        today = dt.date.today()
        rows = []
        for label in ("crypto", "stock"):
            st = state.get(label)
            if not isinstance(st, dict) or st.get("last_psi") is None:
                continue
            psi = st["last_psi"]
            try:
                psi_f = float(psi)
            except (TypeError, ValueError):
                continue
            if psi_f >= DRIFT_PSI_ACTION:
                pcol = T["red"].name()
            elif psi_f >= DRIFT_PSI_WARN:
                pcol = T["yellow"].name()
            else:
                pcol = T["green"].name()
            # last_check is a calendar date string, not an instant — render
            # days-ago from it (hour precision isn't in the data).
            checked = st.get("last_check")
            when = "checked ?"
            try:
                d = dt.date.fromisoformat(str(checked))
                days = (today - d).days
                when = ("checked today" if days <= 0
                        else "checked 1d ago" if days == 1
                        else f"checked {days}d ago")
            except (TypeError, ValueError):
                if checked:
                    when = f"checked {checked}"
            action_days = int(st.get("action_days", 0) or 0)
            if action_days >= DRIFT_CONSECUTIVE_ACTION_DAYS:
                act = (f"<span style='color:{T['red'].name()}'>retrain "
                       f"requested</span>")
            elif action_days >= 1:
                act = (f"<span style='color:{T['yellow'].name()}'>action "
                       f"day {action_days}/{DRIFT_CONSECUTIVE_ACTION_DAYS}</span>")
            else:
                act = f"<span style='color:{muted}'>stable</span>"
            rows.append(
                f"<b>{label}</b>: PSI <span style='color:{pcol}'>"
                f"{psi_f:.3f}</span> · {when} · {act}")
        lbl.setText("<br>".join(rows) if rows
                    else f"<span style='color:{muted}'>no drift data</span>")

    def _refresh_meta_panel(self):
        """Render {prefix}meta_meta.json + {prefix}meta_refused.json per book
        (c26 U1) via chart_core.meta_panel_model. pred_source == 'in_sample'
        is the in-sample-primary leak diagnostic (amber); the refusal
        sidecar's mere presence triggers the loud amber chip."""
        labels = getattr(self, "_meta_gate_labels", None)
        if not labels:
            return
        muted = T.get("muted", T["white"]).name()
        for book, lbl in labels.items():
            meta = None
            refused = None
            try:
                with open(META_META_FILES.get(book)) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError, TypeError):
                meta = None
            try:
                with open(META_REFUSED_FILES.get(book)) as f:
                    refused = json.load(f)
            except (OSError, json.JSONDecodeError, TypeError):
                refused = None
            m = chart_core.meta_panel_model(meta, refused)
            parts = []
            if m['refused']:
                chip = (f"<span style='color:{T['yellow'].name()}; "
                        f"font-weight:700'>⚠ META REFUSED</span>")
                reasons = '; '.join(m['refused_reasons'])
                if reasons:
                    chip += f" {reasons}"
                if m['refused_at']:
                    try:
                        chip += " · " + _ago(dt.datetime.fromisoformat(
                            m['refused_at']).timestamp())
                    except (TypeError, ValueError):
                        pass
                parts.append(chip)
            if not m['present']:
                parts.append(f"<span style='color:{muted}'>no meta model "
                             f"trained</span>")
            else:
                ps = m['pred_source']
                pcol = (T['yellow'] if ps == 'in_sample' else T['green']).name()
                seg = f"pred_source <span style='color:{pcol}'>{ps}</span>"
                if m['val_auc'] is not None:
                    seg += f" · AUC {m['val_auc']}"
                if m['n_trades'] is not None:
                    seg += f" · n {m['n_trades']}"
                if m['oof_note']:
                    seg += (f" · <span style='color:{muted}'>"
                            f"{m['oof_note']}</span>")
                if m['trained_at']:
                    try:
                        seg += (f" · <span style='color:{muted}'>"
                                + _ago(dt.datetime.fromisoformat(
                                    m['trained_at']).timestamp())
                                + "</span>")
                    except (TypeError, ValueError):
                        pass
                parts.append(seg)
            lbl.setText(f"<b>{book}</b>: " + "  ·  ".join(parts))

    def _refresh_reports_freshness(self):
        """Render the reports-freshness strip (c26 U1) from
        chart_core.artifact_freshness(REPORT_FRESHNESS_ITEMS), plus the two
        meta_refused sidecars with INVERTED semantics — for those, presence
        is the alarm and absence is healthy."""
        if not hasattr(self, '_reports_fresh_label'):
            return
        muted = T.get("muted", T["white"]).name()
        segs = []
        for row in chart_core.artifact_freshness(REPORT_FRESHNESS_ITEMS):
            name = row['name']
            if not row['exists']:
                segs.append(f"<span style='color:{muted}'>{name}: —</span>")
            elif row['stale']:
                segs.append(
                    f"{name}: <span style='color:{T['yellow'].name()}'>"
                    f"{chart_core.format_age(row['age_s'])} STALE</span>")
            else:
                segs.append(
                    f"{name}: <span style='color:{T['green'].name()}'>"
                    f"{chart_core.format_age(row['age_s'])}</span>")
        for book, path in META_REFUSED_FILES.items():
            try:
                age = time.time() - os.path.getmtime(path)
                segs.append(
                    f"<span style='color:{T['yellow'].name()}; "
                    f"font-weight:700'>meta_refused {book}: PRESENT "
                    f"({chart_core.format_age(age)})</span>")
            except (OSError, TypeError, ValueError):
                segs.append(f"<span style='color:{muted}'>meta_refused "
                            f"{book}: none</span>")
        self._reports_fresh_label.setText("  ·  ".join(segs))

    def _refresh_models_tab(self):
        now_ts = dt.datetime.now().timestamp()
        configs = []
        for name, cfg_path in CONFIG_FILES.items():
            cfg = read_config(cfg_path)
            mod_time = "\u2014"
            age_hours = None
            mtime = _model_deployed_ts(name)
            if mtime is not None:
                age_hours = (now_ts - mtime) / 3600
                d = dt.datetime.fromtimestamp(mtime, tz=TZ_CENTRAL)
                mod_time = d.strftime("%Y-%m-%d %I:%M %p")
            # Challenger cell: compact shadow_status.json decision summary once
            # the daily eval has run (decision \u00b7 n/min_obs \u00b7 day/window \u00b7 p),
            # else the manifest-mtime fallback. The full DM-HLN breakdown lives
            # in the Shadow/Promotion panel below the table.
            chal_str = self._shadow_cell_text(name)
            configs.append((name, cfg, mod_time, age_hours, chal_str))

        self._model_table.setUpdatesEnabled(False)
        self._model_table.setRowCount(len(configs))
        for row, (name, cfg, mod_time, age_hours, chal_str) in enumerate(configs):
            # Determine status and age display
            if age_hours is None:
                status, age_str = "Missing", "\u2014"
                status_color = T["red"]
            elif age_hours < 24:
                status, age_str = "Fresh", f"{age_hours:.0f}h"
                status_color = T["green"]
            elif age_hours < 168:
                status = "OK"
                age_str = f"{age_hours / 24:.0f}d"
                status_color = T["yellow"]
            else:
                status = "Stale"
                age_str = f"{age_hours / 24:.0f}d"
                status_color = T["red"]

            best_score = _get_best_score(name)
            score_str = f"{best_score:.3f}" if best_score is not None else "\u2014"

            if cfg:
                vals = [name, status, score_str, mod_time, age_str,
                        str(cfg.get("hidden_dim", "?")),
                        str(cfg.get("num_layers", "?")),
                        str(cfg.get("seq_len", "?")),
                        str(cfg.get("trade_threshold", "?")),
                        str(cfg.get("indicator_preset", "N/A")),
                        chal_str]
            else:
                vals = [name, status, score_str, "Not found", age_str,
                        "\u2014", "\u2014", "\u2014", "\u2014", "\u2014", chal_str]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 1:
                    item.setForeground(status_color)
                elif col == 2 and best_score is not None:
                    item.setForeground(T["green"] if best_score > 3 else T["yellow"])
                elif col == 10 and v != "\u2014":
                    item.setForeground(T["yellow"])
                self._model_table.setItem(row, col, item)
        self._model_table.setUpdatesEnabled(True)

        # --- Pipeline Status (from pipeline_status.json) ---
        pinfo = _read_pipeline_status()
        phase = pinfo.get("phase", "idle")
        phase_label = pinfo.get("phase_label", "Idle")
        phase_idx = pinfo.get("phase_idx", -1)
        total_phases = pinfo.get("total_phases", 0)

        # Determine if pipeline is actively running (status file updated recently)
        is_running = False
        age = None
        status_path = BASE_DIR / "pipeline_status.json"
        try:
            age = now_ts - status_path.stat().st_mtime
            is_running = age < PIPELINE_STALE_SEC  # trials can take up to 10 minutes
        except OSError:
            pass

        # Update restart button text and clear stale command status
        if hasattr(self, '_restart_pipeline_btn'):
            self._restart_pipeline_btn.setText(
                "Restart Pipeline" if is_running else "Start Pipeline")
        if hasattr(self, '_bot_cmd_status'):
            # Clear stale "Starting/Stopping" messages once state is reflected
            cur = self._bot_cmd_status.text().lower()
            if cur and ("starting" in cur or "stopping" in cur
                        or "suspending" in cur):
                self._bot_cmd_status.setText("")

        bots_running = pinfo.get("bots_running", False)
        # Combined-bots mode: per-bot flags in the status file don't track
        # the single run_bots.py process — detect it directly.
        combined_bots = (is_running and not bots_running
                         and self._combined_bots_running())
        if combined_bots:
            bots_running = True

        if phase == "idle" or not is_running:
            status_color = T["muted"].name()
            status_text = "IDLE"
        elif phase == "failed":
            status_color = T["red"].name()
            status_text = "FAILED"
        elif phase == "complete":
            status_color = T["green"].name()
            status_text = "COMPLETE"
        elif phase == "trading":
            status_color = T["green"].name()
            status_text = "TRADING"
        elif phase == "suspended":
            status_color = T["accent"].name()
            status_text = "SUSPENDED"
        else:
            status_color = T["green"].name()
            status_text = "TRAINING" if bots_running else "RUNNING"

        if total_phases > 0 and phase_idx >= 0 and phase != "trading":
            status_text += f" ({phase_idx + 1}/{total_phases})"
        if bots_running and phase != "trading":
            status_text += " + BOTS"

        # Kill-switch indicator (flag may also be set via Telegram or ssh)
        halted = halt_active()
        if halted:
            reason = ""
            try:
                reason = json.loads(
                    (BASE_DIR / "trading_halt.flag").read_text()
                    or "{}").get("reason", "")
            except Exception:
                pass  # `touch trading_halt.flag` leaves a non-JSON file
            status_color = T["red"].name()
            status_text += (" | HALTED — entries blocked"
                            + (f" ({reason})" if reason else ""))
        if hasattr(self, '_halt_btn'):
            self._halt_btn.setText(
                "Resume Entries" if halted else "Halt Entries")
        # Settings-tab halt mirror reflects the same state.
        if hasattr(self, '_settings_halt_btn'):
            self._settings_halt_btn.setText(
                "Resume Entries" if halted else "Halt Entries")

        # Show how fresh the status file is while running (U3) — the status
        # file can go quiet for minutes mid-trial without meaning the
        # pipeline died (see PIPELINE_STALE_SEC), so surface the age instead
        # of hiding it.
        if is_running and age is not None:
            status_text += f"  (updated {age:.0f}s ago)"

        stale_prefix = ""
        if age is not None and 120 < age < PIPELINE_STALE_SEC:
            stale_prefix = (
                f"<span style='color:{T['yellow'].name()}'>"
                f"⚠ status {age:.0f}s stale</span>  ")
            # Edge-triggered cockpit alert (static text so the changing age
            # doesn't defeat the identical-newest dedupe).
            if not self._alert_pipe_stale:
                self._push_alert('stale', "pipeline status stale (>2 min)")
                self._alert_pipe_stale = True
        else:
            self._alert_pipe_stale = False

        self._pipeline_status.setText(
            f"{stale_prefix}Status: "
            f"<span style='color:{status_color}'>{status_text}</span>")

        self._pipeline_phase.setText(f"Phase: {phase_label}")

        trial_cur = pinfo.get("trial_current", 0)
        trial_tot = pinfo.get("trial_total", 0)
        cycle = pinfo.get("cycle", 0)

        if trial_tot > 0:
            self._pipeline_trial.setText(f"Trial: {trial_cur} / {trial_tot}")
            self._pipeline_progress.setRange(0, trial_tot)
            self._pipeline_progress.setValue(min(trial_cur, trial_tot))
            pct = trial_cur / trial_tot if trial_tot else 0
            if pct < 0.5:
                bar_color = T["accent"].name()
            elif pct < 0.9:
                bar_color = T["yellow"].name()
            else:
                bar_color = T["green"].name()
            self._pipeline_progress.setStyleSheet(
                f"QProgressBar {{ color: {_on_color(bar_color)}; }}"
                f"QProgressBar::chunk {{ background-color: {bar_color}; }}")
        elif cycle > 0:
            self._pipeline_trial.setText(f"Bot Cycle: {cycle}")
            self._pipeline_progress.setRange(0, 1)
            self._pipeline_progress.setValue(1)
            self._pipeline_progress.setStyleSheet(
                f"QProgressBar {{ color: {_on_color(T['green'].name())}; }}"
                f"QProgressBar::chunk {{ background-color: {T['green'].name()}; }}")
        else:
            self._pipeline_trial.setText("Trial: \u2014")
            self._pipeline_progress.setRange(0, 1)
            self._pipeline_progress.setValue(0)

        best_score = pinfo.get("best_score", 0)
        per_class = pinfo.get("best_per_class", {})
        if best_score > 0:
            pc_str = ""
            if per_class:
                pc_str = (f"  (B:{per_class.get('bear', 0):.0%}"
                          f" N:{per_class.get('neutral', 0):.0%}"
                          f" U:{per_class.get('bull', 0):.0%})")
            self._pipeline_best.setText(f"Best Score: {best_score:.4f}{pc_str}")
        else:
            self._pipeline_best.setText("Best Score: \u2014")

        elapsed = pinfo.get("elapsed_sec", 0)
        if elapsed > 0:
            h, m = divmod(elapsed // 60, 60)
            self._pipeline_elapsed.setText(f"Elapsed: {h:.0f}h {m:.0f}m")
        else:
            self._pipeline_elapsed.setText("Elapsed: \u2014")

        # Show final scores from completed phases. Tri-state (U2): the key
        # can be present-but-None (the search itself failed — status.py
        # writes None instead of echoing a stale/zero best_score) vs. the
        # key being absent entirely (phase hasn't run yet this session) —
        # those two must render differently, not both fall back to blank.
        scores_parts = []
        if "crypto_final_score" in pinfo:
            crypto_final = pinfo.get("crypto_final_score")
            if crypto_final is not None:
                scores_parts.append(f"Crypto: {crypto_final:.4f}")
            else:
                scores_parts.append(
                    f"Crypto: <span style='color:{T['red'].name()}'>"
                    f"search failed</span>")
        if "stock_final_score" in pinfo:
            stock_final = pinfo.get("stock_final_score")
            if stock_final is not None:
                scores_parts.append(f"Stock: {stock_final:.4f}")
            else:
                scores_parts.append(
                    f"Stock: <span style='color:{T['red'].name()}'>"
                    f"search failed</span>")
        self._pipeline_scores.setText("  |  ".join(scores_parts) if scores_parts else "")

        # Per-phase outcome badges (U1) — a gate rollback (model already
        # reverted to .prev by backtest.py --gate) is otherwise invisible
        # in the GUI, so it gets its own badge distinct from a hard failure.
        pr = pinfo.get("phase_results", {})
        if pr:
            badge_parts = []
            for phase_id, info in pr.items():
                if not isinstance(info, dict):
                    continue
                outcome = info.get("outcome")
                rc = info.get("rc")
                attempts = info.get("attempts", 1)
                if outcome == "ok":
                    badge, badge_color = "✓", T["green"].name()
                elif outcome == "gate_failed_rolled_back":
                    badge, badge_color = "⤺ rolled back", T["yellow"].name()
                elif outcome == "failed":
                    badge, badge_color = "✗", T["red"].name()
                else:
                    badge, badge_color = str(outcome), T["muted"].name()
                text = f"{phase_id}: {badge}"
                if attempts and attempts > 1:
                    text += f" (rc{rc}, {attempts}x)"
                badge_parts.append(
                    f"<span style='color:{badge_color}'>{text}</span>")
            self._pipeline_phase_results.setText("  ".join(badge_parts))
        else:
            self._pipeline_phase_results.setText("")

        # --- Command acknowledgement (command_result.json, written by
        # run_pipeline.py). Render the last verdict if fresh (<10 min) so a
        # rejected start/stop is visible instead of a stale optimistic label. ---
        try:
            with open(BASE_DIR / "command_result.json") as f:
                cr = json.load(f)
            age = now_ts - float(cr.get("ts", 0))
            if 0 <= age < PIPELINE_STALE_SEC:
                book = ("both" if cr.get("crypto") and cr.get("stock")
                        else "crypto" if cr.get("crypto")
                        else "stock" if cr.get("stock") else "")
                cmd = str(cr.get("command", "?"))
                cmd_str = f"{cmd}({book})" if book else cmd
                res = str(cr.get("result", "")).upper()
                reason = str(cr.get("reason", "")).strip()
                ago = f"{age:.0f}s ago" if age < 90 else f"{age / 60:.0f}m ago"
                color = (T["green"] if res == "ACCEPTED" else T["red"]).name()
                txt = f"Last command: {cmd_str} — {res}"
                if reason:
                    txt += f": {reason}"
                self._pipeline_cmd_ack.setText(
                    f"<span style='color:{color}'>{txt} ({ago})</span>")
                # Cockpit alert on a rejected command — once per distinct result
                # timestamp so a fresh-but-unchanged ack doesn't re-alert.
                ts_cmd = float(cr.get("ts", 0))
                if (res and res != "ACCEPTED"
                        and ts_cmd != self._alert_last_cmd_ts):
                    self._alert_last_cmd_ts = ts_cmd
                    self._push_alert(
                        'rejected', f"command {cmd_str} {res}"
                        + (f": {reason}" if reason else ""))
            else:
                self._pipeline_cmd_ack.setText("")
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            self._pipeline_cmd_ack.setText("")

        # Next retrain time
        next_retrain = pinfo.get("next_retrain")
        retrain_cycle = pinfo.get("retrain_cycle", 0)
        retrain_text = ""
        if next_retrain:
            try:
                rt = dt.datetime.fromisoformat(next_retrain)
                retrain_text = f"Next retrain: {rt.strftime('%a %m/%d %I:%M %p')}"
                if retrain_cycle:
                    retrain_text += f"  (cycle {retrain_cycle})"
            except (ValueError, TypeError):
                pass
        self._pipeline_retrain.setText(retrain_text)

        # --- Retrain button states ---
        trigger_path = BASE_DIR / "retrain_trigger.json"
        trigger_pending = trigger_path.exists()
        is_actively_training = (is_running and phase not in
                                ("trading", "idle", "failed", "complete", ""))
        # Auto-expire stale trigger (>5 min unconsumed while the pipeline is
        # in a trigger-polling state = pipeline didn't pick it up). During
        # training the engine intentionally doesn't read it (only the trading
        # wait loop does), so keep the mtime fresh instead — the 5-min clock
        # starts once training ends.
        if trigger_pending:
            if is_actively_training or not is_running:
                # During training the engine intentionally doesn't poll;
                # with the pipeline DOWN it will consume the trigger on
                # next startup — keep the queued intent alive in both
                try:
                    trigger_path.touch()
                except OSError:
                    pass
            else:
                try:
                    trigger_age = now_ts - trigger_path.stat().st_mtime
                    if trigger_age > 300:
                        trigger_path.unlink()
                        trigger_pending = False
                except OSError:
                    trigger_pending = False

        if trigger_pending:
            # Trigger written, waiting for pipeline to pick it up — show cancel
            for btn in [self._retrain_crypto_btn, self._retrain_stock_btn, self._retrain_both_btn]:
                btn.setEnabled(False)
            self._retrain_cancel_btn.setVisible(True)
            self._retrain_status.setText("Retrain queued — waiting for pipeline...")
            self._retrain_status.setStyleSheet(f"color: {T['accent'].name()}; font-size: 11px;")
        elif not is_running:
            for btn in [self._retrain_crypto_btn, self._retrain_stock_btn, self._retrain_both_btn]:
                btn.setEnabled(False)
            self._retrain_cancel_btn.setVisible(False)
            self._retrain_status.setText("Pipeline not running")
            self._retrain_status.setStyleSheet(f"color: {T['muted'].name()}; font-size: 11px;")
        else:
            for btn in [self._retrain_crypto_btn, self._retrain_stock_btn, self._retrain_both_btn]:
                btn.setEnabled(True)
            self._retrain_cancel_btn.setVisible(False)
            if is_actively_training:
                # Training in progress — show which phase
                training_what = ""
                if "crypto" in phase:
                    training_what = "crypto"
                elif "stock" in phase:
                    training_what = "stock"
                self._retrain_status.setText(
                    f"Training {training_what} in progress" if training_what
                    else "Training in progress")
                self._retrain_status.setStyleSheet(f"color: {T['accent'].name()}; font-size: 11px;")
            else:
                # Clear stale status messages
                cur = self._retrain_status.text().lower()
                if "queued" in cur or "cancelled" in cur:
                    self._retrain_status.setText("")

        # --- Per-bot status ---
        crypto_running = pinfo.get("crypto_bot_running", False) or combined_bots
        stock_running = pinfo.get("stock_bot_running", False) or combined_bots

        if is_running:
            if crypto_running:
                self._crypto_bot_label.setText(
                    "Crypto Bot: Running (combined)" if combined_bots
                    else "Crypto Bot: Running")
                self._crypto_bot_label.setStyleSheet(
                    f"font-size: 13px; font-weight: bold; color: {T['green'].name()};")
                self._crypto_start_btn.setEnabled(False)
                self._crypto_stop_btn.setEnabled(not combined_bots)
            else:
                self._crypto_bot_label.setText("Crypto Bot: Stopped")
                self._crypto_bot_label.setStyleSheet(
                    f"font-size: 13px; font-weight: bold; color: {T['muted'].name()};")
                self._crypto_start_btn.setEnabled(True)
                self._crypto_stop_btn.setEnabled(False)

            if stock_running:
                self._stock_bot_label.setText(
                    "Stock Bot: Running (combined)" if combined_bots
                    else "Stock Bot: Running")
                self._stock_bot_label.setStyleSheet(
                    f"font-size: 13px; font-weight: bold; color: {T['green'].name()};")
                self._stock_start_btn.setEnabled(False)
                self._stock_stop_btn.setEnabled(not combined_bots)
            else:
                self._stock_bot_label.setText("Stock Bot: Stopped")
                self._stock_bot_label.setStyleSheet(
                    f"font-size: 13px; font-weight: bold; color: {T['muted'].name()};")
                self._stock_start_btn.setEnabled(True)
                self._stock_stop_btn.setEnabled(False)
        else:
            for lbl in [self._crypto_bot_label, self._stock_bot_label]:
                lbl.setText(lbl.text().split(":")[0] + ": --")
                lbl.setStyleSheet(
                    f"font-size: 13px; font-weight: bold; color: {T['muted'].name()};")
            for btn in [self._crypto_start_btn, self._crypto_stop_btn,
                        self._stock_start_btn, self._stock_stop_btn]:
                btn.setEnabled(False)

        # --- LLM Usage ---
        try:
            from llm_client import get_daily_cost, get_budget, GEMINI_MODELS
            spent, limit = get_daily_cost()
            pct = int(spent / limit * 100) if limit > 0 else 0
            self._llm_cost_label.setText(f"Cost: ${spent:.3f} / ${limit:.2f}")
            if pct < 50:
                cost_color = T['green'].name()
            elif pct < 80:
                cost_color = T['yellow'].name()
            else:
                cost_color = T['red'].name()
            self._llm_cost_bar.setValue(min(pct, 100))
            self._llm_cost_bar.setStyleSheet(
                f"QProgressBar::chunk {{ background-color: {cost_color}; }}")

            for model in GEMINI_MODELS:
                remaining, total = get_budget(model)
                # Extract readable short name from model ID
                if "flash-lite" in model or "flash_lite" in model:
                    label, short_name = self._llm_lite_label, "Lite"
                elif "pro" in model:
                    label, short_name = self._llm_pro_label, "Pro"
                else:
                    label, short_name = self._llm_flash_label, "Flash"
                label.setText(f"{short_name}: {remaining}/{total}")
        except Exception:
            pass

        # Shadow/DM-HLN + drift panels piggyback the same 60s Models refresh
        # (both fully guarded — their JSON producers run only on the Jetson).
        try:
            self._refresh_shadow_panel()
        except Exception:
            pass
        try:
            self._refresh_drift_panel()
        except Exception:
            pass
        try:
            self._refresh_meta_panel()
        except Exception:
            pass
        try:
            self._refresh_reports_freshness()
        except Exception:
            pass

        # Cockpit landing piggybacks this 60s timer: refresh the halt echo /
        # heartbeats / risk gauge / DD badge, and reload the recent-trades feed
        # from the journals (both guarded — journals may be absent on this Mac).
        try:
            self._refresh_cockpit()
        except Exception:
            pass
        try:
            self._refresh_last_actions()
        except Exception:
            pass

    def closeEvent(self, event):
        from PySide6.QtCore import QMetaObject

        # 1. Stop main-thread timers immediately
        self._model_timer.stop()
        self._clock_timer.stop()
        self._perf_timer.stop()

        # 2. Ask worker threads to stop their timers (queued → runs before quit)
        QMetaObject.invokeMethod(self._fetcher_hot, "stop_timers", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self._fetcher_slow, "stop_timers", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self._tailer, "stop_timer", Qt.QueuedConnection)

        # 3. Signal threads to stop, then ask event loops to exit
        self._fetcher_hot_thread.requestInterruption()
        self._fetcher_slow_thread.requestInterruption()
        self._tailer_thread.requestInterruption()
        self._fetcher_hot_thread.quit()
        self._fetcher_slow_thread.quit()
        self._tailer_thread.quit()

        # 4. Wait briefly for clean shutdown; os._exit fallback handles stragglers
        threads_stuck = False
        for name, thread in [("fetcher-hot", self._fetcher_hot_thread),
                             ("fetcher-slow", self._fetcher_slow_thread),
                             ("tailer", self._tailer_thread)]:
            if not thread.wait(2000):
                print(f"WARNING: {name} thread did not stop in time")
                threads_stuck = True

        # 5. Flush news cache so recent articles persist (news lives on the slow
        #    fetcher; its thread is already stopped/waited above).
        if self._news_articles:
            try:
                _save_news_cache(self._news_articles, self._news_fng)
            except Exception:
                pass

        super().closeEvent(event)

        # 6. If threads are stuck in blocking network calls, force-exit to
        #    avoid "QThread: Destroyed while thread is still running" crash
        if threads_stuck:
            import os
            os._exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_app_fonts():
    """Register every bundled fonts/*.ttf so the Jetson renders real typography
    (Inter + IBM Plex Mono) instead of the DejaVu Sans fallback. Each file is
    optional — a missing or unloadable font is silently skipped (graceful
    degradation per fonts/README.md); apply_theme's family stack then falls
    back to system fonts. Must run after QApplication exists, before styling."""
    try:
        for path in sorted((BASE_DIR / "fonts").glob("*.ttf")):
            try:
                QFontDatabase.addApplicationFont(str(path))
            except Exception:
                pass
    except Exception:
        pass


def main():
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        print("ERROR: ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env")
        sys.exit(1)

    api = get_api()

    try:
        acct = api.get_account()
        print(f"Connected to Alpaca. Equity: ${float(acct.equity):,.2f}")
    except Exception as e:
        print(f"ERROR: Cannot connect to Alpaca API: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    _load_app_fonts()  # register Inter + IBM Plex Mono before any styling
    app.setApplicationName("Trader Dashboard")
    app.setDesktopFileName("trader-dashboard")
    app_icon = BASE_DIR / "logos" / "circuit_bull.png"
    if app_icon.exists():
        app.setWindowIcon(QIcon(str(app_icon)))

    saved_theme = _load_gui_settings().get('theme', 'Batman')
    if saved_theme not in THEMES:
        saved_theme = 'Batman'
    set_theme(saved_theme)
    apply_theme(app)

    window = TradingDashboard(api, app)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
