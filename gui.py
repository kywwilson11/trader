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
import json
import pickle
import datetime as dt
from pathlib import Path
from collections import defaultdict
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, Slot, QObject,
)
from PySide6.QtGui import QColor, QPalette, QFont, QAction, QPainter, QPixmap, QDesktopServices, QIcon
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPlainTextEdit, QComboBox, QCheckBox, QFrame,
    QSplitter, QGroupBox, QProgressBar, QToolBar,
    QSizePolicy, QLineEdit, QPushButton, QSpinBox,
    QScrollArea,
)
import pyqtgraph as pg
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TZ_CENTRAL = ZoneInfo("America/Chicago")

LOG_FILES = {
    "Pipeline": BASE_DIR / "pipeline_output.log",
    "Crypto Bot": BASE_DIR / "crypto_bot_output.log",
    "Stock Bot": BASE_DIR / "stock_bot_output.log",
}

CONFIG_FILES = {
    "Crypto Bear": BASE_DIR / "bear_config.pkl",
    "Crypto Bull": BASE_DIR / "bull_config.pkl",
    "Stock Bear": BASE_DIR / "stock_bear_config.pkl",
    "Stock Bull": BASE_DIR / "stock_bull_config.pkl",
}

MODEL_FILES = {
    "Crypto Bear": BASE_DIR / "bear_model.pth",
    "Crypto Bull": BASE_DIR / "bull_model.pth",
    "Stock Bear": BASE_DIR / "stock_bear_model.pth",
    "Stock Bull": BASE_DIR / "stock_bull_model.pth",
}

# Tax rates
FED_SHORT_TERM = 0.37
FED_LONG_TERM = 0.20
STATE_RATE = 0.05

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
}

# Active theme — module-level so helpers can reference it
T = THEMES["Batman"]


def set_theme(name):
    """Switch the active theme colors."""
    global T
    T = THEMES[name]


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
}


_THEME_IMAGES = {
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


def generate_theme_logo(theme_name, size=80):
    """Generate a logo icon for the given theme.

    Character themes load image files from logos/.
    Other themes use SVG rendering.
    Returns a QPixmap of the requested size.
    """
    # Try image file first
    img_path = _THEME_IMAGES.get(theme_name)
    if img_path and img_path.exists():
        pix = QPixmap(str(img_path))
        if not pix.isNull():
            return pix.scaled(size, size, Qt.KeepAspectRatio,
                              Qt.SmoothTransformation)

    # Fall back to SVG
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtCore import QByteArray

    svg_data = _THEME_SVGS.get(theme_name)
    if not svg_data:
        pix = QPixmap(size, size)
        pix.fill(QColor(0, 0, 0, 0))
        return pix

    renderer = QSvgRenderer(QByteArray(svg_data.encode("utf-8")))
    pix = QPixmap(size, size)
    pix.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pix)
    renderer.render(painter)
    painter.end()
    return pix


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
    """Return green/red/white QColor based on sign."""
    try:
        v = float(val)
        if v > 0:
            return T["green"]
        elif v < 0:
            return T["red"]
    except (TypeError, ValueError):
        pass
    return T["white"]


def make_card(title, value="\u2014", parent=None):
    """Create a styled info card widget."""
    frame = QFrame(parent)
    frame.setProperty("card", True)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 8, 12, 8)

    lbl_title = QLabel(title)
    lbl_title.setProperty("card_title", True)
    lbl_title.setAlignment(Qt.AlignLeft)

    lbl_value = QLabel(str(value))
    lbl_value.setAlignment(Qt.AlignLeft)
    lbl_value.setObjectName("card_value")

    layout.addWidget(lbl_title)
    layout.addWidget(lbl_value)
    return frame


def read_config(path):
    """Safely read a pickle config file."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _read_pipeline_status():
    """Read pipeline_status.json, returning empty dict on failure."""
    try:
        with open(BASE_DIR / "pipeline_status.json") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Data Fetcher Thread
# ---------------------------------------------------------------------------
class DataFetcher(QObject):
    """Fetches data from Alpaca API on background timers."""

    account_updated = Signal(dict)
    positions_updated = Signal(list)
    orders_updated = Signal(list)
    history_updated = Signal(dict)
    hw_updated = Signal(dict)
    news_updated = Signal(dict)
    stocks_updated = Signal(dict)
    chart_updated = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, api):
        super().__init__()
        self.api = api
        # sym -> {'daily': {closes,timestamps,cached_at}, 'hourly': ..., '15min': ...}
        self._chart_cache = {}
        self._chart_cache_ttl = 300  # 5 min before re-fetching same resolution

    @Slot()
    def start_timers(self):
        """Create and start all polling timers (called after moveToThread)."""
        self._timer_account = QTimer(self)
        self._timer_account.timeout.connect(self.fetch_account)
        self._timer_account.start(10_000)

        self._timer_positions = QTimer(self)
        self._timer_positions.timeout.connect(self.fetch_positions)
        self._timer_positions.start(5_000)

        self._timer_orders = QTimer(self)
        self._timer_orders.timeout.connect(self.fetch_orders)
        self._timer_orders.start(30_000)

        self._timer_history = QTimer(self)
        self._timer_history.timeout.connect(self.fetch_history)
        self._timer_history.start(300_000)

        self._timer_hw = QTimer(self)
        self._timer_hw.timeout.connect(self.fetch_hw)
        self._timer_hw.start(5_000)

        self._timer_news = QTimer(self)
        self._timer_news.timeout.connect(self.fetch_news)
        self._timer_news.start(300_000)  # 5 min

        self._timer_stocks = QTimer(self)
        self._timer_stocks.timeout.connect(self.fetch_stocks)
        self._timer_stocks.start(30_000)  # 30s

        # Load cached news immediately (instant startup)
        cached = _load_news_cache()
        if cached and cached.get('articles'):
            self.news_updated.emit({
                'articles': cached['articles'],
                'fng': cached.get('fng'),
            })

        # Immediate first fetch (news will merge with cache)
        # Check interruption between calls so closeEvent can break the burst
        for fn in (self.fetch_account, self.fetch_positions, self.fetch_orders,
                   self.fetch_history, self.fetch_hw, self.fetch_news,
                   self.fetch_stocks):
            if QThread.currentThread().isInterruptionRequested():
                return
            fn()

    @Slot()
    def stop_timers(self):
        """Stop all timers (must be called from this object's thread)."""
        for attr in ("_timer_account", "_timer_positions", "_timer_orders",
                      "_timer_history", "_timer_hw", "_timer_news",
                      "_timer_stocks"):
            timer = getattr(self, attr, None)
            if timer:
                timer.stop()

    @Slot()
    def fetch_account(self):
        try:
            acct = self.api.get_account()
            self.account_updated.emit({
                "equity": acct.equity,
                "cash": acct.cash,
                "buying_power": acct.buying_power,
                "last_equity": acct.last_equity,
                "portfolio_value": acct.portfolio_value,
            })
        except Exception as e:
            self.error_occurred.emit(f"Account fetch: {e}")

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
        except Exception as e:
            self.error_occurred.emit(f"Positions fetch: {e}")

    @Slot()
    def fetch_orders(self):
        try:
            # Only show orders after clean-slate cutoff (if set)
            after = None
            slate = BASE_DIR / ".clean_slate"
            if slate.exists():
                after = slate.read_text().strip()
            orders = self.api.list_orders(status="all", limit=100, after=after)
            data = []
            for o in orders:
                data.append({
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
            self.orders_updated.emit(data)
        except Exception as e:
            self.error_occurred.emit(f"Orders fetch: {e}")

    @Slot()
    def fetch_history(self):
        try:
            hist = self.api.get_portfolio_history(period="1M", timeframe="1D")
            self.history_updated.emit({
                "equity": list(hist.equity),
                "timestamp": list(hist.timestamp),
                "profit_loss": list(hist.profit_loss) if hist.profit_loss else [],
                "profit_loss_pct": list(hist.profit_loss_pct) if hist.profit_loss_pct else [],
            })
        except Exception as e:
            self.error_occurred.emit(f"History fetch: {e}")

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
            })
        except Exception as e:
            self.error_occurred.emit(f"HW fetch: {e}")

    @Slot()
    def fetch_news(self):
        """Fetch news headlines from Finnhub and Fear & Greed Index."""
        try:
            from sentiment import get_fear_greed, _get_finnhub, score_article_batch
            from crypto_loop import CRYPTO_SYMBOLS
            import datetime as _dt

            # Build base-symbol set for crypto headline matching
            crypto_bases = {s.split('/')[0] for s in CRYPTO_SYMBOLS}

            articles = []
            fng = get_fear_greed()

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

                # Company news for high-activity stocks
                _today = _dt.date.today()
                _from = (_today - _dt.timedelta(days=2)).isoformat()
                _to = _today.isoformat()
                for stock in ['TSLA', 'NVDA', 'META', 'AMD', 'PLTR',
                              'COIN', 'MARA', 'MSTR', 'SOFI', 'HOOD']:
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
                        cached_scores[key] = ca['_sentiment']

            # Split articles into already-scored (from cache) and new
            need_scoring = []
            for a in articles:
                key = a.get('headline', '').strip().lower()
                if key in cached_scores:
                    a['_sentiment'] = cached_scores[key]
                else:
                    need_scoring.append(a)

            # Only score genuinely new articles
            if need_scoring:
                scores = score_article_batch(need_scoring)
                for a, score in zip(need_scoring, scores):
                    a['_sentiment'] = score

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

            # Save merged articles to cache
            _save_news_cache(articles, fng)

            self.news_updated.emit({
                'articles': articles,
                'fng': fng,
            })
        except Exception as e:
            self.error_occurred.emit(f"News fetch: {e}")

    @Slot(str, str)
    def fetch_chart(self, symbol, resolution):
        """Fetch bars for a symbol at a given resolution (background thread).

        resolution: 'daily' | 'hourly' | '15min' | '5min'
        Uses get_crypto_bars() for crypto symbols (containing '/').
        Caches per (symbol, resolution) so zoom switches within the same
        resolution are instant.
        """
        try:
            import datetime as _dt

            now_ts = _dt.datetime.now().timestamp()

            # Check cache
            sym_cache = self._chart_cache.get(symbol, {})
            cached = sym_cache.get(resolution)
            if cached and (now_ts - cached['cached_at']) < self._chart_cache_ttl:
                self.chart_updated.emit({
                    'symbol': symbol, 'resolution': resolution,
                    'closes': cached['closes'],
                    'timestamps': cached['timestamps'],
                })
                return

            # Resolution → API params
            res_config = {
                'daily':  ('1Day',  365, 365),
                'hourly': ('1Hour',  10, 168),   # 7 days of hourly = ~168 bars
                '15min':  ('15Min',   2, 104),   # 2 days of 15-min = ~104 bars
            }
            tf, lookback_days, limit = res_config.get(resolution, ('1Day', 365, 365))

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
            for b in bars:
                closes.append(float(b.c))
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

            self._chart_cache[symbol][resolution] = {
                'closes': closes, 'timestamps': timestamps,
                'cached_at': now_ts,
            }

            self.chart_updated.emit({
                'symbol': symbol, 'resolution': resolution,
                'closes': closes, 'timestamps': timestamps,
            })
        except Exception as e:
            self.chart_updated.emit({
                'symbol': symbol, 'resolution': resolution,
                'closes': [], 'timestamps': [], 'error': str(e),
            })

    @Slot()
    def fetch_stocks(self):
        """Fetch stock + crypto snapshots + prediction cache for Markets tab."""
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
                    self.error_occurred.emit(f"Stock snapshots: {e}")

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
                    self.error_occurred.emit(f"Crypto snapshots: {e}")

            # Read prediction caches (written by stock_loop / crypto_loop in jetson env)
            predictions = {}
            for pred_name in ("stock_predictions.json", "crypto_predictions.json"):
                pred_file = BASE_DIR / pred_name
                try:
                    if pred_file.exists():
                        with open(pred_file) as f:
                            predictions.update(json.load(f))
                except (OSError, json.JSONDecodeError):
                    pass

            self.stocks_updated.emit({
                'symbols': symbols,
                'snapshots': snapshots,
                'predictions': predictions,
            })
        except Exception as e:
            self.error_occurred.emit(f"Stocks fetch: {e}")

    @staticmethod
    def _read_gpu_temp():
        """Read GPU temp from thermal zone (no torch, no tegrastats)."""
        for zone in ("/sys/devices/virtual/thermal/thermal_zone1/temp",
                     "/sys/devices/virtual/thermal/thermal_zone0/temp"):
            try:
                with open(zone) as f:
                    return int(f.read().strip()) / 1000.0
            except (FileNotFoundError, ValueError, OSError):
                continue
        try:
            import subprocess
            proc = subprocess.run(
                ["tegrastats", "--interval", "100"],
                capture_output=True, text=True, timeout=2,
            )
            m = re.search(r"GPU@(\d+(?:\.\d+)?)C", proc.stdout + proc.stderr, re.IGNORECASE)
            if m:
                return float(m.group(1))
        except Exception:
            pass
        return None

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
# Tax Estimator
# ---------------------------------------------------------------------------
def _mintax_sort_key(lot, sell_price, sell_time_str):
    """Sort key: losses first (highest cost), then LT gains, then ST gains."""
    gain = sell_price - lot["price"]
    is_loss = gain < 0
    try:
        buy_t = dt.datetime.fromisoformat(lot["time"].replace("Z", "+00:00"))
        sell_t = dt.datetime.fromisoformat(sell_time_str.replace("Z", "+00:00"))
        long_term = (sell_t - buy_t).days >= 365
    except Exception:
        long_term = False
    if is_loss:
        tier = 0
    elif long_term:
        tier = 1
    else:
        tier = 2
    return (tier, -lot["price"])


def estimate_taxes(orders):
    """MinTax lot matching from filled orders. Returns dict of tax info.

    Lots are matched in tax-optimal order: losses first (highest cost basis),
    then long-term gains, then short-term gains — each sub-sorted by highest
    cost basis to minimize realized gain within the tier.
    """
    buys = defaultdict(list)
    realized = []

    filled = [o for o in orders if o.get("status") == "filled" and o.get("filled_avg_price")]
    filled.sort(key=lambda o: o.get("filled_at", ""))

    for o in filled:
        sym = o["symbol"]
        try:
            qty = abs(float(o.get("filled_qty") or o.get("qty") or 0))
            price = float(o["filled_avg_price"])
        except (TypeError, ValueError):
            continue
        if qty == 0:
            continue

        filled_at = o.get("filled_at", "")

        if o["side"] == "buy":
            buys[sym].append({"qty": qty, "price": price, "time": filled_at})
        elif o["side"] == "sell":
            remaining = qty
            buys[sym].sort(key=lambda lot: _mintax_sort_key(lot, price, filled_at))
            while remaining > 0 and buys[sym]:
                lot = buys[sym][0]
                matched = min(remaining, lot["qty"])
                gain = (price - lot["price"]) * matched
                try:
                    buy_time = dt.datetime.fromisoformat(lot["time"].replace("Z", "+00:00"))
                    sell_time = dt.datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
                    days_held = (sell_time - buy_time).days
                except Exception:
                    days_held = 0
                realized.append({
                    "symbol": sym, "gain": gain, "qty": matched,
                    "long_term": days_held >= 365,
                })
                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] <= 0:
                    buys[sym].pop(0)

    total_gain = sum(r["gain"] for r in realized)
    st_gain = sum(r["gain"] for r in realized if not r["long_term"])
    lt_gain = sum(r["gain"] for r in realized if r["long_term"])
    st_tax = max(0, st_gain) * (FED_SHORT_TERM + STATE_RATE)
    lt_tax = max(0, lt_gain) * (FED_LONG_TERM + STATE_RATE)

    return {
        "realized_gain": total_gain,
        "short_term_gain": st_gain,
        "long_term_gain": lt_gain,
        "estimated_tax": st_tax + lt_tax,
        "net_after_tax": total_gain - (st_tax + lt_tax),
        "num_trades": len(realized),
    }




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

    app.setStyleSheet(f"""
        QToolTip {{ color: {t["white"].name()}; background-color: {t["bg_card"].name()};
                    border: 1px solid {t["accent"].name()}; }}
        QTabWidget::pane {{ border: 1px solid {t["bg_border"].name()}; }}
        QTabBar::tab {{
            background: {t["bg_header"].name()}; color: {t["muted"].name()};
            padding: 10px 20px; border: 1px solid {t["bg_border"].name()};
            border-bottom: none; border-top-left-radius: 6px;
            border-top-right-radius: 6px; margin-right: 2px;
            font-size: 13px; font-weight: bold;
        }}
        QTabBar::tab:selected {{
            background: {t["bg_dark"].name()}; color: {t["accent"].name()};
            border-bottom: 2px solid {t["accent"].name()};
        }}
        QTabBar::tab:hover {{ background: {t["bg_hover"].name()}; color: {t["white"].name()}; }}
        QComboBox {{
            background: {t["bg_header"].name()}; color: {t["white"].name()};
            border: 1px solid {t["bg_border"].name()}; padding: 4px 8px; border-radius: 4px;
        }}
        QComboBox::drop-down {{ border: none; }}
        QComboBox QAbstractItemView {{
            background: {t["bg_header"].name()};
            selection-background-color: {t["accent"].name()};
        }}
        QCheckBox {{ spacing: 6px; color: {t["white"].name()}; }}
        QCheckBox::indicator:checked {{
            background-color: {t["accent"].name()};
            border: 1px solid {t["accent"].name()}; border-radius: 2px;
        }}
        QCheckBox::indicator:unchecked {{
            background-color: {t["bg_header"].name()};
            border: 1px solid {t["bg_border"].name()}; border-radius: 2px;
        }}
        QProgressBar {{
            background: {t["bg_header"].name()};
            border: 1px solid {t["bg_border"].name()}; border-radius: 4px;
        }}
        QProgressBar::chunk {{ border-radius: 3px; }}
        QStatusBar {{
            background: {t["bg_dark"].name()}; color: {t["muted"].name()};
            border-top: 1px solid {t["bg_border"].name()};
        }}
        QGroupBox {{ color: {t["accent"].name()}; }}
        QScrollBar:vertical {{ background: {t["bg_dark"].name()}; width: 10px; margin: 0; }}
        QScrollBar::handle:vertical {{
            background: {t["bg_border"].name()}; border-radius: 4px; min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {t["accent"].name()}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QToolBar {{
            background: {t["bg_dark"].name()}; border-bottom: 1px solid {t["bg_border"].name()};
            spacing: 8px; padding: 4px;
        }}
        QLabel#toolbar_label {{ color: {t["muted"].name()}; font-size: 11px; }}
    """)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class TradingDashboard(QMainWindow):
    def __init__(self, api, app):
        super().__init__()
        self.api = api
        self._app = app
        self.setWindowTitle("Trader Dashboard")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        self._orders_cache = []
        self._hw_cache = {}

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
        self._error_count = 0
        self.statusBar().addWidget(self._status_updated)
        self.statusBar().addWidget(self._status_positions)
        self.statusBar().addWidget(self._status_sentiment)
        self.statusBar().addPermanentWidget(self._status_conn)
        self.statusBar().addPermanentWidget(self._status_gpu)
        self.statusBar().addPermanentWidget(self._status_ram)
        self.statusBar().addPermanentWidget(self._status_gpu_info)

        # Apply initial styling
        self._restyle()

        # Data fetcher on background thread
        self._fetcher_thread = QThread()
        self._fetcher = DataFetcher(api)
        self._fetcher.moveToThread(self._fetcher_thread)
        self._fetcher.account_updated.connect(self.on_account)
        self._fetcher.positions_updated.connect(self.on_positions)
        self._fetcher.orders_updated.connect(self.on_orders)
        self._fetcher.history_updated.connect(self.on_history)
        self._fetcher.hw_updated.connect(self.on_hw)
        self._fetcher.news_updated.connect(self.on_news)
        self._fetcher.stocks_updated.connect(self.on_stocks)
        self._fetcher.chart_updated.connect(self.on_chart)
        self._fetcher.error_occurred.connect(self.on_error)
        self._fetcher_thread.started.connect(self._fetcher.start_timers)
        self._fetcher_thread.start()

        # Log tailer on background thread
        self._tailer_thread = QThread()
        self._tailer = LogTailer()
        self._tailer.moveToThread(self._tailer_thread)
        self._tailer.new_lines.connect(self.on_log_lines)
        self._tailer_thread.started.connect(self._tailer.start_timer)
        self._tailer_thread.start()

        # Model refresh timer (every 60s)
        self._model_timer = QTimer(self)
        self._model_timer.timeout.connect(self._refresh_models_tab)
        self._model_timer.start(60_000)
        self._refresh_models_tab()

        # Clock timer (every 1s)
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1_000)
        self._update_clock()

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

    def _update_clock(self):
        now = dt.datetime.now(TZ_CENTRAL)
        self._clock_label_right.setText(now.strftime("%a %b %d, %I:%M:%S %p CT"))

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
            f" gridline-color: {t['bg_border'].name()}; }}"
            f" QTableWidget::item {{ padding: 4px; }}"
            f" QHeaderView::section {{ background-color: {t['bg_header'].name()};"
            f" padding: 6px; border: 1px solid {t['bg_border'].name()}; }}"
        )
        group_style = (
            f"QGroupBox {{ font-weight: bold; border: 1px solid {t['bg_border'].name()};"
            f" border-radius: 6px; margin-top: 8px; padding-top: 16px; }}"
            f" QGroupBox::title {{ subcontrol-position: top left; padding: 0 6px; }}"
        )
        card_style = (
            f"QFrame {{ background-color: {t['bg_card'].name()};"
            f" border-radius: 10px; padding: 14px;"
            f" border: 1px solid {t['bg_border'].name()}; }}"
        )

        # Cards
        all_cards = [
            self._card_equity, self._card_cash, self._card_buying_power,
            self._card_day_pl, self._card_total_pl,
            self._tax_realized, self._tax_st, self._tax_lt,
            self._tax_owed, self._tax_net,
            self._stat_return, self._stat_best, self._stat_worst, self._stat_drawdown,
        ]
        for card in all_cards:
            card.setStyleSheet(card_style)
            title_lbl = card.layout().itemAt(0).widget()
            if title_lbl:
                title_lbl.setStyleSheet(f"color: {t['muted'].name()}; font-size: 11px;")
            val_lbl = card.findChild(QLabel, "card_value")
            if val_lbl:
                val_lbl.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {t['white'].name()};")

        # Tables
        for table in [self._positions_table, self._open_orders_table,
                       self._fills_table, self._model_table, self._news_table,
                       self._stock_table]:
            table.setStyleSheet(table_style)

        # Group boxes
        for group in self.findChildren(QGroupBox):
            group.setStyleSheet(group_style)

        # Plots
        for plot in [self._equity_plot, self._pnl_plot, self._stock_chart]:
            plot.setBackground(t["bg_dark"].name())

        # Plot pen colors
        self._equity_curve.setPen(pg.mkPen(t["accent"].name(), width=2))
        self._stock_chart_line.setPen(pg.mkPen(t["accent"].name(), width=2))

        # Zoom buttons
        for z, btn in self._stock_zoom_buttons.items():
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

        # Settings tab inputs
        input_style = (
            f"QLineEdit, QSpinBox, QComboBox {{ background-color: {t['bg_table'].name()};"
            f" color: {t['white'].name()}; border: 1px solid {t['bg_border'].name()};"
            f" border-radius: 4px; padding: 4px; }}"
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
            for combo in self._settings_models.values():
                combo.setStyleSheet(input_style)
            self._settings_latency.setStyleSheet(input_style)
            self._settings_provider.setStyleSheet(input_style)
            for toggle in self._settings_key_toggles.values():
                toggle.setStyleSheet(btn_style)
        if hasattr(self, '_settings_indicator_preset'):
            self._settings_indicator_preset.setStyleSheet(input_style)
            self._indicator_feature_list.setStyleSheet(
                f"QPlainTextEdit {{ background-color: {t['bg_table'].name()};"
                f" color: {t['white'].name()}; border: 1px solid {t['bg_border'].name()};"
                f" border-radius: 4px; padding: 4px; }}"
            )

        # Clock
        self._clock_label_right.setStyleSheet(
            f"font-size: 12px; font-weight: bold; padding: 0 8px; color: {t['accent'].name()};"
        )

    # ---- Tab 1: Dashboard ------------------------------------------------
    def _build_dashboard_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        cards_layout = QHBoxLayout()
        self._card_equity = make_card("Equity")
        self._card_cash = make_card("Cash")
        self._card_buying_power = make_card("Buying Power")
        self._card_day_pl = make_card("Day P&L")
        self._card_total_pl = make_card("Total P&L")
        for c in [self._card_equity, self._card_cash, self._card_buying_power,
                   self._card_day_pl, self._card_total_pl]:
            cards_layout.addWidget(c)
        layout.addLayout(cards_layout)

        pos_label = QLabel("Open Positions")
        pos_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 8px;")
        layout.addWidget(pos_label)

        self._positions_table = QTableWidget(0, 8)
        self._positions_table.setHorizontalHeaderLabels(
            ["Symbol", "Qty", "Side", "Avg Entry", "Current Price",
             "Mkt Value", "Unrealized P&L", "P&L %"]
        )
        self._positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._positions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._positions_table.setAlternatingRowColors(True)
        layout.addWidget(self._positions_table)

        tax_group = QGroupBox("Tax Estimation (MinTax)")
        tax_layout = QHBoxLayout(tax_group)
        self._tax_realized = make_card("Realized Gains")
        self._tax_st = make_card("Short-Term Gains")
        self._tax_lt = make_card("Long-Term Gains")
        self._tax_owed = make_card("Est. Tax Owed")
        self._tax_net = make_card("Net After Tax")
        for c in [self._tax_realized, self._tax_st, self._tax_lt, self._tax_owed, self._tax_net]:
            tax_layout.addWidget(c)
        layout.addWidget(tax_group)

        self.tabs.addTab(tab, "Dashboard")

    # ---- Tab 2: Trading --------------------------------------------------
    def _build_trading_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

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

        self._open_orders_table = QTableWidget(0, 6)
        self._open_orders_table.setHorizontalHeaderLabels(
            ["Symbol", "Side", "Qty", "Type", "Status", "Submitted (CT)"]
        )
        self._open_orders_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._open_orders_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._open_orders_table.setAlternatingRowColors(True)
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

        self.tabs.addTab(tab, "Trading")

    # ---- Tab 3: Performance ----------------------------------------------
    def _build_performance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        eq_label = QLabel("Equity Curve (1M Daily)")
        eq_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(eq_label)

        self._equity_plot = pg.PlotWidget()
        self._equity_plot.showGrid(x=True, y=True, alpha=0.3)
        self._equity_plot.setLabel("left", "Equity ($)")
        self._equity_plot.setLabel("bottom", "Date")
        self._equity_curve = self._equity_plot.plot(pen=pg.mkPen(width=2))
        layout.addWidget(self._equity_plot)

        pl_label = QLabel("Daily P&L")
        pl_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 8px;")
        layout.addWidget(pl_label)

        self._pnl_plot = pg.PlotWidget()
        self._pnl_plot.showGrid(x=True, y=True, alpha=0.3)
        self._pnl_plot.setLabel("left", "P&L ($)")
        self._pnl_plot.setLabel("bottom", "Date")
        layout.addWidget(self._pnl_plot)

        stats_group = QGroupBox("Performance Stats")
        stats_layout = QHBoxLayout(stats_group)
        self._stat_return = make_card("Total Return")
        self._stat_best = make_card("Best Day")
        self._stat_worst = make_card("Worst Day")
        self._stat_drawdown = make_card("Max Drawdown")
        for c in [self._stat_return, self._stat_best, self._stat_worst, self._stat_drawdown]:
            stats_layout.addWidget(c)
        layout.addWidget(stats_group)

        self.tabs.addTab(tab, "Performance")

    # ---- Tab 4: Models ---------------------------------------------------
    # ---- Tab 4: News ----------------------------------------------------
    def _build_news_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Fear & Greed header
        fng_layout = QHBoxLayout()
        self._news_fng_label = QLabel("Crypto Fear & Greed Index: —")
        self._news_fng_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        fng_layout.addWidget(self._news_fng_label)
        fng_layout.addStretch()
        self._news_refresh_label = QLabel("")
        self._news_refresh_label.setStyleSheet("font-size: 11px;")
        fng_layout.addWidget(self._news_refresh_label)
        layout.addLayout(fng_layout)

        # Filter combo
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        filter_layout.addWidget(filter_label)
        self._news_filter_combo = QComboBox()
        self._news_filter_combo.addItems(["My Universe", "All News", "Global / Macro"])
        self._news_filter_combo.setCurrentIndex(0)
        self._news_filter_combo.currentIndexChanged.connect(self._apply_news_filter)
        self._news_filter_combo.setFixedWidth(180)
        filter_layout.addWidget(self._news_filter_combo)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Initialize article cache
        self._news_articles = []
        self._news_fng = None

        # News table
        self._news_table = QTableWidget(0, 5)
        self._news_table.setHorizontalHeaderLabels(
            ["Time", "Source", "Category", "Headline", "Sentiment"]
        )
        header = self._news_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._news_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._news_table.setAlternatingRowColors(True)
        self._news_table.setWordWrap(True)
        self._news_table.verticalHeader().setDefaultSectionSize(32)
        self._news_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._news_table.setCursor(Qt.PointingHandCursor)
        self._news_table.cellClicked.connect(self._on_news_row_clicked)
        self._news_filtered = []  # track filtered articles for click lookup
        layout.addWidget(self._news_table)

        self.tabs.addTab(tab, "News")

    # ---- Tab 5: Markets --------------------------------------------------
    def _build_stocks_tab(self):
        from stock_config import load_stock_universe, save_stock_universe
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

        # Zoom buttons — these just change the view range, no API call
        self._stock_zoom = "1M"  # default zoom
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

        main_layout.addLayout(top_layout)

        # --- Middle: chart (left) + heatmap (right) via splitter ---
        splitter = QSplitter(Qt.Horizontal)

        # Price chart with date axis
        date_axis = pg.DateAxisItem(orientation='bottom')
        self._stock_chart = pg.PlotWidget(axisItems={'bottom': date_axis})
        self._stock_chart.showGrid(x=True, y=True, alpha=0.3)
        self._stock_chart.setLabel("left", "Price ($)")
        self._stock_chart_line = self._stock_chart.plot(pen=pg.mkPen(width=2))
        self._stock_chart.setMouseEnabled(x=False, y=False)  # zoom via buttons only
        self._stock_chart_data = {}  # resolution -> {closes, timestamps}
        splitter.addWidget(self._stock_chart)

        # Heatmap panel
        heatmap_container = QWidget()
        heatmap_outer = QVBoxLayout(heatmap_container)
        heatmap_outer.setContentsMargins(4, 0, 4, 0)
        hm_title = QLabel("Heatmap")
        hm_title.setStyleSheet("font-size: 13px; font-weight: bold;")
        heatmap_outer.addWidget(hm_title)

        self._heatmap_widget = QWidget()
        self._heatmap_layout = QGridLayout(self._heatmap_widget)
        self._heatmap_layout.setSpacing(3)
        self._heatmap_layout.setContentsMargins(0, 0, 0, 0)
        self._heatmap_labels = {}  # sym -> QLabel
        heatmap_outer.addWidget(self._heatmap_widget)
        heatmap_outer.addStretch()
        splitter.addWidget(heatmap_container)

        splitter.setSizes([600, 300])
        main_layout.addWidget(splitter, stretch=1)

        # --- Bottom: metrics table ---
        self._stock_table = QTableWidget(0, 8)
        self._stock_table.setHorizontalHeaderLabels(
            ["Symbol", "Price", "Day Chg%", "Volume", "Bear", "Bull", "Score", "Signal"]
        )
        self._stock_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._stock_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._stock_table.setAlternatingRowColors(True)
        self._stock_table.setSortingEnabled(True)
        self._stock_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._stock_table.cellDoubleClicked.connect(self._on_stock_table_dblclick)
        main_layout.addWidget(self._stock_table, stretch=1)

        self._stock_data_cache = {}  # latest data from fetch_stocks
        self.tabs.addTab(tab, "Markets")

    def _on_stock_add(self):
        from stock_config import load_stock_universe, save_stock_universe
        text = self._stock_add_input.text().strip().upper()
        if not text:
            return
        symbols = load_stock_universe()
        if text in symbols:
            self._stock_add_input.clear()
            return
        symbols.append(text)
        save_stock_universe(symbols)
        symbols = load_stock_universe()  # re-read sorted
        self._stock_symbol_combo.blockSignals(True)
        self._stock_symbol_combo.clear()
        self._stock_symbol_combo.addItems(symbols)
        self._stock_symbol_combo.setCurrentText(text)
        self._stock_symbol_combo.blockSignals(False)
        self._stock_universe_label.setText(f"Universe ({len(symbols)})")
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
            self._stock_universe_label.setText(f"Universe ({len(symbols)})")

    def _on_stock_symbol_changed(self, sym):
        """Symbol changed — clear old data and fetch for current zoom."""
        self._stock_chart_data = {}
        self._stock_chart_line.clear()
        self._request_chart()

    def _zoom_resolution(self, zoom=None):
        """Map zoom level to the API resolution needed."""
        z = zoom or self._stock_zoom
        if z in ('1Y', '3M', '1M'):
            return 'daily'
        elif z == '1W':
            return 'hourly'
        else:  # '1D'
            return '15min'

    def _on_zoom_clicked(self, zoom):
        self._stock_zoom = zoom
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
            sym = item.text()
            self._stock_symbol_combo.setCurrentText(sym)

    def _on_heatmap_clicked(self, sym):
        self._stock_symbol_combo.setCurrentText(sym)

    def _request_chart(self):
        """Ask DataFetcher to fetch bars at the resolution needed for current zoom."""
        sym = self._stock_symbol_combo.currentText()
        if not sym:
            return
        res = self._zoom_resolution()
        self._stock_chart.setTitle(f"{sym} — Loading...")
        from PySide6.QtCore import QMetaObject, Q_ARG
        QMetaObject.invokeMethod(
            self._fetcher, "fetch_chart", Qt.QueuedConnection,
            Q_ARG(str, sym), Q_ARG(str, res),
        )

    def _apply_chart_zoom(self):
        """Slice cached data to the zoom window and autoscale."""
        res = self._zoom_resolution()
        data = self._stock_chart_data.get(res)
        if not data:
            return
        closes = data.get('closes', [])
        timestamps = data.get('timestamps', [])
        if not closes or not timestamps:
            return

        import datetime as _dt
        now_ts = _dt.datetime.now(_dt.timezone.utc).timestamp()
        zoom_days = {"1Y": 365, "3M": 90, "1M": 30, "1W": 7, "1D": 1}
        cutoff = now_ts - zoom_days.get(self._stock_zoom, 30) * 86400

        # Find first index >= cutoff
        start_idx = 0
        for i, ts in enumerate(timestamps):
            if ts >= cutoff:
                start_idx = i
                break

        vis_ts = timestamps[start_idx:]
        vis_closes = closes[start_idx:]

        if vis_closes:
            self._stock_chart_line.setData(vis_ts, vis_closes)
            y_min = min(vis_closes)
            y_max = max(vis_closes)
            pad = (y_max - y_min) * 0.05 if y_max > y_min else y_max * 0.02
            self._stock_chart.setXRange(vis_ts[0], vis_ts[-1], padding=0.02)
            self._stock_chart.setYRange(y_min - pad, y_max + pad, padding=0)
        else:
            self._stock_chart_line.clear()

        sym = self._stock_symbol_combo.currentText()
        self._stock_chart.setTitle(f"{sym} ({self._stock_zoom})")

    @Slot(dict)
    def on_chart(self, data):
        """Handle chart_updated signal — store into resolution slot, apply zoom."""
        sym = data.get('symbol', '')
        res = data.get('resolution', 'daily')
        closes = data.get('closes', [])
        timestamps = data.get('timestamps', [])
        error = data.get('error')

        # Only update if this symbol is still selected
        if self._stock_symbol_combo.currentText() != sym:
            return

        if error:
            self._stock_chart_line.clear()
            self._stock_chart.setTitle(f"{sym} — Error: {error}")
            return

        if closes and timestamps:
            self._stock_chart_data[res] = {
                'closes': closes, 'timestamps': timestamps,
            }
            # Only apply if this resolution matches current zoom
            if self._zoom_resolution() == res:
                self._apply_chart_zoom()
        else:
            self._stock_chart_line.clear()
            self._stock_chart.setTitle(f"{sym} — No data")

    @Slot(dict)
    def on_stocks(self, data):
        """Handle stocks_updated signal — update heatmap, table, chart."""
        self._stock_data_cache = data
        symbols = data.get('symbols', [])
        snapshots = data.get('snapshots', {})
        predictions = data.get('predictions', {})

        # --- Heatmap ---
        cols = 7  # grid columns
        for idx, sym in enumerate(symbols):
            snap = snapshots.get(sym, {})
            chg = snap.get('change_pct', 0)

            if sym not in self._heatmap_labels:
                lbl = QLabel()
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setFixedSize(62, 38)
                lbl.setCursor(Qt.PointingHandCursor)
                lbl.mousePressEvent = lambda _, s=sym: self._on_heatmap_clicked(s)
                self._heatmap_labels[sym] = lbl
                r, c = divmod(idx, cols)
                self._heatmap_layout.addWidget(lbl, r, c)

            lbl = self._heatmap_labels[sym]
            # Color intensity scales with magnitude (max ±5%)
            intensity = min(abs(chg) / 5.0, 1.0)
            if chg > 0:
                r_val = int(20 + 20 * (1 - intensity))
                g_val = int(80 + 175 * intensity)
                b_val = int(20 + 20 * (1 - intensity))
            elif chg < 0:
                r_val = int(80 + 175 * intensity)
                g_val = int(20 + 20 * (1 - intensity))
                b_val = int(20 + 20 * (1 - intensity))
            else:
                r_val, g_val, b_val = 60, 60, 60

            short = sym.split('/')[0] if '/' in sym else sym
            lbl.setText(f"{short}\n{chg:+.1f}%")
            lbl.setStyleSheet(
                f"background-color: rgb({r_val},{g_val},{b_val});"
                f" color: white; font-size: 10px; font-weight: bold;"
                f" border-radius: 4px; padding: 2px;"
            )

        # Remove stale heatmap labels
        for sym in list(self._heatmap_labels):
            if sym not in symbols:
                lbl = self._heatmap_labels.pop(sym)
                self._heatmap_layout.removeWidget(lbl)
                lbl.deleteLater()

        # --- Metrics table ---
        tbl = self._stock_table
        tbl.setSortingEnabled(False)
        tbl.setUpdatesEnabled(False)
        tbl.setRowCount(len(symbols))

        for row, sym in enumerate(symbols):
            snap = snapshots.get(sym, {})
            pred = predictions.get(sym, {})

            price = snap.get('price', 0)
            chg = snap.get('change_pct', 0)
            vol = snap.get('volume', 0)
            bear = pred.get('bear')
            bull = pred.get('bull')
            score = pred.get('score')
            signal = pred.get('signal', '')

            chg_color = T['green'] if chg > 0 else (T['red'] if chg < 0 else T['white'])

            items_data = [
                (sym, T['white']),
                (f"${price:.2f}" if price else "\u2014", T['white']),
                (f"{chg:+.2f}%", chg_color),
                (f"{vol:,}" if vol else "\u2014", T['muted']),
                (f"{bear:+.4f}" if bear is not None else "\u2014",
                 T['red'] if bear is not None and bear < 0 else T['muted']),
                (f"{bull:+.4f}" if bull is not None else "\u2014",
                 T['green'] if bull is not None and bull > 0 else T['muted']),
                (f"{score:+.4f}" if score is not None else "\u2014",
                 T['green'] if score is not None and score > 0 else
                 (T['red'] if score is not None and score < 0 else T['muted'])),
                (signal, T['green'] if signal == 'BULL' else
                 (T['red'] if signal == 'BEAR' else
                  (T['yellow'] if signal == 'DISAGREE' else T['muted']))),
            ]

            for col, (val, color) in enumerate(items_data):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(color)
                tbl.setItem(row, col, item)

        tbl.setUpdatesEnabled(True)
        tbl.setSortingEnabled(True)

        # Refresh chart if no data yet
        if not hasattr(self, '_stock_chart_loaded'):
            self._stock_chart_loaded = True
            self._request_chart()

    # ---- Tab 6: Models ---------------------------------------------------
    def _build_models_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        model_label = QLabel("Model Status")
        model_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(model_label)

        self._model_table = QTableWidget(0, 9)
        self._model_table.setHorizontalHeaderLabels(
            ["Model", "Status", "Last Trained", "Age",
             "Hidden Dim", "Layers", "Seq Len", "Threshold", "Preset"]
        )
        self._model_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._model_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._model_table.setAlternatingRowColors(True)
        layout.addWidget(self._model_table)

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
        layout.addWidget(pipeline_group)

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

        layout.addWidget(hw_group)
        layout.addStretch()
        self.tabs.addTab(tab, "Models")

    # ---- Tab 7: Logs -----------------------------------------------------
    def _build_logs_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

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

        self._log_display = QPlainTextEdit()
        self._log_display.setReadOnly(True)
        self._log_display.setFont(QFont("Monospace", 10))
        self._log_display.setMaximumBlockCount(5000)
        layout.addWidget(self._log_display)

        self._log_buffers = {name: "" for name in LOG_FILES}
        self._on_log_selected(self._log_selector.currentText())

        self.tabs.addTab(tab, "Logs")

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
        top_row.addSpacing(16)
        top_row.addWidget(QLabel("Provider:"))
        self._settings_provider = QComboBox()
        self._settings_provider.addItems(["gemini", "claude", "openai"])
        self._settings_provider.setCurrentText(config.get("provider", "gemini"))
        self._settings_provider.setMaximumWidth(120)
        self._settings_provider.currentTextChanged.connect(self._on_settings_changed)
        top_row.addWidget(self._settings_provider)
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
        for i, (provider, label) in enumerate([
            ("gemini", "Gemini"), ("claude", "Claude"),
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

        # Model Selection — non-editable dropdowns
        model_group = QGroupBox("Model Selection")
        model_layout = QGridLayout(model_group)
        model_layout.setColumnStretch(1, 1)
        model_layout.setColumnMinimumWidth(0, 55)

        self._settings_models = {}
        model_options = {
            "gemini": [
                ("gemini-2.5-flash-lite", "Flash Lite (free, fastest)"),
                ("gemini-2.5-flash", "Flash (free, balanced)"),
                ("gemini-2.5-pro", "Pro (free, smartest, low RPM)"),
            ],
            "claude": [
                ("claude-haiku-4-5-20251001", "Haiku 4.5 (cheapest)"),
                ("claude-sonnet-4-5-20250929", "Sonnet 4.5"),
                ("claude-opus-4-6", "Opus 4.6 (most capable)"),
            ],
            "openai": [
                ("gpt-4.1-nano", "GPT-4.1 Nano (cheapest)"),
                ("gpt-4.1-mini", "GPT-4.1 Mini"),
                ("gpt-4.1", "GPT-4.1"),
                ("o4-mini", "o4-mini (reasoning)"),
            ],
        }
        for i, (provider, label) in enumerate([
            ("gemini", "Gemini"), ("claude", "Claude"), ("openai", "OpenAI"),
        ]):
            model_layout.addWidget(QLabel(f"{label}:"), i, 0)
            combo = QComboBox()
            combo.setMaximumWidth(320)
            for model_id, display_name in model_options[provider]:
                combo.addItem(display_name, model_id)
            # Set current selection from config
            current_model = config.get("models", {}).get(provider, {}).get("model", "")
            if current_model:
                idx = combo.findData(current_model)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(self._on_settings_changed)
            model_layout.addWidget(combo, i, 1)
            self._settings_models[provider] = combo

        llm_layout.addWidget(model_group)

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
            "full": "Full (all features)",
        }
        for name in ["minimal", "standard", "full"]:
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
        self._indicator_cross_note.setStyleSheet("color: gray; font-size: 10px;")
        self._indicator_cross_note.setWordWrap(True)
        ind_layout.addWidget(self._indicator_cross_note)

        warn_label = QLabel("Changing presets requires model retraining.")
        warn_label.setStyleSheet("color: orange; font-weight: bold; font-size: 10px;")
        ind_layout.addWidget(warn_label)

        self._update_indicator_feature_display()
        self._settings_indicator_preset.currentIndexChanged.connect(
            self._on_indicator_preset_changed)

        layout.addWidget(ind_group)
        layout.addStretch()

        scroll.setWidget(scroll_widget)
        tab_layout.addWidget(scroll)
        self.tabs.addTab(tab, "Settings")

    def _on_settings_changed(self, *_args):
        """Auto-save settings when any field changes."""
        from llm_config import load_llm_config, save_llm_config

        config = load_llm_config()
        config["enabled"] = self._settings_llm_enabled.isChecked()
        config["provider"] = self._settings_provider.currentText()
        config["journal_enabled"] = self._settings_journal.isChecked()
        config["max_llm_latency_sec"] = self._settings_latency.value()
        config["fmp_api_key"] = self._settings_fmp_key.text().strip()

        for provider, key_edit in self._settings_api_keys.items():
            config.setdefault("models", {}).setdefault(provider, {})["api_key"] = key_edit.text().strip()
        for provider, combo in self._settings_models.items():
            model_id = combo.currentData()
            if model_id:
                config.setdefault("models", {}).setdefault(provider, {})["model"] = model_id

        save_llm_config(config)

    def _on_test_llm(self):
        """Test LLM connection with a trivial prompt."""
        self._settings_test_status.setText("Testing...")
        self._settings_test_status.setStyleSheet("")
        QApplication.processEvents()

        # Force-save first so the client reads current keys
        self._on_settings_changed()

        import time
        start = time.time()
        try:
            from llm_client import call_llm
            result = call_llm("Respond with just the word OK.", max_tokens=16)
            elapsed = (time.time() - start) * 1000

            if result:
                provider = self._settings_provider.currentText()
                self._settings_test_status.setText(
                    f"Connected to {provider} ({elapsed:.0f}ms)")
                self._settings_test_status.setStyleSheet(f"color: {T['green'].name()};")
            else:
                self._settings_test_status.setText("No response — check API key")
                self._settings_test_status.setStyleSheet(f"color: {T['red'].name()};")
        except Exception as e:
            self._settings_test_status.setText(f"Error: {e}")
            self._settings_test_status.setStyleSheet(f"color: {T['red'].name()};")

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
    @Slot(dict)
    def on_account(self, data):
        equity = float(data["equity"])
        cash = float(data["cash"])
        buying_power = float(data["buying_power"])
        last_equity = float(data["last_equity"])
        day_pl = equity - last_equity
        total_pl = equity - 100_000

        self._set_card(self._card_equity, fmt_money(equity))
        self._set_card(self._card_cash, fmt_money(cash))
        self._set_card(self._card_buying_power, fmt_money(buying_power))
        self._set_card(self._card_day_pl, fmt_money(day_pl), pnl_color(day_pl))
        self._set_card(self._card_total_pl, fmt_money(total_pl), pnl_color(total_pl))

        now = dt.datetime.now(TZ_CENTRAL).strftime("%I:%M:%S %p")
        self._status_updated.setText(f"Last update: {now}")
        self._status_conn.setText("API: OK")
        self._status_conn.setStyleSheet(f"color: {T['green'].name()};")
        self._error_count = 0

        # Update sentiment indicator
        try:
            from sentiment import get_fear_greed
            fng = get_fear_greed()
            if fng is not None:
                val = fng['value']
                label = fng['label']
                if val <= 25:
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

    @Slot(list)
    def on_positions(self, positions):
        tbl = self._positions_table
        tbl.setUpdatesEnabled(False)
        tbl.setRowCount(len(positions))
        total_unr = 0.0
        for row, p in enumerate(positions):
            try:
                unr = float(p["unrealized_pl"])
            except (TypeError, ValueError):
                unr = 0.0
            total_unr += unr
            try:
                mkt_val = float(p["qty"]) * float(p["current_price"])
            except (TypeError, ValueError):
                mkt_val = 0.0
            items = [
                p["symbol"], p["qty"], p["side"],
                fmt_money(p["avg_entry_price"]),
                fmt_money(p["current_price"]),
                fmt_money(mkt_val),
                fmt_money(p["unrealized_pl"]),
                fmt_pct(float(p["unrealized_plpc"]) * 100),
            ]
            color = pnl_color(p["unrealized_pl"])
            for col, val in enumerate(items):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                if col >= 6:
                    item.setForeground(color)
                tbl.setItem(row, col, item)
        tbl.setUpdatesEnabled(True)
        self._status_positions.setText(
            f"Pos: {len(positions)} | Unr: {fmt_money(total_unr)}")

    @Slot(list)
    def on_orders(self, orders):
        self._orders_cache = orders
        self._apply_trade_filter(self._trade_filter.currentText())
        self._update_tax(orders)

    @Slot(dict)
    def on_history(self, data):
        equities = data.get("equity", [])
        timestamps = data.get("timestamp", [])
        pnl = data.get("profit_loss", [])

        if equities and timestamps:
            x = np.arange(len(equities))
            self._equity_curve.setData(x, equities)

            axis = self._equity_plot.getPlotItem().getAxis("bottom")
            ticks = []
            for i, ts in enumerate(timestamps):
                try:
                    d = dt.datetime.fromtimestamp(ts, tz=TZ_CENTRAL)
                    ticks.append((i, d.strftime("%m/%d")))
                except Exception:
                    pass
            axis.setTicks([ticks])

        if pnl:
            self._pnl_plot.clear()
            pos_x = [i for i, v in enumerate(pnl) if v >= 0]
            pos_h = [v for v in pnl if v >= 0]
            neg_x = [i for i, v in enumerate(pnl) if v < 0]
            neg_h = [v for v in pnl if v < 0]
            if pos_x:
                self._pnl_plot.addItem(pg.BarGraphItem(
                    x=pos_x, height=pos_h, width=0.6, brush=T["green"].name()))
            if neg_x:
                self._pnl_plot.addItem(pg.BarGraphItem(
                    x=neg_x, height=neg_h, width=0.6, brush=T["red"].name()))

            if timestamps and len(timestamps) == len(pnl):
                axis = self._pnl_plot.getPlotItem().getAxis("bottom")
                ticks = []
                for i, ts in enumerate(timestamps):
                    try:
                        d = dt.datetime.fromtimestamp(ts, tz=TZ_CENTRAL)
                        ticks.append((i, d.strftime("%m/%d")))
                    except Exception:
                        pass
                axis.setTicks([ticks])

            total_return = sum(pnl)
            best_day = max(pnl) if pnl else 0
            worst_day = min(pnl) if pnl else 0

            self._set_card(self._stat_return, fmt_money(total_return), pnl_color(total_return))
            self._set_card(self._stat_best, fmt_money(best_day), T["green"])
            self._set_card(self._stat_worst, fmt_money(worst_day), T["red"])

            if equities:
                peak = equities[0]
                max_dd = 0
                for eq in equities:
                    if eq > peak:
                        peak = eq
                    dd = (peak - eq) / peak * 100 if peak else 0
                    if dd > max_dd:
                        max_dd = dd
                self._set_card(self._stat_drawdown, f"-{max_dd:.2f}%",
                               T["red"] if max_dd > 0 else T["white"])

    @Slot(dict)
    def on_hw(self, data):
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

    @Slot(dict)
    def on_news(self, data):
        import datetime as _dt
        articles = data.get('articles', [])
        fng = data.get('fng')

        # Update Fear & Greed
        if fng is not None:
            val = fng['value']
            label = fng['label']
            if val <= 25:
                color = T['red'].name()
            elif val >= 75:
                color = T['green'].name()
            else:
                color = T.get('yellow', T['white']).name()
            self._news_fng_label.setText(
                f"Crypto Fear & Greed Index: "
                f"<span style='color:{color}; font-size:20px;'>{val}</span> "
                f"<span style='color:{color};'>({label})</span>")

        now = _dt.datetime.now(TZ_CENTRAL).strftime("%I:%M %p")
        self._news_refresh_label.setText(f"Updated {now}")

        # Cache articles and apply current filter
        self._news_articles = articles
        self._news_fng = fng
        self._apply_news_filter()

    def _apply_news_filter(self):
        """Filter cached news articles based on combo selection."""
        import datetime as _dt
        from crypto_loop import CRYPTO_SYMBOLS
        from stock_config import load_stock_universe

        idx = self._news_filter_combo.currentIndex()
        articles = getattr(self, '_news_articles', [])

        if idx == 0:  # My Universe
            crypto_bases = {s.split('/')[0] for s in CRYPTO_SYMBOLS}
            stock_set = set(load_stock_universe())
            universe = crypto_bases | stock_set
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
                    if re.search(r'\b' + re.escape(s) + r'\b', headline_upper):
                        filtered.append(a)
                        break
            articles = filtered
        elif idx == 2:  # Global / Macro
            articles = [a for a in articles if a.get('_category') == 'Market']
        # idx == 1 (All News) — no filtering

        self._news_filtered = articles  # store for click lookup

        tbl = self._news_table
        tbl.setUpdatesEnabled(False)
        tbl.setRowCount(len(articles))
        for row, a in enumerate(articles):
            ts = a.get('datetime', 0)
            if ts:
                time_str = _dt.datetime.fromtimestamp(ts, tz=TZ_CENTRAL).strftime("%m/%d %I:%M %p")
            else:
                time_str = "—"

            source = a.get('source', '—')
            category = a.get('_category', '—')
            headline = a.get('headline', '—')
            summary = a.get('summary', '')
            sentiment = a.get('_sentiment', 0.0)

            if sentiment > 0.1:
                sent_color = T['green']
                sent_text = f"+{sentiment:.2f}"
            elif sentiment < -0.1:
                sent_color = T['red']
                sent_text = f"{sentiment:.2f}"
            else:
                sent_color = T.get('yellow', T['white'])
                sent_text = f"{sentiment:.2f}"

            # Tooltip: show summary on hover for every cell in the row
            tooltip = summary if summary else ''

            items_data = [time_str, source, category, headline, sent_text]
            for col, val in enumerate(items_data):
                item = QTableWidgetItem(str(val))
                if col == 4:  # Sentiment
                    item.setForeground(sent_color)
                    item.setTextAlignment(Qt.AlignCenter)
                elif col <= 2:  # Time, Source, Category
                    item.setTextAlignment(Qt.AlignCenter)
                if tooltip:
                    item.setToolTip(tooltip)
                tbl.setItem(row, col, item)
        tbl.setUpdatesEnabled(True)

    def _on_news_row_clicked(self, row, _col):
        """Open the article URL in the system browser."""
        articles = getattr(self, '_news_filtered', [])
        if 0 <= row < len(articles):
            url = articles[row].get('url', '')
            if url:
                QDesktopServices.openUrl(QUrl(url))

    @Slot(str)
    def on_error(self, msg):
        self._error_count += 1
        self._status_conn.setText(f"API: ERR({self._error_count})")
        self._status_conn.setStyleSheet(f"color: {T['red'].name()};")
        self.statusBar().showMessage(f"Error: {msg}", 10_000)

    @Slot(str, str)
    def on_log_lines(self, name, text):
        self._log_buffers[name] = self._log_buffers.get(name, "") + text
        if len(self._log_buffers[name]) > 200_000:
            self._log_buffers[name] = self._log_buffers[name][-150_000:]

        if self._log_selector.currentText() == name:
            self._log_display.appendPlainText(text.rstrip("\n"))
            if self._auto_scroll.isChecked():
                self._log_display.verticalScrollBar().setValue(
                    self._log_display.verticalScrollBar().maximum())

    # ---- Private Helpers -------------------------------------------------
    def _set_card(self, card, value, color=None):
        lbl = card.findChild(QLabel, "card_value")
        if lbl:
            lbl.setText(str(value))
            if color:
                lbl.setStyleSheet(
                    f"font-size: 22px; font-weight: bold; color: {color.name()};")

    def _apply_trade_filter(self, filter_text):
        orders = self._orders_cache
        crypto_symbols = {
            "BTCUSD", "ETHUSD", "XRPUSD", "SOLUSD", "DOGEUSD",
            "LINKUSD", "AVAXUSD", "DOTUSD", "LTCUSD", "BCHUSD",
            "BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "DOGE/USD",
            "LINK/USD", "AVAX/USD", "DOT/USD", "LTC/USD", "BCH/USD",
        }
        if filter_text == "Crypto":
            orders = [o for o in orders if o["symbol"] in crypto_symbols]
        elif filter_text == "Stock":
            orders = [o for o in orders if o["symbol"] not in crypto_symbols]

        open_orders = [o for o in orders if o["status"] in
                       ("new", "accepted", "partially_filled", "pending_new")]
        filled_orders = [o for o in orders if o["status"] == "filled"]

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

    def _update_tax(self, orders):
        tax = estimate_taxes(orders)
        self._set_card(self._tax_realized, fmt_money(tax["realized_gain"]),
                       pnl_color(tax["realized_gain"]))
        self._set_card(self._tax_st, fmt_money(tax["short_term_gain"]),
                       pnl_color(tax["short_term_gain"]))
        self._set_card(self._tax_lt, fmt_money(tax["long_term_gain"]),
                       pnl_color(tax["long_term_gain"]))
        self._set_card(self._tax_owed, fmt_money(tax["estimated_tax"]), T["red"])
        self._set_card(self._tax_net, fmt_money(tax["net_after_tax"]),
                       pnl_color(tax["net_after_tax"]))

    def _on_log_selected(self, name):
        self._log_display.clear()
        buf = self._log_buffers.get(name, "")
        if not buf:
            path = LOG_FILES.get(name)
            if path and path.exists():
                try:
                    text = path.read_text(errors="replace")
                    if len(text) > 100_000:
                        text = text[-100_000:]
                    self._log_buffers[name] = text
                    buf = text
                except OSError:
                    pass
        if buf:
            self._log_display.setPlainText(buf)
            if self._auto_scroll.isChecked():
                self._log_display.verticalScrollBar().setValue(
                    self._log_display.verticalScrollBar().maximum())

    def _refresh_models_tab(self):
        now_ts = dt.datetime.now().timestamp()
        configs = []
        for name, cfg_path in CONFIG_FILES.items():
            cfg = read_config(cfg_path)
            model_path = MODEL_FILES.get(name)
            mod_time = "\u2014"
            age_hours = None
            if model_path and model_path.exists():
                try:
                    mtime = model_path.stat().st_mtime
                    age_hours = (now_ts - mtime) / 3600
                    d = dt.datetime.fromtimestamp(mtime, tz=TZ_CENTRAL)
                    mod_time = d.strftime("%Y-%m-%d %I:%M %p")
                except OSError:
                    pass
            configs.append((name, cfg, mod_time, age_hours))

        self._model_table.setUpdatesEnabled(False)
        self._model_table.setRowCount(len(configs))
        for row, (name, cfg, mod_time, age_hours) in enumerate(configs):
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

            if cfg:
                vals = [name, status, mod_time, age_str,
                        str(cfg.get("hidden_dim", "?")),
                        str(cfg.get("num_layers", "?")),
                        str(cfg.get("seq_len", "?")),
                        str(cfg.get("bull_threshold", "?")),
                        str(cfg.get("indicator_preset", "N/A"))]
            else:
                vals = [name, status, "Not found", age_str,
                        "\u2014", "\u2014", "\u2014", "\u2014", "\u2014"]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 1:
                    item.setForeground(status_color)
                self._model_table.setItem(row, col, item)
        self._model_table.setUpdatesEnabled(True)

        # --- Pipeline Status (from pipeline_status.json) ---
        pinfo = _read_pipeline_status()
        phase = pinfo.get("phase", "idle")
        phase_label = pinfo.get("phase_label", "Idle")
        phase_idx = pinfo.get("phase_idx", -1)
        total_phases = pinfo.get("total_phases", 0)

        # Determine if pipeline is actively running (status file updated < 60s ago)
        is_running = False
        status_path = BASE_DIR / "pipeline_status.json"
        try:
            age = now_ts - status_path.stat().st_mtime
            is_running = age < 60
        except OSError:
            pass

        bots_running = pinfo.get("bots_running", False)

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
        else:
            status_color = T["green"].name()
            status_text = "TRAINING" if bots_running else "RUNNING"

        if total_phases > 0 and phase_idx >= 0 and phase != "trading":
            status_text += f" ({phase_idx + 1}/{total_phases})"
        if bots_running and phase != "trading":
            status_text += " + BOTS"

        self._pipeline_status.setText(
            f"Status: <span style='color:{status_color}'>{status_text}</span>")

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
                f"QProgressBar {{ color: #111111; }}"
                f"QProgressBar::chunk {{ background-color: {bar_color}; }}")
        elif cycle > 0:
            self._pipeline_trial.setText(f"Bot Cycle: {cycle}")
            self._pipeline_progress.setRange(0, 1)
            self._pipeline_progress.setValue(1)
            self._pipeline_progress.setStyleSheet(
                f"QProgressBar {{ color: #111111; }}"
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

        # Show final scores from completed phases
        bear_final = pinfo.get("bear_final_score")
        bull_final = pinfo.get("bull_final_score")
        stock_bear_final = pinfo.get("stock_bear_final_score")
        stock_bull_final = pinfo.get("stock_bull_final_score")
        scores_parts = []
        if bear_final is not None:
            scores_parts.append(f"C-Bear: {bear_final:.4f}")
        if bull_final is not None:
            scores_parts.append(f"C-Bull: {bull_final:.4f}")
        if stock_bear_final is not None:
            scores_parts.append(f"S-Bear: {stock_bear_final:.4f}")
        if stock_bull_final is not None:
            scores_parts.append(f"S-Bull: {stock_bull_final:.4f}")
        self._pipeline_scores.setText("  |  ".join(scores_parts) if scores_parts else "")

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

    def closeEvent(self, event):
        from PySide6.QtCore import QMetaObject

        # 1. Stop main-thread timers immediately
        self._model_timer.stop()
        self._clock_timer.stop()

        # 2. Ask worker threads to stop their timers (queued → runs before quit)
        QMetaObject.invokeMethod(self._fetcher, "stop_timers", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self._tailer, "stop_timer", Qt.QueuedConnection)

        # 3. Signal threads to stop, then ask event loops to exit
        self._fetcher_thread.requestInterruption()
        self._tailer_thread.requestInterruption()
        self._fetcher_thread.quit()
        self._tailer_thread.quit()

        # 4. Wait briefly for clean shutdown; os._exit fallback handles stragglers
        threads_stuck = False
        for name, thread in [("fetcher", self._fetcher_thread),
                             ("tailer", self._tailer_thread)]:
            if not thread.wait(2000):
                print(f"WARNING: {name} thread did not stop in time")
                threads_stuck = True

        # 5. Flush news cache so recent articles persist
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
def main():
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not api_secret:
        print("ERROR: ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env")
        sys.exit(1)

    api = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")

    try:
        acct = api.get_account()
        print(f"Connected to Alpaca. Equity: ${float(acct.equity):,.2f}")
    except Exception as e:
        print(f"ERROR: Cannot connect to Alpaca API: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
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
