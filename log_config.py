"""Centralized logging configuration with rotating file handler.

Usage in any module:
    from log_config import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_FILE = _LOG_DIR / "trader.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5
_FMT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _setup():
    global _configured
    if _configured:
        return
    _configured = True

    _LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler (INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(ch)

    # Rotating file handler (DEBUG)
    fh = RotatingFileHandler(str(_LOG_FILE), maxBytes=_MAX_BYTES,
                             backupCount=_BACKUP_COUNT)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(fh)

    # Suppress noisy third-party loggers
    for name in ('urllib3', 'httpx', 'httpcore', 'websockets', 'yfinance'):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    _setup()
    return logging.getLogger(name)
