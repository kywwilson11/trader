"""Centralized logging configuration with rotating file handler.

Usage in any module:
    from log_config import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_FILE = _LOG_DIR / "trader.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5
_FMT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured = False
_setup_lock = threading.Lock()


def _setup():
    global _configured
    if _configured:
        return
    with _setup_lock:
        if _configured:
            return

        _LOG_DIR.mkdir(exist_ok=True)

        # Build BOTH handlers fully before adding either to the root logger,
        # and mark configured only after success: a partial failure (logs/
        # unwritable, disk full) then fails loud on the NEXT get_logger call
        # instead of silently leaving the process without handlers for life.

        # Console handler (INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

        # Rotating file handler (DEBUG)
        fh = RotatingFileHandler(str(_LOG_FILE), maxBytes=_MAX_BYTES,
                                 backupCount=_BACKUP_COUNT, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(ch)
        root.addHandler(fh)

        # Suppress noisy third-party loggers. 'numba' matters: every kernel is
        # @njit(cache=True), so each deploy invalidates the cache and numba's
        # byteflow/SSA DEBUG dumps would churn the rotation budget right when
        # post-deploy forensics need the recent history.
        for name in ('urllib3', 'httpx', 'httpcore', 'websockets', 'yfinance',
                     'numba', 'charset_normalizer'):
            logging.getLogger(name).setLevel(logging.WARNING)

        _configured = True


def get_logger(name: str) -> logging.Logger:
    _setup()
    return logging.getLogger(name)
