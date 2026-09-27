"""Centralized logging configuration with rotating file handler.

Usage in any module:
    from log_config import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
import logging.handlers as _stdlib_handlers
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX; lockless fallback
    fcntl = None

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_FILE = _LOG_DIR / "trader.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5
_FMT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# TRADER_LOG_DIR override (2026-09-27, ENGINE r6 W18). Unset or empty (the
# production value: neither run_pipeline, run_bots nor the systemd unit sets
# it) -> <repo>/logs/trader.log exactly as before. Set -> trader.log (and its
# .1-.5 backups and the trader.log.lock flock sibling) live in that directory
# instead; a relative value is resolved against the REPO ROOT, never the CWD
# (same anchoring as every other runtime path). Exists so test processes stop
# writing fake events into the production log (tests/conftest.py sets it).
_LOG_DIR_ENV = 'TRADER_LOG_DIR'
_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_LOG_DIR = _LOG_DIR     # identity sentinels: a module-level
_DEFAULT_LOG_FILE = _LOG_FILE   # reassignment (tests) is detected with `is`

_configured = False
_setup_lock = threading.Lock()
_file_handler = None  # the one shared trader.log handler (set by _setup)


class SharedRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that tolerates OTHER processes writing and rotating
    the same file (2026-09-26 Jetson campaign, G3-5).

    Every process (pipeline parent, bots, backfill worker, every phase child)
    attaches its own handler to the one logs/trader.log. The stdlib handler
    assumes a single writer: after process A renames trader.log ->
    trader.log.1, process B keeps appending to the renamed inode and later
    rotates it AGAIN (undersized backups, history far below the
    (backupCount+1)*maxBytes budget), and concurrent doRollover calls race on
    the renames ("--- Logging error --- FileNotFoundError", record dropped).
    Measured on this box: trader.log.1 = 2.59 MB rotated 58 min after a
    10.49 MB .2; a 4-writer repro gave 31 logging errors and undersized
    backups, this class 0 errors and full-size backups.

    Two stdlib-only fixes, same paths and format:
      * shouldRollover: if the path's inode is not our stream's inode,
        another process rotated -> reopen trader.log before sizing.
      * doRollover: serialized across processes by flock on trader.log.lock,
        re-checking the inode inside the lock (a rotation that happened while
        we waited means: just reopen, do not rotate again).
    Cost: one stat + fstat per record.
    """

    def _foreign_rotation(self):
        try:
            return (self.stream is not None
                    and os.stat(self.baseFilename).st_ino
                    != os.fstat(self.stream.fileno()).st_ino)
        except OSError:
            return True  # path gone (mid-rotation) or stream unusable

    def _reopen(self):
        if self.stream is not None:
            try:
                self.stream.close()
            except OSError:
                pass
        self.stream = self._open()

    def shouldRollover(self, record):
        if self.stream is not None and self._foreign_rotation():
            self._reopen()
        return super().shouldRollover(record)

    def doRollover(self):
        if fcntl is None:  # pragma: no cover - non-POSIX
            return super().doRollover()
        with open(self.baseFilename + '.lock', 'a') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)  # released on close
            if self.stream is not None and self._foreign_rotation():
                self._reopen()  # someone rotated while we waited
                return
            super().doRollover()


def _log_paths():
    """(directory, file) for trader.log -- the ONE source of the log location.

    _setup builds the handler from it; the flock sibling (baseFilename +
    '.lock' in doRollover) and get_file_logger (reuses _file_handler) derive
    from that handler, so all three always agree.

    Read at CALL time (first _setup), so a variable set before the first
    get_logger call is honoured even if log_config was already imported.
    Setting it AFTER the handler exists does nothing: the process keeps its
    file for life (no re-pointing, by design).

    Precedence: a module-level reassignment of _LOG_DIR / _LOG_FILE (tests
    monkeypatch them) > TRADER_LOG_DIR > <repo>/logs. An override directory
    that cannot be created or is not writable falls back to the default with
    ONE stderr line -- it never raises.
    """
    if _LOG_DIR is not _DEFAULT_LOG_DIR or _LOG_FILE is not _DEFAULT_LOG_FILE:
        return _LOG_DIR, _LOG_FILE
    raw = os.environ.get(_LOG_DIR_ENV, '')
    if not raw:
        return _LOG_DIR, _LOG_FILE
    d = Path(raw)
    if not d.is_absolute():
        d = _REPO_ROOT / d
    try:
        os.makedirs(d, exist_ok=True)
        if not os.access(d, os.W_OK | os.X_OK):
            raise PermissionError(f"not writable: {d}")
    except OSError as e:
        print(f"log_config: {_LOG_DIR_ENV}={raw!r} unusable ({e}); "
              f"logging to {_LOG_FILE}", file=sys.stderr)
        return _LOG_DIR, _LOG_FILE
    return d, d / "trader.log"


def _setup():
    global _configured, _file_handler
    if _configured:
        return
    with _setup_lock:
        if _configured:
            return

        log_dir, log_file = _log_paths()
        log_dir.mkdir(exist_ok=True)

        # Build BOTH handlers fully before adding either to the root logger,
        # and mark configured only after success: a partial failure (logs/
        # unwritable, disk full) then fails loud on the NEXT get_logger call
        # instead of silently leaving the process without handlers for life.

        # Console handler (INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

        # Rotating file handler (DEBUG). The multi-process-safe subclass is
        # used unless a test has swapped the module-level RotatingFileHandler
        # (tests/test_review_b20.py injects a failing one) — honor the swap.
        fh_cls = (SharedRotatingFileHandler
                  if RotatingFileHandler is _stdlib_handlers.RotatingFileHandler
                  else RotatingFileHandler)
        fh = fh_cls(str(log_file), maxBytes=_MAX_BYTES,
                    backupCount=_BACKUP_COUNT, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(ch)
        root.addHandler(fh)
        _file_handler = fh

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


def get_file_logger(name: str) -> logging.Logger:
    """A logger that writes to trader.log ONLY (not the console handler).

    For callers that already print the same line to stdout themselves
    (run_pipeline._announce) — propagating to root would add a second copy
    on stderr, which systemd and the GUI launcher merge into the same sink.
    """
    _setup()
    lg = logging.getLogger(name)
    lg.propagate = False
    fh = _file_handler
    if fh is not None and fh not in lg.handlers:
        lg.addHandler(fh)
    lg.setLevel(logging.DEBUG)
    return lg
