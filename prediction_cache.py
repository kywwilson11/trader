"""Bar-keyed prediction memo (wave-8 #5).

Hourly bars + a 30s loop means ~119 of every 120 inference cycles re-run a
bit-identical feature-compute + LSTM + two LightGBM boosters over ~66 symbols.
The closed bars that drive the prediction only change when a new bar closes, so
memoizing the result on the latest CLOSED-bar timestamp (plus a model token, so a
hot-reload invalidates) collapses that wasted recompute — the single largest
source of idle CPU on the 8GB Jetson.

Pure and dependency-free so the key / hit / miss / invalidation logic is
unit-testable without torch or Alpaca; predict_now wires it behind a flag.
"""
import threading

_MISS = object()
MISS = _MISS  # public sentinel: distinguishes "no cached value" from a cached None


def bar_key(last_bar_ts):
    """Hashable key from the latest CLOSED bar's timestamp, or None if unusable.

    Accepts a pandas/py datetime (anything with .timestamp()) or an epoch number.
    None / unparseable -> None, which the cache treats as "do not cache" so a bad
    timestamp can never serve a stale prediction.
    """
    if last_bar_ts is None:
        return None
    try:
        if hasattr(last_bar_ts, 'timestamp'):
            return int(last_bar_ts.timestamp())
        return int(last_bar_ts)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


class PredictionCache:
    """Per-key single-slot memo — only the latest bar's result is retained.

    A slot is invalidated automatically the moment a new bar key arrives, so the
    cache never grows and never serves a prior bar's prediction.
    """

    def __init__(self):
        self._store = {}          # subkey -> (bar_key, value)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, subkey, key):
        """Return the cached value for (subkey, key), or MISS."""
        if key is None:
            self.misses += 1
            return _MISS
        with self._lock:
            entry = self._store.get(subkey)
            if entry is not None and entry[0] == key:
                self.hits += 1
                return entry[1]
            self.misses += 1
            return _MISS

    def put(self, subkey, key, value):
        if key is None:
            return
        with self._lock:
            self._store[subkey] = (key, value)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def clear(self):
        with self._lock:
            self._store.clear()
        self.hits = self.misses = 0
