"""Headline novelty scoring — stop stale reprints from re-triggering signals.

Tetlock (2011) and the staleness literature: price response to FRESH news
is ~1.7x that of reprinted/recombined stories, and stale news that LOOKS
fresh induces overreaction that reverses. The wires republish the same
story across feeds for days; feeding those reprints to the LLM analyst
and sentiment stack re-counts one event many times.

Method: 3-word shingle sets per headline (crc32-hashed — Python's hash()
is salted per process and would break persistence), Jaccard similarity
against everything seen for that symbol in the trailing window:

    novelty = 1 - max Jaccard vs 7-day history

Pure stdlib; scoring is microseconds per headline on the Jetson, and the
store flushes to disk at most once per filter_novel batch — and only when
it actually changed (steady-state reprints rewrite nothing). An embedding
upgrade (MiniLM int8 ONNX) can slot behind the same interface later;
shingles already kill the dominant exact/near-duplicate failure mode.
"""

import json
import logging
import os
import re
import threading
import time
import zlib
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
_STORE_FILE = BASE_DIR / 'novelty_store.json'

WINDOW_DAYS = 7
MAX_PER_SYMBOL = 200
SHINGLE_W = 3
NOVELTY_MIN = 0.4       # below this = stale reprint
_WORD_RE = re.compile(r'[a-z0-9]+')

_store: dict | None = None
_dirty = False
# Combined-bot mode (run_bots.py) scores headlines from BOTH loop threads;
# the lock serializes the whole load -> mutate -> save cycle (a json.dump
# racing a dict insert raises mid-save and both threads share one tmp file).
_LOCK = threading.Lock()
_last_save_warn = 0.0   # rate-limits dead-disk warnings to one per hour


def _shingles(text: str) -> set[int]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < SHINGLE_W:
        return {zlib.crc32(' '.join(words).encode())} if words else set()
    return {zlib.crc32(' '.join(words[i:i + SHINGLE_W]).encode())
            for i in range(len(words) - SHINGLE_W + 1)}


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _load() -> dict:
    global _store
    if _store is None:
        try:
            with open(_STORE_FILE) as f:
                _store = json.load(f)
        except (OSError, json.JSONDecodeError):
            _store = {}
        if not isinstance(_store, dict):
            # Corrupt-but-parseable file: reset and rebuild, same outcome
            # as the JSONDecodeError path (store.get would raise forever).
            logger.warning('novelty store was not a dict — resetting')
            _store = {}
        _sweep(_store, time.time())
    return _store


def _save():
    global _dirty, _last_save_warn
    if not _dirty:
        return
    tmp = str(_STORE_FILE) + '.tmp'
    try:
        _sweep(_store, time.time())
        with open(tmp, 'w') as f:
            json.dump(_store, f)
        os.replace(tmp, _STORE_FILE)
        _dirty = False
    except OSError as e:
        now = time.time()
        if now - _last_save_warn > 3600:
            _last_save_warn = now
            logger.warning('novelty store save failed (history will not '
                           'survive a restart): %s', e)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _prune(entries: list, now: float) -> list:
    cutoff = now - WINDOW_DAYS * 86400
    kept = [e for e in entries if e[0] >= cutoff]
    return kept[-MAX_PER_SYMBOL:]


def _sweep(store: dict, now: float) -> None:
    """Prune ALL symbols, not just the one being scored — names that rotate
    out of the traded set would otherwise pin their history forever."""
    for sym in list(store):
        kept = _prune(store[sym], now)
        if kept:
            store[sym] = kept
        else:
            del store[sym]


def headline_novelty(symbol: str, headline: str,
                     remember: bool = True, flush: bool = True) -> float:
    """Novelty in [0, 1] vs this symbol's trailing 7 days of headlines.

    remember=True adds the headline to the store (call once per headline
    per cycle path; scoring what you just stored would return 0).
    flush=False defers the disk write to the caller — filter_novel flushes
    once per batch instead of once per headline.
    """
    global _dirty
    sh = _shingles(headline)
    if not sh:
        return 0.0
    now = time.time()
    with _LOCK:
        store = _load()
        prior = store.get(symbol, [])
        entries = _prune(prior, now)
        changed = len(entries) != len(prior)
        max_sim = 0.0
        for _, stored in entries:
            sim = _jaccard(sh, set(stored))
            if sim > max_sim:
                max_sim = sim
                if max_sim > 0.999:
                    break
        novelty = 1.0 - max_sim
        if remember and novelty > 0.001:  # exact repeats need no new entry
            entries.append([now, sorted(sh)])
            entries = entries[-MAX_PER_SYMBOL:]
            changed = True
        store[symbol] = entries
        if changed:  # steady-state repeats must not rewrite the flash
            _dirty = True
        if flush:
            _save()
    return novelty


def filter_novel(symbol: str, headlines: list[str],
                 min_novelty: float = NOVELTY_MIN) -> list[str]:
    """Drop stale reprints; always keep at least the single most novel
    headline so downstream consumers retain context. Order preserved.
    Fail open: any error returns the input unchanged."""
    if not headlines:
        return headlines
    try:
        scored = [(h, headline_novelty(symbol, h, flush=False))
                  for h in headlines]
        with _LOCK:
            _save()  # one flush per batch (no-op when nothing changed)
        fresh = [h for h, n in scored if n >= min_novelty]
        if fresh:
            return fresh
        return [max(scored, key=lambda t: t[1])[0]]
    except Exception:
        return headlines
