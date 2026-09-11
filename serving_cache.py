"""mtime-keyed lazy-load kernel for predict_now's per-prefix booster caches.

R2C-01 (H3 serving-integrity repair): legacy retrains write the model
manifest BEFORE train_lgb_ensemble finishes, so base_loop's hot-reload pops
the LGB/q10 caches minutes before the fresh booster files land on disk. The
old presence-keyed caches (``if pfx not in cache``) then loaded LAST WEEK's
booster files — trained against the previous scaler — and pinned them until
the next retrain; a transient load failure was likewise cached as None
forever. shadow.py patched exactly this race for the challenger slot only
(manifest-mtime keyed pops + cached-None eviction); this kernel restores the
same new-stack-together semantics for every slot, champion included.

stdlib-only (os/time) so the reload semantics are unit-testable on the dev
Mac, where predict_now itself cannot be imported (torch).

Contract with predict_now:
- cache VALUES keep their legacy shapes (booster-or-None; (booster, floor)
  -or-None) so base_loop/shadow ``pop(prefix)`` eviction and shadow's
  cached-None checks keep working unchanged;
- the mtime keys and failure timestamps live in side dicts owned by the
  caller, keyed ``(leg_name, prefix)``.
"""

import os
import time

# Retry a failed load (files present but loader raised) after this many
# seconds; matches base_loop's failed-hot-reload backoff. A key change —
# files replaced or newly appeared — retries immediately regardless.
RETRY_SEC = 300.0

_MISSING = object()


def stat_key(paths):
    """mtime_ns tuple for the files backing one cache entry, or None if any
    is missing. The LGB mean save is atomic (tmp + os.replace) so its
    mtime_ns is a clean generation key; the legacy q10 save writes in place
    (hypersearch_v2.py:1171-1172), so a stat/load can catch a mid-write file
    — the recorded key then differs from the finished file's, and the next
    call reloads (stat runs BEFORE load, so a mismatch always resolves
    toward a fresh reload, never toward serving stale forever)."""
    try:
        return tuple(os.stat(p).st_mtime_ns for p in paths)
    except OSError:
        return None


def cache_get(cache, keys, failed_at, name, pfx, paths, loader,
              retry_sec=RETRY_SEC, on_swap=None, now=None):
    """Return ``cache[pfx]``, (re)loading via ``loader()`` when stale.

    The cached object is served only while the backing files' mtime key
    matches the key recorded for ``(name, pfx)``. Otherwise ``loader()`` runs;
    an exception caches None. A cached None is retried after ``retry_sec``
    while the files are unchanged, and immediately once they change or
    appear — never cached permanently. ``on_swap()`` fires when a (re)load
    replaces a DIFFERENT cached object, so the caller can drop bar-keyed
    prediction memos computed under the old booster generation (mirrors
    load_model's _PRED_CACHE.clear for the LSTM); it never raises through.

    ``now`` (epoch seconds) is injectable for tests; defaults to
    ``time.time()``.
    """
    if now is None:
        now = time.time()
    key = stat_key(paths)
    ck = (name, pfx)
    if pfx in cache and keys.get(ck, _MISSING) == key:
        obj = cache[pfx]
        if obj is not None:
            return obj
        # Cached failure under an unchanged key: files still missing (key
        # None) -> wait for them to appear; files present -> backoff retry.
        if key is None or now - failed_at.get(ck, 0.0) < retry_sec:
            return None
    try:
        obj = loader()
    except Exception:
        obj = None
    prev = cache.get(pfx, _MISSING)
    if on_swap is not None and prev is not _MISSING and prev is not obj:
        try:
            on_swap()
        except Exception:
            pass
    cache[pfx] = obj
    keys[ck] = key
    if obj is None:
        failed_at[ck] = now
    else:
        failed_at.pop(ck, None)
    return obj
