"""Wave-8 #5: bar-keyed prediction cache semantics.

The cache must return a HIT only for the exact (subkey, bar-key) pair, invalidate
the instant a new bar arrives (single slot, never grows), isolate different
symbols / model objects / return modes, and refuse to cache on a bad timestamp.
predict_now wires it; a source guard confirms the wiring shape.
"""
import datetime
from pathlib import Path

import pytest

from prediction_cache import PredictionCache, bar_key, MISS

REPO = Path(__file__).resolve().parent.parent


def test_bar_key_from_datetime_int_and_bad():
    dt = datetime.datetime(2026, 6, 18, 14, 0, 0, tzinfo=datetime.timezone.utc)
    assert bar_key(dt) == int(dt.timestamp())
    assert bar_key(1_700_000_000) == 1_700_000_000
    assert bar_key(None) is None
    assert bar_key(object()) is None          # no timestamp(), not int-able


def test_miss_then_hit_for_same_bar():
    c = PredictionCache()
    sub, k = ('BTC/USD', 111, True), bar_key(1000)
    assert c.get(sub, k) is MISS
    c.put(sub, k, (0.42, {'x': 1}))
    assert c.get(sub, k) == (0.42, {'x': 1})


def test_new_bar_invalidates_single_slot():
    c = PredictionCache()
    sub = ('ETH/USD', 7, False)
    c.put(sub, bar_key(1000), 0.1)
    assert c.get(sub, bar_key(1000)) == 0.1
    # A newer bar key for the same subkey is a MISS and overwrites the slot.
    assert c.get(sub, bar_key(2000)) is MISS
    c.put(sub, bar_key(2000), 0.2)
    assert c.get(sub, bar_key(2000)) == 0.2
    assert c.get(sub, bar_key(1000)) is MISS   # old bar no longer cached
    assert len(c._store) == 1                  # never grows


def test_subkey_isolation_symbol_model_and_mode():
    c = PredictionCache()
    k = bar_key(1000)
    c.put(('BTC/USD', 1, True), k, 'a')
    # different symbol / different model id / different return_snapshot all miss
    assert c.get(('ETH/USD', 1, True), k) is MISS
    assert c.get(('BTC/USD', 2, True), k) is MISS      # model hot-reloaded -> new id
    assert c.get(('BTC/USD', 1, False), k) is MISS
    assert c.get(('BTC/USD', 1, True), k) == 'a'


def test_none_key_never_caches():
    c = PredictionCache()
    sub = ('X', 1, True)
    c.put(sub, None, 'nope')               # no-op
    assert c.get(sub, None) is MISS
    assert len(c._store) == 0


def test_counters_and_hit_rate_119_of_120():
    c = PredictionCache()
    sub, k = ('BTC/USD', 1, True), bar_key(1000)
    # First cycle of the hour: miss + compute + store.
    assert c.get(sub, k) is MISS
    c.put(sub, k, 0.5)
    # Next 119 cycles within the same bar: all hits.
    for _ in range(119):
        assert c.get(sub, k) == 0.5
    assert c.hits == 119 and c.misses == 1
    assert c.hit_rate() == pytest.approx(119 / 120)


def test_clear_resets_store_and_counters():
    c = PredictionCache()
    c.put(('X', 1, True), bar_key(1), 1)
    c.get(('X', 1, True), bar_key(1))
    c.clear()
    assert len(c._store) == 0 and c.hits == 0 and c.misses == 0


def test_predict_now_wires_cache_before_feature_compute():
    src = (REPO / "predict_now.py").read_text()
    assert "from prediction_cache import PredictionCache" in src
    assert "_PRED_CACHE = PredictionCache()" in src
    # 2026-07 review: the memo's whole value is skipping the pandas/numba
    # feature pass (~most of the cycle's CPU), so the check must sit BEFORE
    # compute_features — the original wiring checked after it, so an enabled
    # cache skipped only the (cheap) model calls. Order: forming-bar drop ->
    # cache check -> feature compute -> seq_len guard -> ... -> cache put.
    forming = src.index("drop_forming_bar(df)")
    check = src.index("_PRED_CACHE.get(_cache_subkey, _cache_key)")
    features = src.index("# --- Compute technical features ---")
    guard = src.index("Not enough data for sequence")
    store = src.index("_PRED_CACHE.put(_cache_subkey, _cache_key, predicted_return)")
    assert forming < check < features < guard < store


def test_predict_now_clears_cache_on_model_load():
    # a hot-reload must never serve the previous model's memo, and the
    # clear also defuses id()-reuse in the cache subkey
    src = (REPO / "predict_now.py").read_text()
    load = src.index("def load_model(")
    nxt = src.index("def load_models(")
    assert "_PRED_CACHE.clear()" in src[load:nxt]
