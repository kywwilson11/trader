"""R2C-01 (H3) — serving booster-cache integrity tests.

The pure kernel (serving_cache.cache_get) is exercised directly with stub
booster files, injected clocks, and counting loaders — no torch/lightgbm
needed. predict_now.py itself cannot be imported on the dev Mac, so its
wiring is pinned by source-structure assertions, alongside the untouched
challenger-slot pattern in shadow.py and the pop contract in base_loop.py.
"""

import os
from pathlib import Path

import pytest

import serving_cache
from serving_cache import cache_get, stat_key

REPO = Path(__file__).resolve().parents[1]


def _set_mtime(path, ns):
    os.utime(path, ns=(ns, ns))


class _Loader:
    """Counting loader returning self.result (or raising self.exc)."""

    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.fixture
def state():
    """Fresh (cache, keys, failed_at) side-dict triple per test."""
    return {}, {}, {}


class TestStatKey:
    def test_missing_file_is_none(self, tmp_path):
        assert stat_key((str(tmp_path / 'nope.txt'),)) is None

    def test_any_missing_file_is_none(self, tmp_path):
        f = tmp_path / 'lgb_model.txt'
        f.write_text('x')
        assert stat_key((str(f), str(tmp_path / 'nope.json'))) is None

    def test_present_files_tuple(self, tmp_path):
        a = tmp_path / 'a.txt'
        b = tmp_path / 'b.json'
        a.write_text('x')
        b.write_text('y')
        key = stat_key((str(a), str(b)))
        assert isinstance(key, tuple) and len(key) == 2

    def test_key_changes_on_rewrite(self, tmp_path):
        f = tmp_path / 'lgb_model.txt'
        f.write_text('gen1')
        _set_mtime(f, 1_000)
        k1 = stat_key((str(f),))
        _set_mtime(f, 2_000)
        assert stat_key((str(f),)) != k1


class TestCacheGetReload:
    def test_loads_once_and_memoizes(self, tmp_path, state):
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('gen1')
        loader = _Loader(result='booster-1')
        for _ in range(3):
            got = cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                            loader, now=100.0)
            assert got == 'booster-1'
        assert loader.calls == 1
        assert cache[''] == 'booster-1'  # legacy value shape preserved

    def test_mtime_change_triggers_reload(self, tmp_path, state):
        """The H3 core: a retrain landing a fresh booster file minutes after
        the manifest must be picked up on the next cycle, not next week."""
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('gen1')
        _set_mtime(f, 1_000)
        loader = _Loader(result='old-booster')
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=100.0) == 'old-booster'
        # retrain writes the new booster (atomic replace bumps mtime)
        f.write_text('gen2')
        _set_mtime(f, 2_000)
        loader.result = 'new-booster'
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=130.0) == 'new-booster'
        assert loader.calls == 2
        assert cache[''] == 'new-booster'

    def test_default_clock_path(self, tmp_path, state):
        """now=None (production path) uses time.time() — the injectable
        clock must not be load-bearing for the happy path."""
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('gen1')
        loader = _Loader(result='booster-1')
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader) == 'booster-1'
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader) == 'booster-1'
        assert loader.calls == 1

    def test_champion_pop_forces_reload_without_swap(self, tmp_path, state):
        """base_loop._hot_reload_check pops the champion entry (leaving the
        side dicts): the next call must reload even with an unchanged file,
        and on_swap must NOT fire (no cached object was replaced — the LSTM
        reload already cleared the memo via load_model)."""
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('gen1')
        _set_mtime(f, 1_000)
        fired = []
        loader = _Loader(result='old-booster')
        cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader,
                  on_swap=lambda: fired.append(1), now=1.0)
        cache.pop('', None)  # base_loop hot-reload eviction
        loader.result = 'reloaded-booster'
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader,
                         on_swap=lambda: fired.append(1), now=2.0) \
            == 'reloaded-booster'
        assert loader.calls == 2
        assert fired == []

    def test_replace_during_load_converges(self, tmp_path, state):
        """TOCTOU: the retrain replaces the file BETWEEN stat and load. The
        recorded key is the pre-replace one, so the next call reloads once
        more and then stabilizes — stale is never served indefinitely."""
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('gen1')
        _set_mtime(f, 1_000)
        calls = []

        def racing_loader():
            calls.append(1)
            if len(calls) == 1:
                # retrain lands gen2 mid-load (after cache_get's stat, at
                # open time) — the loader reads gen2's bytes
                f.write_text('gen2')
                _set_mtime(f, 2_000)
            return 'booster-gen2'

        got1 = cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         racing_loader, now=1.0)
        assert got1 == 'booster-gen2'
        # key mismatch (recorded pre-replace) -> exactly one more reload
        got2 = cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         racing_loader, now=2.0)
        assert got2 == 'booster-gen2'
        assert len(calls) == 2
        # now stable: no further loader calls
        cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                  racing_loader, now=3.0)
        assert len(calls) == 2

    def test_deleted_file_falls_back_to_none(self, tmp_path, state):
        """A cached booster whose backing file disappears must not keep
        being served: reload runs, the loader reports missing (None, as
        load_lgb_model does), and the blend falls back to LSTM-only."""
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('gen1')
        loader = _Loader(result='booster-1')
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=1.0) == 'booster-1'
        f.unlink()
        loader.result = None
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=2.0) is None
        assert cache[''] is None

    def test_q10_pair_keyed_on_both_files(self, tmp_path, state):
        cache, keys, failed = state
        b = tmp_path / 'lgb_q10.txt'
        m = tmp_path / 'lgb_q10_meta.json'
        b.write_text('q10-gen1')
        m.write_text('{"floor": -1.0}')
        _set_mtime(b, 1_000)
        _set_mtime(m, 1_000)
        paths = (str(b), str(m))
        loader = _Loader(result=('q10-booster-1', -1.0))
        assert cache_get(cache, keys, failed, 'q10', '', paths,
                         loader, now=10.0) == ('q10-booster-1', -1.0)
        # only the META file changes -> still a reload (new floor generation)
        _set_mtime(m, 2_000)
        loader.result = ('q10-booster-1', -2.0)
        assert cache_get(cache, keys, failed, 'q10', '', paths,
                         loader, now=20.0) == ('q10-booster-1', -2.0)
        assert loader.calls == 2


class TestCachedNoneEviction:
    def test_none_evicted_once_files_appear(self, tmp_path, state):
        """Manifest-first ordering: first tick finds no booster file; the
        cached None must be evicted the moment the file lands."""
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        loader = _Loader(result=None)  # load_lgb_model: missing file -> None
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=100.0) is None
        assert cache[''] is None  # shadow's `.get(cp) is None` still sees it
        # cached: no loader storm while the file is still missing
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=101.0) is None
        assert loader.calls == 1
        # train_lgb_ensemble lands the file minutes later
        f.write_text('gen1')
        loader.result = 'fresh-booster'
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=160.0) == 'fresh-booster'
        assert loader.calls == 2

    def test_raising_loader_cached_as_none_not_propagated(self, tmp_path,
                                                          state):
        cache, keys, failed = state
        f = tmp_path / 'lgb_q10.txt'
        f.write_text('half')
        loader = _Loader(exc=RuntimeError('bad model file'))
        assert cache_get(cache, keys, failed, 'q10', '', (str(f),),
                         loader, now=100.0) is None
        assert cache[''] is None

    def test_failure_retries_after_backoff(self, tmp_path, state):
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('x')
        _set_mtime(f, 1_000)
        loader = _Loader(exc=RuntimeError('transient'))
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=100.0) is None
        # inside the backoff window, unchanged file: no retry
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=100.0 + 10.0) is None
        assert loader.calls == 1
        # backoff expired: retry, and this time it succeeds
        loader.exc = None
        loader.result = 'recovered'
        t = 100.0 + serving_cache.RETRY_SEC + 1.0
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=t) == 'recovered'
        assert loader.calls == 2

    def test_failure_retries_immediately_on_file_change(self, tmp_path,
                                                        state):
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('half-written')
        _set_mtime(f, 1_000)
        loader = _Loader(exc=RuntimeError('truncated'))
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=100.0) is None
        # repaired file inside the backoff window: key change wins
        f.write_text('repaired')
        _set_mtime(f, 2_000)
        loader.exc = None
        loader.result = 'good-booster'
        assert cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                         loader, now=105.0) == 'good-booster'
        assert loader.calls == 2

    def test_repeated_failure_keeps_backing_off(self, tmp_path, state):
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('x')
        _set_mtime(f, 1_000)
        loader = _Loader(exc=RuntimeError('still broken'))
        t = 100.0
        cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader, now=t)
        t += serving_cache.RETRY_SEC + 1.0
        cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader, now=t)
        assert loader.calls == 2
        # the second failure re-armed the backoff from ITS timestamp
        cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader,
                  now=t + 10.0)
        assert loader.calls == 2


class TestOnSwap:
    def test_not_fired_on_first_load(self, tmp_path, state):
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('x')
        fired = []
        cache_get(cache, keys, failed, 'lgb', '', (str(f),),
                  _Loader(result='b1'), on_swap=lambda: fired.append(1),
                  now=1.0)
        assert fired == []

    def test_fired_on_generation_swap(self, tmp_path, state):
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('x')
        _set_mtime(f, 1_000)
        fired = []
        loader = _Loader(result='b1')
        cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader,
                  on_swap=lambda: fired.append(1), now=1.0)
        _set_mtime(f, 2_000)
        loader.result = 'b2'
        cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader,
                  on_swap=lambda: fired.append(1), now=2.0)
        assert fired == [1]

    def test_fired_when_cached_none_becomes_booster(self, tmp_path, state):
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        fired = []
        loader = _Loader(result=None)
        cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader,
                  on_swap=lambda: fired.append(1), now=1.0)
        f.write_text('landed')
        loader.result = 'b1'
        cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader,
                  on_swap=lambda: fired.append(1), now=2.0)
        assert fired == [1]

    def test_not_fired_on_repeated_failure(self, tmp_path, state):
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('x')
        _set_mtime(f, 1_000)
        fired = []
        loader = _Loader(exc=RuntimeError('down'))
        t = 1.0
        cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader,
                  on_swap=lambda: fired.append(1), now=t)
        t += serving_cache.RETRY_SEC + 1.0
        cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader,
                  on_swap=lambda: fired.append(1), now=t)
        assert fired == []

    def test_on_swap_exception_never_propagates(self, tmp_path, state):
        cache, keys, failed = state
        f = tmp_path / 'lgb_model.txt'
        f.write_text('x')
        _set_mtime(f, 1_000)
        loader = _Loader(result='b1')
        cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader, now=1.0)
        _set_mtime(f, 2_000)
        loader.result = 'b2'

        def _boom():
            raise RuntimeError('memo clear failed')

        got = cache_get(cache, keys, failed, 'lgb', '', (str(f),), loader,
                        on_swap=_boom, now=2.0)
        assert got == 'b2'


class TestChallengerIsolation:
    """Champion and challenger prefixes share the dicts but never interact —
    the pre-existing per-prefix contract base_loop/shadow rely on."""

    def test_prefixes_independent(self, tmp_path, state):
        cache, keys, failed = state
        champ = tmp_path / 'lgb_model.txt'
        chall = tmp_path / 'challenger_lgb_model.txt'
        champ.write_text('c')
        chall.write_text('h')
        champ_loader = _Loader(result='champ-booster')
        chall_loader = _Loader(result='chall-booster')
        assert cache_get(cache, keys, failed, 'lgb', '', (str(champ),),
                         champ_loader, now=1.0) == 'champ-booster'
        assert cache_get(cache, keys, failed, 'lgb', 'challenger',
                         (str(chall),), chall_loader, now=1.0) \
            == 'chall-booster'
        assert cache[''] == 'champ-booster'
        assert cache['challenger'] == 'chall-booster'

    def test_challenger_pop_leaves_champion_cached(self, tmp_path, state):
        cache, keys, failed = state
        champ = tmp_path / 'lgb_model.txt'
        chall = tmp_path / 'challenger_lgb_model.txt'
        champ.write_text('c')
        chall.write_text('h')
        champ_loader = _Loader(result='champ-booster')
        chall_loader = _Loader(result='chall-booster-1')
        cache_get(cache, keys, failed, 'lgb', '', (str(champ),),
                  champ_loader, now=1.0)
        cache_get(cache, keys, failed, 'lgb', 'challenger', (str(chall),),
                  chall_loader, now=1.0)
        # shadow.py's manifest-keyed eviction: pop the challenger entry only
        cache.pop('challenger', None)
        chall_loader.result = 'chall-booster-2'
        assert cache_get(cache, keys, failed, 'lgb', 'challenger',
                         (str(chall),), chall_loader, now=2.0) \
            == 'chall-booster-2'
        # champion entry untouched: served from cache, no reload
        assert cache_get(cache, keys, failed, 'lgb', '', (str(champ),),
                         champ_loader, now=2.0) == 'champ-booster'
        assert champ_loader.calls == 1

    def test_shadow_none_eviction_pattern_still_works(self, tmp_path, state):
        """shadow.py:227-233 checks `.get(cp) is None` then pops — the new
        cache must keep that pattern functional (values stay legacy-shaped),
        even though the mtime key now makes it redundant."""
        cache, keys, failed = state
        chall = tmp_path / 'challenger_lgb_model.txt'
        loader = _Loader(result=None)
        cache_get(cache, keys, failed, 'lgb', 'challenger', (str(chall),),
                  loader, now=1.0)
        assert cache.get('challenger') is None  # shadow's check sees it
        chall.write_text('landed')
        cache.pop('challenger', None)  # shadow's eviction
        loader.result = 'fresh'
        assert cache_get(cache, keys, failed, 'lgb', 'challenger',
                         (str(chall),), loader, now=2.0) == 'fresh'


class TestPredictNowWiring:
    """Source-structure pins: predict_now.py imports torch, so its wiring is
    asserted from source text (the dev-Mac standard for heavy modules)."""

    SRC = (REPO / 'predict_now.py').read_text()

    def test_kernel_imported(self):
        assert ('from serving_cache import cache_get as _booster_cache_get'
                in self.SRC)

    def test_presence_keyed_caching_gone(self):
        assert 'if pfx not in _lgb_models' not in self.SRC
        assert 'if pfx not in _q10_models' not in self.SRC

    def test_both_legs_use_mtime_keyed_cache(self):
        assert self.SRC.count('_booster_cache_get(') >= 2
        assert ('_lgb_models, _booster_keys, _booster_failed_at, '
                "'lgb', pfx") in self.SRC
        assert ('_q10_models, _booster_keys, _booster_failed_at, '
                "'q10', pfx") in self.SRC

    def test_q10_keyed_on_booster_and_meta(self):
        assert "lgb_q10.txt', f'{_p}lgb_q10_meta.json'" in self.SRC

    def test_lgb_stat_path_matches_model_lgb_loader(self):
        # load_lgb_model opens _MODEL_DIR-joined paths; the stat key must
        # watch the same file
        assert "os.path.join(_MODEL_DIR, f'{_p}lgb_model.txt')" in self.SRC

    def test_swap_clears_prediction_memo(self):
        assert self.SRC.count('on_swap=_PRED_CACHE.clear') == 2

    def test_retry_backoff_matches_base_loop(self):
        # RETRY_SEC deliberately mirrors base_loop's failed-hot-reload
        # backoff (base_loop.py: `time.time() - failed_at < 300`); keep the
        # two in sync if either changes.
        assert serving_cache.RETRY_SEC == 300.0
        src = (REPO / 'base_loop.py').read_text()
        assert 'failed_at < 300' in src

    def test_legacy_cache_dicts_survive(self):
        # base_loop._hot_reload_check and shadow.maybe_log_shadow pop these
        # by name; the dicts (and their value shapes) must remain
        assert '_lgb_models: dict[str, object | None] = {}' in self.SRC
        assert ('_q10_models: dict[str, tuple[object, float] | None] = {}'
                in self.SRC)


class TestChallengerSlotUntouched:
    """R2C-01 ports the shadow pattern to the champion; the challenger-slot
    code itself must be byte-untouched (packet scope: predict_now.py only)."""

    def test_shadow_eviction_block_intact(self):
        src = (REPO / 'shadow.py').read_text()
        assert "predict_now._lgb_models.pop(cp, None)" in src
        assert "predict_now._q10_models.pop(cp, None)" in src
        assert "predict_now._lgb_models.get(cp) is None" in src
        assert "predict_now._q10_models.get(cp) is None" in src

    def test_base_loop_pop_contract_intact(self):
        src = (REPO / 'base_loop.py').read_text()
        assert 'predict_now._lgb_models.pop(self.MODEL_PREFIX, None)' in src
        assert 'predict_now._q10_models.pop(self.MODEL_PREFIX, None)' in src
