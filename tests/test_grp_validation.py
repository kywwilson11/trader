"""Cross-module tripwires for the validation/shadow/panel group.

Two invariants no single module can see break:

1. ARTIFACT-SUFFIX THREE-WAY SYNC — shadow.promote_challenger (copy +
   .prev + meta-delete), backtest.restore_previous_model (--gate
   rollback), and scripts/hypersearch_v2.save_model_atomically (.prev at
   save) each hold their own copy of the model-artifact suffix list.
   Drift means a promotion or rollback ships/restores a stale mix.
2. PANEL PRODUCER/CONSUMER PARITY — panel_ranks.compute_live_panel_ranks
   emits keys that predict_now injects via c.startswith('CS_');
   MS_Interact is the ONE known-uninjected key (ledger P1, model-facing
   fix deferred). Any NEW non-CS_-prefixed producer key would silently
   neutral-fill live.

Pure import + source inspection — runs on the dev Mac and the Jetson.
"""
import re
from pathlib import Path

import pytest

import backtest
import panel_ranks
import shadow

REPO = Path(__file__).resolve().parent.parent

CORE4 = ['model_v2.pth', 'config_v2.pkl', 'scaler_v2.pkl',
         'feature_cols_v2.pkl']
MANIFEST = 'model_v2.manifest.json'


class TestArtifactSuffixThreeWaySync:
    def test_core_four_identical_and_ordered(self):
        # shadow.promote_challenger treats _ARTIFACT_SUFFIXES[:4] as the
        # mandatory core; backtest.restore_previous_model requires
        # prevs[:4] — both depend on the first four being EXACTLY the
        # LSTM core, in this order.
        assert shadow._ARTIFACT_SUFFIXES[:4] == CORE4
        assert backtest.ARTIFACT_SUFFIXES[:4] == CORE4

    def test_backtest_manifest_at_index_4(self):
        # restore_previous_model's i >= 4 optional-leg logic (restore
        # .prev / delete never-gated orphans) starts at the manifest slot.
        assert backtest.ARTIFACT_SUFFIXES[4] == MANIFEST

    def test_shadow_and_backtest_cover_the_same_artifact_set(self):
        # Everything shadow promotion touches (stack + manifest + the
        # meta trio it deletes) must be exactly what a --gate rollback
        # restores, or a rollback after a promotion leaves a stale mix.
        shadow_set = (set(shadow._ARTIFACT_SUFFIXES) | {MANIFEST}
                      | set(shadow._STALE_META_SUFFIXES))
        assert shadow_set == set(backtest.ARTIFACT_SUFFIXES), (
            'shadow.py / backtest.py artifact suffix sets drifted — '
            f'shadow-only={shadow_set - set(backtest.ARTIFACT_SUFFIXES)}, '
            f'backtest-only={set(backtest.ARTIFACT_SUFFIXES) - shadow_set}')

    def test_hypersearch_prev_list_matches_shadow_stack(self):
        # save_model_atomically .prevs the outgoing artifacts so --gate
        # can roll back; the set it touches must equal shadow's stack +
        # manifest. Source-inspected: hypersearch imports torch, so it is
        # not importable on the dev Mac.
        src = (REPO / 'scripts' / 'hypersearch_v2.py').read_text()
        seg = src.split('def save_model_atomically', 1)[1]
        seg = seg.split('\ndef ', 1)[0]
        found = set(re.findall(
            r"""f['\"]\{prefix\}([A-Za-z0-9_.]+)['\"]""", seg))
        expect = set(shadow._ARTIFACT_SUFFIXES) | {MANIFEST}
        assert found == expect, (
            'save_model_atomically artifact references drifted from '
            f'shadow._ARTIFACT_SUFFIXES: hypersearch-only={found - expect},'
            f' shadow-only={expect - found}')
        # hypersearch never writes meta artifacts; .prev-ing them at save
        # time would let a rollback resurrect a never-gated meta pairing.
        assert not (found & set(shadow._STALE_META_SUFFIXES))


class TestPanelProducerConsumerParity:
    def test_ms_interact_is_the_only_non_cs_prefixed_feature(self):
        # Any new panel feature invisible to predict_now's
        # startswith('CS_') filter would be a NEW silent train/serve
        # parity break on every hourly stock prediction.
        non_cs = [c for c in panel_ranks.CS_FEATURE_COLS
                  if not c.startswith('CS_')]
        assert non_cs == ['MS_Interact'], (
            f'non-CS_-prefixed panel features {non_cs}: MS_Interact is the '
            'only documented (deferred, ledger P1) injection gap — a new '
            'one would silently neutral-fill live')

    def test_predict_now_filter_shape_pinned(self):
        src = (REPO / 'predict_now.py').read_text()
        assert "c.startswith('CS_')" in src, (
            "predict_now's panel-injection filter changed — re-verify "
            'every panel_ranks.CS_FEATURE_COLS key is injected, then '
            'update this tripwire')
        if 'MS_Interact' in src:
            pytest.fail(
                'predict_now now references MS_Interact — the known '
                'producer/consumer gap appears closed; update this '
                'tripwire to assert full injection parity instead')
