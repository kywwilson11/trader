"""SIG-R1-A5: the end-of-search PBO warning fires on noise.

validation.pbo_from_fold_scores' own contract: "Under the no-skill null the
IS winner's OOS rank is uniform, so PBO centres near ~0.5 and with n_folds
folds only takes the values k/n_folds — judge it against 0.5 (Bailey et
al.), never a small threshold." hypersearch_v2.main warns at `pbo > 0.25`,
i.e. whenever ANY of the 3 leave-one-fold-out splits lands below median:
P(warn | no skill) = 0.83 measured (40 trials x 3 folds, 2000 null sims),
vs 0.51 at the documented 0.5 bar. Print-only (no gate reads it).

Source-structure test (main() is not callable without the full training
stack); the numeric half runs the real validation kernel on null data.
Module under test: SIG_R1_HS.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get('SIG_R1_REPO',
                           Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
HS_PATH = os.environ.get('SIG_R1_HS',
                         str(REPO / 'scripts' / 'hypersearch_v2.py'))
SRC = Path(HS_PATH).read_text()


def _warn_threshold():
    m = re.search(r"WARNING: [^']*' if pbo > ([0-9.]+) else ''", SRC)
    assert m, 'PBO warning expression not found'
    return float(m.group(1))


def test_warning_threshold_is_the_documented_half():
    assert _warn_threshold() == 0.5


def test_null_false_alarm_rate_at_threshold_is_about_half():
    from validation import pbo_from_fold_scores
    rng = np.random.default_rng(1)
    p = np.array([pbo_from_fold_scores(rng.normal(size=(40, 3)))
                  for _ in range(600)], float)
    rate = float(np.mean(p > _warn_threshold()))
    assert rate < 0.65, f'warning fires on {rate:.0%} of no-skill searches'
