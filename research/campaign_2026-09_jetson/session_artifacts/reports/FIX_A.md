# FIX_A — C extension archived, backend opt-in, golden test pinned per backend (2026-09-26)

## Diff summary
1. **Archive (plain mv, nothing deleted):** `c_ext/indicators_c.c`, `c_ext/build.py`, root
   `indicators_c.cpython-310-aarch64-linux-gnu.so` -> `archive/c_ext/`; empty `c_ext/` removed.
   All untracked, not gitignored (`git status`: `?? archive/c_ext/`). Row added to
   `archive/README.md` Contents table (origin, tracked=no, overflow at `indicators_c.c:428` via ASan +
   the other prefix loops, the `Py_BuildValue` ref leak, ≤1e-12 rel parity vs numba, no speed gain,
   2026-09-26, restore only after the C source is fixed; also notes that `build.py` writes to its
   `parent.parent`, and that even a restored `.so` loads only with `TRADER_INDICATORS_C=1`).
2. **indicators.py (+11/-6):** `import os`; `_HAS_C = False` by default; `import indicators_c` is
   attempted only when `os.environ.get('TRADER_INDICATORS_C') == '1'`. There is a 2-line comment
   (opt-in, archived, cites the archive row, not model-facing). No kernel was touched.
   Verified:
   - default -> `_HAS_C False`, `_HAS_NUMBA True`;
   - `.so` on `PYTHONPATH` without the env var -> False;
   - with `TRADER_INDICATORS_C=1` -> True.
3. **tests/test_indicators_parity.py:**
   - `test_compute_stock_features_golden_fingerprint` is now parametrized over `numba` (with
     `importorskip("numba")`; this is the production pin) and `pure` (`_HAS_NUMBA=False`), with
     `_HAS_C` forced False via monkeypatch. It asserts:
     - the column set;
     - every column sum against `_GOLDEN_COL_SUMS`. For numba, `_NUMBA_COL_SUM_OVERRIDES` supplies
       ATR_Percentile 117.441678, RSI 16356.790603 and RSI_Divergence -9.780238. I recomputed these
       here and they match the report exactly. All other columns match on both backends.
     - `_WARMUP_PINS`, the NaN count and first-valid index. Pure: ATR_Percentile 112, RSI 14,
       RSI_Divergence 18. Numba: 99, 13 and 17.
   - The hash assertion moved into a new separate test,
     `test_compute_stock_features_fingerprint_tripwire[numba|pure]`, keyed as `_GOLDEN_FINGERPRINTS`:
     - `("darwin", any machine, pandas 3, "pure")`: 8972321854121808304
     - `("linux", "aarch64", pandas 2, "numba")`: 422055739916008620

     It asserts only on a key match and otherwise skips with a message. I split it out so that a
     skip cannot hide the passing column-sum and warm-up checks.
   - `test_rsi_kernel_matches_fallback_clean_data` got a docstring. It says the test deliberately
     compares only from bar 270, and that the warm-up divergence is a known model-facing owner item.
     No behaviour change.
   - The module docstring and the golden-section comment were updated: the C ext is now opt-in, and
     the golden was recorded on the Mac pure path.
   - No other test was changed.
4. **Docs (exact lines only):**
   - `docs/MODULES.md:270`, the Known-issues bullet: marked RESOLVED. The extension existed on the
     Jetson, was archived, and is now opt-in via `TRADER_INDICATORS_C`.
   - `docs/MAP.md:880`: the question is answered.
   - `CLAUDE.md` § Running tests: "the suite is green" is replaced by "verified 2026-09-26: 3919
     passed / 5 failed". The five are the C extension, the fingerprint test, untracked residue and
     `bidask`, all addressed this campaign.

## Verification (jetson env, CUDA_VISIBLE_DEVICES='', PYTHONPATH unset, no noc sitecustomize)
```
py_compile indicators.py tests/test_indicators_parity.py -> OK
import indicators -> _HAS_C False, _HAS_NUMBA True
pytest tests/test_indicators_parity.py tests/test_indicators.py tests/test_indicator_reindex.py tests/test_c26_W1.py -q -p no:cacheprovider -rs
tests/test_indicators_parity.py ........xxx...s
tests/test_indicators.py ..................................................
tests/test_indicator_reindex.py ....        <- previously SIGABRT; now passes
tests/test_c26_W1.py ....................................
SKIPPED [1] tests/test_indicators_parity.py:478: no recorded fingerprint for (linux, aarch64, pandas 2, pure) — got 12400458680678781189; ...
101 passed, 1 skipped, 3 xfailed in 15.22s
```
The single skip is the pure-leg tripwire on Linux (expected). The 3 xfails are the pre-existing
documented divergences. The full suite was not run, per the brief.

## Not done / flags for the orchestrator
- **`docs/FLAGS.md` has no row for the new `TRADER_INDICATORS_C` env var.** It owns every `TRADER_*`
  var, but it was outside my file set. Suggested row: default unset/off; read at `indicators.py:14`;
  not model-facing (float-noise vs numba); opt-in only after the archived C source is fixed.
- **Two stale lines were outside my exact-line mandate:**
  - `docs/MODULES.md:266` still says "(C ext > numba > pure)".
  - `CLAUDE.md:90` still says "Jetson/CI stay green".
- **Unverified on other platforms:**
  - The Mac tripwire key (darwin / pandas 3 / pure) comes from the existing golden and the Mac's
    recorded pandas 3.0.3. I could not run it on the Mac.
  - CI (x86_64 Linux) matches no fingerprint key, so both tripwire legs skip there.
  - The CI numba column sums (6-dp) are assumed to match the aarch64 values. That is untested
    because there is no x86 box. If CI's pandas-3 leg differs, the overrides will show it.
- **The opt-in C path still takes priority over numba when enabled.** The function bodies are
  unchanged (`if _HAS_C:` first), per the "minimal diff, do not touch kernels" instruction.
- **Other agents edited the same files concurrently:**
  - `docs/MODULES.md` has other agents' hunks (jetson setup / ops sections).
  - `archive/README.md` has another agent's `jetson_residue_2026-03/` and `local_residue/` rows.

  My edits were single-substring replacements and preserved theirs.
