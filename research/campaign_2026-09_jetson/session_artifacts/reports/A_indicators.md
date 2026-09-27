# A — the feature kernel on the prod box (indicators.py + untracked C extension)

Agent A, 2026-09-26, Jetson (jetson env: py3.10, numpy 1.26.1, pandas 2.3.3, numba 0.63.1).
No production file was touched. All scripts and logs are in `scratchpad/A/`.

## Verdicts

**(1) The SIGABRT is a heap buffer overflow in the untracked C extension.** Six of the nine kernels
write `NAN` into `out[0 .. window-2]` in a plain C loop without clamping to `n`. Any input shorter
than `window-1` bars therefore writes past the end of a numpy buffer. The failing test feeds a
**60-bar** frame, and `compute_features` calls `rolling_percentile(ATR, 100)` (`indicators.py:549`)
and `hurst(Close, 100)` (`indicators.py:580`). Each writes 39 doubles past a 480-byte buffer. ASan
names `c_ext/indicators_c.c:428` (`py_rolling_percentile`) as the first bad write. glibc only
notices the smashed chunk headers at the next malloc, which is pandas `__setitem__` for `Hurst` at
line 580. That is why the traceback points there.

The same bug class is in `hurst` (:506), `linear_slope` (:467), `rolling_std`/bbands (:90),
`rolling_min`/`rolling_max`/stoch (:107/:118). `rsi`/`macd`/`atr`/`obv` overflow only at `n==0`.

There is a second, independent defect: a **reference leak**. `Py_BuildValue("(OOO…)")` INCREFs
arrays that are already owned (`:227, :321, :377`). So macd/bbands/stoch leak 10 arrays on every
`compute_features` call. Measured: 23.4 KB/call on a 274-bar frame.

Normal production frames do not hit the overflow: harvest runs over thousands of bars, and live
frames are 250–320 bars. But no caller guards against the short-frame case, so it is reachable.

**(2) The C extension was numerically equivalent to the numba kernels, so it was never
"model-facing".** It was compared on real BTC-USD (44,484 bars), real TSLA (39,912 bars) and a
300-bar live-sized TSLA tail, all nine kernels:
- NaN placement is identical everywhere, including first-valid indices and a NaN-injected frame.
- Maximum relative difference is ≤1e-12 on O(1) values. The ~1e-8 "relative" outliers are MACD and
  %B crossing zero.
- Full-frame `compute_features` / `compute_stock_features`: 28/38 and 53/63 columns are bit-identical.
  The rest differ by ≤1.6e-8 absolute on values of order 1e5 (FMA/`-O3 -march=native` rounding).
- `Vol_Price_Confirm`, the only feature that could turn rounding noise into a discrete flip, is
  bit-identical.

The training stores (mtime 2026-02-28 02:07) predate the `.so` (2026-02-28 15:51), so the models
were trained on numba features. The bots that ran 2026-02-28 → 2026-05-07 served C features.
That split is within float noise. What the C extension *did* cost production was the memory leak.
From the log call counts, that is about 2.3 GB leaked in the crypto bot and about 19 GB in the
stock bot across process lifetimes (an estimate; there is no RSS telemetry to confirm it).

**(3) The golden-fingerprint failure is not a regression. The golden was recorded on the Mac's
pure-pandas path; the Jetson and CI run the numba path.**
- On the Jetson numba path, exactly three column sums differ from the golden: `ATR_Percentile`,
  `RSI` and `RSI_Divergence`. The causes are the numba vs pandas difference in the rolling-percentile
  warm-up (13 extra NaNs from the ATR warm-up) and the RSI EWM seeding (`gain[0]=0` vs pandas' NaN).
- On the Jetson pure path, all 63 column sums match the golden exactly. The 64-bit fingerprint
  still differs, and it differs identically under numpy 1.26 and numpy 2.2.6. That is platform
  ulp noise: a +1-ulp nudge to one input bar changes the fingerprint and leaves every column sum
  unchanged.
- Production truth is the numba path, fingerprint 422055739916008620.
- CI installs numba in both legs (`requirements-ci.txt`, `ci.yml:48-50`), so this test cannot pass
  on CI either. The "suite is green on Jetson/CI" claim is false for this test.

**Recommendation:**
- **Archive the C extension** (it gives no speed benefit at live size and no memory benefit,
  because numba is loaded anyway).
- **Re-pin the golden test per backend**, using tolerance-based per-column statistics, with the
  numba leg as the production pin. Drop the bit-hash or make it a platform-keyed diagnostic.

---

## Q1 — root cause of the abort

### Code reading (c_ext/indicators_c.c, 573 lines)

| Kernel | Defect | Line | Trigger |
|---|---|---|---|
| `rolling_percentile` | `for (i=0; i<window-1; i++) out[i]=NAN;` with no `i<n` bound, so a **heap write overflow** | `:427-428` | n < window-1 (default 100 → n ≤ 98) |
| `linear_slope` | same | `:466-467` | n < window-1 (5 → n ≤ 3; 100 → n ≤ 98) |
| `hurst` | same | `:505-506` | n ≤ 98 |
| `bbands` → `rolling_std` | same, into `up_arr` | `:89-90` | n ≤ 18 |
| `stoch` → `rolling_min`/`rolling_max` | same, into malloc'd `lowest`/`highest` | `:106-107`, `:117-118` | n ≤ 12 |
| `rsi` | `gain[0]=…; rsi[0]=NAN` | `:159`, `:170` | n == 0 |
| `macd` → `ewm_span` | `out[0]=arr[0]` | `:61` | n == 0 (read + write) |
| `atr` | `tr[0]=high[0]-low[0]` | `:252` | n == 0 |
| `obv` | `out[0]=0.0` | `:400` | n == 0 |
| `macd` / `bbands` / `stoch` | `Py_BuildValue("(OOO)"/"(OOOOO)"/"(OO)", …)` INCREFs new references that are never DECREF'd, so a **reference leak** of every output array (should be `"N"`) | `:227`, `:321`, `:377` | every call |
| `atr` / `stoch` / `obv` | `n` is taken from ONE input, and the others are never length-checked, so an **over-read** is possible | `:244`, `:339`, `:391` | mismatched lengths (never happens from `indicators.py`: all inputs are columns of one frame) |

Not defects:
- dtype and contiguity: `check_array` (`:20-34`) rejects non-C-contiguous, non-1-D and non-float64
  input with a Python exception, and every `indicators.py` wrapper passes
  `series.values.astype(np.float64)`, which is always a fresh contiguous copy. So float32, int and
  strided input cannot reach the kernels unconverted.
- No borrowed-reference returns. The single-array kernels return their own new reference correctly.
- The numba twins (`indicators.py:86-126, 236-318`) use `out[:window-1] = np.nan`. That is a
  **slice**, which clamps at `n`, so the numba path is safe on short frames. This is exactly the
  semantics the C port lost.

The shipped `.so` is this source. Rebuilding `c_ext/indicators_c.c` with the build script's flags
(`gcc -O3 -march=native -fno-math-errno`) gives a byte-identical `.text` section:
`sha256 b5526b3f…f449` for both (`objcopy --only-section=.text`).

### Empirical confirmation

1. **The suite abort, isolated** (C ext loaded vs disabled):
   ```
   $JPY -m pytest tests/test_indicator_reindex.py -q -p no:cacheprovider
   → tests/test_indicator_reindex.py .Fatal Python error: Aborted
       File "/home/kyle/trader/indicators.py", line 580 in compute_features
       File "/home/kyle/trader/tests/test_indicator_reindex.py", line 54 in test_no_deprecation_warning
     exit=134
   PYTHONPATH=$S/noc $JPY -m pytest tests/test_indicator_reindex.py -q -p no:cacheprovider
   → 4 passed, exit=0
   ```
   The first test (n=120) survives. The second (`test_no_deprecation_warning`,
   `tests/test_indicator_reindex.py:39`, **n = 60**) aborts.

2. **ASan, with the exact frame from that test** (`A/asan_driver.py`, ASan build of the same source
   in `A/asan/`, `LD_PRELOAD=libasan.so`):
   ```
   ERROR: AddressSanitizer: heap-buffer-overflow ... WRITE of size 8
     #0 in py_rolling_percentile /home/kyle/trader/c_ext/indicators_c.c:428
   0x… is located 0 bytes to the right of 480-byte region     (480 B = 60 float64)
   ```

3. **ASan matrix, every kernel × boundary length** (`A/asan_case.py`; format `kernel n args |
   result | site`):
   ```
   rsi 60/10/1 CLEAN            rsi 0  → WRITE overflow  indicators_c.c:159
   macd 5 CLEAN                 macd 0 → READ overflow   indicators_c.c:61  (ewm_span)
   atr 10 CLEAN                 atr 0  → READ overflow   indicators_c.c:252
   bbands 19 CLEAN              bbands 18 → WRITE        indicators_c.c:90  (rolling_std)
   stoch 13 CLEAN               stoch 12 → WRITE         indicators_c.c:107 (rolling_min)
   obv 5 CLEAN                  obv 0  → WRITE           indicators_c.c:400
   rolling_percentile 99 CLEAN  98 / 60 → WRITE          indicators_c.c:428
   linear_slope 4 (w=5) CLEAN   3 (w=5), 60 (w=100) → WRITE indicators_c.c:467
   hurst 99 CLEAN               98 / 60 → WRITE          indicators_c.c:506
   ```
   The boundaries are exactly `n = window-2` (overflow) and `n = window-1` (clean), as the code
   predicts.

4. **glibc detection with the production `.so`**:
   `MALLOC_CHECK_=3 $JPY A/repro_min.py {rolling_percentile|hurst|linear_slope} 60 100` →
   `malloc(): invalid size (unsorted)`, core dump, exit 134, for all three. At n=120 all three
   survive.

5. **Reference leak** (`A/leak.py`, `A/leak_cf.py`):
   - After `t = ic.macd(x,…)`, each array has refcount 4 inside the list comprehension; 3 is
     expected. That is one leaked reference per output.
   - 500 calls at n=10,000 cost: macd +114.5 MB, bbands +191.0 MB, stoch +76.8 MB, rsi +0.3 MB.
     That is 3, 5 and 2 arrays × 80 KB × 500, matching exactly.
   - Whole-pipeline: 2,000 `compute_features` calls on a 274-bar BTC frame cost **+46.8 MB with C
     (23.4 KB/call; the theory is 10 × 274 × 8 B = 21.9 KB)** and −8.3 MB (i.e. none) with numba.

### Could harvest or live have been fed such input?

These are structurally reachable, but none happens on typical data:
- **Harvest.** `scripts/harvest_crypto_data.py:143` and `scripts/harvest_stock_data.py:188` compute
  features over each ticker's full history. The only guard is `if ohlcv.empty` (`:139` / `:184`).
  A newly added or thin ticker with 1–98 bars would corrupt the heap (0 bars is excluded).
- **Live `predict_now`.**
  - Crypto `fetch_bars_alpaca(limit=250)` (`market_data.py:195,221`) returns about 250 bars.
  - Stock `fetch_stock_bars_alpaca` covers 45 days (`market_data.py:267,294`), about 200–320 bars.
  - The only guards before `compute_features` / `compute_stock_features` (`predict_now.py:250,287`)
    are `df.empty` (`predict_now.py:216,226`). An illiquid crypto pair with sparse hourly bars, a
    recent IPO, a long halt or an API short-read below 99 bars would overflow.
- **`panel_ranks.compute_live_panel_ranks`.** It explicitly admits frames with **≥60 bars**
  (`panel_ranks.py:184`) into `compute_stock_features` (`:186`). Every frame of 60–98 bars there is
  an overflow.
- **`market_data.get_live_atr`.** Guarded by `len(df) < length+1` (`market_data.py:697`), and ATR
  uses `rolling_mean`, which has no prefix loop. Safe.

Production logs show **no SIGABRT evidence**:
- `pipeline_output.log` records 1,778 bot restarts, **all `crashed (exit 1)`** (Python exceptions),
  none `-6` or `-9`.
- `grep` finds no `malloc()` / `double free` / `corrupted` / `Aborted` in `crypto_bot_output.log`,
  `stock_bot_output.log` or `logs/trader.log*`.

Silent corruption from a short frame cannot be ruled out from the logs, but frames were normally
≥200 bars.

The leak, by contrast, was hit on every call. The logs contain 100,641 crypto and 757,497 stock
`Predicted Return` lines between 2026-02-23 and 2026-05-07, and `PREDICTION_CACHE_ENABLED` did not
exist until 2026-06-18 (default False). Times ~23–26 KB/call, that is roughly 2.3 GB (crypto) and
~19 GB (stock) leaked in total. That is about 260 MB/day for the stock bot, bounded per process by
the weekly cold restart and the crash restarts. This is an estimate from call counts; the logs
contain no RSS telemetry.

## Q2 — numerical parity: C vs numba vs pure

Script: `A/parity.py` (log: `A/parity.log`, JSON: `A/parity.json`) and `A/parity2.py`
(`A/parity2.log`). Data: the real `training_data.parquet` / `stock_training_data.parquet` OHLCV
(read-only). Every output was compared, including the NaN-leading derived inputs exactly as
`compute_features` feeds them: percentile of ATR (13-bar NaN lead) and slope of RSI (13-bar NaN
lead).

**C vs numba, BTC-USD n=44,484** (TSLA n=39,912 and the 300-bar tail are the same or tighter):

| output | NaN mismatches | first-valid C/numba | max abs | bit-exact rows |
|---|---|---|---|---|
| RSI | 0 | 13/13 | 2.8e-14 | 35228/44471 |
| MACD / hist / signal | 0 | 0/0 | 7.3e-11 / 4.1e-11 / 5.7e-11 (price ~1e5) | partial |
| ATR | 0 | 13/13 | 0 | all |
| BB lower/mid/upper | 0 | 19/19 | 1.6e-8 / 0 / 1.6e-8 (price ~1.2e5, rel 6.7e-13) | mid all |
| BB bw / %B | 0 | 19/19 | 1.3e-12 / 1.7e-10 | partial |
| Stoch k / d | 0 | 15/15, 17/17 | 0 | all |
| OBV | 0 | 0/0 | 0 | all |
| rolling_percentile(ATR,100) | 0 | 99/99 | 0 | all |
| linear_slope(RSI,5) / (Close,5) | 0 | 17/17, 4/4 | 8.9e-15 / 0 | partial / all |
| hurst(Close,100) | 0 | 99/99 | 1.1e-16 | 42016/44385 |

**NaN-injected frame** (BTC tail of 3,000 bars, 1% random NaN per OHLCV column plus a 20-bar full
outage): C vs numba has **0 NaN-placement mismatches on all 16 outputs**, and the maximum absolute
difference is ≤7.8e-9 (BB bands at price ~1e5). The C port reproduces the numba NaN semantics
exactly, including the known `_ewm_span` NaN-poisoning and the valid-count percentile.

**Full frame** (`compute_features` on BTC, `compute_stock_features` on TSLA): 28/38 and 53/63
columns are bit-identical. The differing ones are RSI, MACD×3, BBL/BBU/BBB/BBP, RSI_Divergence and
Hurst, all at ≤1.6e-8 absolute. `Vol_Price_Confirm` (sign product) is bit-identical, and there are
0 NaN mismatches.

**Speed** (`A/bench.py`): on a 274-bar frame, numba 21.95 ms vs C 22.72 ms per `compute_features`.
At 44k bars, numba 124.5 ms vs C 95.7 ms. There is no live benefit. There is also no memory
benefit: numba is imported anyway by `policy_exits.py` and `sample_weights.py`.

**Conclusion:** the C extension was never model-facing relative to numba. Its outputs are within
float32 rounding (the LSTM consumes float32) of the numba outputs the models were trained on.

**Pure path vs numba on real data** (context for Q3). On clean, gap-free data there are
**three real divergences**:
- **RSI** — first-valid 13 (numba) vs 14 (pure). Maximum difference 9.7 (BTC) / 13.6 (TSLA) /
  14.5 (300-bar tail) RSI points. It decays at a rate of (13/14) per bar, so on a 300-bar live
  frame **0/286 rows are bit-exact**.
- **RSI_Divergence** — inherits the RSI difference; maximum 1.1–1.5.
- **ATR_Percentile** — 13 NaN mismatches (first-valid 99 vs 112), identical where both are valid.
  Pandas `rolling().apply` needs a full non-NaN window; the kernel counts valid values only.

Everything else agrees to ≤1e-6 relative. There is one spurious "relative 7e287" for Stoch k: an
exact 0 vs 1e-13 from pandas' online rolling sum.

## Q3 — golden fingerprint on the Jetson

**How the test builds it.** `tests/test_indicators_parity.py:366-381`:
`compute_stock_features(_golden_stock_frame())` produces 45 business days × 7 bars = 315 rows,
seed 2026, plus an SPY series. The fingerprint is
`int(pd.util.hash_pandas_object(result[sorted cols].round(10)).sum())`, followed by a per-column
`round(sum, 6)` check against `_GOLDEN_COL_SUMS`. The golden literal was added in `6bb38e7`
(2026-07-13) and has never changed. The test's own comment says it "runs entirely on pure
pandas/numpy (no numba…)". That was true only on the Mac. The test does not force a backend, so on
the Jetson or CI it runs whatever `indicators` dispatches to.

**Measured** (`A/golden.py`; run in the jetson env with numpy 1.26.1, and in the base env with
numpy 2.2.6 / pandas 2.3.3 with identical results):

| backend | fingerprint | column-sum mismatches vs golden |
|---|---|---|
| C ext (default here while the `.so` is present) | 12517820087942437080 | ATR_Percentile, RSI, RSI_Divergence |
| **numba (production truth)** | **422055739916008620** | `ATR_Percentile` 117.441678 vs 109.49 · `RSI` 16356.790603 vs 16416.688909 · `RSI_Divergence` -9.780238 vs -17.204879 |
| pure pandas | 12400458680678781189 | **none — all 63 sums match the Mac golden** |
| Mac golden | 8972321854121808304 | — |

**numba vs pure, per column** (315-row fixture):
- `ATR_Percentile`: 13 NaN mismatches (first-valid 99 vs 112); 0 difference where both are valid.
- `RSI`: 1 NaN mismatch (13 vs 14); maximum 10.4; 301 rows > 1e-10.
- `RSI_Divergence`: 1 NaN mismatch; maximum 1.05; 283 rows.
- `BBL`/`BBU`/`BBP`: ≤2.3e-11, but enough to flip `round(10)` hashes.
- **Stoch does not differ in this fixture**, because it has no interior NaN.

So the documented claim (`docs/MODULES.md:270`, the three xfails at
`tests/test_indicators_parity.py:211-261`) is only partly accurate:
- The rolling-percentile divergence hits **every** frame through ATR's ordinary 13-bar warm-up,
  not just "NaN-bearing windows".
- The Stoch divergence needs interior NaN, which real OHLCV lacks.
- The largest divergence on clean data is the **RSI seeding** (`_rsi_core` seeds `gain[0]=loss[0]=0`,
  `indicators.py:136-137`; pandas `diff()` gives NaN at index 0, `indicators.py:327`). Its test
  (`test_rsi_kernel_matches_fallback_clean_data`) compares only bars 270-299, so it hides the
  warm-up gap.

**Why pure-Jetson ≠ pure-Mac even though every sum matches.** The fingerprint is an exact bit hash:
- `round(10)` does not quantize large-magnitude columns. A 1-ulp change survives `round(10)` in 80%
  of `OBV` cells (|v| up to 1.07e7), 40% of `Volume` and 43% of `Volume_SMA_20`.
- Nudging a single fixture `Close` bar by +1 ulp (bar 200) changes the fingerprint
  (12400458680678781189 → 4348029342516541295) with **zero** column-sum changes (`A/golden_ulp.py`).
- The hash also distinguishes `-0.0` from `0.0` (`Hour_cos` at hour 18 rounds to -0.0), so sign
  conventions of libm cos/sin matter too.

**numpy 1.x vs 2.x is excluded** as the cause on Linux: numpy 1.26.1 and 2.2.6 produce identical
fingerprints for both backends. The residual Mac-vs-Linux difference is therefore ulp-level noise
from the macOS arm64 libm (`np.exp` builds the fixture's Close; sin/cos feed the calendar columns)
and/or pandas 3.0 internals on the Mac. It cannot be isolated further without the Mac. A signed-zero
brute force over the tiny-value columns did not reproduce the golden. Either way it is not a
semantic difference.

**Production truth on this box.** Harvest and `predict_now` both run the numba path once the `.so`
is gone; the training stores were built on it. So the value to pin is the **numba** result:
column sums equal to golden except those three, and fingerprint 422055739916008620 (Jetson, numpy
1.26/2.2, pandas 2.3.3). CI installs numba in both legs (`requirements-ci.txt:18`,
`.github/workflows/ci.yml:48-50`), so CI also takes the numba path and the current golden test must
fail there.

## Recommendations

### C extension: archive it; do not fix it

Why archive:
- It adds no model value (Q2) and no speed or memory benefit at live size.
- It has two live defect classes: a heap overflow on short frames and a ~23 KB/call reference leak.
- Nothing tests it. `_kernel_vs_fallback` forces `_HAS_C=False` (`tests/test_indicators_parity.py:110`),
  and the Mac and CI never have it.
- Because it is untracked, it silently outranks numba (`indicators.py:11-16`, "C > Numba") on any
  box where the `.so` happens to sit.

If the owner ever wants it back, the fix would be about 12 lines:
- Clamp every prefix loop to `min(window-1, n)`.
- Early-return an empty array when `n==0`.
- Use `Py_BuildValue("(NNN)")`.
- Check that all input lengths are equal.
- Add a short-frame and parity test.

It is still not worth carrying.

Move (plain `mv`; both paths are untracked, and CLAUDE.md gotcha #6 forbids deleting):
```
archive/c_ext/indicators_c.c            <- c_ext/indicators_c.c
archive/c_ext/build.py                  <- c_ext/build.py
archive/c_ext/indicators_c.cpython-310-aarch64-linux-gnu.so  <- ./indicators_c.cpython-310-aarch64-linux-gnu.so
```
`archive/` is not on `sys.path`, so after the move `import indicators_c` fails, `_HAS_C=False`, and
the numba path runs.

Proposed row for `archive/README.md` (matching the table's columns):

| `c_ext/` (`indicators_c.c`, `build.py`, `indicators_c.cpython-310-aarch64-linux-gnu.so`) | `c_ext/` + repo root `.so` | no (untracked, not gitignored) | Jetson-only C port (built 2026-02-28) of the 9 numba kernels; `indicators.py` auto-preferred it (`_HAS_C`). Heap-buffer-overflow when n < window-1 (`indicators_c.c:90,107,118,428,467,506`; n==0 in rsi/macd/atr/obv) aborted `tests/test_indicator_reindex.py`; `Py_BuildValue("(O…)")` leaked every macd/bbands/stoch output (~23 KB per `compute_features`). Numerically ≡ numba (≤1e-12 rel, identical NaN placement), so not model-facing; models were trained on numba features. Evidence: campaign 2026-09-26 report A. | 2026-09-26 | Only after fixing the prefix loops and the `N` refs. Restoring the `.so` to the root silently re-enables it (`indicators.py:12-16`). Nothing else reads these paths (`grep indicators_c` hits only `indicators.py` and docs). |

Follow-ups for the owner (not implemented; cross-file):
- **(a)** `indicators.py:11-16`: either drop the `_HAS_C` branch (archive the branch text in
  `08_removed_code.md` per convention) or gate it behind a default-OFF `TRADER_INDICATORS_C` env flag
  and demote it below numba. That way a stray `.so` can never change the backend silently.
- **(b)** Add a minimum-length guard (e.g. `len(df) >= 100` or `>= seq_len + 100`) before
  `compute_features` in `predict_now.py`, the harvest scripts and `panel_ranks.py:184` (which
  admits 60-bar frames). The numba path survives short frames but emits all-NaN long-window
  features.
- **(c)** Update the docs:
  - `docs/MODULES.md:270` ("no `indicators_c` source… unverified whether the Jetson has a local
    extension") is now verified: it existed; archived.
  - `docs/MAP.md:880` (open question on `indicators_c*.so` presence) is answered.
  - CLAUDE.md's "on the full Jetson stack the suite is green" is false for
    `test_compute_stock_features_golden_fingerprint` until the test is fixed.

### Golden-fingerprint test: pin by backend, pin the numba path as production truth

Replace the single bit-hash with a backend-parametrized, tolerance-based check (recommendation
only; not implemented):

1. **Always force `indicators._HAS_C = False`** in the test via monkeypatch, so a stray `.so`
   cannot change the result.
2. **Parametrize over `backend in ("numba", "pure")`**, and set `_HAS_NUMBA` accordingly:
   - The `numba` leg is `skipif(not indicators._HAS_NUMBA)`. It runs on Jetson and CI and is the
     **production pin**.
   - The `pure` leg runs everywhere. It is the Mac's only leg, and still guards the fallback.
3. **Per-backend expectations** = the existing `_GOLDEN_COL_SUMS` plus an override dict for numba:
   `{'ATR_Percentile': 117.441678, 'RSI': 16356.790603, 'RSI_Divergence': -9.780238}`. All other
   columns match on both backends. Also pin, per column:
   - the **NaN count / first-valid index** (numba `ATR_Percentile` first-valid 99 vs pure 112;
     `RSI` 13 vs 14; `RSI_Divergence` 17 vs 18), which makes the known divergence explicit;
   - a second moment (sum of squares) with `rel=1e-9`, to catch real value changes that a sum can
     mask.
4. **Drop the `hash_pandas_object` fingerprint as an assertion.** It is ulp- and signed-zero
   sensitive and platform-specific: `round(10)` does not quantize OBV, Volume or Volume_SMA_20, and
   one 1-ulp input nudge flips it. If a refactor tripwire is still wanted, keep it only as
   `skipif(sys.platform, numpy/pandas major) != recorded` with **two** recorded values: the Mac pure
   value 8972321854121808304 and the Linux numba value 422055739916008620.
5. Separately (queued owner decisions, model-facing, retrain-bundled):
   - the RSI seed and rolling-percentile warm-up divergences mean the **Mac pure path does not
     reproduce production features** for roughly the first 100–270 bars of any frame;
   - `test_rsi_kernel_matches_fallback_clean_data`'s bars-270+ window hides this. Make it explicit
     rather than silent.

## Reproduction index (scratchpad/A/)
- `repro_min.py`: single-kernel short-frame repro with the prod `.so`
  (`MALLOC_CHECK_=3 $JPY repro_min.py hurst 60 100`).
- `asan/`, `asan_driver.py`, `asan_case.py`: ASan build of `c_ext/indicators_c.c` (`-O1 -g
  -fsanitize=address`), the test's exact frame, and the per-kernel boundary matrix. Run with
  `LD_PRELOAD="/usr/lib/gcc/aarch64-linux-gnu/11/libasan.so $LD_PRELOAD"
  ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0`.
- `rebuilt/`: rebuild with the production flags; `.text` sha256 is identical to the shipped `.so`.
- `leak.py`, `leak_cf.py`: refcount and RSS leak measurements.
- `parity.py`/`.log`/`.json`, `parity2.py`/`.log`: Q2.
- `golden.py`, `golden_probe.py`, `golden_ulp.py`, `stubs/pytest.py`: Q3. The base env (numpy
  2.2.6) run uses `PYTHONPATH=A/stubs /home/kyle/miniforge3/bin/python golden.py`.
- `bench.py`: C vs numba timing.
All processes stayed under ~450 MB RSS, with `CUDA_VISIBLE_DEVICES=''`.
