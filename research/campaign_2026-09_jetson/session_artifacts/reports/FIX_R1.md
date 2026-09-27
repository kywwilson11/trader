# FIX_R1 — OI_Chg_24h +inf (74 rows) + non-finite parity guard

## Root cause
- **Pre-fix `oi_archive.py:266`** — `chg = s.ffill().pct_change(24, fill_method=None) * 100`, over a series that holds
  **glitch prints with `oi_value == 0.0`** (Binance metrics files: 125 rows across all 10 perps, `oi` also 0 on 115
  of them; tt_ls/taker columns look normal). Examples: every perp at 2023-04-10 09:00 UTC; bursts on BTC 2024-07-09..15
  (34 BTC rows); 2023-11-11, 2023-11-23, 2024-08-12, 2025-04-11/15, 2025-07-21. ffill does not touch a 0.0, so:
  - at the glitch hour, chg = -100% (74 rows in training_data.parquet sat at exactly -100)
  - 24 rows later, x/0 = **+inf**. These are the 74 inf rows (ETH/SOL/DOGE/BTC/LINK/XRP, 2023-04-11 09:00 through
    2025-07-22 17:00, each glitch timestamp + 24h)
  - OI_Z: the zero enters the 720h rolling mean/std, so z reads around -17 at the glitch and is skewed for 30 days
    (140 rows at z < -5).
- It was not caused by an archive gap or a first print. Those give NaN (fill_method=None), and the harvest then
  fills that NaN with 0.0.
- **Neutral fill at harvest:** `scripts/harvest_crypto_data.py` `_fill_archive_features` (around line 221-233) does
  `fillna(0.0)` on ARCHIVE_FEATURES, which includes OI_Chg_24h and OI_Z. It is not `fill_warmup_features`. Live
  already serves 0.0: `live_oi_features` defaults chg/z to 0.0, and `predict_now` fills 0.0 when the call returns None.

### Can the live path produce this inf today (pre-fix)?
**No inf from a zero denominator.** `live_oi_features` has an `if ref > 0` guard, so a zero or NaN ref gives chg = 0.0,
which is already the offline neutral value. Two latent live defects remained, both fixed here:
1. A zero OKX `oiCcy` print was accepted. It was served as chg = -100 and persisted into `oi_history.json`, where it
   skewed the live z for about 35 days.
2. `float()` parses `'NaN'` and `'inf'`, so a non-finite print could be served as NaN or inf. `json` also round-trips
   NaN, so it would stick in the history and make the z NaN for the full window.

(`oi_history.json` does not exist on this box yet, so nothing is contaminated now.)

## Diff summary (only these files)
- `oi_archive.py`
  - `import math`.
  - `oi_features_for_index` (:265-284): `s = s.where(s > 0)` treats a non-positive notional as a missing print. It is
    masked in place, so the positional 24-row grid is kept and neighbouring rows don't shift. The change then
    ffills over the prior print, and z is NaN at the glitch hour (then 0.0 at harvest). A backstop
    `chg.replace([inf, -inf], nan)` covers any other zero or NaN denominator.
  - `_fetch_okx_oi` (:395-401): a non-finite or ≤0 print is treated like a failed fetch. It returns None, is
    negative-cached and is not persisted.
  - `live_oi_features` (:446+): history samples that are not finite and >0 are ignored for both ref and z. A final
    `math.isfinite` backstop falls back to 0.0.
- `predict_now.py`
  - `import numpy as np`; `NONFINITE_FILL = 0.0`.
  - `_sanitize_nonfinite_features(X, fill, copy)` (:64-95) returns `(X, n_nonfinite, n_inf, bad_col_idx)`. Finite
    or non-float input comes back as the same object, untouched.
  - It is applied to `current_features` just before `scaler_X.transform(current_features[-seq_len:])`, with a
    one-line `[FEATURES] <sym>: N non-finite input value(s) (M inf) in [cols] -> neutral 0.0` print.
  - It runs on the frame matrix, not the slice, so the literal pinned by `tests/test_scaler_slice_equiv.py` stays
    intact. The cost is one isfinite pass.
- `scripts/hypersearch_v2.py`
  - A behaviour-identical twin of the helper (:152-181), kept local so the trainer doesn't import the live module and
    its torch thread side effect.
  - It is applied in place on `all_features` right after the `np.vstack` in `load_data` (:322+), before any per-fold
    `RobustScaler.fit`. It prints `[SANITIZE] N non-finite feature value(s) (M inf) in [cols] -> neutral 0.0`.
  - Because it runs after the float32 cast, it also catches float64→float32 overflow.
- Both guards carry comments saying they are **not model-facing for finite input** (byte-identical, same object).
- `tests/test_oi_inf_2026_09.py` (new, 23 tests). This covers (a) offline: the zero glitch gives no inf and no -100,
  z is NaN at the glitch, non-glitch rows stay bit-equal to the old formula, a NaN denominator gives NaN, and the
  harvest fill is 0.0. It covers (b) live: zero ref → 0.0; a bad sample is skipped for a valid neighbour; z ignores
  persisted NaN/0; `'0'/'-1'/'NaN'/'inf'` prints are rejected and not persisted. It covers (c) both twins, exec'd
  from source via ast (Mac-safe): finite float32/float64 input is the same object and bit-identical; int input is
  untouched; replace+count+columns; `copy` semantics; the twins agree; source order puts the guard before the
  scaler. The end-to-end `get_live_prediction` and `load_data` runs use importorskip on the heavy deps and assert
  the logged count line and bit-equality everywhere else.
- A sanity check with the pre-fix `oi_archive.py` on the same glitch series gives chg[300] = -100, chg[324] = inf
  and min z = -17. Those are exactly the values the new tests reject.

## Verify
```
py_compile oi_archive.py scripts/hypersearch_v2.py predict_now.py tests/test_oi_inf_2026_09.py -> OK
pytest tests/test_oi_inf_2026_09.py tests/test_oi_archive.py tests/test_predict_now.py tests/test_hypersearch_v2.py
       tests/test_c26_P1.py tests/test_review_b08.py tests/test_scaler_slice_equiv.py tests/test_c26_T3.py
       tests/test_c26_T2.py tests/test_grp_deriv.py tests/test_r2c_blend_coherence.py tests/test_r2c_serving_cache.py
       tests/test_prediction_cache.py tests/test_review_b15.py tests/test_ia4_flagged.py tests/test_basis_archive.py
       tests/test_imports.py tests/test_c26_S3.py tests/test_c26_T4.py tests/test_c26_T6.py tests/test_c26_W1.py
       tests/test_base_loop_v3.py tests/test_c26_base_loop_functional.py -q -p no:cacheprovider
====================== 640 passed, 29 warnings in 29.76s =======================
```
(The first pass failed `test_scaler_slice_equiv::test_source_uses_sliced_transform_no_dead_intermediate`, a source-text
pin on the sliced-transform literal. I fixed that by sanitizing the frame matrix instead of the slice; the test is
unchanged.)

## Re-validation on the real archive (read-only)
I recomputed `oi_features_for_index` for the 6 tickers on the exact training_data.parquet index, with the harvest's
NaN→0.0 applied, and did not rewrite the parquet:
- inf rows: **74 → 0**. All 74 formerly-inf rows are now finite ordinary changes, e.g. BTC 14.3, 1.7, 0.08…;
  SOL 26.3, 16.9…
- rows at exactly -100%: 74 → 0.
- OI_Chg_24h changed on 153 rows in total (the glitch hours and their +24h rows). Every other row is bit-identical.
- OI_Z changed on about 26k rows by more than 1e-9 (about 22k by >0.01, 6.4k by >0.1, 391 by >1). These are the
  30-day windows after each zero print, which had been skewed. Rows at z < -5 went 140 → 72, and min z went from
  -16.9 to -6.5 (BTC).
- The other rows differ only at float-ulp level (<1e-9), because the rolling accumulator now sees NaNs.
- **The crypto harvest must be rerun** for these values to reach training_data.parquet, since the stored values are
  still the pre-fix ones. Until then, the hypersearch `[SANITIZE]` guard maps the 74 infs to 0.0 at load. That is a
  different value for those rows than the source fix produces, so rerun before training. Stock is unaffected: it has
  no OI features, and neither the stock files nor the harvest scripts were touched.

## Notes / out of scope
- OI_Z change magnitude is model-facing (the feature was corrupted), but it lands with the scheduled re-harvest +
  retrain (gotcha #2) through the normal challenger path.
- `pct_change(24)` is **positional** (24 rows), not 24 hours. The BTC archive has one 10-hour gap, so the rows around
  it measure about 34h. This is pre-existing and not changed here.
- `_parse_zip` still stores the zero prints. The read-time mask covers both the existing and future archive rows.
  Masking at parse would also pick the last non-zero 5-minute print inside the hour, but would need a re-sync to take
  effect, so I left it alone.
