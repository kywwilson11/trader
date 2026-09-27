# SIG-R1-A1 — evaluate_on_holdout feeds calendar_effective_n epoch SECONDS (contract: HOURS)

**Defect (OBJECTIVE).** scripts/hypersearch_v2.py:1933 (PROMOTION_GATE_V2 branch) and :2006 (legacy
side-by-side instrumentation, runs on EVERY holdout eval) call
`calendar_effective_n(entry_t, exit_t)` with `entry_t = all_times[global_rows]` — int64 epoch
SECONDS. sample_weights._coerce_time_intervals (sample_weights.py:296-297): "Numeric pairs are float64
HOURS by contract" (datetime64 is converted to hours — backtest.py:482-498 passes datetime64, so the
two consumers of ONE estimator disagree on identical trades). Concurrency is binned per second.

**Repro** (`/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/signal_r1/audit/repro_pure.py`, section R3; output `repro_pure.out`):
200 back-to-back fb=4 holds on 2 names offset 1 bar: hours n_eff=80.40, datetime64 (backtest) 80.40,
epoch seconds (hypersearch) **100.24**. Memory: traced peak of one call = **182 MB** at a 44-day holdout,
**829 MB** at 200 days (hours units: 54 KB / 232 KB). Live crypto store's legacy holdout span =
**219.9 days** (`holdout_span.py`) -> ~0.9 GB transient inside the hypersearch process on every
crypto holdout evaluation (stock: 35.6 d, ~150 MB).

**Fix.** Pass `entry_t / 3600.0, exit_t / 3600.0` at both call sites (clustered_effective_n,
the legacy DSR input, is unit-invariant and untouched).

**Test** tests/test_sig_r1_a1.py runs the REAL evaluate_on_holdout on CPU (tiny RegressionLSTM,
synthetic 2-name panel). Live -> FAIL, patched -> PASS (`run_landing_test.sh A1 live|patched`):
```
E         comparison failed
E         Obtained: 29.248038330671513
E         Expected: 23.600000000000023 ± 2.4e-05
2 failed, 1 passed in 6.34s
3 passed in 6.20s
```
(live: n_eff 29.248 from seconds vs 23.600 hours-correct.)

**Blast radius.** Callers: hypersearch main (:2822) and scripts/window_ab.py:316. No existing test
pins the changed lines (grep of tests/ for `calendar_effective_n(entry_t` = 0 hits).
**Model-facing?** Gate-facing ONLY under PROMOTION_GATE_V2 (default OFF, where it is the DSR n_eff).
Flag OFF: the gate's n_eff/DSR are byte-identical (pinned by test_legacy_gate_numbers_untouched); only the
persisted `holdout.n_eff_v2` instrumentation value changes (to the correct one) and the ~0.9 GB
transient disappears. No new flag needed. **Indisputable:** a unit mismatch against the callee's
written contract, and backtest's twin call already uses hours.

**Existing pins vs ALL FIVE patches applied together** (they apply cleanly in sequence with
`patch`; mirror repo `/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/signal_r1/audit/mirror`, `/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/signal_r1/audit/run_pins.sh`, output `/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/signal_r1/audit/pins_combined.out`):
test_r2c_training_repairs 45 passed · test_c26_T1 30 · test_r2c_blend_coherence 39 · test_r2c_lgb_refit 30 ·
test_r2c_holdout_boundary 41 · test_r2c_rankic_ledger 56 · test_hypersearch_v2 15 — 0 failures.
