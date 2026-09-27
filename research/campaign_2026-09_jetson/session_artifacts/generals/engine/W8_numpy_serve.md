# W8 — X7 numpy LSTM serving spike (2026-09-27)
VERDICT vs pre-registered X7 rule: **NOT FEASIBLE (partial — fails criterion 1a only)**. No production file edited.
## LANDED (new files only; research/measurement-only, nothing imports them)
- /home/kyle/trader/lstm_numpy_serve.py — `export_weights` (:153, fail-closed exact key/shape check, atomic npz,
  optional .pth sha256), `export_from_pth` (:201, lazy torch), `is_fresh` (:214), `NumpyLSTM` (:223, fp32 default / fp64),
  op-mapping table in the docstring; `__main__` parity/memory harness (extract runs in its own process).
- tests/test_lstm_numpy_serve.py — 13 tests: fp64 independent reference (1e-12), shape/dtype, npz round-trip, fail-closed
  export, is_fresh, no-torch import, source pin (no module incl. predict_now imports it); torch parity (importorskip) 3 configs
  + jit + saturated inputs (1e-6 fp32 / 1e-10 fp64). Mutation check: q-scale/LN-eps/bias perturbation → Δ 4e-4..1e-1.
- Appendix appended: research_engine.md "## X7 spike results (W8, 2026-09-27)" (metrics table + verdict).
## Metrics (real champions = archive/models/2026-04_pre_rebuild/*, the only ones on disk; torch ref = predict_now.load_model jit, 2 thr)
| | crypto (H288, 1.38M params) | stock (H160, 438k) |
|---|---|---|
| windows (real, live-identical build) | 1,575 / 300 ts | 1,632 / 40 ts |
| max\|Δ\| np-fp32 vs torch-fp32 | **5.7e-6 FAIL** | **6.0e-6 FAIL** |
| max\|Δ\| np-fp64 vs torch-fp32 | 8.0e-6 | 1.17e-5 |
| max\|Δ\| np-fp64 vs torch-fp64 (mapping) | 1.6e-14 | 1.2e-14 |
| torch-fp32 own error vs fp64 | 8.0e-6 | 1.17e-5 |
| sign / threshold / rank changes | 0/0/0 PASS | 0/0/0 PASS |
| RSS min proc, median of 3 (numpy vs torch) | 45 vs 400 → −355 MB PASS | 43 vs 392 → −349 MB |
| RSS bot-like proc (lgb+pandas+repo mods first) | 218 vs 567 → −349 MB | 214 vs 560 → −346 MB |
| ms/fwd torch-jit / numpy (BLAS 6 thr) / numpy (BLAS 1 thr) | 13.4 / 18.8 / 9.7 | 12.6 / 12.8 / 5.9 |
| per-cycle (6 / 46 fwds, all memo misses) torch vs numpy-1thr | 0.08 vs 0.06 s | 0.58 vs 0.27 s |
lightgbm import +53 MB. Timings under CEO training load (noisy; one 6-thr numpy probe hit 74.6 ms).
## FOUND-NOT-FIXED (owner items)
- OBJECTIVE: criterion 1a (1e-6 absolute) is below the fp32 noise floor of the reference itself (torch-fp32 is
  8.0e-6/1.17e-5 from fp64; numpy's own B=1-vs-batched differs 3.8e-6); mirroring torch's bias-add order did not
  help. Re-registering 1a (e.g. "≤ torch's own fp32-vs-fp64 error + zero decision changes") is an OWNER decision —
  goalpost not moved, so no integration proposal was written.
- Any future flip needs SIGNAL's predict_now.py:16/:110/:484/:487 torch uses made conditional; base_loop.py:31 is
  the only bot-path torch importer (hw_monitor.py:140 short-circuits). Saving is ~350 MB per bot process.
- Doc drift: E5.1/M2 and this task's "champion has 293,761 params" = the synthetic H=128 probe; real crypto
  champion is 1,378,497 params (5.5 MB fp32). M2's "54 ms/forward" was per-call weight re-casting in the prototype.
- Etiquette: stock-store extraction peaked 1,421 MB RSS (separate numpy-only process; under the task's 1.5 GB cap).
VERIFIED-CLEAN: torch jit B=1 vs eager batched bit-identical (0.0); numerics unchanged under OPENBLAS_NUM_THREADS=1.
## TEST RUNS (each via hwlock heavy, CUDA_VISIBLE_DEVICES='', one file per process)
- py_compile lstm_numpy_serve.py tests/test_lstm_numpy_serve.py — ok
- pytest tests/test_lstm_numpy_serve.py 13 passed; test_predict_now.py 11 passed; test_imports.py 21 passed (hygiene
  "MOD v2_study.db / tests/test_intel_gui2_2026_09.py" = concurrent writes by others). Raw: w8/work/*_parity*.json, memory_*.json
