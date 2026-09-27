# Mac runbook — atomic fix for the `sys.modules['dotenv']` test leak

*Research note, 2026-09-27 (W14, INTEL dept). **Dev-Mac only.** It carries out `tests/README.md` § Known hygiene
items, item 1, which is an **OWNER DECISION**. Nothing here has been applied. The diffs below come verbatim from the
INTEL W1 scratch (`W1/fix/T2.diff`, `W1/fix/T3.diff`). On the Jetson, on 2026-09-27, both applied cleanly
(`patch --dry-run`). The patched files are byte-identical to W1's tested copies and `py_compile` passes on both.*

## Why this exists, and why it must be one atomic change

- `tests/test_c26_T2.py:21-24` and `tests/test_c26_T3.py:25-28` each put a `dotenv` stub into `sys.modules` when
  the test module is imported, and neither ever removes it. The stub installs only when `python-dotenv` is missing,
  so the leak happens on the dev Mac only. The Jetson and CI have dotenv, so there the fix changes nothing.
- The files run in alphabetical order, so the leaked stub reaches `test_gpu_lock`, `test_imports`, `test_new_modules`
  and `test_wave4` (`c26` sorts before `g`/`i`/`n`/`w`). Those files then import `trading_utils` and
  `harvest_stock_data` against the stub instead of hitting the real dev-Mac `ModuleNotFoundError`.
  The result is that **7 names are masked** (W1 measurement, dotenv blocked by a `sitecustomize`).
- **Both diffs must land together.** Either file alone still leaks. On the Jetson, W1 ran the fixed T2 with the
  unfixed T3 and `test_gpu_lock` still passed (`W1/leak_exp.txt`).
- **The baseline has to be regenerated in the same change.** `scripts/ab_check.sh` compares failure *names*
  with `tests/baseline_failures.txt`. If the fix lands without the new baseline, every later gate reports 7 NEW
  names and exits 1, which looks like a regression. If the baseline is regenerated without the fix, it gets out of
  step with the tests. The name count and the suite counts in `CLAUDE.md` also need to move in the same change.
- **Single-writer tree only.** Do this only when no other Claude session, editor or pytest run is touching this
  checkout. The regeneration step writes down whatever fails *right now*. A concurrent writer's half-finished
  failures would be recorded as "missing-dependency" names. The repo-root hygiene section also reports the whole
  tree. If another session shares the tree, stop. Do not regenerate. Rebuild the old baseline with
  `git show HEAD:tests/baseline_failures.txt` instead.

## The 7 names this uncovers (all are `dotenv`)

Each one fails because its first import that needs a heavy dependency is `from dotenv import load_dotenv`.

| Name | Import chain to `dotenv` |
|---|---|
| `FAILED tests/test_gpu_lock.py::test_choose_inference_device_always_cpu` | test :133 `from trading_utils import …` → `trading_utils.py:14` |
| `FAILED tests/test_new_modules.py::TestKelly::test_compute_kelly_no_history` | `trading_utils` → `trading_utils.py:14` |
| `FAILED tests/test_new_modules.py::TestKelly::test_kelly_position_size_default` | same |
| `FAILED tests/test_new_modules.py::TestKelly::test_shared_constants` | same |
| `FAILED tests/test_new_modules.py::TestKellyScoping::test_asset_type_scopes_the_sample` | same |
| `FAILED tests/test_new_modules.py::TestKellyScoping::test_recency_is_time_ordered_not_dict_ordered` | same |
| `FAILED tests/test_wave4.py::TestWarmupFill::test_long_warmup_features_survive_dropna` | test :138 `from harvest_stock_data import …` → `scripts/harvest_stock_data.py:26` |

Expected: **23 + 7 = 30 names** (23 `FAILED`, 7 `ERROR`, now spread over 11 files instead of 10, because
`test_gpu_lock` is new). The `dotenv` count goes from 1 to 8.

Two side effects produce no name at all:
- `passed` drops by at least 7.
- The dotenv-only skips in `tests/test_imports.py:44-52` go from pass back to **skip**. W1 found this by reading
  the code and did not measure it, so `skipped` is expected to rise.

**Write down the counts the run actually prints.** The figures above are what to expect. They are not the record.

## Step 0 — pre-flight (stop if any check fails)

Run all of these in **one terminal session**, because `$W` has to stay set from step to step.

```bash
cd ~/Desktop/Projects/trader            # adjust to the Mac checkout
W=$(mktemp -d); echo "scratch: $W"
git status --short tests/test_c26_T2.py tests/test_c26_T3.py tests/baseline_failures.txt   # must print NOTHING
python3 -c "import dotenv" 2>/dev/null && echo "dotenv IS installed here: STOP, this runbook is a no-op" || echo "ok: dotenv absent"
shasum -a 256 tests/test_c26_T2.py tests/test_c26_T3.py
#   Jetson pre-image (2026-09-27): T2 e9e2cf1f512cc9516b76418f3ed7fbd0d4b8ca3e1687a793922140e623d0af56
#                                  T3 91d437d6a93bfd39d122806da7b54414600b24a7f72e19abf08ea0eff829229c
#   If they differ, the Mac tree is not synced. The dry-run in step 1 decides whether you can continue.
cp CLAUDE.md tests/README.md tests/baseline_failures.txt "$W"/          # backups for the rollback
bash scripts/ab_check.sh; echo "pre-fix ab_check exit=$?"                # must be 0 (clean start)
```

## Step 1 — apply both diffs (verbatim from W1/fix)

The `+++` lines contain the path to the Jetson scratch copy. That path is harmless, because `patch` gets the
target file name explicitly. Do **not** use `git apply`, which reads the header paths. Dry-run both first:

```bash
cat > "$W/T2.diff" <<'EOF'
--- tests/test_c26_T2.py	2026-09-26 19:51:44.929992485 -0500
+++ /tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/generals/intel/w/W1/fix/test_c26_T2_fixed.py	2026-09-27 00:33:15.270272427 -0500
@@ -16,19 +16,34 @@
 
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
+import data_utils
+import data_sources
+import market_data
+import panel_ranks
+
+# dev-Mac: harvest_* do `from dotenv import load_dotenv` at module scope
+# (scripts/harvest_stock_data.py:26, scripts/harvest_crypto_data.py:33) and are
+# the ONLY modules here that need it. Stub dotenv for these two imports alone,
+# then drop the stub and the two stub-bound modules so later files (test_gpu_lock,
+# test_imports, test_new_modules, test_wave4) still see the real dev-Mac
+# ModuleNotFoundError. Same scoping as test_c26_S3.py / test_c26_base_loop_functional.py.
+_STUB_SCOPE = ('dotenv', 'harvest_stock_data', 'harvest_crypto_data')
 try:
     import dotenv  # noqa: F401
-except ImportError:  # dev-Mac: stub load_dotenv so harvest modules import
+    _saved = None
+except ImportError:
+    _saved = {n: sys.modules.pop(n) for n in _STUB_SCOPE if n in sys.modules}
     _m = types.ModuleType('dotenv')
     _m.load_dotenv = lambda *a, **k: None
     sys.modules['dotenv'] = _m
-
-import data_utils
-import data_sources
-import market_data
-import panel_ranks
-import harvest_stock_data as h
-import harvest_crypto_data as hc
+try:
+    import harvest_stock_data as h
+    import harvest_crypto_data as hc
+finally:
+    if _saved is not None:  # restore sys.modules exactly as it was
+        for _n in _STUB_SCOPE:
+            sys.modules.pop(_n, None)
+        sys.modules.update(_saved)
 
 
 # =====================================================================
EOF
cat > "$W/T3.diff" <<'EOF'
--- tests/test_c26_T3.py	2026-09-26 19:51:44.929992485 -0500
+++ /tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/generals/intel/w/W1/fix/test_c26_T3_fixed.py	2026-09-27 00:31:56.382589213 -0500
@@ -20,13 +20,6 @@
 REPO = Path(__file__).resolve().parent.parent
 sys.path.insert(0, str(REPO))
 sys.path.insert(0, str(REPO / 'scripts'))
-try:
-    import dotenv  # noqa: F401
-except ImportError:  # dev-Mac: stub load_dotenv so scripts import
-    _m = types.ModuleType('dotenv')
-    _m.load_dotenv = lambda *a, **k: None
-    sys.modules['dotenv'] = _m
-
 import cost_regime
 import liquidity
 
@@ -482,10 +475,27 @@
 
 class TestHarvestWiringInert:
     def test_harvest_modules_import(self):
-        import harvest_crypto_data  # noqa: F401
-        import harvest_stock_data as h
-        assert callable(h._minute_edge_overlay)
-        assert h.MINUTE_EDGE_DAYS == 120
+        # dev-Mac: stub dotenv for THIS import only (T2 pattern), then restore
+        # sys.modules exactly — no stub or stub-bound module outlives the test.
+        names = ('dotenv', 'harvest_crypto_data', 'harvest_stock_data')
+        try:
+            import dotenv  # noqa: F401
+            saved = None
+        except ImportError:
+            saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
+            _m = types.ModuleType('dotenv')
+            _m.load_dotenv = lambda *a, **k: None
+            sys.modules['dotenv'] = _m
+        try:
+            import harvest_crypto_data  # noqa: F401
+            import harvest_stock_data as h
+            assert callable(h._minute_edge_overlay)
+            assert h.MINUTE_EDGE_DAYS == 120
+        finally:
+            if saved is not None:
+                for n in names:
+                    sys.modules.pop(n, None)
+                sys.modules.update(saved)
 
 
 # =====================================================================
EOF
patch --dry-run tests/test_c26_T2.py < "$W/T2.diff" && patch --dry-run tests/test_c26_T3.py < "$W/T3.diff" \
  && patch tests/test_c26_T2.py < "$W/T2.diff" && patch tests/test_c26_T3.py < "$W/T3.diff"
python3 -m py_compile tests/test_c26_T2.py tests/test_c26_T3.py && echo compiled
shasum -a 256 tests/test_c26_T2.py tests/test_c26_T3.py
#   Jetson post-image: T2 d32d3b1c7e47a3e91514cc9f9e25dbca27798b9bc2c6286966ff69cf71245006
#                      T3 ef804ef6d4a8251c2912f374297ef0c2a798f34822303c88f59927bc83ee0e1a
python3 -m pytest tests/test_c26_T2.py tests/test_c26_T3.py -q -p no:cacheprovider   # must be 0 failed
```

## Step 2 — prove the delta is exactly the 7 names (before touching the baseline)

```bash
cat > "$W/expected7.txt" <<'EOF'
FAILED tests/test_gpu_lock.py::test_choose_inference_device_always_cpu
FAILED tests/test_new_modules.py::TestKelly::test_compute_kelly_no_history
FAILED tests/test_new_modules.py::TestKelly::test_kelly_position_size_default
FAILED tests/test_new_modules.py::TestKelly::test_shared_constants
FAILED tests/test_new_modules.py::TestKellyScoping::test_asset_type_scopes_the_sample
FAILED tests/test_new_modules.py::TestKellyScoping::test_recency_is_time_ordered_not_dict_ordered
FAILED tests/test_wave4.py::TestWarmupFill::test_long_warmup_features_survive_dropna
EOF
LC_ALL=C sort -u -o "$W/expected7.txt" "$W/expected7.txt"
PY_COLORS=0 python3 -m pytest tests/ --continue-on-collection-errors -q 2>/dev/null > "$W/run.txt"
grep -E '^(FAILED|ERROR)' "$W/run.txt" | sed 's/ - .*//' | LC_ALL=C sort -u > "$W/names.txt"
grep -v '^#' tests/baseline_failures.txt | sed '/^[[:space:]]*$/d' | LC_ALL=C sort -u > "$W/old.txt"
grep -E '[0-9]+ (passed|failed)' "$W/run.txt" | tail -1          # RECORD this counts line
wc -l < "$W/names.txt"                                            # expect 30
comm -13 "$W/old.txt" "$W/names.txt" > "$W/added.txt"             # expect exactly the 7
comm -23 "$W/old.txt" "$W/names.txt" > "$W/gone.txt"              # expect EMPTY
diff "$W/added.txt" "$W/expected7.txt" && [ ! -s "$W/gone.txt" ] && echo "DELTA OK: exactly the 7, none gone"
```

**Stop rule.** If `added.txt` has any name that is not one of the 7, open that test's traceback in `run.txt`. If
the root cause is `ModuleNotFoundError: No module named 'dotenv'`, it is another masked name. W1 measured pairs of
files, not the full suite, so this can happen. Add it to the table above and keep going. **Anything else means a
real regression. Roll back.** If `gone.txt` is not empty, something unrelated changed. Roll back and investigate.
The optional check `bash scripts/ab_check.sh` should exit **1** here and list the same 7 as NEW (persistent).

## Step 3 — regenerate `tests/baseline_failures.txt` (keep the header)

The command from the file's own header is `PY_COLORS=0 python3 -m pytest tests/ --continue-on-collection-errors -q
2>/dev/null | grep -E '^(FAILED|ERROR)' | sed 's/ - .*//' | sort -u`. Step 2 already ran exactly that pipeline into
`$W/names.txt`, so reuse it rather than running the suite again. **Keep the `#` header block.** `CLAUDE.md` calls
it the "3-line header", but the file currently has **8** `#` lines. Keep all 8 and change one fact: line 4's
`dotenv (1) -> test_asof_universe` becomes the 8-name attribution.

```bash
{ grep '^#' tests/baseline_failures.txt; cat "$W/names.txt"; } > "$W/new_baseline.txt"
python3 - "$W/new_baseline.txt" <<'EOF'
import sys; p = sys.argv[1]; s = open(p).read()
old = "dotenv (1) -> test_asof_universe"
new = ("dotenv (8) -> test_asof_universe, test_new_modules::TestKelly + ::TestKellyScoping, "
       "test_gpu_lock::test_choose_inference_device_always_cpu, test_wave4::TestWarmupFill")
assert s.count(old) == 1, "header line 4 not found; edit it by hand"
open(p, "w").write(s.replace(old, new))
EOF
cp "$W/new_baseline.txt" tests/baseline_failures.txt
grep -c '^#' tests/baseline_failures.txt; grep -vc '^#' tests/baseline_failures.txt   # expect 8 and 30
```

## Step 4 — the gate

```bash
bash scripts/ab_check.sh; echo "exit=$?"     # MUST be 0: no NEW names, none DISAPPEARED
```

## Step 5 — update the docs that restate the name set (same change)

- **`CLAUDE.md` § Running tests is the only home of the suite count.** Replace the `Current baseline (verified
  2026-09-08): 16 failed, 3833 passed, 25 skipped, 7 errors` line with the counts line recorded in step 2 and
  today's date. On the next line, `23-name` becomes the new name count. In the dependency list, `dotenv (1)`
  becomes `dotenv (8)`. Further down, the prose "23 name lines" and "23-name baseline" becomes the new count, and
  "3-line header" becomes "8-line header".
- `tests/README.md` § The dev-Mac baseline: change `23 names (7 ERROR, 16 FAILED) across 10 files` to the new
  figures. The `dotenv` table row becomes 8 names (add `test_new_modules::TestKelly` (3) + `::TestKellyScoping`
  (2), `test_gpu_lock` (1) and `test_wave4::TestWarmupFill` (1), with the import chains from the table above).
  Mark item 1 of § Known hygiene items as **resolved (date, commit)**.
- `docs/MAP.md:886` says "the 23-name baseline". Update the count there too. The dated historical records
  (`research/cleanup_2026-09/README.md`, `research/campaign_2026-08/0{3,4}_*.md`) describe the past. Leave them.

Suggested commit, only when the owner asks: `test: scope the c26_T2/T3 dotenv stub; baseline +7 masked dotenv names`.

## Rollback (at any step)

```bash
git checkout -- tests/test_c26_T2.py tests/test_c26_T3.py tests/baseline_failures.txt
cp "$W/CLAUDE.md" CLAUDE.md; cp "$W/README.md" tests/README.md    # these carry uncommitted work: restore from backup, NOT git checkout
bash scripts/ab_check.sh; echo "exit=$?"                          # back to 0 on the old baseline
```

`git checkout` is safe for the three test/baseline files only because step 0 confirmed they were clean at `HEAD`.
