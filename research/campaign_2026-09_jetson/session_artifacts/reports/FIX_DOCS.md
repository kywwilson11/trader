# FIX_DOCS — doc reconciliation for the 2026-09-26 Jetson campaign

I edited only the files I own: docs/{FLAGS,MODULES,MAP,STATE_FILES,GLOSSARY}.md, CLAUDE.md, README.md,
archive/README.md, the llm_config.py docstring, and the 03_jetson_runbook.md annotations. There were no code
changes. Nothing was staged or committed.

Every edit is an exact-substring replacement, checked for exactly one match. Backups of the pre-edit
state are in `scratchpad/docs_fix/*.orig` (FLAGS, MODULES, MAP). Other agents' hunks were preserved.

## Edits (file:line → what; the code line I verified is in brackets)

### docs/FLAGS.md
- **:6-9** Header note. It records that on 2026-09-26 the rows below and the setup-script and SHADOW_MODE cites were
  re-verified, and that other cites were not re-swept. run_pipeline.py shifted +19…+24 lines below :288
  [HEAD BOT_ENV :291 → now :310; SHADOW_MODE :1016 → :1040].
- **:14** Philosophy count changed from 81/68 to **82/69**, because `indicators._HAS_C` is a new OFF flag.
- **:39** Parsing table: `TRADER_INDICATORS_C` added to the strict-`'1'` row [indicators.py:15 `== '1'`].
- **:41** literal-'0' row: `run_pipeline.py:1016` → `:1040` [run_pipeline.py:1040].
- **:131, :160, :166** The env-flag table goes from (26) to **(27)**, with a new row `indicators._HAS_C`: default False, ops,
  "LIVE·inert until the archived .so is fixed". The note below the table changes from 26 to 27 rows
  [indicators.py:14-20, readers :325/341/358/377].
- **:424** §5 intro: 56 → **58** read patterns, 32 → **34** `TRADER_*` (35 concrete), 24 → **25** flags, 8 → **9** settings.
- **:459** New row **`TRADER_INDICATORS_C`**: default unset/OFF (numba > pure), ops, not model-facing (≤1e-12 parity).
  It says to enable only after the C source is fixed, and that the env parse has no test
  [indicators.py:15; tests only monkeypatch `_HAS_C`].
- **:460** New row **`TRADER_PYBIN`**: install-time only. Default is the jetson python (= run_pipeline.PYTHON :44).
  Step 0 exits 2 unless the interpreter imports pyarrow/dotenv/torch. Pass it as `sudo TRADER_PYBIN=… bash …`
  [setup_jetson_system.sh:25-26,51,56-66,197; test_jetson_ops_2026_09.py:119,129].
- **:455** `TRADER_SHADOW_MODE` row:
  - cite `:1016` → `:1040`, and gui `:7489` → `:7578` [gui.py:7578].
  - The stale default text "UNSET==OFF at gui" now reads "'1' (ON) in both readers (§3…)". §3 records that the
    gui chip was fixed on 2026-09-08.
- **:478** `CUDA_VISIBLE_DEVICES`:
  - adds `_training_env` dropping an inherited '' for training while BOT_ENV sets '', and notes that the unit no longer sets it;
  - readers corrected to run_bots.py:45 and run_pipeline.py:300/:310; test_jetson_ops_2026_09 added.
- **:479/:480** `TORCH_NUM_THREADS`/`OMP_NUM_THREADS` cites: run_bots.py :46/:47 and run_pipeline.py :310 [verified].
- **:481** `LD_LIBRARY_PATH`: gui cite `:108` → `:119`, plus a note that `_engine_env` now drops empty elements [gui.py:117-121].
- **:494** `SUDO_USER`: `:155` → `:184` [setup_jetson_system.sh:184].
- **:497** "Not env vars" note:
  - `:154-155` → `:183-184`;
  - adds that `PYBIN`/`UNIT_LD_*` (`:51-55`) are shell locals and only `TRADER_PYBIN` is read from the environment.
- **:552** §7 dynamic-read cite: `run_pipeline.py:1020` → `:1043-1044` [getattr lines].
- **Not added:** a beta_ledger CLI row. FLAGS.md has no CLI section. MODULES.md Appendix A owns the CLI flags, and FIX_D already updated it.

### docs/MODULES.md
- **:266** "three dispatch backends (C ext > numba > pure)" → "two (numba > pure; the archived C ext is tried first only with
  `TRADER_INDICATORS_C=1`)".
- **:194** market_data Known issues: FIXED 2026-09-26, the SIP end-clamp. It cites `_clamp_sip_end` and `SIP_RECENT_DELAY_MIN=16`, notes the
  rebuild-only timing, and names the test [market_data.py:548-563].
- **:482** base_loop Known issues: new **OWNER-decision D13**. A veto-strike survives omission from a partial response, so
  veto → omitted → veto liquidates [base_loop.py:1696-1711]. The narrow and broad fixes are described; no code change.
- **:778** llm_client Goal: Claude "forced tool use" gains a caveat. On Opus ≥5.5 / Fable ≥5.1 the path is auto + strict tool +
  client-side validation with one retry [llm_client.py:1356 `_anthropic_accepts_forced_tool`, :1383, :1544].
- **:786** llm_client Known issues:
  - "gpt-5.4* placeholders" becomes FIXED 2026-09-26, covering the price rows, the family-ceiling fallback, RPD rows, the locked
    rollover and discarded-billing charges [llm_client.py:271-285,295,810,942];
  - OPEN: KNOWN_MODELS lacks the new ids, and RPD is not incremented for discarded-but-billed responses.
- **:800** llm_eval Known issues: the SIP-delay sibling bug from FIX_C is recorded as OBJECTIVE-fix-pending, moot while no bots run
  [llm_eval.py:132-135,218].
- **:804** sentiment_history Goal: stock key `(t_utc−6h).date()−1` (harvest) vs host-local `date.today()−1` (serve); crypto F&G is
  keyed on its UTC publication date [harvest_stock_data.py:540,551; sentiment_history.py:639].
- **:807** Reads/Writes: adds the table `fng_daily_legacy_localtz` and the `state` keys `fng_date_basis`/`_migrated_at` [sentiment_history.py:196-198].
- **:808** Known issues:
  - FIXED 2026-09-26: the F&G local-tz +1-day look-ahead and the `limit=total_days` fetch. It is model-facing and takes effect at the next harvest.
  - OWNER: the serve key depends on the host timezone.
  - OWNER: the articles cache mixes Chicago- and UTC-dated rows.
- **:845** gui Known issues:
  - The three 2026-09-08 fixes (SHADOW polarity, Lead/Lag path, QAction) are now marked RESOLVED. I re-verified them: there is no
    `crypto_training_data`, "orders suppressed" or `QAction` in gui.py, and :7578 is correct.
  - FIXED 2026-09-26: the FIX_F items D1/D2/D3/D4-5/D10 [gui.py:58,125,4530].
  - OPEN: the Trading/Cockpit squeeze, the Markets first-visit "Loading…", and the ~20 req/min orders walk (owner decides the cadence).
  - Salander is still pending.
- **:869** ops Goal: the "no RTC battery" chrony premise is marked unverified (rtc0/rtc1 exist).
- **:888** hw_monitor: "setup script still says wait_for_cool_gpu" → fixed 2026-09-26 [setup_jetson_system.sh:246].

### docs/MAP.md
- **:341** Process tree: trader.service is marked "NOT installed on the prod box as of 2026-09-26".
- **:382-387** Harvest labels: an annotation on the F&G local-tz one-day look-ahead in every store built so far. The repair and the stock key take
  effect at the next harvest; the paragraph points to MODULES.
- **:510** Bot process modes: "Fixed 2026-09-26: a live 'Bots' process ORs `_BOT_SCOPE`" [run_pipeline.py:776-784].
- **:747-752** §5-8 invariant: "every other gate is fail-closed" is corrected. The meta-label gate is fail-open/neutral without
  artifacts by documented intent, as are the stock event checks [meta_label.py:519-544,601-603; base_loop.py:2064-2068; gate 15 at
  §4e]. I did not verify the B_serving `MAP.md:558` cite separately; that row is gate 15.
- **:788-791** §5-12: "learned lexicon over the static one" is replaced. `sentiment.py` uses the static hand-built lexicon, and
  `learned_lexicon.py` is dark: it writes `learned_lexicon.json`, which nothing reads and which does not exist on prod
  [grep: only learned_lexicon.py and scripts/train_lexicon.py mention it; learned_lexicon.py:10-12].
- **:845-847** §6 invariant 3: adds the meta-label exception (absent artifacts ⇒ neutral pass).
- **:892-893** §7: "installed pyarrow version" is now answered: 23.0.1 [`$JPY -c 'import pyarrow'`].
- **:904** §8: annotation that the "Uncommitted, awaiting owner review" paragraph (R2/R3/IA/cleanup) was committed on 2026-09-10 as
  `438f56a` [git show --stat 438f56a].
- **:934-941** §8: a new 8-line paragraph, **"2026-09-26 Jetson campaign"**. It marks the work uncommitted, lists the fixes, and says no
  Phase-2/3 flag was flipped. It points to `research/campaign_2026-09_jetson/README.md`.
- **:975** §9 row 22: "— **fixed 2026-09-26** …", tag "JETSON-FIX (done)".
- **:1055, :1077** The where-is table and the document map each gain a row for `research/campaign_2026-09_jetson/README.md`.

### docs/STATE_FILES.md
- **:269** §11 is retitled "what moved into `archive/` (dev Mac 2026-09-08, Jetson 2026-09-26)". Nothing else referenced the old title.
- **:299-302** The Jetson paragraph gains the `archive/c_ext/` move and a note that it is opt-in via `TRADER_INDICATORS_C`.
- **:215** §7 `sentiment_cache.db` row: it gains the `fng_daily_legacy_localtz` table and the `state.fng_date_basis` key on the first crypto F&G fetch.

### CLAUDE.md
- **:31** Repository map: `research/campaign_2026-09_jetson/README.md` added to the per-directory READMEs.
- **:91** "Jetson/CI stay green" → "must never be ported to the Jetson or CI — their current state is the verified-2026-09-26
  sentence in the baseline paragraph above, the only place it is quoted". The counts stay only in § Running tests.
- **:189-191** LLM line: the Claude forced-tool-use caveat, pointing to MODULES §llm_client.
- **:239** Conventions: annotation that the "Since 20a41db, UNCOMMITTED" R2/R3/IA work was committed on 2026-09-10 as `438f56a`.
  What is uncommitted now is the 2026-09-26 campaign (see MAP §8).

### README.md
- **:160** Jetson setup block: a comment on the `sudo TRADER_PYBIN=… bash …` override.
- **:193** "the suite is green on the full Jetson stack" → "the same section records the full-Jetson-stack state".

### docs/GLOSSARY.md
- **:70** baseline_failures.txt: "…Jetson or CI, where the suite is green" → "(their current suite state: CLAUDE.md § Running tests)".

### archive/README.md
- **:20** c_ext row, last sentence:
  - the `generate_manual.py` path is now `archive/jetson_residue_2026-03/`;
  - the grep hit list is completed (it adds tests/test_indicators_parity.py and docs/graphs/import_graph.json). Re-grepped.
- **:29-37** The "All three local_residue patterns are depth-agnostic" paragraph is rewritten so that it is true:
  - the original rows' patterns are depth-agnostic (no slash);
  - the two 2026-09-26 local_residue rows use directory-anchored patterns (`.gitignore:162-163`, verified);
  - `c_ext/` and `jetson_residue_2026-03/` are untracked and deliberately not ignored.

### llm_config.py (docstring only, :103-107)
- "bills at llm_client's $1.25/$10 fallback" → "bills at its provider family's most expensive tabled row, per
  llm_client._fallback_price" [llm_client.py:295-313]. `py_compile` OK.

### research/campaign_2026-08/03_jetson_runbook.md (annotations only)
- **:25-28** Phase 0 step 2 annotation:
  - bidask was installed on 2026-09-26 (`pip install bidask==2.1.0`) before any harvest [`pip show bidask` → 2.1.0; `import bidask` OK];
  - neither pre-rebuild store carried `Eff_Spread_Pct`, so this is not a mid-stream change.
- **:129-134** Phase 3 "Data-store event" annotation:
  - the 2026-09-26 clean rebuild runs with `TRADER_RAW_SIDECAR=1 TRADER_YF_WINDOW_SLICE=1`, the PIT sentiment repair and the SIP
    end-clamp, with defaults otherwise;
  - the Phase-3 model flags were NOT flipped, and that is an owner decision pending;
  - it points to the campaign README.
- Nothing else in the runbook was changed. The only other diff hunk, at :48, is FIX_D's.

## Verification
- `git diff --stat` over my files: 10 files, +142/−62. This includes other agents' earlier hunks in CLAUDE.md, MAP, MODULES,
  GLOSSARY, STATE_FILES, archive/README and the runbook.
- grep over docs/, CLAUDE.md, README.md, archive/README.md, the runbook and llm_config.py:
  - **0 hits** for "C ext > numba", "Jetson/CI stay green" and "learned lexicon over the static";
  - "$1.25/$10" is gone from llm_config.py.
  - "n ≥ 60" survives only in FIX_D's negations: MAP.md:613 "n ≥ 60 alone is not enough" and MODULES.md:778 "n ≥ 60 alone
    abstains". Neither claims the verdict at n≥60.
- `$JPY -m pytest tests/test_imports_v3.py tests/test_repo_graph.py -q -p no:cacheprovider` → **13 passed, 2 skipped**.
- The other tests that read doc/repo text were also run: test_cscv_gate, test_decision_report(_v3), test_grp_training,
  test_imports, test_policy_exits_v3, test_shadow_status_persist, test_indicators_parity and test_llm_config →
  **172 passed, 1 skipped, 3 xfailed** (the pre-existing documented xfails).

## Not done / outside my files
- **"Suite is green on Jetson/CI" still appears in files I don't own:**
  - `tests/README.md:58`;
  - `research/AGENT_CONTEXT.md:72`;
  - runbook Phase 0 step 3, "expect green", which I was not allowed to rewrite.
  All three should point at CLAUDE.md § Running tests.
- **The training stores are not in the repo root now** (moved aside for the rebuild). The "no Eff_Spread_Pct" claim therefore
  rests on the brief's verified device facts, not a fresh schema read.
- **FIX_D's "left open" items are code-level, not doc items, and were not recorded:**
  - the sizing_cofire header;
  - execution_report's silent shortfall omission;
  - rank_gradient_report ignoring `representative`;
  - no producer for reliability_report's input.
- **Also not recorded:** FIX_E's P2 audit items, and the `shadow.py:875` `sys.executable` note.
- **FLAGS.md's other line cites were not re-swept**, as the header says. That includes the gui.py:74xx notify cites and the
  base_loop/order_utils cites.
