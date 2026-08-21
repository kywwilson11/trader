---
name: panel-improve
description: Run a deep panel-review improvement pass with the module-improve-v3 workflow (N independent Opus reviewers per module on an identical brief -> Fable adjudicates into a verified spec -> Sonnet implements -> Fable hardens -> serialized ab_check gate). Use when the user wants bugs AND improvements AND gaps found on specific modules, with consensus-weighted confidence. For broad cheap sweeps use /improve (group-improve-v2) instead.
---

# /panel-improve — panel-review improvement campaigns

Orchestration: `.claude/workflows/module-improve-v3.js` (invoke as the `module-improve-v3` workflow).
This skill is the operating manual. Template config: `.claude/workflows/modules-v3.example.json`.

## Why this exists (vs `/improve`)

`group-improve-v2` is the cheap broad sweep: **one** designer per group, tiered by risk. It is the
right tool for "clean up 12 groups of modules."

`module-improve-v3` is the deep pass: **N independent reviewers per module, all given the identical
brief**, then adjudicated. It costs ~2-3x more per module and buys three things v2 cannot:

1. **A consensus signal.** Independent samples of the same question. A finding 3 of 3 reviewers reach
   is strong evidence; a finding 1 of 3 reaches with an airtight quoted proof still stands. The
   synthesizer weighs the code, never the vote — consensus is an input to judgment, not a verdict.
2. **Breadth beyond bug-hunting.** The brief demands three classes: **bugs** (with a concrete failure
   scenario), **improvements** (efficiency on the Jetson hot path, structure, dead code, doc rot), and
   **gaps** (what the module *should* have and does not — validation, edge cases, untested branches,
   instrumentation the rest of the system expects, contract holes with callers).
3. **A hardening pass with real license.** The final Fable stage does not merely verify spec
   conformance — it owns the code: fixes bugs it finds while reading, improves efficiency, and
   modernizes idioms *where it matches surrounding style and changes no behavior*.

## Pipeline (per module)

| Phase | Model | Does |
|---|---|---|
| Context | Sonnet | Factual pack: purpose, public API, real caller sites, existing tests (Mac-passing vs Jetson-gated), ledger entries, kill-list hits, hard constraints. Facts only — no opinions, so it cannot correlate reviewer judgment. |
| Review | **Opus x N** | N independent reviewers, **identical brief**, no cross-talk. Bugs + improvements + gaps, each with file:line, a verbatim snippet as proof, a concrete proposal, and a **classification**. |
| Spec | **Fable** | Adjudicates: merges duplicates (records consensus), **re-verifies every anchor against the current file** (reviewers quote from memory and drift), resolves conflicts, gates by classification, orders the work, and emits the implementer-facing spec + evidence pack. |
| Implement | **Sonnet** | Mechanical. Follows the spec verbatim from the evidence pack; opens files only where it edits. Adds tests; self-verifies. |
| Harden | **Fable** | Reads the diff. Duty A correctness, duty B quality (bugs, perf, modern idiom). May edit; may request ONE guided repair round; may revert. |
| Gate | Sonnet | **One serialized** `scripts/ab_check.sh` after all modules — concurrent full-suite runs on a shared tree produce phantom failures. One guided gate-repair (Fable) if it fails. |
| Report | Sonnet | Assembles the campaign report + the owner decision queue. |

## Config contract — the FILE is the primary channel

**The host does not reliably deliver `args` to a workflow invoked by name** (it arrives `undefined`,
which is indistinguishable from "no config" — the run then exits having spawned nothing). So the
config lives in a committed JSON file that the workflow loads itself, and `args` is only an optional
override. Edit the file, then invoke the workflow with no args at all.

Default path: **`.claude/workflows/modules-v3.run.json`** (override with `args.modules_path`).

```json
{ "campaign_title": "...", "baseline_note": "...", "tree_note": "...",
  "reviewers": 5, "workers": 3, "report_path": "research/module_improve_v3_report.md",
  "effort_opus": "max", "effort_fable": "xhigh", "effort_sonnet": "max",
  "modules": [ { "id": "...", "mods": "a.py", "test": "tests/test_a_v3.py", "seed": "..." } ] }
```

- `modules` — **required**. Bare strings (`"indicators.py"` — id and test derived) or objects
  `{id, mods, test, seed, reviewers}`. `mods` may list several space-separated files.
- `reviewers` — panel size per module, clamped 2-5 (default **3**); per-module `reviewers` overrides.
- `workers` — concurrent module chains (default **2**). Peak concurrency ~ `workers x reviewers`.
- `effort_opus` / `effort_fable` / `effort_sonnet` — reasoning effort per model family
  (defaults `max` / `xhigh` / `max`). Fable sits one tier down deliberately: it adjudicates a whole
  panel and hardens a diff, work where over-deliberation costs more than it buys.
- `report_path`, `repo`, `context_pack: false` (skip the Context phase — reviewers read cold).

Resolution order for every tunable: explicit `args` → config file → built-in default.
`.run.json` is the working config; `.example.json` is the template — keep the template intact.

**Agent budget:** `modules x (reviewers + 4) + 2`. 3 modules x 3 reviewers = 23 agents. The workflow
logs the plan up front and stops starting new modules when the turn's token target runs low,
`log()`ing exactly which modules were dropped — it never silently truncates coverage.

## Rules baked into the prompts (never relax them)

- **MODEL-FACING never ships** from this pipeline — feature values, thresholds, entry/exit/sizing
  semantics, label construction, model artifacts. Reviewers classify; Fable gates; the owner decides.
  Misclassifying model-facing as behavior-neutral is called out as the worst possible error.
- **`research/KILL_LIST.md` is checked before any strategy/feature/data-source proposal** — a killed
  item is not a gap.
- `policy_exits.exit_walk` is sacred; heavy modules are READ, never imported on the Mac.
- **New test files must `pytest.importorskip` heavy deps at module level** so they SKIP, not ERROR —
  a collection error is a NEW failure name and fails the gate.
- Strict file ownership: a module's chain edits only its `mods` + its `test`. Cross-file needs are
  reported, never implemented. Never revert or stash work the campaign does not own.
- Nothing is committed. Ever.

## Before launching

1. **Check tree state with the user** and write an honest `tree_note` — if unrelated uncommitted work
   is in flight, name those files and tell the agents to ignore them.
2. Build seeds from `research/module_review_2026-07.json` + `session-state`; do not recycle stale seeds.
3. Prefer targets with a real *consumer contract* (something parses their output) or a hot path — that
   is where a panel earns its cost. A module nothing depends on rarely repays 3 reviewers.
4. After completion: read the report's **OWNER DECISIONS** section to the user, run `/regression-ab`
   independently if the gate looked marginal, and hand over the diff for review.

## Reading the result

An **empty spec is a success**, not a failure: it means the panel read the module and found nothing
shippable. The report's per-module row shows `raw findings` vs `accepted` — a large gap there usually
means the module is healthy but model-facing-constrained, and the value landed in owner decisions.
