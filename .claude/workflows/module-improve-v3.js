// module-improve-v3 — PANEL REVIEW pipeline (successor shape to group-improve-v2).
//
//   Context pack (Sonnet, factual)
//     -> N INDEPENDENT Opus reviewers, identical brief, no cross-talk   [bugs + improvements + gaps]
//     -> Fable adjudicates the panel into a verified, anchored design spec
//     -> Sonnet implements the spec + tests (mechanical)
//     -> Fable hardens the final code (bugs, perf, modern practice) with one optional repair loop
//     -> ONE serialized ab_check gate (+ one guided gate repair)
//     -> Haiku assembles the campaign report + owner decision queue
//
// Why a panel: reviewers get the SAME brief on purpose. Independent samples of the same question
// give a consensus signal the synthesizer can weigh — a finding three reviewers reach alone is
// strong evidence; a lone finding with an airtight quoted proof still stands. Fable judges the
// code, not the vote.
//
// Config: args.modules (inline) or args.modules_path (JSON file).
// Template: .claude/workflows/modules-v3.example.json   Manual: .claude/skills/panel-improve/SKILL.md
export const meta = {
  name: 'module-improve-v3',
  description: 'Panel review: N independent Opus reviewers per module (identical brief) -> Fable design spec -> Sonnet implement -> Fable harden -> serialized regression gate',
  whenToUse: 'Deep improvement pass on specific modules when you want more than bug-hunting: consensus-weighted bugs + improvements + gaps, a Fable-verified spec, and a Fable-hardened final diff. Heavier and slower than group-improve-v2; use that one for broad cheap sweeps.',
  phases: [
    { title: 'Context', detail: 'factual pack: API, callers, tests, ledger, kill-list', model: 'sonnet' },
    { title: 'Review', detail: 'N independent reviewers, identical brief', model: 'opus' },
    { title: 'Spec', detail: 'adjudicate panel -> verified anchored spec', model: 'fable' },
    { title: 'Implement', detail: 'write the code + tests from the spec', model: 'sonnet' },
    { title: 'Harden', detail: 'final review: bugs, perf, modern practice', model: 'fable' },
    { title: 'Gate', detail: 'one serialized full-suite ab_check' },
    { title: 'Report', detail: 'campaign report + owner decision queue', model: 'sonnet' },
  ],
}

// ---------------------------------------------------------------- config ----
// `args` can arrive as a JSON-encoded STRING instead of an object. Untreated, a
// stringified payload is indistinguishable from "no config" and the run exits
// having done nothing, so normalize once here and read everything through A.
let A = args
if (typeof A === 'string') {
  try { A = JSON.parse(A) } catch (err) {
    log(`module-improve-v3: args arrived as a string that is not valid JSON — ${err.message}`)
    A = null
  }
}
if (A && typeof A.modules === 'string') {
  try { A.modules = JSON.parse(A.modules) } catch (err) { /* validated below */ }
}

const REPO = (A && A.repo) || '/Users/kywwilson/Desktop/Projects/trader'
const BUDGET_RESERVE = 80000  // stop starting new modules below this many output tokens

// The host does not always deliver `args` to a named workflow, so the config FILE is the
// primary channel and args are an optional override. Everything tunable therefore lives in
// the JSON: modules, panel size, workers, report path, per-family effort.
const CFG_PATH = (A && A.modules_path) || '.claude/workflows/modules-v3.run.json'

const CFG_SCHEMA = {type: 'object', properties: {
  campaign_title: {type: 'string'}, baseline_note: {type: 'string'}, tree_note: {type: 'string'},
  reviewers: {type: 'integer'}, workers: {type: 'integer'}, report_path: {type: 'string'},
  effort_opus: {type: 'string'}, effort_fable: {type: 'string'}, effort_sonnet: {type: 'string'},
  modules: {type: 'array', items: {type: 'object', properties: {
    id: {type: 'string'}, mods: {type: 'string'}, test: {type: 'string'}, seed: {type: 'string'},
    reviewers: {type: 'integer'}},
    required: ['mods']}}},
  required: ['modules']}

let cfg = null
if (A && Array.isArray(A.modules)) cfg = A
else {
  cfg = await agent(
    `Read the JSON file ${REPO}/${CFG_PATH} and return its contents verbatim as the structured ` +
    `output. Copy every field exactly as written — the seed strings and notes are prompts for ` +
    `later agents, so do not summarize, truncate, or reword them. Read-only; edit nothing.`,
    {label: 'load-config', phase: 'Context', model: 'sonnet', effort: 'max', schema: CFG_SCHEMA})
}
if (!cfg || !Array.isArray(cfg.modules) || cfg.modules.length === 0) {
  log(`module-improve-v3: no modules. Expected a config at ${REPO}/${CFG_PATH} ` +
      `(shape: {campaign_title, baseline_note, tree_note, reviewers, workers, modules:[{id,mods,test,seed}]}), ` +
      `or pass args.modules inline. Template: .claude/workflows/modules-v3.example.json`)
  return {error: 'missing modules config'}
}

// Tunables: explicit args win, then the config file, then the built-in default.
const pick = (k, d) => (A && A[k] !== undefined) ? A[k]
  : ((cfg && cfg[k] !== undefined && cfg[k] !== null) ? cfg[k] : d)
const REVIEWERS = Math.max(2, Math.min(5, pick('reviewers', 3)))
const N_WORKERS = Math.max(1, pick('workers', 2))
const REPORT_PATH = pick('report_path', 'research/module_improve_v3_report.md')
const WANT_PACK = pick('context_pack', true) !== false

// Reasoning effort per model family. Reviewers and implementers run at max; Fable's
// adjudication/hardening runs one tier down at xhigh, which is where it reasons best over a
// large panel without over-deliberating.
const E_OPUS = pick('effort_opus', 'max')
const E_FABLE = pick('effort_fable', 'xhigh')
const E_SONNET = pick('effort_sonnet', 'max')

// Accept bare strings ("indicators.py") or full objects; derive id/test when omitted.
function norm(m) {
  const spec = (typeof m === 'string') ? {mods: m} : m
  const mods = String(spec.mods || '').trim()
  const first = mods.split(/\s+/)[0] || 'module'
  const base = first.split('/').pop().replace(/\.py$/, '')
  return {
    id: spec.id || base,
    mods,
    test: spec.test || `tests/test_${base}_v3.py`,
    seed: spec.seed || '',
    reviewers: Math.max(2, Math.min(5, spec.reviewers || REVIEWERS)),
  }
}
const MODULES = cfg.modules.map(norm).filter(m => m.mods)

const BASELINE_NOTE = cfg.baseline_note || (A && A.baseline_note) ||
  `see ${REPO}/CLAUDE.md "Running tests" for the current pre-existing failure baseline; ` +
  `scripts/ab_check.sh is the authority (it diffs failure NAMES, never counts)`
const TREE_NOTE = cfg.tree_note || (A && A.tree_note) ||
  'The tree may carry unrelated uncommitted work from other sessions — never revert, stash, or ' +
  'clean anything you do not own, and ignore unrelated dirty state when judging your diff.'

const CTX = `Repo: ${REPO} — autonomous Alpaca PAPER-trading system (RegressionLSTM+LightGBM blend per book, hourly bars, crypto 24/7 + US stocks), production on a Jetson Orin Nano 8GB shared with the live bots. Read ${REPO}/research/AGENT_CONTEXT.md if you need the full brief.

TWO-MACHINE REALITY: this dev Mac has numpy/pandas/scipy/yfinance/bidask/requests/pytest ONLY. NO torch, lightgbm, optuna, joblib, numba, sklearn, dotenv, alpaca, finnhub, PySide6. Heavy modules are READ, never imported here (numba code has pure-python fallbacks). Anything needing training/real journals/parquet/GUI rendering is Jetson-gated — flag it, do not attempt it.

NON-NEGOTIABLE CONVENTIONS:
- Train/serve parity is sacred; policy_exits.exit_walk is the one shared exit kernel — never fork it.
- MODEL-FACING changes (feature VALUES, thresholds, entry/exit/sizing semantics, label construction, model artifacts) NEVER ship from this pipeline. They become owner decisions with a fix sketch.
- Instrumentation / measurement / behavior-neutral refactors / unambiguous-bug fixes MAY ship directly.
- Fail-closed in live trading paths; fail-open for the LLM gate (an LLM error can never block a trade).
- PIT discipline: features strictly trailing, sentiment lagged to publication, no survivorship.
- Check ${REPO}/research/KILL_LIST.md before proposing ANY strategy/feature/data-source idea — a killed item is not a gap, and it leaves that list only by owner decision.
- Every module already had a function-by-function review: ${REPO}/research/module_review_2026-07.json. Read your module's entries; do not re-derive what is already recorded.
- ${TREE_NOTE}
- DO NOT COMMIT, ever. The owner reviews and commits.

TESTS: suite baseline — ${BASELINE_NOTE}. Regressions are judged by NEW failure NAMES in the files you own. A NEW test file that cannot import its heavy deps must use module-level pytest.importorskip so it SKIPS rather than ERRORS — a collection error is a new name and fails the gate. Never weaken, skip, or delete an existing test to make your change pass.`

// ---------------------------------------------------------------- schemas ----
const PACK_SCHEMA = {type: 'object', properties: {
  purpose: {type: 'string'},
  public_api: {type: 'array', items: {type: 'string'}},
  callers: {type: 'array', items: {type: 'string'}},
  existing_tests: {type: 'array', items: {type: 'string'}},
  ledger: {type: 'array', items: {type: 'string'}},
  killlist: {type: 'array', items: {type: 'string'}},
  constraints: {type: 'array', items: {type: 'string'}},
  notes: {type: 'string'}},
  required: ['purpose', 'public_api', 'callers', 'existing_tests', 'ledger', 'killlist', 'constraints', 'notes']}

const REVIEW_SCHEMA = {type: 'object', properties: {
  module_summary: {type: 'string'},
  findings: {type: 'array', items: {type: 'object', properties: {
    kind: {type: 'string', enum: ['bug', 'improvement', 'gap']},
    severity: {type: 'string', enum: ['P0', 'P1', 'P2', 'P3']},
    title: {type: 'string'},
    file: {type: 'string'},
    lines: {type: 'string'},
    evidence: {type: 'string'},
    failure_or_cost: {type: 'string'},
    proposal: {type: 'string'},
    classification: {type: 'string', enum: ['instrumentation', 'behavior-neutral', 'model-facing', 'policy-semantics', 'unclear']},
    confidence: {type: 'string', enum: ['high', 'medium', 'low']},
    test_hint: {type: 'string'}},
    required: ['kind', 'severity', 'title', 'file', 'lines', 'evidence', 'failure_or_cost', 'proposal', 'classification', 'confidence', 'test_hint']}},
  clean_verdict: {type: 'string'}},
  required: ['module_summary', 'findings', 'clean_verdict']}

const SPEC_SCHEMA = {type: 'object', properties: {
  accepted: {type: 'array', items: {type: 'object', properties: {
    title: {type: 'string'}, consensus: {type: 'integer'},
    classification: {type: 'string'}, why_it_survives: {type: 'string'}},
    required: ['title', 'consensus', 'classification', 'why_it_survives']}},
  rejected: {type: 'array', items: {type: 'string'}},
  deferred_to_owner: {type: 'array', items: {type: 'string'}},
  conflicts_resolved: {type: 'array', items: {type: 'string'}},
  spec: {type: 'string'},
  evidence: {type: 'array', items: {type: 'object', properties: {
    file: {type: 'string'}, lines: {type: 'string'}, snippet: {type: 'string'}, why: {type: 'string'}},
    required: ['file', 'lines', 'snippet', 'why']}}},
  required: ['accepted', 'rejected', 'deferred_to_owner', 'conflicts_resolved', 'spec', 'evidence']}

const IMPL_SCHEMA = {type: 'object', properties: {
  fixes_applied: {type: 'array', items: {type: 'string'}},
  deferred: {type: 'array', items: {type: 'string'}},
  files_touched: {type: 'array', items: {type: 'string'}},
  tests_added: {type: 'integer'},
  test_result: {type: 'string'}},
  required: ['fixes_applied', 'deferred', 'files_touched', 'tests_added', 'test_result']}

const HARDEN_SCHEMA = {type: 'object', properties: {
  verdict: {type: 'string', enum: ['approved', 'approved-after-hardening', 'needs-repair', 'rejected-reverted']},
  bugs_fixed: {type: 'array', items: {type: 'string'}},
  perf_improved: {type: 'array', items: {type: 'string'}},
  modernized: {type: 'array', items: {type: 'string'}},
  fix_notes: {type: 'string'},
  test_result: {type: 'string'},
  notes: {type: 'string'},
  commit_line: {type: 'string'}},
  required: ['verdict', 'bugs_fixed', 'perf_improved', 'modernized', 'fix_notes', 'test_result', 'notes', 'commit_line']}

const GATE_SCHEMA = {type: 'object', properties: {
  passed: {type: 'boolean'},
  new_failures: {type: 'array', items: {type: 'string'}},
  disappeared: {type: 'array', items: {type: 'string'}},
  summary: {type: 'string'}},
  required: ['passed', 'new_failures', 'disappeared', 'summary']}

// ---------------------------------------------------------------- prompts ----
function renderPack(p) {
  if (!p) return '(no context pack — read what you need yourself)'
  return [
    `purpose: ${p.purpose}`,
    `public API: ${(p.public_api || []).join(' | ')}`,
    `callers: ${(p.callers || []).join(' | ')}`,
    `existing tests: ${(p.existing_tests || []).join(' | ')}`,
    `prior review ledger: ${(p.ledger || []).join(' ;; ')}`,
    `kill-list entries touching this area: ${(p.killlist || []).join(' ;; ')}`,
    `hard constraints: ${(p.constraints || []).join(' ;; ')}`,
    `notes: ${p.notes || ''}`,
  ].join('\n')
}

function renderEvidence(evidence) {
  return (evidence || []).map(e => `--- ${e.file} (lines ${e.lines}) — ${e.why}\n${e.snippet}`).join('\n\n')
}

function packPrompt(g) {
  return `${CTX}

CONTEXT PACK (read-only, facts only — no opinions, no proposals) — module(s): ${g.mods}
Assemble the scaffolding that ${g.reviewers} independent reviewers will each be handed:
1. purpose — what this module is for, in two lines, from the code and its docstring.
2. public_api — the functions/classes other modules actually import (grep the repo for real import sites, do not guess from __all__).
3. callers — "file.py:line calls fn()" for every real consumer; this is how reviewers spot contract gaps.
4. existing_tests — test files/classes covering it, and which of them PASS on this Mac vs need Jetson deps.
5. ledger — this module's entries in research/module_review_2026-07.json, verbatim-ish, one per line.
6. killlist — any research/KILL_LIST.md entries that touch this module's domain (so reviewers do not propose killed ideas).
7. constraints — anything that makes a change dangerous here: model-facing surfaces, shared kernels, hot paths on the Jetson, files whose format other modules parse, byte-compat tests that pin behavior.
Facts only. If something is unknown, say unknown rather than inferring.`
}

// IDENTICAL for every reviewer of a module — that is the point of the panel.
function reviewPrompt(g, pack) {
  return `${CTX}

MODULE REVIEW — ${g.mods}${g.seed ? `\nOwner seed (known items / hints): ${g.seed}` : ''}

You are one of ${g.reviewers} reviewers working this module RIGHT NOW under an identical brief, independently. You will never see their reports and they will never see yours. Do not aim for a safe, obvious answer — report what YOU actually find. Independent agreement is what earns a finding confidence downstream, and a sharp finding only you spotted is exactly what a panel exists to surface.

FACTUAL CONTEXT PACK (scaffolding, not opinion — verify anything you lean on):
${renderPack(pack)}

Read ${g.mods} in full, plus whatever else you need. Find three classes of thing:

1. BUGS — wrong results, crashes, races, silent failure, resource/handle leaks, unbounded growth, PIT/lookahead violations, fail-open where this repo demands fail-closed, train/serve divergence, contract violations with callers. For each: quote the offending code and give a CONCRETE failure scenario (specific inputs/state -> wrong output or crash). "This looks fragile" is not a bug.

2. IMPROVEMENTS — make the existing behavior better without changing it: efficiency on the Jetson hot path (allocation inside loops, repeated file IO/JSON parse/regex compile, O(n^2) where O(n) exists, recomputation that could be cached or vectorized, memory held longer than needed on an 8GB box), clearer structure, dead code, sharper error handling, docstrings that no longer match the behavior.

3. GAPS — what this module SHOULD have and does not: unhandled edge cases, missing input validation, branches no test covers, instrumentation the rest of the system already expects, data it computes and silently drops, contract holes with its callers, missing bounds on anything that grows.

For EVERY finding: exact file + line range, a VERBATIM snippet of the current code as your proof, the failure or the cost, and a concrete proposal. Then CLASSIFY it — this decides whether it may ship at all:
  instrumentation | behavior-neutral | model-facing | policy-semantics | unclear
Model-facing means it changes feature VALUES, thresholds, entry/exit/sizing semantics, label construction, or model artifacts. Those must NOT be proposed as work — classify them honestly so they become owner decisions. Misclassifying a model-facing change as behavior-neutral is the single worst mistake you can make here.

Rank by severity P0 (broken/dangerous) / P1 (high value) / P2 (worthwhile) / P3 (polish), and set confidence honestly — low-confidence findings with real evidence are welcome; padding is not.

READ-ONLY: change nothing, run nothing that mutates state. If the module is genuinely in good shape, say so in clean_verdict and return few or no findings — a clean verdict is a valid, useful result and costs the owner nothing to read.`
}

function specPrompt(g, pack, reviews) {
  const panel = reviews.map((r, i) => `<<<REVIEWER ${i + 1}>>>\n${JSON.stringify(r)}`).join('\n\n')
  return `${CTX}

ADJUDICATE THE PANEL -> DESIGN SPEC — ${g.mods}

${reviews.length} independent Opus reviewers examined this module under an identical brief. Their reports:

${panel}

CONTEXT PACK:
${renderPack(pack)}

Your job is adjudication, not aggregation:
1. MERGE semantically identical findings across reviewers and record consensus = how many reviewers independently reached it. Consensus is evidence, not proof: a 3-of-3 finding that misreads the code is still wrong, and a 1-of-3 finding with an airtight quoted proof is still right. Judge the CODE, not the vote.
2. VERIFY every anchor you intend to keep against the CURRENT file — reviewers quote from memory and drift. Any finding whose evidence does not survive your own read is REJECTED, with the reason. This verification pass is the reason you exist in this pipeline.
3. RESOLVE disagreements explicitly (one reviewer calls something a bug, another calls it correct) in conflicts_resolved.
4. GATE by classification: only instrumentation / behavior-neutral / unambiguous-bug fixes may enter the spec. Everything model-facing or policy-semantic goes to deferred_to_owner as one line + a fix sketch sharp enough for the owner to act on later. Re-check research/KILL_LIST.md before accepting anything that smells strategic.
5. ORDER the accepted work so an implementer executes top-to-bottom without rework, and state for each item what must remain byte-identical.

Produce:
- spec: the implementer-facing instruction text. Exact anchors, the exact intended change, the tests to add in ${g.test}, and the verification commands. Write it to be followed VERBATIM by someone who will not re-derive your reasoning and cannot ask you questions.
- evidence: for EVERY anchor the implementer will touch — {file, lines, verbatim current snippet with enough surrounding context to edit unambiguously, why}. The implementer works FROM this pack and opens files only where it edits; completeness here is what keeps the pipeline both cheap and correct.
- accepted / rejected / deferred_to_owner / conflicts_resolved.

An EMPTY spec is the correct output for a module the panel found clean or whose findings are all model-facing. Do not manufacture work to look productive. READ-ONLY at this stage.`
}

function implPrompt(g, spec, repairNotes) {
  const repair = repairNotes
    ? `\nREPAIR ROUND — a reviewer examined your previous attempt and requires these corrections. Fix these EXACTLY; do not start over and do not re-litigate:\n${repairNotes}\n`
    : ''
  return `${CTX}

IMPLEMENT — STRICT OWNERSHIP: edit ONLY ${g.mods} and ${g.test} (create the test file if absent). A change that needs another file is REPORTED in 'deferred', never implemented.${repair}

SPEC (follow verbatim; verify each anchor against the current file before editing — if an anchor drifted, adapt minimally and say so; if the spec misreads the code, skip that item into 'deferred' with the reason rather than forcing it):
${spec.spec}

EVIDENCE PACK (the exact code at each anchor as of design time — work from this, and read files only around your own edits):
${renderEvidence(spec.evidence)}

Then verify: python3 -m py_compile ${g.mods} && python3 -m pytest ${g.test} -q, plus any existing Mac-passing test module that covers these files. A newly failing existing test means YOUR edit is wrong — fix it; never weaken the test. Another session may be editing other files: re-read immediately before each edit, and ignore dirty state you do not own.

Report honestly. "Deferred 3 of 7 because the spec misread the code" is a good report; silently skipping is not.`
}

function hardenPrompt(g, spec, impl, finalRound) {
  const finality = finalRound
    ? `This is the FINAL pass: 'needs-repair' is NOT available. Either bring it to approvable yourself, or 'git checkout -- ${g.mods} ${g.test}' and return rejected-reverted with the reason.`
    : `If the implementation has correctable defects too large to fix yourself, return verdict 'needs-repair' with precise fix_notes — the implementer gets ONE guided repair round.`
  return `${CTX}

FINAL REVIEW + HARDENING — ${g.mods}

Start from the diff: git diff -- ${g.mods} ${g.test}. Then the implementer's own report:
${JSON.stringify(impl)}
Accepted spec items: ${(spec.accepted || []).map(a => `${a.title} [${a.consensus}/${g.reviewers}, ${a.classification}]`).join(' ;; ')}

You own this code now. Two duties, in order:

A. CORRECTNESS — confirm every change does what the spec intended and introduces nothing new. Deep-read only where the diff exceeds spec scope, touches tests, or looks wrong. Check the things implementers get wrong: off-by-one and boundary conditions, None/NaN/empty-input paths, exception paths that now swallow more than intended, mutable default or shared-state capture, a cache that can grow without bound, a test that asserts the implementation rather than the behavior. FIX what is broken.

B. QUALITY — improve what shipped, within hard bounds:
   - real bugs you find while reading, even if the implementer did not introduce them in this diff;
   - efficiency, especially on the Jetson hot path (hoist invariants out of loops, precompile regexes, avoid re-reading/re-parsing files, vectorize with numpy where the surrounding code already does, drop redundant copies on an 8GB box);
   - modern, idiomatic Python where it genuinely reads better AND matches the surrounding style: pathlib over os.path string surgery, f-strings, dataclasses, comprehensions, context managers, early returns over deep nesting, informative type hints, contextlib.suppress over empty except blocks — but only where the file already leans that way.
   Do NOT churn style for its own sake. A diff that rewrites working code without improving correctness, speed, or clarity is worse than no diff, and it costs the owner review time.

BOUNDS: behavior-neutral or unambiguous-bug-only; no model-facing drift; never weaken, skip, or delete a test; edit ONLY ${g.mods} and ${g.test}. After your edits re-run python3 -m py_compile ${g.mods} && python3 -m pytest ${g.test} -q and leave the tree green — record the real result in test_result.
${finality}
Always fill commit_line with one conventional-commit line for this module.`
}

// ------------------------------------------------------------------ chain ----
async function chain(g) {
  const pack = WANT_PACK
    ? await agent(packPrompt(g), {label: `pack:${g.id}`, phase: 'Context', model: 'sonnet', effort: E_SONNET, schema: PACK_SCHEMA})
    : null

  // Panel: identical brief, N independent reviewers. Barrier is genuine — the synthesizer
  // needs every report before it can weigh consensus.
  const reviews = (await parallel(
    Array.from({length: g.reviewers}, (_, i) => () =>
      agent(reviewPrompt(g, pack),
        {label: `review:${g.id}#${i + 1}`, phase: 'Review', model: 'opus', effort: E_OPUS, schema: REVIEW_SCHEMA}))
  )).filter(Boolean)

  if (reviews.length === 0) return {id: g.id, failed: 'review panel returned nothing'}
  const nFindings = reviews.reduce((n, r) => n + (r.findings || []).length, 0)
  log(`[${g.id}] panel: ${reviews.length}/${g.reviewers} reviewers, ${nFindings} raw findings`)

  const spec = await agent(specPrompt(g, pack, reviews),
    {label: `spec:${g.id}`, phase: 'Spec', model: 'fable', effort: E_FABLE, schema: SPEC_SCHEMA})
  if (!spec) return {id: g.id, reviews, failed: 'spec' }

  if (!spec.spec || !spec.spec.trim() || (spec.accepted || []).length === 0) {
    log(`[${g.id}] no shippable work (${(spec.deferred_to_owner || []).length} owner deferrals)`)
    return {id: g.id, reviews, spec, skipped: 'panel found nothing shippable', n_findings: nFindings}
  }

  let impl = await agent(implPrompt(g, spec, null),
    {label: `impl:${g.id}`, phase: 'Implement', model: 'sonnet', effort: E_SONNET, schema: IMPL_SCHEMA})
  if (!impl) return {id: g.id, reviews, spec, failed: 'implement'}

  let harden = await agent(hardenPrompt(g, spec, impl, false),
    {label: `harden:${g.id}`, phase: 'Harden', model: 'fable', effort: E_FABLE, schema: HARDEN_SCHEMA})

  if (harden && harden.verdict === 'needs-repair') {
    log(`[${g.id}] repair round: ${String(harden.fix_notes).slice(0, 140)}`)
    impl = await agent(implPrompt(g, spec, harden.fix_notes),
      {label: `repair:${g.id}`, phase: 'Implement', model: 'sonnet', effort: E_SONNET, schema: IMPL_SCHEMA})
    harden = await agent(hardenPrompt(g, spec, impl || {}, true),
      {label: `harden2:${g.id}`, phase: 'Harden', model: 'fable', effort: E_FABLE, schema: HARDEN_SCHEMA})
  }
  return {id: g.id, reviews, spec, impl, harden, n_findings: nFindings}
}

// ------------------------------------------------------- run the campaign ----
const planned = MODULES.length * (REVIEWERS + (WANT_PACK ? 4 : 3)) + 2
log(`module-improve-v3: ${MODULES.length} module(s) x ${REVIEWERS} reviewers, ${N_WORKERS} workers ` +
    `-> ~${planned} agents (plus any repair rounds)`)

const queue = [...MODULES]
const results = []
const dropped = []
async function worker(n) {
  while (queue.length) {
    if (budget.total && budget.remaining() < BUDGET_RESERVE) {
      while (queue.length) dropped.push(queue.shift().id)
      break
    }
    const g = queue.shift()
    log(`[worker ${n}] -> ${g.id} (${g.mods})`)
    const r = await chain(g)
    results.push(r)
    log(`[${g.id}] ${r.failed ? 'FAILED at ' + r.failed : (r.harden ? r.harden.verdict : r.skipped || 'done')}`)
  }
}
await parallel(Array.from({length: N_WORKERS}, (_, i) => () => worker(i + 1)))
if (dropped.length) log(`BUDGET STOP — not reviewed: ${dropped.join(', ')}`)

// One serialized gate. Concurrent full-suite runs on a shared tree produce phantom failures,
// so this runs only after every chain is finished.
const owned = results.filter(r => r.impl).map(r => r.id)
let gate = null
if (owned.length) {
  gate = await agent(
`${CTX}

REGRESSION GATE (read-only except as instructed below). Run: cd ${REPO} && bash scripts/ab_check.sh
It reruns the full suite and diffs FAILED/ERROR test NAMES against tests/baseline_failures.txt. Report its verdict verbatim: passed = exit 0 / "NEW failures: none". List any NEW names and any DISAPPEARED names, and for each NEW name say which of these modules plausibly caused it: ${owned.join(', ')}.
Do not edit anything. Do not regenerate the baseline.`,
    {label: 'gate', phase: 'Gate', model: 'sonnet', effort: E_SONNET, schema: GATE_SCHEMA})

  if (gate && gate.passed === false && (gate.new_failures || []).length) {
    log(`GATE FAILED: ${gate.new_failures.join(', ')} — one guided repair`)
    await agent(
`${CTX}

GATE REPAIR — scripts/ab_check.sh reports these NEW failures after this campaign:
${gate.new_failures.join('\n')}
Modules changed by the campaign: ${owned.join(', ')}. Diagnose each failure, then either FIX it in the owning module/test, or if it cannot be fixed cleanly, 'git checkout -- <that module and its test>' to revert just that module's work. Touch only files this campaign changed — never revert unrelated uncommitted work. Re-run bash scripts/ab_check.sh and report the final state honestly.`,
      {label: 'gate-repair', phase: 'Gate', model: 'fable', effort: E_FABLE})
    gate = await agent(
`${CTX}

RE-GATE: run cd ${REPO} && bash scripts/ab_check.sh and report the result verbatim.  Do not edit anything.`,
      {label: 'regate', phase: 'Gate', model: 'sonnet', effort: E_SONNET, schema: GATE_SCHEMA})
  }
}

// ----------------------------------------------------------------- report ----
const summary = results.map(r => ({
  id: r.id,
  verdict: r.harden ? r.harden.verdict : (r.failed || r.skipped || 'done'),
  raw_findings: r.n_findings || 0,
  accepted: r.spec ? (r.spec.accepted || []).length : 0,
  rejected: r.spec ? (r.spec.rejected || []).length : 0,
  deferred_to_owner: r.spec ? (r.spec.deferred_to_owner || []) : [],
  bugs_fixed: r.harden ? (r.harden.bugs_fixed || []) : [],
  perf_improved: r.harden ? (r.harden.perf_improved || []) : [],
  modernized: r.harden ? (r.harden.modernized || []) : [],
  commit_line: r.harden ? r.harden.commit_line : null,
}))

await agent(
`${CTX}

FINAL REPORT — write ${REPO}/${REPORT_PATH} (create/overwrite ONLY that file). Assemble it from this structured campaign result; do not re-read the code or re-derive anything, and do not soften a bad result:
${JSON.stringify({campaign: cfg.campaign_title || 'module-improve-v3', reviewers_per_module: REVIEWERS, gate, dropped, modules: summary}, null, 1)}

Structure:
# Module improvement campaign — panel review (v3)
1. One-paragraph verdict: modules reviewed, panel size, what shipped, gate result (state plainly if the gate failed or if modules were dropped for budget).
2. Per-module table: module | raw findings | accepted | rejected | verdict | bugs fixed / perf / modernized (counts).
3. "Shipped" — the commit_line of each module, as a bulleted list.
4. "OWNER DECISIONS (deferred — model-facing or policy-semantic)" — every deferred_to_owner line, grouped by module. This is the section the owner acts on; keep every line verbatim.
5. "Verification" — the gate summary, plus the standing reminder that nothing is committed and that heavy-dep modules are only verified as far as this dev Mac allows.
Keep it under 120 lines. End with: 'Generated by module-improve-v3 (Opus panel -> Fable spec -> Sonnet implement -> Fable harden).'`,
  {label: 'report', phase: 'Report', model: 'sonnet', effort: E_SONNET})

log(`module-improve-v3 done: ${results.length} module(s), gate ${gate ? (gate.passed ? 'PASS' : 'FAIL') : 'n/a'}`)
return {modules: summary, gate, dropped, report: REPORT_PATH}
