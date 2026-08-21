// Promoted from the 2026-07 campaign (wf_4ec23279-d53) session workflow dir. Config-driven:
// pass args.groups_path (JSON: {campaign_title, baseline_note, tree_note, groups:[...]}) or
// inline args.groups. Template: .claude/workflows/groups-2026-07.example.json.
// Operating manual: .claude/skills/improve/SKILL.md.
export const meta = {
  name: 'group-improve-v2',
  description: 'v2 evidence-pack pipeline: Design+Scout (Fable/Opus by tier) -> script-assembled Sonnet implement -> thin Fable verify with one repair loop; 3 workers, no round barriers; config-driven groups',
  phases: [
    { title: 'Design', detail: 'one read pass: spec + evidence pack + deferrals' },
    { title: 'Implement', detail: 'Sonnet, fed spec+evidence inline; self-verifies', model: 'sonnet' },
    { title: 'Verify', detail: 'Fable reads diff+report only; polish or one repair loop', model: 'fable' },
    { title: 'CommitMsg', detail: 'Haiku assembles commit_msg.txt', model: 'haiku' },
  ],
}

const REPO = (args && args.repo) || '/Users/kywwilson/Desktop/Projects/trader'

const CFG_SCHEMA = {type:'object', properties: {
  campaign_title: {type:'string'}, baseline_note: {type:'string'}, tree_note: {type:'string'},
  groups: {type:'array', items:{type:'object', properties:{
    id:{type:'string'}, tier:{type:'string'}, mods:{type:'string'}, test:{type:'string'},
    seed:{type:'string'}, reuse:{type:'string'}},
    required:['id','tier','mods','test','seed']}}},
  required:['groups']}

let cfg = null
if (args && args.groups) {
  cfg = args
} else if (args && args.groups_path) {
  cfg = await agent(
    `Read the JSON file ${args.groups_path} and return its contents verbatim as the structured output (fields: campaign_title, baseline_note, tree_note, groups). Read-only; do not edit anything.`,
    {label: 'load-config', phase: 'Design', model: 'haiku', schema: CFG_SCHEMA})
}
if (!cfg || !Array.isArray(cfg.groups) || cfg.groups.length === 0) {
  log('group-improve-v2: no groups. Pass args.groups_path (JSON file: {campaign_title, baseline_note, tree_note, groups:[{id,tier,mods,test,seed,reuse?}]}) or inline args.groups. Template: .claude/workflows/groups-2026-07.example.json')
  return {error: 'missing groups config'}
}

const BASELINE_NOTE = cfg.baseline_note || (args && args.baseline_note) ||
  `see ${REPO}/CLAUDE.md "Running tests" for the current pre-existing failure baseline`
const TREE_NOTE = cfg.tree_note || (args && args.tree_note) ||
  'Other sessions may be editing other files concurrently — ignore unrelated dirty state.'

const CTX = `Repo: ${REPO} — autonomous Alpaca paper-trading (LSTM+LightGBM hourly bars, crypto+stocks), prod on Jetson Orin Nano 8GB. ${TREE_NOTE} This dev Mac has NO torch/lightgbm/numba/sklearn/dotenv/alpaca/PySide6 (numpy/pandas/scipy/yfinance/requests OK; numba code has pure fallbacks — heavy modules: READ, never import). Conventions: train/serve parity sacred; MODEL-FACING changes (feature values, thresholds, entry/exit/sizing semantics, model artifacts) NEVER ship silently — defer to owner; measurement/instrumentation/unambiguous-bug fixes ship directly; fail-closed in live paths; policy_exits.exit_walk is the shared exit kernel — never fork it. Every module already passed a function-by-function review (ledger: research/module_review_2026-07.json — read your modules' entries, do not re-derive). Suite baseline: ${BASELINE_NOTE}; judge regressions by NEW failure NAMES in files you own. Do not commit. Never touch files outside stated ownership.`

const GROUPS = cfg.groups

const DESIGN_SCHEMA = {type:'object', properties: {
  accepted: {type:'array', items:{type:'string'}},
  rejected: {type:'array', items:{type:'string'}},
  spec: {type:'string'},
  evidence: {type:'array', items:{type:'object', properties:{
    file:{type:'string'}, lines:{type:'string'}, snippet:{type:'string'}, why:{type:'string'}},
    required:['file','lines','snippet','why']}},
  deferred_to_owner: {type:'array', items:{type:'string'}}},
  required:['accepted','rejected','spec','evidence','deferred_to_owner']}
const IMPL_SCHEMA = {type:'object', properties: {
  fixes_applied: {type:'array', items:{type:'string'}}, deferred: {type:'array', items:{type:'string'}},
  tests_added: {type:'integer'}, test_result: {type:'string'}}, required:['fixes_applied','deferred','tests_added','test_result']}
const VERIFY_SCHEMA = {type:'object', properties: {
  verdict: {type:'string', enum:['approved','approved-after-polish','needs-repair','rejected-reverted']},
  fix_notes: {type:'string'}, polish_applied: {type:'array', items:{type:'string'}},
  notes: {type:'string'}, commit_line: {type:'string'}},
  required:['verdict','fix_notes','polish_applied','notes','commit_line']}

function renderEvidence(evidence) {
  return evidence.map(e => `--- ${e.file} (lines ${e.lines}) — ${e.why}\n${e.snippet}`).join('\n\n')
}

function designPrompt(g, presummary) {
  const pre = presummary ? `\nA prior Opus summary of this group exists — verify its claims from the code rather than re-deriving:\n<prior_summary>${presummary}</prior_summary>\n` : ''
  return `${CTX}

DESIGN+SCOUT (single read pass) — modules: ${g.mods}. Seed: ${g.seed}${pre}
Read the ledger entries for these modules, then the code ONCE, verifying as you go. Produce:
1. spec — the final improvement set an implementer follows verbatim: ONLY behavior-neutral changes or unambiguous-bug fixes; exact anchors; the tests to add in ${g.test}; verification commands. Go beyond bug-hunting: efficiency (Jetson hot paths), structure, dead code, doc rot.
2. evidence — for EVERY anchor the implementer needs: {file, lines, verbatim snippet (the exact current code, enough context to edit unambiguously), why}. The implementer will work FROM THIS PACK and read files only where it edits — completeness here is what makes the whole pipeline cheap.
3. deferred_to_owner — every model-facing/behavior-changing idea, one line + fix sketch each.
An empty spec is a valid answer for a clean group. READ-ONLY at this stage.`
}

function implPrompt(g, design, repairNotes) {
  const repair = repairNotes ? `\nREPAIR ROUND — a reviewer examined your previous attempt and requires these corrections (fix these EXACTLY; do not start over):\n${repairNotes}\n` : ''
  return `${CTX}

IMPLEMENT — STRICT OWNERSHIP: edit ONLY ${g.mods} and ${g.test} (create if absent).${repair}
Spec (follow verbatim; verify each anchor against the current file before editing — if an anchor has drifted, adapt minimally and note it; if the spec misreads the code, skip that item into 'deferred' with the reason):
${design.spec}

EVIDENCE PACK (the exact code at each anchor as of design time — work from this; read files only around your edits):
${renderEvidence(design.evidence)}

Then run: python3 -m py_compile ${g.mods} && python3 -m pytest ${g.test} -q, plus any existing Mac-passing test module covering these files. A broken existing test means fix YOUR edit — never weaken a test. Report honestly.`
}

function verifyPrompt(g, design, impl, finalRound) {
  const finality = finalRound
    ? `This is the FINAL verification: 'needs-repair' is NOT available — either polish it yourself to approvable, or git checkout -- the owned files and return rejected-reverted.`
    : `If the implementation has correctable defects too large to polish yourself, return verdict 'needs-repair' with precise fix_notes (the implementer gets ONE guided repair round).`
  return `${CTX}

VERIFY+POLISH (thin pass) — modules: ${g.mods}. Read the DIFF first (git diff -- ${g.mods} ${g.test}) and the implementer report; deep-read code ONLY where the diff exceeds the spec's scope, touches tests, or looks wrong. Spec summary of accepted items:
${design.accepted.join('\n')}
Implementer report: ${JSON.stringify(impl)}
Checks: every change spec-conformant and behavior-neutral/unambiguous-bug-only; python3 -m pytest ${g.test} -q green; full suite (--continue-on-collection-errors) shows NO new failure NAMES in owned files (concurrent chains own other files — ignore them). You MAY edit ${g.mods}/${g.test} to polish. ${finality}
Always fill commit_line: one conventional-commit line for this group.`
}

async function tierC(g) {
  // light groups: one agent designs AND implements; independent Sonnet verifier
  const impl = await agent(
`${CTX}

DESIGN+IMPLEMENT (light tier) — STRICT OWNERSHIP: edit ONLY ${g.mods} and ${g.test} (create if absent). Seed: ${g.seed}
Read the ledger entries + the code once; decide the behavior-neutral/unambiguous-bug improvements yourself (model-facing ideas go to 'deferred'); implement them with tests in ${g.test}; run python3 -m py_compile ${g.mods} && python3 -m pytest ${g.test} -q. Report honestly.`,
    {label: `design+impl:${g.id}`, phase: 'Implement', model: 'opus', schema: IMPL_SCHEMA})
  if (!impl) return {id: g.id, failed: 'design+implement'}
  const verify = await agent(
`${CTX}

INDEPENDENT VERIFY (light tier) — modules: ${g.mods}. Author's report: ${JSON.stringify(impl)}
Read the diff (git diff -- ${g.mods} ${g.test}); confirm behavior-neutral/bug-fix-only and tests green (run them); check the full suite for new failure NAMES in these files. Small fixes: apply yourself. Unsalvageable: git checkout -- the owned files, verdict rejected-reverted. Fill commit_line.`,
    {label: `verify:${g.id}`, phase: 'Verify', model: 'sonnet', schema: VERIFY_SCHEMA})
  return {id: g.id, impl, review: verify}
}

async function chain(g) {
  if (g.tier === 'C') return tierC(g)

  let design = null
  const rp = args && args.reuse_path
  if (g.reuse === 'spec' && rp) {
    // paid round-1 design exists on disk — hand the implementer a pointer;
    // it Reads the real spec itself (no evidence pack: verify anchors in-file)
    design = {
      spec: `REUSED ROUND-1 DESIGN: before doing ANYTHING, Read the JSON file ${rp} and extract YOUR spec from .reuse.${g.id}.spec — follow that spec verbatim (it contains exact anchors and quoted code). There is no evidence pack; verify each anchor against the current file before editing.`,
      evidence: [],
      accepted: [`reused round-1 design — full spec at ${rp} -> .reuse.${g.id}.spec (Read it)`],
      rejected: [],
      deferred_to_owner: [`see .reuse.${g.id}.deferred_to_owner in ${rp}`],
    }
    log(`[${g.id}] reusing paid round-1 design via ${rp}`)
  } else {
    const presummary = (g.reuse === 'summary' && rp)
      ? `The full summary text is on disk — Read the JSON file ${rp} and use .reuse.${g.id}.summary as the prior summary to verify.`
      : null
    const model = g.tier === 'A' ? 'fable' : 'opus'
    design = await agent(designPrompt(g, presummary),
      {label: `design:${g.id}`, phase: 'Design', model, schema: DESIGN_SCHEMA})
    if (!design) return {id: g.id, failed: 'design'}
  }
  if (!design.spec || !design.spec.trim() || (design.accepted || []).length === 0) {
    return {id: g.id, design, skipped: 'no safe improvements worth shipping'}
  }

  let impl = await agent(implPrompt(g, design, null),
    {label: `impl:${g.id}`, phase: 'Implement', model: 'sonnet', schema: IMPL_SCHEMA})
  if (!impl) return {id: g.id, design, failed: 'implement'}

  let review = await agent(verifyPrompt(g, design, impl, false),
    {label: `verify:${g.id}`, phase: 'Verify', model: 'fable', schema: VERIFY_SCHEMA})
  if (review && review.verdict === 'needs-repair') {
    log(`[${g.id}] repair round: ${review.fix_notes.slice(0, 120)}`)
    impl = await agent(implPrompt(g, design, review.fix_notes),
      {label: `repair:${g.id}`, phase: 'Implement', model: 'sonnet', schema: IMPL_SCHEMA})
    review = await agent(verifyPrompt(g, design, impl, true),
      {label: `verify2:${g.id}`, phase: 'Verify', model: 'fable', schema: VERIFY_SCHEMA})
  }
  return {id: g.id, design, impl, review}
}

const queue = [...GROUPS]
const results = []
async function worker(n) {
  while (queue.length) {
    const g = queue.shift()
    log(`[worker ${n}] -> ${g.id} (tier ${g.tier})`)
    const r = await chain(g)
    results.push(r)
    log(`[${g.id}] ${r.failed ? 'FAILED at ' + r.failed : (r.review ? r.review.verdict : r.skipped || 'done')}`)
  }
}
const N_WORKERS = Math.max(1, (args && args.workers) || 3)
await parallel(Array.from({length: N_WORKERS}, (_, i) => () => worker(i + 1)))

const lines = results.filter(r => r.review && r.review.commit_line).map(r => r.review.commit_line)
const extra = (args && args.round1_lines) || []
const TITLE = cfg.campaign_title || 'feat: group improve campaign (v2 pipeline)'
await agent(
`${CTX}

FINAL: write ${REPO}/commit_msg.txt (create/overwrite ONLY that file; if write-protected, write commit_msg_new.txt and say so): conventional commit for this campaign — title '${TITLE}', then a bulleted body including these lines where they fit:\n${extra.concat(lines).join('\n')}\nGround scope with 'git status --short | head -40'. Under 35 lines. End body with: 'Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>'.`,
  {label: 'commit-msg', phase: 'CommitMsg', model: 'haiku'})

log(`v2 campaign done: ${results.length} groups`)
return {results: results.map(r => ({id: r.id, verdict: r.review ? r.review.verdict : (r.failed || r.skipped),
  n_fixes: r.impl ? (r.impl.fixes_applied || []).length : 0,
  deferred_to_owner: r.design ? r.design.deferred_to_owner : []}))}
