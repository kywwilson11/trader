# GLOSSARY — trader

Every term a future agent will meet in this codebase and in its docs, defined **as used here** —
the local meaning, not the textbook definition alone. Each entry names the module (and the
function or constant, where one exists) that owns the concept, so the code is always one grep
away.

Conventions:
- `module.symbol` citations are stable; line numbers are deliberately avoided.
- Facts have one home. Flag inventory → `docs/FLAGS.md`; runtime/generated files →
  `docs/STATE_FILES.md`; per-module reference → `docs/MODULES.md`; suite counts and how to run
  things → `CLAUDE.md`. This file defines vocabulary and never repeats those numbers.
- Entries tagged *(unverified)* could not be confirmed against code from the dev Mac.

[A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) ·
[J](#j) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) ·
[S](#s) · [T](#t) · [V](#v) · [W](#w)

## A

- **ab_check** — the repo's one-command regression verdict: rerun the dev-Mac suite, extract the
  `FAILED`/`ERROR` test IDs, and diff the NAME sets against the committed baseline. Exit 0 means
  no NEW failing name appeared; counts are explicitly not the signal.
  Lives in: `scripts/ab_check.sh`. See also: baseline_failures.txt, NEW vs DISAPPEARED.
- **acquisition evidence** — the order object carrying the largest observed `filled_qty` seen
  across a multi-stage entry, returned by the maker ladder so a later zero-fill poll cannot erase
  an earlier partial fill. Lives in: `order_utils.place_maker_buy`. See also: maker ladder.
- **admission floor (edge floor)** — the minimum expected edge a candidate must clear to be
  admitted: `MIN_EDGE_MULTIPLE` (2.0) × round-trip cost. It is a *margin* requirement, not
  break-even. Lives in: `fees.required_edge_pct`, `fees.MIN_EDGE_MULTIPLE`.
  See also: cost gate, round-trip cost, cost-multiple ladder.
- **admitted-k** — how many candidates a cycle's full gate stack actually admitted; a Stage-0
  measurement of funnel width rather than a policy knob.
  Lives in: `decision_report` (journal rollup). See also: Stage-0, GATE_REASONS.
- **advisor-v2 (decision dossier)** — the opt-in extended LLM schema (`p_up`, `conviction`,
  `abstain`, `key_risks`, `event_flags`). It is shadow-journaled and scored offline; it never
  gates or sizes a trade. Lives in: `llm_analyst`, scored by `llm_eval.py --advisor`.
  See also: LLM roles, prompt version, `s`.
- **AKL lagged beta** — Asness-Krail-Liew (2001) beta: sum the contemporaneous and lagged
  benchmark betas from one joint regression. Ignoring the lags roughly halves measured exposure
  and manufactures alpha, so the ledger always reports the summed figure.
  Lives in: `beta_ledger.ols_hac` and its `--lags` CLI flag. See also: beta ledger, HAC alpha.
- **archive/** — the root directory that receives anything the repo stops using. Nothing is
  deleted; stale scratch files, dead run-state and local residue are MOVED here with a row added
  to `archive/README.md`. Lives in: `archive/`. See also: delete-nothing, 08_removed_code.
- **as-of universe (as-of membership)** — keeping a training row only when the name would have
  been selected by a mechanical rule *on that day* (tradability floors, or top-`AS_OF_TOP_K` by
  30-day median dollar volume). This is the survivorship / look-ahead mask.
  Lives in: `stock_config.AS_OF_TOP_K`, applied in `scripts/harvest_stock_data.py`.
  See also: survivorship, PIT, DV30.
- **average uniqueness** — AFML's `ū`: the fraction of a label's holding window not shared with
  other overlapping labels. Summing it over rows gives the effective sample size that deflates
  every Sharpe-based promotion statistic.
  Lives in: `sample_weights.average_uniqueness`. See also: effective-n, DSR, Kish design effect.

## B

- **`b2`** — the coefficient on the z-scored LLM conviction score in
  `realized = a + b1·pred + b2·z_s`. Its significance under Driscoll-Kraay standard errors at
  n ≥ 60 IS the keep-or-kill-the-LLM-spend verdict. Lives in: `llm_eval.py`.
  See also: echo gap, `s`, LLM roles.
- **backfill role** — the batched historical article-scoring LLM role. It is pinned to Gemini's
  Batch API (50% price, separate quota) and does not follow the provider switch.
  Lives in: `sentiment_history` (Gemini batch path), `llm_client`. See also: Batch API, tier.
- **bad print** — an isolated close more than 6 robust (MAD-scaled) sigmas from the rolling
  11-bar median; dropped before features are computed.
  Lives in: `market_data._filter_bad_prints`. See also: closed bar, harvest.
- **baseline_failures.txt** — the committed list of dev-Mac-only failing test NAMES (the
  missing-heavy-dependency set). It is machine-specific by design and is deliberately not
  portable to the Jetson or CI, where the suite is green.
  Lives in: `tests/baseline_failures.txt`. See also: ab_check, two-machine reality.
- **Batch API** — Gemini's batched generation endpoint (half price, its own quota). The only
  role wired to it is backfill; no other provider has a batch path.
  Lives in: `sentiment_history` (`models/{model}:batchGenerateContent`). See also: backfill role.
- **bear / bull** — vestigial vocabulary. The old dual bear/bull model ensemble is GONE; the
  names survive only as (a) regime *diagnostics*, (b) the training-objective regime penalty mask,
  and (c) LLM-vs-ML disagreement buckets in the scorecard. Never read them as two live models.
  Lives in: `regime_detector` (bear/neutral/bull states), `llm_eval` (`llm_bull_ml_bear` …),
  `strategy_config` (regime-penalty note). See also: regime, champion, challenger.
- **beta ledger** — the measurement-only report of realized daily equity beta against SPY and
  BTC: lagged AKL betas, a HAC alpha t-stat, and up/down plus trend-conditional betas.
  Lives in: `beta_ledger.py` (CLI `python beta_ledger.py --days N`). See also: AKL, HAC alpha.
- **blend (blend weight `w`, `lstm_weight`)** — the single deployed predictor per book:
  `pred = w·LSTM + (1−w)·LGB`, default `w = 0.6`. Under the V3 path `w` is fitted, smoothed and
  clamped to [0.25, 0.75]; under the legacy path 0.6/0.4 is hardcoded and never certified.
  Lives in: `blend_fit`, served by `predict_now`, replayed by `backtest`.
  See also: certificate, stale-w vs refit-w, q10 booster.
- **book** — one tradable universe with its own model stack, journals, flags and loop: crypto
  (24/7, artifact prefix `''`) or US stocks (RTH, prefix `stock_`). "Per book" always means
  these two. Lives in: `crypto_loop`, `stock_loop` over `base_loop`. See also: prefix.
- **book risk budget** — the maximum stop-risk a NEW position may add before the per-book cap
  (`MAX_BOOK_RISK_PCT`) binds, measured on the correlation-adjusted (ENB) aggregate rather than
  the naive sum. Lives in: `portfolio.diversified_book_risk`, `strategy_config.MAX_BOOK_RISK_PCT`.
  See also: ENB, stop-risk, GATE-1 / GATE-2.
- **byte-pin test** — a test asserting that a new flag's default-OFF path produces output
  byte-identical to pre-flag behavior. Every model-facing flag in the 2026-08 campaign has one;
  never relax a byte-pin to make a new feature fit. Lives in: `tests/` (e.g.
  `tests/test_r2c_holdout_boundary.py`). See also: default-OFF, golden fingerprint, flag family.

## C

- **c26** — the 2026-08 campaign packet family. Packet IDs run D01–D40 (defects) and B02–B24
  (builds); the tests that pin them are named `tests/test_c26_*.py`.
  Lives in: `research/campaign_2026-08/01_state_map.md`, `tests/test_c26_*.py`.
  See also: packet ID, r2c, IA-1..IA-4.
- **candidate pool** — sector-diverse liquid names that are harvested and trained on but never
  traded, included to dilute the winner-tilted hand-picked universe.
  Lives in: `stock_config.TRAINING_CANDIDATE_POOL`. See also: as-of universe, survivorship.
- **capped IOC** — a marketable limit with `time_in_force='ioc'` priced at most `cap_bps` past
  the touch, auto-cancelling the remainder. Entries only — never liquidations.
  Lives in: `order_utils` (`TRADER_IOC_ENTRY_CAP`). See also: `entry_tactic`, maker ladder.
- **certificate ("cert == deploy")** — the holdout report stored as `config['holdout']` and in
  the manifest. The invariant it encodes: the certified predictor (blend + q10 veto + threshold)
  is EXACTLY what serving and the replay run. Defect H1 is a breach of it.
  Lives in: `scripts/hypersearch_v2.py` (`evaluate_on_holdout`). See also: blend, holdout pin, H1.
- **challenger** — a freshly trained candidate model saved into a parallel artifact slot
  (prefix `challenger_` / `stock_challenger_`, via `hypersearch_v2 --shadow`) instead of
  overwriting the champion. Lives in: `scripts/hypersearch_v2.py --shadow`, `shadow.py`.
  See also: champion, shadow slot, DM-HLN, promotion path.
- **champion** — the live slot: the artifacts the running bots actually serve (prefix `''` /
  `stock_`). A challenger replaces the champion only through the DM-HLN promotion test.
  Lives in: `predict_now`, `shadow.py`. See also: challenger, `.prev` rollback.
- **champion serving race** — defect H3: after a legacy retrain the bot hot-reloads on manifest
  mtime and serves the NEW LSTM with LAST week's LightGBM and q10 boosters, because the booster
  caches were presence-keyed rather than mtime-keyed. Repaired by the serving cache (packet
  R2C-01). Lives in: `predict_now` + `serving_cache`, described in
  `research/campaign_2026-08/06_signal_model_plan.md` (H3). See also: serving cache, hot-reload.
- **checkpoint soup (`SOUP_K`)** — the uniform average of the K best-validation-loss epoch
  state_dicts inside the fold loop, or of the LAST K epochs in the refit ("SWA tail soup").
  Lives in: `scripts/hypersearch_v2.py`. See also: fold-max checkpoint, refit epoch budget.
- **circuit breaker** — the account-drawdown kill switch that halts new entries (and, per book
  under `BREAKER_PER_BOOK`, measures each book's own P&L). Exits are never breaker-gated.
  Lives in: `order_utils.check_circuit_breaker`, `base_loop`. See also: lockout, cooldown.
- **closed bar / forming bar** — Alpaca labels an hourly bar with its OPEN time and serves it
  while still forming. Training windows contain only closed bars; live must drop the forming one.
  Lives in: `market_data.drop_forming_bar` (`TRADER_CLOSED_BARS_V2`). See also: bad print, M1.
- **command file** — the GUI's only write-path into the running system: an atomically-replaced
  JSON (`pipeline_command.json`, `retrain_trigger.json`) or a flag file, consumed by the
  orchestrator. Lives in: `gui.py`, consumed by `run_pipeline.py`. See also: engine subprocess.
- **conviction journal** — the per-buy journal of the conviction tier and its inputs, written so
  conviction calibration can be measured offline before any sizing change ships.
  Lives in: `base_loop._conviction_tier` + `trade_journal`, gated by
  `strategy_config.CONVICTION_JOURNAL_ENABLED`; read by `decision_report`.
  See also: Stage-0, sizing co-fire, tilt.
- **cooldown** — the per-symbol minimum gap between ENTRIES after a trade (`cooldown_min`, 60 min
  for crypto). Since packet IA-2, cooldown never gates an EXIT — an entry throttle that delayed
  risk reduction would invert the tool's purpose.
  Lives in: `strategy_config.CRYPTO_POLICY['cooldown_min']`, enforced in `base_loop`/`stock_loop`,
  mirrored in `meta_label` as `cooldown_bars`. See also: trade budget, lockout.
- **cost gate** — the admission test that a candidate's predicted edge exceeds
  `MIN_EDGE_MULTIPLE` × round-trip cost, using the live quoted spread. It is the gate the
  backtest, the objective and both loops must price identically.
  Lives in: `fees.required_edge_pct`, called from `order_utils` and both loops.
  See also: admission floor, required edge, two-tier cost, EDGE.
- **cost-multiple ladder** — the family of multiples applied to ONE economic quantity
  (round-trip cost): 1.0× break-even, 2.0× admission, +1.5× stacking. Keeping them in one place
  is why tuning the canonical multiple is safe. Lives in: `fees.py` (module header constants).
- **counterfactual P&L ("saved")** — a veto's replayed net return; "saved" is its NEGATIVE, so a
  veto that dodged a loser shows a positive saved figure. Lives in:
  `decision_report._replay_grouped`. See also: episode dedup, GATE_REASONS, Stage-0.
- **CS rank (cross-sectional rank)** — a signed rank in [−1, 1] computed across the names alive
  in a bar, used both as a model feature family and as an ordering key.
  Lives in: `panel_ranks.add_panel_ranks`, `panel_ranks.cs_size_tilt`.
  See also: panel ranks, CS-IC, near-miss, DV30.
- **CS-IC / rank-IC** — the Spearman rank information coefficient between prediction and realized
  forward return, computed cross-sectionally per bar (CS-IC) or per name over time (IC by name).
  It is the Stage-0 signal-quality statistic, not a P&L statistic.
  Lives in: `ic_diagnostic.rank_ic`, `ic_diagnostic.ic_by_name`, CLI `scripts/ic_by_name.py`.
  See also: Stage-0, stage0 preds, rank gradient.
- **CSCV** — Combinatorially-Symmetric Cross-Validation, the López de Prado construction that
  estimates PBO by recombining performance across submatrix splits.
  Lives in: `validation.pbo_cscv` (library; a coarse 3-fold screen is what hypersearch prints).
  See also: PBO, DSR, selection pressure.
- **`cum_trials`** — cumulative hyperparameter trials spent against an overlapping holdout,
  persisted so it survives deletion of the Optuna study DB. It is the deflation pool that keeps
  DSR honest across retrains. Lives in: `adaptive_config.record_trials`.
  See also: selection pressure, DSR, noisy ratchet, gotcha #2.

## D

- **dark flag / dark artifact** — shipped code that is inert behind a default-OFF flag whose OFF
  path is byte-identical and pinned by a test (dark flag), or an offline output that nothing
  consumes yet (dark artifact, e.g. `learned_lexicon.json`). Promotion of either requires an
  owner decision. Lives in: `strategy_config` + `TRADER_*` env reads; inventory in
  `docs/FLAGS.md`. See also: default-OFF, byte-pin test, evidence gate.
- **decision queue** — the 90 owner-decision items from the completed 2026-07 module review
  (P0/P1/P2). Queue items are decisions for the owner, never auto-fixes.
  Lives in: `research/module_review_2026-07.json`, rendered by the `/decision-queue` skill.
  See also: pending owner ask, kill list.
- **default-OFF** — the shipping rule for anything model-facing or gate-facing: land the code
  behind a flag that is off, with the off-path byte-pinned, and let the owner flip it against
  evidence. Lives in: `strategy_config`, `TRADER_*` env vars; see `docs/FLAGS.md`.
  See also: model-facing vs measurement-only, evidence gate, byte-pin test.
- **degraded mode** — when two or more advisory sizing inputs are unavailable, the composite tilt
  is capped at 0.5×, so missing information can only shrink a position.
  Lives in: `base_loop` (tilt composition). See also: tilt, fail-open / fail-closed.
- **de-leveraging ladder** — the account-drawdown size ladder (10% / 15% / 20% → 0.75 / 0.50 /
  0.25×), measured against a *persisted* high-water mark (Grossman-Zhou 1993).
  Lives in: `drawdown.DRAWDOWN_LADDER`. See also: circuit breaker, HWM.
- **delete-nothing** — the standing repo rule: code, docs and scratch files that stop being used
  are MOVED (`git mv` when tracked) to `archive/` or a dated `research/` subdirectory, with a row
  added to the receiving README — never removed. Verbatim removed code is archived too.
  Lives in: `research/AGENT_CONTEXT.md` (rule 5), `research/campaign_2026-08/08_removed_code.md`.
  See also: archive/, one writer per tree.
- **DM-HLN** — the Diebold-Mariano paired forecast-error test with the Harvey-Leybourne-Newbold
  small-sample correction; the statistic that promotes a challenger over the champion on LIVE
  shadow predictions. It tests forecast errors, so using it to gate POLICY changes is a category
  error and is on the kill list. Lives in: `shadow.dm_hln`.
  See also: champion, challenger, shadow slot, policy gate.
- **DSR (Deflated Sharpe) / `DSR_MIN`** — Bailey-López de Prado's probability that an observed
  Sharpe beats the expected maximum of `n_trials` null configurations, computed on effective
  (overlap-corrected) trade counts. The promotion bar is `DSR_MIN = 0.60`.
  Lives in: `validation.deflated_sharpe_ratio`, `validation.dsr_from_trade_returns`,
  `validation.DSR_MIN`. See also: effective-n, MinTRL, `cum_trials`, PBO.
- **DV30 (`_DV30`)** — trailing 30-day median daily dollar volume. It is exposed to the rank
  layer and then DROPPED: only its cross-sectional rank survives as a feature.
  Lives in: harvest scripts + `panel_ranks`. See also: as-of universe, CS rank.

## E

- **echo gap** — `raw Spearman(s, realized) − partial Spearman(s, realized | pred)`. A large
  positive gap means the LLM is echoing the ML prediction rather than adding information.
  Lives in: `llm_eval.py`. See also: `b2`, `s`, LLM roles.
- **EDGE spread** — the Ardia-Guidotti-Kroencke (2024, JFE) effective-spread estimator computed
  from OHLC bars alone (the `bidask` package). Its trailing per-name output `Eff_Spread_Pct` is
  a PERCENT and is a COST input, deliberately excluded from the model features.
  Lives in: `liquidity` (`from bidask import edge_rolling`), stamped by the harvest scripts and
  `liquidity.stamp_crypto_spreads`. See also: effective vs quoted spread, `FLAT_SPREAD_PCT`.
- **effective-n (`n_eff`)** — the overlap-corrected trade or row count that feeds DSR, computed
  as `Σ ū` per ticker, as a cross-name cluster count, or (v2) as calendar-concurrency uniqueness.
  Using raw n instead inflates every promotion statistic.
  Lives in: `sample_weights.average_uniqueness`, `sample_weights.clustered_effective_n`.
  See also: average uniqueness, Kish design effect, DSR.
- **effective vs quoted spread** — EDGE estimates what trades actually PAID; the live gate prices
  the QUOTED `ask − bid`. The stamp closes per-name dispersion but is a lower bound on the live
  hurdle. Lives in: `liquidity` (module header). See also: EDGE spread, cost gate.
- **embargo** — the gap inserted after a training-fold boundary before validation starts, so no
  validation row's features overlap the training window. Legacy = `seq_len` hours; under
  `TRAINING_REPAIRS_V1` it counts distinct bars.
  Lives in: `scripts/hypersearch_v2.EMBARGO_MULTIPLIER` + `get_walk_forward_folds`; the lexicon
  trainer has its own in `learned_lexicon.purged_folds`. See also: purge, holdout pin.
- **ENB (effective number of bets)** — the equicorrelation aggregation of position stop-risks,
  `sqrt((1−ρ)·Σr² + ρ·(Σr)²)`, which sits between sqrt-sum-of-squares (independent) and the plain
  sum (lockstep). It is what the book risk cap is measured against.
  Lives in: `portfolio.diversified_book_risk`. See also: book risk budget, `MAX_BOOK_RISK_PCT`.
- **endpoint (LLM)** — a user-declared OpenAI-compatible provider entry
  `{name, base_url, api_key, model, free, enabled}` that joins the candidate set for the provider
  chain. Lives in: `llm_config.py`, resolved in `llm_client.resolve_provider_chain`.
  See also: provider chain, selection mode, free-first.
- **engine subprocess** — any child process the GUI launches through its engine-python helpers,
  i.e. under the *Jetson* interpreter and library path, so the GUI's own (lighter) Python never
  imports torch or lightgbm. Lives in: `gui.py` (`_engine_python`, `_engine_env`).
  See also: command file, spawn edge, two-machine reality.
- **`entry_tactic`** — the *realized, ex-post* fill outcome journaled on a buy row (`maker`,
  `maker_reprice`, `maker_partial`, `maker_unknown`, `taker_fallback`, `unfilled`, `marketable`,
  `marketable_bracket`). Do NOT confuse it with `execution_policy.choose_entry_tactic`'s
  *ex-ante* `cross`/`post`/`ladder` vocabulary — different words, different stage.
  Lives in: `order_utils` (journaled), prefix-matched by `fees.realized_crypto_maker_share`.
  See also: maker ladder, capped IOC, implementation shortfall.
- **entry windows** — the intraday ET windows in which stock ENTRIES are allowed (exits are
  never window-gated). Both the loop and the backtest read the same list, so the replay validates
  the policy that trades. Lives in: `strategy_config.STOCK_ENTRY_WINDOWS_ET` +
  `ENTRY_WINDOWS_ENABLED`, applied in `stock_loop._in_entry_window` and `backtest`.
  See also: overnight sleeve, EOD flatten, GATE_REASONS.
- **EOD flatten** — the stock-side exit at the day's last bar (≈15:50 ET live; the last in-day
  bar in labels and replay). Lives in: `policy_exits.eod_mask_from_index`, `stock_loop`.
  See also: exit stack, overnight sleeve, vertical barrier.
- **episode dedup** — collapsing a journal to one row per (symbol, reason, calendar-day) before
  counterfactual replay, so a symbol skipped on every cycle of a day does not emit ~24
  overlapping replays. Lives in: `decision_report._replay_grouped`.
  See also: counterfactual P&L, GATE_REASONS.
- **evidence gate** — an activation flip that is blocked until a named in-repo instrument has
  produced its measurement. The runbook states, per flip, which report must exist and what it
  must show. Lives in: `research/campaign_2026-08/03_jetson_runbook.md`.
  See also: default-OFF, Stage-0, pending owner ask.
- **exit code 3** — the promotion-gate failure signal from the policy replay. It means one of:
  the champion was rolled back to `.prev`; a challenger is HELD (no rollback); or an established
  champion is held because no challenger exists. Lives in: `backtest.py` (module docstring +
  `--gate`). See also: policy gate, `.prev` rollback, challenger.
- **exit kernel (`policy_exits.exit_walk`)** — the single Numba kernel that walks an entry
  forward bar by bar and returns the realized exit. It is the ONE implementation shared by the
  backtester, the harvest's triple-barrier labels and the meta-labeler, which is what guarantees
  label semantics == backtest == live. Lives in: `policy_exits.exit_walk` →
  `policy_exits._exit_walk_kernel`. See also: live mirror, exit stack, short mirror.
- **exit stack** — the ORDERED barrier check the kernel runs at every bar's close: hard stop →
  take-profit → high-water-mark trailing → signal exit → EOD flatten → vertical barrier. Order
  matters and is pinned by tests. Lives in: `policy_exits._exit_walk_kernel`,
  `policy_exits.REASON_NAMES` (codes 0..6). See also: exit kernel, reason code, HWM.
- **extract-and-exec test** — a Mac-runnable test technique for modules that cannot be imported
  (heavy deps): pull one method's source text out of the file, `compile`/`exec` it in a stub
  namespace, and call it for real. Stronger than a source-text assertion, weaker than a true
  import. Lives in: `tests/test_review_b01.py` (`_extract_method`).
  See also: source-text contract test, byte-pin test, Mac-runnable.

## F

- **fail-open / fail-closed** — the two error contracts. Advisory inputs FAIL OPEN: an LLM,
  sentiment, EDGAR or SPY-trend failure yields a neutral result and can never block a trade.
  Live trading paths FAIL CLOSED: a missing prediction or quote means no entry.
  Lives in: `llm_analyst`/`sentiment` (open), `base_loop`/`stock_loop` entry paths (closed).
  See also: degraded mode, LLM veto.
- **fan-in / fan-out** — the count of distinct non-test repo modules that import a module
  (fan-in) or that it imports (fan-out). Test importers are counted separately.
  Lives in: `docs/graphs/import_graph.json`, generated by `scripts/repo_graph.py`.
  See also: eager vs lazy import, spawn edge.
- **fee sweep (λ\*)** — replaying the policy at a ladder of fee multipliers and reporting λ\*,
  the multiplier at which net P&L crosses zero. λ\* is the cost headroom: how wrong the cost
  model can be before the edge dies. Lives in: `backtest.breakeven_fee_mult`, CLI
  `python backtest.py --fee-sweep`. See also: round-trip cost, cost gate, naive baseline.
- **`FIXED_HOLDOUT_DAYS` (holdout pin)** — pinning the search/holdout boundary to a fixed number
  of calendar days instead of the floating `HOLDOUT_FRACTION` (0.12) row quantile, so a training
  window A/B changes only the window and not the thing being scored.
  Lives in: `strategy_config.FIXED_HOLDOUT_DAYS` (default `None`; env
  `TRADER_FIXED_HOLDOUT_DAYS` wins), consumed by `scripts/hypersearch_v2.py`.
  See also: window A/B, purge, certificate, selection pressure.
- **flag family** — a named group of flags landed together by one packet, sharing a rationale and
  a runbook flip order: `c26` (2026-08 campaign), `r2c` (R2 signal wave), `IA-1..4`
  (decision-influence audit), plus the `TRADER_*` env family. The inventory of every flag and its
  default lives in `docs/FLAGS.md` — never quote a count from prose.
  Lives in: `strategy_config.py` + `TRADER_*` env reads. See also: default-OFF, `TRADER_*`.
- **`FLAT_SPREAD_PCT`** — the flat per-asset spread fallback used when no per-name EDGE stamp is
  available. It has sibling copies (the backtest's `SPREAD_PCT`, an inline literal in the meta
  row generator) that tests keep in sync. Lives in: `fees.FLAT_SPREAD_PCT`.
  See also: EDGE spread, two-tier cost, round-trip cost.
- **fold-max checkpoint (winner's curse)** — the legacy shipping artifact: the souped weights of
  the single BEST fold. Because the fold was chosen by its own score, its Sharpe overstates
  deployable edge. Lives in: `scripts/hypersearch_v2.py`. See also: checkpoint soup, DSR, PBO.
- **`forward_bars` / `fb`** — the label horizon in bars. The adaptive search space is
  `[12, 18, 24, 32, 48]`; it is also the vertical-barrier distance.
  Lives in: harvest scripts, `scripts/hypersearch_v2.py`, `policy_exits` (`max_hold`).
  See also: triple-barrier, vertical barrier, horizon transfer.
- **free-first** — the operating posture of routing LLM traffic to qualified free providers
  before paid ones; expressed through `selection_mode` values `free-only` / `best-free`.
  Lives in: `llm_config.py`, `llm_client.resolve_provider_chain`.
  See also: selection mode, provider chain, endpoint, tier.
- **FR-xx** — frontier-research candidate IDs (FR-01…FR-20) from the 2025-26 literature survey;
  several became R2C build packets (e.g. FR-01 fixed holdout, FR-16 fee sweep).
  Lives in: `research/campaign_2026-08/05_frontier_research.md`. See also: packet ID, r2c.

## G

- **gap-through** — the loss BEYOND a resting stop when an overnight gap opens past it,
  quantified as `E[max(0, |gap| − stop_dist)]`. Lives in: `gap_audit.gap_stats`.
  See also: overnight sleeve, stop-risk.
- **GATE-1 / GATE-2** — GATE-1 is the cross-book stop-risk *measurement* journal (does the
  account actually stack toward `ACCOUNT_RISK_CAP`?). GATE-2 is the concurrent-equity promotion
  check the repo previously lacked. Lives in: `risk_budget.ACCOUNT_RISK_CAP` (GATE-1 target),
  `risk_budget.simulate_two_books` (GATE-2). See also: book risk budget, ENB.
- **`GATE_REASONS`** — the canonical taxonomy of skip reasons that the gate-attribution report
  prices (`llm_veto`, `meta_veto`, `q10_tail_veto`, `vix_halt`, `vix25_block`,
  `macro_standdown`, `rank_near_miss`, …). Entries are retained even after a gate is removed, so
  historical rows stay priceable. Lives in: `decision_report.GATE_REASONS`.
  See also: counterfactual P&L, episode dedup, Stage-0.
- **golden fingerprint** — a test that hashes the full feature frame produced from a fixed
  synthetic input, so ANY change to any feature's values fails loudly. It is the primary
  train/serve-parity tripwire on the Mac.
  Lives in: `tests/test_indicators_parity.py::test_compute_stock_features_golden_fingerprint`.
  See also: train/serve parity, byte-pin test, gotcha #2.
- **gotcha #2** — the retrain hygiene rule: any objective, feature or cost change invalidates the
  stored Optuna scores, so delete `v2_study.db` + `stock_v2_study.db` and reset the adaptive
  `best_score`/`cum_trials`, bundling everything into ONE retrain. Cited by name from many code
  sites — do not renumber the list. Lives in: `CLAUDE.md` § Gotchas.
  See also: study DB, `cum_trials`, model-facing vs measurement-only.

- **guarded heavy dep** — a heavy import (torch, lightgbm, numba, sklearn …) wrapped in
  `try/except ImportError` with a pure fallback, or placed inside a function body. Guarded means
  the module still imports on the Mac; a module can still be *transitively un-importable* when
  its own imports are clean but an eager internal chain reaches an unguarded heavy dep.
  Lives in: `indicators.py`, `policy_exits.py`, `model_lgb.py`, `meta_label.py` (numba/lightgbm
  fallbacks). See also: Mac-runnable, Jetson-gated, import graph, source-text contract test.
## H

- **HAC alpha** — the intercept of the equity-vs-benchmark regression with Newey-West
  heteroskedasticity-and-autocorrelation-consistent standard errors (automatic plug-in lag by
  default). Its t-stat is the honest "is there alpha" reading.
  Lives in: `beta_ledger.ols_hac`. See also: beta ledger, AKL.
- **HAR-RV** — the heterogeneous autoregressive realized-volatility forecast built on daily
  realized range (Parkinson RRV), used for sizing sigma. Forecasting vol for SIZING is live and
  first-class; betting on vol as an alpha source and strategy-level vol-targeting are both on the
  kill list — do not conflate. Lives in: `volatility.har_forecast_sigma`,
  `volatility.daily_realized_range`, flag `strategy_config.HAR_VOL_ENABLED`.
  See also: realized range, vol target, kill list.
- **harvest** — the data-collection stage: one year of hourly OHLCV per name plus every computed
  feature, written to the training panel. It is where PIT discipline, the as-of mask and the
  triple-barrier labels are applied. Lives in: `scripts/harvest_crypto_data.py`,
  `scripts/harvest_stock_data.py`. See also: training_data parquet, raw sidecar, PIT.
- **honest floor** — the smallest sample size n at which the meta-labeler's AUC is within 0.01 of
  its plateau AND its veto flip-rate is under 10%; i.e. the point where the meta gate stops being
  noise. Lives in: `meta_curve` (`honest_floor`), CLI `scripts/meta_learning_curve.py`.
  See also: meta gate, Stage-0.
- **horizon transfer** — the diagnostic that asks whether edge at horizon δ survives to horizon
  Δ, comparing `corr(r^δ, r^Δ)` against the IID null `sqrt(δ/Δ)` with weekly block-bootstrap
  standard errors. Lives in: `horizon_transfer.transfer_matrix`, `horizon_transfer.iid_null`;
  CLI `scripts/horizon_transfer_report.py`. See also: `forward_bars`, naive baseline, Stage-0.
- **hot-reload** — the live loops picking up newly trained artifacts by manifest mtime WITHOUT
  restarting, so the bots never stop across a weekly retrain. Its failure mode is the champion
  serving race. Lives in: `base_loop._hot_reload_check`.
  See also: champion serving race, serving cache, prediction cache.
- **HTB risk score** — a supply-side (market-cap) proxy for hard-to-borrow status, built to
  exclude-when-uncertain rather than assume borrowability.
  Lives in: `borrow_proxy`. See also: regime-dated borrow, short mirror.
- **`HURST_ON_RETURNS`** — the flag that feeds RETURNS (the correct R/S input) rather than price
  LEVELS to the Hurst exponent. Default off because flipping it moves a feature's values, which
  is model-facing and triggers gotcha #2.
  Lives in: `indicator_config.HURST_ON_RETURNS`, read by `indicators`.
  See also: gotcha #2, model-facing vs measurement-only, IA-1..IA-4.
- **HWM / LWM** — the high-water mark (long) or low-water mark (short mirror) that the percentage
  trailing stop rides. Note the documented denominator asymmetry: the kernel scales the trail by
  ATR/entry while `base_loop` scales by ATR/hwm.
  Lives in: `policy_exits._exit_walk_kernel`, `base_loop._desired_stop_for`.
  See also: exit stack, de-leveraging ladder.
- **hysteresis (Schmitt trigger)** — asymmetric enter/exit thresholds that prevent whipsaw: a
  state turns on at one level and off at a lower one. Used for the VIX ladder (enter 25/35, exit
  22/31), the crypto trend state and live universe membership.
  Lives in: `macro_indicators._VIX_TIER_ENTER` / `_VIX_TIER_EXIT`,
  `crypto_trend.hysteresis_state`. See also: VIX ladder, regime.

## I

- **IA-1 … IA-4** — the four packets of the 2026-08 decision-influence audit: IA-1 owner-ruled
  removals (pseudo-CAPE, two provably-dead branches), IA-2 undesigned-behavior and safety fixes,
  IA-3 influence journaling, IA-4 the flag family that makes each remaining influence
  individually switchable. Lives in: `research/campaign_2026-08/07_decision_influences.md`,
  `strategy_config` (IA-4 flag block), `tests/test_ia{1,2,3,4}_*.py`.
  See also: flag family, packet ID, kill list.
- **implementation shortfall** — the realized `fill_price` measured against the `decision_price`
  (the quote midpoint at decision time), signed so positive ALWAYS means worse.
  Lives in: `execution_report.run_report`. See also: `entry_tactic`, maker ladder.
- **import graph (eager vs lazy)** — eager = a module-level import not inside `try:`; lazy = an
  import inside a function body. This repo leans hard on lazy imports both to break cycles and to
  keep Jetson import-time memory down, which is why "module X imports torch" is rarely the whole
  story. Lives in: `docs/graphs/import_graph.json`, `scripts/repo_graph.py`.
  See also: fan-in / fan-out, guarded heavy dep, spawn edge.

## J

- **Jetson-gated** — work that cannot be done on the dev Mac because it needs torch, lightgbm,
  optuna, joblib, numba, sklearn, dotenv, real journals or parquet. Write it, unit-test the pure
  parts, and hand it to the owner to run. Lives in: `CLAUDE.md` § Two-machine reality.
  See also: two-machine reality, Mac-runnable, guarded heavy dep.
- **journal** — an append-only JSONL record of what the system decided and did (decisions, buys,
  exits, sizing multipliers, vetoes). Journals are the substrate every Stage-0 measurement reads;
  they live outside the repo under `journals/`.
  Lives in: `trade_journal` (writer), `journal_stats` / `decision_report` (readers).
  See also: Stage-0, conviction journal, GATE_REASONS.

## K

- **Kelly cap** — the fractional-Kelly ceiling (`KELLY_CAP = 0.25`, MacLean-Thorp-Ziemba) applied
  to the edge-derived size multiplier, so an over-confident win-rate estimate cannot lever the
  book. Lives in: `strategy_config.KELLY_CAP`, applied in `base_loop._compute_position_size`.
  See also: risk-per-trade, tilt, ENB.
- **kill list** — the ONE canonical do-not-rebuild list, consolidating every KILLED / REJECTED /
  refuted verdict from the waves, reviews and literature research, with source tags. Every
  research or build agent MUST check it (including its PENDING OWNER ASKS appendix) before
  proposing anything; an entry leaves only by explicit owner decision. It also carries a
  "commonly confused survivors" section so a live feature is not killed by association.
  Lives in: `research/KILL_LIST.md`. See also: pending owner ask, wave, decision queue.
- **Kish design effect** — `deff = 1 + (G−1)·ρ̄`, the standard-error inflation from clustered
  observations. Available as an optional divisor on the blend fit; it is an ALTERNATIVE to, not a
  companion of, the uniqueness effective-n — stacking both double-counts (gotcha #4).
  Lives in: `blend_fit` (`kish_divisor`, `rho_bar`). See also: effective-n, average uniqueness.

## L

- **λ\* (lambda-star)** — see fee sweep.
- **lead/lag** — the per-feature diagnostic separating PREDICTIVE information content (IC at
  1–48h ahead, overlap-adjusted, FDR-controlled) from merely REACTIVE coupling to recent price,
  plus redundancy clusters and exact duplicate columns. Measurement-only.
  Lives in: `indicator_leadlag.py` (CLI `python indicator_leadlag.py --data F`).
  See also: CS-IC, Stage-0, kill list.
- **learned lexicon** — the data-fitted sentiment word/phrase weights trained with purged,
  embargoed walk-forward folds, as an alternative to the hand-written static lexicon. It is DARK
  by construction: it writes its artifacts and nothing consumes them until an owner decision.
  Lives in: `learned_lexicon.py` (`purged_folds`), CLI `scripts/train_lexicon.py`.
  See also: sentiment lexicon, dark artifact, sentiment triple-count.
- **live mirror** — the property that `policy_exits.exit_walk(side=+1)` IS the long path the live
  loops implement, so labels, backtest and production share one exit definition. `side=-1` is the
  offline short mirror only. Lives in: `policy_exits.exit_walk`.
  See also: exit kernel, short mirror, train/serve parity.
- **`llm_mult`** — the sizing multiplier derived from the LLM conviction score as `0.5 + s`
  (range 0.65–1.5), folded into the composite tilt.
  Lives in: `base_loop` (tilt composition), from `llm_analyst`. See also: `s`, tilt, size tilt.
- **LLM roles** — the four distinct jobs the LLM stack performs, each separately configurable:
  the **veto** gate, the **size tilt** (`llm_mult`), the shadow **advisor-v2 dossier**, and the
  batched **backfill** scorer. Only veto and size tilt touch a live trade.
  Lives in: `llm_analyst`, `llm_client`, `llm_config.py`.
  See also: LLM veto, advisor-v2, backfill role, fail-open / fail-closed.
- **LLM veto** — `s < LLM_VETO_THRESHOLD` (0.15) blocks a new entry immediately, and liquidates
  an open position only on the SECOND consecutive vetoing analysis. Fail-open: any error yields a
  neutral score. Lives in: `trading_utils.LLM_VETO_THRESHOLD`, enforced in
  `base_loop._execute_llm_veto_sells` and `llm_analyst`.
  See also: veto strike, `s`, fail-open / fail-closed.
- **lockout** — the post-hard-stop cooling-off period during which a symbol cannot be re-entered
  (`lockout_hours`, 24 for crypto). Since IA-2 the lockout state file is per-book, so the two
  books cannot clobber each other's state.
  Lives in: `strategy_config.CRYPTO_POLICY['lockout_hours']`, state in
  `{prefix}_hard_stop_lockout.json`. See also: cooldown, circuit breaker, trade budget.

## M

- **Mac-runnable** — provable on this dev Mac with numpy/pandas/scipy/statsmodels/bidask and
  synthetic data. Everything else is Jetson-gated. Numba-decorated repo code has pure-Python
  fallbacks, which is what keeps the kernels Mac-runnable.
  Lives in: `CLAUDE.md` § Two-machine reality. See also: Jetson-gated, extract-and-exec test.
- **macro stand-down** — a scheduled-macro window (FOMC, CPI) in which new entries are blocked
  while exits keep running. Fail-open, and it raises a loud alarm when the static event table is
  exhausted. Lives in: `macro_calendar.macro_standdown`, consumed by the loops and
  `llm_analyst`. See also: VIX ladder, GATE_REASONS, entry windows.
- **maker ladder (bid-join)** — the crypto entry tactic: rest a limit at the live bid, reprice
  once, then escalate the remainder. Alpaca crypto is 15 bps maker vs 25 bps taker per side, plus
  the saved half-spread. Lives in: `order_utils.place_maker_buy`, flag
  `strategy_config.MAKER_ENTRIES_ENABLED`. See also: `entry_tactic`, capped IOC, CONFIRM-ONLY.
- **`MAX_BOOK_RISK_PCT`** — the correlation-adjusted stop-risk cap per book (0.025). It is
  measured on the ENB aggregate, not the naive sum of position risks.
  Lives in: `strategy_config.MAX_BOOK_RISK_PCT`. See also: ENB, book risk budget, GATE-1 / GATE-2.
- **meta-label / meta gate** — the secondary classifier that estimates `p = P(this trade profits
  net of costs)` given the primary model already said "enter". Two effects: **veto** when
  `p < META_VETO_PROB = 0.30`, and **size tilt** `clip(2p, 0.6, 1.3)`.
  Lives in: `meta_label.py` (`META_VETO_PROB`), trained on the OOF pack.
  See also: OOF pack, honest floor, size tilt, q10 veto.
- **meta triple** — the three artifacts a trained meta-labeler ships as a unit:
  `{p}meta_model.txt`, `{p}meta_calib.pkl`, `{p}meta_meta.json` (plus `.staged` / `.prev` /
  `.stale` variants). A **refusal** — the labeler declining to ship — is recorded in
  `{p}meta_refused.json`. Lives in: `meta_label.py`. See also: meta gate, `.prev` rollback.
- **MinTax** — optimistic specific-identification lot selection (losses first, then long-term
  gains, then short-term; highest cost basis within each tier). It is NOT the IRS-default FIFO.
  Lives in: `tax_lots.py`. See also: journal.
- **MinTRL** — the minimum track record length: how many effective trades are needed before the
  observed Sharpe could clear the DSR bar at all.
  Lives in: `validation` (reported alongside DSR). See also: DSR, effective-n.
- **model-deploy fence** — on a manifest content-hash change, truncate the prediction history and
  reset the drift action streak, so a new model is never judged against the old model's
  distribution. Lives in: `monitor_drift` (`_manifest_hash`, history reset).
  See also: PSI, hot-reload.
- **model-facing vs measurement-only** — the deployment dichotomy. Model-facing (anything that
  changes a feature value, a prediction, or a gate decision) ships ONLY as default-OFF behind a
  byte-pinned flag, then through challenger → shadow → DM-HLN. Measurement-only (journals,
  reports, offline research kernels) ships directly.
  Lives in: `CLAUDE.md` § Conventions, `research/AGENT_CONTEXT.md` rule 2.
  See also: default-OFF, evidence gate, Stage-0.

## N

- **naive baseline (Nagel)** — the deliberately trivial EWMA-momentum-over-trailing-vol
  one-liner, scored on the SAME stage-0 rows as the blend. It is a falsification test: if the
  blend cannot beat it on purged IC and DSR, the complexity is not earning its keep.
  Lives in: `naive_baseline.naive_signal`, CLI `scripts/naive_vs_blend.py`.
  See also: stage0 preds, horizon transfer, fee sweep.
- **`name_class`** — the `mega` / `mid` / `spec` liquidity tier that execution policy would key
  off. The seed table does not exist yet, so everything currently resolves to `mid`.
  Lives in: `execution_policy`. See also: orphan, `entry_tactic`.
- **near-miss** — a candidate that failed only the top-N rank cut, journaled so the cost of the
  ordering rule itself can be measured. Lives in: `stock_loop` (top-N near-miss substrate),
  taxonomy entry `rank_near_miss` in `decision_report.GATE_REASONS`.
  See also: rank gradient, CS rank, admitted-k.
- **neutral fill** — the "no data reads as neutral" convention: a missing archive feature, CS
  rank or live injection becomes 0.0 (0.5 for `Pos_Range_*`), so train and serve agree on missing
  data instead of one side dropping the row. Lives in: harvest scripts + `predict_now` injection.
  See also: warmup fill, train/serve parity, PIT.
- **NEW vs DISAPPEARED** — how `ab_check` reports its diff. NEW failing names are regressions and
  block; DISAPPEARED names (baseline failures that now pass) are a bonus, not a problem.
  Lives in: `scripts/ab_check.sh`. See also: ab_check, baseline_failures.txt.
- **noisy ratchet (Thresholdout)** — the Dwork-2015-shaped acceptance rule for the persisted best
  score: accept a new best only if `new > stored + 2σ + Laplace(σ/2)`. It stops the adaptive
  search from ratcheting on noise. Lives in: `adaptive_config.noisy_ratchet`.
  See also: selection pressure, `cum_trials`, DSR.

## O

- **`OBJECTIVE_LONG_ONLY`** — scores only the deployable long side in the hypersearch objective.
  Default off; flipping it IS an objective change and therefore a gotcha #2 event.
  Lives in: `strategy_config.OBJECTIVE_LONG_ONLY`. See also: gotcha #2, short mirror.
- **one writer per tree** — the concurrency rule for agents: another session may be working the
  same checkout, so re-read a file immediately before editing it, never `git stash` a shared tree
  for an A/B (reconstruct baselines with `git show HEAD:<file>`), and edit only the files you own.
  Lives in: `research/AGENT_CONTEXT.md` rule 7. See also: delete-nothing, ab_check.
- **OOF pack (`{p}oof_preds.npz`)** — the winner's purged walk-forward validation-fold
  predictions, fingerprinted to the champion manifest (`saved_at`, `score`). It is the meta
  labeler's training input, which is why the fingerprint matters.
  Lives in: written by `scripts/hypersearch_v2.py`, read by `meta_label.py`.
  See also: meta gate, purge, certificate.
- **Optuna trial** — one hyperparameter configuration evaluated by the TPE search. Trials are the
  unit of selection pressure: their cumulative count is the deflation pool for DSR.
  Lives in: `scripts/hypersearch_v2.py`, counted in `adaptive_config`.
  See also: study DB, `cum_trials`, selection pressure, gotcha #2.
- **orphan (not-wired)** — a shipped, tested kernel with no production caller. These are DORMANT,
  not dead: they are kill-list survivors awaiting an owner wiring decision (e.g.
  `squeeze_features`, `crypto_trend`, `basis_archive`, `panel_ranks.add_crypto_panel_ranks`,
  `bet_sizing`, `portfolio_backtest`). Lives in: the named modules; each self-documents its
  status. See also: dark flag, kill list, pending owner ask.
- **overnight sleeve** — the exception to EOD flatten: up to `OVERNIGHT_SLEEVE_MAX_POSITIONS` (2)
  stock positions may be held overnight, capped per position at 5% of equity and only while still
  predicted up. It fails CLOSED — missing information means flatten.
  Lives in: `strategy_config.OVERNIGHT_SLEEVE_*`, honored by `policy_exits` and `stock_loop`.
  See also: EOD flatten, gap-through, entry windows.

## P

- **packet ID** — the stable identifier of a unit of campaign work, used as the citation key in
  code comments and test names. Families: `D01–D40` defects and `B02–B24` builds (2026-08
  campaign), `FR-01..20` frontier candidates, `H1–H3 / M1–M4 / L1–L9` signal-model defects,
  `R2C-01..08` build packets, `IA-1..4` influence packets.
  Lives in: `research/campaign_2026-08/`. See also: c26, r2c, FR-xx, IA-1..IA-4.
- **panel ranks** — the cross-sectional rank feature layer computed per bar across the names
  alive in that bar, plus the live-serving equivalents used for ordering and size tilting.
  Lives in: `panel_ranks.py` (`add_panel_ranks`, `cs_size_tilt`).
  See also: CS rank, CS-IC, near-miss, DV30.
- **PBO (probability of backtest overfitting)** — the probability that the configuration selected
  as best in-sample underperforms the median out-of-sample. Complements DSR: DSR asks "is this
  Sharpe real", PBO asks "did selection manufacture it".
  Lives in: `validation.pbo_cscv`; CLI `scripts/cscv_audit.py`. See also: CSCV, DSR.
- **pending owner ask** — a formal request to REOPEN a kill-list entry. Filing one changes
  nothing: the item stays killed until the owner rules, and the ruling is recorded inline in the
  kill list. Lives in: `research/KILL_LIST.md` § PENDING OWNER ASKS.
  See also: kill list, decision queue, evidence gate.
- **phase** — `run_pipeline`'s unit of work: a dict carrying a `cmd` command vector, built by the
  harvest/training phase builders and run serially. Phases are how the orchestrator sequences
  harvest → train → gate without holding heavy imports itself.
  Lives in: `run_pipeline._build_harvest_phases`, `_build_training_phases`, `_run_training`.
  See also: spawn edge, engine subprocess, command file.
- **PIT (point-in-time)** — the non-negotiable discipline that a feature value at bar *t* uses
  only information a live system could have had at *t*. All features strictly trailing;
  sentiment and short interest lagged to publication date; borrow cost regime-dated; universe
  membership as-of. Lives in: enforced across `indicators`, `sentiment_history`, `short_cost`,
  harvest scripts. See also: as-of universe, survivorship, train/serve parity, regime-dated borrow.
- **policy gate (promotion gate)** — the replay of REAL entries and exits over recent data that a
  model must pass before it is allowed to keep trading: `n ≥ 10`, Sharpe > 0, `DSR ≥ DSR_MIN`.
  Lives in: `backtest.py --gate`; sidecar `{slot}_policy_gate.json`.
  See also: exit code 3, `.prev` rollback, challenger, DM-HLN.
- **prediction cache** — two different things that share the name. (a) `prediction_cache.py`: a
  bar-keyed in-process MEMO that skips re-running a bit-identical inference while no new bar has
  closed (the largest source of idle Jetson CPU), invalidated by a model token on hot-reload.
  (b) `{book}_predictions.json`: the published snapshot the loop writes for the GUI and
  measurement tools — a publication artifact, not a serving path.
  Lives in: `prediction_cache.bar_key` (memo), `crypto_loop` / `stock_loop` (JSON).
  See also: serving cache, hot-reload, stage0 preds.
- **prefix** — the artifact filename stem that selects a book and slot: `''` (crypto champion),
  `stock_`, `challenger_`, `stock_challenger_`. Note `hypersearch --prefix stock` becomes
  `stock_`, while some scripts' `--prefix` already carries the underscore.
  Lives in: `data_utils`, `scripts/hypersearch_v2.py`. See also: book, champion, challenger.
- **preset** — a named feature-column subset (`standard`, `stationary`, `stationary_lean`). The
  production pipeline hardcodes `stationary`; the persisted preset and the GUI picker are
  advisory. Hypersearch intersects the preset with the columns actually present in the loaded
  panel — there is no asset-type filter.
  Lives in: `indicator_config.py`, selected via `--preset` in `scripts/hypersearch_v2.py`.
  See also: harvest, training_data parquet, gotcha #2.
- **`.prev` rollback** — the promotion gate's undo: rename the previous generation of all
  artifact legs back into the live slot. Lives in: `backtest.restore_previous_model`,
  `backtest.ARTIFACT_SUFFIXES`. See also: exit code 3, policy gate, meta triple.
- **prompt version** — the `analyst-v1` / `advisor-v2` tag journaled on every LLM row, so a
  prompt change is a segmentable measured event rather than a silent regime shift.
  Lives in: `llm_analyst`. See also: advisor-v2, LLM roles, evidence hash.
- **provider chain** — the ordered candidate list of LLM providers/models for a role, built from
  the configured selection mode plus declared endpoints, with cross-provider fallback. All
  providers are schema-enforced (Gemini responseSchema, Claude forced tool use, OpenAI strict
  structured outputs). Lives in: `llm_client.resolve_provider_chain`, `llm_config.py`.
  See also: selection mode, free-first, endpoint, tier.
- **PSI (Population Stability Index)** — `Σ_b (live_b − ref_b)·ln(live_b/ref_b)` over the
  holdout's prediction deciles: < 0.10 stable, 0.10–0.25 warn, > 0.25 act. The daily drift check
  that decides whether live predictions still look like the certified ones.
  Lives in: `monitor_drift.compute_psi`; a feature-side twin is
  `scripts/funding_drift_audit.psi_from_train_deciles`. See also: model-deploy fence, Stage-0.
- **purge** — dropping a training row whose LABEL window ends after the fold or holdout boundary,
  so no training label can peek across the split. Always paired with an embargo.
  Lives in: `scripts/hypersearch_v2.get_walk_forward_folds` (`purge_val_labels`,
  `all_label_times`). See also: embargo, holdout pin, OOF pack, effective-n.

## Q

- **q10 booster / q10 veto** — a LightGBM 10th-percentile quantile regressor estimating the LEFT
  TAIL of this state's return. A long entry is vetoed when `q10` falls below a floor calibrated as
  the 15th percentile of q10 on the fold-validation slice. Distinct from the meta gate: q10 vetoes
  on tail risk, meta on net-of-cost profitability.
  Lives in: trained in `scripts/hypersearch_v2.py`, served in `predict_now` (`_q10_models`),
  taxonomy entry `q10_tail_veto`. See also: meta gate, certificate, blend.

## R

- **r2c (R2C-01 … R2C-08)** — the R2 signal-wave build packets: serving cache integrity (01),
  blend/certificate coherence (02), LGB full refit + honest q10 floor (03), training-loop repairs
  and seed plumbing (04), fixed holdout + window A/B (05), the measurement-kernel suite (06),
  rank-IC certificate lines + retrain-gain ledger (07), and the breakeven fee sweep (08).
  Lives in: `research/campaign_2026-08/06_signal_model_plan.md`, `tests/test_r2c_*.py`.
  See also: packet ID, c26, holdout pin, serving cache.
- **rank (top-N)** — the per-cycle ordering of candidates by prediction. The two books treat it
  differently and this is the single biggest structural difference between them: the stock loop
  iterates ONLY `TOP_N = 7`, so rank is a hard admission cut, while the crypto loop iterates the
  full universe and rank is annotation only. Lives in: `stock_loop.TOP_N`, `base_loop._execute_buys`.
  See also: near-miss, rank gradient, CS rank, panel ranks.
- **rank gradient** — the Stage-0 test of whether prediction RANK carries monotone economic
  value: compare mean net return of ranks 1–3 against ranks 6–7. It is the gate for any
  concentration or edge-Kelly change. Lives in: `rank_gradient.rank_gradient_from_panel`,
  `rank_gradient.rank_gradient_verdict`; CLI `scripts/rank_gradient_report.py`.
  See also: near-miss, CS-IC, stage0 preds.
- **raw sidecar (`raw_ohlcv`)** — the store of venue bars exactly as fetched, BEFORE features, so
  feature warmups can be recomputed from full raw history instead of eating the store's head.
  Lives in: `data_utils._RAW_STEMS` (`raw_ohlcv`, `stock_raw_ohlcv`), flag `TRADER_RAW_SIDECAR`.
  See also: harvest, training_data parquet, warmup fill.
- **reason code** — the integer 0..6 the exit kernel returns naming which barrier fired, mapped
  to text for labels, replay and journals. Lives in: `policy_exits.REASON_NAMES`.
  See also: exit stack, triple-barrier.
- **regime** — a coarse market-state label used for SIZING and diagnostics, never as a standalone
  signal: the VIX ladder tier, the macro regime object, the crypto trend state, and the (unwired,
  kill-list-pending) HMM bull/bear/neutral layer.
  Lives in: `macro_indicators.get_macro_regime`, `macro_indicators.vix_tier_mult_v2`,
  `crypto_trend`, `regime_detector`. See also: bear / bull, VIX ladder, hysteresis.
- **regime-dated borrow** — keying borrow cost to a dated regime boundary
  (`BORROW_REGIME_START = 2025-10-01`), because granting today's $0 easy-to-borrow rate to a 2024
  simulation is itself look-ahead. Lives in: `short_cost.BORROW_REGIME_START`.
  See also: PIT, HTB risk score.
- **required edge** — the percent-of-notional predicted return a candidate must show to clear the
  cost gate: `MIN_EDGE_MULTIPLE × round_trip_cost_pct(asset, spread)`.
  Lives in: `fees.required_edge_pct`. See also: cost gate, admission floor, round-trip cost.
- **retrain ledger** — the measurement that answers "did this retrain actually help?": score the
  incumbent stack and the new stack on the same recent purged rows and record the paired gain.
  Lives in: `retrain_ledger.record_retrain_gain`, `retrain_ledger.paired_scores`.
  See also: window A/B, DM-HLN, Stage-0.
- **risk-per-trade** — `RISK_PCT_PER_TRADE = 0.005`: 0.5% of equity at risk to the stop on a new
  position, before any tilt or Kelly adjustment. The base unit all sizing multiplies.
  Lives in: `strategy_config.RISK_PCT_PER_TRADE`. See also: stop-risk, Kelly cap, tilt.
- **round-trip cost** — the total percent-of-notional cost of an entry plus exit: venue fees, one
  full spread, and optionally a square-root impact term. Every gate prices it from this one
  function. Lives in: `fees.round_trip_cost_pct`. See also: cost gate, fee sweep, EDGE spread.

## S

- **`s`** — the LLM analyst's conviction score in [0.0, 1.0]. It is the ONLY LLM output that
  gates or sizes a trade. Lives in: `llm_analyst`, consumed by `base_loop`.
  See also: LLM veto, `llm_mult`, `b2`, echo gap.
- **selection mode** — how the provider chain is ordered: `auto`, `single`, `free-only`,
  `best-free`. Lives in: `llm_config.py` (`selection_mode`), honored by
  `llm_client.resolve_provider_chain`. See also: provider chain, free-first, endpoint.
- **selection pressure** — the cumulative number of configurations a winner was chosen from,
  which is exactly the deflation pool DSR needs. Tracked persistently so it survives study-DB
  deletion. Lives in: `adaptive_config.record_trials` (`cum_trials`).
  See also: `cum_trials`, DSR, PBO, noisy ratchet, gotcha #2.
- **sentiment lexicon (learned vs static)** — two implementations of the same job: the
  hand-written phrase/word lists that score articles today, and the purged-walk-forward learned
  weights that are built but dark. Note a known static-lexicon defect: phase-2 re-tokenizes the
  full text without masking phase-1 phrase matches, so `('rate cut', +1.0)` is cancelled by
  `'cut'`. Lives in: `sentiment.py` (static), `learned_lexicon.py` (learned).
  See also: learned lexicon, sentiment triple-count, dark artifact.
- **sentiment triple-count** — the structural warning that one piece of symbol news can influence
  size three times: as a trained model feature (`Daily_Sentiment`), as the sentiment gate
  multiplier, and as prompt evidence behind `llm_mult`.
  Lives in: `sentiment.sentiment_gate`, harvest `Daily_Sentiment`, `llm_analyst` prompt.
  See also: tilt, sizing co-fire, `llm_mult`.
- **serving cache** — the mtime-keyed cache that serves a loaded booster only while the file it
  came from is unchanged, closing the champion serving race (packet R2C-01).
  Lives in: `serving_cache.cache_get`, `serving_cache.stat_key`, used by `predict_now`.
  See also: champion serving race, hot-reload, prediction cache.
- **shadow slot** — the parallel artifact slot a challenger is saved into (`--shadow`) so it can
  make LIVE predictions alongside the champion without trading. Accumulated paired errors feed
  the DM-HLN promotion test. Lives in: `shadow.py`, `scripts/hypersearch_v2.py --shadow`, env
  `TRADER_SHADOW_MODE`. See also: champion, challenger, DM-HLN.
- **short mirror (`side=-1`)** — the offline short version of the exit kernel, used for research
  only. The system is long-only in production; `side=+1` is the live path.
  Lives in: `policy_exits._exit_walk_kernel_short`. See also: live mirror, `OBJECTIVE_LONG_ONLY`.
- **size tilt** — any multiplier that scales a position without changing the entry decision. The
  named ones are the meta tilt `clip(2p, 0.6, 1.3)`, `llm_mult`, the regime/VIX multiplier, and
  the CS-rank tilt. Lives in: `meta_label`, `base_loop` (composition).
  See also: tilt, `llm_mult`, sizing co-fire, `TILT_MAX`.
- **sizing co-fire** — the measurement of how often the advisory sizing multipliers move
  TOGETHER, i.e. how much of the composite tilt is one signal counted several times.
  Lives in: `scripts/sizing_cofire_report.py`, gated by `CONVICTION_JOURNAL_ENABLED` journals.
  See also: sentiment triple-count, tilt, degraded mode.
- **source-text contract test** — a test that cannot import its target (heavy deps) so it reads
  the source as TEXT and asserts on it — that a call exists, that a constant is unchanged, that
  a removed branch stays removed. Weaker than executing code; used where nothing else is possible
  on the Mac. Lives in: `tests/` (e.g. `tests/test_base_loop_v3.py`).
  See also: extract-and-exec test, byte-pin test, Mac-runnable.
- **spawn edge** — a parent module launching a child module as a separate OS process
  (`subprocess.Popen`). It is a real dependency that import tooling cannot see, so the graph
  records it separately. Lives in: `run_pipeline.py`, `gui.py`; recorded by
  `scripts/repo_graph.py`. See also: import graph, engine subprocess, command file.
- **Stage-0** — the measurement-first discipline: before any activation flip, an in-repo
  instrument must produce evidence from real journals. Stage-0 outputs are measurement-only and
  ship directly. Lives in: `decision_report.py`, `stage0_preds.py`, `ic_diagnostic.py`,
  `rank_gradient.py`, `scripts/*_report.py`. See also: evidence gate, GATE_REASONS, journal.
- **stage0 preds (`{slot}_stage0_preds.json`)** — the non-overlapping per-(symbol, bar)
  prediction dump `{ts, symbol, pred, signal, fwd_return, close, horizon_bars, lstm_pred,
  lgb_pred, meta_p, q10, pred_thresh_ratio}`, auto-emitted by the weekly backtest. It is the
  common input for IC-by-name, rank-gradient, naive-baseline and horizon-transfer measurement.
  Lives in: `stage0_preds.build_rows` / `write_rows`, emitted by `backtest.py`.
  See also: Stage-0, CS-IC, rank gradient, naive baseline.
- **stop-risk** — a position's `(entry_price − stop) × qty / equity`, expressed as a FRACTION of
  equity and anchored at ENTRY (initial-risk bookkeeping, not marked to market).
  Lives in: `portfolio`, `risk_budget`. See also: book risk budget, ENB, risk-per-trade.
- **study DB** — the Optuna SQLite study (`v2_study.db`, `stock_v2_study.db`) holding every
  trial's parameters and score. It MUST be deleted after any objective, feature or cost change,
  because the old scores are no longer comparable (gotcha #2).
  Lives in: written by `scripts/hypersearch_v2.py`. See also: Optuna trial, gotcha #2, `cum_trials`.
- **survivorship** — the bias introduced by training on the universe as it looks TODAY. It is
  removed by the as-of membership mask plus the harvested-but-never-traded candidate pool.
  Lives in: `stock_config.AS_OF_TOP_K`, `stock_config.TRAINING_CANDIDATE_POOL`.
  See also: as-of universe, PIT, DV30.

## T

- **tier (LLM)** — the Gemini free-vs-paid account tier, auto-detected from the
  `x-ratelimit-limit-requests` response header; it selects the RPM and per-model RPD budgets.
  Lives in: `llm_client` (`probe_tier`). See also: provider chain, free-first, Batch API.
- **tilt** — the single advisory multiplier PRODUCT applied to a position's base size. Boosts are
  clamped at `TILT_MAX = 1.30`; de-risking is honored all the way down to 0.1. Kelly and vol
  multipliers are applied OUTSIDE the tilt product.
  Lives in: `strategy_config.TILT_MAX`, composed in `base_loop`.
  See also: size tilt, degraded mode, sizing co-fire, Kelly cap.
- **trade budget** — the per-symbol daily cap on entries
  (`MAX_TRADES_PER_SYMBOL_PER_DAY`), a backstop that works alongside the cooldown.
  Lives in: `strategy_config.MAX_TRADES_PER_SYMBOL_PER_DAY`, flag `TRADE_BUDGET_BACKSTOP`;
  taxonomy entry `trade_budget`. See also: cooldown, lockout, GATE_REASONS.
- **trade threshold** — the predicted-return level above which a long entry is admitted
  (`pred > threshold`, in percent). Known defect H2: it is SEARCHED on raw-LSTM fold predictions
  but SERVED against the blend. Note the strict-vs-inclusive asymmetry — the objective uses `>`,
  the replay and `base_loop` admit `>=` (measure-zero for float predictions).
  Lives in: `scripts/hypersearch_v2.py` (selection), `objective_utils.simulate_trades_core`,
  `predict_now`, `backtest`. See also: blend, certificate, q10 veto.
- **`TRADER_*` env flag** — the environment-variable half of the flag surface (the other half is
  `strategy_config` constants). Most accept `1`/`true`/`yes` case-insensitively and are read at
  call time; two older ones (`TRADER_ORDER_STREAM`, `TRADER_USE_ALPACA_PY`) accept only the
  literal `'1'` and are captured at IMPORT time — so `TRADER_ORDER_STREAM=true` is a silent
  no-op. Lives in: `docs/FLAGS.md` (inventory). See also: flag family, default-OFF.
- **train/serve parity** — the sacred invariant: features are computed by the SAME functions at
  harvest time and at live serving time, so a value can never differ between the two. Changing a
  feature's values is therefore always model-facing.
  Lives in: `indicators.py` shared by harvest and `predict_now`; pinned by the golden fingerprint.
  See also: golden fingerprint, PIT, neutral fill, gotcha #2.
- **training_data parquet** — the harvested feature+label panel per book: `training_data.*`
  (crypto) and `stock_training_data.*` (stock). The crypto stem has NO `crypto_` prefix — a
  recurring source of wrong literals. Lives in: `data_utils._FILE_STEMS`, written by the harvest
  scripts. See also: harvest, raw sidecar, preset.
- **triple-barrier (TB) label** — the return the LIVE exit stack would actually realize from an
  entry at each bar: `TB_Ret_{fb}` (percent), `TB_Bars_{fb}` (bars held), `TB_Reason_{fb}`
  (which barrier fired). Contrast `Target_Return_{fb}`, the naive hold-exactly-`fb`-bars return.
  Produced by the same kernel the backtester runs. Lives in: harvest scripts calling
  `policy_exits.exit_walk`. See also: exit stack, vertical barrier, reason code, average uniqueness.
- **two-machine reality** — the operational fact that shapes every plan: the dev Mac (py3.13, no
  torch/lightgbm/optuna/joblib/numba/sklearn/dotenv/alpaca/PySide6) can do pure-algorithm work
  and synthetic-data tests; the Jetson Orin Nano 8 GB (py3.10, full stack) does training,
  harvest, live trading and the GUI. The sync between them is user-driven and not encoded in the
  repo. Lives in: `CLAUDE.md` § Two-machine reality. See also: Jetson-gated, Mac-runnable.
- **two-tier cost** — pricing a trade with a per-name EDGE stamp when one exists and a flat
  per-asset fallback when it does not, so cost is never silently zero and never uniformly
  optimistic. Lives in: `fees.round_trip_cost_pct` + `fees.FLAT_SPREAD_PCT` + `liquidity`
  stamps. See also: EDGE spread, `FLAT_SPREAD_PCT`, cost gate.

## V

- **vertical barrier** — the maximum-hold exit at `fb` bars. Its value differs by consumer:
  labels use `fb`; the backtest runs unlimited (`max_hold=0`); live enforces it as a loop-layer
  position-age check. Lives in: `policy_exits` (`max_hold`), `base_loop` (crypto fb-anchored
  check, IA-4). See also: exit stack, triple-barrier, `forward_bars`.
- **veto strike** — the per-symbol counter of CONSECUTIVE vetoing LLM analyses. It is cleared by
  any non-vetoing score and by TTL expiry; two strikes are required to liquidate an open
  position. Lives in: `base_loop`. See also: LLM veto, `s`.
- **VIX ladder** — the volatility-tier size multiplier with hysteresis: enter tiers at VIX 25 and
  35, exit at 22 and 31. One tier map, one owner — sizing must not double-count VIX.
  Lives in: `macro_indicators.vix_tier_mult_v2`, `_VIX_TIER_ENTER` / `_VIX_TIER_EXIT`.
  See also: VIX25 block, hysteresis, regime, tilt.
- **VIX25 block** — the separate hard ENTRY block in the 25–35 band, distinct from the ladder's
  size multiplier; the `VIX25_BLOCK_REMOVED` flag (default off) removes it so the band's cost can
  be measured. Lives in: `strategy_config.VIX25_BLOCK_REMOVED`, consumed by `base_loop` /
  `stock_loop`; taxonomy entry `vix25_block`. See also: VIX ladder, GATE_REASONS, IA-1..IA-4.
- **vol target** — the annualized portfolio volatility target used to derive a book-level size
  scalar. Applied at exactly ONE scope by design; per-position ratios compose at 1.0.
  Lives in: `strategy_config.PORTFOLIO_VOL_TARGET`, applied in `base_loop`.
  See also: tilt, HAR-RV, Kelly cap.

## W

- **warmup fill** — replacing a long-window feature's NaN warmup with a neutral constant (0.0, or
  0.5 for `Pos_Range_*`) so the harvest's `dropna()` does not discard a year of history AND live
  serving can produce the same value from a short frame.
  Lives in: `indicators.py`, harvest scripts. See also: neutral fill, train/serve parity.
- **wave** — a numbered research campaign, each frozen into a JSON record plus a memory file
  holding its survivors and its KILL list. Waves are historical: current state lives in
  `session-state`, and every wave's kills are consolidated into the canonical kill list.
  Lives in: `research/waves/wave{N}_research.json`, `research/README.md`.
  See also: kill list, packet ID, decision queue.
- **window A/B** — a fixed-config comparison of two training-window lengths against a PINNED
  holdout, with no hyperparameter search in between, so the only thing that varies is the window.
  Its point is to answer "more history or fresher history" without adding selection pressure.
  Lives in: `scripts/window_ab.py`, enabled by `FIXED_HOLDOUT_DAYS`.
  See also: holdout pin, selection pressure, retrain ledger.
- **winner's curse** — the general failure mode this repo defends against everywhere: any
  statistic reported for the configuration that was CHOSEN by that same statistic is biased
  upward. The named defenses are DSR deflation, PBO, the noisy ratchet, the pinned holdout, and
  the policy-replay promotion gate. Lives in: `validation.py`, `adaptive_config.py`,
  `backtest.py --gate`. See also: DSR, PBO, fold-max checkpoint, selection pressure.

