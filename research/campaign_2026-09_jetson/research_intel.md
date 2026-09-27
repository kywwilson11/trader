# research_intel — INTEL department research log (2026-09 Jetson campaign)

Append-only, dated, cited. Each entry is a scout brief adjudicated by the INTEL general; experiments carry pre-registered decision rules. Nothing here is shipped by being written here.


---
## 2026-09-27 R1 · Scout A — LLM-as-analyst in finance + news sentiment beyond lexicons

# SCOUT A — LLM-as-analyst + news sentiment beyond lexicons (survey 2026-09-27, Opus scout; zero LLM spend, no repo edits)

## Sources
1. Gao, Jiang & Yan, "Detecting Lookahead Bias in LLM Forecasts" — arXiv 2512.23847 v2 — 2026-06 — https://arxiv.org/html/2512.23847v2 — a date-only recall probe (Lookahead Propensity, LAP = P(up)+P(down)) explains ~32% of an LLM headline signal's next-day effect; the interaction goes to zero after the cutoff.
2. Benhenda, "Look-Ahead-Bench" — arXiv 2601.13770 — 2026-01-20 — https://arxiv.org/abs/2601.13770 — standard LLMs show large alpha decay across temporally distinct regimes; only PiT-trained models generalise.
3. "HindsightBench" — arXiv 2607.18867 — 2026-07 — https://arxiv.org/abs/2607.18867 — parametric hindsight shows up in every 2026-generation model tested (15 models, 7 vendors); effective cutoffs sit up to 8 months BEFORE vendor-stated ones; quantisation/serving changes the measurement.
4. Li, Wang & Ma, "Summoning the Oracle to Slay It" (FinCAD) — arXiv 2605.24564 — 2026-05-23, rev. 2026-08-29 — https://arxiv.org/abs/2605.24564 — decoding-time de-memorisation cut in-sample backtest returns by up to 67.1%; OOS (2025) was unchanged, i.e. that part of the in-sample alpha was memory.
5. Lopez-Lira & Tang, "Can ChatGPT Forecast Stock Price Movements?" — arXiv 2304.07619 v6 / SSRN 4412788 (2026 rev.) — https://arxiv.org/html/2304.07619v6 — on post-cutoff headlines, GPT-4 catches the (non-tradable) initial reaction, and the tradable drift is concentrated in SMALL stocks and NEGATIVE news; returns fall as LLM adoption rises.
6. Xia et al., "Agentic Trading: When LLM Agents Meet Financial Markets" — arXiv 2605.19337 — 2026-05-19 — https://arxiv.org/abs/2605.19337 — audit of 77 studies: in the 19-study closed-loop core, only 2 have time-consistent splits and 1 has a transaction-cost model; none are reproducible.
7. Asaad et al., "The Analyst in the Prompt" — arXiv 2609.03218 — 2026-09-02 — https://arxiv.org/pdf/2609.03218 — 3,575 SEC filings × 12 LLMs: context moves a "neutral evidence judgment" mainly by changing how the SAME evidence is interpreted, not which evidence is retrieved. Separating the evidence judgment from the context-conditioned output reduces the spill but does not remove it.
8. Zhao, Balagopalan et al., "The Price of Agreement" — arXiv 2604.24668 — 2026-04-27 (ICLR'26 FinAI) — https://arxiv.org/abs/2604.24668 — financial agents lose little under user rebuttal, but most models fail when the supplied preference contradicts the right answer; LLM input filtering helps.
9. Pres, …, Andreas, "Position: It's Time to Optimize LLMs for Self-Consistency" — arXiv 2608.05188 — 2026-07-31 (ICML) — https://arxiv.org/abs/2608.05188 — framing bias and overconfidence can only be detected across repeated or related responses, not from single I/O pairs.
10. "From Financial Sentiment Classification to Return Predictability: a QLoRA benchmark" — arXiv 2608.04200 — 2026-08 — https://arxiv.org/abs/2608.04200 — 7 scorers on 2019 Benzinga S&P-100 headlines: all 1-day rank ICs are small and positive; the best is 0.0143 for FinBERT; classification accuracy rank does NOT carry over to return IC.
11. Bisharat & Hean, "Evaluating Financial Sentiment in the Age of AI" — arXiv 2609.20198 — 2026-07-29 — https://arxiv.org/abs/2609.20198 — 12 scorers (dictionary, finance transformers, open LLMs): general LLMs ≈ finance encoders; several relate to earnings surprises, NONE is significantly associated with next-day returns.
12. Renney et al., "Cloud to Edge: LLM inference on SBCs" — arXiv 2604.24785 — 2026-04 — https://arxiv.org/html/2604.24785v1 — Jetson Orin Nano CPU-only, Q4_K_M, 100-token generation: Qwen2.5-0.5B 13.0 tok/s, Qwen2.5-1.5B 4.2, Llama3.2-1B 9.5, Llama3.2-3B 4.3.
13. Gemini API pricing — updated 2026-09-24 — https://ai.google.dev/gemini-api/docs/pricing — 3.1-flash-lite $0.25/$1.50, 3.5-flash-lite $0.30/$2.50, 2.5-flash-lite $0.10/$0.40 (per G_llm); Batch and Flex are 50% off; the free tier is "free of charge" but model access is "limited".
14. Gemini rate limits — updated 2026-09-02 — https://ai.google.dev/gemini-api/docs/rate-limits — limits are PER PROJECT; RPD resets at midnight PT; per-model numbers appear only in AI Studio, not in the docs.
15. Gemini context caching — updated 2026-09-02 — https://ai.google.dev/gemini-api/docs/caching — implicit caching is on by default for 2.5+; minimum prefix is 2,048 tokens (2.5 Flash) and 4,096 (3.5+ Flash).
16. Gemini Flex inference docs + Google blog — 2026 — https://ai.google.dev/gemini-api/docs/flex-inference — Flex is synchronous at 50% off, with a 1–15 min target latency; requests can be preempted and there is no server-side fallback.
17. Anthropic pricing — accessed 2026-09-27 — https://platform.claude.com/docs/en/about-claude/pricing — Haiku 4.5 $1/$5; Batch 50%; cache read 0.1×, 5m write 1.25×, 1h write 2×; forced `tool` choice adds 588 system tokens on Haiku 4.5; Sonnet 5 $2/$10 is now the standard price.
18. OpenAI pricing — accessed 2026-09-27 — https://developers.openai.com/api/docs/pricing — gpt-5.4-nano $0.20/$1.25 (cached input $0.02); Batch and Flex are half price.
19. OpenRouter limits — accessed 2026-09-27 — https://openrouter.ai/docs/api-reference/limits — `:free` models: 20 RPM; 50 RPD without purchases, 1,000 RPD after ≥$10 of lifetime credit purchases.
20. Groq rate limits — accessed 2026-09-27 — https://console.groq.com/docs/rate-limits — free gpt-oss / qwen3.8-27b: 30 RPM, 1K RPD, 8K TPM, **200K TPD**.
21. Modern-FinBERT (ModernBERT-large, finance-tuned) HF card — accessed 2026-09-27 — https://huggingface.co/beethogedeon/Modern-FinBERT — the current FinBERT-successor class. Card only; no return evidence.
22. Glasserman & Lin (2023, pre-2025), "Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis" — SSRN 4586726 — in-window LLM sentiment scores embed outcome knowledge.
23. IN-HOUSE M1 (this scout, 2026-09-27, read-only `mode=ro`, under hwlock): `sentiment_cache.db` now holds **105,753 articles, dated to 2026-09-26** (H saw 63,743 frozen at 02-21, so the cache has been refreshed since). **660 headlines contain `n’t` (U+2019) vs 1,434 with ASCII `n't`: 31.5% of negated-contraction headlines**. For summaries it is 1,047 vs 1,284 (45%). 8,088 headlines contain U+2019 in total, 350 contain "rate cut", and 6,350 carry an `llm_score`.
24. IN-HOUSE M2 (this scout, 2026-09-27, CUDA off, 2 torch threads = the bots' TORCH_NUM_THREADS, random-weight shape-equivalent encoders, seq 32): MiniLM-L6 shape (22M) 84 ms/headline at bs1, 25 ms at bs16; BERT-base/FinBERT shape (110M) 320 ms at bs1, **134 ms at bs16** (≈40 GFLOP/s effective). Process max-RSS was 1.11 GB including torch, which is over the 600 MB guidance (a disclosed overrun). The int8 dynamic-quant run failed on this torch 2.8 build (torch.ao deprecated), so int8 is UNMEASURED.

## Findings that apply to THIS system
F1. **Only the live post-cutoff journal can judge the analyst. Enforce that mechanically.**
- Claim: LLM headline and forecast skill is substantially memorisation [1][2][3][4].
- Wrinkle: effective cutoffs can precede vendor-stated ones by up to 8 months [3]. Gating on the vendor date is therefore conservative in the safe direction.
- Applies because llm_eval's b2 verdict is the keep/kill-spend rule (llm_eval.py:73-77 power floor), and FR-20c's cutoff guard (05_frontier §FR-20c) is **not built**: `grep cutoff llm_eval.py` returns 0 hits.
- It also touches the stored feature: the 6,350 `llm_score` rows [23] are in-window re-scores (Feb–Apr 2026 Gemini, 2020–25 articles) and feed `daily_sentiment` score_type='llm' [22]. H already flagged this; [1] supplies the test.
F2. **Anchoring on the evidence pack is the dominant failure mode, and our prompt hands the model an anchor.**
- The prompt states "ML model prediction: ±x% (bullish/bearish)" (llm_analyst.py:1053-1058) whenever `include_pred=True`, which is the live default (:448).
- [7] shows context shifts the interpretation of identical evidence. [8] shows models fail when the supplied stance contradicts the truth.
- The echo-gap b2 test in llm_eval measures the damage after the fact. The pred-blind arm already exists in `scripts/prompt_ab.py` (`--hide-pred-b`, :199/:285) and has never been run on live evidence.
F3. **Repeat-call noise has never been measured.** [9] argues that consistency failures are invisible in single calls. The analyst runs at `_ANALYST_TEMPERATURE=0.2` (llm_analyst.py:47). A veto at s<0.15 and the 2-strike liquidation (base_loop, per G_llm §4) act on ONE sample.
- If test-retest |Δs| is comparable to the veto margin, vetoes are partly coin flips. This is measurable by pairing the production model with itself in `llm_qualify --shadow` (scripts/llm_qualify.py:22,74).
F4. **Cadence correction.** The brief says "hourly". The code is `LLM_INTERVAL_SEC = 600` (base_loop.py:94, stock_loop.py:66), so up to 144 calls/day per book and 288 combined.
F5. **Price trap in the "move off legacy 2.5" plan.**
- Assumptions: ~5k input tokens + 0.5–1.2k output per call, 288 calls/day.
- Paid cost per day [13]: **2.5-flash-lite ≈ $0.20–0.28**, **3.1-flash-lite ≈ $0.58–0.88**, **3.5-flash-lite ≈ $0.79–1.30**.
- So a paid 3.5-flash-lite analyst alone breaches `_DAILY_COST_LIMIT = 1.00` (llm_client.py:261, not :175 as G_llm cites; the file moved), and the cap is global, so it would starve sentiment scoring.
- The free tier is "free of charge", but its RPD is unpublished and applies per project, resetting at midnight PT [14]. That matches `_maybe_reset_quota`'s PT reset, and it means G_llm's D7 (per-process budgets) under-models it.
- Conclusion: a free-first 3.x switch needs its budget rows sized from the AI Studio numbers, not guessed.
F6. **Free OpenAI-compatible endpoints cannot carry 288 calls/day at 5k tokens.** Groq's 200K TPD allows ≈40 calls/day [20].
- OpenRouter `:free` allows 50 RPD, or 1,000 RPD only after a one-time ≥$10 credit purchase [19].
- `llm_qualify` should test at the real prompt size, and Phase-4's "free-first" should treat OpenRouter as viable only with that $10 (Owner ask A2).
F7. **Cheap-mechanics fits.**
- Gemini implicit caching is already automatic on 2.5+ [15], but only above a 2,048-token prefix on 2.5 Flash and 4,096 on 3.5+. The static system prompt must be ordered first for any hit.
- Gemini **Flex** (50% off, synchronous, 1–15 min latency) [16] is wrong for the 45 s analyst but right for the sentiment_history background re-scorer. Unlike Batch it returns inline usage, so it can be priced into the ledger, which sidesteps G_llm D11.
- Anthropic: Haiku 4.5 + 5-minute cache + forced tool (+588 tokens) [17] is still ≈$0.006–0.01/call (≈$2–3/day at 288). That exceeds the cap, so it is a spend decision, not a hygiene fix.
F8. **A local LLM judge is clearly NOT feasible on the 8 GB Orin CPU within the cycle.**
- Generation: 4–13 tok/s CPU-only for 0.5–1.5B at Q4 [12].
- Prompt ingestion: M2 measures ≈40 GFLOP/s on 2 threads, which implies ≈13 tok/s for a 1.5B model (≈3 GFLOP/token).
- One 5k-token analyst prompt therefore takes ≈6 min on 2 threads (≈2–3 min on all 6), plus minutes to decode ~1k tokens. Add ~1–1.5 GB RSS against today's ~3.3 GB available (`free -m`, 2026-09-27).
- A 0.5B per-headline YES/NO classifier (~60-token prompt) fits compute-wise (~1–2 s/headline), but [10][11] give no reason to expect IC above the encoder class.
F9. **A FinBERT-class encoder is feasible but not justified by evidence.**
- Cost: 134 ms/headline batched on 2 threads, ~440 MB fp32 weights; MiniLM is 25 ms and ~90 MB [24].
- Benefit, best case: one-day rank IC ≈0.014 [10], with no significant next-day association [11], on universes 2–60× broader than our ~46 names.
- It would also be a new dependency (transformers/onnxruntime are not installed in the jetson env; checked 2026-09-27).
- This matches the B13 ruling (02_research §B13: "no local FinBERT without a Jetson memory feasibility pass"). M2 is the first half of that pass.
F10. **Novelty via embeddings is also not justified yet.** novelty.py's Jaccard shingles (header :9-19) already remove exact and near reprints.
- MiniLM would cost ~25 ms/headline [24] to catch paraphrases. There is no 2025–26 return evidence at hourly horizons that paraphrase-level dedup adds IC.
F11. **G4-05 (curly apostrophe) is material, not cosmetic.**
- Mechanism: `_PUNCT = [^\w\s'-]` (sentiment.py:182) strips U+2019, so `isn’t` becomes the token `isnt`, which misses `_NEGATORS` (:175-179) and `endswith("n't")` (:241) and flips the sign.
- Scale [23]: this affects ~31% of negated-contraction headlines and ~45% of summaries.
- Consequence: this layer is the stored `keyword_score` → `Daily_Sentiment` (model-facing), so the fix belongs in the same gotcha-#2 re-harvest the H audit already requires.
- Modern subword tokenisers split `’` and `'` identically, so the defect is lexicon-specific. I found NO 2025–26 paper on typographic negation in finance lexicons; the survey found nothing new here.
- The fix is normalisation (U+2018/2019/02BC → `'`) before `.lower()`. It adds no lexicon entries, so it is consistent with the B13 "do not expand" rule.
F12. **B13 needs a data-level test, not a literature test.** 350 headlines contain "rate cut" [23], and G_llm showed these cancel to 0.000. [10] shows that classification gains do not imply IC gains. Measure impact before deciding (E3).

## Findings that do NOT apply (and why)
- **LLM-as-directional-forecaster / upsizing the analyst:** [5]'s residual drift is small-cap and negative-news, which is outside this liquid long-only book. It is already ruled NO in 05_frontier N10. The analyst stays veto/size-tilt only.
- **Historical backtests of LLM outputs (including with FinCAD de-biasing [4]):** FinCAD needs logit access to open 7–14B models we cannot host (F8). This is also N10.
- **Mechanistic overconfidence fixes (arXiv 2604.01457, Zhao et al., 2026-04-01):** they need white-box activations, and the providers are closed APIs.
- **QLoRA/fine-tuned 7–8B sentiment LLMs [10]:** won't fit 8 GB alongside the bots, and the best IC came from FinBERT anyway.
- **Local llama.cpp analyst:** not feasible (F8). **Groq/OpenRouter free as the primary analyst:** not viable at 288 × 5k tokens (F6).
- **Kill-list overlap:** none. The kill list has no LLM or sentiment-encoder entries. The adjacent items ("Analyst-revision/investor-attention momentum", "On-chain flow", "Order-book") are untouched by anything here. The attention-factor stat-arb kill is unrelated (long-short).

## Proposed experiments
**E1 — Test-retest noise floor of the analyst `s`.**
- Hypothesis: identical-evidence repeat calls at temperature 0.2 disagree enough to flip vetoes.
- Instrument: existing `scripts/llm_qualify.py --shadow --replay`, with the production model id listed as its own candidate (prod vs prod), plus a ~40-line pure summariser (new, measurement-only `scripts/llm_retest_report.py`) over `journals/llm_qualify/shadow_scores.jsonl`. Data: ≥300 symbol-pairs from ≥30 distinct replay cycles, both books.
- PRE-REGISTERED rule: Let V = veto-disagreement rate: among pairs where either call is < 0.25, the fraction where exactly one call is < LLM_VETO_THRESHOLD (0.15).
  - If median |Δs| ≥ 0.05 **or** V ≥ 5%, then **FAIL** → owner item: temperature 0, or a 2-sample mean before any veto/strike (default-OFF flag, shadow).
  - Otherwise **PASS** → record the noise floor in llm_eval's report header and close.
- Cost: ≈300 extra flash-lite calls ≈ **$0.20 total**, ≈10 Jetson-min of wall-clock (network-bound). Ships as measurement-only. Kill-list: clear.
**E2 — Pred-anchoring (echo) A/B on live evidence.**
- Hypothesis: showing the ML pred pulls `s` toward the pred's sign ([7][8]), so part of any b2 is echo.
- Instrument: existing `scripts/prompt_ab.py --hide-pred-b` over replayed live cycles.
- Data: ≥400 symbol-cycles (≈2 weeks of replay journals).
- PRE-REGISTERED statistic: anchoring index A = ρ_s(s_A, sign(pred)) − ρ_s(s_B, sign(pred)), with a cycle-block bootstrap 95% CI (2,000 reps).
- PRE-REGISTERED rule: If A ≥ 0.15 and the CI excludes 0, then **FAIL (anchored)** → owner item: a default-OFF `include_pred=False` live flag, graded later by llm_eval b2 on ≥120 t0 clusters under shadow.
  - If the CI includes 0 → keep the pred line and record "no anchoring detected at n".
- Cost: 2 calls per cycle-symbol (800 flash-lite calls) ≈ **$0.55**, ≈15 Jetson-min. Ships as measurement-only; any change goes behind a default-OFF flag. Kill-list: clear.
**E3 — Lexicon-correctness impact (G4-05 apostrophe + B13 phrase-mask), offline.**
- Hypothesis: the two defects move `Daily_Sentiment` materially on the refreshed 105k-article cache.
- Instrument: new measurement-only `scripts/lexicon_fix_impact.py`. It re-scores `articles` in-process with (a) the current `_score_text`, (b) apostrophe normalisation, (c) (b) + phrase-span masking. The variants are local copies, and sentiment.py is untouched.
- Outputs: per-article sign-flip rate; ticker-day |Δ aggregate|; and PIT IC vs next-trading-day return using learned_lexicon's entry rule (strictly after the publication day).
- Data: all 105,753 articles and ~10k+ ticker-days.
- PRE-REGISTERED rule: If (b) or (c) changes the ticker-day aggregate by >0.05 on ≥1% of non-zero ticker-days, then **bundle the fix** (default-OFF flag + byte-compat test) into the H-audit re-harvest (already a gotcha-#2 event).
  - IC deltas are **reported, not gating**: the expected IC magnitude (~0.01 [10]) is unresolvable at this n.
  - Otherwise, log it as a correctness-only owner item.
- Cost: $0, ≈3 Jetson-min CPU, <300 MB. Kill-list: clear. It is not a lexicon expansion.
**E4 — Go/no-go gate for ANY learned or encoder sentiment build.**
- Hypothesis: no headline scorer has hourly-to-daily IC here worth an encoder dependency.
- Instrument: existing `scripts/train_lexicon.py`, which writes `lexicon_eval_report.json` (static vs learned vs `llm_score` IC). Run it on the refreshed cache, AFTER the H FnG/date repair.
- PRE-REGISTERED rule: If no scorer's purged-WF 1-day IC has a 95% CI lower bound > 0.01 on ≥5,000 ticker-days, then record a NEGATIVE and do NOT build FinBERT, Modern-FinBERT, MiniLM novelty or a local LLM classifier (F8–F10).
  - If one does, schedule an int8 ONNX feasibility pass (M2 left int8 unmeasured) as the next step.
- Cost: $0, minutes of Jetson CPU. Ships as a dark artifact / measurement-only. Kill-list: clear.

## Owner asks
- **A1 (money, conditional):** E1 + E2 need ≈$0.75 of analyst-model calls outside the live path. `llm_qualify` bypasses the ledger by design, so approve the spend explicitly. It is within the spirit of the $1/day cap but not metered by it.
- **A2 (money):** OpenRouter free models are only a credible 288/day fallback after a one-time ≥$10 credit purchase (1,000 RPD) [19]. Buy it or drop OpenRouter from the free-first plan.
- **A3 (money/cap):** any PAID 3.x flash-lite or Haiku analyst needs `_DAILY_COST_LIMIT` raised, or an analyst sub-budget (F5/F7). Stay on free-tier 3.x only after reading the per-project RPD in AI Studio.
- **A4 (data integrity, no spend):** rule on excluding in-window `llm_score` rows (article date < scoring model's cutoff − 8 months margin [3]) from `daily_sentiment` training aggregates at the next re-harvest. **No kill-list re-open is requested.**

---
## 2026-09-27 R1 · Scout B — robust small-sample statistics for live evaluation + operator-console UX

# SCOUT B — small-sample live-evaluation statistics + operator-console UX (2026-09-27)
Scope: llm_eval / beta_ledger / decision_report / validation / shadow / evidence_reads / gui.py. Web survey plus code reading, and one synthetic size sim
(`scratchpad/generals/intel/w/scoutB_sim/dk_size.py`, run under hwlock with <100 MB and CPU only; `boot_cov.py` was written but not run, see F5). No repo edits, no LLM spend. [F] = foundational (pre-2024).
Note: gui.py's real tabs are Cockpit, Trading, Performance, News, Markets, Models, Logs, Settings (addTab gui.py:3807-7953). The brief's "Positions" and "LLM" tabs do not exist.

## Sources
1. Cluster-robust inference: a guide to empirical practice (MacKinnon, Nielsen, Webb) — J. Econometrics 232 — 2023 — https://arxiv.org/abs/2205.03285 — CRVE t-tests over-reject when clusters are few or unequal. What counts is the number of *effectively independent* clusters, not the raw G.
2. Fixed-b asymptotics for panel models with two-way clustering (Chen, Vogelsang) — arXiv v4 — 2024-08-23 — https://arxiv.org/abs/2309.08707 — the Driscoll-Kraay component is biased in finite T. A bias correction plus fixed-b critical values restores coverage.
3. HAR inference: recommendations for practice (Lazarus, Lewis, Stock, Watson) — JBES 36(4) — 2018 [F] — https://discovery.ucl.ac.uk/id/eprint/10160486/1/har_JBES_v8.pdf — Newey-West needs bandwidth S=1.3·T^½ with fixed-b critical values. Alternative: EWC with ν=0.4·T^⅔ cosines, judged against t_ν.
4. Robust two-sample inference under serial dependence (Hounyo, Kim) — arXiv — 2025-12-12, rev 2026-07-28 — https://arxiv.org/abs/2512.11259 — chi²/normal HAR over-rejects in small samples. Welch-type fixed-K t/F with adjusted dof fixes it, and the paper covers Diebold-Mariano directly.
5. Inference with few heterogeneous clusters (Ibragimov, Müller) — REStat 98 — 2016 [F] — https://www.princeton.edu/~umueller/BehrensFisher.pdf — estimate per block and t-test the q block estimates against t_{q-1}. Valid with fixed small q.
6. Automatic block-length selection + Correction (Politis-White 2004; Patton-Politis-White 2009) [F] — https://mathweb.ucsd.edu/~politis/PAPER/SBblockCORRECTION.pdf — the operative block-length rule. I found no 2025–26 refinement.
7. Hypothesis testing with e-values (Ramdas, Wang) — Foundations & Trends in Statistics 1 — 2024-10, v2 2025-09 — https://arxiv.org/abs/2410.23614 — e-processes stay valid under optional stopping and continuation. This is the textbook reference.
8. Comparing sequential forecasters (Choe, Ramdas) — Operations Research 72(4):1368 — 2024 — https://doi.org/10.1287/opre.2021.0792 — anytime-valid confidence sequences and e-processes on the mean score difference of two forecasters, for bounded scores under a weak null.
9. Valid sequential inference on probability forecast performance (Henzi, Ziegel) — Biometrika 109 — 2022 [F] — https://arxiv.org/abs/2103.08402 — for lag-h forecasts, average h interleaved e-products (Prop. 3.4). Power falls as h grows (§4, Fig. 2).
10. E-backtesting (Wang, Wang, Ziegel) — Management Science 72(6):4952 — 2026 — https://arxiv.org/abs/2209.00991 — model-free sequential e-process monitoring of risk forecasts (VaR/ES).
11. On e-backtesting: generalizations and sample-size determination (Oestmann, Dickhaus) — arXiv — 2026-09-04 — https://arxiv.org/abs/2609.05089 — first sample-size lower bounds for reaching a prescribed power with an e-backtest.
12. Anytime validity is free: inducing sequential tests (Koning, van Meer) — arXiv / JRSSB adv. — 2025-01, rev 2025-12 — https://arxiv.org/abs/2501.03982 — any valid N-sample test can be embedded in an anytime-valid sequential test that matches it at N.
13. Near-optimal nonparametric sequential tests with possibly dependent observations (Bibaut, Kallus, Lindon) — arXiv — 2022, rev 2024-03 — https://arxiv.org/abs/2212.14411 — delayed-start normal-mixture SPRT is asymptotically level-α under dependence. Used at Netflix.
14. Estimating means of bounded random variables by betting (Waudby-Smith, Ramdas) — JRSSB 86(1) — 2024 — https://academic.oup.com/jrsssb/article/86/1/1/7043257 — the tightest practical confidence sequences for a bounded mean (hedged-capital betting).
15. What survives honest evaluation? Search-aware assessment of LLM-driven strategy discovery (Gençay) — arXiv — 2026-08-27 — https://arxiv.org/abs/2608.27734 — DSR and PBO work as complementary filters over a full trial registry. Every LLM-found strategy failed, and so did a leaky oracle with SR 35.
16. The Deflated Sharpe Ratio (Bailey, López de Prado) — JPM — 2014 [F] — https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf — covers selection bias, non-normality and MinTRL. I found no 2024–26 revision of the method itself.
17. Simply better market betas (Welch) — Critical Finance Review 11:37 — 2022 [F] — https://cfr.ivo-welch.info/published/papers/welch2022simply.pdf — winsorize each daily return to [−2, +4]×r_m and add a ~4-month half-life decay. RMSE is ~40–45% better than OLS or Vasicek.
18. Crypto market betas: the limits of predictability and hedging (Sila, Mark, Kristoufek, Weber) — Financial Innovation 11 — 2025 — https://link.springer.com/article/10.1186/s40854-025-00777-w — crypto betas are far less predictable than equity betas. Winsorization and shrinkage help, but beta hedging cut variance for only ~17% of names.
19. Dimson (1979) JFE; Asness-Krail-Liew (2001) JPM [F] — summed lagged betas for non-synchronous prices. These are already implemented.
20. Best Practices for Automated Trading Risk Controls and System Safeguards — FIA — 2024-07 — https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf — §1.5: a kill switch blocks new orders **and cancels working orders**, may still allow risk-reducing orders, sits apart from the trading app, and must warn "of the consequences of activating".
21. ANSI/ISA-18.2 + EEMUA 191 alarm benchmarks — https://www.instrumentationblog.in/alarm-management-isa-18-2/ — steady state ≤1 alarm per 10 min per operator; a flood is >10 per 10 min. Target priority mix is ~5/15/80 (high/med/low), with no chattering alarms.
22. ANSI/ISA-101 high-performance HMI — https://control.com/technical-articles/going-gray/ — the normal state is low-contrast grey and colour is reserved for abnormal. Uses a 4-level display hierarchy.
23. EU AI Act Art. 14(4) (Reg. 2024/1689) — 2024 — https://artificialintelligenceact.eu/article/14/ — oversight must build automation-bias awareness, let the operator override, and provide a "stop button… to a halt in a safe state".
24. WCAG 2.2 SC 1.4.1 / 1.4.11 — W3C 2023 — https://webaim.org/articles/contrast/ — colour must never be the only carrier of meaning. Non-text UI and state indicators need ≥3:1 contrast, rechecked for each theme.
25. Explanations can reduce overreliance on AI systems (Vasconcelos et al.) — CSCW 2023 — https://arxiv.org/abs/2212.06823 — explanations cut over-reliance when checking them is cheaper than the task.

## Findings that apply to THIS system
F1. **The llm_eval keep/kill p-value is anti-conservative at its own power floor.** Sources: #3, #4, #1, #2. The DK sandwich uses Bartlett lag = fb−1 = 23 (llm_eval.py:303-331, 532-537) and is judged against t_{G−1} (dof=G−1 at :537). Bartlett down-weights exactly the MA(23) overlap it is meant to cover. At G≥120, t_{G−1} ≈ normal, so it adds no small-sample protection. Measured null size in synthetic panels (K=6, h=24, persistent pred/s, nominal 5%): **production DK/t_{G−1} = 0.124–0.138** for n_eff 20/30/60 (0.170 with K=1). It does *not* improve with T, because the bandwidth is fixed at 23. DK with M=max(1.3√T,2h) plus KV fixed-b gives 0.070–0.090. EWC/t_ν gives 0.084–0.108 (ν too large for MA(23)). **IM K=8 gives 0.040–0.062, the only one on nominal.** R=500 per cell, MC s.e. ≈0.010. The IM K=8 cross-check (llm_eval.py:333-369) is report-only today. **Touches** llm_eval.py:537-544 (dof/p) and :553-555 (IM).
F2. **The "≥120 hourly clusters" floor is not the binding quantity; n_eff = span/fb is.** Sources: #1, #5, #9. With 24-hour overlap, adjacent hourly clusters are not independent, so the effectively independent units are roughly days. evidence_reads' `n_eff ≥ 20` (scripts/evidence_reads.py:84-86) is defensible only if the reference distribution has about n_eff dof (IM t_{q−1}, EWC t_ν, fixed-b). With t_{G−1} it is not. Also, `hac_lag_hours` treats one cluster as one hour (llm_eval.py:503); if cycles are sparser, the lag over-covers in hours.
F3. **Anytime-valid monitoring fits the shadow's peeking problem, but at h=24 the evidence is ~daily.** Sources: #8, #9, #12. shadow.py:36-41 documents ~14 unadjusted daily looks at p<0.05, and DM v2 (flag OFF) cuts that to 2 scheduled looks. By Henzi-Ziegel Prop. 3.4, a lag-24 e-process averages 24 interleaved products, so it should be *slower* than the fixed-N DM. By Koning-van Meer, the fixed-N final look can stay and an anytime-valid monitor can be layered on top at no loss at N.
F4. **The keep/kill-LLM-spend read can be sequential if it runs on disjoint daily blocks.** Sources: #14, #7, #13. Per-day net value-add is a bounded, non-overlapping series (the fb=24 overlap ends at day boundaries). A betting confidence sequence gives a valid stop at any day with no fixed n, which avoids the lag-h merge penalty in F3.
F5. **decision_report verdict CIs: iid percentile bootstrap at n≥9 (decision_report.py:76, 398-420).** Episode dedup removes intraday overlap but not same-day cross-name market moves. False-verdict rate at true mean 0 (nominal 10%, t3 tails): not measured this round (the CEO HW throttle). Command: `bash scratchpad/hwlock.sh heavy intel-scoutB2 -- timeout 280 /home/kyle/miniforge3/envs/jetson/bin/python scratchpad/generals/intel/w/scoutB_sim/boot_cov.py`. Expected direction [F]: the percentile interval is too narrow at small n (factor √((n−1)/n) plus skew), and same-day correlation widens the gap. Sources: #1, #6.
F6. **validation.politis_white_block_length returns 1.0 (iid) below 20 obs** (validation.py:647-649). So `stationary_bootstrap_sharpe_pvalue` silently becomes an iid bootstrap on tiny live samples. With known label overlap h, block length should be floored at h. Sources: #6, #3.
F7. **Gotcha #4 is correct in literature terms.** Sources: #3, #16. Uniqueness n_eff (sample_weights.py:166) and Lo-2002 f (validation.py:496) are two estimators of one long-run-variance inflation when overlap causes the autocorrelation. Use one LRV estimator, never both.
F8. **beta_ledger alpha t-stats cannot become decisive in a paper-trading horizon.** Sources: #16, #17. Measured SR ≈ 0.97 (D report, bad print removed) gives MinTRL for t=2 ≈ (2/0.97)² ≈ 4.3 yr. Beta is estimable (BTC summed β +0.62, t 3.5). Welch slope-winsorization would also neutralise vendor bad prints like the 2026-08-13 print (beta_ledger.py:47, 457-477 only warn). **Touches** beta_ledger.py:68-99 (ols_hac), 237-316.
F9. **A long-only SPY/BTC hedge built on the ledger's betas will be noisy.** Source: #18: crypto betas are poorly predictable, and hedging helped only ~17% of names. This bears on the survivor-#12 roadmap step 2 (SPY hedge). Size any hedge on a winsorized beta that has passed a stability check, not on the raw OLS beta.
F10. **Halt is entries-only and fails open** (base_loop.py:2922-2932; gui.py:10073-10093). Nothing on the halt path cancels working orders. FIA §1.5 defines a kill switch as block-new **plus** cancel-working, with explicit consequence warnings. Sources: #20, #23.

## Findings that do NOT apply (and why)
- **Wild cluster bootstrap / CR2 (Bell-McCaffrey)** (#1): these fix few *cross-sectional* clusters. Our problem is temporal overlap *across* clusters. WCR with independent cluster draws breaks that dependence, and the day-block IM test already gives exact-dof inference at no cost.
- **Chen-Vogelsang two-way (symbol×time) clustering** (#2): needs a large cross-section. Crypto has K≈6. Revisit only for stock-book llm_eval if K≥30 names per cycle.
- **E-backtesting of VaR/ES** (#10, #11): the system issues no VaR/ES forecasts. It would become relevant only if HAR-vol sizing forecasts were monitored (future; measurement-only).
- **Consumer A/B mSPRT defaults** (#13, Johari et al.): these assume iid users. They are unusable on hourly overlapping returns unless blocked to days (F4).
- **Conformal abstention (CQR/ACI)** is KILLED (KILL_LIST wave-4). Confidence sequences here are *evaluation* instruments, not a trade gate. Keep that boundary explicit.
- **Using an e-process on forecast loss to decide the LLM gate** would repeat the KILLED "DM-HLN gates POLICY" category error. The LLM read must be on *economic* value (F4), not forecast error.
- **Vasicek shrinkage of the portfolio beta** (#17, #18): there is no cross-section to calibrate a prior for one book. Use winsorization instead. This is not Ledoit-Wolf (killed), just an unhelpful prior.
- **New DSR/PBO variants**: none found for 2024–26 (#15 and #16 confirm existing practice). The repo's DSR+PBO pairing already matches #15.

## Proposed experiments/modules
**E1 — HAR size audit + verdict co-signature.** *Hypothesis:* llm_eval's DK/t_{G−1} rejects a true null at >7.5% when n_eff ∈ [20, 40] (F1). *Instrument:* NEW measurement-only `scripts/har_size_audit.py`. It reuses `llm_eval._driscoll_kraay_se` / `_im_block_pvalue` on synthetic nulls calibrated to live geometry: K, fb, and the AR(1) of pred and s estimated from realized llm_analysis rows. It also adds two report-only fields to llm_eval: `b2_ewc_p` (ν=⌊0.4T^⅔⌋, t_ν) and `b2_dk_fixedb_p` (M=max(1.3√T, 2h), KV critical value). *Data:* synthetic data (runs today), then real rows once n≥60 and n_eff≥20. *Pre-registered rule:* the synthetic audit already shows DK size 0.12–0.17 (F1). If the live-geometry audit (R=2000) confirms DK size >0.075, the keep/kill verdict carrier becomes IM K=8 at p<0.05, with DK p<0.05 required as co-sign. If DK size ≤0.075, DK alone stands. *Jetson cost:* <100 MB, <2 min CPU. *Ships as:* measurement-only (verdict rule = owner ask 1). *Kill-list:* not listed; distinct from killed conformal and sequential-bootstrap items.
**E2 — Anytime-valid LLM-spend ledger.** *Hypothesis:* the LLM size-tilt's daily net value is ≤ 0. *Instrument:* NEW measurement-only `llm_eprocess.py`. Per UTC day, d_t = Σ_buys (1−1/llm_multiplier)·final_notional·r_trade / equity − llm_cost_t / equity, clipped to [−B, B] with B=0.5%. It computes a Waudby-Smith-Ramdas hedged-capital confidence sequence and an e-process for H0: E[d]≤0. *Data:* journal buy rows (llm_multiplier, final_notional; present even in legacy rows), realized trade returns, llm_cost.json; ≥1 buy/day. *Pre-registered rule (α=0.05; first look at day 10, last at day 90):* **KILL** spend if the CS upper bound is <0 on any day. **KEEP** if the e-value is ≥20. At day 90 with neither, defer to the fixed-N llm_eval verdict (the Koning-van Meer embedding). Veto value stays with decision_report. *Jetson cost:* stdlib/numpy, <50 MB, seconds. *Ships as:* measurement-only. *Kill-list:* clear (evaluation, not a gate; economic, not forecast-loss).
**E3 — decision_report CI honesty.** *Hypothesis:* the iid percentile CI over-issues REVIEW/OK verdicts under same-day cross-name correlation (F5). *Instrument:* existing `decision_report.py`, plus a report-only `ci90_dayclust` from a day-cluster bootstrap (resample calendar days) and `verdict_dayclust`. *Data:* ≥30 priced episodes over ≥10 distinct days. *Pre-registered rule:* on the first post-retrain 30-day report, if verdict ≠ verdict_dayclust on ≥20% of gates with n≥9, switch the GUI verdict source to the day-cluster CI and raise MIN_VERDICT_N to 20 (owner). *Jetson cost:* negligible. *Ships as:* measurement-only. *Kill-list:* clear (not the killed sequential bootstrap for LGB bagging).
**E4 — Robust beta leg.** *Hypothesis:* the OLS summed BTC beta is fragile to single bad prints. *Instrument:* existing `beta_ledger.py`, plus a report-only Welch slope-winsorized beta per benchmark (with and without 120-day half-life WLS) and `alpha_mintrl_years` = (t*/SR̂)². *Data:* equity history, ≥60 daily obs (evidence_reads.py:97). *Pre-registered rule:* the beta read is "stable" iff |β_OLS − β_winsor| < 0.15 for BTC and SPY at n≥60. Alpha is printed as "not estimable" whenever alpha_mintrl_years exceeds the window length. No hedge sizing (roadmap step 2) on an unstable beta. *Jetson cost:* negligible. *Ships as:* measurement-only. *Kill-list:* clear (not Ledoit-Wolf or vol-timing; the SPY hedge is survivor #12).

## Console UX recommendations (ranked)
1. **Kill-switch consequences plus a safe-state read-back** (Models tab bot box gui.py:7195-7211, toolbar :3163, `_toggle_halt_clicked` :10073, `_flatten_all_clicked` :10095). The halt confirm and banner should state "entries blocked; N working orders NOT cancelled; M positions open" (#20, #23), and a chip should show when the halt-flag check is failing open (base_loop.py:2929-2932). *Outcome:* seconds from click to "0 working entry orders" is logged per activation, target ≤1 bot cycle.
2. **"What would it take to act" readiness panel** on the Models tab beside `_refresh_reports_freshness` (:10367). Render the evidence_reads table: per read, current/threshold for n, n_clusters and n_eff, projected ETA at current accrual, and the pre-registered action. *Outcome:* zero decisions taken from reports flagged `insufficient_power`, `quality.representative=false` or `stale`, auditable from the promotion_ledger/notes.
3. **Content-aware freshness.** `REPORT_FRESHNESS_ITEMS` (gui.py:361-373) and `chart_core.artifact_freshness` (:41) age by mtime only. In the F run, a `stale:true` stub read "decision_report: 1s". Use 3 states: fresh-valid / fresh-VOID / aged, and carry the no_data / insufficient_power flags. *Outcome:* the false-fresh rate on the D-run stub fixtures (decision_report, llm_eval, llm_advisor no_data) is 0/3 in a headless test.
4. **Alarm hierarchy on the Cockpit feed** (`_push_alert` :3828, `_alert_color` :3818). Today: colour-only kinds, no priority, no ack or shelve, dedupe only against the top item. Add 3 priorities with a text tag or glyph (#21, #24 SC 1.4.1), ack/shelve, per-kind dedupe over a 10-min window, and flood collapse at >10/10 min. *Outcome:* steady-state alert rate ≤1 per 10 min over a week (measured from the list), and zero colour-only encodings.
5. **Structured veto explanations** (Trading tab `_refresh_gate_attribution` :4311; Markets LLM column). Each veto shows (gate, value, threshold) and a priced CI with n (#25). The data prerequisite is structured skip reasons: 100% of April/May skips were free-text `llm_below_buy_min (x<0.60)`, which is unclassifiable. *Outcome:* `_unclassified_skip_reasons` is empty on every post-retrain report.
6. **ISA-101 "Ops" theme plus a contrast test** (#22, #24). Grey normal state, saturated colour only for abnormal. Add a unit test in tests/test_design_tokens.py asserting ≥4.5:1 text and ≥3:1 status-chip contrast for all 12 themes. *Outcome:* the test passes for 12/12 themes, and the Ops theme has 0 saturated widgets in the nominal state.

## Owner asks
1. Approve E1: make IM K=8 (llm_eval.py:333, today report-only) the keep/kill-LLM-spend verdict carrier, with DK as co-sign. The synthetic DK size is 0.12–0.17 at nominal 0.05. This changes the pre-registered read in runbook 03:48.
2. Freeze E2's parameters (B=0.5% clip, α=0.05, day-10 first look, day-90 horizon, cost source) **before** post-retrain journals accrue. Otherwise the thresholds are not pre-registered.
3. Decide whether Halt should cancel working entry orders (FIA §1.5). This is live-path behaviour, not UX.
4. Accept "alpha not estimable in this horizon; beta only" as the beta_ledger policy (F8), and rule that no SPY hedge is sized on an unstable beta (F9, E4).
5. Decide whether shadow gets a report-only anytime-valid monitor (F3) or relies on DM v2's two scheduled looks (TRADER_SHADOW_DM_V2, default OFF).

---
## 2026-09-27 R2 · Scout C — anytime-valid / sequential evaluation design (LLM-spend ledger + shadow monitor)

# SCOUT C — anytime-valid / sequential evaluation design for the LLM-spend read and the shadow (2026-09-27)
Scope: design only. No repo edits, no LLM spend. I ran one synthetic sizing sim under hwlock: `scoutC_sim/eproc_size.py`, numpy only, <100 MB, about 3 min in total. [F] marks a pre-2024 foundational source. Repo HEAD is 438f56a plus the uncommitted tree.

## Sources
1. Waudby-Smith & Ramdas, *Estimating means of bounded random variables by betting*, JRSSB 86(1):1-27, 2024-02 — https://academic.oup.com/jrsssb/article/86/1/1/7043257. Gives the hedged-capital CS, the aGRAPA/predictable-plug-in bets and the weak (conditional-mean) null.
2. Ramdas, Grünwald, Vovk & Shafer, *Game-theoretic statistics and SAVI*, Stat. Sci. 38(4):576-601, 2023-11 — https://arxiv.org/abs/2210.01948. Survey. Ville's inequality makes a threshold of 1/α on a test supermartingale valid at any stopping time.
3. Ramdas & Wang, *Hypothesis testing with e-values*, FnT Stat., 2024-10, v2 2025-09 — https://arxiv.org/abs/2410.23614. Textbook treatment of e-processes, optional continuation and merging.
4. Grünwald, de Heide & Koolen, *Safe testing*, JRSSB 86(5):1091-1128, 2024 (with discussion) — https://doi.org/10.1093/jrsssb/qkae059. GRO/GROW as the power notion under optional continuation.
5. Henzi & Ziegel, *Valid sequential inference on probability forecast performance*, Biometrika 109, 2022 [F] — https://arxiv.org/abs/2103.08402. For lag-h forecasts, Prop 3.4 averages h interleaved e-products. Under Prop 3.5 a stopped e-value is only determined h−1 steps later. In §4 / Fig. 2, e-value power falls faster with lag than DM's does.
6. Choe & Ramdas, *Comparing sequential forecasters*, Oper. Res. 72(4):1368-1387, 2024 — https://doi.org/10.1287/opre.2021.0792. CS and e-processes for the running mean score difference under a weak null, with lag handling by stream splitting.
7. Koning & van Meer, *Anytime validity is free: inducing sequential tests*, arXiv 2501.03982, 2025-01, v5 2025-12 — https://arxiv.org/abs/2501.03982. Any valid N-sample test can be induced into an anytime-valid test that matches it at N. Early-look power is not guaranteed, and some results assume iid.
8. Johari, Koomen, Pekelis & Walsh, *Always valid inference*, Oper. Res. 70(3):1806-1821, 2022 [F] — https://doi.org/10.1287/opre.2021.2135. mSPRT. Misspecifying the mixing variance by 2 orders of magnitude costs about 20% power and about 40% more run length. Assumes iid units.
9. Schultzberg, *Closed-form sample-size correction for always-valid inference*, arXiv 2606.18366, 2026-08 — https://arxiv.org/html/2606.18366. Sizing an anytime-valid test by endpoint power alone oversizes it by 8-20%. Endpoint sizing is therefore conservative, and I use it that way below.
10. Taga, Oymak & Shekhar, *Learning to bet for horizon-aware anytime-valid testing*, arXiv 2603.19551, 2026-03/06 — https://arxiv.org/abs/2603.19551. With a hard deadline, pure Kelly is not optimal: bet harder when behind schedule. This is a v2 option only; I pre-register the simple aGRAPA bet.
11. Durand & Wintenberger, *Power comparison of sequential testing by betting procedures*, arXiv 2504.00593, 2025-04/10 — https://arxiv.org/abs/2504.00593. Second-order (variance-aware) bets widen the set of detectable alternatives for bounded means.
12. Bibaut, Kallus & Lindon, *Near-optimal nonparametric sequential tests with dependent observations*, arXiv 2212.14411, rev. 2024-03 — https://arxiv.org/abs/2212.14411. Delayed-start mixture SPRT under dependence. Cited as the fallback if the lag-1 autocorrelation of d_t is material.

## Design A — anytime-valid LLM-spend ledger (`llm_eprocess.py`, NEW, measurement-only)
**What the LLM does to trades, per the code.** A score s < 0.15 is a veto (base_loop.py:3170-3177, trading_utils.py:30). Otherwise the size tilt is m = 0.5 + s (base_loop.py:3178), fed into `_compute_position_size` (:3202-3204). Buy rows journal `llm_multiplier`, `final_notional`, `fill_price` and `ts` (base_loop.py:3360-3373). Sell rows journal `fill_price`, `pnl_pct` and `exit_reason` (base_loop.py:1916-1926, 1537-1544).
**The daily statistic must be mark-to-market (MTM), not trade-close attribution.** Scout B's F4 said the fb=24 overlap "ends at day boundaries". That holds only for disjoint MTM days: a scored candidate's 24-bar forward window, or a round trip, straddles midnight. If P&L is attributed on the close day, the part accrued before day t is already F_{t−1}-measurable, and the martingale-difference structure is lost. Daily MTM makes the lag 1 in day units, so Henzi-Ziegel's h-way merge penalty (Src 5) disappears.
**d_t (bp of equity, per UTC day t, both books combined because the spend is shared: llm_client.py:269-286):**
- Lot i (one buy row) has qty q_i = N_i/P_i^fill. Its tilt qty is τ_i = q_i·(1 − m̄/m_i). m̄ is the mean multiplier over the burn-in buys, frozen. This is the *budget-neutral* primary: it is the value of the LLM's reallocation versus a free constant multiplier. The secondary, as-deployed variant sets m̄ = 1, matching llm_eval's neutral-1.0× baseline (llm_eval.py:925-934).
- G_t = Σ_i τ_i·(P_i^end,t − P_i^start,t) − Σ_{entries on t} φ_e|τ_i|P^fill − Σ_{exits on t} φ_x|τ_i|P^exit. Here start = max(entry, day start) and end = min(exit, day end). Marks are the last hourly close ≤ 24:00 UTC, taken from read-only Alpaca bars (stocks use the last RTH close). φ_e and φ_x are half of `fees.round_trip_cost_pct` each (fees.py:187).
- **d_t = 10⁴·(G_t − c_t)/E_{t−1}.** E_{t−1} is the prior daily equity from `beta_ledger.load_equity_alpaca` (beta_ledger.py:692-698). Days with no exposure still count at −c_t: the spend happens whether or not anything is held.
- **Cost c_t.** llm_cost.json holds today only: `{"date","cost"}` is overwritten (llm_client.py:861) and resets at LA midnight (llm_client.py:879-887). It is therefore not a history source. The ledger brackets cost instead:
  - c_t^lo = Σ `cost_usd` over the day's llm_analysis rows (base_loop.py:1757-1760, taken as a shared-ledger delta at llm_analyst.py:647-652). This is analyst-role only, rounded to $1e-4, and excludes sentiment-role calls.
  - c_t^hi = $1.00, the hard daily cap `_DAILY_COST_LIMIT` (llm_client.py:261).
  - KEEP uses c^hi and KILL uses c^lo, which is conservative in each direction. At $100k equity, c ≤ 0.1 bp/day against a simulated sd(d) ≈ 2.8 bp/day, so cost barely moves the read. What decides is the tilt.
- **Clipping.** B = max(2.5·δ, 5·1.4826·MAD(d_1..d_10)), frozen at the end of burn-in. Then x_t = (clip(d_t, −B, B) + B)/(2B) ∈ [0, 1]. The floor 2.5δ keeps every null point strictly inside (0, 1); the sim crashed without it. A decision counts only if the clip rate *in the direction that favours it* is ≤ 5% (lower clips for KEEP, upper clips for KILL). The sim clip rate was about 2% per side.
**Nulls (weak, time-averaged: μ̄_t = t⁻¹Σ E[d_s | F_{s−1}]; Src 1, 6).**
- H_K: μ̄ ≤ 0, with cost c^hi.
- H_H: μ̄ ≥ 0, with c^lo. Rejecting it means the tilt destroys value.
- H_F: μ̄ ≥ δ with δ = 1.0 bp/day, about $10/day at $100k (10× the spend cap), with c^lo. Rejecting it means futility.
**E-processes (Src 1 aGRAPA; Src 2 Ville).** For a null point m₀ = (a + B)/(2B):
- Upward test: K_t = Π_{s≤t} (1 + λ_s(x_s − m₀)), with λ_s = clip((μ̂_{s−1} − m₀)/(σ̂²_{s−1} + (μ̂_{s−1} − m₀)²), 0, c/m₀) and c = 0.5.
- Downward tests use the mirror (m₀ − x_s) with cap c/(1 − m₀). μ̂ and σ̂² are running estimates with priors ½ and ¼.
- Under the weak null each K is a nonnegative supermartingale, so P(sup_t K_t ≥ 1/α) ≤ α. This is not "HAC-robust": it is valid for any dependence, because only the conditional mean is constrained.
- For display only, the report prints the two-sided hedged-capital CS [L_t, U_t] in bp/day (θ = ½, α = 0.05).
**α and thresholds.** Each decision direction gets α = 0.05, split as 0.025 sequential (threshold 1/0.025 = 40) plus 0.025 for the fixed-N fallback (union bound).
**Schedule.** Burn-in covers days 1-10 (freezes m̄ and B; not tested). The CS starts on day 11. First decision look: CS day 10 **and** ≥ 30 lots. Looks are daily after that. Horizon: CS day 180, with an interim status at day 90 that runs no fixed test. Sim basis: 90 days is under-powered (see Simulation plan).
**Stop rules (evaluated daily after the first look):**
- **KILL-HARM** if E_H ≥ 40.
- **KILL-FUTILITY** if E_F ≥ 40 (U_t < δ at 97.5%).
- **KEEP** if E_K ≥ 40.
- Otherwise **CONTINUE**.
- At CS day 180, the fallback is a one-sided t on the clipped daily d at α = 0.025 against t_{n−1}. If the lag-1 autocorrelation ρ₁ has |ρ₁| > 0.2, use IM K=8 day-blocks instead (llm_eval's `_im_block_pvalue`, llm_eval.py:333-369). Still inconclusive → the owner's pre-signed default (Owner ask 3).
- **Veto clause.** KILL ends the *spend* only if decision_report's `llm_veto` gate CI on the same date does not exclude positive value. Otherwise the recommendation is "kill the tilt, keep the veto". The veto is rare: 3 `llm_veto` rows in the Apr-May journals.
**What "spend nothing" means for the read.** Zero LLM calls. It never imports `llm_client` call paths and reads llm_cost.json and journals only as files. No orders. The only network use is read-only Alpaca bars and portfolio history, the same as beta_ledger. It writes only its own JSON, records the input hashes, and is deterministic. Adding it cannot change any live decision, so it ships directly.
**Data window.** Only post-retrain journals written by the current gate code are used. The Apr-May 2026 journals ran a since-removed `llm_below_buy_min (s<0.60)` gate: 11,815 of 11,818 skips, journals/2026-04-06…05-06. Today's code has only s < 0.15 (base_loop.py:3170-3178), so those rows are *not the same policy* and are excluded. The pre-registration records the activation date and git sha.
**How it coexists with the fixed-N llm_eval read (Src 7).**
- The two reads answer different questions. The llm_eval b2 read (power floor n≥60, ≥120 t0-clusters, n_eff≥20: llm_eval.py:36-48, 73-77; evidence_reads.py:83-89) asks whether s carries information beyond pred on *scored candidates*. The ledger asks whether the *deployed* tilt pays.
- The spend decision is owned by the ledger plus its fallback. b2 is reported alongside as the mechanism diagnostic and does **not** vote, so there are never two α's on one decision.
- This changes the pre-registered read in runbook 03:48 ("this is the keep/kill-LLM-spend read"). That is the owner's call (Owner ask 1).
- Koning-van Meer option: instead of the α split, induce the day-180 t-test into an anytime-valid test that equals it at N=170. That gives the full α = 0.05 at the horizon but weaker early looks. Defer it to v2.

## Design B — shadow: anytime-valid monitor vs DM v2's two scheduled looks → **recommend DM v2, no monitor now**
- **Status quo.** Legacy runs about 14 unadjusted daily looks at p < 0.05 from day 14 (shadow.py:25-39, 75-79). DM v2 (flag `TRADER_SHADOW_DM_V2`, default OFF, computed side by side; shadow.py:40-48, 85-95) uses an IM cluster t with 48h blocks, q ≥ 6. Its looks are a crypto interim at day 21 (α = 0.025) and a final at α = 0.10 requiring mean_d > 0 (shadow.py:462-530). The stock final is at 56d.
- **Evidence accrual at h=24.** Shadow records are hourly forecasts at a 24-bar horizon. Henzi-Ziegel Prop 3.4 averages 24 interleaved products, so each sub-process sees 1/24 of the records, about one per day. Prop 3.5 adds that a stopped e-value is known only h−1 = 23 steps later, and §4 / Fig. 2 shows e-power falling faster with lag than DM. The best case collapses to days, at lag 2. On my sim, reaching 1/α then takes about 60-150 days at daily SNR ≈ 0.2-0.35. That exceeds the 28-day (crypto) and 56-day (stock) windows. The monitor would almost never fire before DM v2's fixed look does.
- **Nothing to stop early.** The challenger is silent ("no trading impact", shadow.py:14-15). Anytime monitoring earns its keep by stopping *harm* early, and here there is no harm to stop, only delay.
- **Verdict.** Flip DM v2 per the runbook: two scheduled, α-budgeted looks. Build no monitor.
- **Revisit trigger.** If shadow windows become open-ended or ≥ 90d, add a report-only Choe-Ramdas e-process on daily-collapsed loss differences, lag-2 interleaved (Src 6). Koning-van Meer (Src 7) guarantees the final DM look can then be embedded at no loss at N.

## Pre-registration sheet (owner signs before post-retrain journals accrue; any later change = a new registration and a restart of the clock)
| # | Parameter | Value | Reason |
|---|---|---|---|
| 1 | Start | first post-retrain buy under the current gate code; git sha recorded | legacy journals ran a removed gate |
| 2 | Unit / day | bp of E_{t−1}; UTC day; both books combined | spend is shared (llm_client.py:269-286) |
| 3 | Statistic | daily MTM tilt P&L net of fees and cost (formula above) | day-disjoint, so lag 1 |
| 4 | m̄ | burn-in mean multiplier (primary); 1.0 (secondary, report only) | value of skill, not leverage |
| 5 | Fees | fees.round_trip_cost_pct, split half/half | one cost model |
| 6 | Cost | c^hi = $1.00 cap (KEEP); c^lo = Σ journaled cost_usd (KILL) | conservative each way; llm_cost.json has no history |
| 7 | Burn-in | days 1-10, untested | freezes m̄ and B without look-ahead |
| 8 | Clip B | max(2.5δ, 5·1.4826·MAD₁₋₁₀), frozen; decision void if favouring clip > 5% | bounded-mean betting needs a range; the floor keeps nulls inside |
| 9 | δ (futility) | 1.0 bp/day ≈ $10/day | 10× the spend cap; smallest edge worth an LLM dependency |
| 10 | Bet | aGRAPA, c = 0.5, priors μ̂ = ½, σ̂² = ¼ | Src 1; simple and fixed (not the RL bets of Src 10) |
| 11 | α / threshold | 0.025 sequential per direction → E ≥ 40; 0.025 fallback | union bound = 0.05 |
| 12 | First look | CS day 10 and ≥ 30 lots | robustness only; validity holds at any time |
| 13 | Horizon | CS day 180 (interim status at 90) | sim: 90d gives 43% KEEP power at 1 bp/day |
| 14 | Fallback | one-sided t_{n−1} on clipped d at α = 0.025; IM K=8 if abs(ρ₁) > 0.2 | Src 7 embedding logic |
| 15 | Veto | KILL ends spend only if decision_report llm_veto CI does not exclude > 0 | the veto is part of what the spend buys |
| 16 | llm_eval b2 | diagnostic, non-voting | one α per decision |
| 17 | Inconclusive default | owner picks KEEP-status-quo or KILL | must be fixed ex ante |

## Simulation plan (sizing; I ran a small version under hwlock, R = 300-400, about 3 min in total)
- **Model.** Poisson(5) open lots; tilt notional N(0, 0.13·$3k); t₃ daily returns at 3.5% vol with a 0.5 common factor; $100k equity; drift μ added in bp/day. Command: `bash scratchpad/hwlock.sh heavy intel-scoutC -- timeout 240 $JPY scratchpad/generals/intel/w/scoutC_sim/eproc_size.py <δ> <H> <R> <μ-list>`. This takes about 1 min per 300 paths × 6 μ.
- **Result at δ = 1, H = 90:**
  | μ (bp/day) | KILL-FUTILITY fires | KEEP fires |
  |---|---|---|
  | 0 | 55% (median day 62) | — |
  | −0.5 | 85% | — |
  | 1 | — | 43% (median day 64) |
  | 2 | — | 97% |
- **Result at δ = 1, H = 180:**
  | μ (bp/day) | KILL-FUTILITY fires | KEEP fires |
  |---|---|---|
  | 0 | 94% | — |
  | 1 | — | 91% |
  | 0.5 | 36% | 21% (46% undecided) |
- **False-fire rates at every null boundary were ≤ 1.7%**, against the 2.5% allowed:
  - KEEP at μ = 0.1: 0%.
  - HARM at μ = 0: 0-0.3%.
  - FUTILITY at μ = δ: 0.7-1.7%.
- **Still to run before signing** (R = 2000, about 7 min total, one hwlock slot):
  1. Calibrate sd(d) and the lot count from the first 10 real post-retrain days, if the owner will wait. Otherwise sweep TF ∈ {0.08, 0.13, 0.2}.
  2. A left-skewed d (stop cascades), to confirm the clip rule stops a false KEEP.
  3. AR(1) d with ρ₁ = 0.3, to confirm weak-null validity empirically.

## Kill-list check (research/KILL_LIST.md, binding)
- "DM-HLN gates POLICY" (:106, a category error): **clear.** Design A is an *economic* value ledger, not a forecast-loss test. Design B keeps DM-HLN / DM v2 in its legitimate role, model promotion only.
- Conformal abstention CQR/ACI (:105): **clear.** The CS here is an evaluation instrument and gates no trade.
- Sequential bootstrap for LGB bagging (:103): unrelated.
- "Shipping concentration/sizing changes on literature priors alone" (:107): Design A *is* the required in-house measurement, and it changes no sizing.

## Owner asks
1. Make the ledger (plus its day-180 fallback) the keep/kill-spend decision owner, with llm_eval b2 non-voting. This amends runbook 03:48. It is compatible with Scout B's ask 1, since IM is still used in the fallback.
2. Sign the pre-registration table as is, or edit it now: δ, horizon (180 vs 90), primary m̄.
3. Fix the inconclusive default in row 17.
4. Capital basis. Paper P&L is notional but the LLM spend is real dollars. Normalise to paper equity (the default) or to an intended live capital K_live?
5. Cross-file, measurement-only (not in my scope to edit): at cost rollover (llm_client.py:884-885), append `{date, cost}` to `llm_cost_history.jsonl`. Today yesterday's spend survives only as a stdout print. With this, c_t becomes exact rather than bracketed.
6. Design B: flip `TRADER_SHADOW_DM_V2` per the runbook and add no anytime monitor. The revisit trigger is shadow windows of 90 days or more.

---
## 2026-09-27 R3 · Scout D — console accessibility: 12-theme contrast matrix + ISA-101 Ops theme proposal

# SCOUT D — console contrast across 12 themes + ISA-101 "Ops" theme (2026-09-27)
Landed: NEW `/home/kyle/trader/tests/test_intel_contrast_2026_09.py` (measurement only; nothing else edited).
Proof: one hwlock run, `pytest -q -p no:cacheprovider` → **214 passed, 46 xfailed(strict), 1.9 s**; py_compile OK. Line numbers
are from 2026-09-27; gui.py was being edited concurrently (cockpit lines moved ~+100 during this scout).

## Sources
1. W3C WCAG 2.2 (Rec 2023) — https://www.w3.org/TR/WCAG22/ — SC 1.4.3: text ≥4.5:1 (large ≥18pt/14pt-bold ≥3:1). SC 1.4.11: UI components/graphical objects ≥3:1 vs adjacent colours. SC 1.4.1: colour never the only carrier.
2. W3C WAI "Relative luminance" — https://www.w3.org/WAI/GL/wiki/Relative_luminance — L=.2126R+.7152G+.0722B. WCAG 2.x keeps the 0.03928 threshold, where sRGB uses 0.04045; the difference is "not significant" for 8-bit.
3. control.com "Going Gray" (ISA-101) — https://control.com/technical-articles/going-gray/ — the normal state is low-contrast grey; colour appears only when something is abnormal.
4. Industrial Monitor Direct, ISA-101 colour strategy (2026-06-18) — https://industrialmonitordirect.com/blogs/knowledgebase/isa-101-high-performance-hmi-design-principles-color-strategy — colour levels: chrome 0–5% saturation, normal values ~15% slate, alarms 85–100%. Run/stop is encoded by shape, not colour. Typical priority colours: P1 red, P2 amber, P3 yellow, P4 blue/cyan.
5. processcontrolguide / merobix on ISA-18.2 (2026) — https://processcontrolguide.com/isa-18-2-alarm-management/ , https://www.merobix.com/blog/what-is-alarm-priority-color-coding — alarm colours are reserved for alarm priorities, never reused elsewhere, and backed by a non-colour cue.
6. LADX "ISA-101: the grey screen" (2026-08-21) — https://ladx.ai/resources/isa-101-hmi-design — "Colour only for deviation… nothing else uses those colours. Ever."
7. Bloomberg UX, "Designing the Terminal for color accessibility" (2021; still the reference trading example) — https://www.bloomberg.com/ux/2021/10/14/designing-the-terminal-for-color-accessibility/ — separate deuteranopia and protanomaly schemes, because red and green carry up and down.
8. Grafana thresholds docs (current) — https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/configure-thresholds/ — the Base threshold defaults to green, so a healthy panel is saturated green. That is the anti-pattern ISA-101 removes; you have to set the base to grey yourself.
The repo helper `chart_core.contrast_ratio` (chart_core.py:1268-1277, threshold 0.03928) is WCAG-2.x exact. The test cross-checks it against an independent re-implementation to 1e-12 and checks the anchors 21.0 and #777/#fff = 4.478.

## Contrast matrix (today; `*` = below floor; text 4.5:1, nt.* 3:1)
```
pair            Batman BlkMetal Bubblegu  Dark Harley  Joker  Money  Paper Salander Space Terminal TwoFace
text.hi/base     15.00  10.97  14.06  10.33  16.24  13.52  14.76  14.17  15.69  14.86  13.33  13.63
text.hi/raised   13.18  10.39  12.52   8.68  14.53  12.27  13.07  15.93  14.71  13.45  12.27  11.96
text.hi/inset    13.96  10.65  13.21   9.35  15.43  13.03  13.87  13.05  15.25  14.21  12.89  12.66
text.mid/base     5.58   2.96*  6.13   5.41   5.24   5.09   6.10   4.82   4.58   5.06   4.94   4.49*
text.mid/raised   4.91   2.80*  5.46   4.55   4.69   4.61   5.40   5.41   4.29*  4.58   4.55   3.95*
text.mid/inset    5.20   2.87*  5.76   4.90   4.98   4.90   5.73   4.44*  4.45*  4.84   4.78   4.18*
accent/base      14.12   7.79   7.32   6.39   5.15  14.43  13.56   5.55  13.68  12.71   6.61   8.35
warn/base        14.12   7.79  11.20   8.69  16.10  16.39  13.56   2.70* 11.37  10.39   8.35  11.67
danger/base       5.38   3.05*  5.52   4.15*  5.15   5.18   6.86   4.52   5.34   6.13   4.37*  4.21*
ok/base           7.12   9.83  11.60   6.33   9.15  14.43  11.40   3.64* 11.96  11.72   5.72   9.04
chip.hb_ok        6.26   9.31  10.33   5.32   8.18  13.09  10.09   4.09* 11.22  10.61   5.27   7.94
chip.hb_stale     4.73   2.89*  4.92   3.49*  4.60   4.70   6.07   5.08   5.01   5.55   4.02*  3.69*
chip.shadow      12.41   7.38   9.97   7.30  14.40  14.86  12.01   3.04* 10.66   9.40   7.68  10.25
chip.regime      11.34   7.16   6.03   5.14   4.35*  12.21 10.33   4.73  12.21  10.52   5.69   6.73
chip.mode        14.12   7.79   7.32   6.39   5.15  14.43  13.56   5.55  13.68  12.71   6.61   8.35
nt.hb_chip_edge   1.74*  1.38*  1.69*  1.90*  1.69*  1.67*  1.91*  1.45*  1.52*  1.58*  1.43*  2.05*
nt.regime_edge    1.74*  1.38*  1.69*  1.90*  1.69*  1.67*  1.91*  1.45*  1.52*  1.58*  1.43*  2.05*
nt.mode_fill     14.12   7.79   7.32   6.39   5.15  14.43  13.56   5.55  13.68  12.71   6.61   8.35
nt.risk_ok_bar    5.72   9.04   9.56   5.08   7.73  12.21   8.68   3.10  10.68   9.69   4.93   7.29
```
How the pairs map to the code: chips use `_chip_style` (gui.py:3829: fg on bg_card, border bg_border). hb ok→green and stale→red at gui.py:4039. Regime chips: accent on bg_header (gui.py:3355). Mode chip: bg_dark on an accent fill (gui.py:4136). Risk gauge <70%: green chunk (gui.py:4112) on the bg_header groove (gui.py:2721). The cockpit sits on bg_dark (QMainWindow, apply_theme gui.py:2571).

## Failing pairs (owner items) — 46 of 228, each xfail(strict) with its ratio
- **O1 (P1). Status-chip boundary fails SC 1.4.11 in 12/12 themes** (1.38–2.05). bg_card/bg_header and bg_border all sit within about 2:1 of bg_dark. Strictly, 1.4.11 only binds when the boundary is needed to identify the component; here the chip text carries the state. Fix: give chips a ≥3:1 border (bg_border lifted, or a per-chip border in the state colour).
- **O2 (P1). The stale/danger colour, the one that matters, fails in 5 themes.** chip.hb_stale: Black Metal 2.89, Dark 3.49, Two-Face 3.69, Terminal 4.02. danger/base: Black Metal 3.05, Dark 4.15, Two-Face 4.21, Terminal 4.37. The red used for halt/flatten/stale is the least legible colour in exactly the themes built to be restrained.
- **O3 (P2). Muted text fails in 4 themes**: Black Metal all three surfaces (2.80–2.96), Two-Face all three (3.95–4.49), Salander raised/inset (4.29/4.45), Paper inset (4.44).
- **O4 (P2). Paper (the only light theme) fails 5 pairs**: warn 2.70, shadow chip 3.04, ok 3.64, hb_ok 4.09, muted/inset 4.44. It needs darker amber and green text roles.
- **O5 (P3).** Harley Quinn regime chip 4.35.
- Every fix flips its xfail to XPASS, which strict mode makes a failure, so the entry must then be deleted from `KNOWN_FAIL`.

## Ops theme proposal (design_tokens Phase B + gui.THEMES entry; measured, pinned in the test's `TestOpsProposal`)
Dark neutral grey. That is a dim-console variant of ISA-101's grey: [4] specifies light #B0–#C8 chrome, but the other 11 themes and the Jetson console are dark (owner ask A3).
| key | hex | HSV S | vs base / vs raised | role |
|---|---|---|---|---|
| bg_dark / bg_card / bg_table | #1f2124 / #292c30 / #24272a | .14/.15/.14 | — | L1 chrome |
| bg_header / bg_hover / bg_log | #313439 / #363a3f / #1a1c1f | ≤.16 | — | chrome |
| bg_border | #80868f | .10 | 4.40 / 3.82 | chip + control edges (fixes O1) |
| white (text.hi) | #e3e5e8 | .02 | 12.79 / 11.11 (inset 11.90) | text |
| muted (text.mid) | #a3a8b0 | .07 | 6.75 / 5.87 (inset 6.28) | labels |
| accent | #aeb6c2 | .10 | 7.89 / 6.86 | grey-slate selection. Mode chip text on it: 7.89 |
| green (pnl.up) | #93b59f | .19 | 7.19 / 6.25; risk bar 5.56 | data, not state |
| red (ALARM P1 only) | #ff5c5c | .64 | 5.33 / 4.63 | halt, flatten, order-error, in-hours stale |
| yellow (ALARM P2 only) | #f2b233 | .79 | 8.60 / 7.47 | warn |
| *new* ok_nominal | #a3a8b0 | .07 | 6.75 / 5.87 | "alive / at peak / ✓" |
| *new* loss (pnl.down) | #b98e8e | .23 | 5.63 / 4.89 | P&L < 0; luminance gap to green 0.10 ≥ 0.08 |
All 19 pairs pass for Ops. Minimum text ratio 4.63 (red on raised); minimum non-text 4.40.
**Nominal-state widgets that break "no saturated colour when nominal" in the 11 chromatic themes** (Black Metal is achromatic but fails O2/O3):
heartbeat ok→green (`_refresh_heartbeats` gui.py:4039); DD badge "at peak"→green (`_refresh_dd_badge` gui.py:4220); risk gauge <70%→green (`_refresh_risk_gauge` gui.py:4112); static PAPER mode chip on an accent fill (`_refresh_cockpit_banner` gui.py:4136); regime chips in accent (gui.py:3355); progress chunk and QGroupBox titles in accent (apply_theme gui.py:2726, 2735); shadow chip in **alarm yellow** in its *default ON* state (gui.py:8029); READY chip green (gui.py:10603); phase badge ✓ green (gui.py:10946); P&L tables in saturated `PAL` up/down (pnl_color gui.py:1498, from chart_core.py:1335-1336).
**Schema gap (why the Ops tokens alone are not enough):** the 13-key schema (design_tokens.py:105, 124-129) uses one key, `red`, for both alarm P1 and P&L loss, and one key, `green`, for both "healthy" and profit. ISA-18.2 exclusivity [5] cannot hold until `ok_nominal` and `loss` exist as tokens and those 10 call sites route to them. That goes against the Phase-A "do not invent more" contract (design_tokens.py:37), so it is an owner decision.

## Acceptance rule (pre-registered; evaluated by tests/test_intel_contrast_2026_09.py)
An "Ops" entry landing in gui.THEMES is auto-parametrized with **no** xfail marks. It ships only if:
(a) all 19 pairs pass: text ≥4.5:1, nt.* ≥3:1;
(b) every nominal-role colour (the 7 bg keys, white, muted, accent, green, ok_nominal, loss) has HSV S ≤ 0.25;
(c) red and yellow have S ≥ 0.50 and share no hex with any nominal role;
(d) the profit/loss luminance gap is ≥0.08;
(e) a grep of the 10 call sites above shows zero `T['green']`/`T['accent']`/`T['yellow']` in a nominal branch while Ops is active (0 saturated nominal widgets);
(f) every alarm colour also carries text or a glyph (SC 1.4.1). This is already true of the halt/STALE/DD strings.
(a)–(d) run today against `OPS_PROPOSAL` (all green). (e) needs a gui test once O6 lands.

## Kill-list check
research/KILL_LIST.md (190 lines) has no GUI/theme/contrast/HMI/alarm entry (grep: gui|theme|console|hmi|alarm|dashboard → 0 hits). This agrees with gui_review_2026-07 S8/S11 (Phase B themes are planned). Not model-facing: this is ship-direct instrumentation and UI.

## Owner asks
A1. Fix O1–O5 by tuning palette values only (Phase B). Each fix is proven by an xfail flipping.
A2. O6: approve 2 new tokens (`ok_nominal`, `loss`) plus routing the 10 nominal call sites. This is a design_tokens contract change.
A3. Ops as dark grey (proposed) or ISA-classic light grey (#B0–#C8 chrome, dark text)?
A4. Fold SC 1.4.1 into Scout B UX #4: alert-feed priority glyphs, since `_alert_color` (gui.py:3837) is colour-only per kind.
Not proven: nothing is rendered. The ratios are token-level, so Qt anti-aliasing, alpha and hover/selection states (`acc_soft`) are unmeasured.

---
## 2026-09-27 R4 · Scout E — instrument SPEC: LLM structured-output reliability replay (deferred until journals/llm_replay exists)

# SCOUT E — instrument SPEC: `scripts/llm_schema_reliability.py` (LLM structured-output reliability replay)
*INTEL W14, 2026-09-27. A SPEC, not code; no repo edits. Measurement only: reads journals, never changes a gate, config or live state. Build only after the §6 journal-field asks land — §1 explains why the headline rates cannot be computed otherwise.*

## 0. Purpose
- **Schema enforcement exists in every provider** (llm_client.py:33-36): Gemini `responseMimeType`+`responseSchema` (:2181-2182); Claude forced tool (`tool_choice` type=tool :1735, or auto+`strict` :1743-1745); OpenAI `response_format` json_schema `strict:True` (:1872-1878).
- **Output can still silently go neutral.** `_parse_response` (llm_analyst.py): fence-strip :1106-1108; missing or non-numeric `s`→0.5 :1125-1129; non-finite `s`→0.5 :1130-1135; clamp :1136; p_up clamp/non-finite→None :1140-1146; conviction clamp/inf→None :1148-1157. Transport discards (llm_client.py): Gemini MAX_TOKENS :2214-2217 and no-text/blockReason :2221-2233; Claude max_tokens :1794-1796; OpenAI `length` :1906-1908.
- **Why it matters:** one sample drives the veto (s<0.15, trading_utils.py:30) and the 2-strike liquidation (base_loop.py:1805-1809). The instrument reports failure-class rates per provider/model, plus repeat-call agreement using Scout A's E1 statistic.

## 1. Can it run today? No
- **Failures are never journaled.** `_journal_replay` runs only `if result and persist` (llm_analyst.py:611-616); the `llm_analysis` row is written only inside `if new_scores:` (base_loop.py:1795, row :1823-1842); a failure writes only a `logger.warning` (:1843-1850). No denominator ⇒ schema-invalid, truncation and refusal rates are **uncomputable**.
- **No data yet:** `journals/llm_replay/` does not exist (checked 2026-09-27; W5 ITEM 3); the 02-23…05-07 journals have 0 `llm_analysis` rows.
- **Degraded mode** once bots write (labelled in output): latency from `latency_ms`; exact-0.5 share of `s` as an UPPER bound on fallback hits; natural repeat pairs (§4a).

## 2. Inputs (read-only, gitignored `journals/`)
| Source | Written at | Fields used |
|---|---|---|
| `journals/YYYY-MM-DD.jsonl[.gz]` (trade_journal.py:3, `log_decision` :73; `.gz` only if `TRADER_JOURNAL_ROTATE_DAYS`>0, :63) | base_loop.py:1823-1842 | `action=="llm_analysis"`, `asset_type`, `forward_bars`, `scores{sym:{s,pred}}`, `model`, `prompt_sha256`, `dedup_hit`, `latency_ms`, `cost_usd` (from `_LAST_CALL_META`, llm_analyst.py:653-661; latency at :604 = call_model + any call_llm fallback + parse) |
| same files, `action=="llm_advisor_v2"` (advisor_v2 on only) | llm_analyst.py:422-437 | `scores{sym:{s,p_up,conviction,abstain}}`, `model`, `prompt_version`, `prompt_sha256`, `dedup_hit` |
| `journals/llm_replay/YYYY-MM-DD.jsonl` (pruned >45 d, :1398) | llm_analyst.py:1377-1391 | `ts`, `asset_type`, `candidates`, `live_scores{sym:s}`, `live_model` |
| `journals/llm_qualify/shadow_scores.jsonl` (optional) | scripts/llm_qualify.py:856-866 | `evidence_id`, `symbol`, `prod_model`, `prod_s`, `free_model`, `free_s`, `*_latency_s`, `*_fallback` |

Exclude `dedup_hit==True` everywhere (cache replays, latency 0; llm_analyst.py:545-550). Window opens at `--since` (pre-registered activation timestamp + git sha, SCOUT_C pre-reg row 1). Rows before the §6 producer change go under `legacy`.

## 3. Metrics, per provider/model
Grouped by `provider` (pure prefix map mirroring `llm_client._provider_for`, :144), answering `model`, `asset_type`, `path`. Every rate is k/n with a Clopper-Pearson 95% CI.

| Metric | Definition | Needs |
|---|---|---|
| schema_invalid_rate | outcome ∈ {parse_fail, not_object, empty, transport_discard} / attempts | A1, A2 |
| truncated_rate | finish ∈ {MAX_TOKENS, length, max_tokens} / attempts | A2 |
| refusal_rate | finish ∈ {SAFETY, RECITATION, PROHIBITED_CONTENT, refusal, content_filter} or non-empty blockReason, / attempts | A2 |
| partial_rate | n_symbols_returned < n_symbols_sent | A1 |
| fence_fallback_rate | fence-strip needed to parse | A3 |
| s_default_rate ("fallback-0.5") | per symbol: `s` missing / non-numeric / non-finite → 0.5 | A3 (degraded: exact-0.5 upper bound) |
| nonfinite_rate / out_of_range_rate | per symbol: raw s, p_up or conviction non-finite / raw s∉[0,1], p_up∉[0,1], conviction∉{1..5} | A3 |
| latency p50/p95/max | non-dedup attempts, `latency_ms`; failures via A1 | degraded / A1 |
| cost_per_attempt | `cost_usd` + discards charged by `_charge_discarded` (llm_client.py:1088) | A2 |

## 4. Repeat-call agreement (Scout A E1, reused verbatim)
- **Statistics:** median |Δs|; V = among pairs with min(s1,s2)<0.25, share where exactly one side < `LLM_VETO_THRESHOLD` 0.15; plus sign agreement at `NEUTRAL_BAND` 0.02 (llm_qualify.py:90). Power floor: ≥300 pairs from ≥30 evidence ids, else `insufficient_n`.
- **(a) Natural pairs, $0:** two non-dedup `llm_analysis` rows with the same `asset_type`, `prompt_sha256` and `model` at different ts (same sha = identical prompt bytes). They arise when the dedup TTL lapses, or when a cached s within 0.05 of the veto forces a fresh call (llm_analyst.py:355-360). Veto-enriched sample ⇒ reported separately.
- **(b) Deliberate pairs:** `shadow_scores.jsonl` rows with `free_model == prod_model` (llm_qualify `--shadow`, prod listed as its own candidate). Those calls are a **separate owner-approved spend run** (≈$0.20, Scout A ask A1); this instrument never calls anything.

## 5. Pre-registered thresholds
PASS if CI upper ≤ threshold; FAIL if CI lower > threshold; else `insufficient_n`. Minimum n per model: 300 attempts for 1% thresholds, 600 for 0.5% (rule of three). **The instrument only REPORTS; every action below is an OWNER call.**

| Metric | Threshold | Anchor | FAIL → owner item |
|---|---|---|---|
| schema_invalid_rate | ≤1.0% | stricter than qualification's `SCHEMA_VALID_MIN_PCT` 98 (llm_qualify.py:82) — this is prod | demote provider / reorder fallback in llm_config.json |
| truncated_rate | ≤0.5% | — | revisit `max_tok` formula (llm_analyst.py:569); gate-affecting ⇒ shadow first |
| refusal_rate | ≤0.5% | — | inspect untrusted-headline/injection path; possible demotion |
| s_default_rate | ≤0.5% | c26 D33 intent | treat a defaulted s as *missing*, not neutral; default-OFF flag + shadow |
| nonfinite_rate | ≤0.1% | G4-07 guards exist | provider demotion |
| out_of_range_rate | ≤0.5% | clamp contains it | report only; feeds demotion discussion |
| latency p95 | ≤45 s | `BUDGET_S = _ANALYST_TIMEOUT_SEC` (llm_qualify.py:79, :84; llm_analyst.py:48); >60 s (`P95_MARGINAL_MAX_S`, :85) = severe | change provider or timeout |
| repeat median \|Δs\| / V | <0.05 and <5% | Scout A E1 | temperature 0, or 2-sample mean before veto/strike (default-OFF flag, shadow) |

## 6. Journal fields MISSING today — asks for the journal producer (ENGINE/INTEL), before bots restart
All measurement-only and fail-soft; none changes `analyze_trades`' return value.
- **A1 — attempt record for every non-dedup call, failures included.** Producer llm_analyst.analyze_trades :575-604 (returns `{}` at :600-601 without a record; replay gated at :611). New `action:"llm_call"` row with `ts`, `asset_type`, `requested_model` (:567), `model_used`, `path` (call_model :584 | call_llm fallback :594), `outcome` ∈ {ok, empty, transport_error, parse_fail, not_object, partial}, `n_symbols_sent`, `n_symbols_returned`, `latency_ms`, `response_chars`, `max_tokens` (:569), `temperature`, `prompt_sha256`, `advisor_v2`.
- **A2 — transport finish/stop reason and usage.** Gemini `finishReason`/`promptFeedback.blockReason` (llm_client.py:2209, :2222, :2227-2228), Claude `stop_reason` (:1787), OpenAI `finish_reason` (:1905) reach only `print` today. Expose `get_last_call_meta()` beside `get_last_model_used` (:246) → `{finish_reason, block_reason, usage_in, usage_out, http_status}`.
- **A3 — per-symbol parse flags from `_parse_response` (llm_analyst.py:1089-1185):** `fence_stripped`, `raw_s`, `s_defaulted`, `s_nonfinite`, `s_clamped`, `raw_p_up`, `raw_conviction`, `p_up_nonfinite`, `conviction_nonfinite`. Carry into the replay record (:1377-1388) as `parse_flags{sym:{…}}`, and `s_defaulted` into `llm_analysis` scores (base_loop.py:1829-1831).
- **A4 — replay record lacks `prompt_sha256`, `latency_ms`, `dedup_hit`** (llm_analyst.py:1377-1388), so it cannot be joined to `llm_analysis` rows. Copy them from `_LAST_CALL_META`.
- **Cross-file note (not applied; owner/ENGINE):** base_loop.py:1827-1828 says `s` is "journaled as null when the provider omitted it". That never happens: `_parse_response` always substitutes 0.5 (:1125), and an omitted symbol is simply absent from `new_scores`. D33 is inert; A3 is the real fix.

## 7. CLI and output
- **CLI:** `--days 14`, `--since ISO`, `--asset {crypto,stock,all}`, `--journals DIR`, `--shadow FILE`, `--json F` (default: stdout table only, no file). Always exits 0 (fail-open, like llm_qualify).
- **JSON:** `{schema_version:1, generated_at, git_sha, since, mode:"full"|"degraded", input_files:[{path,sha256,rows}], models:{"<prov>/<model>":{attempts, n_success, rates:{<metric>:{k,n,rate,ci_lo,ci_hi,threshold,verdict}}, latency:{n,p50,p95,max,threshold_s:45,verdict}, cost_per_attempt}}, repeat:{natural:{n_pairs,n_evidence, median_abs_ds,V,sign_agree_pct,verdict}, shadow_self:{…}}, legacy:{…}, owner_items:[…], notes:[…]}`. Deterministic: sorted keys, input hashes recorded.
- **When built:** rows in docs/MODULES.md + scripts/README.md, and `tests/test_llm_schema_reliability.py` (pure helpers, synthetic rows, Mac-runnable).

## 8. Cost, Mac-safety, kill list
- **Cost: $0.** Reads journals only, zero LLM calls. Any re-call (e.g. §4b) is a separate owner-approved llm_qualify run.
- **Mac-safe:** stdlib only (json, gzip, math, statistics, hashlib, argparse). Does **not** import `llm_client`/`llm_analyst` — their read-shaped getters write `<repo>/llm_cost.json` (tests/README hygiene item 3). Constants 45 / 0.15 / 0.02 copied as literals, pinned by a source-text test to llm_analyst.py:48, trading_utils.py:30, llm_qualify.py:90. One journal day in memory at a time: <100 MB.
- **Kill list (research/KILL_LIST.md, read 2026-09-27):** no entry covers LLM output reliability, provider scoring or repeat-call consistency. Adjacent kills respected: "shadow DM-HLN gating POLICY" (this gates nothing) and "ship on literature priors alone" (this is the in-house measurement). **CLEAR.**
