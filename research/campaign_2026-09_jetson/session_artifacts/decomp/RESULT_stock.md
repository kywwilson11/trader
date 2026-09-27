# DECOMP-2 RESULT — stock study decomposition (2026-09-27, DB copy 07:33, 11 COMPLETE / 1 RUNNING / 0 PRUNED)
Tool: /home/kyle/trader/scripts/trial_score_decomposition.py (unmodified). Inputs: stock_v2_study_copy_0733.db (cp of live DB, no -wal; tool
re-snapshots into run_*/ and opens `mode=ro`), log snapshot train_stock_log_0733.txt, store stock_training_data.parquet (mtime 09-26 23:38, before
the 06:29 start), --max-rows 200000 (= trainer: 2222/ticker, 191251 rows). Flags per log banner: LONG_ONLY on, REPAIRS/V3 off, FAILED_TRIAL_PRUNE.eff=True.
Run A (tool as landed, L3): 81 MB/1.0 s. Run B (proxy_wrap.py hides L3 keys -> forces L2): 618 MB peak (18 MB OVER cap; 18.7 s; the tool reads
12 projected cols x 1.54M rows before its per-ticker cap). hwlock heavy, threads 1. Fidelity: L2 val rows == recorded L3 n_rows 11/11; L2 train
rows == log [CACHE] (#0 87358/112152/136979). #11 finished 07:34 after the copy (-0.390, thr 0.06, fb48) — excluded; 20 COMPLETE not reached.
PRE-REGISTERED CLASSIFICATION (applied mechanically; PRIMARY = first match in order d > c > economic)
 (d) THRESHOLD/FREQUENCY MISMATCH if ANY of:
     d_floor  a fold has n_trades < 10 [L3]  |  a fold Sharpe is exactly 0.0 (compute_sharpe's floor) [L1]
     d_rate   a fold's threshold_pass_rate (share of val rows with pred > thr) < 0.5 % or > 20 % [L3 only]
              (20 % ~ the live cap MAX_TRADES_PER_SYMBOL_PER_DAY crypto 4/24h = 16.7 % of hourly bars;
               0.5 % = < 1 signal-hour per 8 days per name => ~<10 non-overlapping trades per fold)
     d_cost   trade_threshold < TXN_COST_PCT[book] (the searched rule admits trades its own forecast
              says lose money after the round trip) [L1, definitive]
     (flag only, not primary: d_admit = thr < fees.required_edge_pct — the LIVE admission floor)
 (c) SCORING ARTEFACT if ANY of:
     c_sign   sign(score) != sign(mean fold Sharpe) (the -0.5*std term sets the sign) [L1]
     c_rank   the regime penalty changes THIS penalised trial's rank among COMPLETE trials [L1]
     c_ann    trades_per_year < 1 in a fold (sqrt(max(tpy,1)) floor degenerate) [L3]
 ECONOMIC class (reported for EVERY trial, also as its own histogram — the 'why negative' answer):
   L3: pooled over folds, gross = trade-weighted mean, SE = pooled sd / sqrt(sum n_trades)
     a   no edge        gross <= +1 SE   (sub-tag a0 |gross| <= 1 SE, a- gross < -1 SE)
     b   over-trading   gross > +1 SE and cost_drag >= gross (net <= 0)
     +   net edge       gross > +1 SE and net > 0
     ((a) is tested before (b) so a gross indistinguishable from 0 is NO EDGE, never 'over-trading')
   L2 PROXY (no L3 attrs): per fold, the gross band [g(10), g(n_possible)] consistent with the observed
     fold Sharpe S (g increasing in n when S < 0):
     a   g(n_possible) <= +1 SE(n_possible)          (even the most favourable trade count shows no edge)
     b   g(10) > +1 SE(10) and S <= 0                 (every feasible trade count has gross > 0, net <= 0)
     +   S > 0                                        (net-positive fold)
     a~  band straddles 0 and the zero-skill trade count n* is feasible (10 <= n* <= n_possible):
         the observed Sharpe is exactly what a RANDOM long ranker paying the cost earns at n* trades
     ?   otherwise UNRESOLVED
     trial = majority of its folds (tie => '?'); verdict tagged PROXY.
## Histograms (11 COMPLETE)
L3 path (tool default — uses the attrs; L2 is SKIPPED when every row has L3, trial_score_decomposition.py:937): PRIMARY d 7 | a 2 | c 2     ECONOMIC [L3] a0 6 | a- 1 | + 4 | b 0     flags d_rate 5 d_floor 2 d_cost 2 d_admit 2 c_sign 3 c_rank 2 c_ann 2
L2-PROXY path (forced): PRIMARY d 4 | c 4 | a 2 | ? 1     ECONOMIC [PROXY]  ? 5 | a~ 2 | a 2 | + 2
  Proxy vs L3: every RESOLVED proxy call (6/11) agrees with L3 (+,+ / a,a~,a,a~ -> a0/a-); 5 unresolved (?) — validates DECOMP-1's crypto proxy direction.
Layer 3 AVAILABLE: all 9 LAYER3_KEYS + regime_trade_decomp present on trials 0-10 (RUNNING #11 had only cfg); per-fold lists aligned with fold_sharpes.
## L3 per-trial table (pooled over folds; gross/net/SE in % per trade; rand = zero-skill random long ranker gross from Run B, trade-weighted) #4 (11th: -0.528, gross -0.635 z -2.36, a-, d_floor) and all folds: l3_table_0733.txt
 #  score  thr  fb tk   N (folds)        gross    SE     z     net  cost  hit | rand   excess z_ex | L3  proxy P
 6 +0.000 0.27  24 tb     6 (0/0/6)      +2.934 2.057 +1.43 +2.824 0.110 0.67 | -0.039 +2.97 +1.45 | +   ?     d(floor)
 2 -0.018 0.62  48 tb   198 (125/40/33)  +0.126 0.216 +0.58 -0.002 0.128 0.40 | +0.017 +0.11 +0.50 | a0  ?     a
 5 -0.035 0.33  18 raw 2044 (789/698/557)+0.244 0.090 +2.72 +0.138 0.105 0.52 | +0.196 +0.05 +0.53 | +   +     d(rate)/c_sign
 3 -0.037 0.96  18 raw 1178 (538/375/265)+0.366 0.142 +2.58 +0.256 0.110 0.51 | +0.234 +0.13 +0.93 | +   +     d(rate)/c_sign
10 -0.047 0.68  18 raw 1840 (1007/599/234)+0.237 0.099 +2.38 +0.134 0.102 0.51 | +0.279 -0.04 -0.42 | +   ?     d(rate)/c_sign
 7 -0.109 0.73  12 tb   693 (501/71/121) +0.034 0.088 +0.39 -0.076 0.110 0.46 | +0.019 +0.02 +0.18 | a0  a~    a
 8 -0.225 0.92  12 tb   503 (339/82/82)  -0.011 0.107 -0.11 -0.121 0.110 0.45 | +0.000 -0.01 -0.11 | a0  ?     c(rank)
 9 -0.306 0.68  48 tb   163 (132/18/13)  -0.091 0.217 -0.42 -0.201 0.110 0.43 | +0.070 -0.16 -0.74 | a0  a     c(ann)
 0 -0.324 0.05  12 tb   605 (515/10/80)  -0.026 0.093 -0.28 -0.136 0.110 0.47 | +0.027 -0.05 -0.57 | a0  ?     d(cost)
 1 -0.355 0.09  48 tb   349 (176/127/46) -0.075 0.125 -0.59 -0.183 0.108 0.44 | +0.008 -0.08 -0.66 | a0  a~    d(cost)
## Other requested numbers (OBJECTIVE unless marked)
- [FAILED-TRIAL]/PRUNED: 0 log lines, 0 PRUNED rows. But the study BEST is #6 = 0.0 via compute_sharpe's <10-trade floor (hypersearch_v2.py:753)
  on every fold (n 0/0/6, pass 0/0/0.3 %) — a no-trade model outranks all 10 traded trials; FAILED_TRIAL_PRUNE does not cover this door.
- Thresholds: 2/11 below TXN 0.11 (0.05, 0.09) and the same 2/11 below F = required_edge_pct('stock', 0.05) = 0.226 (rt 0.113 x 2.0); 9/11 above F.
- Fold Sharpes: 7 > 0, 4 == 0.0, 22 < 0 (of 33); fold 1 is <= 0 in 11/11 (its random-ranker gross is -0.05..-0.15 %: a negative-drift slice).
- Regime penalty (legacy *0.7, :1527): fires 6/11 (#0,3,5,7,8,10) — BEAR 5, SIDEWAYS 1 (#7); c_rank 2 (#0,#8); c_sign 3 (#3,5,10: mean fold +0.19..+0.28,
  score < 0 via -0.5*std, :1476). Regime decomp: raw-fb18 trials gross bull +1.8..+2.2 %, bear -1.1..-1.5 % — long beta.
- Spearman(score, thr/fb/seq_len) +0.18 (p .60)/-0.09 (.79)/-0.22 (.53), pre-penalty +0.16/+0.05/-0.23; Spearman(random-ranker Sharpe,
  mean fold Sharpe) = +0.65 (p 0.032) — zero-skill arithmetic orders the study (crypto 0.51).
- Definitive (a)/(b) split (L3 vs zero): a 7 (a0 6, a- 1), b 0, + 4. Against the DRIFT benchmark: 0/11 trials have excess gross > 1 SE over the
  random ranker (max z_ex +0.93, #3; #6 is 6 trades). The 3 raw '+' are the random ranker's drift (+0.20..+0.28 %/trade > 0.11 cost) — not selection.
## Tool defects / caveats (measurement tool; not fixed here — landing protocol)
- D1 (material): the L3 path skips L2 (:937), so L3 '+' means gross > 0, never gross > drift — a zero-skill long book in a rising slice reads
  "net edge". Run both paths (as here) or add the random-ranker benchmark to L3. D2: economic_class_l3 (:780) has no min-N (#6 '+' on 6 trades).
- D3 (JUDGMENT): PASS_RATE_HI 0.20 (:95) is crypto-calibrated; stock cap 3/day (strategy_config.py:74-77) over unsettled bars/day (BPY owner item).
  Without d_rate the L3 primary is d 4 | c 5 | a 2; at 3/6.5 = 46 % only #5, #10 still flag.
## Crypto vs stock
Same mechanism at the core, different surface. Both: no selection skill (crypto a~ 13/16 proxy; stock 0/11 beat the random ranker at L3), a
zero-skill baseline that predicts the ordering (rho 0.51 vs 0.65), the regime penalty firing on BEAR for a long-only model, legacy *0.7 flattering.
Different: crypto's drift < cost (0.60 %) so zero-skill is net-negative and 16/16 thresholds sit below the 1.20 floor; stock's cost is 0.11 %,
9/11 thresholds clear F, and the raw-target drift (+0.2..0.3 %/trade) EXCEEDS cost — so stock has net-positive-gross trials whose score is
negative only through the fold dispersion (-0.5*std, c_sign) and the penalty. Stock's top of the study is set by the 0.0 floor (#6) and by target_kind
(raw trials are scored on drift-carrying returns; tb trials on barrier returns with ~0 drift) — not by model quality.
## What this implies for the next stock retrain (levers, not flips — owner's call)
- Zero point: the <10-trade 0.0 floor wins the study; a min-trade prune (FAILED_TRIAL_PRUNE-style) is the lever.
- Yardstick: score excess over the random/drift ranker (benchmark-relative) — raw vs tb target_kind currently changes the P&L measure itself.
- Fold 1 (negative-drift slice) sinks every trial: regime conditioning / entry masks are where selection could show; more same-design trials rank drift.
- Thresholds are NOT the stock bottleneck (9/11 >= F); -0.5*std and the penalty form (TRAINING_REPAIRS_V1) set sign/rank of the top trials.
