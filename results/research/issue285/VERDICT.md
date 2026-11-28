# Issue #285 — VERDICT

**Decision: the conditional rank association is real and supported by the pre-registered interaction
test; it is a probabilistic rule confined to the low-N∩low-C cell, partly explained by range
restriction, and it does not produce a predictor increment. P is at least as informative as chain ALL;
the two low axes are one source. #277 is retained; the best pre-registered arm is frozen only as an
exploratory research candidate. No Stage3 was started; no public infra was modified.**

## C1–C5

| C | evidence | verdict |
|---|---|---|
| C1 | ER=P/N exact (0.0); N–C 0.864, N–P 0.801, C–P 0.739, PC1 0.888; fold-local tertiles and per-fold edges recorded; cells LL 34/LN 10/LC 7/HH 39/other 39; low-set Jaccard 0.667; #284 global bands reproduced (0.907/0.846); 129/120/9/13/6 and 127 clusters unchanged; 2 time-postordered labels as sensitivity | PASS |
| C2 | joint grid with CIs and reversal rates; interaction Δρ (P +0.711 [0.334, 1.075], ALL +0.844 [0.429, 1.210]) supported; P ≥ ALL within LL (0.951 vs 0.926); partial after N,C (P 0.962, ALL 0.925); no strict monotonicity (7.7 %/10.9 % inversions) with counterexamples; low axes alone AUC(−N)=0.9025, AUC(−C)=0.9038; 11/13 highs in LL; robustness strong-only/without-high-tail/LOO; range restriction partial (truncated P 0.878, ALL 0.815) | PASS |
| C3 | A reproduced exactly 0.0029719/0.6666; B/C/D/B_C and the #284 raw-ALL reference executed; all paired CIs contain 0 (C is the best at [−7.63e−5, +6.36e−4]); B/D worse; 13/0/0; 0.065 NOT_EVALUABLE; power 383 SIDs (C), 847 (selected), within-LL MDE 2.2e−4 at n=34 | PASS |
| C4 | one clearly-labelled candidate C (ALL+ALL·G_N, G_N = 1[N ≤ 0.0089066], alpha 0.01, coefficients and hashes recorded) with 122 pending + 2 sensitivity predictions; #277/#284 untouched; replacement rule stated | PASS |
| C5 | CONTRACT/HYPOTHESES/DIAGNOSIS; threshold and grid data; counterexample list; per-SID features; OOF/CI; formula; power; 5 deterministic tests; logs; infra drift 0; commit+push | PASS |

## Hypotheses

| id | status | reason |
|---|---|---|
| H46 (conditional law stronger under low N/C) | **SUPPORTED (descriptive)** | interaction Δρ CIs exclude 0 for P/ER/ALL; within-LL P 0.951 [0.861, 0.987] vs HH 0.240 [−0.128, 0.602] |
| H47 (only aggregation/range restriction) | **REFUTED (as "only")** | truncation to the common recall interval lowers P 0.951→0.878 and ALL 0.926→0.815, but the gap to HH (0.240/0.082) remains; the low axes alone already separate the high tail (AUC(−N) 0.9025) |
| H48 (chain ALL extra over P) | **REFUTED** | P 0.951 ≥ ALL 0.926 within LL; partial 0.962 vs 0.925; ALL·gate ≈ raw ALL (ΔMAE 4.2e−6) |
| H49 (gate passes the MAE gate) | **REFUTED** | all candidate CIs contain 0; B/D worse than A; selected [−1.51e−4, +4.69e−4] |
| H50 (N and C are one source) | **SUPPORTED** | N–C 0.864, PC1 0.888, low-set Jaccard 0.667, lowN alone 11/13 high recalls |
| H37/H38 (from #285 text: conditional law / range restriction) | mapped to H46/H47 above | same evidence |

## Knowledge increment

1. The low-N and low-C strata are essentially **one axis**; the "double condition" is a concentration
   of the same low-activity/low-collision regime, not an independent second condition.
2. Under that condition the recall ranking of P (and of ALL) is strong but **probabilistic**
   (7.7 %/10.9 % inversions), partly range-restriction, and **not a linear-predictor gain**.
3. **P suffices**; chain ALL adds nothing (H39 refuted), and the gate interaction is redundant with
   the raw ALL term.
4. The low axes alone already identify the high-recall tail (AUC ≈ 0.90); the extra P/ALL information
   is a within-cell ranking, not a global discriminator.
5. Power: ≈383 independent SIDs for the global C-arm effect; the 34-SID conditional cell cannot be
   validated at this scale — a future independent-label campaign is required.

## Closure checklist (skill §6.4)

1. Production code implemented: yes — `common285.py`, `audit285.py`, `grid285.py`, `model285.py`,
   `freeze285.py`, tests; all executed with logs/exit codes.
2. Required tests executed: yes — 5 deterministic tests (fold quantiles/ramps, cell labels, ER
   identity, model matrix, AUC direction) pass; all run-time assertions (exact reproductions, ER
   identity, support thresholds) held.
3. Required training completed: **N/A** — analysis-only issue.
4. Infra boundary respected: yes — FROZEN; start/close snapshots added/removed/changed = 0; assets and
   #277/#284 artifacts only read.
5. Executable unfinished work: none in scope.
