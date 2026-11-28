# Issue #285 — CONTRACT (frozen 2026-11-28, before any cell correlation)

## 1. Variables and identities

| symbol | definition |
|---|---|
| N | `l1_neg_hit` (validation legal-negative L1 hit rate) — the negative-sharing axis |
| C | `l3_collision` (catalogue three-code collision) — the collision axis |
| P | `l1_pos_hit` — positive sharing |
| ER | `l1_hard_er` = **P/N exactly** (identity asserted per SID; ER carries no independent information) |
| ALL | `all1_chain` (raw chain L1 triple intersection; #283/#284 definition, row-weighted) |
| ALL_fork/ALL_merge | the same statistic on the fork/merge classes (cross-structure controls) |

All 129 SIDs are evaluable at every stratum; the low condition is a stratifier, **not a hard gate**.

## 2. Frozen strata (training-fold quantiles only)

For each outer fold f: quantiles `q1/3, q2/3` of N and of C are computed on the **training rows only**
(fold != f); the fold's test SIDs are binned as low (`<= q1/3`), high (`>= q2/3`), else mid. No bin edge
uses recall. The pooled descriptive grid aggregates these fold-local bins. A global-tercile variant is
reported only as a cross-check of the #284 numbers.

Cells: `LL = lowN & lowC`; `LN = lowN & ~lowC`; `LC = ~lowN & lowC`; `HH = highN & highC`;
`MM = mid & mid`. **Minimum support: 10 independent SIDs per cell**; a cell below it is reported but
marked **unidentifiable** and cannot support an interaction claim. Cells are never merged to chase a result.

## 3. Frozen hypotheses and statistics

- **H37**: under low N (and/or low C) the L1 behavioural-fidelity association is stronger (conditional law).
- **H38**: the apparent conditioning is metric aggregation / range restriction from the strong-behaviour
  sampling or a low-recall-end regression shape (no mechanism label needed).
- **H39**: chain ALL has incremental value over the plain L1 positive rate P under the same condition.
- **H40**: the conditional relation does not generalize (driven by a few SIDs/weak identities/high-recall
  tail; or only in-sample).

Statistics per cell: Spearman/Kendall of P, ER, ALL, ALL_fork, ALL_merge with recall; slope (OLS of
recall on the standardised feature); rank-reversal proportion; SID-cluster bootstrap 95 % CI (2,000 reps,
127 clusters). **Interaction test**: Δρ = ρ(LL) − ρ(HH) for the same feature; supported iff the 95 %
cluster-bootstrap CI of Δρ excludes 0 **and** both cells have ≥10 SIDs. "One significant, one not" is not
an interaction. Strict monotonicity = 0 rank reversals (reported as such, in-sample only); probabilistic
rank rules are reported as reversal proportions, not as laws. Counterexamples = the largest-|Δfeature|
rank-inverted pairs, listed and never dropped. LowN/lowC-alone discrimination: AUC of N and C for
`recall > 0.06`, plus the cell recall ranges. Incremental information of P/ALL beyond N,C: partial
Spearman after rank-residualising out N and C (within LL), with the same cluster bootstrap. Robustness:
strong-only, leave-one-SID-out ranges, and the 2 time-postordered labels as an explicitly non-blind
sensitivity (they are not in the 129).

## 4. Pre-registered predictors (training-fold quantities only)

| id | features (extra terms over the six #277 metrics) |
|---|---|
| A | — (the #277 corrected Ridge6 baseline) |
| B | `G_N` + `P*G_N` (2 terms) — the finite conditional interaction on the N gate; `G_N=1[N<=q1/3(N)]` |
| C | `ALL` + `ALL*G_N` (2 terms) — the chain-ALL main effect and its interaction with the same gate |
| D | `P*G_N*G_C` (1 term) — the smooth double-condition model, with continuous ramps
`G(x) = clip((q2/3 - x)/(q2/3 - q1/3), 0, 1)` from the training-fold quantiles |

A pre-registered alternative `B_C = G_C + P*G_C` (the C gate instead of N) is reported as a fixed arm,
not selected. Selection: per outer fold, inner OOF (the four remaining fold ids as inner groups) among
{B, C, D}; pick the single best if it beats A's inner MAE, else A. All standardization, alpha and gate
edges are fitted inside the training part. Fixed arms B/C/D/B_C are reported for ablation, with BH over
the three family comparisons (A vs B, A vs C, A vs D).

Primary criterion: paired SID-cluster bootstrap 95 % CI of MAE(A)−MAE(model) lower bound > 0 **and**
Spearman/Kendall, reversal proportion and the >0.06 TP/FP/FN not worse. Strict 120-strong subset,
0.065 has no positive labels → NOT_EVALUABLE. Selection optimism: the 129 cohort has been reused across
#276-#284; every OOF number is exploratory/internal and cannot prove future validity.

## 5. Prospective freeze

One clearly-labeled research candidate (the best pre-registered arm) is frozen with formula, thresholds,
domain, input hashes and predictions for the truly unlabeled SIDs (the 122 pending from the 124-row
queue, minus the 2 time-postordered labels). #277 and the #284 files are never overwritten. Replacement of
#277 requires the historical gate **and** a future separately-authorized independent-label validation;
otherwise the candidate stays exploratory. Power: per-SID paired error differences → required independent
SID count at 80 % power (one-sided 0.05), plus the conditional strata effective sample.
