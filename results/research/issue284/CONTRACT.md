# Issue #284 — CONTRACT (frozen 2026-11-28, before any new statistic)

## 1. Definitions (locked to #283)

- Chain class: `chain` = pure induced `a→m→b` (`in1&out2&!in2&!out1&!e12&!e21`); the
  reciprocal/closed variants (`chain_both`) are reported separately and never merged into the main
  variable.
- `all1_chain` = row-weighted P(L1_a = L1_m = L1_b) over the `chain` rows of a SID
  (`counts[ALL]/support` at k=1, exactly the #283 JOIN_TABLE definition). Centre-balanced
  (`macro`) values are reported separately.
- L2/L3 cumulative ALL: the same ALL partition at cumulative prefix k=2,3.
- Three marginals per SID: `p_am`, `p_mb`, `p_ab` (sums of the partition counts); the five patterns
  AM/MB/AB/NONE are affine functions of `ALL` given the marginals.
- **Feasible ALL interval** given the marginals (five-partition non-negativity):
  `lo = max(0, (p_am+p_mb+p_ab−1)/2)`, `hi = min(p_am,p_mb,p_ab)`. The marginal-conditional residual
  is `ALL_norm = (ALL−lo)/(hi−lo)` (0 if `hi<=lo`), and the raw residual `ALL−lo`. Both are frozen.
- `cond_refine` = P(L2 non-ALL | L1 ALL) = 1 − trans(ALL→ALL at k2)/Σ_{p2} trans(ALL→p2 at k2);
  `p3_split` = P(L3 pattern NONE | L1 ALL) computed exactly from the k2→k3 transitions with the
  per-pattern k2 denominators (no approximation).
- `true_three_code_collision` = 1 − (# distinct (L1,L2,L3) triples)/#items; `l3_collision` from the
  #277 feature set is the catalogue collision; the two are never substituted for each other.
- Training population: the full 28,391,948-row triplet asset; validation population: unseen
  validation pairs with training-graph intermediates (population 2 of #283).

## 2. Matching (pre-frozen tolerances, not chosen on recall)

Primary matched set (chain-marginal-equal + overall-metric-near-equal + discrimination controls):
| control | tolerance |
|---|---|
| chain p_am, p_mb, p_ab | each abs ≤ 0.005 |
| l1_pos_hit | abs ≤ 0.01 |
| l1_hard_er | relative ≤ 20 % |
| l2_pos_hit | relative ≤ 20 % |
| l3_collision | abs ≤ 0.01 |
| l3_refine_bits | relative ≤ 10 % |
| L1 distinct codes (occupancy) | relative ≤ 10 % |
| L1 code entropy | relative ≤ 5 % |

Secondary matched set: the same six-metric/L2/collision/L3/occupancy controls **without** the chain
marginal equality (the #283 146-pair style set). Degree/popularity are SID-invariant (all SIDs assign
codes to the same 24,587 items), so they cannot confound SID-level contrasts; this is stated, not
"controlled". Pairs are resampled by SID cluster (127 label clusters), never as independent pairs.
Leave-one-out: recompute the matched Spearman dropping each SID to check single-SID dominance.

## 3. Conditional analysis (strata frozen before labels are split)

- Bands: terciles of the 129 SIDs' feature values (features only, label-free): `l1_neg_hit`,
  `l2_pos_hit`, `l3_refine_bits`, `l3_collision`. The same band edges are used for all models; within
  each band the ALL-vs-recall association is reported with SID-cluster bootstrap CIs.
- Conditional metrics computed per SID from the #283 transition tables: `cond_refine`, `p3_split`,
  `P(L2 ALL | L1 ALL)`, `P(L3 ALL | L2 ALL)`, and the L2/L3 five-pattern distributions after L1
  ALL. Denominators and NA are explicit; zero-denominator SIDs are NA, never 0.

## 4. Candidates (pre-registered, at most 2 extra free terms)

| id | features |
|---|---|
| A | the six #277 features (frozen baseline) |
| B | A + `all1_chain` (the single main hypothesis) |
| C | A + `ALL_norm` (marginal-conditional residual diagnostic) |
| D | A + `all1_chain` + `cond_refine` (main effect + one conditional-refinement feature) |

Procedure: identical to #277/#283 — same 129 rows, same 5 folds, StandardScaler fitted inside each
training part, inner alpha grid {0.01,0.1,1,10} by inner SID-group CV, no outer-label use. Selection:
per outer fold, inner OOF among {B,C,D}; choose the single candidate with the lowest inner MAE if it
beats A; otherwise A. Fixed arms B/C/D are also reported (no selection) for ablation. Secondary
correction: BH over the three candidate family tests (A vs B, A vs C, A vs D).

Primary criterion (same as #283): SID-cluster paired bootstrap 95 % CI of MAE(A)−MAE(model) has
lower bound > 0 **and** Spearman/Kendall, ranking reversals and >0.06 fp/fn do not worsen.
Strict 120-strong split reported; 0.065 has no positive labels → NOT_EVALUABLE.

## 5. Power analysis (frozen formula)

Per-SID absolute-error differences `d_i = |y_i−A_i| − |y_i−B_i|`; cluster mean and SD. Required
independent SID count for 80 % power at one-sided α=0.05:
`n = ((1.645+0.8416)·SD/mean)^2` when mean>0; minimum detectable mean at n=129:
`(1.645+0.8416)·SD/√129`. Cluster-level (127 label clusters) values are reported as the conservative
variant. No new labels are fabricated.

## 6. Prospective freeze (no Stage3)

Before reading any truly unknown Stage3 label, freeze the research candidate B: full-cohort fit
(6 features + `all1_chain`) with the frozen alpha procedure; record coefficients, scaler, training
domain (129 SIDs, support thresholds), input sha256; write predictions for the label-pending SIDs
(124 prospective minus the 2 SIDs whose Stage3 results are already bindable = 122, plus recovered
unlabeled SIDs from #274 R2/R3 trainer exports and #272 archived runs if they have an exact SID
file). The #277 predictions and the 124-row prospective file are never overwritten. Replacement of
#277 requires the historical CI gate **and** a future separately-authorized independent-label
validation; otherwise the deliverable is an explicitly labeled research candidate with frozen
exploratory predictions.
