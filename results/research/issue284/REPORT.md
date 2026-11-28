# Issue #284 — REPORT

Analysis-only, INFRA_MODE=FROZEN, no training, no Stage3. Reference commit #283 =
`801048ca8153fd765e38b045a4c0578cb6343a0c`; all nine #283 artifacts hash-match their
`STATS_MANIFEST` (audit in `AUDIT_284.json`).

## C1 — data, identity and #283 reconciliation

- 253 SIDs / 28,391,948 triplet rows / 36 role classes; per-SID recomputation of `all1_chain`,
  L2/L3 cumulative ALL, the three marginals and AM/MB/AB/NONE from the code files against the #283
  `JOIN_TABLE`: **253/253 exact, 0 mismatches** (`CHAIN_TABLE.json`).
- 129 labels: 120 strong / 9 weak; 13 high (>0.06) with 6 strong; folds 27/27/26/25/24 (seed 42, same
  rule); 124 prospective SIDs, none labelled at the #276 freeze.
- #283 reproduction: baseline **0.002971939 / 0.66661**; fixed `all1_chain` **0.002699006 / 0.766546**
  (exact); the #283 inner-selected arm **0.002821808 / 0.715612** with picks 0/2/4 `all1_chain`,
  1/3 `am_mb_chain` (exact). The two are not conflated: the fixed arm is the post-hoc candidate, the
  selected arm is the #283 selection outcome.
- **New time-postordered labels (exactly bound SID + run identity)** from the R2 ledger:
  - R2 114 `TIGER_pairrank_enc_rq`, SID `9f6bf673…` (Stage2 run `stage2_5809…`), test recall@10
    **0.0479465** (test_final sha recorded);
  - R2 115 `TIGER_codebook_budget_j_rq`, SID `61f898…` (Stage2 run `stage2_0a2be3…`), recall@10
    **0.0484688**.
  Both SIDs sit in the 253-SID table with empty labels; they are in the 124 prospective set and are
  therefore *public now, not blind*. Their chain statistics are complete (same-mouth reuse).
- **Export-version separation (never merged)**: #274 R2 trainer `b9c3e8de…` vs checkpoint export
  `36032024…` (75 items differ at the repair, `same_l1=1.0`, `same_l3=1.0`); R3 trainer `67b5a47c…`
  vs checkpoint `9d9f3d68…` (85 items differ). The R3 checkpoint export is the labelled one
  (0.0505928); the R2 checkpoint export is unlabelled; the trainer exports are unconnected to any
  Stage3 run. #272's six recent runs were audited the same way (e.g. `active_j` SID `c8d5a088…`
  unlabelled; the codebook SID `61f898…` is the new label above; trainer exports listed separately).
- No missing metric was invented: `#274 R2` remains label-less (the earlier #283 `NOT_AVAILABLE` is
  explained by the export-version distinction, not by a missing Stage3 run of the checkpoint export).

## C2 — what the chain joint sharing provides

### Collinearity and decomposition (129 labelled SIDs)

| quantity | Spearman with recall | 95 % CI |
|---|---|---|
| raw `all1_chain` | 0.395 | [0.206, 0.556] |
| marginal-conditional `ALL_norm` | 0.645 | [0.484, 0.778] |
| `cond_refine` = P(L2 non-ALL \| L1 ALL) | 0.001 | [−0.202, 0.205] |
| `p3_split` = P(L3 all-different \| L1 ALL) | −0.028 | [−0.224, 0.168] |

Raw ALL is nearly the same variable as the existing L1 hit rate: ρ(`all1_chain`, `l1_pos_hit`) =
**0.984**; with `l1_neg_hit` 0.852, `l2_pos_hit` 0.899, `l3_collision` 0.763, `l1_entropy` −0.867.
`ALL_norm` reduces the collinearity (0.742 with `l1_pos_hit`) and is the stronger univariate ranker,
but see C3 for its linear-model behaviour. Residual association of raw ALL with the #277 OOF residual
is 0.082 [−0.110, 0.266] — not significant.

### Structure specificity

`all1` vs recall: chain 0.395 [0.206, 0.556]; fork 0.388 [0.197, 0.550]; merge 0.401 [0.209, 0.562];
revchain 0.399 [0.209, 0.560]; chain_both 0.438 (28k-row class). Chain is **not special** — all role
structures' ALL co-vary with recall at the same level, consistent with a shared L1-coarseness factor.

### Matched contrasts (frozen tolerances)

Primary set = same chain marginals (|Δ|≤0.005 each) + six metrics + L2/collision/L3 + L1
occupancy/entropy: **100 pairs / 48 SIDs**. Secondary set = the same controls without the marginal
equality: **113 pairs / 59 SIDs**.

| contrast | primary ρ [CI] | secondary ρ [CI] | LOO range (secondary) |
|---|---|---|---|
| `all1_chain` | 0.122 [−0.239, 0.451] | 0.192 [−0.119, 0.476] | 0.123 … 0.295 |
| `ALL_norm` | 0.149 [−0.186, 0.462] | 0.192 [−0.117, 0.462] | 0.118 … 0.251 |
| `cond_refine` | −0.284 [−0.558, 0.140] | −0.213 [−0.487, 0.221] | −0.254 … −0.090 |

Under the stricter primary controls the #283 marginal-matched association (0.191 [0.045, 0.325])
**does not survive**: every CI covers 0. Leave-one-out over the 48–59 covered SIDs stays in a narrow
positive band (no single-SID dominance); strong-only subsets are 0.067/0.091 (primary) and
0.169/0.163 (secondary). Degree/popularity are SID-invariant and cannot confound these contrasts.

### Conditional bands (frozen feature-only terciles, 43 SIDs each)

| band variable | low band ρ [CI] | middle | high band ρ [CI] |
|---|---|---|---|
| `l1_neg_hit` | **0.907 [0.779, 0.963]** | 0.474 [0.160, 0.701] | 0.039 [−0.296, 0.400] |
| `l2_pos_hit` | **0.807 [0.604, 0.918]** | 0.295 [−0.016, 0.539] | −0.033 [−0.373, 0.311] |
| `l3_refine_bits` | 0.686 [0.363, 0.929] | 0.388 [0.073, 0.663] | 0.332 [−0.024, 0.646] |
| `l3_collision` | **0.846 [0.665, 0.941]** | 0.336 [0.052, 0.580] | 0.151 [−0.199, 0.502] |

The ALL–recall association is concentrated in the **low-collision / low-L1-negative-sharing** regime
and vanishes in the high bands; mean ALL rises from 0.003–0.006 to 0.115 in the high bands (the
signal is saturated there). This is a conditional rule with support, not a universal one.

## C3 — conditional refinement and the predictive comparison

- `cond_refine` = 0.960 (low-recall half) vs 0.983 (high-recall half); correlation with recall
  0.001 [−0.202, 0.205]; `p3_split` −0.028; matched pairs show a *negative* (n.s.) association.
  The **over-merging hypothesis is not supported** at the SID level: nearly all L1-ALL triples are
  split at L2, and the amount of splitting does not predict recall.
- Three-item three-way distinction vs catalogue collision: `all3_chain` (triple L3 ALL) correlates
  0.883 with the catalogue `l3_collision`; the chain partition is not the catalogue collision, but the
  two co-move. The conditional metrics above use the exact transition denominators (no NA substitution).

### Predictive candidates (frozen protocol, same 129 rows / 5 folds / inner alpha)

| model | OOF MAE | ρ | reversals | strict MAE (120) | strict ρ | paired CI of MAE(A)−model | gate |
|---|---|---|---|---|---|---|---|
| A (#277) | 0.0029719 | 0.6666 | 0.2485 | 0.003096 | 0.6133 | — | — |
| B: A+`all1_chain` | **0.0026990** | 0.7665 | 0.2029 | 0.002778 | 0.7322 | [−7.65e−5, +6.41e−4] | fail |
| C: A+`ALL_norm` | 0.0031463 | 0.6178 | 0.2629 | 0.003277 | 0.5542 | [−3.57e−4, −1.40e−5] | worse (CI<0) |
| D: A+`all1_chain`+`cond_refine` | 0.0029465 | 0.7712 | 0.2014 | 0.003047 | 0.7387 | [−4.13e−4, +4.22e−4] | fail |
| selected (B/C/D, inner folds) | 0.0030260 | 0.7329 | 0.2154 | 0.003047* | 0.7387* | [−4.89e−4, +3.10e−4] | fail |

\* strict numbers for the D and selected arms coincide in the rounded report. `>0.06` confusion is
13/0/0 for every arm; 0 rows > 0.065 → **0.065 NOT_EVALUABLE**. Family tests A−B, A−C, A−D: only C is
significantly negative; B/D cover 0.

**Collinearity sign flip**: in the full-cohort fit of B the coefficient of `all1_chain` is
**−0.1125** while `l1_pos_hit` is +0.0662 — the fold-wise OOF gain comes from unstable partial fits of
a feature that is 0.984-correlated with an existing predictor. This is the mechanistic explanation of
the fragile #283 point estimate: the "new" information is the same L1-coarseness axis, and the split
between the two coefficients is not identified.

**Power** (frozen formula, per-SID absolute-error differences): A−B mean 2.73e−4, SD 2.12e−3 →
**374 independent SIDs for 80 % power** (SID-cluster variant 353); minimum detectable mean at 129 =
4.64e−4 > the observed 2.73e−4. A−C mean is negative; A−D mean 2.5e−5 needs ≈57k SIDs. The 129-SID
cohort cannot resolve the observed B effect.

**Time-postordered sensitivity (2 new labels, not blind)**: #277 predicts 0.05319 / 0.05083; B predicts
0.05295 / 0.05005; actual 0.04795 / 0.04847. Both over-predict (B's absolute errors 0.0050 / 0.0016
vs #277's 0.0052 / 0.0024). n=2 — anecdotal, not used to modify any formula.

## C4 — frozen research candidate and prospective predictions

Because the historical gate fails for every candidate, **#277 is not replaced**. A clearly labelled
research candidate B is frozen (`PROSPECTIVE_284.json`): alpha 0.01, raw-unit coefficients
[0.06625, 0.05878, 0.001078, −8.36e−7, −0.01872, −0.001125, **−0.11254**], training domain (129 SIDs,
120/9, 5 folds, chain support 22k rows/SID), input hashes (`CHAIN_TABLE`, `SID_TABLE`, `FOLDS`,
`JOIN_TABLE`). It carries: predictions for the **122 label-pending SIDs** (124 prospective minus the
two now-labelled), the two known-label sensitivities, and **9 recovered unlabelled SIDs** (R2/R3
trainer exports and #272 run exports) with `all1_chain` and the B prediction where the six features
are unavailable for those exports (explicitly marked `six_features_not_available`). The #277
predictor and the 124-row file are unchanged. Replacement requires the historical CI gate **and** a
future separately-authorized independent-label validation.

## Answers

1. **Does chain ALL have independent value after fixed marginals and the six metrics?** No at the
   pre-registered standard: the primary matched CI covers 0, the raw variable is 0.984-collinear with
   `l1_pos_hit`, and the full-cohort coefficient is negative. The descriptive signal exists only in
   the low-L1-collision regime (band ρ up to 0.91).
2. **Does it depend on subsequent L2/L3 refinement?** No: `cond_refine` has no recall association
   (0.001 [−0.202, 0.205]) and the matched sign is negative; the over-merging hypothesis is not
   supported. `all3_chain` is 0.883-correlated with the catalogue collision — the deep prefix behaves
   like a collision measure, not a refinement benefit.
3. **Which conditional rules are supported?** The band rule (ALL ranks recall only when L1 negative
   sharing is low / L2 discrimination is weak-to-moderate / collisions are low); the structure rule
   is absent (chain ≈ fork/merge/revchain); the over-merging rule is absent.
4. **Predictive increment with generalization evidence?** Not established; B's point estimate is
   −9.1 % MAE but needs ≈374 independent SIDs (3× the current 129) for 80 % power. The correct
   boundary is "underpowered at 129 labels", not "impossible" — the tested set is not the allowed
   space, and no infeasibility claim is made.
