# Issue #283 — REPORT

Analysis-only, INFRA_MODE=FROZEN, no training, no Stage3. Population 1 = the full 28,391,948-row
triplet asset (all `valid`) on the visible directed graph `ac7155…`; population 2 = the 57,439
validation rows (53,661 unseen) with training-graph intermediates. All derived tables carry sha256 in
`STATS_MANIFEST.json`; commands/logs/exit codes were produced through
`tools/iteration/execution.py` (logs under `results/research/issue283/logs/`).

## C1 — definitions and invariants (CONTRACT §2/§3)

Ordered roles are positional: `a = row.b`, `m = row.centre`, `b = row.c`. The 6-bit directed code is
recomputed from `assets/train/visible_forward.npz`; `canonical_code(code)` equals the stored
`structure_class` on **every** row (0 mismatches). 36 role classes cover every feasible code; the 28
infeasible codes (a or b not adjacent to m) never occur.

Seven deterministic tests (`tests/test_joint283.py`), all passing:

| check | result |
|---|---|
| equal-marginal hand example D1={ALL,ALL,NONE,NONE} vs D2={ALL,AM,MB,AB} | p_am=p_mb=p_ab=0.5 both; ALL 0.5 vs 0.25 |
| partition counts sum to support (253 SIDs × 36 classes × 3 levels) | asserted, no violation |
| cross-layer refinement only (12 allowed, 13 forbidden transitions) | asserted per SID/class/level, no forbidden cell |
| codeword global-rename invariance (independent permutation per level) | exact |
| LCP feasibility (22 feasible triples of 64) | asserted 0 infeasible |
| type classification hand cases (chain/revchain/fork/merge/ff/fb/recip) | exact |
| uniform null hand case N=6, groups [3,3] | pair 0.4, triple 0.1 exact |

Algebra (CONTRACT §3): with the pairwise marginals fixed, AM = p_am−ALL, MB = p_mb−ALL,
AB = p_ab−ALL, NONE = 1−p_am−p_mb−p_ab+2·ALL — the joint 5-partition has exactly **one** free
parameter beyond the marginals, ALL = P(a=m=b). AM/MB/AB are affine functions of ALL and are not
independent information. Verified exactly on the pooled counts (e.g. chain k=1: ALL 0.06296,
AM 0.05693 = (0.06296+0.05693)−0.06296).

## C2 — full graph, coverage and null models

- 13 shards, **28,391,948 rows**, all valid, matching `assets/index.npz`; per-shard sha256 in
  `ROWS_MANIFEST.json` (class counts, orbit counts, definition sha256 recorded).
- Role-class totals: chain 6,347,269; revchain 5,795,800; chain_both 28,114; revchain_both 27,266;
  fork 6,063,454 (+variants 173,718); merge 5,940,844 (+variants 170,136); triangle family 296,841;
  reciprocal family 3,548,506. Orbit roles per row from `graphlet.orbit_of_position`; no class
  filtered.
- Support, degree and popularity are SID-invariant (all SIDs assign a code to all 24,587 items), so
  they cannot confound SID-level contrasts; stated instead of "controlled".
- #282 negative-sample audit: level 2 (type∩degree) 1.35 %, level 1 11.2 %, level 0 87.4 %
  (`issue282/TYPE_AUDIT.json`). #283 uses **no item-level negative sampling**: exact full-population
  counts plus analytic permutation nulls.
- Null models: uniform permutation E[pair]=Σ n_g(n_g−1)/(N(N−1)), E[ALL]=Σ n_g(n_g−1)(n_g−2)/
  (N(N−1)(N−2)), N=24,587, per-level group sizes (codebooks up to 1024 handled); degree-stratified:
  deciles of the undirected degree, within-stratum permutation with per-(class, stratum-triple) row
  counts. Empirical check (`PERM_CHECK.json`, 3 SIDs × 10 permutations × 3 levels): 320 non-degenerate
  tests, **316 within 3σ**, median |z| 0.74, 4 degenerate (SE=0); the single 7.3σ case is the small
  `recip_both_both` class (15,024 rows) where the 10-draw SE is itself noisy — reported as a caveat,
  the analytic expectation is used.
- Population-2 coverage (validation rows with ≥1 intermediate): chain 19,516 rows / 140,945
  instances; revchain 19,993 / 141,543; merge 19,113 / 134,963; fork 19,849 / 143,612; pair-uniform
  1/|M| weighting frozen, inverse-degree weighting as sensitivity.

## C3 — joint structural fidelity and recommendation differences

### Pooled per-class/level patterns (253 SIDs; `TABLES.md`, `JOIN_SUMMARY.json`)

| class | level | ALL | AM | MB | AB | NONE | ALL lift(u) | ALL lift(d) |
|---|---|---|---|---|---|---|---|---|
| chain | k=1 | 0.06296 | 0.05693 | 0.05711 | 0.02470 | 0.79830 | +0.01808 | +0.01538 |
| chain | k=2 | 0.01783 | 0.01053 | 0.01129 | 0.00464 | 0.95570 | −0.00109 | −0.00282 |
| chain | k=3 | 0.00618 | 0.00408 | 0.00454 | 0.00243 | 0.98278 | −0.00625 | −0.00713 |
| revchain | k=1 | 0.06329 | 0.05398 | 0.05973 | 0.02491 | 0.79809 | +0.01841 | +0.01567 |
| fork | k=1 | 0.06229 | 0.05404 | 0.05638 | 0.02435 | 0.80295 | +0.01741 | +0.01473 |
| merge | k=1 | 0.06353 | 0.05619 | 0.05989 | 0.02476 | 0.79562 | +0.01865 | +0.01587 |
| ff | k=1 | 0.08422 | 0.05768 | 0.05953 | 0.05912 | 0.73943 | +0.03934 | +0.03460 |
| fb | k=1 | 0.08609 | 0.05944 | 0.06150 | 0.05979 | 0.73318 | +0.04121 | +0.03654 |
| recip_both | k=1 | 0.08520 | 0.08010 | 0.08370 | 0.03317 | 0.71783 | +0.04032 | +0.03541 |

- At k=1 the triple intersection is **1.4–2.0× the uniform null** for chain/wedge and **≈1.9×** for
  triangles; at k=2/k=3 the observed ALL falls **below** the uniform null (chain k=2 −0.0011,
  k=3 −0.0063): the SID preserves joint structure at the coarse prefix but depletes exact deeper
  coincidences.
- LCP (chain, pooled): `(0,0,0)` 1.28e9, `(0,0,1)` 3.65e7, `(0,0,2)` 2.79e6, `(0,0,3)` 4.1e5,
  `(0,1,0)` 8.1e7 — deeper joint depths are rare but populated; transitions are exactly the 12
  allowed cells (ALL→{ALL,AM,MB,AB,NONE}, AM→{AM,NONE}, MB→{MB,NONE}, AB→{AB,NONE}, NONE→NONE).
- Centre-macro vs row-micro differ (e.g. chain k=1 macro ALL 0.0103 vs pooled 0.0630 in the 2-SID
  smoke; across 253 SIDs the same direction): small-centre triples share more, so row-weighted and
  centre-weighted statistics are reported separately (`JOIN_TABLE`).
- Marginally matched chain pairs (509; frozen tolerance 0.005 on each pairwise rate): ALL vs recall
  Spearman **+0.191, 95 % CI [0.045, 0.325]**, sign agreement 0.56. With near-equal marginals, a
  larger triple intersection goes with slightly higher recall — a weak but non-zero descriptive
  association.
- Matched SIDs (146 pairs passing the frozen L1/L2/collision/L3 tolerances):

| contrast | Spearman | 95 % CI (SID-cluster bootstrap) |
|---|---|---|
| all1_chain_lift | 0.085 | [−0.181, 0.334] |
| all3_chain_lift | 0.357 | **[0.043, 0.598]** |
| am_mb_chain | −0.058 | [−0.370, 0.295] |
| amb_ab_merge | 0.170 | [−0.114, 0.419] |
| amb_ab_fork | 0.164 | [−0.126, 0.417] |
| am_mb_ab_chain | 0.206 | [−0.077, 0.455] |
| all_ab_triangle | −0.066 | [−0.311, 0.224] |
| all_ab_recip | 0.072 | [−0.154, 0.281] |
| all1_merge_lift | 0.061 | [−0.203, 0.323] |

- #277 residual associations (OOF residuals of the 6-feature model): all_ab_triangle −0.306
  (BH q=0.0039), all3_chain_lift +0.242 (q=0.026), all_ab_recip −0.194 (q=0.083), amb_ab_fork +0.178
  (q=0.095), all1_chain_lift +0.099 (q=0.30), am_mb_chain −0.011 (q=0.90). These are descriptive:
  the features are associated with the residual structure, not an increment (C4).
- High-recall vs low-recall (median split 0.05416): all1_chain_lift 0.0221 vs 0.0135 (Spearman all
  0.465); all3_chain_lift −2.5e−5 vs −1.1e−5 (0.510); amb_ab_merge 0.123 vs 0.081 (0.517);
  all_ab_triangle −0.020 vs +0.015 (−0.141).
- Counterexamples (largest |Δfeature| with opposite sign to Δrecall) are recorded per contrast in
  `MATCHED.json`; e.g. marginal-matched pair (9,30): ΔALL −0.0037 vs Δrecall +0.0004.
- #274 R2: `NOT_AVAILABLE` — the R2 SID (`b9c3e8de…`) is not in the #276 table and has no label or
  features (`AUDIT_274.json`); it cannot enter a matched diagnostic.

### Population-2 (validation context, independent of the training triplets)

- **Tautology check**: the same-level hit (a=b) is mechanically determined by the partition
  (ALL/AB ⇒ hit; AM/MB/NONE ⇒ no hit); the hit-difference in `VALIDATION_CONTEXT.json` (~0.95 at
  k=1) is this implication, not an effect. The nontrivial part is the composition
  (`VALIDATION_COMPOSITION.json`): among unseen hit pairs with intermediates, the intermediate shares
  the anchor/target L1 code (ALL) in **≈36–37 %** and sits in another code (AB) in **≈63–64 %**
  (chain/revchain/merge/fork; k=1).
- Coverage selection is weak: hit rate on covered vs uncovered unseen rows 0.0404 vs 0.0380 (chain),
  0.0409 vs 0.0377 (revchain), 0.0394 vs 0.0384 (merge), 0.0406 vs 0.0379 (fork).
- Redundancy: the validation ALL1 rate is **ρ=0.980 (chain) / 0.981 (merge)** with the #277
  `l1_pos_hit` feature across 253 SIDs, and ρ=0.49/0.48 with recall — the population-2 joint pattern
  re-measures the same information as `l1_pos_hit` and adds no independent signal (BH q=0).

## C4 — predictive acceptance (frozen protocol)

Baseline A = #277 corrected Ridge6 (same 129 rows, same folds, same nested inner-alpha procedure):
**MAE 0.0029719, ρ 0.6666** — reproduced exactly. 120 strong / 9 weak; 13 high (>0.06) with 6 strong;
**0 rows > 0.065 → 0.065 NOT_EVALUABLE**.

| model | OOF MAE | ρ | Kendall | reversals | strict MAE (120) | strict ρ |
|---|---|---|---|---|---|---|
| A (277) | 0.0029719 | 0.6666 | 0.5028 | 0.2485 | 0.003096 | 0.6133 |
| A + inner-selected joint feature | 0.0028218 | 0.7156 | 0.5494 | 0.2252 | 0.002914 | 0.6745 |
| A + all1_chain (fixed) | 0.0026990 | 0.7665 | — | — | — | — |
| A + all1_chain_lift (fixed) | 0.0032500 | 0.5559 | — | — | — | — |
| A + all3_chain_lift (fixed) | 0.0034395 | 0.5604 | — | — | — | — |
| A + am_mb_chain (fixed) | 0.0028235 | 0.6881 | — | — | — | — |
| A + all1_merge_lift (fixed) | 0.0032495 | 0.5566 | — | — | — | — |
| #282 F1 (comparison) | 0.0029815 | 0.6591 | — | — | — | — |
| #282 F2 (comparison) | 0.0029652 | 0.6631 | — | — | — | — |

Inner selection (only training folds): fold 0 all1_chain, fold 1 am_mb_chain, fold 2 all1_chain,
fold 3 am_mb_chain, fold 4 all1_chain. Paired SID-cluster bootstrap (127 units, 2,000 reps) of
MAE(A) − MAE(new): **+1.50e−4, 95 % CI [−1.57e−4, +4.57e−4]** — the CI contains 0, so the
**pre-registered primary criterion fails**; the fixed `all1_chain` arm also fails
([−1.06e−4, +6.35e−4]). All secondary criteria improve for the selected model (ρ 0.716 vs 0.667,
reversals 0.225 vs 0.248, fp/fn unchanged 13/0/0, strict MAE/ρ improve), but the frozen rule requires
the CI gate; **#277 is retained, no new formula/coefficients are frozen, and no predictions are
modified for unlabeled SIDs**. The joint lift features (`all1_chain_lift`, `all1_merge_lift`) are
significantly *worse* (CI excludes 0 on the negative side), i.e. subtracting the SID-specific null
removes the useful part.

## Answers

1. **Is the joint triplet assignment relevant under equal marginals?** Yes descriptively: ALL is the
   unique non-marginal degree of freedom, it is 1.4–2.0× its permutation null at k=1, and with chain
   marginals matched (509 pairs) ALL vs recall is +0.191 [0.045, 0.325]. But the effect is weak
   (sign agreement 0.56) and not redundant with the marginals.
2. **Which role/level patterns have predictive value?** The raw ALL at k=1 (chain/merge) is the only
   candidate with a positive point estimate (fixed `all1_chain`: MAE −9.1 %, ρ +0.10); the role
   contrasts (AM−MB, AMB−AB) and the triangle/reciprocal contrasts show no matched-pair signal; deeper
   levels (k=2/3) are depleted below the null and do not help.
3. **Increment beyond the six metrics?** Not established at the pre-registered gate: the paired
   bootstrap CI contains 0. The increment is not absent by redundancy (ALL is above the null) nor by
   support (6.3M rows) but is **not statistically resolvable at 129 SIDs**; the population-2 ALL is
   ρ=0.98 redundant with `l1_pos_hit`. No freeze; #277 stays.
