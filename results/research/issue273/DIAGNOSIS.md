# Issue #273 — diagnosis (round 1)

## Observed phenomenon (facts, not root causes)

| arm | local MAE | Spearman | sign | L2 | rank | full-ruler J | recon |
|---|---|---|---|---|---|---|---|
| TIGER (frozen baseline) | 7.309217 | 0.015851 | 0.432453 | 7,374 | 11.5 | 0.739635 | 6.15e-4 |
| #269 `TIGER_calibrated_kappa_rq` | 1.997261 | −0.070083 | 0.363704 | 6,246 | — | 0.338336 | 8.18e-5 (same criterion) |
| #271 `TIGER_global_j_rq` | 1.893506 | −0.005483 | 0.392646 | 6,097 | 4.501 | 0.351361 | — |
| #265 `TIGER_euclid_local_curve_supervision_rq` | 2.245950 | −0.071041 | 0.332350 | 10,918 | 3.05 | 0.189476 | — |
| #272 round 1 `TIGER_active_j_rq` | 2.343824 | −0.098838 | 0.348062 | 6,848 | 2.77 | 0.789 (training estimate) | 8.57e-5 (training) |
| #272 round 3 (codebook budget, running) | — | — | — | — | — | 0.221 (training estimate) | 8.73e-5 (epoch 1502) |

Every arm that improves the pointwise MAE **loses** Spearman, sign and effective rank relative to
TIGER. The issue asks specifically about the **signed difference magnitude** between triplets that
share a centre.

## #272 criterion inconsistency (verified, per issue §1)

- **J**: the #272 round-1 run reported the **training estimator** J (0.789) while the acceptance
  threshold was stated on the **full common ruler**. The same run's full common-ruler value on the
  1-epoch smoke was **0.64866** (`results/stage2_RQ-VAE/_conformance/TIGER_active_j_rq/geometry_final.json`,
  299,402,685 pairs) — the two criteria differ by ≈0.14, i.e. the estimator offset is of the same
  magnitude as the acceptance margin. #273 therefore computes C9–C11 on the **full ruler at full
  precision** and never mixes the two.
- **reconstruction**: the #272 verdict quoted the **training** reconstruction (8.57e-5) while the
  acceptance baseline was the **#265 same-criterion** error (8.18e-5, ×1.05). #273 recomputes the
  #265 same-criterion error at full precision and compares like with like.

## Candidate root causes and competing explanations

- **R1 (primary) — the pointwise Huber loss is satisfied by a uniform shrink.** A per-triplet
  absolute-value regression can be minimised by pulling every prediction towards the target's mean,
  which flattens the **differences** between triplets of the same centre. The measured signature
  matches: MAE falls (7.31 → 1.89–2.34) while Spearman/sign fall below TIGER and the predicted
  variance collapses (#271 `var_ratio` 0.335, `slope` 0.041 per the #272 contract).
- **R2 — the pointwise term cannot see the differences at all.** For triplets t, s of one centre the
  pointwise loss constrains `p_t` and `p_s` separately; a common offset `c` applied to every
  prediction changes the pointwise loss but leaves every difference `(p_t − p_s)` unchanged. The
  statistic that *is* sensitive to the differences while blind to the offset is the same-centre
  difference `(p_t − p_s) − (y_t − y_s)`; it is exactly `e_t − e_s` with `e = p − y`.
- **R3 — the MAE gains are shrink artefacts.** The #271 MAE (the source of the C1 threshold) came
  with `var_ratio` 0.335 / `slope` 0.041, i.e. a degenerate constant-ish fit; C4/C5 exclude it.
- **R4 — the SID capacity loss is assignment-driven, not geometry-driven.** #265 (Euclidean local)
  reaches L2 10,918 with rank 3.05, while TIGER (no local term) has L2 7,374 with rank 11.5; the
  local objective changes the assignment's diversity in a family-dependent way. A delta term on the
  same κ_t geometry may therefore not move L2/rank at all.
- **R5 — the #272 global constraint is a separate failure.** #272's tightened global term harmed the
  1-hop relation and its 293x local weight collapsed the assignment; #273 does **not** use a global
  training term (the parent is the #269 fixed local supervision), so C9–C11 are acceptance-only.

## Competing explanations and the discriminating measurement

| explanation | prediction if true | falsifier |
|---|---|---|
| R1/R2 (differences are compressed) | the within-centre difference RMSE (C6) and the within-centre ordering (C7) are worse than #269; adding `L_delta` lowers C6 and raises C7 while C4/C5 rise | C6/C7 unchanged or C4/C5 stay below the threshold |
| R3 (MAE is a shrink artefact) | the frozen baselines already show `var_ratio`/slope below the thresholds; the candidate's C4/C5 do not recover | the candidate's C4/C5 pass |
| R4 (capacity is assignment-driven) | the candidate's L2/rank track the #269 values regardless of the delta term | the candidate's L2/rank move with the delta weight |

## Low-cost verification before the formal run

1. Measure the **frozen baselines** (TIGER, #269, #271) on the new statistics (C6, C7, C4, C5) with
   the issue-273 analysis script; it must reproduce the frozen evaluator's MAE/Spearman/sign exactly
   (self-check) before the new statistics are used.
2. Unit tests of the delta loss on hand-computed triplets (see `CONTRACT.md` §4).
3. A **gradient-budget probe** on representative real batches at two training stages: the
   unweighted/weighted gradient norms of `λ_point L_point` and `λ_delta L_delta`, their angle, the
   actual parameter update and the post-step objective change. The budget is set from these probes,
   never from a single-batch loss value.
4. A 1-epoch real-entry smoke (the framework's existing admission) before the 3000-epoch run.

## Six admission questions (before the formal Stage2 run)

1. **Problem**: the pointwise curvature supervision compresses the within-centre curvature
   differences, degrading ordering/sign/variance/rank while the MAE improves; and #272's global
   constraint is out of scope for this round.
2. **Evidence**: every κ_t arm improves MAE and degrades Spearman/sign/rank (table above); the #271
   MAE came with a shrink (0.335/0.041); the pointwise loss is provably blind to the common offset
   (R2).
3. **Why the mechanism acts on it**: `L_delta` is exactly the within-centre error variance
   (2×Var_c(e) under independent same-centre pairing); it is blind to the common offset and
   sensitive to the differences, so it pushes the differences to match while leaving the offset to
   the pointwise term.
4. **Difference from the previous rounds**: #269/#271 changed the pointwise weight; #272 added a
   global constraint and a 293x weight jump (collapse); #273 adds a **new difference term** with a
   low incremental gradient budget and progressive enablement, and **no** global term and **no**
   weight jump.
5. **Metrics**: C1–C15 above, computed at full precision on the frozen ruler and the exported hard
   SID; the P-table in `CONTRACT.md`.
6. **Knowledge if it fails**: whether the within-centre differences are recoverable at all under the
   κ_t geometry (C6/C7 move or not), whether the MAE gains were shrink artefacts (C4/C5), and
   whether the SID capacity is assignment-driven (C12/C13 unchanged) — each of these selects the
   next round's single change.
