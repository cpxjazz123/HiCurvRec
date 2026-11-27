# Issue #272 — round 3 diagnosis

## Observed failures (not yet root causes)

| round | P1 J | P2 MAE / Spearman / sign | P3 L2 / rank | P4 recon | P5 mu |
|---|---|---|---|---|---|
| round 1 `TIGER_active_j_rq` | 0.789 | 2.343824 / −0.098838 / 0.348062 | 6,848 / 2.77 | 8.57e-5 | active |
| round 2 `TIGER_balanced_j_rq` | diverged 1.598 | n/a | collision 0.983 | 2.01e-4 | active |
| TIGER (lambda=0) | 0.739635 | 7.309217 / 0.015851 / 0.432453 | 7,374 / 11.5 | 6.15e-4 | n/a |
| #271 `TIGER_global_j_rq` | 0.351361 | 1.893506 / −0.005483 / 0.392646 | 6,097 / 4.501 | n/a | inactive |
| #269 `TIGER_calibrated_kappa_rq` | 0.338336 | 1.997261 / −0.070083 / 0.363704 | 6,246 / n/a | n/a | active |
| #265 Euclidean local | 0.189476 | 2.245950 / −0.071041 / 0.332350 | 10,918 / 3.05 | 8.18e-5 | n/a |

## Candidate root causes and competing explanations

- **R1 (primary): codebook-side gradient.** Both the local curvature term and the global J term
  update the encoder and the codebook. The codebook-side update compresses code usage and
  perturbs the assignment ordering; the encoder-side fit is the useful part. Round 2's 10%
  local codebook share collapsed the assignment, while round 1's ~0.05% did not; the 0.5–2%
  band was never tested.
- **R2 (null): encoder-side local fit.** The local term on the encoder itself degrades the
  ordering; the codebook budget cannot help.
- **R3: scale shrink.** The local MAE gain is a shrink; the P2 non-degeneracy excludes it
  (#271 var_ratio 0.335, slope 0.041). The local ordering is below the TIGER in every local arm.
- **R4: base incompatibility.** The P3 L2>=10,918 belongs to the Euclidean-local #265; the frozen
  kappa_t local reduces L2 in every tested point (6,097–6,848). This is evidence over X_tested,
  not an X_allowed proof; it is the competing explanation the round must separate.

## Evidence (measured, paths)

- `results/research/issue271/assets/geometry/*.json`: the local MAE/Spearman/sign above, 28,391,948
  rows, frozen evaluator.
- `results/stage2_RQ-VAE/TIGER_global_j_rq/.../stage2_metrics/*.json`: L2 6,097, rank 4.501.
- `results/stage2_RQ-VAE/TIGER_RQ-VAE/...`: L2 7,374, rank 11.5.
- Round 1 `stage2_metrics`/`VERDICT.md`: L2 6,848, rank 2.77, recon 8.57e-5, J 0.789.
- Round 2 `training_metrics.jsonl`: collision 0.983, J 1.598 at epoch 556.
- The local-weight trend (round 1 → #269 → #271): MAE 2.34 → 1.997 → 1.894, Spearman
  −0.0988 → −0.0701 → −0.0055, sign 0.348 → 0.364 → 0.393; stronger local improves the local
  metrics but still stays below the TIGER's Spearman/sign.

## Low-cost verification before the full run

- The helper tests: known answers, the beta=0/1, the inactive constraint, the missing grad, the
  independent composition, the six mutations (including the no-budget red baseline).
- The first-step `codebook_budget` record in the smoke confirms `beta=0.01` and the codebook
  parameters are touched.
- The smoke's first-epoch local/SID readout distinguishes the codebook-budget effect from R2/R4
  before the 3000-epoch budget is spent.

## Remaining uncertainty

The P2 non-degeneracy (R3) is not addressed by the codebook budget; the round measures it. If the
P3 rank/L2 improve but the P2 ordering stays below the TIGER, the next round targets the
encoder-side local loss (variance-preserving/ranking) or the base (R4). If the P3 stays at the
round-1 level, R1 is excluded.

## Six questions (formal-training admission)

1. **Specific problem**: whether the codebook-side local/global gradient is the channel that
   compresses L2/rank and degrades the SID ordering.
2. **Evidence it caused the failure**: round 2's 10% codebook share collapsed the assignment;
   round 1's ~0.05% did not; all tested arms applied the terms fully to the codebook; the
   0.5–2% band is untested.
3. **Why the new mechanism acts on it**: it scales exactly that channel by beta=0.01 while the
   encoder keeps the full fit; the reconstruction keeps the codebook's code usage.
4. **Substantive difference**: previous rounds changed lambda/bound/dual; this round changes the
   codebook-side gradient routing, which no prior arm did.
5. **Metrics**: P3 rank >= 8 and L2 >= 10,918; P2; P1; P4; P5; the independent evaluator.
6. **Knowledge if it fails**: the encoder/codebook attribution of the L2/rank/ordering loss; the
   next round targets the encoder-side loss (R3) or the base (R4) with the measured direction.
