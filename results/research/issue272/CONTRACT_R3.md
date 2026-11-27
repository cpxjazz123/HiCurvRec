# Issue #272 — round 3 contract: codebook-side gradient budget

Frozen before the round-3 3000-epoch run. Issue #272 body sha256
`8814eae7356ac36ba26c0812b51a1d4327b499171327eaa31be3b70bbeca84f7`. The acceptance
thresholds are the ones frozen in `CONTRACT.md` (round 1); this contract only adds the round-3
mechanism and its falsifiers.

## 1. Observed failures

- Round 1 (`TIGER_active_j_rq`): P1 J 0.789 (fail), P2 MAE 2.343824 / Spearman −0.098838 /
  sign 0.348062 (fail), P3 L2 6,848 / rank 2.77 (fail), P4 recon 8.57e-5 (pass), P5 mu active.
- Round 2 (`TIGER_balanced_j_rq`): the 10%-of-global local codebook gradient collapsed the
  assignment at epoch 556 (collision 0.9832, J 1.598, recon 2.01e-4); stopped as a negative
  diagnostic.
- Prior arms: TIGER (lambda=0) L2 7,374 / rank 11.5 / Spearman 0.015851 / sign 0.432453;
  #271 `TIGER_global_j_rq` MAE 1.893506 / Spearman −0.005483 / sign 0.392646 / L2 6,097 /
  rank 4.501; #269 `TIGER_calibrated_kappa_rq` MAE 1.997261 / Spearman −0.070083 / sign
  0.363704 / L2 6,246; #265 Euclidean local L2 10,918 / rank 3.05.

## 2. Root cause under test (R1) and competing explanations

**R1 (primary).** The local curvature term and the global J term both flow through the codebook
(the assignment). Their codebook-side gradient compresses code usage (L2/rank) and perturbs the
SID ordering; the encoder-side fit is the part the mechanism needs. All tested arms applied both
terms fully to the codebook. Round 2 already showed that a large local codebook share is
destructive; the 0.5–2% codebook share was never tested.

**R2 (null).** The degradation comes from the encoder-side local fit itself; protecting the
codebook changes nothing.

**R3 (shrink).** The local MAE gain is a scale shrink; the P2 non-degeneracy (var_ratio >= 0.5,
slope >= 0.1) excludes it; the ordering target is unreachable through the local term.

**R4 (base).** The P3 L2>=10,918 target belongs to the Euclidean-local #265, not the frozen
kappa_t base; no legal lambda/dual/schedule setting can reach it in this base.

R1 is distinguishable from R2 by the codebook budget: if R1 holds, the rank/L2 and the ordering
should improve while the encoder keeps the local fit; if R2/R4 hold, they will not. R3 is
measured by the independent local evaluator's var_ratio/slope.

## 3. Mechanism (single primary change)

`L = L_RQVAE + lambda L_curv + mu (J - J_max)`; architecture, selection, Sinkhorn, floor,
reconstruction, kappa_t and S* unchanged; lambda and mu are the round-1 values; the only change
is the **codebook-side gradient budget**:

```
g_codebook = g_codebook(L_RQVAE) + beta * (lambda * g_codebook(L_curv)
                                          + mu * g_codebook(J - J_max))
g_encoder  = full gradient of L
beta       = 0.01
```

Implementation: after `loss.backward(retain_graph=True)`, `scale_codebook_terms` subtracts
`(1-beta)` times the local+constraint codebook gradient from the codebook embeddings' `.grad`.
The codebook remains updated by the reconstruction/commitment term; the local+global terms still
reach the codebook at `beta > 0` (the issue's gradient-reach requirement is preserved).

## 4. Data, shapes, boundaries, tolerances

- Codebook parameters: `raw_module.rq.vq_layers[l].embed.weight`, shape (256, 32), float32, l=0,1,2.
- Encoder/decoder: unchanged, full gradients.
- `beta` in [0, 1]; invalid values raise; an empty parameter list raises.
- Inactive constraint (`J <= J_max`): the zero tensor; the helper reduces to the local term only.
- `allow_unused=True`: a parameter with no connection or no `.grad` is skipped.
- The first-step record `codebook_budget` stores `beta`, `parameters`, `touched`.
- Tolerance: exact arithmetic on float32; the tests use 1e-7.

## 5. Predictions and falsifiers

| id | prediction | falsifier |
|---|---|---|
| P3 rank | rank >= 8 | rank < 8 |
| P3 L2 | L2 >= 10,918 | L2 < 10,918 |
| P2 | MAE <= 1.893506, Spearman > 0.015851, sign > 0.432453, var_ratio >= 0.5, slope >= 0.1 | any fails |
| P1 | full common-ruler J <= 0.338336, 1-hop/2-hop within 1.05x #269 | any fails |
| P4 | same-criterion recon <= 8.589e-5 | above |
| P5 | mu > 0 material fraction and the constraint violated at some point | mu = 0 |

Stage3 is forbidden by the issue; no recall is produced.

## 6. Evaluation and acceptance

The common-ruler J and the local A.1 use the frozen evaluators and the #270 scale; the SID metrics
come from the exported `item_sids.json`; the reconstruction and mu from the run. The close
condition is the PASS table of P1–P5 (with the non-degeneracy), exactly as in `CONTRACT.md`.
