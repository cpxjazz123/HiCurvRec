# Issue #273 — contract (round 1): same-centre difference protection for sectional curvature

Frozen before the round-1 3000-epoch run. Issue body sha256
`fc85224d759f58eb7dab113bfa50b4018d0c8721b3d45bde37d05c543bc4ec44`. Acceptance thresholds are the
C-list in `RUN_LOG.md`; the diagnostics are `DIAGNOSIS.md`.

## 1. Scope

Parent = the **#269 fixed local supervision** (`TIGER_calibrated_kappa_rq`): frozen κ_t A.1 pointwise
Huber on the final hard q, the #269 calibrated `lambda_point = 5.883564881514758e-06`, no global
distance training term (the #272 global term is **not** inherited). Round 1 adds exactly one
mechanism: the same-centre difference protection `L_delta` with a low incremental gradient budget and
progressive enablement. Nothing else changes.

## 2. Objects (kept separate, per issue §1)

| symbol | meaning | gradient |
|---|---|---|
| `y_t` | the frozen graph A.1 curvature **target** of triplet t (from the frozen target table) | none (host data) |
| `kappa_t` | the frozen per-triple curvature **parameter** used by the stereographic distance | none (host lookup) |
| `p_t` | the **prediction** computed from the final hard q of triplet t under its own `kappa_t` and the frozen scale `w = q/(2 S*)` | yes (through the hard q / STE) |

`kappa_t` is never a target. All predictions and targets of a pair use the same graph target unit and
the same frozen mapping; **no trainable scale** is introduced (the pointwise term's frozen
`target_scale` is not used to rescale the differences either — the difference term lives in the raw
frozen units, and the scale only enters through the calibrated weight).

## 3. The difference protection

For a training step, the structural sampler draws `TRIPLES_PER_UPDATE = 32` triplets **from one
centre** (`StructuralSampler.draw`: a centre cycle, `row_ids` drawn from that centre's block,
`triples[:, 0] = centre`). Write `e_t = p_t − y_t`. The new term is

```
L_delta = mean over the centre's pairs (t, s), t != s of  ( (p_t - p_s) - (y_t - y_s) )^2
        = mean over the pairs of ( e_t - e_s )^2
```

with the **centre-balanced** normalisation used by the frozen evaluator
(`FrozenCurvature.loss`): the per-centre contribution is
`sum_pairs (Δe)^2 / (denominator_c * n_pairs_c)`, where `denominator_c` is the centre's valid
triplet count (`sample['triple_population']`); one centre is drawn per step, so averaging over the
steps weights every centre equally and no high-frequency centre dominates. The pairing enumerates
`32*31/2 = 496` pairs per step — no quadratic enumeration over the full triplet population.

**Variance equivalence (proof).** Let `t, s` be drawn independently (with replacement of the error
realisation) from the same centre. Then
`E[((p_t − p_s) − (y_t − y_s))^2] = E[(e_t − e_s)^2] = E[e_t^2] + E[e_s^2] − 2E[e_t e_s] = 2 Var_c(e)`
for independent draws (`Cov(e_t, e_s) = 0`). Hence `L_delta` is (up to the pair-count normalisation)
**twice the within-centre error variance**, divided by the centre population: the realised loss is
`2 Var_c(e) / population`. Consequences, stated explicitly as the issue requires:
the term constrains only the **within-centre spread** of the errors; a common offset `c` added to
every prediction changes the pointwise loss but leaves `Var_c(e)` and therefore `L_delta` unchanged,
so the offset stays the pointwise term's responsibility (the pointwise loss is retained); the term
uses the **same** `y_t` as the pointwise loss and therefore introduces **no new independent graph
information**; and it does **not** guarantee that the item vectors avoid collapse.

## 4. Gradient budget and schedule

`L_total = L_RQVAE + lambda_point * L_point + lambda_delta * L_delta`.

- `lambda_delta` is **calibrated by gradient norms**, never by loss values: on the #269-style
  calibration batch (real data, the same first-batch probe) the gradient norms of the **weighted**
  pointwise term (`lambda_point * ∇L_point`) and of `L_delta` are measured on the real modules
  (encoder + the three codebooks) and

  ```
  lambda_delta = min over the active modules m of
                 rho * (lambda_point * ‖∇L_point‖_m) / ‖∇L_delta‖_m
  ```

  with `rho = 0.02` (2 % of the pointwise gradient budget). The **per-module minimum** is used rather
  than a single global ratio so that *every* module's delta gradient stays below `rho` times its own
  weighted pointwise gradient; a global ratio can satisfy the cap in aggregate while one codebook is
  pushed far past it. A module with a zero pointwise gradient or a zero delta gradient contributes
  nothing to either side and is excluded from the minimum. The value is frozen in a buffer before
  training, exactly like `curvature_weight`; the realised shares (the global `weighted_share` and the
  per-module `weighted_ratio`) are recorded at the calibration.
- **Progressive enablement**: the effective weight is `lambda_delta * min(1, global_step / RAMP)` with
  `RAMP = 200` optimizer steps, so the term starts at zero budget and grows into its 2 % share.
- **Two-stage probe**: the gradient norms, the angle between the two terms' gradients, the actual
  parameter update and the post-step objective change are measured (a) at the calibration batch and
  (b) at a representative later stage (epoch 50 probe). If the realised share exceeds `rho`, the
  calibration is redone with a smaller share before the formal run; the share is never enlarged on
  the basis of a single batch.
- The gradient budget is the **only** new knob in round 1.

## 5. Shapes, boundaries, tolerances

| item | value / rule |
|---|---|
| sampled triplets per step | `m = 32`, all with the same centre (`sample['triples'][:, 0]`) |
| pairs per step | `m(m-1)/2 = 496` (upper triangle, `offset=1`) |
| prediction / target dtype | float32 (the model's), targets cast to the prediction dtype |
| centre denominator | `sample['triple_population']` (the centre's valid triplet count, ≥ m) |
| `m < 2` | `L_delta = 0` exactly (no pairs, no gradient contribution) |
| identical triplets | the difference is exactly 0; the term contributes 0 (the sampler draws without replacement, so this is a boundary test only) |
| constant prediction | the error spread equals the target's own spread, so `L_delta = mean_pairs (y_t - y_s)^2 / population` — positive, so the term **counters a collapsed prediction** — and it is **independent of the constant offset** (the offset stays the pointwise term's responsibility); the pair is used with the pointwise term |
| different-centre mispairing | pairing rows of different centres is forbidden; a mispair test asserts the implementation refuses it (the sampler guarantees one centre per step) |
| invalid mask | the sampler only draws valid rows of the centre block; the target finiteness and cross-component checks stay as in the parent |
| coincident points | zero/identical endpoints give identical predictions; the difference is 0 |
| fixed kappa | `kappa_t` is a host lookup; `autograd.grad` on it is never requested; the frozen `target_scale` is constant |
| hard q / STE | the gradient of `L_delta` reaches the encoder and the codebooks through the existing STE path; the test asserts non-zero parameter deltas |
| numerical stability | the loss and the gradients must be finite; a non-finite value aborts the run (the parent's behaviour) |
| degenerate prediction | a NaN prediction propagates to a non-finite loss and is caught by the parent's finite check |

## 6. Paper / official-implementation comparison

The module rows of `results/research/issue271/CONTRACT.md` §6 are inherited unchanged (TIGER RQ-VAE,
κ-stereographic distance, per-triple κ_t, the inequality constraint, the global graph distance). The
new module:

| module | paper position | official implementation | our choice |
|---|---|---|---|
| same-centre difference protection | no external position: derived here from the frozen assets (the within-centre error variance identity of §3) | the repository has no ranking/contrastive local loss on the curvature; #267's relative-ordering loss (historical, read-only) ordered the magnitudes but not the **signed differences** | `L_delta` on the signed differences of the same centre, centre-balanced, with the pointwise loss retained; no new information, no trainable scale |

## 7. Pre-registered predictions

| id | prediction | falsifier |
|---|---|---|
| P1 | the within-centre difference RMSE (C6) falls below the measured #269 baseline and the within-centre ordering (C7) rises above it | C6/C7 unchanged |
| P2 | C1 ≤ 1.8935063111536612 **with** C2 > 0.015850898303185326 and C3 > 0.4324533490974272 (the local error is not bought by compressing the prediction) | Spearman/sign stay below TIGER |
| P3 | C4 ≥ 0.5 and C5 ≥ 0.1 (the variance/slope recover) | the shrink persists |
| P4 | C12 ≥ 10,918 and C13 ≥ 8 and C15 ≤ 1.05 × the #265 same-criterion error (the delta term does not cost SID capacity or reconstruction) | L2/rank stay at the #269 level or the reconstruction regresses |
| P5 | C9 ≤ 0.338335882630658 with C10 ≤ 0.798295 and C11 ≤ 0.278106, computed on the **full ruler** | the full-ruler J is above the bound |
| P6 | the realised gradient share of the delta term stays ≤ rho at both probes (no amplification) | the share exceeds the cap |

## 8. Evaluation

The frozen evaluator `results/research/issue271/local_geometry.py` (assets, scale and algorithm
unchanged) provides C1–C3, C8, C9–C11; the issue-273 analysis script
`results/research/issue273/within_centre_metrics.py` adds C4–C7 and must reproduce the frozen
evaluator's MAE/Spearman/sign exactly on the frozen baselines before its new statistics are used.
C12–C15 come from `results/research/issue271/collect_metrics.py` on the run's exported hard SID and
the run's reconstruction. Stage3 is forbidden in every outcome.
