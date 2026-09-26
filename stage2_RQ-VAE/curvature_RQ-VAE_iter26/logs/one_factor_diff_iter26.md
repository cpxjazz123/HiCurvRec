# Iter26 One-Factor Diff Manifest

```
PARENT_ITER = iter16
PARENT_COMMIT = 945f20d159d21b33026b3be3127c4d9416bbec5c (skill bump; no functional change to Iter16 source tree)
CANONICAL_BASELINE_ITER = iter18
EXPERIMENT_TYPE = single_factor
ACTIVE_MECHANISMS_BEFORE = iter16 closed-form fixed curvature (residual_norm = [0.001, 0.932889, 1.0])
NEW_MECHANISM = raw_residual_median substitution (residual_norm = [1.0, 0.10941, 0.09331])
ACTIVE_MECHANISMS_AFTER = iter26 closed-form fixed curvature (residual_norm = [1.0, 0.10941, 0.09331])
```

## Only conceptual change

The Iter16 closed-form mapping
`s_l = log1p(B_l) / log1p(m_l / m_min)`, `c_l = clip(c_base * exp(alpha * z_l), 0.05, 1.5)`
with `c_base = 0.5`, `alpha = 0.2`, `m_min = min(m)` is preserved byte-identical.
The single conceptual change is the **substituted `m` vector**: Iter16 used the
Iter1-calibrated `[0.001, 0.932889, 1.0]`; Iter26 uses the raw residual medians
`[1.0, 0.10941, 0.09331]`. The final `c_l` vector therefore changes from
`[0.6634, 0.4347, 0.4334]` (Iter16) to `[0.6145, 0.5333, 0.3814]` (Iter26).

No other conceptual change is made. Every other line of `curvature_RQ-VAE.py`,
`modules/quantize.py`, and `modules/rqvae.py` is byte-identical to Iter16,
except for three cosmetic edits (MECHANISM_NAME label, log lines, and the new
`_check_fixed_curvature_invariant` helper that does not change curvature
values).

## Changed source files

- `scripts/computed_behavior_branching.json`: `residual_norm` (and
  `raw_residual_medians`) field rewritten from `[0.001, 0.932889, 1.0]` to
  `[1.0, 0.10941, 0.09331]`. `closed_form_c_l` recomputed and locked at
  `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]`. `m_min`
  recomputed as `0.09331`.
- `curvature_RQ-VAE.py`: declared `FIXED_CURVATURE_INVARIANCE_STEPS`,
  `FIXED_CURVATURE_INVARIANCE_TOL`, `_check_fixed_curvature_invariant`,
  and two `_check_fixed_curvature_invariant(...)` call sites at the
  training start and inside the step loop. `iter16_curvature` snapshots are
  already emitted at the same mandatory steps; the new invariant check is
  an additive layer of safety, not a source of new mechanism.

## Changed equations / constants

| Symbol | Iter16 | Iter26 |
|---|---|---|
| `B_l` (branching) | `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` | identical |
| `m_l` (residual) | `[0.001, 0.932889, 1.0]` | `[1.0, 0.10941, 0.09331]` |
| `c_base` | `0.5` | identical |
| `alpha` | `0.2` | identical |
| `m_min` | `0.001` | `0.09331` |
| `s_l` | `[4.3452, 0.1316, 0.1014]` | `[1.2238, 1.1605, 1.0106]` |
| `z_l` | `[1.4142, -0.6995, -0.7147]` | `[1.0313, 0.3224, -1.3537]` |
| `c_l` | `[0.6634, 0.4347, 0.4334]` | `[0.6145, 0.5333, 0.3814]` |

## Inherited mechanisms (unchanged from Iter16)

- Cyclic `c(t)` is disabled (`c_cyclic_min`, `c_cyclic_max`,
  `c_cyclic_period` are still in `curvature_RQ-VAE.py` but unused by
  `Quantize.get_c()` when `fixed_curvature` is provided).
- Learnable `c_layer_scale` is replaced by `_fixed_c` register_buffer;
  `curvature_regularization` is identically zero.
- Behavior-loss weight = 0.20, behavior temperature = 0.07 (unchanged).
- Sinkhorn ε = 0.05, Sinkhorn iters = 3 (unchanged).
- Optimizer AdamW lr = 1e-3, weight_decay = 1e-4 (unchanged).
- Warm-start from iter8's `rqvae_best.pth` (unchanged; `_fixed_c` and
  `c_layer_scale` are both excluded from the transfer).

## Optimizer differences

None.

## Loss differences

None.

## Stage 1 / Stage 3 differences

None. Stage1 embeddings are read from the immutable
`/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy`.
Stage3 trainer and wrapper are unchanged from Iter25; the only edit is
`trainer.RQVAE_VARIANT = "iter26_closed_form_with_raw_residual"` and the
output directory `results/stage3_T5Train/curvature_RQ-VAE_iter26/`.

## Explicit statement

The only conceptual change is the registered mechanism: the substitution of
the Iter1-calibrated residual norm `[0.001, 0.932889, 1.0]` for the raw
residual medians `[1.0, 0.10941, 0.09331]` inside Iter16's closed-form mapping.
All other components are inherited verbatim from Iter16. No stacking,
no second novelty, no Stage1/Stage3 modification.