# Issue #267 mechanism contract

Frozen 2026-11-26 after `DIAGNOSIS.md` and before candidate implementation.
Parent: #265 R2 109. The parent point-value structural term is retained; the only
new structural term is a same-centre relative ranking loss.

## Relative loss

For one training-side structural sample, the predicted curvature endpoints are
`p_t = κ_q(t)` (the existing `sampled_values` output on hard q) and the frozen
targets are `y_t = κ_graph(t)`. For every unordered pair `(t,u)` in the sample:

```
Δ_tu = y_t - y_u
d_tu = p_t - p_u
```

Only pairs with the same centre (`m_t == m_u`) and `|Δ_tu| > 1e-6` are used.
Ties, cross-centre pairs and non-finite inputs are excluded. Let `N` be the valid
pair count. The term is

```
L_rel = (1/N) * Σ softplus(-sign(Δ_tu) * d_tu / tau)
```

with `tau = 1.0516836255159168` (the target-difference scale measured from
187,102 within-centre pairs over 518 deterministic centres of the frozen training
target table; frozen before training). If `N=0`, the term is a differentiable zero
and the count is reported; the calibration path rejects an all-tie calibration
sample.

`alpha` is not chosen by a score/Recall search. On the fixed initial structural
sample (seed 42, `draw(0,0,1,track=False)`) compute the parameter-gradient norms
of the point term and `L_rel` over all non-decoder trainable parameters, then set

```
alpha = 0.2 * ||grad L_point||_2 / ||grad L_rel||_2
```

The 0.2 share is a pre-registered gradient-budget allocation; it is recorded,
not tuned. It is frozen before the first optimizer step.

The structural scalar used by the parent calibration is

```
L_struct = L_point + alpha * L_rel
L_total  = L_RQVAE + lambda * L_struct
```

where `lambda` remains the parent's module-budget calibration applied to the
actual `L_struct`. No point loss is dropped; no gate-only shortcut is accepted.

## Forward and gradient route

- Hard q is the exact sum of the three selected codebooks at the exported SID.
- Euclidean residual selection is unchanged and gate-independent.
- `L_rel` reads `p_t` and therefore routes gradients through the real hard-q STE
  to the encoder, the selected codebooks and the gate that determines the local
  mixed distance. The frozen GNN/token cache has no forward, backward or optimizer
  participation.
- The target `y_t` is detached; target-side graph assets are read-only.
- No new parameter, no online GNN, no extra codebook layer, no InfoNCE/edge/prefix
  loss.

## Bounds and edge cases

- `tau>0`; `alpha>0`; `1e-6` tie threshold; no epsilon added to gradients.
- `N=0`: zero value, zero gradient, explicit `valid_pairs=0`; calibration rejects
  it because a relative gradient cannot be defined.
- All pairs are within one centre; cross-centre pair count must be zero.
- Non-finite predicted/target or a negative/zero tau raises.
- The pair reduction is the arithmetic mean over valid pairs; ordered duplicates
  are not counted (upper triangle only).
- The implementation must preserve the parent's dtype/STE and use no Python-side
  device readback in the hot path beyond the frozen sample arrays.

## Test and admission contract

- Independent reference: a separate closed-form Python implementation of the pair
  loss and its gradient, hand values in tests.
- Known answers: sign-correct ordering loss < sign-wrong ordering loss; tied target
  excluded; singleton/empty returns zero; exact pair count for 28 rows = 378.
- Gradient tests: nonzero encoder, gate and codebook gradients from the actual
  training loss; relative term alone changes q in the correct direction.
- Mutation tests: flip sign, remove tie mask, include cross-centre pairs, use
  mean over all pairs including ties, and swap the predicted difference; each must
  be caught by a named assertion.
- Real-entry checks: one-epoch smoke/export, DDP four-rank, collision <0.40,
  §63 layer record and the parent's fixed independent evaluation.
- `alpha`, `tau`, pair count and gradient ratios are persisted before Stage2.

## Paper/official comparison

| module | source | choice |
|---|---|---|
| parent hard-q local mixed curvature | #265 `model/mechanism.py`; #262 contract | retained unchanged |
| point-value Huber | @265 `sampled_curvature` | retained |
| relative pair loss | issue #267 §2, no external official implementation | softplus sign ranking with target-derived tau, exact upper-triangle mean |
| selection | #265 Euclidean + floor/last-level Sinkhorn | unchanged |

## Non-goals

No test-guided weight tuning, no second seed, no extra baseline, no GNN/TIGER
retraining, no public-infra change, no test-set mechanism selection, no claim that
a larger structural gradient is an optimization benefit.
