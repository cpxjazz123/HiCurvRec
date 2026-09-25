# direction_decision_iter18 (Agent B — direction judge)

## Candidate scores (1–5; higher is better)

| Candidate | Gap repair | 2023+ evidence | Curriculum fit | Scope / ceiling risk | Curvature compliance | Decision |
|---|---:|---:|---:|---:|---:|---|
| P17A Riemannian Adam on live cyclic-curvature codebooks | 4 | 4 | 1 | 1 | 5 | Defer: parameterization, moment transport, and changing-metric semantics are unspecified; fixed-curvature literature does not directly cover this schedule |
| P18B initial-curvature-conditioned per-layer AdamW `β₂` | 3 | 3 | 4 | 4 | 5 | **Recommend** |
| P17C product-manifold preconditioned metric | 3 | 4 | 2 | 1 | 5 | Defer: cited method constructs Hessian-block approximations; a layer scalar is not its proposed preconditioner |
| Repeat P17B live `1/(c_l+1e-3)` learning-rate scaling | 3 | 3 | 5 | 5 | 5 | Reject: already completed, Stage3 target missed |

## Recommendation

Test exactly one new Stage2 mechanism: assign a fixed AdamW second-moment decay per quantizer layer from that layer's initial curvature. Let `c_l(0)` be the existing model curvature after `set_curriculum_step(0)`, and compute

`u_l = clamp(2 * log(c_l(0) / C_CYCLIC_MIN) / log(C_CYCLIC_MAX / C_CYCLIC_MIN), 0, 1)`

`β₂,l = 0.999 - 0.009 * u_l`.

Thus the layer-scale endpoints map to `β₂=0.999` (longer second-moment memory) and `β₂=0.990` (shorter memory). Keep `β₁=0.9`, `eps=1e-8`, base learning rate `1e-3`, and weight decay `1e-4` unchanged. Keep the existing cyclic curvature schedule and all loss, assignment, and Stage3 settings unchanged. Do not retain P17B's live per-layer learning-rate multiplier. Use each quantizer-layer parameter group for its corresponding `β₂,l`; parameters outside those groups retain the ordinary AdamW `β₂=0.999`.

Freeze `β₂,l` for the run after initialization. This avoids changing Adam's second-moment decay while its bias correction is accumulating and makes the intervention strictly one optimizer hyperparameter per factor. It does not claim to implement Riemannian Adam or to have a convergence guarantee on the cyclic schedule.

## Rationale and evidence limits

P17B passed its registered direct-effect and pipeline checks but its complete Stage3 run scored `test_recall@10=0.059071362662999005`, below the strict `>0.065` target. This is a downstream miss, not proof that P17B caused the result. Repeating its LR scaling would not test a new mechanism.

The recommended intervention changes AdamW's second-moment memory by quantizer layer, rather than multiplying the already-adapted update by `1/c`. The curvature schedule supplies a fixed, reproducible layer scale at initialization; shorter memory at larger scale is a falsifiable hypothesis, not an established best setting. Bécigneul–Ganea support adaptive optimization across factors of a product manifold, but their paper does not establish this exact `β₂(c_l(0))` rule. Whiting et al. discuss Riemannian Adam on fixed-curvature hyperbolic factors; that is related context, not direct validation of the selected rule. The exact formula requires the registered direct-effect check and complete Stage3 result.

This is safer to implement than P17A because standard AdamW already accepts a fixed `β₂` per parameter group and needs no custom transport, ball-valued parameterization, or nonstandard bias correction. The tradeoff: `β₂` is tied to each layer's initial scale and does not track the within-run cyclic phase.

## Required checks before Stage2 training

1. Compute and log `c_l(0)`, `u_l`, and `β₂,l`; assert finite values, `u_l ∈ [0,1]`, `β₂,l ∈ [0.990,0.999]`, and correct monotonicity (larger `c_l(0)` gives no larger `β₂,l`).
2. Confirm the optimizer has complete, non-overlapping parameter coverage; each quantizer group uses its registered fixed `β₂,l`, non-quantizer parameters retain `0.999`, and no P17B LR multiplier is present.
3. On the required one-checkpoint/one-batch gradient-path check, verify `total_loss.requires_grad`, non-null `grad_fn`, and at least one nonzero gradient for every applicable loss item; after five steps, every quantizer layer has finite nonzero parameter updates.
4. Same-seed/same-batch 200-step ON/OFF comparison: ON uses the per-layer `β₂,l`; OFF uses `0.999` for every group. Require finite losses and `abs(L_on-L_off)>1e-6`.
5. If checks pass, train all 100k Stage2 steps. Record SID quality descriptively only; do not gate or stop Stage2 on SID metrics or `hitrate@50`. Then run complete Stage3 and adopt only if final `test_recall@10 > 0.065`.

## Sources

- Bécigneul & Ganea, *Riemannian Adaptive Optimization Methods* (2019): https://arxiv.org/abs/1810.00760
- Whiting, Wang, Xin, *Convergence of Hyperbolic Neural Networks Under Riemannian Stochastic Gradient Descent* (online 2023; journal 2024): https://doi.org/10.1007/s42967-023-00302-9
- Gao, Peng, Yuan, *Optimization on product manifolds under a preconditioned metric* (2023): https://arxiv.org/abs/2306.08873
