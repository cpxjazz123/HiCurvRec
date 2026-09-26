# direction_decision_iter17 (Agent B — direction judge)

## Candidate scores (1–5; higher is better)

| Candidate | Gap repair (a/e) | 2023+ evidence (b) | Curriculum fit (c) | Scope / ceiling risk (d) | Curvature compliance (f) | Decision |
|---|---:|---:|---:|---:|---:|---|
| P17A Riemannian Adam on hyperbolic factors | 4 | 4 | 3 | 2 | 5 | Backup; more invasive parameter-manifold change |
| P17B normalized per-layer `1/(c_l+1e-3)` AdamW update scaling | 5 | 3 | 5 | 5 | 5 | **Recommend** |
| P17C product-manifold preconditioned metric | 3 | 4 | 4 | 4 | 5 | Backup; cited operator is Hessian-block based, not `1/c` |

## Recommendation

Choose **P17B: per-layer inverse-curvature scaling of the AdamW parameter update**, with the existing cyclic `c_l(t)` schedule unchanged. Apply the factor after Adam's adaptive moments (as a per-layer effective learning-rate multiplier), not to raw gradients: scalar gradient scaling is largely canceled by Adam's moment normalization and would risk a silent no-op. Normalize by the **initial mean** inverse curvature and hold that reference fixed for the run; this keeps the step-0 mean multiplier at 1 while retaining the cyclic schedule's time-varying effect.

The 2023 hyperbolic-network study discusses Riemannian Adam and geometric updates; the 2023 product-manifold paper supports metric preconditioning. Neither paper directly validates this exact normalized `1/c` rule, so Stage3 evidence is still required. No Stage1 embedding, Stage3 trainer, or extra loss change is authorized.

## Gap-closing relevance

iter16's fixed high-curvature `c0=0.663` regressed despite healthy SID geometry. P17B restores cyclic low/high curvature and makes the optimizer update respond to its current per-layer curvature, targeting the specific failure without repeating a fixed-curvature mechanism.

## Required observable checks

1. Per-layer effective update multipliers equal `(1/(c_l(t)+1e-3)) / mean_j(1/(c_j(0)+1e-3))`, are finite/positive, and vary at the curriculum midpoint versus step 0.
2. Each layer receives a finite, nonzero parameter update after five optimizer steps.
3. Same-seed, same-batch ON/OFF runs for 200 steps differ in final total loss by more than `1e-6`.

## Sources

- Whiting, Wang, Xin (2023 online), *Convergence of Hyperbolic Neural Networks Under Riemannian Stochastic Gradient Descent*: https://link.springer.com/article/10.1007/s42967-023-00302-9
- Gao, Peng, Yuan (2023), *Optimization on Product Manifolds under a Preconditioned Metric*: https://arxiv.org/abs/2306.08873
- Bécigneul, Ganea, *Riemannian Adaptive Optimization Methods*: https://arxiv.org/abs/1810.00760
