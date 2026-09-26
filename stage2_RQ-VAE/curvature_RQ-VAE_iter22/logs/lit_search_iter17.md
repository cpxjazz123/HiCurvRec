# lit_search_iter17 (Agent A — targeted literature search)

## Bottleneck from Agent G

iter16 kept a healthy SID but `test_R@10=0.05688`, below iter11 (`0.05977`) and iter15 (`0.05824`). Its fixed `c0=0.663` removed the cyclic low-curvature regime. Search scope: curvature-dependent optimization that preserves the existing cyclic schedule and changes only the Stage2 optimizer.

## Targeted queries and primary hits

1. Query: `Riemannian Adam hyperbolic neural networks Poincare ball adaptive optimization curvature`
   - Whiting, Wang, Xin, *Convergence of Hyperbolic Neural Networks Under Riemannian Stochastic Gradient Descent*, published online 2023-10-05; discusses Riemannian Adam and reports hyperbolic benchmark simulations. https://link.springer.com/article/10.1007/s42967-023-00302-9
   - Bécigneul and Ganea, *Riemannian Adaptive Optimization Methods*, product-manifold adaptive moments and Poincaré-ball WordNet experiments. https://arxiv.org/abs/1810.00760
   - The 2023 Springer paper's Riemannian update uses a metric gradient and manifold exponential-map step; this supports optimizer-level geometry rather than changing the curvature schedule. https://link.springer.com/article/10.1007/s42967-023-00302-9

2. Query: `product manifold metric preconditioning Riemannian optimization curvature blocks 2023`
   - Gao, Peng, Yuan, *Optimization on Product Manifolds under a Preconditioned Metric*, arXiv 2306.08873 (2023); develops product-manifold methods with an operator-defined preconditioned metric and reports accelerated numerical results. https://arxiv.org/abs/2306.08873
   - The paper's operator approximates diagonal Riemannian-Hessian blocks; it is evidence for metric-based preconditioning, not direct evidence for the specific `1/c` rule. https://arxiv.org/abs/2306.08873
   - Bécigneul and Ganea's product-manifold adaptive method remains a directly relevant optimizer reference, but predates the 2023+ evidence preference. https://arxiv.org/abs/1810.00760

## Candidate mechanisms (no ranking)

- **P17A — Riemannian Adam on hyperbolic factors:** compute metric gradients and use a manifold-consistent adaptive update. It could preserve cyclic curvature while making optimizer steps respond to the live geometry; it requires a larger change to parameter-manifold handling.
- **P17B — per-layer `1/(c_l+eps)` adaptive-update scaling:** keep the cyclic `c_l(t)` schedule and apply a normalized inverse-curvature factor to each layer's AdamW parameter update. It could restore a curvature-dependent optimizer effect without freezing `c_l`; placing the factor after Adam's moment normalization avoids scalar-gradient cancellation.
- **P17C — product-manifold preconditioned metric:** use a layer-block metric preconditioner informed by curvature. It could adapt update scales across factors, but the cited paper's Hessian-based operator is not the project's `1/c` rule.

Evidence is optimizer-focused; it does not establish a downstream recommendation gain. Stage3 `test_R@10` remains the decision metric.
