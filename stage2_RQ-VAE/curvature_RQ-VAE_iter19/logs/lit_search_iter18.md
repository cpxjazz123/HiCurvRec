# Literature and implementation review — iter18 optimizer direction

## Prior candidate: P17A Riemannian Adam

Bécigneul and Ganea formulate Riemannian adaptive optimization for a Cartesian product of Riemannian manifolds, with adaptivity across product factors; their experiments include Poincaré-ball WordNet embeddings. The construction is not merely a scalar learning-rate rescale. It requires a Riemannian gradient and optimizer state interpreted in the relevant tangent spaces. [Bécigneul & Ganea, *Riemannian Adaptive Optimization Methods*, 2019](https://arxiv.org/abs/1810.00760).

Whiting, Wang, and Xin describe hyperbolic neural-network training on a fixed-curvature Poincaré ball, explain exponential-map updates and parallel transport, and discuss/evaluate Riemannian Adam. This supports the fixed-manifold algorithmic ingredients, not an optimizer whose manifold curvature changes cyclically each step. [Whiting et al., *Convergence of Hyperbolic Neural Networks Under Riemannian Stochastic Gradient Descent*, online 2023 / journal 2024](https://doi.org/10.1007/s42967-023-00302-9).

The optimizer candidate is therefore mathematically motivated for fixed-curvature factors, but the cited sources do not establish its behavior or convergence for the repository's live cyclic `c_l(t)`. Treating each step as if it belonged to a fixed metric would be an extrapolation, not a directly supported application. This is a limitation, not evidence that the experiment cannot work.

## Product-manifold metric preconditioning is not a drop-in substitute

Gao, Peng, and Yuan construct preconditioned metrics for product-manifold optimization using operators that approximate diagonal Riemannian-Hessian blocks, with exact block, left/right, and Gauss–Newton constructions. Their work does not justify replacing those operators with an inverse-curvature scalar. [Gao et al., *Optimization on product manifolds under a preconditioned metric*, 2023](https://arxiv.org/abs/2306.08873). This direction would require its own approximation and validation design.

## Repository facts that constrain a port

- In the iter17 code, codebook weights are ordinary Euclidean `nn.Embedding` parameters. `QuantizeLoss` maps query and codebook vectors through `_expmap0_t` to compute Poincaré distances; codebook storage itself is not a ball-valued parameter.
- The existing `_transport_between_t` changes the curvature representation of a tangent vector at the origin. It is not parallel transport of optimizer momentum between two codeword locations.
- The model's curvature schedule varies with training step, while iter17's optimizer is AdamW with a normalized per-layer `1/(c_l + 1e-3)` learning-rate multiplier. Iter17 completed Stage3 but reached `test_recall@10=0.059071362662999005`, below the strict `>0.065` target; its score alone does not establish the mechanism's cause.

A literal Riemannian Adam implementation would need an explicit choice of parameter manifold/storage, gradient metric, first-moment transport, second-moment meaning under changing curvature, and the curvature at which transport/update are defined. Those choices are absent from the bridge's one-line P17A description. In particular, the repository's curvature-change helper cannot be reused as point-to-point moment transport.

## Direction decision implication

Do not silently implement P17A by applying a Riemannian formula to the existing Euclidean embedding tensor or by renaming `_transport_between_t`. Agent B should either specify and justify every missing cyclic-metric choice before selecting P17A, or select a distinct, better-specified optimizer mechanism. In all cases preserve the cyclic curvature schedule, change one mechanism only, verify its direct effect and nonzero gradient path before Stage2 training, and use complete Stage3 `test_recall@10 > 0.065` as the sole adoption objective. Stage2 SID statistics remain descriptive; `hitrate@50` is not a gate.
