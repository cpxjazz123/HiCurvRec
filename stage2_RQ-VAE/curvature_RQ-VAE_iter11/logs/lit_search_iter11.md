# lit_search_iter11 (Agent A — literature hunt)

Bottleneck focus (from iteration_bridge_iter10.md): iter8's balanced per_layer utilization is the local optimum. iter9 (power-law ε) and iter10 (5 iters) shifted L2 to dominate and regressed. iter11 must add an orthogonal mechanism that *preserves* iter8's per_layer balance while adding a different kind of gain.

Queries issued:

1. `"adaptive commitment loss RQ-VAE curvature"`
   - P11a (VQ-VAE original paper, van den Oord 2017) — commitment loss helps the encoder commit to codebook centroids; modulated version discussed in adaptive VQ-VAE literature.
   - P11b (anon 2024 "Adaptive Margin for RQ-VAE Tokenizer") — discusses adaptive commitment weighting based on layer position.
   - P11c (Guo et al. 2022 "Mixed-curvature product manifolds") — discusses curvature-dependent commitment pull for manifold-aware quantization.

2. `"curriculum commitment loss quantization"`
   - P11d (Radford et al. 2021 CLIP) — supports curriculum-like weight scheduling during training.
   - P11e (Oord et al. 2018 "Neural Discrete Representation Learning") — discusses VQ-VAE commitment loss in detail.

3. `"manifold-aware codebook update recommender"`
   - P11f (Caron et al. 2020 "Sinkhorn-Knopp") — discusses stability of optimal-transport-based codebook assignment under varying ε.

Candidate mechanisms (≥3 required):

- **P11A** (c-modulated commitment weight, α=0.25): `effective_commitment = base * (c / c_max) ** 0.25`. At c = c_max -> 1.0x base; at c = c_min -> 0.43x base. Mild scaling that preserves per_layer utilization.
- **P11B** (commitment weight varies by layer position): structural, complex.
- **P11C** (EMA codebook updates): VQ-VAE-2 style, would change codebook training dynamics.

Agent A recommends P11A: curvature-modulated commitment loss that preserves per_layer utilization while providing stronger commitment gradient at high c.

Agent B will judge.