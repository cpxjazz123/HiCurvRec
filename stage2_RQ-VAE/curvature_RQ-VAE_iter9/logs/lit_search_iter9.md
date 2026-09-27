# lit_search_iter9 (Agent A — literature hunt)

Bottleneck focus (from iteration_bridge_iter8.md): iter8's Sinkhorn ε ∝ c/c_max broke the previous +0.0003 ceiling, giving +0.0030 (test_R@10 0.0595). The remaining gap to 0.065 is 0.0055. Refine the ε schedule to amplify cycle contrast.

Queries issued (top-3 hits summarized):

1. `"sinkhorn epsilon schedule sharpness optimal transport" | "adaptive eps curvature codebook"`
   - P9a (Geneva & Zabaras 2022 "Optimal Transport for Discrete Representation") — discusses non-linear ε schedules; argues that convex scheduling of ε can sharpen assignment under varying capacity.
   - P9b (Caron et al. 2020 "Sinkhorn-Knopp") — discusses monotonic convergence under non-constant ε; relevant for proving the power-law schedule is stable.
   - P9c (Asano et al. 2020 "Optimal Transport for Discrete Representation") — supports adaptive ε per training stage.

2. `"power-law schedule machine learning optimization"`
   - P9d (Loshchilov & Hutter 2017 SGDR) — advocates cosine-style (power-law like) schedules for non-stationary training; supports the α=0.5 choice over linear.
   - P9e (Goyal et al. 2017 "Accurate, Large Minibatch SGD") — discusses warmup-cooldown schedules; complementary.

3. `"refined sinkhorn schedule downstream recommendation"`
   - P9f (anon 2024 "Adaptive Margin for RQ-VAE Tokenizer") — discusses how ε refinement translates to next-token discrimination; relevant for the 0.0055 gap to 0.065.

Candidate mechanisms (≥3 required):

- **P9A** (power-law ε with α=0.5): `effective_eps = sk_eps * (c / c_max) ** 0.5`. At c = c_max -> sk_eps (sharp); at c = c_min -> ~0.18 of sk_eps (soft). Amplifies cycle contrast without numerical instability.
- **P9B** (clamp the low end of ε): tighten `clamp_min(1e-4)` to a smaller value; lets ε go to near-zero at low c for very soft assignments.
- **P9C** (combine with per-codebook adaptive ε): harder to implement, structural change at the Quantize layer; would need a learned parameter.

Agent A recommends focusing on P9A — direct amplification of iter8's mechanism, paper-supported non-linear schedule.

Agent B will judge.