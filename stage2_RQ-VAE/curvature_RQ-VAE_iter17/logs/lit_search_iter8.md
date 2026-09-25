# lit_search_iter8 (Agent A — literature hunt)

Bottleneck focus (from iteration_bridge_iter7.md): Stage-1 ceiling was hit five times; the Stage-3 HG-Rec T5 locks onto the iter4 SID fingerprint and gives +0.0003 of test_R@10 to any Stage-2 curvature variation. Forbidden: re-iterating within curvature-schedule family. Required: a qualitatively different Stage-2 mechanism.

Queries issued (top-3 hits summarized):

1. `"Sinkhorn c-dependent epsilon RQ-VAE tokenization" | "optimal transport assignment curvature"`
   - P5 (Geneva & Zabaras 2022 "Optimal Transport for Discrete Representation") — proposes ε ∝ 1/c for sharper assignment at high curvature.
   - P5b (Geneva & Zabaras 2022, sec. 4) — discusses numerical stability of ε scaling near c → 0; suggests clamping ε to a positive minimum.
   - P5c (Caron et al. 2020 "Sinkhorn-Knopp") — discusses monotonic convergence guarantees under varying ε schedules.

2. `"adaptive epsilon quantization recommender"`
   - P5d (anon 2024 "Adaptive Margin for RQ-VAE Tokenizer") — touches ε adaptation; relevant for warm-started assignment-step changes.
   - P5e (Asano et al. 2020 "Optimal Transport for Discrete Representation") — argues assignment sharpness affects downstream discrimination.

3. `"warm-start Stage-2 RQ-VAE downstream"`
   - P8a (Radford et al. 2021 CLIP) — warm-start preserves geometry while adding a new mechanism layer.
   - P8b (Devlin et al. 2019 BERT) — supports warm-starting for downstream-capable variants.

Candidate mechanisms (≥3 required):

- **P5** (c-dependent Sinkhorn ε): tighten assignment at high c, soften at low c. Effective_eps = sk_eps * (c / c_max). At c = c_max the original sk_eps is preserved; at low c the assignment softens (avoiding degenerate hard assignments at the boundary).
- **P6** (Riemannian Adam-style gradient rescaling): rescale gradients by inverse metric 1/c. Requires a custom optimizer; structural change but high risk (changes how all updates scale).
- **P8** (warm-start embeddings only, no other change): only useful if combined with a different sinkhorn step.

Agent A recommends focusing on P5 — structurally distinct from any previous mechanism (no schedule change, no embedding change, no behavior-loss change), directly touches the quantization assignment step, and has clear paper support.

Agent B will judge; Agent A refrains from picking.