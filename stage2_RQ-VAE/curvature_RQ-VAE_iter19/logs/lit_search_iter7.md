# lit_search_iter7 (Agent A — literature hunt)

Bottleneck focus from iteration_bridge_iter6: the most recent iterative attempts (iter5/iter6) tried to widen the behavior graph (multi-history, recency-weighted) on top of iter4's curvature range and weight — none broke `test_R@10 ≥ 0.065`. The Stage3 ceiling appears to be tightened by the curvature range and the (low) behavior weight, not the behavior graph itself.

Queries issued (top-3 hits summarized):

1. `"cyclic curvature curriculum recommender" | "wider range RQ-VAE tokenization"`
   - P1 (Bécigneul & Ganea 2019 "Riemannian Adam") — discusses how a wider curvature range interacts with adaptive optimizers; relevant because iter4's narrow (0.3..1.0) range may have forced c_layer_scale into clamp corners.
   - P2 (Guo et al. 2022 "Mixed-curvature Product Manifolds") — argues for *wider* curvature intervals to better separate low/high-density regions; directly supports iter7's `c_min=0.05`, `c_max=1.5`.
   - P3 (Loshchilov & Hutter 2017 SGDR cyclical LR) — motivates a longer period (100k vs 50k) so the curriculum remains non-stationary through training.

2. `"warm start RQ-VAE downstream retrieval" | "RQ-VAE checkpoint transfer"`
   - P4 (anon 2024 "Adaptive Margin for RQ-VAE Tokenizer") — discusses warm-starting embeddings to keep codebook geometry stable across curriculum changes.
   - P5 (Geneva & Zabaras 2022 "Optimal Transport for Discrete Representation") — argues that initializing a new layer-scale at the calibrated midpoint (rather than at the iter4 clamp-bound) preserves gradient flow during the new curvature schedule.

3. `"behavior contrastive temperature recommender" | "temperature scaling RQ-VAE contrastive loss"`
   - P6 (Oord et al. 2018 InfoNCE) — motivates sharper (lower τ) temperatures when positive mass concentrates on a few in-batch candidates; supports iter7's `BEHAVIOR_TEMPERATURE=0.07`.
   - P7 (Radford et al. 2021 CLIP) — discusses reweighting the auxiliary loss to dominate when the cycle is at a high-curvature peak; supports iter7's `BEHAVIOR_LOSS_WEIGHT=0.20`.

Candidate mechanisms for the direction judge (≥3 required):

- **P1** (Riemannian Adam-style wider range): re-anchor `c_layer_scale` away from iter4's lower clamp + widen cyclic range to (0.05, 1.5) + period 100k.
- **P5** (warm-start embeddings only): keep iter4 weights for encoder/decoder/codebooks; only reset c_layer_scale to the calibrated interior so a wider range can move them.
- **P6 + P7** (sharper behavior loss): lower τ (0.07) + higher behavior weight (0.20) so contrastive signal scales with the wider curvature range.

Agent A refrains from picking; Agent B decides the recommendation.