# Issue255: layer-wise three-curvature gating with direction-aware ego tokens

## 1. What was built

One new candidate, `stage2_RQ-VAE/TIGER_prefix_curvature_gate_rq`, is an independent copy of the archived
#253 three-expert arm (itself a copy of the frozen `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` plus the #253
mechanism). Only `ARM_NAME` identifies it. Everything else that the issue freezes is preserved: the 768-d
semantic input, the 32-d latent, three 256-word levels, #253's actual assignment protocol (L1/L2
floor-constrained nearest, L3 Sinkhorn), the original reconstruction/commitment/codebook losses, four cards
with global batch 4096, seed 42, 3000/150 epochs, the expert curvatures `{-1,0,+1}` only, and an
end-to-end trainable GNN and gate.

Three mechanisms were added inside the existing framework, and nothing else:

1. **Direction-aware ego tokens.** The shared directed GNN now produces four token types per item: the
   frozen full one-hop ego, the frozen complete two-hop ego, the centre with all in-neighbours, and the
   centre with all out-neighbours. The two new populations keep every real directed edge whose endpoints are
   inside the induced node set, are not claimed to be mutually exclusive, and never invent a neighbour: an
   item with an empty neighbourhood yields a root-only token. A fixed per-type encoding keeps the four
   slots distinguishable. The new populations are a new frozen asset, `assets/v10_direction_ego/`, built
   from the frozen visible graph and audited item by item.

2. **Layer-wise three-curvature gating.** One shared GNN computation feeds a per-level expert attention.
   Every level owns trainable queries, and the returned geometry weights `w_i^(l)` have shape
   `[N, 3 levels, 3 experts]`, normalised over the experts for that level; the token attention is
   normalised over the token axis. No level is hand-assigned a curvature. Each level's mixed cost is
   `Σ_k w_i^(l,k) d_k²(r_i^(l), e_{l,j})` with the frozen hard selection and the same-cost soft backward
   unchanged; no neighbour-distance term was added.

3. **Cumulative-prefix sectional supervision.** The hard prefix `q_i^(l) = Σ_{j≤l} e_{i,j}` is bitwise the
   actual SID's first `l` codes, and each level's GraphMoRE A.1 proxy is computed with a fixed,
   gate-independent equal-weight three-expert distance over the same frozen triples and reference sets.
   The structural term became `λ Σ_l α_l Huber(K_l/target_scale, K_G/target_scale)` with `α_l = 1/3`.
   For the corrected generation, `λ` is calibrated exactly once at initialization by the frozen rule
   `λ = ‖∇_encoder L_original‖ / ‖∇_encoder L_curv‖`, and then held fixed throughout optimization.
   Per-level distance scales are frozen from the training-side initial cumulative prefixes.

## 2. Protocol and honesty boundaries

- The training objective uses the frozen finite sampling and quantisation population, so the training
  gradient is a biased estimate of the full-population objective; the evaluation keeps the independent full
  28,391,948-triple measurement under one fixed cross-model scale.
- Compared with #253, this arm jointly changes direction-token populations, per-level queries and
  cumulative-prefix supervision. Both arms retain the same three experts. The joint intervention and its
  training trajectory cannot identify any single module's causal contribution; catalogue-derived
  collision suffixes can also change a downstream T5 vocabulary.
- No behaviour-neighbourhood constraint, pair ranking, InfoNCE, edge prediction, additional behaviour loss,
  PPR, four-node exhaustive graphlet, teacher/GNN pretraining, TIGER retraining, multi-seed or change of
  the 768/32 dimensions or the SID protocol was introduced.

## 3. Invalidated first run

`issue255_formal` completed3000 epochs but used a reconstruction-only calibration numerator. It is
`INVALID_IMPLEMENTATION`, not a scientific comparison result. Its immutable source and artifacts are
preserved. The owner authorized exactly one corrected same-arm generation, `issue255_formal_r1`,
with unchanged optimizer, budget, seed and mechanism; dispatch requires complete current-source
test, actual-loss four-rank, admission and performance credentials. No score from the invalid run
is used to tune the correction, and no Stage3 was launched for it.

