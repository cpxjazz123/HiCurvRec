# hypothesis_iter11 (Agent C — hypothesis designer)

Mechanism: keep iter8's assignment step (linear Sinkhorn ε, 3 iters); only modulate the commitment loss by `effective_commitment = base * (c / c_max) ** 0.25`. The encoder pulls residuals toward codebook centroids slightly more strongly at high-c phases and slightly more softly at low-c phases.

Direct effects (DE-1..3 — must be observed for ALIGNED):
- DE-1: commitment loss weight varies mildly with c (between 0.43x base at low c and 1.0x base at high c).
- DE-2: warm-start loads iter8 weights for embeddings/codebooks/c_layer_scales. First-step rl ≈ iter8's.
- DE-3: per-layer utilization stays in iter8's range [0.20, 0.23, 0.22] (the assignment step is unchanged).

Proxy hypothesis (D, descriptive):
- Stage3 `test_R@10` ≥ 0.0615 (≥ +0.002 over iter8 0.0595) for gap-closing.
- If gap-closing fails, classify as `SID_GEOMETRY_OK_DOWNSTREAM_FLAT` (类 3).

Forbidden directions: changing ε schedule (forbidden); changing Sinkhorn iterations (forbidden); changing per_layer balance (forbidden).

Verification plan:
- MVG PASS on a 640-pair batch with iter8 warm-start.
- Stage2: 100k steps, observe DE-1..3.
- Stage3: 150 epochs, beam=20, four-GPU DDP.