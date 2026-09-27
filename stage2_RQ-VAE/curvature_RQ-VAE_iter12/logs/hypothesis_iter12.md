# hypothesis_iter12 (Agent C — hypothesis designer)

Mechanism: keep iter11's assignment step (linear Sinkhorn ε, 3 iters) and c-modulated commitment weight (α=0.25); only change cyclic period from 100k to 50k. Doubles the curvature cycle frequency within the 100k step budget.

Direct effects (DE-1..3 — must be observed for ALIGNED):
- DE-1: 2× curvature cycles within 100k steps vs iter11 (i.e., the curriculum step reaches |sin(πt/50k)| peak twice instead of once).
- DE-2: warm-start loads iter8 weights for embeddings/codebooks/c_layer_scales. First-step rl ≈ iter8's.
- DE-3: per-layer utilization stays in iter8's range [0.20, 0.23, 0.22] (the assignment step is unchanged).

Proxy hypothesis (D, descriptive):
- Stage3 `test_R@10` ≥ 0.0615 (≥ +0.002 over iter11 0.0598) for gap-closing.
- If gap-closing fails, classify as `SID_GEOMETRY_OK_DOWNSTREAM_FLAT` (类 3).

Forbidden directions: changing any other Stage-2 parameter (final attempt).

Verification plan:
- MVG PASS on a 640-pair batch with iter8 warm-start.
- Stage2: 100k steps, observe DE-1..3 (curriculum step log will show 2 cycles vs iter11's 1).
- Stage3: 150 epochs, beam=20, four-GPU DDP.