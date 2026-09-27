# hypothesis_iter9 (Agent C — hypothesis designer)

Mechanism: power-law ε schedule. iter8 used `effective_eps = sk_eps * (c / c_max)` (linear). iter9 uses `effective_eps = sk_eps * (c / c_max) ** 0.5` (square-root). At c = c_max the original `sk_eps` is preserved; at c = c_min the eps shrinks by `sqrt(c_min/c_max) ≈ 0.183`, vs iter8's `c_min/c_max ≈ 0.033`. The square-root makes the cycle twice as contrastive at high c and three times as contrastive at low c.

Direct effects (DE-1..3 — must be observed for ALIGNED):
- DE-1: Sinkhorn ε at high-c phases (~0.05 of cycle) is approximately equal to iter8's (both ≈ sk_eps). At low-c phases (~0.5 of cycle) it is ~5x larger than iter8's, leading to slightly less-degenerate low-c assignments.
- DE-2: warm-start loads iter8 weights for embeddings/codebooks/c_layer_scales. First-step rl should match iter8's (~1.27-1.29).
- DE-3: per-layer Sinkhorn assignments converge under the softer power-law ε schedule; observed quantization loss should be slightly lower than iter8's because the schedule is less aggressive at the very-low-c end.

Proxy hypothesis (D, descriptive):
- Stage3 `test_R@10` ≥ 0.0615 (≥ +0.002 over iter8 0.0595) for gap-closing.
- If gap-closing fails, classify as `SID_GEOMETRY_OK_DOWNSTREAM_FLAT` (类 3).

Forbidden directions: reverting to a linear ε schedule (forbidden, fails gap-closing); trying per-codebook adaptive ε (P9C, structural change without iteration_bridge backing).

Verification plan:
- MVG PASS on a 640-pair batch with iter8 warm-start.
- Stage2: 100k steps, observe DE-1..3.
- Stage3: 150 epochs, beam=20, four-GPU DDP.