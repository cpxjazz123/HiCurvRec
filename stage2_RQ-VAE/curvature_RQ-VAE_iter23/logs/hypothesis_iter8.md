# hypothesis_iter8 (Agent C — hypothesis designer)

Mechanism: Sinkhorn assignment epsilon scaled by current layer curvature: `effective_eps = sk_eps * (c / c_max).clamp_min(1e-4)`. At c = c_max the original `sk_eps` is preserved (high-c sharpen); at low c the eps shrinks, softening assignments (avoids degenerate hard assignments at the boundary).

Direct effects (DE-1..3 — must be observed for ALIGNED):
- DE-1: per-layer Sinkhorn iterations converge to measurably different assignment distributions across the cyclic-c schedule (compared to iter7's constant sk_eps). Verify by reading the assignments matrix entropy over the cycle.
- DE-2: warm-start loads iter7 weights for embeddings/codebooks/c_layer_scales. First-step reconstruction loss should remain comparable to iter7's (~1.30).
- DE-3: quantization loss (rqvae_loss) should stay in iter7's range (≈0.5–0.7) since the assignment softmax is still applied.

Proxy hypothesis (D, descriptive):
- Stage3 `test_R@10` ≥ 0.065 (hard target). If DE-1..3 are observed and Stage3 fails, classify as `SID_GEOMETRY_OK_DOWNSTREAM_FLAT` (类 3).

Forbidden directions: re-encoding Stage1 embeddings (out of scope); changing Stage3 architecture (out of scope); reverting to iter4-style ε (forbidden, fails gap-closing).

Verification plan:
- MVG PASS on a 640-pair batch with iter7 warm-start (c_layer_scale reset) and curriculum step at mid-cycle.
- Stage2: 100k steps, observe DE-1..3 in log tail at step ≈ 50k.
- Stage3: 150 epochs, beam=20, four-GPU DDP.