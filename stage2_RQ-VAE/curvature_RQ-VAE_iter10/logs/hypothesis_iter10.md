# hypothesis_iter10 (Agent C — hypothesis designer)

Mechanism: keep iter8's linear ε schedule (`effective_eps = sk_eps * (c / c_max)`); only change `sk_iters` from 3 to 5. The Sinkhorn iteration count affects convergence tightness; 5 iters is the canonical value for adaptive-ε setups (Cuturi 2013).

Direct effects (DE-1..3 — must be observed for ALIGNED):
- DE-1: per-layer Sinkhorn assignments converge more tightly (lower transport residual after 5 iters vs 3 iters).
- DE-2: warm-start loads iter8 weights for embeddings/codebooks/c_layer_scales. First-step rl ≈ iter8's (~1.28).
- DE-3: per-layer quantization loss (rqvae_loss) stays in iter8's range; only the convergence sharpness changes.

Proxy hypothesis (D, descriptive):
- Stage3 `test_R@10` ≥ 0.0615 (≥ +0.002 over iter8 0.0595) for gap-closing.
- If gap-closing fails, classify as `SID_GEOMETRY_OK_DOWNSTREAM_FLAT` (类 3).

Forbidden directions: changing ε schedule (forbidden, iter9 regressed); reducing iters (reverts to iter4-baseline territory).

Verification plan:
- MVG PASS on a 640-pair batch with iter8 warm-start.
- Stage2: 100k steps, observe DE-1..3.
- Stage3: 150 epochs, beam=20, four-GPU DDP.