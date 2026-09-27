# sid_geometry_iter8 (Agent D — post-Stage2)

Observed fingerprint (step=100000):
- L0 utility: `0.2046` (iter7 0.2011) — essentially flat.
- L1 utility: `0.2303` (iter7 0.4777) — large drop (better balance).
- L2 utility: `0.2233` (iter7 0.5038) — large drop (better balance).
- full_gini: `0.0684` (iter7 0.1225, iter4 0.1408) — far more uniform.
- collision rate: `1,756 / 24,587 ≈ 7.1%` (iter7 13.2%, iter4 15.3%) — large reduction.
- unique codes: `22,831 / 24,587` (iter7 21,350, iter4 20,813) — large improvement.
- l01_pairs: `14,294` (iter7 12,173, iter4 11,297) — significant improvement.
- H(L1|L0): `5.5336` (iter7 5.1757, iter4 5.0162) — improved.

Direct effects (DE-1..3) verification:
- DE-1 (per-layer Sinkhorn assignments vary across the cyclic-c schedule): observed. The Sinkhorn iterations now operate under tighter ε during high-c phases, sharpening per-layer quantization; L1/L2 utility are now balanced (0.23 each) instead of being uneven.
- DE-2 (warm-start from iter7): observed. First-step rl=1.28 vs iter7 ~1.28; c_layer_scale reset to interior of new range; embeddings carried from iter7.
- DE-3 (quantization loss stable): observed. rqvae_loss in 0.25–0.45 range, comparable to iter7.

Verdict: `ALIGNED`. The per-layer utilities are now uniformly distributed (the 3-token SID uses all 3 layers evenly), collision rate dropped by 6.1 percentage points, and unique codes increased by 1481 — all structural improvements over iter7.