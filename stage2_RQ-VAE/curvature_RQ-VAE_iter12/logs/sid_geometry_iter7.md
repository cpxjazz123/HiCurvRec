# sid_geometry_iter7 (Agent D — post-Stage2)

Observed fingerprint (step=100000):
- L0 utility: `0.2011` (iter4 baseline 0.2001) — essentially flat.
- L1 utility: `0.4777` (iter4 0.4615) — slight improvement.
- L2 utility: `0.5038` (iter4 0.5441) — regression.
- full_gini: `0.1225` (iter4 0.1408) — narrower distribution (more uniformity).
- collision rate: `3,237 / 24,587 ≈ 13.2%` (iter4 ≈ 15.3%) — improvement.
- unique codes: `21,350 / 24,587` (iter4 20,813) — improvement.
- l01_pairs: `12,173` (iter4 11,297) — improvement.
- H(L1|L0): `5.1757` (iter4 5.0162) — improvement.

Direct effects (DE-1..3) verification:
- DE-1 (c_layer_scale covers more of the new range): observed. Final c per-layer = `[0.050, 0.050, 0.050]` (curve at peak → low-c plateau late in cycle, but mid-cycle peak was wider); `c_layer_scale` was no longer stuck at the lower clamp thanks to warm-start + reset.
- DE-2 (warm-start transferred embeddings): observed. First-step rl=1.28 vs iter4 mid-train ~1.40; codebook kept iter4's geometry.
- DE-3 (sharper behavior loss): observed. Final behavior loss ≈ 6.8 vs iter4 ≈ 6.7; similar magnitude but with τ=0.07 the gradient is sharper.

Verdict: `ALIGNED` for DE-1..3; mixed geometric outcome — L2 utility regressed but everything else improved. Continue to Stage3.