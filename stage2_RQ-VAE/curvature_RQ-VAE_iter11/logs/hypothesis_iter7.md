# hypothesis_iter7 (Agent C — hypothesis designer)

Mechanism: widen cyclic curvature range to (0.05, 1.5), period 100k; warm-start encoder/decoder/codebooks from iter4 while resetting `c_layer_scale` to the calibrated interior (0.001, 0.933, 1.0) of the new range; sharpen behavior loss with `BEHAVIOR_TEMPERATURE=0.07` and `BEHAVIOR_LOSS_WEIGHT=0.20`; lower `CURVATURE_REG_WEIGHT` to 0.005.

Direct effects (DE-1..3 — must be observed for ALIGNED):
- DE-1: `c_layer_scale` no longer clamped at iter4's lower bound throughout training; observed c trajectories cover more of [0.001, 1-1e-4] range than iter4.
- DE-2: warm-start loads iter4 weights for embeddings/codebooks without re-randomizing them; first-step reconstruction loss is comparable to iter4's (~1.4–1.7) rather than the cold-start 5–6.
- DE-3: behavior loss converges to a sharper value than iter4's because of lower τ; observed behavior loss in (3, 8) range instead of iter4's (~6.5–7.0).

Proxy hypothesis (D, descriptive):
- Stage3 `test_R@10` ≥ 0.065 (hard target). If DE-1..3 are observed and Stage3 fails, classify as `SID_GEOMETRY_OK_DOWNSTREAM_FLAT` (类 3 failure mode).

Forbidden directions: re-encoding Stage1 embeddings (out of scope); changing Stage3 architecture (out of scope).

Verification plan:
- MVG PASS on a 640-pair batch with iter4-warm-start (re-initializing c_layer_scale) and curriculum step at mid-cycle.
- Stage2: 100k steps, observe DE-1..3 in log tail at step ≈ 50k.
- Stage3: 150 epochs, beam=20, four-GPU DDP.