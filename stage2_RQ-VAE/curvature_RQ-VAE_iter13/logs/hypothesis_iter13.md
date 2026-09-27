# hypothesis_iter13 (Agent C — hypothesis designer)

Mechanism: keep iter8's assignment step (linear Sinkhorn ε ∝ c/c_max); add a learnable per-layer `log_tau_l` parameter initialized to 0 (tau_l=1 identity). The effective_eps becomes `sk_eps * c_scale / tau_l`. The optimizer can drive per-layer tau_l to values other than 1, sharpening or softening per-layer assignments.

Direct effects (DE-1..3 — must be observed for ALIGNED):
- DE-1: per-layer log_tau_l diverges from 0 during training (gradient flows).
- DE-2: warm-start loads iter8 weights for embeddings/codebooks/c_layer_scales (log_tau_l reset to 0). First-step rl ≈ iter8's.
- DE-3: per-layer utilization stays in iter8's range [0.20, 0.23, 0.22] (the assignment step is iter8's at start; per-layer temperature can drift this, but optimizer needs evidence to change).

Proxy hypothesis (D, descriptive):
- Stage3 `test_R@10` ≥ 0.0615 (≥ +0.002 over iter8 0.0595) for gap-closing.
- If gap-closing fails, classify as `SID_GEOMETRY_OK_DOWNSTREAM_FLAT` (类 3).

Forbidden directions: changing ε schedule (forbidden); changing Sinkhorn iterations (forbidden); changing per_layer balance deliberately (forbidden).

Verification plan:
- MVG PASS on a 640-pair batch with iter8 warm-start.
- Stage2: 100k steps, observe DE-1..3 (log_tau_l drift).
- Stage3: 150 epochs, beam=20, four-GPU DDP.