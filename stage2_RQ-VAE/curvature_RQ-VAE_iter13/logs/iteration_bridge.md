# iteration_bridge (Agent G — root cause + gap analyst)

Inputs:
- Stage2 SID: full_gini 0.0482 (iter8 0.0684, best ever), collision 4.9% (best), unique 23382 (best), H(L1|L0)=5.8442 (best), per_layer [0.19, 0.17, 0.16] (L2 dropped to 0.16).
- Stage3: test_R@10=0.0596 (iter8 0.0595, +0.0001; iter4 baseline 0.0562, +0.0034).

Mechanism execution: PASS. log_tau_l drift led to ~4x commitment_weight boost (vl=1.36 vs iter8 0.30). SID geometry improved dramatically.

Dominant bottleneck: even with best-ever SID stats, Stage3 is locked at ~0.060. The HG-Rec T5 (Stage-3) has a fixed capacity ceiling. 13 distinct Stage-2 mechanisms have been tried.

Forbidden next directions:
- Continuing Stage-2 mechanism variations (exhausted, ceiling confirmed).
- Modifying Stage-3 (forbidden by project rules).

Recommended next action (for the user):
- Accept iter8 (test_R@10=0.0595) as the best Stage-2 mechanism.
- The 0.0054 gap to 0.065 cannot be closed without modifying Stage-3.

If the user insists on more Stage-2 attempts:
- iter14 could try a structural orthogonal mechanism (e.g., product manifold M = H^{c1} × H^{c2} × S — per-layer different manifolds). But this would require major refactoring of the encoder/decoder (since Poincaré distance is replaced by per-layer distance), which would break iter8's warm-start.