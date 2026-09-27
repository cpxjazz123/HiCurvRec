# iteration_bridge (Agent G — root cause + gap analyst)

Inputs:
- Stage2 SID: full_gini 0.0665 (iter8 0.0684), collision 6.9% (vs 7.1%), unique 22879 (vs 22831), per_layer [0.17, 0.25, 0.32].
- Stage3: test_R@10=0.0594 (vs iter8 0.0595, -0.0001; vs iter4 0.0562, +0.0032).

Mechanism execution: PASS. Sinkhorn 5 iters activated; per-layer L2 utility shifted to dominate.

Dominant bottleneck: both iter9 (power-law ε) and iter10 (5 iters) shift per-layer utility toward L2; the downstream T5 doesn't pick up the gain. iter8's balanced [0.20, 0.23, 0.22] utilization is the local optimum for Stage-3. The 0.0056 gap to 0.065 is now likely structural: Stage-2 SID quality is no longer the bottleneck, but Stage-3 architecture (HG-Rec T5) is read-only and has a fixed capacity ceiling.

Forbidden next directions:
- Any change that shifts per-layer utilization away from [0.20, 0.23, 0.22] (proven local optimum).
- Any further ε refinement (linear was the optimum).
- Any Sinkhorn iteration count change (3 was the optimum).
- Pure curvature-schedule changes (iter4-iter7 territory, all < iter8).

Next iteration objective (must be falsifiable):
- iter11 = orthogonal mechanism that *preserves* iter8's balanced per-layer utility while adding a different kind of gain. Candidates:
  - Add a small temperature-scaled softmax on the reconstruction loss (annealed by c).
  - Add a stop-gradient / EMA on the codebook (VQ-VAE-2 style codebook update).
  - Add per-item adaptive ε based on item density (different from c-dependent).

If iter11 doesn't break iter8, the Stage-1 ceiling at ~0.060 is structural — only Stage-3 changes (forbidden) could break it.

Iteration count: 10 attempts done (iter4 through iter10), 2 remaining in the 12-iter cap. Need to be careful.