# iteration_bridge (Agent G — root cause + gap analyst)

Inputs:
- Stage2 SID: full_gini 0.0618 (vs 0.0684 iter8, marginally better), collision 6.5% (vs 7.1%), unique 22997 (vs 22831).
- Stage3: test_R@10=0.0568 (vs 0.0595 iter8, REGRESSION of -0.0027).

Mechanism execution: PASS (DE-1/2/3 all observed). But Stage3 regressed despite marginally improved SID stats.

Dominant bottleneck: the power-law ε schedule over-amplified cycle contrast. iter8's linear ε schedule was the local optimum — refining toward power-law 0.5 shifts more discrimination into the deeper L2 layer (L2 utility jumped from 0.22 → 0.285), but the downstream T5 cannot capitalize on this concentration. The downstream training prefers balanced per-layer utility (iter8: [0.20, 0.23, 0.22]) over concentrated (iter9: [0.11, 0.14, 0.285]).

Forbidden next directions:
- Any further ε schedule refinement (linear was the optimum).
- Pure curvature-schedule changes (iter4-iter7 territory).

Next iteration objective (must be falsifiable):
- iter10 = revert to iter8's mechanism family (linear ε) but attack a different quantization aspect: tighten the Sinkhorn iterations count (iter8 used 3 iters; try 5 iters at high-c only) OR add a small c-dependent temperature on the quantization softmax. The goal is to keep iter8's balanced per-layer utility while adding a small gain.
- Target: ≥ +0.002 over iter8 (test_R@10 ≥ 0.0615).

If iter10 doesn't break iter8, the alternative is to combine iter8's Sinkhorn mechanism with a different Stage-2 family entirely (e.g., codebook-temperature curriculum at the high-c phase).