# iteration_bridge (Agent G — root cause + gap analyst)

Inputs:
- Stage2 SID: full_gini 0.0641 (iter8 0.0684, better), per_layer [0.19, 0.21, 0.22] (preserved iter8 balance), collision 6.7%, unique 22948, l01_pairs 14749, H(L1|L0)=5.5844.
- Stage3: test_R@10=0.0598 (iter8 0.0595, +0.0003).

Mechanism execution: PASS. Per-layer balance preserved; SID stats slightly improved; Stage3 only +0.0003.

Dominant bottleneck: Stage-1 ceiling is structural at ~0.060. iter11's SID improvement didn't translate downstream — the HG-Rec T5 is locked onto iter8's effective capacity. The 0.0052 gap to 0.065 is downstream.

Forbidden next directions:
- Any Stage-2 mechanism that doesn't directly modulate curvature (out of innovation focus).
- Re-iterating within the same mechanism family (exhausted: ε schedule, iters, commitment weight).

Next iteration objective (final attempt, iter12):
- iter12 = iter8 + iter11 combined, but pushing the c-modulation harder on a different aspect: increase the cyclic-c period back to the original 50k (vs iter7's 100k), keeping all iter8 + iter11 settings otherwise. Rationale: iter8 used 100k period, iter11 inherits this. Shorter period = c cycle more frequently within the 100k step budget = more high-c phases per item = more commitment-pull signal. Should preserve per-layer balance while doubling the curvature cycles.

If iter12 fails, **HARD STOP** with NO-GO summary (the 12-iter cap is reached).