# iteration_bridge (Agent G — root cause + gap analyst)

Inputs:
- Stage2 SID fingerprint: full_gini 0.0684 (vs 0.1225 iter7, vs 0.1408 iter4), collision 7.1% (vs 13.2% / 15.3%), unique 22831/24587 (vs 21350 / 20813), H(L1|L0)=5.5336 (vs 5.1757 / 5.0162), per_layer balanced.
- Stage3: test_R@10=0.0595 (vs 0.0565 iter7, +0.0030); test_R@5=0.0400 (+0.0019); NDCG@10=0.0330 (+0.0019).

Mechanism execution: PASS (DE-1/2/3 all observed).

ΔU_L0 ≈ +0.004, ΔH(L1|L0) ≈ +0.36, Δcollision ≈ -6.1%, Δoracle ≈ +1481 unique, Δtest_R@10 ≈ +0.0030.

Dominant bottleneck: iter8 successfully moved the SID geometry (uniform per-layer utilization, halved collision) AND broke the previous +0.0003 stage-3 ceiling, gaining +0.0030. The remaining gap to 0.065 is 0.0055. The Sinkhorn c-dependent ε mechanism is the right direction — continue refining it.

Forbidden next directions:
- Reverting to iter4/iter5/iter6/iter7-style curvature-schedule changes — those have been exhaustively tested.
- Going back to a constant ε (revert iter8's mechanism).

Next iteration objective (must be falsifiable):
- iter9 = iter8 + tightening the ε scale (e.g., ε_min clamp lower; allow ε to be more aggressive at high c).
- Target: ≥ +0.002 over iter8 (test_R@10 ≥ 0.0615).

If iter9 doesn't deliver, the alternative is combining iter8's Sinkhorn mechanism with a learned per-codebook c (mechanism pool #5 #8 product manifold) — structural change at the codebook level rather than just the assignment step.