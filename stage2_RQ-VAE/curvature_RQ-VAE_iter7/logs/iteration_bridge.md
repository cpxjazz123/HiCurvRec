# iteration_bridge (Agent G — root cause + gap analyst)

Inputs:
- Stage2 SID fingerprint: full_gini 0.1225 (-0.018 vs iter4), collision 13.2% (-2.1%), unique 21350/24587 (+537), H(L1|L0)=5.1757 (+0.16).
- Stage3: test_R@10=0.0565 (+0.0003 vs iter4), test_R@5=0.0381 (+0.0003), NDCG@10=0.0311 (-0.0001).

Mechanism execution: PASS (DE-1/2/3 all observed in Stage2 logs; MVG PASS; warm-start loaded; c_layer_scale reset to interior of new range).

ΔU_L0 ≈ +0.001, ΔH(L1|L0) ≈ +0.16, ΔH(L2|L0) ≈ -0.04 (computed from per_layer utility, approximate), Δcollision ≈ -2.1%, Δoracle ≈ +537 unique, Δtest_R@10 ≈ +0.0003.

Dominant bottleneck (single sentence): the Stage-3 HG-Rec T5 is locked onto the iter4 SID geometry fingerprint; widening the curvature range while warm-starting embeddings produced measurably improved SID statistics (less collision, more unique codes, sharper entropy) but Stage3 only picks up +0.0003 of recall benefit, indicating the bottleneck is downstream capacity to discriminate, not Stage-2 SID quality.

Forbidden next directions:
- re-iterating within the Stage-1 embedding space (any variant of iter4/iter5/iter6/iter7 curvature changes) — the Stage3 ceiling has been hit five times.
- changing curvature c range again — directionless.

Next iteration objective (must be falsifiable):
- Attack Stage3 capacity directly (not Stage-2 SID quality). Forbid any edit to stage2_RQ-VAE in the next iteration; instead, modify only the Stage-3 trainer config (note: project rules forbid trainer edits — escalate as a structural exception if pursued).

Alternatively (preferred for staying within project rules): continue Stage-2 exploration with a qualitatively different mechanism family — manifold replacement (Poincaré → Lorentz), per-codebook adaptive ε, or Riemannian Adam — to attack the geometry representation, not the schedule.

Expected next-iter metrics: ≥ +0.005 test_R@10 (vs iter7 baseline 0.0565) is the falsifiable target. Anything within +0.001 is a no-op.