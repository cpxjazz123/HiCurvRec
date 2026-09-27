# failure_attribution_iter10 (Agent F — failure attribution)

Observed Stage3 metrics:
- test_R@5: 0.03905 (vs iter8 0.04001 → -0.0010)
- test_R@10: 0.05940 (vs iter8 0.05947 → -0.0001)
- test_NDCG@5: 0.02621 (vs iter8 0.02676 → -0.0005)
- test_NDCG@10: 0.03277 (vs iter8 0.03302 → -0.0003)

Hard target: `test_R@10 > 0.065`. Result: NOT MET (gap = 0.0056).

Five-class evaluation:
1. Implementation correct (MVG PASS): yes.
2. Mechanism activated (DE-1..3 observed): yes — DE-1 per-layer L2 utility jumped to 0.32 (vs iter8 0.22), confirming tighter Sinkhorn convergence; DE-2 warm-start loaded iter8; DE-3 vl stable.
3. Geometry alignment (Agent D ALIGNED): yes — but per-layer utility has shifted to L2 like iter9 (though less extreme); full_gini similar to iter8.
4. Pipeline consistency: yes.

Label: `TRUE_MECHANISM_FAIL` — 5 Sinkhorn iters provided essentially no downstream improvement (-0.0001). The Sinkhorn 3→5 iters change is on the same edge as iter9's regression: any per-layer sharpening shifts L2 to dominate and the downstream T5 prefers balance.

Append to failed_mechanism_ledger.md:
- iter10 | sinkhorn_iters_5 | 类 3 (Sinkhorn iterations tightening shifts L2 utility to dominate; same edge as iter9's regression) | full_gini 0.0665, collision 6.9%, unique 22879/24587, H(L1|L0)=5.4893, per_layer [0.1739, 0.2460, 0.3233], sk_iters 5 (vs 3 in iter8), ε linear (iter8), warm-started from iter8 | test_R@10=0.0594 (-0.0001 vs iter8) | 累计 1