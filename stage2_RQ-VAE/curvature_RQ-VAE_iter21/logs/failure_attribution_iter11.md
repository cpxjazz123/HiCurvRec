# failure_attribution_iter11 (Agent F — failure attribution)

Observed Stage3 metrics:
- test_R@5: 0.03889 (vs iter8 0.04001 → -0.0011)
- test_R@10: 0.05977 (vs iter8 0.05947 → +0.0003)
- test_NDCG@5: 0.02564 (vs iter8 0.02676 → -0.0011)
- test_NDCG@10: 0.03236 (vs iter8 0.03302 → -0.0007)

Hard target: `test_R@10 > 0.065`. Result: NOT MET (gap = 0.0052).

Five-class evaluation:
1. Implementation correct (MVG PASS): yes.
2. Mechanism activated (DE-1..3 observed): yes — commitment weight varies mildly with c; per-layer utilization preserved within 0.02 of iter8.
3. Geometry alignment (Agent D ALIGNED): yes — full_gini 0.0641 (vs 0.0684), per_layer [0.19, 0.21, 0.22] preserved, collision 6.7%, unique 22948.
4. Pipeline consistency: yes.

Label: `TRUE_MECHANISM_FAIL`. iter11's c-modulated commitment weight preserved iter8's per-layer balance and improved SID stats slightly, but Stage3 gave only +0.0003 vs iter8. The 0.0052 gap remains.

Append to failed_mechanism_ledger.md:
- iter11 | curvature_scaled_commitment | 类 3 (c-modulated commitment preserved iter8 balance but only +0.0003 gain) | full_gini 0.0641, collision 6.7%, unique 22948/24587, H(L1|L0)=5.5844, per_layer [0.1922, 0.2108, 0.2240] (preserved), commitment weight α=0.25 c-modulation, warm-started from iter8 | test_R@10=0.0598 (+0.0003 vs iter8) | 累计 1