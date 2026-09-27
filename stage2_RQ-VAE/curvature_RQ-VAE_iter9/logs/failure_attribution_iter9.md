# failure_attribution_iter9 (Agent F — failure attribution)

Observed Stage3 metrics:
- test_R@5: 0.03776 (vs iter8 0.04001 → -0.0023)
- test_R@10: 0.05677 (vs iter8 0.05947 → -0.0027)
- test_NDCG@5: 0.02499 (vs iter8 0.02676 → -0.0018)
- test_NDCG@10: 0.03112 (vs iter8 0.03302 → -0.0019)

Hard target: `test_R@10 > 0.065`. Result: NOT MET (gap = 0.0082).

Five-class evaluation:
1. Implementation correct (MVG PASS): yes.
2. Mechanism activated (DE-1..3 observed): yes — power-law ε applied; L2 utility jumped to 0.285; Stage2 SID stats marginal improvement.
3. Geometry alignment (Agent D ALIGNED): yes — full_gini 0.0618 (vs iter8 0.0684), collision 6.5% (vs 7.1%), unique 22997 (vs 22831). All marginally better than iter8 in pure SID terms.
4. Pipeline consistency: yes.

Label: `TRUE_MECHANISM_FAIL`. The power-law schedule over-amplified cycle contrast in a way that hurt downstream discrimination despite improving raw SID stats. iter8's linear ε schedule was the better local optimum.

Append to failed_mechanism_ledger.md:
- iter9 | power_law_sinkhorn | 类 3 (different mechanism refinement within same family) | full_gini 0.0618, collision 6.5%, unique 22997/24587, H(L1|L0)=5.5042, per_layer [0.1120, 0.1420, 0.2850], ε ∝ (c/c_max)**0.5, warm-started from iter8 | test_R@10=0.0568 (-0.0027 vs iter8) | 累计 1