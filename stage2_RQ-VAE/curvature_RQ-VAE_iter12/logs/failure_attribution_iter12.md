# failure_attribution_iter12 (Agent F — failure attribution)

Observed Stage3 metrics:
- test_R@5: 0.03907 (vs iter11 0.03889 → +0.0002)
- test_R@10: 0.05851 (vs iter11 0.05977 → -0.0013)
- test_NDCG@5: 0.02613 (vs iter11 0.02564 → +0.0005)
- test_NDCG@10: 0.03239 (vs iter11 0.03236 → +0.0000)

Hard target: `test_R@10 > 0.065`. Result: NOT MET (gap = 0.0065).

Five-class evaluation:
1. Implementation correct (MVG PASS): yes.
2. Mechanism activated (DE-1..3 observed): yes — 2× curvature cycles within 100k steps; per-layer balance preserved.
3. Geometry alignment (Agent D ALIGNED): yes — full_gini 0.0629 (vs 0.0641 iter11), per_layer [0.19, 0.20, 0.22] preserved.
4. Pipeline consistency: yes.

Label: `TRUE_MECHANISM_FAIL`. iter12's doubling of cycle frequency did not break the Stage-3 ceiling. iter8 is still the best Stage-2 mechanism.

Append to failed_mechanism_ledger.md:
- iter12 | short_period_faster_cycle | 类 3 (cycle period 100k→50k preserved iter11 balance but Stage3 regressed) | full_gini 0.0629, collision 6.5%, unique 22982/24587, H(L1|L0)=5.5935, per_layer [0.1877, 0.2040, 0.2220], period 50k (vs 100k in iter11), warm-started from iter8 | test_R@10=0.0585 (-0.0013 vs iter11) | 累计 1