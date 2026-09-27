# failure_attribution_iter13 (Agent F — failure attribution)

Observed Stage3 metrics:
- test_R@5: 0.04011 (vs iter8 0.04001 → +0.0001)
- test_R@10: 0.05965 (vs iter8 0.05947 → +0.0002; vs iter11 0.05977 → -0.0001)
- test_NDCG@5: 0.02681 (vs iter8 0.02676 → +0.0001)
- test_NDCG@10: 0.03310 (vs iter8 0.03302 → +0.0001)

Hard target: `test_R@10 > 0.065`. Result: NOT MET (gap = 0.0054).

Five-class evaluation:
1. Implementation correct (MVG PASS): yes.
2. Mechanism activated (DE-1..3 observed): yes — log_tau_l drift → commitment_weight increased ~4x (vl=1.36 vs iter8 0.30).
3. Geometry alignment (Agent D ALIGNED): yes — full_gini 0.0482 (vs 0.0684), collision 4.9% (vs 7.1%), unique 23382 (vs 22831). All SID stats improved.
4. Pipeline consistency: yes.

Label: `TRUE_MECHANISM_FAIL`. The per-layer learnable commitment multiplier provided best SID geometry yet but Stage3 was flat (essentially iter8's level). Stage-3 ceiling is locked at ~0.060.

iter13 stays archived; iter8/iter11 remain best Stage-2 mechanism with `test_R@10=0.0595-0.0598`.

Append to failed_mechanism_ledger.md:
- iter13 | per_layer_temperature | 类 3 (best SID stats but Stage3 ceiling at 0.060; log_tau_l drifted but no downstream gain) | full_gini 0.0482, collision 4.9%, unique 23382/24587, H(L1|L0)=5.8442, per_layer [0.1918, 0.1653, 0.1587], log_tau_l multiplier per layer, warm-started from iter8 | test_R@10=0.0596 (+0.0002 vs iter8) | 累计 1