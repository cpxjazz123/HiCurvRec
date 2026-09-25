# failure_attribution_iter15 (Agent F — failure attribution)

Observed Stage3 metrics:
- test_R@5: 0.03787 (vs iter11 0.03889 → -0.0010; vs iter14 0.03936 → -0.0015)
- test_R@10: 0.05824 (vs iter11 0.05977 → -0.0015; vs iter14 0.05944 → -0.0012; vs iter8 0.05947 → -0.0012)
- test_NDCG@5: 0.02538 (vs iter11 0.02564 → -0.0003)
- test_NDCG@10: 0.03194 (vs iter11 0.03236 → -0.0004; vs iter14 0.03291 → -0.0010)

Hard target: `test_R@10 > 0.065`. Result: NOT MET (gap = 0.0068).

Five-class evaluation:
1. Implementation correct (MVG PASS): yes — warm-start iter8, frozen `c_layer_scale`, export wiring PASS.
2. Mechanism activated (DE-1..3 observed): yes — `u_learned` fixed at residual calibration; `c0<c1<c2` at mid-cycle; `curv_reg=0`.
3. Geometry alignment (Agent D ALIGNED): yes — unique 23095/24587 (slightly above iter14), hitrate@50=0.9968, H(L1|L0)=5.5962.
4. Pipeline consistency: yes — same Stage3 HG-Rec trainer; `item_sids` from iter15 export.

Label: `TRUE_MECHANISM_FAIL` — freezing residual `u_l` was intended to remove confounding from learnable layer-scale drift, but Stage3 **underperformed** both learnable-`u_l` (iter11) and behavior-branching (iter14). The paper A/B/C proxy does not support frozen residual as the better inductive bias at the Stage-1 ceiling.

Append to failed_mechanism_ledger.md:
- iter15 | fixed_residual_calibrated_u_l | 类 3 (SID_GEOMETRY_OK_DOWNSTREAM_REGRESSED) | unique 23095/24587, hitrate@50=0.9968, H(L1|L0)=5.5962, freeze u_l=[0.001,0.933,1.0], warm-start iter8 | test_R@10=0.0582 (-0.0015 vs iter11, -0.0012 vs iter14) | 累计 1
