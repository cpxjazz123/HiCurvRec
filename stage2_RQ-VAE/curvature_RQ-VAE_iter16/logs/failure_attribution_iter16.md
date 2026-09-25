# failure_attribution_iter16 (Agent F — failure attribution)

Observed Stage3 metrics:
- test_R@5: 0.03794 (vs iter11 0.03889 → -0.0010; vs iter15 0.03787 → +0.0001)
- test_R@10: **0.05688** (vs iter11 0.05977 → **-0.00289**; vs iter15 0.05824 → -0.00136; vs iter8 0.05947 → -0.00259)
- test_NDCG@5: 0.02551 (vs iter11 0.02564 → -0.0001)
- test_NDCG@10: 0.03159 (vs iter11 0.03236 → -0.00077)

Hard target: `test_R@10 > 0.065`. Result: **NOT MET** (gap = -0.00812).

Five-class evaluation:
1. **Implementation correct (MVG PASS)**: yes — `closed_form_c_l = [0.663, 0.435, 0.433]`; `c_live` matches at steps 0/10k/50k/100k.
2. **Mechanism activated (DE-1..3 observed)**: yes — fixed `_fixed_c` buffers, no cyclic drift, `curv_reg=0`, c₀ > c₁ > c₂.
3. **Geometry alignment (Agent D ALIGNED)**: yes — unique 22827/24587, hitrate@50=0.9963, H(L1|L0)=5.4549, full_gini=0.0681.
4. **Pipeline consistency**: yes — same Stage3 HG-Rec trainer, 4-token SID extension wired into Stage2 `curvature_RQ-VAE.py` end-of-run (no separate post-script), `item_sids.json` matches SID npy.

**Label: `TRUE_MECHANISM_FAIL`** — closed-form fixed curvature was intended to remove cyclic/learnable confound, but Stage3 underperformed **all** previous iters. Likely root cause: the formula's c₀=0.663 is 13× larger than the mid-cycle cyclic c₀~0.05 that the assignment step benefits from; fixed curvature freezes the high-c regime and prevents the c(t) range that the Stage-3 HG-Rec T5 was implicitly adapted to.

Append to `failed_mechanism_ledger.md`:
- iter16 | closed_form_layer_curvature | 类 3 (SID_GEOMETRY_OK_DOWNSTREAM_REGRESSED) | unique 22827/24587, hitrate@50=0.9963, H(L1|L0)=5.4549, c_l=[0.663,0.435,0.433], warm-start iter8 | test_R@10=0.0569 (-0.0029 vs iter11, -0.0014 vs iter15) | 累计 1
