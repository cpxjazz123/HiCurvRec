# gate_decision_iter15

## Decision: **NO-GO** (hard target not met)

- **Mechanism**: `iter15_fixed_residual_calibrated_curvature` (freeze `c_layer_scale` at residual `u_l=[0.001, 0.932889, 1.0]`, no behavior `b_l`)
- **Stage2**: `global_step=100000`; SID export `item_sids.json` (unique 23095/24587, hitrate@50=0.9968); Agent D **ALIGNED**
- **Stage3 run**: `Sep-25-2026_22-46-35`, 150 epochs train + beam test complete

## Stage3 test (n_eval=57439)

| Metric | Value |
|--------|-------|
| test_recall@5 | 0.03787 |
| test_recall@10 | **0.05824** |
| test_ndcg@5 | 0.02538 |
| test_ndcg@10 | 0.03194 |

**Hard gate**: `test_recall@10 > 0.065` → **FAIL** (0.05824)

## A/B/C proxy (paper)

| Iter | Mechanism | test_R@10 | Δ vs iter15 |
|------|-----------|-----------|-------------|
| iter11 | learnable `u_l` | 0.0598 | +0.0016 |
| iter14 | + behavior `b_l` | 0.0594 | +0.0012 |
| **iter15** | frozen residual `u_l` | **0.0582** | — |

vs iter8 best reference (0.0595): **-0.0013**

## Notes

- Mechanism hypotheses DE-1/DE-2/DE-3 satisfied (Stage2), but downstream **regressed** vs iter11/iter14 and below iter8.
- Freezing `u_l` removed learnable layer-scale capacity without closing the 0.065 gap.
- Do not PROMOTE; keep **iter11** (peak 0.0598) / **iter8** (0.0595) as references.
- Failure attribution: `TRUE_MECHANISM_FAIL` — see `failure_attribution_iter15.md`.
