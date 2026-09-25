# gate_decision_iter14

## Decision: **NO-GO** (hard target not met)

- **Mechanism**: `iter14_soft_behavior_branching_curvature` (soft behavior-calibrated per-layer curvature multiplier, fixed alpha=0.1)
- **Stage2**: `global_step=100000`; SID export `item_sids.json` (unique 23003/24587, hitrate@50=0.9968)
- **Stage3 run**: `Sep-25-2026_21-07-42`, 150 epochs, beam eval complete

## Stage3 test (n_eval=57439)

| Metric | Value |
|--------|-------|
| test_recall@5 | 0.03936 |
| test_recall@10 | **0.05944** |
| test_ndcg@5 | 0.02643 |
| test_ndcg@10 | 0.03291 |

**Hard gate**: `test_recall@10 > 0.065` → **FAIL** (0.05944)

## Notes

- vs iter13 peak: R@10 ≈ flat; R@5 / NDCG@10 slightly below iter13.
- Do not PROMOTE; keep iter8/iter13 as reference baselines for next direction.

- Audit commit: `438b50a276cd0a4133049cb562e9f00d57b48d9f`
