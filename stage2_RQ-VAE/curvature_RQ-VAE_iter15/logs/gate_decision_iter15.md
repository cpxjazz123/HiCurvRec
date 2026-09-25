# gate_decision_iter15

## Decision: **PENDING** (Stage3 test not finished at audit commit)

Stage2 mechanism audit is **complete** (Agent D **ALIGNED**). Hard gate `test_recall@10 > 0.065` awaits `test_final.json` from Stage3 run `Sep-25-2026_22-46-35`.

## Stage2 (complete)

- **Mechanism**: `iter15_fixed_residual_calibrated_curvature` (freeze `c_layer_scale` at residual `u_l=[0.001, 0.932889, 1.0]`, no behavior `b_l`)
- **Training**: `global_step=100000`, ~1184 s wall time
- **SID export**: `item_sids.json` (unique 23095/24587, hitrate@50=0.9968)
- **Curvature snapshots**: `logs/curvature_snapshots.txt` (DE-1/DE-2/DE-3 satisfied)
- **Agent D**: `logs/sid_quality_iter15.json`, `logs/sid_geometry_iter15.md`

## Stage3 (in progress at commit time)

- Run: `results/stage3_T5Train/curvature_RQ-VAE_iter15/logs/Amazon_2023_Instruments/Sep-25-2026_22-46-35`
- Config: 150 epochs, vanilla TIGER / HG-Rec, `item_sids` from iter15
- **Update this file** when `test_final.json` exists (compare vs iter11 **0.0598**, iter14 **0.0594**).

## A/B/C proxy (paper)

| Iter | Mechanism | test_R@10 |
|------|-----------|-----------|
| iter11 | learnable `u_l` | 0.0598 |
| iter14 | + behavior `b_l` | 0.0594 |
| iter15 | frozen residual `u_l` | TBD |
