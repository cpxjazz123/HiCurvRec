# Does the existing SID partition already follow the AF3 preference?

Pre-training diagnostic for the `curvature_prefix_ranking` arm
(`stage2_RQ-VAE/TIGER_curvature-prefix-ranking`). On the frozen TRAIN behaviour
graph, every anchor that has both a cohesive (high-AF3) and a bridge (low-AF3)
neighbour is classified by whether each neighbour shares the anchor's **hard**
prefix at L1/L2/L3, for all 108 recorded SID arms. No model, no training and no
Stage3 outcome is read.

`measure_pressure.py` -> `pressure_by_level.csv`, `pressure.json`.

## Result (108 arms)

| level | cohesive share | bridge share | difference | arms with a positive difference |
|---|---|---|---|---|
| L1 | 16.0% | 7.2% | **+8.8 pp** | **98.1%** |
| L2 | 3.04% | 1.58% | +1.46 pp | 77.8% |
| L3 | 1.10% | 0.49% | +0.61 pp | 74.1% |

The existing SID partitions already rank cohesive behaviour neighbours above
bridges, and they do so almost uniformly at the coarse level. The mechanism's
ranking target is therefore consistent with the learned hierarchy rather than
opposed to it; the headroom and the expected effect are concentrated at L1/L2,
while L3 shares almost no prefix for either class (the fine level distinguishes
items, as designed).

This is a diagnostic, not an admission gate and not evidence that the trained
arm will improve Stage3 Recall.
