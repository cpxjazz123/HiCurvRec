# Issue #261 — complete eight-bundle Stage2→Stage3 correlation audit

## Scope and identity

Analysis only. No Stage2/Stage3/TIGER training, no optimizer update, no SID rewrite.
Eight bundles 100–107, one row per Stage2 run and its one Stage3 result. The body
sha256 is `022ad858552a4d067b9a0e15abdff5d8f99e58404ad3d5574efe31cd59c38e9d`;
the frozen metric rules are in `METRIC_CONTRACT.md`.

`INVENTORY.json` records actual file hashes, not just ledger names. The R2
`recall.json` for each number was downloaded and hash-checked; local checkpoint,
SID, archive binding, export manifest, 3000/3000 training stream, Stage3
`n_eval=57,439`, prediction-shard row count and layer-record SID hash all pass.
Candidate source/config hashes pass for all eight. For 100–102 a later public
infra commit changed four historical `infrastructure_sources`; that drift is
reported in `INFRA_FAULTS.md`, while the archived candidate source and checkpoint
remain unchanged. It did not change the mechanism being measured.

## Eight completed arms

| idx | arm | test R@10 | test R@5 | NDCG@10 | L3 collision | distinct L2 | L1 positive hit | L1 ER | quant error | eff. rank z |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | gnn_attention_sectional | 0.049287 | 0.032783 | 0.026926 | 0.0728 | 9,130 | 0.0533 | 6.846 | 0.0081 | 3.01 |
| 101 | three_expert_attention | 0.046432 | 0.031425 | 0.025634 | 0.0716 | 10,386 | 0.0370 | 5.961 | 0.0087 | 3.08 |
| 102 | module_budget_prefix | 0.052734 | 0.034106 | 0.028523 | 0.0657 | 8,559 | 0.0667 | 8.516 | 0.0072 | 5.39 |
| 103 | periodic_budget_prefix | 0.052665 | 0.035028 | 0.028953 | 0.0649 | 8,840 | 0.0643 | 8.536 | 0.0077 | 5.05 |
| 104 | role_conditioned_l1 | 0.049009 | 0.031895 | 0.026459 | 0.0545 | 9,257 | 0.0596 | 7.850 | 0.0065 | 4.42 |
| 105 | role_conditioned_l3 | 0.051516 | 0.034054 | 0.028115 | 0.0647 | 8,781 | 0.0625 | 8.401 | 0.0076 | 5.14 |
| 106 | role_conditioned_l2 | 0.050349 | 0.033270 | 0.027328 | 0.0553 | 9,609 | 0.0600 | 7.102 | 0.0071 | 4.33 |
| 107 | topology_residual_l2 | 0.049600 | 0.033026 | 0.027279 | 0.1005 | 7,714 | 0.0646 | 8.284 | 0.0086 | 5.05 |

No arm reaches the 0.065 adoption line; these are exploratory correlations among
completed runs, not a new model-selection criterion.

## Frozen full-table geometry/reconstruction

All eight were re-evaluated from their own checkpoint/SID with the same independent
three-expert metric and every frozen component triple (28,391,948; no sampling).
The SID is the hard decoded authorititative table; the collision counter is excluded.

| arm | hard macro MAE | hard micro MAE | sign acc | hard recon MSE | L1 code spacing |
|---|---:|---:|---:|---:|---:|
| 100 | 2.117 | 2.344 | 0.340 | 8.84e-5 | 0.193 |
| 101 | 2.088 | 2.360 | 0.363 | 9.86e-5 | 0.223 |
| 102 | 2.582 | 1.882 | 0.344 | 7.78e-5 | 0.122 |
| 103 | 2.552 | 1.911 | 0.341 | 7.88e-5 | 0.132 |
| 104 | 2.563 | 1.939 | 0.340 | 8.36e-5 | 0.139 |
| 105 | 2.553 | 1.906 | 0.338 | 7.94e-5 | 0.132 |
| 106 | 2.559 | 1.925 | 0.335 | 8.65e-5 | 0.138 |
| 107 | 2.541 | 1.900 | 0.347 | 8.02e-5 | 0.134 |

The macro/micro split is not interchangeable: macro gives each centre equal weight,
micro weights every triple. The correlation signs below can differ; no single
"geometry quality" scalar is claimed.

## Main exploratory associations with test R@10 (n=8)

| predictor | Pearson | Spearman | 95% Fisher CI | nominal p |
|---|---:|---:|---|---:|
| validation L1 positive hit | +0.856 | +0.833 | [+0.382,+0.974] | 0.0067 |
| validation L1 hard ER | +0.839 | +0.857 | [+0.329,+0.970] | 0.0092 |
| L2 distinct prefixes | −0.571 | −0.619 | [−0.910,+0.224] | 0.140 |
| L3 collision | −0.193 | −0.119 | [−0.790,+0.592] | 0.647 |
| hard-q curvature micro MAE | −0.724 | −0.810 | [−0.946,−0.040] | 0.042 |
| hard-q reconstruction L2 error | −0.885 | −0.881 | [−0.979,−0.481] | 0.0034 |
| L1 codebook spacing | −0.812 | −0.952 | [−0.965,−0.250] | 0.014 |
| effective rank of z | +0.796 | +0.810 | [+0.208,+0.961] | 0.018 |
| held-out test shared-in jaccard distance Spearman | +0.958 | +0.905 | [+0.782,+0.993] | 0.00017 |
| direct-edge L1 prefix rate | +0.833 | +0.833 | [+0.310,+0.969] | 0.010 |

The preliminary #260 values are replicated in the complete matrix: L1 positive
is **+0.856** (the earlier n=7 +0.686 was not the complete eight-arm value);
L2 distinct is **−0.571**; L3 collision is **−0.193**.

The held-out structural correlation is strong but is not independent of the same
SID construction and has only eight points; it is a diagnostic, not a causal
predictor. Full per-predictor rows, constants/NA, nominal p, leave-one-out, and
exclude-107/exclude-100 variants are in `CORRELATIONS.csv` / `SENSITIVITY.csv`.
L1 positive remains positive when either 107 or 100 is excluded (+0.923 / +0.851);
L2 distinct becomes −0.893 when 107 is excluded, so the L2 relation is sensitive
to the R06b outlier and is not robust by itself.

## Mechanism chain

The observed ordering is consistent with: **GNN/token geometry and gate summary →
hard SID/prefix structure → validation prefix behaviour → Stage3 recall**, but
the eight points cannot separate architecture, capacity and training variance.
The largest SID-side association is at L1, not L2/L3. Deep-prefix metrics are
weak or unstable. The full-table reconstruction/curvature associations are confounded with arm
architecture and capacity: arm 100 has relatively high hard reconstruction error
and low R@10, arm 102 has lower error and higher R@10, but arm 101 has the worst
error with a low R@10. Eight runs cannot identify a causal direction; do not read
these correlations as "better reconstruction causes worse recommendation".

## R06b counterexample (107)

The #260 R06b work is the concrete negative case. It changed 52.19% of L2 codes
under a same-checkpoint token switch; the per-role L2 change was 0.413–0.575.
Its final collision was 0.1005, its SID has 7,714 distinct L2 prefixes, and its
Stage3 test R@10 was 0.049600, below R05 (0.050349) and below 0.065. The code
change was real but did not align with next-item behaviour. Competing
explanations remain: the token direction is not behaviour-relevant; the role
table is too coarse; Stage3 variance; or the measured prefixes are not the
causal bottleneck. The data can reject "any large L2 code movement raises recall";
it cannot select among those causes.

## Missing measurements and limits

`MISSING.csv` lists every unavailable item with the concrete evidence and impact:
Stage3 validation-only metrics (no artifact), historical per-module gradients,
per-level commitment/codebook loss decomposition, top1/top2 margin vectors,
predicted-vs-target curvature Pearson/Spearman/RMSE, role/degree-stratified
curvature, and common switch-off diagnostics for all eight. These are not filled
with zero or imputed. The three historical infra receipt drifts are separately
reported in `INFRA_FAULTS.md`.

## Evidence map

- `INVENTORY.json` — actual identity/hash/row/epoch checks.
- `METRIC_CONTRACT.md` — frozen formulas and scope.
- `METRICS.csv` / `METRICS.json` — wide per-arm matrix.
- `FULL_METRICS.csv` / `.json` — compact full-table geometry/reconstruction.
- `CORRELATIONS.csv` / `.json`, `SENSITIVITY.csv` — associations and sensitivity.
- `HEATMAP.svg`, `SCATTER_L1.svg` — selected visual checks.
- `TOKEN_GATE.json` — read-only GNN-token/gate statistics.
- `MISSING.csv`, `INFRA_FAULTS.md` — unavailable evidence and drift.
