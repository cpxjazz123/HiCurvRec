# Issue #270 run inventory

Generated from `RUN_INVENTORY.json` and `COHORT.json`; every number below is a
re-read of an archived artefact, no imputation.

## Coverage

- Stage3 runs with `test_final.json`: **204**.
- R2 ledger arms with a four-piece entry: **100**.
- Usable geometry cohort (exact SID + checkpoint + 57,439 predictions): **42**.
  - record-verified immutable-run bindings: **12**;
  - legacy arm-directory bindings (no SID/checkpoint hash in the Stage3 record): **30**.
- Duplicate SID groups: **0**.
- Protocol proxy (rank-file count): single-card **3**, four-card **39**.

Families:
- `af3_curvature`: 1
- `af3_geometry`: 1
- `alignment`: 1
- `attention_geometry`: 1
- `baseline`: 1
- `behavior_contrast`: 1
- `budget`: 2
- `control`: 4
- `curvature_budget`: 1
- `curvature_calibrated`: 1
- `curvature_point`: 1
- `curvature_rank`: 1
- `curve_contrast`: 2
- `gnn`: 2
- `graphlet_closure`: 4
- `graphlet_codebook`: 2
- `graphlet_role`: 5
- `i234`: 5
- `other`: 1
- `role_condition`: 3
- `semantic`: 1
- `topology_residual`: 1

## Frozen TIGER baseline and reruns

| run | R@10 |
|---|---|
| `Nov-01-2026_10-15-01` | 0.040199167813 |
| `Nov-12-2026_21-04-04` | 0.055450129703 |
| `Nov-23-2026_10-18-36` | 0.053482825258 |
| `Nov-23-2026_10-57-48` | 0.055206392869 |
| `Oct-26-2026_06-26-20` | 0.055537178572 |

Range: **0.040199167813 – 0.055537178572**. The canonical latest run is
`Nov-23-2026_10-57-48` and is used as the pairing baseline; the spread is reported
so no single-run difference is interpreted as significant on its own.

## Usable cohort

| arm | family | records | binding | SID sha256 | checkpoint sha256 | R@10 | predictions |
|---|---|---|---|---|---|---|---|
| `TIGER_RQ-VAE` | baseline | legacy | arm_dir_legacy | `025ad8bd5809ce62` | `af8efc29a8bcbbe5` | 0.055206392869 | 57439 |
| `TIGER_Uniform_KNN_RQ` | control | legacy | arm_dir_legacy | `13decf4263fd64c8` | `7aaf896178ed3be7` | 0.055293441738 | 57439 |
| `TIGER_Uniform_KNN_control` | control | legacy | arm_dir_legacy | `b49c2b484065ea09` | `094e38068ae533eb` | 0.055763505632 | 57439 |
| `TIGER_af3_curvature_drive_rq` | af3_curvature | legacy | arm_dir_legacy | `e911c9fc93843068` | `474f70657c590ee1` | 0.054805968070 | 57439 |
| `TIGER_af3_graphgeo_rq` | af3_geometry | legacy | arm_dir_legacy | `e8f4486f3343d4d6` | `f73cce9c67a80f32` | 0.054579641010 | 57439 |
| `TIGER_baseline_recal_rq` | control | legacy | arm_dir_legacy | `06be252ded1e4d3c` | `a5e3d2b302902fc2` | 0.053552464353 | 57439 |
| `TIGER_behavior_edge_contrastive_rq` | behavior_contrast | legacy | arm_dir_legacy | `de83e2b29ff6865c` | `9cccd7757c88104f` | 0.056250979300 | 57439 |
| `TIGER_calibrated_kappa_rq` | curvature_calibrated | verified | immutable_run | `a89110b86168eb09` | `896e69b7bfc701a4` | 0.045230592455 | 57439 |
| `TIGER_chain_align_rq` | alignment | legacy | arm_dir_legacy | `9020ea99dc218cfe` | `fd814660b0868a4d` | 0.056111701109 | 57439 |
| `TIGER_cos_contrastive_rq` | curve_contrast | legacy | arm_dir_legacy | `e2f16d7306ad8635` | `af0e5d6a374f92b6` | 0.054562231237 | 57439 |
| `TIGER_curve_contrastive_rq` | curve_contrast | legacy | arm_dir_legacy | `588ecd46c2675066` | `8ec8d843283f7af5` | 0.053500235032 | 57439 |
| `TIGER_degree_temp_rq` | control | legacy | arm_dir_legacy | `ae67fbc0729e238f` | `645317d6707339bd` | 0.052490468149 | 57439 |
| `TIGER_drq_D1` | other | legacy | arm_dir_legacy | `a47b78967d0b8580` | `b12374b7caeada13` | 0.044377513536 | 57439 |
| `TIGER_euclid_local_curve_supervision_rq` | curvature_point | verified | immutable_run | `8961e4a2062ee892` | `1f8bfecba0c74bb3` | 0.052594926792 | 57439 |
| `TIGER_frozen_topology_gnn_rq` | gnn | verified | immutable_run | `a4d7241a12d4e8e5` | `46ad65fb14866525` | 0.051898535838 | 57439 |
| `TIGER_g237_sem_rq` | semantic | legacy | arm_dir_legacy | `e992b716bbf95d68` | `44b1799a94c4e99d` | 0.052786434304 | 57439 |
| `TIGER_g4_closure_A` | graphlet_closure | legacy | arm_dir_legacy | `dfd3caf996646e98` | `8c98646fbe58ca8b` | 0.055519768798 | 57439 |
| `TIGER_g4_closure_B` | graphlet_closure | legacy | arm_dir_legacy | `103857faac7a3369` | `21052830776c1f6b` | 0.054161806438 | 57439 |
| `TIGER_g4_closure_C` | graphlet_closure | legacy | arm_dir_legacy | `68699907173c4add` | `b1f98db131a81c1e` | 0.040860739219 | 57439 |
| `TIGER_g4_closure_D` | graphlet_closure | legacy | arm_dir_legacy | `fb909f7b0c878e0c` | `0359630203f36752` | 0.038231863368 | 57439 |
| `TIGER_gnn_attention_sectional_rq` | gnn | verified | immutable_run | `037766704565c514` | `0ca5cd32c307460c` | 0.049287069761 | 57439 |
| `TIGER_graphlet_codebook_global_rq` | graphlet_codebook | legacy | arm_dir_legacy | `b5a7536d48f6c262` | `b320496a22db7e73` | 0.046379637529 | 57439 |
| `TIGER_graphlet_codebook_rq` | graphlet_codebook | legacy | arm_dir_legacy | `11f46ff80e134083` | `3c73fa3b898b9f7a` | 0.047093438256 | 57439 |
| `TIGER_graphlet_role_E` | graphlet_role | legacy | arm_dir_legacy | `1143f6f832cbcb23` | `ffbbcb7be96057d4` | 0.054335904177 | 57439 |
| `TIGER_graphlet_role_Q` | graphlet_role | legacy | arm_dir_legacy | `69e6c04729d83370` | `3f47881752adc4b9` | 0.045073904490 | 57439 |
| `TIGER_graphlet_role_QR` | graphlet_role | legacy | arm_dir_legacy | `05d8bfdab885f30e` | `6c044428c0844200` | 0.054109577117 | 57439 |
| `TIGER_graphlet_role_R` | graphlet_role | legacy | arm_dir_legacy | `cb0bb662cdd1d3bc` | `b06230405d06614d` | 0.048172844235 | 57439 |
| `TIGER_graphlet_role_T` | graphlet_role | legacy | arm_dir_legacy | `5be85adad0458e7c` | `cf3edc80144c5e76` | 0.043367746653 | 57439 |
| `TIGER_i234_A1` | i234 | legacy | arm_dir_legacy | `0da95bb0091a295d` | `089090a7becfce03` | 0.053012761364 | 57439 |
| `TIGER_i234_A2` | i234 | legacy | arm_dir_legacy | `2afffe90b7acba39` | `ea4218743bf7514a` | 0.053012761364 | 57439 |
| `TIGER_i234_A3` | i234 | legacy | arm_dir_legacy | `864afe3c0739ac6d` | `de71e28737cf0c33` | 0.052403419280 | 57439 |
| `TIGER_i234_A4` | i234 | legacy | arm_dir_legacy | `c3e77f6665d22afb` | `ead3c3252c3856c9` | 0.051915945612 | 57439 |
| `TIGER_i234_A5` | i234 | legacy | arm_dir_legacy | `4d7075fa1313990d` | `c29058eb937bfbc8` | 0.043089190271 | 57439 |
| `TIGER_kappa_sk_only_rq` | curvature_budget | legacy | arm_dir_legacy | `7ccaf8c438851bff` | `6e8cd303c585038e` | 0.055049704904 | 57439 |
| `TIGER_module_budget_prefix_rq` | budget | verified | immutable_run | `201d2cc537dfdcc2` | `680439b57bfdbec6` | 0.052734204983 | 57439 |
| `TIGER_periodic_budget_prefix_rq` | budget | verified | immutable_run | `e18fc0460eea2872` | `e9ee3214e118682a` | 0.052664565887 | 57439 |
| `TIGER_relative_curvature_ranking_rq` | curvature_rank | verified | immutable_run | `0b25f3b5e0a9e43d` | `541736b475083aef` | 0.050784310312 | 57439 |
| `TIGER_role_conditioned_l1_rq` | role_condition | verified | immutable_run | `f6754f385fe7ea46` | `37e7198921824f6a` | 0.049008513379 | 57439 |
| `TIGER_role_conditioned_l2_rq` | role_condition | verified | immutable_run | `dc206dba7fbdadc2` | `f93da1a7466d62f0` | 0.050349065966 | 57439 |
| `TIGER_role_conditioned_l3_rq` | role_condition | verified | immutable_run | `26368f3e7690c683` | `5c6b1557f004a194` | 0.051515520813 | 57439 |
| `TIGER_three_expert_attention_rq` | attention_geometry | verified | immutable_run | `24c17aee0818373f` | `ecc37a67753bee1b` | 0.046431866850 | 57439 |
| `TIGER_topology_residual_l2_rq` | topology_residual | verified | immutable_run | `c5f0e739b11b3ee7` | `ec5271f4d60e84c7` | 0.049600445690 | 57439 |

## Known gaps

- #259–#264 geometry arms (`role_geom*`, `mixed_rq_F0`) exist with SIDs and checkpoints
  but their Stage3 prediction shards are absent locally, so they cannot enter the
  recommendation linkage; they are listed in `RUN_INVENTORY.json` and excluded here.
- Legacy bindings lack a recorded SID/checkpoint hash in the Stage3 record; the
  files are still read exactly and their hashes are recorded in `COHORT.json`.
- No validation predictions exist (Stage3 `NO_EVAL=True`); all recommendation
  linkage is explanatory error analysis on the frozen test set.
