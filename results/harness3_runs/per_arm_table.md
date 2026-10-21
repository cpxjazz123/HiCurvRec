# The per-arm table

Generated 2026-10-21T11:16:12.700657+00:00 from the evidence store. The frozen Hard-L1 ER gate is 16.38953942232631 and the historical R4-08 reference is 16.402627511591962. The matched control for this window is the first row.

| arm | issue | declaration sha | judges | probe | export | epochs | Hard-L1 ER | vs gate | vs R4-08 | ER delta vs control | ER CI95 | Stage-3 R@10 | Stage-3 NDCG@10 | run id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r8_control | #105 (the matched control for this window) | - | - | - | reference | 3000 | 16.334505666275888 | -0.05503375605042038 | -0.06812184531607457 | +0.0000 | [15.563964614011141, 17.180504770538473] | 0.06406796775709883 | 0.03530651701940677 | control-r8_control-20261021T110437 |
| behavior_conditioned_codebook_directional_transport | the run's own manifest | - | - | - | active | 3000 | 15.280216070222822 | -1.1093233521034858 | -1.12241144136914 | -1.0543 | [14.606288126555599, 15.988142948718563] | - | - | stage2-behavior_conditioned_codebook_directional_transport-20261021T110527 |
| behavior_conditioned_pairwise_curvature | #103 (repaired, judged in #104) | 43556fd2a76e39bc | all 11 declared checks hold | the real entry ran, the declared parameters moved, the exported hard SIDs changed and the code did not collapse | active | 3000 | 2.619839290662233 | -13.769700131664075 | -13.78278822092973 | -13.7147 | [2.5664184162314267, 2.671704369111437] | - | - | stage2-behavior_conditioned_pairwise_curvature-20261021T110513 |
| behavior_conditioned_pairwise_margin | #103 (repaired, judged in #104) | 318d06cd8b82fad5 | all 10 declared checks hold | the real entry ran, the declared parameters moved, the exported hard SIDs changed and the code did not collapse | inert | 3000 | 16.334505666275888 | -0.05503375605042038 | -0.06812184531607457 | +0.0000 | [15.563964614011141, 17.180504770538473] | - | - | stage2-behavior_conditioned_pairwise_margin-20261021T110515 |
| behavior_directed_code_assignment_cross_entropy | #105 (ledger-derived, judged and probed) | 364110e655e26d3f | - | - | collapsed | 3000 | 1.478111600364957 | -14.91142782196135 | -14.924515911227004 | -14.8564 | [1.4698992388140568, 1.4860194613649877] | - | - | stage2-behavior_directed_code_assignment_cross_entropy-20261021T110532 |
| behavior_directed_soft_code_transition_kl | the run's own manifest | - | - | - | active | 3000 | 16.40282131661442 | 0.013281894288112994 | 0.00019380502245880393 | +0.0683 | [15.638471833878368, 17.220618022328548] | 0.06225735127700691 | 0.03384902483338121 | stage2-behavior_directed_soft_code_transition_kl-20261021T110524 |
| behavior_residual_geometry_transport | #104 (versioned redesign) | ba35076a6423e646 | all 10 declared checks hold | the real entry ran, the declared parameters moved, the exported hard SIDs changed and the code did not collapse | active | 3000 | 16.616 | 0.22646057767369143 | 0.21337248840803724 | +0.2815 | [15.832937361772315, 17.540387471250334] | 0.06338898657706436 | 0.03454331085500618 | stage2-behavior_residual_geometry_transport-20261021T110522 |
| behaviour_conditioned_residual_target | #103 (repaired, judged in #104) | 05c319f174359669 | 1 of 10 declared checks failed | the real entry ran, the declared parameters moved, the exported hard SIDs changed and the code did not collapse | active | 3000 | 16.503445774392453 | 0.11390635206614519 | 0.100818262800491 | +0.1689 | [15.79346905356989, 17.260442889558657] | 0.0610038475600202 | 0.03381020016512428 | stage2-behaviour_conditioned_residual_target-20261021T110518 |
| curvature_weighted_transition_transport | #103 (repaired, judged in #104) | 8520b4e281bcbc66 | all 11 declared checks hold | the real entry ran, the declared parameters moved, the exported hard SIDs changed and the code did not collapse | inert | 3000 | 16.334505666275888 | -0.05503375605042038 | -0.06812184531607457 | +0.0000 | [15.563964614011141, 17.180504770538473] | - | - | stage2-curvature_weighted_transition_transport-20261021T110520 |
| directed_st_code_transition_geometry | #104 (versioned redesign, refused at the probe) | 4b2067831eedb6ec | all 11 declared checks hold | the code collapsed: the arm uses 1 level-0 entries against the host's 199 (0.5% of it) | - | 3000 | None | - | - | - | - | - | - | stage2-directed_st_code_transition_geometry-20261021T110533 |
| geometry_transport_seed43 | the run's own manifest | - | - | - | active | 3000 | 16.73965936739659 | 0.35011994507028277 | 0.3370318558046286 | +0.4052 | [15.986512944138077, 17.616393302090106] | - | - | stage2-geometry_transport_seed43-20261021T110530 |
| r8_control_seed43 | the run's own manifest | - | - | - | reference | 3000 | 16.036655211912944 | -0.3528842104133645 | -0.3659722996790187 | -0.2979 | [15.305116499648957, 16.845240399976646] | - | - | control-r8_control_seed43-20261021T110528 |

## The immutable artifacts

### r8_control (control-r8_control-20261021T110437)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r8_control/sids_for_hgrec.npy`
* `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/r8_control/logs/Amazon_2023_Instruments/Oct-21-2026_14-55-22/test_final.json`

### behavior_conditioned_codebook_directional_transport (stage2-behavior_conditioned_codebook_directional_transport-20261021T110527)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_conditioned_codebook_directional_transport/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_conditioned_codebook_directional_transport/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_conditioned_codebook_directional_transport/r8_control/sids_for_hgrec.npy`

### behavior_conditioned_pairwise_curvature (stage2-behavior_conditioned_pairwise_curvature-20261021T110513)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_curvature/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_curvature/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_curvature/r8_control/sids_for_hgrec.npy`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/behavior_conditioned_pairwise_curvature/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/v2_judges/behavior_conditioned_pairwise_curvature.json`

### behavior_conditioned_pairwise_margin (stage2-behavior_conditioned_pairwise_margin-20261021T110515)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_margin/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_margin/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_margin/r8_control/sids_for_hgrec.npy`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/behavior_conditioned_pairwise_margin/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/v2_judges/behavior_conditioned_pairwise_margin.json`

### behavior_directed_code_assignment_cross_entropy (stage2-behavior_directed_code_assignment_cross_entropy-20261021T110532)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_directed_code_assignment_cross_entropy/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_directed_code_assignment_cross_entropy/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_directed_code_assignment_cross_entropy/r8_control/sids_for_hgrec.npy`

### behavior_directed_soft_code_transition_kl (stage2-behavior_directed_soft_code_transition_kl-20261021T110524)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_directed_soft_code_transition_kl/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_directed_soft_code_transition_kl/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_directed_soft_code_transition_kl/r8_control/sids_for_hgrec.npy`
* `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/TIGER_c1_policy_behavior_directed_soft_code_transition_kl__r8_control/logs/Amazon_2023_Instruments/Oct-21-2026_20-54-52/test_final.json`

### behavior_residual_geometry_transport (stage2-behavior_residual_geometry_transport-20261021T110522)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_residual_geometry_transport/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_residual_geometry_transport/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_residual_geometry_transport/r8_control/sids_for_hgrec.npy`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/behavior_residual_geometry_transport/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/redesign_geometry/verification.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/TIGER_behavior_residual_geometry_transport__r8_control/logs/Amazon_2023_Instruments/Oct-21-2026_14-01-57/test_final.json`

### behaviour_conditioned_residual_target (stage2-behaviour_conditioned_residual_target-20261021T110518)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behaviour_conditioned_residual_target/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behaviour_conditioned_residual_target/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behaviour_conditioned_residual_target/r8_control/sids_for_hgrec.npy`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/behaviour_conditioned_residual_target/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/v2_judges/behaviour_conditioned_residual_target.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/TIGER_c1_behaviour_conditioned_residual_target__r8_control/logs/Amazon_2023_Instruments/Oct-21-2026_12-07-27/test_final.json`

### curvature_weighted_transition_transport (stage2-curvature_weighted_transition_transport-20261021T110520)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_curvature_weighted_transition_transport/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_curvature_weighted_transition_transport/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_curvature_weighted_transition_transport/r8_control/sids_for_hgrec.npy`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/curvature_weighted_transition_transport/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/v2_judges/curvature_weighted_transition_transport.json`

### directed_st_code_transition_geometry (stage2-directed_st_code_transition_geometry-20261021T110533)

* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/directed_st_code_transition_geometry/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/redesign/verification.json`

### geometry_transport_seed43 (stage2-geometry_transport_seed43-20261021T110530)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_geometry/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_geometry/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_geometry/r8_control/sids_for_hgrec.npy`

### r8_control_seed43 (control-r8_control_seed43-20261021T110528)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_control/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_control/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_control/r8_control/sids_for_hgrec.npy`

