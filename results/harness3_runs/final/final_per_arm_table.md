# The final per-arm table (issue #106)

Generated 2026-10-21T14:33:31.863266+00:00 from the evidence store and the candidate records. The frozen Hard-L1 ER gate is 16.38953942232631; the historical R4-08 reference is 16.402627511591962; the archived control of this window is ER 16.334505666275888 with Stage-3 recall@10 0.06406796775709883 and NDCG@10 0.03530651701940677. The control is evidence and was never re-run.

**Uncertainty**: every ER is ONE run at the single frozen seed, and every Stage-3 number is ONE run of the frozen protocol. The bootstrap interval of each ER is given in the row. No number here is a claim of statistical significance or of reproducibility across seeds.

| arm | issue | proposal sha | judges | export | epochs | Hard-L1 ER | vs gate | vs R4-08 | vs control | Stage-3 R@10 | delta | Stage-3 NDCG@10 | delta | run id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r8_control | #103/#104/#105 (archived) | - | - | reference | 3000 | 16.334505666275888 | -0.05503375605042038 | -0.06812184531607457 | 0.0 | 0.06406796775709883 | 0.0 | 0.03530651701940677 | 0.0 | control-r8_control-20261021T143221 |
| behavior_conditioned_codebook_directional_transport | #103/#104/#105 (archived) | - | - | active | 3000 | 15.280216070222822 | -1.1093233521034858 | -1.12241144136914 | -1.0542895960530654 | None | - | None | - | stage2-behavior_conditioned_codebook_directional_transport-20261021T143321 |
| behavior_conditioned_pairwise_curvature | #103/#104/#105 (archived) | - | all 11 declared checks hold | active | 3000 | 2.619839290662233 | -13.769700131664075 | -13.78278822092973 | -13.714666375613655 | None | - | None | - | stage2-behavior_conditioned_pairwise_curvature-20261021T143307 |
| behavior_conditioned_pairwise_margin | #103/#104/#105 (archived) | - | all 10 declared checks hold | inert | 3000 | 16.334505666275888 | -0.05503375605042038 | -0.06812184531607457 | 0.0 | None | - | None | - | stage2-behavior_conditioned_pairwise_margin-20261021T143309 |
| behavior_directed_code_assignment_cross_entropy | #103/#104/#105 (archived) | - | - | collapsed | 3000 | 1.478111600364957 | -14.91142782196135 | -14.924515911227004 | -14.85639406591093 | None | - | None | - | stage2-behavior_directed_code_assignment_cross_entropy-20261021T143331 |
| behavior_directed_soft_code_transition_kl | #103/#104/#105 (archived) | - | - | active | 3000 | 16.40282131661442 | 0.013281894288112994 | 0.00019380502245880393 | 0.06831565033853337 | 0.06225735127700691 | -0.0018106164800919186 | 0.03384902483338121 | -0.0014574921860255607 | stage2-behavior_directed_soft_code_transition_kl-20261021T143319 |
| behavior_residual_geometry_transport | #103/#104/#105 (archived) | - | all 10 declared checks hold | active | 3000 | 16.616 | 0.22646057767369143 | 0.21337248840803724 | 0.2814943337241118 | 0.06338898657706436 | -0.0006789811800344686 | 0.03454331085500618 | -0.0007632061644005927 | stage2-behavior_residual_geometry_transport-20261021T143316 |
| behaviour_conditioned_residual_target | #103/#104/#105 (archived) | - | 1 of 10 declared checks failed | active | 3000 | 16.503445774392453 | 0.11390635206614519 | 0.100818262800491 | 0.16894010811656557 | 0.0610038475600202 | -0.0030641201970786347 | 0.03381020016512428 | -0.0014963168542824895 | stage2-behaviour_conditioned_residual_target-20261021T143311 |
| curvature_weighted_transition_transport | #103/#104/#105 (archived) | - | all 11 declared checks hold | inert | 3000 | 16.334505666275888 | -0.05503375605042038 | -0.06812184531607457 | 0.0 | None | - | None | - | stage2-curvature_weighted_transition_transport-20261021T143314 |
| directed_assignment_usage_balance | #106 (new candidate) | 9d2f0b648562c763 | - | active | 3000 | 16.651583710407238 | 0.2620442880809293 | 0.2489561988152751 | 0.31707804413134966 | 0.06241403924163025 | -0.0016539285154685834 | 0.034312283480564885 | -0.000994233538841885 | stage2-directed_assignment_usage_balance-20261021T143329 |
| directed_codeword_margin_geometry | #106 (new candidate) | 6a37f2c036538873 | - | active | 3000 | 16.591820368885326 | 0.20228094655901785 | 0.18919285729336366 | 0.25731470260943823 | 0.06373718205400512 | -0.00033078570309370836 | 0.03466763666330205 | -0.0006388803561047235 | stage2-directed_codeword_margin_geometry-20261021T143326 |
| directed_st_code_transition_geometry | #103/#104/#105 (archived) | - | all 11 declared checks hold | - | - | None | - | - | - | None | - | None | - | stage2-directed_st_code_transition_geometry-20261021T143331 |
| geometry_transport_seed43 | #103/#104/#105 (archived) | - | - | active | 3000 | 16.73965936739659 | 0.35011994507028277 | 0.3370318558046286 | 0.40515370112070315 | None | - | None | - | stage2-geometry_transport_seed43-20261021T143324 |
| r8_control_seed43 | #103/#104/#105 (archived) | - | - | reference | 3000 | 16.036655211912944 | -0.3528842104133645 | -0.3659722996790187 | -0.2978504543629441 | None | - | None | - | control-r8_control_seed43-20261021T143322 |

## The immutable artifacts

### r8_control (control-r8_control-20261021T143221)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r8_control/sids_for_hgrec.npy`
* `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/r8_control/logs/Amazon_2023_Instruments/Oct-21-2026_14-55-22/test_final.json`

### behavior_conditioned_codebook_directional_transport (stage2-behavior_conditioned_codebook_directional_transport-20261021T143321)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_conditioned_codebook_directional_transport/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_conditioned_codebook_directional_transport/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_conditioned_codebook_directional_transport/r8_control/sids_for_hgrec.npy`

### behavior_conditioned_pairwise_curvature (stage2-behavior_conditioned_pairwise_curvature-20261021T143307)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_curvature/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_curvature/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_curvature/r8_control/sids_for_hgrec.npy`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/behavior_conditioned_pairwise_curvature/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/v2_judges/behavior_conditioned_pairwise_curvature.json`

### behavior_conditioned_pairwise_margin (stage2-behavior_conditioned_pairwise_margin-20261021T143309)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_margin/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_margin/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_margin/r8_control/sids_for_hgrec.npy`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/behavior_conditioned_pairwise_margin/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/v2_judges/behavior_conditioned_pairwise_margin.json`

### behavior_directed_code_assignment_cross_entropy (stage2-behavior_directed_code_assignment_cross_entropy-20261021T143331)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_directed_code_assignment_cross_entropy/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_directed_code_assignment_cross_entropy/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_directed_code_assignment_cross_entropy/r8_control/sids_for_hgrec.npy`

### behavior_directed_soft_code_transition_kl (stage2-behavior_directed_soft_code_transition_kl-20261021T143319)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_directed_soft_code_transition_kl/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_directed_soft_code_transition_kl/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_policy_behavior_directed_soft_code_transition_kl/r8_control/sids_for_hgrec.npy`
* `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/TIGER_c1_policy_behavior_directed_soft_code_transition_kl__r8_control/logs/Amazon_2023_Instruments/Oct-21-2026_20-54-52/test_final.json`

### behavior_residual_geometry_transport (stage2-behavior_residual_geometry_transport-20261021T143316)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_residual_geometry_transport/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_residual_geometry_transport/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_behavior_residual_geometry_transport/r8_control/sids_for_hgrec.npy`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/behavior_residual_geometry_transport/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/redesign_geometry/verification.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/TIGER_behavior_residual_geometry_transport__r8_control/logs/Amazon_2023_Instruments/Oct-21-2026_14-01-57/test_final.json`

### behaviour_conditioned_residual_target (stage2-behaviour_conditioned_residual_target-20261021T143311)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behaviour_conditioned_residual_target/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behaviour_conditioned_residual_target/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_behaviour_conditioned_residual_target/r8_control/sids_for_hgrec.npy`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/behaviour_conditioned_residual_target/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/v2_judges/behaviour_conditioned_residual_target.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/TIGER_c1_behaviour_conditioned_residual_target__r8_control/logs/Amazon_2023_Instruments/Oct-21-2026_12-07-27/test_final.json`

### curvature_weighted_transition_transport (stage2-curvature_weighted_transition_transport-20261021T143314)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_curvature_weighted_transition_transport/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_curvature_weighted_transition_transport/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c1_curvature_weighted_transition_transport/r8_control/sids_for_hgrec.npy`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/curvature_weighted_transition_transport/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/v2_judges/curvature_weighted_transition_transport.json`

### directed_assignment_usage_balance (stage2-directed_assignment_usage_balance-20261021T143329)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c106_directed_assignment_usage_balance/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c106_directed_assignment_usage_balance/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c106_directed_assignment_usage_balance/r8_control/sids_for_hgrec.npy`
* `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/TIGER_c106_directed_assignment_usage_balance__r8_control/logs/Amazon_2023_Instruments/Oct-22-2026_00-56-45/test_final.json`

### directed_codeword_margin_geometry (stage2-directed_codeword_margin_geometry-20261021T143326)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c106_directed_codeword_margin_geometry/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c106_directed_codeword_margin_geometry/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_c106_directed_codeword_margin_geometry/r8_control/sids_for_hgrec.npy`
* `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/TIGER_c106_directed_codeword_margin_geometry__r8_control/logs/Amazon_2023_Instruments/Oct-22-2026_00-00-42/test_final.json`

### directed_st_code_transition_geometry (stage2-directed_st_code_transition_geometry-20261021T143331)

* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/probes/directed_st_code_transition_geometry/probe_report.json`
* `/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/redesign/verification.json`

### geometry_transport_seed43 (stage2-geometry_transport_seed43-20261021T143324)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_geometry/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_geometry/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_geometry/r8_control/sids_for_hgrec.npy`

### r8_control_seed43 (control-r8_control_seed43-20261021T143322)

* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_control/r8_control/item_sids.json`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_control/r8_control/rqvae_best.pth`
* `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_seed43_control/r8_control/sids_for_hgrec.npy`

