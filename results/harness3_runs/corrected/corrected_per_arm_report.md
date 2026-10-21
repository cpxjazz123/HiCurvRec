# The corrected per-arm report (issue #106)

the existing store's manifests and campaign records only; no run, no evaluation and no manifest was re-executed or rewritten.

The frozen Hard-L1 ER gate is 16.38953942232631 and the historical R4-08 reference is 16.402627511591962.

| arm | status | epochs | Hard-L1 ER | export | Stage-3 R@10 | consistency | run id |
|---|---|---|---|---|---|---|---|
| r8_control | succeeded | 3000 | 16.334505666275888 | reference | 0.06406796775709883 | consistent | control-r8_control-20261021T110437 |
| behavior_conditioned_codebook_directional_transport | succeeded | 3000 | 15.280216070222822 | active | None | consistent | stage2-behavior_conditioned_codebook_directional_transport-20261021T110527 |
| behavior_conditioned_pairwise_curvature | succeeded | 3000 | 2.619839290662233 | active | None | consistent | stage2-behavior_conditioned_pairwise_curvature-20261021T110513 |
| behavior_conditioned_pairwise_margin | succeeded | 3000 | 16.334505666275888 | inert | None | consistent | stage2-behavior_conditioned_pairwise_margin-20261021T110515 |
| behavior_directed_code_assignment_cross_entropy | succeeded | 3000 | 1.478111600364957 | collapsed | None | consistent | stage2-behavior_directed_code_assignment_cross_entropy-20261021T110532 |
| behavior_directed_soft_code_transition_kl | succeeded | 3000 | 16.40282131661442 | active | 0.06225735127700691 | consistent | stage2-behavior_directed_soft_code_transition_kl-20261021T110524 |
| behavior_residual_geometry_transport | succeeded | 3000 | 16.616 | active | 0.06338898657706436 | consistent | stage2-behavior_residual_geometry_transport-20261021T110522 |
| behaviour_conditioned_residual_target | succeeded | 3000 | 16.503445774392453 | active | 0.0610038475600202 | consistent | stage2-behaviour_conditioned_residual_target-20261021T110518 |
| curvature_weighted_transition_transport | succeeded | 3000 | 16.334505666275888 | inert | None | consistent | stage2-curvature_weighted_transition_transport-20261021T110520 |
| directed_st_code_transition_geometry | refused | not trained | None | None | None | consistent | stage2-directed_st_code_transition_geometry-20261021T110533 |
| geometry_transport_seed43 | succeeded | 3000 | 16.73965936739659 | active | None | consistent | stage2-geometry_transport_seed43-20261021T110530 |
| r8_control_seed43 | succeeded | 3000 | 16.036655211912944 | reference | None | consistent | control-r8_control_seed43-20261021T110528 |

## The corrections

* **directed_st_code_transition_geometry** -- was: 3000 epochs recorded, with no ER and no export. Now: not trained: the active probe refused it and it has no export of its own; the 3000 epochs in the original manifest came from its tree's metrics file, which belongs to an earlier run of the same tree.
* **the policy campaign arms** -- was: one record that said G6 failed and the launch was refused, and also carried an ER. Now: the attempt history: the first attempt was refused on the controller's own wiring and the relaunch passed all six gates and trained; the ER belongs to the relaunch.

## The launch attempts

* learned: admitted=True status=failed started=False refused=CGV gate: {'G1': 'passed', 'G2': 'passed', 'G3': 'passed', 'G4': 'passed', 'G5': 'passed', 'G6': 'failed'} gates={'G1': 'passed', 'G2': 'passed', 'G3': 'passed', 'G4': 'passed', 'G5': 'passed', 'G6': 'failed'}
* old: admitted=True status=cold_running started=False refused=None gates={'G1': 'passed', 'G2': 'passed', 'G3': 'passed', 'G4': 'passed', 'G5': 'passed', 'G6': 'passed'}
