# R8 ten-mechanism leaderboard (issue #90)

Frozen baseline **16.38953942232631**; historical R4-08 **16.402627511591962**; failed R7-F **14.071573261309926**.

Every row is present. `NOT_RUN` means no final checkpoint exists; it is neither an excluded row nor a null result.

| Issue | Lane | Mechanism | Status | Steps | Active | pos_hits | neg_hits | Hard-L1 ER | Gate | vs R4-08 | Stage3 | CGV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| #0 | r8_control | CONTROL: untouched R4-08 host, no auxiliary term | ER_MEASURED | 0 | 0 | 4250 | 2598 | 16.358737490377 | FAIL | no | NOT_RUN | not_completed |
| #91 | r8_01_directed_l1_transition_nce | Directed L1 code-transition NCE | ER_MEASURED | 18000 | 9000 | 22580 | 81366 | 2.775114912863 | FAIL | no | NOT_RUN | not_completed |
| #92 | r8_02_l1_next_item_prediction | L1 next-item semantic prediction | ER_MEASURED | 18000 | 9000 | 3948 | 2619 | 15.074455899198 | FAIL | no | NOT_RUN | not_completed |
| #93 | r8_03_successor_profile_js | Successor-profile JS co-clustering | ER_MEASURED | 18000 | 9000 | 11750 | 17159 | 6.847718398508 | FAIL | no | NOT_RUN | not_completed |
| #94 | r8_04_behavior_conditioned_ot | Behaviour-conditioned entropic OT quantizer | ER_MEASURED | 18000 | 9000 | 9316 | 10544 | 8.835356600910 | FAIL | no | NOT_RUN | not_completed |
| #95 | r8_05_three_level_distillation | Three-level teacher to L1 distillation | ER_MEASURED | 18000 | 9000 | 18217 | 47424 | 3.841303981107 | FAIL | no | NOT_RUN | not_completed |
| #96 | r8_06_train_graph_diffusion | Train-graph diffusion distance preservation | ER_MEASURED | 18000 | 9000 | 4026 | 2530 | 15.913043478261 | FAIL | no | NOT_RUN | not_completed |
| #97 | r8_07_hyperbolic_gyro | Hyperbolic directed gyro-translation | ER_MEASURED | 18000 | 9000 | 4595 | 3056 | 15.035994764398 | FAIL | no | NOT_RUN | not_completed |
| #98 | r8_08_hyperbolic_frechet | Hyperbolic Frechet successor barycentres | ER_MEASURED | 18000 | 9000 | 10654 | 15675 | 6.796810207337 | FAIL | no | NOT_RUN | not_completed |
| #99 | r8_09_hyperbolic_branching | Hyperbolic branching radial hierarchy | ER_MEASURED | 18000 | 9000 | 4318 | 2722 | 15.863335782513 | FAIL | no | NOT_RUN | not_completed |
| #100 | r8_10_hyperbolic_gw | Hyperbolic graph-code Gromov-Wasserstein | ER_MEASURED | 18000 | 9000 | 52727 | 526904 | 1.000694623689 | FAIL | no | NOT_RUN | not_completed |
