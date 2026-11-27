# Issue #271 mutations

Each mutation is injected into a copy of the formula; the named assertion must fail for the mutated
version while the production assertion passes. The tests are in
`stage2_RQ-VAE/TIGER_global_j_rq/tests/test_global_mutation.py`; the production checks are in
`tests/test_global_constraint.py`.

| id | mutation | expected failure | evidence |
|---|---|---|---|
| M1 | drop the factor that turns `w=q/(2S)` into `d=||q||/S` (use `||q||/(2Sg)`) | `test_weighted_j_exact_zero_and_one`: the exact-zero case becomes J=0.25 instead of 0 | `test_mutation_ruler_factor_dropped` |
| M2 | replace the weighted mean by an unweighted mean | `test_weighted_j_reference_loop` disagrees on non-uniform weights | `test_mutation_weight_normalisation_dropped` |
| M3 | invert the dual sign (μ falls on violation) | `test_dual_step_sign_and_cap` expects μ=0.21 on a violation | `test_mutation_dual_sign_inverted` |
| M4 | include the diagonal/self pairs in the full J | `test_full_j_known_answer_and_masks` expects 3 pairs on the 3-node path (15 would fail) | `test_mutation_diagonal_included` |
| M5 | detach the hard q before the global loss | `test_gradient_reaches_q` requires a nonzero gradient through q | `test_mutation_hard_q_detached` |

All five mutations are caught by the specified assertion; none is caught only by a crash. The
gradient test also checks the coincident-point case: a zero-distance pair must produce a finite zero
subgradient (no NaN), because a naive `sqrt(0)` produces NaN and would poison the encoder.
