# Mutation tests — issue #274's pair-ranking loss

Every mutation is injected into a **copy** of `pair_rank_loss.py`; the copy and the test files run in
a temporary directory (`tests/test_mutation.py`).  A mutation counts as caught only when the run
reports a **failed assertion** (a crash or an import error does not count).  The machine-readable
record is `MUTATIONS.json` (written by the test) and the raw outcomes are below.

| mutation | the injected error | the catching test (the named assertion) |
|---|---|---|
| M1_sign_flip | `d_pos - d_neg` becomes `d_neg - d_pos` | `test_hand_computed_hinge_mean` (the hand-computed hinge mean), `test_inverted_order_is_penalised`, `test_insufficient_margin_is_zero` |
| M2_margin_sign | the margin enters with the opposite sign | `test_hand_computed_hinge_mean`, `test_inverted_order_is_penalised`, `test_insufficient_margin_is_zero` |
| M3_wrong_reduction | the mean over the groups becomes a sum | `test_hand_computed_hinge_mean`, `test_reference_consistency_on_random_inputs` |
| M4_single_level_hard_q | the hard q uses only the first VQ level | `test_reference_consistency_on_random_inputs`, `test_the_hard_q_is_the_plain_sum_of_the_three_codewords`, `test_the_gradient_reaches_the_codebooks_and_not_the_encoder` |
| M5_swapped_pairs | the positive distance reads the negative pair | `test_hand_computed_hinge_mean`, `test_inverted_order_is_penalised`, `test_insufficient_margin_is_zero` |
| M6_holdout_leak | the hold-out groups enter the training sampling | `test_the_index_sampling_is_deterministic_and_uses_only_the_training_groups` |
| M7_detached_codebook | the codebook gradient is detached, so the loss cannot train the code geometry | `test_the_gradient_reaches_the_codebooks_and_not_the_encoder` |

The three classes the skill asks for are covered: **numerical** (M1, M2, M3), **structural**
(M4, M5 — a wrong reduction and a wrong routing of the pair) and **routing/leakage** (M6, M7 — the
hold-out leak and the detached codebook gradient).
