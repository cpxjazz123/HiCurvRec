# Mutation tests — issue #287 round 6 (the hard-negative ranking loss)

Every mutation is injected into a **copy** of `hard_negative_loss.py`; the copy and the test files run
in a temporary directory (`tests/test_mutation.py`).  A mutation counts as caught only when the run
reports a **failed assertion** (a crash or an import error does not count).  The machine-readable
record is `MUTATIONS.json` (written by the test) and the raw outcomes are below.

| mutation | the injected error | the catching test (the named assertion) |
|---|---|---|
| M1_sign_flip | `d_pos - d_neg + margin` becomes `d_neg - d_pos + margin` | `test_hand_computed_hinge_mean`, `test_inverted_order_is_penalised` |
| M2_margin_sign | the margin enters with the opposite sign | `test_hand_computed_hinge_mean`, `test_insufficient_margin_is_zero` |
| M3_wrong_reduction | the mean over the groups becomes a sum | `test_hand_computed_hinge_mean` |
| M4_swapped_pairs | the positive distance reads the negative item | `test_hand_computed_hinge_mean`, `test_inverted_order_is_penalised` |
| M5_detached_representation | the positive is detached, so the loss cannot train the encoder | `test_the_gradient_reaches_the_representation_and_pushes_the_negative_away` |
| M6_holdout_leak | the hold-out groups enter the training sampling | `test_the_index_sampling_is_deterministic_and_uses_only_the_training_groups` |
| M7_mining_keeps_the_neighbours | the miner stops excluding the anchor's direct neighbours | `test_the_mining_excludes_the_direct_neighbours` |
| M8_no_margin_validation | the loss accepts a non-positive margin | `test_the_margin_must_be_positive_and_is_frozen` |

The three classes the skill asks for are covered: **numerical** (M1, M2, M3), **structural**
(M4 - a wrong routing of the pair) and **routing/leakage** (M5, M6, M7, M8 - the detached
representation, the hold-out leak, the miner that stops excluding the behaviourally related items
and the margin validation).
