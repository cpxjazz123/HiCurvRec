# Issue254 designated mutation checks

Acceptance: each deliberately incorrect value/route must raise the assigned assertion. Import errors, crashes and timeouts do not count. Actual execution outcome and source hashes are in TEST_RESULTS.json and GREEN.log.

| Mutation | Category | Designated detector / assertion |
|---|---|---|
| Reverse improvement-minus-regression sign | numeric | `KnownAnswers.test_four_groups_signed_delta`: delta is improvements minus regressions |
| Divide item net count by item frequency instead of global N | numeric/reduction | `test_population_denominator_and_repeated_target`: item contribution denominator is global N |
| Treat zero-based rank as display rank | numeric/off-by-one | `test_rank_is_one_based_display_and_censored`: display ranks are one-based |
| Route fourth collision suffix into three-level prefix | structural/routing | `test_prefix_ignores_collision_suffix`: fourth token is not a geometry level |
| Overwrite received user/sample identity with row-number lookup before validation | identity routing | `test_identity_mismatch_rejected`: expected ValueError is not raised |
| Replace missing curvature MAE with0 | missing-data mask | `MechanismTests.test_curvature_explicit_identity_missing`: missing curvature mask must remain undefined |

The initial mask mutation targeted the producer module while the test held an imported callable alias; it therefore did not reach the consumer. The detector was repaired to mutate the actual consumer binding, not counted as a caught mutation until the numeric assertion failed. A subsequent failure was only NumPy2's `np.False_` representation differing from a generic exception regex; the designated assertion now names the invariant explicitly rather than matching an incidental boolean representation. Both failed full runs are retained as GREEN_FAILURE_1/2.log; none is claimed as a pass.

The initial RED.log independently reproduces the interrupted field-name KeyError, missing/duplicate identity acceptance, fourth-level prefix bug and display-rank off-by-one. Training-frequency input-schema failure and its regression are retained in STATISTICS_FAILURE_1.log and `test_training_frequency_input_hash_not_asset_entry`. No production training/math code changed.
