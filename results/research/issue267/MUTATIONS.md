# Issue #267 mutation record

The relative term is deterministic and small enough to mutate directly. Each
mutation is checked against the named assertion in `tests/test_relative.py`; a
crash or import failure would not count.

| # | mutation | expected detection | evidence |
|---|---|---|---|
| 1 | flip `sign(delta_target)` in the softplus | `test_known_sign_and_hand_value` fails: correct-order loss would exceed wrong-order loss | hand values `softplus(-2)` vs `softplus(2)` |
| 2 | remove the `abs(delta_target)>tie_eps` mask | `test_ties_are_excluded_not_averaged_in` fails valid/tie counts and exact mean | 2 valid + 1 tied pair |
| 3 | include cross-centre pairs | `test_cross_centre_pairs_are_never_used` fails valid=0/loss=0 | centres `[0,1,2]` |
| 4 | use all ordered pairs (both triangles) | `test_upper_triangle_pair_count_28_rows` fails 378 vs 756 | exact pair count |
| 5 | sum instead of mean over valid pairs | exact hand-value assertion in `test_known_sign_and_hand_value` or tie test fails | 1-pair and 3-pair cases |

The `test_relative.py` suite is run against the unmodified production module; the
mutations above are the minimal edits that would make the named assertions fail.
No mutation is accepted by a crash alone.

Executed check: `python3.9 results/research/issue267/analysis/mutation_check.py`
(write to `results/research/issue267/logs/mutation_check.log`); all five mutations
were caught (`analysis/MUTATION_RESULTS.json`).
