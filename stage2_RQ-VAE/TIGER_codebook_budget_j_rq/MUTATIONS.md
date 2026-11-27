# Mutations — issue #272 round 3 codebook budget

Each mutation is a deliberate wrong variant of `model.budget.scale_codebook_terms`. The
"caught by" entry is the production assertion that fails for the mutation; a crash or an import
error is not a catch.

| id | mutation | caught by |
|---|---|---|
| M1 | `alpha=beta` instead of `1-beta` (the codebook keeps `1-beta` of the local+constraint) | `tests/test_codebook_budget.py::test_beta_one_percent_matches_the_hand_computed_codebook_gradient` |
| M2 | `p.grad.add_` instead of `sub_` (the codebook gradient grows) | the same known-answer test |
| M3 | the encoder is passed into the budget (the encoder loses its full gradient) | `tests/test_codebook_budget.py::test_beta_one_percent_matches_the_hand_computed_codebook_gradient` (the encoder assertion) |
| M4 | the local term only (the constraint is not scaled) | the same known-answer test |
| M5 | the constraint only (the local is not scaled) | the same known-answer test |
| M6 | no budget (the round-1 behaviour, the red baseline) | the same known-answer test |

Every mutation is caught by its specified assertion; see
`tests/test_codebook_budget_mutations.py`. M6 is the red baseline: the round-1 behaviour fails
the same assertion that the helper passes.
