# Issue #263 reference mutation checks

The production GNN is not re-implemented here. The independent `reference.py`
covers the label/distribution math that the pretraining loss consumes. Five
mutants were checked against the known-answer tests:

| Mutant | Change | Expected detectable failure |
|---|---|---|
| M1 | drop the last histogram edge | bin count/length mismatch in `test_histogram_known_answer` |
| M2 | normalise a missing row to uniform | `test_normalize_does_not_impute_missing_rows` sees nonzero row |
| M3 | average CE per class instead of per row | identity CE differs from the known `-sum(p log p)` |
| M4 | ignore the mask in CE | masked rows change the scalar |
| M5 | count sign 0 as wrong | `test_sign_accuracy_and_edge` fails |

`tests/test_mutation.py` checks that these structural mistakes are not accepted
by the independent reference tests; it does not train or mutate a model.
