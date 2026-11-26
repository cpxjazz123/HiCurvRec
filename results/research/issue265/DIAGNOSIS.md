# Issue #265 diagnosis

## Observations inherited

- #263: frozen-token Stage2 collision 0.0607, Stage3 `test_R@10=0.0518985`; mixed cost was used for
  selection while fixed equal-weight curvature supervised prefixes.
- #260/R05: changing role mixture changed only 0.11% of L2 codewords, suggesting the selection
  channel is insensitive to gate/curvature changes.
- #262: curvature MAE and curvature magnitude discrimination were not aligned; a single mean/role is
  not sufficient.

## Competing explanations

- **A (selection insensitivity):** mixed curvature in the codeword cost is nearly constant/ordering
  neutral; gate changes move the cost scale but not the argmin, so SID is unchanged.
- **B (structural supervision has no sufficient behavior information):** even perfect local
  curvature supervision cannot improve recommendation because the frozen target/reference structure
  is not predictive of next-item behavior.
- **Null:** changes are numerical/optimization noise; no actionable difference.

## Pre-registered low-cost checks

1. Compare hard SIDs and pairwise margins from the frozen #263 checkpoint under mixed versus
   Euclidean selection on training/validation catalogue data, without retraining.
2. Fix q and vary gate weights; measure the fixed equal-weight curvature error. Fix gate and vary q;
   measure fixed curvature and gradients.
3. Check whether the gate/curvature signal changes the selected prefix at all; report the fraction,
   not only a loss value.

A positive signal requires a measurable selection/representation change and a real gradient route;
a loss decrease from gate-only scale changes is not evidence of SID improvement. The mechanism is
still falsifiable and a negative recommendation result is acceptable.

## Low-cost result: mixed vs Euclidean selection

Frozen #263 checkpoint, full 24,587-item catalogue, same token cache and no test split:
mixed-curvature selection and Euclidean selection produce **44 different hard SIDs
(0.17896%)**; the two SID hashes are in `DIAGNOSIS_SELECTION.json`. This is a direct
measurement that the old selection channel is nearly ordering-neutral on the
frozen representation, supporting explanation A while not establishing whether
structural supervision has enough behavior information (B remains untested here).
No retraining is involved.
