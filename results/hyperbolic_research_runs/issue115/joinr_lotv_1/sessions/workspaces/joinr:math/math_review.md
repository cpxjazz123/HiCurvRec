# Mathematical review — LOTV

## Verdict: **REVISE**

The declared operator is a signed determinant of four ambient-coordinate columns, not an unsigned 2-simplex area. The document's framing is internally consistent on that point. I do not review an unsigned area, second score term, or any other candidate: those are not in `candidate.md`.

### 1. Definition and degeneracies

For finite vectors, the specified `D` and `A` are scalar-valued, including when points repeat, columns are dependent, or `D` is negative. The finite-precision rules define a deterministic result for those cases (subject to the stated invalidation on non-finite intermediates). `D` is a 4-by-4 minor, not the full ambient Lorentzian simplex volume; it is coordinate-chart dependent, as the proposal acknowledges.

There is, however, a serious domain/numerics hole in the assertion that every finite valid hyperboloid input can be processed. Binary64 finite coordinates do not guarantee finite intermediate products or sums. Large finite coordinates can overflow in the norm check or determinant products, and cancellation in the norm check can also make validity dependent on the prescribed arithmetic. The fail-closed behavior is deterministic, but this means the operator is not defined with a finite scalar for every input in the stated finite-input domain. The proposal needs either a bounded input domain with a proven no-overflow bound, or an explicit narrower admissible domain and an acknowledgment that outside it the operator is invalid.

### 2. Epsilon regularization and gradients

There is no epsilon-regularized training form in this candidate. It defines `D/(1+|D|)` above the determinant threshold and exactly zero at/below it. It has no training loss and makes no gradient claim; the discrete selection/unchanged STE account is correct for that declared path.

The hard threshold creates a discontinuity at `|D| = eps_D`: the nonzero-side limit is approximately `eps_D`, while the threshold branch returns zero. This is not a differentiability issue for the stated non-training tie-break, but should not be described as a smooth regularization. `A` is also not evaluated for gradient propagation.

### 3. Flat limit and curvature control

No flat-limit identity or `c -> 0` control is stated in the candidate. The curvature is fixed at `-1`; therefore there is no algebraic flat-limit claim to validate, and no curvature control that can discriminate this mechanism from a flat counterpart. Such claims must not be attributed to this document.

### 4. Capacity and metric-swap distinguishability

The operator adds no learned parameters or token capacity. It is a deterministic, fixed-coordinate tie-break over the existing candidates. It is mathematically distinguishable from the identity ordinary assignment on constructed tied fixtures with differing determinants, but it is not an extra score term used outside the declared tie set.

It is not invariant under general rotations of the spatial coordinates: its first-four-coordinate minor changes under coordinate transformations, and its sign convention is tied to the fixed ambient coordinate order. Thus it is not a full Lorentz-invariant simplex volume and can be interpreted as a fixed coordinate-dependent assignment heuristic. The proposed odd/even column-permutation invariants concern the determinant calculation, not a transformation the ordinary assignment system necessarily performs. The claimed non-identity fixture is valid only if the candidate enumeration actually supplies the constructed tied alternatives.

### Minimal revision required

1. Narrow the numerical domain to inputs for which every specified norm-check and determinant intermediate is finite, or specify a safe bounded-coordinate domain and demonstrate that its binary64 operations cannot overflow. Keep the fail-closed rule for inputs outside it; do not claim a scalar for all finite binary64 inputs.
2. State explicitly that `A` is a coordinate-dependent 4-by-4 minor and not a Lorentz-invariant volume; retain fixed-coordinate dependence as part of the mechanism.
3. Do not claim smooth epsilon regularization, a flat limit, or curvature discrimination. None is in this proposal.

With these corrections, the determinant and discrete tie-break are mathematically coherent for the admitted domain. The strict experimental thresholds and novelty/overlap audit remain empirical gates, not conclusions of this review.
