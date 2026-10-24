# Mathematical review — candidate v7

## Verdict: REVISE

The unsigned tangent-space wedge area is mathematically sound, and the regularised formula and flat-limit algebra are correct under the stated conventions. The proposal is not yet a fully defined assignment operator for every input: it specifies how to break ties in the listed zero-area cases, but not ties between equal scores in the general case. The minimal fix is to make the stated lexicographic ordering `(A_k, d_c(p,z_k), k)` the tie-break for *every* exact score tie (or specify another deterministic tie rule). No second mechanism or signed area is being reviewed.

## Findings

1. **Scalar and degeneracies.** For finite tangent vectors, the Gram matrix is positive semidefinite by Cauchy–Schwarz, so its determinant is nonnegative in exact arithmetic. Thus `A_k = ½√D_k` is a well-defined, nonnegative scalar. It is zero when either vector is zero or the vectors are linearly dependent, including the listed `q=p`, `z_k=p`, and collinear cases; otherwise it is positive. The specified lower clamp handles floating-point negative round-off, though it cannot make an invalid/non-finite logarithm valid. The near-boundary input clamp makes the evaluated quantity a clamped implementation of the operator, not the unclamped mathematical logarithm; its reported clamp gate appropriately exposes that distinction.

   The listed degenerate tie rule is internally consistent: all zero-area candidates have score zero, and `(A,d,k)` then selects the smallest distance and index. However, equal `S_k` values can also occur outside those cases, and `argmin` alone does not define a unique winning code. Extend the tie rule to all exact score ties for a total, reproducible operator.

2. **Regularisation and gradients.** With `D≥0`, `Ã=½√(D+ε²)` is the correct epsilon regularisation for `ε>0`; `ε` has the units of `√D`, i.e. squared tangent length. It is smooth as a function of the operands wherever those operands are smooth, including at `D=0`. At `ε=0` it reduces to the original nonsmooth norm. The exact error identity
   `Ã−A = ε²/[4(Ã+A)]`
   is correct; `0≤Ã−A≤ε/2`, it converges pointwise to zero, and for `D>0` it is bounded by `ε²/(4√D)`. The claimed `O(ε²)` off the degenerate set is therefore correct (locally for fixed `D>0`). The original area is locally Lipschitz and generally nondifferentiable where the wedge vanishes; the surrogate removes that norm singularity for positive epsilon. This does not guarantee smooth gradients through any separate clamp/projection operations.

3. **Flat limit and control.** Given the stated fixed-coordinate convention, `log_p^c(q)→q−p`, `log_p^c(z_k)→z_k−p`, and `d_c(p,z_k)→‖z_k−p‖`; hence both displayed flat-limit expressions, including the factor `½`, are algebraically right. The control is a meaningful comparison of the same coordinate rows under the stated flat metric, but it is not a clean curvature-only intervention on a fixed embedding procedure: the points are deliberately not re-embedded as `c` changes. The text acknowledges this limited interpretation. A flat arm that reproduces assignments is a reasonable falsifier under the preregistered rule; a differing assignment alone does not prove that curvature, rather than the changed distance/geometry jointly, is the useful cause.

4. **Capacity and identity.** The formula adds no learned parameters; it changes the assignment rule. It depends jointly on target and candidate directions, so it is not merely a replacement by a target-independent distance between `p` and `z_k`. It can nevertheless be described as a target-conditioned dissimilarity/metric-like score, and the candidate appropriately avoids claiming separation from every possible reformulation. There is no signed area, orientation, frame, or second score term in the document.

## Minimal required edit

In §1 or §2, state that whenever multiple codes attain the minimum `S_k`, choose the lexicographically smallest `(A_k, d_c(p,z_k), k)`, including non-degenerate ties. Keep the existing special-case explanation if desired. With that edit, I would approve the mathematical definition for JOIN_E, subject to the frozen empirical gates and audit requirements.
