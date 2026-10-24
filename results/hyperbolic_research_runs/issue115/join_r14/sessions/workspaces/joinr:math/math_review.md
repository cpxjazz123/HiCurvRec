# Mathematical review of `candidate.md`

## Verdict: REVISE

The single declared operator is reviewed here; there is no signed-area term or second score term in `candidate.md`. The frozen protocol and conjunctive thresholds do not affect the mathematical findings below.

The formula is a valid scalar **at each valid input**, but it is not the area of a geodesic 2-simplex in the curved host. Its main mathematical defect is therefore a mismatch between the named geometric object and the quantity actually computed. There are also inconsistencies in the tie-breaking specification and in the stated interpretation of the flat control. These are minimal, document-level revisions; they do not require a different candidate mechanism.

## Findings

1. **Scalar and degeneracies.** For `c>0`, valid points strictly inside the Poincaré ball, and the Riemannian inner product on `T_p`, `u=log_p(q)` and `v=log_p(z_k)` are finite tangent vectors. Cauchy–Schwarz gives
   `D=||u||²||v||²-<u,v>² >= 0`, so `A=½√D` is a well-defined nonnegative scalar. It vanishes exactly for linearly dependent vectors, including `u=0`, `v=0`, and the general collinear case. Thus the listed zero/collinear degeneracies are algebraically handled without division. This assumes the inner product is the host's tangent-space metric; the document should say so explicitly. In floating-point arithmetic, the Gram subtraction can produce a small negative value despite the exact inequality, so a numerical clamp or stable wedge-norm evaluation is needed to guarantee no NaN.

   **But:** `½√D` is the area of the Euclidean triangle formed by the two tangent vectors in `T_p` (equivalently, half their wedge norm). In nonzero curvature it is not, in general, the area of the geodesic triangle with vertices `p,q,z_k`. The logarithm map does not preserve areas. Calling it the “unsigned 2-simplex area the two geodesics ... span” is therefore false as written. Either rename it consistently as *tangent-space wedge area* (the minimal fix, matching the flat-limit formula), or replace the formula with the actual curved geodesic-triangle area and redo the limit/gradient analysis.

2. **Regularization and gradient.** With `ε>0`, `Ã=½√(D+ε²)` is well-defined. Its difference from `A` is exactly `ε²/[4(Ã+A)]`; it is bounded by `ε/2`, converges pointwise to zero, and for `D>0` is at most `ε²/(4√D)`. Hence the stated convergence and off-degenerate `O(ε²)` claim are correct (with fixed positive `D`). `D` is smooth in its vector arguments and `Ã` is smooth for positive epsilon. The unregularized wedge norm is locally Lipschitz and generally nondifferentiable where the wedge vanishes; this supports the stated Lipschitz/nondifferentiability claim. Add that training requires `ε>0`; at `ε=0` the surrogate does not smooth the degeneracy. Also specify the units of epsilon: since `D` has units of tangent-length to the fourth power, `ε` has units of tangent-length squared.

3. **Flat limit and control.** Under a finite-point, fixed-coordinate flat limit, the logarithms tend to `q-p` and `z_k-p`, and the distance tends to Euclidean distance. The displayed limiting expression is algebraically correct as a tangent wedge-area score. It is not a limit to curved geodesic-simplex area (which is the naming issue above). The control can test whether curvature changes assignments, provided its implementation specifies how `p,q,z_k` are held/scaled as `c→0`; that is not fully specified by merely saying “everything else identical.” The prose is internally contradictory: §3 calls agreement between flat and curved arms a necessary condition for non-triviality, while the decisive criterion and §6 reject when the flat arm reproduces assignments. Agreement is not the needed condition; **non-reproduction** is the stated discriminating criterion. Correct the sentence to say that the control tests whether curvature changes decisions, and that the mechanism passes this discriminator only if the flat arm does not reproduce them.

4. **Capacity, metric swap, and assignment rule.** The score introduces no learned parameters, so the no-added-learned-capacity statement is correct. The area is not a function of `d_c(p,z_k)` alone, so it is not reducible to a distance-only score on that same geometry. However, “not a metric swap” is too categorical: this is a hand-designed, target-dependent dissimilarity used for assignment, not a learned representation or a new metric, and no experiment can establish a broad distinction from every possible metric-based reformulation. State the narrow claim: it is not a distance-only replacement because it depends jointly on the target direction and candidate direction.

   Tie handling also needs one consistent definition. §2 specifies `(A_k,d_c(p,z_k),k)` for degeneracies, whereas schema 2 specifies exact minimum-full-score ties and `(margin, lowest index)`. These are not the same rule; moreover, the §2 rule is only stated for degeneracies while ordinary equal-score minima are not resolved. Define one deterministic tie rule for every assignment and use it in the implementation and artifact. The artifact's per-example key must report that same rule. Do not describe a tie in score as necessarily a tie in area.

## Minimal required revision

1. Rename the operator throughout as **unsigned tangent-space wedge area** (or compute actual geodesic-simplex area); explicitly identify the Riemannian tangent inner product and the stable numerical treatment of `D`.
2. Make the tie-breaking rule total and identical in §§2 and 8, including for non-degenerate equal-score minima.
3. Correct the flat-control interpretation and specify the coordinate/scaling convention for the `c→0` arm.
4. State `ε>0` during training and its units; narrow the metric-swap claim as above.

With these corrections, the scalar formula, its regularization limit, and the stated thresholds can be evaluated without changing the frozen protocol. Until then, the geometric claim and the audit's assignment semantics are not sufficiently well-defined for approval.
