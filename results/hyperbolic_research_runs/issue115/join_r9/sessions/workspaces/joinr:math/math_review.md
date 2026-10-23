# Mathematical review of `candidate.md`

## Verdict: REVISE

The proposed unsigned Gram area is a well-defined nonnegative scalar for finite tangent vectors, including the zero and collinear cases. However, the declared training surrogate is not a regularisation of that area: it omits the factor `1/2`. There are also contradictions in the degeneracy/tie-breaking description and an unstated requirement that code points lie in the Poincaré ball. These are minimal, repairable issues; no other mechanism is reviewed here.

## 1. Operator and degeneracies

For Euclidean tangent vectors `u,v` at the same point, Cauchy–Schwarz gives

`G = ||u||² ||v||² - <u,v>² ≥ 0`,

and `A = (1/2) sqrt(G)` is defined for every finite pair. It is zero if either vector is zero or they are linearly dependent, and positive otherwise. Thus the four stated cases (`u=0`, `v=0`, collinearity, and the boundary-handling case) do not require division and are mathematically well-defined, provided the logarithms themselves are defined.

There are two problems with the written degeneracy account:

* Since `A ≥ 0` and `1+d > 0`, `S ≥ 0`, not `S ≤ 0`. If `z_k=p`, then `S_k=0`; under the stated `argmin`, this is a minimum (possibly tied), so it is not true that the distance term decides. The score contains no distance-only tie-break. Likewise, when `q=p`, all scores are zero, so the stated distance tie-break does not follow from `argmin S`.
* `z_k` is declared to be in `R^32`, but `log_p^c(z_k)` is only defined when `z_k` is a point in the Poincaré ball (or is first mapped/projected there). State that domain/mapping explicitly. The clamp on the norm of a logarithm does not by itself make an out-of-ball argument valid.

## 2. Training surrogate and gradients

The declared surrogate is

`Ã = sqrt(G + ε²)`,

whereas the proposed score uses `A=(1/2)sqrt(G)`. For every fixed `G>0`, `Ã → sqrt(G)=2A` as `ε→0`; it does **not** converge to `A`. The claimed `O(ε²)` agreement is therefore false as written. The minimal correction is

`Ã = (1/2) sqrt(G + ε²)`.

With that correction, for fixed `G>0`, `Ã-A = ε²/(2(sqrt(G+ε²)+sqrt(G)))`, hence the difference is `O(ε²)`. The corrected surrogate is smooth in `u,v` for `ε>0`, since its radicand is strictly positive. At a degenerate pair it has a nonzero smoothing value `ε/2`, and its gradient is finite; it does not reproduce the nonsmooth exact area there. The exact norm/area is Lipschitz but generally not differentiable at zero area, so that part of the gradient description is sound. A numerical implementation should also guard the computed Gram determinant against small negative roundoff; the algebraic nonnegativity alone does not prevent floating-point NaNs.

## 3. Flat limit and control

The flat-limit algebra is correct under the stated common tangent-space identification: `log_p → (·−p)`, and the Gram determinant becomes the squared Euclidean wedge norm. Thus the limit of the score is the displayed half-wedge area divided by `1+||z_k−p||`.

The control is a valid empirical falsifier if curvature is the only changed factor and the arm is actually specified/implemented at a finite near-zero curvature (or by the exact Euclidean limit). But “if its assignments match, the mechanism is a re-parameterisation” is too strong as a mathematical inference: matching on the evaluated examples establishes observational agreement on those examples, not equivalence of the scoring rules over the domain. Define the matching criterion and report the comparison; describe the conclusion as failure of the registered discrimination test, rather than proof of re-parameterisation.

## 4. Capacity and metric identity

The score adds no learned parameters, so the no-added-parameter claim is correct. It does change the assignment rule. The area is not determined by the single scalar distance `d_c(p,z_k)`, so it is not merely a scalar distance rescaling of that distance. But “not a metric swap” is not established by that observation: the score is a nonnegative, pair/target-dependent function and is not asserted to be a metric. If “metric swap” means replacing the host distance with another point-to-code metric, the proposal needs a precise definition and a control to rule that out; otherwise narrow the claim to “not a distance-only score.”

## Minimal required edits before approval

1. Put the `1/2` factor in the regularised form and correct the convergence statement.
2. Correct the tie behavior: specify an explicit tie-break rule if distance is intended, and say how zero-area codes are treated under minimisation.
3. Specify how every `z_k` is guaranteed to be in the ball before applying `log_p^c`.
4. Weaken the flat-control and metric-swap conclusions as above (or define their tests precisely).

After these edits, the core area operator is mathematically coherent. This review does not certify the frozen empirical thresholds or the required audit artifact.
