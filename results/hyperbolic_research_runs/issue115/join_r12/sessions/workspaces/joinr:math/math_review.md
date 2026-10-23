# Mathematical review

## Verdict: REVISE

The declared unsigned 2-simplex operator is mathematically coherent, but the document overstates the surrogate error bound and the evidentiary force of the flat control. These are minimal, local corrections; they do not require a second mechanism or a change to the frozen protocol.

## 1. Definition and degeneracies

For finite tangent vectors in the same inner-product space,
\[
D=\|u\|^2\|v_k\|^2-\langle u,v_k\rangle^2\ge 0
\]
by Cauchy–Schwarz, so `A_k = ½√D` is a well-defined nonnegative scalar. It is zero exactly when either vector is zero or the vectors are linearly dependent; otherwise it is positive. Thus the four listed cases are covered: `q=p`, `z_k=p`, collinearity, and near-boundary evaluation (provided the stated clamps produce finite tangent vectors). The lexicographic tie rule makes the assignment defined when the area score ties, including `q=p` and `z_k=p`.

Two qualifications: “no NaN” is a mathematical property only in exact arithmetic; a direct subtraction of nearly equal Gram terms can produce a small negative floating-point value. An implementation should use a stable wedge/norm computation where feasible, or clamp `D` to `max(D,0)` with a recorded numerical-error policy. Also, clamping the arguments of `log_p` changes the declared geometric quantities; the resulting score remains computable, but it is a clamped approximation, not the exact area for the original points.

## 2. Regularisation and gradients

For fixed `ε>0`, `Ã=½√(D+ε²)` is smooth as a function of the finite tangent-vector components, and its gradient is finite. It converges pointwise to `A` both off and on the degenerate set; on that set it equals `ε/2`. The stated convergence order `O(ε²)` for fixed `D>0` is correct.

The explicit bound in §4b is not correct. Rationalising gives
\[
|Ã-A|=\frac{\epsilon^2}{2(\sqrt{D+\epsilon^2}+\sqrt D)}
\leq \frac{\epsilon^2}{4\sqrt D},\qquad D>0,
\]
not `ε²/(8√D)`. Replace the bound with this one. The bound is not uniform as `D→0`; convergence is nevertheless valid, with error `ε/2` at `D=0`.

The nonsmoothed area is locally Lipschitz (it is a norm of the wedge/bivector) but is generally not differentiable where the wedge vanishes. The finite-gradient claim applies to the fixed-ε surrogate, not uniformly as `ε→0`; its gradients can become poorly conditioned near degeneracy. State the value of ε and how it is scaled relative to tangent-vector units, since `ε` has the units of `√D` (area).

## 3. Flat limit and control

The displayed flat-limit algebra is right, under the stated identification of points as curvature tends to zero: `log_p^c(q)→q−p`, and the Gram determinant becomes the squared Euclidean wedge norm. Likewise `d_c(p,z_k)→‖z_k−p‖`, giving the displayed limit of `S_k`.

The inference about the control is too strong as written. Matching assignments in the flat and curved arms is a useful falsifier under the registered rule, but it does not establish that the mechanisms are reparameterisations: distinct score functions can induce the same finite argmin partition. Conversely, a difference in assignments establishes only that the realized assignments differ, not why. Keep this as an operational kill criterion if intended, but describe it as a pre-registered assignment-equivalence criterion, not a mathematical proof of reparameterisation. Specify whether “match” means exact equality of all per-example winners and how ties are handled.

## 4. Capacity and metric identity

The formula introduces no learned parameters; it changes the assignment rule and uses the already-defined target `q`. The area is not a function of `d_c(p,z_k)` alone in general: it depends on the direction of `log_p(z_k)` relative to `log_p(q)`. Hence the stated distinction from a score based solely on host-to-code distance is mathematically defensible. It is still an additional target-conditioned scoring signal, even without learned capacity. “Not a metric swap” should be limited to that precise claim; it does not follow merely from calling distance a tempering factor.

## Minimal required edits

1. Correct the §4b bound to `ε²/(4√D)` (or give the exact rationalised expression), and qualify the gradient-conditioning statement near `D=0`; specify ε and its units.
2. Qualify the flat-control interpretation as an assignment-level falsifier, define assignment matching/tie handling, and remove the claim that matching by itself proves reparameterisation.
3. Qualify “no NaN” as an exact-arithmetic claim and specify a stable/nonnegative floating-point handling policy for the Gram determinant.
