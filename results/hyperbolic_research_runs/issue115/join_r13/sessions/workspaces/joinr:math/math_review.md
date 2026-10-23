# Mathematical review — candidate v7

## Verdict: REVISE

The declared mechanism is the unsigned 2-simplex area, tempered by distance, exactly as written in `candidate.md`. I find no signed-area operator or second score term in that document; I do not review either.

The Gram-determinant expression is mathematically sound on its intended finite domain, and the epsilon surrogate and flat-limit calculation are essentially correct. However, the artifact schema does not contain enough information to recompute assignments or the claimed ablation/control comparison, and two geometric claims need precise qualification. These are minimal, fixable revisions; they do not require changing the mechanism.

## 1. Scalar and degeneracies

For real tangent vectors with a positive-definite Riemannian inner product,

\[
D=\|u\|^2\|v_k\|^2-\langle u,v_k\rangle^2\geq 0
\]

by Cauchy–Schwarz, so \(A_k=\tfrac12\sqrt D\) is a real, nonnegative scalar. The four listed cases behave as follows:

* **\(u=0\):** \(D=A_k=S_k=0\) for every code. The stated lexicographic tie-break is well-defined if distances and indices are finite.
* **\(v_k=0\):** likewise \(D=A_k=S_k=0\); the stated rule applies.
* **Collinear vectors:** including antiparallel vectors, \(D=0\) exactly and there is no division by zero in the exact formula.
* **Near-boundary inputs/clamping:** clamping can make the logarithm finite only if the implementation defines what is clamped (the point argument, tangent norm, or an intermediate denominator) and applies the same convention consistently. The current phrase “every `log` … `‖·‖ ≤ …`” is not an operational definition of the clamp. Also state explicitly that all targets and code points are in the open ball before logarithms are taken; a post-hoc clamp should not silently redefine the geometric input.

There is an additional dimensional notation issue: in 32 dimensions, \((q-p)\times(z_k-p)\) is not an ordinary vector cross product. It must mean the norm of the exterior/wedge product (equivalently the square root of the Euclidean Gram determinant). The candidate's preceding Gram expression supplies that meaning, but the flat-limit line should use \(\|(q-p)\wedge(z_k-p)\|\), not `×`.

## 2. Epsilon surrogate and gradient

With \(\widetilde A=\tfrac12\sqrt{D+\epsilon^2}\), \(\epsilon>0\), the stated identity is correct:

\[
\widetilde A-A=\frac{\epsilon^2}{4(\widetilde A+A)}.
\]

Since \(\widetilde A+A\geq\epsilon/2\), the error is at most \(\epsilon/2\). For \(D>0\), it is also at most \(\epsilon^2/(4\sqrt D)\), and for fixed nondegenerate inputs it is \(O(\epsilon^2)\). It converges pointwise to \(A\) everywhere, including the degenerate set (where the surrogate is \(\epsilon/2\)). The convergence is not uniformly second-order near degeneracy, as the text correctly cautions.

The exact area is a norm of the bivector \(u\wedge v_k\), hence locally Lipschitz, and is generally nondifferentiable where that bivector is zero. The regularized form is differentiable in the bivector and removes that norm cusp for fixed positive epsilon. This does **not** establish that the complete training objective has a useful or everywhere well-behaved gradient: the log map, projection/clamp, and hard `argmin` assignment can still be nonsmooth or have zero/undefined gradients depending on the actual training estimator. Narrow the gradient claim to the area subexpression, and specify how assignment gradients are handled. Also report the value of epsilon and its units/scaling; adding \(\epsilon^2\) to a squared-area quantity requires epsilon to have area units.

## 3. Flat limit and control

For fixed finite tangent inputs (and fixed scale), as \(c\to0\), \(\log_p^c(y)\to y-p\) and \(d_c(p,z)\to\|z-p\|\). Thus the limit is

\[
S_k\to\frac{\tfrac12\|(q-p)\wedge(z_k-p)\|}{1+\|z_k-p\|},
\]

so the algebra is right when the cross-product notation is corrected. Clarify that the control varies curvature while holding the underlying coordinates, scale, and all other score conventions fixed; otherwise “everything else identical” is ambiguous because ball radii and projection conventions depend on curvature.

The proposed control can discriminate curvature-dependent assignments from the flat-limit assignments, but it is not by itself a proof of a specifically hyperbolic mechanism. A finite-curvature arm and a limiting flat arm can differ because of coordinate mapping, boundary projection, or numerical clamps as well as curvature. More importantly, the schema only records winners for the full and area-zero scores; it records no flat-arm winners. It therefore cannot verify the stated joint decision rule that the flat arm “does not reproduce those same decisions.” Define “reproduce” (per-example equality or an aggregate threshold) in advance and emit the flat-arm winners and relevant threshold in the audit.

## 4. Capacity and metric identity

The formula introduces no trainable parameters, so the narrow statement “adds no parameters” is correct. It does change the assignment rule and thus can change the induced partition/code usage; “no learned capacity” should not be used to imply “no additional modeling choice.”

The score is not a function of geodesic distance alone: at fixed lengths, the Gram determinant varies with the angle between the two tangent vectors. Consequently it is not literally a metric substitution of the form \(f(d_c(p,z_k))\). Still, “not a metric swap” is underspecified unless the comparator is named. The area uses the same geometry and may be correlated with distance; the evidence needed is the assignment ablation/control, not the assertion that the quantities are algebraically distinct.

## Required minimal revisions

1. Replace the 32-D cross-product notation with the wedge/Gram norm and make the clamp and valid-input convention operational.
2. Limit the gradient claim to the area subexpression; specify epsilon (including units) and the hard-assignment gradient treatment.
3. Define the flat-control comparison and add flat winners to the audit.
4. Make the audit genuinely recomputable: as specified, it lacks per-code scores/components, `margin_runner_up` despite mentioning runner-up margins, tie details, and flat winners. Per-example winning indices plus one winning area/distance cannot establish that those indices minimize the declared score over all 256 candidates. Emit enough per-candidate inputs or scores (with deterministic tie handling) to verify the winners, ablation, and control without rerunning.

These changes preserve the single declared mechanism and frozen protocol. Until they are made, the math is plausible but the promised auditable evidence is not sufficient for approval.
