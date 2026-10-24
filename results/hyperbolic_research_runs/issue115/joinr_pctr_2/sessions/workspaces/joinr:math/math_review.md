# Mathematical review

## Verdict: REVISE

**Not approved for JOIN_E as described in the request.** The staged `candidate.md` does not declare the unsigned 2-simplex area operator

\[
A_k=\tfrac12\sqrt{\|u\|^2\|v_k\|^2-\langle u,v_k\rangle^2}.
\]

It declares PCTR: a within-prefix score correction using a **signed, normalized batch-group residual inner product** (`candidate.md` §§13–21). It explicitly says that PCTR does *not* use an unsigned tangent wedge/2-simplex area (§39). There is no area epsilon-regularization, area flat-limit argument, or four area degeneracies specified in this candidate. I therefore do not attribute those claims to the document or certify an area mechanism from it. The request's frozen protocol description also does not match the candidate's stated SID shape: the candidate/frozen artifact describe three scored levels in the diagnostics but the artifact lists `level0`–`level3`; this should be reconciled before a run.

## Requested area-operator questions

1. **Scalar and degeneracies:** In exact real inner-product arithmetic, the radicand is nonnegative by Cauchy–Schwarz, so the formula defines a finite nonnegative scalar for finite vectors, including zero vectors and linearly dependent vectors (where it is zero). It is not, however, a robust finite-precision definition as written: cancellation can make the radicand slightly negative, making the square root undefined. The document supplies neither an explicit clamp/error policy nor a stable wedge-norm evaluation. Since the four degeneracies are not listed in `candidate.md`, I cannot verify any document-specific handling for them.
2. **Epsilon form and gradients:** No epsilon-regularized area expression is present, so its correctness, convergence, and claimed gradient cannot be audited from this declaration. For the unregularized area, differentiability fails at zero area; away from that set the usual derivative exists. Any regularization must be stated explicitly, including where epsilon is inserted, and its gradient must be checked against that exact expression.
3. **Flat limit and control:** No area flat-limit algebra or `c→0` control is declared. Neither can be approved. The candidate's actual signed normalized product is a different operator; its behavior cannot establish an area flat-limit control.
4. **Capacity / metric swap:** This cannot be answered for the absent area score. For the actual PCTR declaration, it adds no learned parameters, but it explicitly adds batch-dependent functional scoring capacity (§21). It is not merely a fixed metric swap: its group mean depends on the current batch's discrete memberships and residuals, and can change within-prefix ordering. The cosine-like term can become insensitive to overall `c` when both norms exceed the floor, while the fixed `1e-6` floors break that invariance near zero. This is a scoring rule, not extra learned embedding capacity.

## Minimal revision

Replace `candidate.md` with a single, internally consistent declaration of the mechanism actually intended for review. If the intended mechanism is the area operator, state its domain and all four degeneracies, give the exact epsilon formula and numerical evaluation policy, derive its off-degenerate-set limit and gradient, show the flat-limit algebra, and define a `c→0` control that changes only the intended geometry. Also reconcile the SID-level count with the frozen protocol. If PCTR is intended, correct the review request to PCTR and do not describe it as an unsigned simplex area. Until that mismatch is resolved, the specified area mechanism is not auditable and cannot be approved.
