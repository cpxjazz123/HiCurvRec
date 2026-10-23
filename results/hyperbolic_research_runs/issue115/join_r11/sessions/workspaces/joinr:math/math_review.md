# Mathematical review — candidate v7

## Verdict: REVISE

The unsigned Gram-determinant area is mathematically coherent on the stated domain, including the listed degeneracies. However, the training surrogate is inconsistent between §§2 and 4b: as written in §2 it converges to twice the declared area, not to the area. The flat-limit notation and the inference attached to its control also need correction. Minimal revisions are listed below. No claim about a signed area or a second score term is being reviewed; neither occurs in `candidate.md`.

## 1. Operator and degeneracies

With `u,v_k` in the same tangent space equipped with the same positive-definite Riemannian inner product,

\[
D_k=\|u\|^2\|v_k\|^2-\langle u,v_k\rangle^2\geq 0,
\qquad A_k=\tfrac12\sqrt{D_k}
\]

is a well-defined nonnegative scalar. It is zero exactly for linearly dependent vectors, including either zero vector, and is positive otherwise. Thus `q=p`, `z_k=p`, and `u∥v_k` are finite and produce zero area; no division by the area occurs. The distance denominator is positive. The tie-breaking statements are consistent with the score (though when `u=0`, the distance factor ranks codes by distance, with the stated index rule for exact ties).

The boundary-clamp statement is implementable only if the clamped arguments still yield valid points for each logarithm; it should specify whether the clamp is applied to the point, the Möbius/log-map intermediate, or both. This does not invalidate the Gram formula, but it matters to the claimed domain and clamp audit.

## 2. Epsilon surrogate and gradients

There is a factor-of-two error. Section 2 gives

\[
\widetilde A_k=\sqrt{D_k+\epsilon^2},
\]

whereas §4b gives `½√(D_k+ε²)`. Only the latter regularizes the declared `A_k=½√D_k` and converges to it. The §2 expression converges to `√D_k=2A_k` for every `D_k>0`; it is not an `O(ε²)` approximation to `A_k`.

Use the consistent definition

\[
\widetilde A_k(\epsilon)=\tfrac12\sqrt{D_k+\epsilon^2}.
\]

For `D_k>0`, the correct bound is

\[
0\leq \widetilde A_k-A_k
=\frac{\epsilon^2}{2(\sqrt{D_k+\epsilon^2}+\sqrt{D_k})}
\leq \frac{\epsilon^2}{4\sqrt{D_k}}.
\]

The §4b bound with denominator `8√D_k` is too strong by a factor of two. On `D_k=0`, the corrected surrogate is `ε/2` and tends to zero. For fixed positive ε it is smooth as a function of `u,v_k` and has finite gradients; the unsmoothed area is Lipschitz but generally nondifferentiable on the zero-area set. The finite-ε surrogate is not exactly zero there, so the training score must explicitly use this same corrected convention, while the audit may separately report the unsmoothed area as proposed.

## 3. Flat limit and control

The limiting Gram expression is right, but `×` is not a defined vector cross product in the declared 32-dimensional latent space. Write the dimension-independent expression instead:

\[
A_k\longrightarrow \tfrac12\sqrt{\|q-p\|^2\|z_k-p\|^2-
\langle q-p,z_k-p\rangle^2}
=\tfrac12\|(q-p)\wedge(z_k-p)\|.
\]

Consequently the stated flat-limit score is algebraically correct when its numerator is understood as the exterior-product norm (not a 32-D cross product). The `c→0` limit also presumes the points/scaling remain in a regime where the maps and parameters have a well-defined flat limit.

Matching assignments in one flat-control run does **not** establish that the mechanism is a re-parameterisation: distinct scores can induce the same argmin on a finite dataset, and assignment equality is not equality of mechanisms. It can be a useful empirical falsifier for this dataset, but the conclusion should be limited to “no assignment change under this control”; it cannot logically establish re-parameterisation. Compare score rankings/margins or evaluate a pre-specified distribution of inputs if the intended claim is broader.

## 4. Capacity and metric identity

The score introduces no learned parameters, so the no-added-parameter statement is correct. The area is not a function of `d_c(p,z_k)` alone: it depends on the target direction `u` as well as the candidate vector `v_k`. Thus it is not merely a swap of the host-to-code distance in the ordinary sense. But “not a metric swap” is too categorical without defining the comparison class: the score is a target-conditioned dissimilarity/reranking rule, and one could describe it as replacing the assignment dissimilarity with a non-metric score. State the narrower, verifiable claim: it adds no learned capacity and is not representable as a function of host–code distance alone in general.

## Minimal required edits

1. Change §2 to `Ã_k = ½·sqrt(D_k + ε²)` and use this same definition for training and §4b.
2. Change the §4b error bound from `ε²/(8√D_k)` to `ε²/(4√D_k)`.
3. Replace the 32-D `×` notation in §3 with the Gram/exterior-product norm above.
4. Reword the flat-control conclusion so assignment matching is treated as an empirical no-change result, not proof of re-parameterisation; narrow the “not a metric swap” statement as above.

After these mathematical corrections, the mechanism is reviewable for JOIN_E; this verdict is not experimental approval and does not waive the frozen conjunctive thresholds or required audit artifact.
