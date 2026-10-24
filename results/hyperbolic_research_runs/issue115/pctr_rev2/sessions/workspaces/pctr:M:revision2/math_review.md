# Mathematical review — PCTR

## Verdict: REVISE

The declared candidate is PCTR, not a simplex-area operator. In particular, `candidate.md` contains neither
\(A_k=\tfrac12\sqrt{\|u\|^2\|v_k\|^2-\langle u,v_k\rangle^2}\), nor an epsilon-regularised version of it. It declares a signed residual/mean cosine term added to the frozen distance. The four area degeneracies, area convergence, and area flat-limit calculation therefore do not apply to this document; reviewing them as if present would review a nonexistent mechanism.

### 1. Area definition and degeneracies

Not applicable: there is no area scalar in the declaration. The actual score is
\[
S_{p,k}(x)=D(x,q_{p,k})-\lambda\,
\frac{\langle r_{p,k},\mu_{p,k}\rangle_o}
{\max(\|r_{p,k}\|_o,\epsilon)\max(\|\mu_{p,k}\|_o,\epsilon)}.
\]
For finite tangent vectors and positive \(\epsilon\), its normalized term is defined, including when either vector is zero. This does not establish finite-precision safety for every intermediate: the declaration appropriately errors on non-finite values, but requires implementation checks.

### 2. Regularisation and gradients

There is no regularised area or area-limit claim to validate. For the actual normalized inner product, the two separately clipped norms give a bounded denominator away from zero. When both norms exceed \(\epsilon\), it is the cosine similarity. Below the cutoff it is not a cosine and need not have magnitude at most one; this is a deliberate regularisation behavior that should be stated/tested. The expression is continuous at the cutoff, but generally not differentiable there. The log-map identity branch at \(\theta\le10^{-6}\), the max cutoffs, and discrete assignments/group membership also make the full operation only piecewise differentiable.

The gradient claim is conditionally sound: with group membership held fixed, differentiating through \(r\) and the batch mean \(\mu\) propagates gradients to participating residuals. It does not propagate through the integer assignment decisions. The document should clarify whether the mean includes the current sample and whether its gradient is retained (the current wording implies yes), and verify the actual training graph. If it includes the current residual, singleton groups yield a self-cosine (or its clipped version), an important edge case rather than independent group evidence.

### 3. Flat limit and curvature control

Not applicable: the declaration gives no flat-limit algebra and no \(c\to0\) control. Curvature is the frozen positive Stage2 value. No claim that this candidate discriminates via a curvature/flat-limit comparison can be inferred.

### 4. Capacity and metric distinguishability

No new learned embedding or codebook parameter is introduced. However, “no added learned capacity” is too broad if intended to mean no added functional capacity: the batch-computed \(\mu_{p,k}\), the selected nonzero \(\lambda\), and the resulting batch-dependent reranking add a data-dependent scoring operation. \(\lambda\) is a validation-selected hyperparameter, not a gradient-trained parameter, but it is still a selected degree of freedom.

This is not generally a metric swap. The correction depends on the batch/group mean and can change when other samples in the batch change, so it is not a fixed pairwise distance on \((x,q)\); in general it need not satisfy metric axioms. It is distinguishable from the frozen distance by its within-prefix, batch-dependent signed cosine reranking. The claim that equal frozen distances can be reordered is algebraically plausible when the corresponding correction terms differ.

## Minimal required revisions

1. Keep the review and experiment declaration explicitly about PCTR; do not import an area operator, its degeneracies, gradient claims, or flat-limit claims from another text.
2. Specify the exact grouping/assignment used to form \(\mu\), whether the current sample contributes, and whether gradients flow through that mean; define behavior for singleton groups and groups changing during assignment.
3. Qualify the capacity statement as “no added learned parameters/codebooks,” not as absence of added functional scoring capacity. Record the validation-selected \(\lambda\) as a hyperparameter.
4. Audit the cutoff and identity-branch behavior in the implementation, including gradients at the boundaries and the clipped-normalization regime.

Until those clarifications and the actual computation-graph/finite-precision checks are supplied, the candidate is mathematically plausible as written but not sufficiently specified for approval under the requested mathematical audit.
