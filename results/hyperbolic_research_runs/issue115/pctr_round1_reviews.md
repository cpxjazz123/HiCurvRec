# PCTR -- JOIN R round 1, the three reviews

### math_review.md

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

### invariants_review.md

# Measurability review — candidate.md

## Verdict: REVISE

I cannot approve the stated measurement plan from the supplied declaration. `candidate.md` does **not** name `assignment_audit.emit`, identify its call site or written artifact, or state the proposed flip-rate, tie-share, and clamp-rate thresholds. Its §AUDIT instead gives a broad list of operands and counts that an audit should emit. No implementation or artifact schema is present here. Therefore a reviewer cannot verify the mechanism's effect from the named artifact alone—the named artifact is not declared in this document.

## Scope and document-grounded reading

I reviewed the single mechanism declared in `candidate.md`; I did not assume a second candidate, revision, or withdrawn proposal. The actual declaration is PCTR: within-prefix candidate ordering uses the signed tangent-space residual/mean inner-product term in its displayed score (§§OPERATOR, FINITE-PRECISION DEFINITION). It is not an unsigned 2-simplex-area operator. The document also explicitly defines a score consisting of the frozen distance plus that residual-consistency contribution. I do not substitute an unsigned area or review other, absent mechanisms. If the intended declaration is an unsigned 2-simplex area, `candidate.md` must be reconciled with that requirement before this review can approve it.

The frozen thresholds are recorded in §PRE-REGISTERED FALSIFIERS WITH NUMBERS: Hard-L1 ER, Recall@10, and NDCG@10 must each strictly exceed their stated values. The declaration separately requires at least 1.00% changed candidate comparisons and specifies a non-finite/domain-error tolerance. Those are not the flip/tie/clamp criteria in the review request, and the document does not establish that the requested `assignment_audit.emit` artifact checks them.

## Why the proposed artifact is not yet decisive

Even if the prose list in §AUDIT were emitted, it does not specify a machine-checkable, self-contained record sufficient to determine the requested falsifiers without rerunning:

1. **No artifact contract or provenance.** Specify the exact artifact path/format/schema, run and split identifiers, `assignment_audit.emit` call site and invocation conditions, code/config/data hashes, and a completion marker/count reconciliation. Define behavior on exceptions and partial writes. A list of desired fields is not evidence they were actually emitted.
2. **No explicit flip denominator or per-level flip table.** Define the comparison unit (sample, candidate pair, or assignment), the eligible population, treatment of absent/empty candidate sets, and the numerator/denominator per level. Emit baseline and candidate integer assignments and enough before/after ranking data to independently recompute flips. The requested threshold is at least 5% **per level**; a pooled rate can conceal a failing level. Candidate §53 instead states a 1.00% threshold on same-prefix comparisons.
3. **No tie-share definition/evidence.** Define whether ties are score ties under the declared `1e-7` tolerance, ties in selected assignments, or another event; state denominator and per-level aggregation. Emit the raw score differences and tie-resolution path for all relevant comparisons so a reviewer can recompute the share and apply the requested `<=10%` bound. Merely recording a “tolerance branch” is not enough without a clear tie-share statistic and its population.
4. **No clamp-rate accounting.** The declaration has norm truncation at epsilon and a `max(1, ...)` in the arcosh argument, but §AUDIT does not enumerate clamp events or define a clamp rate. Record each clamp type, count, affected comparison/sample IDs, and its denominator; specify how the combined rate is calculated. The requested failure rule is clamp rate `>0.1%` (including the exact boundary convention).
5. **Operand recoverability is underspecified.** “Actual float32 values or a verifiable lossless hash with source-file location” is not sufficient unless the artifact also makes those source files available and immutable. For verification without rerunning, include the actual operands/results or a content-addressed, accessible payload plus a deterministic decoding/schema contract. Hashes alone cannot reveal values or establish the computed score.
6. **Outcome evidence is incomplete for the performance thresholds.** §AUDIT asks for paired ER scores and Stage3 raw counts/configuration, but does not define the exact evaluation population, aggregation, metric implementation/version, or provide the actual results. The artifact must contain the per-example/per-pair inputs or sufficient raw counts and metric definitions to independently recompute all three metrics and compare strictly against the frozen thresholds. Audit evidence alone cannot establish that the candidate passed them.
7. **Control/identity and training-statistic traceability.** Emit a directly comparable lambda=0 control and candidate audit over identical examples, with exact SID streams and a verifi
### priorart_review.md

# Prior-art / novelty review — candidate.md

## Verdict

**REVISE — ADJACENT; do not make a novelty claim yet.** The available campaign record identifies HiHPQ (DOI [10.1609/aaai.v38i5.28261](https://doi.org/10.1609/aaai.v38i5.28261), arXiv:2401.07212) as the only verified match, but only at **ABSTRACT_ONLY** depth. That is not enough to establish that HiHPQ discloses the particular operator, nor enough to establish that it does not. A claim that an unsigned triangle-area quantiser-selection term is novel is therefore **not presently supportable**.

## Scope and declaration check

I treated `candidate.md` as the complete declaration, as requested. There is an important mismatch between the review prompt and that file: the file does **not** declare an unsigned 2-simplex/triangle-area operator. Its stated candidate is PCTR: a signed tangent-space residual–group-mean inner-product term in the score (candidate.md, §§OPERATOR, INVARIANTS THAT FAIL FOR THE IDENTITY). It explicitly distinguishes itself from an unsigned tangent-space wedge area (candidate.md, §WHY NOT THE SIX CLOSED FAMILIES). I found no second candidate or second score term in the file.

Accordingly, the answer to “is an unsigned triangle-area term used to choose a quantiser code already occupied?” is **not established by this candidate or the verified scan record**. I will not attribute that area operator to candidate.md or silently review it as if declared. If the intended operator really is unsigned triangle area, the complete declaration needs to be corrected to say so before that mechanism can receive a substantive review. If candidate.md is authoritative, the mechanism to review is the stated signed inner-product reranking instead.

## Evidence status

- **Verified from the supplied materials:** `candidate.md` describes the PCTR signed inner-product score and says the unsigned tangent-space wedge-area family is excluded. The supplied `frozen_artifacts.json` records control statistics and shapes, not prior-art evidence. No other local prior-art report or source was present in the workspace.
- **Verified campaign-scan result as supplied in the request:** HiHPQ is the sole match, with the scan explicitly limited to **ABSTRACT_ONLY**. This supports treating HiHPQ as relevant adjacent art to check; it does not verify a triangle-area code-selection operator in HiHPQ.
- **Unknown:** Whether HiHPQ's full paper, supplementary material, code, or cited work uses an unsigned triangle-area term to choose a quantiser code in a recommender SID. The supplied abstract-only status cannot settle the operator-level comparison.
- **No recollection used as evidence:** I have not relied on memory of HiHPQ or any other paper to fill in the missing operator details.

## Novelty assessment and required next check

No “novel” claim should be made on this record. This is a limitation of the evidence, not a finding that the area operator is already occupied. Nor does the record justify a confident “not occupied” claim. The defensible status is **ADJACENT / unresolved**, pending (1) reconciliation of the unsigned-area prompt with the signed-inner-product declaration in `candidate.md`, and (2) full-text and, where available, code-level review of HiHPQ and relevant references for the exact operator and its use in quantiser-code selection for recommender SIDs.

This is a scoped technical prior-art screen, not a legal patentability opinion.
