# Adversarial prior-art review

## Scope and standard

This review concerns the two operators in `proposals.md`, not whether the prescribed experiments will pass. The frozen 32-dimensional / three 256-entry codebook / three-integer-SID protocol and strict metric gates are unchanged. The supplied `literature.md` is a targeted scan, not a systematic review; its one verified primary-source match is **ABSTRACT_ONLY**. Claims below distinguish that evidence from recollection and from inference. No independent full-text or database search was performed for this review, so absence from the supplied scan is not evidence of priority.

The five already-refuted families (radius-dominated assignment, horospheric level sets, transported-residual frame, role-assigned layout, and Euclidean reformulation) are excluded as candidates; their rejection does not establish novelty of either surviving mechanism.

## Candidate 1 — ordered-simplex signed-area assignment

**Verdict: ADJACENT (provisional; priority unresolved).**

### Prior-art overlap

The verified closest item in `literature.md` is **HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval**, DOI `10.1609/aaai.v38i5.28261`, arXiv `2401.07212`. The scan verifies from its abstract that it has hyperbolic product quantization, hyperbolic codebook attention, and quantized contrastive learning on a hyperbolic product manifold. It is explicitly **ABSTRACT_ONLY**: the available evidence does not show a signed geodesic-triangle-area assignment term. It occupies the broad territory “hyperbolic geometry inside retrieval quantization,” but does not, on the verified evidence, occupy this exact operator. No recommendation-specific instance of this exact operator is verified in the supplied scan.

**Recollection, not verified:** hyperbolic geometry has standard triangle-area/angle-defect constructions, and oriented area or determinant-like terms occur in geometric representation learning. That general mathematical use is not evidence that this precise three-vertex area has been used to select quantization codes. I cannot name a verified recommender, quantizer, or hyperbolic-representation paper implementing this assignment score; status for exact prior art is **UNKNOWN**, not “none.”

### Pipeline-location test

The proposed area is computed from host, target, and each candidate code, then enters the **hard assignment score** alongside squared hyperbolic distance. This is materially different in location from using hyperbolic geometry only in the encoder, decoder, ranking distance, or a contrastive loss. However, moving a familiar geometric quantity from a representation/loss into assignment is not automatically inventive: once a code assignment compares a query residual to candidate code points, a triangle invariant involving those points is a foreseeable scoring feature. The proposal needs evidence that the *oriented* area term changes useful partitions, not just that area is nonzero.

### Is it more than curvature or metric substitution?

Formally, it is distinguishable from a curvature sweep or a metric swap: `-A_c(p,q,z)` is a three-point, ordering-sensitive term, whereas `d_c(q,z)^2` is a two-point distance. It vanishes in the stated fixed-chart flat limit, while the distance term becomes Euclidean. The score is invariant under orientation-preserving isometries but can change under reflection because the ordered area changes sign. This makes it a genuine additional operator, not a reparameterization of the distance, **if implemented as specified**.

That mathematical distinction does not establish a useful or robust novelty. The sign convention makes the area positive for the ordered nondegenerate simplex by construction; swapping the final vertices reverses it. Since the candidate code is one of those vertices, the score rewards one orientation of candidate relative to host-target. Review must test whether this is stable under the model's data symmetries and whether its effect survives learned code placement. The registered area-zero paired arm is necessary; if the decision-level thresholds fail, the deployed mechanism is operationally just the distance control. The flat limit and Gauss–Bonnet check are correctness controls, not novelty evidence.

### Precise difference and closest work

Closest verified work: **HiHPQ** (DOI above), adjacent at the level of hyperbolic retrieval quantization and codebook operations. Difference established by the currently inspected evidence: this proposal selects among codes using an **ordered host–target–code geodesic 2-simplex signed area plus target–code distance**; the HiHPQ abstract does not establish such an area-based assignment. Because HiHPQ full text was not inspected and the literature scan is bounded, the difference is an evidence gap, not a priority finding.

**Kill condition:** locate an earlier paper using oriented geodesic triangle area (or an equivalent angle-defect/oriented-volume term) in vector/code assignment, or show that reflection/data symmetries make the signed preference arbitrary, or fail the preregistered materiality/task gates. On present evidence, do not claim NOVEL.

## Candidate 2 — origin-chart ordered composition with tangent gyration

**Verdict: ADJACENT (provisional; priority unresolved).**

### Prior-art overlap

The closest verified source in `literature.md` is again **HiHPQ**, DOI `10.1609/aaai.v38i5.28261`, arXiv `2401.07212`, **ABSTRACT_ONLY**. It establishes hyperbolic product quantization and codebook attention in image retrieval, but the inspected abstract does not establish ordered Möbius composition or a differential gyration penalty in code assignment. No recommendation-specific exact match is verified in the supplied scan.

**Recollection, not verified:** Möbius addition and gyrations are standard in gyrovector-space treatments of hyperbolic geometry; gyrations express non-associativity/Thomas-precession-like rotations, and Möbius operations have appeared in hyperbolic representation and recommendation models. I do not have a verified paper identifier for the exact combination `D_o Gyr[p,z](Log_o x)` used to rank quantizer codes. Exact prior-art status is therefore **UNKNOWN**. The basic ingredients themselves are plainly not new; any novelty is narrowly the assignment construction and its demonstrated role, not “using gyrations” or “using Möbius addition.”

### Pipeline-location test

The score uses ordered host–code–target Möbius composition in the **assignment**, with an additional squared norm of the gyration's tangent derivative acting on the target's origin-chart tangent. This differs from using Möbius operations only to encode/decode representations, using a hyperbolic distance in ranking, or using hyperbolic losses. But those earlier uses make the proposal more—not less—obvious as a candidate assignment feature: once points are in a gyrovector model, ordered composition and gyration are natural operations to test. Reusing a known operation at assignment is not by itself a novelty argument.

### Is it more than curvature or metric substitution?

The first score term is an order-sensitive three-point composition, and the second explicitly measures the derivative action of a gyration on a named tangent vector at the origin. Under the stated flat limit, composition becomes vector addition and the gyration derivative tends to identity, leaving `||h+v_k+r||²`. Thus the full expression is distinguishable from merely sweeping curvature or swapping a pairwise metric, provided the operator and its paired ablation materially change assignments.

There is a serious adversarial confound: the first term alone is an ordered Euclidean sum in the flat limit, and at finite curvature it is also a candidate-dependent composition score. A gain over a conventional residual quantizer could come from this altered base score, not from gyration. The `γ=0` arm isolates the incremental gyration contribution, but comparing only full versus `γ=0` does not isolate the novelty of ordered composition. Nor does the registered materiality threshold prove utility or non-obviousness. Report and compare the composition-only arm against a matched conventional assignment if making a claim specifically about gyration; do not alter the frozen task recipe or declare success from geometry identities.

The proposal's `PT^{rad}` is diagnostic and does not enter the score; it cannot be claimed as the operative novelty. The derivative definition avoids silently identifying points with vectors, but type correctness is implementation rigor, not prior-art distinction.

### Precise difference and closest work

Closest verified work: **HiHPQ** (DOI above), adjacent through hyperbolic retrieval quantization/codebook machinery. Difference established by the abstract-level evidence: this proposal's assignment ranks each code using **ordered `(p ⊕ z) ⊕ x` composition plus `||D_oGyr[p,z](Log_o x) - Log_o x||²`**; HiHPQ's abstract does not establish this expression or differential gyration assignment. Exact priority remains **UNKNOWN** because the scan is bounded and the HiHPQ full text was not reviewed.

**Kill condition:** find prior use of ordered Möbius composition or gyration action for code selection (including an equivalent gyrovector formulation), show the result is just the composition-only control with no material incremental gyration effect, or fail the preregistered assignment/clamp/task gates. On present evidence, do not claim NOVEL.

## Bottom line

Neither candidate is **OCCUPIED** by an exact match in the supplied, verified evidence; neither merits **NOVEL** on that evidence. Both are **ADJACENT** to verified hyperbolic product quantization and draw on established hyperbolic/gyrovector operations. The defensible statement is narrow: the exact assignment formulas are not shown by the supplied abstract-level scan, but their priority is unverified. A full-text HiHPQ check plus targeted searches for triangle-area code assignment and gyrovector/gyration quantization are required before any novelty claim. Passing the frozen metric gates would establish task performance under the protocol, not priority or non-obviousness.
