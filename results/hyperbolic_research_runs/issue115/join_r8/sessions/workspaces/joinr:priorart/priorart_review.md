# Adversarial prior-art review

## Scope and standard

This review evaluates the two Stage-2 assignment candidates in `proposals.md` against the bounded scan in `literature.md` and established geometric constructions. The frozen dimensions, codebooks, tokens, data, recipes, and strict metric gates are assumed unchanged; they do not establish novelty. `literature.md` verified HiHPQ only at **ABSTRACT_ONLY** level and explicitly says its search was targeted rather than exhaustive. Accordingly, “not found” below is not treated as evidence of absence. Recollections of broader mathematical/ML prior art are marked as such, not presented as source-verified paper matches.

## Candidate 1 — Signed geodesic-triangle-area assignment

### Prior-art overlap

- **Verified in the supplied scan:** HiHPQ, *Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval*, DOI `10.1609/aaai.v38i5.28261`, arXiv `2401.07212` (**ABSTRACT_ONLY**). It already places hyperbolic structure in product quantization/codebook attention for retrieval. It does not, on the inspected abstract, establish signed triangle-area assignment or recommendation use. This is important adjacent quantization prior art, not an exact-match claim.
- **Recommendation-specific exact match:** none verified in `literature.md`; the scan expressly says no recommendation-specific match was verified. Status for this exact operator in recommender quantization is **UNKNOWN**, not “novel by proof.”
- **Broader prior art (recollection, not independently verified here):** oriented/geodesic triangle area, angle defect, Gauss–Bonnet, and area-based geometric scores are classical differential geometry. The proposal itself invokes this identity. I do not have a verified hyperbolic-recommender or hyperbolic-quantizer paper identifier for this exact score.

### Where the operator sits, and obviousness

The proposed area is used directly in **assignment** (with a distance penalty), and as a soft-assignment training term. It is not merely an encoder embedding or a decoder distance. The same triangle-area/angle-defect construction is already used in geometry to characterize curvature, and area-like/orientation-sensitive quantities are familiar in geometric learning; moving a geometric score into a nearest-codeword decision is a real pipeline placement change, but not by itself a strong conceptual novelty argument. Given HiHPQ's verified quantization-level hyperbolic overlap, a reviewer can reasonably view this as a specialized assignment energy within an already occupied broad design space.

### Distinguishability and fatal definition issue

If fully and consistently defined, a signed curved-simplex-area term is **not** just a curvature sweep or a metric swap: it is a three-point, orientation-sensitive score, unlike pairwise distance. Its flat-limit disappearance is a useful ablation, but does not by itself establish novelty.

As written, the defining signed area is not well-defined in 32 dimensions. The three vertices determine a 2-plane and an **unsigned** simplex area, but not a canonical sign. A “fixed ordered orthonormal 2-frame at `p`” does not resolve this: a frame at `p` is not an orientation of every candidate triangle plane. Projecting the frame onto each plane, or choosing an orientation from it, can be singular and generally breaks invariance under common hyperbolic isometries. In addition, the claim that swapping `q,z` reverses sign conflicts with an orientation convention that is not specified independently of the triple. Gauss–Bonnet's angle-defect formula gives the unsigned area (or an oriented version only after an orientation of the simplex has been supplied); it does not supply the missing 32-D sign convention. The stated invariant (iii) is therefore not established, and may be incompatible with a fixed external frame.

There is also a degeneracy in the intended interpretation: for fixed host `p` and target `q`, a signed area is effectively an orientation-weighted angular/wedge term. In the flat limit it becomes a familiar oriented Euclidean parallelogram/triangle-area feature (scaled by curvature conventions), rather than an intrinsically new quantization principle; the proposal's claim that area “vanishes” needs careful scaling and must be checked against the chosen area normalization. `A_c=(π−sum angles)/c` tends to zero for fixed-size triangles as `c→0`, but the normalized quantity and score scale matter.

### Verdict: **ADJACENT** (high risk; not ready as specified)

**Closest verified work:** HiHPQ (arXiv:2401.07212; DOI above), **ABSTRACT_ONLY**: hyperbolic product quantization for image retrieval. **Precise difference claimed:** signed host-target-code geodesic simplex area in hard code assignment, rather than the abstract-described hyperbolic product manifold/codebook-attention mechanism. That difference could be substantive if made mathematically coherent and empirically non-redundant, but the supplied scan cannot verify HiHPQ's detailed operators, and the proposed sign is currently underdefined. I would not accept a “NOVEL” verdict until the oriented area is defined for arbitrary 2-planes in the ball, its isometry behavior is proved, and an exact prior-art full-text comparison is made. No verified exact match found; exact-match status remains **UNKNOWN**.

## Candidate 2 — Non-associative gyro-composition quantization

### Prior-art overlap

- **Verified in the supplied scan:** HiHPQ (arXiv `2401.07212`; DOI `10.1609/aaai.v38i5.28261`) is again the closest verified quantization/retrieval work, at **ABSTRACT_ONLY**. Its abstract establishes hyperbolic product quantization, but does not mention Möbius gyration or this ordered residual-composition score.
- **Recommendation-specific exact match:** none verified by `literature.md`; exact use of this score in recommender SID assignment is **UNKNOWN**.
- **Broader prior art (recollection, not source-verified here):** Möbius addition and gyration are foundational operations in gyrovector/hyperbolic geometry and have been used in hyperbolic neural networks/representation learning. Gyration is precisely the correction encoding non-associativity of Möbius addition. No verified paper identifier is claimed here for the exact quantizer.

### Where the operator sits, and obviousness

The proposal puts Möbius composition and a gyration penalty in the **assignment score**, and also differentiates through them during training. Möbius operations/gyrations elsewhere in an encoder or representation layer do not automatically disclose their use to select a code. But once hyperbolic representations are composed, using gyration to model order effects is a fairly direct consequence of the standard gyrogroup formalism; applying that known operation to residual/code composition is an adaptation, not an entirely new geometric operator. HiHPQ makes the broad “hyperbolic quantization for retrieval” territory occupied, while the exact assignment mechanism is not shown by its abstract.

### Distinguishability and technical concerns

The intended mechanism is distinguishable in principle from a curvature sweep or metric swap: the score is order-sensitive, and a nonzero gyration correction cannot be reproduced by simply changing pairwise distance while preserving the same operations. It is not merely a reparameterized Euclidean residual norm if the measured gyration contribution changes actual assignments and survives controlled flat-limit and ablation tests.

However, the proposal does not cleanly implement the semantics it claims. `G_k` logs `(p ⊕ code) ⊕ Exp_p(r)` at `p`, while the second term compares `gyr[p, code] Exp_p(r)` to `Exp_p(r)`. These expressions mix ball points, tangent vectors, and tangent spaces: `Exp_p(r)` is a point, not a tangent vector; the displayed gyration is normally an action on a gyrovector/point in the ball, not automatically a metric-preserving linear map in `T_p`. The invariant “transport preserves tangent norms” does not validate the point-level gyration expression. Also, the transported `v_k` is defined but the score recomputes a code point via `Exp_o(T_s Log_o(z))`; the role of `PT_{o→p}` and `u_k` is not reflected consistently in the score. A convention-specific derivation is required before these terms are mathematically comparable.

Even with those repairs, the first score term can behave like a geometry-dependent composition/reconstruction residual, while the second explicitly adds a gyration penalty. A gain could arise from this changed objective/assignment energy rather than a novel quantization family. The frozen recipe's allowance for an auxiliary softmin does not remove that attribution issue; report the ablation and assignment changes as proposed, and ensure the unchanged reconstruction/quantization loss is not silently replaced.

### Verdict: **ADJACENT**

**Closest verified work:** HiHPQ (arXiv:2401.07212; DOI above), **ABSTRACT_ONLY**, for hyperbolic quantization in retrieval. **Precise difference claimed:** using ordered Möbius composition and its gyration correction in the residual-stage code decision, rather than the abstract-described hyperbolic product manifold/codebook attention. Classical gyrovector operations are established; the potentially differentiating contribution is their specific assignment-level use. The exact recommender/quantizer match is **UNKNOWN** on this scan, not verified absent. This is distinguishable from a curvature sweep/reparameterization in principle, but the current point/tangent-space inconsistencies make the proposed mechanism insufficiently specified to test or credit as novel.

## Cross-candidate kill criteria

1. **Do not infer novelty from the metric gates.** ER, Recall@10, and NDCG@10 thresholds test utility under this protocol, not prior-art distinction.
2. **Do not infer novelty from nonzero curvature alone.** Both proposals need operator ablations and assignment-level evidence; a curvature-dependent score can still be an obvious geometric reparameterization.
3. **Resolve definitions before running expensive experiments.** Candidate 1 needs a canonical signed-area convention compatible with the claimed isometry invariance in dimension 32. Candidate 2 needs one consistent gyrovector convention, correct point-versus-tangent typing, and a derivation of the score's basepoint behavior.
4. **Expand prior-art verification before any strong novelty claim.** Inspect the full HiHPQ paper and search cited/follow-on work for hyperbolic codebook assignment, gyrovector quantization, and area/orientation-based quantizers. The current scan is explicitly not exhaustive; exact-match status for both candidates remains UNKNOWN.

## Bottom line

Neither candidate earns **NOVEL** on the supplied evidence. Both are **ADJACENT** to verified hyperbolic retrieval quantization (HiHPQ, ABSTRACT_ONLY). Candidate 1 has a potentially differentiating three-point assignment functional but its signed 32-D area is underdefined. Candidate 2 uses established gyrovector machinery in a potentially differentiating pipeline location, but mixes incompatible geometric object types as written. Neither can be responsibly called **OCCUPIED** by an exact verified paper from the available material; absence of an exact match in this bounded scan is not proof of novelty.
