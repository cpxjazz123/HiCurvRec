# Prior-art review — candidate v7

## Scope and question

This review covers the single mechanism declared in `candidate.md`: for each residual host `p`, target `q`, and candidate code `z_k`, compute the **unsigned tangent-space wedge area**

\[
A_k=\tfrac12\sqrt{\|\log_p^c(q)\|^2\|\log_p^c(z_k)\|^2-\langle\log_p^c(q),\log_p^c(z_k)\rangle^2},
\]

then choose the code by minimizing `A_k / (1 + d_c(p,z_k))`. This is a target-conditioned code assignment in the recommender's residual-quantisation SID. It is not a signed area, a curved geodesic-simplex area, or an additional score term beyond the declared area-over-distance score.

The question is narrow: **is an unsigned triangle-area term used to choose a quantiser code in a recommender SID already occupied by prior art?** This is a prior-art/novelty screen, not a patentability opinion or an empirical validation of the operator.

## Findings and confidence

| Status | Finding |
|---|---|
| **Verified in the supplied campaign record** | The sole verified match is HiHPQ, DOI [10.1609/aaai.v38i5.28261](https://doi.org/10.1609/aaai.v38i5.28261), arXiv [2401.07212](https://arxiv.org/abs/2401.07212), and its evidence level is explicitly **ABSTRACT_ONLY**. |
| **Not verified** | The available record does not establish that HiHPQ uses an unsigned triangle-area term to select individual quantiser codes, nor that it uses the candidate's target-conditioned tangent wedge area, area-over-geodesic-distance score, or recommender SID setting. An abstract-only match cannot resolve those implementation-level details. |
| **Recollection / unsupported here** | No recollection is treated as evidence. This review does not infer the HiHPQ mechanism from its title, abstract, or neighboring literature. |
| **Unknown** | Whether the full HiHPQ paper or other unscanned literature contains this exact or a materially equivalent code-selection rule remains unknown. The campaign scan's single verified match is not proof that no other prior art exists. |

## Attempt to falsify novelty

The obvious route to killing the claim is that HiHPQ, or an unreviewed related quantisation method, uses a triangle/wedge-area criterion in its code assignment. The current evidence does **not** confirm that route—but it also does not rule it out. In particular, the abstract-only status leaves the decisive question (what quantity actually selects a code) unanswered.

There is also a scope distinction that must be maintained in any later comparison: candidate v7's area is the ordinary unsigned Gram-determinant area of two tangent vectors based at the current host. It is not signed/oriented, is not a geodesic simplex area, and is not by itself the whole assignment score. A future match should be compared against the complete declared rule, while also checking whether the area component itself is already disclosed as a code-selection criterion.

## Novelty conclusion

**A NOVEL claim may not be made on this record.** The correct classification is **ADJACENT**, not NOVEL and not OCCUPIED: HiHPQ is a verified nearby match, but the available abstract-only evidence does not establish occupation of the specific mechanism. “Not shown occupied” is not the same as “shown novel.”

Before asserting novelty, retrieve and inspect the full HiHPQ paper and search its methods, equations, and code-selection procedure for area/wedge/Gram-determinant terms; then broaden the search to quantisation and recommender-SID literature. Record exact passages and distinguish an area used for code assignment from area used only for analysis, regularisation, or representation geometry.

## Verdict

**REVISE — ADJACENT.** Do not approve a novelty claim or describe the operator as unprecedented yet. The candidate can proceed to the next experimental/join gate only if that gate does not treat prior-art novelty as established; otherwise hold it pending full-text verification of HiHPQ and a broader search. This verdict does not assert that the mechanism is occupied.
