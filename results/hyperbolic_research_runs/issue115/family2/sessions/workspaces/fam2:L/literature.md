# Literature scan: hyperbolic operations at discrete SID selection

## Scope and constraints

The target is to place a genuinely non-trivial hyperbolic operation in the decision that selects a discrete SID code, while preserving the frozen protocol: 32-dimensional latent vectors; exactly three 256-entry codebooks and exactly three integer SID tokens; original dataset, splits, Stage2 recipe, and Stage3 recipe. Any eventual candidate must strictly exceed all three reported gates: Hard-L1 ER > 16.38953942232631, Stage3 Recall@10 > 0.06406796775709883, and NDCG@10 > 0.03530651701940677. The literature below is not evidence that any such candidate will do so.

The six already-closed families supplied in the task are excluded from the candidate list. This is a targeted prior-art scan, not an exhaustive novelty or patentability search.

## Verified prior art

| Identifier | Title | Year | Inspection level | Precise overlap |
|---|---|---:|---|---|
| DOI: [10.1609/aaai.v38i5.28261](https://doi.org/10.1609/aaai.v38i5.28261); arXiv: [2401.07212](https://arxiv.org/abs/2401.07212) | **HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval** | 2024 | **ABSTRACT_ONLY** | Directly overlaps hyperbolic product quantization and image retrieval. The abstract identifies a hyperbolic product quantizer, hyperbolic codebook attention, and quantized contrastive learning on a hyperbolic product manifold. Thus hyperbolic structure is used in quantization/codebook construction, not only in an upstream embedding. It is unsupervised image retrieval, not recommendation. The inspected abstract does not establish the frozen 32-d / 3×256 / 3-token configuration, the given dataset or training recipes, nor any of the three required thresholds. |

## Evidence boundary

- The arXiv identifier, title, year, and abstract-level method description were checked against the arXiv record for 2401.07212. The DOI and AAAI 2024 venue association were cross-checked against the AAAI DOI record / publication metadata.
- The full paper was not inspected in this pass. Accordingly, the entry is deliberately marked **ABSTRACT_ONLY**; no details about its exact assignment equation, numerical boundary conventions, or experimental compatibility are inferred.
- Hyperbolic recommendation/ranking work that does not place a hyperbolic operation in discrete code selection is outside the requested overlap and is not presented as a match.
- “Not identified” below means not found in this bounded scan, not proof of absence or a novelty opinion.

## Candidate mechanism ideas not occupied by the inspected abstract

These are research directions only, not validated mechanisms, and are stated to distinguish their decision operator from the abstract-level HiHPQ overlap. Each would need an explicit finite-precision contract (domain, epsilons, clamps, comparison tolerance, and boundary behavior) before implementation, and then a test under the frozen protocol.

1. **Lorentz-score SID selection.** Select each code by a Lorentz-model bilinear score between the current item state and code prototypes, with a declared time-orientation and finite-precision admissibility rule. This is not assignment by product-geodesic radius. The inspected abstract does not specify this assignment rule.
2. **Hyperbolic barycentric context-conditioned selection.** Form a context point by a declared hyperbolic barycenter of already-selected code-conditioned states, then choose the next code using a hyperbolic comparison against that context. The state update must be specified independently of codeword-frame residual transport. No such sequential barycentric SID rule is established by the inspected abstract.
3. **Gyrotranslation-conditioned prototype selection (without an associator score).** Apply one explicitly ordered Möbius/gyrotranslation to the query or prototype using a previously selected code, then choose the next integer code by a separately defined score. The operation order must be fixed; no associator term is used. The inspected abstract does not establish this ordered, prior-code-conditioned assignment mechanism.
4. **Hyperbolic triangle-defect decision.** Choose a code using a signed, orientation-defined defect formed from three hyperbolic distances involving the query, a code prototype, and a fixed anchor, with degeneracies and ties specified numerically. This is not the parked unsigned tangent-space wedge area. The inspected abstract does not disclose this decision statistic.

These ideas remain unoccupied only relative to the single abstract inspected here; they are not cleared against full texts or the broader literature. In particular, item 4 must not be treated as a revival of the parked wedge-area candidate without a fresh definition and independent review.
