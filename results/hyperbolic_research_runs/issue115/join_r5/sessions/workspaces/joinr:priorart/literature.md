# Literature scan: hyperbolic operations inside discretisation

**Scope.** Candidate mechanisms must perform a genuinely hyperbolic operation in code assignment, quantisation, or codebook construction, and be relevant to recommendation or retrieval. The target is retrieval/recommendation under the frozen setup: 32-dimensional latent, exactly three 256-entry codebooks and three integer SID tokens, with the original data/splits/Stage2/Stage3 recipes. Nothing below proposes changing those constraints or asserts that a method will clear the strict ER / Recall@10 / NDCG@10 gates.

**Search boundary.** Searches used OpenAlex full-text discovery and the arXiv API for hyperbolic product quantisation/retrieval terms. This is a targeted scan, not an exhaustive systematic review. The clear primary-source match found is an image-retrieval paper; no recommendation-specific match was verified in this pass. A candidate is included only where an identifier and primary-source metadata could be checked.

## Verified match

| Identifier | Title | Year | Access checked | Precise overlap |
|---|---|---:|---|---|
| DOI: [10.1609/aaai.v38i5.28261](https://doi.org/10.1609/aaai.v38i5.28261); arXiv: [2401.07212](https://arxiv.org/abs/2401.07212) | **HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval** | 2024 (AAAI 2024; arXiv preprint posted 2024) | **ABSTRACT_ONLY** | Direct retrieval and product-quantisation overlap. Its abstract explicitly describes a *hyperbolic product quantizer*, a *hyperbolic codebook attention mechanism*, and quantized contrastive learning on a *hyperbolic product manifold*. Thus hyperbolic structure is in the quantisation/codebook mechanism, not merely in the upstream embedding or a curvature sweep. It is unsupervised **image** retrieval, not recommendation. The abstract does not establish compatibility with 32 dimensions, three 256-way codebooks, three integer SID tokens, the specified dataset/recipes, or the stated thresholds. Do not treat it as evidence those constraints can be met. |

## Screening notes

- The identifier, title, date, venue note (“Accepted by AAAI 2024”), and abstract wording above were verified against the [arXiv API record](https://export.arxiv.org/api/query?id_list=2401.07212); the DOI and AAAI 2024 venue metadata were cross-checked against [OpenAlex's work record](https://api.openalex.org/works/https://doi.org/10.1609/aaai.v38i5.28261).
- The published AAAI PDF is openly linked from the venue/OpenAlex record, but its full text was **not** successfully inspected in this run. Therefore the status remains ABSTRACT_ONLY; details beyond the abstract are intentionally not claimed.
- Nearby hyperbolic recommendation papers that use geometry only in representation learning or ranking were not counted: the requested overlap is specifically an operation inside discretisation. Likewise, generic Euclidean product quantisation and hyperbolic embedding papers without quantisation were excluded.
- “Not found” means not identified by this bounded search, not proof that no such work exists.

## Unoccupied mechanism ideas (names only)

- Möbius-Residual Code Assignment
- Geodesic Voronoi Codebook Learning
- Hyperbolic Tangent-Space Residual Quantisation
- Curvature-Aware Product-Code Coordination
