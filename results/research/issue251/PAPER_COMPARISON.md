# Paper / official code / issue251 specification

Read sources: https://arxiv.org/html/2412.11085v1 and https://github.com/RingBDStack/GraphMoRE/blob/main/models.py (GitHub blob 5352bddb1b8a4eb6c32f6585af956b477eb836bd).

| Module | Paper | Official code | Issue251 choice |
|---|---|---|---|
| Local topology | GraphMoRE Eq4–6 multiresolution subgraphs, pooled readouts, concatenated T, classifier | Sampler uses ego graphs; Gating concatenates scale readouts then classifier | Two full directed induced egos processed separately by a shared GNN; keep two tokens until expert attention. Not a claim to enumerate all Graphlets |
| Gating | Eq6 classifier Softmax(T) | Gating.classifier and expert-axis softmax | Owner's replacement: five independent Queries, shared K/V, token-axis masked softmax then shared score and expert-axis softmax. This Cross-Attention is not attributed to the original GraphMoRE implementation |
| Geometry | Section4.1 stereographic curvature experts | Separate Riemannian Encoders per expert | Shared32-dimensional tangent/codebooks, five fixed metric computations; no fivefold latent/codebook capacity |
| Pair metric | Eq9–10 pair-dependent expert alignment | compute_distortion uses softmax gate-product and squared tangent differences | Assignment item gate changes costs; curvature uses fixed equal-weight symmetric expert distances. No direct trainable-metric route to curvature loss |
| Target | Appendix A.1 squared graph-distance statistic | Official models.py instead implements edge distortion objective | Frozen #248 full-component A.1 targets and masks; moment identity is algebraically equivalent to reference enumeration |
| Quantization | Not part of GraphMoRE | None | Frozen local TIGER actual per-level rules, real VQLayer._indices and launcher-installed guards; same-cost soft backward with exact hard forward |
| Loss coefficient | Eq11 task + lambda distortion | Configurable | One real initial reconstruction/curvature encoder-gradient ratio, frozen without search |

The graph target is an undirected-projection finite-distance statistic, not intrinsic or directed sectional curvature. Continuous z never substitutes for decoded hard SID q. Initial Query roles carry no topology or geometry labels. Independent Lorentz/sphere references, hand attention calculations, finite differences and real selection-path negative controls are required evidence.
