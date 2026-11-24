# Paper and official implementation cross-check

Sources read on 2026-11-24:
- https://arxiv.org/html/2412.11085v1 (GraphMoRE, Sections 4.1–4.4, Appendix A.1)
- https://github.com/RingBDStack/GraphMoRE/blob/main/models.py (API blob SHA 5352bddb1b8a4eb6c32f6585af956b477eb836bd)

| Module | Paper | Official implementation | This candidate / reason |
|---|---|---|---|
| Local gate | Eq 4–6 multiresolution sampled ego, pooling, concatenation, softmax | Sampler.sample_ego builds separate k-hop egos; Gating uses two GCNConv layers, global_mean_pool then classifier | Full deterministic visible directed 2-hop ego, separate incoming/outgoing messages, closed 0/1/2-hop pools; no sampled target or dropout; retain direction in structure branch |
| Experts | Section 4.1 stereographic experts | Experts has one independent Encoder per curvature; concatenates K outputs | One shared 32-dimensional tangent latent and original shared codebooks, no K-fold capacity enlargement; fixed expert metric calculations only |
| Pair distance | Eq9 softmax pair gate products; Eq10 weighted squared expert distances | compute_distortion uses squared tangent differences and softmax pair-products | Assignment uses per-item gates on candidate costs; curvature supervision uses fixed uniform expert mixture independent of gate, preventing metric-only gradient bypass; explicitly not a faithful reproduction of official distortion loss |
| Target | Appendix A.1 finite squared-distance expression | models.py compute_distortion is an edge distance-ratio objective, not A.1 | Reuse frozen full component A.1 targets from #248; no edge-ratio loss, no InfoNCE, no independent GNN supervision |
| Loss weighting | Eq11 task + lambda distortion | configurable distortion loss | lambda determined once from initial real loss gradient norm ratio, not tuned |
| Quantizer | Not a GraphMoRE component | No RQ-VAE/Sinkhorn | Frozen local TIGER layers.py has argmin on levels1/2, training Sinkhorn last level (.003/50). Preserve this actual source contract |

Graph A.1 is an undirected-projection discrete statistic, not intrinsic manifold curvature and not direct supervision of direction. Curvature sign uses negative hyperbolic, zero Euclidean, positive spherical. Common tangent coordinates use geodesic origin distances equal to tangent norm for all experts (half-factor in Poincare exponential); no hidden 2x hyperbolic scale.
