# Issue 247 report — leak-free sectional curvature and node-local topology assets

## 1. What was built
A CPU-only, no-training, no-PPR infrastructure release (`v3-sectional-247`) on top of the frozen
#246 v2 assets, which were read but never modified.

Production modules (new): `tools/graph_infra/sectional.py`, `interfaces_v3.py`, `prepare_v3.py`,
`module_checks_v3.py`; tests `tests/test_sectional_v3.py`, `test_interfaces_v3.py`, `test_mutation_v3.py`;
entry `tools/graph_infra/verify_v3.sh`.

## 2. Audit of #246 (files govern, not the earlier reports)
| Item | Earlier report | Measured | Verdict |
|---|---|---|---|
| visible users | 51,695 | 45,951 (val 5,744 / test 5,744) | earlier number wrong |
| probe pairs | 64 | 62 | earlier number wrong |
| strict bank sizes | ≤2,938 | 2,726–2,940 | earlier number wrong |
| v2 curvature asset | "approximate Ollivier–Ricci generated" | **no curvature file exists**; `curvature.py` is Ricci | new asset required |
| 3-node instance total | 28,391,948 | 28,391,948 (exact pairwise count) | consistent |

Method: 47/47 v2 manifest hashes verified unchanged; every user split re-counted with an independent
`Counter` over the raw transitions (self-loops dropped: 0) and compared with the stored CSR;
strict-vs-training undirected key intersection = 0. Evidence: `AUDIT_246.json`.

## 3. Sectional curvature (GraphMoRE Appendix A.1)
Statistic (Eq. A.1, arXiv 2412.11085v1):
`kappa(m;b,c) = mean_a[ d(a,m)^2 + d(b,c)^2/4 − (d(a,b)^2 + d(a,c)^2)/2 ]`,
`d` = unweighted shortest path on the **undirected projection** of the training-visible directed graph.
Consequences recorded in the assets: this is **not** a directed sectional curvature and **not** a Ricci
curvature; direction is preserved separately (`directed_edges.npz` reciprocal flag, directed 1/2-hop
neighbourhoods).

Triples: one uniformly drawn unordered neighbour pair per eligible centre (seed 247001).
References: a fixed global permutation without replacement (seed 247002), nested prefixes 16/64/256
for the sensitivity study. Missing references are masked, never zero-filled; centres without a
neighbour pair have no triple and a NaN / `valid=false` summary.

Official implementation: the public GraphMoRE repository (commit `1948d3c`) ships
`utils.cal_shortest_dis` (networkx shortest paths) but **no sectional-curvature estimator**. The
numerical comparator is therefore an independent implementation of the paper equation
(`results/research/issue247/reference.py`, networkx BFS + exact `Fraction` arithmetic), not a claim
about official curvature code.

### Measured on the real visible graph
| Quantity | Value |
|---|---|
| items | 24,587 |
| components / isolated nodes | 117 / 116 |
| centres with a triple | 24,326 (98.94 %) |
| items with valid curvature | 24,326; missing 261 (116 isolated + small components without a reference) |
| eligible neighbour-pair population | 28,391,948; sampled 0.086 % |
| reference sensitivity 16→64 | mean abs change 0.475, max 2.811, sign flips 4.67 % |
| reference sensitivity 64→256 | mean abs change 0.233, max 1.243, sign flips 2.18 % |
| distance computation | 2.0 s (256 references × 24,587 nodes) |

The sensitivity table is the honest cost statement: with 16 references the sign of the statistic is
unstable for ~5 % of triples; 256 references still move 2.2 % of signs. Any downstream use must fix a
reference budget and report it.

## 4. Node-local topology
* exact 3-node family counts per item (reused from #246) and their normalised distribution;
* directed one/two-hop sparse neighbourhoods, verified for all 24,587 centres;
* exact induced radius-1/radius-2 ego loader with `root_local`, true hop, in/out/mutual roles,
  `node_mask`, `edge_mask` — deterministic, no sampling, no GNN;
* orbit columns = the centre's orbit participation in the **sampled** 3-node instance bank
  (2,000 centres), explicitly not full-population orbit totals;
* 4-node instances remain a **sample**: 500 centres (481 with instances), 1.96 % of centres,
  9.3e-7 of the neighbour-triple population, selection biased by node-ID order — no unbiased
  population estimate or confidence interval is claimed (`coverage.json`).

No hand-computed feature is called a learned embedding; no GNN is trained.

## 5. Leak-free interfaces
`TrainingAssets` can only load assets whose declared use is `training`; any attempt to read a
`strict_` evaluation bank raises. `HeldOutEvaluation` only serves `strict_` banks. The strict
evaluation keys and the training bank keys are disjoint (0 shared pairs). Curvature, topology,
semantic embeddings and the visible graph are aligned by item index and can be loaded independently
with their masks.

## 6. Frozen TIGER provenance
Stage3 read `results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json` (sha256 `025ad8bd…`): prefix-3
collision 0.169276, duplicate share 0.265303, per-level codebook usage/entropy in bits, prefix
sharing per structure bank. `sids_for_hgrec.npy` (`6f2682c1…`) and the checkpoint (epoch 300) belong
to another generation and are recorded separately. Historical recall and reconstruction error are
**NA**: no immutable SID-hash binding to a completed recall exists, and the frozen checkpoint cannot
reproduce the Stage3 SID table.

## 7. Verification
* unit/contract tests: 9 PASS (path 0, star −0.5, triangle 0.25, four-cycle 1, isolated component,
  random graph equal to the independent reference within 1e-12, node reordering, direction reversal,
  self-coincident reference, duplicate triple/reference, NaN/Inf, ID mismatch, manifest tampering);
* mutation testing: **7/7 caught** by the designated assertion (numeric 3 / structural 2 / routing 2);
* per-module verification on real data: **12 PASS + 1 NA** (NA = no training ⇒ no gradient/update check);
* the 17 pre-existing #246 suites: all PASS;
* rebuild from scratch reproduces the sampled triples, the reference distances, every curvature value
  and the whole per-item summary, and re-verifies the directed two-hop membership and the manifest hashes.

One command: `bash tools/graph_infra/verify_v3.sh`.

## 8. Storage
99 MB under `results/research/issue247/assets/` (largest: `reference_distances.npy` 50 MB, the two
directed two-hop CSR files 47 MB). Generated arrays stay out of git under the existing
`results/research/**` rules; the manifest, coverage report, acceptance report and this documentation
are committed.
