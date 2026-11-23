# Issue 247 verdict

No training (issue forbids every model training, Stage2/Stage3, and TIGER retraining). No PPR was
computed, read or referenced. The frozen #246 assets were read only; every v2 manifest hash still
matches.

## C1–C10 reconciliation

| # | Requirement | Result | Evidence | Verdict |
|---|---|---|---|---|
| C1 | Audit #246 consistency without overwriting frozen assets | 47/47 v2 hashes unchanged; splits re-counted independently (visible 45,951 / val 5,744 / test 5,744, pairwise disjoint, union complete); three earlier report numbers corrected (51,695→45,951; 64→62 probe pairs; ≤2,938→2,940 strict max); v2 curvature asset does not exist and `curvature.py` is Ricci, not A.1 | `AUDIT_246.json`, `INFRA.md` | PASS |
| C2 | Freeze the training-visible directed graph with ID map, direction, weight, reciprocal flag and split isolation; curvature/neighbourhood/reference/features only from the visible graph; hidden edges evaluation-only; graph hash, build rules, isolated/disconnected nodes | `assets/item_ids.npy`, `assets/directed_edges.npz` (source/target/weight/reciprocal), `assets/directed_{out,in}_within2.npz`, `assets/manifest.json` (graph_sha256, inputs), `assets/coverage.json` (build rules, 117 components, 116 isolated) | PASS |
| C3 | True GraphMoRE-style sectional curvature from paper and official implementation; no Ricci/Forman/triangle-count substitute; undirected projection stated; degenerate cases defined | `CONTRACT.md` (formula, triple, centre, reference rule, distance, normalisation, averaging order, degeneracies, paper/official crosswalk), `tools/graph_infra/sectional.py`, official repo commit `1948d3c` inspected — no official curvature estimator exists, recorded as a provenance limitation | PASS |
| C4 | Reusable triple-level curvature table with centre, partners, reference rule + fixed seed, effective reference count, value, invalid reason, local structure type; per-item counts, positive/zero/negative ratios, mean, std, quantiles; zero tolerance, empty-sample mask, coverage; no missing→zero; deterministic reference-count sensitivity study with cost | `assets/sectional_triples.npz` (triples, references, value, effective_references, valid, invalid_reason, endpoint_distance, directed_graphlet, neighbour_pair_population, per-reference contributions), `assets/sectional_summary.npz` (count, valid, mean, std, negative/zero/positive, quantiles), `assets/coverage.json` (tolerance 1e-12, sampling rules, 16/64/256 sensitivity with cost) | PASS |
| C5 | Node-local topology: reuse #246 3/4-node graphlet/orbit/ego; directed 1/2-hop neighbourhood, centre position, adjacency, node indices, valid masks, deterministic topological features (in/out degree, reciprocal, role counts, normalised distributions); 4-node sampling coverage/mechanism/uncertainty declared; multi-scale ego loader for a future GNN; no GNN training; no "learned embedding" claim | `assets/topology_features.npz`, `assets/orbit_schema.npz`, `TrainingAssets.ego(item, radius)` (exact induced radius 1/2 with `root_local`, hop, in/out masks, `node_mask`, `edge_mask`), `coverage.json` (4-node: 500 centres, 481 with instances, 1.96 % centre coverage, ID-biased selection, uncertainty NA) | PASS |
| C6 | Reusable data interface for future RQ-VAE supervision, aligned by item index, per-asset loading with masks; strict held-out evaluation interface reusing #246 samples; training side cannot read test targets; no new loss | `tools/graph_infra/interfaces_v3.py`, `tests/test_interfaces_v3.py`, `ACCEPTANCE.json` (strict ∩ training = 0), no loss defined | PASS |
| C7 | Unified frozen TIGER assets: the actual Stage3 `item_sids.json` with hash, other NPY/checkpoint separated; prefix-3 collision, duplicate share, per-level codebook usage/entropy, prefix sharing, structure-group metrics; only same-source provable reconstruction/recall bound, otherwise NA; no retrain | `assets/tiger.json` (sid sha256 `025ad8bd…`, collision 0.169276, duplicate 0.265303, codebooks in bits, prefix by structure bank, separate checkpoint/NPY, recall and reconstruction MSE NA with reasons) | PASS |
| C8 | Hand-computable small graphs; repeatable numerical comparison against a paper-derived reference; tests for reordering, direction reversal, distance degeneracy, self-coincident reference, empty neighbourhood, NaN/Inf, ID mismatch, data leakage, duplicate samples, manifest tampering; numerical/invariant assertions; regression test per defect | `tests/test_sectional_v3.py` (path 0, star −0.5, triangle 0.25, four-cycle 1, isolated component, random graph vs independent reference ≤1e-12, reordering, reversal, self-reference, duplicates, NaN/Inf), `tests/test_interfaces_v3.py` (ID mismatch, manifest tampering, routing), `tests/test_mutation_v3.py` (7/7 caught) | PASS |
| C9 | Versioned manifest (path, schema, dtype, shape, source commit, input hashes, config, seed, output hashes, coverage, command, runtime, storage cost); machine-readable acceptance report; README; one repeatable prepare/verify entry; large files follow repo storage rules | `assets/manifest.json` (50 entries, version `v3-sectional-247`, source commit, per-module source hashes, input hashes, seed, coverage, command, runtime, storage bytes), `ACCEPTANCE.json`, `README.md`, `tools/graph_infra/verify_v3.sh`; 99 MB of arrays excluded from git by the existing `results/research/**` rules | PASS |
| C10 | Full-data rebuild, reload, ID alignment, split isolation, finiteness, hashes, reproducibility; locatable code/assets/logs; per-item PASS/FAIL/NA | `ACCEPTANCE.json` (PASS; 24,587 items; 50 manifest entries; 24,326 finite valid curvature; 261 missing; strict∩training 0; triples/distance/two-hop reproduced), `MODULE_CHECKS.json` (12 PASS / 0 FAIL / 1 NA), `VERIFY.log`, `REPORT.md`, `CHECKLIST.md` | PASS |

One item is explicitly **NA** rather than PASS: the gradient / parameter-update module check, because
this round defines no loss and trains nothing (the issue forbids both). It is recorded as NA in
`MODULE_CHECKS.json`, not silently skipped.

## Headline numbers, independently recomputed

| Number | Recomputation path | Matches |
|---|---|---|
| valid curvature items 24,326 / missing 261 | `prepare_v3.validate` rebuilds triples, references, distances and the summary from the visible graph | yes |
| per-item summary | `sectional.summarize` re-run on the stored triple table, compared key by key with the published `sectional_summary.npz` | yes (`MODULE_CHECKS.json`) |
| strict ∩ training undirected keys = 0 | set intersection over all `strict_*` and `train_*` banks | yes |
| TIGER prefix-3 collision 0.1692764469 / duplicate share 0.2653028023 | recomputed from `item_sids.json` inside `prepare_v3.validate` | yes |
| curvature on the hand-computed graphs (0, −0.5, 0.25, 1) | `tests/test_sectional_v3.py` vs the networkx/`Fraction` reference | yes |

## Preconditions the conclusion depends on
* The statistic is defined on the **undirected projection** of the training-visible graph; it is not a
  directed sectional curvature, and the sign convention of the paper is used unchanged.
* The reference budget (256) and the triple sampling seed (247001) are part of the asset identity; a
  different budget changes 2.2 % of triple signs (see `coverage.json`).
* 261 items have no valid curvature. They are marked invalid and are **not** imputed; a future model
  must handle them with a mask.
* The curvature values are a **descriptive** local-geometry statistic of the visible graph. Nothing
  here is evidence that a curvature-conditioned RQ-VAE improves recall.

Sample-internal vs generalisation: every published curvature value is computed on the visible
training graph; the strict evaluation banks remain unseen by the training interface, and the two
evaluation splits were never used to build any asset.

## Closing four questions (§6.4)
1. All production code implemented? **Yes** — `tools/graph_infra/{sectional,interfaces_v3,prepare_v3,module_checks_v3}.py`, no stubs, no placeholders, no TODOs.
2. All necessary tests actually executed? **Yes** — 9 v3 assertions-suites pass, 7/7 mutations caught, 12 module checks PASS / 1 NA, the 17 pre-existing #246 suites pass, `bash tools/graph_infra/verify_v3.sh` exits 0.
3. All prescribed training completed? **Yes (this round prescribes no training)** — the issue forbids it; only deterministic graph computation and read-only reuse of frozen assets were performed.
4. Any executable task left? **No** — every C1–C10 item is PASS or explicitly NA with a reason, and no executable step remains (the forbidden items are forbidden by the issue, not by resource limits).

## Evidence index
`results/research/issue247/`: `CONTRACT.md`, `INFRA.md`, `AUDIT_246.json`, `CHECKLIST.md`, `REPORT.md`,
`README.md`, `MUTATIONS.md`, `MUTATIONS.json`, `MODULE_CHECKS.json`, `ACCEPTANCE.json`, `RUN_LOG.md`,
`VERIFY.log`, `BUILD.log`, `GREEN.log`, `MUTATIONS.log`, `REGRESSION.log`, `COMMENT_01.md`–`COMMENT_03.md`,
`assets/` (manifest + 11 data assets).
Production: `tools/graph_infra/{sectional.py,interfaces_v3.py,prepare_v3.py,module_checks_v3.py,verify_v3.sh}`,
`tools/graph_infra/tests/test_{sectional,interfaces,mutation}_v3.py`.
