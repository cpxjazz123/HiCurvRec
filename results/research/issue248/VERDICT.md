# Issue 248 verdict

No training (the issue forbids Teacher/GNN/RQ-VAE/Stage2/Stage3 training and TIGER retraining), no
PPR, no new loss, no change to the 768-dim input or the three-layer SID protocol. The frozen #246
and #247 assets were read only; nothing was overwritten.

## C1–C11 reconciliation

| # | Requirement | Result | Evidence | Verdict |
|---|---|---|---|---|
| C1 | Freeze and audit the inputs; keep #247 as history | 47/47 #246 hashes and every readable #247 hash verified unchanged; #247's sampled assets referenced by hash as version `v3-sectional-247` and never mixed with the v4 table; the audit block is stored in the manifest | `assets/manifest.json` (`audit`), `INFRA.md` | PASS |
| C2 | Full enumeration of every legal neighbour pair, total recomputed from the visible graph, no sampling/truncation/omission | `Σ_m C(deg(m),2)` = **28,391,948**, recomputed from the visible graph and matching the produced table row for row; 24,326 centres have triples, 116 items are isolated and 145 have a single neighbour (no legal pair) | `assets/index.npz` (`expected_pairs_per_centre`, `expected_total_rows`), `assets/coverage.json`, `MODULE_CHECKS.json` (enumeration ×3) | PASS |
| C3 | The #247-verified A.1 formula with all conforming reachable references; explicit exclusion rules; no sampling/approximation/Ricci; exact simplifications allowed; equivalence proven on small graphs | Eq. A.1 with `R(m)` = the centre's whole connected component, no vertex excluded (`a = m, b, c` are valid references, as in the paper's sum over `V`); the mean reduces exactly to `( S2(m) + N·g(b,c)²/4 − (S2(b)+S2(c))/2 )/N`; the production code and the independent networkx reference agree to 1e-12 on four hand-computed graphs and on random graphs; the official repository ships no curvature estimator, which the contract records | `CONTRACT.md` §1–§2, `tools/graph_infra/sectional_full.py`, `results/research/issue248/reference.py`, `tests/test_full_curvature_v4.py` | PASS |
| C4 | Full triple table persisted with centre, pair, curvature, reference count, reconstructible intermediates, mask, invalid reason, structure type | 13 shards with `centre`, `b`, `c`, `bc_dist_sq`, `curvature`, `effective_references`, `valid`, `invalid_reason`, `structure_class`; the exact per-node distance histogram is persisted, and every moment (`S1`, `S2`, eccentricity, component size) is recomputed from it, so any triple's basis is reconstructible without a distance matrix | `assets/shards/*.npz`, `assets/distance_histogram.npz`, `assets/node_moments.npz` | PASS |
| C5 | Per-item distribution persisted (counts, positive/zero/negative counts and ratios, mean, variance, quantiles, extremes, missing reason) with queries by item, triple and structure class | `assets/item_summary.npz` with all required fields; missing items carry the reason `isolated` / `no_triple` and **NaN** statistics, never zeros; queries: `FullAssets.item_triples`, `.triple`, `.structure_class_counts`, `.rows_of_structure_class` | `assets/item_summary.npz`, `tools/graph_infra/full_assets.py`, `tests/test_v4_reload.py` | PASS |
| C6 | All existing infra assets consolidated and persisted; reuse by immutable hash without duplicate uploads; coverage and sampling attributes kept; the sampled 4-node asset never called full | 30 reused assets referenced by hash (`origin: reused by immutable hash`), 32 built assets; the 4-node bank is declared "sampled in #246 (500 centres) ... NOT full coverage" and the ego assets as "#246 sampled and full 1-hop" | `assets/manifest.json`, `assets/coverage.json` (`four_node_assets`, `ego_assets`) | PASS |
| C7 | Reusable exact distance intermediates; training/evaluation isolation; no hidden structure as a training feature | the exact distance histogram and the per-node moments are persisted; `TrainingView` refuses any `evaluation` asset, `EvaluationView` serves only the strict banks and the held-out support; every v4 asset derives from the training-visible graph (261,132 directed edges) and the full-description graph is never read | `assets/distance_histogram.npz`, `tools/graph_infra/interfaces_v4.py`, `MODULE_CHECKS.json` (isolation ×3) | PASS |
| C8 | Persist under the repository's storage norms; capacity and quota checked first; no empty files or unuploaded pointers | `/fs04` has 1.4 P free (84 % used), `results/research` 8.6 GB; the assets live on the shared research volume under `results/research/issue248/assets/` (200 MB), not in a temp directory or machine cache; the R2 bucket is reachable but its numbered layout is reserved by §6b for Stage3 mechanism archives, so the assets stay in the repository's research-asset tree while the manifest and evidence are committed; every file has a non-zero byte count and a sha256 | `INFRA.md`, `assets/manifest.json`, `assets/coverage.json` (`storage`) | PASS |
| C9 | One manifest/index with path, version, schema, shape, dtype, rows, bytes, hash, input hashes, commit, command, coverage, use; runtime and cost; old version provenance kept | `assets/manifest.json` (62 entries with path/use/schema/bytes/sha256/origin/command, plus `inputs`, `source_commit`, `source_hashes`, `audit`, `version`), `assets/coverage.json` (runtime and storage cost), `assets/index.npz`; the #247 version stays referenced by hash | `assets/manifest.json`, `assets/coverage.json` | PASS |
| C10 | Full acceptance: per-centre counts, global no-duplicate/no-omission, complete reference sets, agreement with the independent implementation, independent recomputation of the per-item distributions, id alignment, direction, isolation, finiteness, masks, all hashes; the existing infra regressions run | per-centre counts equal `C(deg,2)` for all 24,587 centres; 28,391,948 rows with no duplicate `(centre,b,c)` and `b < c` everywhere; every triple's reference count equals its centre's component size; five sampled triples recomputed from explicit distance rows match to 0.0; `S2` recomputed from the persisted histogram matches to 0.0 and matches an independent BFS to 0.0; per-item means recomputed from the shards match to 0.0; 62 hashes verified; 14 module checks PASS / 0 FAIL; the 16 #246 suites and the 3 #247 suites pass | `ACCEPTANCE.json`, `MODULE_CHECKS.json`, `MUTATIONS.json`, `VERIFY.log` | PASS |
| C11 | Reload from the persisted assets (not a temporary generation directory); verify the full scan, random access, the interfaces, shard merging and the hashes; report the actual counts, coverage, file list, sizes and persistence location | a fresh process re-reads the manifest and finds the same 62 hashes; the shard merge reproduces the global row count and every per-class count; 200 random accesses agree with the full scan; the structure-class row sets sum to the published counts; the training/evaluation views behave as declared; the report states 28,391,948 rows, 24,326 items with curvature, 261 without, 117 components, 200 MB and the persistence path | `tests/test_v4_reload.py` (6 PASS), `REPORT.md`, `README.md` | PASS |

## Headline numbers, independently recomputed

| Number | Recomputation path | Agrees |
|---|---|---|
| 28,391,948 triples | `Σ_m C(deg(m),2)` from the visible graph, and the stored per-centre counts | yes |
| per-centre counts | `index.item_count` vs `C(deg(m),2)` element-wise | yes |
| `S2` | from the persisted histogram, and from an independent BFS on 12 sampled nodes | yes (0.0) |
| five sampled curvatures | explicit per-source distance rows, mean over the finite references | yes (0.0) |
| per-item means | recomputed from the shards | yes (0.0) |
| class counts | merged shards vs the published index | yes |
| TIGER provenance | the #247 `tiger.json` (Stage3-read SID `025ad8bd…`, collision 0.169276, duplicate 0.265303) is referenced by hash, unchanged | yes |

## Preconditions the conclusion depends on
* The statistic is the **undirected projection** of the training-visible graph. It is not a directed
  sectional curvature; direction survives only in the reciprocal flags and the per-triple structure
  class (13 of the 64 canonical directed 3-node classes occur).
* The reference set is the centre's whole connected component, so the estimand is
  **component-conditional**. 116 items are isolated and 145 have a single neighbour; their curvature
  is missing (NaN with a reason), never zero.
* The #248 estimand **differs from #247's** (full vs 256-node sampled reference set). The two tables
  are not interchangeable; #247 stays as the historical version.
* The curvature is a **descriptive** statistic of the visible graph. Nothing here is evidence that a
  curvature-conditioned RQ-VAE improves recall.

## Closing four questions (§6.4)
1. All production code implemented? **Yes** — `tools/graph_infra/{sectional_full,full_assets,interfaces_v4,build_assets_v4,module_checks_v4}.py` plus `verify_v4.sh`; no stubs, no placeholders.
2. All necessary tests actually executed? **Yes** — 6 curvature tests, 5 asset tests, 6 reload tests, 8 mutations (all caught by their named assertion), 14 module checks, the 16 #246 suites and the 3 #247 suites, and `bash tools/graph_infra/verify_v4.sh` exiting 0.
3. All prescribed training completed? **Yes (this round prescribes none)** — the issue forbids training and PPR; only deterministic graph computation and read-only reuse were performed.
4. Any executable task left? **No** — every C1–C11 item is PASS with executed evidence.

## Evidence index
`results/research/issue248/`: `CONTRACT.md`, `INFRA.md`, `CHECKLIST.md`, `REPORT.md`, `README.md`,
`MUTATIONS.md`, `MUTATIONS.json`, `MODULE_CHECKS.json`, `ACCEPTANCE.json`, `RUN_LOG.md`,
`issue.json`, `BUILD.log`, `VERIFY.log`, `reference.py`, `assets/` (manifest, coverage, index,
item summary, distance histogram, node moments, 13 shards, train/).
Production: `tools/graph_infra/{sectional_full,full_assets,interfaces_v4,build_assets_v4,module_checks_v4}.py`,
`tools/graph_infra/verify_v4.sh`,
`tools/graph_infra/tests/{test_full_curvature_v4,test_v4_assets,test_v4_reload,test_mutation_v4}.py`.
