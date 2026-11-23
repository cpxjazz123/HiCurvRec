# INVENTORY — reviewed artefacts, reuse decisions, known errors (issue #245 §1)

Hashes are truncated to 16 hex characters here; the full values are in
`results/research/infra245/assets/manifest.json` (new assets) and reproducible with `sha256sum`.
`REV` = reviewed, `REUSE` = consumed by the new infrastructure as-is, `NEW` = produced by this pass.

## 1. Graph and feature assets

| artefact | what | sha256 (16) | decision |
|---|---|---|---|
| `results/research/behavior_graph/graph.npz` | canonical undirected CSR, 315,391 edges | `6dbac4545f23779c` | REV; counts cross-checked, not re-read as input |
| `results/research/behavior_graph/edges.parquet` | edge list with AF3 values | `57a4477331019274` | REV; AF3 explicitly **not** a graph distance |
| `results/research/behavior_graph/node_features.npy` | 768-dim raw T5 matrix | `6490c71753952b2d` | REV; identical hash to `stage1_GeneEmbedding/output/sentence_t5.npy`, i.e. no GNN-derived features were ever frozen there |
| `results/research/behavior_graph/metadata.json` | counts + source hashes | `786a12813f89aad0` | REUSE as the cross-check target |

## 2. Curvature and geometry

| artefact | what | sha256 (16) | decision |
|---|---|---|---|
| `results/research/issue236/node_curvature_exact_geodesic.csv` | per-item curvature proxy, 24,587 rows | `da2697a4c3a3c1b2` | REUSE with an explicit caliber note (see §5d) |
| issue #243 geometry adapters | kappa-stereographic ops, kappa-GCN | — | REUSE, consolidated into `tools/graph_infra/` with reference tests |

## 3. Frozen TIGER arm

| artefact | what | sha256 (16) | decision |
|---|---|---|---|
| `results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json` | SID table, 20,425 distinct triples | `025ad8bd5809ce62` | REUSE as the baseline SID table |
| `results/stage2_RQ-VAE/TIGER_RQ-VAE/sids_for_hgrec.npy` | exported matrix, 21,873 distinct triples | `6f2682c18d20ba88` | **rejected as the same object** (see §5a) |
| `results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth` | checkpoint | `af8efc29a8bcbbe5` | REV only; the infra layer does not import a Stage2 tree |
| `results/stage2_RQ-VAE/TIGER_RQ-VAE/stage2_metrics/025ad8…json` | §63 metrics record | — | REV; its `l3_collision` is irreproducible (see §5b) |

## 4. New frozen assets (26 entries, 29.8 MB, all hash-verified)

| group | files | notes |
|---|---|---|
| graph | `forward_full`, `reverse_full`, `undirected_full`, `visible_forward`, `visible_undirected`, `eval_forward`, `stats.json` | 323,633 directed / 315,391 undirected edges, identical to the canonical graph |
| ego | `ego_one_hop`, `ego_two_hop` | deterministic, cap 32, seed 245002, coverage 1.0 |
| motifs | `motif_labels.npz`, `motif_report.json` | chain/fork/merge/triangle/reciprocal + degree strata |
| pairs | 5 visible banks + 5 eval banks + `types.json` + `probe.npz` | 12,000 pairs each, mirror-held-out pairs removed |
| distance | `bfs_visible_root_cache.npz` (2,000 roots), `ppr_visible_sources.npz` (256 sources, top-32) | unweighted hops; PPR is a score, not a distance |
| curvature | `curvature_proxy_236` | reuse, caliber recorded in the manifest |

## 5. Known errors found during the review

**(a) The two frozen TIGER exports are not the same table.**
`item_sids.json` holds 20,425 distinct triples and matches the training log (`collision=0.169276`,
`raw_unique=20425/24587`). `sids_for_hgrec.npy` holds 21,873 and shares only **17** triples with the
JSON; 24,587 of 24,587 rows differ. The JSON is newer (2026-11-20) than the NPY (2026-11-19). Any later
work must name which artefact it reads. Measured by `tools/graph_infra/baseline_report.py`.

**(b) The §63 record's `l3_collision` — CORRECTED by issue #246.**
This entry originally claimed the value was irreproducible from the record's own fields. **That claim
was wrong.** The record documents `methodology.collision = "item-weighted share of repeated three-token
SIDs"`, and the share of *items* whose three-token code is shared with another item is **0.265303** —
exactly the stored value. The §89 gate uses a different caliber, `1 − distinct/items = 0.169276`, which
is the value confirmed by the training log. Both numbers are correct under their own caliber; the
defect is that the two calibers share the word "collision". See
`results/research/infra246/CONTRACT.md` §13 and `results/research/infra246/tiger_audit.json`.

**(c) Three arms sit above the Stage3 collision gate (0.4).**
Measured on every `results/stage2_RQ-VAE/*/item_sids.json` (54 arms):
`TIGER_g237_struct_rq` 0.9655, `TIGER_kappa_cost_rq` 0.8251, `TIGER_i234_A6` 0.4119. They cannot enter
Stage3 until the collision is fixed; the next worst are `TIGER_Uniform_KNN_RQ` 0.3748 and
`TIGER_behavior_edge_contrastive_rq` 0.3375. The frozen baseline `TIGER_RQ-VAE` is 0.1693.

**(d) The #236 curvature is undirected and computed on the full train graph.**
It is a structural description, not a generalisation asset, and must not be read as a directed
curvature. Both facts are recorded in the manifest entry's `caliber` field.

**(e) Duplicated geometry/graph implementations were scattered across issue directories.**
`manifold`, `kappa_gcn`, `metrics`, `pairs`, `topology`, `graph_distance` and the asset plumbing now live
in `tools/graph_infra/` only. The issue-243 adapters were moved, their tests re-pointed at the package,
and the originals removed; new experiments import the package instead of copying code.

## 6. Dependencies

```
train.parquet ─┬─► forward/visible/eval CSR ─┬─► ego blocks
               │                             ├─► motif labels
               │                             ├─► pair banks (+ eval banks)
               │                             ├─► BFS cache, PPR
               │                             └─► baseline diagnostics
sentence_t5.npy ─► (reference only; not consumed by any new asset)
#236 CSV ─────────► curvature asset (reuse, caliber recorded)
```
