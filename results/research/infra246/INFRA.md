# INFRA v2 — the shared graph infrastructure after #246 (issue #246)

One package, `tools/graph_infra/`, still the only home for the geometry, graph, pair-bank and
metric code. v2 adds the modules the issue asked for and fixes the defects it named; nothing
imports a checkpoint, a Stage2 tree or a training entry.

| module | responsibility | key entry points |
|---|---|---|
| `config.py` | frozen constants: version, split seed and ratios, caps, seeds, asset roots; the repository root is found by the `stage2_RQ-VAE/baseline` marker so a copied package still points at the real data | `VERSION`, `SPLIT_SEED`, `SPLIT_RATIOS`, `EGO_CAP`, `ASSET_ROOT` |
| `graph_build.py` | sequences from the parquet, weighted edge multiset, the actual item id set, self-loop count | `sequences_from_parquet`, `count_edges`, `EdgeCounts.to_csr`, `item_index`, `self_loops` |
| `splits.py` | the frozen 80/10/10 user split inside train, with its audit | `three_way`, `audit` |
| `isolation.py` | held-out support on booleans, undirected pair blocking, bank audit | `held_out_support`, `exclude_undirected`-usage, `pair_keys`, `blocked_pairs`, `audit_banks` |
| `topology.py` | ego subgraphs with real roles, true hops, explicit root, edges and features | `ego_induced`, `ego_full`, `root_roles`, `reference_hops`, `ego_features`, `batch_egos` |
| `graphlet.py` | exact 3-node canonical forms, automorphism orbits, families, relaxed fork/merge, instance enumeration, vectorised counts, sampled 4-node patterns | `canonical_code`, `orbits`, `family_of`, `fork_relaxed`, `merge_relaxed`, `motif_instances`, `family_counts`, `four_node_instances` |
| `pairs.py` | typed banks, undirected exclusion, label evidence, degree verification | `direct_pairs`, `shared_successor_pairs`, `shared_predecessor_pairs`, `two_hop_pairs`, `degree_matched_pairs`, `exclude_undirected`, `label_evidence`, `pair_record`, `verify_degree_match` |
| `bank_features.py` | per-pair graph features on the visible graph | `graph_hops`, `subgraph_hops`, `two_hop_counts`, `jaccard_fields`, `bhattacharyya_out`, `witness_fields`, `compute` |
| `ppr.py` | exact personalised PageRank by sparse factorisation, with the dangling correction | `solve`, `solve_selected` |
| `curvature.py` | exact Ollivier-Ricci on a graph via a small transport LP | `orc_edge`, `node_curvature`, `transport_w1`, `summarise` |
| `graph_distance.py` | sparse BFS, full vs subgraph distance, PPR (iterative), pair diagnostics | `bfs_distances`, `subgraph_distances`, `reference_bfs`, `ppr`, `pair_metrics` |
| `metrics.py` | metrics with explicit direction and denominators; SID structure and audit | `spearman`, `pearson`, `jaccard`, `prefix_at_k`, `collision_rate`, `duplicate_item_fraction`, `codebook_stats`, `subgraph_summary`, `stratified`, `sid_structure`, `sid_audit` |
| `assets.py` | v2 manifest entries, schema audit, legal-use enforcement, strict id-keyed join | `entry_v2`, `audit_manifest`, `assert_legal_use`, `verify_manifest`, `join_scores_v2` |
| `manifold.py`, `reference.py`, `kappa_gcn.py` | geometry, its numpy reference twin, the directed kappa-GCN layer | `expmap0`, `logmap0`, `dist`, `proju`, `KappaGCNLayer` |
| `build_assets_v2.py` | one-command rebuild of every v2 asset and the v2 manifest | `python3 tools/graph_infra/build_assets_v2.py` |
| `baseline_v2.py` | TIGER baseline recomputed on the v2 banks | `python3 tools/graph_infra/baseline_v2.py` |
| `tiger_audit.py` | TIGER SID provenance: schema, read-only re-export, Stage3 binding | `python3 tools/graph_infra/tiger_audit.py` |

## Test map (v2)

| suite | tests | covers |
|---|---|---|
| `test_v2_graph.py` | 5 | held-out support, the 80/10/10 split, edge-count add-up, the independent cross-check, the actual item id set |
| `test_v2_ego.py` | 9 | roles from the CSR, true hop vs traversal depth, explicit root, cap semantics, full ego, batching with edges and features |
| `test_v2_graphlet.py` | 8 | relaxed fork/merge, canonicalisation and orbits against the reference on all 64 patterns, 16 classes, instance enumeration against brute force |
| `test_v2_metrics.py` | 8 | the distortion known answer, direction handling, prefix@k, prefix vs full collision, SID audit, entropy unit |
| `test_v2_assets.py` | 3 | strict join validation, id-keyed reorder, manifest schema and legal use, verify-never-writes |
| `test_v2_pairs.py` | 6 | missing vs zero, degree matching per pair, label evidence, unobserved is not a negative |
| `test_v2_ppr.py` | 3 | mass conservation with a dangling row, agreement with the iterative definition, source independence |
| `test_v2_geometry.py` | 4 | ORC known answers against an enumeration reference, curvature summary, the curvature/flat-limit boundaries |
| `test_v2_assets_real.py` | 10 | the real frozen v2 assets: manifest, cross-check, splits, held-out support, bank isolation, ego consistency, motif agreement, distance masks, curvature report |
| `test_mutation_v2.py` | 14 mutations | each mutation must be caught by a designated `FAIL` line |
