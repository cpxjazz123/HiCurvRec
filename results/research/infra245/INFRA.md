# INFRA — the shared graph infrastructure (issue #245 §7)

One package, `tools/graph_infra/`, replaces the geometry, graph, pair-bank and metric code that had
been copied into individual issue directories. Nothing in it imports a model, a checkpoint or a
training entry; a CPU-only interpreter is enough.

## Module map

| module | responsibility | key entry points |
|---|---|---|
| `manifold.py` | κ-stereographic exp/log/projection/distance for Euclidean (`κ=0`), hyperbolic (`κ>0`) and spherical (`κ<0`) | `expmap0`, `logmap0`, `proju`, `dist`, `mobius_add`, `tan_k_inv` |
| `kappa_gcn.py` | directed κ-GCN layer with curvature-bounded aggregation | `KappaGCNLayer.forward` |
| `topology.py` | deterministic and budgeted induced ego subgraphs; frozen node features; batching | `ego_induced`, `root_roles`, `ego_features`, `batch_egos` |
| `graphlet.py` | typed structural motifs computed from the graph, plus degree strata | `root_motifs`, `label_nodes`, `degree_stratified` |
| `pairs.py` | typed pair banks, undirected leak blocking, neighbour distributions, gradient probe | `direct_pairs`, `shared_successor_pairs`, `shared_predecessor_pairs`, `two_hop_pairs`, `degree_matched_pairs`, `exclude_undirected`, `bhattacharyya`, `split_banks`, `probe_bank`, `verify_group` |
| `graph_distance.py` | sparse BFS, full vs subgraph distance, PPR, pair diagnostics | `undirected_projection`, `bfs_distances`, `subgraph_distances`, `reference_bfs`, `ppr`, `pair_metrics`, `masked_mean` |
| `assets.py` | manifest, freeze/verify, strict join | `graph_stats`, `save_sparse`, `load_sparse`, `entry`, `write_manifest`, `verify_manifest`, `join_scores` |
| `metrics.py` | reproducible metrics with explicit denominators | `spearman`, `pearson`, `jaccard`, `prefix_at_k`, `collision_rate`, `duplicate_item_fraction`, `codebook_stats`, `reconstruction_error`, `compare_sid_tables`, `distortion`, `subgraph_summary`, `stratified` |
| `build_assets.py` | one-command rebuild of every frozen asset | `python3 tools/graph_infra/build_assets.py` |
| `baseline_report.py` | recompute the frozen TIGER baseline under the new caliber | `python3 tools/graph_infra/baseline_report.py` |

## Invariants enforced by code

* the centre of an ego block is addressed by an explicit index, never by row 0;
* a motif type is a predicate on the real adjacency, so two samplings of one ego cannot change it;
* a held-out edge blocks **both** directions of its undirected pair from every training bank;
* an unreachable pair is `+inf` and masked, never 0;
* a constant input yields `None`, never a fabricated correlation;
* a model's per-pair numbers enter only through `assets.join_scores`, which raises on any mismatch;
* PPR carries its own unit and is never reported as a distance.

## Test map

| suite | tests | covers |
|---|---|---|
| `test_known_answers.py` | 17 | hand-computed distances, projections, ranking |
| `test_reference.py` | 12 | independent reference for manifold and κ-GCN, gradients |
| `test_topology.py` | 10 | ego construction, centre index, features, padding masks |
| `test_graphlet.py` | 9 | five predicates, negative cases, sampling independence |
| `test_pairs.py` | 12 | typed banks, reverse-edge exclusion, Bhattacharyya, frozen probe |
| `test_metrics.py` | 15 | ties, constant NA, denominators, subgraph aggregation, strata |
| `test_graph_distance.py` | 7 | BFS vs reference, depth cap, PPR mass, pair masking |
| `test_assets.py` | 6 | manifest round-trip, tamper detection, strict join |
| `test_isolation.py` | 14 | real-asset coverage, index consistency, leak rules, canonical cross-check |
| `test_mutation.py` | 17 mutations | each mutation must be caught by a **FAIL** line, never by a crash |

## Rebuild and verify

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 tools/graph_infra/build_assets.py
python3 -c "import json;print(len(json.load(open('results/research/infra245/assets/manifest.json'))['assets']))"
```

`build_assets.py` rewrites the manifest and verifies every hash before returning.
