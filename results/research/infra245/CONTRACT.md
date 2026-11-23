# CONTRACT — frozen data and interface contract (issue #245)

This document freezes the conventions every later experiment must use. Changing any row below is a
**new asset version** (new manifest), never an in-place edit. Frozen historical artefacts are
read-only; nothing here overwrites them.

## 1. Identifiers and indexing

| item | rule |
|---|---|
| item id | integer `0 … 24586`, the row order of `results/stage0_build_parquet/items.parquet`, identical to the order of `stage1_GeneEmbedding/output/sentence_t5.npy` and of `item_sids.json` rows |
| user id | integer from `train.parquet`; 57,439 distinct users |
| pair | `np.int64[2]`, `[left, right]`; directed relations keep their order, undirected relations are emitted as `(min, max)` |
| graph index | `csr_matrix` with `indptr/indices/data` in `int64`; the same item id is the row index, no local renumbering except inside an ego block |
| ego node | `nodes` holds **global** item ids; `root_local` (or `root_local_index` after `ego_features`) is the row of the centre — the centre is never assumed to be row 0 |
| ego offsets | `offsets[block] … offsets[block+1]` slices `nodes`; `edges` are local indices into that slice, offset by `offsets[block]` |

## 2. Edge semantics

| rule | value |
|---|---|
| edge | adjacency between consecutive purchases inside one user's train sequence |
| weight | integer transition count; **never** an AF3 value, a distance or a similarity |
| self loops | dropped (measured: 0 in this dataset) |
| duplicate handling | summed into the weight, one CSR entry per ordered pair |
| directed graph | 323,633 edges (verified equal to `behavior_graph/metadata.json`) |
| undirected projection | simple, one edge per unordered pair: 315,391 (verified equal to the canonical metadata) |
| reverse graph | exact transpose of the forward graph |

AF3 (or any attribute) must not be read as a graph distance. Distance means unweighted hop count
unless the field says otherwise.

## 3. Splits and leakage

| split | definition |
|---|---|
| visible graph | transitions of 90 % of users, `USER_SPLIT_SEED = 245001`, 51,695 users, 291,954 edges |
| eval graph | transitions of the remaining 5,744 users, 33,971 edges |
| user isolation | the two user sets are disjoint and their union is every user (`test_visible_and_eval_users_are_disjoint_by_count`) |
| held-out blocking | an edge present in the eval graph but not in the visible graph is blocked **in both directions** from every training bank (`pairs.exclude_undirected`) |
| eval-side banks | `bank_*_eval.npy` are built on the eval graph and are evaluation-only; the manifest purpose string says so and `test_eval_banks_mark_eval_use_in_the_manifest` enforces it |
| structural description vs generalisation | assets computed on the full train graph (motifs, curvature) are *descriptive*; assets for generalisation use only the visible graph (banks, PPR, distances) |

Validation/test labels are never used for propagation, supervision, calibration or sampling.

## 4. Feature order (frozen)

`topology.ego_features` emits, in this order:

```
0 is_root        the node is the centre
1 hop            0 for the centre, 1 for a one-hop neighbour, 2 for a two-hop node
2 degree_out     global out-degree on the graph the ego was built from
3 degree_in      global in-degree on the same graph
4 has_in_edge    the node points to the centre
5 has_out_edge   the centre points to the node
6 is_mutual      both of the above
```

Any new feature is appended; existing positions never move.

## 5. Masks and empty cases

| case | policy |
|---|---|
| padded batch positions | `node_mask = False`; they are never attendable and never enter a mean |
| ego with no neighbour | block size 1 (the centre), `empty = True`; the block stays in the asset |
| isolated item in a distance task | distance `+inf`, excluded from means; `+inf` is never replaced by 0 |
| unreachable pair in distortion | counted under `unreachable`, excluded from `mean`/`median` |
| constant input to Spearman/Pearson | `None`, never 0.0 |
| absent stratum | absent from the report, never reported as 0 |

## 6. Distance units and scale calibration

* graph distance: unweighted hops, integer, `+inf` for unreachable.
* model distance: whatever unit the model emits; `pair_metrics` and `metrics.distortion` **require**
  the caller to pass the graph hops and report the unit next to the number.
* subgraph distance (`subgraph_distances`) and full-graph distance are different names and are never
  substituted for one another.
* PPR is a propagation score in `(0, 1)`; it is not a distance and must not be reported as one.
* scale calibration is a *future* interface: `pairs.probe_bank` freezes the probe pairs and their
  graph hops. No scale, curvature weight or temperature is chosen here.

## 7. Metric denominators

| metric | numerator / denominator |
|---|---|
| SID collision | `1 − distinct(first 3 codes) / items` |
| duplicate item fraction | share of items whose **full** code tuple is shared with another item (different quantity from collision) |
| prefix@k | pairs whose first `k` positions match position by position (not an arbitrary token intersection) |
| subgraph pass fraction | passing subgraphs / all subgraphs |
| centre all-pass fraction | centres where **every** subgraph passes / centres (never mixed with the previous row) |
| Jaccard | intersection / union of neighbour sets |

A common neighbour without a direct edge implies undirected distance exactly 2 by definition; Jaccard
on such pairs cannot be used alone as evidence that a complex model is unnecessary.

## 8. Strict join

`assets.join_scores(bank_pairs, model_pairs, model_scores)` is the only supported way to attach model
numbers to a bank. Missing pairs, duplicate pairs, misaligned ids or a length mismatch raise; there is
no silent zero fill and no reordering.

## 9. Manifest schema

Every frozen file has one entry with:

```
name, path, schema, purpose, command, sha256, bytes, dependencies,
plus optional fields (shape, dtype, pairs, coverage, caliber, …)
```

`assets.verify_manifest` recomputes every hash and fails closed on a missing file or drift.

## 10. What this contract does not decide

Curvature weights, temperatures, gate values, expert scales, λ, thresholds and any model output are
deliberately absent. They belong to a future candidate, which must state them as its own constants and
join its per-pair numbers onto the frozen banks through §8.
