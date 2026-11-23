# Issue 248 assets — full-resolution sectional curvature (v4)

`results/research/issue248/assets/` holds the versioned, persisted asset family `v4-full-sectional-248`.
Everything is derived from the **training-visible** graph only. No training, no PPR, no sampling of
triples, no new loss.

Rebuild/verify with one command:

```bash
bash tools/graph_infra/verify_v4.sh
```

The builder (`python3 -m tools.graph_infra.build_assets_v4`) writes the assets; the verifier
(`--verify`) recomputes every hash, re-counts the triples per centre, checks the global
no-duplicate/no-omission property and re-reads every shard.

## What the curvature is

GraphMoRE (arXiv 2412.11085v1, Appendix A.1, Eq. A.1) — the sectional curvature of the geodesic
triangle formed by a centre `m` and two of its neighbours `b, c`:

```
kappa(m;b,c) = mean_{a in R(m)} [ g(a,m)^2 + g(b,c)^2/4 - ( g(a,b)^2 + g(a,c)^2 )/2 ]
```

* `g` = unweighted shortest-path distance on the **undirected projection** of the visible directed
  graph. This is therefore *not* a directed sectional curvature; direction survives in the directed
  edge table and in the per-triple structure class.
* `R(m)` = every node reachable from `m` (its connected component). Unreachable nodes are excluded
  from the mean and counted, never imputed as zero. No vertex is excluded from the reference set:
  `a = m, b, c` are valid references, as in the paper's sum over `V`.
* The mean over the full reference set has the exact closed form
  `( S2(m) + N·g(b,c)²/4 − (S2(b)+S2(c))/2 ) / N` with `S2(v) = Σ_{a∈R(v)} g(a,v)²` and `N = |R(m)|`.
  The closed form is an algebraic rearrangement, verified against the independent reference
  implementation (networkx BFS, term-by-term summation) to 1e-12.
* **This estimand differs from #247**, which averaged over a sample of 256 references. The #247
  numbers are a sampled approximation of these; the two tables are not interchangeable.

## Files

| Path | Content |
|---|---|
| `manifest.json` | every asset: path, use, schema, bytes, sha256, origin, command; plus input hashes, source commit, source-code hashes and the audit of the reused #246/#247 assets |
| `coverage.json` | triple counts, per-centre expectations, component statistics, sampling declarations, shard inventory, runtime and storage cost |
| `index.npz` | `item_shard`, `item_start`, `item_count`, `shard_names`, `total_rows`, `structure_class_counts`, `expected_pairs_per_centre`, `expected_total_rows` |
| `item_summary.npz` | per item: total/valid/invalid triple counts, positive/zero/negative counts and ratios, mean, variance, minimum, maximum, quantiles (0,.25,.5,.75,1), missing reason |
| `distance_histogram.npz` | the exact per-node distance histogram (ragged `offsets`/`counts`) — the lossless intermediate from which every distance moment is recomputed |
| `node_moments.npz` | per node: `s1`, `s2`, eccentricity, component, component size, in/out/reciprocal degrees |
| `shards/shard_NNN.npz` | the triple table, one shard per 2,000 centres: `centre`, `b`, `c`, `bc_dist_sq`, `curvature`, `effective_references`, `valid`, `invalid_reason`, `structure_class` |
| `train/` | the visible forward/reverse/undirected graphs and the item-id map, copied so the training view is self-contained |
| (no `eval/` directory) | The evaluation material is deliberately **not** part of this tree: the strict banks, the held-out support and the user splits stay in the frozen #246 assets and are referenced by immutable hash, so the evaluation data is physically outside the training asset tree and is never uploaded twice. |

Reused assets (visible graph, splits, held-out support, ego, graphlet/orbit banks, relation banks,
semantic embeddings, the frozen TIGER SID) are referenced from the frozen #246/#247 trees by
immutable hash and are **not** re-uploaded.

## Queries

```python
from tools.graph_infra.full_assets import FullAssets
assets = FullAssets(repo)                 # reads the persisted files; never rebuilds
assets.verify()                           # recompute every hash
rows = assets.item_triples(item_id)       # every triple of one item, contiguous rows
one  = assets.triple(m, b, c)             # exact lookup (raises KeyError when absent)
counts = assets.structure_class_counts()  # rows per canonical directed 3-node class
subset = assets.rows_of_structure_class(k)
```

Training/evaluation separation:

```python
from tools.graph_infra.interfaces_v4 import TrainingView, EvaluationView
training = TrainingView(repo)      # refuses any asset declared for evaluation
held    = EvaluationView(repo)     # serves only the strict evaluation banks
```

## Coverage declarations

* The triple enumeration is **full**: `sum_m C(deg(m), 2)` triples, one per unordered neighbour
  pair of every centre, with no sampling, no per-centre truncation and no omission.
* The 4-node graphlet assets remain the **sampled** #246 assets (500 centres) and are declared as
  such; they are not full coverage and are not presented as such.
* The ego assets are the #246 full 1-hop and sampled ego, referenced by hash.
* PPR is not computed, not read and not referenced anywhere in this asset family.
