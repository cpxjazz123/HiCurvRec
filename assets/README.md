# `assets/` — persisted graph infrastructure assets

This directory holds a **flattened copy** of the issue #248 asset family
(`v4-full-sectional-248`). The canonical build stays in
`results/research/issue248/assets/`; every file here is byte-identical to it (same sha256),
and `manifest.json` here has its `path` fields repointed at `assets/` while keeping a
`copy_of` field for each copied asset.

## What is here

| Path | Content |
|---|---|
| `manifest.json` | Versioned assets: path, use (`training` / `evaluation` / `description`), schema, bytes, sha256, origin, command; plus input hashes and source provenance |
| `coverage.json` | triple counts, per-centre expectations, component statistics, sampling declarations, shard inventory, runtime and storage cost |
| `index.npz` | `item_shard`, `item_start`, `item_count`, `shard_names`, `total_rows`, `structure_class_counts`, `expected_pairs_per_centre`, `expected_total_rows` |
| `item_summary.npz` | per item: total/valid/invalid triple counts, positive/zero/negative counts and ratios, mean, variance, minimum, maximum, quantiles, missing reason |
| `distance_histogram.npz` | the exact per-node distance histogram (ragged `offsets`/`counts`) — the lossless intermediate every distance moment is recomputed from |
| `node_moments.npz` | per node: `s1`, `s2`, eccentricity, component, component size, in/out/reciprocal degrees |
| `shards/shard_000..012.npz` | the full triple table (28,391,948 rows): centre, b, c, `bc_dist_sq`, curvature, effective reference count, validity mask, invalid reason, directed structure class |
| `train/` | the visible forward/reverse/undirected graphs and the item-id map |

The frozen #246 and #247 assets (visible graph, splits, held-out support, ego, graphlet/orbit
banks, relation banks, semantic embeddings, the frozen TIGER SID) are **referenced** by immutable
hash from their own trees, not copied here.

## Reading it

```python
from pathlib import Path
from tools.graph_infra.full_assets import FullAssets
assets = FullAssets(Path('.'), Path('assets'))
assets.verify()                              # recompute every hash
assets.item_triples(item_id)                 # every triple of one item
assets.triple(m, b, c)                       # exact lookup
assets.structure_class_counts()              # rows per canonical directed 3-node class
```

Training and evaluation stay separated:

```python
from tools.graph_infra.interfaces_v4 import TrainingView, EvaluationView
training = TrainingView(Path('.'), Path('assets'))   # refuses any evaluation asset
held    = EvaluationView(Path('.'), Path('assets'))  # serves only the strict banks
```

## What the curvature is

GraphMoRE (arXiv 2412.11085v1, Appendix A.1, Eq. A.1) — the sectional curvature of the geodesic
triangle formed by a centre `m` and two of its neighbours `b, c`:

```
kappa(m;b,c) = mean_{a in R(m)} [ g(a,m)^2 + g(b,c)^2/4 - ( g(a,b)^2 + g(a,c)^2 )/2 ]
```

`g` is the unweighted shortest-path distance on the **undirected projection** of the visible
directed graph, so this is not a directed sectional curvature; direction survives in the
reciprocal flags and in the per-triple structure class. `R(m)` is the centre's whole connected
component; unreachable nodes are excluded from the mean and never imputed as zero, and no vertex
is excluded from the reference set (`a = m, b, c` are valid references, as in the paper's sum
over `V`). Items with no legal neighbour pair carry a missing reason and NaN statistics.

The binaries stay out of git under the repository's `results/research/**` rules; the manifest,
coverage report and this note are the committed evidence.

## Issue #249: version `v5-gnn-geometry-249`

`v5_gnn_geometry/` adds the ego batching index, permitted node features,
training-visible standardization statistics, evaluation coverage and fidelity reports.
The root manifest registers these additions; frozen graph targets remain unchanged.

```python
from tools.graph_infra.ego_gnn import EgoStore
from tools.graph_infra.curvature_stream import CurvatureReader
from tools.graph_infra.heldout_eval import HeldOutEvaluation
store = EgoStore(Path('.'), Path('assets'))
batch = store.collate([0, 1234], hops=1)
chunks = store.collate_sparse([0, 1234], hops=1, chunk=1)
reader = CurvatureReader(Path('.'), Path('assets'))
rows = reader.item_rows(1234)
evaluation = HeldOutEvaluation(Path('.'), Path('assets'))
```

Deterministic verification: `bash tools/graph_infra/verify_v5.sh`.
Numerical APIs live in `diff_sectional.py`, gradient diagnostics in `grad_diag.py`,
hard-SID decoding and metrics in `fidelity.py`. The finite-difference statistic
is distance-induced, not intrinsic manifold curvature. Hard SID and continuous
encoder fidelity are distinct measurements. No training is performed.

## Issue #251: version `v6-cross-attention-251`

`v6_attention_tokens/token_schema.json` declares two distinct full induced ego
tokens (one hop and two hops), each produced by the same directional GNN and
its own subgraph readout. Existing graph, features, targets and standardization
statistics are reused by immutable hash; no binary asset is rebuilt.
The root manifest now contains 68 assets.

The candidate `TIGER_gnn_attention_sectional_rq` keeps both tokens until expert
cross-attention. Token attention has shape `[items,5,3]` (two ego slots plus a
masked null slot); final geometry weights have shape `[items,5]`. Null is valid
only when both real tokens are invalid. Isolates have two valid root-only ego
tokens. Learned attention outputs are run evidence, not reusable static assets.
`tokens_for_items(store, ids, device, chunk=...)` restores global item order and
duplicates after deterministic batching. Complete two-hop induced subgraphs are
constructed on demand from the same visible directed CSR, without truncation.

Verification-only asset checks are read-only: `build_assets_v5 --verify` preserves
newer manifest versions and exact manifest bytes.
