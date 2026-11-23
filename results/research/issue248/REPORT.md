# Issue 248 report — full-resolution sectional curvature and persistent infrastructure

## 1. What changed relative to #247

#247 froze a *sampled* curvature table: one uniform neighbour pair per eligible centre (0.086 % of
the 28,391,948 legal triples) and a 256-node reference sample. #248 replaces it with the **full**
table and the **full** reachable reference set, and persists everything under a versioned manifest.

| | #247 (`v3-sectional-247`) | #248 (`v4-full-sectional-248`) |
|---|---|---|
| triples | 24,326 (one per centre, sampled pair) | **28,391,948** (every unordered neighbour pair) |
| reference set | 256 sampled nodes | **every reachable node** (the centre's connected component) |
| estimand | mean over a sampled reference set | mean over the full reachable reference set (exact closed form) |
| storage | 197 MB, sampled arrays | 13 shards, one per 2,000 centres, contiguous rows per item |

The two tables are **not interchangeable**: #247's numbers are a sampled approximation of #248's.
Both are kept on disk; #247 is referenced by hash as the historical version.

## 2. The statistic and why it is now cheap and exact

Eq. A.1 of GraphMoRE (arXiv 2412.11085v1, Appendix A.1):

```
kappa(m;b,c) = mean_{a in R(m)} [ g(a,m)^2 + g(b,c)^2/4 - ( g(a,b)^2 + g(a,c)^2 )/2 ]
```

With `R(m)` = the centre's whole connected component, the mean separates:

```
kappa(m;b,c) = ( S2(m) + N·g(b,c)²/4 − ( S2(b) + S2(c) )/2 ) / N
```

`S2(v) = Σ_{a∈R(v)} g(a,v)²` and `N = |R(m)|`. This is an algebraic rearrangement, not an
approximation, so the full reference set costs one BFS per node instead of one pass per triple.
The per-node **distance histogram** is stored, and every moment (`S1`, `S2`, eccentricity, component
size) is recomputed from it — the intermediate is lossless, not a summary.

The production code and the independent reference implementation agree to 1e-12 on hand-computed
graphs (path 0, star −0.5, triangle 0.25, four-cycle 1) and on random graphs.

## 3. Enumeration

* Every unordered neighbour pair of every centre, `b < c`, no sampling and no per-centre truncation.
* The population is **recomputed** from the visible graph, not taken from the earlier report:
  `Σ_m C(deg(m),2) = 28,391,948`, which matches the previously reported number.
* 24,326 centres have at least one triple; 116 items are isolated and 145 have exactly one
  neighbour (no legal pair, no triple, recorded with the reason `no_triple` / `isolated`).

## 4. Persisted assets

| Asset | Content | Size |
|---|---|---|
| `shards/shard_000..012.npz` | the full triple table: centre, b, c, `g(b,c)²`, curvature, effective reference count, validity mask, invalid reason, directed structure class | 185 MB |
| `item_summary.npz` | per item: counts, positive/zero/negative counts and ratios, mean, variance, min, max, quantiles, missing reason | ~1 MB |
| `distance_histogram.npz` | exact per-node distance histogram (ragged) | ~1 MB |
| `node_moments.npz` | per node: `S1`, `S2`, eccentricity, component, component size, degrees | ~1 MB |
| `index.npz` | item → (shard, start, count), shard names, total rows, per-class counts, expected per-centre counts | ~1 MB |
| `manifest.json` | every asset with path, use, schema, bytes, sha256, origin; input hashes, source commit, source-code hashes, audit of the reused assets | 60 KB |
| `coverage.json` | counts, coverage, component statistics, sampling declarations, runtime and storage cost | 6 KB |
| `train/` | the visible forward/reverse/undirected graphs and the item-id map | 1 MB |

The reused assets (visible graph, splits, held-out support, ego, graphlet/orbit banks, relation
banks, semantic embeddings, the frozen TIGER SID) are referenced from the frozen #246/#247 trees by
immutable hash and are **not** re-uploaded. The 4-node graphlet bank remains the sampled #246 asset
and is declared as such.

## 5. Isolation

* Every v4 asset is derived from the training-visible graph only (45,951 visible users); the
  `struct_val` / `struct_test` users and the full-description graph are never read by a v4 builder.
* `TrainingView` refuses any asset declared `evaluation`; `EvaluationView` serves only the strict
  banks and the held-out support.
* The strict evaluation keys and the training bank keys are disjoint (checked on the loaded banks).

## 6. Verification

* Unit and contract tests: hand-computed graphs, full enumeration against a brute-force pair
  enumeration, the closed form against the independent reference implementation, the missing-reference
  mask, the directed structure class.
* Mutation testing: **8/8 caught by the designated assertion** (numeric 3, structural 3, routing 2).
* Per-module verification on the real data: recomputation of `S2` from the persisted histogram, an
  independent BFS on sampled nodes, an independent explicit recomputation of sampled triples,
  per-item distribution recomputation from the shards, per-class counts, and the isolation checks.
* Asset verification: 61 hashes recomputed, per-centre counts equal to `C(deg,2)` for every centre,
  no duplicate triple globally, every curvature finite, every reference set non-empty.

One command: `bash tools/graph_infra/verify_v4.sh`.

## 7. Honest limits

* The curvature is defined on the **undirected projection**; it is not a directed sectional
  curvature. Direction survives only in the reciprocal flags and the per-triple structure class.
* 116 items have no curvature at all (isolated); 145 more have no legal neighbour pair. They are
  marked, never imputed.
* 0.47 % of all node pairs are unreachable — the other 116 components. A triple's reference set is
  its centre's component, so the statistic is component-conditional; this is stated in the contract
  rather than hidden.
* The curvature is a **descriptive** local-geometry statistic of the visible graph. Nothing here is
  evidence that a curvature-conditioned RQ-VAE improves recall.
