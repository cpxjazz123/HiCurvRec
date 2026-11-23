# Issue 249 frozen contract

2026-11-23. Infrastructure only: no training of any kind (GNN / Teacher / gating / RQ-VAE / Stage2 /
Stage3), no PPR, no new full 4-node graphlet/orbit enumeration, no re-training of TIGER, no loss
weight design and no architecture selection. The 768-dim semantic input and the three-layer SID
protocol are unchanged.

## 0. What is reused, and what is genuinely new

Reused **by immutable hash** from `assets/manifest.json` (never recomputed, never copied):

| Asset | Use |
|---|---|
| `assets/shards/shard_000..012.npz` | the 28,391,948-row full-resolution curvature table (#248) |
| `assets/index.npz` | item → (shard, start, count); expected per-centre counts |
| `assets/item_summary.npz` | per-item curvature distribution |
| `assets/distance_histogram.npz`, `assets/node_moments.npz` | exact distance intermediates |
| `assets/train/visible_{forward,reverse,undirected}.npz`, `assets/train/item_ids.npy` | the training-visible graph |
| `results/research/infra246/assets/ego/ego_full_1hop_visible.npz` | the full 1-hop induced ego (referenced, not rebuilt) |
| `results/research/infra246/assets/graph/{visible_*,split_users,held_out_support}.npz` | frozen splits and held-out support |
| `results/research/infra246/assets/pairs/*.npz` | relation banks (train / strict / relaxed) |
| `results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json` | the SID table Stage3 actually read (sha256 `025ad8bd…`) |

Genuinely new, and persisted under `assets/v5_gnn_geometry/` with its own manifest entries:
the ego batching index, the constant node-feature table and its standardization statistics.

## 1. The statistic (unchanged formula, two sides)

GraphMoRE Appendix A.1, Eq. A.1, over a reference set `R`:

```
kappa(m;b,c) = (1/|R|) * sum_{a in R} [ d(a,m)^2 + d(b,c)^2/4 - ( d(a,b)^2 + d(a,c)^2 )/2 ]
```

* **Target side** (`d` = unweighted hop distance on the undirected projection of the training-visible
  graph, `R` = the centre's connected component): exactly the #248 definition, frozen.
* **Quantized side** (`d` = a distance in the space the quantized vectors live in): the *same* formula,
  with the same reference indices `R` and the same triples `(m,b,c)`. The reference indices on the
  quantized side **must equal** the ones on the target side; the caller supplies them.
* The quantized-side value is a **distance-induced finite-difference statistic**, not the intrinsic
  sectional curvature of a manifold. It is labelled as such everywhere.
* The Euclidean metric is **not** identically zero: the centre `m` need not be the geometric midpoint
  of `b` and `c`, and `R` is finite. A test pins a non-zero Euclidean value on a hand-computed graph.
* Node-dependent weights are **caller-specified**, never baked in: the statistic is a weighted mean
  `sum_a w_a f_a / sum_a w_a` with `w_a >= 0`; uniform weights are the default. A weight is a
  modelling choice, **not** a curvature label.

## 2. Metrics: scale, sign, domain

* `d` is squared inside the statistic, so the statistic has units of (metric length)^2.
* The graph side is in **hop^2**; a quantized metric is in its own units. The two are **not** on the
  same scale unless the metric's scale is aligned explicitly.
* Alignment rule, fixed in advance: the reported scale factor is
  `s = sum(kappa_q * kappa_g) / sum(kappa_q^2)` over the evaluated set (a least-squares ratio, one
  global number), and the scale-aligned error is `|kappa_g - s * kappa_q|`. The unaligned error is
  reported too. **Sign agreement is the scale-free headline number** and is always reported.
* A metric must declare: family, curvature magnitude `k` and its sign convention, scale factor,
  domain, symmetry, behaviour at zero distance, and its stability (gradient magnitude near
  saturation).

## 3. Metrics: families

| Family | Definition | Domain |
|---|---|---|
| `euclidean` | `d(x,y) = ||x - y||_2` | all of R^d |
| `hyperbolic` | Poincaré ball, curvature `-1/kappa` with `kappa > 0`; `d = 2/sqrt(kappa) * artanh(sqrt(kappa) * ||x (+) y||)` | ball of radius `1/sqrt(kappa)` |
| `spherical` | stereographic sphere of radius `R = 1/sqrt(kappa)`, `kappa > 0`; `d = R * arccos(1 - 2||x-y||^2/((1+||x||^2)(1+||y||^2)))` | all of R^d |
| `mixed` | caller-supplied per-node weights over the families above, renormalised to sum to one | intersection of the used domains |

The `mixed` metric is a **pluggable callable**, not a frozen model. Families are implemented
independently of each other and each is unit-tested on hand-computed pairs.

## 4. Numerical contract

* Everything is computed in float64 unless the caller asks for float32; the tests use float64.
* Zero distance: `d(x,x) = 0` exactly for every family.
* Symmetry: `d(x,y) == d(y,x)` to 1e-12 for every family.
* Saturation: for `kappa` at the boundary of the domain, the distance may saturate; the caller must
  clamp inputs into the domain and the module **raises** on a non-finite input rather than
  propagating NaN.
* Gradients: the statistic must be differentiable in the quantized vectors; `d` at exactly
  coincident points is non-differentiable and the test uses distinct points.
* Tolerances: `atol = rtol = 1e-9` for float64 comparisons; `1e-6` for finite-difference gradient
  checks on a well-conditioned point.

## 5. Ego / GNN input contract

* Source: the training-visible directed graph only. The full-description graph and the
  struct-val / struct-test edges are never read.
* For every item: the centre index, the ego's **global** node ids and their **local** positions,
  the induced edges with their direction (out / in / both), the node and edge validity masks, and
  the per-node in-degree, out-degree and reciprocal degree.
* The full 1-hop ego is **reused** from the frozen #246 asset; the 2-hop ego is produced on demand
  from the visible CSR, deterministically, **never truncated**; a batch may therefore contain
  egos of very different sizes.
* Optional constant node features: per-node in/out/reciprocal degree and their logarithms, plus an
  isolated-node flag. **Standardization statistics are fitted on the training-visible items only**
  and persisted; no validation/test item contributes.
* **Forbidden as GNN input**: graph structure labels (family / orbit / graphlet class), the
  validation/test edges, and the 768-dim semantic embedding as a forced input.
* `collate` is deterministic and order preserving: the batch's item order is recoverable, the
  centre of every ego is recoverable, and one item collated alone equals the same item inside a
  batch (same local ids, same features).

## 6. Curvature reading contract

* The reader streams the 13 shards; it never materialises all 28,391,948 rows at once.
* Per-centre access: rows of one centre are contiguous inside one shard.
* The valid mask and the effective-reference count are exposed; missing references are never
  imputed as zero.
* A full traversal must visit every row exactly once (no duplicate, no omission); the traversal
  reports rows read, distinct centres, memory high-water mark and wall time.
* Batched exact aggregation (e.g. per-item means) must equal the single-item computation bit for
  bit on the same data.

## 7. TIGER comparison contract

* The SID table is the one Stage3 read: `results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json`,
  sha256 `025ad8bd…`. It is never re-trained.
* **Pure SID metrics** (prefix-3 collision, duplicate share, per-level codebook usage and entropy,
  prefix sharing) are recomputed directly from that file.
* **Geometry metrics** need a codebook / `q` / checkpoint **provably from the same SID generation**.
  The frozen `rqvae_best.pth` belongs to a *different* generation (#246 measured the re-export as
  differing in every one of the 24,587 rows), so the geometry metrics for the Stage3 table are
  reported as **NA** with the reason.
* The checkpoint-consistent pair (that checkpoint with its own re-exported table) is reported
  separately and explicitly labelled as *not* the Stage3 table.

## 8. Evaluation isolation

* The evaluation entry point serves the strict held-out banks only; the training side cannot read
  the held-out targets through it.
* Continuous `z` and hard-SID-decoded `q` are reported **separately**; continuous performance never
  substitutes for the discrete SID.

## 9. Independent reference and mutation testing

The reference implementation (`results/research/issue249/reference.py`) does not import the
production modules: it evaluates Eq. A.1 with explicit Python loops over hand-written distance
matrices, and it recomputes the graph-side target from the persisted distance histogram.

Module-level mutations (≥5, covering numeric / structural / routing) must each be caught by a
**named assertion**; crashes and import errors do not count.
