# Issue 248 frozen contract

2026-11-23. Infrastructure only: no training, no PPR, no new loss, no TIGER retraining, no change
to the 768-dim input or the three-layer SID protocol.

## Scope of the asset family
Version `v4-full-sectional-248`. It supersedes the **sampled** curvature of #247
(`results/research/issue247/assets/`, version `v3-sectional-247`) for curvature only. The #247
assets stay on disk untouched and are referenced by hash as the historical version. Every other
#246/#247 asset is reused by immutable hash reference; nothing is re-uploaded or duplicated.

## 1. The statistic (unchanged from #247, which was itself verified against the paper)

GraphMoRE, arXiv 2412.11085v1, Appendix A.1, Eq. A.1 — sectional curvature of the geodesic triangle
composed of the centre `m` and two of its neighbours `b, c`:

```
kappa(m;b,c) = mean_{a in R(m)} [ g(a,m)^2 + g(b,c)^2/4 - ( g(a,b)^2 + g(a,c)^2 )/2 ]
```

* `g` = unweighted shortest-path distance on the **undirected projection** of the training-visible
  directed graph. The result is therefore *not* a directed sectional curvature; direction is kept
  separately (reciprocal flags, directed neighbourhoods, directed 3-node structure classes).
* Ricci curvature (Ollivier/Forman) and triangle counts are **not** substitutes and are not used.
* The official GraphMoRE repository (commit `1948d3c5f19f7251cadd8d1118c06f0e294f96b3`) publishes no
  sectional-curvature estimator (only `utils.cal_shortest_dis`, a networkx shortest-path helper);
  the numerical comparator is therefore an independent implementation of the paper equation
  (`results/research/issue248/reference.py`, networkx BFS + explicit summation over references),
  never a claim about official curvature code.

### Reference set and exclusion rules (this is the #248 change)
#247 estimated the mean over a fixed sample of 256 references. #248 uses the **full** reference
set, so the estimand changes and the old numbers are not comparable without recomputation:

* `R(m)` = every node reachable from `m` in the undirected projection = the connected component of
  `m`. Unreachable nodes are excluded from the mean (distance is undefined there) and their count is
  reported as `component_outside`, never imputed as zero.
* **No vertex is excluded from the reference set**: `a = m`, `a = b` and `a = c` are valid references,
  exactly as in the paper's sum over `a in V` and as in the #247 contract. The reference count is
  `|R(m)|` = the component size.
* Both conventions are reconstructible from the stored assets: the per-node distance histogram
  (Section 3) plus the per-triple `g(b,c)^2` reproduce the paper-literal value published in the
  table, and the triple-excluded variant is obtained by the closed form in Section 4. The table
  publishes the **paper-literal** value.

### Exact closed form used by the production code
With `S2(v) = sum_{a in R(v)} g(a,v)^2` and `N = |R(m)|` (and `b, c` in the same component as `m`):

```
kappa(m;b,c) = ( S2(m) + N * g(b,c)^2 / 4 - ( S2(b) + S2(c) ) / 2 ) / N
```

This is an algebraic rearrangement of the mean over the full reference set, not an approximation.
It is verified against four hand-computed graphs (path 0, star -0.5, triangle 0.25, four-cycle 1) and
against the independent reference implementation on small graphs and on sampled real triples.
The triple-excluded convention is
`kappa_excl = ( S2(m) - 2 + (N-3) * g(b,c)^2/4 - ( (S2(b) - 1 - g) + (S2(c) - 1 - g) )/2 ) / (N-3)`
with `g = g(b,c)^2` (neighbours of `m` are at distance 1 from `m`).

## 2. Triple enumeration (full, no sampling)

* The neighbour set of `m` is the neighbour set of the **undirected projection** (the union of the
  in- and out-neighbours of the visible directed graph).
* Every unordered pair `{b, c}` of distinct neighbours of `m` is one triple. `b < c` by index.
* Total count is recomputed from the visible graph itself; the previously reported 28,391,948 is
  treated as a claim to check, not as a constant. It must equal `sum_m C(deg(m), 2)`.
* No random sampling, no per-centre truncation, no silent omission. Per-centre counts must equal
  `C(deg(m), 2)` exactly, and the global count must equal the sum of the per-centre counts.
* Triple identity is `(m, b, c)` with `m` the centre; the pair `{b, c}` is unordered and stored with
  `b < c`. The same 3-node set can appear with different centres; those are different triples.

## 3. Reusable exact distance intermediates

For every node `v`: the **distance histogram** `h_v[k] = |{a in R(v) : g(a,v) = k}|` (exact
integers), from which `S2(v)`, the linear distance sum, the eccentricity and the component size
follow exactly. The histogram is stored in a ragged layout (`offsets`, `counts`) and is lossless:
any moment of the distance distribution can be recomputed from it.

Also stored per node: component id, component size, `S2`, `S1`, eccentricity, undirected degree,
directed in/out degree, reciprocal degree.

## 4. Stored tables

### 4.1 Triple table (sharded)
Columns per row: `centre`, `b`, `c` (int32), `bc_dist_sq` (int8, `g(b,c)^2` in {1, 4}),
`curvature` (float64, the paper-literal value), `effective_references` (int32, `|R(m)|`),
`valid` (bool), `invalid_reason` (int8 code, 0 = none), `structure_class` (int8, the canonical
directed 3-node graphlet class of the induced subgraph on `{m, b, c}` in the visible directed graph,
reusing the #246 canonical-code machinery).

### 4.2 Per-item distribution
Per item: total triples, valid/invalid counts, positive/zero/negative counts **and** ratios
(zero tolerance 1e-12), mean, variance (population), quantiles (0, .25, .5, .75, 1), minimum,
maximum, and the missing reason (`no_triple` when the item has fewer than two neighbours;
`isolated` when it has no neighbours at all).

### 4.3 Index / query interfaces
* by item: `(shard, start, stop)` into the shard table, plus the aggregates;
* by triple: exact lookup of `(m, b, c)`;
* by structure class: counts per class and an iterator over the rows of a class.

## 5. Boundary conditions and tolerances
* Empty neighbour set -> no triple, item marked `isolated`; degree 1 -> no triple, `no_triple`.
* Unreachable reference nodes -> excluded from the mean, reported as `component_outside`, never
  imputed as zero.
* `b == c`, `b == m`, `c == m` -> rejected at construction (not representable).
* Zero classification tolerance: 1e-12 (absolute, on the squared-hop scale).
* Reference comparison tolerance: atol 1e-12, rtol 1e-12 (float64 on sums of at most 24,587 terms).
* Variance is the population variance (ddof = 0); quantiles use the linear interpolation of
  `numpy.quantile` and the implementation is recorded.
* Missing values are `NaN` with an explicit mask; `NaN`/`Inf` never enter a mean.

## 6. Isolation
* Every curvature, distance, neighbourhood and structure asset is derived from the **training-visible
  graph only** (45,951 visible users; `struct_val` and `struct_test` users never enter).
* Evaluation assets (strict banks, `held_out_support`, struct splits) are physically separate
  directories and are marked `use = evaluation` in the manifest; the training interface refuses to
  load them.
* The full-description graph is never read by any v4 builder.

## 7. Storage and manifest
* Binary assets live under `results/research/issue248/assets/` and stay out of git under the
  repository's existing `results/research/**` rules (§92); the manifest, coverage report, acceptance
  report and documentation are committed.
* Shards are compressed `.npz`; the triple table is split by centre ranges so a single item's rows
  are contiguous and randomly accessible, and the shard merge is verified.
* The manifest records for every asset: path, version, schema, shape, dtype, rows, bytes, sha256,
  input hashes, generating commit, generating command, coverage and legal use
  (`training` / `evaluation` / `description`), plus total runtime and storage cost.

## 8. Independent reference and mutation testing
The reference implementation (`results/research/issue248/reference.py`) does not import the
production module; it builds the graph with `networkx`, computes distances with BFS and sums the
Eq. A.1 terms explicitly over the reference set (no closed form, no precomputed `S2`). On the real
graph, a second independent recomputation uses explicit per-source distance rows from SciPy BFS
(no `S2` precomputation) on sampled triples.

At least five mutations must be caught by a **named assertion** (crashes and import errors do not
count): numeric (wrong coefficient / wrong sign / wrong reduction), structural (missing mask,
wrong denominator, wrong `g(b,c)`), routing (wrong reference set, wrong graph, wrong ID alignment).
