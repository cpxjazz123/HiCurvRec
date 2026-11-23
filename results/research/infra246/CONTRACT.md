# CONTRACT v2 — issue #246 (supersedes v1 for all new assets)

v1 (`results/research/infra245/CONTRACT.md`) stays readable for traceability. **v2 assets never mix with
v1 assets**: every v2 file carries `version: v2` and its own `graph_version`; `assets.assert_legal_use`
refuses a cross-version join.

## 1. Identifiers

| item | rule |
|---|---|
| item id | actual set read from `items.parquet` (not `max+1`); must equal `0..n-1` and match `sentence_t5.npy` row count and `item_sids.json` keys — otherwise the build stops |
| user id | integer from `train.parquet` |
| pair | `pair_directed = (src, dst)` int64; `pair_undirected = (min, max)` int64; **separately stored** |
| graph version | `v2/{split}/{sha256-of-source-parquet}`; every asset records it |
| node index | global item id everywhere except inside an ego block, where `root_local` gives the centre row |

## 2. Splits (frozen before any asset is built)

Three disjoint user sets inside **train**, `split_seed = 246001`, ratios 80/10/10:

| name | role | may be used for |
|---|---|---|
| `visible` | propagation graph | ego, degree, motif, distance, supervision pairs, PPR, reference points |
| `struct_val` | structural validation | evaluation only; never propagation or supervision |
| `struct_test` | structural test | evaluation only; never propagation or supervision |

Invariants (all checked): pairwise intersection = ∅; union = every train user; edge-weight sum of the
three graphs = the full graph's sum; these are **not** the Stage3 val/test splits.

`full_description` = all train users. It exists for descriptive statistics only.

## 3. Graph construction

```
edge (i→j)  ⇔  j immediately follows i inside one user's train sequence
weight      =  number of occurrences (int64)
self loops  =  dropped; the actual dropped count is recorded (not assumed 0)
undirected projection = simple: one entry per unordered pair, weight = w(i→j) + w(j→i)
```

Formulas:

* `nnz(forward) = |{distinct (i,j), i≠j}|`
* `nnz(undirected) = |{{i,j} : i≠j, (i→j) or (j→i)}|` and `nnz(undirected) = 2 · |edges|` for a
  symmetric store.
* cross-check: an **independent** per-sequence counter (`reference_hand.sequence_edges`) must equal the
  CSR edge set exactly (symmetric difference = 0), not just the total.

## 4. Held-out blocking (the v1 bug)

v1 used `eval - eval.multiply(visible)`, i.e. **weighted** subtraction; a visible weight larger than the
eval weight drives the difference negative and the held-out direction is destroyed.

v2 definition, on boolean support:

```
V   = support(visible)            # bool
E   = support(eval)
H   = setdiff(E, V)               # ordered held-out-only edges
H_u = sym_union(H) = H ∨ Hᵀ       # undirected block set
blocked(p) = H_u[min(p), max(p)]
```

A training bank, the gradient probe and any future sampler must contain **no** pair p with
`blocked(p)`. Criterion: for every pair p in a training bank, `H_u` is 0 in both directions.

Known answer (used as a test): visible `(7→8)` weight 3, eval `(7→8)` weight 1, eval `(9→8)` only →
`H = {(9→8)}`, `H_u` contains `{8,9}` and does **not** contain `{7,8}`.

`user-disjoint` and `pair-disjoint` are different claims:

* user-disjoint: `struct_val`/`struct_test` users never enter `visible` (checked on user sets);
* pair-disjoint strict: the strict eval bank's undirected keys are absent from every training bank, the
  probe and the sampler (checked on pair sets);
* a pair that exists in the visible graph **and** is observed by an eval user is user-isolated only —
  it is listed in a separately named relaxed bank, never merged into the strict one.

## 5. Ego subgraphs

```
nodes         = root ∪ sampled/full neighbours
edges         = ALL directed edges of the source graph with both ends in nodes (induced)
hop           = shortest hop in the SOURCE graph: 0 root, 1 direct neighbour, 2 otherwise (recomputed)
traversal_depth = BFS depth at which the node was discovered (kept separately)
role_out / role_in = real CSR membership of the root (never another sample)
empty         = no neighbour
cap           = maximum number of NEIGHBOURS; the root is extra (audited: max block size = cap+1)
```

`root_local` is the row of the centre; no assumption on id order.

`batch_egos` returns `nodes`, `node_mask`, `edges` (local), `edge_mask`, `features` (n×7, masked),
`root_row`. Padding is right-side and never attendable.

Two versions per graph: `full_1hop` (deterministic, complete, for every item) and `sampled` (seeded,
cap, frozen). Full 2-hop blocks exceed the storage budget, so they are produced deterministically on
demand by `ego_full(radius=2)` and the frozen asset records per-item 2-hop **size** statistics plus a
declared sample; that is stated, never called full.

## 6. Exact 3-node graphlets and orbits

For a centre `c` and two distinct neighbours `a,b` (union of in/out neighbours, excluding `c`), the
induced directed pattern is the 6-bit code over the ordered pairs `(c,a),(c,b),(a,c),(a,b),(b,c),(b,a)`
in that fixed order.

```
canonical(code) = min over the 6 permutations π of code∘π     # labeled → unlabeled
graphlet_id     = canonical(code)                              # 0..63
family          = name of the canonical unlabeled pattern (16 of them)
orbit(position) = the automorphism orbit of that position in the canonical pattern
centre_role     = (in-degree, out-degree) of the centre inside the instance
```

The 16 unlabeled classes map to: `empty`, `single_edge`, `chain`, `out_star` (fork), `in_star` (merge),
`reciprocal_pair`, `ff_triangle` (transitive), `fb_triangle` (directed cycle), `path_rev_extra`, … and
`other_3node` for the remainder; the mapping is generated from the canonical codes, not hand-typed.

Definitions the contract distinguishes and never conflates:

| name | definition |
|---|---|
| `fork_relaxed(c)` | ∃ two out-neighbours `a,b` with **no edge between them** (either direction). The rest of the neighbourhood may be anything |
| `merge_relaxed(c)` | symmetric over in-neighbours |
| `strict out_star instance` | an instance whose induced pattern **is exactly** `c→a, c→b` and no other edge |
| `undirected_triangle` | three nodes pairwise adjacent in the undirected projection (orientation-agnostic) |

Known answer (test): successors `{a,b,d}` with `a—b` adjacent and `d` adjacent to neither → `fork_relaxed
= True` (v1 returned False).

Automorphism cross-check: `canonical()` is verified against brute force on all 64 labeled patterns and
against an isomorphism search on random 3-node graphs.

## 7. Distances and propagation

| field | unit | note |
|---|---|---|
| `hop_full_visible` | hops | undirected shortest path on the visible graph; `-1` = unreachable, mask column `reachable` |
| `hop_directed_src_dst` | hops | directed reachability src→dst; `-1` = not reachable |
| `hop_subgraph` | hops | shortest path inside the fixed local ego; `-1` = not connected inside |
| `ppr` | probability mass | restart `α=0.15`, tolerance `1e-10`, no truncation; residual reported. **本轮 owner 指示不做**:bank 不带 ppr 列(见 VERDICT 的 owner-withdrawn 行);`tools/graph_infra/ppr.py` 仍保留且有测试,后续需要时可直接启用 |
| `two_hop_count` | integer | number of directed 2-paths src→k→dst |
| `jaccard_out`, `jaccard_in` | [0,1] | empty denominator = `missing` flag, never 0 |
| `bhattacharyya_out` | [0,1] | empty distribution = `missing` flag |

PPR is **not** a distance and is stored under its own name and unit. PPR is computed by solving
`(I − (1−α)Pᵀ) x = α e_s` once per source with a sparse LU factorisation; residual ‖LHS·x − RHS‖∞ is
reported per source. **Every unique source of every bank** is covered, not a probe subset.

Unobserved pairs are never labelled "negative": a bank row carries `label_evidence` (witness nodes or
path) and `label_kind ∈ {observed_edge, common_successor, common_predecessor, two_hop_chain,
degree_matched_unobserved}`; `unobserved` is a fact, not a negative label.

## 8. Curvature on the visible graph

`#236` curvature is undirected and computed on the **full** train graph; it cannot be transplanted.

v2 builds its own fixed reference set on the visible graph: `reference_count = 2000` items chosen by a
frozen seed, and for each, exact Ollivier–Ricci on a sampled incident edge set:

```
W1(μ_x, μ_y) = min over couplings π of E_{(u,v)~π}[d(u,v)],  μ_x = uniform(N(x)),  μ_y = uniform(N(y))
κ(x,y) = 1 − W1(μ_x, μ_y) / d(x,y)
node κ(x) = mean of κ over the sampled edges incident to x
```

`solve_small_lp` uses `scipy.optimize.linprog(method="highs")`; the edge sample is stratified by degree
and its coverage is recorded. Unit: dimensionless. Missing (isolated / dangling) is reported, not zero.

## 9. Structural metrics

```
direction="minimize"  →  pass(v) = v ≤ threshold;  worst = max;  per-centre p90 = quantile(v, 0.9)
direction="maximize"  →  pass(v) = v ≥ threshold;  worst = min;  per-centre p90 = quantile(v, 0.1)
subgraph_pass_fraction      = #{passing subgraphs} / #{valid subgraphs}
center_all_pass_fraction    = #{centres with every subgraph passing} / #{centres with ≥1 valid subgraph}
center_any_pass_fraction    = #{centres with ≥1 passing subgraph} / same denominator
centers_without_valid_subgraph  reported separately
```

Known answer: `E=[0.3, 0.4, 0.8]`, `threshold=0.5`, `direction="minimize"` → subgraph pass `2/3`,
`center_all_pass 0`, `worst 0.8`, `p90 0.72` (numpy linear interpolation).

`prefix_at_k`: both codes must have length ≥ k and the first k positions must match pairwise.

SID metrics are reported twice: over the first three codes and over the full code (including the
disambiguation token). `entropy` states its unit: nats by default, bits when requested.

SID audit validates: length per row, value range, key coverage of the item set, duplicates, and the
mapping to the item-id order.

## 10. Join and manifest

`join_scores_v2(bank, model_pairs, model_scores, *, expected_graph_version)` requires:

* `model_pairs` shape `(N,2)`, dtype integer; `model_scores` shape `(N,)`, dtype float, all finite;
* the **set** of model pairs equals the bank's set exactly (missing, extra and duplicates each raise);
* reordering is allowed only through the pair id; positional alignment is never assumed;
* `expected_graph_version` must equal the asset's own version.

Manifest v2 entries carry: `name, path (repo-relative), schema, shape, dtype, graph_version, legal_use,
version, purpose, command, source_sha256, code_sha256, dependencies, sha256, bytes`.
`legal_use` ∈ {`description`, `generalisation`, `evaluation`, `reuse`};
`assert_legal_use(name, use)` raises when a generalisation reader asks for a description-only asset or
when a version mismatch is detected. `verify_manifest` never rebuilds and never writes: on drift it
raises, leaving the frozen expectation intact.

## 11. Paper / official-implementation comparison

| module | reference | what the reference says | our choice | why |
|---|---|---|---|---|
| Ollivier–Ricci | Ollivier 2009; Lin–Lu–Yau 2011 | `κ = 1 − W1/ d` with uniform neighbour measures | same, exact LP per sampled edge | small degrees make the LP exact and cheap |
| PPR | Brin–Page 1998 | `x = α e_s + (1−α) Pᵀ x`, `α` restart | solved exactly via sparse LU, residual reported | avoids "did it converge" ambiguity |
| Bhattacharyya | Bhattacharyya 1943 | `Σ √(p_i q_i)` over the shared support | same; empty support = missing | avoids silent zero |
| graphlet orbits | Pržulj 2004/2006 | induced subgraph isomorphism classes; orbits = automorphism classes | exact 3-node canonicalisation by min-code over permutations | 64 labels enumerable, verifiable |
| Spearman | standard | average ranks for ties | `metrics.spearman` with average ranks; constant → NA | matches scipy on non-tied inputs |
| Jaccard | standard | `|A∩B|/|A∪B|` | same; empty ∪ = missing | avoids 0/0 = 0 |
| kNN/kappa-stereographic | Geoopt `Stereographic` | `exp/log/proj` with the curvature sign convention | this module: k<0 = Poincare ball, k>0 = stereographic sphere, k=0 = flat limit; **both** branches use the 2x convention so that `d(0, expmap0(v)) = 2|v|` and k=0 is continuous | v1 mixed the conventions (the ball used 1x exp/log with a 2x distance) and its k=0 branch returned the plain Euclidean norm, a factor-two jump; corrected here |

Tolerances: float64, `1e-9` for closed-form geometry, `1e-6` for iterative quantities (LP, LU solves),
`1e-12` for integer-count equalities.

## 12. Curvature convention (v2 correction, supersedes v1 §1)

The public module uses `k` as the *model parameter*: `k < 0` is the Poincare ball of radius
`1/sqrt(|k|)`, `k > 0` is the stereographic sphere of radius `1/sqrt(k)`, and `k = 0` is flat.

* distance: ball `2 * tan_k^{-1}(|(-x) +_k y|)`, sphere `R * arccos(...)`, flat `2 * |x - y|`;
* exponential map: ball `tanh(sqrt|k| |v|)/sqrt|k|`, sphere `R tan(|v|/R)`, flat `v`;
* logarithm: ball `atanh(sqrt|k| |p|)/sqrt|k|`, sphere `R atan(|p|/R)`, flat `p`.

Invariants, all asserted in `tests/test_v2_geometry.py` and `tests/test_known_answers.py`:

```
d(0, expmap0(v)) = 2 |v|        for every k, including the limit
lim_{k->0} d_k(x, y) = 2 |x - y|
```

v1 violated both (`d(0, expmap0(v)) = |v|` on the sphere branch, and the k=0 branch returned
`|x - y|`). The v2 convention keeps every v1 distance value for `k != 0` and only changes the
sphere's exp/log and the flat branch. The sphere's injectivity radius is `pi R`, i.e.
`|v| < pi R / 2` under this convention.

## 13. Two collision calibers, both reported

| name | formula | TIGER JSON | TIGER NPY | checkpoint re-export |
|---|---|---|---|---|
| `collision_prefix3` | `1 - distinct(first 3 codes) / items` | 0.169276 | 0.110384 | 0.109936 |
| `duplicate_prefix3` | share of items whose first 3 codes are shared with another item | 0.265303 | 0.184813 | 0.184244 |

The Stage3 gate (CLAUDE.md §89) uses `collision_prefix3`; the §63 metric record's `l3_collision`
equals `duplicate_prefix3` for the same table (its methodology line says "item-weighted share of
repeated three-token SIDs"). **The v1 report claimed the record's value was irreproducible; that
claim was wrong** and is corrected in `INVENTORY.md` (v2 correction note).

## 14. What v2 does not decide

Curvature weights, temperatures, gates, expert scales and λ stay with a future candidate. v2 fixes the
probe pairs, the banks and the join, nothing else.
