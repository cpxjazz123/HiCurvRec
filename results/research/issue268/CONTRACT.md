# Issue #268 frozen math and data contract

Frozen before candidate implementation. Parent: #265 (`TIGER_euclid_local_curve_supervision_rq`,
R@10 `0.052594926791900976`). #267 is evidence only; its relative-ranking arm is not a parent.

## 1. Three different objects (never interchangeable)

| symbol | meaning | source | may be a training target? |
|---|---|---|---|
| `kappa_graph(m;b,c)` | frozen discrete graph sectional proxy, Eq. A.1 | `assets/v8_runtime_layout/targets.npy` (unchanged) | yes, this is the target |
| `kappa_t` | continuous constant curvature fitted for one legal triple under the frozen reference representation and the frozen graph distances | `results/research/issue268/kappa_cache/kappa.npy` | no, it enters the distance only |
| `kappa_hat(q;kappa_t)` | the same moment formula evaluated on hard `q` with the kappa-stereographic distance | model forward | no, it is the prediction |

The frozen target definition is not modified:

```
kappa_graph(m;b,c) = ( S2(m) + N*g(b,c)^2/4 - (S2(b)+S2(c))/2 ) / N
S2(v) = sum_{a in R(v)} g(a,v)^2 ,  N = |R(v)|
```

`g` is the unweighted shortest-path distance on the undirected projection of the training-visible
graph (`assets/train/visible_undirected.npz`, 24587 nodes, 510464 undirected edges). Directed labels
are never treated as symmetric distances. The packed target table has 28,391,948 rows,
24,326 centres with a valid curvature (`assets/index.npz`, `assets/item_summary.npz`).

## 2. Frozen reference representation Z_ref

- `Z_ref` = final continuous 32-d encoder output of the parent #265 checkpoint on all 24,587 frozen
  768-d semantic embeddings (PCA is off, `PCA_DIM=0`).
- Checkpoint: `results/stage2_RQ-VAE/_runs/stage2_4891fa8df3e843839de770e52e156892/TIGER_euclid_local_curve_supervision_rq/result/rqvae_best.pth`,
  sha256 `1f8bfecba0c74bb3cbf9da11fe6925983cbae012753100401a7168e246da7bef`.
- Source snapshot: the immutable `src/` of the same run.
- `artifacts/z_ref.npy` sha256 `a2643a9bc09bbedffe4c249fea7d07d54153c9c8fd6aa5e476fe4f7b63ae4a66`,
  shape `[24587, 32]`, float32, max norm `0.5954992175102234`.
- Z_ref was produced before any candidate weight existed; no candidate `q`, no test interaction and
  no label from test is used.

## 3. Frozen coordinate map and distance

Global scale `S = max_i ||Z_ref_i|| = 0.5954992175102234`; coordinates `w = Z_ref / (2*S)`, so
`||w_i|| <= 0.5` and the Poincaré ball of curvature `-1` (radius 1) contains every point with
margin. The kappa-stereographic distance follows the repo's `tools/graph_infra/manifold.py`
convention (Bachmann et al., ICML 2020; GraphMoRE Appendix A.1):

```
k = 0:      d_0(x,y) = 2*||x-y||                (squared: 4*||x-y||^2)
k < 0:      d_k(x,y) = (2/sqrt(a)) * asinh( sqrt( a*||x-y||^2 / ((1-a||x||^2)(1-a||y||^2)) ) ),  a=-k
k > 0:      d_k(x,y) = (2/sqrt(k)) * asin ( sqrt( k*||x-y||^2 / ((1+k||x||^2)(1+k||y||^2)) ) )
```

With `w = Z_ref/(2S)` this gives `d_0(w_i,w_j) = ||Z_i - Z_j|| / S`, the same Euclidean scale as the
`k=0` member of the family. `d_k^2` is continuous at `k=0`. Arguments of `asin`/`asinh` are clamped
to their legal domains; a zero denominator (only possible outside the frozen coordinate box) is a
hard error, never silently repaired. Coincident points give distance 0 and are excluded from the
relative distortion by the `g>0` mask.

Paper / implementation comparison:

| module | paper | official code | this project |
|---|---|---|---|
| distance | Bachmann et al. 2020, Eq. 1/κ-stereographic | `geoopt` `Stereographic`; repo `manifold.dist` | same convention, re-implemented in the candidate with a stable `asin`/`asinh` branch and a k=0 branch that matches exactly |
| curvature domain | κ ∈ R; κ>0 sphere, κ<0 ball | continuous | fit domain from the frozen coordinate box: `[-0.125, 1.0]`, grid step `0.005`, `κ=0` always included |
| target | GraphMoRE A.1 (graph statistic) | `tools/graph_infra/sectional_full.py` | unchanged table, never refitted |

## 4. Frozen distortion and fitting procedure

For a target row `t = (m;b,c)` let `R` be the frozen deterministic reference subset of the item's
connected component (256 references, `numpy.random.default_rng(268)`, sorted; the giant component
has 24,471 members, so a subset is used and reported). The pair population is

```
P_t = { (m,r), (b,r), (c,r) : r in R, g(.,r) > 0 }  union  { (b,c) }
J_t(k) = sum_{pairs in P_t} ( d_k(w_i,w_j)/g(i,j) - 1 )^2  /  |P_t|
```

Exact node decomposition used by the fit (algebraic identity, no approximation):

```
f_k(v) = sum_{r in R, g(v,r)>0} ( d_k(w_v,w_r)/g(v,r) - 1 )^2
n_k(v) = #{r in R : g(v,r) > 0}
J_t(k) = ( f_k(m) + f_k(b) + f_k(c) + (d_k(w_b,w_c)/g(b,c) - 1)^2 ) / ( n_k(m)+n_k(b)+n_k(c)+1 )
```

- One-dimensional bounded search: uniform grid `k = -0.125, -0.120, ..., 1.000` (226 values, `k=0`
  exact), each value evaluated exactly from the decomposition. The lower end is set by the
  frozen coordinate box (`max||w|| = 2.8118`, so `|k| < 1/max||w||^2 = 0.1265`, rounded toward zero),
  not by a tuned constant; the executed cache records this domain. `kappa_t = argmin J_t(k)`, with
  `kappa_t = 0` on exact ties (a tie is resolved toward the Euclidean baseline, never toward noise).
- Reported per row: `kappa_t`, `J_t(0)`, `J_t(kappa_t)`, improvement fraction, `argmin` uniqueness
  (gap to the second-best grid value), boundary flag (`|kappa_t| = 1`).
- Failure/degenerate policy: rows with non-finite target are marked invalid and never assigned a
  curvature; rows whose component has fewer than 2 references are marked degenerate; no invented
  value, no imputation.
- The fit uses **only** the training-visible graph, the frozen target row identity and Z_ref. It
  never reads Stage2/Stage3 results, never reads test interactions, and is frozen before the
  candidate is trained.

## 5. Independent validation of the fit

1. A second deterministic reference subset (seed 269, same size) recomputes `J_t(0)` and
   `J_t(kappa_t)` on a deterministic sample of 20,000 rows; report the subset-to-subset change of
   the improvement fraction and of the sign of `J(0)-J(kappa)`.
2. A full-reference audit recomputes both distortions on 512 deterministic rows using **all**
   component references (24,471) instead of the subset; report the difference from the subset
   estimate (`analysis/FULL_AUDIT.json`).
3. The decomposition identity is verified against a direct pair-wise recomputation on a
   deterministic sample (`analysis/probe_J.json:decomposition_max_abs_error`).
4. Sign/structure stratification: fitted curvature by graph structure class
   (`assets/shards/*:structure_class`) and by `kappa_graph` sign, reported but not used for tuning.

## 6. Training mechanism (candidate tree only)

- Euclidean residual selection is unchanged: `C_l = ||r_l - E_l||_2^2`, floor + last-level
  Sinkhorn, original residual update, hard IDs.
- The trainable gate/token attention is removed; the candidate has no GNN, attention or gate
  parameters and therefore no unused trainable parameter under DDP. Encoder + three codebooks are
  the trainable modules.
- `hard q_i = sum_l c_i,l` is the forward value; the backward path is the parent STE surrogate.
  The curvature term never uses per-level prefixes during training.
- For the sampled triple `t` (row id supplied by the frozen structural sampler), all distances in
  that triple use the same frozen `kappa_t` and the same frozen scale `S`:

```
pred_t = mean_r d_{kappa_t}(q_m,q_r)^2 - (mean_r d_{kappa_t}(q_b,q_r)^2
        + mean_r d_{kappa_t}(q_c,q_r)^2)/2 + d_{kappa_t}(q_b,q_c)^2/4
L_curvature = mean_t Huber( pred_t / target_scale , kappa_graph_t / target_scale , delta=1 )
L_total = L_RQVAE + lambda * L_curvature
```

- `kappa_t` is a non-trainable table indexed by target row id; it is hashed at model init and is
  identical before and after training. No adaptive kappa, no free scale, no ranking loss.
- `lambda` calibration keeps the parent rule: one global budget computed on the actual first batch
  before the first optimizer step, over `{encoder, codebook0, codebook1, codebook2}`, with the
  parent's `min(||g_original|| / ||g_curvature||)` and the parent's 0.2 factor on the point term
  is not used as a free knob. With the gate removed the budget groups are exactly the encoder and
  the three codebooks; the budget construction is `calibration.norm_budget_weight`, unchanged.

## 7. Gradient route

`q` carries gradient to the encoder and the selected codewords through the parent STE; the
curvature term adds a gradient to the same encoder/codebook parameters through the kappa distance.
`kappa_t`, `Z_ref`, the graph and the target table have `requires_grad=False` and no optimizer entry;
their sha256 is checked before and after training. Detaching the kappa index must not detach `q`:
the test asserts `q.grad`-carrying parameters receive nonzero gradient from `L_curvature` alone.

## 8. Pass conditions before formal training

1. `kappa=0` reproduces `2*||x-y||` to `1e-6` relative; `kappa`-continuity across `±1e-3` has no
   jump larger than `1e-4`; sphere and ball branches match `tools.graph_infra.manifold.dist` within
   `1e-4` on random points in the frozen box.
2. Hand-computed 3-node geometry, coincident points, boundary `||w||=0.5`, `k=±1` and triple-index
   tests pass.
3. Mutation tests (≥5, numeric/sign/mask/reduction/routing) are each caught by a named assertion.
4. Module check on a real training batch with the actual loss: curvature gradient reaches encoder
   and codebooks; kappa table and Z_ref have no gradient; parameter update changes q.
5. Pre-train probe on the frozen parent checkpoint: fixing q freezes `pred_t` under any kappa-table
   change; changing a row's kappa changes `pred_t` where the triple is active; the geometry differs
   from Euclidean.
6. Real-entry 1-epoch smoke with the frozen single-card entry mode (admission `check_entry`) passes
   if the host has one card; the formal four-card Stage2 additionally requires four visible cards.

## 9. Non-goals

No GNN/attention/gate, no adaptive curvature, no per-triple scale, no ranking loss, no extra
baseline, no seed search, no re-fit after seeing Stage3, no public infra or asset modification.
