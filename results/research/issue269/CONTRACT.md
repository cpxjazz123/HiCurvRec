# Issue #269 frozen math and data contract

Frozen before the Stage2 candidate is dispatched (2026-11-27). Parent: #265
(`TIGER_euclid_local_curve_supervision_rq`, R@10 `0.052594926791900976`); #268 supplies the
frozen reference representation, the target table and the negative graph-unit-scale result.

## 0. Targeted root cause, causal path, falsifiable prediction, decision metric

- **Root cause under test.** #268 failed because the per-triple fit used the graph-unit scale:
  `max||Z/(2S)|| = 0.5` put 99.78 % of the legal triples on the coordinate floor, and one global
  Euclidean scale (`S* = 0.068284`) reached lower distortion than the whole per-triple family.
  The failure is a *scale* failure of the ruler, not evidence that local curvature carries no
  information.
- **Causal path.** Fix the global scale by weighted least squares (`S*`), refit `kappa_t` in the
  calibrated chart `w = Z/(2S*)`, and use the same fixed `S*` and `kappa_t` for every distance of
  a sampled triple in the point-value curvature objective. `S*` is frozen; no per-triple scale,
  no adaptive kappa, no gate, no GNN.
- **Falsifiable prediction.** Relative to one global curvature fitted on the same rows, the
  per-triple fit lowers the held-out mean distortion by at least 1 %, its centre bootstrap lower
  bound is above zero, and a reference-perturbation refit has Spearman >= 0.5. If any fails, the
  candidate must degrade to `kappa = 0` at `S*` and must not be described as local curvature.
- **Decision metric.** Stage3 `test_R@10` against frozen TIGER, the parent and the 0.065 target.
  Continuous geometry error is a diagnostic, never a substitute for SID/recall.

## 1. Objects (never interchangeable)

| symbol | meaning | source | training target? |
|---|---|---|---|
| `kappa_graph(m;b,c)` | frozen discrete graph sectional proxy, Eq. A.1 | `assets/v8_runtime_layout/targets.npy` (unchanged) | yes |
| `S*` | one global Euclidean/coordinate scale, weighted least squares | `results/research/issue269/analysis/GATE.json`, `0.06828367045803868` | no, frozen |
| `kappa_t` | continuous constant curvature of one legal triple under `Z_ref` and `S*` | `results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284/kappa.npy` | no, enters the distance only |
| `kappa_hat(q;kappa_t)` | the A.1 moment formula evaluated on hard `q` in the calibrated chart | candidate forward | no, prediction |

The target definition is unchanged from #265/#268:

```
kappa_graph(m;b,c) = ( S2(m) + N*g(b,c)^2/4 - (S2(b)+S2(c))/2 ) / N
S2(v) = sum_{a in R(v)} g(a,v)^2 ,  N = |R(v)|
```

`g` is the unweighted shortest path on the undirected projection of the training-visible graph
(`assets/train/visible_undirected.npz`, sha256 `097a719a…`). Directed labels are never treated as
symmetric distances. 28,391,948 rows, all valid.

## 2. Frozen reference representation and calibrated scale

- `Z_ref`: final 32-d encoder output of the #265 checkpoint
  (`sha256 1f8bfecb…`, 24,587 x 32, float32), sha256
  `a2643a9bc09bbedffe4c249fea7d07d54153c9c8fd6aa5e476fe4f7b63ae4a66`.
- One global scale `S* = 0.06828367045803868`. It is the weighted least-squares optimum of
  `J(c) = sum (c*a - 1)^2`, `a = ||Z_i-Z_j||/(edge_scale*g(i,j))`, computed on the same 20,000 rows
  and reference set as #268 and reproduced here from the closed form
  `c* = sum(t1 + a_bc) / sum(t2 + a_bc^2)`, `S* = edge_scale/c*`. Relative error vs the #268
  value `2.86e-7` (`analysis/GATE.json`).
- Coordinate map `w = Z/(2*S*)`; `max||w|| = 4.360480406776883`. The `kappa=0` distance is
  `d_0(w_i,w_j) = ||Z_i - Z_j|| / S*`, the same Euclidean scale for fit and training.

## 3. Distance and legal domain

```
k = 0:  d_0^2 = 4*||x-y||^2
k < 0:  d_k^2 = (4/a) * asinh( sqrt( a*||x-y||^2 / ((1-a||x||^2)(1-a||y||^2)) ) )^2, a=-k
k > 0:  d_k^2 = (4/k) * asin ( sqrt( k*||x-y||^2 / ((1+k||x||^2)(1+k||y||^2)) ) )^2
```

Grid: `k = -0.050, -0.045, ..., 1.000` (211 values, `k=0` exact). The negative end is the largest
grid value with `|k|*max||w||^2 < 0.99`, so every reference coordinate is inside the ball; the
grid is not widened to manufacture legality. `asin` is evaluated at `1-1e-7` for the fit and
`1-1e-4` in the candidate; `sqrt` at zero uses an exact forward with a floored backward.

Boundary conditions frozen in code: coincident points give exactly 0; the coordinate projection
`project` clamps only points outside the ball of *negative* kappa and is the identity for `k >= 0`;
a zero denominator is a hard error; `kappa` carries no gradient; the table is hashed at load.

Paper / implementation comparison:

| module | paper | official code | this project |
|---|---|---|---|
| distance | Bachmann et al., ICML 2020, kappa-stereographic | `geoopt` `Stereographic`; repo `manifold.dist` | same convention, stable `asin`/`asinh` branches with an exact k=0 limit |
| curvature | continuous real kappa | continuous | fit on the frozen legal grid `[-0.05, 1.0]`, step 0.005 |
| target | GraphMoRE A.1 graph statistic | `tools/graph_infra/sectional_full.py` | unchanged table, never refit |
| scale | none (the model learns representations) | none | one frozen global `S*`, never per-triple |

## 4. Fit and identifiability procedure (frozen before running)

- Reference set A: deterministic 256 references per component, `default_rng(268)`; set B:
  `default_rng(269)`. Both on the undirected training graph.
- `J_t(k)` is the exact node decomposition over the pair population `(m,r),(b,r),(c,r), (b,c)`
  with `g>0`, equal weight per pair, denominator `n(m)+n(b)+n(c)+1`. Per-row `argmin` over the
  grid, exact ties resolve to `k=0`.
- The gate uses 2,048 deterministic target rows (`default_rng(269)`): fit on A, evaluate the same
  grid row on B. Global kappa is the single grid value with minimal mean `J` on A, evaluated on B.
- Pre-registered pass conditions: mean held-out improvement vs global >= 1 %; centre-bootstrap
  (2,000 draws, resampling centres) 2.5 % quantile > 0; Spearman of the A-fit and B-fit per-row
  kappa >= 0.5. Boundary fractions reported, not used to relax the gate.
- The full fit writes `kappa`, `J(0)`, `J(kappa)` for every one of the 28,391,948 rows with the
  manifest hashes; no refit after seeing Stage3.

## 5. Training mechanism

```
q = sum_l c_l                      (hard q, unchanged Euclidean selection/floor/Sinkhorn)
w_q = q / (2*S*)                   (frozen coordinate map)
pred_t = mean_r d_{k_t}(w_m,w_r)^2 - (mean_r d_{k_t}(w_b,w_r)^2 + mean_r d_{k_t}(w_c,w_r)^2)/2
         + d_{k_t}(w_b,w_c)^2/4
L_curvature = mean_t Huber(pred_t/target_scale, kappa_graph_t/target_scale, delta=1)
L_total = L_RQVAE + lambda * L_curvature
```

`target_scale` is the frozen RMS of the graph target table (unchanged from #265). `S*` and
`kappa_t` are constants/host lookups; no per-triple free scale; `lambda` is set by the parent's
one-batch norm-budget rule (`min(||g_original||/||g_curvature||)`, 0.2 factor absent), computed
before the first optimizer step over encoder + three codebooks.

## 6. Gradient route

`q` reaches the encoder and the selected codewords through the parent STE; the curvature term
adds gradients through `d_{k_t}` to the same parameters. `kappa_t`, `S*`, `Z_ref`, the graph and
the target table have no optimizer entry and no gradient; their hashes are checked before and
after training. Detaching the kappa lookup must not detach `q`.

## 7. Pass conditions before formal training

1. `k=0` reproduces `||x-y||/S*` exactly; +-1e-3 continuity; sphere/ball match
   `tools.graph_infra.manifold.dist` within 1e-4.
2. Hand-computed 3-node geometry, coincident points, boundary, `k=+-1`, triple-index tests.
3. Mutation tests (>=5) each caught by a named assertion.
4. Module check on a real batch with the actual loss: curvature gradient reaches encoder and
   codebooks; kappa/scale have no gradient; a parameter update changes q.
5. Pre-train probe: fixing q freezes `pred_t`; changing the frozen scale changes `pred_t` by the
   documented `1/S^2` factor; changing a row's kappa changes `pred_t`.
6. Real-entry 1-epoch smoke with the frozen single-card entry mode passes; Stage2 additionally
   requires four visible cards unless the owner's single-card authorisation is recorded.
