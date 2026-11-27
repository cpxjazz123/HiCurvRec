# Issue #271 contract — globally constrained local sectional curvature

Status: frozen before the official Stage2 run. Issue #271, body sha256
`198ac63423e330b159ab6b0cee3ab57e283ea5714ab48286a8b8b2b5bbe09823`.

## 1. Mechanism

The candidate keeps the frozen TIGER RQ-VAE (768→32 encoder, three 256-code levels, Euclidean
selection, residual/floor/last-level Sinkhorn, collision suffix) and the #269 local objective, and
adds one inequality constraint on the **global** behavioural distance of the same final hard `q`:

```
q_i = sum_l c_{i,l}                                  (hard code sum, STE through selection)
L_local = mean over sampled κ-triples of Huber( A1(q)/S* - target/S* )
J(q)    = sum_{pairs} w_p ( ||q_i - q_j|| / (S* g_ij) - 1 )^2 / sum_pairs w_p
min_θ   L_original + λ L_local + μ (J - J_max),      μ >= 0
μ       <- max(0, μ + η (Ĵ - J_max))                  every K optimizer steps
```

`S* = 0.06828367045803868` is the scale of the frozen #269 κ cache. The Euclidean common ruler is
the κ→0 limit of the #269 map `w = q/(2S*)`: the κ-stereographic distance carries a factor 2, so
`d_κ → 2||w_i - w_j|| = ||q_i - q_j||/S*`. The issue's literal `||q_i - q_j||/(S* g)` is therefore
correct; multiplying by an extra 2 is a double scaling and is rejected by the unit test
`test_kappa_zero_limit`. The local and global terms use the same `q` and the same scale.

## 2. Frozen constants and thresholds

| constant | value | source |
|---|---|---|
| `S*` | `0.06828367045803868` | `kappa_cache_s0p068284/manifest.json` |
| `J_TIGER` | `0.739634552674212` | full-graph evaluation on the frozen TIGER checkpoint (`geometry/threshold.json`) |
| `ε` | `0.04230598283915425` | `max(1e-4, 2·0.021152991419577, 0.025190340031445) = 2·SE` |
| `J_max` | `0.7819405355133663` | `0.739634552674212 + 0.04230598283915425` |
| `λ` | #269 gradient-norm calibration on the first real batch | inherited `calibrate_curvature_weight` |
| `μ` init / step / every / cap | `0` / `0.1` / `50` / `10` | fixed dual ascent; no tuning |
| pair table | 2,048 centres × one partner per distance layer; seed 271 | fixed at init; stored as a result artifact |
| training pairs per step | 512 | gradient estimate; epoch J = mean of step estimates |
| `J_max` validation | exact full-graph J from the exported hard codes at the end | acceptance, not a training estimate |

ε is computed **before** the candidate is trained, from the frozen TIGER checkpoint and the fixed pair
table: bootstrap standard error of the weighted pair estimator, and the absolute sample-vs-full
difference on TIGER (whichever is larger). No post-hoc widening is allowed.

## 3. Local term (inherited #269, frozen)

- `L_local` is the #269 A.1 origin-value error: for each sampled target row, the frozen per-row κ_t
  and scale S* map the hard `q` to `w=q/(2S*)`, project into the κ ball, take the stereographic
  distances to the frozen reference set, and compare `mean(m)-mean(b)/2-mean(c)/2+pair/4` with the
  frozen target; Huber δ=1 after dividing both sides by `target_scale`.
- `κ_t`, S*, the reference set, the target table and `target_scale` are host inputs with no gradient.
- The weight λ is the #269 gradient-norm budget (`norm_budget_weight`); it is calibrated on the real
  first batch from the original and local gradients of the encoder/codebook groups and frozen.
- No change to selection, codebook shape, residual/floor/Sinkhorn, reconstruction or export.

## 4. Global term

- Pairs are ordered `(i,j)`, `g_ij` is the exact shortest-path distance on the undirected visible
  training graph; self pairs (`g=0`) and unreachable pairs (`g=inf`) are excluded. The graph is
  undirected; the directed triplet table is not used for this constraint, and this distinction is
  recorded because the local term uses directed triplets while J uses symmetric distances.
- Full evaluation: all `i<j` valid pairs, chunked BFS + chunked `q` distances; no approximation is
  used for the acceptance number. Strata by `g`: 1, 2, 3, 4–5, ≥6, plus the mean and tail share.
- Training estimate: a fixed centre-uniform stratified pair table. For every sampled centre and every
  exact distance layer, one partner is drawn uniformly; the layer size is the weight, so
  `sum w·err / sum w` is an unbiased estimate of the ordered-pair mean, and the graph is undirected so
  it equals the unordered-pair mean. Coverage and weights are saved in `global_pairs.json`.
- Per-step J uses 512 sampled pairs; the dual update uses the mean of the last 50 per-step estimates.
- The constraint is an inequality: when `Ĵ ≤ J_max`, μ decreases toward 0; when `Ĵ > J_max`, μ rises.
  A decreasing penalty alone is not accepted as feasibility; the pass criterion is the full J.

## 5. Shapes, dtypes, boundary conditions, tolerances

| object | shape/dtype | boundary test |
|---|---|---|
| `q` | `[N, 32]` float32 | coincident points: zero subgradient, no NaN |
| pair `i,j` | int64 global item ids | self pairs excluded at table build and evaluation |
| `g` | float64, 1..diameter | unreachable excluded, no division by zero |
| `J` | scalar float32 (training), float64 (evaluation) | all pairs equal to reference: J=0 ±1e-12 |
| μ | scalar float32 ≥ 0, cap 10 | satisfied constraint lowers μ, violated raises it |
| full J | float64 | tolerance vs independent NumPy reference 1e-9 relative on known cases |

## 6. Paper / official-implementation comparison

| module | paper position | official implementation | our choice |
|---|---|---|---|
| TIGER RQ-VAE | Rajput et al., TIGER, §3 (semantic IDs) | RecBole TIGER RQ-VAE | frozen baseline, unchanged |
| κ-stereographic distance | Bachmann et al., ICML 2020, §2 | #269 `model/stereographic.py` | frozen; factor 2 verified against κ=0 |
| per-triple κ_t | #269 local objective | #269 `sampled_curvature` | inherited unchanged |
| inequality constraint | standard primal-dual / augmented Lagrangian (Boyd et al., 2011, §5) | none in the repo | fixed-step dual ascent, μ≥0, feasibility by full J |
| global graph distance | issue §1 | `assets/train/visible_undirected.npz` | exact BFS, no learned distance |

## 7. Pre-registered predictions

| id | prediction | falsifier |
|---|---|---|
| P1 | full-graph `J ≤ J_max` at the final checkpoint | `J > J_max` |
| P2 | independent common-ruler local A.1 error < TIGER, with Spearman/sign/strata reported | local error ≥ TIGER |
| P3 | the exported SID avoids #269-style prefix compression (L2 groups and split/merge reported) | L2 groups collapse while J passes |
| P4 | Stage3 test R@10 improves over TIGER/canonical or the loss is explained by SID/protocol | R@10 below canonical with no SID/protocol explanation |

P4 is exploratory: the joint candidate cannot attribute an effect to local curvature alone; a matched
global-constraint–only intervention is required for that and is not run here.

## 8. Failure policy

- A failed P1/P2/P3 is a scientific result: recorded in `VERDICT.md`, never fixed by relaxing
  `J_max`, ε, the collision gate or the Stage3 protocol.
- A Stage3 hard-gate failure (collision ≥ 0.40) is repaired within the legal mechanism only; the
  unrun Stage3 is reported as not run, never as PASS.
- No TIGER retrain, no extra seeds, no second candidate, no ablation arms.
