# Issue #272 contract — active global constraint under the Stage2-only rules

Frozen before the round-1 3000-epoch run. Issue #272 body sha256
`8814eae7356ac36ba26c0812b51a1d4327b499171327eaa31be3b70bbeca84f7`.

## 1. Root cause, causal path, falsifiable predictions

**Root cause under test (R1).** #271's global bound was TIGER + eps (`0.781941`), far above the #269
common-ruler J (`0.338336`); the candidate ended at `0.351359` with `mu = 0` for all 3000 epochs. The
constraint was never active, so the run could not test whether a global-distance constraint restores
SID capacity/local ranking.

**Causal path.** `J_max` is set to the #269 level minus the frozen pair-estimator offset; when the batch
estimate exceeds it, the non-negative multiplier rises and the term `mu (J - J_max)` sends gradients
through the same hard `q` to the encoder and the three codebooks. The local term is unchanged; lambda is
the inherited #269 calibration in round 1. The single primary change is the bound (0.7819 -> 0.3200);
nothing else changes.

| id | prediction | falsifier |
|---|---|---|
| P1 | full common-ruler `J <= 0.338336` at the final checkpoint | `J > 0.338336` |
| P2 | independent local A.1 `MAE <= 1.893506`, global Spearman `> 0.015851`, sign `> 0.432453`; not a constant/shrink fit (`var_ratio >= 0.5`, `slope >= 0.1`) | any of these fails |
| P3 | exported SID `L2 >= 10,918`, effective rank `>= 8`, prefix discrimination and behaviour neighbourhood `>= TIGER` | any of these fails |
| P4 | same-criterion reconstruction `<= 1.05 x 8.18e-5` (recomputed #265 final `recon`) | reconstruction above |
| P5 | `mu > 0` for a material fraction of training and the constraint is genuinely violated at some point | `mu = 0` throughout; the run then cannot claim a global-gradient effect |

Collision is recorded only; it is not a #272 gate. Stage3 is forbidden by the issue; no recall is produced.

## 2. Frozen thresholds and definitions

| quantity | value | source/rationale |
|---|---:|---|
| common ruler | `d = ||q_i-q_j|| / (S* g_ij)`, `S*=0.06828367045803868` | the Euclidean member of the frozen #269 map; factor 2 verified against kappa->0 |
| #269 full J (reference, recomputed) | `0.338335882630658` | `results/research/issue272/ruler_inventory.json`; #271 0.351361, #265 0.189476, TIGER 0.739635 |
| training bound `J_MAX` | `0.320000000000000` | #269 reference minus the frozen estimator offset 0.018336 (the #271 pair estimate 0.3354 vs full 0.3514) |
| acceptance full J | `<= 0.338336` | issue #272 C2.1 |
| 1-hop / 2-hop tolerance | `<= 0.798295 / 0.278106` (1.05 x #269 0.760281 / 0.264863) | pre-frozen relative tolerance; catches near-neighbour degradation hidden by the mean |
| local ruler | #270 frozen evaluator, SCALE 0.08433422446250916, full 28.4M triplets, independent of the training kappa ruler | issue C2.2 |
| local MAE / Spearman / sign | `<= 1.893506` / `> 0.015851` / `> 0.432453` | frozen from the #271 MAE and the TIGER rank/sign (strictly better than TIGER) |
| scale-degeneration exclusion | `var(pred)/var(target) >= 0.5` and `slope >= 0.1` | prevents a constant/shrunk prediction from lowering MAE; #271 was 0.335/0.041 |
| SID L2 / rank | `>= 10,918` / `>= 8` | issue C2.3 (#265 L2, target rank) |
| prefix / behaviour | not below TIGER on the same metric | issue C2.3 |
| reconstruction | `<= 1.05 x 8.18e-5` | same-criterion recomputed #265 final training `recon`; the issue's quoted 1.23e-4 is a different metric and is not used |
| collision | recorded only | issue C2.3 |
| Stage3 | not run | issue §4 |

## 3. Mechanism and constants

`min L_RQVAE + lambda L_curv + mu (J - J_MAX)`, `mu >= 0`; `lambda = 5.883564881514758e-06` (the
inherited #269 gradient-norm calibration); `mu` starts at 0, updates every 50 optimizer steps with
`mu <- max(0, min(10, mu + 0.1 (J_est - J_MAX)))`; the estimate is the mean of the last 50 step
estimates on the fixed centre-uniform pair table (2,048 centres, seed 271). The local term is the
inherited #269 kappa_t A.1 Huber loss; kappa_t and S* are frozen host inputs; the architecture, selection,
floor, Sinkhorn, reconstruction and export are unchanged.

Round 1 changes **only** the bound. A later round may change lambda or the dual schedule, one primary
change per round, after the first round's failing criterion is identified.

## 4. Data, shapes, boundaries, tolerances

- `q`: `[24587, 32]` float32; pair `i,j` int64; `g` float64; self/unreachable pairs excluded.
- `J` training scalar float32 on sampled pairs; acceptance float64 on all 299,402,685 valid pairs.
- Coincident points: finite zero subgradient (tested in #271); the fixed estimator weights are frozen.
- Identities: SID/checkpoint/sha256 in `ruler_inventory.json`, `sid_inventory.json`, and each run's
  `provenance`.

## 5. Paper / official-implementation comparison

| module | paper | official | choice |
|---|---|---|---|
| RQ-VAE | Rajput et al., TIGER, §3 | RecBole TIGER | frozen baseline |
| kappa-stereographic | Bachmann et al., ICML 2020 §2 | #269 `stereographic.py` | inherited, factor 2 verified |
| dual inequality | Boyd et al. 2011 §5 | none in repo | fixed-step dual ascent, feasibility by full J |

## 6. Pass rule

A candidate is PASS only if P1-P5 all hold with the frozen thresholds, the legal non-collision admission
passes, and the full SID/geometry artifacts are persisted. A failed criterion is a result, not a reason
to relax a threshold. The issue forbids Stage3; none is run.

## 7. Round-2 addendum (frozen before the round-2 run)

**Round-1 root cause (measured).** At the round-1 calibration point the local A.1 gradient norm was
`6.827e-5` while the active constraint gradient at `mu=0.1` was `0.1485` (ratio 4.6e-4); the inherited
#269 lambda matched the original-loss gradient, not the global one. The local term was effectively
absent once the bound was active, so its MAE/rank/sign degraded.

**Round-2 primary change (one).** Keep the mechanism, the bound (0.320) and every other constant; after
the inherited calibration, measure the real local and global gradients on the calibration batch (local
sample + fixed pair sample, hard q) and re-scale lambda so the local gradient is
`LOCAL_GLOBAL_FRACTION = 0.1` of the global gradient at `mu_probe = 0.1`:

`lambda = lambda0 * fraction * ||grad global|| / ||grad local||`

`lambda0` is the inherited 5.8836e-6; the balance is frozen once and logged. No sweep.

**Expected/falsifiable.** P2 (local MAE/Spearman/sign) improves relative to round 1 (1.894/ -0.0055 /
0.3926 at best from #271; round 1 was 2.344/ -0.0988/0.3481); P1 and P3/P4 must remain within the frozen
thresholds. If P2 improves but P1/P3/P4 fail, the local/global/SID trade-off is confirmed at a second
point; if P2 does not improve, the fixed evaluation ruler (not the gradient budget) is the cause.
