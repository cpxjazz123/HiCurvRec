# Issue #268 diagnosis

Frozen before the mechanism decision. Every command and log is under
`results/research/issue268/receipts/checks/`; every number below is reproducible
from `results/research/issue268/` and the frozen cache.

## 1. Phenomenon (observation, not root cause)

| arm | Stage3 test_R@10 | geometry-side change |
|---|---:|---|
| TIGER `three_expert_attention` (#253) | 0.04643 | mixed gate |
| #265 `TIGER_euclid_local_curve_supervision_rq` | 0.05259 | gate mixed distance, final hard q, point target |
| #267 `TIGER_relative_curvature_ranking_rq` | 0.05078 | + same-centre relative ranking term |

#267 raised full-table q Pearson from `-0.0331` to `+0.0213` and the
within-centre median Spearman from `0.0870` to `0.0912`, yet lost `0.00181`
R@10. The observation to explain is: **local geometric fidelity improves while
the discrete SID loses recommendation gain**, and the gate-based arm is not
separable from the target definition.

## 2. Competing explanations (required before any mechanism)

| id | explanation | test | status |
|---|---|---|---|
| E1 | The geometry ruler has too much freedom: a global scale change, not local curvature, explains the distortion reduction | closed-form global scale control on the independent subset (`analysis/SCALE_CONTROL.json`) | **SUPPORTED** |
| E2 | The curvature proxy carries too little information: per-triple curvature is unidentifiable | full fit, boundary/flat statistics, global-κ comparison, full-reference audit (`kappa_cache/manifest.json`, `analysis/TRANSFER_VALIDATION.json`) | **SUPPORTED** |
| E3 | The frozen reference representation itself lacks behavior geometry | residual distortion floor `0.1746` at the best global scale (≈0.42 mean absolute relative pair error), not separable from E1/E2 | **INCONCLUSIVE** |
| E4 | A gate-only change explains #265/#267's geometric gain while hard q/SID stay fixed | code path: selection is Euclidean and ignores the gate (#265), and #267 keeps that selection | **SUPPORTED for #265/#267** |

## 3. Evidence

### 3.1 Gate-only explanation (E4)

#265 routes the gate **only** into the curvature distance. `assignment_cost`
deletes the weights and returns the Euclidean cost; hard ids come from that
cost. #267 keeps the same selection. Therefore the gate can rescale the training
objective but cannot move `q` or the SID, so any gain attributed to it is an
objective-scale effect. This matches #267's own verdict.

### 3.2 Reference representation and the fit (E2)

`Z_ref` = the #265 final encoder on the frozen catalogue (sha256
`a2643a9bc09bbedffe4c249fea7d07d54153c9c8fd6aa5e476fe4f7b63ae4a66`, 24587×32,
max norm `0.595499`). Frozen scale `S = 0.1058943657241384` (median Euclidean
length of a visible undirected edge), coordinates `w = Z_ref/(2S)`, so `κ=0` is
the Euclidean distance `||Z_i - Z_j||/S`. The legal curvature interval is
`[-0.125, 1.0]` because every coordinate must satisfy `|κ|·||w||² < 1` and
`max||w|| = 2.8118`.

Full fit over all 28,391,948 legal rows (`kappa_cache/manifest.json`):

| statistic | value |
|---|---:|
| mean J at κ=0 | 0.268071 |
| mean J at fitted κ_t | 0.242248 |
| improvement median | 9.568 % |
| κ_t at the domain floor (`-0.125`) | **99.78 %** |
| κ_t mean / std | -0.124651 / 0.010070 |
| flat fraction (J unchanged) | 0.00046 % |
| median gap to the second-best grid value | 0.001004 |

The fit saturates the negative boundary: the data wants a stronger hyperbolic
distortion than the frozen coordinate box allows.

### 3.3 Independent transfer and full-reference audit

Independent reference subset (seed 269, 256 references, 20,000 rows) and the
full-reference audit (all 24,471 references, 512 rows):

| comparison | independent subset | full-reference audit |
|---|---:|---:|
| J at κ=0 | 0.280895 | 0.282668 |
| J at per-triple κ_t | 0.253693 | 0.257418 |
| J at one global κ = -0.125 | 0.254044 | 0.257418 |
| per-triple beats global | **0.18 %** of rows | **0.0 %** of rows |

The improvement over `κ=0` transfers (99.995 % of rows improve), but the
**per-triple difference over one global curvature does not exist**. The cache
verification recomputes J at κ=0 and at the stored κ_t directly from the raw
pairs: maximum absolute error `5.5e-8` / `3.9e-8` on 1,000 rows
(`analysis/CACHE_VERIFICATION.json`), so the cache is self-consistent.

### 3.4 Global-scale control (E1, decisive)

On the same 20,000 independent rows a single global Euclidean multiplier has a
closed-form quadratic objective (`analysis/SCALE_CONTROL.json`):

| setting | mean J |
|---|---:|
| κ=0 at the frozen scale `S=0.10589` | 0.278700 |
| best single global scale `S*=0.068284` | **0.174575** |
| per-triple κ at the frozen scale | 0.253693 |
| per-triple free scale (diagnostic, forbidden) | 0.163897 |

One scalar beats the whole per-triple curvature family by `0.0791` J (45 %
relative), while per-triple curvature is only `0.00035` better than one global
curvature. The curvature family's gain is the frozen global scale being wrong;
the box does not let κ compensate.

The post-hoc probe at the best global scale
(`analysis/probe_J_scale_0p068284.json`) points the same way: `J0 = 0.1674`,
per-triple `Jfit = 0.1616`, median improvement `2.27 %`, boundary fraction
`38.3 %`. Even with a well-calibrated ruler, triplet-specific curvature adds a
couple of percent, far below what a 3000-epoch arm would need to beat the
`0.0018` R@10 split between #265 and #267.

Probe consistency check: at the frozen scale the 2,048-row probe gives
`J0 = 0.2913`, `Jfit = 0.2641`, median `9.28 %`, boundary `99.95 %`
(`analysis/probe_J_edge.json`), consistent with the full-table fit. At the
max-norm scale the fit is 100 % at the spherical ceiling (`+1`), confirming that
this scale is not a meaningful ruler (`analysis/probe_J.json`).

### 3.5 Stratification

From the cache verification: the floor fraction is `100 %` for triples with
adjacent endpoints (`bc=1`), `99.69 %` for two-hop endpoints, `100 %` for degree
≤ 30 centres, `99.68 %` for degree > 30, `99.73 %` for negative graph targets and
`99.63 %` for non-negative ones. No stratum shows a different behaviour, so the
boundary solution is not an artefact of one graph region.

## 4. Root cause and mechanism target

The root cause is **not** "the gate is too free" or "the loss lacks a ranking
term". The local geometry is being asked to absorb a **global scale mismatch**
between the frozen semantic representation and the graph hop metric. Once that
single scalar is corrected, triplet-specific curvature has no measurable
content under this reference representation. A future mechanism must change the
global calibration or the reference representation; a per-triple geometry table
cannot act on this cause.

## 5. Six pre-training questions (answered; the answer is "do not train")

1. **What problem?** Give each legal triple a local continuous curvature that
   matches real graph distances.
2. **Evidence this is the failure cause?** The fitted κ sits at the coordinate
   box boundary for 99.78 % of rows and is indistinguishable from one global
   curvature (0.18 % / 0.0 % of rows).
3. **Why would the mechanism act on it?** It does not: the gain is a global
   scale, and a per-triple table cannot change a global ruler.
4. **What differs from previous failures?** It is not a gate or a ranking term,
   but the diagnostics show the difference is immaterial.
5. **Metric to judge it?** Distance distortion J and its independent-subset /
   full-reference transfer; both say "no per-triple signal".
6. **Knowledge if it fails?** The global ruler, not local curvature, is the
   binding constraint; future work must calibrate or replace the reference
   representation.

## 6. Test-access disclosure

Earlier issues' Stage3 runs read the test split. Issue #268 read no test
interaction, no Stage3 prediction and no test label: all evidence above uses the
training-visible graph, the frozen target rows and the frozen parent encoder.
The one planned test read (Stage3 `test_final.json`) was not executed because
the geometry gate stopped the mechanism before training. Any reuse must keep
that distinction.

## 7. Decision

`STOP_BEFORE_TRAINING`. The issue's own gate says "若无法证实有效几何适配，
继续低成本诊断，不默认正式训练". The diagnostics disprove per-triple adaptation
(E2) and put the dominant effect in the global ruler (E1), so formal Stage2 and
Stage3 are not launched. The cache, its verification, the transfer audit and
the negative conclusion are delivered; no unrun item is reported as a pass.
