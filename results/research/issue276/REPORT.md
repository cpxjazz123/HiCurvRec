# Issue #276 — REPORT

Analysis-only, `INFRA_MODE=FROZEN`, no training, no Stage3 (issue text). All numbers are recomputed
from stored artifacts; nothing is imputed. Design frozen in `ANALYSIS_CONTRACT.md` (+ one dated
amendment on the encoder bias) before any fit; the fold file is bound to the table hash.

## Deliverables index

| deliverable | path |
|---|---|
| frozen acceptance list / protocol | `RUN_LOG.md` |
| root-cause diagnosis + admission six | `DIAGNOSIS.md` |
| hypotheses (inherited from #275) | `HYPOTHESES.json` |
| pre-registered contract (+ amendment 1) | `ANALYSIS_CONTRACT.md` |
| recovery (disk / R2 ledger / git history) | `recover.py`, `recover_git.py`, `recovery/RECOVERY.json`, `recovery/git_sids.json`, `recovery/AUDIT.md`/`.json` |
| per-SID table + manifest | `SID_TABLE.json`, `TABLE_MANIFEST.json`, `build_table276.py` |
| frozen folds | `FOLDS.json` (`table_sha256` bound), `folds.py` |
| corrected-forward vector series | `vector_forward.json`, `vector_forward.py` |
| SID-group analysis | `run_analysis276.py`, `ANALYSIS_276.json` |
| descriptive extras (not pre-registered) | `extras276.py`, `EXTRAS.json` |
| #274 R1/R2/R3 identity audit | `AUDIT_274.json` |
| infra start snapshot | `INFRA_SNAPSHOT_START.json` |

## C1 — recovery audit

Channels: L1 = R2-ledger sha256 of `item_sids.json`; L2 = computed L3 collision equals the stored
`metrics_corrected[arm].collision`; L3 = git era + directory name only. The recovered payloads are
**not** committed (205 MB); `recovery/git_sids.json` stores `commit:path + sha256` so every file is
re-extractable (`git show <commit>^:<path>`).

| arm | label (recall@10) | identity | in table |
|---|---|---|---|
| `TIGER_r48k2_rq` | 0.064765 / 0.063685 (2 SIDs) | L1 ledger | yes |
| `TIGER_r48k1_rq` | 0.064660 | L1 ledger | yes |
| `TIGER_r48k2u_rq` | 0.063685 | L1 ledger | yes |
| `TIGER_BEHAVIOR_R48` | 0.063110 | L1 ledger | yes |
| `behavior_residual_geometry_transport/r8_control` | 0.063995 (mean of 2 runs) | L1 ledger | yes |
| `resample` | 0.063302 / 0.062379 / … (8 SID rows) | L1 ledger + L2 | yes |
| `crowd_couser_w3` | 0.063528 | L2 collision | yes |
| `crowd_posattract_w3` | 0.064259 | L2 collision | yes |
| `resample_fn_mask` | 0.063563 | L2 collision | yes |
| `resample_l1only` | 0.062832 | L2 collision | yes |
| `resample_s43` | 0.063023 | L2 collision | yes |
| `resample_s44` | 0.062379 | L2 collision | yes |

**12 / 12 named high-recall arms recovered with content identity** (6 L1, 6 L2). Full audit:
`recovery/AUDIT.md`. High-recall records >0.06 in inventory∪cohort: 91; of them 26 have an arm that is
now content-bound; the rest are duplicate runs or arms whose SID files never existed in the repo or in
the R2 cache/git history (recorded, not guessed). One SID (`bootstrap_code_stability__resample`) has a
computed collision different from its stored record (0.04132 vs 0.03709) and is kept secondary.
`_r2_download/25` (`behaviour_code_cohesion`) has codes outside the 256-wide codebook; features are
computed from the first three codes and the row is flagged, its recall 0.0546 is below the band.

## C2 — duplicates, repeated runs, aggregation

Unit = discrete SID (ordered first three codes). Labels are the mean of protocol-consistent records
per SID; multi-run spreads are stored in `label_sources` (e.g. `behavior_residual…` 0.06376–0.06422;
`resample` 8 SID rows with distinct contents). No duplicate content crosses folds: folds are assigned
by SID group (mapping hash when known, content sha otherwise). Same-SID-different-checkpoint vector
features are joined by content sha only.

## C3–C5 — feature table (`TABLE_MANIFEST.json`)

| queue | n | definition |
|---|---|---|
| SID-only | 253 rows with features (129 with label) | frozen evaluator `tools/issue252/layer_metrics.py` on the same parquet inputs as #275 |
| vector | 74 (73 with label) | corrected forward with Linear bias: `PR(z)`, `PR(hard q)`, `quant_err`, `recon_mse`, `recon_cos` |
| geometry | 3 | `results/research/issue271/assets/geometry/*.json` (sha-keyed) |

Collision cross-check between the stored report and the computed SID: 73 rows equal, 1 differs
(the secondary row above). **Protocol audit (C4):** the stored `vector_stats_all.json` `pr` is a
**bias-less** forward (SID 48: stored 4.248629 = no-bias, corrected 6.729273); 71 of 74 rows differ.
The corrected series is used; the stored one is kept as a flagged secondary series. The #275 `recon`
was the final-epoch training batch loss — not a decoder MSE — and is used nowhere in Q_VEC.

**#274 audit:** R1/R2/R3 SIDs (`549773e3…`, `b9c3e8de…`, `67b5a47c…`) no longer exist on disk and are
not in the table; their checkpoints/sha are recorded in `AUDIT_274.json`. They cannot serve as blind
predictions and are excluded.

## C7 — frozen folds

5 folds by SID group, recall-band stratified, seed 42: bands `<0.045` 17, `[0.045,0.06]` 99, `>0.06` 13;
per-fold high-recall counts 3/3/3/2/2 (`FOLDS.json`, bound to table sha `4c927914…`).

## C8/C9 — results (SID-group OOF, 129 rows, 13 high)

### Q_SID (8 frozen SID-only features)

| model | MAE | RMSE | Spearman | Kendall | reversals |
|---|---|---|---|---|---|
| fold mean | 0.004680 | 0.006638 | −0.055 | −0.042 | 0.422 |
| fold median | 0.004424 | 0.006800 | −0.000 | +0.001 | 0.362 |
| best single (train-selected) | 0.004857 | 0.007111 | −0.149 | −0.102 | 0.551 |
| ridge (8 features, inner-CV α) | 0.006478 | 0.037502 | +0.593 | +0.437 | 0.282 |
| **ridge6 (#275 six features, inner-CV α)** | **0.003059** | **0.004869** | **+0.616** | **+0.467** | **0.266** |
| prior `l1_pos_hit` | 0.004909 | 0.007205 | −0.283 | −0.196 | 0.598 |
| PySR | 0.004067 | 0.005841 | +0.251 | +0.173 | 0.414 |

Strongest control = ridge6. **PySR − ridge6 = +0.001008, SID-group bootstrap 95% CI
[+0.000490, +0.001538]** → PySR is significantly worse. PySR expressions flip between
`1/l2_pos_hit`, `1/(l2_distinct·l2_pos_hit)`, `a/(b−l1_hard_er)` and an affine `l1_hard_er` term
across folds (5 folds, 3–5 nodes each).

### >0.06 threshold (13 positives, OOF)

| model | tp | fp | fn | precision | recall |
|---|---|---|---|---|---|
| ridge6 | **13** | **0** | **0** | 1.00 | 1.00 |
| ridge (8) | 12 | 0 | 1 | 1.00 | 0.92 |
| PySR | 4 | 0 | 9 | 1.00 | 0.31 |
| best_single / prior / mean / median | ≤2 | ≤3 | ≥11 | ≤0.40 | ≤0.15 |

### Q_VEC (same folds, + 5 corrected vector features)

n=73 (12 high). mean 0.004738/−0.137 · median 0.004607/−0.145 · best single (`l1_hard_er`) 0.003910/+0.494 ·
ridge13 0.005696/+0.699 · ridge6 0.003423/+0.552 · PySR 0.004094/+0.225.
PySR − ridge6 = +0.000671, CI [−0.000310, +0.001687] → **no PySR increment**; the 13-feature ridge
raises ρ (.552→.699) but worsens MAE (0.00342→0.00570) (calibration loss from outliers).
Threshold: ridge6 12/12, ridge13 and best_single 10/12, PySR 2/12.

### 0.065 criterion
Max label 0.064765 → **0 positives ≥ 0.065 → NOT_EVALUABLE** (as pre-registered).

### Descriptive extras (explicitly not pre-registered; `EXTRAS.json`)
* fidelity-only ridge (3 L1 features, α=10): MAE 0.003507 / ρ 0.473 → capacity features add
  (6-feature: 0.003221 / 0.512) under the same folds and fixed α.
* `PR(z)` bins: recall 0.0463 (<4), 0.0549 (4–6), **0.0595 (6–8)**, 0.0548 (8–12), 0.0529 (≥12);
  overall ρ −0.031 → window shape without a monotone relation.
* corrected `recon_mse` vs recall: ρ −0.087 (n=73), high band +0.287 (n=12) → the #275 −0.858 is not
  reproduced with the unified decoder MSE.
* largest ridge6 OOF residuals: `TIGER_cogeo_region_argmax` (y 0.0398, pred 0.0574),
  `TIGER_dul_rq` (y 0.0260, pred 0.0424) — over-prediction of low-recall arms with high fidelity.

## Answer to the issue question

* **No stable symbolic formula.** PySR loses to the simple linear control on the SID queue and adds
  nothing on the vector queue; its forms are unstable across folds.
* **A simple usable predictor exists and it is linear**: the six frozen SID-only features, ridge with
  inner SID-group CV → OOF ρ **0.616**, MAE **0.003059** (vs fold median 0.004424), and the >0.06 band
  is identified **13/13 with 0 false positives**. Vector features do not improve calibration; geometry
  remains unidentifiable (n=3).
* This holds under SID-identity grouping with the recovered high-recall band — i.e. it is not a
  mechanism-family artifact, and the recovered >0.06 assets do not overturn the #275 ordering of
  PySR vs the linear control.

## Limitations

1. 129 labeled SIDs of 253 table rows; 124 unlabeled SIDs have no accessible recall record.
2. Label range 0.0260–0.0648; no ≥0.065; the >0.06 band has 13 SIDs (band-stratified into every fold
   but small).
3. 6 of the 12 named arms bind by L2 collision (content-equal collision) rather than the ledger hash;
   one SID is flagged secondary (collision disagreement).
4. History reuse: labels come from stored Stage3 results produced by different code eras; only the
   protocol-majority label per SID is used, spread recorded (`label_sources`).
5. Geometry is descriptive (3 SIDs).
