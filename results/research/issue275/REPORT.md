# Issue #275 — REPORT

Analysis-only, `INFRA_MODE=FROZEN`, no training, no Stage3 (issue text). All numbers below are
recomputed from stored artifacts; nothing is imputed.

## Deliverables index

| deliverable | path |
|---|---|
| frozen contract + scope | `RUN_LOG.md` (C1–C10, INFRA_MODE, no-training decision, body sha256 `57677cb6…`) |
| run inventory | `RUN_INVENTORY.json`, `build_inventory.py` |
| feature contract | `FEATURE_CONTRACT.md` |
| joint table + manifest | `JOINT_TABLE.json`, `build_table.py`, `features/layer/*.json` (54), `features/layer_run.json` |
| pre-registered analysis contract | `ANALYSIS_CONTRACT.md` |
| PySR environment | `PYSR_ENV.md` |
| analysis code + results | `run_analysis.py`, `ANALYSIS_RESULTS.json`, `analysis.log` |
| hypotheses | `HYPOTHESES.json` |
| verdict | `VERDICT.md` |

## C1 — inventory

* 239 stored `test_final.json` records → 234 label units. All 239 records verified to have been
  produced by the frozen Stage3 entry (`stage3_trainer_sha256`, `stage3_completed_epochs`,
  `stage3_input_config` present in the identity-era records).
* Binding channels: **54 SIDs by the record's own identity** (`sid_sha256` + `stage2_checkpoint_sha256`),
  **8 by the R2 ledger** (`results/research/issue139/r2_ledger.json`, name → recorded SID sha + R2 id),
  **172 units remain unbound**: their arms are legacy/archived trees whose SID files no longer exist in
  the working tree (`TIGER_cogeo_region_argmax`, `curvature_RQ-VAE`, `tiger_sample_*`, the `c10x…__r8_control`
  series, …). They are counted, never guessed, and excluded from matched comparisons.
* Protocol mix (all 234 units): world_size 4 — 11, world_size 1 — 7, unknown (legacy) — 216.
  The 62 bound units are the analysis cohort; protocol is recorded per unit and the label uses the
  majority world_size of that SID (frozen rule).
* Duplicate SIDs: grouped; every Stage3 record of the same SID is retained; `noise_range` = max−min.

## C2 — features (frozen evaluators only)

Recomputed for the 54 bound SIDs whose SID file exists (`features/layer/<sid>.json`, evaluator
`tools/issue252/layer_metrics.py` sha recorded, inputs `results/stage0_build_parquet/{train,valid}.parquet`
with sha256):

| feature | source | coverage (of 62 units) |
|---|---|---|
| `l1_pos_hit`, `l1_neg_hit`, `l1_hard_er` | `validation_metrics` | 54 |
| `l2_distinct`, `l3_collision`, `l3_refine_bits` | `catalogue_metrics` | 54 |
| `quant_error`, `effective_rank` | frozen gate cache (`~/.cache/genrec-mcp/stage3__*.json`) | 26 |
| `recon` (final-epoch Stage2 MSE) | run `training_metrics.jsonl` | 23 |
| `geo_mae`, `geo_spearman`, `geo_sign_acc` | `results/research/issue271/assets/geometry/*.json` (sha-matched) | 3 |

One definition per quantity: `l3_collision` is the catalogue definition (1 − distinct triples/items);
the old `valid_l1_*` records use a different negative-sampling definition and are **not** mixed in.
`recon` is never mixed with `quant_error`.

## C3 — queues and identity audit

| queue | definition | n | families |
|---|---|---|---|
| Q_SID | label + layer features | 40 (outer-fold eligible) / 54 total | 19 (8 with n≥3) |
| Q_VEC | Q_SID ∩ quant/rank | 16 (outer-fold eligible) / 26 total | small |
| Q_GEO | Q_SID ∩ geometry | **3** — unidentifiable | 3 |

The 12 high-recall arms named in the issue body (encoder PR 5.99–6.93) are legacy: their SID files are
absent from the working tree, so their identity cannot be strictly bound; they are part of the 172-unit
gap and are **not** used to fit or select anything.

## C4 — pre-registered contract
`ANALYSIS_CONTRACT.md`: outer LOFO by family over families with n≥3 (8 folds), inner grouped selection,
outer labels never used for screening/constants/expression choice; time split as a report-only check;
controls = fold mean/median, best single feature, ridge (inner-CV alpha), and a trivial `l1_pos_hit`
prior; PySR frozen exactly as `PYSR_ENV.md`; success criterion fixed before the search.

## C5 — PySR search (outer folds)

8 folds fitted, expressions (training-fold only):

| held-out family | selected expression |
|---|---|
| af3 | `0.05489857 − 0.025077298 / l1_hard_er` |
| g4_closure | `0.054442838 − 0.0211172 / l1_hard_er` |
| graphlet | `l1_pos_hit·0.01701999·l1_hard_er + 0.04121583` |
| i234 | `0.052601658 − 0.00010457544 / l1_pos_hit` |
| j_budget | `0.05348301 − 0.0001391851 / l1_pos_hit` |
| other | `0.052216865 − 9.8146265e−05 / l1_pos_hit` |
| role | `0.0011421308·l1_hard_er + 0.042936057` |
| role_geom | `0.0012594291·l1_hard_er + 0.04167024` |

Recurring variables: `l1_hard_er`, `l1_pos_hit` — the same inputs as the trivial prior; no
capacity/curvature term survives selection. All expressions are finite on the training domain; the
`1/l1_hard_er` form has a pole at 0 (never reached; `l1_hard_er > 0` on all rows) and is reported as a
counterexample-prone shape.

## C6 — results

### Q_SID outer LOFO (n=40, 8 folds)

| model | MAE | RMSE | Spearman | Kendall | reversals |
|---|---|---|---|---|---|
| mean | 0.003997 | 0.004925 | −0.397 | −0.290 | 0.564 |
| median | 0.003670 | 0.005202 | −0.454 | −0.369 | 0.551 |
| best single (train-selected) | 0.004150 | 0.005192 | +0.198 | +0.201 | 0.399 |
| ridge (6 SID features) | **0.002497** | **0.003529** | **+0.600** | **+0.479** | **0.261** |
| prior `l1_pos_hit` | 0.003716 | 0.004760 | +0.452 | +0.412 | 0.294 |
| PySR | 0.003491 | 0.004552 | +0.110 | +0.096 | 0.452 |

Paired (family-cluster bootstrap, 2000 reps, seed 42): **PySR − ridge = +0.000994, 95% CI
[+0.000614, +0.001925]** → PySR is significantly *worse* than the strongest control.

### Q_VEC ablation (n=16, 3 folds)

| model | MAE | RMSE | Spearman | Kendall | reversals |
|---|---|---|---|---|---|
| mean | 0.002768 | 0.003434 | −0.237 | −0.187 | 0.358 |
| median | **0.002383** | 0.003113 | −0.308 | −0.253 | 0.383 |
| best single | 0.002700 | 0.003331 | +0.497 | +0.383 | 0.308 |
| ridge | 0.002860 | 0.003511 | −0.285 | −0.233 | 0.617 |
| PySR | 0.002945 | 0.003689 | −0.605 | −0.447 | 0.708 |
| prior `l1_pos_hit` | 0.003703 | 0.004637 | −0.785 | −0.600 | 0.800 |

PySR − median = **+0.000562, CI [−0.00086, +0.00103]** (covers zero) → adding quantisation error and
effective rank does **not** improve out-of-fold error. No vector increment.

### Descriptive single-indicator Spearman (all 54 Q_SID rows; descriptive, not selection)
`l1_pos_hit` +0.501 · `l3_collision` +0.483 · `l2_distinct` −0.391 · `l1_hard_er` +0.380 ·
`l1_neg_hit` +0.353 · `l3_refine_bits` +0.322 · `effective_rank` +0.577 (n=26) · `quant_error` +0.306 (n=26) ·
**`recon` −0.858 (n=23)**.

The last row is the strongest association in the whole table: lower final reconstruction error goes with
higher recall. It was **not** in the pre-registered core feature set, so it is reported as a descriptive
finding and a next-round hypothesis, never used to fit or select the formula.

### Secondary criteria
* 0.06 threshold: **0 positives** among 62 labels (range 0.0382–0.0573) → `NOT_EVALUABLE`.
* Time split: labels exist in a single month (2026-11; 54 rows) → temporal validation **unavailable**
  (report-only; not used to claim generalisation).

## C7 — ablations / hypotheses
* H1 (L1 fidelity strongest single predictor): supported among SID features (`l1_pos_hit` +0.501).
* H2 (capacity adds beyond fidelity): inconclusive — the ridge uses capacity and reaches ρ 0.60, but no
  fold-level ablation isolates its increment.
* H3 (vector adds beyond SID): **refuted** (CI covers zero).
* H4 (geometry adds beyond SID): **untested** (3 sha-matched SIDs).
* Null N (nothing beats the fold mean/median): refuted — ridge beats it by 0.0015 MAE with ρ 0.60.

## C8 — prospective prediction
**Not applicable**: the success criterion for a usable formula was not met (PySR worse than the control,
CI excluding zero), so per the pre-registered contract no formula was frozen and no prospective
prediction table was produced. Predicting with a rejected model would break the pre-registration.
The Stage2-complete arms without Stage3 (the remaining queue) are listed in
`results/research/stage3_queue/queue_dispatch.json` and remain future validation only.

## Limitations (explicit)
1. Strictly bound cohort is small (62 units; 40 outer-fold eligible) and its label range is narrow
   (0.038–0.057); no ≥0.06 positive exists, so the issue's secondary threshold metric cannot be scored.
2. 172 legacy records cannot be bound to a SID in the working tree (identity gap, counted).
3. Geometry features cover 3 SIDs only — the curvature-increment question is unidentifiable here.
4. Protocol mix (single-card vs 4-card and legacy unknown) is recorded but not fully separable with
   this sample; the label uses the majority protocol per SID.
5. Family singletons (11 of 19) cannot serve as outer folds; LOFO covers 8 families.
