# Issue #278 — DIAGNOSIS (root cause candidates for the #277 over-prediction)

Round start 2026-11-28 14:3x; INFRA_MODE=FROZEN; analysis-only.

## Phenomenon

The #277 frozen Ridge6 predicted `0.0531902589` for #274 R2 (`TIGER_pairrank_enc_rq`, SID
`9f6bf673…`); the actual Stage3 label is `0.0479465172` — a **+0.0052437418 over-prediction**
(1.7× the OOF MAE 0.003059), inside the empirical interval, with the correct `< 0.06` call.
The prediction also over-predicts several low-recall/high-fidelity arms (e.g. `cogeo_region_argmax`
0.0398 → 0.0574, `dul_rq` 0.0260 → 0.0424, #277 extras).

## Candidate causes (competing)

| id | explanation | discriminating measurement | status |
|---|---|---|---|
| R1 | **Weight-blind fidelity**: L1 sharing counts *how many* relations are kept, not *which*; weighting by training transition importance should explain part of the residual | compute `P_w/N_w/ER_w`; correlate the residual with the weighted minus uniform difference; refit with 1–2 weighted features | UNTESTED |
| R2 | **Noise**: one label, residual 1.7× MAE — could be ordinary | all other prospective labels; residual distribution vs the OOF interval | UNTESTED |
| R3 | **Protocol/identity**: R2 is a re-run SID (`9f6bf673…`, not the early `7923…`); its label is a single run of a different export generation | verify the SID hash and run identity; compare with same-SID repeats where they exist | UNTESTED |
| R4 | **Missing features**: popularity/degree/SID-group size may drive the residual instead of fidelity weighting | add popularity/degree as diagnostics; check whether importance weighting survives | UNTESTED |
| R5 | **Regression shrinkage/selection**: the fit moves extreme labels toward the mean; low-recall arms with good fidelity are pulled up | residual-vs-label slope; calibration curve | UNTESTED |

R1 is the issue's hypothesis; R2/R4/R5 are explicit competitors, so the round must report whether the
weighted fidelity carries information *beyond* the six features and whether any improvement survives
the paired CI (not just the R2 case).

## Low-cost verification (no training)

All of this is computable from the existing train/valid parquet rows, the frozen SID tables and the
stored OOF predictions: transition counts/conditional probabilities from the training graph; weighted
`h+`/`h−` on the same validation rows; cross-fitted residual correlations; refits of the frozen model
family with at most two added features.
