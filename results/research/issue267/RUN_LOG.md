# Issue #267 run log

- Issue: #267 — Relative sectional-curvature supervision on frozen-GNN Euclidean RQ-VAE and prospective Stage2 score validation
- Issue body sha256: `fe40b2ba18899a41b2ef62a121a315923cb60f4cced22f2b233d255b5fb4301e`
- Frozen at: 2026-11-26
- `INFRA_MODE=FROZEN` — issue explicitly says `INFRA_MODE=FROZEN`, reuse #259/#264 public Infra, no new public Infra.
- Training required: **yes**. The acceptance requires one Stage2 3000-epoch four-card run and one frozen Stage3 150-epoch run; Stage2/Stage3 metrics cannot be produced from existing weights.

## Acceptance checklist (issue §1–§4)

| ID | Requirement | Criterion / output | Status |
|---|---|---|---|
| C1 | Root diagnosis on 109 hard q and z | full 28,391,948-triple fixed-distance recomputation; MAE/RMSE/Pearson/Spearman; target/prediction moments; sign/confusion; centre/degree/role strata; q and z separately | pending |
| C2 | Competing hypotheses and low-cost evidence | H1 point-value error loses relative order; H2 degree/scale/target definition; H3 gate scaling not q; fixed-q/gate probes, centre differences, gradients; `DIAGNOSIS.md` + `HYPOTHESES.json` | PASS |
| C3 | One new relative-ranking loss | frozen formula/tau/alpha/tie rule; `L_total=L_RQVAE+lambda*(L_curv+alpha*L_rel)`; hard q + precise STE route; tests/known answers/mutations/gradients; no new Infra | PASS |
| C4 | Pre-Stage3 score prediction | freeze formula/weights/ranges; compute S on the new 11th arm before Stage3 or reading new Recall; persist `SCORE_CONTRACT.md`, `SCORE_PREDICTION.json`, comment | PASS |
| C5 | Formal Stage2 | 3000 epochs, four cards, seed42, final checkpoint; collision <0.40; export/receipt; §63 layer record | PASS |
| C6 | Formal Stage3 | one 150-epoch frozen run, beam20, seed42; test/recall/NDCG; 11-arm monotonic/inversions/correlations; no retry | PASS |
| C7 | Independent full comparison | new arm vs #265 full hard-q curvature MAE/RMSE/Pearson/Spearman, centre ordering, variance, reconstruction, quant error/rank, prefix rates/ER, SID fidelity, gate entropy, gradient budget | PASS |
| C8 | Performance | report relative-loss overhead and actual inclusive wall; no skipped required checks | PASS |
| C9 | Evidence/delivery | R2 four pieces with frozen GNN deps; commit/push; issue closed; public Infra/assets unchanged | PASS |

## Scope

Candidate-only changes under the new Stage2 arm, issue-local diagnostics/tests and the regular
outputs. Public `tools/`, `stage2_RQ-VAE/baseline/`, Stage3 entry, assets and Skill are read-only.
No GNN/TIGER retraining, no extra baseline runs, no seed search, no test-guided weight selection.

## Notes

- Parent is #265 R2 109 (`TIGER_euclid_local_curve_supervision_rq`, test R@10
  `0.052594926791900976`); the only main mechanism change is the relative ranking term.
- #262 and #266 are historical evidence; their test results are not used to choose the new mechanism.

## Evidence updates

- Root diagnosis: PASS for C1/C2 (`DIAGNOSIS.md`, `HYPOTHESES.json`, full q/z and centre/gate JSONs).
- Candidate implementation/tests: `TIGER_relative_curvature_ranking_rq`; `pytest -q tests` = **139 passed** before the DDP-only broadcast fix; smoke after the fix = PASS, collision `0.051694`.
- Relative calibration at formal Stage2 init: alpha `0.3075788718799832`, tau `1.0516836255159168`, 378 valid pairs, 0 tied, 0 cross-centre; point/relative gradient norms `7.125525... / 4.633299...`.
- Stage2 run: `stage2_88816eeca29442c7b3284f4d4a68f610`; live until 3000. Earlier startup failure `stage2_caa891...` was the broadcast-list bug and is preserved as a failed dispatch, not reused.

## Completed evidence

- C1/C2 root diagnosis: PASS; full 28,391,948 q/z recomputation and stratum/centre/gate evidence.
- C3 mechanism/tests: 139 candidate tests passed; relative known-answer/edge tests and mutation checks pass.
- C5 Stage2: PASS, final collision `0.04534916825964941`, wall `811.659 s`.
- C4 pre-Stage3 score: PASS before reading new Recall; S `1.2921335084032926`, predicted above the ten-arm range.
- C7 initial full comparison: q Spearman `-0.03373` vs parent `-0.09342`; MAE `2.20395` vs `2.24595`; within-centre q median `0.09119` vs `0.08695`.
- C6 Stage3: PASS, `test_recall@10 0.05078431031180905`, `@5 0.03300893121398353`,
  `ndcg@10 0.027244785026023777`, `n_eval 57439`; parent 109 `0.052594926791900976`,
  delta `-0.0018106164800919256` (`-3.4426%`).
- 11-arm check: recall-ascending rank 6/11; Pearson `0.513532`, Spearman `0.863636`,
  Kendall `50/5` tau `0.818182` vs historical `0.928526` / `45/0`; frozen S again
  out-of-range (predicted highest, actual 6th) with 5 inversions.
- C8: Stage2 inclusive wall `811.659 s` vs parent `813.357 s`; one extra softplus +
  backward per step, alpha calibration once before step 1; no skipped checks.
- C9: R2 `r2:genrec/110/` (115 files, recall keys 23); infra end diff PASS
  (`INFRA_DIFF.json`: 0 changed, 0 missing).
