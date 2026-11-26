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
| C2 | Competing hypotheses and low-cost evidence | H1 point-value error loses relative order; H2 degree/scale/target definition; H3 gate scaling not q; fixed-q/gate probes, centre differences, gradients; `DIAGNOSIS.md` + `HYPOTHESES.json` | pending |
| C3 | One new relative-ranking loss | frozen formula/tau/alpha/tie rule; `L_total=L_RQVAE+lambda*(L_curv+alpha*L_rel)`; hard q + precise STE route; tests/known answers/mutations/gradients; no new Infra | pending |
| C4 | Pre-Stage3 score prediction | freeze formula/weights/ranges; compute S on the new 11th arm before Stage3 or reading new Recall; persist `SCORE_CONTRACT.md`, `SCORE_PREDICTION.json`, comment | pending |
| C5 | Formal Stage2 | 3000 epochs, four cards, seed42, final checkpoint; collision <0.40; export/receipt; §63 layer record | pending |
| C6 | Formal Stage3 | one 150-epoch frozen run, beam20, seed42; test/recall/NDCG; 11-arm monotonic/inversions/correlations; no retry | pending |
| C7 | Independent full comparison | new arm vs #265 full hard-q curvature MAE/RMSE/Pearson/Spearman, centre ordering, variance, reconstruction, quant error/rank, prefix rates/ER, SID fidelity, gate entropy, gradient budget | pending |
| C8 | Performance | report relative-loss overhead and actual inclusive wall; no skipped required checks | pending |
| C9 | Evidence/delivery | R2 four pieces with frozen GNN deps; commit/push; issue closed; public Infra/assets unchanged | pending |

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
