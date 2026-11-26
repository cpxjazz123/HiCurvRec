## Issue #267 candidate ready for Stage2 admission

- Old #265 arm archived to `stage2_RQ-VAE/_archive/TIGER_euclid_local_curve_supervision_rq_issue267_parent`; top level has only the new `TIGER_relative_curvature_ranking_rq`.
- New mechanism: same-centre softplus sign-ranking `L_rel` on the actual hard q, `tau=1.0516836255159168`, `alpha = 0.2 * ||grad L_point|| / ||grad L_rel||` calibrated before the first optimizer step; point term retained.
- `CONTRACT_MECHANISM.md` and `SCORE_CONTRACT.md` frozen before training.
- Candidate tests: 139 passed; new relative known-answer/edge tests pass; actual path calibration smoke gave 378 same-centre pairs, no ties/cross-centre pairs.
- Public infra start snapshot 61 files, 0 missing; no asset or framework change.
- Next: `stage2_start` four-card 3000-epoch run, then pre-Stage3 score prediction before any new Recall read.
