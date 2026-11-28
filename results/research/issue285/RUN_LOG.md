# Issue #285 — RUN_LOG

- Issue: #285 "Analysis-only: conditional SID quality law under low negative sharing and low collision"
- Body sha256: `6e308d5f14a4f1cec8b0709afcb55fa7a27bb41551da986ab85ae7fd858f7151` (2474 chars, frozen 2026-11-28).
- **INFRA_MODE = FROZEN** (issue: "INFRA_MODE=FROZEN …禁止修改公共infra/资产,重训…启动Stage2/Stage3或新test").
- **Training required? NO** — analysis-only; all acceptance quantities recomputable from the frozen 129 SIDs,
  the 253 computable SIDs, the 5 folds and the #283/#284 derived tables.
- References fixed: #284 `b106d01e61a73d903aad7e58f77e63a42afaa745`; #277 predictor; #283 `801048ca…`.
- Units: independent SID; near-duplicate label clusters (127 units); the 2 time-postordered labels are
  sensitivity only (not the 129 primary cohort).

## Frozen acceptance checklist (C1..C5)

| C | § | requirement | criterion |
|---|---|---|---|
| C1 | §1 | recompute low/mid/high N and C tertiles **from training folds**, thresholds, sample counts, CIs; check the overlap of the two low intervals and whether the four conditions are the same source; clarify missing/120-9/13-6/near-duplicates/protocol; keep the 5 folds; freeze hypotheses H37-H40 and the pre-registration | all numbers + CIs; overlap matrix; frozen thresholds in CONTRACT |
| C2 | §2 | joint grid lowN∩lowC / lowN-only / lowC-only / both-high; per-cell independent SIDs, strong/weak, high-recall counts, ranges; within-cell P/ER/chainALL/fork/merge ALL associations; interaction difference test (not one-significant-one-not); controls; ER=P/N identity; strict monotonicity/reversals and counterexamples; lowN/lowC-alone discrimination; incremental P/ALL beyond N,C; robustness (strong-only, LOO, clusters) | every cell with ≥10 SIDs or marked unidentifiable; CI-backed claims; counterexamples listed |
| C3 | §3 | pre-registered A/B/C/D with training-fold gates; single inner selection; exact #277 reproduction 0.002972/0.6666; compare #284 fixed B 0.002699/0.7665; metrics/CI/BH/strict/0.065; selection-optimism disclosure | gate rule (CI lower>0, secondaries not worse); explicit conditional range if only a stratum works |
| C4 | §4 | one clearly-labeled research candidate + predictions for truly unlabeled SIDs, frozen formula/thresholds/domain/input hash, old/new side by side; #277/#284 files untouched; power/effect and the required independent SIDs | frozen PROSPECTIVE_285.json; no Stage3 |
| C5 | §5 | CONTRACT/HYPOTHESES/DIAGNOSIS, threshold/grid data, counterexamples, per-SID features, OOF/CI, formula (if applicable), power, tests, logs, REPORT/VERDICT, infra drift 0, commit+push | all files + zero drift |
