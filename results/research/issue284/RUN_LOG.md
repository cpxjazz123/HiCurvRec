# Issue #284 — RUN_LOG

- Issue: #284 "Analysis-only: validate chain triplet L1 joint retention and its interaction with SID discrimination"
- Body sha256: `26fa8b4496d76f4e01a485e3216894522b7597925daf4213021ae2fd8f27a168` (3035 chars, frozen 2026-11-28).
- **INFRA_MODE = FROZEN** (issue: "单Agent,analysis-only,INFRA_MODE=FROZEN;复用#277/#283…不得修改公共infra或原资产").
- **Training required? NO** — "不重训TIGER/RQ-VAE/T5,不启动Stage2/Stage3、不运行新的最终test"; acceptance is recomputable from existing artifacts.
- Unit: independent SID (`sid_sha256`), near-duplicate label clusters (127 units of 129 labels); the 124-SID prospective queue; recovered unlabeled versions kept separate by export sha.
- #283 reference commit fixed: `801048ca8153fd765e38b045a4c0578cb6343a0c`.

## Frozen acceptance checklist (C1..C5)

| C | issue § | requirement | criterion |
|---|---|---|---|
| C1 | §1 | verify #283 artifacts (JOIN_SUMMARY/parts/MATCHED/MODEL_COMPARISON/validation/manifest); per-SID recompute all1_chain + L2/L3 cumulative ALL, three marginals, AM/MB/AB/NONE, exact predictions and paired CI; verify fixed 0.002699/0.7665 vs selected 0.002822/0.7156 (not conflate); labels 129/120/9/13/6 + near-duplicates + 5 folds; new Stage3 only with exact SID+run binding and same-mouth chain stats → time-postordered queue, public/unknown separated; recover #274 R2/R3 and #272 actual SIDs without mixing export versions | all reconciliations exact or documented; recovered SIDs identified by sha |
| C2 | §2 | decompose chain ALL (raw, marginal-conditional residual + feasible bounds, null lift); match same three marginals + six metrics + L2/collision/L3/L1 occupancy/degree (degree SID-invariant); frozen tolerances; coverage/effective SIDs/matching quality/dependence/SID bootstrap; high-recall/weak-binding dominance; strata; ALL vs AB; chain vs fork/merge/revchain + matched random; residual collinearity with L1 pos/neg/Hard ER/L1 concentration/L2/L3; counterexamples | every claim has support + CI; no triplet-as-independent |
| C3 | §3 | conditional refinement: P(L2 non-ALL\|L1 ALL), P(L3 all-different\|L1 ALL), refinement gain, L2/L3 partitions after L1 sharing, true three-code collision; separate three-item vs catalogue collision; recall association of chain ALL within L1-neg/L2/L3 discrimination bands (thresholds from training folds); candidates A/B/C/D frozen; 5-fold leak-free inner selection; OOF metrics/CI/strict/0.065; secondary correction | all metrics + CIs; frozen candidates executed |
| C4 | §4 | before reading truly unknown labels: freeze research candidate formula/coefficients/domain/input hash + old/new predictions for pending SIDs; no Stage3; known new labels only time-postordered sensitivity; replacement condition; power analysis; final answers | frozen file with sha256; power numbers |
| C5 | §5 | CONTRACT/HYPOTHESES/DIAGNOSIS, audit, per-SID chain + conditional table, matched pairs/counterexamples, OOF/CI, conditional response data, frozen research formula or non-applicability, power analysis, tests, persistent logs, REPORT/VERDICT, infra drift 0, commit+push | all files + zero drift |

## Scope of mechanism changes

Analysis-only; no candidate GNN/quantizer/loss changes, no training. New files under
`results/research/issue284/` only. Public infra and assets are read-only; #283 derived artifacts are
referenced by sha256.
