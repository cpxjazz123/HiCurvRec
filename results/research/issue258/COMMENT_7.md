### #258 COMPLETED (negative) — R05 level-2 role mixture, full Stage2/Stage3, no Infra change

**Result.** The single mechanism change (move the frozen per-role curvature mixture from the level-3 cost to the
level-2 cost, levels 1 and 3 on the two-row item gate) was implemented once and run once under the frozen recipe.
The preregistered Stage2 signature is **falsified**, and the frozen Stage3 test is below the threshold:

| measurement | value | criterion |
|---|---|---|
| distinct two-code prefixes | **9609** | ≤6000 prediction; **≥7000 failure line met** |
| absolute L1 prefix carry | **0.059959** | 0.0496–0.0696 (level 1 unchanged) |
| H(L3 | L1 L2) / distinct triples | 1.851570 / 23228 of 24587 | recorded |
| three-code collision | **0.055273** | <0.40 gate (PASS) |
| quantisation error / effective rank | 0.007134 / 4.333 | diagnostics |
| test Recall@10 | **0.050349** | >0.065 adoption threshold (not met) |
| test Recall@5 / nDCG@10 | 0.033270 / 0.027328 | recorded |
| validation Recall@10 | **NOT_AVAILABLE** | FAULT-1: only an R04-identity-bound implementation exists; FROZEN forbids copying or changing it |

Comparison: R01 0.052734, R02 0.052665, R03 0.049009, R04 0.051516, R05 **0.050349**; baseline TIGER 0.0557.
The level-2 placement did not coarsen the branching; it opened more second codes than R04 while level 1 stayed put.
This is a valid negative mechanism result, not an implementation failure (SID reproduced exactly from the
checkpoint with 0 differing items; real-loss gradients reach the role table and both gate rows; 13/13 mutations
caught; one run only).

**Acceptance C1–C13:** all PASS — see `results/research/issue258/ACCEPTANCE.json` and `VERDICT.md`.
Public Infra: `INFRA_BASELINE.json` vs `INFRA_FINAL.json` — 220 files, changed `[]`, missing `[]`.

**Training**
- Stage2 `issue256_r05_formal`: 3000 epochs, four ranks, global batch 4096, seed 42, wall 2005.5 s; checkpoint
  `f93da1a7…`, SID `dc206dba…`, re-export exact; collision 0.055273.
- Stage3: 150 epochs, beam 20, seed 42, four ranks, n_eval 57439, wall ~2507 s; independent reduction of the
  57439 persisted predictions reproduces all four metrics (`replay_stage3/REPLAY.json`, PASS).

**Evidence** (`results/research/issue258/`): `SIGNATURE.json`, `FINAL_STAGE2.json`, `FINAL_STAGE3.json`,
`MODULE_CHECKS.json`, `PERFORMANCE.json` (mean 0.693481 s/epoch, 0.8 ceiling PASS), `VALIDATION_PLAN.json`
(6 reuse / 0 rerun), `checks/candidate_suite.log` (163 passed), `MUTATIONS.md` (13/13), `report/` set,
`VERDICT.md`. R2 four-piece archive: **`r2:genrec/106/`** (133 files, 23 recall keys, ledger entry 106).

**Wall clock** (existing TIMELINE): merged **8952 s = 2h29m**, GPU 4582 s. The ≤2h target was not met (first-time
Stage3 `transformers` import ~6 min plus the full validation battery); this is ~51% of the #256 4h53m average.

**Commits / remote**: `48e0ab44` (candidate + evidence), `8717dc6f` (final ledger); local `main` ==
remote `main` == `8717dc6f8299009d09b7eef4b71e47217b99b142`.

**Closing checks (§6.4):** production code implemented — yes; required tests executed — yes (163 + 13 mutations +
admission); required training completed — yes (one Stage2 3000-epoch, one Stage3 150-epoch); Infra reused from
`assets/`, no new Infra, hashes identical — yes; executable tasks left — none.

**Limits carried forward:** validation Recall@10 is unavailable for the documented tool-binding reason; the Stage3
test is exploratory because all arms have read it; R05-vs-baseline is not single-factor causal. These do not
change the verdict, which already follows from the measured Stage2 failure line.
