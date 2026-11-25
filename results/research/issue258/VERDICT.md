# Issue #258 VERDICT — R05 role-conditioned level-2 mixed-curvature refinement

Status: **COMPLETED_NEGATIVE**. The preregistered Stage2 signature is falsified, the frozen Stage3 test is
below the adoption threshold, and every compute, reduction and archive step of the round is complete.
Validation Recall@10 is `NOT_AVAILABLE` for a documented Infra reason (below); it cannot change the verdict
because the Stage2 failure line already fired on a measured criterion.

## 1. Acceptance checklist (issue #258)

| id | requirement | status | evidence |
|---|---|---|---|
| C1 | preregistration intact, nothing dispatched, one registration | **PASS** | `results/research/issue256/rounds/r05/STATUS.json` (`REGISTERED_NOT_DISPATCHED`); `_runs/` contains only `issue256_r05_formal` |
| C2 | role mixture at level 2; levels 1 and 3 use the item gate; three switches wired | **PASS** | `model/mechanism.py::level_weights`; `MODULE_CHECKS.json` switches; 163 candidate tests passed |
| C3 | frozen training-only role table, no new graph/orbit/cache | **PASS** | `role_table.json` sha256 `32f986dac9bf8e382c11aedd62cd54e7defbf1853ee0b930b6fe51d4f334d3eb`; no graph asset added |
| C4 | public Infra byte-identical; only mechanism/tests/evidence changed | **PASS** | `INFRA_BASELINE.json` vs `INFRA_FINAL.json`: 220 files, changed `[]`, missing `[]` |
| C5 | dependency-aware revalidation of all affected items | **PASS** | `VALIDATION_PLAN.json` 6 reuse / 0 rerun; `ADMISSION.json`, `PERFORMANCE.json`, `MODULE_CHECKS.json` |
| C6 | 0.8 s/epoch ceiling, target recorded truthfully | **PASS** | `PERFORMANCE.json`: mean 0.693481 s, ceiling PASS; 0.4/0.5 not met |
| C7 | preregistered predictions recorded unchanged; failure line applied; unavailable metrics marked | **PASS** | `SIGNATURE.json`: distinct L2 **9609** ≥ 7000 failure line; L1 carry 0.059959 inside window; validation R@10 `NOT_AVAILABLE` |
| C8 | one formal four-card 3000-epoch Stage2 + one frozen Stage3 150 epochs, collision < 0.4 | **PASS** | `FINAL_STAGE2.json` (3000 epochs, collision 0.055273, re-export exact), `FINAL_STAGE3.json` (150 epochs, n_eval 57439) |
| C9 | analysis/replay via existing tools; missing metrics marked | **PASS** | `ANALYSIS.json`; `replay_stage3/REPLAY.json` PASS, independent reduction of 57439 predictions reproduces all four metrics |
| C10 | wall clock via existing TIMELINE, comparison with #256 | **PASS** | `TIMELINE.json`: merged 8952 s (2h29m), GPU 4582 s; target ≤2h **not met**, reported |
| C11 | reports/VERDICT/R2 four-piece set via existing interfaces; negative preserved; causal wording limited | **PASS** | `report/` set; R2 `r2:genrec/106/` (133 files, 23 recall keys, ledger entry 106) |
| C12 | before/after public hashes equal; pushed; local == remote | **PASS** | `INFRA_FINAL.json` equal; commit/push recorded in the close comment |
| C13 | complete real Stage2/Stage3 + independent reduction + archive | **PASS** | all of the above |

## 2. Locked predictions and measured result

| measurement | R04 reference | R05 measured | prediction / failure line | verdict |
|---|---|---|---|---|
| distinct two-code prefixes | 8781 | **9609** | ≤ 6000 / ≥ 7000 | **failure line met** |
| absolute L1 prefix carry | 0.062501 | **0.059959** | 0.0496–0.0696 | level 1 unchanged |
| H(L3 | L1 L2) | 1.9957 | **1.8516** | recorded only | — |
| distinct triples | 22995 | **23228** | recorded only | triple stays near-unique |
| three-code collision | 0.06475 | **0.05527** | < 0.40 gate | gate passed |
| validation Recall@10 | 0.05590 | **NOT_AVAILABLE** | ≥ 0.0583 | not measurable with an existing interface |
| test Recall@10 | 0.051516 | **0.050349** | > 0.065 adoption | below threshold |
| test Recall@5 / nDCG@10 | — | 0.033270 / 0.027328 | recorded | — |

**Conclusion**: applying the same frozen role mixture to the level-2 cost did not coarsen the level-2 branching.
It opened more distinct second codes than R04 (9609 vs 8781) while level 1 stayed exactly where the contract
expected, and the final test is below R01, R02, R04 and the 0.065 threshold. This is a valid negative mechanism
result: the SID is reproduced exactly from the checkpoint (0 differing items), the real-loss gradients reach the
role table and both gate rows, 13/13 mutations are caught, and the frozen recipe was used once.

Causal wording: R05 vs R04 isolates *the level that the same frozen role table drives*. R05 vs the TIGER
baseline changes the mechanism family as well and is **not** single-factor causal.

## 3. Independent recomputation (existing tools only)

- `tools/iteration/analysis.py` on the final record: distinct L1 256, distinct L2 9609, triples 23228,
  collision 0.055273, H(L3|L1L2) 1.851570.
- `tools/iteration/replay.py` on `test_final.json`: **PASS**; the independent reduction of 57439 persisted
  predictions reproduces `test_recall@5`, `test_recall@10`, `test_ndcg@5`, `test_ndcg@10` exactly
  (tolerance 1e-10), and all artefact hashes match.
- Legal re-export from `rqvae_best.pth`: 0 differing items, identical SID sha256.
- Stage2 artefacts: checkpoint `f93da1a7…`, SID `dc206dba…`, export `export_manifest.json`,
  binding `archive_binding.json` `28055c28…`.

## 4. Wall clock (existing TIMELINE)

| phase | wall |
|---|---|
| registration + contract | 917 s |
| mechanism implementation / TDD / red-green | 540 s |
| validation (tests, admission, re-export, module diagnostics) | 7495 s, overlapping the compute |
| Stage2 formal 3000 epochs | 2075 s |
| Stage3 frozen 150 epochs + test | 2507 s |

Merged wall **8952 s = 2h29m**, GPU wall 4582 s. The ≤2h target was **not met**; the main extra cost was the
first-time `transformers` import on the Stage3 node (~6 min) and the full validation battery. Compared with the
historical #256 average of 4h53m per round, this round used ~51% of that wall.

## 5. Limits

1. **Validation Recall@10 is not available.** The only implementation is `tools/issue256/r04/external_validation.py`,
   bound to the R04 run identity; no parameterised public entry exists and `FROZEN` forbids copying or modifying
   it. It is recorded as `NOT_AVAILABLE`, never replaced by the test metric or a fabricated value. The Stage2
   failure line fired on the measured L2-prefix count, so this gap does not change the verdict.
2. **The Stage3 test is exploratory.** The test split has been read by all earlier arms; it is reported once and
   cannot serve as an unbiased selection signal.
3. **No Infra development.** Public components are byte-identical before/after; only the candidate tree and this
   evidence directory changed.

## 6. Closing checks (skill §6.4)

1. **All production code implemented?** Yes — the candidate mechanism is real code; no stub or reference-only path.
2. **All necessary tests actually executed?** Yes — 163 candidate tests (exit 0), 13/13 mutations caught, 14
   pipeline units, admission entry + behaviour graph, four-rank real-loss module diagnostics.
3. **All required training completed?** Yes — one four-card 3000-epoch Stage2 run and one frozen 150-epoch
   Stage3 run, both with logs, checkpoint and metrics.
4. **Infra from `assets/` and unchanged?** Yes — all data assets referenced by hash in `INFRA.md`,
   `INFRA_BASELINE.json` vs `INFRA_FINAL.json` identical (220 files, 0 changes), no new Infra.
5. **Any executable task left?** No — compute, independent reduction, reports, R2 archive and remote push are
   complete; the remaining issue state is the close action itself.
