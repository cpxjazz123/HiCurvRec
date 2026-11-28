# Issue #279 — VERDICT

**Decision: negative at the prediction level, positive at the relation level.** Higher-order topology
discriminates unseen validation transitions (AUC 0.62–0.63; degree-matched 0.66), but no SID-level
model passes the pre-registered criterion; #277 remains the frozen predictor.

## C1–C13

| C | evidence | verdict |
|---|---|---|
| C1 | 129/120/9/13/6 verified; 92.41 % unseen (2 698 387 edges, 323 633 distinct) | PASS |
| C2 | H1–H4 frozen in `DIAGNOSIS.md` before results; metrics/CI/budget registered | PASS |
| C3 | relation unit and legal negative identical to the evaluator; training graph only; distances not used as directional | PASS |
| C4 | 2-hop, common succ/pred, raw + Jaccard + degree-normalised; curvature from the real asset with NA; 3-node classes only, 4-node documented as coverage-limited | PASS |
| C5 | all 57 439 rows scored; fixed negatives seed 42; coverage/sparsity/runtime recorded | PASS |
| C6 | AUC/PR-AUC + anchor-clustered CIs + degree-matched AUC + seen/unseen split | PASS |
| C7 | per-SID L1/L1+L2 tables for 253 SIDs; frozen stratum threshold 4.5326; high/low 9 347/9 291 | PASS |
| C8 | candidates C1/C2/C3 (≤2 features) frozen before the Recall association; R2 not used for tuning | PASS |
| C9 | A/C1/C2/C3/E278 same queue/folds/leak-free scaler | PASS |
| C10 | all CIs cover 0 and every candidate adds one FP → fail; strict subset 6/6 tp with 1 fp; 0.065 NOT_EVALUABLE | PASS (negative) |
| C11 | curvature unidentifiable (coverage 12.9 %/2.65 %, 100 pairs); no claim | PASS |
| C12 | no V2, no new prospective predictions; #277 preserved | PASS |
| C13 | deliverables + infra drift | PASS |

## Hypotheses

* **H16 SUPPORTED**: higher-order topology carries real association strength on unseen transitions
  (`two_hop` unseen AUC 0.624, `common_pred` 0.631, both with CIs far from 0.5).
* **H18 REFUTED**: the signal is not a degree/popularity proxy — degree-matched AUC is 0.66, higher
  than the raw 0.64.
* **H17 SUPPORTED**: the SID retains the higher-order relations only weakly; the weighted fidelity
  does not improve Recall prediction (all CIs cover 0, +1 FP).
* **H19 INCONCLUSIVE**: curvature coverage is too low (100 pairs) to test independent value.

## Closure checklist (§6.4)

1. Production code: `rellevel279.py`, `sidfeat279.py`, `model279.py`, `curv279.py` executed. **YES**
2. Verification: AUC/CI/confusion recomputed from the stored predictions and scores; inputs hashed. **YES**
3. Training: none (issue forbids Stage2/Stage3). **YES (N/A)**
4. Infra: FROZEN; snapshot comparison added/removed/changed = 0. **YES**
5. Executable work left in scope: none — negative result, no V2, #277 preserved. **NO**
