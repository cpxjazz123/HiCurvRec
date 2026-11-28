# Issue #280 — VERDICT

**Decision: negative.** Pure discrete SIDs do not decode unseen directed behavioural transitions; the
apparent supervised signal is target-popularity leakage. No SID-level model passes; #277 remains frozen.

## C1–C11

| C | evidence | verdict |
|---|---|---|
| C1 | 57 439 rows, 53 081 unseen (92.41 %), graph rebuilt from train only, negatives fixed seed 42 | PASS |
| C2 | decoder tables from training edges only; source-anchored 80/20 internal split; identical candidate set for all SIDs; reverse-edge diagnostic | PASS |
| C3 | SID codes only (no z/q, item ids, text, adjacency); Recall hash-bound; duplicate exports unified | PASS |
| C4 | count decoder + smoothing/backoff; baselines (chance, popularity, prefix); supervised diagnostics with a target-only leakage control; permutation/direction tests | PASS |
| C5 | higher-order association: decoder at chance vs graph 2-hop AUC 0.62 (#279) | PASS |
| C6 | relation AUC/degree-matched margin/direction accuracy; MRR/Hits recorded NOT_RUN with the at-chance reason | PASS |
| C7 | two frozen SID-level features (`dec_auc_unseen`, `dec_margin_degmatched`) | PASS |
| C8 | A/D1/D2/D3/G279 same folds; collapse + residual analysis | PASS |
| C9 | all CIs cover 0, +1 FP, ρ degrades → fail; strict subset consistent; 0.065 NOT_EVALUABLE | PASS (negative) |
| C10 | no V2, no new prospective predictions | PASS |
| C11 | deliverables + infra drift | PASS |

## Hypotheses

* **H20 REFUTED**: the pure SID directional decoder is at chance (calibration 0.499; validation mean
  AUC 0.493; direction accuracy 0.487).
* **H21 SUPPORTED (as the explanation)**: the only high-AUC signal comes from target-code identity —
  the target-only control reaches AUC 0.93–0.99, so the supervised "decoder" measures popularity.
* **H22 REFUTED**: the SID codes do not reproduce the graph higher-order scores (decoder at chance;
  the graph scores themselves discriminate at 0.62–0.63).
* **H23 REFUTED**: D1–D3 lower MAE by 5e-5…1.6e-4 but all CIs cover 0, each adds one FP, and Spearman
  falls (0.636/0.644/0.613 vs 0.667).

## Closure checklist (§6.4)

1. Production code: `decoder280.py`, `model280.py`, `logistic280.py` executed as written. **YES**
2. Verification: decoder tables hand-checked, permutation/direction/zero-support tests (4/4 pass),
   metrics recomputed from stored predictions. **YES**
3. Training: none (issue forbids Stage2/Stage3). **YES (N/A)**
4. Infra: FROZEN; snapshot comparison added/removed/changed = 0. **YES**
5. Executable work left in scope: none — negative result, #277 preserved. **NO**
