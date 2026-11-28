# Issue #282 — VERDICT

**Decision: negative.** Type-conditioned L1 retention (chain/merge/fork/revchain, closed on seen edges)
adds no reproducible predictive increment; the descriptive type differences are tiny and, if anything,
point the opposite way. #277 remains the frozen predictor.

## C1–C11

| C | evidence | verdict |
|---|---|---|
| C1 | 129/120/9/13/6 verified; 92.41 % unseen; baseline A reproduces 0.002972/0.667 | PASS |
| C2 | pair-level types with multi-label counts; closed only on observed edges; 4-node/curvature excluded with reasons | PASS |
| C3 | supports/overlaps/coverage; matched-negative level distribution reported (level 2 1.35 %) | PASS |
| C4 | main types and contrasts pre-registered; exploratory items excluded | PASS |
| C5 | per-SID type table (253 SIDs); P/N/ER/lift; L1 primary; L1+L2 secondary possible but not needed for the verdict | PASS |
| C6 | similar-overall contrast: 477 pairs, Spearman ΔF1/ΔF2 vs Δrecall −0.198/−0.154 (weak, non-independent) | PASS |
| C7 | baseline A = #277 corrected; D1/D2/D3 with same folds/scaler; no #281 baseline mixing | PASS |
| C8 | all CIs cover 0; ρ degrades in D1/D2; strict subset consistent; 0.065 NOT_EVALUABLE | PASS (negative) |
| C9 | residual associations +0.076/+0.056; no FDR-significant contrast | PASS |
| C10 | no V2, no new prospective predictions | PASS |
| C11 | deliverables + infra zero drift | PASS |

## Hypotheses

* **H28 INCONCLUSIVE**: the type-conditioned ERs differ by ~4 % on average; the similar-overall contrast
  is weak and negative (ρ −0.20/−0.15) with non-independent pairs and without CIs.
* **H29 SUPPORTED**: no type-specific information beyond the overall rate survives.
* **H30 REFUTED**: D1/D2/D3 CIs all cover 0 (best D2 +6.8e-6); ρ degrades.
* **H31 SUPPORTED**: the type-matched negative coverage (1.35 %) and the near-collinearity of the type
  rates with the overall rates prevent a clean type-specific estimate.

## Closure checklist (§6.4)

1. Production code: `types282.py`, `model282.py` executed. **YES**
2. Verification: hand-computed type definitions (3 tests pass); metrics recomputed from the stored OOF.
   **YES**
3. Training: none (issue forbids Stage2/Stage3). **YES (N/A)**
4. Infra: FROZEN; snapshot comparison added/removed/changed = 0. **YES**
5. Executable work left in scope: none — negative result, #277 preserved. **NO**
