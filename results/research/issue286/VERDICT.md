# Issue #286 — VERDICT

**Decision: the gated two-regime predictor does not improve on the six-metric Ridge6; every arm and
ablation is equal or worse, and the main candidate's paired CI lower bound is < 0. The gate is an
excellent region classifier but that does not translate into regression accuracy. #277 is retained;
candidate E is frozen only as an explicitly exploratory negative result. No Stage3; infra drift 0.**

## C1–C5

| C | evidence | verdict |
|---|---|---|
| C1 | 129/120/9, 13/6, 127 clusters, folds; label range [0.02601,0.064764], 0>0.065; baseline exact 0.002971939/0.666609, 13/0/0; #285 LL 34/11 high/4 high-strong, P 0.951, reversals 43/560; #277 domain/residual interval/label aggregation recorded | PASS |
| C2 | primary region gate (label-free) AUC 0.9808/Brier 0.0662, per-fold b (b2 3.47–3.80 > b1 0.52–0.81); diagnostic label gate AUC 0.8846/Brier 0.0750; λ [0.5,0.25,0.25,0.25,0.25]; support 25–30 ≥ 20; cross-fitted residuals; comparisons B/C/D and four ablations executed | PASS |
| C3 | A 0.0029719/0.6666; B 0.0040165 (CI <0), C_hard 0.0030846, D_soft 0.0031631, E_main 0.0030271 all with CI lower <0; strict subset worse; cells/bias/counterexamples recorded; 13/0/0; 0.065 NOT_EVALUABLE; gate calibration separated from regression; power (negative mean, MDE 8.3e−5) | PASS (negative) |
| C4 | exploratory candidate E frozen (λ=0.25, gate b=[−2.6746,0.7038,3.9573], R6 α=0.01, experts, domain, hashes); 122 pending predictions side by side; #277/#284/#285 untouched; no Stage3 | PASS |
| C5 | CONTRACT/HYPOTHESES/DIAGNOSIS; audit; formulas/coefficients; per-fold predictions; OOF/CI/ablations/calibration/counterexamples; tests (5 pass); logs; REPORT/VERDICT; infra drift 0; commit+push | PASS |

## Hypotheses

| id | status | reason |
|---|---|---|
| H51 (gated two-regime generalizes) | **REFUTED** | E 0.0030271 vs A 0.0029719, CI [−1.20e−4,+9.7e−6]; ρ 0.641 vs 0.667; all arms/ablations equal or worse |
| H52 (gate redundant; six metrics absorb) | **SUPPORTED** | E ≈ A with a worse mean; single-gate/single-expert ablations worse; gate C-dominated with N–C 0.864 |
| H53 (experts overfit the 34 LL SIDs) | **REFUTED (support not the cause)** | high-g training support 25–30 ≥ 20; inner λ shrinks to 0.25; strict subset also worse — the architecture adds variance without signal, not a support artefact |
| H54 (main candidate passes the CI gate) | **REFUTED** | CI lower −1.20e−4 < 0; secondaries worse |
| H55 (gate classification ≠ regression) | **SUPPORTED** | region gate AUC 0.9808 and label gate AUC 0.8846, yet E is worse than A on MAE/ρ/reversals |

## Knowledge increment

1. A near-perfect region gate (AUC 0.98) plus a good label gate (AUC 0.88) does **not** imply a
   regression gain; classification and regression are distinct targets.
2. The gate is C-dominated (b2 ≈ 3.5 vs b1 ≈ 0.7) because N and C are 0.864-collinear: the "two
   conditions" are one axis, and a second expert does not create a second signal.
3. The residual-expert shrink to λ = 0.25 (selected by the inner CV) shows the optimal correction is
   near zero; the six-metric Ridge6 already carries the available information.
4. The failure is not a support artefact (25–30 high-g training SIDs) and does not improve the
   low/high-end biases materially; the candidate remains a documented negative result.

## Closure checklist (skill §6.4)

1. Production code implemented: yes — `audit286.py`, `gate286.py`, `freeze286.py`, tests; all
   executed with logs/exit codes.
2. Required tests executed: yes — 5 deterministic tests (constrained gate signs, soft mixture,
   gate extremes, hard gate, cross-fit exclusion) pass; run-time assertions (baseline reproduction)
   held.
3. Required training completed: **N/A** — analysis-only issue.
4. Infra boundary respected: yes — FROZEN; start/close snapshots added/removed/changed = 0; assets and
   #277/#284/#285 artifacts only read.
5. Executable unfinished work: none in scope.
