# Issue #286 — REPORT

Analysis-only, INFRA_MODE=FROZEN, no training, no Stage3. References fixed: #285
`a433172f…`, #277 `d1cb5cbc…`, #284 `b106d01e…`, #283 `801048ca…`; hashes in `AUDIT_286.json`.
All numbers were produced by `genrec_env/bin/python3.10` through `tools/iteration/execution.py`
(logs under `logs/`).

## C1 — audit and reproduction

- 129 labels: **120 strong / 9 weak**, 13 high (>0.06) with **6 strong**, 127 near-duplicate clusters,
  folds 27/27/26/25/24; label range **[0.02601, 0.064764]**, 0 rows > 0.065 → 0.065 NOT_EVALUABLE.
- #277 corrected Ridge6 reproduced exactly: **MAE 0.002971939, ρ 0.666609**, confusion **13/0/0**
  (TP/FP/FN).
- #285 headline cells reproduced: **LL 34** SIDs (11 high, 4 high-strong), LL P ρ **0.951**, rank
  reversals **43/560 = 7.7 %**.
- #277 predictor audit: target "Stage3 test recall@10 (protocol-majority per SID)"; feature domain
  ranges recorded (`l1_pos_hit` 0.0085–0.9998, `l1_neg_hit` 0.0039–0.9997, `er` 1.0–23.9,
  `l2_distinct` 13–16686, `l3_collision` 0.0098–0.9977, `l3_refine_bits` 0.33–6.63); empirical MAE
  interval [0.002376, 0.003620]; 0 FP in 2,000 bootstrap resamples. The two time-postordered labels
  (pairrank_enc 0.04795, codebook 0.04847) are diagnostic only.

## C2 — the gated design as executed

- **Primary gate** (label-free): `g = sigmoid(b0 − b1·N_std − b2·C_std)`, `b1,b2 ≥ 0`, fitted to the
  training-fold feature region `lowN & lowC` (the region uses the training-fold bottom tertiles).
  OOF region calibration: **AUC 0.9808, Brier 0.0662** — the gate is nearly a perfect region
  classifier. Per-fold coefficients: b0 −2.37…−2.70, b1 0.52–0.81, **b2 3.47–3.80** — the gate is
  C-dominated, consistent with N–C 0.864 and PC1 0.888 (#285).
- **Diagnostic label gate** (trained on `y>0.06` inside each training fold): OOF **AUC 0.8846,
  Brier 0.0750** — good classification, but see the regression below.
- **Effective support**: the high-g training branch has 25–30 SIDs per fold (pre-registered minimum
  20 met); the gate's mean weight is 0.26–0.30; the selected shrink factor λ per fold is
  [0.5, 0.25, 0.25, 0.25, 0.25] → the inner CV systematically chooses a small residual correction.
- **Main candidate E**: `R6 + λ·[g·e_good(P,ER) + (1−g)·e_other(x6)]`, experts on cross-fitted
  residuals (inner cross-fitting), α_expert = 1.0, λ inner-selected from {0.25,0.5,0.75,1.0}.

## C3 — acceptance (same 129 SIDs, same folds)

| arm | OOF MAE | ρ | reversals | strict MAE | paired CI of MAE(A)−arm |
|---|---|---|---|---|---|
| A (#277) | 0.0029719 | 0.6666 | 0.2485 | 0.003096 | — |
| B fixed single branch [P,ER] | 0.0040165 | 0.3620 | 0.3775 | — | [−1.59e−3, −5.40e−4] **worse** |
| C hard-gate two-branch | 0.0030846 | 0.5880 | 0.2740 | — | [−3.43e−4, +1.05e−4] |
| D soft-gate two-branch | 0.0031631 | 0.6517 | 0.2549 | — | [−4.07e−4, +4.1e−6] |
| **E main (residual experts + shrink)** | **0.0030271** | **0.6408** | **0.2549** | 0.003147 | **[−1.20e−4, +9.7e−6]** |
| D ablation P only (good expert) | 0.0039922 | 0.6284 | 0.2624 | — | [−1.49e−3, −5.82e−4] worse |
| D ablation ER only | 0.0032055 | 0.6325 | 0.2647 | — | [−4.82e−4, −4.6e−6] worse |
| D ablation N gate | 0.0030843 | 0.5987 | 0.2786 | — | [−2.85e−4, +5.5e−5] |
| D ablation C gate | 0.0031631 | 0.6520 | 0.2548 | — | [−4.07e−4, +4.3e−6] |

- **No arm passes**: every point estimate is worse than A; E is the closest (mean difference
  −5.5e−5, CI [−1.20e−4, +9.7e−6]) but the lower bound is < 0 and ρ/reversals are worse
  (0.641 vs 0.667; 0.255 vs 0.248). The strict 120 subset is also worse (E 0.003147/0.582 vs A
  0.003096/0.613).
- **Gate classification ≠ regression**: the primary gate is a near-perfect region classifier
  (AUC 0.981) and the label gate is good (AUC 0.885), yet the regression does not improve — the
  branch structure adds variance without signal.
- **Cell errors** (fold-local cells): LL MAE A 0.001944 vs E 0.002014; HH 0.003961 vs 0.004104;
  other 0.002842 vs 0.002862. **Bias**: low-end A −1.97e−3 vs E −1.89e−3; high-end +1.52e−3 vs
  +1.42e−3 (marginally smaller magnitude, but MAE/ranking worsen).
- **Counterexamples** (largest |error| changes): SID 49 y=0.0419 A=0.0532 E=0.0550 (+1.80e−3);
  SID 2 y=0.0353 A=0.0521 E=0.0505 (−1.59e−3); SID 35 y=0.0579 A=0.0591 E=0.0605 (+1.39e−3).
- **Power**: E's mean difference is **negative** (−5.5e−5) → no sample size recovers it; the minimum
  detectable mean at 129 is 8.3e−5. The gate's effective training support (25–30) is above the
  pre-registered minimum, so the failure is not a support artefact.

## C4 — frozen exploratory candidate

Because the main criterion fails, **#277 is not replaced**. `PROSPECTIVE_286.json` freezes E as an
explicitly exploratory candidate: λ = 0.25, gate b = [−2.6746, 0.7038, 3.9573] on the full-cohort
region tertiles (N ≤ 0.008907, C ≤ 0.116932), R6 α = 0.01, expert coefficients/scalers, the feature
domain ranges and input hashes; predictions for the **122** truly unlabeled SIDs, side by side with
#277/#284/#285. The two time-postordered labels: #277 0.05319/0.05083 vs E 0.05310/0.05077 vs actual
0.04795/0.04847 (both over-predict, n=2 diagnostic). No Stage3 was started; the #277/#284/#285 files
are unchanged.

## Answers

1. **Does the gated two-regime predictor improve the #277 error?** No. The main candidate E is slightly
   worse (0.003027 vs 0.002972, CI lower < 0, ρ and reversals worse); all comparisons and ablations
   are equal or worse. No arm passes.
2. **Is the failure due to gate redundancy or expert overfitting?** Neither alone: the gate is
   highly accurate as a *region* classifier (AUC 0.981) and its training support is adequate
   (25–30 ≥ 20), while the inner CV shrinks the residual correction to λ = 0.25; the two experts
   simply do not carry additional signal beyond R6. The gate is C-dominated (b2 ≈ 3.5 vs b1 ≈ 0.7)
   because N and C are collinear (0.864) — the "two conditions" are one axis (#285), and the
   two-expert architecture does not change that.
3. **Does it reduce the low-recall overestimation / high-recall underestimation?** Only nominally
   (bias magnitudes shrink ~4 %), while MAE and ranking worsen; the cell-level biases are nearly
   identical. No material improvement.
4. **Power / independent SIDs?** The observed effect is negative, so no number of independent SIDs
   validates it; the MDE at 129 is 8.3e−5. A future independent-label campaign could only test the
   *opposite* direction; this candidate is retained only as a documented negative result.
