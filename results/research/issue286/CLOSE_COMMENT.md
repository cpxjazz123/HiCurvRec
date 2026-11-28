**Result: the gated two-regime predictor does not improve the six-metric Ridge6; every arm is equal or
worse; #277 retained.**

Analysis-only over #285 `a433172f` and #277 `d1cb5cbc`. Audit: 129 labels (120/9; 13 high/6 strong),
127 clusters, folds unchanged; label range [0.02601,0.064764], 0>0.065; baseline reproduced exactly
(0.002971939/0.666609, 13/0/0); #285 LL 34 (11 high, 4 high-strong), P 0.951, reversals 43/560;
#277 domain/residual interval [0.002376,0.003620]/label aggregation recorded.

**Design:** primary gate `g=sigmoid(b0−b1·N_std−b2·C_std)`, b1,b2≥0, trained on the **feature region**
`lowN&lowC` (label-free; training-fold tertiles) — OOF **AUC 0.9808, Brier 0.0662**; diagnostic
label gate on `y>0.06` — **AUC 0.8846, Brier 0.0750**. Gate coefficients per fold b0 −2.37…−2.70,
b1 0.52–0.81, **b2 3.47–3.80** (C-dominated; N–C 0.864). Main candidate E: soft-gated residual experts
`R6 + λ[g·e_good(P,ER)+(1−g)·e_other(x6)]` on cross-fitted residuals; inner λ = [0.5,0.25,0.25,0.25,0.25];
high-g training support 25–30 (≥ pre-registered 20).

**Acceptance (same 129/folds):** A 0.0029719/0.6666; B fixed [P,ER] 0.0040165 (CI [−1.59e−3,−5.40e−4]
worse); C hard gate 0.0030846 [−3.43e−4,+1.05e−4]; D soft gate 0.0031631 [−4.07e−4,+4.1e−6];
**E main 0.0030271/0.6408, CI [−1.20e−4,+9.7e−6]** — no arm passes; ρ/reversals worse; strict 120
subset worse (0.003147/0.582 vs 0.003096/0.613). Ablations P-only/ER-only significantly worse; N/C
single gates no better. **Gate classification ≠ regression**: AUC 0.98/0.88 but no MAE gain.
Cells: LL MAE A 0.001944 vs E 0.002014, HH 0.003961 vs 0.004104; bias magnitudes shrink only ~4 %.
Counterexamples listed (SID 49 +1.80e−3, SID 2 −1.59e−3, SID 35 +1.39e−3). Power: the E mean
difference is negative → no n recovers it; MDE at 129 = 8.3e−5.

**Frozen:** E as an explicitly exploratory negative result (λ=0.25, gate b=[−2.6746,0.7038,3.9573],
R6 α=0.01, expert coefficients/scalers, domain, hashes) with 122 pending predictions side by side with
#277/#284/#285; the 2 time-postordered labels (0.04795/0.04847) are diagnostic only. #277/#284/#285
files unchanged; no Stage3; infra drift 0.
