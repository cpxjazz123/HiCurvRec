# Issue #270 diagnosis

## 1. Phenomenon (kept separate from causes)

#269 moved the hard-q A.1 curvature ranking from Spearman `0.084` (frozen `Z_ref`) to
`0.353`, kept the SID inside the collision gate (`0.1563 < 0.40`), and still lost
recommendation: R@10 `0.045231` vs the #265 parent `0.052595` and canonical TIGER `0.055206`.
At the same time the number of distinct L2 prefixes fell from `10,918` to `6,246` and the
Stage3 training loss was lower. The phenomenon is a **dissociation** between continuous
curvature fidelity and discrete recommendation.

## 2. Competing explanations and the measurement that can separate them

| ID | explanation | measurement in this issue | status before analysis |
|---|---|---|---|
| E1 | **SID capacity/prefix compression** caused the loss; curvature improved but the code space collapsed, so behaviourally different items share prefixes | per-arm L2 prefix count/split, L1/L2/L3 collision, per-item prefix purity, the four quadrants (ΔA.1 error x ΔSID fidelity), and the #265 vs #269 lost-prefix/merged-item audit | **CONSISTENT: #265→#269 L2 170→113, split 127, merge 93, ΔR −0.00736; curvature→SID ρ −0.28, SID→ΔR ρ −0.47** |
| E2 | **No causal channel**: the A.1 proxy is evaluated on a ruler that is not the Euclidean selection cost, so its improvement cannot move hard q/SID | correlation of ΔA.1 error with ΔSID and Δrecall at arm and item level; own-scale sensitivity; the #269 per-triple κ ruler | **SUPPORTED as a limitation: arm Spearman −0.068; item ΔA.1 vs Δhit ρ −0.009 (81 % ties); the κ_t ruler reverses #269 vs #265 (J 0.288 vs 0.168)** |
| E3 | **Protocol/family confounding**: single-card #269 vs four-card history, and mechanism families differ; apparent curvature-recall associations are driven by these | single vs four-card stratification, family-cluster bootstrap, leave-one-family-out | **SUPPORTED: between/within family SD 0.00356/0.00094; single-card n=3 (degenerate); leave-one-family-out in [−0.26,+0.01]** |
| E4 | **Item-level heterogeneity**: mean A.1 improvement hides items that improve and items that degrade; recommendation follows the per-item sign, not the mean | per-item ΔA.1 vs Δhit/Δrank, sign test, four quadrants, counterexample list | **NEGATIVE for a usable item-level law: pooled ρ −0.026, 84 % zero-Δhit ties; class 5 (#265) is the explicit improve-error/gain-hit counterexample** |
| E5 | **Baseline noise**: historical TIGER reruns span `0.040199`–`0.055537`; a single R@10 delta may be within rerun noise | TIGER rerun range reported; effect sizes with intervals; no single-run significance claim | **USED: 0.040199–0.055537; all deltas are reported with arm/family-level intervals** |

## 3. Why this issue is analysis-only

The issue explicitly forbids Stage2/Stage3, retraining and mechanism changes, and asks for an
observational, triplet-level analysis of archived runs. No GPU training is required; the only
compute is reading archived checkpoints/SIDs/predictions and the frozen graph assets. The
"six training questions" therefore do not gate this round; instead the pre-registered
falsifiable predictions P1–P4 in `CONTRACT.md` are the acceptance criteria.

## 4. Confounders to control (pre-registered)

Training protocol (single vs four cards), mechanism family, dataset split (all use the frozen
Amazon 2023 Instruments split), target popularity (training frequency), centre degree, sequence
length (not stored in predictions: recorded as a gap), semantic similarity (frozen embeddings),
reconstruction error (available only through the checkpoint q; reported where possible), SID
prefix group size, collision, and the number of triplets per centre. Test predictions are used
only as explanatory error analysis; no weight, threshold or model selection uses them.

## 5. What would count as a stable relationship

A relation must survive: (i) arm-level sign and family-cluster CI, (ii) item-level paired sign
test with a centre-cluster CI, (iii) leave-one-family-out without sign flips, (iv) FDR across the
pre-registered strata, and (v) an explicit counterexample audit. Failing any of these is reported
as "not identifiable with the available independent runs", not smoothed into a monotone claim.

## 6. Known evidence gaps (recorded before computing)

- **G1**: no validation predictions exist (Stage3 `NO_EVAL=True`); the saved predictions are the
  frozen test set, so the recommendation linkage is explanatory, not an independent validation.
- **G2**: 30 of the 42 usable arms have only the legacy arm-directory SID/checkpoint binding (no
  immutable-run hash in the Stage3 record); they are marked `arm_dir_legacy`.
- **G3**: the #259–#264 geometry arms (`role_geom*`, `mixed_rq_F0`) have no saved Stage3
  predictions locally, so they appear in the inventory but cannot enter the recommendation
  linkage.
- **G4**: sequence length per test row is not stored in the predictions; it is not reconstructed
  from parquet (the frozen test order is not exposed), so it stays uncontrolled and is listed as a
  limitation.

## 7. Resolution (2026-11-27)

The dissociation is **not** explained by a monotone geometry→recommendation channel (P1 null,
P2 null). Three facts survive the pre-registered checks: (i) the SID layer is an observable
concomitant of the geometric change (curvature→SID ρ −0.28, SID→ΔR ρ −0.47), but the conditional
interaction is not significant; (ii) protocol/family dominates the arm-level variance
(between/within SD ≈ 3.8×; the only single-card stratum is n=3); (iii) the "improvement" premise
is ruler-dependent — the Σ absolute MAE favours #269 over #265 by 0.249, while the relative J and
the fitted κ_t ruler favour #265. No causal claim is made; the controlled export-geometry design
is in `VERDICT.md`.
