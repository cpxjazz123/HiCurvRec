# Issue #276 — DIAGNOSIS (root cause of the #275 negative, competing explanations, admission six)

Round start 2026-11-28 13:2x; INFRA_MODE=FROZEN. Analysis-only: no training, no Stage3 (issue text).

## Phenomenon (what #275 actually observed)

* Outer family-LOFO (n=40, 8 families) gave PySR MAE 0.003491 / ρ +0.110 vs ridge 0.002497 / ρ +0.600;
  paired Δ +0.000994 CI [+0.000614, +0.001925] → PySR significantly worse. The ridge *did* beat the
  fold mean/median (0.00250 vs 0.00400, ρ −0.40), so not "no signal at all".
* Zero labels ≥ 0.06 among the 62 bound SIDs (range 0.0382–0.0573): the threshold criterion was
  NOT_EVALUABLE and only 40 rows entered the outer folds.
* Descriptive only: final-epoch `recon` MSE vs recall ρ = −0.858 (n=23), the strongest association.

## Candidate root causes (competing, not settled)

| id | explanation | discriminating measurement | status before issue-276 work |
|---|---|---|---|
| R1 | Family-LOFO with 3–5 rows per fold made both the controls and PySR unstable; the design, not the SID features, produced the negative | rerun the *same* 54 bound SIDs under SID-identity grouped CV: if ridge/PySR ordering changes materially, R1 holds | UNTESTED |
| R2 | Label-range truncation: the 43–91 historical >0.06 records had no SID identity in the working tree, so the fit never saw a positive and the >0.06 band (where a window/non-monotone relation could live) was absent | recover and bind high-recall SIDs; fit on the extended range | UNTESTED (recovery: 0 of 91 high-recall records bound on disk/ledger in the first pass) |
| R3 | Protocol mixing: the `recon` ρ −0.858 was carried by world_size/export-generation differences (majority-world_size label rule) rather than by reconstruction quality | recompute `recon` vs recall inside one protocol slice and inside the high-recall band | UNTESTED |
| N | The features genuinely carry only weak signal; ridge's ρ 0.60 was itself a family-composition artifact (families are heterogeneous recall groups) | SID-group CV on the same rows: if ridge ρ collapses to ≈0 under SID groups, N is supported | UNTESTED |

Distinguishing R1 from N is the central experiment: both use the same rows and features; only the
grouping unit changes. R2 needs the recovery pass. R3 needs the protocol audit.

## Admission six (this is an analysis round; answered so the design is not an undirected retry)

1. **Problem**: #275's negative was produced on a design with zero positives and family grouping;
   it cannot answer whether SID features predict recall over the full observed range.
2. **Evidence it matters**: 43 cohort rows with recall > 0.06 exist and none of them had a bindable
   SID in the fitted cohort; the fitted band is 0.038–0.057. Controls were degenerate (mean had
   ρ −0.40, i.e. the folds were ordered adversarially by family), so the "beat the control" test was
   asymmetric.
3. **Why the new design acts on the cause**: grouping by SID identity removes the family prior that
   drove the degenerate controls; recovering high-recall SIDs extends the label range beyond 0.06 so
   a threshold and a window test become possible; protocol slicing tests R3.
4. **Substantive difference from #275**: unit of generalization (SID groups vs mechanism families),
   label range (recovered positives), and an explicit high-recall-band analysis. Not a new loss or
   a hyperparameter change.
5. **Metrics**: OOF MAE/RMSE/ρ/τ/reversals vs controls, >0.06 precision/recall where positives exist,
   SID-group bootstrap CI; R1 is decided by the ridge/PySR ordering change; R2 by the recovered
   >0.06 coverage; R3 by the protocol-sliced recon correlation.
6. **If still negative**: the knowledge increment is a stronger negative — SID features do not predict
   recall out-of-SID-group even with recovered positives — plus the protocol dependence of the recon
   relation and the exact identity gap of the historical high-recall assets.

## Low-cost verification (no training)

* Recovery: content-hash binding of SID files from disk, R2 ledger/cache, and git history (deleted
  blobs) — evidence for R2 and for the C1 audit.
* SID-group CV on the same bound rows — decides R1 vs N without any new model.
* Protocol-sliced recompute of `recon` vs recall — decides R3 from stored metrics.
