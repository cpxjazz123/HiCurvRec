**Result: chain-ALL predictive increment not established; conditional descriptive rule found; #277 retained.**

Analysis-only over the frozen #283 commit `801048ca`: 9/9 artifacts hash-verified, 253/253 per-SID
recompute against `JOIN_TABLE` (0 mismatches), #283 baseline and both its candidates reproduced
exactly (0.0029719/0.6666; fixed `all1_chain` 0.0026990/0.7665; selected 0.0028218/0.7156 with
identical fold picks).

**Identity:** 129 labels (120 strong / 9 weak; 13 high / 6 strong); folds 27/27/26/25/24; 124
prospective. Two **new time-postordered labels** with exact SID+run binding (R2 114 pairrank_enc
`9f6bf673…` = 0.0479465; R2 115 codebook `61f898…` = 0.0484688) — public now, not blind, used only as
sensitivity. #274 R2/R3 and #272 exports are separated by version and never merged (R2 trainer
`b9c3e8de…` vs checkpoint `36032024…`, 75 items differ; R3 trainer `67b5a47c…` vs checkpoint
`9d9f3d68…`, 85 differ; the R3 checkpoint is the labelled one).

**Findings:**
- Raw `all1_chain` is 0.984-correlated with `l1_pos_hit`; `ALL_norm` (marginal-conditional residual)
  reduces this to 0.742 and ranks recall at 0.645 [0.484, 0.778], but in the linear model it is
  significantly **worse** than the baseline (0.0031463; CI [−3.57e−4, −1.40e−5]).
- The stricter primary matched set (same chain marginals + six metrics + L2/collision/L3 +
  L1 occupancy/entropy; 100 pairs / 48 SIDs) gives all1 0.122 [−0.239, 0.451] and ALL_norm 0.149
  [−0.186, 0.462]: the #283 marginal-matched 0.191 **does not survive**. LOO ranges are narrow
  (0.032–0.246 / 0.123–0.295) — no single-SID dominance; strong-only 0.067/0.169.
- Chain is not structurally special: chain 0.395 ≈ fork 0.388 / merge 0.401 / revchain 0.399.
- Over-merging hypothesis not supported: `cond_refine` vs recall 0.001 [−0.202, 0.205]; matched sign
  negative (n.s.); `all3_chain` correlates 0.883 with the catalogue collision.
- **Conditional rule (supported):** ALL ranks recall only in the low L1-negative / low-collision
  bands (0.907 [0.779, 0.963] and 0.846 [0.665, 0.941]) and vanishes in the high bands (0.039,
  0.151). Descriptive, with frozen feature-only terciles.
- **Predictive gate:** B 0.0026990/0.7665 CI [−7.65e−5, +6.41e−4]; D 0.0029465/0.7712 CI
  [−4.13e−4, +4.22e−4]; selected 0.0030260/0.7329 CI [−4.89e−4, +3.10e−4] — all cover 0; C is
  significantly worse. Full-cohort B coefficients: `l1_pos_hit` +0.0662, `all1_chain` **−0.1125**
  (collinearity sign flip). 13/0/0 at >0.06; 0 rows >0.065 → NOT_EVALUABLE.
- **Power:** A−B mean 2.73e−4, SD 2.12e−3 → **374 independent SIDs for 80 %** (cluster 353);
  MDE at 129 = 4.64e−4. Underpowered at 129, not impossible.
- **Time-postordered sensitivity (n=2, not blind):** #277 0.05319/0.05083 vs B 0.05295/0.05005 vs
  actual 0.04795/0.04847 — both over-predict; no formula change.

**Frozen research candidate (not a replacement):** B with alpha 0.01 and the recorded coefficients;
predictions for the 122 label-pending SIDs + 9 recovered unlabelled exports are in
`PROSPECTIVE_284.json`; the #277 predictor and its 124-row prospective file are unchanged.
Replacement requires the historical CI gate **and** a future separately-authorized independent-label
validation. No Stage3 was started. Infra drift: 0.

Commit: b106d01e61a73d903aad7e58f77e63a42afaa745 (main); issue CLOSED.
