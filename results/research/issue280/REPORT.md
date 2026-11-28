# Issue #280 — REPORT

Analysis-only, `INFRA_MODE=FROZEN`, no training, no Stage3. Inputs pinned: #276 `SID_TABLE.json`
`4c927914…`, `FOLDS.json` `1a8fb81d…`; #277 predictor `ec339f4a…`; #278/#279 tables.

## C1 — data isolation

Validation rows 57 439; unseen positives 53 081 (92.41 %), matching #279. Training graph built from
`train.parquet` only (2 698 387 edges, 323 633 distinct); negatives are the frozen legal-uniform draws
(seed 42), identical for every SID; the reverse pair is a separate diagnostic. No validation edge enters
the decoder tables.

## C4/C5 — SID-only directional decoder (pre-registered main model)

Frozen form: level-wise log-likelihood ratio of the target code given the source prefix, normalised by
the target prefix alone; Laplace `α`, ε from a **training-internal** source-anchored 80/20 split on five
fixed SIDs, grid α ∈ {0.1, 1, 10}, ε ∈ {1e-3, 1e-2}; selected (α, ε) = (0.1, 1e-3) with calibration
AUC **0.4992** (held-out training edges vs random non-edges) — i.e. the statistic has no signal even
in-distribution.

Validation evaluation on the unseen candidate set, all 253 SIDs:

| quantity | value |
|---|---|
| mean per-SID AUC | **0.4931** (min 0.347, max 0.714) |
| direction accuracy (mean over SIDs) | **0.4867** |
| calibration AUC | 0.4992 |

**Pure SID codes do not predict unseen directed edges**: the decoder is at chance, and the directional
likelihood ratio adds nothing over the target-prefix marginal.

## C4 — supervised interaction diagnostic and its leakage control

Hashed ordered code-pair logistic (10 fixed SIDs: canonical + 4 strong high-recall + 5 by sha order):

| variant | AUC range over the 10 SIDs |
|---|---|
| full-pair hash (degenerate mod 4096) | 0.947–0.956 |
| prefix-pair hash (degenerate mod 4096) | 0.795–1.000 |
| corrected multiplicative hash | 0.625–0.956 |
| **target-only control (no source)** | **0.933–0.986** |

The target-only control matches or exceeds every pair model: the high AUCs are **target identity /
popularity leakage** through the candidate set (true targets are popular; negatives are random items),
not source-conditional structure. Per the issue's constraint (no target-leaking features; unobserved
is not a true negative) these supervised numbers are recorded as a leakage demonstration and **excluded
from the evidence**; the pre-registered count decoder is the valid SID-only test.

## C6 — relation-level metrics and diagnostics

The main decoder's AUC is the relation-level metric (0.49); MRR/Hits@5 on a fixed 5-negative sample was
not completed because the decoders are at chance and the sample construction would not change the
conclusion (recorded as NOT_RUN with this reason). Direction asymmetry exists numerically (swap changes
the score) but carries no predictive content (direction accuracy ≈ 0.49). Code-permutation invariance
is asserted in `tests/test_decoder.py` (exact equality). Higher-order association: the count decoder's
scores are uncorrelated with the graph 2-hop/common-neighbour scores in any usable way (the decoder is
at chance while the graph scores reach AUC 0.62 in #279); this is consistent with the graph structure
not being recoverable from the SID code layout.

## C7/C8 — SID-level comparison (129 rows, fixed folds)

| model | features | OOF MAE | ρ | >0.06 tp/fp/fn |
|---|---|---|---|---|
| A control | 6 | 0.002972 | 0.667 | 13/0/0 |
| D1 | A + `dec_auc_unseen` | 0.002919 | 0.636 | 13/1/0 |
| D2 | A + `dec_margin_degmatched` | 0.002834 | 0.644 | 13/1/0 |
| D3 | A + both | 0.002813 | 0.613 | 13/1/0 |
| G279 (control) | A + #279 sharing | 0.002866 | 0.711 | 13/1/0 |

Paired bootstrap (`MAE(A) − MAE(candidate)`): D1 +5.3e-5 CI [−1.25e-4, +2.19e-4]; D2 +1.38e-4
[−6.2e-5, +3.48e-4]; D3 +1.59e-4 [−9.8e-5, +4.31e-4]; G279 +1.06e-4 [−1.0e-4, +3.37e-4].
All CIs cover 0; every candidate adds one FP; D1–D3 also **degrade Spearman** (0.636/0.644/0.613 vs
0.667). No candidate passes.

Residual correlations (cross-fitted): `dec_auc` +0.115, `dec_margin` +0.150, #279 sharing −0.085/+0.146.

## Answers

1. **Can pure SID predict unseen directed edges?** No — the pre-registered directional code decoder is
   at chance both in-distribution (calibration AUC 0.499) and on the unseen validation candidates
   (mean AUC 0.493, direction accuracy 0.487).
2. **What does it retain beyond prefix sharing?** Nothing beyond the target-prefix marginal: the
   source-conditional refinement adds zero discriminative power. Supervised models that appear to work
   are target-popularity leakage (target-only control AUC 0.93–0.99).
3. **Can higher-order structure be decoded from SID?** Not with this code layout; the graph higher-order
   scores discriminate (AUC 0.62–0.63 in #279) but the SID decoder cannot reproduce them.
4. **Does it improve Recall prediction?** No: all SID-level candidates have CIs covering 0, add one FP,
   and degrade the rank correlation. #277 remains frozen; no new predictor and no new prospective
   predictions.
