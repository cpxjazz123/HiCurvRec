# Issue #282 — DIAGNOSIS (competing hypotheses, frozen before any type/Recall association)

Phenomenon: #278–#281 failed to improve the six-metric predictor with frequency, 2-hop/common-neighbour,
edge-decoding and nonlinear terms. The open question: when the **overall** L1 sharing is similar, does
the **type** of retained relation (chain / merge / fork / closed, with a/b roles) explain Recall
differences?

| id | explanation | discriminating measurement | status |
|---|---|---|---|
| H28 | type-conditioned retention differs with Recall beyond the overall L1 rate | per-SID per-type P/N/ER vs the overall rate; similar-overall SID contrasts | UNTESTED |
| H29 | no type-specific information beyond the overall rate (null) | type contrasts do not correlate with Recall after matching the overall L1 metrics; model CIs cover 0 | UNTESTED |
| H30 | type contrasts add a Recall prediction increment | pre-registered D1/D2/D3 vs the corrected baseline A | UNTESTED |
| H31 | coverage/collinearity prevents identification | type support floors, overlap between types, matched-negative level distribution | UNTESTED |

The key design element is the **similar-overall contrast**: SIDs whose overall L1 P/N/ER are close
(frozen tolerance: |Δl1_pos_hit| ≤ 0.01 and relative ER within 20 %) but whose type composition and
Recall differ. If type retention moves with Recall there, H28 has support; if the matched pairs show no
consistent ordering, H29 holds.
