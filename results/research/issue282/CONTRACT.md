# Issue #282 — pre-registered contract (frozen before any type/Recall association)

Frozen 2026-11-28 16:0x. Training graph only (same builder as #278–#280: consecutive history pairs plus
`(history[-1], target)`); validation rows and the legal-negative definition are the frozen #277 ones.

## 1. Pair-level directed types (multi-label, counts)

For a directed pair `(a,b)`:
| id | definition | role note |
|---|---|---|
| `chain` | `\|succ(a) ∩ pred(b)\|` (a→m→b, directed 2-hop) | a source, b target, m bridge |
| `revchain` | `\|succ(b) ∩ pred(a)\|` (b→m→a) | reversed direction |
| `merge` | `\|succ(a) ∩ succ(b)\|` (a→m, b→m) | m shared successor |
| `fork` | `\|pred(a) ∩ pred(b)\|` (m→a, m→b) | m shared predecessor |
| `closed` | only when `c(a,b)>0`: `\|succ(a) ∩ pred(b)\|` with the direct edge present; `mutual` when `c(b,a)>0` | feed-forward vs mutual |

A pair may carry several types simultaneously; no unique label is forced. A potential closure on an
**unseen** pair is an open-structure candidate, never an observed triangle. 4-node graphlets are
**excluded** (the existing asset is sampled and cannot certify pair-level roles); node-level Orbit
totals are never used as pair-level structure. Curvature is **excluded** (coverage 12.9 % positives /
2.65 % negatives in #279).

## 2. Matched negatives

For every validation row (positive), five legal negatives are drawn with seed 42, attempts ≤ 20 each:
priority level 2 = same (anchor out-degree decile × target in-degree decile) **and** type set intersects
the positive's type set; level 1 = degree decile match only; level 0 = any legal negative. The level
distribution per type is reported; this is a sampled approximation (265 k pairs), stated as such.

## 3. Per-SID per-type fidelity

`P_t` = mean `1[L1(a)=L1(b)]` over positives with type t; `N_t` = mean over those rows' matched
negatives of `1[L1(a)=L1(neg)]` (empirical, sampled); `ER_t = P_t/N_t`, `Δ_t = P_t − N_t`; support =
number of positive rows of type t. L1+L2 as secondary. A type enters a SID feature only if its support
≥ 100 rows. Anchor-clustered CIs (500 bootstrap resamples of anchors) are reported for the aggregate
type P/N/ER across SIDs, not per SID.

## 4. Frozen SID-level features (main, at most two)

`F1 = ER_merge − ER_chain`; `F2 = ER_fork − ER_chain` (degree-matched negatives, L1). NA when either
term is undefined or below the support floor.

## 5. Model comparison and criterion

Baseline `A` = the **#277 corrected** Ridge6 OOF (procedure of `issue277/AUDIT_277.json`
`inner_alpha_fixed`, reproduced: MAE 0.002972, ρ 0.6666). The #281 baseline is not mixed in.
`D1` = A + F1; `D2` = A + F2; `D3` = A + both; same folds, scaler inside inner folds, one OOF per SID.
Pass = OOF MAE decrease vs A with the SID-group paired bootstrap 95 % CI **lower bound > 0** and
Spearman/Kendall and the >0.06 FP/FN not worse, on the full 129 and the strict 120 subsets; 0.065
NOT_EVALUABLE. Otherwise keep #277; no new predictor.
