# Issue #283 — DIAGNOSIS

## Failure of the previous round (#282)

Observation: pair-level directed type-conditioned L1 fidelity (chain/revchain/merge/fork) produced ERs
within ≈4 % of each other and no predictive increment over #277 (D1/D2/D3 CIs cover 0; ρ degrades).
The item-level matched-negative sample was dominated by non-type-matched draws (level 2 = 1.35 %).

## Candidate root causes (competing)

| id | explanation | discriminating measurement |
|---|---|---|
| R1 | The triplet's joint assignment carries no information beyond the pairwise marginals — the pairwise rates are the whole signal. | joint distribution given marginals has exactly one free parameter (ALL); measure its variance and predictive value. |
| R2 | The joint information exists but only at deeper prefixes (k=2,3) where support/ENTROPY collapses. | per-level ALL support, lift, and CI; level-wise comparison. |
| R3 | The label budget (129 SIDs, 13 high-recall) cannot resolve a small joint effect. | paired bootstrap CI width and power at the observed effect size. |
| R4 | The signal is structure-specific (e.g., only merge/fork outer roles) and is diluted by pooling. | per-structure ALL contrasts with support, matched pairs, FDR. |
| R5 | The triplet asset (undirected-projection wedges around a centre) does not represent the behavioral chain/revchain structure (which needs distance-2 intermediates). | compare the asset's role-class coverage with the validation-pair intermediate context; report coverage limits. |

R1 is the null; R1 vs R4 is distinguishable by per-structure ALL contrasts; R2 by level-wise support;
R3 by CI width; R5 by the coverage audit. No claim of infeasibility is made from #282: its tested
set is a subset of the joint hypothesis space.

## Evidence review (existing artifacts, no retraining)

- `issue282/TYPE_TABLE.json`: per-SID type ERs (chain 9.99, merge 9.79, fork 9.61) — close.
- `issue282/TYPE_AUDIT.json`: matched-negative level 2 = 1.35 %.
- Algebra: with p_am, p_mb, p_ab fixed, AM = p_am−ALL, MB = p_mb−ALL, AB = p_ab−ALL, NONE =
  1−p_am−p_mb−p_ab+2·ALL, so the joint distribution's only free parameter is ALL = P(a=m=b). #282
  never measured ALL; the joint round is the first direct test of the non-marginal degree of freedom.
- #277 baseline reproduces exactly (0.002972/0.667), so the comparison instrument is stable.

## New hypotheses (inherit #282 H28–H31; see HYPOTHESES.json)

- H32: ALL is measurable and varies across SIDs/structures beyond its null expectation (descriptive).
- H33: With pairwise marginals matched, ALL differences associate with recall (joint assignment matters).
- H34: A pre-frozen ALL/role contrast adds OOF MAE increment beyond the six #277 features.
- H35: ALL lift is concentrated in high-recall SIDs / strong identities (descriptive, FDR-corrected).
- H36: Joint patterns are redundant with the marginals or support-limited (null outcome, keeps #277).

## Formal-training admission six questions (analysis-only; answered for completeness)

1. **Problem**: #282's pair-level types did not add predictive value; the untested alternative is the
   joint triple partition.
2. **Evidence**: the algebra above shows ALL is the only non-marginal dof; #282 measured only pairwise
   aggregates; its negatives were not type-matched.
3. **Mechanism**: the joint analysis computes ALL exactly over the full 28,391,948-row asset, per
   structure/level, and tests the pre-frozen AL features against the #277 baseline.
4. **Difference from the failed mechanism**: not a reparameterization of the pairwise rates — ALL is
   the unique residual dof; the negatives are exact structural strata, not sampled items.
5. **Metrics**: OOF MAE paired SID-cluster bootstrap CI; descriptive contrasts with FDR.
6. **Knowledge if it fails**: the joint assignment carries no predictive information under the
   observed support and label budget; the direction closes with that boundary stated.

No Stage2/Stage3 training is performed (issue §范围 and §4).
