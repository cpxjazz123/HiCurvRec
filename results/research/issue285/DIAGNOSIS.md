# Issue #285 — DIAGNOSIS

## Observation carried from #284

Under strict matching (same chain marginals + six metrics + L2/collision/L3/occupancy) the chain-ALL
association vanishes (0.122 [−0.239, 0.451]); the only surviving descriptive structure is the
**band** result: within the low `l1_neg_hit` tercile ρ=0.907 [0.779, 0.963], within the low
`l3_collision` tercile ρ=0.846 [0.665, 0.941], while the high bands are 0.039 and 0.151. The two
band analyses were computed **separately**, so the joint condition was never tested; #284's chain-ALL
raw variable is also 0.984-collinear with `l1_pos_hit`, so the apparent conditioning may be a
range-restriction/aggregation artefact of the strong-behaviour sampling rather than a mechanism-like law.

## Candidate explanations (distinguishable)

| id | explanation | discriminating measurement |
|---|---|---|
| E1 (H37) | A genuine conditional law: controlling N (and/or C) makes P/ALL rank recall more strongly. | joint-grid cells, interaction Δρ with a cluster CI, partial association after N,C |
| E2 (H38) | Range restriction / sampling: high-recall SIDs are concentrated in the low cells; the low cell's wider range inflates rank correlation. | recall ranges per cell, variance/range restriction, exclusion of the high tail, AUC of N/C alone |
| E3 (H39) | Chain ALL adds over P under the condition. | within-cell P vs ALL comparisons; candidate C vs B/D; collinearity |
| E4 (H40) | The conditional relation is driven by a few SIDs / weak identities / the high tail and does not generalize. | leave-one-out, strong-only, near-duplicate clusters, the 2 time-postordered labels as sensitivity |
| E5 | N and C are the same source (ρ≈0.86), so the "joint" condition is not a two-axis law. | N–C correlation, overlap/Jaccard of the low sets, cell support |

## Cheap evidence already visible (pre-freeze, no cell correlations used)

Fold-local tertiles (training-fold quantiles): lowN 44, lowC 41, highN 44, highC 45; lowN∩lowC 34,
lowN-only 10, lowC-only 7, highN∩highC 39. Spearman N–C 0.864, N–P 0.801, C–P 0.739. High-recall
(>0.06) SIDs: 11 in lowN∩lowC, 2 in lowC-only, 0 elsewhere. The cells have similar mean recall
(0.050–0.055) but different upper tails/ranges — the range-restriction explanation is live.

## Formal-training admission six questions (analysis-only; answered for completeness)

1. **Problem**: is there a reproducible conditional SID-quality law when N and C are controlled, and
   does P or chain ALL add independent predictive information under it?
2. **Evidence**: #284's two separate band correlations; the joint condition was never tested; N–C 0.864
   and ALL–P 0.984 warn of same-source/aggregation explanations.
3. **Approach**: fold-local joint grid, interaction Δρ with cluster CIs, within-cell P/ALL comparisons,
   partial associations, counterexamples and robustness — all replayable from frozen artifacts.
4. **Difference from #284**: tests the *joint* condition and the interaction, distinguishes the
   range-restriction explanation, and pre-registers conditional predictors instead of the raw ALL.
5. **Metrics**: paired SID-cluster MAE gate + secondary ranking/confusion; cell-level Δρ CIs.
6. **Knowledge if it fails**: whether N/C are one axis, whether the band result is range restriction,
   and whether P/ALL add anything beyond N,C — each narrows the next round.

No Stage2/Stage3 training is performed.
