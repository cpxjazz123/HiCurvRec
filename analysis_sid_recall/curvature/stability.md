# The curvature-recall association is not stable across measurement budgets

The corrected (undirected, exact-verified) sweep was re-run with the sampled
edges doubled (16 → 32) and the number of null replicates raised (2 → 3), on
the same 182 arms. Seeds, features and the `1/2`-hop verification are unchanged.

| feature | 16-edge rho (CI95) | 32-edge rho (CI95) | shift |
|---|---|---|---|
| z_curv_L1 | −0.061 [−0.207, +0.093] | −0.091 [−0.232, +0.066] | −0.030 |
| z_curv_L2 | +0.051 [−0.083, +0.198] | **+0.150 [+0.001, +0.306]** | +0.100 |
| z_curv_L3 | −0.049 [−0.206, +0.108] | **+0.145 [+0.006, +0.285]** | +0.194 |
| z_curv_L1_minus_L3 | −0.037 [−0.191, +0.117] | −0.095 [−0.237, +0.060] | −0.058 |
| sign_change_L1_L3 | +0.098 [−0.058, +0.242] | −0.059 [−0.219, +0.111] | −0.157 |

What this says:

* **The association is not established.** Every feature's confidence interval
  contains zero in at least one budget, and no feature survives a Bonferroni
  threshold for five tests (0.01) in either budget.
* **The estimates are budget-dependent.** The level-3 feature changes sign
  between budgets (from −0.049 to +0.145) and the level-2 feature moves by
  +0.100; the largest shift is 0.194, which is the same order as the effect
  the original (pre-fix) analysis claimed (ρ ≈ −0.20).
* **The two features that reach p ≈ 0.05 at the larger budget sit at the edge
  of zero** (lower bounds +0.001 and +0.006) and are the ones whose sign flips —
  i.e. the noise of the per-arm curvature estimator dominates the association.

Conclusion: on the corrected graphs, with a verified metric, the curvature of
the SID's prefix quotient **cannot be claimed to track the downstream recall**,
positively or negatively. The pre-fix "−0.20 profile correlation" is withdrawn
(Issue #128) and the re-measurement does not resurrect it: the point estimates
move by up to 0.19 with the sampling budget alone.

Artefacts: `curvature_summary.json` (the 32-edge / 3-null run, authoritative),
`curvature_summary_16edge.json` (the saved 16-edge summary), the per-arm
features `per_arm_features_32edge.json`, and the earlier 16-edge report in the
commit `f5b95977` (the features themselves were replaced by the larger run; the
summary's statistics are preserved here and in `curvature_summary_16edge.json`).

## The pair table the numbers rest on

These statistics were computed on all 182 rows of `sid_recall.csv`. The pair
audit (`analysis_sid_recall/PAIR_AUDIT_FINDINGS.md`) shows only 73 of them are
verified by the same-commit and same-directory criteria, and on those no feature
correlates with the recall (`z_curv_L2` +0.129 p 0.28, `z_curv_L3` −0.044
p 0.71, the profile +0.141 p 0.24; the same-commit set gives +0.074 and +0.029).
The budget instability reported above and the pair-verification result point the
same way: the association is not reproducible.

## The replicate budget moves the coefficient as much as the edge budget

On the same arms and the same observed sample, only the number of null
replicates changed (nested): the level-2 coefficient runs −0.136 (2 replicates) →
−0.206 (3) → −0.263 (4, p 0.028) → −0.250 (6) → −0.185 (8, p 0.112), and the
level-1 coefficient drifts +0.061 → +0.154. The significance therefore belongs
to the budget, not to the data; see `../CONVERGENCE_FINDINGS.md`.
