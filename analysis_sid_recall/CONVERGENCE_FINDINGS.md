# Does the level curvature estimate converge? (73 verified pairs)

Set: the 73 pairs that pass the pair-audit admission rule (repo rule 64b).
Nested sample: one seeded random order of each quotient's edges per (arm, level),
reduced to its first 8/16/32/64/128 entries, so every budget is the same draw.
Null: 1-8 size-preserving label shuffles, 32 edges each. 1,152 edge solves per arm.

| level | mean @64 | mean @128 | \|Δmean\|/SE | \|Δρ\| 64→128 | ρ @64 | ρ @128 | SE | sd across arms | signal/noise |
|---|---|---|---|---|---|---|---|---|---|
| L1 | +0.4317 | +0.4317 | 0.02 | 0.042 | +0.169 | +0.127 | 0.0042 | 0.0359 | 8.59 |
| L2 | -0.1564 | -0.1566 | 0.01 | 0.019 | -0.012 | -0.030 | 0.0079 | 0.1258 | 15.89 |
| L3 | -0.2434 | -0.2431 | 0.02 | 0.008 | -0.106 | -0.115 | 0.0077 | 0.0823 | 10.67 |

## Verdict

**The three level curvature estimates converge; the correlation with the recall
does not exist at any budget.**

* The estimate is a mean over sampled edges, and by 64 edges its shift against
  the 128-edge value is within 0.02 of the estimator's own standard error in all
  three levels; the across-arm spread is 8.6-15.9 times that standard error, so
  the level differences between arms are resolvable and are not sampling noise.
* The null mean and standard deviation move by 0.1-1.1% between 4 and 8
  replicates at every level, so 4 replicates already fix the standardisation.
* The correlation is null at every rung: level 1 +0.185 (p 0.116) → +0.127
  (p 0.298), level 2 ±0.06 (p 0.58-0.94), level 3 −0.063 (p 0.597) → −0.115
  (p 0.335). Its total movement across a 16-fold change of the sample is at most
  0.06, i.e. the null is stable, not merely under-powered at one budget.
* The level-1 null mean (0.4958) sits far from the observed mean (0.4317): the
  standardised z is large because the null is tight, which is why the z alone
  was never evidence of downstream value.

Consequence for the curvature line of work: the estimator is trustworthy at 64
edges and 4 null replicates, so the negative result of the analysis cannot be
attributed to a noisy measurement.

Artefacts: `analysis_sid_recall/curvature/convergence/convergence.md` (all
tables), `convergence_summary.json`, `per_arm_convergence.json` (every arm's
ladder and null replicates).
