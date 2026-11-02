# Issue #129 - layerwise behaviour curvature, controls and the matched arm

110 admitted pairs; 109 with every feature defined. Provenance: HASH_PROVEN 0, RECORD_HASH_MATCH 108, MISSING 2.

## Q1 - does each level's measured curvature predict Recall?

| feature | n | Spearman | p | Holm p | Kendall | 95% CI |
|---|---|---|---|---|---|---|
| `g1` | 109 | +0.099 | 0.306 | 0.612 | +0.090 | [-0.128, +0.317] |
| `z1` | 100 | -0.074 | 0.454 | 0.612 | -0.063 | [-0.278, +0.138] |
| `g2` | 109 | -0.401 | 0.000 | 0.002 | -0.285 | [-0.576, -0.203] |
| `z2` | 107 | +0.179 | 0.069 | 0.253 | +0.131 | [-0.024, +0.384] |
| `g3` | 109 | -0.371 | 0.000 | 0.001 | -0.249 | [-0.540, -0.180] |
| `z3` | 109 | +0.180 | 0.063 | 0.253 | +0.124 | [-0.005, +0.370] |

## Q2/Q3 - does curvature add beyond the behaviour alignment and collision?

Controls `A1, A2, collision`: n=100, out-of-series RMSE 0.00349; with the three curvatures: n=100, out-of-series RMSE 0.00926; relative change -165.3%.

| feature | coefficient | cluster-bootstrap 95% CI |
|---|---|---|
| `A1` | +0.0674 | [-0.0085, +0.1438] |
| `A2` | +0.0846 | [-0.0350, +0.5275] |
| `collision` | -0.0141 | [-0.0261, -0.0062] |
| `z1` | -0.0000 | [-0.0000, +0.0000] |
| `z2` | +0.0001 | [-0.0001, +0.0004] |
| `z3` | -0.0001 | [-0.0006, +0.0002] |

### Is the raw curvature a proxy for SID resolution?

Partial Spearman controls the level-2 quotient size, the level-2 code count, A2 and collision (rank-residualised).

| feature | partial Spearman | Spearman vs log(edges) |
|---|---|---|
| `g1` | +0.364 | +0.119 |
| `z1` | -0.029 | -0.425 |
| `g2` | -0.005 | -0.929 |
| `z2` | +0.135 | +0.249 |
| `g3` | +0.022 | -0.821 |
| `z3` | +0.259 | -0.024 |

## Nonlinearity - is the relation an interior optimum?

| feature | n | linear RMSE | quadratic RMSE | improvement | vertex | inside range | supported |
|---|---|---|---|---|---|---|---|
| `z1` | 100 | 0.00827 | 0.28147 | -3302.3% | -2278.7017 | True | False |
| `z2` | 107 | 0.00387 | 0.01970 | -409.3% | -68.0302 | True | False |
| `z3` | 109 | 0.00333 | 0.00429 | -28.9% | -2.4837 | True | False |

## WP3 - the intra- minus inter-prefix gap on the original graph

Fixed sample: 1500 edges of 20264 eligible (315391 total; degree cap 60), mean curvature not recorded.
Degree-preserving rewiring null: NOT_MEASURED (the exact-transport rewiring null needs the accelerator interpreter's POT and the shared filesystem stalled; the prefix-label-shuffle null is measured for every arm instead).

| level | defined arms | undefined arms | mean Delta | median null p | Spearman vs Recall (defined) | partial given A_l + collision |
|---|---|---|---|---|---|---|
| delta1 | 108 | 2 | +0.0109 | 0.502 | 0.254 (p 0.008, n 108) | 0.267 |
| delta2 | 31 | 79 | +0.0095 | 0.542 | 0.229 (p 0.207, n 31) | 0.379 |
| delta3 | 10 | 100 | +0.0080 | 0.450 | -0.073 (p 0.838, n 10) | n/a |

Per-arm label-shuffle nulls, counts and coverage are in `experiment_table.csv` (`delta{1,2,3}_null_p`, `delta{1,2,3}_n_within`).

## Sensitivity across the tiers

| tier | feature | n | Spearman | p |
|---|---|---|---|---|
| strict_A2_73 | `z1` | 70 | +0.150 | 0.207 |
| strict_A2_73 | `z2` | 73 | +0.129 | 0.271 |
| strict_A2_73 | `z3` | 73 | -0.044 | 0.714 |
| wide_code_path_178 | `z1` | 167 | -0.091 | 0.227 |
| wide_code_path_178 | `z2` | 176 | +0.161 | 0.034 |
| wide_code_path_178 | `z3` | 178 | +0.140 | 0.061 |

## Q4 - does a curve-driven model curvature improve matched Stage 3 Recall?

**Not claiming an arm.** No arm was launched: the only level-dependent signal that survives the checks is the raw-curvature-versus-recall association, and it is a proxy for the quotient's size (see below), not evidence that a different model curvature would help. The pre-registered mapping and the four arms are recorded in `PREREGISTRATION.md` and `results/research/layerwise_curvature/manifest.json`.

## Verdict

**Q1.** The raw curvature at levels 2 and 3 is strongly *negatively* correlated with Recall (Spearman -0.401 and -0.371; Holm-adjusted p 0.002 and 0.001), but it is nearly a measurement of the quotient's size (Spearman with log quotient edges -0.929 and -0.821) and it vanishes once that size, the code count, the alignment and collision are partialled out (partial -0.005 and 0.022). The size-preserving standardised curvature z_l is null at every level (raw p 0.45, 0.07, 0.06; Holm-adjusted 0.61, 0.25, 0.25). **The layer's curvature does not predict Recall beyond SID resolution.**

**Q2.** Curvature adds nothing over the controls: the leave-one-series-out RMSE is 0.00349 with the behaviour alignment and collision alone and 0.00926 with the three curvatures added (-165.3%). The joint-model coefficients of z1/z2/z3 are indistinguishable from zero (cluster-bootstrap CIs in the table above).

**Q3.** The relation is not different by level in a way that survives controls: the level-2 and level-3 raw associations have the same sign and the same resolution proxy, level 1 is null. Across the tiers (see the sensitivity table) the level-2 coefficient changes sign, and the strict A2 set is 99% one series, so no cross-mechanism claim is possible.

**WP3.** On the fixed item-graph sample, the intra-minus-inter gap is defined for 98% of arms at level 1, 28% at level 2 and 9% at level 3 (the L3 singleton trap is reported as UNDEFINED, never as zero). Delta1 correlates with Recall (0.254, p 0.008) and keeps a partial correlation given A1 and collision (0.267), but it does **not** improve out-of-series prediction over the controls (RMSE 0.00368 -> 0.00377, -2.5%). Delta2 and Delta3 have too few defined arms to carry a conclusion.

**Q4.** No matched Euclidean / shared / grid / adaptive arm is launched: the diagnostics do not support a level-dependent signal, so the expensive claim-producing experiment has no pre-registered justification. This is the issue's negative verdict, recorded with the mapping that would have been used.

**Provenance and limits.** 108 pairs match their recorded content hash but not the run's own stored path (0 do), so the bytes are content-verified rather than path-attested; a prospective Stage 3 must use `attest.py`, which fails closed on a hash or item-order mismatch. The degree-preserving rewiring null is not measured (see WP3). Historical recall is observational.
