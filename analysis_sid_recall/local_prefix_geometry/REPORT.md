# Issue #131 - singleton-aware local geometry of the SID prefix groups

110 (SID, run) rows, joined by SID content hash. Provenance: HASH_PROVEN 0, RECORD_HASH_MATCH 110, MISSING 0.

## Q1 - what local geometry do multi-item groups contain?

Support audit (median across arms): L1: occupied 256, singleton 0, multi 256, item coverage by multi groups 1.000; L2: occupied 13428, singleton 8472, multi 3472, item coverage by multi groups 0.655; L3: occupied 22408, singleton 20819, multi 1167, item coverage by multi groups 0.152.

| level | four-point dev (norm) | transitivity | cycle surplus / node | largest component | conductance | detour ratio |
|---|---|---|---|---|---|---|
| L1 | +0.0372 | +0.140 | +0.187 | +0.401 | +0.9931 | +1.060 |
| L2 | +0.0186 | +0.098 | +0.002 | +0.440 | +0.9987 | +1.000 |
| L3 | +0.0022 | +0.111 | +0.001 | +0.508 | +0.9984 | +1.000 |

## Q2 - what changes from L1 to L2 to L3 after accounting for singletons?

| level | singleton rate (items) | item coverage by multi-item groups | item coverage by >=4 groups | geometry summary vs L1 |
|---|---|---|---|---|
| L1 | 0.000 | 1.000 | 1.000 | +0.000 |
| L2 | 0.631 | 0.655 | 0.312 | +0.173 |
| L3 | 0.929 | 0.152 | 0.022 | -0.082 |

## Q3 - is the geometry different from size/degree-matched nulls?

| feature | global null delta | activity-stratified delta | parent-conditioned delta |
|---|---|---|---|
| `transitivity` L1 | +0.124 | +0.112 | n/a |
| `cycle_surplus_normalised` L1 | +0.187 | +0.185 | n/a |
| `largest_component_fraction` L1 | +0.363 | +0.338 | n/a |
| `transitivity` L2 | +0.076 | +0.057 | +0.051 |
| `cycle_surplus_normalised` L2 | +0.002 | +0.002 | +0.001 |
| `largest_component_fraction` L2 | +0.043 | +0.045 | +0.027 |
| `transitivity` L3 | +0.057 | +0.036 | +0.000 |
| `cycle_surplus_normalised` L3 | +0.001 | +0.001 | +0.000 |
| `largest_component_fraction` L3 | +0.049 | +0.049 | +0.016 |

## Q4 - does geometry predict Recall beyond alignment, collision and resolution?

Nested models on the matched rows (n=89), leave-one-series-out RMSE:

| model | features | n | RMSE | in-sample R2 |
|---|---|---|---|---|
| `A_controls` | 5 | 89 | 0.00472 | +0.7230 |
| `B_plus_local_geometry` | 8 | 89 | 0.00437 | +0.7905 |
| `C_plus_hierarchy_change` | 11 | 89 | 0.00723 | +0.8285 |

| step | n | RMSE before | RMSE after | relative |
|---|---|---|---|---|
| A_controls->B_plus_local_geometry | 89 | 0.00472 | 0.00437 | +7.5% |
| B_plus_local_geometry->C_plus_hierarchy_change | 89 | 0.00437 | 0.00723 | -65.6% |

Strict A2 sensitivity: A_controls: n 73, UNDEFINED_SINGLE_SERIES; C_plus_hierarchy_change: n 58, UNDEFINED_SINGLE_SERIES.
The strict subset is a single mechanism series, so leave-one-series-out prediction is undefined there; that is reported, not hidden.

Is the geometry a resolution proxy? (partial Spearman given log occupied prefixes, A_l, collision and L1 code usage):

| feature | partial given resolution | Spearman vs log occupied |
|---|---|---|
| `geo_l1` | +0.526 | +0.012 |
| `four_point_norm_l1` | +0.307 | -0.026 |
| `cycle_surplus_normalised_l1` | +0.611 | +0.132 |
| `largest_component_fraction_l1` | +0.412 | +0.011 |
| `geo_l2` | +0.126 | +0.765 |
| `four_point_norm_l2` | +0.210 | +0.190 |
| `cycle_surplus_normalised_l2` | +0.466 | -0.467 |
| `largest_component_fraction_l2` | +0.065 | +0.905 |
| `geo_l3` | +0.329 | +0.503 |
| `four_point_norm_l3` | +0.234 | +0.081 |
| `cycle_surplus_normalised_l3` | +0.331 | -0.364 |
| `largest_component_fraction_l3` | +0.466 | +0.597 |

## Q5 - stability across mechanism families and supports

The strict A2 subset is 99% one series; the leave-one-series-out numbers above are the only family-honest estimate, and no independent-replication claim is made.

## Figures

| figure | defined values | feature non-missing |
|---|---|---|
| `scatter_geometry_L1.svg` | 89 | 89 |
| `scatter_geometry_L2.svg` | 108 | 108 |
| `scatter_geometry_L3.svg` | 110 | 110 |

## Verdict

**Q1 (what geometry is actually there) - MEASURED, with L3 uninformative.** At L1 the median arm has 256 multi-item groups covering 100.0% of the items; their induced subgraphs are sparse and far from trees (four-point deviation +0.037 normalised, transitivity 0.140, cycle surplus 0.187 per node, largest component 0.401, conductance 0.993). At L3 the median arm has 20819 singleton groups covering 84.8% of the items, so **L3 local geometry is STRUCTURALLY_UNINFORMATIVE as a clustering signal** — it describes near-singleton groups, not SID-induced clusters, and no L3 geometric claim is made.

**Q2 (what changes L1 -> L2 -> L3) - MEASURED.** Refinement replaces multi-item structure with singletons: item coverage by multi-item groups falls 1.000 -> 0.655 -> 0.152. The hierarchy table counts the parents that split into several multi-item children versus into one multi-item child plus singletons versus only singletons (columns `L1_to_L2_*` and `L2_to_L3_*`).

**Q3 (different from the nulls) - MEASURED at L1, weak at L2/L3.** Against the fixed-size global, activity-stratified and parent-conditioned shuffles the L1 geometry shows the largest deltas (transitivity +0.124 global, +0.112 stratified), while the L2 parent-conditioned deltas are +0.051 and the L3 +0.000. Standardised effects are only reported where the null spread is at least 1e-3.

**Q4 (predicting Recall beyond alignment, collision, resolution) - MEASURED, L1 survives the controls.** The geometry block improves the leave-one-series-out RMSE from 0.00472 to 0.00437 (+7.5%), while adding the hierarchy-change block makes it worse (-65.6%). Individually, the L1 geometry summary keeps a partial correlation of +0.526 given log occupied prefixes, A1, collision and L1 code usage (vs log occupied +0.012); the L1 cycle surplus keeps +0.611. The L2 summary is largely resolution (partial +0.126, size correlation +0.765).

**Q5 (stability and supports) - NOT ESTABLISHED across families.** The strict A2 subset is a single mechanism series, so leave-one-series-out prediction is undefined there (`UNDEFINED_SINGLE_SERIES`); the admitted set is dominated by the same series and the wider tier was not measured. The L1 result is therefore an in-sample-family, out-of-series estimate, not independent replication, and no causal claim about hyperbolic RQ-VAE follows from it.

**Scope.** Diagnostic only; no Stage 2 loss, codebook or Stage 3 job was touched, and no geometry-to-model-curvature causality is claimed.
