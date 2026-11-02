# Issue #132 - full-census local geometry (no sampling caps)

108 (SID, run) rows; excluded for a measured digest mismatch: 2. Provenance: HASH_PROVEN 0, RECORD_HASH_MATCH 108, MISSING 0, HASH_MISMATCH 0.

## Census completeness

| quantity | L1 | L2 | L3 |
|---|---|---|---|
| occupied prefix groups (median) | 256 | 13470 | 22488 |
| singleton groups (median) | 0 | 8479 | 20970 |
| multi-item groups (median) | 256 | 3519 | 1167 |
| items in multi-item groups (median) | 1.000 | 0.655 | 0.146 |
| item coverage, transitivity (median) | 0.994 | 0.067 | 0.005 |
| item coverage, four-point (median) | 0.234 | 0.034 | 0.002 |

Every occupied group of every level enters the structural census (no group count or group-size cap; singleton groups are handled analytically and count the same). The four-point metric is exact on every connected component with at most EXACT up to 24 nodes per component, else BLOCKED nodes and is `BLOCKED_NOT_FULL_CENSUS` above it; the detour metric is `BLOCKED_NOT_FULL_CENSUS` (exact all-pairs needs one full-graph BFS per item, about 2.7 million BFS for this table) and is not reported. Neither is estimated from a sample.

## Q1/Q3 - geometry vs the nulls (full census, structural metrics)

| level | transitivity | cycle rank / node | conductance | largest component |
|---|---|---|---|---|
| L1 | 0.091 | 0.249 | 0.985 | 0.430 |
| L2 | 0.111 | 0.001 | 0.999 | 0.793 |
| L3 | 0.141 | 0.000 | 1.000 | 0.967 |

| level | metric | real − global null | real − stratified null | real − parent null |
|---|---|---|---|---|
| L1 | `transitivity` | +0.077 | +0.072 | n/a |
| L1 | `cycle_rank_normalised` | +0.247 | +0.247 | n/a |
| L1 | `conductance` | -0.014 | -0.014 | n/a |
| L2 | `transitivity` | +0.102 | +0.098 | +0.059 |
| L2 | `cycle_rank_normalised` | +0.001 | +0.001 | +0.001 |
| L2 | `conductance` | -0.001 | -0.001 | -0.000 |
| L3 | `transitivity` | +0.106 | +0.101 | +0.022 |
| L3 | `cycle_rank_normalised` | +0.000 | +0.000 | +0.000 |
| L3 | `conductance` | -0.000 | -0.000 | -0.000 |

## Q2/Q4 - does geometry track Recall, and beyond the controls?

| feature | value | partial given resolution/A/collision/codes |
|---|---|---|
| `transitivity_l1` | +0.704 (p 0.000, Holm 0.005, n 105) | +0.614 |
| `cycle_rank_normalised_l1` | +0.468 (p 0.000, Holm 0.005, n 108) | +0.421 |
| `conductance_l1` | -0.662 (p 0.000, Holm 0.005, n 108) | -0.499 |
| `largest_component_fraction_l1` | +0.555 (p 0.000, Holm 0.005, n 108) | +0.443 |
| `geo_l1` | +0.634 (p 0.000, Holm 0.005, n 108) | +0.554 |
| `transitivity_l2` | +0.814 (p 0.000, Holm 0.005, n 82) | +0.568 |
| `cycle_rank_normalised_l2` | -0.028 (p 0.769, Holm 1.000, n 108) | +0.285 |
| `conductance_l2` | -0.036 (p 0.713, Holm 1.000, n 108) | -0.398 |
| `largest_component_fraction_l2` | +0.416 (p 0.000, Holm 0.005, n 108) | -0.438 |
| `geo_l2` | +0.479 (p 0.000, Holm 0.005, n 108) | -0.021 |
| `transitivity_l3` | +0.603 (p 0.000, Holm 0.005, n 81) | +0.180 |
| `cycle_rank_normalised_l3` | -0.091 (p 0.353, Holm 1.000, n 108) | +0.320 |
| `conductance_l3` | +0.184 (p 0.061, Holm 0.424, n 108) | -0.308 |
| `largest_component_fraction_l3` | +0.491 (p 0.000, Holm 0.005, n 108) | +0.182 |
| `geo_l3` | +0.553 (p 0.000, Holm 0.005, n 108) | +0.267 |

Nested models on the matched rows (n=108), leave-one-series-out RMSE:

| model | features | n | RMSE | in-sample R2 |
|---|---|---|---|---|
| `A_controls` | 5 | 108 | 0.00354 | +0.7716 |
| `B_L1_geometry` | 6 | 108 | 0.00319 | +0.8697 |
| `B_L2_geometry` | 6 | 108 | 0.00357 | +0.8064 |
| `B_L3_geometry` | 6 | 108 | 0.00376 | +0.7717 |
| `C_all_levels` | 8 | 108 | 0.00347 | +0.8715 |

| step | n | RMSE before | RMSE after | relative |
|---|---|---|---|---|
| A_controls->B_L1_geometry | 108 | 0.00354 | 0.00319 | +9.8% |
| A_controls->B_L2_geometry | 108 | 0.00354 | 0.00357 | -0.8% |
| A_controls->B_L3_geometry | 108 | 0.00354 | 0.00376 | -6.4% |
| A_controls->C_all_levels | 108 | 0.00354 | 0.00347 | +1.9% |

Strict A2 sensitivity: A_controls: n 73, UNDEFINED_SINGLE_SERIES; B_L1_geometry: n 73, UNDEFINED_SINGLE_SERIES.

## Side by side with #131's sampled numbers

| metric | level | #131 sampled rho | full-census rho |
|---|---|---|---|
| `transitivity_l1` | | +0.536 | +0.704 |
| `transitivity_l2` | | +0.322 | +0.814 |
| `transitivity_l3` | | +0.381 | +0.603 |
| `cycle_rank_normalised_l1` | | n/a | +0.468 |
| `cycle_rank_normalised_l2` | | n/a | -0.028 |
| `cycle_rank_normalised_l3` | | n/a | -0.091 |
| `conductance_l1` | | -0.615 | -0.662 |
| `conductance_l2` | | -0.399 | -0.036 |
| `conductance_l3` | | -0.437 | +0.184 |
| `largest_component_fraction_l1` | | +0.640 | +0.555 |
| `largest_component_fraction_l2` | | +0.438 | +0.416 |
| `largest_component_fraction_l3` | | +0.657 | +0.491 |

## Figures

| figure | defined | feature non-missing |
|---|---|---|
| `scatter_full_census_L1.svg` | 108 | 108 |
| `scatter_full_census_L2.svg` | 108 | 108 |
| `scatter_full_census_L3.svg` | 108 | 108 |

## Verdict

**MEASURED (full census).** Every occupied prefix group of all three levels is in the structural census, singleton groups included and handled analytically; no 64-item or 300-group cap exists. The metrics reported as all-group values (transitivity, cycle rank, conductance, largest component, internal/boundary edges, triangles, components) are exact for every group, and their measured coverage is in the completeness table.

**BLOCKED_NOT_FULL_CENSUS.** The four-point deviation is exact only on connected components up to the declared size and is reported as `PARTIAL_BLOCKED` with counts; the detour is not reported at all. Neither is substituted by a sample.

**NEGATIVE / not established.** See the tables: the L1 geometry's association with Recall and its partial correlation given resolution, alignment, collision and code usage; only a positive and control-surviving coefficient with a positive leave-one-series-out improvement would support a predictive claim, and the strict subset cannot provide it because it is a single mechanism series (`UNDEFINED_SINGLE_SERIES`).

**Scope.** Offline analysis only; no Stage 2 loss, codebook or Stage 3 job was touched, and no model curvature is inferred from graph curvature.
