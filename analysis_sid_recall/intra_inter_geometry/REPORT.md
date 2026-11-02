# Issue #133 - corrected intra-group, new inter-group and curvature geometry

108 (SID, run) rows joined by SID content hash; excluded for a measured digest mismatch: 2. Provenance: HASH_PROVEN 0, RECORD_HASH_MATCH 108, MISSING 0, HASH_MISMATCH 0.

**Primary analysis**: raw (unadjusted) Spearman with permutation p, bootstrap 95% CI and Holm over the predeclared family of 76 tests. No control variable gates or reorders it; the partial correlations are secondary.

## What was corrected relative to #132

| metric (level) | #132 (singletons mixed, mixed units) | #133 corrected |
|---|---|---|
| `transitivity_l1` | +0.704 | +0.828 |
| `transitivity_l2` | +0.814 | +0.814 |
| `transitivity_l3` | +0.603 | +0.583 |
| `cycle_rank_normalised_l1` | +0.468 | +0.449 |
| `cycle_rank_normalised_l2` | -0.028 | +0.031 |
| `cycle_rank_normalised_l3` | -0.091 | +0.145 |
| `largest_component_fraction_l1` | +0.555 | +0.573 |
| `largest_component_fraction_l2` | +0.416 | +0.465 |
| `largest_component_fraction_l3` | +0.491 | +0.724 |
| `conductance_l1` | -0.662 | -0.604 |
| `conductance_l2` | -0.036 | -0.164 |
| `conductance_l3` | +0.184 | -0.406 |

* Singleton prefix groups are counted in the census and kept in the inter-group quotient, but they no longer enter the intra-group averages (their largest-component fraction 1, cycle rank 0 and numeric conductance were contaminating L2/L3).
* Triangles use an exact sparse `int32` accumulation instead of a dense `int8` cube (no integer overflow; a 15-clique returns 455 triangles and transitivity 1 in the tests).
* Conductance is now the binary cut over binary volumes; the previous weighted-degree/unweighted-cut mix produced values above 1 and up to 1e18. A zero complement volume is `UNDEFINED_ZERO_VOLUME`, and the tests assert 0 <= kappa <= 1.

## Intra-group geometry (multi-item groups only)

| feature | n | Spearman rho | permutation p | Holm p | 95% CI | Kendall |
|---|---|---|---|---|---|---|
| `intra:transitivity:L1` | 108 | +0.828 | 0.0002 | 0.0152 | [+0.729, +0.887] | +0.632 |
| `intra:cycle_rank_normalised:L1` | 108 | +0.449 | 0.0002 | 0.0152 | [+0.211, +0.673] | +0.416 |
| `intra:largest_component_fraction:L1` | 108 | +0.573 | 0.0002 | 0.0152 | [+0.353, +0.755] | +0.477 |
| `intra:conductance:L1` | 107 | -0.604 | 0.0002 | 0.0152 | [-0.784, -0.403] | -0.522 |
| `intra:internal_edges:L1` | 108 | +0.155 | 0.1162 | 1.0000 | [-0.092, +0.393] | +0.156 |
| `intra:boundary_edges:L1` | 108 | -0.369 | 0.0004 | 0.0164 | [-0.578, -0.156] | -0.312 |
| `intra:triangles:L1` | 108 | +0.144 | 0.1316 | 1.0000 | [-0.103, +0.381] | +0.131 |
| `intra:components:L1` | 108 | -0.501 | 0.0002 | 0.0152 | [-0.680, -0.295] | -0.388 |
| `intra:transitivity:L2` | 82 | +0.814 | 0.0002 | 0.0152 | [+0.698, +0.886] | +0.625 |
| `intra:cycle_rank_normalised:L2` | 108 | +0.031 | 0.7431 | 1.0000 | [-0.197, +0.249] | +0.018 |
| `intra:largest_component_fraction:L2` | 108 | +0.465 | 0.0002 | 0.0152 | [+0.266, +0.642] | +0.343 |
| `intra:conductance:L2` | 108 | -0.164 | 0.0960 | 1.0000 | [-0.380, +0.062] | -0.136 |
| `intra:internal_edges:L2` | 108 | -0.194 | 0.0466 | 1.0000 | [-0.400, +0.018] | -0.114 |
| `intra:boundary_edges:L2` | 108 | -0.470 | 0.0002 | 0.0152 | [-0.642, -0.281] | -0.334 |
| `intra:triangles:L2` | 108 | -0.120 | 0.2260 | 1.0000 | [-0.332, +0.095] | -0.078 |
| `intra:components:L2` | 108 | -0.466 | 0.0002 | 0.0152 | [-0.644, -0.272] | -0.336 |
| `intra:transitivity:L3` | 81 | +0.583 | 0.0002 | 0.0152 | [+0.385, +0.734] | +0.419 |
| `intra:cycle_rank_normalised:L3` | 108 | +0.145 | 0.1418 | 1.0000 | [-0.077, +0.362] | +0.092 |
| `intra:largest_component_fraction:L3` | 108 | +0.724 | 0.0002 | 0.0152 | [+0.586, +0.829] | +0.544 |
| `intra:conductance:L3` | 108 | -0.406 | 0.0004 | 0.0164 | [-0.581, -0.210] | -0.294 |
| `intra:internal_edges:L3` | 108 | +0.081 | 0.4051 | 1.0000 | [-0.142, +0.298] | +0.068 |
| `intra:boundary_edges:L3` | 108 | -0.425 | 0.0002 | 0.0152 | [-0.588, -0.233] | -0.292 |
| `intra:triangles:L3` | 108 | -0.005 | 0.9568 | 1.0000 | [-0.230, +0.215] | -0.012 |
| `intra:components:L3` | 108 | -0.600 | 0.0002 | 0.0152 | [-0.746, -0.424] | -0.436 |

## Inter-group quotient geometry (nodes = occupied prefixes, singletons kept)

| feature | n | Spearman rho | permutation p | Holm p | 95% CI | Kendall |
|---|---|---|---|---|---|---|
| `quot:nodes:L1` | 108 | +0.132 | 0.1680 | 1.0000 | [-0.105, +0.362] | +0.082 |
| `quot:edges:L1` | 108 | +0.144 | 0.1314 | 1.0000 | [-0.089, +0.360] | +0.094 |
| `quot:components:L1` | 107 | undefined: DEGENERATE | — | — | — | — |
| `quot:largest_component_fraction:L1` | 107 | undefined: DEGENERATE | — | — | — | — |
| `quot:density:L1` | 107 | +0.133 | 0.1766 | 1.0000 | [-0.096, +0.370] | +0.104 |
| `quot:degree_median:L1` | 107 | +0.172 | 0.0780 | 1.0000 | [-0.052, +0.382] | +0.116 |
| `quot:degree_p90:L1` | 107 | +0.086 | 0.3825 | 1.0000 | [-0.149, +0.314] | +0.054 |
| `quot:degree_max:L1` | 107 | +0.015 | 0.8776 | 1.0000 | [-0.226, +0.249] | +0.009 |
| `quot:triangles:L1` | 107 | +0.161 | 0.1002 | 1.0000 | [-0.063, +0.372] | +0.106 |
| `quot:transitivity:L1` | 106 | +0.182 | 0.0630 | 1.0000 | [-0.045, +0.400] | +0.141 |
| `quot:cycle_rank_per_node:L1` | 107 | +0.172 | 0.0746 | 1.0000 | [-0.052, +0.387] | +0.115 |
| `quot:forman_mean:L1` | 107 | -0.161 | 0.0990 | 1.0000 | [-0.373, +0.064] | -0.109 |
| `quot:forman_median:L1` | 107 | -0.158 | 0.1078 | 1.0000 | [-0.369, +0.065] | -0.105 |
| `quot:forman_negative_fraction:L1` | 107 | undefined: DEGENERATE | — | — | — | — |
| `quot:singleton_nodes:L1` | 108 | -0.106 | 0.2843 | 1.0000 | [-0.311, +0.112] | -0.050 |
| `quot:nodes:L2` | 108 | +0.452 | 0.0002 | 0.0152 | [+0.260, +0.630] | +0.322 |
| `quot:edges:L2` | 108 | +0.354 | 0.0004 | 0.0164 | [+0.158, +0.542] | +0.231 |
| `quot:components:L2` | 108 | undefined: DEGENERATE | — | — | — | — |
| `quot:largest_component_fraction:L2` | 108 | undefined: DEGENERATE | — | — | — | — |
| `quot:density:L2` | 108 | -0.451 | 0.0002 | 0.0152 | [-0.631, -0.260] | -0.326 |
| `quot:degree_median:L2` | 108 | -0.267 | 0.0068 | 0.2312 | [-0.491, -0.049] | -0.211 |
| `quot:degree_p90:L2` | 108 | -0.200 | 0.0438 | 1.0000 | [-0.427, +0.007] | -0.154 |
| `quot:degree_max:L2` | 108 | +0.374 | 0.0004 | 0.0164 | [+0.174, +0.556] | +0.274 |
| `quot:triangles:L2` | 108 | -0.145 | 0.1370 | 1.0000 | [-0.377, +0.063] | -0.118 |
| `quot:transitivity:L2` | 108 | -0.465 | 0.0002 | 0.0152 | [-0.643, -0.274] | -0.337 |
| `quot:cycle_rank_per_node:L2` | 108 | -0.235 | 0.0142 | 0.4685 | [-0.458, -0.019] | -0.180 |
| `quot:forman_mean:L2` | 108 | +0.170 | 0.0800 | 1.0000 | [-0.039, +0.399] | +0.123 |
| `quot:forman_median:L2` | 108 | +0.203 | 0.0366 | 1.0000 | [-0.007, +0.428] | +0.159 |
| `quot:forman_negative_fraction:L2` | 108 | -0.211 | 0.0266 | 0.8244 | [-0.384, -0.017] | -0.128 |
| `quot:singleton_nodes:L2` | 108 | +0.419 | 0.0002 | 0.0152 | [+0.218, +0.615] | +0.319 |
| `quot:nodes:L3` | 108 | +0.478 | 0.0002 | 0.0152 | [+0.288, +0.646] | +0.344 |
| `quot:edges:L3` | 108 | +0.415 | 0.0002 | 0.0152 | [+0.211, +0.600] | +0.294 |
| `quot:components:L3` | 108 | undefined: DEGENERATE | — | — | — | — |
| `quot:largest_component_fraction:L3` | 108 | undefined: DEGENERATE | — | — | — | — |
| `quot:density:L3` | 108 | -0.479 | 0.0002 | 0.0152 | [-0.649, -0.292] | -0.345 |
| `quot:degree_median:L3` | 108 | -0.562 | 0.0002 | 0.0152 | [-0.721, -0.385] | -0.386 |
| `quot:degree_p90:L3` | 108 | -0.476 | 0.0002 | 0.0152 | [-0.649, -0.285] | -0.340 |
| `quot:degree_max:L3` | 108 | +0.398 | 0.0002 | 0.0152 | [+0.202, +0.581] | +0.287 |
| `quot:triangles:L3` | 108 | -0.420 | 0.0002 | 0.0152 | [-0.609, -0.226] | -0.299 |
| `quot:transitivity:L3` | 108 | -0.438 | 0.0002 | 0.0152 | [-0.615, -0.247] | -0.317 |
| `quot:cycle_rank_per_node:L3` | 108 | -0.478 | 0.0002 | 0.0152 | [-0.649, -0.290] | -0.346 |
| `quot:forman_mean:L3` | 108 | +0.420 | 0.0002 | 0.0152 | [+0.217, +0.608] | +0.296 |
| `quot:forman_median:L3` | 108 | +0.462 | 0.0002 | 0.0152 | [+0.277, +0.633] | +0.323 |
| `quot:forman_negative_fraction:L3` | 108 | -0.218 | 0.0220 | 0.7039 | [-0.408, -0.002] | -0.129 |
| `quot:singleton_nodes:L3` | 108 | +0.485 | 0.0002 | 0.0152 | [+0.297, +0.652] | +0.353 |

## Graph curvature (Ollivier-Ricci sampled; Forman is in the inter table)

| feature | n | Spearman rho | permutation p | Holm p | 95% CI | Kendall |
|---|---|---|---|---|---|---|
| `orc:mean:L1` | 107 | +0.112 | 0.2507 | 1.0000 | [-0.113, +0.349] | +0.095 |
| `orc:median:L1` | 107 | +0.150 | 0.1284 | 1.0000 | [-0.077, +0.385] | +0.117 |
| `orc:sd:L1` | 107 | -0.118 | 0.2318 | 1.0000 | [-0.360, +0.108] | -0.097 |
| `orc:negative_fraction:L1` | 107 | undefined: DEGENERATE | — | — | — | — |
| `orc:edge_coverage:L1` | 108 | -0.093 | 0.3335 | 1.0000 | [-0.313, +0.136] | -0.059 |
| `orc:mean:L2` | 108 | -0.451 | 0.0002 | 0.0152 | [-0.633, -0.257] | -0.332 |
| `orc:median:L2` | 108 | -0.448 | 0.0002 | 0.0152 | [-0.630, -0.254] | -0.327 |
| `orc:sd:L2` | 108 | +0.269 | 0.0048 | 0.1680 | [+0.072, +0.459] | +0.199 |
| `orc:negative_fraction:L2` | 108 | +0.441 | 0.0002 | 0.0152 | [+0.246, +0.622] | +0.318 |
| `orc:edge_coverage:L2` | 108 | -0.354 | 0.0004 | 0.0164 | [-0.542, -0.158] | -0.231 |
| `orc:mean:L3` | 108 | -0.420 | 0.0002 | 0.0152 | [-0.598, -0.227] | -0.293 |
| `orc:median:L3` | 108 | -0.442 | 0.0002 | 0.0152 | [-0.611, -0.261] | -0.313 |
| `orc:sd:L3` | 108 | -0.052 | 0.5979 | 1.0000 | [-0.271, +0.167] | -0.031 |
| `orc:negative_fraction:L3` | 108 | +0.335 | 0.0006 | 0.0216 | [+0.135, +0.513] | +0.211 |
| `orc:edge_coverage:L3` | 108 | -0.415 | 0.0002 | 0.0152 | [-0.600, -0.211] | -0.294 |

## Coverage and exactness

* Intra: exact for every group; groups with >= 2 items only enter the averages. Singleton exclusion counts are in `features_by_level.csv`.
* Triangles: exact sparse; `BLOCKED_NOT_FULL_CENSUS` only above the declared nnz bound exact sparse; BLOCKED above nnz 200000000.
* Quotient: complete (no sampling); Forman on **every** quotient edge.
* Ollivier-Ricci: `SAMPLED_ESTIMATE` on SAMPLED_ESTIMATE with 512 edges; the measured edge coverage per row is `orc_edge_coverage_l*` and the edge counts are `orc_edges_measured_l*` / `orc_edges_total_l*`. All-edge ORC is not claimed.
* Gromov: exact on components <= 24, else BLOCKED; the quotient is one component larger than the exact limit, so the quotient Gromov column is `BLOCKED_NOT_FULL_CENSUS` and is not interpreted.

## Performance

Per-SID wall time: median 9.97 s, max 21.8 s; phase totals {'intra': 111.9, 'quotient': 240.0, 'orc': 735.2}. The exact sparse triangle path replaces the dense cube; `bench.json` records the same-input before/after measurement.

## Verdict

**Corrected and measured.** The intra-group geometry now uses binary units, excludes singletons from the averages and counts triangles exactly; the quotient geometry is complete at all three levels with Forman curvature on every edge.

**Raw associations.** See the three tables above; every sign is reported, including null and `TOO_FEW` cases, and the Holm column is over the whole predeclared family rather than a chosen subset.

**Not claimed.** Ollivier-Ricci is a sampled estimate (coverage per row); the quotient Gromov deviation is blocked; and an observed correlation is not evidence that graph geometry causes Recall. Model hyperbolic curvature `-c_l` is a separate object and is not trained or selected here.
