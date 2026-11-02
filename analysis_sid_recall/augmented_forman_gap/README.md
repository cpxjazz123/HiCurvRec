# AF3 curvature and the original-graph within/cross edge gap (Issue #136)

On the same 108 content-hash-matched SID–Stage3 pairs as #134 (two digest
mismatches excluded), measure the **frozen triangle-only augmented Forman
curvature** and the **original-graph within-vs-cross-prefix curvature gap**, and
correlate them (raw Pearson **and** raw Spearman) with R@5, R@10, NDCG@5 and
NDCG@10 at L1/L2/L3. Offline measurement only: no Stage2/3 change, no ORC
recomputation, no @20 synthesis.

## Frozen definitions (`definitions.json`)

On a binary, unweighted, simple graph `H`, for every undirected edge `e=(u,v)`:

```
F_H(e)   = 4 - deg_H(u) - deg_H(v)
T_H(e)   = |N_H(u) ∩ N_H(v)|
AF3_H(e) = F_H(e) + 3*T_H(e)
```

Degrees and triangles are always taken **in the graph the edge belongs to**
(induced internal degree, quotient degree, or the original item graph). AF3 is a
chosen graph-curvature convention; it is **not** the hyperbolic sectional
curvature `-c_l`. Square/4-cycle terms, weights, normalisations and model
curvatures are out of scope.

`Gap_AF3(l) = mean(AF3_G(e), e ∈ E_in(l)) − mean(AF3_G(e), e ∈ E_cross(l))` on
the **same** original edges, using one cached global curvature array; per-SID
variation comes only from prefix membership. Edges touching an item without a
hard prefix are counted separately and never classified as cross; an empty class
is `UNDEFINED_INSUFFICIENT_*_EDGES`, never zero.

## Files

| file | contents |
|---|---|
| `definitions.json` | the frozen convention, gap, nulls and the NOT_EVALUATED list |
| `af3.py` | edge arrays, exact `T_e` from `(A^2)⊙A`, F/AF3, induced-group aggregation, gap, label-shuffle nulls |
| `run_measure.py` | one global curvature cache, then per SID/level intra/quotient/gap/null features with checkpoint/resume |
| `run_stats.py` | raw Pearson and Spearman for every metric × level × outcome × type, permutation p (20k), bootstrap CI, Holm over the full family, cohorts, redundancy, REPORT |
| `features_by_level.csv|json` | per SID and level: intra AF3/Forman aggregates, quotient AF3/Forman/density/transitivity, gap components, retention, null z |
| `support.csv` | groups, eligible/singleton/no-edge groups, intra edges, quotient nodes/edges, within/cross/unlabelled edge counts |
| `correlations_pearson_spearman.csv` | every test with n, coverage, both coefficients, both p-values, Holm, both CIs, status |
| `metric_redundancy.csv` | AF3 vs transitivity/conductance/density, gap vs edge retention, gap vs #134 ORC |
| `tests/test_af3.py` | 17 oracles: P4, K3 (F=0, AF3=+3), K4 (F=-2, AF3=+4), two triangles + bridge, star, isolated nodes, one-child parent, all-singleton labels, empty within/cross classes, `AF3-F=3T`, single-count edges, frozen provenance |

## Reproduce

```bash
python3 analysis_sid_recall/augmented_forman_gap/run_measure.py --resume
python3 analysis_sid_recall/augmented_forman_gap/run_stats.py
python3 -m unittest discover -s analysis_sid_recall/augmented_forman_gap/tests
```

## Blocked / not claimed

* R@20 and NDCG@20: `NOT_EVALUATED` (the saved artifacts only contain topk 5/10).
* D3 / prefix alignment: `NOT_AVAILABLE` (no such implementation exists in the
  repository); hard prefix edge retention `E_in/(E_in+E_cross)` is used as the
  explicitly named comparison.
* AF3-vs-ORC on the identical sampled edges: `NOT_AVAILABLE_BY_DESIGN` (#134
  stores SID-level ORC aggregates only). The SID-level comparison is labelled
  `SAMPLED_ESTIMATE`.
* Gromov stays `BLOCKED_NOT_FULL_CENSUS`; no model-training claim; the gap is not
  deduced from `mean intra − mean quotient`.
