# Corrected intra-group and new inter-group geometry (Issue #133)

Offline, full-census measurement of (a) the **intra-group** geometry of every
multi-item SID prefix group, (b) the **inter-group** quotient graph of the
occupied prefixes, and (c) **graph curvature** on the quotient edges, each with a
raw Spearman comparison against the frozen Stage 3 Recall@10.

## What is corrected relative to #132

1. **Singletons**: every occupied group is still counted, and singleton prefixes
   remain nodes of the inter-group quotient, but they no longer enter the
   intra-group averages (their largest-component fraction 1, cycle rank 0 and
   numeric conductance were contaminating L2/L3).
2. **Triangles**: exact sparse `(A @ A).multiply(A).sum() // 6` with an `int32`
   accumulator. The old dense `int8` cube overflows: on a real 1,000-item group
   it reports 481 triangles where the exact answer is 524 (`bench.json`), and it
   is ~800x slower. A 15-clique returns 455 triangles and transitivity 1 in the
   tests.
3. **Conductance**: binary cut `sum(deg_B(S)) - 2E_B(S)` over binary volumes,
   asserted to lie in [0, 1]; a zero complement volume is
   `UNDEFINED_ZERO_VOLUME`. The old weighted-degree/unweighted-cut mix produced
   values above 1 (and up to 1e18).

## What is new

* **Quotient graph Q_l** at all three levels: one node per occupied prefix
  (singletons included), an undirected edge iff at least one source item edge
  crosses the prefixes, no self-loops, unweighted. Complete metrics: nodes,
  edges, components, largest-component share, density, degree quantiles,
  triangles/transitivity, cycle rank per node, and the combinatorial **Forman
  curvature** `4 - deg(u) - deg(v)` on **every** quotient edge.
* **Ollivier–Ricci curvature** on a declared sampled edge set (512 edges) with
  the exact 0/1/2-hop transport (validated max-flow identity), reported as
  `SAMPLED_ESTIMATE` with its edge coverage — never as a full statistic.
* **Raw Spearman** (rho, permutation p, bootstrap 95% CI, Holm over the
  predeclared family of all features × levels, n) for every measured feature;
  partial correlations are secondary only.

## Files

| file | contents |
|---|---|
| `geom.py` | binary graph, exact sparse triangles, binary conductance, quotient construction/metrics, Forman, sampled ORC, quotient Gromov status |
| `run_measure.py` | full run with per-SID checkpoint and an idempotent support-table checkpoint |
| `run_stats.py` | raw Spearman family, secondary partials, the #132 comparison, figures, REPORT |
| `bench.py` | same-input dense-vs-sparse triangle benchmark with the overflow detection |
| `features_by_level.csv|json` | per (SID, level) intra/inter/curvature features |
| `support_L1_L2_L3.csv` | census accounting per SID and level |
| `timing.json`, `completeness.json`, `bench.json` | performance, exactness/coverage ledger, benchmark |
| `statistics.json`, `figure_audit.json`, `REPORT.md` | results and the write-up |
| `tests/test_geom.py` | 17 contract tests (singleton exclusion/retention, 15-clique, 2,500-node sparse group, 301+ prefixes, disconnected, full-graph zero volume, quotient triangle/Forman, ORC vs exact OT, Gromov statuses, resume, figure honesty) |

## Reproduce

```bash
python3 analysis_sid_recall/intra_inter_geometry/run_measure.py --resume
python3 analysis_sid_recall/intra_inter_geometry/run_stats.py
python3 analysis_sid_recall/intra_inter_geometry/bench.py
python3 -m unittest discover -s analysis_sid_recall/intra_inter_geometry/tests
```

## Exactness ledger

`completeness.json` states it for the run: intra metrics exact for every group;
triangles exact up to the declared nnz bound; the quotient complete; Forman on
every edge; ORC sampled with per-row coverage; quotient Gromov blocked when the
component exceeds the exact limit. No sampling is ever reported as a full-census
value.
