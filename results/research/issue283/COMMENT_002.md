**C1/C2 node — full triplet classification verified**

- Hand tests pass (`tests/test_joint283.py`, 7 deterministic cases): the equal-marginal example
  D1={ALL,ALL,NONE,NONE} vs D2={ALL,AM,MB,AB} has p_am=p_mb=p_ab=0.5 in both but ALL 0.5 vs 0.25;
  partition sum=1, cross-layer refinement only, codeword-rename invariance, LCP feasibility
  (22 feasible combos), type-classification hand cases, null-rate hand case `N=6, groups [3,3]`
  (pair 0.4, triple 0.1).
- Full asset: 13 shards = **28,391,948 rows** (matches the manifest), all `valid`; every row's 6-bit
  directed code recomputed from `assets/train/visible_forward.npz` (`ac7155…`) reproduces the stored
  `structure_class` under `canonical_code` (**0 mismatches**), and every row maps to one of 36 role
  classes (no infeasible code).
- Role classes (36) with per-row Orbit ids: chain 6,347,269 + revchain 5,795,800 + chain_both/
  revchain_both 55,380; fork 6,063,454 + merge 5,940,844 + wedge variants 291,854;
  triangle family 296,841; reciprocal family 3,548,506. Manifest: `rows/ROWS_MANIFEST.json`
  (per-shard sha256, class counts, orbit counts, definition sha256).
- Independent validation context: each relation covers ~19.1-20.0k of 57,439 rows (~140k instances);
  pair-uniform weighting frozen. Diagnostic running.

So the full graphlet+Orbit traversal is complete and no undirected wedge is treated as a directed
chain — chains require the actual `a→m→b` edges.
