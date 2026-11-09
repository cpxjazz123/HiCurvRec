# The train behaviour graph is tree-like and community-like at once

`graph_shape.py` reads the TRAIN behaviour graph (an edge for every consecutive
item pair inside a user history, deduplicated, sampled to 400k edges at seed 42)
and reports on one run, `graph_shape.json`:

| reading | value |
|---|---|
| nodes / edges | 24,474 / 264,326 |
| mean / median degree | 21.6 / 11 |
| clustering mean / median | 0.039 / 0.028 |
| clustering p90 / p99 | 0.095 / 0.222 |
| clustering kurtosis | 77.1 |
| nodes with clustering 0 | **32.0%** |
| nodes with 0 < C < 0.1 | 59.2% |
| nodes with 0.1 <= C < 0.3 | 8.1% |
| nodes with C >= 0.3 | **0.7%** |
| edges in no triangle | **46.0%** |
| edges in >= 5 triangles | 19.1% |
| max k-core / share core >= 5 / >= 10 | 40 / 89.1% / 43.9% |
| triangles held by the top 1% of nodes | **60.6%** |
| clustering of hubs (deg >= 50) vs leaves (deg <= 3) | 0.045 vs 0.038 |
| share of nodes in the largest component | 1.00 |

A third of the nodes have no neighbour pair that knows each other and 46% of the
edges sit in no triangle: that is the tree-like branch structure.  At the same
time 0.7% of the nodes carry a clustering of 0.3 or more and the top 1% of nodes
hold 60% of all triangles: that is the dense community structure.  Clustering on
hubs is nearly the same as on leaves, so the density is not a property of degree
alone -- it lives inside local groups, and both shapes coexist in one connected
component (a core-periphery graph).

Consequence for Stage 2: a single curvature cannot fit both regimes at once,
which is why the intra-versus-inter Forman-Ricci separation, not a global
curvature, is what tracks recall (see `report.md`).
