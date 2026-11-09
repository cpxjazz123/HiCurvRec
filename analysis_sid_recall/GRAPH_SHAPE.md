# The train behaviour graph is tree-like and community-like at once

`graph_shape.py` reads the TRAIN behaviour graph: every consecutive item pair
inside a user history, deduplicated (a simple graph cannot hold the same edge
twice) and **not sampled** -- all 264,326 edges are used.  One run,
`graph_shape.json`:

| reading | value |
|---|---|
| nodes / edges | 24,474 / 264,326 |
| mean / median degree | 21.6 / 11 |
| clustering mean / median | 0.0390 / 0.0278 |
| clustering p90 / p99 | 0.0948 / 0.2222 |
| clustering kurtosis | 77.3 |
| nodes with clustering 0 (including the 210 of degree < 2) | **32.5%** |
| nodes with 0 < C < 0.1 | 59.2% |
| nodes with 0.1 <= C < 0.3 | 8.1% |
| nodes with C >= 0.3 | **0.71%** |
| edges in no triangle | **46.0%** |
| edges in >= 5 triangles | 19.1% |
| max k-core / share core >= 5 / >= 10 | 40 / 89.1% / 43.9% |
| triangles held by the top 1% of nodes | **60.6%** |
| clustering of hubs (deg >= 50) vs leaves (deg <= 3) | 0.0451 vs 0.0323 |
| share of nodes in the largest component | 1.00 |

Clustering is the usual local coefficient with degree < 2 counted as 0: a node
with a single neighbour cannot close a triangle, so dropping it would quietly
report the mean over only the nodes that could.  (210 nodes, 0.86%, are in that
bucket.)  Under that convention a third of the nodes have no neighbour pair that
knows each other and 46% of the
edges sit in no triangle: that is the tree-like branch structure.  At the same
time 0.7% of the nodes carry a clustering of 0.3 or more and the top 1% of nodes
hold 60% of all triangles: that is the dense community structure.  Clustering on
hubs is nearly the same as on leaves, so the density is not a property of degree
alone -- it lives inside local groups, and both shapes coexist in one connected
component (a core-periphery graph).

Consequence for Stage 2: a single curvature cannot fit both regimes at once,
which is why the intra-versus-inter Forman-Ricci separation, not a global
curvature, is what tracks recall (see `report.md`).

## Do the high-closure nodes form a region?

`closure_regions.py` -> `closure_regions.json` takes the top 1%, 5% and 10% of
nodes by local clustering and asks whether they sit together, against two nulls:
degree-matched random nodes, and a degree-preserving rewiring of the same graph.

Closure itself is real -- it is not a degree artefact:

| | real graph | degree-preserving rewiring |
|---|---|---|
| triangles | 367,434 | 170,855 (x0.47) |
| mean clustering | 0.0390 | 0.0231 (x0.59) |
| share of nodes with C >= 0.3 | 0.71% | 0.38% (x0.53) |

But the high-closure nodes do **not** form a region:

| selection | inside edges | density vs degree-matched | neighbour share vs matched | hops to nearest selected (matched) |
|---|---|---|---|---|
| top 1% (244) | 3 | 0.11x | 0.006 vs 0.011 (0.58x) | 2.08 (1.89) |
| top 5% (1223) | 83 | 0.11x | 0.018 vs 0.051 (0.35x) | 1.90 (1.51) |
| top 10% (2447) | 846 | 0.32x | 0.056 vs 0.101 (0.55x) | 1.59 (1.33) |

Their spread across communities is the same as the degree-matched sample (0.79-
0.83 of the selection in the top three communities, matched 0.75-0.78), and they
are, if anything, slightly **farther** apart than random nodes of the same
degree.  So the honest statement is: **triangle closure is real and concentrated
on a minority of nodes, but those nodes are not a contiguous high-closure
region.**  They are scattered, each sitting in its own small neighbourhood; that
does not by itself support a picture of one dense zone next to one tree zone.
