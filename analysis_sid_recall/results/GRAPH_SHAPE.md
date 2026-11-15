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

`closure_regions.py` -> `closure_regions.json`.  Correction (2026-11-09): the
first version of this section compared a high-clustering selection (mean degree
5.3) against random nodes of mean degree 19.2 -- the "degree matched" sampler was
drawing the wanted degrees from the whole graph.  The table below is the fixed
version: the control now reproduces the selection's degree multiset exactly
(5.25 vs 5.25, 7.83 vs 7.83, 11.35 vs 11.35), with 20 replicates per share.
The earlier claim that the high-closure nodes are *more dispersed* than random
does not survive that fix.

Closure itself is far beyond a degree-preserving null (5 rewirings, mean +- sd):

| | real graph | rewired (5 runs) | z |
|---|---|---|---|
| triangles | 367,434 | 168,485 +- 905 | ~220 |
| mean clustering | 0.0390 | 0.0232 +- 0.0002 | ~79 |
| share of nodes with C >= 0.3 | 0.71% | 0.40% +- 0.03% | ~10 |

Triangles are contiguous, not scattered: 2,414 patches share edges, but one
patch holds **99.2%** of all 367,434 triangles and only four patches carry 10 or
more; the other 2,410 patches are isolated single triangles.

Do the high-closure nodes sit inside a particular patch or region?  After the
degree fix the answer is "weakly, if at all":

| selection | inside edges | density z | neighbour-share z | hops z (negative = closer) | top-3 communities (matched) |
|---|---|---|---|---|---|
| top 1% (244) | 3 | +0.08 | +0.5 | **-6.1** (2.10 vs 2.51) | 0.83 (0.69) |
| top 5% (1223) | 83 | -2.4 | -2.1 | +0.6 | 0.81 (0.73) |
| top 10% (2447) | 846 | +0.15 | -4.1 | +1.9 | 0.79 (0.76) |

So: at the top 1% the nodes are closer to each other than degree-matched random
nodes and fall into fewer communities, but they still share almost no edges
(3 of 244); at 5% and 10% those effects shrink or reverse sign, and the
neighbour-share is consistently *below* the degree-matched control.  The honest
reading is that **triangle closure is real, statistically overwhelming, and
essentially one contiguous mass; the nodes with the highest local clustering are
not a separate dense zone beside a tree zone**, and this analysis does not by
itself support or refute the heterogeneous-manifold hypothesis.
