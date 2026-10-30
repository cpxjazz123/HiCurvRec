# The SVD-guided hyperbolic SID consistency: the protocol, fixed before the numbers

Everything below is decided BEFORE any correlation with Recall is computed.

## Why this route

The previous experiment asked a hyperbolic model whether its own geometry was
tree-like, with the reference's curvature-1 ball as the data. The behaviour
reference here comes from the RAW interactions instead: an SVD of the training
user-item matrix gives every item a behaviour embedding, and the item graph built
from those embeddings gives the tree-likeness of the REAL purchase behaviour. The
curvature is then chosen so that a hyperbolic space reproduces THAT structure.

## Step 1 -- the SVD behaviour embedding

* the matrix: the training interactions only (`train.parquet`), users x items,
  binary;
* the embedding: a truncated SVD of rank **64**, centred, with the fixed seed 42;
* the items: every item of the catalogue, the zero rows (items that never occur)
  kept as the zero vector.

## Step 2 -- the behaviour graph and its distances

* the graph: each item is joined to its **10** nearest neighbours in the SVD space
  by cosine similarity, the edge weight being `1 - cosine`;
* the distances: the shortest-path distances in that graph, from the sources the
  sampled quadruples need, with a hop limit of **4**;
* the data's four-point delta: the standard `(S_max - S_second) / 2` over the
  three pairing sums of those graph distances.

The graph, not the SVD distances, is the reference: a shortest-path distance is a
tree-like distance by construction, so the curvature is not attracted to the flat
limit the way it is when the target is a Euclidean cloud.

## Step 3 -- the curvature

* the grid: `c in {0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8}`;
* the model at curvature c: the SAME SVD vectors mapped into the ball of that
  curvature, `x = exp_c(t) = tanh(sqrt(c)|t|) t / (sqrt(c)|t|)`, so the mapping is
  the identity as c goes to zero and no curvature is favoured by the mapping;
* the scale-free delta: `delta_4 / m`, with `m` the median pairwise distance of
  the quadruple. The delta depends on the overall scale of the distances, so the
  comparison is made on the normalised quantity;
* the criterion: minimise `|median normalised delta_model(c) - median normalised
  delta_graph|`;
* the tie-break: the smaller curvature;
* the sample: **4000** quadruples for the estimation (the graph distances are the
  expensive part), drawn with the seed 42.

`c_delta` is computed ONCE, from the interactions alone. No recall enters it.

## Step 4 -- the metric

For every SID table, over **20000** quadruples (the same for every table):

* the model's three pairing sums in the ball of curvature `c_delta` (or the
  Euclidean distances of the SVD space for the control);
* the SID tree's three pairing sums, in PATH lengths (the codes from the first
  mismatch onwards);
* the preferred pairing of each: the one with the smallest sum;
* the weight: `1 / (1 + |normalised delta| / m)`, with `m` the median normalised
  delta at `c_delta`, computed once and shared by every table and every variant;
* the quadruples whose three tree sums are equal carry no information and are
  left out of both the numerator and the denominator;
* the score: the weighted agreement, in [0, 1].

## Step 5 -- the controls

| variant | geometry |
|---|---|
| A | `c_delta` (the hypothesis) |
| B | `0.5 c_delta` |
| C | `2 c_delta` |
| D | nearly flat, `c = 0.001` |
| E | the Euclidean SVD space |
| F | the item-SID correspondence permuted (seed 42) |

## Step 6 -- the decision

The Spearman of each variant with Recall@10 on the **107** confirmed pairs, with
its bootstrap interval and its p-value. The claim that would be supported is: the
curvature chosen from the BEHAVIOUR's own tree-likeness makes the SID's
hierarchical consistency predict the recall better than the Euclidean control and
better than the other curvatures.

The Gromov delta is not computed on the SID prefix tree itself: a tree's delta is
zero by definition, so it cannot separate the tables.

## v3 -- what the first execution showed, and what had to be corrected

The first execution reported a four-point obligation of +0.3656 for this family.
Re-running it against the protocol's own wording exposed four defects, all fixed
in `metrics_four_point_svd.py`:

1. `scipy.sparse.csgraph.dijkstra(..., limit=4)` bounds the weighted path
   LENGTH, not the number of hops. The protocol says "a hop limit of 4". A true
   hop-bounded shortest path is now a min-plus relaxation over the edges.
2. The k-NN graph was built with the out-edges only, so two neighbours of a
   source could not reach each other and almost every quadruple was dropped. The
   behaviour graph is undirected now.
3. The min-plus relaxation was written as a sparse matrix product, which SUMS
   the incoming contributions and is not a shortest path at all; the write-back
   also went through a fancy-indexed copy. Both are fixed.
4. The graph is fragmented: at k=10 a random partner is reachable within four
   hops in only about 4% of the pairs, and the fraction FALLS with a denser
   graph because the extra edges stay inside the cluster. Random quadruples over
   the whole catalogue therefore cannot be scored against four-hop graph
   distances. The quadruples are drawn where the structure exists: a source node
   and three of its direct neighbours, so every pair has a finite distance.

With those corrections the family is much stronger on the 107 confirmed pairs,
and the curvature claim is refuted:

| variant | rho | 95% CI |
| --- | --- | --- |
| `svd_delta_four_point` (c_delta = 0.05) | +0.5777 | [+0.419, +0.710] |
| half (c = 0.025) | +0.5478 | [+0.382, +0.688] |
| double (c = 0.1) | +0.5653 | [+0.406, +0.699] |
| nearly flat (c = 0.001) | +0.5955 | [+0.437, +0.727] |
| Euclidean control | +0.5869 | [+0.424, +0.723] |
| permuted control | +0.0150 | [-0.173, +0.211] |

Paired bootstrap: c_delta - flat = -0.0175 [-0.064, +0.025] (P(not better) 0.78);
c_delta - Euclidean = -0.0086 [-0.065, +0.048] (P 0.61).

**The selection criterion is degenerate, structurally.** The behaviour graph's
median normalised delta is 0.5387, while the hyperbolic model reaches at most
0.0597 at any curvature on the grid: the behaviour graph is far LESS tree-like
than any hyperbolic embedding of the same points. "Match the median delta" can
only be satisfied at the grid's flattest end, which is why the estimate sits at
c = 0.05, the smallest grid value, with an unattainable gap of 0.479. A flat or
even Euclidean control is therefore as good or better, and the curvature that
delta selects is not the curvature that ranks SIDs.

The positive result stands and is stronger than before: the four-point
obligation of the BEHAVIOUR graph, computed on the SVD space, predicts
Recall@10 at rho = 0.5777 on the 107 confirmed pairs, and the permuted control
is at 0.0150. The curvature-selection part of the protocol is what fails, not the
metric.
