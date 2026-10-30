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
