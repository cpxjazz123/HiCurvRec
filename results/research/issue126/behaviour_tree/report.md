# Stage one: is the behaviour graph a tree metric?

**Verdict: insufficient evidence.** The behaviour graph's shortest-path metric IS
tree-like, and a degree-matched random graph is exactly as tree-like, so the
tree-likeness is a property of shortest paths on a sparse graph and not evidence
that the interactions carry a hierarchy.

## What was measured

The graph is built from the **sequential** interactions of the training split
only: a user's sequence A, B, C, D contributes the edges A-B, B-C and C-D, never
A-C. The edges have unit length, the graph is symmetric, and nothing is
truncated (no top-k, no text, no teacher, no RQ-VAE output). Quadruples are
drawn from the largest connected component and the six shortest-path distances
give `delta = (S_max - S_mid) / 2`, which is zero exactly for a tree metric.

The manifest records the data hash, the construction rules, the node and edge
counts, the component structure, the sampled diameter and the distance
distribution: `graph_manifest.json`.

| | value |
| --- | --- |
| training sequences | 339,519 |
| edges | 630,782 |
| items in the largest component | 24,556 of 24,587 (99.87%) |
| components | 32 |
| sampled diameter | 42 |

## The three sampling schemes (`four_point_results.csv`)

| scheme | quadruples | connected | median delta | p90 | mean | delta = 0 | delta <= 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global random | 50,000 | 50,000 | 0.0 | 0.5 | 0.2530 | 53.27% | 99.93% |
| locally associated | 25,000 | 25,000 | 0.0 | 0.5 | 0.2971 | 52.04% | 98.73% |
| popularity stratified | 25,000 | 25,000 | 0.0 | 0.5 | 0.2543 | 52.98% | 99.94% |

Every scheme agrees: the graph's shortest-path metric satisfies the four-point
condition exactly for about half of the quadruples and within delta = 1 for
almost all of them. The normalised median (diameter 42) is 0.0.

## The control that decides it (`rewiring_controls.csv`)

Ten degree-preserving rewirings of the same graph, by the double-edge swap, the
weights travelling with the edges, each scored on the SAME 20,000 quadruples.
Every seed keeps the degree sequence, the edge count and the weight
distribution; the successful swap counts are 37,665 of 40,000 per seed and the
largest component stays at 24,556 items.

| statistic | the real graph | the ten rewirings (mean +- sd) | difference |
| --- | --- | --- | --- |
| median delta | 0.0 | 0.0 +- 0.0 | 0.0 |
| mean delta | 0.25257 | 0.25211 +- 0.00183 | +0.00046 |
| p90 delta | 0.5 | 0.5 +- 0.0 | 0.0 |
| share delta = 0 | 0.53285 | 0.53216 +- 0.00367 | +0.00069 |
| share delta <= 1 | 0.99915 | 0.99945 +- 0.00010 | -0.00030 |

The effect size is an order of magnitude smaller than the spread between the
random graphs, so the real graph and its rewirings are indistinguishable. The
paired bootstrap over the quadruples gives a difference of 0.0 with a confidence
interval of [0.0, 0.0]: the median is degenerate (both are 0) and every finer
statistic separates the two by less than the seed-to-seed spread.

**Reading it.** The graph is a union of user paths, so it is close to a forest,
and the shortest-path metric of a sparse graph is close to a tree metric. The
rewiring keeps the degrees and the sparsity and loses the interaction
structure, and the metric stays exactly as tree-like. The four-point condition
therefore does not distinguish the behaviour structure from a degree-matched
random graph, and it cannot support the hyperbolic motivation on this data.

## What this does and does not say

* It does **not** say the behaviour data has no structure: the metrics of this
  project that DO predict Recall (+0.7104 for the plain distance contrast,
  +0.7170 for its Euclidean control, +0.6593 for the unweighted four-point
  agreement) are all behaviour-contrast results.
* It does say that the specific evidence the hyperbolic motivation would want --
  that the behaviour geometry is a tree that a hyperbolic space can reproduce --
  is **not** there. A sparse graph's shortest-path metric looks tree-like
  whether or not the interactions carry a hierarchy.
* The honest next step is RQ2 (does the tree-likeness, such as it is, survive in
  the text embedding and in a behaviour embedding?) with the same controls, and
  then the matched Euclidean/hyperbolic training comparison. Both are worth
  running, but the first layer of the evidence chain is missing.

## Reproducing it

```bash
python -m pytest results/research/issue126/behaviour_tree/tests/test_behaviour_tree.py
python results/research/issue126/behaviour_tree/run_stage1.py
python results/research/issue126/behaviour_tree/run_controls.py
```

The tests are the hard requirement of the brief: a path and a star must give
delta = 0, a four-cycle must give delta = 1, the shortest-path reads must use
the item IDs and not a position in the sampled array, the rewiring must keep
the degrees and the weights, and the same seed must reproduce the same graph.
