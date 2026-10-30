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

## v4 -- the reference distances had to be local, and then the curvature works

v3 recorded that the curvature selection was degenerate. That conclusion was an
artefact of the graph-side distances, not of the criterion. Four further defects
were found and fixed:

5. A single global relaxation over the ~15,000 sources of a 4,000-quadruple
   sample silently dropped almost every quadruple (18 of 4,000 survived), and a
   hand-written min-plus loop does not scale. The reference is now exact and
   local: each quadruple builds the induced subgraph of its four items plus their
   neighbours and takes one small Dijkstra on it. Every pair of a
   (source + three direct neighbours) quadruple is at most two hops apart and the
   middle node of a two-hop path is a neighbour of one of the four items, so the
   local distance EQUALS the graph distance. 4,000 quadruples take 1.1 s.
6. The grid stopped at c = 8, where the criterion landed, so a boundary could
   masquerade as an optimum. It now runs to c = 64 and the selection is interior.

With the reference computed this way the graph's median normalised delta is
0.00103 -- the local quadruples of the behaviour graph are almost exactly a tree
-- and the level criterion selects c = 8.0. The curvature it selects does carry
information, on the 107 confirmed SID+recall pairs:

| variant | rho | 95% CI |
| --- | --- | --- |
| `svd_delta_four_point` (c_delta = 8.0) | +0.6605 | [+0.519, +0.774] |
| half (c = 4.0) | +0.6605 | [+0.519, +0.774] |
| double (c = 16.0) | +0.6605 | [+0.519, +0.774] |
| shape-selected c = 2.0 | +0.6383 | [+0.495, +0.758] |
| nearly flat (c = 0.001) | +0.5318 | [+0.366, +0.670] |
| Euclidean control | +0.5037 | [+0.334, +0.646] |
| permuted control | +0.0011 | [-0.182, +0.194] |

Paired bootstrap over the 107 runs: c_delta - flat = +0.1284
[+0.0717, +0.1898], P(not better) < 1e-4; c_delta - Euclidean = +0.1565
[+0.0833, +0.2383], P < 1e-4; c_delta - shape-selected = +0.0218
[+0.0021, +0.0447], P = 0.016; c_delta - half = -0.0016 [-0.0182, +0.0162].

So the four-point obligation of the behaviour graph estimates a curvature that
RANKS SIDs better than both a near-flat and a Euclidean control, the optimum is
interior (c = 8 with the grid running to 64) and broad (4 and 16 tie), and the
permuted control sits at zero. The estimator is admissible as a data-side
constant: it is computed from the training behaviour alone and never sees recall.

The profile-shape comparison (delta against distance scale, bucket by quantile on
each side) is the more principled criterion and selects c = 2.0, but it ranks
slightly worse (+0.6383, paired -0.0218, P = 0.016): with the local reference the
level itself is meaningful, so the shape correction adds nothing here.

## v5 -- the plateau above c = 8 is saturation, and the curvature is a gauge

The sweep was extended to c = 256 and each step was measured on two additional
quantities: the largest per-SID change of the metric against the previous grid
point, and the share of items that ``exp_c`` has pushed onto the ball's
boundary. Artifact: ``../curve_sweep_107.json``, script ``curve_sweep.py``.

| c | spearman | max per-SID change | items at the boundary |
| --- | --- | --- | --- |
| 0.001 | +0.6000 | -- | 0.04 |
| 0.01 | +0.5471 | 1.3e-02 | 0.15 |
| 0.05 | +0.5716 | 1.6e-02 | 0.18 |
| 0.1 | +0.5583 | 7.2e-03 | 0.20 |
| 0.25 | +0.5658 | 7.1e-03 | 0.22 |
| 0.5 | +0.5586 | 7.3e-03 | 0.25 |
| 1 | +0.5983 | 7.5e-03 | 0.31 |
| 2 | +0.6409 | 7.5e-03 | 1.00 |
| 4 | +0.6641 | 4.8e-03 | 1.00 |
| 8 | +0.6609 | 4.1e-03 | 1.00 |
| 16 | +0.6609 | **7.8e-12** | 1.00 |
| 32 | +0.6609 | **1.8e-14** | 1.00 |
| 64 | +0.6609 | **2.9e-14** | 1.00 |
| 128 | +0.6609 | **6.0e-14** | 1.00 |
| 256 | +0.6609 | **6.0e-14** | 1.00 |

Two facts settle what the plateau means.

1. **The values stop moving at c = 16**, by twelve orders of magnitude (4e-3 per
   step up to c = 8, then 8e-12, then 1e-14). Every item is already on the
   boundary from c = 2 on, and ``tanh(sqrt(c)||t||)`` has saturated: the reported
   identity of c = 4, 8 and 16 to 1e-15 is pure floating point.
2. **The curvature is a gauge, not a parameter of the data.** Scaling the
   tangents by ``1/sqrt(c)`` and using curvature ``c`` reproduces the c = 1
   configuration exactly: the metric returns bit-identical values
   (max |difference| = 0.000e+00 for c = 4, 16 and 64). Curvature and radial scale
   are one degree of freedom, so a fixed point set cannot identify a curvature;
   what it can identify is one scalar -- how far toward the boundary the points
   are compressed.

Read honestly, the sweep supports exactly two statements: **compression toward
the boundary ranks better than the near-flat limit** (c = 0.001 to c = 4 gains
0.06 in Spearman) and **the useful region is a plateau** (c = 4 and c = 8 differ
by 0.003, which is noise). It does not show that the correlation keeps rising,
and it cannot show that c = 8 is optimal: beyond c = 8 there is nothing left to
measure.

The score also carried an order dependence that this sweep exposed. Its
normalising median was cached once per process, so the value of a variant
depended on which variant ran first. It is now cached per curvature, which moves
the nearly-flat control from +0.5318 to **+0.6011**; the family is re-measured
under the fixed score:

| variant | rho | 95% CI |
| --- | --- | --- |
| c_delta = 8.0 | +0.6605 | [+0.519, +0.774] |
| nearly flat (c = 0.001) | +0.6011 | [+0.445, +0.731] |
| Euclidean | +0.5037 | [+0.334, +0.646] |
| permuted | -0.0055 | [-0.204, +0.181] |

Paired bootstrap: c_delta - flat = **+0.0591 [+0.0215, +0.1016]**, P(not better)
= 0.001; c_delta - Euclidean = +0.1565 [+0.0833, +0.2383].

## v6 -- the ingredient is the radial weight, not the curvature

v5 left two explanations standing: the hyperbolic geometry helps, or the radial
compression toward the boundary does. The compression sweep separates them with
three distance laws on the same quadruples, the same items and the same scale,
with the curvature held at 8 and the tangent scale sweeping the compression
(artifact `../compression_sweep_107.json`, script `compression_sweep.py`):

| tangent scale | median radius | hyperbolic | Euclidean | radially weighted Euclidean |
| --- | --- | --- | --- | --- |
| 0.01 | 0.033 | +0.5912 | +0.5860 | **+0.6423** |
| 0.02 | 0.066 | +0.5373 | +0.5850 | **+0.6504** |
| 0.05 | 0.155 | +0.5442 | +0.5831 | **+0.6922** |
| 0.1 | 0.260 | +0.5509 | +0.5792 | **+0.6849** |
| 0.2 | 0.338 | +0.5773 | +0.5631 | **+0.6751** |
| 0.35 | 0.353 | +0.5980 | +0.5416 | +0.6459 |
| 0.5 | 0.354 | +0.6409 | +0.5213 | +0.5829 |
| 1.0 | 0.354 (saturated) | +0.6609 | +0.4992 | +0.5419 |

The radially weighted Euclidean is the linear distance divided by the very factor
the hyperbolic distance carries, `sqrt((1-c||x||^2)(1-c||y||^2))`, with the
`arccosh` removed. Three readings:

1. **The `arccosh` never wins.** At every compression the weighted Euclidean is
   equal to or better than the hyperbolic distance; in the middle of the range it
   is better by 0.10 to 0.15 (+0.69 against +0.55). Whatever the four-point
   agreement measures here, the curvature contributes none of it.
2. **At low compression the two coincide** (+0.5912 against +0.5860 at radius
   0.033), as they must: at one radius the hyperbolic distance is a strictly
   increasing transform of the Euclidean one and the score depends on an argmin.
3. **The operating point is not even the best of its own family.** The saturated
   point (radius 0.354) gives +0.6609, while the weighted Euclidean at radius
   0.155 gives +0.6922. That value is NOT reported as a metric: the radius was
   picked by looking at the recall, which is the tuning the plan forbids. It is
   evidence about the ingredient, not a result.

**Verdict.** The +0.6605 is produced by the radial compression and the per-point
radial weight, not by the hyperbolic curvature. The admissible statement is:
*a metric that weights the linear item geometry by how deep each item sits in the
ball predicts the 107 recalls at about +0.66; the arccosh adds nothing.*

A genuine hyperbolic claim would need a property the weighted Euclidean cannot
reproduce -- the tree-likeness of a genuinely hierarchical embedding, or a
radial-angular coupling that no diagonal weight can mimic. This construction has
neither: its ingredient is a diagonal weight.

## v7 -- the leader passes the control, and its curvature does not

HBAA (+0.7623) had never been run against the permutation control that caught
every hyperbolic family in `HYPERBOLIC_FAMILIES.md`. It has now
(`metrics_hbaa.py::hbaa_permuted`, artifact
`../metric_bench_hbaa_107.json`):

| comparison | paired bootstrap | P(not better) |
| --- | --- | --- |
| hbaa - hbaa_permuted | **+0.7655 [+0.5702, +0.9705]** | 0.0000 |
| hbaa - hbaa_anchor | +0.1031 [+0.0342, +0.1840] | 0.0024 |
| hbaa - hbaa_flat (c = 1e-3) | +0.0214 [-0.0305, +0.0761] | 0.2036 |
| hbaa - hbaa_euclid | +0.0451 [-0.0089, +0.1062] | 0.0508 |
| hbaa - hbaa_size_matched | +0.2573 [+0.1421, +0.3935] | 0.0000 |

HBAA permuted is -0.0065, so the leader is not an artefact of the SID
composition: the item-SID correspondence is what makes it work, and the history
barycentre adds a significant +0.10 over using the anchor alone.

But the hyperbolic increment does not survive: against a nearly flat curvature
the leader gains +0.021 with a confidence interval that straddles zero
(P = 0.20), and against the Euclidean control +0.045 with P = 0.051. The
curvature is a gauge (v5) and the family is insensitive to it, exactly as the
seven-property round found. `hbaa_teacher_curvature` returning the identical
+0.7623 is the same statement from another direction.

**So the strongest construction in this project is a behaviour contrast, not a
hyperbolic one.** Its power is the behaviour pairs and the history barycentre;
the curvature contributes a statistically indistinguishable +0.02 to +0.05.

`hbaa_size_matched` (the prefixes shrunk to a common size) drops to +0.5049 and
`hbaa_size_matched_kept` to +0.0701, so part of the leader's value rides on the
prefix-size structure -- a caveat worth keeping in view, but the permutation
control shows that the correspondence is not merely that structure.

## v8 -- the Lorentz chart retracts the headline number

The Poincare ball and the Lorentz hyperboloid are two charts of one space, so for
a sound implementation they must return the same distance and the same metric.
They do not above c = 4, and the Lorentz chart is the one that is right. Artifact
`../lorentz_check_107.json`, script `lorentz_check.py`.

The Lorentz distance is computed from the tangent directly,
``cosh(sqrt(c) d) = cosh(r1-r2) + sinh(r1) sinh(r2) (1-cos)`` with
``r = 2 sqrt(c) ||t||`` (the ball point is ``tanh(sqrt(c)||t||)/sqrt(c)``, so the
hyperbolic radius carries the factor two), ``1-cos = 0.5||u1-u2||^2``, and the
whole expression in the log domain so that nothing overflows up to c = 65536.

| c | Poincare | **Lorentz** | pairing agreement | chart disagreement |
| --- | --- | --- | --- | --- |
| 0.001 | +0.6000 | **+0.6016** | 0.999 | 1.8e-10 |
| 0.05 | +0.5716 | +0.5333 | 0.991 | 0.79 |
| 1 | +0.5983 | +0.5266 | 0.995 | 0.95 |
| 4 | +0.6641 | +0.5402 | 0.993 | 0.98 |
| 16 | +0.6609 | +0.5441 | 1.000 | 0.99 |
| 256 | +0.6609 | +0.5449 | 1.000 | 1.00 |
| 4096 | +0.6609 | +0.5462 | 1.000 | 1.00 |
| 65536 | +0.6609 | +0.5461 | 1.000 | 1.00 |

Three conclusions, in order of importance.

1. **The headline +0.6605 is retracted.** It is what the saturated Poincare chart
   returns. The equivalent, numerically sound Lorentz chart gives **+0.545** for
   the same construction on the same points. The difference of 0.12 is not
   geometry; it is ``tanh`` hitting 1 and the denominator clip taking over.
2. **The curvature genuinely does not matter.** The Lorentz number moves from
   +0.533 to +0.546 across four orders of magnitude of curvature. The plateau the
   Poincare chart showed above c = 16 was an artefact, but there IS a real
   plateau, at +0.545, and it starts at c of order 0.05.
3. **Where both charts agree, and only there, the number is admissible**: at
   c = 0.001 the two differ by 1.8e-10 and return +0.6000 and +0.6016. So the
   four-point family's honest value is **about +0.60 at the nearly flat end, and
   it DECLINES as the points are compressed toward the boundary** -- the opposite
   of what the saturated sweep suggested. Against the Euclidean control (+0.5037)
   the hyperbolic form adds about 0.04 to 0.09, the same order as the leader's
   own hyperbolicity increment (+0.045, P = 0.051, protocol v7).

The lesson generalises: a curvature sweep on a Poincare chart measures the chart's
saturation past c of order 2. Any curvature claim in this project has to be
re-checked in the Lorentz chart before it is reported.

## v9 -- the weight was a liability, and the effect is Euclidean

The ablation the weighted mean needed: same quadruples, same informative filter,
same pairing rule, with and without the Gromov-delta weight, in both charts, plus
the Euclidean law and the permuted control. Artifact
`../unweighted_ablation_107.json`, script `unweighted_ablation.py`.

At c = 16:

| variant | rho | p |
| --- | --- | --- |
| Poincare, weighted | +0.6609 | 9.4e-15 |
| Lorentz, weighted | +0.5441 | 1.4e-09 |
| Poincare, unweighted | +0.6593 | 1.2e-14 |
| **Lorentz, unweighted** | **+0.6593** | 1.2e-14 |
| **Euclidean, unweighted** | **+0.6618** | 8.4e-15 |
| Lorentz, unweighted, SID permuted | +0.1341 | 0.168 (not significant) |

The unweighted Poincare and Lorentz scores are bit-identical: the maximum per-SID
difference is **0.000e+00**. That is the expected result -- the pairing choice is
an argmin over sums of distances, and the two charts agree on it completely
(pairing agreement 1.000) -- and it validates both implementations from the other
side. At c = 1 the two differ by 3.2e-03, i.e. only as far as the charts disagree.

Three conclusions.

1. **The delta weight was a liability.** Removing it lifts the Lorentz number from
   +0.5441 to +0.6593 (+0.115) and leaves the Poincare number where the
   saturation had already put it (+0.6609 to +0.6593). The weight `1/(1+delta/median)`
   concentrates the average on the quadruples whose delta is near the median, and
   under a saturated chart that is precisely the distorted part of the
   distribution.
2. **The effect is Euclidean.** The Euclidean unweighted score is +0.6618, at
   least as high as either hyperbolic one. This is not a coincidence and not a
   failure of the implementation: when the points sit at similar radii the
   hyperbolic distance is a strictly increasing function of the Euclidean one, so
   the argmin of the pairing sums -- which is all the unweighted score uses -- is
   invariant. The four-point family is a Euclidean four-point pairing test
   wearing a hyperbolic chart.
3. **It passes its control**: the permuted SID gives +0.1341 with p = 0.168.

**The admissible number for this family is +0.66, and it is not a hyperbolic
number.** The honest description is: *the agreement between the four-point pairing
of the SVD item geometry and the SID prefix tree predicts the 107 recalls at
about +0.66; the curvature, the chart and the delta weight contribute nothing.*

## v10 -- the leader is chart-fragile

The same swap was applied to HBAA: identical points, identical prefix centroids,
identical history barycentres, identical queries; only the distance law changes
from the Poincare ``arccosh(1 + ...)`` to the Lorentz log-domain form. Artifact
`../lorentz_hbaa_107.json`, scripts `lorentz_hbaa.py` and `lorentz_hbaa_run.py`.

| chart | rho | per-SID difference |
| --- | --- | --- |
| Poincare | +0.7623 | -- |
| Lorentz | **+0.6880** | median 6.3e-04, max 5.1e-03 |

Paired bootstrap: Poincare - Lorentz = **+0.0743 [+0.0117, +0.1459]**,
P(not better) = 0.010. The two value series correlate at 0.9933, so the ranking is
nearly the same; a sub-percent numerical difference moves the rank correlation by
0.074. Two consequences:

1. **+0.7623 is chart-inflated.** The admissible HBAA number is **+0.6880**. It
   still passes the permutation control (-0.0065) and it still beats the
   four-point family, but the margin is smaller than reported.
2. **The leader is fragile.** A reformulation that computes the same geometry
   better moves its headline by 0.074 with p = 0.010. The four-point family is not:
   its unweighted score is bit-identical between the charts (v9). A number whose
   ranking moves by that much under a numerically equivalent implementation should
   not be quoted to four decimals.

Where that leaves the family ranking, all numbers now in the numerically sound
chart or the Euclidean law:

| construction | rho | control |
| --- | --- | --- |
| HBAA, Lorentz | +0.6880 | permuted -0.0065 |
| four-point, Euclidean, unweighted | +0.6618 | permuted +0.1341 (n.s.) |
| four-point, Lorentz, unweighted | +0.6593 | -- |
| four-point, Lorentz, weighted | +0.5441 | -- |

The two independent constructions converge on +0.66 to +0.69, and neither needs a
curvature to get there: the Euclidean forms are +0.7170 (HBAA) and +0.6618
(four-point), i.e. at least as high as the hyperbolic forms in both families.

## v11 -- the leader's gap was a missing factor of two, not the chart

v10 attributed the gap between the two charts to the Poincare chart's
saturation. **That attribution was wrong**, and the operator found the cause: the
radial distance in `metrics_hbaa.py` was half of what it must be.

```python
def _norm(x, curvature):                       # before
    return np.arccosh(1.0 / np.sqrt(1.0 - c * np.sum(x * x, axis=-1))) / np.sqrt(c)
```

That is ``arctanh(sqrt(c)||x||) / sqrt(c)``. The distance from the origin to a
point of the Poincare ball of curvature ``-c`` is
``arccosh((1+u)/(1-u)) / sqrt(c) = 2 arctanh(sqrt(u)) / sqrt(c)`` with
``u = c||x||^2``, so the function returned exactly half of it -- and the same
file's ``_distance(0, x)`` returned the full value. The module contradicted
itself, and the Gromov product, which is half of
``d(0,x) + d(0,y) - d(x,y)``, was built on the halved radial term. The Lorentz
implementation used the full radius ``2 arctanh(sqrt(c)||x||)``, so the two
"charts" were computing different quantities.

The fix removes the second definition instead of correcting it: ``_norm`` now
delegates to ``_distance(0, x)``, so the radial distance and the pairwise
distance cannot drift apart again.

**After the fix the two charts are the same number to machine precision:**

| | rho | per-SID difference |
| --- | --- | --- |
| Poincare | +0.6880 | -- |
| Lorentz | +0.6880 | median **0.000e+00**, max **5.6e-16** |

So the +0.0743 gap of v10 was a definition bug, and the claim "the leader is
chart-fragile" is withdrawn: the metric is in fact chart-invariant to the last
bit, like the four-point family's unweighted score.

The corrected numbers, on the same 107 pairs:

| variant | rho | note |
| --- | --- | --- |
| hbaa (Poincare or Lorentz, they agree) | +0.6880 | the admissible value |
| hbaa_euclid | **+0.7170** | paired -0.0292 [-0.0588, -0.0049], P(not better) 0.992 |
| hbaa_flat (c = 1e-3) | +0.5969 | was reported as +0.7407 before the fix |
| hbaa_permuted | +0.0647 | p = 0.508, not significant |

Paired bootstrap: hbaa - euclid = **-0.0292 [-0.0588, -0.0049]**; hbaa - flat =
+0.0903 [+0.0358, +0.1582], P = 0.0003; hbaa - permuted = +0.6194 [+0.4191,
+0.8212].

Two things survive the correction, and they point the same way:

* the **Euclidean control is significantly better than the hyperbolic form**
  (-0.0292, interval clear of zero): the chart is not what makes the metric work;
* a **finite curvature is significantly better than a nearly flat one** (+0.0903):
  the metric does want curvature, but not more than the Euclidean limit gives.

The four-point family's retraction is unaffected: that one rested on the
weighted/unweighted contrast, which never used a radial distance to the origin,
and the unweighted score is bit-identical across the charts.

## v12 -- why the saturated chart ranks the SIDs better, and what it really says

The question was not "why does saturation raise the correlation" but "why do the
WRONG weights predict Recall better than the right ones". Three experiments on the
same 107 pairs, the same quadruples and the same items, at c = 16. Artifact
`../weight_experiment_107.json`, script `weight_experiment.py`. One table has no
informative quadruple at all and is excluded from the paired comparisons, so the
series below are over 106 tables.

### 1. The pairing judgement is untouched; only the weights move

The two charts choose the SAME pairing for every quadruple: the agreement between
their pairing choices is **1.0000**. The weights, however, correlate at only
0.6866, and the saturated chart's delta median is 3.6 times the true one
(0.00389 against 0.00107).

| correlation of the weight with | Poincare weight | Lorentz weight |
| --- | --- | --- |
| the TRUE delta | -0.666 | **-0.845** |
| the pairing scale | +0.111 | +0.626 |

and the true delta is itself anti-correlated with the pairing scale (-0.398).

### 2. The whole gap is the weights

Holding the shared agreement A_k fixed and swapping only the weights:

| weights | rho |
| --- | --- |
| Poincare | +0.6513 |
| Lorentz | +0.5312 |
| uniform | +0.6497 |

Paired bootstrap: uniform - Lorentz = **+0.1181 [+0.0693, +0.1755]**,
P(not better) < 1e-4; Poincare - Lorentz = +0.1195 [+0.0700, +0.1781],
P < 1e-4; **Poincare - uniform = +0.0013 [-0.0345, +0.0347]**, P = 0.46.

The saturated weights are statistically indistinguishable from uniform weights,
and both beat the correct delta weighting by about 0.12.

### 3. The clipping threshold is a numerical parameter, and it moves the delta

| clip | rho | median delta |
| --- | --- | --- |
| 1e-6 | +0.6472 | 0.00867 |
| 1e-9 | +0.6508 | 0.00506 |
| 1e-12 | +0.6513 | 0.00358 |
| 1e-15 | +0.6477 | 0.00276 |
| 1e-18 | +0.6477 | 0.00225 |

The delta moves by a factor 3.9 across a parameter that is pure numerics, while
the ranking moves by 0.004. The weights are an artefact; the agreement rate is
not.

### The answer to the question that was asked

**The Gromov delta weighting is harmful for this prediction task, and the
saturated chart accidentally removed it.** The weight `1/(1+delta/median)` is
largest for the quadruples whose delta sits at the median, i.e. for the quadruples
that are NOT distinctive in tree-likeness; and because the true delta falls as the
pairing scale grows (-0.398), that weighting hands the decisive influence to the
geometrically distant quadruples, whose agreement with the SID tree is the least
reliable signal. Saturation inflates the delta in proportion to the scale (the
log weight ratio correlates -0.642 with the scale, and -0.038 with the tree
spread and -0.037 with the shared prefixes: it has nothing to do with the SID
structure), which cancels part of that scale effect and lands the weighting near
uniform.

So the honest metric of this family is the **unweighted agreement rate**
(+0.6593 on the 107 tables), and the delta weighting -- the part of the
construction that was supposed to make it principled -- is what has to be
dropped. That is a statement about the metric, not about the geometry: it says
that what predicts Recall is *how often the geometry's four-point pairing agrees
with the SID tree at all*, not *how tree-like the geometry is on the quadruples
that disagree most*.

It also closes the loop on the curvature: the weights were the only thing the
curvature touched, and the correct ones make things worse, so nothing in this
family rewards a curvature.

## v13 -- the radial term helps, but at half its Gromov weight

The bug multiplied the radial term of the Gromov product by a half, which is the
same as scoring with a radial coefficient of 0.5 instead of 1. Writing the
contrast with a general coefficient,

    g_alpha(x,y) = ( alpha*d(0,x) + alpha*d(0,y) - d(x,y) ) / 2,
    Delta_alpha  = ( d(h,n) - d(h,p) ) / 2 + (alpha/2) * ( r_p - r_n ),

makes the bug a point on a one-parameter family. The sweep holds everything else
fixed -- the items, the SID trees, the prefix centroids, the history barycentres,
the queries, the distances, the temperature -- and moves only alpha. Artifact
`../hbaa_radial_sweep_107.json`, script `hbaa_radial_sweep.py`.

| alpha | rho | paired vs alpha=1 | P(not better) |
| --- | --- | --- | --- |
| 0 | +0.7104 | +0.0219 | 0.299 |
| 0.25 | +0.7367 | +0.0491 | 0.086 |
| **0.5** | **+0.7623** | +0.0747 | **0.0094** |
| **0.75** | **+0.7615** | +0.0739 | **0.0011** |
| 1 | +0.6880 | -- | -- |
| 1.25 | +0.5660 | -0.1210 | 0.9999 |
| 1.5 | +0.5276 | -0.1600 | 1.0000 |
| 2 | +0.3916 | -0.2952 | 1.0000 |
| 3 | +0.2301 | -0.4563 | 1.0000 |

Both end points check the arithmetic: alpha = 1 reproduces the corrected HBAA
(+0.6880) and **alpha = 0.5 reproduces the historical number (+0.7623) exactly**.
The historical headline was therefore this scoring rule, not the Gromov product.

The shape answers the question that was posed:

1. **The radial term is not harmful.** alpha = 0 (no radial term at all) gives
   +0.7104, and the peak beats it by +0.0522 [+0.0009, +0.1112], P = 0.023. How
   deep the positive's prefix sits relative to the negatives' is real signal.
2. **The Gromov weight over-credits it.** alpha = 1 loses +0.075 against the peak
   (P = 0.001 to 0.009), and the curve collapses from there: -0.12 at 1.25 and
   -0.46 at 3. The metric is robust to UNDER-weighting the radial term (alpha = 0
   and 0.25 stay within noise of the peak) and fragile to OVER-weighting it.
3. **The optimum is a plateau**, alpha = 0.5 and alpha = 0.75 differ by -0.0004
   [-0.0315, +0.0308], P = 0.51.

The Gromov product asserts that the deeper prefix shares a deeper ancestor with
the history, and it credits that with the full radial difference. The data says
the ancestor-depth reading is right but the credit is too large: a prefix that
is somewhat deeper than the negatives' is a better positive, and the full
Gromov weight over-credits it.

Alpha was chosen by looking at the recall, so the peak is a DIAGNOSTIC, not a
metric: the admissible choices remain alpha = 0 (+0.7104) and alpha = 1
(+0.6880), neither of them tuned.

**The theme that now runs through both families.** The four-point metric's
delta weighting -- a tree-likeness weighting -- hurt, and the unweighted
agreement rate won. Here the ancestor-depth weighting of the Gromov product
hurts as well, and a damped version of it wins. In both families the part that
fails is the part that reads the hyperbolic construction as a TREE, and the part
that works is the plain geometric contrast.

## v14 -- is the radial information hierarchy, or is it group compactness?

HBAA reads the radius of a prefix centroid as depth. The frozen reference puts
almost every item on a thin shell (radius p10 0.9050, median 0.9115, p90 0.9176),
so the radius of a group's centroid could be measuring how TIGHT the group is.
Artifact `../radius_vs_hierarchy_107.json`, script `radius_vs_hierarchy.py`.

For every level-1 prefix group of every table: the radius ``r = d(0, mu)`` of the
Frechet mean, the dispersion ``D = mean d(x_i, mu)^2``, the group size, the
members' popularity in the training interactions, the members' depth in an
INDEPENDENT hierarchy, and the same radius measured from a RANDOM point of the
ball. All statistics are computed within table and then pooled, and the
significance comes from a bootstrap over the 107 tables (the groups inside a table
are dependent).

**The product taxonomy is unavailable in this environment.** The parquet's
``categories`` column is entirely null and the raw ``dataset/Instruments.item.json``
the builder reads has been cleaned up, so the Amazon category tree could not be
used as a reference. The reference used instead is a behaviour hierarchy built
from the TRAINING interactions (an average-linkage clustering of the item
co-occurrence profiles), which the brief allows explicitly. Two readings of it
are used: the depth of a leaf in the tree and the height at which it joins.

| relation | value |
| --- | --- |
| radius vs dispersion | **-0.9106** |
| radius vs log group size | +0.2778 |
| radius vs log popularity | +0.1360 |
| radius vs the radius from a RANDOM centre | **+0.9798** |
| radius vs behaviour tree depth (raw) | +0.0729 |
| radius vs behaviour tree depth (partial, dispersion+size+popularity controlled) | **+0.2115**, 95% CI [+0.1885, +0.2320] |
| the same with a RANDOM centre | **+0.1921** |
| radius vs behaviour join height (partial) | **-0.0877**, 95% CI [-0.0966, -0.0774] |

**Verdict: the compactness explanation wins, and the hierarchy explanation fails
the brief's own criteria.**

1. The radius is 91% the dispersion. A group whose members are spread out has a
   centroid pulled toward the origin, so its radius shrinks; nothing about depth
   is needed to produce the number.
2. The radius from the true origin and the radius from a RANDOM point of the ball
   correlate at **0.980**. Whatever the quantity is, it is not "distance from a
   root": it is the same number whether the reference is the origin or an
   arbitrary point.
3. After controlling for compactness, size and popularity, a residual relation
   with the behavioural tree depth survives (+0.2115, CI clear of zero) -- **but a
   random centre reproduces 91% of it** (+0.1921). The residual does not need a
   meaningful root.
4. The second reading of the same hierarchy, the height at which an item joins,
   has the OPPOSITE sign (-0.0877, CI clear of zero). A quantity that means
   "depth" cannot correlate one way with the depth and the other way with the
   height.

So the radial term's contribution to HBAA is best read as group tightness and
the group's position relative to an arbitrary point -- and the Gromov product's
ancestor-depth semantics, which is what the alpha sweep showed to be
over-weighted, is not what the radius measures.

**The prediction value is nevertheless stable (the brief's experiment C).**
Splitting the 107 pairs into two halves and repeating the alpha sweep on each:

| alpha | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| even half | +0.7369 | +0.7568 | **+0.7779** | +0.7728 | +0.7098 | +0.5977 | +0.5605 | +0.4100 | +0.2015 |
| odd half | +0.6897 | +0.7148 | +0.7488 | **+0.7527** | +0.6639 | +0.5363 | +0.4957 | +0.3917 | +0.2677 |

Both halves show the same shape, and the best alpha is one grid step apart (0.5
and 0.75) inside the same plateau. The radial term's value therefore does not
depend on the sample; what it measures, on the evidence above, is not hierarchy.

Even if all three experiments had supported the hierarchical reading, the
correct claim would have been "empirical evidence for the radial-depth reading",
not "hierarchical information causes better recommendations": that needs an
intervention under matched training conditions, which this project has not run.

## v15 -- the radial term is group tightness, and v13's peak was a dimension

Two corrections and one decisive experiment.

### The decisive experiment: replace the radial term with the tightness

The radius is 91% the group's dispersion (v14), so the question the brief poses
is whether the radius carries HBAA's extra predictive power or whether the
tightness it stands for does. The two were put in the same scoring rule and
measured on the same 107 pairs:

    Delta = ( d(h,n) - d(h,p) ) / 2 + (coef/2) * ( z(term_p) - z(term_n) )

with ``term`` either the radius or the tightness ``D = mean d(x_i, mu)^2``, each
standardised over the table's own groups so that one coefficient scale serves
both families, and with both signs of the tightness term tried. Artifact
`../hbaa_tightness_swap_107.json`, script `hbaa_tightness_swap.py`.

| coef | -1.00 | -0.75 | -0.50 | -0.25 | **0.00** | +0.25 | +0.50 | +0.75 | +1.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| radial | +0.1305 | +0.2233 | +0.3969 | +0.5243 | **+0.7104** | +0.1881 | -0.0112 | -0.1386 | -0.2584 |
| tightness | -0.1043 | -0.0054 | +0.0904 | +0.2935 | **+0.7104** | +0.4755 | +0.3257 | +0.1428 | +0.0685 |

Zero is the peak of BOTH families, and the fall is on both sides, so this is not
a sign error. The split-half protocol agrees: the coefficient chosen on the even
half is 0.0 for both families, the coefficient chosen on the odd half is 0.0 for
both, and the held-out values are exactly the zero-coefficient values (+0.6897
on the odd half, +0.7369 on the even half). Neither the radius nor the
tightness adds anything; the plain distance contrast is the best version, at
**+0.7104**.

### Correction 1: v13's peak was a dimension, not a weight

v13 swept the coefficient on the RAW radius, whose scale is set by the ball
(radius median 2.785) and not by anything comparable to the distance
difference. The effective coefficient in each table was therefore
``alpha_raw / std_r(table)``, which varies from table to table without limit.
Under a scale-free coefficient the optimum is exactly zero on both independent
halves. So "+0.7623 at alpha = 0.5" is not evidence that half-weighting the
radial term helps; it is what a coefficient whose magnitude is set by the
dimensional accident of the radius does. v13's conclusion is withdrawn.

### Correction 2: v14's residual was an artefact of the reference

The hierarchy of v14 was built over at most 4,000 items and left the rest at a
default depth. The reference now covers every item (12,000 in the tree, the
remaining 18,000 inheriting the depth and the height of their nearest tree item
in the co-occurrence profile) and the analysis uses TEN random reference points
instead of one. Artifact `../radius_statistics_107.json`, script
`radius_statistics.py`.

| relation | v14 (broken reference, one random point) | v15 (full reference, ten points) |
| --- | --- | --- |
| radius vs the group's dispersion | -0.9106 | **-0.9019** |
| radius vs the distance to a random point | +0.9798 | **+0.9806** (mean of ten) |
| radius vs the tree depth (partial) | +0.2115 | **+0.0097** |
| the same from a random centre | +0.1921 | +0.0089 (mean of ten) |
| the origin's excess over a random centre | -- | **+0.0009** |
| radius vs the join height (partial) | -0.0877 | **-0.0001** |

The residual that v14 reported is gone: with a reference that covers every item
the radius has NO relation to the hierarchy depth (+0.0097) or to the join height
(-0.0001), and the true origin exceeds a random centre by 0.0009, which is
nothing. The tree is unbalanced (depth median 46, maximum 175, because average
linkage chains near-duplicate items), so the depth reading remains a weak
reference; but the verdict does not depend on it, since the height reading, the
random-centre control and the tightness control all agree.

### Where this leaves the construction

The admissible HBAA is the plain distance contrast: the Gromov product's radial
term is not only uninformative about hierarchy, it is harmful in every
scale-free setting tried, and the compactness it secretly measures is harmful in
the same way. The best numbers in this project are therefore

| construction | rho |
| --- | --- |
| HBAA without the radial term (alpha = 0) | **+0.7104** |
| HBAA, Euclidean control | +0.7170 |
| four-point agreement rate, unweighted | +0.6593 |

all of them Euclidean in content, and none of them rewarded by a curvature.

## v16 -- the six ablations: what the +0.0522 actually is

The brief's question, answered on the same 107 pairs with everything but the
scoring term held fixed (same items, SID trees, prefix centroids, history
barycentres, queries, temperature, recalls; no Stage 2 or Stage 3 touched).
Artifact `../hbaa_ablation_AF_107.json`, script `hbaa_ablation_AF.py`.

| variant | rho | paired vs plain | P(not better) |
| --- | --- | --- | --- |
| A plain (no radial term) | +0.7104 | -- | -- |
| B raw radial, alpha = 0.5 | **+0.7623** | +0.0522 [+0.0186, +0.0952] | 0.0005 |
| B' the standardised radius with the per-table weight | **+0.7623** | -- | -- |
| E the radius from TEN random origins | +0.7612 | -- | -- |
| F the radius shuffled among compactness-matched groups | +0.7506 | +0.0407 [+0.0086, +0.0801] | 0.0057 |
| C the radius PREDICTED from compactness, size and popularity | +0.7465 | +0.0361 [+0.0054, +0.0755] | 0.0106 |
| D2 the residual r - rhat, at the radius' own influence | +0.7458 | +0.0357 [+0.0137, +0.0646] | 0.0006 |
| D1 the residual at the same units and weight | +0.7337 | +0.0234 [+0.0093, +0.0442] | 0.0000 |
| G a term of the same scale with NO information (random) | +0.6954 | -0.0151 [-0.0373, +0.0047] | 0.9309 |

**The reproduction is exact.** B' replaces the raw radius with the standardised
one and uses the per-table equivalent weight ``alpha_std,t = 0.5 sigma_r(t)``;
the maximum difference over every per-query score of every table is
**2.2e-16**, i.e. machine precision, and the correlation is +0.7623 as before.
(The two formulations are the same rule; the earlier failure to reproduce was an
implementation error in the standardisation, not a difference of definitions.)

**What the excess is.** It is reproduced, almost to the digit, by a radius
measured from ten ARBITRARY points of the ball (+0.7612) and by a radius
SHUFFLED among compactness-matched groups (+0.7506). Neither carries any
group-specific radial information, so the excess is not about which prefix sits
where in the ball. It is also reproduced by the radius PREDICTED from the
compactness (+0.7465), which is a table-level aggregate: what matters is how
tight the groups are in aggregate, not which group has which radius.

**What the excess is not.** It is not a conditioning artefact: a random term of
the same scale makes things WORSE (+0.6954, paired -0.0151). So the gain is a
real effect of a table-level property -- but see the next section.

**The gain does not survive the honest protocol.** With the coefficient chosen
on the other families and evaluated on the held-out family (the families come
from the run names: curvature 77, mechanisms 18, resample 8, fork_hyper 3,
baseline 1):

| family | B raw radial | C compactness | D2 residual | F shuffled | G noise |
| --- | --- | --- | --- | --- | --- |
| curvature | +0.0706 | +0.0374 | +0.0500 | +0.0159 | -0.0077 |
| resample | +0.0476 | +0.2143 | -0.2143 | -0.1429 | 0.0000 |
| mechanisms | -0.0062 | -0.0041 | +0.0640 | -0.0145 | 0.0000 |

The radial term is better on two families and worse on one; the residual and the
shuffled radius change sign from family to family; the noise control is never
better. The pooled +0.0522 measured on all 107 pairs is therefore a SELECTION
EFFECT of the coefficient and of the sample, not a signal that transfers.

**Consequence.** The admissible HBAA is the plain distance contrast (+0.7104),
or its Euclidean control (+0.7170), and the +0.7623 headline is withdrawn for a
third reason -- it is not radial information, it is not compactness
information, and it does not generalise. The history of this family now reads:
the two-copy radial bug (v11), the chart comparison (v8, v9), the weight
analysis (v12), the hierarchy question (v14, v15), and this ablation, which
closes it: **the Gromov product's radial term is a table-level correction that
was tuned on the same 107 pairs.**

## v17 -- which feature of the SID table carries the gain, and the three fixes

The six ablations (v16) left one question: the gain is reproduced by an
arbitrary origin and by a shuffled radius, so it is not group-specific radial
information -- but WHICH group-level feature is it? Artifact
`../hbaa_feature_probe_107.json`, script `hbaa_feature_probe.py`.

Three defects in the first version of this probe were found and fixed first:

1. **The separation feature was identically zero.** The self-distance was set to
   infinity, the mean was taken (infinite) and the infinity was then replaced by
   zero, so the feature carried no information at all and the experiment measured
   nothing. It is now the sum over the OTHER groups divided by n-1.
2. **The table-level calibration could not cancel out.** A constant per group
   cancels in the positive-minus-negative difference, so the "radial mean"
   substitution could never test a calibration effect. It is now an offset added
   to the query difference itself, standardised against the pooled groups so that
   its size is comparable to the contrast, in two versions: the table's mean
   radius and the table's radius spread.
3. **The families were one member each.** Every run has its own variant name
   (``curvature_RQ-VAE_iter53``), so the held-out split was skipped everywhere.
   The families are now the mechanism prefixes: curvature 77, other 16, resample
   8, fork_hyper 3, crowd 2, behavior 1.

| candidate | best rho on the 107 | held-out: curvature | other | resample |
| --- | --- | --- | --- | --- |
| H1 group compactness | **+0.7636** | +0.6984 | +0.5971 | +0.7143 |
| the raw radius | +0.7623 | +0.6956 | +0.5941 | +0.7143 |
| H2 group separation | +0.7389 | +0.6319 | +0.6000 | +0.5952 |
| H4 group shape | +0.7210 | +0.6272 | +0.6059 | +0.6905 |
| H3 group size | +0.7123 | +0.6414 | +0.6176 | +0.8571 |
| H5 table offset (radius mean) | +0.7104 | alpha = 0 | alpha = 0 | alpha = 0 |
| H5 table offset (radius spread) | +0.7104 | alpha = 0 | alpha = 0 | alpha = 0 |
| a random term of the same scale | +0.7125 | worse | equal | worse |
| plain | +0.7104 | -- | -- | -- |

**The full permutation of the radius values** (every group's radius replaced by
another group's radius in the same table, 100 draws, the distribution preserved
exactly and the correspondence destroyed): mean **+0.7078**, p5 +0.6910, p95
+0.7215, and **no draw reaches 0.74**.

**What follows.**

1. **The table-level calibration is excluded.** Both offset variants choose alpha
   = 0 on every family and return the baseline exactly. The gain is not "adding a
   term of this size changes how the score spreads over the SIDs".
2. **The group separation does carry something** (+0.7389 once the bug is fixed,
   against the +0.7104 the broken version reported), but it is weaker than the
   compactness and it does not survive the held-out split: better on curvature,
   clearly worse on resample.
3. **The compactness and the raw radius behave identically under the held-out
   protocol** (+0.6984/+0.5971/+0.7143 against +0.6956/+0.5941/+0.7143). They are
   two writings of the same signal, which is what the brief's hypothesis says:
   the radial gain IS the compactness of the groups.
4. **The correspondence is required.** A full permutation of the radius values
   collapses to +0.7078 with no draw above 0.74, so the gain is not a pure
   distribution effect.
5. **The gain still does not generalise across families.** Both the radius and the
   compactness are better on curvature and resample and WORSE on the "other"
   family (16 members, 16 mechanism candidates), and the held-out numbers on the
   small families are unstable: the group size reaches +0.8571 on the eight
   resample members, which is what a sample of eight can do. The pooled +0.05 on
   the 107 pairs therefore still carries a selection component, and the only way
   to settle it is a fresh set of SID-recall pairs.
