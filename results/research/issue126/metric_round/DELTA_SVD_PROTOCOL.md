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
