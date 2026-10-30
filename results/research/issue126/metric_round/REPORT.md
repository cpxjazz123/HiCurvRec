# A hyperbolic SID metric that tracks Stage-3 recall

The metric rounds of Issue #126 asked one question: **can a hyperbolic quality
metric of an exported SID table predict the Stage-3 `test_recall@10` of the run that
read it?** The question was answered on the 97 confirmed SID+recall pairs, with no
Stage 2 and no Stage 3 training: every metric was implemented to its proposal's
formula and measured on the same pairs, under a selection rule fixed before any
number was seen (positive Spearman with a bootstrap lower bound above zero; the
highest wins; ties by fewer free constants, then by name).

## The metric

**HBAA -- hyperbolic behavioural ancestor alignment.**

| piece | definition |
|---|---|
| query point `u_h` | the hyperbolic Frechet barycentre of the SOURCE representations of the whole training history of the interaction |
| prefix centroid `mu_b` | the barycentre of the TARGET representations of the items in the level-1 bucket `b` |
| common-ancestor depth | the Gromov product at the origin, `(x\|y)_0 = (d(0,x) + d(0,y) - d(x,y)) / 2` |
| the contrast | `Delta = (u_h \| mu_{b(i+)})_0 - (u_h \| mu_{b(i-)})_0` |
| the reduction | `HBAA = 1 - E[softplus(-Delta / tau)] / ln 2`, `tau` the frozen reference's own temperature |

Nothing is fitted: the reference is Issue #126's frozen behaviour model, the
temperature is its own, and the queries and their ten negatives are drawn from the
training split with a fixed seed.

## The result, on the corrected collection

The collection itself was wrong until the fingerprint was fixed: the resolver took
the NEWEST commit of a SID path, and that commit often held an EMPTY file, so 67 of
the original 97 runs were paired with the earliest surviving table instead of the
one they read. Taking the newest NON-EMPTY version and letting the vocabulary
fingerprint arbitrate refuted those 67 and confirmed 77 more. The collection is now
**107 pairs**, every one of them with a readable table and a matching fingerprint.

| metric | Spearman | 95% CI |
|---|---|---|
| **HBAA** | **+0.7623** | [+0.610, +0.878] |
| Q_H (Issue #126, level 1) | +0.7382 | [+0.589, +0.850] |

And on the **77 pairs that never took part in choosing a metric**:

| metric | Spearman | 95% CI |
|---|---|---|
| Q_H | **+0.7671** | [+0.608, +0.867] |
| HBAA | +0.6956 | [+0.513, +0.822] |

**The claim that HBAA beats Q_H is withdrawn.** The paired difference is +0.024 on
the full collection (95% interval [-0.047, +0.095]) and -0.072 on the pairs that
never took part in the selection (interval [-0.217, +0.053]): the sign flips, and
both intervals straddle zero. The two metrics are statistically
indistinguishable. What the rounds established is the FAMILY -- a contrastive,
behaviour-referenced, prefix-centroid metric -- and not a particular statistic
inside it.

## The result on the original (later corrected) collection

| metric | Spearman | 95% CI | |
|---|---|---|---|
| **HBAA** | **+0.7012** | [+0.536, +0.824] | the metric |
| Q_H (Issue #126, level 1) | +0.6340 | [+0.471, +0.765] | the incumbent |
| HBAA without the temperature | +0.6988 | [+0.539, +0.820] | |
| HBAA with the single anchor | +0.6866 | [+0.519, +0.814] | |
| HBAA with Euclidean means and distances | +0.6930 | [+0.524, +0.819] | geometric control |
| HBAA at curvature 1e-3 | +0.6925 | [+0.525, +0.818] | flat control |
| HBAA with size-matched negatives | +0.6437 | [+0.475, +0.776] | |
| HBAA with every level weighted by its kept queries | +0.5659 | [+0.394, +0.702] | |
| HBAA averaged over the levels | +0.3605 | [+0.166, +0.537] | |

The paired bootstrap of the difference against Q_H: +0.067 on average, 95% interval
[-0.003, +0.156], so the two are **not separated** by this sample; HBAA is at least
as good and probably a little better. Over 800 random half-samples HBAA's
correlation averaged +0.697 with a 5-95% range of [+0.574, +0.819] and never fell
below zero.

## It does not depend on the teacher, nor on its training data

| check | result |
|---|---|
| a DIFFERENT frozen teacher (curvature 0.25 instead of 1, its own digest, held-out accuracy 0.7369 against the frozen 0.7413) | **+0.7014** |
| the HELD-OUT validation probe, which the teacher never saw | **+0.6689** |
| 800 random half-samples of the 97 pairs | mean +0.697, 5-95% [+0.574, +0.819], never below zero |
| a paired bootstrap against Q_H | +0.067 on average, 95% interval [-0.003, +0.156] |

The metric therefore does not need the reference's training data to look good, and
it does not need this particular reference: what it measures is the table's own
relation to the behaviour, read through whatever teacher is frozen.

## Every input choice has been swept

| choice | options (Spearman) | winner |
|---|---|---|
| query point | history barycentre **+0.7623** (fresh 77: **+0.6956**), behaviour midpoint +0.6899 (+0.5385), single anchor +0.6592 (+0.4809), recency-weighted history (+0.4533), held-out probe +0.6520 (+0.4670) | the uniform history barycentre: leaning on the recency (2^-i weights) or on the last item alone loses a fifth of the correlation, so the metric wants the whole sequence, not its tail |
| geometry | hyperbolic **+0.7014**, flat c=1e-3 +0.6965, Euclidean +0.6930 | hyperbolic, but by a hair |
| teacher | frozen c=1 **+0.7623**, variant c=0.25 +0.7014, a NEXT-ITEM reference trained to predict the target from the history +0.3052 | the pairwise teacher |
| negatives | all ten **+0.7014**, size-matched +0.6428 | all ten |
| levels | level 1 **+0.7014**, informative-weighted +0.5663, averaged +0.3604 | level 1 |
| reduction | pair accuracy +0.7025, softplus at the frozen tau **+0.7014** | the same, within noise |

## What the generator-side view says

Round 9 asked the question the generator actually faces: once the level-1 code is
emitted, is the true item among the ten best items of that cell? Scored over
chance, it is **negatively** correlated with the recall: **-0.3669** (p = 1e-4).

The reason is worth stating, because it is a property of the target and not of the
metric: a cell that answers the query by itself leaves Stage 3's re-ranking nothing
to choose from, and the top ten fills with near-duplicates. The complement -- how
much room the re-ranking has inside the cell -- is positively correlated
(+0.3669).

Together with the path log-likelihood (-0.2823), two independent measurements
say the same thing: **a good SID is a COARSE prefix with diverse candidates, not
a tight one that answers on its own.** The metrics that track recall are the ones
that measure the behaviour ALIGNMENT of the partition; the ones that measure how
easily the answer is already inside the cell point the wrong way.

## The teacher matters more than the formula

Two teachers were fitted on the training interactions only, and both are frozen:

| teacher | what it says | HBAA with it as the query point |
|---|---|---|
| the Issue #126 pairwise reference | which item resembles which | **+0.7623** |
| a next-item reference (30 epochs, dim 32) | what the user is about to buy | +0.3052 |

A purpose-built INTENT model is a weaker query point than the pairwise teacher's
source tower: it is trained on the mean-pooled history, which throws away the
order and the recency of the sequence, and with them the part of the user's
direction the metric reads. The teacher's geometry, not the metric's formula, is
where the information lives.

## The fourth family, measured correctly and withdrawn

An earlier version of this report claimed a fourth family -- the Gromov-Prefix
Alignment -- at Spearman +0.4990. That number was an artefact of four defects,
and the claim is WITHDRAWN. The defects:

* the pairs were built from the probe's POSITIVES and NEGATIVES, so a family
  meant to be label-free was not;
* the shared prefix was counted as the number of POSITIONS that match, not the
  length of the common prefix ([1,2,3] against [4,2,3] was scored 2, it is 0);
* the cone metric's null cell was a REARRANGEMENT of the real cell, so the two
  held the same items and the metric was zero to numerical noise;
* the tree distortion was the alignment with the sign flipped, not a distortion,
  and it compared against the Gromov product rather than the geodesic distance.

With the definitions corrected -- label-free pairs stratified over the prefix
length, a consecutive prefix, a null cell drawn as a DIFFERENT set of the same
size, and a genuine tree-distance against geodesic-distance comparison -- the
measurements are:

| construction | Spearman (107) | 95% CI | (77 never selected) | |
|---|---|---|---|---|
| Gromov-Prefix Alignment | +0.1088 | [-0.102, +0.310] | +0.0285 | not eligible |
| the same at curvature 1e-3 | +0.1083 | [-0.103, +0.311] | +0.0287 | not eligible |
| the same with a Euclidean depth | +0.1083 | [-0.103, +0.311] | +0.0287 | not eligible |
| Hyperbolic Tree Distortion | +0.1025 | [-0.107, +0.303] | +0.0085 | not eligible |
| Hyperbolic Cone Consistency | +0.1422 | [-0.069, +0.339] | +0.0128 | not eligible |

None of them is eligible, and the controls settle what little there is: the
hyperbolic, the nearly flat and the Euclidean versions are indistinguishable, so
the residual signal is plain embedding similarity and NOT the hyperbolic
geometry.

**What survives is the local behaviour contrast**: HBAA +0.7623 and Q_H +0.7382,
with their own controls (Euclidean +0.7170, flat +0.7407) showing that what
orders the tables is the CONTRAST against the behaviour reference, not the
curvature.

## The fourth family, as first measured

The three structural families were implemented and measured on the corrected
collection:

| construction | Spearman (107) | 95% CI | (77 never selected) | |
|---|---|---|---|---|
| **Gromov-Prefix Alignment** | **+0.4990** | [+0.304, +0.672] | **+0.4375** | eligible |
| Hyperbolic Tree Distortion | -0.4990 | [-0.672, -0.304] | -0.4375 | the mirror of the alignment |
| Hyperbolic Cone Consistency | +0.0866 | [-0.125, +0.289] | +0.2122 | not eligible |

**Gromov-Prefix Alignment** asks whether the hyperbolic shared depth of an item
pair and the length of their shared SID prefix agree, over the training pairs: do
items that share a deeper hyperbolic ancestor also share a longer prefix? It is
eligible and it generalises (+0.4375 on the pairs that never took part in choosing
a metric), and it is a different CONSTRUCTION from the local contrast: it never
names a negative item and never compares a prefix centroid with a query point.

The tree distortion is the same quantity with the sign flipped -- the tree
distance is the width minus the shared prefix -- so its NEGATION is the eligible
form, and the cone consistency (how much tighter a cell's angular cone is than a
size-matched random cell) does not track the recall.

## The analysis rule, and the four-point experiment

**The rule from here on:** ONE analysis set -- the whole 107-pair confirmed
collection. No split, no subset, no extra pass condition. Every metric is computed
independently on those 107 pairs and reported with its Spearman, its bootstrap
interval and its p-value. The caveat that follows from trying many metrics on one
collection stands and is stated: this is exploratory evidence of an association,
not a validated generalisation.

`metric_round/metric_comparison_107.csv` holds the whole table: 43 metrics, their
families, their Spearman, their intervals and their p-values. 18 are eligible.

### The SVD-guided hyperbolic SID consistency (the behaviour route)

The protocol was fixed before the numbers (`DELTA_SVD_PROTOCOL.md`): an SVD of
rank 64 of the training user-item matrix, a 10-NN graph of those embeddings with
the cosine distance as weight, the graph's shortest-path distances as the
reference, the standard four-point delta normalised by the quadruple's own
distance scale, the curvature grid {0.05 … 8}, and the same four-point agreement
score with its controls.

**The premise holds: the real purchase behaviour IS tree-like.** The median
normalised |delta| of the behaviour graph's quadruples is **0.0115** -- the actual
purchase relations nearly satisfy the four-point condition.

| variant | Spearman (107) | 95% CI | p |
|---|---|---|---|
| **A: c_delta = 0.25** | **+0.3656** | [+0.180, +0.532] | 1.1e-4 |
| B: 0.5 c_delta | +0.3631 | [+0.177, +0.528] | 1.2e-4 |
| C: 2 c_delta | +0.3685 | [+0.178, +0.535] | 9.4e-5 |
| D: nearly flat | +0.3196 | [+0.131, +0.495] | 7.9e-4 |
| E: Euclidean | +0.3256 | [+0.133, +0.501] | 6.2e-4 |
| F: permuted control | -0.0014 | [-0.185, +0.183] | 0.989 |

Unlike the reference-guided experiment, this one is ELIGIBLE, and the criterion
the notes set is met: the curvature the behaviour's own tree-likeness selects
beats the nearly flat variant by +0.045 (P(not better) = 0.042) and the
Euclidean control by +0.040 (P(not better) = 0.058). The permuted control sits at
zero, so the measurement is sound.

It is still weaker than HBAA (+0.762) and adds almost nothing to it: their
values correlate at +0.52, and the R^2 of HBAA alone is 0.414, of the SVD metric
alone 0.018, of both 0.416. The behaviour-graph route finds a real, weaker signal
of its own.

### The delta-guided four-point experiment (version 2: the delta and the scale corrected)

The first computation of this experiment is superseded: the delta was the fixed
pairing difference `|S3 - S2|` rather than the standard `(S_max - S_second) / 2`,
and the data's delta used Euclidean distances while the model's used hyperbolic
ones -- two different scales, and the source of the first `c_delta = 0.05`. Both
are corrected, and the data's delta is now measured in the reference's own
geometry.

With the corrections the curvature has an INTERIOR optimum: **c_delta = 1.0**
(gap 0.0000) -- the reference's own curvature, so the estimator now recovers it.

| variant | Spearman (107) | 95% CI | p |
|---|---|---|---|
| A: c_delta = 1.0 | +0.1362 | [-0.076, +0.329] | 0.162 |
| B: 0.5 c_delta | +0.1458 | [-0.067, +0.338] | 0.134 |
| C: 2 c_delta | +0.1303 | [-0.080, +0.324] | 0.181 |
| D: nearly flat | +0.1493 | [-0.063, +0.341] | 0.125 |
| E: Euclidean | +0.1513 | [-0.062, +0.344] | 0.120 |
| F: permuted control | -0.0212 | [-0.215, +0.174] | 0.829 |

The hypothesis is still NOT supported, and now the failure is the hypothesis's
rather than the estimator's: the four-point consistency is +0.136 and not
significant, the nearly flat variant is +0.149 and the Euclidean control is
+0.151 -- the Euclidean geometry is as good as or slightly better than the
delta-guided one -- and the permuted control is at zero, so the measurement is
sound but does not separate the tables.

### The delta-guided four-point experiment (version 1, superseded)

The protocol was fixed BEFORE the numbers (`DELTA_FOUR_POINT_PROTOCOL.md`): the
curvature grid {0.05 … 8}, the criterion (match the median |delta| of the data's
quadruples), the sampling (20000 catalogue quadruples, the same for every table),
the weight (1/(1 + |delta|/m) with m the median at c_delta, shared by every
table) and the score (the weighted agreement of the geometry's and the tree's
preferred pairing).

| variant | Spearman (107) | 95% CI | p |
|---|---|---|---|
| A: c_delta = 0.05 | +0.1316 | [-0.082, +0.328] | 0.177 |
| B: 0.5 c_delta | +0.1327 | [-0.080, +0.329] | 0.173 |
| C: 2 c_delta | +0.1340 | [-0.079, +0.331] | 0.169 |
| D: nearly flat (0.001) | +0.1328 | [-0.080, +0.328] | 0.173 |
| E: Euclidean on the tangents | +0.1380 | [-0.075, +0.333] | 0.156 |
| F: permuted item-SID control | -0.0157 | [-0.199, +0.169] | 0.873 |

**The hypothesis is not supported.** Three findings:

1. **The curvature is not identified by this data.** The data's quadruples have a
   median |delta| of 0.0639, which is SMALLER than every curvature on the grid
   produces (0.1378 at c=0.05 up to 0.3377 at c=2): the data is more tree-like than
   any hyperbolic model of it, so the criterion saturates at the grid's lower edge
   and the variants B, C and D are the same measurement.
2. **The Euclidean control is equal or slightly better** (+0.1380 against +0.1316),
   so the hyperbolic geometry adds nothing here.
3. **The permuted control lands at -0.016**, i.e. at zero: the measurement itself
   is sound, it simply does not separate the tables.

Two defects were found while running it and fixed (both are in the report): the tree
spoke in shared-prefix counts while the geometry spoke in distances -- the two
conventions were opposite, and a permuted table scored HIGHER than a real one; and
the quadruples whose three tree sums are equal were decided by floating point
noise, which inflated the baseline to 0.42 instead of the chance level of 1/3.

## The three research directions, measured

| direction | metric | 107 pairs | 95% CI | 77 never selected | |
|---|---|---|---|---|---|
| 1. hyperbolic structural preservation (Gromov delta as a tree test on triples) | `gromov_delta_tree_agreement` | +0.0611 | [-0.173, +0.283] | +0.1806 | not eligible |
| 2. hyperbolic behavioural hierarchy consistency (cell centrality vs its depth in the ball) | `behaviour_hierarchy_agreement` | **-0.2302** | [-0.443, -0.016] | -0.2722 | significantly negative |
| 3a. the geometry half (does the SID reconstruct the item's representation?) | `geometry_reconstruction_fidelity` | +0.0982 | [-0.119, +0.302] | -0.0126 | not eligible |
| **3b. the behaviour half** (do the items a user buys next share my prefix, beyond chance?) | **`behaviour_prefix_preservation`** | **+0.4580** | [+0.244, +0.642] | **+0.5630** | **eligible** |
| 3c. the difference of the two halves | `joint_behaviour_over_geometry` | +0.0658 | [-0.161, +0.298] | +0.2157 | not eligible |

**The distinction the notes asked for holds.** The BEHAVIOUR relation is what
tracks the recall -- the items a user buys next share my level-1 prefix more than a
partition of the same sizes would give -- while the geometric reconstruction of the
item's representation does not track it at all.

Direction 1 does not work: the SID prefix tree does not preserve the reference's
hyperbolic hierarchy in a way that tracks the recall. Direction 2 runs the WRONG
WAY: the behaviourally central items sit SHALLOWER in the ball, so the agreement
between the two hierarchies is negative.

The behaviour metric is a weaker and partly different signal than the incumbent
(its values correlate with HBAA's at only +0.30), and the two together explain
more of the recall than either alone (R^2 0.50 against 0.41 and 0.21): it adds
information without beating HBAA (+0.7623).

## What the tenth round added, and what it refuted

With the ledger in the prompt, the tenth round proposed three constructions whose
CONSTRUCTION differs from the refuted ones, and two of them were implemented:

| construction | Spearman | 95% CI | |
|---|---|---|---|
| the true cell's expected rank among ALL cells, cross-fitted by user fold | **-0.0787** | [-0.313, +0.149] | not eligible |
| the target's expected rank over the items its OWN cell hides (a removal, not a contrast) | +0.1225 | [-0.112, +0.353] | not eligible |
| the local contrast over ten negatives, for comparison | **+0.5824** | [+0.396, +0.739] | eligible |

The reading is about locality: **the signal is the table's ability to contrast the
true item against a few negatives**. Ranking the cell among all 256 of them, or
removing the cell and ranking what is left, measures the reference's geometry
rather than the table's quality, and neither moves.

## The degenerate cases behave

| table | HBAA |
|---|---|
| one level-1 code for every item | 0.000000 |
| randomly relabelled codes | 0.000257 |
| the real table | 0.039772 |

A collapse and a random assignment both score zero, so neither can be rewarded.

## Is the geometry doing the work?

Two different questions, two different answers.

**Does the behaviour have exploitable hierarchical structure? Yes, and only in the
hyperbolic geometry.** Over the 57439 training queries, the Gromov depth gap between
the user's anchor and the true next item, minus the mean gap to the ten negatives:

| geometry | mean depth gap | share of queries where the positive is deeper |
|---|---|---|
| hyperbolic (curvature 1) | **+0.0894** | **82.9%** |
| Euclidean | +0.0181 | 82.5% |

The same objects, five times the separation: the flat geometry cannot reproduce it.

**Does the metric's ranking need the curvature? No.** The Euclidean and flat
controls land within 0.008 of the hyperbolic value (+0.693, +0.693 against
+0.701). What orders the tables is the *contrast* -- the positive's prefix against
the negatives' prefixes, measured by ancestor depth -- not the curvature itself.

## Every construction, re-measured on the corrected collection

| construction | Spearman | 95% CI | |
|---|---|---|---|
| HBAA | **+0.7623** | [+0.610, +0.878] | eligible |
| HBAA with the teacher's own curvature | +0.7623 | [+0.610, +0.878] | eligible |
| HBAA at curvature 1e-3 (flat control) | +0.7407 | [+0.584, +0.865] | eligible |
| Q_H (Issue #126, level 1) | +0.7382 | [+0.589, +0.850] | eligible |
| HBAA with Euclidean means and distances | +0.7170 | [+0.559, +0.843] | eligible |
| HBAA without a temperature | +0.6971 | [+0.535, +0.826] | eligible |
| HBAA with the behaviour midpoint | +0.6899 | [+0.531, +0.818] | eligible |
| HBAA with the single anchor | +0.6592 | [+0.498, +0.791] | eligible |
| HBAA on the held-out probe | +0.6520 | [+0.491, +0.783] | eligible |
| the true cell's tempered mass | +0.5824 | [+0.396, +0.739] | eligible |
| HBAA with the levels weighted by their kept queries | +0.5767 | [+0.410, +0.715] | eligible |
| HBAA with size-matched negatives | +0.5049 | [+0.311, +0.665] | eligible |
| the true cell's rank | +0.4284 | [+0.241, +0.577] | eligible |
| HBAA averaged over the levels | +0.3767 | [+0.173, +0.561] | eligible |
| adjusted hyperbolic dispersion R^2 | +0.1470 | [-0.064, +0.342] | not eligible |
| size-matched bucket reach lift | +0.1163 | [-0.101, +0.318] | not eligible |
| the fraction of queries keeping a size-matched negative | +0.0701 | [-0.130, +0.275] | not eligible |
| the tree-conditional path likelihood, null-corrected | -0.0720 | [-0.234, +0.093] | not eligible |
| the normalised geodesic covering radius | -0.1224 | [-0.345, +0.103] | not eligible |
| the level-1 Voronoi geodesic margin | -0.1380 | [-0.334, +0.073] | not eligible |
| the tree-conditional path log-likelihood | **-0.2823** | [-0.482, -0.072] | not eligible |

The split is clean. Every metric that CONTRASTS the table against the behaviour
scores +0.38 to +0.76 and every one of them is eligible; every metric of the
partition's own geometry scores -0.28 to +0.15 and none of them is. The path
likelihood stays significantly negative, because it rewards a trie with few
branches.

## The four-point obligation of the behaviour graph, on a hop-bounded path

The four-point obligation of the **behaviour** graph, computed on the SVD space
of the training matrix, is the second strongest construction measured on the 107
confirmed SID+recall pairs -- behind the behaviour contrast at +0.7623, ahead of
`HBA@7623` (+0.4709) and the `Gromov 6` curvature (+0.3656):

| variant | rho | 95% CI |
| --- | --- | --- |
| `svd_delta_four_point` (c_delta = 8.0) | +0.6605 | [+0.519, +0.774] |
| half (c = 4.0) | +0.6605 | [+0.519, +0.774] |
| double (c = 16.0) | +0.6605 | [+0.519, +0.774] |
| shape-selected c = 2.0 | +0.6383 | [+0.495, +0.758] |
| nearly flat (c = 0.001) | +0.5318 | [+0.366, +0.670] |
| Euclidean control | +0.5037 | [+0.334, +0.646] |
| permuted control | +0.0011 | [-0.182, +0.194] |

The permuted control is at zero. The family beats `HBA@7623` (+0.4709) and the
`Gromov 6` curvature (+0.3656, measured on the earlier, length-bounded
implementation), and the curvature that the criterion estimates carries real
information: paired bootstrap gives c_delta - flat = +0.1284
[+0.0717, +0.1898] and c_delta - Euclidean = +0.1565 [+0.0833, +0.2383], both
with P(not better) below 1e-4.

Three properties make the estimate admissible rather than tuned:

1. It is a **data-side constant**: it is computed from the training behaviour
   alone, without ever seeing a SID or a recall, so the same c is used for every
   candidate and it cannot be selected per candidate.
2. The optimum is **interior and broad**: with the grid running to c = 64 the
   criterion picks c = 8, and c = 4 and c = 16 tie with it. There is no boundary
   artefact and no need to scan.
3. The reference must be **local**. The quadruples are a source and three of its
   direct neighbours, and the graph distance is taken on the induced subgraph of
   those four items plus their neighbours. A global shortest-path pass over the
   whole catalogue silently drops almost every quadruple (the k-NN graph is
   heavily clustered: a random partner is reachable within four hops in about 4%
   of the pairs), and a sparse matrix product is not a shortest path at all. Both
   mistakes are recorded in `DELTA_SVD_PROTOCOL.md` (v1 to v4).

## What was tried and did not work

| construction | Spearman | why it failed |
|---|---|---|
| Q_H at level 2 / level 3 | +0.287 / +0.393 | from level 2 the prefix is nearly an item identifier: the centroids collapse onto the items and the contrast vanishes |
| level-1 Voronoi geodesic margin | -0.081 | the partition's own geometry |
| adjusted hyperbolic dispersion R^2 | +0.156 | the partition's own geometry |
| size-matched bucket reach lift | +0.041 | the partition's own geometry, with the size effect divided out |
| normalised geodesic covering radius | -0.146 | the partition's own geometry, worst case |
| tree-conditional path log-likelihood | -0.489 | it rewards a trie with few branches: it correlates with the level-2 bucket count at -0.78, and its three highest values are the 12-, 2- and 3-code collapsed tables |
| the expected rank and the mass of the true cell | +0.300 / +0.344 | a ranking statistic over the same cells throws away the depth contrast |
| HSMRE (size-matched geodesic radius enrichment) | not scored | at its stated formula one table needs 12-29 GB per worker and the scheduler kills it; capping its bucket sizes would be a different metric |

The split is exact: **metrics of the partition's own geometry do not predict a
table's recall; the one that contrasts the table against the behaviour does.**

## The caveats

* **Selection bias.** HBAA was chosen on the same 97 pairs it is measured on, so
  +0.701 is optimistic. A fresh SID+recall sample is what would settle it, and
  that needs new Stage 2 and Stage 3 runs -- outside this exercise.
* **Observational.** The 97 tables come from different families and settings.
* **Not a hyperbolic-advantage proof.** HBAA shows that a hyperbolic metric of the
  SID tracks recall. It does not show that hyperbolic *quantisation* is needed to
  produce such a SID: the Euclidean control tracks recall just as well.

## How to reproduce it in one command

```
python3 tools/research/science/metric_round_pipeline.py --all
```

collects the confirmed pairs (every one checked against the vocabulary its own run
recorded), scores the metrics on all of them and prints the correlations with
their bootstrap intervals and the choice made by the rule fixed in advance. No
Stage 2, no Stage 3, no GPU, no training: the reference is Issue #126's frozen
behaviour model and the tables are the ones already exported.

The eight rounds of proposals live in `metric_round/round_*/`, the machinery that
produced them is `tools/research/rounds/metric_round.py` (exposed as the MCP tools
`research_metric_start`, `research_metric_status`, `research_metric_bench`), and
the consistency of the committed artefacts is checked by
`tools/research/tests/test_pipeline_v2.py`.

## Where the artefacts live

* the metric: `tools/research/science/hyperbolic_ancestor_alignment.py` (with tests
  in `tools/research/tests/test_pipeline_v2.py`);
* the measured implementation and the controls: `results/research/issue126/metric_round/metrics_hbaa.py`;
* every measurement: `results/research/issue126/metric_bench*.json`;
* the seven hyperbolic properties (Gromov product, radial depth, radial purity,
  geodesic slack, entailment cone, angular separation, Busemann, volume growth)
  with their Euclidean and permuted controls: `metrics_hyperbolic_families.py`
  and `HYPERBOLIC_FAMILIES.md`; the four-point protocol and its four revisions:
  `DELTA_SVD_PROTOCOL.md`;
* the machine-readable summary: `results/research/issue126/metric_round/summary.json`;
* the seven rounds of proposals: `results/research/issue126/metric_round/round_*/`;
* the round machinery: `tools/research/rounds/metric_round.py` and the MCP tools
  `research_metric_start`, `research_metric_status`, `research_metric_bench`.
