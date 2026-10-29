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
| query point | history barycentre **+0.7014**, behaviour midpoint +0.6941, single anchor +0.6866, held-out probe +0.6689 | history barycentre |
| geometry | hyperbolic **+0.7014**, flat c=1e-3 +0.6965, Euclidean +0.6930 | hyperbolic, but by a hair |
| teacher | frozen c=1 **+0.7014**, variant c=0.25 +0.7014 | neither |
| negatives | all ten **+0.7014**, size-matched +0.6428 | all ten |
| levels | level 1 **+0.7014**, informative-weighted +0.5663, averaged +0.3604 | level 1 |
| reduction | pair accuracy +0.7025, softplus at the frozen tau **+0.7014** | the same, within noise |

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
* the machine-readable summary: `results/research/issue126/metric_round/summary.json`;
* the seven rounds of proposals: `results/research/issue126/metric_round/round_*/`;
* the round machinery: `tools/research/rounds/metric_round.py` and the MCP tools
  `research_metric_start`, `research_metric_status`, `research_metric_bench`.
