# Behavior Graph Geometry Analysis: Does User Purchase Behavior Exhibit Hyperbolic Structure?

Independent study on `cpxjazz123/HiCurvRec`. Seed 42 throughout, single seed, no
replication. Real data only: `results/stage0_build_parquet/train.parquet` builds the
graph, `valid.parquet` is never used here, `test.parquet` is not touched. No RQ-VAE
model, Stage 2 configuration or checkpoint was modified.

Provenance: git `63378a3` (analyses) on top of the repository at `a1c8fac`;
`genrec_env` python 3.10; numpy/scipy/networkx/matplotlib/sklearn/torch (L40S).
Artifacts in this directory: `behavior_graph_statistics.json`,
`behavior_hierarchy_analysis.json`, `gromov_hyperbolicity.json`,
`embedding_distortion_comparison.csv` / `.json`, `graph_arrays.npz`,
`lcc_edges.npz`, `ccdf_arrays.npz`, and the figures `ccdf_powerlaw.png`,
`hierarchy_communities.png`, `gromov_delta.png`, `embedding_distortion.png`.
Code and unit tests in `analysis/behavior_graph_geometry/`.

---

## 1. The graph

| quantity | value |
|---|---|
| train rows / users / distinct item ids | 396958 / 57439 / 24556 |
| distinct directed transitions | 323633 |
| self loops (same item bought twice in a row) | 0 |
| undirected nodes / edges | 24556 / 315391 |
| largest connected component | the entire graph (1 component) |
| mean / median degree | 25.69 / 13 |
| density | 0.001046 |
| average clustering / transitivity | 0.0434 / 0.0388 |
| degree assortativity | -0.00168 |
| diameter / mean distance | 7 / 3.187 ± 0.014 (400 seeded BFS sources) |
| unordered pairs seen in BOTH directions | 8242 of 315391 (2.61%) |

Two cross-checks. The 323633 distinct directed transitions match the 323633 positives
of the repository's own canonical behaviour-pair set exactly, so the sequence
extraction reproduces the repository's transition semantics. The parquet carries **no
timestamp**; the order is encoded by `len(seen_history)`, recovered by sorting each
user's rows on that length and verified rather than assumed.

Direction matters and the symmetrisation used below loses it: 97.39% of the unordered
pairs occur in one direction only. Every community, distance and embedding result in
this report is about the **symmetrised** relation.

---

## 2. RQ1 — Power-law / long-tail structure

| distribution | Gini | top 1% | top 10% | xmin | alpha | KS | R(lognorm vs PL) | R(expon vs PL) |
|---|---|---|---|---|---|---|---|---|
| item interaction count | 0.5778 | 0.183 | 0.500 | 24 | 2.376 | 0.060 | **-27.71** | +7.59 |
| item degree (undirected) | 0.5585 | 0.157 | 0.475 | 29 | 2.382 | 0.046 | **-27.73** | +9.25 |
| item strength | 0.5813 | 0.183 | 0.501 | 30 | 2.339 | 0.044 | **-31.11** | +9.97 |
| out-degree (weighted) | 0.5819 | 0.182 | 0.500 | 22 | 2.393 | 0.062 | **-30.21** | +7.17 |
| in-degree (weighted) | 0.5842 | 0.184 | 0.503 | 24 | 2.412 | 0.062 | **-27.66** | +7.04 |
| out-neighbours | 0.5672 | 0.164 | 0.483 | 22 | 2.441 | 0.065 | **-27.65** | +6.34 |
| in-neighbours | 0.5688 | 0.164 | 0.485 | 24 | 2.467 | 0.063 | **-25.91** | +6.26 |
| transition weight | 0.0457 | 0.035 | 0.142 | — | — | — | no admissible fit | |

Method: xmin by minimum KS, alpha by the discrete MLE, alternatives by **conditional**
ML on the same tail (`log L = Σ log pdf - n log S(xmin)`) scored on the conditional CDF,
models compared by a normalised log-likelihood ratio R (~N(0,1) under the null). R > 0
favours the power law. Transition weights are almost all 1, so there is no tail to fit.

Three separate statements, as RQ1 requires:

1. **Long tail: yes.** Gini 0.56–0.58 everywhere, top 1% of items carry 15.7–18.4% of
   the volume, top 10% carry 47.5–50.3%, and the 99th percentile is 16–20× the median.
2. **Power law: no.** The lognormal wins by R = -25.9 to -31.1, far outside noise, on
   all seven distributions and on the identical tail. The power law does beat the
   exponential (R = +6.3 to +10.0), so the tail is heavier than exponential — which is
   not the same claim.
3. **Only popularity concentration: no.** alpha sits at 2.34–2.47 with KS 0.044–0.065
   consistently across seven views of the same graph.

**H1 → PARTIALLY SUPPORTED.** Long-tailed, but not power-law-tailed.

---

## 3. RQ2 — Community, branching and hierarchical structure

| resolution | communities | modularity | largest community | cross-community edges |
|---|---|---|---|---|
| 0.5 | 3 | 0.0814 | 89.53% | 2.94% |
| 1.0 | 9 | **0.3512** | 28.37% | 39.21% |
| 1.5 | 20 | 0.3381 | 18.14% | 55.10% |
| 2.0 | 37 | 0.3042 | 9.74% | 63.96% |
| 3.0 | 102 | 0.2607 | 6.83% | 71.21% |
| degree-preserving null (res 1.0) | 10–15 | **0.1567–0.1589** | 16.97–17.36% | 69.72–72.85% |

Branching: distinct successors mean 13.22, median 7, p90 25, p99 113, max 1295; 2.17%
of items have a single successor and 33.73% have at least ten; the top 1% own 16.41% of
all successor slots.

Tree-likeness: **290836 of 315391 edges are non-tree edges (92.21%)**; the BFS tree
reproduces only **0.69%** of sampled pair distances exactly, against **0.33–1.88%** for
the degree-preserving nulls — the real graph sits inside the null range.

1. **Communities are real and beyond the degree sequence:** modularity 0.3512 against
   0.1567–0.1589, a much larger largest community, and far fewer cross-community edges.
2. **Multi-scale:** modularity stays in 0.26–0.35 while the count runs 9 → 102.
3. **Branching is strong:** median 7 distinct successors, a third with ≥10.
4. **Tree-likeness: no.** Not closer to a tree than a random graph of the same degrees.

**H2 → PARTIALLY SUPPORTED.** Communities and branching yes; tree-like hierarchy no.

---

## 4. RQ3 — Gromov hyperbolicity

Four-point delta on the unweighted undirected LCC, seeded Monte Carlo over quadruples
of a seeded 2000-source sample.

| graph | diameter | median delta | mean | p95 | delta/diameter |
|---|---|---|---|---|---|
| real, uniform anchors | 7 | **0.0000** | 0.1732 | 0.5 | 0.0000 |
| real, top-1% most popular | 7 | 0.0000 | 0.1629 | 0.5 | 0.0000 |
| real, bottom-half popularity | 7 | 0.0000 | 0.1993 | 0.5 | 0.0000 |
| degree-preserving null 0 / 1 / 2 | 6 / 5 / 5 | **0.0000** | — | — | 0.0000 |
| BFS tree of the real graph | 2 | 0.0000 | — | — | — |
| random labelled tree, n=5000 | 256 | 0.0000 | — | — | — |
| induced subgraph 5% (n=462) | 15 | 0.5000 | — | — | — |
| induced subgraph 20% (n=4370) | 11 | 0.5000 | — | — | — |
| induced subgraph 50% (n=12170) | 8 | 0.0000 | — | — | — |

Sample-size sensitivity: median delta 0.0000 at 25k, 50k and 200k quadruples, bootstrap
SE of the mean falling 0.00156 → 0.00055.

1. The real median is 0 and so is the degree-preserving null's, at identical
   delta/diameter. Whatever tree-likeness exists is what the degree sequence already
   produces.
2. The **smaller** subgraphs have the **larger** delta (0.5 at 5% and 20% against 0.0 at
   50%) precisely because they are sparser and have larger diameters. If the graph
   carried negative curvature the dense full graph would be the more hyperbolic one; the
   opposite happens, so delta here is diameter-driven.
3. Both tree references give delta = 0 on every sampled quadruple, which is the property
   a tree metric must have and which calibrates the estimator.

**H3 → NOT SUPPORTED.** The sampled estimate is not the exact hyperbolicity constant
(C(24556,4) ≈ 1.4e17), but the real graph and the null are measured with the same
estimator, so the absence of a gap is not an artefact.

---

## 5. RQ4 — Euclidean vs Poincaré embedding distortion

Identical protocol on both sides: same 24556 nodes, same 8000 sampled nodes, the same
300000 training pairs with the same ground-truth shortest-path distances, 100000
evaluation pairs separated at the **pair** level and verified disjoint (0 overlaps),
same optimiser (AdamW), learning rate, 4000 steps, batch 2048, seed, one global
positive scale per geometry **fitted on evaluation pairs only**, and no recommendation
term anywhere.

| geometry | dim | normalised stress ↓ | mean rel. distortion ↓ | Spearman ↑ | radius median | radius > 0.99 |
|---|---|---|---|---|---|---|
| euclid | 8 | **0.27220** | **0.22621** | **0.28897** | — | — |
| poincare | 8 | 0.27295 | 0.22678 | 0.28803 | 0.0815 | 0.0000 |
| euclid | 16 | **0.21488** | **0.17958** | **0.39116** | — | — |
| poincare | 16 | 0.23553 | 0.19617 | 0.34914 | 0.1097 | 0.0000 |
| euclid | 32 | **0.17652** | 0.14845 | **0.49760** | — | — |
| poincare | 32 | 0.18129 | **0.14440** | **0.01866** | 0.1073 | 0.3258 |

Neighbourhood preservation is ~0 on both sides at every k (0.0 at k=1, 0.0004–0.0017 at
k=10). Held-out edge prediction, 31539 removed edges against degree-matched non-edges
from the reduced graph:

| geometry | d=8 | d=16 | d=32 |
|---|---|---|---|
| euclid | 0.4974 | 0.4980 | 0.5008 |
| poincare | 0.4946 | 0.5057 | 0.4960 |

1. **No hyperbolic advantage.** Equal at d=8 (hyperbolic marginally worse on all three
   metrics) and worse at d=16 and d=32. No dimension favours Poincaré.
2. **The hyperbolic arm degrades with dimension:** Spearman 0.288 → 0.349 → 0.019 while
   32.6% of its points sit past radius 0.99 at d=32. Its marginally better MRD at d=32
   alongside a collapsed rank correlation is the signature of a fit that kept the
   scale and lost the ordering.
3. **Held-out edge prediction carries no signal in either space** (AUC 0.4946–0.5057).

**H4 → NOT SUPPORTED.**

---

## 6. What this does and does not license

None of the following was used, and each would have been false here:

- power law ⇒ hierarchy — H1 is only partially supported, and hierarchy failed anyway;
- hierarchy ⇒ negative-curvature advantage — hierarchy failed;
- low Gromov delta ⇒ better recommendation — delta is identical to the null;
- lower hyperbolic distortion ⇒ higher behaviour enrichment — RQ4 found no advantage;
- better embedding metric ⇒ better RQ-VAE or T5 — not tested here and not implied.

The cross-sectional correlation this programme started from is worth stating plainly
because it is the strongest-looking piece of evidence for hyperbolic modelling and it
does not survive intervention. Over 38 trained models, L1+L2 prefix enrichment
correlated with Recall@10 at Spearman 0.891. On the RQ-VAE side, the L1 codebook collapse
was diagnosed, fixed by a single-variable intervention, and the fix moved L1 codebook
utilisation from 1 of 256 codewords to 256 of 256, full-SID collision from 0.6208 to
0.0450 (better than the healthy Euclidean control's 0.0508), and validation-set L1
enrichment from 1.00 to 11.79 — while **Recall@10 moved from 0.0614043 to 0.0613695,
a change of -0.06%, i.e. noise**. Restoring a genuinely selective L1 level bought
nothing downstream. Across-model correlation is not an intervention lever.

---

## 7. Answer to the study's question

**Does the observed structure of real-world item purchase transitions provide
empirical justification for using hyperbolic geometry in behaviour-guided Semantic ID
learning?**

**On this dataset, at these scales: no.** Three independent lines agree. The degree
distribution is heavy-tailed but lognormal rather than power-law (RQ1). The metric is
not tree-like beyond what the degree sequence already explains — median Gromov delta 0
against 0 for the degree-preserving null, and the subgraph sweep shows the value is
driven by diameter (RQ3). And when the geometry is put under test directly, the
Poincaré embedding matches Euclidean at d=8 and is worse at d=16 and d=32, degrading
into boundary saturation as the dimension grows (RQ4).

Why the three plausible explanations all point the same way:

- **The behaviour graph lacks the structure.** Communities are real (modularity 0.351 vs
  0.157 for the null) and branching is strong (median 7 successors), but communities are
  not curvature, and none of it shows up in the four-point condition beyond degrees.
- **Dimension is not the bottleneck.** At d=32 Euclidean reaches Spearman 0.498; more
  dimensions keep helping the Euclidean arm and actively hurt the hyperbolic one.
- **Method and scale are a real limitation, stated below**, and they are the one place
  where a hyperbolic advantage could still be hiding.

---

## 8. Limitations

1. **The fits are weak.** Stress 0.18–0.27, neighbourhood preservation indistinguishable
   from zero at every k, held-out edge AUC at chance. This is a comparison of two weak
   fits. "No hyperbolic advantage was observed" is not "none exists".
2. **One dataset, one aspect ratio, one domain** (Amazon Instruments).
3. **4000 optimisation steps, batch 2048.** A longer budget was not run.
4. **The whole metric analysis is on the symmetrised graph**, which discards the
   direction of 97.39% of the edges. A directed metric, which the purchase order
   actually has, was not tested.
5. **The δ estimate is Monte Carlo**, not the exact hyperbolicity constant; the null
   comparison uses the same estimator, which is what makes the comparison valid.
6. **The 10% held-out-edge test resolves only 31% of the edges** (about 13.3k matched
   negatives for 31.5k positives), so its AUC has wide confidence intervals.
7. **This study says nothing about L2/L3.** The Semantic ID has three levels and only
   the first was examined.

## 9. Recommended next experiments, in the order they are worth running

1. **Establish that the metric embedding is identifiable here at all.** Both arms sit at
   Spearman ≤ 0.50 with neighbourhood preservation ≈ 0, so the next honest step is to
   raise the budget (steps, and a schedule) on the Euclidean arm alone and find where its
   ceiling is. Comparing two geometries is meaningless until at least one is fitting
   well.
2. **Test the residual chain, not the prefix.** The one place the RQ-VAE-side data
   suggests looking is L2/L3: restoring L1 selectivity changed nothing downstream,
   which makes the L1 prefix the least likely place for a hyperbolic mechanism to pay.
3. **Use a directed or sequential distance objective**, since 97.4% of the item pairs
   are asymmetric and a symmetric metric discards that entirely.
4. **Replicate on a second dataset and a second aspect ratio** before any general claim.
5. Only if 1–3 produce a well-fitting hyperbolic arm should the RQ-VAE side take a
   hyperbolic quantisation objective seriously — and then it must be justified against
   the flat Recall result in §6, not against the 38-model correlation.

---

## 10. A note on how these numbers were produced

Thirteen defects were found between writing the first version of this analysis and
these results. Seven of them would have produced numbers that looked reasonable while
being wrong: the alternative distributions were fitted untruncated to a truncated tail;
neighbourhood preservation counted each node as its own neighbour; the sampled mean
distance included the zero self-distance; a quadruple of source indices was looked up in
a distance matrix whose columns are node positions, which made every four-point delta
wrong (a random tree reported delta 15); the Poincaré denominator broadcast to a
transposed matrix whenever the two sides had different sizes; the evaluation nodes were
disjoint from the training nodes, so the evaluation scored untrained embeddings; and the
training objective compared a (batch, batch) outer product against a (batch,) target,
which torch only warns about. The last one reversed the apparent conclusion of RQ4.

Each is now pinned by a unit test or a runtime assertion, and the wrong intermediate
results were withdrawn rather than kept. The comparison protocol in §5 is built so that
the only difference between the two arms is the distance function.