# round_20261027_031345_66 — hyperbolic x behaviour, iteration 1

## What was designed

Five independent ideation agents proposed five coupling mechanisms; three scorers judged all
five and the declared selection rule (max 1, weighted score >= 3.0, two reviewers above
threshold) selected `incoming_context_barycenter` with a weighted score of 3.25. The five
medians were within 0.14 of each other (3.00-3.14), so the selection was a tie-break, not a
clear win.

The selected mechanism, as verified:

* behaviour signal: train-only directed predecessor counts `N[j -> i]` and a co-consumption
  Jaccard hard negative per item (the highest-overlap item that is *not* an observed
  predecessor);
* curve: fixed-curvature Poincare maps (`project`, `log_map`, `exp_map`, Mobius addition), a
  Frechet mean of the predecessors' codepoints, and the geodesic distance;
* coupling: the first-level quantizer's straight-through assignment consumes
  `first_assignment_logits`, built as `d_c(positive barycentre, code) - 0.5 d_c(hard negative
  barycentre, code)`; the later levels keep the Euclidean path;
* ablations: `BEHAVIOUR_ON = False` falls back to the plain Euclidean assignment, and
  `GEOMETRY = "euclidean"` keeps the behaviour term with a Euclidean metric.

## Defects the review found (9 revisions)

| rev | finding | class |
|---|---|---|
| 1 | the coupled score only fitted the Euclidean argmin (one-way coupling) | real |
| 2 | `NameError: first_assignment_logits` in `RQLayer.forward` | real |
| 3 | the pathway computed but disconnected from the quantizer | real |
| 5-7 | `acosh` derivative at zero distance, then the concentration measure had to read regions rather than predecessor ids | real |
| 9 | `'DistributedDataParallel' object has no attribute 'rq'` in the per-epoch refresh (found by the four-rank run, not by the one-rank smoke) | real |
| 10 | a positional forward read as a keyword mismatch (rebutted with the four-rank run; made explicit anyway) | review artifact |

## Runtime work (RESOURCE_BLOCKED -> optimisation route)

The verified per-sample implementation did not finish an epoch in 900 s: the hard-negative
search intersected every pair of user sets (O(items^2) set work) and the Frechet mean was
rebuilt per sample per step. The same mathematics was vectorised: the sparse item x item
Jaccard, padded predecessor/negative tensors, one barycentre refresh per epoch, and the
geodesic distance through the `|a-b|^2` identity instead of a materialised
(batch, codebook, dim) tensor.

Measured: 7.241 s -> 0.714 s per epoch in the smoke, and **0.153 s per epoch** on the real
four-rank run (1191 epochs in 240 s, loss finite and falling). Full budget 3000 epochs in
about nine minutes.

## Stage 2 result: the mechanism collapsed the partition

| quantity | this arm | baseline `TIGER_RQ-VAE` | best arm in the 67-table sample |
|---|---|---|---|
| largest L1 bucket share | **0.985** | ~0.02 | — |
| `L1` codes | 116 | 247 | 204-248 in the good family |
| `L3` triples | 6,460 | ~20,000 | 15,000-24,000 |
| extension alphabet | 136 | 42 | 7-10 in the good family |
| `L1 Hard ER` | 1.003 | 7.4-8.3 | 15-16 |
| `cooccur_overlap_L3` (the target) | **0.0584** | 0.0756 | 0.168 |
| `adjacency_L3` | 0.0212 | 0.0506 | 0.23-0.27 |

The prediction from the measured curve at 0.0584 is a recall near 0.049, i.e. **below the
baseline**; the Stage 3 run is dispatched and recorded regardless, because admission is
technical validity and the experiment registry keeps failed branches.

## Diagnosis and the next iteration

The logits **replaced** the assignment: `argmax` over
`d_c(positive barycentre, code) - 0.5 d_c(negative barycentre, code)`. Nothing in that
objective keeps the partition spread, so the argmax drifts towards the few codes whose
barycentre is closest for most items and the first level collapses into one bucket holding
98.5% of the catalogue. The reconstruction loss cannot counteract it, because the assignment
it would have to fight is no longer the one it sees.

Two concrete fixes, both small on this tree:

1. **modulate instead of replace**: `logits = -d_euclid(x, code)^2 + lambda * bias` with the
   bias normalised to O(1) against the distance spread and `lambda` around 0.1-0.3, so the
   Euclidean term still orders the codes and the behaviour/curvature term only re-weights;
2. **keep the replacement but make it balanced**: route the assignment through the Sinkhorn
   normalisation the tree already carries (`sk_epsilon`, `sk_iters`) so the marginal is
   uniform and a single bucket cannot absorb the catalogue.

The measured evidence that should decide between them: `cooccur_overlap_L3` is the target
(0.902 rank correlation, isotonic out-of-sample R2 0.705, a step at 0.0585, and the best
observed arm sits at 0.168), while `ext_distinct` is negative (-0.471) and the L1 code count
is irrelevant (-0.199, p = 0.117). A next iteration must show the target moving above 0.0756
before any Stage 3 budget is spent on it.

## Stage 3 result (recorded after the SID metrics, per the measurement contract)

| metric | value |
|---|---|
| `test_recall@10` | **0.058497** |
| `test_recall@5` | 0.038893 |
| `test_ndcg@10` | 0.032057 |
| baseline `TIGER_RQ-VAE` | 0.055537 (delta **+0.0030**) |
| target | 0.065 (not met) |

**This is a counter-example to the coherence curve, and it is the most useful thing this
round produced.** The table's `cooccur_overlap_L3` is 0.0584, *below* the baseline's 0.0756,
and the curve fitted on 67 tables predicts about 0.049 at that value; the measured recall is
0.0585, *above* the baseline. The reason is the collapse: the first level holds 98.5% of the
catalogue in one bucket, so the beam search discriminates on the 572 L2 groups and the 6,460
L3 triples, and the first token stops carrying information altogether. That is the regime the
`l1only` arms in the sample already hinted at (0.0628 with a single L1 code), and it means
`cooccur_overlap_L3` is a population-level predictor and **not** a sufficient statistic for a
particular table's recall.

The consequence for the programme: a mechanism has to be judged on the Stage 3 number, with
the coherence metric as a prior, not as a gate. The next iteration should therefore keep the
L1 spread (so the metric can be read at all) and test whether the *discrimination* coupling
raises the metric into the 0.15 band, rather than taking a collapse as a pass.
