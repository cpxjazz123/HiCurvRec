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

## Iteration 2 (same idea, the diagnosed fixes)

The first iteration's failure mode was that the behaviour/curvature cost *replaced* the
assignment. Iteration 2 keeps the same behaviour signal and the same curvature, and changes
where they act:

* **modulate instead of replace**: `scores = squared Euclidean cost of the item's own latent +
  CONTEXT_WEIGHT * concentration * bias`, where the bias is the geodesic contrast
  `d_c(positive barycentre, code) - 0.5 d_c(negative barycentre, code)`, zero-centred and
  rescaled to unit mean magnitude, so the Euclidean term still orders the codes and the
  behaviour term only re-weights them;
* **the concentration gate is multiplied by `mask.any(-1)`**, so the ~7k items with no observed
  training predecessor keep the plain Euclidean assignment instead of receiving a bias built
  from a zero-vector barycentre;
* **the behaviour-off control is non-detached**, so it differs from the coupled arm only by the
  behaviour term rather than by a broken gradient path;
* **the Euclidean ablation removes the curvature everywhere**, including the context
  barycentre: the control averages raw codepoints and skips the Karcher iterations.

### Review state

Fourteen revisions. Every finding but one was a real defect and each of the four fixes above
came out of this loop:

| rev | finding | real? |
|---|---|---|
| 1-3 | one-way coupling, `NameError`, disconnected path | yes |
| 5-7 | `acosh` derivative at zero distance, concentration read the wrong quantity | yes |
| 9 | DDP wrapper not unwrapped in the per-epoch refresh (only a four-rank run shows it) | yes |
| 10 | a positional forward read as a keyword mismatch | no, rebutted with a four-rank run |
| 11 | the behaviour-off control detached the encoder and codebook | yes |
| 12 | the Euclidean ablation still built a Poincaré barycentre | yes |
| 13 | items without a predecessor got a zero-vector barycentre and a full-strength bias | yes |
| 14 | `torch.where` constant-zero branch disconnects the gradient at exact coincidence | edge case, measure zero |

The loop was stopped at revision 14. The remaining finding is a measure-zero numerical
refinement (exact coincidence of a codepoint and a barycentre), and the per-candidate repair
budget is long past its cap, so the candidate is recorded as `IMPLEMENTATION_NEEDS_FIX` with
the evidence rather than spending more review rounds on it.

**Iteration 2 has no Stage 3 number.** The fixes are smoke-tested (24 steps, finite loss, one
epoch) but the arm was not trained, because the review gate did not return a verified digest.
The next session should either (a) fix the measure-zero gradient branch and take the candidate
through the review once more, or (b) run the same modulate-not-replace design through the
issue #117 executable-verification flow, whose gate executes tests instead of reviewing
prose, and then `stage2_start` it.

## Iteration 3: Sinkhorn-balanced assignment

The balance was added so the behaviour/curvature cost could enter at full strength without
collapse: the coupled logits are Sinkhorn-normalised (differentiable variant, `SK_EPSILON`
0.003, `SK_ITERS` 50) before the `argmax`, so a near-uniform code marginal makes a single
bucket expensive.

| run | L1 codes | largest L1 share | L2 | L3 | ext | `cooccur_overlap_L3` | adjacency L3 |
|---|---|---|---|---|---|---|---|
| baseline `TIGER_RQ-VAE` | 247 | ~0.02 | — | ~20,000 | 42 | **0.0756** | 0.0506 |
| iteration 1 (replace) | 116 | 0.985 | 572 | 6,460 | 136 | 0.0584 | 0.0212 |
| iteration 3a (balance, sign error) | 124 | 0.821 | 1,388 | 7,805 | 143 | 0.0560 | 0.0176 |
| iteration 3b (balance, sign fixed) | 104 | 0.693 | 853 | 8,545 | 117 | **0.0571** | 0.0192 |

The balance softens the collapse (0.985 -> 0.821 -> 0.693) and does nothing for the target
metric, which stays near 0.057 against the 0.0756 baseline. **The Stage 3 budget was not
spent**, per the rule set after iteration 1: the target has to clear the baseline first.

### Why the collapse is structural, not a matter of weights or balancing

Every variant shares one property: the item's assignment is driven by a **barycentre of its
predecessors' codepoints**, and that barycentre is a function of *which region the
predecessors fall in*, not of the item itself. Items whose predecessors land in the same few
regions therefore receive **the same cost vector** and, necessarily, the same `argmax`. The
partition cannot have more distinct outcomes than the behaviour signal has distinct
contexts -- a few hundred -- while a usable SID needs about 20,000 distinct triples. No
weight, temperature, or marginal balancing can create information that the shared context
does not carry.

The fix is therefore not another knob but a different composition: the context must be an
**offset on the item's own latent** (`x_i + offset(context_i)`, or a residual gate that
multiplies the per-item Euclidean cost by a bounded factor derived from the context) so that
two items with identical contexts still separate by their own embeddings. That keeps the
behaviour signal in the assignment while preserving 24.6k distinct outcomes, and it is the
design to try next.
